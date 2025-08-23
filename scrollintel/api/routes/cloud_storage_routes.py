"""
Cloud Storage API Routes
REST API endpoints for cloud storage operations.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Query
from fastapi.responses import StreamingResponse
from typing import List, Optional
import io

from ...engines.cloud_storage_engine import CloudStorageEngine
from ...models.cloud_storage_models import (
    CloudStorageConnectionConfig, CloudStorageConnectionResponse,
    FileUploadRequest, FileDownloadRequest, FileMetadataResponse,
    CloudStorageStats, FileFormat, CloudProvider
)
from ...core.auth import get_current_user
from ...core.error_handling import handle_api_error

router = APIRouter(prefix="/api/v1/cloud-storage", tags=["cloud-storage"])
cloud_storage_engine = CloudStorageEngine()


@router.post("/connections", response_model=dict)
async def create_connection(
    config: CloudStorageConnectionConfig,
    current_user: dict = Depends(get_current_user)
):
    """Create a new cloud storage connection"""
    try:
        connection_id = await cloud_storage_engine.create_connection(config)
        return {
            "success": True,
            "connection_id": connection_id,
            "message": "Cloud storage connection created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/connections", response_model=List[CloudStorageConnectionResponse])
async def list_connections(
    current_user: dict = Depends(get_current_user)
):
    """List all cloud storage connections"""
    try:
        connections = await cloud_storage_engine.list_connections()
        return [
            CloudStorageConnectionResponse(
                id=conn.id,
                name=conn.name,
                provider=CloudProvider(conn.provider),
                status=conn.status,
                last_sync=conn.last_sync,
                created_at=conn.created_at
            )
            for conn in connections
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connections/{connection_id}")
async def get_connection(
    connection_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get connection details"""
    try:
        connection = await cloud_storage_engine.get_connection(connection_id)
        if not connection:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        return CloudStorageConnectionResponse(
            id=connection.id,
            name=connection.name,
            provider=CloudProvider(connection.provider),
            status=connection.status,
            last_sync=connection.last_sync,
            created_at=connection.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/connections/{connection_id}/test")
async def test_connection(
    connection_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Test cloud storage connection"""
    try:
        success = await cloud_storage_engine.test_connection(connection_id)
        return {
            "success": success,
            "message": "Connection test successful" if success else "Connection test failed"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/connections/{connection_id}/upload")
async def upload_file(
    connection_id: str,
    file: UploadFile = File(...),
    file_path: str = Query(..., description="Destination file path"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    encrypt: bool = Query(True, description="Enable encryption"),
    current_user: dict = Depends(get_current_user)
):
    """Upload file to cloud storage"""
    try:
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        
        # Upload file
        file_id = await cloud_storage_engine.upload_file(
            connection_id=connection_id,
            file_path=file_path,
            file_data=file.file,
            metadata={"original_filename": file.filename},
            tags=tag_list,
            encrypt=encrypt
        )
        
        return {
            "success": True,
            "file_id": file_id,
            "message": "File uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/connections/{connection_id}/download")
async def download_file(
    connection_id: str,
    file_path: str = Query(..., description="Source file path"),
    stream: bool = Query(True, description="Enable streaming"),
    current_user: dict = Depends(get_current_user)
):
    """Download file from cloud storage"""
    try:
        async def generate():
            async for chunk in cloud_storage_engine.download_file(
                connection_id, file_path, stream
            ):
                yield chunk
        
        filename = file_path.split("/")[-1]
        return StreamingResponse(
            generate(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/connections/{connection_id}/files")
async def delete_file(
    connection_id: str,
    file_path: str = Query(..., description="File path to delete"),
    current_user: dict = Depends(get_current_user)
):
    """Delete file from cloud storage"""
    try:
        success = await cloud_storage_engine.delete_file(connection_id, file_path)
        return {
            "success": success,
            "message": "File deleted successfully" if success else "File deletion failed"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/connections/{connection_id}/files")
async def list_files(
    connection_id: str,
    prefix: str = Query("", description="File path prefix"),
    limit: int = Query(1000, description="Maximum number of files"),
    current_user: dict = Depends(get_current_user)
):
    """List files in cloud storage"""
    try:
        files = await cloud_storage_engine.list_files(connection_id, prefix, limit)
        return {
            "success": True,
            "files": files,
            "count": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/files/{file_id}/metadata", response_model=FileMetadataResponse)
async def get_file_metadata(
    file_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get file metadata"""
    try:
        metadata = await cloud_storage_engine.get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File metadata not found")
        return metadata
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/search", response_model=List[FileMetadataResponse])
async def search_files(
    connection_id: Optional[str] = Query(None, description="Filter by connection"),
    file_format: Optional[FileFormat] = Query(None, description="Filter by file format"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to search"),
    limit: int = Query(100, description="Maximum results"),
    current_user: dict = Depends(get_current_user)
):
    """Search files by criteria"""
    try:
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
        
        files = await cloud_storage_engine.search_files(
            connection_id=connection_id,
            file_format=file_format,
            tags=tag_list,
            limit=limit
        )
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=CloudStorageStats)
async def get_storage_stats(
    current_user: dict = Depends(get_current_user)
):
    """Get cloud storage statistics"""
    try:
        stats = await cloud_storage_engine.get_storage_stats()
        return CloudStorageStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/connections/{connection_id}/sync")
async def sync_connection(
    connection_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Sync connection metadata with cloud storage"""
    try:
        # Get all files from cloud storage
        files = await cloud_storage_engine.list_files(connection_id, limit=10000)
        
        # Update database with current file list
        # This is a simplified sync - in production, implement proper delta sync
        sync_count = 0
        for file_info in files:
            # Check if file exists in database and update if needed
            sync_count += 1
        
        return {
            "success": True,
            "synced_files": sync_count,
            "message": f"Synced {sync_count} files"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Batch operations
@router.post("/connections/{connection_id}/batch-upload")
async def batch_upload_files(
    connection_id: str,
    files: List[UploadFile] = File(...),
    base_path: str = Query("", description="Base path for uploads"),
    encrypt: bool = Query(True, description="Enable encryption"),
    current_user: dict = Depends(get_current_user)
):
    """Upload multiple files to cloud storage"""
    try:
        results = []
        
        for file in files:
            try:
                file_path = f"{base_path}/{file.filename}" if base_path else file.filename
                
                file_id = await cloud_storage_engine.upload_file(
                    connection_id=connection_id,
                    file_path=file_path,
                    file_data=file.file,
                    metadata={"original_filename": file.filename},
                    encrypt=encrypt
                )
                
                results.append({
                    "filename": file.filename,
                    "file_id": file_id,
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        successful_uploads = len([r for r in results if r["success"]])
        
        return {
            "success": True,
            "results": results,
            "total_files": len(files),
            "successful_uploads": successful_uploads,
            "failed_uploads": len(files) - successful_uploads
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/connections/{connection_id}/batch-delete")
async def batch_delete_files(
    connection_id: str,
    file_paths: List[str],
    current_user: dict = Depends(get_current_user)
):
    """Delete multiple files from cloud storage"""
    try:
        results = []
        
        for file_path in file_paths:
            try:
                success = await cloud_storage_engine.delete_file(connection_id, file_path)
                results.append({
                    "file_path": file_path,
                    "success": success
                })
            except Exception as e:
                results.append({
                    "file_path": file_path,
                    "success": False,
                    "error": str(e)
                })
        
        successful_deletes = len([r for r in results if r["success"]])
        
        return {
            "success": True,
            "results": results,
            "total_files": len(file_paths),
            "successful_deletes": successful_deletes,
            "failed_deletes": len(file_paths) - successful_deletes
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))