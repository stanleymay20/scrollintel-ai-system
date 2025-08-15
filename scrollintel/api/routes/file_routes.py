"""
File upload routes for ScrollIntel API.
Handles file upload, processing, and dataset creation.
"""

import os
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ...core.config import get_config
from ...models.database_utils import get_sync_db as get_db
from ...models.schemas import (
    FileUploadResponse, FileProcessingStatus, DataPreviewResponse,
    DataQualityReport, DatasetCreate, DatasetResponse
)
from ...engines.file_processor import FileProcessorEngine
from ...security.middleware import require_permission
from ...security.permissions import Permission
from ...security.audit import audit_logger, AuditAction
from ...core.interfaces import SecurityContext


def create_file_router() -> APIRouter:
    """Create file upload router."""
    
    router = APIRouter()
    config = get_config()
    file_processor = FileProcessorEngine()
    
    @router.post("/upload", response_model=FileUploadResponse)
    async def upload_file(
        file: UploadFile = File(...),
        name: Optional[str] = Form(None),
        description: Optional[str] = Form(None),
        auto_detect_schema: bool = Form(True),
        generate_preview: bool = Form(True),
        context: SecurityContext = Depends(require_permission(Permission.DATA_UPLOAD)),
        db: Session = Depends(get_db)
    ):
        """Upload and process a data file."""
        
        try:
            # Log file upload attempt
            await audit_logger.log(
                action=AuditAction.FILE_UPLOAD,
                resource_type="file",
                resource_id=file.filename,
                user_id=context.user_id,
                session_id=context.session_id,
                ip_address=context.ip_address,
                details={
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "file_size": file.size,
                    "auto_detect_schema": auto_detect_schema,
                    "generate_preview": generate_preview
                },
                success=True
            )
            
            # Process file upload
            result = await file_processor.process_upload(
                file=file,
                user_id=str(context.user_id),
                storage_path=config.system.storage_path,
                db_session=db,
                auto_detect_schema=auto_detect_schema,
                generate_preview=generate_preview
            )
            
            # Log successful upload
            await audit_logger.log(
                action=AuditAction.FILE_UPLOAD_COMPLETE,
                resource_type="file",
                resource_id=result.upload_id,
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "upload_id": result.upload_id,
                    "detected_type": result.detected_type,
                    "file_size": result.file_size
                },
                success=True
            )
            
            return result
            
        except Exception as e:
            # Log upload error
            await audit_logger.log(
                action=AuditAction.FILE_UPLOAD,
                resource_type="file",
                resource_id=file.filename if file else "unknown",
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "filename": file.filename if file else "unknown",
                    "error": str(e)
                },
                success=False,
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File upload failed: {str(e)}"
            )
    
    @router.get("/upload/{upload_id}/status", response_model=FileProcessingStatus)
    async def get_upload_status(
        upload_id: str,
        context: SecurityContext = Depends(require_permission(Permission.DATA_READ)),
        db: Session = Depends(get_db)
    ):
        """Get processing status for an uploaded file."""
        
        try:
            status_info = await file_processor.get_processing_status(upload_id, db)
            return status_info
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Upload status not found: {str(e)}"
            )
    
    @router.get("/upload/{upload_id}/preview", response_model=DataPreviewResponse)
    async def get_file_preview(
        upload_id: str,
        context: SecurityContext = Depends(require_permission(Permission.DATA_READ)),
        db: Session = Depends(get_db)
    ):
        """Get data preview for an uploaded file."""
        
        try:
            from ...models.database import FileUpload
            
            file_upload = db.query(FileUpload).filter_by(upload_id=upload_id).first()
            
            if not file_upload:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Upload with ID {upload_id} not found"
                )
            
            if file_upload.processing_status != "completed":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File processing not completed. Status: {file_upload.processing_status}"
                )
            
            if not file_upload.preview_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Preview data not available for this upload"
                )
            
            return DataPreviewResponse(**file_upload.preview_data)
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get file preview: {str(e)}"
            )
    
    @router.get("/upload/{upload_id}/quality", response_model=DataQualityReport)
    async def get_quality_report(
        upload_id: str,
        context: SecurityContext = Depends(require_permission(Permission.DATA_READ)),
        db: Session = Depends(get_db)
    ):
        """Get data quality report for an uploaded file."""
        
        try:
            from ...models.database import FileUpload
            
            file_upload = db.query(FileUpload).filter_by(upload_id=upload_id).first()
            
            if not file_upload:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Upload with ID {upload_id} not found"
                )
            
            if file_upload.processing_status != "completed":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File processing not completed. Status: {file_upload.processing_status}"
                )
            
            if not file_upload.quality_report:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Quality report not available for this upload"
                )
            
            return DataQualityReport(**file_upload.quality_report)
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get quality report: {str(e)}"
            )
    
    @router.post("/upload/{upload_id}/create-dataset", response_model=DatasetResponse)
    async def create_dataset_from_upload(
        upload_id: str,
        dataset_name: str = Form(...),
        dataset_description: Optional[str] = Form(None),
        context: SecurityContext = Depends(require_permission(Permission.DATA_CREATE)),
        db: Session = Depends(get_db)
    ):
        """Create a dataset from an uploaded file."""
        
        try:
            # Log dataset creation attempt
            await audit_logger.log(
                action=AuditAction.DATASET_CREATE,
                resource_type="dataset",
                resource_id=dataset_name,
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "upload_id": upload_id,
                    "dataset_name": dataset_name,
                    "dataset_description": dataset_description
                },
                success=True
            )
            
            # Create dataset from upload
            dataset_id = await file_processor.create_dataset_from_upload(
                upload_id=upload_id,
                dataset_name=dataset_name,
                dataset_description=dataset_description,
                db_session=db
            )
            
            # Get created dataset
            from ...models.database import Dataset
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            
            if not dataset:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Dataset created but could not be retrieved"
                )
            
            # Log successful dataset creation
            await audit_logger.log(
                action=AuditAction.DATASET_CREATE_COMPLETE,
                resource_type="dataset",
                resource_id=str(dataset.id),
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "dataset_id": str(dataset.id),
                    "dataset_name": dataset.name,
                    "source_type": dataset.source_type,
                    "row_count": dataset.row_count
                },
                success=True
            )
            
            return DatasetResponse(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                source_type=dataset.source_type,
                data_schema=dataset.data_schema,
                dataset_metadata=dataset.dataset_metadata,
                row_count=dataset.row_count,
                file_path=dataset.file_path,
                connection_string=dataset.connection_string,
                table_name=dataset.table_name,
                query=dataset.query,
                refresh_interval_minutes=dataset.refresh_interval_minutes,
                last_refreshed=dataset.last_refreshed,
                is_active=dataset.is_active,
                created_at=dataset.created_at,
                updated_at=dataset.updated_at
            )
            
        except Exception as e:
            # Log dataset creation error
            await audit_logger.log(
                action=AuditAction.DATASET_CREATE,
                resource_type="dataset",
                resource_id=dataset_name,
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "upload_id": upload_id,
                    "error": str(e)
                },
                success=False,
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create dataset: {str(e)}"
            )
    
    @router.get("/uploads", response_model=List[FileUploadResponse])
    async def list_uploads(
        skip: int = 0,
        limit: int = 100,
        status_filter: Optional[str] = None,
        context: SecurityContext = Depends(require_permission(Permission.DATA_READ)),
        db: Session = Depends(get_db)
    ):
        """List uploaded files for the current user."""
        
        try:
            from ...models.database import FileUpload
            
            query = db.query(FileUpload).filter_by(user_id=context.user_id)
            
            if status_filter:
                query = query.filter_by(processing_status=status_filter)
            
            uploads = query.offset(skip).limit(limit).all()
            
            return [
                FileUploadResponse(
                    upload_id=upload.upload_id,
                    filename=upload.filename,
                    original_filename=upload.original_filename,
                    file_path=upload.file_path,
                    file_size=upload.file_size,
                    content_type=upload.content_type,
                    detected_type=upload.detected_type,
                    schema_info=upload.schema_info or {},
                    preview_data=upload.preview_data,
                    quality_report=upload.quality_report,
                    dataset_id=upload.dataset_id,
                    created_at=upload.created_at
                )
                for upload in uploads
            ]
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list uploads: {str(e)}"
            )
    
    @router.delete("/upload/{upload_id}")
    async def delete_upload(
        upload_id: str,
        context: SecurityContext = Depends(require_permission(Permission.DATA_DELETE)),
        db: Session = Depends(get_db)
    ):
        """Delete an uploaded file."""
        
        try:
            from ...models.database import FileUpload
            
            file_upload = db.query(FileUpload).filter_by(
                upload_id=upload_id,
                user_id=context.user_id
            ).first()
            
            if not file_upload:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Upload with ID {upload_id} not found"
                )
            
            # Delete physical file
            try:
                if os.path.exists(file_upload.file_path):
                    os.remove(file_upload.file_path)
            except Exception as e:
                # Log but don't fail if file deletion fails
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to delete physical file {file_upload.file_path}: {str(e)}")
            
            # Delete database record
            db.delete(file_upload)
            db.commit()
            
            # Log deletion
            await audit_logger.log(
                action=AuditAction.FILE_DELETE,
                resource_type="file",
                resource_id=upload_id,
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "upload_id": upload_id,
                    "filename": file_upload.filename
                },
                success=True
            )
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": f"Upload {upload_id} deleted successfully"}
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete upload: {str(e)}"
            )
    
    @router.post("/validate")
    async def validate_file(
        file: UploadFile = File(...),
        context: SecurityContext = Depends(require_permission(Permission.DATA_READ))
    ):
        """Validate a file without uploading it."""
        
        try:
            validation_result = await file_processor.validate_file_format(file)
            return validation_result
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File validation failed: {str(e)}"
            )
    
    @router.get("/uploads/search")
    async def search_uploads(
        q: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        context: SecurityContext = Depends(require_permission(Permission.DATA_READ)),
        db: Session = Depends(get_db)
    ):
        """Search and filter uploaded files."""
        
        try:
            uploads = await file_processor.list_user_uploads(
                user_id=str(context.user_id),
                db_session=db,
                skip=skip,
                limit=limit,
                status_filter=status,
                search_term=q
            )
            
            return {
                "uploads": uploads,
                "total": len(uploads),
                "skip": skip,
                "limit": limit
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to search uploads: {str(e)}"
            )
    
    @router.get("/supported-formats")
    async def get_supported_formats(
        context: SecurityContext = Depends(require_permission(Permission.DATA_READ))
    ):
        """Get list of supported file formats with detailed information."""
        
        try:
            return await file_processor.get_supported_formats()
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get supported formats: {str(e)}"
            )
    
    return router

# Create the router instance
router = create_file_router()