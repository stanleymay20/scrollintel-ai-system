"""
API routes for content export and format conversion functionality.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import os
import tempfile
import shutil

from scrollintel.engines.visual_generation.utils.content_exporter import (
    ContentExporter,
    FormatConverter,
    ExportSettings,
    ContentMetadata,
    ExportFormat,
    QualityLevel,
    CompressionType,
    ExportResult
)

router = APIRouter(prefix="/api/v1/export", tags=["Content Export"])

# Initialize exporter
content_exporter = ContentExporter()
format_converter = FormatConverter(content_exporter)


class ExportRequest(BaseModel):
    """Request model for content export."""
    content_path: str = Field(..., description="Path to content file")
    format: ExportFormat = Field(..., description="Target export format")
    quality: QualityLevel = Field(QualityLevel.HIGH, description="Quality level")
    compression: CompressionType = Field(CompressionType.BALANCED, description="Compression type")
    resolution: Optional[tuple] = Field(None, description="Target resolution (width, height)")
    frame_rate: Optional[int] = Field(None, description="Target frame rate for video")
    bitrate: Optional[str] = Field(None, description="Target bitrate for video")
    preserve_metadata: bool = Field(True, description="Preserve original metadata")
    watermark: Optional[str] = Field(None, description="Watermark text")
    output_filename: Optional[str] = Field(None, description="Custom output filename")


class BatchExportRequest(BaseModel):
    """Request model for batch export."""
    content_paths: List[str] = Field(..., description="List of content file paths")
    format: ExportFormat = Field(..., description="Target export format")
    quality: QualityLevel = Field(QualityLevel.HIGH, description="Quality level")
    compression: CompressionType = Field(CompressionType.BALANCED, description="Compression type")
    max_concurrent: int = Field(3, description="Maximum concurrent exports")


class MetadataRequest(BaseModel):
    """Request model for content metadata."""
    title: Optional[str] = None
    description: Optional[str] = None
    creator: Optional[str] = None
    generation_prompt: Optional[str] = None
    model_used: Optional[str] = None
    tags: Optional[List[str]] = None
    copyright: Optional[str] = None
    custom_fields: Optional[Dict[str, Any]] = None


class ArchiveRequest(BaseModel):
    """Request model for creating archives."""
    content_paths: List[str] = Field(..., description="List of content file paths")
    archive_format: ExportFormat = Field(ExportFormat.ZIP, description="Archive format")
    archive_name: Optional[str] = Field(None, description="Custom archive name")


class ConversionRequest(BaseModel):
    """Request model for format conversion."""
    input_path: str = Field(..., description="Path to input file")
    target_format: ExportFormat = Field(..., description="Target format")
    quality: QualityLevel = Field(QualityLevel.HIGH, description="Quality level")
    preserve_metadata: bool = Field(True, description="Preserve metadata")


@router.post("/export", response_model=ExportResult)
async def export_content(
    request: ExportRequest,
    metadata: Optional[MetadataRequest] = None
):
    """
    Export content with specified settings.
    
    Args:
        request: Export configuration
        metadata: Optional content metadata
        
    Returns:
        Export result with details
    """
    try:
        # Convert request to export settings
        settings = ExportSettings(
            format=request.format,
            quality=request.quality,
            compression=request.compression,
            resolution=request.resolution,
            frame_rate=request.frame_rate,
            bitrate=request.bitrate,
            preserve_metadata=request.preserve_metadata,
            watermark=request.watermark
        )
        
        # Convert metadata if provided
        content_metadata = None
        if metadata:
            content_metadata = ContentMetadata(
                title=metadata.title,
                description=metadata.description,
                creator=metadata.creator,
                generation_prompt=metadata.generation_prompt,
                model_used=metadata.model_used,
                tags=metadata.tags or [],
                copyright=metadata.copyright,
                custom_fields=metadata.custom_fields or {}
            )
        
        # Perform export
        result = await content_exporter.export_content(
            request.content_path,
            settings,
            content_metadata,
            request.output_filename
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/batch-export", response_model=List[ExportResult])
async def batch_export_content(request: BatchExportRequest):
    """
    Export multiple content files with the same settings.
    
    Args:
        request: Batch export configuration
        
    Returns:
        List of export results
    """
    try:
        settings = ExportSettings(
            format=request.format,
            quality=request.quality,
            compression=request.compression
        )
        
        results = await content_exporter.batch_export(
            request.content_paths,
            settings,
            max_concurrent=request.max_concurrent
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch export failed: {str(e)}")


@router.post("/convert", response_model=ExportResult)
async def convert_format(request: ConversionRequest):
    """
    Convert content to a different format.
    
    Args:
        request: Conversion configuration
        
    Returns:
        Conversion result
    """
    try:
        result = await format_converter.convert_format(
            request.input_path,
            request.target_format,
            request.quality,
            request.preserve_metadata
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Format conversion failed: {str(e)}")


@router.post("/batch-convert", response_model=List[ExportResult])
async def batch_convert_format(
    input_paths: List[str],
    target_format: ExportFormat,
    quality: QualityLevel = QualityLevel.HIGH
):
    """
    Convert multiple files to the same format.
    
    Args:
        input_paths: List of input file paths
        target_format: Target format for all files
        quality: Quality level for conversion
        
    Returns:
        List of conversion results
    """
    try:
        results = await format_converter.batch_convert(
            input_paths,
            target_format,
            quality
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch conversion failed: {str(e)}")


@router.post("/optimize-web", response_model=ExportResult)
async def optimize_for_web(input_path: str):
    """
    Optimize content for web delivery.
    
    Args:
        input_path: Path to input file
        
    Returns:
        Optimization result
    """
    try:
        result = await format_converter.optimize_for_web(input_path)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Web optimization failed: {str(e)}")


@router.post("/archive", response_model=ExportResult)
async def create_archive(request: ArchiveRequest):
    """
    Create an archive of multiple content files.
    
    Args:
        request: Archive configuration
        
    Returns:
        Archive creation result
    """
    try:
        result = await content_exporter.create_archive(
            request.content_paths,
            request.archive_format,
            request.archive_name
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Archive creation failed: {str(e)}")


@router.get("/formats/{content_type}", response_model=List[ExportFormat])
async def get_supported_formats(content_type: str):
    """
    Get supported export formats for a content type.
    
    Args:
        content_type: Type of content (image, video)
        
    Returns:
        List of supported formats
    """
    try:
        formats = content_exporter.get_supported_formats(content_type)
        return formats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get formats: {str(e)}")


@router.get("/format-info/{format}", response_model=Dict[str, Any])
async def get_format_info(format: ExportFormat):
    """
    Get information about a specific format.
    
    Args:
        format: Export format
        
    Returns:
        Format information
    """
    try:
        info = content_exporter.get_format_info(format)
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get format info: {str(e)}")


@router.get("/download/{filename}")
async def download_exported_file(filename: str):
    """
    Download an exported file.
    
    Args:
        filename: Name of the exported file
        
    Returns:
        File response for download
    """
    try:
        file_path = content_exporter.output_directory / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.post("/upload-and-export")
async def upload_and_export(
    file: UploadFile = File(...),
    format: ExportFormat = ExportFormat.JPEG,
    quality: QualityLevel = QualityLevel.HIGH,
    compression: CompressionType = CompressionType.BALANCED
):
    """
    Upload a file and export it with specified settings.
    
    Args:
        file: Uploaded file
        format: Target export format
        quality: Quality level
        compression: Compression type
        
    Returns:
        Export result
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # Export the file
            settings = ExportSettings(
                format=format,
                quality=quality,
                compression=compression
            )
            
            result = await content_exporter.export_content(temp_path, settings)
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload and export failed: {str(e)}")


@router.get("/export-status/{export_id}")
async def get_export_status(export_id: str):
    """
    Get the status of an export operation.
    
    Args:
        export_id: Export operation ID
        
    Returns:
        Export status information
    """
    # This would typically check a database or cache for export status
    # For now, return a placeholder response
    return {
        "export_id": export_id,
        "status": "completed",
        "progress": 100,
        "message": "Export completed successfully"
    }


@router.delete("/cleanup")
async def cleanup_exports(older_than_days: int = 7):
    """
    Clean up old exported files.
    
    Args:
        older_than_days: Delete files older than this many days
        
    Returns:
        Cleanup summary
    """
    try:
        import time
        from pathlib import Path
        
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
        deleted_count = 0
        total_size = 0
        
        for file_path in content_exporter.output_directory.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_size = file_path.stat().st_size
                file_path.unlink()
                deleted_count += 1
                total_size += file_size
        
        return {
            "deleted_files": deleted_count,
            "freed_space_bytes": total_size,
            "freed_space_mb": round(total_size / (1024 * 1024), 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")