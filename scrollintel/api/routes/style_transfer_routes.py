"""
API routes for style transfer operations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

from ...engines.style_transfer_engine import StyleTransferEngine, StyleType
from ...models.style_transfer_models import (
    StyleTransferRequest, StyleTransferResult, StyleTransferStatus,
    ArtisticStyle, ContentPreservationLevel, StyleTransferConfig,
    BatchProcessingRequest, StyleTransferJob, StylePreset,
    create_default_presets, validate_style_transfer_request,
    calculate_estimated_processing_time
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/style-transfer", tags=["style-transfer"])

# Global style transfer engine instance
style_engine: Optional[StyleTransferEngine] = None
active_jobs: Dict[str, StyleTransferJob] = {}


# Pydantic models for API
class StyleTransferRequestAPI(BaseModel):
    """API model for style transfer requests."""
    content_paths: List[str] = Field(..., description="Paths to content images")
    style_path: Optional[str] = Field(None, description="Path to style reference image")
    style_type: Optional[str] = Field(None, description="Predefined artistic style")
    style_types: Optional[List[str]] = Field(None, description="Multiple styles for comparison")
    batch_processing: bool = Field(False, description="Enable batch processing")
    multiple_styles: bool = Field(False, description="Apply multiple styles to single image")
    preserve_original_colors: bool = Field(False, description="Preserve original image colors")
    output_format: str = Field("png", description="Output image format")
    quality: int = Field(95, description="Output quality (1-100)")
    config: Optional[Dict[str, Any]] = Field(None, description="Custom style transfer configuration")


class StyleTransferConfigAPI(BaseModel):
    """API model for style transfer configuration."""
    content_weight: float = Field(1.0, description="Weight for content preservation")
    style_weight: float = Field(1000.0, description="Weight for style application")
    num_iterations: int = Field(1000, description="Number of optimization iterations")
    max_image_size: int = Field(512, description="Maximum image size for processing")
    preserve_colors: bool = Field(False, description="Preserve original colors")
    blend_ratio: float = Field(1.0, description="Blend ratio with original (0-1)")
    content_preservation_level: str = Field("medium", description="Content preservation level")


class BatchStyleTransferRequestAPI(BaseModel):
    """API model for batch style transfer requests."""
    content_paths: List[str] = Field(..., description="Paths to content images")
    style_path: Optional[str] = Field(None, description="Path to style reference image")
    style_type: Optional[str] = Field(None, description="Predefined artistic style")
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    max_concurrent: int = Field(3, description="Maximum concurrent processes")
    config: Optional[StyleTransferConfigAPI] = Field(None, description="Style transfer configuration")


class StyleTransferResponseAPI(BaseModel):
    """API response model for style transfer results."""
    id: str
    status: str
    result_paths: List[str] = []
    result_urls: List[str] = []
    processing_time: float = 0.0
    style_consistency_score: float = 0.0
    content_preservation_score: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


async def get_style_engine() -> StyleTransferEngine:
    """Get or initialize the style transfer engine."""
    global style_engine
    
    if style_engine is None:
        style_engine = StyleTransferEngine()
        await style_engine.initialize()
    
    return style_engine


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        engine = await get_style_engine()
        return {
            "status": "healthy",
            "engine_initialized": engine.is_initialized,
            "capabilities": engine.get_capabilities()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Style transfer service unavailable")


@router.get("/styles")
async def get_available_styles():
    """Get list of available artistic styles."""
    try:
        styles = [
            {
                "name": style.value,
                "display_name": style.value.replace("_", " ").title(),
                "description": f"{style.value.replace('_', ' ').title()} artistic style"
            }
            for style in ArtisticStyle
        ]
        
        return {
            "styles": styles,
            "total_count": len(styles)
        }
    except Exception as e:
        logger.error(f"Failed to get available styles: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve available styles")


@router.get("/presets")
async def get_style_presets():
    """Get predefined style presets."""
    try:
        presets = create_default_presets()
        
        return {
            "presets": [preset.to_dict() for preset in presets],
            "total_count": len(presets)
        }
    except Exception as e:
        logger.error(f"Failed to get style presets: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve style presets")


@router.post("/transfer", response_model=StyleTransferResponseAPI)
async def create_style_transfer(
    request: StyleTransferRequestAPI,
    background_tasks: BackgroundTasks
):
    """Create a new style transfer request."""
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Validate and convert request
        transfer_request = StyleTransferRequest(
            id=request_id,
            content_paths=request.content_paths,
            style_path=request.style_path,
            style_type=ArtisticStyle(request.style_type) if request.style_type else None,
            style_types=[ArtisticStyle(s) for s in request.style_types] if request.style_types else None,
            batch_processing=request.batch_processing,
            multiple_styles=request.multiple_styles,
            preserve_original_colors=request.preserve_original_colors,
            output_format=request.output_format,
            quality=request.quality,
            config=StyleTransferConfig.from_dict(request.config) if request.config else None
        )
        
        # Create job
        job = StyleTransferJob(
            id=request_id,
            request=transfer_request,
            status=StyleTransferStatus.PENDING
        )
        
        # Calculate estimated completion time
        estimated_time = calculate_estimated_processing_time(transfer_request)
        job.estimated_completion_time = estimated_time
        
        # Store job
        active_jobs[request_id] = job
        
        # Start processing in background
        background_tasks.add_task(process_style_transfer_job, request_id)
        
        return StyleTransferResponseAPI(
            id=request_id,
            status=job.status.value,
            metadata={
                "estimated_processing_time": estimated_time,
                "num_images": len(transfer_request.content_paths),
                "batch_processing": transfer_request.batch_processing,
                "multiple_styles": transfer_request.multiple_styles
            }
        )
        
    except ValueError as e:
        logger.error(f"Invalid style transfer request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create style transfer request: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create style transfer request")


@router.post("/batch", response_model=StyleTransferResponseAPI)
async def create_batch_style_transfer(
    request: BatchStyleTransferRequestAPI,
    background_tasks: BackgroundTasks
):
    """Create a batch style transfer request."""
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Convert to standard request format
        transfer_request = StyleTransferRequest(
            id=request_id,
            content_paths=request.content_paths,
            style_path=request.style_path,
            style_type=ArtisticStyle(request.style_type) if request.style_type else None,
            batch_processing=True,
            config=StyleTransferConfig.from_dict(request.config.dict()) if request.config else None
        )
        
        # Create job
        job = StyleTransferJob(
            id=request_id,
            request=transfer_request,
            status=StyleTransferStatus.PENDING
        )
        
        # Store job
        active_jobs[request_id] = job
        
        # Start processing in background
        background_tasks.add_task(process_style_transfer_job, request_id)
        
        return StyleTransferResponseAPI(
            id=request_id,
            status=job.status.value,
            metadata={
                "batch_size": len(request.content_paths),
                "parallel_processing": request.parallel_processing,
                "max_concurrent": request.max_concurrent
            }
        )
        
    except ValueError as e:
        logger.error(f"Invalid batch style transfer request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create batch style transfer request: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create batch style transfer request")


@router.post("/upload-and-transfer")
async def upload_and_transfer(
    content_file: UploadFile = File(...),
    style_file: Optional[UploadFile] = File(None),
    style_type: Optional[str] = Form(None),
    config: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """Upload images and create style transfer request."""
    try:
        # Save uploaded content file
        content_path = f"./temp/content_{uuid.uuid4()}_{content_file.filename}"
        Path("./temp").mkdir(exist_ok=True)
        
        with open(content_path, "wb") as f:
            content = await content_file.read()
            f.write(content)
        
        # Save uploaded style file if provided
        style_path = None
        if style_file:
            style_path = f"./temp/style_{uuid.uuid4()}_{style_file.filename}"
            with open(style_path, "wb") as f:
                style_content = await style_file.read()
                f.write(style_content)
        
        # Parse config if provided
        transfer_config = None
        if config:
            import json
            config_dict = json.loads(config)
            transfer_config = StyleTransferConfig.from_dict(config_dict)
        
        # Create style transfer request
        request_id = str(uuid.uuid4())
        transfer_request = StyleTransferRequest(
            id=request_id,
            content_paths=[content_path],
            style_path=style_path,
            style_type=ArtisticStyle(style_type) if style_type else None,
            config=transfer_config
        )
        
        # Create and store job
        job = StyleTransferJob(
            id=request_id,
            request=transfer_request,
            status=StyleTransferStatus.PENDING
        )
        active_jobs[request_id] = job
        
        # Start processing in background
        if background_tasks:
            background_tasks.add_task(process_style_transfer_job, request_id)
        else:
            asyncio.create_task(process_style_transfer_job(request_id))
        
        return {
            "id": request_id,
            "status": job.status.value,
            "message": "Files uploaded and style transfer started"
        }
        
    except Exception as e:
        logger.error(f"Upload and transfer failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload and process files")


@router.get("/jobs/{job_id}", response_model=StyleTransferResponseAPI)
async def get_job_status(job_id: str):
    """Get status of a style transfer job."""
    try:
        if job_id not in active_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = active_jobs[job_id]
        
        response = StyleTransferResponseAPI(
            id=job.id,
            status=job.status.value,
            processing_time=0.0,
            metadata={
                "progress": job.progress,
                "current_step": job.current_step,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
        )
        
        if job.result:
            response.result_paths = job.result.result_paths
            response.result_urls = job.result.result_urls
            response.processing_time = job.result.processing_time
            response.style_consistency_score = job.result.style_consistency_score
            response.content_preservation_score = job.result.content_preservation_score
            response.error_message = job.result.error_message
            response.metadata.update(job.result.metadata)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job status")


@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List style transfer jobs."""
    try:
        jobs = list(active_jobs.values())
        
        # Filter by status if provided
        if status:
            jobs = [job for job in jobs if job.status.value == status]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        total_count = len(jobs)
        jobs = jobs[offset:offset + limit]
        
        return {
            "jobs": [
                {
                    "id": job.id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "created_at": job.created_at.isoformat(),
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "num_images": len(job.request.content_paths),
                    "style_type": job.request.style_type.value if job.request.style_type else "custom"
                }
                for job in jobs
            ],
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve jobs")


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a style transfer job."""
    try:
        if job_id not in active_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = active_jobs[job_id]
        
        if job.status in [StyleTransferStatus.COMPLETED, StyleTransferStatus.FAILED]:
            raise HTTPException(status_code=400, detail="Cannot cancel completed or failed job")
        
        job.status = StyleTransferStatus.CANCELLED
        job.completed_at = job.created_at.__class__.now()
        
        return {
            "id": job_id,
            "status": job.status.value,
            "message": "Job cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel job")


@router.get("/results/{job_id}/download")
async def download_result(job_id: str, image_index: int = 0):
    """Download a result image from a completed job."""
    try:
        if job_id not in active_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = active_jobs[job_id]
        
        if job.status != StyleTransferStatus.COMPLETED or not job.result:
            raise HTTPException(status_code=400, detail="Job not completed or no results available")
        
        if image_index >= len(job.result.result_paths):
            raise HTTPException(status_code=400, detail="Invalid image index")
        
        result_path = job.result.result_paths[image_index]
        
        if not Path(result_path).exists():
            raise HTTPException(status_code=404, detail="Result file not found")
        
        return FileResponse(
            result_path,
            media_type="image/png",
            filename=f"style_transfer_result_{job_id}_{image_index}.png"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download result: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download result")


@router.get("/capabilities")
async def get_capabilities():
    """Get style transfer engine capabilities."""
    try:
        engine = await get_style_engine()
        return engine.get_capabilities()
    except Exception as e:
        logger.error(f"Failed to get capabilities: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve capabilities")


async def process_style_transfer_job(job_id: str):
    """Background task to process style transfer job."""
    try:
        if job_id not in active_jobs:
            logger.error(f"Job {job_id} not found in active jobs")
            return
        
        job = active_jobs[job_id]
        job.start_processing()
        
        # Get style transfer engine
        engine = await get_style_engine()
        
        # Update progress
        job.update_progress(10.0, "Initializing style transfer")
        
        # Process the request
        result = await engine.process_style_transfer_request(job.request)
        
        # Update progress
        job.update_progress(90.0, "Finalizing results")
        
        # Complete job with result
        job.complete_with_result(result)
        
        logger.info(f"Style transfer job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Style transfer job {job_id} failed: {str(e)}")
        
        if job_id in active_jobs:
            active_jobs[job_id].fail_with_error(str(e))


# Cleanup completed jobs periodically
@router.on_event("startup")
async def startup_event():
    """Initialize the style transfer service."""
    try:
        # Initialize the engine
        await get_style_engine()
        logger.info("Style transfer service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize style transfer service: {str(e)}")


@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on service shutdown."""
    try:
        # Cancel any pending jobs
        for job in active_jobs.values():
            if job.status in [StyleTransferStatus.PENDING, StyleTransferStatus.PROCESSING]:
                job.status = StyleTransferStatus.CANCELLED
        
        logger.info("Style transfer service shutdown completed")
    except Exception as e:
        logger.error(f"Error during style transfer service shutdown: {str(e)}")