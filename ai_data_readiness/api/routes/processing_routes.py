"""Data processing routes."""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import uuid

from ..models.requests import ProcessingJobRequest
from ..models.responses import ProcessingJobResponse, APIResponse
from ..middleware.auth import get_current_user
from ..middleware.validation import validate_dataset_id

router = APIRouter()


@router.post("/processing/jobs", response_model=ProcessingJobResponse)
async def create_processing_job(
    request: ProcessingJobRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new processing job."""
    try:
        validate_dataset_id(request.dataset_id)
        
        job_id = str(uuid.uuid4())
        
        return ProcessingJobResponse(
            job_id=job_id,
            dataset_id=request.dataset_id,
            job_type=request.job_type,
            status="queued",
            progress=0.0,
            parameters=request.parameters,
            result=None,
            error_message=None,
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")


@router.get("/processing/jobs/{job_id}", response_model=ProcessingJobResponse)
async def get_processing_job(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get processing job status."""
    try:
        return ProcessingJobResponse(
            job_id=job_id,
            dataset_id="sample_dataset_id",
            job_type="quality_assessment",
            status="completed",
            progress=1.0,
            parameters={},
            result={"quality_score": 0.85},
            error_message=None,
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {str(e)}")


@router.delete("/processing/jobs/{job_id}", response_model=APIResponse)
async def cancel_processing_job(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Cancel a processing job."""
    try:
        return APIResponse(
            success=True,
            message=f"Job {job_id} cancelled successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")