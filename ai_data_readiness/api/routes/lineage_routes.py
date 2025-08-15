"""Data lineage routes."""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime

from ..models.responses import LineageResponse
from ..middleware.auth import get_current_user
from ..middleware.validation import validate_dataset_id

router = APIRouter()


@router.get("/datasets/{dataset_id}/lineage", response_model=LineageResponse)
async def get_lineage(
    dataset_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get data lineage for a dataset."""
    try:
        validate_dataset_id(dataset_id)
        
        return LineageResponse(
            dataset_id=dataset_id,
            source_datasets=["source_dataset_1", "source_dataset_2"],
            transformations=[
                {
                    "type": "join",
                    "description": "Inner join on customer_id",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ],
            downstream_datasets=["processed_dataset_1"],
            models_trained=["model_v1", "model_v2"],
            created_by="data_engineer",
            created_at=datetime.utcnow()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get lineage: {str(e)}")