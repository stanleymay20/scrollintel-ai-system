"""Drift monitoring routes."""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime

from ..models.requests import DriftMonitoringRequest
from ..models.responses import DriftReportResponse
from ..middleware.auth import get_current_user
from ..middleware.validation import validate_dataset_id

router = APIRouter()


@router.post("/datasets/{dataset_id}/drift", response_model=DriftReportResponse)
async def monitor_drift(
    dataset_id: str,
    request: DriftMonitoringRequest,
    current_user: dict = Depends(get_current_user)
):
    """Set up drift monitoring for a dataset."""
    try:
        validate_dataset_id(dataset_id)
        validate_dataset_id(request.reference_dataset_id)
        
        return DriftReportResponse(
            dataset_id=dataset_id,
            reference_dataset_id=request.reference_dataset_id,
            drift_score=0.08,
            feature_drift_scores={
                "feature_1": 0.05,
                "feature_2": 0.12,
                "feature_3": 0.03
            },
            statistical_tests={
                "ks_test": {"statistic": 0.08, "p_value": 0.15},
                "chi2_test": {"statistic": 12.5, "p_value": 0.02}
            },
            alerts=[],
            recommendations=[
                {
                    "type": "model_retraining",
                    "description": "Consider retraining model due to feature_2 drift",
                    "priority": "medium"
                }
            ],
            generated_at=datetime.utcnow()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift monitoring failed: {str(e)}")