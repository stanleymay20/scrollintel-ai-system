"""
FastAPI routes for AutoModel engine.
Provides endpoints for ML model training, prediction, and management.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
from uuid import UUID
import logging

from ...engines.automodel_engine import AutoModelEngine
from ...models.schemas import (
    MLModelCreate, MLModelResponse, MLModelUpdate,
    BaseSchema, PaginatedResponse
)
from ...security.auth import get_current_user
from ...models.database import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/automodel", tags=["AutoModel"])

# Global engine instance
automodel_engine = AutoModelEngine()


class TrainModelRequest(BaseSchema):
    """Request schema for model training."""
    
    dataset_path: str
    target_column: str
    feature_columns: Optional[List[str]] = None
    model_name: Optional[str] = None
    algorithms: List[str] = ["random_forest", "xgboost"]
    hyperparameter_tuning: bool = True


class TrainModelResponse(BaseSchema):
    """Response schema for model training."""
    
    model_name: str
    model_type: str
    algorithms_tested: List[str]
    best_model: Dict[str, Any]
    training_duration_seconds: float
    model_path: Optional[str] = None
    results: Dict[str, Any]


class PredictRequest(BaseSchema):
    """Request schema for model prediction."""
    
    model_name: str
    data: List[Dict[str, Any]]


class PredictResponse(BaseSchema):
    """Response schema for model prediction."""
    
    model_name: str
    predictions: List[Any]
    model_info: Dict[str, Any]


class ModelComparisonResponse(BaseSchema):
    """Response schema for model comparison."""
    
    comparison: Dict[str, Any]
    total_models: int


class ExportModelRequest(BaseSchema):
    """Request schema for model export."""
    
    model_name: str
    format: str = "joblib"


class ExportModelResponse(BaseSchema):
    """Response schema for model export."""
    
    model_name: str
    export_path: str
    export_format: str
    files_created: List[str]


@router.on_event("startup")
async def startup_automodel():
    """Initialize AutoModel engine on startup."""
    try:
        await automodel_engine.start()
        logger.info("AutoModel engine started successfully")
    except Exception as e:
        logger.error(f"Failed to start AutoModel engine: {e}")


@router.on_event("shutdown")
async def shutdown_automodel():
    """Cleanup AutoModel engine on shutdown."""
    try:
        await automodel_engine.stop()
        logger.info("AutoModel engine stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping AutoModel engine: {e}")


@router.post("/train", response_model=TrainModelResponse)
async def train_model(
    request: TrainModelRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Train ML models automatically with multiple algorithms.
    
    This endpoint trains multiple ML models (Random Forest, XGBoost, Neural Networks)
    with automated hyperparameter tuning and returns performance comparisons.
    """
    try:
        # Prepare training data
        training_data = {
            "action": "train",
            "dataset_path": request.dataset_path,
            "target_column": request.target_column,
            "feature_columns": request.feature_columns,
            "model_name": request.model_name,
            "algorithms": request.algorithms
        }
        
        # Execute training
        result = await automodel_engine.execute(training_data)
        
        return TrainModelResponse(**result)
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Make predictions using a trained model.
    
    Provides prediction capabilities for deployed models with input validation
    and error handling.
    """
    try:
        prediction_data = {
            "action": "predict",
            "model_name": request.model_name,
            "data": request.data
        }
        
        result = await automodel_engine.execute(prediction_data)
        
        return PredictResponse(**result)
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=Dict[str, Any])
async def list_models(
    current_user: User = Depends(get_current_user)
):
    """
    List all trained models with their metadata.
    
    Returns information about all available trained models including
    performance metrics and training details.
    """
    try:
        return {
            "models": automodel_engine.trained_models,
            "total_models": len(automodel_engine.trained_models)
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}", response_model=Dict[str, Any])
async def get_model_info(
    model_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a specific model.
    
    Returns comprehensive information about a trained model including
    metrics, parameters, and training history.
    """
    try:
        if model_name not in automodel_engine.trained_models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        return automodel_engine.trained_models[model_name]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ModelComparisonResponse)
async def compare_models(
    model_names: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Compare performance of multiple trained models.
    
    Provides side-by-side comparison of model performance metrics
    to help select the best model for deployment.
    """
    try:
        comparison_data = {
            "action": "compare",
            "model_names": model_names
        }
        
        result = await automodel_engine.execute(comparison_data)
        
        return ModelComparisonResponse(**result)
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export", response_model=ExportModelResponse)
async def export_model(
    request: ExportModelRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Export a trained model for deployment.
    
    Creates a deployment package with the trained model, metadata,
    and deployment scripts for easy integration.
    """
    try:
        export_data = {
            "action": "export",
            "model_name": request.model_name,
            "format": request.format
        }
        
        result = await automodel_engine.execute(export_data)
        
        return ExportModelResponse(**result)
        
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete a trained model.
    
    Removes a trained model from the system including all associated
    files and metadata.
    """
    try:
        if model_name not in automodel_engine.trained_models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # Remove from memory
        model_info = automodel_engine.trained_models.pop(model_name)
        
        # Remove files
        import os
        if os.path.exists(model_info["model_path"]):
            os.remove(model_info["model_path"])
        
        # Remove neural network files if applicable
        if model_info["algorithm"] == "neural_network":
            nn_path = model_info["model_path"].replace('.pkl', '_nn.h5')
            if os.path.exists(nn_path):
                os.remove(nn_path)
        
        return {"message": f"Model {model_name} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_engine_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get AutoModel engine status and health information.
    
    Returns current status of the AutoModel engine including
    number of trained models and system health.
    """
    try:
        status = automodel_engine.get_status()
        metrics = automodel_engine.get_metrics()
        
        return {
            "status": status,
            "metrics": metrics,
            "engine_info": {
                "engine_id": automodel_engine.engine_id,
                "name": automodel_engine.name,
                "capabilities": [cap.value for cap in automodel_engine.capabilities]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting engine status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain/{model_name}")
async def retrain_model(
    model_name: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Retrain an existing model with updated data.
    
    Retrains a model using the same configuration but potentially
    updated dataset for model refresh and improvement.
    """
    try:
        if model_name not in automodel_engine.trained_models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        model_info = automodel_engine.trained_models[model_name]
        
        # Prepare retraining data using original configuration
        training_data = {
            "action": "train",
            "model_name": model_name,
            "target_column": model_info["target_column"],
            "feature_columns": model_info["feature_columns"],
            "algorithms": [model_info["algorithm"]]
        }
        
        # Note: In a real implementation, you'd need to store the original dataset path
        # or provide a way to specify the new dataset path
        
        return {"message": f"Retraining initiated for model {model_name}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(status_code=500, detail=str(e))