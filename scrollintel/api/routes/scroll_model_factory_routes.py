"""
API routes for ScrollModelFactory engine.
Provides endpoints for custom model creation, validation, and deployment.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ...models.database_utils import get_db
from ...models.database import MLModel, Dataset
from ...models.schemas import MLModelCreate, MLModelResponse
from ...engines.scroll_model_factory import ScrollModelFactory
from ...security.auth import get_current_user
from ...security.permissions import require_permission
from ...models.database import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/model-factory", tags=["model-factory"])

# Global engine instance
model_factory_engine = None


async def get_model_factory_engine() -> ScrollModelFactory:
    """Get or create ScrollModelFactory engine instance."""
    global model_factory_engine
    if model_factory_engine is None:
        model_factory_engine = ScrollModelFactory()
        await model_factory_engine.start()
    return model_factory_engine


# Pydantic models for request/response
class ModelTemplateResponse(BaseModel):
    """Response model for model templates."""
    templates: Dict[str, Any]
    count: int


class AlgorithmResponse(BaseModel):
    """Response model for algorithms."""
    algorithms: Dict[str, Any]
    count: int


class CustomModelRequest(BaseModel):
    """Request model for custom model creation."""
    model_name: str = Field(..., min_length=1, max_length=255)
    dataset_id: UUID
    algorithm: str
    template: Optional[str] = None
    target_column: str
    feature_columns: Optional[List[str]] = None
    custom_params: Dict[str, Any] = Field(default_factory=dict)
    validation_strategy: str = "train_test_split"
    hyperparameter_tuning: bool = False


class CustomModelResponse(BaseModel):
    """Response model for custom model creation."""
    model_id: str
    model_name: str
    algorithm: str
    template: Optional[str]
    target_column: str
    feature_columns: List[str]
    model_path: str
    is_classification: bool
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    validation_strategy: str
    training_duration: float
    created_at: str


class ModelValidationRequest(BaseModel):
    """Request model for model validation."""
    model_id: str
    validation_data: Optional[List[List[Any]]] = None


class ModelValidationResponse(BaseModel):
    """Response model for model validation."""
    model_id: str
    validation_status: str
    predictions: Optional[List[Any]] = None
    validation_timestamp: str


class ModelDeploymentRequest(BaseModel):
    """Request model for model deployment."""
    model_id: str
    endpoint_name: Optional[str] = None


class ModelDeploymentResponse(BaseModel):
    """Response model for model deployment."""
    model_id: str
    endpoint_name: str
    api_endpoint: str
    model_path: str
    deployment_timestamp: str
    status: str


@router.get("/templates", response_model=ModelTemplateResponse)
async def get_model_templates(
    current_user: User = Depends(get_current_user),
    engine: ScrollModelFactory = Depends(get_model_factory_engine)
):
    """Get available model templates."""
    try:
        result = await engine.process(
            input_data=None,
            parameters={"action": "get_templates"}
        )
        return ModelTemplateResponse(**result)
    
    except Exception as e:
        logger.error(f"Error getting model templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/algorithms", response_model=AlgorithmResponse)
async def get_algorithms(
    current_user: User = Depends(get_current_user),
    engine: ScrollModelFactory = Depends(get_model_factory_engine)
):
    """Get available algorithms."""
    try:
        result = await engine.process(
            input_data=None,
            parameters={"action": "get_algorithms"}
        )
        return AlgorithmResponse(**result)
    
    except Exception as e:
        logger.error(f"Error getting algorithms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models", response_model=CustomModelResponse)
async def create_custom_model(
    request: CustomModelRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    engine: ScrollModelFactory = Depends(get_model_factory_engine)
):
    """Create a custom model with user-defined parameters."""
    try:
        # Check permissions
        require_permission(current_user, "model:create")
        
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset
        if dataset.file_path:
            if dataset.file_path.endswith('.csv'):
                data = pd.read_csv(dataset.file_path)
            elif dataset.file_path.endswith('.xlsx'):
                data = pd.read_excel(dataset.file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
        else:
            raise HTTPException(status_code=400, detail="Dataset file not found")
        
        # Prepare parameters
        parameters = {
            "action": "create_model",
            "model_name": request.model_name,
            "algorithm": request.algorithm,
            "template": request.template,
            "target_column": request.target_column,
            "feature_columns": request.feature_columns,
            "custom_params": request.custom_params,
            "validation_strategy": request.validation_strategy,
            "hyperparameter_tuning": request.hyperparameter_tuning
        }
        
        # Create model
        result = await engine.process(input_data=data, parameters=parameters)
        
        # Save to database
        ml_model = MLModel(
            name=result["model_name"],
            algorithm=result["algorithm"],
            dataset_id=request.dataset_id,
            parameters=result["parameters"],
            metrics=result["metrics"],
            feature_columns=result["feature_columns"],
            target_column=result["target_column"],
            model_path=result["model_path"],
            training_duration_seconds=result["training_duration"],
            is_active=True
        )
        
        db.add(ml_model)
        db.commit()
        db.refresh(ml_model)
        
        return CustomModelResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating custom model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/validate", response_model=ModelValidationResponse)
async def validate_model(
    model_id: str,
    request: ModelValidationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    engine: ScrollModelFactory = Depends(get_model_factory_engine)
):
    """Validate a trained model."""
    try:
        # Check permissions
        require_permission(current_user, "model:validate")
        
        # Prepare parameters
        parameters = {
            "action": "validate_model",
            "model_id": model_id,
            "validation_data": request.validation_data
        }
        
        # Validate model
        result = await engine.process(input_data=None, parameters=parameters)
        
        return ModelValidationResponse(**result)
    
    except Exception as e:
        logger.error(f"Error validating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/deploy", response_model=ModelDeploymentResponse)
async def deploy_model(
    model_id: str,
    request: ModelDeploymentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    engine: ScrollModelFactory = Depends(get_model_factory_engine)
):
    """Deploy a model with API endpoint generation."""
    try:
        # Check permissions
        require_permission(current_user, "model:deploy")
        
        # Check if model exists in database
        ml_model = db.query(MLModel).filter(MLModel.id == model_id).first()
        if not ml_model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Prepare parameters
        parameters = {
            "action": "deploy_model",
            "model_id": model_id,
            "endpoint_name": request.endpoint_name
        }
        
        # Deploy model
        result = await engine.process(input_data=None, parameters=parameters)
        
        # Update model in database
        ml_model.api_endpoint = result["api_endpoint"]
        ml_model.is_deployed = True
        db.commit()
        
        return ModelDeploymentResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/predict")
async def predict_with_model(
    model_id: str,
    features: List[float],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    engine: ScrollModelFactory = Depends(get_model_factory_engine)
):
    """Make predictions using a deployed model."""
    try:
        # Check if model exists and is deployed
        ml_model = db.query(MLModel).filter(
            MLModel.id == model_id,
            MLModel.is_deployed == True,
            MLModel.is_active == True
        ).first()
        
        if not ml_model:
            raise HTTPException(status_code=404, detail="Deployed model not found")
        
        # Load model and make prediction
        import joblib
        from pathlib import Path
        
        model_path = Path(ml_model.model_path)
        if not model_path.exists():
            raise HTTPException(status_code=500, detail="Model file not found")
        
        model = joblib.load(model_path)
        
        # Prepare features for prediction
        import numpy as np
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        
        # Get prediction probability if available
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                prediction_proba = model.predict_proba(features_array).tolist()
            except:
                pass
        
        return {
            "model_id": model_id,
            "prediction": prediction.tolist(),
            "prediction_proba": prediction_proba,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_engine_status(
    current_user: User = Depends(get_current_user),
    engine: ScrollModelFactory = Depends(get_model_factory_engine)
):
    """Get ScrollModelFactory engine status."""
    try:
        status = engine.get_status()
        return status
    
    except Exception as e:
        logger.error(f"Error getting engine status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(
    engine: ScrollModelFactory = Depends(get_model_factory_engine)
):
    """Health check endpoint."""
    try:
        is_healthy = await engine.health_check()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }