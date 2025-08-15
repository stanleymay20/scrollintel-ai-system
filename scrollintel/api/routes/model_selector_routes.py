"""
API routes for Model Selector functionality.

Provides endpoints for model selection, performance tracking, A/B testing,
and analytics.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from scrollintel.core.model_selector import (
    ModelSelector,
    ModelCapabilities,
    GenerationRequest,
    ModelSelection,
    ModelType,
    QualityMetric,
    initialize_default_models
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/model-selector", tags=["Model Selector"])

# Global model selector instance
model_selector: Optional[ModelSelector] = None


# Pydantic models for API

class ModelCapabilitiesModel(BaseModel):
    """Model capabilities for API."""
    model_id: str = Field(..., description="Unique model identifier")
    model_type: str = Field(..., description="Type of model (image_generation, video_generation, etc.)")
    supported_resolutions: List[List[int]] = Field(..., description="Supported resolutions [[width, height], ...]")
    supported_formats: List[str] = Field(..., description="Supported output formats")
    max_prompt_length: int = Field(..., gt=0, description="Maximum prompt length")
    supports_negative_prompts: bool = Field(default=False, description="Whether model supports negative prompts")
    supports_style_control: bool = Field(default=False, description="Whether model supports style control")
    supports_batch_processing: bool = Field(default=False, description="Whether model supports batch processing")
    gpu_memory_required: float = Field(default=4.0, gt=0, description="GPU memory required in GB")
    estimated_processing_time: float = Field(default=30.0, gt=0, description="Estimated processing time in seconds")
    cost_per_generation: float = Field(default=0.10, ge=0, description="Cost per generation in USD")


class GenerationRequestModel(BaseModel):
    """Generation request for API."""
    request_id: str = Field(..., description="Unique request identifier")
    model_type: str = Field(..., description="Type of model needed")
    prompt: str = Field(..., min_length=1, description="Generation prompt")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    resolution: List[int] = Field(default=[1024, 1024], description="Desired resolution [width, height]")
    format: str = Field(default="jpg", description="Output format")
    style: Optional[str] = Field(None, description="Style preference")
    quality_preference: str = Field(default="balanced", description="Quality preference (speed, balanced, quality)")
    budget_limit: Optional[float] = Field(None, ge=0, description="Budget limit in USD")
    user_id: Optional[str] = Field(None, description="User identifier")
    priority: str = Field(default="normal", description="Request priority")


class ModelSelectionResponse(BaseModel):
    """Model selection response."""
    selected_model: str
    confidence_score: float
    estimated_cost: float
    estimated_time: float
    estimated_quality: float
    selection_reason: str
    alternative_models: List[str]


class MetricsUpdateModel(BaseModel):
    """Model for updating performance metrics."""
    model_id: str = Field(..., description="Model identifier")
    processing_time: float = Field(..., gt=0, description="Actual processing time in seconds")
    quality_score: float = Field(..., ge=0, le=1, description="Quality score (0-1)")
    cost: float = Field(..., ge=0, description="Actual cost in USD")
    success: bool = Field(..., description="Whether generation was successful")
    user_satisfaction: Optional[float] = Field(None, ge=0, le=1, description="User satisfaction score (0-1)")


class ABTestCreateModel(BaseModel):
    """Model for creating A/B test."""
    test_id: str = Field(..., description="Unique test identifier")
    model_a: str = Field(..., description="First model to test")
    model_b: str = Field(..., description="Second model to test")
    traffic_split: float = Field(default=0.5, ge=0, le=1, description="Traffic split for model A")


class ModelRankingResponse(BaseModel):
    """Model ranking response."""
    model_id: str
    score: float


# Dependency functions

async def get_model_selector() -> ModelSelector:
    """Get the model selector instance."""
    global model_selector
    if model_selector is None:
        model_selector = await initialize_default_models()
    return model_selector


def parse_model_type(model_type_str: str) -> ModelType:
    """Parse model type string to ModelType enum."""
    try:
        return ModelType(model_type_str.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type_str}")


def parse_quality_metric(metric_str: str) -> QualityMetric:
    """Parse quality metric string to QualityMetric enum."""
    try:
        return QualityMetric(metric_str.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid quality metric: {metric_str}")


# API Routes

@router.on_event("startup")
async def startup_model_selector():
    """Initialize model selector on startup."""
    global model_selector
    model_selector = await initialize_default_models()
    logger.info("Model selector initialized with default models")


@router.post("/models", response_model=Dict[str, str])
async def register_model(
    model_caps: ModelCapabilitiesModel,
    selector: ModelSelector = Depends(get_model_selector)
) -> Dict[str, str]:
    """Register a new model with its capabilities."""
    try:
        # Convert API model to internal model
        model_type = parse_model_type(model_caps.model_type)
        
        capabilities = ModelCapabilities(
            model_id=model_caps.model_id,
            model_type=model_type,
            supported_resolutions=[tuple(res) for res in model_caps.supported_resolutions],
            supported_formats=model_caps.supported_formats,
            max_prompt_length=model_caps.max_prompt_length,
            supports_negative_prompts=model_caps.supports_negative_prompts,
            supports_style_control=model_caps.supports_style_control,
            supports_batch_processing=model_caps.supports_batch_processing,
            gpu_memory_required=model_caps.gpu_memory_required,
            estimated_processing_time=model_caps.estimated_processing_time,
            cost_per_generation=model_caps.cost_per_generation
        )
        
        await selector.register_model(capabilities)
        
        return {"model_id": model_caps.model_id, "status": "registered"}
        
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select", response_model=ModelSelectionResponse)
async def select_model(
    request: GenerationRequestModel,
    strategy: str = "performance",
    selector: ModelSelector = Depends(get_model_selector)
) -> ModelSelectionResponse:
    """Select the best model for a generation request."""
    try:
        # Convert API model to internal model
        model_type = parse_model_type(request.model_type)
        
        generation_request = GenerationRequest(
            request_id=request.request_id,
            model_type=model_type,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            resolution=tuple(request.resolution),
            format=request.format,
            style=request.style,
            quality_preference=request.quality_preference,
            budget_limit=request.budget_limit,
            user_id=request.user_id,
            priority=request.priority
        )
        
        selection = await selector.select_model(generation_request, strategy)
        
        return ModelSelectionResponse(
            selected_model=selection.selected_model,
            confidence_score=selection.confidence_score,
            estimated_cost=selection.estimated_cost,
            estimated_time=selection.estimated_time,
            estimated_quality=selection.estimated_quality,
            selection_reason=selection.selection_reason,
            alternative_models=selection.alternative_models
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error selecting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/update")
async def update_metrics(
    metrics: MetricsUpdateModel,
    selector: ModelSelector = Depends(get_model_selector)
) -> Dict[str, str]:
    """Update model performance metrics."""
    try:
        await selector.update_model_metrics(
            model_id=metrics.model_id,
            processing_time=metrics.processing_time,
            quality_score=metrics.quality_score,
            cost=metrics.cost,
            success=metrics.success,
            user_satisfaction=metrics.user_satisfaction
        )
        
        return {"model_id": metrics.model_id, "status": "metrics_updated"}
        
    except Exception as e:
        logger.error(f"Error updating metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models(
    selector: ModelSelector = Depends(get_model_selector)
) -> Dict[str, Any]:
    """List all registered models with their capabilities."""
    try:
        models = {}
        for model_id, capabilities in selector.model_capabilities.items():
            metrics = selector.model_metrics.get(model_id)
            
            models[model_id] = {
                "capabilities": {
                    "model_type": capabilities.model_type.value,
                    "supported_resolutions": capabilities.supported_resolutions,
                    "supported_formats": capabilities.supported_formats,
                    "max_prompt_length": capabilities.max_prompt_length,
                    "supports_negative_prompts": capabilities.supports_negative_prompts,
                    "supports_style_control": capabilities.supports_style_control,
                    "supports_batch_processing": capabilities.supports_batch_processing,
                    "gpu_memory_required": capabilities.gpu_memory_required,
                    "estimated_processing_time": capabilities.estimated_processing_time,
                    "cost_per_generation": capabilities.cost_per_generation
                },
                "metrics": {
                    "total_generations": metrics.total_generations if metrics else 0,
                    "success_rate": metrics.success_rate if metrics else 0.0,
                    "average_quality_score": metrics.average_quality_score if metrics else 0.0,
                    "average_processing_time": metrics.average_processing_time if metrics else 0.0,
                    "average_cost": metrics.average_cost if metrics else 0.0,
                    "user_satisfaction_score": metrics.user_satisfaction_score if metrics else 0.0,
                    "last_updated": metrics.last_updated.isoformat() if metrics else None
                } if metrics else None
            }
        
        return {"models": models, "total_count": len(models)}
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rankings/{model_type}", response_model=List[ModelRankingResponse])
async def get_model_rankings(
    model_type: str,
    metric: str = "overall_score",
    selector: ModelSelector = Depends(get_model_selector)
) -> List[ModelRankingResponse]:
    """Get model rankings for a specific type and metric."""
    try:
        model_type_enum = parse_model_type(model_type)
        metric_enum = parse_quality_metric(metric)
        
        rankings = await selector.get_model_rankings(model_type_enum, metric_enum)
        
        return [
            ModelRankingResponse(model_id=model_id, score=score)
            for model_id, score in rankings
        ]
        
    except Exception as e:
        logger.error(f"Error getting rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-tests", response_model=Dict[str, str])
async def create_ab_test(
    test_config: ABTestCreateModel,
    selector: ModelSelector = Depends(get_model_selector)
) -> Dict[str, str]:
    """Create a new A/B test between two models."""
    try:
        success = await selector.create_ab_test(
            test_id=test_config.test_id,
            model_a=test_config.model_a,
            model_b=test_config.model_b,
            traffic_split=test_config.traffic_split
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Test ID already exists")
        
        return {"test_id": test_config.test_id, "status": "created"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-tests/{test_id}")
async def get_ab_test_status(
    test_id: str,
    selector: ModelSelector = Depends(get_model_selector)
) -> Dict[str, Any]:
    """Get A/B test status and results."""
    try:
        status = await selector.get_ab_test_status(test_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Test not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting A/B test status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics")
async def get_selection_analytics(
    selector: ModelSelector = Depends(get_model_selector)
) -> Dict[str, Any]:
    """Get analytics on model selection patterns."""
    try:
        analytics = await selector.get_selection_analytics()
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def list_selection_strategies() -> Dict[str, List[str]]:
    """List available selection strategies."""
    return {
        "strategies": ["performance", "cost", "quality"],
        "descriptions": {
            "performance": "Select based on historical performance metrics",
            "cost": "Select most cost-effective model above quality threshold",
            "quality": "Select highest quality model regardless of cost"
        }
    }


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.delete("/models/{model_id}")
async def unregister_model(
    model_id: str,
    selector: ModelSelector = Depends(get_model_selector)
) -> Dict[str, str]:
    """Unregister a model (remove from capabilities)."""
    try:
        if model_id not in selector.model_capabilities:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Remove from capabilities and metrics
        del selector.model_capabilities[model_id]
        if model_id in selector.model_metrics:
            del selector.model_metrics[model_id]
        
        return {"model_id": model_id, "status": "unregistered"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/metrics")
async def get_model_metrics(
    model_id: str,
    selector: ModelSelector = Depends(get_model_selector)
) -> Dict[str, Any]:
    """Get detailed metrics for a specific model."""
    try:
        if model_id not in selector.model_capabilities:
            raise HTTPException(status_code=404, detail="Model not found")
        
        metrics = selector.model_metrics.get(model_id)
        if not metrics:
            return {"model_id": model_id, "metrics": None}
        
        return {
            "model_id": model_id,
            "metrics": {
                "total_generations": metrics.total_generations,
                "successful_generations": metrics.successful_generations,
                "failed_generations": metrics.failed_generations,
                "success_rate": metrics.success_rate,
                "average_processing_time": metrics.average_processing_time,
                "average_quality_score": metrics.average_quality_score,
                "average_cost": metrics.average_cost,
                "user_satisfaction_score": metrics.user_satisfaction_score,
                "efficiency_score": metrics.efficiency_score,
                "cost_effectiveness": metrics.cost_effectiveness,
                "last_updated": metrics.last_updated.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))