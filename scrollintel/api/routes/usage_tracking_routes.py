"""
API routes for usage tracking and billing system.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ...engines.usage_tracker import UsageTracker
from ...models.usage_tracking_models import (
    GenerationType, ResourceType, GenerationUsage, UserUsageSummary,
    BudgetAlert, UsageForecast, CostOptimizationRecommendation
)


# Request/Response models
class StartTrackingRequest(BaseModel):
    user_id: str
    generation_type: GenerationType
    model_used: str
    prompt: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class TrackResourceRequest(BaseModel):
    session_id: str
    resource_type: ResourceType
    amount: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EndTrackingRequest(BaseModel):
    session_id: str
    success: bool = True
    quality_score: Optional[float] = None
    error_message: Optional[str] = None


class CostEstimateRequest(BaseModel):
    generation_type: GenerationType
    model_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class UsageSummaryResponse(BaseModel):
    user_id: str
    period_start: datetime
    period_end: datetime
    total_generations: int
    successful_generations: int
    failed_generations: int
    image_generations: int
    video_generations: int
    enhancement_operations: int
    batch_operations: int
    total_gpu_seconds: float
    total_cpu_seconds: float
    total_storage_gb: float
    total_bandwidth_gb: float
    total_api_calls: int
    total_cost: float
    average_cost_per_generation: float
    average_quality_score: Optional[float]
    average_generation_time: float


# Initialize router and tracker
router = APIRouter(prefix="/api/v1/usage", tags=["usage-tracking"])
usage_tracker = UsageTracker()


@router.post("/tracking/start")
async def start_tracking(request: StartTrackingRequest) -> Dict[str, str]:
    """Start tracking a new generation request."""
    try:
        session_id = await usage_tracker.start_generation_tracking(
            user_id=request.user_id,
            generation_type=request.generation_type,
            model_used=request.model_used,
            prompt=request.prompt,
            parameters=request.parameters
        )
        
        return {
            "session_id": session_id,
            "status": "tracking_started",
            "message": f"Started tracking generation for user {request.user_id}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start tracking: {str(e)}")


@router.post("/tracking/resource")
async def track_resource(request: TrackResourceRequest) -> Dict[str, str]:
    """Track resource usage for an active generation session."""
    try:
        await usage_tracker.track_resource_usage(
            session_id=request.session_id,
            resource_type=request.resource_type,
            amount=request.amount,
            metadata=request.metadata
        )
        
        return {
            "status": "resource_tracked",
            "message": f"Tracked {request.amount} {request.resource_type.value}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track resource: {str(e)}")


@router.post("/tracking/end")
async def end_tracking(request: EndTrackingRequest) -> Dict[str, Any]:
    """End tracking for a generation session."""
    try:
        session = await usage_tracker.end_generation_tracking(
            session_id=request.session_id,
            success=request.success,
            quality_score=request.quality_score,
            error_message=request.error_message
        )
        
        return {
            "session_id": session.id,
            "status": "tracking_completed",
            "duration_seconds": session.duration_seconds,
            "total_cost": session.total_cost,
            "success": session.success,
            "quality_score": session.quality_score
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end tracking: {str(e)}")


@router.get("/summary/{user_id}")
async def get_usage_summary(
    user_id: str,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
) -> UsageSummaryResponse:
    """Get usage summary for a user."""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        summary = await usage_tracker.get_user_usage_summary(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return UsageSummaryResponse(
            user_id=summary.user_id,
            period_start=summary.period_start,
            period_end=summary.period_end,
            total_generations=summary.total_generations,
            successful_generations=summary.successful_generations,
            failed_generations=summary.failed_generations,
            image_generations=summary.image_generations,
            video_generations=summary.video_generations,
            enhancement_operations=summary.enhancement_operations,
            batch_operations=summary.batch_operations,
            total_gpu_seconds=summary.total_gpu_seconds,
            total_cpu_seconds=summary.total_cpu_seconds,
            total_storage_gb=summary.total_storage_gb,
            total_bandwidth_gb=summary.total_bandwidth_gb,
            total_api_calls=summary.total_api_calls,
            total_cost=summary.total_cost,
            average_cost_per_generation=summary.average_cost_per_generation,
            average_quality_score=summary.average_quality_score,
            average_generation_time=summary.average_generation_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get usage summary: {str(e)}")


@router.get("/budget-alerts/{user_id}")
async def get_budget_alerts(
    user_id: str,
    budget_limit: float = Query(..., gt=0, description="Budget limit in USD"),
    period_days: int = Query(30, ge=1, le=365, description="Budget period in days")
) -> List[Dict[str, Any]]:
    """Get budget alerts for a user."""
    try:
        alerts = await usage_tracker.check_budget_alerts(
            user_id=user_id,
            budget_limit=budget_limit,
            period_days=period_days
        )
        
        return [
            {
                "id": alert.id,
                "alert_type": alert.alert_type,
                "threshold_percentage": alert.threshold_percentage,
                "current_usage": alert.current_usage,
                "budget_limit": alert.budget_limit,
                "usage_percentage": (alert.current_usage / alert.budget_limit) * 100,
                "triggered_at": alert.triggered_at,
                "acknowledged": alert.acknowledged
            }
            for alert in alerts
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get budget alerts: {str(e)}")


@router.get("/forecast/{user_id}")
async def get_usage_forecast(
    user_id: str,
    forecast_days: int = Query(30, ge=1, le=90, description="Days to forecast"),
    historical_days: int = Query(90, ge=7, le=365, description="Historical data period")
) -> Dict[str, Any]:
    """Get usage forecast for a user."""
    try:
        forecast = await usage_tracker.generate_usage_forecast(
            user_id=user_id,
            forecast_days=forecast_days,
            historical_days=historical_days
        )
        
        return {
            "user_id": forecast.user_id,
            "forecast_period_days": forecast.forecast_period_days,
            "predicted_usage": forecast.predicted_usage,
            "predicted_cost": forecast.predicted_cost,
            "confidence_interval": {
                "lower": forecast.confidence_interval[0],
                "upper": forecast.confidence_interval[1]
            },
            "usage_trend": forecast.usage_trend,
            "seasonal_pattern": forecast.seasonal_pattern,
            "generated_at": forecast.generated_at,
            "historical_data_points": len(forecast.historical_usage)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate forecast: {str(e)}")


@router.get("/recommendations/{user_id}")
async def get_cost_optimization_recommendations(
    user_id: str,
    analysis_days: int = Query(30, ge=7, le=90, description="Days to analyze for recommendations")
) -> List[Dict[str, Any]]:
    """Get cost optimization recommendations for a user."""
    try:
        recommendations = await usage_tracker.generate_cost_optimization_recommendations(
            user_id=user_id,
            analysis_days=analysis_days
        )
        
        return [
            {
                "id": rec.id,
                "recommendation_type": rec.recommendation_type,
                "title": rec.title,
                "description": rec.description,
                "potential_savings": rec.potential_savings,
                "implementation_effort": rec.implementation_effort,
                "priority": rec.priority,
                "current_cost": rec.current_cost,
                "optimized_cost": rec.optimized_cost,
                "savings_percentage": ((rec.current_cost - rec.optimized_cost) / rec.current_cost * 100) if rec.current_cost > 0 else 0,
                "affected_operations": rec.affected_operations,
                "created_at": rec.created_at,
                "implemented": rec.implemented
            }
            for rec in recommendations
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.post("/cost-estimate")
async def get_cost_estimate(request: CostEstimateRequest) -> Dict[str, Any]:
    """Get real-time cost estimate for a generation request."""
    try:
        estimate = await usage_tracker.get_real_time_cost_calculation(
            generation_type=request.generation_type,
            model_name=request.model_name,
            parameters=request.parameters
        )
        
        return {
            "generation_type": request.generation_type.value,
            "model_name": request.model_name,
            "base_cost": estimate["base_cost"],
            "multiplier": estimate["multiplier"],
            "estimated_cost": estimate["estimated_cost"],
            "currency": estimate["currency"],
            "parameters_considered": list(request.parameters.keys())
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate cost estimate: {str(e)}")


@router.get("/analytics/{user_id}")
async def get_usage_analytics(
    user_id: str,
    days: int = Query(30, ge=1, le=365, description="Analysis period in days")
) -> Dict[str, Any]:
    """Get comprehensive usage analytics for a user."""
    try:
        analytics = await usage_tracker.get_usage_analytics(
            user_id=user_id,
            days=days
        )
        
        return {
            "user_id": user_id,
            "analysis_period": analytics["period"],
            "usage_summary": {
                "total_generations": analytics["summary"].total_generations,
                "success_rate": (analytics["summary"].successful_generations / analytics["summary"].total_generations * 100) if analytics["summary"].total_generations > 0 else 0,
                "total_cost": analytics["summary"].total_cost,
                "average_cost_per_generation": analytics["summary"].average_cost_per_generation,
                "average_quality_score": analytics["summary"].average_quality_score,
                "average_generation_time": analytics["summary"].average_generation_time
            },
            "forecast": {
                "predicted_cost": analytics["forecast"].predicted_cost,
                "usage_trend": analytics["forecast"].usage_trend,
                "confidence_interval": analytics["forecast"].confidence_interval
            },
            "recommendations_count": len(analytics["recommendations"]),
            "top_recommendations": [
                {
                    "title": rec.title,
                    "potential_savings": rec.potential_savings,
                    "priority": rec.priority
                }
                for rec in sorted(analytics["recommendations"], key=lambda x: x.potential_savings, reverse=True)[:3]
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for usage tracking service."""
    return {
        "status": "healthy",
        "service": "usage_tracking",
        "timestamp": datetime.utcnow().isoformat()
    }