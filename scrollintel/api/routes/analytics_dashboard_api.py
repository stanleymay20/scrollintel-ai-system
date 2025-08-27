"""
Comprehensive REST API for Advanced Analytics Dashboard System.
Provides complete API access to all dashboard and analytics functionality.
"""
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import asyncio
import json
from io import BytesIO

from ...core.dashboard_manager import DashboardManager
from ...engines.roi_calculator import ROICalculator
from ...engines.insight_generator import InsightGenerator
from ...engines.predictive_engine import PredictiveEngine
from ...core.data_connector import DataConnector
from ...core.template_engine import TemplateEngine
from ...core.websocket_manager import websocket_manager
from ...models.dashboard_models import ExecutiveRole, DashboardType
from ...models.roi_models import ROIAnalysis
from ...models.insight_models import Insight
from ...models.predictive_models import Forecast
from ...security.auth import get_current_user, verify_api_key
from ...core.rate_limiter import rate_limit


router = APIRouter(prefix="/api/v1/analytics", tags=["analytics-dashboard"])


# Request/Response Models
class DashboardCreateRequest(BaseModel):
    name: str = Field(..., description="Dashboard name")
    role: str = Field(..., description="Executive role (CTO, CFO, CEO, etc.)")
    template_id: Optional[str] = Field(None, description="Template ID to use")
    config: Dict[str, Any] = Field(default_factory=dict, description="Dashboard configuration")
    widgets: List[Dict[str, Any]] = Field(default_factory=list, description="Widget configurations")


class DashboardUpdateRequest(BaseModel):
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    widgets: Optional[List[Dict[str, Any]]] = None


class MetricRequest(BaseModel):
    name: str
    category: str
    value: float
    unit: str
    source: str
    context: Dict[str, Any] = Field(default_factory=dict)


class ROICalculationRequest(BaseModel):
    project_id: str
    costs: List[Dict[str, Any]]
    benefits: List[Dict[str, Any]]
    time_period: Dict[str, Any]


class InsightGenerationRequest(BaseModel):
    data_sources: List[str]
    analysis_type: str = "comprehensive"
    focus_areas: List[str] = Field(default_factory=list)


class PredictionRequest(BaseModel):
    metric_name: str
    historical_data: List[Dict[str, Any]]
    forecast_horizon: int = 30
    confidence_level: float = 0.95


class DataSourceConnectionRequest(BaseModel):
    source_type: str
    connection_config: Dict[str, Any]
    credentials: Dict[str, Any]


class TemplateRequest(BaseModel):
    name: str
    category: str
    role: str
    description: str
    config: Dict[str, Any]


# Initialize services
dashboard_manager = DashboardManager()
roi_calculator = ROICalculator()
insight_generator = InsightGenerator()
predictive_engine = PredictiveEngine()
data_connector = DataConnector()
template_engine = TemplateEngine()


# Dashboard Management Endpoints
@router.post("/dashboards", response_model=Dict[str, Any])
@rate_limit(requests_per_minute=30)
async def create_dashboard(
    request: DashboardCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create a new analytics dashboard."""
    try:
        user_id = current_user["id"]
        
        # Create dashboard
        dashboard = await dashboard_manager.create_executive_dashboard_async(
            role=ExecutiveRole(request.role),
            config=request.config,
            user_id=user_id,
            name=request.name,
            template_id=request.template_id
        )
        
        # Add widgets if provided
        if request.widgets:
            for widget_config in request.widgets:
                await dashboard_manager.add_widget_async(dashboard.id, widget_config)
        
        # Schedule initial data population
        background_tasks.add_task(
            dashboard_manager.populate_initial_data,
            dashboard.id
        )
        
        return {
            "id": dashboard.id,
            "name": dashboard.name,
            "role": dashboard.role,
            "status": "created",
            "created_at": dashboard.created_at.isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dashboard: {str(e)}")


@router.get("/dashboards", response_model=List[Dict[str, Any]])
@rate_limit(requests_per_minute=60)
async def list_dashboards(
    role: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """List user's dashboards with filtering and pagination."""
    try:
        user_id = current_user["id"]
        
        dashboards = await dashboard_manager.get_dashboards_paginated(
            user_id=user_id,
            role=role,
            category=category,
            limit=limit,
            offset=offset
        )
        
        return [
            {
                "id": dashboard.id,
                "name": dashboard.name,
                "role": dashboard.role,
                "type": dashboard.type,
                "created_at": dashboard.created_at.isoformat(),
                "updated_at": dashboard.updated_at.isoformat(),
                "widget_count": len(dashboard.widgets),
                "last_accessed": dashboard.last_accessed.isoformat() if dashboard.last_accessed else None
            }
            for dashboard in dashboards
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboards/{dashboard_id}", response_model=Dict[str, Any])
@rate_limit(requests_per_minute=120)
async def get_dashboard(
    dashboard_id: str,
    include_data: bool = Query(True),
    time_range: Optional[str] = Query(None, description="ISO format: start/end"),
    current_user: dict = Depends(get_current_user)
):
    """Get dashboard details with optional data."""
    try:
        dashboard_data = await dashboard_manager.get_dashboard_with_data_async(
            dashboard_id=dashboard_id,
            include_data=include_data,
            time_range=time_range
        )
        
        if not dashboard_data:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        return dashboard_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/dashboards/{dashboard_id}", response_model=Dict[str, Any])
@rate_limit(requests_per_minute=30)
async def update_dashboard(
    dashboard_id: str,
    request: DashboardUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update dashboard configuration."""
    try:
        success = await dashboard_manager.update_dashboard_async(
            dashboard_id=dashboard_id,
            updates=request.dict(exclude_unset=True)
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        return {"status": "updated", "dashboard_id": dashboard_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/dashboards/{dashboard_id}")
@rate_limit(requests_per_minute=10)
async def delete_dashboard(
    dashboard_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a dashboard."""
    try:
        success = await dashboard_manager.delete_dashboard_async(dashboard_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        return {"status": "deleted", "dashboard_id": dashboard_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Metrics Management
@router.post("/dashboards/{dashboard_id}/metrics")
@rate_limit(requests_per_minute=100)
async def add_metrics(
    dashboard_id: str,
    metrics: List[MetricRequest],
    current_user: dict = Depends(get_current_user)
):
    """Add metrics to dashboard."""
    try:
        success = await dashboard_manager.add_metrics_async(dashboard_id, metrics)
        
        if not success:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        # Trigger real-time updates
        await websocket_manager.broadcast_metric_update(dashboard_id, metrics)
        
        return {"status": "metrics_added", "count": len(metrics)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboards/{dashboard_id}/metrics")
@rate_limit(requests_per_minute=120)
async def get_metrics(
    dashboard_id: str,
    category: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """Get dashboard metrics with filtering."""
    try:
        metrics = await dashboard_manager.get_metrics_async(
            dashboard_id=dashboard_id,
            category=category,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "dashboard_id": dashboard_id,
            "metrics": metrics,
            "count": len(metrics),
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ROI Analysis Endpoints
@router.post("/roi/calculate", response_model=Dict[str, Any])
@rate_limit(requests_per_minute=20)
async def calculate_roi(
    request: ROICalculationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Calculate ROI for a project."""
    try:
        roi_analysis = await roi_calculator.calculate_roi_async(
            project_id=request.project_id,
            costs=request.costs,
            benefits=request.benefits,
            time_period=request.time_period
        )
        
        return {
            "project_id": request.project_id,
            "roi_percentage": roi_analysis.roi_percentage,
            "total_investment": roi_analysis.total_investment,
            "total_benefits": roi_analysis.total_benefits,
            "payback_period": roi_analysis.payback_period,
            "npv": roi_analysis.npv,
            "irr": roi_analysis.irr,
            "analysis_date": roi_analysis.analysis_date.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roi/projects/{project_id}")
@rate_limit(requests_per_minute=60)
async def get_project_roi(
    project_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get ROI analysis for a specific project."""
    try:
        roi_data = await roi_calculator.get_project_roi_async(project_id)
        
        if not roi_data:
            raise HTTPException(status_code=404, detail="Project ROI data not found")
        
        return roi_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Insight Generation Endpoints
@router.post("/insights/generate", response_model=Dict[str, Any])
@rate_limit(requests_per_minute=10)
async def generate_insights(
    request: InsightGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Generate AI insights from data sources."""
    try:
        # Start insight generation in background
        task_id = await insight_generator.start_analysis_async(
            data_sources=request.data_sources,
            analysis_type=request.analysis_type,
            focus_areas=request.focus_areas
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/{task_id}")
@rate_limit(requests_per_minute=60)
async def get_insights(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get generated insights by task ID."""
    try:
        insights = await insight_generator.get_insights_async(task_id)
        
        if not insights:
            raise HTTPException(status_code=404, detail="Insights not found")
        
        return {
            "task_id": task_id,
            "insights": insights,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Predictive Analytics Endpoints
@router.post("/predictions/forecast", response_model=Dict[str, Any])
@rate_limit(requests_per_minute=20)
async def create_forecast(
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create predictive forecast for metrics."""
    try:
        forecast = await predictive_engine.create_forecast_async(
            metric_name=request.metric_name,
            historical_data=request.historical_data,
            horizon=request.forecast_horizon,
            confidence_level=request.confidence_level
        )
        
        return {
            "metric_name": request.metric_name,
            "forecast_horizon": request.forecast_horizon,
            "predictions": forecast.predictions,
            "confidence_intervals": forecast.confidence_intervals,
            "model_accuracy": forecast.model_accuracy,
            "created_at": forecast.created_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/{metric_name}")
@rate_limit(requests_per_minute=60)
async def get_predictions(
    metric_name: str,
    days: int = Query(30, le=365),
    current_user: dict = Depends(get_current_user)
):
    """Get existing predictions for a metric."""
    try:
        predictions = await predictive_engine.get_predictions_async(metric_name, days)
        
        return {
            "metric_name": metric_name,
            "predictions": predictions,
            "forecast_period": days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Data Source Management
@router.post("/data-sources/connect")
@rate_limit(requests_per_minute=10)
async def connect_data_source(
    request: DataSourceConnectionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Connect a new data source."""
    try:
        connection_id = await data_connector.connect_source_async(
            source_type=request.source_type,
            config=request.connection_config,
            credentials=request.credentials
        )
        
        return {
            "connection_id": connection_id,
            "source_type": request.source_type,
            "status": "connected"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-sources")
@rate_limit(requests_per_minute=60)
async def list_data_sources(
    current_user: dict = Depends(get_current_user)
):
    """List connected data sources."""
    try:
        sources = await data_connector.list_sources_async()
        
        return {
            "sources": sources,
            "count": len(sources)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/data-sources/{connection_id}")
@rate_limit(requests_per_minute=10)
async def disconnect_data_source(
    connection_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Disconnect a data source."""
    try:
        success = await data_connector.disconnect_source_async(connection_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        return {"status": "disconnected", "connection_id": connection_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Template Management
@router.post("/templates")
@rate_limit(requests_per_minute=10)
async def create_template(
    request: TemplateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new dashboard template."""
    try:
        template_id = await template_engine.create_template_async(
            name=request.name,
            category=request.category,
            role=request.role,
            description=request.description,
            config=request.config,
            creator_id=current_user["id"]
        )
        
        return {
            "template_id": template_id,
            "name": request.name,
            "status": "created"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
@rate_limit(requests_per_minute=60)
async def list_templates(
    role: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """List available dashboard templates."""
    try:
        templates = await template_engine.list_templates_async(
            role=role,
            category=category
        )
        
        return {
            "templates": templates,
            "count": len(templates)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Export and Reporting
@router.get("/dashboards/{dashboard_id}/export")
@rate_limit(requests_per_minute=5)
async def export_dashboard(
    dashboard_id: str,
    format: str = Query("pdf", regex="^(pdf|excel|json)$"),
    current_user: dict = Depends(get_current_user)
):
    """Export dashboard data in various formats."""
    try:
        export_data = await dashboard_manager.export_dashboard_async(
            dashboard_id=dashboard_id,
            format=format
        )
        
        if format == "json":
            return JSONResponse(content=export_data)
        else:
            # Return file stream for PDF/Excel
            return StreamingResponse(
                BytesIO(export_data),
                media_type=f"application/{format}",
                headers={"Content-Disposition": f"attachment; filename=dashboard_{dashboard_id}.{format}"}
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Real-time Updates
@router.get("/dashboards/{dashboard_id}/stream")
async def stream_dashboard_updates(
    dashboard_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Stream real-time dashboard updates."""
    try:
        async def event_stream():
            async for update in dashboard_manager.stream_updates(dashboard_id):
                yield f"data: {json.dumps(update)}\n\n"
        
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health and Status
@router.get("/health")
async def health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "dashboard_manager": "operational",
            "roi_calculator": "operational",
            "insight_generator": "operational",
            "predictive_engine": "operational",
            "data_connector": "operational"
        }
    }


@router.get("/stats")
@rate_limit(requests_per_minute=30)
async def get_api_stats(
    current_user: dict = Depends(get_current_user)
):
    """Get API usage statistics."""
    try:
        stats = await dashboard_manager.get_usage_stats_async(current_user["id"])
        
        return {
            "user_id": current_user["id"],
            "dashboards_created": stats.get("dashboards_created", 0),
            "api_calls_today": stats.get("api_calls_today", 0),
            "data_sources_connected": stats.get("data_sources_connected", 0),
            "insights_generated": stats.get("insights_generated", 0),
            "last_activity": stats.get("last_activity")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))