"""
API routes for Advanced Analytics Dashboard System.
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from ...core.dashboard_manager import DashboardManager, DashboardConfig, SharePermissions, TimeRange
from ...core.dashboard_templates import DashboardTemplates
from ...core.websocket_manager import websocket_manager
from ...models.dashboard_models import ExecutiveRole, DashboardType, BusinessMetric
from ...security.auth import get_current_user


router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


# Pydantic models for request/response
class CreateDashboardRequest(BaseModel):
    name: str
    role: str
    config: Dict[str, Any] = {}
    template_id: Optional[str] = None


class UpdateDashboardRequest(BaseModel):
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class ShareDashboardRequest(BaseModel):
    user_ids: List[str]
    permission_type: str = "view"
    expires_in_days: Optional[int] = None


class MetricUpdateRequest(BaseModel):
    metrics: List[Dict[str, Any]]


class DashboardResponse(BaseModel):
    id: str
    name: str
    type: str
    role: Optional[str]
    owner_id: str
    config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DashboardDataResponse(BaseModel):
    dashboard: DashboardResponse
    widgets_data: Dict[str, Any]
    metrics: List[Dict[str, Any]]
    last_updated: datetime


# Initialize dashboard manager
dashboard_manager = DashboardManager()


@router.post("/create", response_model=DashboardResponse)
async def create_dashboard(
    request: CreateDashboardRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new executive dashboard."""
    try:
        user_id = current_user["id"]
        
        # Create dashboard config
        config = DashboardConfig(
            layout=request.config.get("layout", {}),
            theme=request.config.get("theme", "default"),
            auto_refresh=request.config.get("auto_refresh", True),
            refresh_interval=request.config.get("refresh_interval", 300)
        )
        
        if request.template_id:
            # Create from template
            dashboard = dashboard_manager.create_dashboard_from_template(
                request.template_id, user_id, request.name
            )
        else:
            # Create executive dashboard
            try:
                role = ExecutiveRole(request.role)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}")
            
            dashboard = dashboard_manager.create_executive_dashboard(
                role, config, user_id, request.name
            )
        
        return DashboardResponse(
            id=dashboard.id,
            name=dashboard.name,
            type=dashboard.type,
            role=dashboard.role,
            owner_id=dashboard.owner_id,
            config=dashboard.config,
            created_at=dashboard.created_at,
            updated_at=dashboard.updated_at
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=List[DashboardResponse])
async def list_dashboards(
    role: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """Get all dashboards accessible to the current user."""
    try:
        user_id = current_user["id"]
        dashboards = dashboard_manager.get_dashboards_for_user(user_id, role)
        
        return [
            DashboardResponse(
                id=dashboard.id,
                name=dashboard.name,
                type=dashboard.type,
                role=dashboard.role,
                owner_id=dashboard.owner_id,
                config=dashboard.config,
                created_at=dashboard.created_at,
                updated_at=dashboard.updated_at
            )
            for dashboard in dashboards
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dashboard_id}/data", response_model=DashboardDataResponse)
async def get_dashboard_data(
    dashboard_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """Get dashboard data with widgets and metrics."""
    try:
        # Create time range if dates provided
        time_range = None
        if start_date and end_date:
            time_range = TimeRange(start_date, end_date)
        elif start_date:
            time_range = TimeRange(start_date, datetime.utcnow())
        
        dashboard_data = dashboard_manager.get_dashboard_data(dashboard_id, time_range)
        
        if not dashboard_data:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        # Convert metrics to dict format
        metrics_data = [
            {
                "id": metric.id,
                "name": metric.name,
                "category": metric.category,
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp,
                "source": metric.source,
                "context": metric.context
            }
            for metric in dashboard_data.metrics
        ]
        
        return DashboardDataResponse(
            dashboard=DashboardResponse(
                id=dashboard_data.dashboard.id,
                name=dashboard_data.dashboard.name,
                type=dashboard_data.dashboard.type,
                role=dashboard_data.dashboard.role,
                owner_id=dashboard_data.dashboard.owner_id,
                config=dashboard_data.dashboard.config,
                created_at=dashboard_data.dashboard.created_at,
                updated_at=dashboard_data.dashboard.updated_at
            ),
            widgets_data=dashboard_data.widgets_data,
            metrics=metrics_data,
            last_updated=dashboard_data.last_updated
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{dashboard_id}/metrics")
async def update_dashboard_metrics(
    dashboard_id: str,
    request: MetricUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update dashboard metrics and trigger real-time updates."""
    try:
        success = dashboard_manager.update_dashboard_metrics(dashboard_id, request.metrics)
        
        if not success:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        # Trigger real-time update via WebSocket
        metrics = [BusinessMetric(**metric_data) for metric_data in request.metrics]
        await websocket_manager.send_metric_update(dashboard_id, metrics)
        
        return {"status": "success", "message": "Metrics updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dashboard_id}/share")
async def share_dashboard(
    dashboard_id: str,
    request: ShareDashboardRequest,
    current_user: dict = Depends(get_current_user)
):
    """Share dashboard with other users."""
    try:
        user_id = current_user["id"]
        
        permissions = SharePermissions(
            users=request.user_ids,
            permission_type=request.permission_type,
            expires_in_days=request.expires_in_days
        )
        
        share_link = dashboard_manager.share_dashboard(dashboard_id, permissions, user_id)
        
        return {
            "share_url": share_link.url,
            "expires_at": share_link.expires_at.isoformat() if share_link.expires_at else None,
            "shared_with": request.user_ids
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_dashboard_templates(
    role: Optional[str] = Query(None),
    category: Optional[str] = Query(None)
):
    """Get available dashboard templates."""
    try:
        if role:
            template = DashboardTemplates.get_template_by_role(role)
            return [template] if template else []
        
        templates = DashboardTemplates.get_all_templates()
        
        if category:
            templates = {k: v for k, v in templates.items() if v.get("category") == category}
        
        return list(templates.values())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates")
async def create_dashboard_template(
    name: str,
    category: str,
    role: str,
    description: str,
    template_config: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Create a new dashboard template."""
    try:
        user_id = current_user["id"]
        
        template = dashboard_manager.create_dashboard_template(
            name, category, role, description, template_config, user_id
        )
        
        return {
            "id": template.id,
            "name": template.name,
            "category": template.category,
            "role": template.role,
            "description": template.description,
            "created_at": template.created_at
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dashboard_id}/alerts")
async def get_dashboard_alerts(
    dashboard_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get active alerts for a dashboard."""
    try:
        # This would integrate with the alerting system
        # For now, return mock alerts
        alerts = [
            {
                "id": "alert_1",
                "type": "warning",
                "title": "ROI Below Target",
                "message": "Technology ROI has dropped below 15% target",
                "timestamp": datetime.utcnow().isoformat(),
                "dashboard_id": dashboard_id
            }
        ]
        
        return alerts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dashboard_id}/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    dashboard_id: str,
    alert_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Acknowledge a dashboard alert."""
    try:
        # This would integrate with the alerting system
        return {"status": "success", "message": f"Alert {alert_id} acknowledged"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/websocket/stats")
async def get_websocket_stats(current_user: dict = Depends(get_current_user)):
    """Get WebSocket connection statistics."""
    try:
        stats = await websocket_manager.get_connection_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{dashboard_id}")
async def delete_dashboard(
    dashboard_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a dashboard."""
    try:
        # This would implement dashboard deletion
        # For now, return success
        return {"status": "success", "message": f"Dashboard {dashboard_id} deleted"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))