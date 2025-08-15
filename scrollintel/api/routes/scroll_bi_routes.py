"""
FastAPI routes for ScrollBI agent - dashboard creation and management.

Provides REST API endpoints for dashboard creation, real-time updates,
alert management, and sharing functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
import json
import asyncio
import logging

from pydantic import BaseModel, Field
from scrollintel.agents.scroll_bi_agent import ScrollBIAgent, DashboardType, AlertType, SharePermission
from scrollintel.core.interfaces import AgentRequest, ResponseStatus
from scrollintel.security.auth import get_current_user
from scrollintel.models.database import User, Dashboard

logger = logging.getLogger(__name__)

# Initialize router and agent
router = APIRouter(prefix="/api/v1/bi", tags=["ScrollBI"])
security = HTTPBearer()
bi_agent = ScrollBIAgent()

# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, dashboard_id: str):
        await websocket.accept()
        if dashboard_id not in self.active_connections:
            self.active_connections[dashboard_id] = []
        self.active_connections[dashboard_id].append(websocket)
        logger.info(f"WebSocket connected for dashboard {dashboard_id}")
    
    def disconnect(self, websocket: WebSocket, dashboard_id: str):
        if dashboard_id in self.active_connections:
            self.active_connections[dashboard_id].remove(websocket)
            if not self.active_connections[dashboard_id]:
                del self.active_connections[dashboard_id]
        logger.info(f"WebSocket disconnected for dashboard {dashboard_id}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast_to_dashboard(self, message: str, dashboard_id: str):
        if dashboard_id in self.active_connections:
            for connection in self.active_connections[dashboard_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to WebSocket: {e}")

manager = ConnectionManager()

# Pydantic models for request/response

class DashboardCreateRequest(BaseModel):
    """Request model for dashboard creation."""
    name: str = Field(..., description="Dashboard name")
    description: Optional[str] = Field(None, description="Dashboard description")
    dashboard_type: DashboardType = Field(DashboardType.CUSTOM, description="Dashboard type")
    layout: str = Field("grid", description="Dashboard layout")
    theme: str = Field("light", description="Dashboard theme")
    refresh_interval: int = Field(60, description="Refresh interval in minutes")
    auto_refresh: bool = Field(True, description="Enable auto refresh")
    real_time_enabled: bool = Field(False, description="Enable real-time updates")
    charts: List[Dict[str, Any]] = Field(default_factory=list, description="Chart configurations")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Dashboard filters")
    data_sources: List[Dict[str, Any]] = Field(default_factory=list, description="Data sources")

class DashboardResponse(BaseModel):
    """Response model for dashboard operations."""
    dashboard_id: str
    name: str
    description: Optional[str]
    dashboard_type: str
    layout: str
    theme: str
    refresh_interval: int
    auto_refresh: bool
    real_time_enabled: bool
    charts: List[Dict[str, Any]]
    filters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class AlertRuleRequest(BaseModel):
    """Request model for alert rule creation."""
    name: str = Field(..., description="Alert rule name")
    description: Optional[str] = Field(None, description="Alert rule description")
    dashboard_id: str = Field(..., description="Dashboard ID")
    metric_name: str = Field(..., description="Metric to monitor")
    alert_type: AlertType = Field(AlertType.THRESHOLD_EXCEEDED, description="Alert type")
    threshold_value: float = Field(..., description="Threshold value")
    comparison_operator: str = Field(">", description="Comparison operator")
    notification_channels: List[str] = Field(default_factory=list, description="Notification channels")
    is_active: bool = Field(True, description="Alert rule active status")

class AlertRuleResponse(BaseModel):
    """Response model for alert rules."""
    id: str
    name: str
    description: Optional[str]
    dashboard_id: str
    metric_name: str
    alert_type: str
    threshold_value: float
    comparison_operator: str
    notification_channels: List[str]
    is_active: bool
    created_at: datetime

class ShareRequest(BaseModel):
    """Request model for dashboard sharing."""
    dashboard_id: str = Field(..., description="Dashboard ID to share")
    user_emails: List[str] = Field(default_factory=list, description="User emails to share with")
    permission: SharePermission = Field(SharePermission.VIEW_ONLY, description="Permission level")
    expires_at: Optional[datetime] = Field(None, description="Share expiration date")
    public_access: bool = Field(False, description="Enable public access")

class ShareResponse(BaseModel):
    """Response model for dashboard sharing."""
    share_id: str
    dashboard_id: str
    share_link: str
    permission: str
    expires_at: Optional[datetime]
    created_at: datetime

class OptimizationRequest(BaseModel):
    """Request model for dashboard optimization."""
    dashboard_id: str = Field(..., description="Dashboard ID to optimize")
    performance_data: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    usage_analytics: Dict[str, Any] = Field(default_factory=dict, description="Usage analytics")

class BIQueryRequest(BaseModel):
    """Request model for BI query analysis."""
    query: str = Field(..., description="BI query to analyze")
    data_context: Dict[str, Any] = Field(default_factory=dict, description="Data context")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")

# API Routes

@router.post("/dashboards", response_model=Dict[str, Any])
async def create_dashboard(
    request: DashboardCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new dashboard with instant generation."""
    try:
        # Create agent request
        agent_request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=str(current_user.id),
            agent_id="scroll-bi",
            prompt=f"Create {request.dashboard_type.value} dashboard named '{request.name}'",
            context={
                "dashboard_config": request.dict(),
                "user_id": str(current_user.id)
            },
            priority=1,
            created_at=datetime.now()
        )
        
        # Process request
        response = await bi_agent.process_request(agent_request)
        
        if response.status == ResponseStatus.ERROR:
            raise HTTPException(status_code=500, detail=response.error_message)
        
        return {
            "success": True,
            "dashboard_id": f"dashboard-{uuid4()}",
            "message": "Dashboard created successfully",
            "content": response.content,
            "execution_time": response.execution_time
        }
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboards/{dashboard_id}")
async def get_dashboard(
    dashboard_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get dashboard configuration and data."""
    try:
        # In a real implementation, this would fetch from database
        return {
            "dashboard_id": dashboard_id,
            "name": "Sample Dashboard",
            "description": "Sample dashboard description",
            "dashboard_type": "executive",
            "layout": "grid",
            "theme": "light",
            "refresh_interval": 60,
            "auto_refresh": True,
            "real_time_enabled": False,
            "charts": [],
            "filters": {},
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error fetching dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/dashboards/{dashboard_id}")
async def update_dashboard(
    dashboard_id: str,
    request: DashboardCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Update existing dashboard configuration."""
    try:
        # Create agent request for dashboard update
        agent_request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=str(current_user.id),
            agent_id="scroll-bi",
            prompt=f"Update dashboard {dashboard_id} with new configuration",
            context={
                "dashboard_id": dashboard_id,
                "dashboard_config": request.dict(),
                "user_id": str(current_user.id)
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(agent_request)
        
        if response.status == ResponseStatus.ERROR:
            raise HTTPException(status_code=500, detail=response.error_message)
        
        return {
            "success": True,
            "dashboard_id": dashboard_id,
            "message": "Dashboard updated successfully",
            "content": response.content,
            "execution_time": response.execution_time
        }
        
    except Exception as e:
        logger.error(f"Error updating dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/dashboards/{dashboard_id}")
async def delete_dashboard(
    dashboard_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete dashboard."""
    try:
        # In a real implementation, this would delete from database
        return {
            "success": True,
            "message": f"Dashboard {dashboard_id} deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dashboards/{dashboard_id}/real-time")
async def setup_real_time(
    dashboard_id: str,
    config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Set up real-time updates for dashboard."""
    try:
        agent_request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=str(current_user.id),
            agent_id="scroll-bi",
            prompt=f"Set up real-time updates for dashboard {dashboard_id}",
            context={
                "dashboard_id": dashboard_id,
                "real_time_config": config,
                "user_id": str(current_user.id)
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(agent_request)
        
        if response.status == ResponseStatus.ERROR:
            raise HTTPException(status_code=500, detail=response.error_message)
        
        return {
            "success": True,
            "dashboard_id": dashboard_id,
            "message": "Real-time updates configured successfully",
            "content": response.content,
            "websocket_endpoint": f"/ws/dashboard/{dashboard_id}"
        }
        
    except Exception as e:
        logger.error(f"Error setting up real-time for dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/dashboard/{dashboard_id}")
async def websocket_endpoint(websocket: WebSocket, dashboard_id: str):
    """WebSocket endpoint for real-time dashboard updates."""
    await manager.connect(websocket, dashboard_id)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get("type") == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    websocket
                )
            elif message_data.get("type") == "subscribe":
                # Subscribe to specific chart updates
                await manager.send_personal_message(
                    json.dumps({
                        "type": "subscribed",
                        "dashboard_id": dashboard_id,
                        "charts": message_data.get("charts", [])
                    }),
                    websocket
                )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, dashboard_id)
    except Exception as e:
        logger.error(f"WebSocket error for dashboard {dashboard_id}: {e}")
        manager.disconnect(websocket, dashboard_id)

@router.post("/dashboards/{dashboard_id}/alerts", response_model=Dict[str, Any])
async def create_alert_rule(
    dashboard_id: str,
    request: AlertRuleRequest,
    current_user: User = Depends(get_current_user)
):
    """Create alert rule for dashboard metrics."""
    try:
        agent_request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=str(current_user.id),
            agent_id="scroll-bi",
            prompt=f"Create alert rule for {request.metric_name} in dashboard {dashboard_id}",
            context={
                "dashboard_id": dashboard_id,
                "alert_config": request.dict(),
                "user_id": str(current_user.id)
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(agent_request)
        
        if response.status == ResponseStatus.ERROR:
            raise HTTPException(status_code=500, detail=response.error_message)
        
        return {
            "success": True,
            "alert_id": f"alert-{uuid4()}",
            "dashboard_id": dashboard_id,
            "message": "Alert rule created successfully",
            "content": response.content
        }
        
    except Exception as e:
        logger.error(f"Error creating alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboards/{dashboard_id}/alerts")
async def get_alert_rules(
    dashboard_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get all alert rules for dashboard."""
    try:
        # In a real implementation, this would fetch from database
        return {
            "dashboard_id": dashboard_id,
            "alert_rules": [],
            "total_count": 0
        }
        
    except Exception as e:
        logger.error(f"Error fetching alert rules for dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/alerts/{alert_id}")
async def update_alert_rule(
    alert_id: str,
    request: AlertRuleRequest,
    current_user: User = Depends(get_current_user)
):
    """Update alert rule configuration."""
    try:
        # In a real implementation, this would update in database
        return {
            "success": True,
            "alert_id": alert_id,
            "message": "Alert rule updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating alert rule {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/alerts/{alert_id}")
async def delete_alert_rule(
    alert_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete alert rule."""
    try:
        # In a real implementation, this would delete from database
        return {
            "success": True,
            "message": f"Alert rule {alert_id} deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting alert rule {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dashboards/{dashboard_id}/share", response_model=Dict[str, Any])
async def share_dashboard(
    dashboard_id: str,
    request: ShareRequest,
    current_user: User = Depends(get_current_user)
):
    """Share dashboard with users or create public link."""
    try:
        agent_request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=str(current_user.id),
            agent_id="scroll-bi",
            prompt=f"Share dashboard {dashboard_id} with specified users and permissions",
            context={
                "dashboard_id": dashboard_id,
                "sharing_config": request.dict(),
                "user_id": str(current_user.id)
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(agent_request)
        
        if response.status == ResponseStatus.ERROR:
            raise HTTPException(status_code=500, detail=response.error_message)
        
        return {
            "success": True,
            "share_id": f"share-{uuid4()}",
            "dashboard_id": dashboard_id,
            "share_link": f"https://scrollintel.com/dashboard/{dashboard_id}/share",
            "message": "Dashboard shared successfully",
            "content": response.content
        }
        
    except Exception as e:
        logger.error(f"Error sharing dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboards/{dashboard_id}/shares")
async def get_dashboard_shares(
    dashboard_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get all shares for dashboard."""
    try:
        # In a real implementation, this would fetch from database
        return {
            "dashboard_id": dashboard_id,
            "shares": [],
            "total_count": 0
        }
        
    except Exception as e:
        logger.error(f"Error fetching shares for dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/shares/{share_id}")
async def revoke_dashboard_share(
    share_id: str,
    current_user: User = Depends(get_current_user)
):
    """Revoke dashboard share."""
    try:
        # In a real implementation, this would delete from database
        return {
            "success": True,
            "message": f"Dashboard share {share_id} revoked successfully"
        }
        
    except Exception as e:
        logger.error(f"Error revoking share {share_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dashboards/{dashboard_id}/optimize")
async def optimize_dashboard(
    dashboard_id: str,
    request: OptimizationRequest,
    current_user: User = Depends(get_current_user)
):
    """Optimize dashboard performance and user experience."""
    try:
        agent_request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=str(current_user.id),
            agent_id="scroll-bi",
            prompt=f"Optimize performance for dashboard {dashboard_id}",
            context={
                "dashboard_id": dashboard_id,
                "performance_data": request.performance_data,
                "usage_analytics": request.usage_analytics,
                "user_id": str(current_user.id)
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(agent_request)
        
        if response.status == ResponseStatus.ERROR:
            raise HTTPException(status_code=500, detail=response.error_message)
        
        return {
            "success": True,
            "dashboard_id": dashboard_id,
            "message": "Dashboard optimization completed",
            "content": response.content,
            "execution_time": response.execution_time
        }
        
    except Exception as e:
        logger.error(f"Error optimizing dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bi/analyze-query")
async def analyze_bi_query(
    request: BIQueryRequest,
    current_user: User = Depends(get_current_user)
):
    """Analyze BI query and recommend dashboard layout."""
    try:
        agent_request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=str(current_user.id),
            agent_id="scroll-bi",
            prompt="Analyze BI query and recommend optimal dashboard layout",
            context={
                "bi_query": request.query,
                "data_context": request.data_context,
                "user_preferences": request.user_preferences,
                "user_id": str(current_user.id)
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(agent_request)
        
        if response.status == ResponseStatus.ERROR:
            raise HTTPException(status_code=500, detail=response.error_message)
        
        return {
            "success": True,
            "message": "BI query analysis completed",
            "content": response.content,
            "execution_time": response.execution_time
        }
        
    except Exception as e:
        logger.error(f"Error analyzing BI query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def get_dashboard_templates(
    current_user: User = Depends(get_current_user)
):
    """Get available dashboard templates."""
    try:
        templates = [
            {
                "id": "executive_summary",
                "name": "Executive Summary Dashboard",
                "description": "High-level KPIs and metrics for executives",
                "dashboard_type": "executive",
                "preview_image": "/static/templates/executive_preview.png"
            },
            {
                "id": "sales_dashboard",
                "name": "Sales Performance Dashboard",
                "description": "Sales metrics and pipeline analysis",
                "dashboard_type": "sales",
                "preview_image": "/static/templates/sales_preview.png"
            },
            {
                "id": "financial_dashboard",
                "name": "Financial Dashboard",
                "description": "Financial metrics and analysis",
                "dashboard_type": "financial",
                "preview_image": "/static/templates/financial_preview.png"
            },
            {
                "id": "operational_dashboard",
                "name": "Operational Dashboard",
                "description": "Operational metrics and monitoring",
                "dashboard_type": "operational",
                "preview_image": "/static/templates/operational_preview.png"
            }
        ]
        
        return {
            "templates": templates,
            "total_count": len(templates)
        }
        
    except Exception as e:
        logger.error(f"Error fetching dashboard templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint for ScrollBI agent."""
    try:
        is_healthy = await bi_agent.health_check()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "agent_id": bi_agent.agent_id,
            "agent_name": bi_agent.name,
            "capabilities": [cap.name for cap in bi_agent.get_capabilities()],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Background task for sending real-time updates
async def send_dashboard_update(dashboard_id: str, update_data: Dict[str, Any]):
    """Send real-time update to dashboard subscribers."""
    try:
        message = json.dumps({
            "type": "dashboard_update",
            "dashboard_id": dashboard_id,
            "data": update_data,
            "timestamp": datetime.now().isoformat()
        })
        
        await manager.broadcast_to_dashboard(message, dashboard_id)
        logger.info(f"Sent real-time update to dashboard {dashboard_id}")
        
    except Exception as e:
        logger.error(f"Error sending dashboard update: {e}")

# Utility function to trigger dashboard updates
@router.post("/dashboards/{dashboard_id}/trigger-update")
async def trigger_dashboard_update(
    dashboard_id: str,
    update_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Trigger real-time dashboard update."""
    try:
        background_tasks.add_task(send_dashboard_update, dashboard_id, update_data)
        
        return {
            "success": True,
            "message": f"Update triggered for dashboard {dashboard_id}"
        }
        
    except Exception as e:
        logger.error(f"Error triggering dashboard update: {e}")
        raise HTTPException(status_code=500, detail=str(e))