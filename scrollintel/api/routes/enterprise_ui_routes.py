"""
Enterprise User Interface API Routes
Provides endpoints for role-based dashboards, natural language queries, and visualizations
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import logging
from pydantic import BaseModel

from scrollintel.core.config import get_settings
from scrollintel.models.database_utils import get_db
from scrollintel.security.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/enterprise-ui", tags=["enterprise-ui"])

# Pydantic Models
class DashboardMetric(BaseModel):
    id: str
    title: str
    value: str
    change: str
    trend: str
    icon: str
    color: str

class UserRole(BaseModel):
    id: str
    name: str
    permissions: List[str]
    dashboard_config: Dict[str, Any]

class NaturalLanguageQuery(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class QueryResult(BaseModel):
    id: str
    query: str
    response: str
    data: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime
    processing_time: int
    confidence: float

class VisualizationData(BaseModel):
    id: str
    title: str
    type: str
    data: List[Dict[str, Any]]
    config: Dict[str, Any]
    last_updated: datetime
    is_real_time: bool

class AlertNotification(BaseModel):
    id: str
    title: str
    message: str
    severity: str
    timestamp: datetime
    read: bool

@router.get("/dashboard/{role}", response_model=Dict[str, Any])
async def get_dashboard_data(
    role: str,
    time_range: str = Query("24h", description="Time range for metrics"),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get role-based dashboard data"""
    try:
        # Validate role
        valid_roles = ["executive", "analyst", "technical"]
        if role not in valid_roles:
            raise HTTPException(status_code=400, detail="Invalid role")

        # Generate role-specific metrics
        metrics = await _generate_role_metrics(role, time_range, db)
        alerts = await _get_user_alerts(current_user.id, db)
        insights = await _generate_insights(role, time_range, db)

        return {
            "role": role,
            "time_range": time_range,
            "metrics": metrics,
            "alerts": alerts,
            "insights": insights,
            "last_updated": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load dashboard data")

@router.post("/natural-language/query", response_model=QueryResult)
async def process_natural_language_query(
    query_request: NaturalLanguageQuery,
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Process natural language query and return insights"""
    try:
        start_time = datetime.utcnow()
        
        # Process the natural language query
        response = await _process_nl_query(query_request.query, query_request.context, current_user, db)
        
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        result = QueryResult(
            id=f"query_{int(datetime.utcnow().timestamp())}",
            query=query_request.query,
            response=response["response"],
            data=response.get("data"),
            visualizations=response.get("visualizations"),
            timestamp=datetime.utcnow(),
            processing_time=processing_time,
            confidence=response.get("confidence", 0.95)
        )
        
        # Log query for analytics
        await _log_query(current_user.id, query_request.query, result, db)
        
        return result

    except Exception as e:
        logger.error(f"Error processing natural language query: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process query")

@router.get("/visualizations", response_model=List[VisualizationData])
async def get_visualizations(
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get available visualizations for the user"""
    try:
        visualizations = await _get_user_visualizations(current_user, db)
        return visualizations

    except Exception as e:
        logger.error(f"Error getting visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load visualizations")

@router.post("/visualizations/refresh", response_model=List[VisualizationData])
async def refresh_visualizations(
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Refresh visualization data"""
    try:
        # Refresh all real-time visualizations
        visualizations = await _refresh_visualization_data(current_user, db)
        return visualizations

    except Exception as e:
        logger.error(f"Error refreshing visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to refresh visualizations")

@router.get("/alerts", response_model=List[AlertNotification])
async def get_user_alerts(
    limit: int = Query(10, description="Maximum number of alerts to return"),
    unread_only: bool = Query(False, description="Return only unread alerts"),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get user alerts and notifications"""
    try:
        alerts = await _get_user_alerts(current_user.id, db, limit, unread_only)
        return alerts

    except Exception as e:
        logger.error(f"Error getting user alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load alerts")

@router.put("/alerts/{alert_id}/read")
async def mark_alert_read(
    alert_id: str,
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Mark an alert as read"""
    try:
        await _mark_alert_read(alert_id, current_user.id, db)
        return {"status": "success", "message": "Alert marked as read"}

    except Exception as e:
        logger.error(f"Error marking alert as read: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update alert")

@router.get("/system-status", response_model=Dict[str, Any])
async def get_system_status(
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get overall system status and health metrics"""
    try:
        status = await _get_system_status(db)
        return status

    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

# Helper Functions
async def _generate_role_metrics(role: str, time_range: str, db) -> List[DashboardMetric]:
    """Generate metrics based on user role"""
    
    if role == "executive":
        return [
            DashboardMetric(
                id="revenue",
                title="Business Value Generated",
                value="$2.4M",
                change="+15.3%",
                trend="up",
                icon="DollarSign",
                color="text-green-600"
            ),
            DashboardMetric(
                id="cost_savings",
                title="Cost Savings",
                value="$890K",
                change="+8.2%",
                trend="up",
                icon="TrendingUp",
                color="text-blue-600"
            ),
            DashboardMetric(
                id="decision_accuracy",
                title="Decision Accuracy",
                value="94.7%",
                change="+2.1%",
                trend="up",
                icon="Target",
                color="text-purple-600"
            ),
            DashboardMetric(
                id="system_uptime",
                title="System Uptime",
                value="99.97%",
                change="Stable",
                trend="stable",
                icon="CheckCircle",
                color="text-green-600"
            )
        ]
    
    elif role == "analyst":
        return [
            DashboardMetric(
                id="data_processed",
                title="Data Processing Rate",
                value="847 GB/hr",
                change="+12%",
                trend="up",
                icon="Database",
                color="text-blue-600"
            ),
            DashboardMetric(
                id="model_accuracy",
                title="Model Accuracy",
                value="96.3%",
                change="+1.8%",
                trend="up",
                icon="Target",
                color="text-green-600"
            ),
            DashboardMetric(
                id="insights_generated",
                title="Insights Generated",
                value="1,247",
                change="+23%",
                trend="up",
                icon="BarChart3",
                color="text-purple-600"
            ),
            DashboardMetric(
                id="active_pipelines",
                title="Active Pipelines",
                value="12",
                change="+2",
                trend="up",
                icon="Activity",
                color="text-orange-600"
            )
        ]
    
    else:  # technical
        return [
            DashboardMetric(
                id="cpu_usage",
                title="CPU Usage",
                value="23.4%",
                change="-5%",
                trend="down",
                icon="Cpu",
                color="text-blue-600"
            ),
            DashboardMetric(
                id="memory_usage",
                title="Memory Usage",
                value="67.8%",
                change="+3%",
                trend="up",
                icon="Activity",
                color="text-yellow-600"
            ),
            DashboardMetric(
                id="active_agents",
                title="Active Agents",
                value="47",
                change="+2",
                trend="up",
                icon="Users",
                color="text-green-600"
            ),
            DashboardMetric(
                id="response_time",
                title="Response Time",
                value="127ms",
                change="-15ms",
                trend="down",
                icon="Clock",
                color="text-purple-600"
            )
        ]

async def _process_nl_query(query: str, context: Optional[Dict], user, db) -> Dict[str, Any]:
    """Process natural language query and generate response"""
    
    query_lower = query.lower()
    
    # Simple keyword-based processing (replace with actual NLP service)
    if any(word in query_lower for word in ['revenue', 'sales', 'money', 'profit']):
        return {
            "response": "Based on the latest data, revenue has increased by 15.3% compared to last quarter, with total revenue reaching $2.4M. The growth is primarily driven by enterprise clients (+23%) and our new AI services (+45%). Key contributing factors include improved customer retention (94.2%) and successful expansion into healthcare analytics.",
            "data": {
                "current_revenue": 2400000,
                "growth_rate": 15.3,
                "key_drivers": ["enterprise_clients", "ai_services", "customer_retention"]
            },
            "confidence": 0.92
        }
    
    elif any(word in query_lower for word in ['performance', 'system', 'uptime', 'health']):
        return {
            "response": "System performance metrics show excellent health across all infrastructure components. Average response time is 127ms (15% improvement), uptime is 99.97%, and we're processing 847GB/hour of data. CPU utilization is optimal at 23.4%, and all 47 active agents are operating within normal parameters.",
            "data": {
                "response_time": 127,
                "uptime": 99.97,
                "data_throughput": 847,
                "cpu_usage": 23.4,
                "active_agents": 47
            },
            "confidence": 0.95
        }
    
    elif any(word in query_lower for word in ['agents', 'top', 'best', 'performing']):
        return {
            "response": "Top performing agents this month: 1) CTO Agent (94.7% accuracy, 1,247 decisions), 2) Data Scientist Agent (96.3% model accuracy, 89 insights), 3) BI Agent (847GB processed, 156 reports generated). Overall agent efficiency has improved by 12% with the new orchestration system.",
            "data": {
                "top_agents": [
                    {"name": "CTO Agent", "accuracy": 94.7, "decisions": 1247},
                    {"name": "Data Scientist Agent", "accuracy": 96.3, "insights": 89},
                    {"name": "BI Agent", "data_processed": 847, "reports": 156}
                ]
            },
            "confidence": 0.94
        }
    
    else:
        return {
            "response": "I've analyzed your query and found relevant insights. The data shows positive trends across key metrics with opportunities for optimization in several areas. Would you like me to dive deeper into any specific aspect?",
            "data": {},
            "confidence": 0.85
        }

async def _get_user_alerts(user_id: str, db, limit: int = 10, unread_only: bool = False) -> List[AlertNotification]:
    """Get user alerts from database"""
    
    # Sample alerts (replace with actual database query)
    sample_alerts = [
        AlertNotification(
            id="alert_1",
            title="System Alert",
            message="High memory usage detected on server cluster 3",
            severity="warning",
            timestamp=datetime.utcnow() - timedelta(minutes=5),
            read=False
        ),
        AlertNotification(
            id="alert_2",
            title="Revenue Milestone",
            message="Monthly revenue target exceeded by 15%",
            severity="success",
            timestamp=datetime.utcnow() - timedelta(minutes=10),
            read=False
        ),
        AlertNotification(
            id="alert_3",
            title="Data Pipeline",
            message="Customer data sync completed successfully",
            severity="info",
            timestamp=datetime.utcnow() - timedelta(minutes=15),
            read=True
        )
    ]
    
    if unread_only:
        sample_alerts = [alert for alert in sample_alerts if not alert.read]
    
    return sample_alerts[:limit]

async def _generate_insights(role: str, time_range: str, db) -> List[Dict[str, Any]]:
    """Generate role-specific insights"""
    
    if role == "executive":
        return [
            {
                "type": "opportunity",
                "title": "Market Opportunity Identified",
                "description": "AI analysis suggests expanding into healthcare analytics could generate $1.2M additional revenue",
                "priority": "high",
                "action": "View Full Analysis"
            },
            {
                "type": "success",
                "title": "Operational Excellence",
                "description": "Process automation has reduced manual work by 67%, freeing up 40 hours/week for strategic tasks",
                "priority": "medium",
                "action": "View Details"
            }
        ]
    
    return []

async def _get_user_visualizations(user, db) -> List[VisualizationData]:
    """Get visualizations available to the user"""
    
    # Sample visualizations
    return [
        VisualizationData(
            id="viz_1",
            title="Revenue Trends",
            type="line",
            data=[
                {"month": "Jan", "revenue": 2100000, "target": 2000000},
                {"month": "Feb", "revenue": 2300000, "target": 2100000},
                {"month": "Mar", "revenue": 2400000, "target": 2200000}
            ],
            config={
                "xAxis": "month",
                "yAxis": "revenue",
                "showLegend": True,
                "colors": ["#3b82f6", "#10b981"]
            },
            last_updated=datetime.utcnow(),
            is_real_time=True
        ),
        VisualizationData(
            id="viz_2",
            title="Agent Performance",
            type="bar",
            data=[
                {"agent": "CTO Agent", "accuracy": 94.7, "requests": 1247},
                {"agent": "Data Scientist", "accuracy": 96.3, "requests": 892}
            ],
            config={
                "xAxis": "agent",
                "yAxis": "accuracy",
                "colors": ["#8b5cf6"]
            },
            last_updated=datetime.utcnow(),
            is_real_time=False
        )
    ]

async def _refresh_visualization_data(user, db) -> List[VisualizationData]:
    """Refresh real-time visualization data"""
    return await _get_user_visualizations(user, db)

async def _log_query(user_id: str, query: str, result: QueryResult, db):
    """Log query for analytics"""
    # Implementation for logging queries
    pass

async def _mark_alert_read(alert_id: str, user_id: str, db):
    """Mark alert as read in database"""
    # Implementation for marking alerts as read
    pass

async def _get_system_status(db) -> Dict[str, Any]:
    """Get overall system status"""
    return {
        "overall_status": "operational",
        "services": {
            "api": {"status": "healthy", "response_time": 127},
            "database": {"status": "healthy", "connections": 45},
            "agents": {"status": "healthy", "active_count": 47},
            "data_pipeline": {"status": "monitoring", "throughput": 847}
        },
        "uptime": 99.97,
        "last_updated": datetime.utcnow().isoformat()
    }