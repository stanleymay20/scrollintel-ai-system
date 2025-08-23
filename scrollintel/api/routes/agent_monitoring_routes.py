"""
API routes for Agent Steering System monitoring and analytics
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging
import asyncio

from scrollintel.core.agent_performance_monitor import (
    AgentSteeringMonitoringSystem,
    AgentMetrics,
    BusinessImpactMetrics,
    Alert,
    AlertSeverity,
    MetricType
)

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/monitoring", tags=["Agent Monitoring"])

# Global monitoring system instance (would be dependency injected in production)
monitoring_system: Optional[AgentSteeringMonitoringSystem] = None


# Pydantic models for API requests/responses
class AgentMetricsRequest(BaseModel):
    """Request model for submitting agent metrics"""
    agent_id: str = Field(..., description="Unique agent identifier")
    response_time: float = Field(..., ge=0, description="Response time in seconds")
    throughput: float = Field(..., ge=0, description="Throughput in requests per second")
    accuracy: float = Field(..., ge=0, le=100, description="Accuracy percentage")
    reliability: float = Field(..., ge=0, le=100, description="Reliability percentage")
    resource_utilization: Dict[str, float] = Field(..., description="Resource utilization metrics")
    business_impact: Dict[str, float] = Field(..., description="Business impact metrics")
    error_rate: float = Field(..., ge=0, le=100, description="Error rate percentage")
    success_rate: float = Field(..., ge=0, le=100, description="Success rate percentage")
    active_tasks: int = Field(..., ge=0, description="Number of active tasks")
    completed_tasks: int = Field(..., ge=0, description="Number of completed tasks")


class BusinessImpactRequest(BaseModel):
    """Request model for submitting business impact metrics"""
    cost_savings: float = Field(..., description="Cost savings in dollars")
    revenue_increase: float = Field(..., description="Revenue increase in dollars")
    risk_reduction: float = Field(..., description="Risk reduction percentage")
    productivity_gain: float = Field(..., description="Productivity gain percentage")
    customer_satisfaction: float = Field(..., ge=0, le=5, description="Customer satisfaction score")
    compliance_score: float = Field(..., ge=0, le=100, description="Compliance score percentage")
    roi_percentage: float = Field(..., description="ROI percentage")
    time_to_value: float = Field(..., ge=0, description="Time to value in hours")


class AlertResponse(BaseModel):
    """Response model for alerts"""
    id: str
    severity: str
    title: str
    description: str
    timestamp: datetime
    agent_id: Optional[str]
    metric_type: str
    threshold_value: float
    actual_value: float
    resolved: bool
    resolution_time: Optional[datetime]


class DashboardResponse(BaseModel):
    """Response model for dashboard data"""
    timestamp: datetime
    business_impact: Dict[str, Any]
    roi_metrics: Dict[str, Any]
    active_alerts: List[AlertResponse]
    alert_counts: Dict[str, int]
    system_status: str


class ExecutiveReportResponse(BaseModel):
    """Response model for executive reports"""
    report_generated: datetime
    time_period_days: int
    executive_summary: Dict[str, Any]
    business_impact: Dict[str, Any]
    system_performance: Dict[str, Any]
    alerts_and_issues: Dict[str, Any]
    recommendations: List[str]


# Dependency to get monitoring system
async def get_monitoring_system() -> AgentSteeringMonitoringSystem:
    """Get the monitoring system instance"""
    global monitoring_system
    if monitoring_system is None:
        # Initialize with default configuration
        redis_url = "redis://localhost:6379/0"
        database_url = "postgresql://user:password@localhost:5432/scrollintel"
        monitoring_system = AgentSteeringMonitoringSystem(redis_url, database_url)
        await monitoring_system.start_monitoring()
    return monitoring_system


@router.post("/agents/{agent_id}/metrics", 
             summary="Submit agent performance metrics",
             description="Submit real-time performance metrics for a specific agent")
async def submit_agent_metrics(
    agent_id: str,
    metrics_data: AgentMetricsRequest,
    monitoring: AgentSteeringMonitoringSystem = Depends(get_monitoring_system)
) -> Dict[str, str]:
    """Submit agent performance metrics"""
    try:
        # Create AgentMetrics object
        metrics = AgentMetrics(
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            response_time=metrics_data.response_time,
            throughput=metrics_data.throughput,
            accuracy=metrics_data.accuracy,
            reliability=metrics_data.reliability,
            resource_utilization=metrics_data.resource_utilization,
            business_impact=metrics_data.business_impact,
            error_rate=metrics_data.error_rate,
            success_rate=metrics_data.success_rate,
            active_tasks=metrics_data.active_tasks,
            completed_tasks=metrics_data.completed_tasks
        )
        
        # Store metrics
        await monitoring.metrics_collector.store_metrics(metrics)
        
        # Check for alerts
        alerts = await monitoring.alerting_system.check_metrics_for_alerts(metrics)
        
        return {
            "status": "success",
            "message": f"Metrics stored for agent {agent_id}",
            "alerts_generated": len(alerts)
        }
        
    except Exception as e:
        logger.error(f"Error submitting metrics for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/metrics",
            summary="Get agent performance metrics",
            description="Retrieve recent performance metrics for a specific agent")
async def get_agent_metrics(
    agent_id: str,
    hours: int = Query(24, ge=1, le=168, description="Number of hours of metrics to retrieve"),
    monitoring: AgentSteeringMonitoringSystem = Depends(get_monitoring_system)
) -> Dict[str, Any]:
    """Get agent performance metrics"""
    try:
        # Get recent metrics for the agent
        metrics = await monitoring.metrics_collector.collect_agent_metrics(agent_id)
        
        return {
            "agent_id": agent_id,
            "current_metrics": metrics.to_dict() if metrics else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/business-impact",
             summary="Submit business impact metrics",
             description="Submit business impact and ROI metrics")
async def submit_business_impact(
    impact_data: BusinessImpactRequest,
    monitoring: AgentSteeringMonitoringSystem = Depends(get_monitoring_system)
) -> Dict[str, str]:
    """Submit business impact metrics"""
    try:
        # Create BusinessImpactMetrics object
        impact_metrics = BusinessImpactMetrics(
            timestamp=datetime.utcnow(),
            cost_savings=impact_data.cost_savings,
            revenue_increase=impact_data.revenue_increase,
            risk_reduction=impact_data.risk_reduction,
            productivity_gain=impact_data.productivity_gain,
            customer_satisfaction=impact_data.customer_satisfaction,
            compliance_score=impact_data.compliance_score,
            roi_percentage=impact_data.roi_percentage,
            time_to_value=impact_data.time_to_value
        )
        
        # Track business impact
        await monitoring.impact_tracker.track_business_impact(impact_metrics)
        
        return {
            "status": "success",
            "message": "Business impact metrics stored successfully"
        }
        
    except Exception as e:
        logger.error(f"Error submitting business impact metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/business-impact/roi",
            summary="Get ROI analysis",
            description="Get ROI analysis for specified time period")
async def get_roi_analysis(
    days: int = Query(30, ge=1, le=365, description="Number of days for ROI analysis"),
    monitoring: AgentSteeringMonitoringSystem = Depends(get_monitoring_system)
) -> Dict[str, Any]:
    """Get ROI analysis"""
    try:
        time_period = timedelta(days=days)
        roi_data = await monitoring.impact_tracker.calculate_roi(time_period)
        
        return {
            "time_period_days": days,
            "roi_analysis": roi_data,
            "calculated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting ROI analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts",
            summary="Get active alerts",
            description="Retrieve active system alerts")
async def get_active_alerts(
    severity: Optional[str] = Query(None, description="Filter by alert severity"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of alerts to return"),
    monitoring: AgentSteeringMonitoringSystem = Depends(get_monitoring_system)
) -> List[AlertResponse]:
    """Get active alerts"""
    try:
        # Parse severity filter
        severity_filter = None
        if severity:
            try:
                severity_filter = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
        
        # Get alerts
        alerts = await monitoring.alerting_system.get_active_alerts(severity_filter)
        
        # Convert to response format and limit results
        alert_responses = []
        for alert in alerts[:limit]:
            alert_responses.append(AlertResponse(
                id=alert.id,
                severity=alert.severity.value,
                title=alert.title,
                description=alert.description,
                timestamp=alert.timestamp,
                agent_id=alert.agent_id,
                metric_type=alert.metric_type.value,
                threshold_value=alert.threshold_value,
                actual_value=alert.actual_value,
                resolved=alert.resolved,
                resolution_time=alert.resolution_time
            ))
        
        return alert_responses
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve",
             summary="Resolve alert",
             description="Mark an alert as resolved")
async def resolve_alert(
    alert_id: str,
    monitoring: AgentSteeringMonitoringSystem = Depends(get_monitoring_system)
) -> Dict[str, str]:
    """Resolve an alert"""
    try:
        success = await monitoring.alerting_system.resolve_alert(alert_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Alert {alert_id} resolved successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
            
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard",
            summary="Get dashboard data",
            description="Get real-time dashboard data for monitoring interface")
async def get_dashboard_data(
    monitoring: AgentSteeringMonitoringSystem = Depends(get_monitoring_system)
) -> DashboardResponse:
    """Get real-time dashboard data"""
    try:
        dashboard_data = await monitoring.get_dashboard_data()
        
        # Convert alerts to response format
        alert_responses = []
        for alert_dict in dashboard_data.get("active_alerts", []):
            alert_responses.append(AlertResponse(**alert_dict))
        
        return DashboardResponse(
            timestamp=datetime.fromisoformat(dashboard_data["timestamp"]),
            business_impact=dashboard_data["business_impact"],
            roi_metrics=dashboard_data["roi_metrics"],
            active_alerts=alert_responses,
            alert_counts=dashboard_data["alert_counts"],
            system_status=dashboard_data["system_status"]
        )
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/executive",
            summary="Generate executive report",
            description="Generate comprehensive executive report with business metrics")
async def generate_executive_report(
    days: int = Query(7, ge=1, le=90, description="Number of days for report period"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    monitoring: AgentSteeringMonitoringSystem = Depends(get_monitoring_system)
) -> ExecutiveReportResponse:
    """Generate executive report"""
    try:
        report_data = await monitoring.generate_executive_report(days)
        
        return ExecutiveReportResponse(
            report_generated=datetime.fromisoformat(report_data["report_generated"]),
            time_period_days=report_data["time_period_days"],
            executive_summary=report_data["executive_summary"],
            business_impact=report_data["business_impact"],
            system_performance=report_data["system_performance"],
            alerts_and_issues=report_data["alerts_and_issues"],
            recommendations=report_data["recommendations"]
        )
        
    except Exception as e:
        logger.error(f"Error generating executive report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/prometheus",
            summary="Get Prometheus metrics",
            description="Get metrics in Prometheus format for monitoring integration",
            response_class=PlainTextResponse)
async def get_prometheus_metrics(
    monitoring: AgentSteeringMonitoringSystem = Depends(get_monitoring_system)
) -> str:
    """Get Prometheus-formatted metrics"""
    try:
        return monitoring.get_prometheus_metrics()
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health",
            summary="Health check",
            description="Check monitoring system health")
async def health_check(
    monitoring: AgentSteeringMonitoringSystem = Depends(get_monitoring_system)
) -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        # Basic health check
        dashboard_data = await monitoring.get_dashboard_data()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_active": monitoring._monitoring_active,
            "system_status": dashboard_data.get("system_status", "unknown"),
            "components": {
                "metrics_collector": "operational",
                "impact_tracker": "operational",
                "alerting_system": "operational",
                "reporting_engine": "operational"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.post("/system/start",
             summary="Start monitoring system",
             description="Start the real-time monitoring system")
async def start_monitoring_system(
    interval: int = Query(30, ge=10, le=300, description="Collection interval in seconds"),
    monitoring: AgentSteeringMonitoringSystem = Depends(get_monitoring_system)
) -> Dict[str, str]:
    """Start the monitoring system"""
    try:
        await monitoring.start_monitoring(interval)
        return {
            "status": "success",
            "message": f"Monitoring system started with {interval}s interval"
        }
    except Exception as e:
        logger.error(f"Error starting monitoring system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/stop",
             summary="Stop monitoring system",
             description="Stop the real-time monitoring system")
async def stop_monitoring_system(
    monitoring: AgentSteeringMonitoringSystem = Depends(get_monitoring_system)
) -> Dict[str, str]:
    """Stop the monitoring system"""
    try:
        await monitoring.stop_monitoring()
        return {
            "status": "success",
            "message": "Monitoring system stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping monitoring system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents",
            summary="Get monitored agents",
            description="Get list of agents being monitored")
async def get_monitored_agents(
    monitoring: AgentSteeringMonitoringSystem = Depends(get_monitoring_system)
) -> Dict[str, Any]:
    """Get list of monitored agents"""
    try:
        # This would get actual agent list from registry
        # For now, return sample data
        agents = await monitoring._get_active_agents()
        
        agent_data = []
        for agent_id in agents:
            try:
                metrics = await monitoring.metrics_collector.collect_agent_metrics(agent_id)
                agent_data.append({
                    "agent_id": agent_id,
                    "status": "active",
                    "last_metrics": metrics.to_dict() if metrics else None
                })
            except Exception as e:
                agent_data.append({
                    "agent_id": agent_id,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "total_agents": len(agents),
            "agents": agent_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting monitored agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cleanup function for application shutdown
async def cleanup_monitoring():
    """Cleanup monitoring system on shutdown"""
    global monitoring_system
    if monitoring_system:
        await monitoring_system.cleanup()
        monitoring_system = None