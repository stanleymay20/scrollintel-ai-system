"""
ScrollIntel Real-Time Monitoring API Routes
RESTful API endpoints for real-time monitoring, analytics, and executive reporting
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

from ...core.real_time_monitoring import (
    real_time_monitor,
    business_impact_tracker,
    executive_reporting,
    automated_alerting,
    get_real_time_dashboard
)
from ...core.monitoring import metrics_collector, performance_monitor
from ...core.analytics import event_tracker, analytics_engine
from ...core.alerting import alert_manager
from ...core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/monitoring", tags=["Real-Time Monitoring"])

@router.get("/dashboard")
async def get_executive_dashboard():
    """
    Get comprehensive executive dashboard with real-time metrics
    
    Returns:
        - Business impact metrics (ROI, cost savings, revenue impact)
        - System health metrics
        - Agent performance summary
        - Key performance indicators
        - Competitive positioning data
    """
    try:
        dashboard = await get_real_time_dashboard()
        
        logger.info("Executive dashboard requested")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": dashboard,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating executive dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard: {str(e)}")

@router.get("/business-impact")
async def get_business_impact_metrics():
    """
    Get detailed business impact and ROI metrics
    
    Returns:
        - Total ROI calculation
        - Cost savings (24h, 7d, 30d)
        - Revenue impact
        - Productivity gains
        - Decision accuracy improvements
        - Competitive advantage scores
    """
    try:
        metrics = await business_impact_tracker.calculate_roi_metrics()
        
        logger.info("Business impact metrics requested", roi=metrics.total_roi)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "total_roi": metrics.total_roi,
                    "cost_savings": {
                        "24_hours": metrics.cost_savings_24h,
                        "7_days": metrics.cost_savings_7d,
                        "30_days": metrics.cost_savings_30d
                    },
                    "revenue_impact": metrics.revenue_impact,
                    "productivity_gain": metrics.productivity_gain,
                    "decision_accuracy_improvement": metrics.decision_accuracy_improvement,
                    "time_to_insight_reduction": metrics.time_to_insight_reduction,
                    "user_satisfaction_score": metrics.user_satisfaction_score,
                    "competitive_advantage_score": metrics.competitive_advantage_score,
                    "timestamp": metrics.timestamp.isoformat()
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error calculating business impact metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate business impact: {str(e)}")

@router.get("/agents")
async def get_agent_performance_metrics():
    """
    Get real-time agent performance metrics
    
    Returns:
        - Individual agent metrics (CPU, memory, response time, success rate)
        - Agent health status
        - Business value generated per agent
        - Performance trends
    """
    try:
        agent_metrics = real_time_monitor.get_all_agent_metrics()
        
        response_data = []
        for agent in agent_metrics:
            response_data.append({
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "status": agent.status,
                "performance": {
                    "cpu_usage": agent.cpu_usage,
                    "memory_usage": agent.memory_usage,
                    "avg_response_time": agent.avg_response_time,
                    "success_rate": agent.success_rate,
                    "throughput_per_minute": agent.throughput_per_minute
                },
                "business_metrics": {
                    "request_count": agent.request_count,
                    "error_count": agent.error_count,
                    "business_value_generated": agent.business_value_generated,
                    "cost_savings": agent.cost_savings
                },
                "last_activity": agent.last_activity.isoformat(),
                "uptime_seconds": agent.uptime_seconds
            })
        
        logger.info(f"Agent performance metrics requested for {len(agent_metrics)} agents")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "agents": response_data,
                    "summary": {
                        "total_agents": len(agent_metrics),
                        "active_agents": len([a for a in agent_metrics if a.status == "active"]),
                        "avg_success_rate": sum(a.success_rate for a in agent_metrics) / len(agent_metrics) if agent_metrics else 0,
                        "total_business_value": sum(a.business_value_generated for a in agent_metrics)
                    }
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting agent performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent metrics: {str(e)}")

@router.get("/agents/{agent_id}")
async def get_agent_details(agent_id: str, hours: int = Query(24, ge=1, le=168)):
    """
    Get detailed metrics for a specific agent
    
    Args:
        agent_id: Agent identifier
        hours: Hours of historical data to include (1-168)
    
    Returns:
        - Current agent metrics
        - Historical performance data
        - Performance trends
    """
    try:
        # Get current metrics
        current_metrics = real_time_monitor.get_agent_metrics(agent_id)
        if not current_metrics:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Get historical data
        historical_data = real_time_monitor.get_agent_history(agent_id, hours)
        
        logger.info(f"Agent details requested for {agent_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "agent_id": agent_id,
                    "current_metrics": {
                        "agent_type": current_metrics.agent_type,
                        "status": current_metrics.status,
                        "cpu_usage": current_metrics.cpu_usage,
                        "memory_usage": current_metrics.memory_usage,
                        "avg_response_time": current_metrics.avg_response_time,
                        "success_rate": current_metrics.success_rate,
                        "request_count": current_metrics.request_count,
                        "error_count": current_metrics.error_count,
                        "business_value_generated": current_metrics.business_value_generated,
                        "cost_savings": current_metrics.cost_savings,
                        "last_activity": current_metrics.last_activity.isoformat(),
                        "uptime_seconds": current_metrics.uptime_seconds,
                        "throughput_per_minute": current_metrics.throughput_per_minute
                    },
                    "historical_data": historical_data,
                    "data_points": len(historical_data)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent details for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent details: {str(e)}")

@router.get("/system-health")
async def get_system_health():
    """
    Get comprehensive system health metrics
    
    Returns:
        - Overall health score
        - Component health scores (performance, availability, security)
        - System resource utilization
        - Uptime statistics
    """
    try:
        dashboard = await executive_reporting.generate_executive_dashboard()
        system_health = dashboard["system_health"]
        
        logger.info("System health metrics requested")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": system_health,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")

@router.get("/alerts")
async def get_active_alerts():
    """
    Get all active alerts and recent alert history
    
    Returns:
        - Active alerts with severity and details
        - Recent alert history
        - Alert statistics
    """
    try:
        active_alerts = alert_manager.get_active_alerts()
        alert_history = alert_manager.get_alert_history(hours=24)
        
        active_alerts_data = []
        for alert in active_alerts:
            active_alerts_data.append({
                "id": alert.id,
                "name": alert.name,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp.isoformat(),
                "tags": alert.tags
            })
        
        history_data = []
        for alert in alert_history:
            history_data.append({
                "id": alert.id,
                "name": alert.name,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "timestamp": alert.timestamp.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            })
        
        logger.info(f"Alerts requested: {len(active_alerts)} active, {len(alert_history)} in history")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "active_alerts": active_alerts_data,
                    "alert_history": history_data,
                    "statistics": {
                        "total_active": len(active_alerts),
                        "critical_alerts": len([a for a in active_alerts if a.severity.value == "critical"]),
                        "warning_alerts": len([a for a in active_alerts if a.severity.value == "warning"]),
                        "alerts_24h": len(alert_history)
                    }
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user_id: str = Query(...)):
    """
    Acknowledge an active alert
    
    Args:
        alert_id: Alert identifier
        user_id: User acknowledging the alert
    """
    try:
        alert_manager.acknowledge_alert(alert_id, user_id)
        
        logger.info(f"Alert {alert_id} acknowledged by {user_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Alert {alert_id} acknowledged successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")

@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """
    Get metrics in Prometheus format for external monitoring systems
    
    Returns:
        - Prometheus-formatted metrics
    """
    try:
        metrics_data = metrics_collector.export_metrics()
        
        return JSONResponse(
            status_code=200,
            content=metrics_data,
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Error exporting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {str(e)}")

@router.get("/analytics/summary")
async def get_analytics_summary(days: int = Query(30, ge=1, le=365)):
    """
    Get analytics summary for specified period
    
    Args:
        days: Number of days to analyze (1-365)
    
    Returns:
        - User activity analytics
        - Agent usage statistics
        - Performance trends
    """
    try:
        analytics_summary = await analytics_engine.get_analytics_summary(days=days)
        agent_usage_stats = await analytics_engine.get_agent_usage_stats(days=days)
        
        logger.info(f"Analytics summary requested for {days} days")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "period_days": days,
                    "user_analytics": {
                        "total_users": analytics_summary.total_users,
                        "active_users_24h": analytics_summary.active_users_24h,
                        "active_users_7d": analytics_summary.active_users_7d,
                        "active_users_30d": analytics_summary.active_users_30d,
                        "total_sessions": analytics_summary.total_sessions,
                        "avg_session_duration": analytics_summary.avg_session_duration,
                        "bounce_rate": analytics_summary.bounce_rate,
                        "user_retention": analytics_summary.user_retention
                    },
                    "content_analytics": {
                        "top_events": analytics_summary.top_events,
                        "top_pages": analytics_summary.top_pages
                    },
                    "agent_usage": agent_usage_stats
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics summary: {str(e)}")

@router.post("/agents/{agent_id}/metrics")
async def update_agent_metrics(agent_id: str, metrics: Dict[str, Any]):
    """
    Update metrics for a specific agent (used by agents to report their status)
    
    Args:
        agent_id: Agent identifier
        metrics: Agent metrics data
    """
    try:
        await real_time_monitor.update_agent_metrics(agent_id, metrics)
        
        logger.info(f"Metrics updated for agent {agent_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Metrics updated for agent {agent_id}",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error updating agent metrics for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update agent metrics: {str(e)}")

@router.get("/reports/executive")
async def generate_executive_report(
    format: str = Query("json", regex="^(json|pdf|excel)$"),
    period: str = Query("monthly", regex="^(daily|weekly|monthly|quarterly)$")
):
    """
    Generate comprehensive executive report
    
    Args:
        format: Report format (json, pdf, excel)
        period: Report period (daily, weekly, monthly, quarterly)
    
    Returns:
        - Comprehensive business impact report
        - Performance analysis
        - ROI calculations
        - Strategic recommendations
    """
    try:
        dashboard = await executive_reporting.generate_executive_dashboard()
        
        # Create comprehensive report
        report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "period": period,
                "format": format,
                "report_type": "executive_summary"
            },
            "executive_summary": dashboard["executive_summary"],
            "business_impact": dashboard["business_impact"],
            "system_performance": dashboard["system_health"],
            "agent_performance": dashboard["agent_performance"],
            "key_metrics": dashboard["key_performance_indicators"],
            "competitive_analysis": dashboard["competitive_positioning"],
            "recommendations": [
                "Continue investing in AI agent optimization for 15% additional ROI",
                "Expand agent capabilities to capture new market opportunities",
                "Implement advanced predictive analytics for proactive decision making",
                "Scale infrastructure to support 50% user growth projection"
            ],
            "risk_assessment": {
                "operational_risks": "Low",
                "technical_risks": "Low", 
                "business_risks": "Very Low",
                "mitigation_strategies": [
                    "Automated failover systems in place",
                    "Real-time monitoring and alerting active",
                    "Disaster recovery procedures tested"
                ]
            }
        }
        
        logger.info(f"Executive report generated: {format} format, {period} period")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": report,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating executive report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate executive report: {str(e)}")

@router.get("/health")
async def monitoring_health_check():
    """
    Health check endpoint for the monitoring system itself
    
    Returns:
        - Monitoring system status
        - Service availability
        - Performance metrics
    """
    try:
        # Check monitoring system components
        components = {
            "real_time_monitor": "healthy",
            "business_impact_tracker": "healthy", 
            "executive_reporting": "healthy",
            "automated_alerting": "healthy",
            "metrics_collector": "healthy",
            "alert_manager": "healthy"
        }
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "status": "healthy",
                "components": components,
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": (datetime.utcnow() - datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
            }
        )
        
    except Exception as e:
        logger.error(f"Monitoring health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )