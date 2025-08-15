"""
ScrollIntel Monitoring API Routes
API endpoints for monitoring dashboard and system administrators
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from ...core.monitoring import metrics_collector, performance_monitor
from ...core.alerting import alert_manager, Alert, AlertRule, AlertSeverity
from ...core.analytics import analytics_engine, event_tracker
from ...core.resource_monitor import system_monitor, database_monitor, redis_monitor
from ...core.logging_config import get_logger
from ...security.auth import get_current_user
from ...models.database import User

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
logger = get_logger(__name__)

@router.get("/metrics")
async def get_prometheus_metrics():
    """Get Prometheus metrics endpoint"""
    try:
        return metrics_collector.export_metrics()
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export metrics")

@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Basic health check
        system_metrics = system_monitor.get_current_metrics()
        db_metrics = await database_monitor.collect_metrics()
        redis_metrics = await redis_monitor.collect_metrics()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "system": "healthy" if system_metrics else "unhealthy",
                "database": "healthy" if db_metrics else "unhealthy", 
                "redis": "healthy" if redis_metrics else "unhealthy"
            }
        }
        
        # Check if any component is unhealthy
        overall_healthy = all(status == "healthy" for status in health_status["components"].values())
        
        if not overall_healthy:
            health_status["status"] = "degraded"
            
        return health_status
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/dashboard")
async def get_monitoring_dashboard(
    current_user: User = Depends(get_current_user),
    hours: int = Query(default=24, ge=1, le=168)  # 1 hour to 1 week
):
    """Get comprehensive monitoring dashboard data"""
    try:
        # System metrics
        system_metrics = system_monitor.get_current_metrics()
        system_history = system_monitor.get_metrics_history(hours=hours)
        
        # Database metrics
        db_metrics = await database_monitor.collect_metrics()
        
        # Redis metrics
        redis_metrics = await redis_monitor.collect_metrics()
        
        # Analytics summary
        analytics_summary = await analytics_engine.get_analytics_summary(days=min(hours//24, 30))
        
        # Agent usage stats
        agent_stats = await analytics_engine.get_agent_usage_stats(days=min(hours//24, 30))
        
        # Active alerts
        active_alerts = alert_manager.get_active_alerts()
        alert_history = alert_manager.get_alert_history(hours=hours)
        
        # Performance summary
        performance_summary = metrics_collector.get_metrics_summary()
        
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "current": asdict(system_metrics) if system_metrics else None,
                "history": [asdict(m) for m in system_history],
                "summary": system_monitor.get_resource_summary()
            },
            "database": asdict(db_metrics) if db_metrics else None,
            "redis": asdict(redis_metrics) if redis_metrics else None,
            "analytics": asdict(analytics_summary),
            "agents": agent_stats,
            "alerts": {
                "active": [asdict(alert) for alert in active_alerts],
                "history": [asdict(alert) for alert in alert_history],
                "count": len(active_alerts)
            },
            "performance": performance_summary
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring dashboard")

@router.get("/alerts")
async def get_alerts(
    current_user: User = Depends(get_current_user),
    status: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    hours: int = Query(default=24, ge=1, le=168)
):
    """Get alerts with optional filtering"""
    try:
        if status == "active":
            alerts = alert_manager.get_active_alerts()
        else:
            alerts = alert_manager.get_alert_history(hours=hours)
            
        # Filter by severity if specified
        if severity:
            try:
                severity_enum = AlertSeverity(severity.lower())
                alerts = [alert for alert in alerts if alert.severity == severity_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
                
        return {
            "alerts": [asdict(alert) for alert in alerts],
            "count": len(alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alerts")

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user)
):
    """Acknowledge an alert"""
    try:
        alert_manager.acknowledge_alert(alert_id, current_user.id)
        
        logger.info(
            f"Alert acknowledged: {alert_id}",
            alert_id=alert_id,
            user_id=current_user.id,
            operation="acknowledge_alert"
        )
        
        return {"message": "Alert acknowledged successfully"}
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")

@router.post("/alerts/{alert_id}/suppress")
async def suppress_alert(
    alert_id: str,
    duration_minutes: int = Query(default=60, ge=1, le=1440),  # 1 minute to 24 hours
    current_user: User = Depends(get_current_user)
):
    """Suppress an alert for specified duration"""
    try:
        alert_manager.suppress_alert(alert_id, duration_minutes)
        
        logger.info(
            f"Alert suppressed: {alert_id} for {duration_minutes} minutes",
            alert_id=alert_id,
            duration_minutes=duration_minutes,
            user_id=current_user.id,
            operation="suppress_alert"
        )
        
        return {"message": f"Alert suppressed for {duration_minutes} minutes"}
        
    except Exception as e:
        logger.error(f"Error suppressing alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to suppress alert")

@router.get("/analytics")
async def get_analytics_summary(
    current_user: User = Depends(get_current_user),
    days: int = Query(default=30, ge=1, le=90)
):
    """Get analytics summary"""
    try:
        summary = await analytics_engine.get_analytics_summary(days=days)
        agent_stats = await analytics_engine.get_agent_usage_stats(days=days)
        
        return {
            "summary": asdict(summary),
            "agent_usage": agent_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics summary")

@router.get("/analytics/user/{user_id}")
async def get_user_analytics(
    user_id: str,
    current_user: User = Depends(get_current_user),
    days: int = Query(default=30, ge=1, le=90)
):
    """Get user-specific analytics"""
    try:
        # Check if user can access this data (admin or own data)
        if current_user.role != "admin" and current_user.id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
            
        user_journey = await analytics_engine.get_user_journey_analysis(user_id, days=days)
        
        return {
            "user_id": user_id,
            "journey": user_journey,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting user analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user analytics")

@router.get("/system/resources")
async def get_system_resources(
    current_user: User = Depends(get_current_user),
    hours: int = Query(default=1, ge=1, le=24)
):
    """Get system resource metrics"""
    try:
        current_metrics = system_monitor.get_current_metrics()
        history = system_monitor.get_metrics_history(hours=hours)
        process_metrics = system_monitor.get_process_metrics()
        resource_summary = system_monitor.get_resource_summary()
        
        return {
            "current": asdict(current_metrics) if current_metrics else None,
            "history": [asdict(m) for m in history],
            "processes": {str(pid): asdict(metrics) for pid, metrics in process_metrics.items()},
            "summary": resource_summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system resources: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system resources")

@router.get("/database/metrics")
async def get_database_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get database performance metrics"""
    try:
        current_metrics = await database_monitor.collect_metrics()
        
        return {
            "metrics": asdict(current_metrics) if current_metrics else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting database metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get database metrics")

@router.get("/redis/metrics")
async def get_redis_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get Redis cache metrics"""
    try:
        current_metrics = await redis_monitor.collect_metrics()
        
        return {
            "metrics": asdict(current_metrics) if current_metrics else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Redis metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get Redis metrics")

@router.post("/alerts/rules")
async def create_alert_rule(
    rule_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Create a new alert rule"""
    try:
        # Validate user has admin permissions
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
            
        # Create alert rule
        rule = AlertRule(
            name=rule_data["name"],
            metric_name=rule_data["metric_name"],
            condition=rule_data["condition"],
            threshold=float(rule_data["threshold"]),
            severity=AlertSeverity(rule_data["severity"]),
            duration=int(rule_data.get("duration", 0)),
            description=rule_data["description"],
            enabled=rule_data.get("enabled", True),
            tags=rule_data.get("tags", {})
        )
        
        alert_manager.add_rule(rule)
        
        logger.info(
            f"Alert rule created: {rule.name}",
            rule_name=rule.name,
            user_id=current_user.id,
            operation="create_alert_rule"
        )
        
        return {"message": "Alert rule created successfully", "rule": asdict(rule)}
        
    except Exception as e:
        logger.error(f"Error creating alert rule: {e}")
        raise HTTPException(status_code=500, detail="Failed to create alert rule")

@router.delete("/alerts/rules/{rule_name}")
async def delete_alert_rule(
    rule_name: str,
    current_user: User = Depends(get_current_user)
):
    """Delete an alert rule"""
    try:
        # Validate user has admin permissions
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
            
        alert_manager.remove_rule(rule_name)
        
        logger.info(
            f"Alert rule deleted: {rule_name}",
            rule_name=rule_name,
            user_id=current_user.id,
            operation="delete_alert_rule"
        )
        
        return {"message": "Alert rule deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting alert rule: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete alert rule")

@router.get("/logs")
async def get_logs(
    current_user: User = Depends(get_current_user),
    level: Optional[str] = Query(default=None),
    component: Optional[str] = Query(default=None),
    hours: int = Query(default=1, ge=1, le=24),
    limit: int = Query(default=100, ge=1, le=1000)
):
    """Get application logs with filtering"""
    try:
        # Validate user has admin permissions
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
            
        # This would typically read from log files or a log aggregation system
        # For now, return a placeholder response
        return {
            "logs": [],
            "filters": {
                "level": level,
                "component": component,
                "hours": hours,
                "limit": limit
            },
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Log retrieval not yet implemented - would read from structured log files"
        }
        
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get logs")

# Add missing import
from dataclasses import asdict