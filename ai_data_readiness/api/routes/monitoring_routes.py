"""API routes for platform monitoring and metrics."""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from ...core.platform_monitor import get_platform_monitor
from ...core.resource_optimizer import get_resource_optimizer
from ...models.monitoring_models import (
    Alert, AlertSeverity, HealthStatus, MonitoringReport,
    CapacityPlan, MetricDefinition, ALL_METRICS
)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def get_health_status():
    """Get overall platform health status."""
    try:
        monitor = get_platform_monitor()
        health_status = monitor.get_health_status()
        
        return JSONResponse(
            status_code=200 if health_status['status'] == 'healthy' else 503,
            content=health_status
        )
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/current")
async def get_current_metrics():
    """Get current system and platform metrics."""
    try:
        monitor = get_platform_monitor()
        optimizer = get_resource_optimizer()
        
        system_metrics = monitor.get_current_system_metrics()
        platform_metrics = monitor.get_current_platform_metrics()
        resource_usage = optimizer.get_current_resource_usage()
        efficiency_metrics = optimizer.get_resource_efficiency_metrics()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_metrics": system_metrics.to_dict() if system_metrics else None,
            "platform_metrics": platform_metrics.to_dict() if platform_metrics else None,
            "resource_usage": resource_usage.to_dict() if resource_usage else None,
            "efficiency_metrics": efficiency_metrics
        }
    except Exception as e:
        logger.error(f"Error getting current metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/history")
async def get_metrics_history(
    hours: int = Query(default=1, ge=1, le=168, description="Hours of history to retrieve"),
    metric_type: Optional[str] = Query(default=None, description="Filter by metric type")
):
    """Get historical metrics data."""
    try:
        monitor = get_platform_monitor()
        optimizer = get_resource_optimizer()
        
        metrics_history = monitor.get_metrics_history(hours=hours)
        resource_history = optimizer.get_resource_history(hours=hours)
        
        response = {
            "period_hours": hours,
            "system_metrics": metrics_history.get("system_metrics", []),
            "platform_metrics": metrics_history.get("platform_metrics", []),
            "performance_metrics": metrics_history.get("performance_metrics", []),
            "resource_usage": resource_history
        }
        
        if metric_type:
            # Filter by metric type if specified
            filtered_response = {"period_hours": hours}
            if metric_type in response:
                filtered_response[metric_type] = response[metric_type]
            response = filtered_response
        
        return response
    except Exception as e:
        logger.error(f"Error getting metrics history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(
    severity: Optional[AlertSeverity] = Query(default=None, description="Filter by severity"),
    active_only: bool = Query(default=True, description="Show only active alerts"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of alerts")
):
    """Get system alerts."""
    try:
        # This would typically query a database of alerts
        # For now, we'll generate sample alerts based on current metrics
        monitor = get_platform_monitor()
        health_status = monitor.get_health_status()
        
        alerts = []
        
        if health_status['status'] in ['warning', 'critical']:
            alert_severity = AlertSeverity.WARNING if health_status['status'] == 'warning' else AlertSeverity.CRITICAL
            
            for issue in health_status.get('issues', []):
                alert = Alert(
                    severity=alert_severity,
                    title=f"System Health Issue",
                    description=issue,
                    created_at=datetime.utcnow()
                )
                alerts.append(alert.to_dict())
        
        # Filter by severity if specified
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity.value]
        
        # Filter by active status
        if active_only:
            alerts = [a for a in alerts if a['is_active']]
        
        # Apply limit
        alerts = alerts[:limit]
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "active_count": len([a for a in alerts if a['is_active']]),
            "critical_count": len([a for a in alerts if a['severity'] == 'critical'])
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    user: str = Query(..., description="User acknowledging the alert")
):
    """Acknowledge an alert."""
    try:
        # In a real implementation, this would update the alert in the database
        return {
            "alert_id": alert_id,
            "acknowledged_by": user,
            "acknowledged_at": datetime.utcnow().isoformat(),
            "status": "acknowledged"
        }
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/benchmarks")
async def get_performance_benchmarks(
    operation_type: Optional[str] = Query(default=None, description="Filter by operation type"),
    days: int = Query(default=7, ge=1, le=90, description="Days of benchmark data")
):
    """Get performance benchmark data."""
    try:
        # This would typically query benchmark results from database
        # For now, return sample data structure
        return {
            "benchmarks": [],
            "summary": {
                "avg_throughput_ops_per_sec": 0.0,
                "avg_response_time_ms": 0.0,
                "avg_cpu_utilization": 0.0,
                "avg_memory_utilization": 0.0
            },
            "period_days": days
        }
    except Exception as e:
        logger.error(f"Error getting performance benchmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimization/recommendations")
async def get_optimization_recommendations(
    category: Optional[str] = Query(default=None, description="Filter by category (cpu, memory, disk, network)")
):
    """Get resource optimization recommendations."""
    try:
        optimizer = get_resource_optimizer()
        recommendations = optimizer.get_optimization_recommendations(category=category)
        
        return {
            "recommendations": recommendations,
            "total_count": len(recommendations),
            "categories": list(set(r['category'] for r in recommendations)),
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capacity/planning")
async def get_capacity_planning(
    component: Optional[str] = Query(default=None, description="Component to analyze (cpu, memory, storage, network)"),
    time_horizon_days: int = Query(default=30, ge=7, le=365, description="Planning time horizon in days")
):
    """Get capacity planning recommendations."""
    try:
        monitor = get_platform_monitor()
        optimizer = get_resource_optimizer()
        
        # Get current resource utilization
        current_metrics = monitor.get_current_system_metrics()
        efficiency_metrics = optimizer.get_resource_efficiency_metrics()
        
        if not current_metrics:
            raise HTTPException(status_code=404, detail="No current metrics available")
        
        # Generate capacity plans for different components
        capacity_plans = []
        
        components = ['cpu', 'memory', 'storage', 'network'] if not component else [component]
        
        for comp in components:
            if comp == 'cpu':
                current_util = current_metrics.cpu_percent / 100
                projected_util = min(1.0, current_util * 1.2)  # Simple projection
                recommendation = "Monitor CPU usage trends" if projected_util < 0.8 else "Consider CPU upgrade"
            elif comp == 'memory':
                current_util = current_metrics.memory_percent / 100
                projected_util = min(1.0, current_util * 1.15)
                recommendation = "Memory usage stable" if projected_util < 0.85 else "Consider memory upgrade"
            elif comp == 'storage':
                current_util = current_metrics.disk_usage_percent / 100
                projected_util = min(1.0, current_util * 1.1)
                recommendation = "Storage usage stable" if projected_util < 0.9 else "Consider storage expansion"
            else:  # network
                current_util = 0.3  # Placeholder
                projected_util = 0.35
                recommendation = "Network capacity adequate"
            
            plan = CapacityPlan(
                component=comp,
                current_utilization=current_util,
                projected_utilization=projected_util,
                time_horizon_days=time_horizon_days,
                recommendation=recommendation,
                confidence_level=0.75
            )
            capacity_plans.append(plan.to_dict())
        
        return {
            "capacity_plans": capacity_plans,
            "time_horizon_days": time_horizon_days,
            "generated_at": datetime.utcnow().isoformat(),
            "methodology": "Linear projection based on current trends"
        }
    except Exception as e:
        logger.error(f"Error getting capacity planning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/generate")
async def generate_monitoring_report(
    report_type: str = Query(default="daily", description="Report type (daily, weekly, monthly)"),
    include_recommendations: bool = Query(default=True, description="Include optimization recommendations")
):
    """Generate a comprehensive monitoring report."""
    try:
        monitor = get_platform_monitor()
        optimizer = get_resource_optimizer()
        
        # Determine time period based on report type
        if report_type == "daily":
            period_hours = 24
        elif report_type == "weekly":
            period_hours = 168
        elif report_type == "monthly":
            period_hours = 720
        else:
            raise HTTPException(status_code=400, detail="Invalid report type")
        
        # Get metrics for the period
        metrics_history = monitor.get_metrics_history(hours=period_hours)
        
        # Calculate summary statistics
        platform_metrics = metrics_history.get("platform_metrics", [])
        system_metrics = metrics_history.get("system_metrics", [])
        
        if not platform_metrics or not system_metrics:
            raise HTTPException(status_code=404, detail="Insufficient data for report generation")
        
        # Create monitoring report
        report = MonitoringReport(
            report_type=report_type,
            period_start=datetime.utcnow() - timedelta(hours=period_hours),
            period_end=datetime.utcnow(),
            total_datasets_processed=sum(m.get('active_datasets', 0) for m in platform_metrics),
            avg_processing_time_seconds=sum(m.get('avg_processing_time_seconds', 0) for m in platform_metrics) / len(platform_metrics),
            avg_cpu_utilization=sum(m.get('cpu_percent', 0) for m in system_metrics) / len(system_metrics),
            avg_memory_utilization=sum(m.get('memory_percent', 0) for m in system_metrics) / len(system_metrics),
            error_count=sum(m.get('failed_datasets', 0) for m in platform_metrics)
        )
        
        # Add recommendations if requested
        if include_recommendations:
            recommendations = optimizer.get_optimization_recommendations()
            report.optimization_recommendations = [r['title'] for r in recommendations[:5]]
        
        return report.to_dict()
    except Exception as e:
        logger.error(f"Error generating monitoring report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/definitions")
async def get_metric_definitions():
    """Get all available metric definitions."""
    try:
        return {
            "metrics": [metric.to_dict() for metric in ALL_METRICS],
            "total_count": len(ALL_METRICS),
            "categories": {
                "system": len([m for m in ALL_METRICS if m.type.value == "system"]),
                "platform": len([m for m in ALL_METRICS if m.type.value == "platform"]),
                "performance": len([m for m in ALL_METRICS if m.type.value == "performance"])
            }
        }
    except Exception as e:
        logger.error(f"Error getting metric definitions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/start")
async def start_monitoring(
    interval_seconds: int = Query(default=60, ge=10, le=3600, description="Monitoring interval in seconds")
):
    """Start platform monitoring."""
    try:
        monitor = get_platform_monitor()
        optimizer = get_resource_optimizer()
        
        monitor.start_monitoring(interval_seconds=interval_seconds)
        optimizer.start_optimization(interval_seconds=interval_seconds * 5)  # Less frequent optimization
        
        return {
            "status": "started",
            "monitoring_interval_seconds": interval_seconds,
            "optimization_interval_seconds": interval_seconds * 5,
            "started_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/stop")
async def stop_monitoring():
    """Stop platform monitoring."""
    try:
        monitor = get_platform_monitor()
        optimizer = get_resource_optimizer()
        
        monitor.stop_monitoring()
        optimizer.stop_optimization()
        
        return {
            "status": "stopped",
            "stopped_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/export")
async def export_metrics(
    hours: int = Query(default=24, ge=1, le=168, description="Hours of data to export"),
    format: str = Query(default="json", description="Export format (json, csv)")
):
    """Export metrics data."""
    try:
        monitor = get_platform_monitor()
        
        if format == "json":
            # Export to temporary file and return file info
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                filepath = f.name
            
            monitor.export_metrics(filepath, hours=hours)
            
            return {
                "export_file": filepath,
                "format": format,
                "hours": hours,
                "exported_at": datetime.utcnow().isoformat(),
                "message": f"Metrics exported to {filepath}"
            }
        else:
            raise HTTPException(status_code=400, detail="Only JSON format is currently supported")
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))