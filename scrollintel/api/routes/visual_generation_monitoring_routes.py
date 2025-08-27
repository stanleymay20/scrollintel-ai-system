"""
API routes for visual generation monitoring and observability.
"""

from fastapi import APIRouter, HTTPException, Query, Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from scrollintel.engines.visual_generation.monitoring import (
    metrics_collector,
    distributed_tracer,
    observability_dashboard
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/visual-generation/monitoring", tags=["Visual Generation Monitoring"])


@router.get("/dashboard")
async def get_dashboard_data():
    """Get comprehensive dashboard data for visual generation monitoring."""
    try:
        dashboard_data = await observability_dashboard.get_dashboard_data()
        return {
            "success": True,
            "data": dashboard_data
        }
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/performance")
async def get_performance_metrics():
    """Get current performance metrics."""
    try:
        performance = metrics_collector.get_performance_metrics()
        return {
            "success": True,
            "data": {
                "total_requests": performance.total_requests,
                "successful_requests": performance.successful_requests,
                "failed_requests": performance.failed_requests,
                "average_response_time": performance.average_response_time,
                "p95_response_time": performance.p95_response_time,
                "p99_response_time": performance.p99_response_time,
                "requests_per_second": performance.requests_per_second,
                "average_quality_score": performance.average_quality_score,
                "error_rate": performance.error_rate,
                "cpu_usage": performance.cpu_usage,
                "memory_usage": performance.memory_usage,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/summary")
async def get_metrics_summary():
    """Get summary of all collected metrics."""
    try:
        summary = metrics_collector.get_metrics_summary()
        return {
            "success": True,
            "data": summary
        }
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/trends")
async def get_performance_trends(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to look back")
):
    """Get performance trends over time."""
    try:
        trends = await observability_dashboard.get_performance_trends(hours)
        return {
            "success": True,
            "data": trends
        }
    except Exception as e:
        logger.error(f"Error getting performance trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces")
async def search_traces(
    service_name: Optional[str] = Query(None, description="Filter by service name"),
    operation_name: Optional[str] = Query(None, description="Filter by operation name"),
    has_errors: Optional[bool] = Query(None, description="Filter traces with errors"),
    min_duration: Optional[float] = Query(None, ge=0, description="Minimum duration in seconds"),
    max_duration: Optional[float] = Query(None, ge=0, description="Maximum duration in seconds"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of traces to return")
):
    """Search traces based on criteria."""
    try:
        traces = distributed_tracer.collector.search_traces(
            service_name=service_name,
            operation_name=operation_name,
            has_errors=has_errors,
            min_duration=min_duration,
            max_duration=max_duration,
            limit=limit
        )
        
        return {
            "success": True,
            "data": {
                "traces": traces,
                "total_count": len(traces),
                "filters": {
                    "service_name": service_name,
                    "operation_name": operation_name,
                    "has_errors": has_errors,
                    "min_duration": min_duration,
                    "max_duration": max_duration
                }
            }
        }
    except Exception as e:
        logger.error(f"Error searching traces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces/{trace_id}")
async def get_trace_details(
    trace_id: str = Path(..., description="Trace ID to analyze")
):
    """Get detailed analysis of a specific trace."""
    try:
        trace_analysis = await observability_dashboard.get_trace_analysis(trace_id)
        
        if not trace_analysis:
            raise HTTPException(status_code=404, detail="Trace not found")
        
        return {
            "success": True,
            "data": trace_analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trace details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/errors/analysis")
async def get_error_analysis(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to analyze")
):
    """Get error analysis and patterns."""
    try:
        error_analysis = await observability_dashboard.get_error_analysis(hours)
        return {
            "success": True,
            "data": error_analysis
        }
    except Exception as e:
        logger.error(f"Error getting error analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_health_status():
    """Get health status of all monitored components."""
    try:
        health_checks = list(observability_dashboard.health_status.values())
        
        overall_status = "healthy"
        if any(check.status == "unhealthy" for check in health_checks):
            overall_status = "unhealthy"
        elif any(check.status == "degraded" for check in health_checks):
            overall_status = "degraded"
        
        return {
            "success": True,
            "data": {
                "overall_status": overall_status,
                "components": [
                    {
                        "component": check.component,
                        "status": check.status,
                        "message": check.message,
                        "timestamp": check.timestamp.isoformat(),
                        "response_time": check.response_time,
                        "details": check.details
                    }
                    for check in health_checks
                ],
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(
    active_only: bool = Query(False, description="Return only active alerts")
):
    """Get monitoring alerts."""
    try:
        if active_only:
            alerts = list(observability_dashboard.active_alerts.values())
        else:
            # Return recent alerts from history
            recent_alerts = [
                alert for alert in observability_dashboard.alert_history
                if alert.timestamp > datetime.now() - timedelta(hours=24)
            ]
            alerts = recent_alerts
        
        return {
            "success": True,
            "data": {
                "alerts": [
                    {
                        "id": alert.id,
                        "name": alert.name,
                        "description": alert.description,
                        "severity": alert.severity.value,
                        "metric_name": alert.metric_name,
                        "threshold": alert.threshold,
                        "current_value": alert.current_value,
                        "timestamp": alert.timestamp.isoformat(),
                        "resolved": alert.resolved,
                        "resolution_time": alert.resolution_time.isoformat() if alert.resolution_time else None
                    }
                    for alert in alerts
                ],
                "active_count": len(observability_dashboard.active_alerts),
                "total_count": len(alerts)
            }
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capacity/planning")
async def get_capacity_planning():
    """Get capacity planning data and recommendations."""
    try:
        capacity_data = await observability_dashboard.get_capacity_planning_data()
        return {
            "success": True,
            "data": capacity_data
        }
    except Exception as e:
        logger.error(f"Error getting capacity planning data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/custom")
async def record_custom_metric(
    name: str,
    value: float,
    metric_type: str,
    labels: Optional[Dict[str, str]] = None,
    unit: str = "",
    description: str = ""
):
    """Record a custom metric."""
    try:
        from scrollintel.engines.visual_generation.monitoring.metrics_collector import MetricType
        
        # Validate metric type
        try:
            mt = MetricType(metric_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid metric type: {metric_type}")
        
        # Record the metric based on type
        if mt == MetricType.COUNTER:
            metrics_collector.increment_counter(name, value, labels)
        elif mt == MetricType.GAUGE:
            metrics_collector.set_gauge(name, value, labels)
        elif mt == MetricType.HISTOGRAM:
            metrics_collector.record_histogram(name, value, labels)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported metric type for manual recording: {metric_type}")
        
        return {
            "success": True,
            "message": f"Recorded {metric_type} metric '{name}' with value {value}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording custom metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/traces/start")
async def start_trace(
    operation_name: str,
    tags: Optional[Dict[str, Any]] = None
):
    """Start a new trace (for testing/debugging)."""
    try:
        context = distributed_tracer.start_trace(operation_name, **(tags or {}))
        
        return {
            "success": True,
            "data": {
                "trace_id": context.trace_id,
                "span_id": context.span_id,
                "operation_name": operation_name
            }
        }
    except Exception as e:
        logger.error(f"Error starting trace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/status")
async def get_system_status():
    """Get overall system status and key metrics."""
    try:
        performance = metrics_collector.get_performance_metrics()
        health_checks = list(observability_dashboard.health_status.values())
        active_alerts = len(observability_dashboard.active_alerts)
        
        # Determine overall system status
        system_status = "healthy"
        if active_alerts > 0:
            critical_alerts = [
                alert for alert in observability_dashboard.active_alerts.values()
                if alert.severity.value in ["error", "critical"]
            ]
            if critical_alerts:
                system_status = "critical"
            else:
                system_status = "warning"
        
        # Check if any components are unhealthy
        unhealthy_components = [check for check in health_checks if check.status == "unhealthy"]
        if unhealthy_components:
            system_status = "critical"
        
        return {
            "success": True,
            "data": {
                "system_status": system_status,
                "performance_summary": {
                    "requests_per_second": performance.requests_per_second,
                    "average_response_time": performance.average_response_time,
                    "error_rate": performance.error_rate,
                    "success_rate": (performance.successful_requests / max(performance.total_requests, 1)) * 100
                },
                "resource_usage": {
                    "cpu_usage": performance.cpu_usage,
                    "memory_usage": performance.memory_usage
                },
                "alerts": {
                    "active_count": active_alerts,
                    "critical_count": len([
                        alert for alert in observability_dashboard.active_alerts.values()
                        if alert.severity.value == "critical"
                    ])
                },
                "components": {
                    "total": len(health_checks),
                    "healthy": len([check for check in health_checks if check.status == "healthy"]),
                    "degraded": len([check for check in health_checks if check.status == "degraded"]),
                    "unhealthy": len(unhealthy_components)
                },
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))