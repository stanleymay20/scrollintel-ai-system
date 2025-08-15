"""
API Routes for Production Monitoring and Alerting System

This module provides REST API endpoints for accessing production monitoring data,
alerts, failure patterns, and comprehensive reports.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from scrollintel.core.production_monitoring import production_monitor, AlertSeverity, MetricType
from scrollintel.core.ux_quality_monitor import ux_quality_monitor, UXMetricType
from scrollintel.core.failure_pattern_detector import failure_pattern_detector, FailureType, PatternSeverity
from scrollintel.core.comprehensive_reporting import comprehensive_reporter, ReportType, ReportFrequency

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/monitoring", tags=["Production Monitoring"])

# Pydantic models for request/response

class MetricRequest(BaseModel):
    user_id: str
    session_id: str
    metric_type: str
    metric_name: str
    value: float
    context: Optional[Dict[str, Any]] = None

class FailureRequest(BaseModel):
    failure_type: str
    component: str
    error_message: str
    stack_trace: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    severity: Optional[str] = "medium"

class UserActionRequest(BaseModel):
    user_id: str
    action: str
    response_time: float
    success: bool
    satisfaction: Optional[float] = None

class ReportRequest(BaseModel):
    report_type: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    filters: Optional[Dict[str, Any]] = None

class AlertResponse(BaseModel):
    id: str
    title: str
    description: str
    severity: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool

class SystemHealthResponse(BaseModel):
    timestamp: str
    health_score: float
    status: str
    metrics: Dict[str, Any]
    active_alerts: int
    alert_breakdown: Dict[str, int]
    failure_patterns: int
    active_sessions: int

# System Health Endpoints

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Get current system health status"""
    try:
        health_data = production_monitor.get_system_health()
        return SystemHealthResponse(**health_data)
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system health")

@router.get("/health/history")
async def get_health_history(
    hours: int = Query(24, ge=1, le=168, description="Hours of history to retrieve")
):
    """Get system health history"""
    try:
        # Get metrics summary for the specified period
        summary = production_monitor.get_metrics_summary(hours=hours)
        return {
            "period_hours": hours,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting health history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get health history")

# Metrics Endpoints

@router.post("/metrics/record")
async def record_metric(request: MetricRequest):
    """Record a system or UX metric"""
    try:
        # Determine if it's a UX metric or system metric
        if request.metric_type in [t.value for t in UXMetricType]:
            ux_metric_type = UXMetricType(request.metric_type)
            ux_quality_monitor.record_ux_metric(
                request.user_id,
                request.session_id,
                ux_metric_type,
                request.metric_name,
                request.value,
                request.context
            )
        else:
            # Record as system metric through production monitor
            production_monitor.record_user_action(
                request.user_id,
                request.metric_name,
                request.value,
                True,  # Assume success for generic metrics
                request.context.get('satisfaction') if request.context else None
            )
        
        return {"status": "success", "message": "Metric recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to record metric")

@router.post("/user-action")
async def record_user_action(request: UserActionRequest):
    """Record a user action for monitoring"""
    try:
        production_monitor.record_user_action(
            request.user_id,
            request.action,
            request.response_time,
            request.success,
            request.satisfaction
        )
        
        return {"status": "success", "message": "User action recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording user action: {e}")
        raise HTTPException(status_code=500, detail="Failed to record user action")

# Alert Endpoints

@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    resolved: Optional[bool] = Query(None, description="Filter by resolution status"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of alerts to return")
):
    """Get system alerts"""
    try:
        alerts = production_monitor.get_alerts(resolved=resolved)
        
        # Filter by severity if specified
        if severity:
            alerts = [alert for alert in alerts if alert.get('severity') == severity]
        
        # Limit results
        alerts = alerts[:limit]
        
        # Convert to response model
        alert_responses = []
        for alert in alerts:
            alert_responses.append(AlertResponse(
                id=alert['id'],
                title=alert['title'],
                description=alert['description'],
                severity=alert['severity'],
                metric_name=alert['metric_name'],
                current_value=alert['current_value'],
                threshold=alert['threshold'],
                timestamp=alert['timestamp'],
                resolved=alert['resolved']
            ))
        
        return alert_responses
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alerts")

@router.get("/alerts/summary")
async def get_alerts_summary():
    """Get alerts summary"""
    try:
        all_alerts = production_monitor.get_alerts()
        active_alerts = [alert for alert in all_alerts if not alert['resolved']]
        
        # Count by severity
        severity_counts = {}
        for alert in active_alerts:
            severity = alert['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_alerts": len(all_alerts),
            "active_alerts": len(active_alerts),
            "resolved_alerts": len(all_alerts) - len(active_alerts),
            "severity_breakdown": severity_counts,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting alerts summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alerts summary")

# Failure Pattern Endpoints

@router.post("/failures/record")
async def record_failure(request: FailureRequest):
    """Record a failure event"""
    try:
        failure_type = FailureType(request.failure_type)
        severity = PatternSeverity(request.severity)
        
        event_id = failure_pattern_detector.record_failure(
            failure_type,
            request.component,
            request.error_message,
            request.stack_trace,
            request.user_id,
            request.session_id,
            request.request_id,
            request.context,
            severity
        )
        
        return {
            "status": "success",
            "event_id": event_id,
            "message": "Failure recorded successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid failure type or severity: {e}")
    except Exception as e:
        logger.error(f"Error recording failure: {e}")
        raise HTTPException(status_code=500, detail="Failed to record failure")

@router.get("/failures/patterns")
async def get_failure_patterns(
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    component: Optional[str] = Query(None, description="Filter by component"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of patterns to return")
):
    """Get detected failure patterns"""
    try:
        severity_filter = PatternSeverity(severity) if severity else None
        patterns = failure_pattern_detector.get_detected_patterns(severity=severity_filter)
        
        # Filter by component if specified
        if component:
            patterns = [p for p in patterns if p.get('component') == component]
        
        # Limit results
        patterns = patterns[:limit]
        
        return {
            "patterns": patterns,
            "total_count": len(patterns),
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid severity: {e}")
    except Exception as e:
        logger.error(f"Error getting failure patterns: {e}")
        raise HTTPException(status_code=500, detail="Failed to get failure patterns")

@router.get("/failures/component-health")
async def get_component_health(component: Optional[str] = Query(None, description="Specific component to check")):
    """Get component health information"""
    try:
        health_data = failure_pattern_detector.get_component_health(component=component)
        return {
            "component_health": health_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting component health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get component health")

@router.get("/failures/prevention-status")
async def get_prevention_status():
    """Get failure prevention system status"""
    try:
        status = failure_pattern_detector.get_prevention_status()
        return {
            "prevention_status": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting prevention status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get prevention status")

# User Experience Endpoints

@router.get("/ux/dashboard")
async def get_ux_dashboard():
    """Get UX quality dashboard data"""
    try:
        dashboard_data = ux_quality_monitor.get_ux_dashboard()
        return dashboard_data
    except Exception as e:
        logger.error(f"Error getting UX dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get UX dashboard")

@router.get("/ux/optimizations")
async def get_ux_optimizations():
    """Get UX optimization recommendations"""
    try:
        optimizations = ux_quality_monitor.get_optimization_recommendations()
        return {
            "optimizations": optimizations,
            "count": len(optimizations),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting UX optimizations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get UX optimizations")

@router.post("/ux/session/start")
async def start_user_session(
    user_id: str,
    session_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
):
    """Start tracking a user session"""
    try:
        session_id = ux_quality_monitor.start_user_session(user_id, session_id, context)
        return {
            "status": "success",
            "session_id": session_id,
            "message": "User session started"
        }
    except Exception as e:
        logger.error(f"Error starting user session: {e}")
        raise HTTPException(status_code=500, detail="Failed to start user session")

@router.post("/ux/session/{session_id}/end")
async def end_user_session(session_id: str, final_satisfaction: Optional[float] = None):
    """End a user session"""
    try:
        ux_quality_monitor.end_user_session(session_id, final_satisfaction)
        return {
            "status": "success",
            "message": "User session ended"
        }
    except Exception as e:
        logger.error(f"Error ending user session: {e}")
        raise HTTPException(status_code=500, detail="Failed to end user session")

@router.get("/ux/sessions/summary")
async def get_user_sessions_summary(user_id: Optional[str] = Query(None, description="Filter by user ID")):
    """Get user sessions summary"""
    try:
        summary = ux_quality_monitor.get_user_session_summary(user_id=user_id)
        return summary
    except Exception as e:
        logger.error(f"Error getting user sessions summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user sessions summary")

# Reporting Endpoints

@router.post("/reports/generate")
async def generate_report(request: ReportRequest):
    """Generate a comprehensive report"""
    try:
        report_type = ReportType(request.report_type)
        
        # Set default time period if not provided
        if not request.start_time or not request.end_time:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            time_period = (start_time, end_time)
        else:
            time_period = (request.start_time, request.end_time)
        
        report = await comprehensive_reporter.generate_report(
            report_type,
            time_period,
            request.filters
        )
        
        return {
            "status": "success",
            "report_id": report.id,
            "message": "Report generated successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid report type: {e}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate report")

@router.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Get a specific report"""
    try:
        report = comprehensive_reporter.get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report: {e}")
        raise HTTPException(status_code=500, detail="Failed to get report")

@router.get("/reports")
async def list_reports(
    report_type: Optional[str] = Query(None, description="Filter by report type"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of reports to return")
):
    """List generated reports"""
    try:
        report_type_filter = ReportType(report_type) if report_type else None
        reports = comprehensive_reporter.list_reports(report_type=report_type_filter, limit=limit)
        
        return {
            "reports": reports,
            "count": len(reports),
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid report type: {e}")
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(status_code=500, detail="Failed to list reports")

@router.get("/reports/schedules")
async def get_report_schedules():
    """Get report schedules"""
    try:
        schedules = comprehensive_reporter.get_report_schedules()
        return {
            "schedules": schedules,
            "count": len(schedules),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting report schedules: {e}")
        raise HTTPException(status_code=500, detail="Failed to get report schedules")

@router.put("/reports/schedules/{schedule_id}")
async def update_report_schedule(schedule_id: str, updates: Dict[str, Any]):
    """Update a report schedule"""
    try:
        success = comprehensive_reporter.update_report_schedule(schedule_id, updates)
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return {
            "status": "success",
            "message": "Schedule updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating report schedule: {e}")
        raise HTTPException(status_code=500, detail="Failed to update report schedule")

@router.get("/insights/summary")
async def get_insights_summary(days: int = Query(7, ge=1, le=30, description="Days of insights to analyze")):
    """Get summary of insights from recent reports"""
    try:
        summary = comprehensive_reporter.get_insights_summary(days=days)
        return summary
    except Exception as e:
        logger.error(f"Error getting insights summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get insights summary")

# System Control Endpoints

@router.post("/system/start")
async def start_monitoring_system():
    """Start the production monitoring system"""
    try:
        # Start all monitoring components
        await production_monitor.start_monitoring()
        await ux_quality_monitor.start_monitoring()
        await failure_pattern_detector.start_detection()
        await comprehensive_reporter.start_reporting()
        
        return {
            "status": "success",
            "message": "Production monitoring system started"
        }
    except Exception as e:
        logger.error(f"Error starting monitoring system: {e}")
        raise HTTPException(status_code=500, detail="Failed to start monitoring system")

@router.post("/system/stop")
async def stop_monitoring_system():
    """Stop the production monitoring system"""
    try:
        # Stop all monitoring components
        await production_monitor.stop_monitoring()
        await ux_quality_monitor.stop_monitoring()
        await failure_pattern_detector.stop_detection()
        await comprehensive_reporter.stop_reporting()
        
        return {
            "status": "success",
            "message": "Production monitoring system stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping monitoring system: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop monitoring system")

@router.get("/system/status")
async def get_system_status():
    """Get monitoring system status"""
    try:
        return {
            "production_monitor_active": production_monitor.monitoring_active,
            "ux_monitor_active": ux_quality_monitor.monitoring_active,
            "pattern_detector_active": failure_pattern_detector.detection_active,
            "reporter_active": comprehensive_reporter.reporting_active,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")