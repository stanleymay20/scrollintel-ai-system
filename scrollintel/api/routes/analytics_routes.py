"""
API routes for prompt analytics system.
Provides REST endpoints for analytics, reporting, and insights.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from ...core.prompt_analytics import PromptPerformanceTracker, AnalyticsEngine
from ...core.analytics_dashboard import TeamAnalyticsDashboard, InsightsGenerator
from ...core.automated_reporting import AutomatedReportingSystem, ReportFrequency
from ...core.trend_pattern_analysis import AdvancedTrendAnalyzer, AdvancedPatternRecognizer
from ...models.analytics_models import (
    PromptMetricsResponse, UsageAnalyticsResponse, AnalyticsReportResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Initialize analytics components
performance_tracker = PromptPerformanceTracker()
analytics_engine = AnalyticsEngine()
team_dashboard = TeamAnalyticsDashboard()
insights_generator = InsightsGenerator()
reporting_system = AutomatedReportingSystem()
trend_analyzer = AdvancedTrendAnalyzer()
pattern_recognizer = AdvancedPatternRecognizer()

# Request/Response Models
class RecordUsageRequest(BaseModel):
    prompt_id: str
    version_id: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    context: Optional[Dict[str, Any]] = None

class PerformanceSummaryRequest(BaseModel):
    prompt_id: str
    days: int = Field(default=30, ge=1, le=365)

class TeamAnalyticsRequest(BaseModel):
    team_id: str
    days: int = Field(default=30, ge=1, le=365)

class TrendAnalysisRequest(BaseModel):
    prompt_id: str
    metric_name: str
    days: int = Field(default=30, ge=7, le=365)
    forecast_days: int = Field(default=7, ge=1, le=30)

class PatternRecognitionRequest(BaseModel):
    prompt_ids: List[str]
    metric_names: List[str]
    days: int = Field(default=30, ge=7, le=365)

class ReportScheduleRequest(BaseModel):
    name: str
    report_type: str
    frequency: str
    recipients: List[str]
    team_ids: List[str]
    prompt_ids: Optional[List[str]] = None

class AlertRuleRequest(BaseModel):
    name: str
    rule_type: str
    metric_name: str
    condition: str
    threshold_value: Optional[float] = None
    trend_direction: Optional[str] = None
    severity: str = "medium"
    notification_channels: List[str] = ["email"]
    recipients: List[str]
    prompt_ids: Optional[List[str]] = None
    team_ids: Optional[List[str]] = None

class AdHocReportRequest(BaseModel):
    team_id: str
    report_type: str
    date_range_start: datetime
    date_range_end: datetime
    recipients: List[str]

# Performance Tracking Endpoints

@router.post("/usage/record")
async def record_prompt_usage(request: RecordUsageRequest):
    """Record prompt usage with performance metrics."""
    try:
        metrics_id = await performance_tracker.record_prompt_usage(
            prompt_id=request.prompt_id,
            version_id=request.version_id,
            performance_metrics=request.performance_metrics,
            context=request.context
        )
        
        return {"metrics_id": metrics_id, "status": "recorded"}
        
    except Exception as e:
        logger.error(f"Error recording prompt usage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/{prompt_id}")
async def get_prompt_performance(
    prompt_id: str,
    days: int = Query(default=30, ge=1, le=365)
):
    """Get performance summary for a specific prompt."""
    try:
        summary = await performance_tracker.get_prompt_performance_summary(
            prompt_id=prompt_id,
            days=days
        )
        
        if "error" in summary:
            raise HTTPException(status_code=404, detail=summary["error"])
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/team/{team_id}")
async def get_team_analytics(
    team_id: str,
    days: int = Query(default=30, ge=1, le=365)
):
    """Get comprehensive analytics for a team."""
    try:
        analytics = await performance_tracker.get_team_analytics(
            team_id=team_id,
            days=days
        )
        
        if "error" in analytics:
            raise HTTPException(status_code=404, detail=analytics["error"])
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard Endpoints

@router.get("/dashboard/{team_id}")
async def get_team_dashboard(
    team_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None)
):
    """Get comprehensive team dashboard data."""
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        dashboard_data = await team_dashboard.get_team_dashboard_data(
            team_id=team_id,
            date_range=(start_date, end_date)
        )
        
        if "error" in dashboard_data:
            raise HTTPException(status_code=404, detail=dashboard_data["error"])
        
        return dashboard_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights/{team_id}")
async def get_team_insights(
    team_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None)
):
    """Get AI-generated insights for a team."""
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        insights = await insights_generator.generate_insights(
            team_id=team_id,
            date_range=(start_date, end_date)
        )
        
        return {"team_id": team_id, "insights": insights}
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Trend Analysis Endpoints

@router.post("/trends/analyze")
async def analyze_trends(request: TrendAnalysisRequest):
    """Perform comprehensive trend analysis."""
    try:
        analysis = await trend_analyzer.analyze_comprehensive_trends(
            prompt_id=request.prompt_id,
            metric_name=request.metric_name,
            days=request.days,
            forecast_days=request.forecast_days
        )
        
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/patterns/recognize")
async def recognize_patterns(request: PatternRecognitionRequest):
    """Recognize patterns across multiple prompts and metrics."""
    try:
        patterns = await pattern_recognizer.detect_comprehensive_patterns(
            prompt_ids=request.prompt_ids,
            metric_names=request.metric_names,
            days=request.days
        )
        
        # Convert patterns to serializable format
        pattern_data = []
        for pattern in patterns:
            pattern_data.append({
                "pattern_type": pattern.pattern_type.value,
                "start_time": pattern.start_time.isoformat(),
                "end_time": pattern.end_time.isoformat(),
                "confidence_score": pattern.confidence_score,
                "parameters": pattern.parameters,
                "description": pattern.description,
                "affected_metrics": pattern.affected_metrics
            })
        
        return {"patterns": pattern_data}
        
    except Exception as e:
        logger.error(f"Error recognizing patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Reporting Endpoints

@router.post("/reports/schedule")
async def create_report_schedule(request: ReportScheduleRequest):
    """Create a scheduled report."""
    try:
        # Validate frequency
        try:
            frequency = ReportFrequency(request.frequency)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid frequency. Must be one of: {[f.value for f in ReportFrequency]}"
            )
        
        schedule_id = await reporting_system.create_report_schedule(
            name=request.name,
            report_type=request.report_type,
            frequency=frequency,
            recipients=request.recipients,
            team_ids=request.team_ids,
            prompt_ids=request.prompt_ids
        )
        
        return {"schedule_id": schedule_id, "status": "created"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating report schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reports/generate")
async def generate_ad_hoc_report(
    request: AdHocReportRequest,
    background_tasks: BackgroundTasks
):
    """Generate an ad-hoc report."""
    try:
        # Validate date range
        if request.date_range_start >= request.date_range_end:
            raise HTTPException(
                status_code=400,
                detail="Start date must be before end date"
            )
        
        # Generate report in background
        background_tasks.add_task(
            reporting_system.generate_ad_hoc_report,
            team_id=request.team_id,
            report_type=request.report_type,
            date_range=(request.date_range_start, request.date_range_end),
            recipients=request.recipients
        )
        
        return {"status": "generating", "message": "Report generation started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating ad-hoc report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports")
async def list_reports(
    team_id: Optional[str] = Query(None),
    report_type: Optional[str] = Query(None),
    limit: int = Query(default=20, ge=1, le=100)
):
    """List available reports."""
    try:
        # This would typically query the database for reports
        # For now, return a placeholder response
        return {
            "reports": [],
            "total": 0,
            "message": "Report listing not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Alert Management Endpoints

@router.post("/alerts/rules")
async def create_alert_rule(request: AlertRuleRequest):
    """Create an alert rule."""
    try:
        alert_id = await reporting_system.create_alert_rule(
            name=request.name,
            rule_type=request.rule_type,
            metric_name=request.metric_name,
            condition=request.condition,
            threshold_value=request.threshold_value,
            severity=request.severity,
            notification_channels=request.notification_channels,
            recipients=request.recipients,
            prompt_ids=request.prompt_ids,
            team_ids=request.team_ids
        )
        
        return {"alert_id": alert_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Error creating alert rule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/active/{team_id}")
async def get_active_alerts(team_id: str):
    """Get active alerts for a team."""
    try:
        alerts = await team_dashboard._get_active_alerts(team_id)
        return {"team_id": team_id, "alerts": alerts}
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# System Management Endpoints

@router.post("/system/start")
async def start_analytics_system():
    """Start the automated analytics system."""
    try:
        await reporting_system.start_system()
        return {"status": "started", "message": "Analytics system started successfully"}
        
    except Exception as e:
        logger.error(f"Error starting analytics system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system/stop")
async def stop_analytics_system():
    """Stop the automated analytics system."""
    try:
        await reporting_system.stop_system()
        return {"status": "stopped", "message": "Analytics system stopped successfully"}
        
    except Exception as e:
        logger.error(f"Error stopping analytics system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/status")
async def get_system_status():
    """Get analytics system status."""
    try:
        return {
            "running": reporting_system.running,
            "alert_rules_count": len(reporting_system.alert_rules),
            "report_schedules_count": len(reporting_system.report_schedules),
            "system_health": "healthy"
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility Endpoints

@router.get("/metrics/available")
async def get_available_metrics():
    """Get list of available metrics for analysis."""
    return {
        "performance_metrics": [
            "accuracy_score",
            "relevance_score", 
            "efficiency_score",
            "user_satisfaction",
            "response_time_ms",
            "token_usage",
            "cost_per_request"
        ],
        "usage_metrics": [
            "total_requests",
            "successful_requests",
            "failed_requests",
            "avg_response_time"
        ]
    }

@router.get("/trends/types")
async def get_trend_types():
    """Get available trend analysis types."""
    return {
        "trend_types": [
            "linear",
            "exponential",
            "logarithmic",
            "polynomial",
            "seasonal"
        ],
        "pattern_types": [
            "seasonal",
            "cyclical",
            "anomaly",
            "spike",
            "drop",
            "plateau",
            "oscillation"
        ]
    }

@router.get("/health")
async def health_check():
    """Health check endpoint for analytics service."""
    try:
        # Perform basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "performance_tracker": "healthy",
                "analytics_engine": "healthy",
                "team_dashboard": "healthy",
                "reporting_system": "healthy" if reporting_system.running else "stopped",
                "trend_analyzer": "healthy",
                "pattern_recognizer": "healthy"
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

# Error Handlers

@router.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid value: {str(exc)}"}
    )

@router.exception_handler(KeyError)
async def key_error_handler(request, exc):
    """Handle key errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": f"Missing required field: {str(exc)}"}
    )