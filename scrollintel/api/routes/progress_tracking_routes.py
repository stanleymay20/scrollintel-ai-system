"""
Progress Tracking API Routes for Cultural Transformation Leadership

This module provides REST API endpoints for progress tracking functionality.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from ...engines.progress_tracking_engine import ProgressTrackingEngine
from ...models.progress_tracking_models import (
    ProgressReport, ProgressDashboard, TransformationMilestone,
    ProgressMetric, ProgressAlert
)
from ...models.cultural_assessment_models import CulturalTransformation

# Initialize router and engine
router = APIRouter(prefix="/api/v1/progress-tracking", tags=["progress-tracking"])
progress_engine = ProgressTrackingEngine()
logger = logging.getLogger(__name__)


@router.post("/initialize/{transformation_id}")
async def initialize_progress_tracking(
    transformation_id: str,
    transformation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Initialize progress tracking for a cultural transformation"""
    try:
        # Create transformation object from data
        transformation = CulturalTransformation(
            id=transformation_id,
            organization_id=transformation_data.get('organization_id'),
            current_culture=transformation_data.get('current_culture'),
            target_culture=transformation_data.get('target_culture'),
            vision=transformation_data.get('vision'),
            roadmap=transformation_data.get('roadmap'),
            interventions=transformation_data.get('interventions', []),
            progress=0.0,
            start_date=datetime.now(),
            target_completion=datetime.fromisoformat(transformation_data.get('target_completion'))
        )
        
        result = progress_engine.initialize_progress_tracking(transformation)
        
        logger.info(f"Initialized progress tracking for transformation {transformation_id}")
        return {
            "status": "success",
            "message": "Progress tracking initialized successfully",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error initializing progress tracking: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/milestones/{transformation_id}/{milestone_id}")
async def update_milestone_progress(
    transformation_id: str,
    milestone_id: str,
    progress_update: Dict[str, Any]
) -> Dict[str, Any]:
    """Update progress on a specific milestone"""
    try:
        milestone = progress_engine.track_milestone_progress(
            transformation_id, milestone_id, progress_update
        )
        
        logger.info(f"Updated milestone {milestone_id} progress")
        return {
            "status": "success",
            "message": "Milestone progress updated successfully",
            "data": {
                "milestone_id": milestone.id,
                "progress_percentage": milestone.progress_percentage,
                "status": milestone.status.value,
                "completion_date": milestone.completion_date.isoformat() if milestone.completion_date else None
            }
        }
        
    except ValueError as e:
        logger.error(f"Validation error updating milestone: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating milestone progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/metrics/{transformation_id}")
async def update_progress_metrics(
    transformation_id: str,
    metric_updates: Dict[str, float]
) -> Dict[str, Any]:
    """Update progress metrics for a transformation"""
    try:
        updated_metrics = progress_engine.update_progress_metrics(
            transformation_id, metric_updates
        )
        
        metrics_data = []
        for metric in updated_metrics:
            metrics_data.append({
                "id": metric.id,
                "name": metric.name,
                "current_value": metric.current_value,
                "target_value": metric.target_value,
                "completion_percentage": metric.completion_percentage,
                "trend": metric.trend,
                "last_updated": metric.last_updated.isoformat()
            })
        
        logger.info(f"Updated {len(updated_metrics)} metrics for transformation {transformation_id}")
        return {
            "status": "success",
            "message": "Progress metrics updated successfully",
            "data": {
                "updated_metrics": metrics_data,
                "update_count": len(updated_metrics)
            }
        }
        
    except ValueError as e:
        logger.error(f"Validation error updating metrics: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating progress metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/{transformation_id}")
async def get_progress_report(transformation_id: str) -> Dict[str, Any]:
    """Generate comprehensive progress report"""
    try:
        report = progress_engine.generate_progress_report(transformation_id)
        
        # Convert report to API response format
        report_data = {
            "id": report.id,
            "transformation_id": report.transformation_id,
            "report_date": report.report_date.isoformat(),
            "overall_progress": report.overall_progress,
            "milestones": [
                {
                    "id": m.id,
                    "name": m.name,
                    "status": m.status.value,
                    "progress_percentage": m.progress_percentage,
                    "target_date": m.target_date.isoformat(),
                    "completion_date": m.completion_date.isoformat() if m.completion_date else None,
                    "is_overdue": m.is_overdue
                }
                for m in report.milestones
            ],
            "metrics": [
                {
                    "id": m.id,
                    "name": m.name,
                    "current_value": m.current_value,
                    "target_value": m.target_value,
                    "completion_percentage": m.completion_percentage,
                    "trend": m.trend,
                    "category": m.category
                }
                for m in report.metrics
            ],
            "achievements": report.achievements,
            "challenges": report.challenges,
            "next_steps": report.next_steps,
            "risk_indicators": report.risk_indicators,
            "recommendations": report.recommendations
        }
        
        logger.info(f"Generated progress report for transformation {transformation_id}")
        return {
            "status": "success",
            "message": "Progress report generated successfully",
            "data": report_data
        }
        
    except ValueError as e:
        logger.error(f"Validation error generating report: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating progress report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/{transformation_id}")
async def get_progress_dashboard(transformation_id: str) -> Dict[str, Any]:
    """Get progress visualization dashboard"""
    try:
        dashboard = progress_engine.create_progress_dashboard(transformation_id)
        
        # Convert dashboard to API response format
        dashboard_data = {
            "transformation_id": dashboard.transformation_id,
            "dashboard_date": dashboard.dashboard_date.isoformat(),
            "overall_health_score": dashboard.overall_health_score,
            "progress_charts": dashboard.progress_charts,
            "milestone_timeline": dashboard.milestone_timeline,
            "metric_trends": dashboard.metric_trends,
            "alert_indicators": dashboard.alert_indicators,
            "executive_summary": dashboard.executive_summary
        }
        
        logger.info(f"Created progress dashboard for transformation {transformation_id}")
        return {
            "status": "success",
            "message": "Progress dashboard created successfully",
            "data": dashboard_data
        }
        
    except ValueError as e:
        logger.error(f"Validation error creating dashboard: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating progress dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/milestones/{transformation_id}/{milestone_id}/validate")
async def validate_milestone_completion(
    transformation_id: str,
    milestone_id: str
) -> Dict[str, Any]:
    """Validate milestone completion against success criteria"""
    try:
        validation_result = progress_engine.validate_milestone_completion(
            transformation_id, milestone_id
        )
        
        logger.info(f"Validated milestone {milestone_id} completion")
        return {
            "status": "success",
            "message": "Milestone validation completed",
            "data": validation_result
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error validating milestone completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/milestones/{transformation_id}")
async def get_transformation_milestones(transformation_id: str) -> Dict[str, Any]:
    """Get all milestones for a transformation"""
    try:
        if transformation_id not in progress_engine.progress_data:
            raise HTTPException(status_code=404, detail="Transformation not found")
        
        tracking_data = progress_engine.progress_data[transformation_id]
        milestones = list(tracking_data['milestones'].values())
        
        milestones_data = []
        for milestone in milestones:
            milestones_data.append({
                "id": milestone.id,
                "name": milestone.name,
                "description": milestone.description,
                "milestone_type": milestone.milestone_type.value,
                "status": milestone.status.value,
                "progress_percentage": milestone.progress_percentage,
                "target_date": milestone.target_date.isoformat(),
                "completion_date": milestone.completion_date.isoformat() if milestone.completion_date else None,
                "success_criteria": milestone.success_criteria,
                "dependencies": milestone.dependencies,
                "is_overdue": milestone.is_overdue
            })
        
        logger.info(f"Retrieved {len(milestones)} milestones for transformation {transformation_id}")
        return {
            "status": "success",
            "message": "Milestones retrieved successfully",
            "data": {
                "milestones": milestones_data,
                "total_count": len(milestones_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving milestones: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{transformation_id}")
async def get_transformation_metrics(transformation_id: str) -> Dict[str, Any]:
    """Get all metrics for a transformation"""
    try:
        if transformation_id not in progress_engine.progress_data:
            raise HTTPException(status_code=404, detail="Transformation not found")
        
        tracking_data = progress_engine.progress_data[transformation_id]
        metrics = list(tracking_data['metrics'].values())
        
        metrics_data = []
        for metric in metrics:
            metrics_data.append({
                "id": metric.id,
                "name": metric.name,
                "description": metric.description,
                "current_value": metric.current_value,
                "target_value": metric.target_value,
                "unit": metric.unit,
                "category": metric.category,
                "completion_percentage": metric.completion_percentage,
                "trend": metric.trend,
                "last_updated": metric.last_updated.isoformat()
            })
        
        logger.info(f"Retrieved {len(metrics)} metrics for transformation {transformation_id}")
        return {
            "status": "success",
            "message": "Metrics retrieved successfully",
            "data": {
                "metrics": metrics_data,
                "total_count": len(metrics_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/{transformation_id}")
async def get_transformation_alerts(
    transformation_id: str,
    active_only: bool = True
) -> Dict[str, Any]:
    """Get alerts for a transformation"""
    try:
        alerts = progress_engine.alerts.get(transformation_id, [])
        
        if active_only:
            alerts = [alert for alert in alerts if not alert.resolved_date]
        
        alerts_data = []
        for alert in alerts:
            alerts_data.append({
                "id": alert.id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "created_date": alert.created_date.isoformat(),
                "resolved_date": alert.resolved_date.isoformat() if alert.resolved_date else None,
                "action_required": alert.action_required,
                "assigned_to": alert.assigned_to
            })
        
        logger.info(f"Retrieved {len(alerts_data)} alerts for transformation {transformation_id}")
        return {
            "status": "success",
            "message": "Alerts retrieved successfully",
            "data": {
                "alerts": alerts_data,
                "total_count": len(alerts_data),
                "active_count": len([a for a in alerts_data if not a["resolved_date"]])
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{transformation_id}")
async def get_transformation_status(transformation_id: str) -> Dict[str, Any]:
    """Get overall transformation status"""
    try:
        if transformation_id not in progress_engine.progress_data:
            raise HTTPException(status_code=404, detail="Transformation not found")
        
        tracking_data = progress_engine.progress_data[transformation_id]
        
        # Calculate summary statistics
        milestones = list(tracking_data['milestones'].values())
        metrics = list(tracking_data['metrics'].values())
        alerts = progress_engine.alerts.get(transformation_id, [])
        
        completed_milestones = len([m for m in milestones if m.status.value == "completed"])
        overdue_milestones = len([m for m in milestones if m.is_overdue])
        active_alerts = len([a for a in alerts if not a.resolved_date])
        
        avg_metric_completion = sum(m.completion_percentage for m in metrics) / len(metrics) if metrics else 0
        
        status_data = {
            "transformation_id": transformation_id,
            "overall_progress": tracking_data['overall_progress'],
            "start_date": tracking_data['start_date'].isoformat(),
            "status": tracking_data['status'].value,
            "milestones": {
                "total": len(milestones),
                "completed": completed_milestones,
                "overdue": overdue_milestones,
                "completion_rate": (completed_milestones / len(milestones) * 100) if milestones else 0
            },
            "metrics": {
                "total": len(metrics),
                "average_completion": avg_metric_completion
            },
            "alerts": {
                "total": len(alerts),
                "active": active_alerts
            }
        }
        
        logger.info(f"Retrieved status for transformation {transformation_id}")
        return {
            "status": "success",
            "message": "Transformation status retrieved successfully",
            "data": status_data
        }
        
    except Exception as e:
        logger.error(f"Error retrieving transformation status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))