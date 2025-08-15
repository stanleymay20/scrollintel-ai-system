"""
Performance Monitoring API Routes for Crisis Leadership Excellence

This module provides REST API endpoints for real-time team performance tracking,
issue identification, and optimization during crisis situations.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from datetime import datetime
import logging

from ...engines.performance_monitoring_engine import PerformanceMonitoringEngine
from ...models.performance_monitoring_models import (
    TeamPerformanceOverview, PerformanceIssue, PerformanceIntervention,
    SupportProvision, PerformanceOptimization, PerformanceAlert,
    PerformanceReport, InterventionType, SupportType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/performance-monitoring", tags=["performance-monitoring"])

# Global engine instance
performance_engine = PerformanceMonitoringEngine()


@router.post("/track-team/{crisis_id}")
async def track_team_performance(
    crisis_id: str,
    team_members: List[str]
) -> TeamPerformanceOverview:
    """Track real-time performance of crisis team members"""
    try:
        if not team_members:
            raise HTTPException(status_code=400, detail="Team members list cannot be empty")
        
        team_overview = await performance_engine.track_team_performance(crisis_id, team_members)
        
        logger.info(f"Team performance tracked for crisis {crisis_id}")
        return team_overview
        
    except Exception as e:
        logger.error(f"Error tracking team performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to track team performance: {str(e)}")


@router.get("/performance/{crisis_id}")
async def get_team_performance(crisis_id: str) -> TeamPerformanceOverview:
    """Get current team performance overview"""
    try:
        if crisis_id not in performance_engine.performance_data:
            raise HTTPException(status_code=404, detail=f"No performance data found for crisis {crisis_id}")
        
        team_overview = performance_engine.performance_data[crisis_id]
        
        logger.info(f"Retrieved team performance for crisis {crisis_id}")
        return team_overview
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving team performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve team performance: {str(e)}")


@router.get("/issues/{crisis_id}")
async def identify_performance_issues(crisis_id: str) -> List[PerformanceIssue]:
    """Identify performance issues requiring intervention"""
    try:
        issues = await performance_engine.identify_performance_issues(crisis_id)
        
        logger.info(f"Identified {len(issues)} performance issues for crisis {crisis_id}")
        return issues
        
    except Exception as e:
        logger.error(f"Error identifying performance issues: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to identify performance issues: {str(e)}")


@router.post("/intervention/{crisis_id}/{member_id}")
async def implement_intervention(
    crisis_id: str,
    member_id: str,
    intervention_type: InterventionType
) -> PerformanceIntervention:
    """Implement performance intervention for team member"""
    try:
        intervention = await performance_engine.implement_intervention(
            crisis_id, member_id, intervention_type
        )
        
        logger.info(f"Implemented {intervention_type.value} intervention for member {member_id}")
        return intervention
        
    except Exception as e:
        logger.error(f"Error implementing intervention: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to implement intervention: {str(e)}")


@router.post("/support/{crisis_id}/{member_id}")
async def provide_support(
    crisis_id: str,
    member_id: str,
    support_type: SupportType,
    provider: str = Query(..., description="Support provider identifier")
) -> SupportProvision:
    """Provide support to team member"""
    try:
        support = await performance_engine.provide_support(
            crisis_id, member_id, support_type, provider
        )
        
        logger.info(f"Provided {support_type.value} support to member {member_id}")
        return support
        
    except Exception as e:
        logger.error(f"Error providing support: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to provide support: {str(e)}")


@router.get("/optimization/{crisis_id}")
async def get_performance_optimization(crisis_id: str) -> List[PerformanceOptimization]:
    """Get performance optimization recommendations"""
    try:
        optimizations = await performance_engine.optimize_team_performance(crisis_id)
        
        logger.info(f"Generated {len(optimizations)} optimization recommendations for crisis {crisis_id}")
        return optimizations
        
    except Exception as e:
        logger.error(f"Error generating optimization recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate optimization recommendations: {str(e)}")


@router.get("/alerts/{crisis_id}")
async def get_performance_alerts(crisis_id: str) -> List[PerformanceAlert]:
    """Get performance monitoring alerts"""
    try:
        alerts = await performance_engine.generate_performance_alerts(crisis_id)
        
        logger.info(f"Generated {len(alerts)} performance alerts for crisis {crisis_id}")
        return alerts
        
    except Exception as e:
        logger.error(f"Error generating performance alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate performance alerts: {str(e)}")


@router.post("/alerts/{crisis_id}/{alert_id}/acknowledge")
async def acknowledge_alert(crisis_id: str, alert_id: str) -> dict:
    """Acknowledge a performance alert"""
    try:
        if crisis_id not in performance_engine.active_alerts:
            raise HTTPException(status_code=404, detail=f"No alerts found for crisis {crisis_id}")
        
        alerts = performance_engine.active_alerts[crisis_id]
        alert = next((a for a in alerts if a.alert_id == alert_id), None)
        
        if not alert:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        alert.acknowledged_at = datetime.now()
        
        logger.info(f"Acknowledged alert {alert_id} for crisis {crisis_id}")
        return {"message": "Alert acknowledged successfully", "acknowledged_at": alert.acknowledged_at}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")


@router.get("/report/{crisis_id}")
async def generate_performance_report(
    crisis_id: str,
    time_period_hours: int = Query(24, description="Time period for report in hours")
) -> PerformanceReport:
    """Generate comprehensive performance report"""
    try:
        if time_period_hours <= 0 or time_period_hours > 168:  # Max 1 week
            raise HTTPException(status_code=400, detail="Time period must be between 1 and 168 hours")
        
        report = await performance_engine.generate_performance_report(crisis_id, time_period_hours)
        
        logger.info(f"Generated performance report for crisis {crisis_id}")
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate performance report: {str(e)}")


@router.get("/interventions/{crisis_id}")
async def get_intervention_history(crisis_id: str) -> List[PerformanceIntervention]:
    """Get intervention history for crisis"""
    try:
        if crisis_id not in performance_engine.intervention_history:
            return []
        
        interventions = performance_engine.intervention_history[crisis_id]
        
        logger.info(f"Retrieved {len(interventions)} interventions for crisis {crisis_id}")
        return interventions
        
    except Exception as e:
        logger.error(f"Error retrieving intervention history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve intervention history: {str(e)}")


@router.get("/support/{crisis_id}")
async def get_support_provisions(crisis_id: str) -> List[SupportProvision]:
    """Get support provisions for crisis"""
    try:
        support_provisions = [
            support for support in performance_engine.support_provisions.values()
            if support.crisis_id == crisis_id
        ]
        
        logger.info(f"Retrieved {len(support_provisions)} support provisions for crisis {crisis_id}")
        return support_provisions
        
    except Exception as e:
        logger.error(f"Error retrieving support provisions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve support provisions: {str(e)}")


@router.put("/intervention/{intervention_id}/effectiveness")
async def update_intervention_effectiveness(
    intervention_id: str,
    effectiveness_score: float = Query(..., ge=0.0, le=10.0, description="Effectiveness score (0-10)")
) -> dict:
    """Update intervention effectiveness score"""
    try:
        # Find intervention across all crises
        intervention = None
        for crisis_interventions in performance_engine.intervention_history.values():
            intervention = next((i for i in crisis_interventions if i.intervention_id == intervention_id), None)
            if intervention:
                break
        
        if not intervention:
            raise HTTPException(status_code=404, detail=f"Intervention {intervention_id} not found")
        
        intervention.effectiveness_score = effectiveness_score
        intervention.completion_status = "completed"
        
        logger.info(f"Updated effectiveness score for intervention {intervention_id}: {effectiveness_score}")
        return {
            "message": "Intervention effectiveness updated successfully",
            "intervention_id": intervention_id,
            "effectiveness_score": effectiveness_score
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating intervention effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update intervention effectiveness: {str(e)}")


@router.put("/support/{support_id}/feedback")
async def update_support_feedback(
    support_id: str,
    effectiveness_rating: float = Query(..., ge=0.0, le=10.0, description="Effectiveness rating (0-10)"),
    member_feedback: Optional[str] = Query(None, description="Member feedback")
) -> dict:
    """Update support provision feedback"""
    try:
        if support_id not in performance_engine.support_provisions:
            raise HTTPException(status_code=404, detail=f"Support provision {support_id} not found")
        
        support = performance_engine.support_provisions[support_id]
        support.effectiveness_rating = effectiveness_rating
        if member_feedback:
            support.member_feedback = member_feedback
        
        logger.info(f"Updated feedback for support provision {support_id}: {effectiveness_rating}")
        return {
            "message": "Support feedback updated successfully",
            "support_id": support_id,
            "effectiveness_rating": effectiveness_rating
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating support feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update support feedback: {str(e)}")


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "performance-monitoring",
        "timestamp": datetime.now(),
        "active_crises": len(performance_engine.performance_data),
        "total_alerts": sum(len(alerts) for alerts in performance_engine.active_alerts.values()),
        "total_interventions": sum(len(interventions) for interventions in performance_engine.intervention_history.values()),
        "total_support_provisions": len(performance_engine.support_provisions)
    }