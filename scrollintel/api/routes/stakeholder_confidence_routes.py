"""
API routes for stakeholder confidence management system.
Provides endpoints for monitoring, assessing, and maintaining stakeholder confidence during crisis.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ...engines.stakeholder_confidence_engine import StakeholderConfidenceEngine
from ...models.stakeholder_confidence_models import (
    StakeholderProfile, ConfidenceMetrics, ConfidenceBuildingStrategy,
    TrustMaintenanceAction, CommunicationPlan, ConfidenceAssessment,
    StakeholderFeedback, ConfidenceAlert, StakeholderType, ConfidenceLevel
)

router = APIRouter(prefix="/api/v1/stakeholder-confidence", tags=["stakeholder-confidence"])
logger = logging.getLogger(__name__)

# Global engine instance
confidence_engine = StakeholderConfidenceEngine()


@router.post("/monitor", response_model=Dict[str, ConfidenceMetrics])
async def monitor_stakeholder_confidence(
    crisis_id: str,
    stakeholder_ids: List[str]
):
    """Monitor confidence levels across specified stakeholders"""
    try:
        confidence_data = await confidence_engine.monitor_stakeholder_confidence(
            crisis_id=crisis_id,
            stakeholder_ids=stakeholder_ids
        )
        
        logger.info(f"Monitored confidence for {len(stakeholder_ids)} stakeholders in crisis {crisis_id}")
        return confidence_data
        
    except Exception as e:
        logger.error(f"Error monitoring stakeholder confidence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess", response_model=ConfidenceAssessment)
async def assess_overall_confidence(crisis_id: str):
    """Assess overall stakeholder confidence situation"""
    try:
        assessment = await confidence_engine.assess_overall_confidence(crisis_id)
        
        logger.info(f"Completed confidence assessment for crisis {crisis_id}")
        return assessment
        
    except Exception as e:
        logger.error(f"Error assessing overall confidence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategy/build", response_model=ConfidenceBuildingStrategy)
async def build_confidence_strategy(
    stakeholder_type: StakeholderType,
    current_confidence: ConfidenceLevel,
    target_confidence: ConfidenceLevel
):
    """Build strategy for improving stakeholder confidence"""
    try:
        strategy = await confidence_engine.build_confidence_strategy(
            stakeholder_type=stakeholder_type,
            current_confidence=current_confidence,
            target_confidence=target_confidence
        )
        
        logger.info(f"Created confidence building strategy for {stakeholder_type.value}")
        return strategy
        
    except Exception as e:
        logger.error(f"Error building confidence strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trust/maintain", response_model=List[TrustMaintenanceAction])
async def maintain_stakeholder_trust(
    stakeholder_id: str,
    crisis_context: Dict[str, Any]
):
    """Maintain trust with specific stakeholder during crisis"""
    try:
        actions = await confidence_engine.maintain_stakeholder_trust(
            stakeholder_id=stakeholder_id,
            crisis_context=crisis_context
        )
        
        logger.info(f"Generated {len(actions)} trust maintenance actions for stakeholder {stakeholder_id}")
        return actions
        
    except Exception as e:
        logger.error(f"Error maintaining stakeholder trust: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/communication/plan", response_model=CommunicationPlan)
async def create_communication_plan(
    crisis_id: str,
    stakeholder_segments: List[StakeholderType]
):
    """Create comprehensive communication plan for stakeholder confidence"""
    try:
        plan = await confidence_engine.create_communication_plan(
            crisis_id=crisis_id,
            stakeholder_segments=stakeholder_segments
        )
        
        logger.info(f"Created communication plan for crisis {crisis_id}")
        return plan
        
    except Exception as e:
        logger.error(f"Error creating communication plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/process", response_model=Dict[str, Any])
async def process_stakeholder_feedback(feedback: StakeholderFeedback):
    """Process and respond to stakeholder feedback"""
    try:
        result = await confidence_engine.process_stakeholder_feedback(feedback)
        
        logger.info(f"Processed stakeholder feedback: {feedback.feedback_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing stakeholder feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/{stakeholder_id}", response_model=Optional[StakeholderProfile])
async def get_stakeholder_profile(stakeholder_id: str):
    """Get stakeholder profile information"""
    try:
        profile = confidence_engine.stakeholder_profiles.get(stakeholder_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Stakeholder profile not found")
        
        logger.info(f"Retrieved stakeholder profile: {stakeholder_id}")
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving stakeholder profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profiles", response_model=StakeholderProfile)
async def create_stakeholder_profile(profile: StakeholderProfile):
    """Create or update stakeholder profile"""
    try:
        confidence_engine.stakeholder_profiles[profile.stakeholder_id] = profile
        
        logger.info(f"Created/updated stakeholder profile: {profile.stakeholder_id}")
        return profile
        
    except Exception as e:
        logger.error(f"Error creating stakeholder profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{stakeholder_id}", response_model=List[ConfidenceMetrics])
async def get_confidence_metrics(stakeholder_id: str):
    """Get confidence metrics history for stakeholder"""
    try:
        metrics = confidence_engine.confidence_metrics.get(stakeholder_id, [])
        
        logger.info(f"Retrieved {len(metrics)} confidence metrics for stakeholder {stakeholder_id}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving confidence metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=List[ConfidenceAlert])
async def get_active_alerts():
    """Get active confidence alerts"""
    try:
        alerts = confidence_engine.active_alerts
        
        logger.info(f"Retrieved {len(alerts)} active confidence alerts")
        return alerts
        
    except Exception as e:
        logger.error(f"Error retrieving confidence alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve")
async def resolve_confidence_alert(alert_id: str):
    """Resolve a confidence alert"""
    try:
        # Find and remove the alert
        alert_found = False
        for i, alert in enumerate(confidence_engine.active_alerts):
            if alert.alert_id == alert_id:
                confidence_engine.active_alerts.pop(i)
                alert_found = True
                break
        
        if not alert_found:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        logger.info(f"Resolved confidence alert: {alert_id}")
        return {"status": "resolved", "alert_id": alert_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving confidence alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assessments", response_model=List[ConfidenceAssessment])
async def get_confidence_assessments():
    """Get confidence assessment history"""
    try:
        assessments = confidence_engine.assessments
        
        logger.info(f"Retrieved {len(assessments)} confidence assessments")
        return assessments
        
    except Exception as e:
        logger.error(f"Error retrieving confidence assessments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback", response_model=List[StakeholderFeedback])
async def get_stakeholder_feedback():
    """Get stakeholder feedback queue"""
    try:
        feedback = confidence_engine.feedback_queue
        
        logger.info(f"Retrieved {len(feedback)} feedback items")
        return feedback
        
    except Exception as e:
        logger.error(f"Error retrieving stakeholder feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies", response_model=List[ConfidenceBuildingStrategy])
async def get_confidence_strategies():
    """Get all confidence building strategies"""
    try:
        strategies = list(confidence_engine.building_strategies.values())
        
        logger.info(f"Retrieved {len(strategies)} confidence building strategies")
        return strategies
        
    except Exception as e:
        logger.error(f"Error retrieving confidence strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/actions/{stakeholder_id}", response_model=List[TrustMaintenanceAction])
async def get_trust_actions(stakeholder_id: str):
    """Get trust maintenance actions for stakeholder"""
    try:
        actions = confidence_engine.trust_actions.get(stakeholder_id, [])
        
        logger.info(f"Retrieved {len(actions)} trust maintenance actions for stakeholder {stakeholder_id}")
        return actions
        
    except Exception as e:
        logger.error(f"Error retrieving trust maintenance actions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/actions/{action_id}/complete")
async def complete_trust_action(action_id: str):
    """Mark a trust maintenance action as completed"""
    try:
        action_found = False
        
        # Find and update the action
        for stakeholder_actions in confidence_engine.trust_actions.values():
            for action in stakeholder_actions:
                if action.action_id == action_id:
                    action.status = "completed"
                    action_found = True
                    break
            if action_found:
                break
        
        if not action_found:
            raise HTTPException(status_code=404, detail="Trust maintenance action not found")
        
        logger.info(f"Completed trust maintenance action: {action_id}")
        return {"status": "completed", "action_id": action_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing trust maintenance action: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard", response_model=Dict[str, Any])
async def get_confidence_dashboard():
    """Get stakeholder confidence dashboard data"""
    try:
        # Calculate overall metrics
        total_stakeholders = len(confidence_engine.stakeholder_profiles)
        active_alerts = len(confidence_engine.active_alerts)
        pending_feedback = len([f for f in confidence_engine.feedback_queue if f.resolution_status == "open"])
        
        # Get recent assessment
        recent_assessment = confidence_engine.assessments[-1] if confidence_engine.assessments else None
        
        dashboard_data = {
            "total_stakeholders": total_stakeholders,
            "active_alerts": active_alerts,
            "pending_feedback": pending_feedback,
            "recent_assessment": recent_assessment,
            "confidence_trends": await confidence_engine._analyze_confidence_trends(),
            "risk_areas": await confidence_engine._identify_confidence_risks(),
            "improvement_opportunities": await confidence_engine._identify_improvement_opportunities()
        }
        
        logger.info("Generated stakeholder confidence dashboard")
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error generating confidence dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))