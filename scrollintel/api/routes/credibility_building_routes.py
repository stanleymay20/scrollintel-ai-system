"""
API Routes for Credibility Building System

This module provides REST API endpoints for credibility building and management
functionality in the Board Executive Mastery system.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ...engines.credibility_building_engine import CredibilityBuildingEngine
from ...models.credibility_models import (
    CredibilityLevel, CredibilityAssessment, CredibilityPlan, 
    StakeholderProfile, RelationshipEvent, CredibilityReport
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/credibility", tags=["credibility"])

# Initialize engine
credibility_engine = CredibilityBuildingEngine()


@router.post("/assess/{stakeholder_id}")
async def assess_credibility(
    stakeholder_id: str,
    evidence_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assess credibility level with specific stakeholder
    
    Args:
        stakeholder_id: ID of the stakeholder
        evidence_data: Evidence and data for credibility assessment
        
    Returns:
        Credibility assessment results
    """
    try:
        assessment = credibility_engine.assess_credibility(stakeholder_id, evidence_data)
        
        return {
            "success": True,
            "assessment": {
                "stakeholder_id": assessment.stakeholder_id,
                "overall_score": assessment.overall_score,
                "level": assessment.level.value,
                "strengths": assessment.strengths,
                "improvement_areas": assessment.improvement_areas,
                "assessment_date": assessment.assessment_date.isoformat(),
                "metrics": [
                    {
                        "factor": metric.factor.value,
                        "score": metric.score,
                        "evidence": metric.evidence,
                        "trend": metric.trend
                    }
                    for metric in assessment.metrics
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error assessing credibility: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Credibility assessment failed: {str(e)}")


@router.post("/plan/{stakeholder_id}")
async def develop_credibility_plan(
    stakeholder_id: str,
    assessment_data: Dict[str, Any],
    target_level: str
) -> Dict[str, Any]:
    """
    Develop comprehensive credibility building plan
    
    Args:
        stakeholder_id: ID of the stakeholder
        assessment_data: Current credibility assessment data
        target_level: Target credibility level
        
    Returns:
        Credibility building plan
    """
    try:
        # Convert assessment data to CredibilityAssessment object
        assessment = credibility_engine.assess_credibility(stakeholder_id, assessment_data)
        target = CredibilityLevel(target_level)
        
        plan = credibility_engine.develop_credibility_plan(assessment, target)
        
        return {
            "success": True,
            "plan": {
                "id": plan.id,
                "stakeholder_id": plan.stakeholder_id,
                "current_level": plan.current_assessment.level.value,
                "target_level": plan.target_level.value,
                "timeline": plan.timeline,
                "actions": [
                    {
                        "id": action.id,
                        "title": action.title,
                        "description": action.description,
                        "target_factor": action.target_factor.value,
                        "expected_impact": action.expected_impact,
                        "timeline": action.timeline,
                        "resources_required": action.resources_required,
                        "success_metrics": action.success_metrics,
                        "status": action.status
                    }
                    for action in plan.actions
                ],
                "milestones": plan.milestones,
                "monitoring_schedule": plan.monitoring_schedule
            }
        }
        
    except Exception as e:
        logger.error(f"Error developing credibility plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Plan development failed: {str(e)}")


@router.get("/track/{plan_id}")
async def track_credibility_progress(
    plan_id: str,
    recent_events: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Track progress on credibility building plan
    
    Args:
        plan_id: ID of the credibility plan
        recent_events: Recent relationship events affecting credibility
        
    Returns:
        Progress tracking data
    """
    try:
        # In a real implementation, you would retrieve the plan from storage
        # For now, we'll create a mock plan for demonstration
        from ...models.credibility_models import CredibilityPlan, CredibilityAction, CredibilityFactor
        
        mock_plan = CredibilityPlan(
            id=plan_id,
            stakeholder_id="stakeholder_1",
            current_assessment=None,  # Would be populated from storage
            target_level=CredibilityLevel.HIGH,
            timeline="6 months",
            actions=[
                CredibilityAction(
                    id="action_1",
                    title="Demonstrate Technical Expertise",
                    description="Showcase technical knowledge",
                    target_factor=CredibilityFactor.EXPERTISE,
                    expected_impact=0.15,
                    timeline="2 months",
                    resources_required=["Documentation"],
                    success_metrics=["Positive feedback"],
                    status="in_progress"
                )
            ],
            milestones=[],
            monitoring_schedule=[],
            contingency_plans=[]
        )
        
        # Convert recent events if provided
        events = []
        if recent_events:
            for event_data in recent_events:
                event = RelationshipEvent(
                    id=event_data.get("id", ""),
                    stakeholder_id=event_data.get("stakeholder_id", ""),
                    event_type=event_data.get("event_type", ""),
                    description=event_data.get("description", ""),
                    date=datetime.fromisoformat(event_data.get("date", datetime.now().isoformat())),
                    credibility_impact=event_data.get("credibility_impact", 0.0),
                    trust_impact=event_data.get("trust_impact", 0.0),
                    lessons_learned=event_data.get("lessons_learned", []),
                    follow_up_actions=event_data.get("follow_up_actions", [])
                )
                events.append(event)
        
        progress = credibility_engine.track_credibility_progress(mock_plan, events)
        
        return {
            "success": True,
            "progress": progress
        }
        
    except Exception as e:
        logger.error(f"Error tracking credibility progress: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Progress tracking failed: {str(e)}")


@router.post("/optimize/{stakeholder_id}")
async def optimize_credibility_strategy(
    stakeholder_id: str,
    stakeholder_profile: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Optimize credibility building strategy for specific stakeholder
    
    Args:
        stakeholder_id: ID of the stakeholder
        stakeholder_profile: Stakeholder profile data
        context: Current context and situation
        
    Returns:
        Optimized credibility strategy
    """
    try:
        # Convert profile data to StakeholderProfile object
        profile = StakeholderProfile(
            id=stakeholder_profile.get("id", stakeholder_id),
            name=stakeholder_profile.get("name", ""),
            role=stakeholder_profile.get("role", ""),
            background=stakeholder_profile.get("background", ""),
            values=stakeholder_profile.get("values", []),
            communication_preferences=stakeholder_profile.get("communication_preferences", {}),
            decision_making_style=stakeholder_profile.get("decision_making_style", ""),
            influence_level=stakeholder_profile.get("influence_level", 0.5),
            credibility_assessment=None,
            trust_assessment=None,
            relationship_events=[]
        )
        
        optimization = credibility_engine.optimize_credibility_strategy(profile, context)
        
        return {
            "success": True,
            "optimization": optimization
        }
        
    except Exception as e:
        logger.error(f"Error optimizing credibility strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Strategy optimization failed: {str(e)}")


@router.post("/report")
async def generate_credibility_report(
    assessments_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate comprehensive credibility status report
    
    Args:
        assessments_data: List of credibility assessment data
        
    Returns:
        Comprehensive credibility report
    """
    try:
        # Convert assessment data to CredibilityAssessment objects
        assessments = []
        for data in assessments_data:
            # In a real implementation, you would properly reconstruct the assessment
            # For now, we'll create a simplified version
            assessment = credibility_engine.assess_credibility(
                data.get("stakeholder_id", ""),
                data.get("evidence_data", {})
            )
            assessments.append(assessment)
        
        report = credibility_engine.generate_credibility_report(assessments)
        
        return {
            "success": True,
            "report": {
                "id": report.id,
                "report_date": report.report_date.isoformat(),
                "overall_credibility_score": report.overall_credibility_score,
                "key_achievements": report.key_achievements,
                "areas_for_improvement": report.areas_for_improvement,
                "recommended_actions": report.recommended_actions,
                "trend_analysis": report.trend_analysis,
                "next_review_date": report.next_review_date.isoformat(),
                "stakeholder_count": len(report.stakeholder_assessments)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating credibility report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/factors")
async def get_credibility_factors() -> Dict[str, Any]:
    """
    Get available credibility factors and their weights
    
    Returns:
        Credibility factors and weights
    """
    try:
        return {
            "success": True,
            "factors": {
                factor.value: weight 
                for factor, weight in credibility_engine.credibility_factors.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting credibility factors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get factors: {str(e)}")


@router.get("/levels")
async def get_credibility_levels() -> Dict[str, Any]:
    """
    Get available credibility levels
    
    Returns:
        Available credibility levels
    """
    try:
        return {
            "success": True,
            "levels": [level.value for level in CredibilityLevel]
        }
        
    except Exception as e:
        logger.error(f"Error getting credibility levels: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get levels: {str(e)}")


@router.post("/action/{action_id}/update")
async def update_action_status(
    action_id: str,
    status: str,
    progress_notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update status of credibility building action
    
    Args:
        action_id: ID of the action
        status: New status of the action
        progress_notes: Optional progress notes
        
    Returns:
        Updated action status
    """
    try:
        # In a real implementation, you would update the action in storage
        return {
            "success": True,
            "action_id": action_id,
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "progress_notes": progress_notes
        }
        
    except Exception as e:
        logger.error(f"Error updating action status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Action update failed: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for credibility building system"""
    return {
        "status": "healthy",
        "service": "credibility_building",
        "timestamp": datetime.now().isoformat()
    }