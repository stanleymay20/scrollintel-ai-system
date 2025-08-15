"""
API routes for Crisis Leadership Guidance System
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from pydantic import BaseModel
from datetime import datetime

from ...engines.leadership_guidance_engine import LeadershipGuidanceEngine
from ...models.leadership_guidance_models import (
    CrisisType, DecisionUrgency, DecisionContext,
    LeadershipRecommendation, LeadershipAssessment, CoachingGuidance
)

router = APIRouter(prefix="/api/leadership-guidance", tags=["leadership-guidance"])

# Pydantic models for API
class DecisionContextRequest(BaseModel):
    crisis_id: str
    crisis_type: str
    severity_level: int
    stakeholders_affected: List[str]
    time_pressure: str
    available_information: Dict[str, Any]
    resource_constraints: List[str]
    regulatory_considerations: List[str]

class PerformanceDataRequest(BaseModel):
    decision_quality: float
    communication_effectiveness: float
    stakeholder_confidence: float
    team_morale: float
    resolution_speed: float
    additional_metrics: Dict[str, Any] = {}

class LeadershipGuidanceResponse(BaseModel):
    recommendation_id: str
    recommended_style: str
    key_actions: List[str]
    communication_strategy: str
    stakeholder_priorities: List[str]
    risk_mitigation_steps: List[str]
    success_metrics: List[str]
    confidence_score: float
    rationale: str

class LeadershipAssessmentResponse(BaseModel):
    assessment_id: str
    leader_id: str
    crisis_id: str
    assessment_time: datetime
    decision_quality_score: float
    communication_effectiveness: float
    stakeholder_confidence: float
    team_morale_impact: float
    crisis_resolution_speed: float
    overall_effectiveness: float
    strengths: List[str]
    improvement_areas: List[str]
    coaching_recommendations: List[str]

class CoachingGuidanceResponse(BaseModel):
    focus_area: str
    current_performance: float
    target_performance: float
    improvement_strategies: List[str]
    practice_exercises: List[str]
    success_indicators: List[str]
    timeline: str
    resources: List[str]

def get_leadership_guidance_engine():
    """Dependency to get leadership guidance engine instance"""
    return LeadershipGuidanceEngine()

@router.post("/guidance", response_model=LeadershipGuidanceResponse)
async def get_leadership_guidance(
    request: DecisionContextRequest,
    engine: LeadershipGuidanceEngine = Depends(get_leadership_guidance_engine)
):
    """Get leadership guidance for crisis situation"""
    try:
        # Convert request to DecisionContext
        context = DecisionContext(
            crisis_id=request.crisis_id,
            crisis_type=CrisisType(request.crisis_type),
            severity_level=request.severity_level,
            stakeholders_affected=request.stakeholders_affected,
            time_pressure=DecisionUrgency(request.time_pressure),
            available_information=request.available_information,
            resource_constraints=request.resource_constraints,
            regulatory_considerations=request.regulatory_considerations
        )
        
        # Get leadership recommendation
        recommendation = engine.get_leadership_guidance(context)
        
        return LeadershipGuidanceResponse(
            recommendation_id=recommendation.id,
            recommended_style=recommendation.recommended_style.value,
            key_actions=recommendation.key_actions,
            communication_strategy=recommendation.communication_strategy,
            stakeholder_priorities=recommendation.stakeholder_priorities,
            risk_mitigation_steps=recommendation.risk_mitigation_steps,
            success_metrics=recommendation.success_metrics,
            confidence_score=recommendation.confidence_score,
            rationale=recommendation.rationale
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get leadership guidance: {str(e)}")

@router.post("/assessment/{leader_id}/{crisis_id}", response_model=LeadershipAssessmentResponse)
async def assess_leadership_effectiveness(
    leader_id: str,
    crisis_id: str,
    performance_data: PerformanceDataRequest,
    engine: LeadershipGuidanceEngine = Depends(get_leadership_guidance_engine)
):
    """Assess leadership effectiveness during crisis"""
    try:
        # Convert performance data to dict
        performance_dict = {
            "decision_quality": performance_data.decision_quality,
            "communication_effectiveness": performance_data.communication_effectiveness,
            "stakeholder_confidence": performance_data.stakeholder_confidence,
            "team_morale": performance_data.team_morale,
            "resolution_speed": performance_data.resolution_speed,
            **performance_data.additional_metrics
        }
        
        # Get assessment
        assessment = engine.assess_leadership_effectiveness(
            leader_id, crisis_id, performance_dict
        )
        
        return LeadershipAssessmentResponse(
            assessment_id=f"{leader_id}_{crisis_id}_{int(assessment.assessment_time.timestamp())}",
            leader_id=assessment.leader_id,
            crisis_id=assessment.crisis_id,
            assessment_time=assessment.assessment_time,
            decision_quality_score=assessment.decision_quality_score,
            communication_effectiveness=assessment.communication_effectiveness,
            stakeholder_confidence=assessment.stakeholder_confidence,
            team_morale_impact=assessment.team_morale_impact,
            crisis_resolution_speed=assessment.crisis_resolution_speed,
            overall_effectiveness=assessment.overall_effectiveness,
            strengths=assessment.strengths,
            improvement_areas=assessment.improvement_areas,
            coaching_recommendations=assessment.coaching_recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to assess leadership: {str(e)}")

@router.get("/coaching/{leader_id}/{crisis_id}", response_model=List[CoachingGuidanceResponse])
async def get_coaching_guidance(
    leader_id: str,
    crisis_id: str,
    engine: LeadershipGuidanceEngine = Depends(get_leadership_guidance_engine)
):
    """Get detailed coaching guidance based on assessment"""
    try:
        # For demo purposes, create a sample assessment
        # In production, this would retrieve the actual assessment
        sample_performance = {
            "decision_quality": 0.65,
            "communication_effectiveness": 0.70,
            "stakeholder_confidence": 0.60,
            "team_morale": 0.75,
            "resolution_speed": 0.65
        }
        
        assessment = engine.assess_leadership_effectiveness(
            leader_id, crisis_id, sample_performance
        )
        
        # Get coaching guidance
        coaching_guidance = engine.provide_coaching_guidance(assessment)
        
        return [
            CoachingGuidanceResponse(
                focus_area=guidance.focus_area,
                current_performance=guidance.current_performance,
                target_performance=guidance.target_performance,
                improvement_strategies=guidance.improvement_strategies,
                practice_exercises=guidance.practice_exercises,
                success_indicators=guidance.success_indicators,
                timeline=guidance.timeline,
                resources=guidance.resources
            )
            for guidance in coaching_guidance
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get coaching guidance: {str(e)}")

@router.get("/best-practices/{crisis_type}")
async def get_best_practices(
    crisis_type: str,
    engine: LeadershipGuidanceEngine = Depends(get_leadership_guidance_engine)
):
    """Get best practices for specific crisis type"""
    try:
        crisis_enum = CrisisType(crisis_type)
        practices = engine._get_relevant_practices(crisis_enum)
        
        return {
            "crisis_type": crisis_type,
            "best_practices": [
                {
                    "id": practice.id,
                    "name": practice.practice_name,
                    "description": practice.description,
                    "implementation_steps": practice.implementation_steps,
                    "success_indicators": practice.success_indicators,
                    "common_pitfalls": practice.common_pitfalls,
                    "effectiveness_score": practice.effectiveness_score,
                    "applicable_scenarios": practice.applicable_scenarios
                }
                for practice in practices
            ]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid crisis type: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get best practices: {str(e)}")

@router.get("/crisis-types")
async def get_crisis_types():
    """Get available crisis types"""
    return {
        "crisis_types": [
            {
                "value": crisis_type.value,
                "name": crisis_type.name.replace("_", " ").title()
            }
            for crisis_type in CrisisType
        ]
    }

@router.get("/leadership-styles")
async def get_leadership_styles():
    """Get available leadership styles"""
    from ...models.leadership_guidance_models import LeadershipStyle
    
    return {
        "leadership_styles": [
            {
                "value": style.value,
                "name": style.name.replace("_", " ").title(),
                "description": {
                    "directive": "Clear, authoritative leadership for urgent situations",
                    "collaborative": "Team-based decision making for complex situations",
                    "supportive": "Empathetic leadership focused on team wellbeing",
                    "transformational": "Visionary leadership for major organizational change",
                    "adaptive": "Flexible leadership that adjusts to situation needs"
                }.get(style.value, "Effective leadership approach")
            }
            for style in LeadershipStyle
        ]
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "leadership-guidance"}