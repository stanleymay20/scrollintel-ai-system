"""
Cultural Leadership API Routes

REST API endpoints for cultural leadership assessment, development, and effectiveness measurement.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ...engines.cultural_leadership_assessment_engine import CulturalLeadershipAssessmentEngine
from ...models.cultural_leadership_models import (
    CulturalLeadershipAssessment, LeadershipDevelopmentPlan,
    CulturalLeadershipProfile, LeadershipEffectivenessMetrics
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cultural-leadership", tags=["cultural-leadership"])

# Initialize engine
assessment_engine = CulturalLeadershipAssessmentEngine()


@router.post("/assess", response_model=Dict[str, Any])
async def assess_cultural_leadership(
    leader_id: str,
    organization_id: str,
    assessment_data: Dict[str, Any]
):
    """Conduct comprehensive cultural leadership assessment"""
    try:
        logger.info(f"Starting cultural leadership assessment for leader {leader_id}")
        
        assessment = assessment_engine.assess_cultural_leadership(
            leader_id=leader_id,
            organization_id=organization_id,
            assessment_data=assessment_data
        )
        
        # Generate insights
        insights = assessment_engine.get_assessment_insights(assessment)
        
        return {
            "success": True,
            "assessment": {
                "id": assessment.id,
                "leader_id": assessment.leader_id,
                "organization_id": assessment.organization_id,
                "assessment_date": assessment.assessment_date.isoformat(),
                "overall_score": assessment.overall_score,
                "leadership_level": assessment.leadership_level.value,
                "cultural_impact_score": assessment.cultural_impact_score,
                "vision_clarity_score": assessment.vision_clarity_score,
                "communication_effectiveness": assessment.communication_effectiveness,
                "change_readiness": assessment.change_readiness,
                "team_engagement_score": assessment.team_engagement_score,
                "competency_scores": [
                    {
                        "competency": score.competency.value,
                        "current_level": score.current_level.value,
                        "target_level": score.target_level.value,
                        "score": score.score,
                        "strengths": score.strengths,
                        "development_areas": score.development_areas
                    }
                    for score in assessment.competency_scores
                ],
                "recommendations": assessment.recommendations
            },
            "insights": insights
        }
        
    except Exception as e:
        logger.error(f"Error in cultural leadership assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/development-plan", response_model=Dict[str, Any])
async def create_development_plan(
    assessment_id: str,
    preferences: Dict[str, Any]
):
    """Create personalized leadership development plan"""
    try:
        logger.info(f"Creating development plan for assessment {assessment_id}")
        
        # Note: In a real implementation, you would retrieve the assessment from storage
        # For now, we'll create a mock assessment for demonstration
        mock_assessment_data = {
            "vision_creation": {"self_rating": 60, "peer_ratings": [65, 70], "manager_rating": 68},
            "communication": {"self_rating": 75, "peer_ratings": [80, 78], "manager_rating": 82},
            "change_leadership": {"self_rating": 45, "peer_ratings": [50, 48], "manager_rating": 52}
        }
        
        assessment = assessment_engine.assess_cultural_leadership(
            leader_id=preferences.get("leader_id", "demo_leader"),
            organization_id=preferences.get("organization_id", "demo_org"),
            assessment_data=mock_assessment_data
        )
        
        plan = assessment_engine.create_development_plan(assessment, preferences)
        
        return {
            "success": True,
            "development_plan": {
                "id": plan.id,
                "leader_id": plan.leader_id,
                "assessment_id": plan.assessment_id,
                "created_date": plan.created_date.isoformat(),
                "target_completion": plan.target_completion.isoformat(),
                "priority_competencies": [comp.value for comp in plan.priority_competencies],
                "development_goals": plan.development_goals,
                "learning_activities": [
                    {
                        "id": activity.id,
                        "title": activity.title,
                        "description": activity.description,
                        "type": activity.activity_type,
                        "duration": activity.estimated_duration,
                        "target_competencies": [comp.value for comp in activity.target_competencies],
                        "status": activity.status
                    }
                    for activity in plan.learning_activities
                ],
                "coaching_sessions": [
                    {
                        "id": session.id,
                        "session_date": session.session_date.isoformat(),
                        "duration": session.duration,
                        "focus_areas": [area.value for area in session.focus_areas],
                        "objectives": session.objectives
                    }
                    for session in plan.coaching_sessions
                ],
                "milestones": [
                    {
                        "id": milestone.id,
                        "title": milestone.title,
                        "description": milestone.description,
                        "target_date": milestone.target_date.isoformat(),
                        "completion_criteria": milestone.completion_criteria,
                        "success_metrics": milestone.success_metrics,
                        "status": milestone.status
                    }
                    for milestone in plan.progress_milestones
                ],
                "success_metrics": plan.success_metrics,
                "status": plan.status
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating development plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/effectiveness-measurement", response_model=Dict[str, Any])
async def measure_leadership_effectiveness(
    leader_id: str,
    measurement_period: str,
    metrics_data: Dict[str, Any]
):
    """Measure cultural leadership effectiveness"""
    try:
        logger.info(f"Measuring leadership effectiveness for leader {leader_id}")
        
        metrics = assessment_engine.measure_leadership_effectiveness(
            leader_id=leader_id,
            measurement_period=measurement_period,
            metrics_data=metrics_data
        )
        
        return {
            "success": True,
            "effectiveness_metrics": {
                "leader_id": metrics.leader_id,
                "measurement_period": metrics.measurement_period,
                "team_engagement_score": metrics.team_engagement_score,
                "cultural_alignment_score": metrics.cultural_alignment_score,
                "change_success_rate": metrics.change_success_rate,
                "vision_clarity_rating": metrics.vision_clarity_rating,
                "communication_effectiveness": metrics.communication_effectiveness,
                "influence_reach": metrics.influence_reach,
                "retention_rate": metrics.retention_rate,
                "promotion_rate": metrics.promotion_rate,
                "peer_leadership_rating": metrics.peer_leadership_rating,
                "direct_report_satisfaction": metrics.direct_report_satisfaction,
                "cultural_initiative_success": metrics.cultural_initiative_success,
                "innovation_fostered": metrics.innovation_fostered,
                "conflict_resolution_success": metrics.conflict_resolution_success,
                "measurement_date": metrics.measurement_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error measuring leadership effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/competencies", response_model=Dict[str, Any])
async def get_cultural_competencies():
    """Get list of cultural leadership competencies and their descriptions"""
    try:
        competencies = {
            "vision_creation": {
                "name": "Vision Creation",
                "description": "Ability to create compelling organizational visions that inspire and guide",
                "key_behaviors": [
                    "Develops clear and inspiring organizational vision",
                    "Communicates vision effectively to all stakeholders",
                    "Aligns team activities with organizational vision",
                    "Creates emotional connection to vision"
                ]
            },
            "values_alignment": {
                "name": "Values Alignment",
                "description": "Ensures organizational values are lived and demonstrated consistently",
                "key_behaviors": [
                    "Models organizational values in daily actions",
                    "Helps others understand and embrace values",
                    "Makes decisions aligned with organizational values",
                    "Addresses values conflicts constructively"
                ]
            },
            "change_leadership": {
                "name": "Change Leadership",
                "description": "Leads organizational and cultural transformation effectively",
                "key_behaviors": [
                    "Champions organizational change initiatives",
                    "Helps others navigate change successfully",
                    "Adapts leadership style to change context",
                    "Builds change resilience in teams"
                ]
            },
            "communication": {
                "name": "Communication",
                "description": "Communicates with clarity, influence, and cultural sensitivity",
                "key_behaviors": [
                    "Communicates clearly and persuasively",
                    "Listens actively and empathetically",
                    "Adapts communication style to audience",
                    "Facilitates difficult conversations"
                ]
            },
            "influence": {
                "name": "Influence",
                "description": "Influences others through trust, credibility, and inspiration",
                "key_behaviors": [
                    "Builds trust and credibility with others",
                    "Influences without formal authority",
                    "Inspires others to take action",
                    "Negotiates win-win solutions"
                ]
            },
            "empathy": {
                "name": "Empathy",
                "description": "Understands and responds to others' emotions and perspectives",
                "key_behaviors": [
                    "Shows genuine concern for others",
                    "Understands diverse perspectives",
                    "Responds appropriately to emotional cues",
                    "Creates inclusive environment"
                ]
            },
            "authenticity": {
                "name": "Authenticity",
                "description": "Demonstrates genuine, consistent, and transparent leadership",
                "key_behaviors": [
                    "Acts consistently with stated values",
                    "Admits mistakes and learns from them",
                    "Shows vulnerability when appropriate",
                    "Maintains integrity under pressure"
                ]
            },
            "resilience": {
                "name": "Resilience",
                "description": "Maintains effectiveness under pressure and bounces back from setbacks",
                "key_behaviors": [
                    "Stays calm under pressure",
                    "Recovers quickly from setbacks",
                    "Maintains optimism during challenges",
                    "Helps others build resilience"
                ]
            },
            "adaptability": {
                "name": "Adaptability",
                "description": "Adjusts approach based on changing circumstances and feedback",
                "key_behaviors": [
                    "Adjusts leadership style as needed",
                    "Embraces new ideas and approaches",
                    "Learns from feedback and experience",
                    "Helps others adapt to change"
                ]
            },
            "systems_thinking": {
                "name": "Systems Thinking",
                "description": "Understands organizational complexity and interconnections",
                "key_behaviors": [
                    "Sees big picture and connections",
                    "Considers long-term implications",
                    "Understands stakeholder impacts",
                    "Balances competing priorities"
                ]
            }
        }
        
        return {
            "success": True,
            "competencies": competencies
        }
        
    except Exception as e:
        logger.error(f"Error getting competencies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assessment-frameworks", response_model=Dict[str, Any])
async def get_assessment_frameworks():
    """Get available cultural leadership assessment frameworks"""
    try:
        frameworks = {
            "comprehensive": {
                "name": "Comprehensive Cultural Leadership Assessment",
                "description": "Full 360-degree assessment covering all cultural leadership competencies",
                "duration": "2-3 hours",
                "participants": ["Self", "Manager", "Peers", "Direct Reports"],
                "competencies_covered": 10,
                "assessment_methods": ["Self-rating", "360 feedback", "Behavioral observation", "Case studies"]
            },
            "360_feedback": {
                "name": "360-Degree Feedback Assessment",
                "description": "Multi-source feedback focused on leadership behaviors and impact",
                "duration": "1-2 hours",
                "participants": ["Self", "Manager", "Peers", "Direct Reports"],
                "competencies_covered": 8,
                "assessment_methods": ["Self-rating", "360 feedback"]
            },
            "self_assessment": {
                "name": "Self-Assessment Tool",
                "description": "Individual reflection and self-evaluation of cultural leadership capabilities",
                "duration": "45-60 minutes",
                "participants": ["Self"],
                "competencies_covered": 10,
                "assessment_methods": ["Self-rating", "Reflection exercises"]
            },
            "manager_assessment": {
                "name": "Manager Assessment",
                "description": "Manager-led assessment of direct report's cultural leadership",
                "duration": "30-45 minutes",
                "participants": ["Manager"],
                "competencies_covered": 8,
                "assessment_methods": ["Manager rating", "Performance review"]
            }
        }
        
        return {
            "success": True,
            "frameworks": frameworks
        }
        
    except Exception as e:
        logger.error(f"Error getting assessment frameworks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/development-resources", response_model=Dict[str, Any])
async def get_development_resources(
    competency: Optional[str] = Query(None, description="Filter by specific competency")
):
    """Get available development resources for cultural leadership"""
    try:
        all_resources = {
            "vision_creation": [
                {
                    "type": "workshop",
                    "title": "Vision Development Masterclass",
                    "description": "Learn to create compelling organizational visions that inspire action",
                    "duration": "2 days",
                    "format": "In-person/Virtual",
                    "level": "All levels"
                },
                {
                    "type": "book",
                    "title": "The Vision-Driven Leader",
                    "description": "Comprehensive guide to developing and communicating organizational vision",
                    "duration": "8-10 hours reading",
                    "format": "Self-study",
                    "level": "Intermediate"
                },
                {
                    "type": "coaching",
                    "title": "Vision Creation Coaching",
                    "description": "One-on-one coaching to develop personal vision creation skills",
                    "duration": "6 sessions",
                    "format": "Individual coaching",
                    "level": "All levels"
                }
            ],
            "communication": [
                {
                    "type": "workshop",
                    "title": "Influential Communication for Leaders",
                    "description": "Master the art of persuasive and inspiring communication",
                    "duration": "1 day",
                    "format": "In-person/Virtual",
                    "level": "All levels"
                },
                {
                    "type": "online_course",
                    "title": "Executive Communication Skills",
                    "description": "Online course covering advanced communication techniques",
                    "duration": "12 hours",
                    "format": "Self-paced online",
                    "level": "Advanced"
                }
            ],
            "change_leadership": [
                {
                    "type": "certification",
                    "title": "Change Leadership Certification",
                    "description": "Comprehensive certification program in organizational change leadership",
                    "duration": "3 months",
                    "format": "Blended learning",
                    "level": "Advanced"
                }
            ]
        }
        
        if competency:
            resources = all_resources.get(competency, [])
            return {
                "success": True,
                "competency": competency,
                "resources": resources
            }
        else:
            return {
                "success": True,
                "all_resources": all_resources
            }
        
    except Exception as e:
        logger.error(f"Error getting development resources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training-recommendation", response_model=Dict[str, Any])
async def get_training_recommendations(
    assessment_results: Dict[str, Any],
    preferences: Dict[str, Any]
):
    """Get personalized training recommendations based on assessment results"""
    try:
        logger.info("Generating training recommendations")
        
        # Extract competency scores
        competency_scores = assessment_results.get("competency_scores", [])
        
        # Identify development priorities (lowest scores)
        development_priorities = sorted(
            competency_scores,
            key=lambda x: x.get("score", 0)
        )[:3]
        
        # Generate recommendations
        recommendations = []
        
        for comp in development_priorities:
            competency_name = comp.get("competency", "")
            current_score = comp.get("score", 0)
            
            if current_score < 40:
                level = "foundational"
            elif current_score < 70:
                level = "intermediate"
            else:
                level = "advanced"
            
            recommendations.append({
                "competency": competency_name,
                "current_score": current_score,
                "priority": "high",
                "recommended_level": level,
                "suggested_resources": [
                    f"{level.title()} {competency_name.replace('_', ' ').title()} Workshop",
                    f"One-on-one coaching for {competency_name.replace('_', ' ')}",
                    f"Peer learning group for {competency_name.replace('_', ' ')}"
                ],
                "estimated_timeline": "3-6 months",
                "success_indicators": [
                    f"20+ point improvement in {competency_name.replace('_', ' ')} score",
                    "Positive feedback from team on improved capability",
                    "Successful application in work context"
                ]
            })
        
        return {
            "success": True,
            "recommendations": recommendations,
            "overall_development_plan": {
                "duration": "6-12 months",
                "focus_areas": [comp["competency"] for comp in development_priorities],
                "learning_approach": preferences.get("learning_style", "blended"),
                "coaching_support": preferences.get("coaching_preference", True),
                "peer_learning": preferences.get("peer_learning", True)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating training recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))