"""
API Routes for Executive Communication System
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

from ...engines.executive_communication_engine import ExecutiveCommunicationSystem
from ...engines.strategic_narrative_engine import (
    StrategicNarrativeSystem, StrategicContext, NarrativeType, NarrativeStructure
)
from ...engines.information_synthesis_engine import (
    InformationSynthesisSystem, ComplexInformation, InformationType, SynthesisLevel
)
from ...models.executive_communication_models import (
    ExecutiveAudience, Message, AdaptedMessage, CommunicationEffectiveness,
    ExecutiveLevel, CommunicationStyle, MessageType
)

router = APIRouter(prefix="/api/v1/executive-communication", tags=["executive-communication"])
logger = logging.getLogger(__name__)

# Initialize the executive communication system
communication_system = ExecutiveCommunicationSystem()
narrative_system = StrategicNarrativeSystem()
synthesis_system = InformationSynthesisSystem()


@router.post("/adapt-message", response_model=Dict[str, Any])
async def adapt_message_for_executive(
    message_data: Dict[str, Any],
    audience_data: Dict[str, Any]
):
    """
    Adapt a message for executive audience
    """
    try:
        # Create message object
        message = Message(
            id=message_data.get("id", "msg_001"),
            content=message_data["content"],
            message_type=MessageType(message_data.get("message_type", "strategic_update")),
            technical_complexity=message_data.get("technical_complexity", 0.5),
            urgency_level=message_data.get("urgency_level", "medium"),
            key_points=message_data.get("key_points", []),
            supporting_data=message_data.get("supporting_data", {}),
            created_at=message_data.get("created_at")
        )
        
        # Create audience object
        audience = ExecutiveAudience(
            id=audience_data.get("id", "exec_001"),
            name=audience_data["name"],
            title=audience_data["title"],
            executive_level=ExecutiveLevel(audience_data["executive_level"]),
            communication_style=CommunicationStyle(audience_data["communication_style"]),
            expertise_areas=audience_data.get("expertise_areas", []),
            decision_making_pattern=audience_data.get("decision_making_pattern", "analytical"),
            influence_level=audience_data.get("influence_level", 0.8),
            preferred_communication_format=audience_data.get("preferred_communication_format", "email"),
            attention_span=audience_data.get("attention_span", 10),
            detail_preference=audience_data.get("detail_preference", "medium"),
            risk_tolerance=audience_data.get("risk_tolerance", "medium"),
            created_at=audience_data.get("created_at")
        )
        
        # Adapt the message
        adapted_message = communication_system.process_executive_communication(message, audience)
        
        return {
            "success": True,
            "adapted_message": {
                "id": adapted_message.id,
                "adapted_content": adapted_message.adapted_content,
                "executive_summary": adapted_message.executive_summary,
                "key_recommendations": adapted_message.key_recommendations,
                "tone": adapted_message.tone,
                "language_complexity": adapted_message.language_complexity,
                "estimated_reading_time": adapted_message.estimated_reading_time,
                "effectiveness_score": adapted_message.effectiveness_score,
                "adaptation_rationale": adapted_message.adaptation_rationale
            }
        }
        
    except Exception as e:
        logger.error(f"Error adapting message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adapting message: {str(e)}")


@router.post("/track-effectiveness", response_model=Dict[str, Any])
async def track_communication_effectiveness(
    message_id: str,
    audience_id: str,
    engagement_data: Dict[str, Any]
):
    """
    Track communication effectiveness
    """
    try:
        effectiveness = communication_system.track_communication_effectiveness(
            message_id, audience_id, engagement_data
        )
        
        return {
            "success": True,
            "effectiveness": {
                "id": effectiveness.id,
                "engagement_score": effectiveness.engagement_score,
                "comprehension_score": effectiveness.comprehension_score,
                "action_taken": effectiveness.action_taken,
                "response_time": effectiveness.response_time,
                "follow_up_questions": effectiveness.follow_up_questions,
                "decision_influenced": effectiveness.decision_influenced
            }
        }
        
    except Exception as e:
        logger.error(f"Error tracking effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error tracking effectiveness: {str(e)}")


@router.get("/optimization-recommendations/{message_id}/{audience_id}", response_model=Dict[str, Any])
async def get_optimization_recommendations(message_id: str, audience_id: str):
    """
    Get optimization recommendations for communication
    """
    try:
        # In a real implementation, we would retrieve the effectiveness data from database
        # For now, we'll create a sample effectiveness object
        sample_effectiveness = CommunicationEffectiveness(
            id=f"eff_{message_id}_{audience_id}",
            message_id=message_id,
            audience_id=audience_id,
            engagement_score=0.7,
            comprehension_score=0.8,
            action_taken=True,
            feedback_received="Good summary, but could be more concise",
            response_time=30,
            follow_up_questions=2,
            decision_influenced=True,
            measured_at=None
        )
        
        recommendations = communication_system.get_optimization_recommendations(sample_effectiveness)
        
        return {
            "success": True,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")


@router.get("/audience-profiles", response_model=Dict[str, Any])
async def get_executive_audience_profiles():
    """
    Get available executive audience profiles
    """
    try:
        # Sample audience profiles
        profiles = [
            {
                "id": "ceo_001",
                "name": "CEO Profile",
                "executive_level": "ceo",
                "communication_style": "direct",
                "attention_span": 5,
                "detail_preference": "low"
            },
            {
                "id": "cto_001", 
                "name": "CTO Profile",
                "executive_level": "cto",
                "communication_style": "analytical",
                "attention_span": 15,
                "detail_preference": "high"
            },
            {
                "id": "board_001",
                "name": "Board Chair Profile",
                "executive_level": "board_chair",
                "communication_style": "strategic",
                "attention_span": 10,
                "detail_preference": "medium"
            }
        ]
        
        return {
            "success": True,
            "profiles": profiles
        }
        
    except Exception as e:
        logger.error(f"Error getting audience profiles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting audience profiles: {str(e)}")


@router.get("/communication-styles", response_model=Dict[str, Any])
async def get_communication_styles():
    """
    Get available communication styles
    """
    try:
        styles = [
            {
                "value": "analytical",
                "label": "Analytical",
                "description": "Data-driven, logical approach with detailed analysis"
            },
            {
                "value": "strategic", 
                "label": "Strategic",
                "description": "High-level, forward-thinking with focus on long-term impact"
            },
            {
                "value": "diplomatic",
                "label": "Diplomatic", 
                "description": "Tactful, collaborative approach that builds consensus"
            },
            {
                "value": "direct",
                "label": "Direct",
                "description": "Straightforward, action-oriented communication"
            },
            {
                "value": "collaborative",
                "label": "Collaborative",
                "description": "Team-focused, inclusive communication style"
            },
            {
                "value": "authoritative",
                "label": "Authoritative",
                "description": "Confident, decisive communication with clear direction"
            }
        ]
        
        return {
            "success": True,
            "styles": styles
        }
        
    except Exception as e:
        logger.error(f"Error getting communication styles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting communication styles: {str(e)}")


@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "executive-communication"}

@
router.post("/create-strategic-narrative", response_model=Dict[str, Any])
async def create_strategic_narrative(
    strategy_data: Dict[str, Any],
    audience_data: Dict[str, Any],
    context_data: Dict[str, Any]
):
    """
    Create strategic narrative for board presentation
    """
    try:
        # Create audience object
        audience = ExecutiveAudience(
            id=audience_data.get("id", "exec_001"),
            name=audience_data["name"],
            title=audience_data["title"],
            executive_level=ExecutiveLevel(audience_data["executive_level"]),
            communication_style=CommunicationStyle(audience_data["communication_style"]),
            expertise_areas=audience_data.get("expertise_areas", []),
            decision_making_pattern=audience_data.get("decision_making_pattern", "analytical"),
            influence_level=audience_data.get("influence_level", 0.8),
            preferred_communication_format=audience_data.get("preferred_communication_format", "presentation"),
            attention_span=audience_data.get("attention_span", 10),
            detail_preference=audience_data.get("detail_preference", "medium"),
            risk_tolerance=audience_data.get("risk_tolerance", "medium"),
            created_at=audience_data.get("created_at")
        )
        
        # Create strategic context
        context = StrategicContext(
            company_position=context_data["company_position"],
            market_conditions=context_data["market_conditions"],
            competitive_landscape=context_data["competitive_landscape"],
            key_challenges=context_data["key_challenges"],
            opportunities=context_data["opportunities"],
            stakeholder_concerns=context_data.get("stakeholder_concerns", []),
            success_metrics=context_data.get("success_metrics", []),
            timeline=context_data.get("timeline", "12 months")
        )
        
        # Create strategic narrative
        narrative = narrative_system.create_strategic_narrative(strategy_data, audience, context)
        
        return {
            "success": True,
            "narrative": {
                "id": narrative.id,
                "title": narrative.title,
                "narrative_type": narrative.narrative_type.value,
                "structure": narrative.structure.value,
                "key_messages": narrative.key_messages,
                "emotional_arc": narrative.emotional_arc,
                "call_to_action": narrative.call_to_action,
                "supporting_visuals": narrative.supporting_visuals,
                "impact_score": narrative.impact_score,
                "personalization_notes": narrative.personalization_notes,
                "elements": [
                    {
                        "type": element.element_type,
                        "content": element.content,
                        "emotional_tone": element.emotional_tone,
                        "audience_relevance": element.audience_relevance
                    }
                    for element in narrative.elements
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating strategic narrative: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating strategic narrative: {str(e)}")


@router.post("/assess-narrative-impact", response_model=Dict[str, Any])
async def assess_narrative_impact(
    narrative_id: str,
    audience_id: str,
    engagement_data: Dict[str, Any]
):
    """
    Assess narrative impact and get optimization recommendations
    """
    try:
        # In a real implementation, we would retrieve the narrative from database
        # For now, we'll create a sample narrative for demonstration
        from ...engines.strategic_narrative_engine import StrategicNarrative, NarrativeElement
        from datetime import datetime
        
        sample_narrative = StrategicNarrative(
            id=narrative_id,
            title="Strategic Transformation Initiative",
            narrative_type=NarrativeType.TRANSFORMATION_JOURNEY,
            structure=NarrativeStructure.THREE_ACT,
            audience_id=audience_id,
            elements=[
                NarrativeElement(
                    element_type="setup",
                    content="Our strategic position requires transformation",
                    emotional_tone="anticipation",
                    supporting_data={},
                    audience_relevance=0.8
                )
            ],
            key_messages=["Transformation required", "Strategic advantage", "Market leadership"],
            emotional_arc="anticipation → challenge → triumph",
            call_to_action="Board approval required for strategic initiative",
            supporting_visuals=["transformation_roadmap", "market_analysis"],
            impact_score=0.85,
            personalization_notes="Tailored for board presentation",
            created_at=datetime.now()
        )
        
        # Create sample audience
        sample_audience = ExecutiveAudience(
            id=audience_id,
            name="Board Member",
            title="Board Member",
            executive_level=ExecutiveLevel.BOARD_MEMBER,
            communication_style=CommunicationStyle.STRATEGIC,
            expertise_areas=["governance"],
            decision_making_pattern="consensus",
            influence_level=0.9,
            preferred_communication_format="presentation",
            attention_span=15,
            detail_preference="medium",
            risk_tolerance="low",
            created_at=datetime.now()
        )
        
        # Assess impact and get recommendations
        impact, recommendations = narrative_system.assess_and_optimize(
            sample_narrative, sample_audience, engagement_data
        )
        
        return {
            "success": True,
            "impact": {
                "id": impact.id,
                "engagement_level": impact.engagement_level,
                "emotional_resonance": impact.emotional_resonance,
                "message_retention": impact.message_retention,
                "action_likelihood": impact.action_likelihood,
                "credibility_score": impact.credibility_score,
                "overall_impact": impact.overall_impact
            },
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error assessing narrative impact: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error assessing narrative impact: {str(e)}")


@router.get("/narrative-types", response_model=Dict[str, Any])
async def get_narrative_types():
    """
    Get available narrative types
    """
    try:
        types = [
            {
                "value": "vision_story",
                "label": "Vision Story",
                "description": "Compelling future vision and path to achievement"
            },
            {
                "value": "transformation_journey",
                "label": "Transformation Journey",
                "description": "Organizational change and evolution narrative"
            },
            {
                "value": "competitive_advantage",
                "label": "Competitive Advantage",
                "description": "Market positioning and differentiation story"
            },
            {
                "value": "growth_strategy",
                "label": "Growth Strategy",
                "description": "Expansion and scaling narrative"
            },
            {
                "value": "crisis_response",
                "label": "Crisis Response",
                "description": "Challenge navigation and resolution story"
            },
            {
                "value": "innovation_showcase",
                "label": "Innovation Showcase",
                "description": "Technology and innovation leadership narrative"
            },
            {
                "value": "market_opportunity",
                "label": "Market Opportunity",
                "description": "Market potential and capture strategy"
            }
        ]
        
        return {
            "success": True,
            "narrative_types": types
        }
        
    except Exception as e:
        logger.error(f"Error getting narrative types: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting narrative types: {str(e)}")


@router.get("/narrative-structures", response_model=Dict[str, Any])
async def get_narrative_structures():
    """
    Get available narrative structures
    """
    try:
        structures = [
            {
                "value": "problem_solution",
                "label": "Problem-Solution",
                "description": "Identify challenge and present solution"
            },
            {
                "value": "hero_journey",
                "label": "Hero's Journey",
                "description": "Transformation through challenge and growth"
            },
            {
                "value": "before_after",
                "label": "Before-After",
                "description": "Contrast current state with future vision"
            },
            {
                "value": "three_act",
                "label": "Three-Act Structure",
                "description": "Setup, conflict, and resolution narrative"
            },
            {
                "value": "pyramid",
                "label": "Pyramid Structure",
                "description": "Build to climactic conclusion"
            },
            {
                "value": "chronological",
                "label": "Chronological",
                "description": "Time-based progression of events"
            }
        ]
        
        return {
            "success": True,
            "narrative_structures": structures
        }
        
    except Exception as e:
        logger.error(f"Error getting narrative structures: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting narrative structures: {str(e)}")

@router.post("/synthesize-information", response_model=Dict[str, Any])
async def synthesize_complex_information(
    information_data: Dict[str, Any],
    audience_data: Dict[str, Any],
    synthesis_level: str = "executive_summary"
):
    """
    Synthesize complex information into executive-friendly format
    """
    try:
        # Create complex information object
        complex_info = ComplexInformation(
            id=information_data.get("id", "info_001"),
            title=information_data["title"],
            information_type=InformationType(information_data.get("information_type", "technical_report")),
            raw_content=information_data["raw_content"],
            data_points=information_data.get("data_points", []),
            source=information_data.get("source", "internal"),
            complexity_score=information_data.get("complexity_score", 0.7),
            urgency_level=information_data.get("urgency_level", "medium"),
            stakeholders=information_data.get("stakeholders", []),
            created_at=information_data.get("created_at")
        )
        
        # Create audience object
        audience = ExecutiveAudience(
            id=audience_data.get("id", "exec_001"),
            name=audience_data["name"],
            title=audience_data["title"],
            executive_level=ExecutiveLevel(audience_data["executive_level"]),
            communication_style=CommunicationStyle(audience_data["communication_style"]),
            expertise_areas=audience_data.get("expertise_areas", []),
            decision_making_pattern=audience_data.get("decision_making_pattern", "analytical"),
            influence_level=audience_data.get("influence_level", 0.8),
            preferred_communication_format=audience_data.get("preferred_communication_format", "report"),
            attention_span=audience_data.get("attention_span", 10),
            detail_preference=audience_data.get("detail_preference", "medium"),
            risk_tolerance=audience_data.get("risk_tolerance", "medium"),
            created_at=audience_data.get("created_at")
        )
        
        # Synthesize the information
        synthesized = synthesis_system.process_complex_information(complex_info, audience)
        
        return {
            "success": True,
            "synthesized_information": {
                "id": synthesized.id,
                "title": synthesized.title,
                "executive_summary": synthesized.executive_summary,
                "key_insights": synthesized.key_insights,
                "critical_data_points": synthesized.critical_data_points,
                "recommendations": synthesized.recommendations,
                "decision_points": synthesized.decision_points,
                "risk_factors": synthesized.risk_factors,
                "next_steps": synthesized.next_steps,
                "synthesis_level": synthesized.synthesis_level.value,
                "readability_score": synthesized.readability_score,
                "confidence_level": synthesized.confidence_level
            }
        }
        
    except Exception as e:
        logger.error(f"Error synthesizing information: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error synthesizing information: {str(e)}")


@router.post("/prioritize-information", response_model=Dict[str, Any])
async def prioritize_information_batch(
    information_list: List[Dict[str, Any]],
    audience_data: Dict[str, Any]
):
    """
    Prioritize a batch of information items for executive attention
    """
    try:
        # Create audience object
        audience = ExecutiveAudience(
            id=audience_data.get("id", "exec_001"),
            name=audience_data["name"],
            title=audience_data["title"],
            executive_level=ExecutiveLevel(audience_data["executive_level"]),
            communication_style=CommunicationStyle(audience_data["communication_style"]),
            expertise_areas=audience_data.get("expertise_areas", []),
            decision_making_pattern=audience_data.get("decision_making_pattern", "analytical"),
            influence_level=audience_data.get("influence_level", 0.8),
            preferred_communication_format=audience_data.get("preferred_communication_format", "report"),
            attention_span=audience_data.get("attention_span", 10),
            detail_preference=audience_data.get("detail_preference", "medium"),
            risk_tolerance=audience_data.get("risk_tolerance", "medium"),
            created_at=audience_data.get("created_at")
        )
        
        # Create complex information objects
        complex_info_list = []
        for info_data in information_list:
            complex_info = ComplexInformation(
                id=info_data.get("id", f"info_{len(complex_info_list)}"),
                title=info_data["title"],
                information_type=InformationType(info_data.get("information_type", "technical_report")),
                raw_content=info_data.get("raw_content", ""),
                data_points=info_data.get("data_points", []),
                source=info_data.get("source", "internal"),
                complexity_score=info_data.get("complexity_score", 0.7),
                urgency_level=info_data.get("urgency_level", "medium"),
                stakeholders=info_data.get("stakeholders", []),
                created_at=info_data.get("created_at")
            )
            complex_info_list.append(complex_info)
        
        # Prioritize the information
        priorities = synthesis_system.prioritize_information_batch(complex_info_list, audience)
        
        return {
            "success": True,
            "priorities": [
                {
                    "id": priority.id,
                    "information_id": priority.information_id,
                    "relevance_score": priority.relevance_score,
                    "urgency_score": priority.urgency_score,
                    "impact_score": priority.impact_score,
                    "priority_level": priority.priority_level,
                    "reasoning": priority.reasoning
                }
                for priority in priorities
            ]
        }
        
    except Exception as e:
        logger.error(f"Error prioritizing information: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error prioritizing information: {str(e)}")


@router.post("/generate-executive-summary", response_model=Dict[str, Any])
async def generate_executive_summary(
    synthesized_data: Dict[str, Any],
    audience_data: Dict[str, Any]
):
    """
    Generate optimized executive summary from synthesized information
    """
    try:
        # Create audience object
        audience = ExecutiveAudience(
            id=audience_data.get("id", "exec_001"),
            name=audience_data["name"],
            title=audience_data["title"],
            executive_level=ExecutiveLevel(audience_data["executive_level"]),
            communication_style=CommunicationStyle(audience_data["communication_style"]),
            expertise_areas=audience_data.get("expertise_areas", []),
            decision_making_pattern=audience_data.get("decision_making_pattern", "analytical"),
            influence_level=audience_data.get("influence_level", 0.8),
            preferred_communication_format=audience_data.get("preferred_communication_format", "report"),
            attention_span=audience_data.get("attention_span", 10),
            detail_preference=audience_data.get("detail_preference", "medium"),
            risk_tolerance=audience_data.get("risk_tolerance", "medium"),
            created_at=audience_data.get("created_at")
        )
        
        # Create synthesized information object
        from ...engines.information_synthesis_engine import SynthesizedInformation
        
        synthesized_info = SynthesizedInformation(
            id=synthesized_data.get("id", "synth_001"),
            original_id=synthesized_data.get("original_id", "orig_001"),
            title=synthesized_data["title"],
            executive_summary=synthesized_data.get("executive_summary", ""),
            key_insights=synthesized_data.get("key_insights", []),
            critical_data_points=synthesized_data.get("critical_data_points", []),
            recommendations=synthesized_data.get("recommendations", []),
            decision_points=synthesized_data.get("decision_points", []),
            risk_factors=synthesized_data.get("risk_factors", []),
            next_steps=synthesized_data.get("next_steps", []),
            synthesis_level=SynthesisLevel(synthesized_data.get("synthesis_level", "executive_summary")),
            audience_id=audience.id,
            readability_score=synthesized_data.get("readability_score", 0.8),
            confidence_level=synthesized_data.get("confidence_level", 0.8),
            created_at=synthesized_data.get("created_at")
        )
        
        # Generate optimized summary
        optimized_summary = synthesis_system.generate_optimized_summary(synthesized_info, audience)
        
        return {
            "success": True,
            "executive_summary": optimized_summary,
            "optimization_notes": f"Optimized for {audience.executive_level.value} with {audience.communication_style.value} style"
        }
        
    except Exception as e:
        logger.error(f"Error generating executive summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating executive summary: {str(e)}")


@router.get("/information-types", response_model=Dict[str, Any])
async def get_information_types():
    """
    Get available information types for synthesis
    """
    try:
        types = [
            {
                "value": "technical_report",
                "label": "Technical Report",
                "description": "Technical documentation and analysis"
            },
            {
                "value": "financial_data",
                "label": "Financial Data",
                "description": "Financial reports and metrics"
            },
            {
                "value": "market_analysis",
                "label": "Market Analysis",
                "description": "Market research and competitive intelligence"
            },
            {
                "value": "operational_metrics",
                "label": "Operational Metrics",
                "description": "Operational performance data"
            },
            {
                "value": "risk_assessment",
                "label": "Risk Assessment",
                "description": "Risk analysis and mitigation strategies"
            },
            {
                "value": "strategic_plan",
                "label": "Strategic Plan",
                "description": "Strategic planning documents"
            },
            {
                "value": "competitive_intelligence",
                "label": "Competitive Intelligence",
                "description": "Competitive analysis and market positioning"
            }
        ]
        
        return {
            "success": True,
            "information_types": types
        }
        
    except Exception as e:
        logger.error(f"Error getting information types: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting information types: {str(e)}")


@router.get("/synthesis-levels", response_model=Dict[str, Any])
async def get_synthesis_levels():
    """
    Get available synthesis levels
    """
    try:
        levels = [
            {
                "value": "executive_summary",
                "label": "Executive Summary",
                "description": "High-level summary for executive consumption"
            },
            {
                "value": "detailed_analysis",
                "label": "Detailed Analysis",
                "description": "Comprehensive analysis with supporting details"
            },
            {
                "value": "key_insights",
                "label": "Key Insights",
                "description": "Focus on critical insights and findings"
            },
            {
                "value": "action_items",
                "label": "Action Items",
                "description": "Actionable recommendations and next steps"
            },
            {
                "value": "decision_points",
                "label": "Decision Points",
                "description": "Key decisions requiring executive attention"
            }
        ]
        
        return {
            "success": True,
            "synthesis_levels": levels
        }
        
    except Exception as e:
        logger.error(f"Error getting synthesis levels: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting synthesis levels: {str(e)}")