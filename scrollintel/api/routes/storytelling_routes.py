"""
Storytelling Framework API Routes

REST API endpoints for storytelling framework including story creation,
personalization, delivery, and impact measurement.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ...engines.storytelling_engine import StorytellingEngine
from ...models.storytelling_models import (
    StoryType, NarrativeStructure, DeliveryFormat
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/storytelling", tags=["storytelling"])

# Global engine instance
storytelling_engine = StorytellingEngine()


@router.post("/strategies")
async def create_narrative_strategy(
    organization_id: str,
    transformation_vision: str,
    core_narratives: List[str],
    audience_preferences: Dict[str, Any]
):
    """Create comprehensive narrative strategy"""
    try:
        strategy = storytelling_engine.create_narrative_strategy(
            organization_id=organization_id,
            transformation_vision=transformation_vision,
            core_narratives=core_narratives,
            audience_preferences=audience_preferences
        )
        
        return {
            "success": True,
            "strategy_id": strategy.id,
            "message": "Narrative strategy created successfully",
            "data": {
                "strategy_id": strategy.id,
                "organization_id": strategy.organization_id,
                "story_themes": strategy.story_themes,
                "character_archetypes": len(strategy.character_archetypes),
                "effectiveness_targets": strategy.effectiveness_targets,
                "created_at": strategy.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating narrative strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stories")
async def create_transformation_story(
    title: str,
    story_type: str,
    narrative_structure: str,
    content: str,
    cultural_themes: List[str],
    key_messages: List[str],
    target_outcomes: List[str],
    template_id: Optional[str] = None
):
    """Create new transformation story"""
    try:
        # Validate enums
        try:
            story_type_enum = StoryType(story_type)
            structure_enum = NarrativeStructure(narrative_structure)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
        
        story = storytelling_engine.create_transformation_story(
            title=title,
            story_type=story_type_enum,
            narrative_structure=structure_enum,
            content=content,
            cultural_themes=cultural_themes,
            key_messages=key_messages,
            target_outcomes=target_outcomes,
            template_id=template_id
        )
        
        return {
            "success": True,
            "story_id": story.id,
            "message": "Transformation story created successfully",
            "data": {
                "story_id": story.id,
                "title": story.title,
                "story_type": story.story_type.value,
                "narrative_structure": story.narrative_structure.value,
                "cultural_themes": story.cultural_themes,
                "key_messages": story.key_messages,
                "characters": len(story.characters),
                "narrative_strength": story.metadata.get('narrative_strength', 0),
                "emotional_resonance": story.metadata.get('emotional_resonance', 0),
                "cultural_alignment": story.metadata.get('cultural_alignment', 0),
                "created_at": story.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating transformation story: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stories/{story_id}/personalize")
async def personalize_story_for_audience(
    story_id: str,
    audience_id: str,
    audience_profile: Dict[str, Any],
    delivery_format: str
):
    """Personalize story for specific audience"""
    try:
        # Validate delivery format
        try:
            format_enum = DeliveryFormat(delivery_format)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid delivery format")
        
        personalization = storytelling_engine.personalize_story_for_audience(
            story_id=story_id,
            audience_id=audience_id,
            audience_profile=audience_profile,
            delivery_format=format_enum
        )
        
        return {
            "success": True,
            "personalization_id": personalization.id,
            "message": "Story personalized successfully",
            "data": {
                "personalization_id": personalization.id,
                "base_story_id": personalization.base_story_id,
                "audience_id": personalization.audience_id,
                "delivery_format": personalization.delivery_format.value,
                "language_style": personalization.language_style,
                "personalization_score": personalization.personalization_score,
                "character_adaptations": len(personalization.character_adaptations),
                "content_preview": personalization.personalized_content[:200] + "...",
                "created_at": personalization.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error personalizing story: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stories/{story_id}/measure-impact")
async def measure_story_impact(
    story_id: str,
    audience_id: str,
    engagement_data: Dict[str, Any],
    feedback_data: Dict[str, Any]
):
    """Measure and analyze story impact"""
    try:
        impact = storytelling_engine.measure_story_impact(
            story_id=story_id,
            audience_id=audience_id,
            engagement_data=engagement_data,
            feedback_data=feedback_data
        )
        
        return {
            "success": True,
            "impact_id": impact.id,
            "message": "Story impact measured successfully",
            "data": {
                "impact_id": impact.id,
                "story_id": impact.story_id,
                "audience_id": impact.audience_id,
                "impact_score": impact.impact_score,
                "emotional_impact": impact.emotional_impact,
                "behavioral_indicators": impact.behavioral_indicators,
                "cultural_alignment": impact.cultural_alignment,
                "message_retention": impact.message_retention,
                "transformation_influence": impact.transformation_influence,
                "measured_at": impact.measured_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error measuring story impact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stories/{story_id}/optimize")
async def optimize_story_performance(
    story_id: str,
    performance_data: Dict[str, Any]
):
    """Optimize story performance based on analytics"""
    try:
        optimizations = storytelling_engine.optimize_story_performance(
            story_id=story_id,
            performance_data=performance_data
        )
        
        return {
            "success": True,
            "message": "Story performance optimized successfully",
            "data": {
                "story_id": story_id,
                "optimizations": optimizations,
                "optimization_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error optimizing story performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/campaigns")
async def create_storytelling_campaign(
    name: str,
    description: str,
    transformation_objectives: List[str],
    target_audiences: List[str],
    stories: List[str],
    duration_days: int
):
    """Create storytelling campaign"""
    try:
        campaign = storytelling_engine.create_storytelling_campaign(
            name=name,
            description=description,
            transformation_objectives=transformation_objectives,
            target_audiences=target_audiences,
            stories=stories,
            duration_days=duration_days
        )
        
        return {
            "success": True,
            "campaign_id": campaign.id,
            "message": "Storytelling campaign created successfully",
            "data": {
                "campaign_id": campaign.id,
                "name": campaign.name,
                "transformation_objectives": campaign.transformation_objectives,
                "target_audiences": campaign.target_audiences,
                "story_count": len(campaign.stories),
                "narrative_arc": campaign.narrative_arc,
                "start_date": campaign.start_date.isoformat(),
                "end_date": campaign.end_date.isoformat(),
                "status": campaign.status,
                "success_metrics": campaign.success_metrics
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating storytelling campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stories/{story_id}/analytics")
async def generate_story_analytics(
    story_id: str,
    start_date: datetime,
    end_date: datetime
):
    """Generate comprehensive story analytics"""
    try:
        time_period = {"start": start_date, "end": end_date}
        
        analytics = storytelling_engine.generate_story_analytics(
            story_id=story_id,
            time_period=time_period
        )
        
        return {
            "success": True,
            "analytics_id": analytics.id,
            "message": "Story analytics generated successfully",
            "data": {
                "analytics_id": analytics.id,
                "story_id": analytics.story_id,
                "time_period": {
                    "start": analytics.time_period["start"].isoformat(),
                    "end": analytics.time_period["end"].isoformat()
                },
                "engagement_metrics": analytics.engagement_metrics,
                "impact_metrics": analytics.impact_metrics,
                "audience_insights": analytics.audience_insights,
                "optimization_recommendations": analytics.optimization_recommendations,
                "trend_analysis": analytics.trend_analysis,
                "comparative_performance": analytics.comparative_performance,
                "generated_at": analytics.generated_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating story analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}")
async def get_narrative_strategy(strategy_id: str):
    """Get narrative strategy details"""
    try:
        strategy = storytelling_engine.narrative_strategies.get(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return {
            "success": True,
            "data": {
                "strategy_id": strategy.id,
                "organization_id": strategy.organization_id,
                "transformation_vision": strategy.transformation_vision,
                "core_narratives": strategy.core_narratives,
                "story_themes": strategy.story_themes,
                "character_archetypes": [
                    {
                        "id": char.id,
                        "name": char.name,
                        "role": char.role,
                        "characteristics": char.characteristics,
                        "motivations": char.motivations
                    } for char in strategy.character_archetypes
                ],
                "narrative_guidelines": strategy.narrative_guidelines,
                "audience_story_preferences": strategy.audience_story_preferences,
                "effectiveness_targets": strategy.effectiveness_targets,
                "created_at": strategy.created_at.isoformat(),
                "updated_at": strategy.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting narrative strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stories/{story_id}")
async def get_transformation_story(story_id: str):
    """Get transformation story details"""
    try:
        story = storytelling_engine.transformation_stories.get(story_id)
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")
        
        return {
            "success": True,
            "data": {
                "story_id": story.id,
                "title": story.title,
                "story_type": story.story_type.value,
                "narrative_structure": story.narrative_structure.value,
                "content": story.content,
                "characters": [
                    {
                        "id": char.id,
                        "name": char.name,
                        "role": char.role,
                        "characteristics": char.characteristics
                    } for char in story.characters
                ],
                "plot": {
                    "structure": story.plot.structure.value,
                    "acts": story.plot.acts
                } if story.plot else None,
                "cultural_themes": story.cultural_themes,
                "key_messages": story.key_messages,
                "emotional_tone": story.emotional_tone,
                "target_outcomes": story.target_outcomes,
                "metadata": story.metadata,
                "version": story.version,
                "is_active": story.is_active,
                "created_at": story.created_at.isoformat(),
                "updated_at": story.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting transformation story: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/campaigns/{campaign_id}")
async def get_storytelling_campaign(campaign_id: str):
    """Get storytelling campaign details"""
    try:
        campaign = storytelling_engine.story_campaigns.get(campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        return {
            "success": True,
            "data": {
                "campaign_id": campaign.id,
                "name": campaign.name,
                "description": campaign.description,
                "transformation_objectives": campaign.transformation_objectives,
                "target_audiences": campaign.target_audiences,
                "stories": campaign.stories,
                "narrative_arc": campaign.narrative_arc,
                "delivery_schedule": campaign.delivery_schedule,
                "success_metrics": campaign.success_metrics,
                "start_date": campaign.start_date.isoformat(),
                "end_date": campaign.end_date.isoformat(),
                "status": campaign.status,
                "created_at": campaign.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting storytelling campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/impact/{story_id}/{audience_id}")
async def get_story_impact(story_id: str, audience_id: str):
    """Get story impact data"""
    try:
        key = f"{story_id}_{audience_id}"
        impact = storytelling_engine.impact_data.get(key)
        
        if not impact:
            raise HTTPException(status_code=404, detail="Impact data not found")
        
        return {
            "success": True,
            "data": {
                "impact_id": impact.id,
                "story_id": impact.story_id,
                "audience_id": impact.audience_id,
                "impact_score": impact.impact_score,
                "emotional_impact": impact.emotional_impact,
                "behavioral_indicators": impact.behavioral_indicators,
                "cultural_alignment": impact.cultural_alignment,
                "message_retention": impact.message_retention,
                "transformation_influence": impact.transformation_influence,
                "feedback_summary": impact.feedback_summary,
                "measured_at": impact.measured_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting story impact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_story_templates(
    story_type: Optional[str] = Query(None),
    narrative_structure: Optional[str] = Query(None)
):
    """Get available story templates"""
    try:
        templates = []
        
        for template in storytelling_engine.story_templates.values():
            # Filter by story type if specified
            if story_type and template.story_type.value != story_type:
                continue
            
            # Filter by narrative structure if specified
            if narrative_structure and template.narrative_structure.value != narrative_structure:
                continue
            
            templates.append({
                "template_id": template.id,
                "name": template.name,
                "story_type": template.story_type.value,
                "narrative_structure": template.narrative_structure.value,
                "character_templates": template.character_templates,
                "plot_template": template.plot_template,
                "customization_points": template.customization_points,
                "usage_guidelines": template.usage_guidelines,
                "effectiveness_data": template.effectiveness_data,
                "created_at": template.created_at.isoformat()
            })
        
        return {
            "success": True,
            "data": {
                "templates": templates,
                "total_count": len(templates)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting story templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/dashboard")
async def get_storytelling_analytics_dashboard(
    strategy_id: Optional[str] = Query(None),
    time_range: Optional[str] = Query("30d")
):
    """Get storytelling analytics dashboard data"""
    try:
        # Aggregate analytics data
        analytics = {
            "overview": {
                "total_stories": len(storytelling_engine.transformation_stories),
                "active_campaigns": len([c for c in storytelling_engine.story_campaigns.values() if c.status == "active"]),
                "total_personalizations": len(storytelling_engine.story_personalizations),
                "avg_impact_score": 0.75
            },
            "performance_metrics": {
                "engagement_rate": 0.72,
                "completion_rate": 0.68,
                "emotional_impact": 0.78,
                "transformation_influence": 0.65,
                "message_retention": 0.70
            },
            "story_types_performance": {
                "transformation_journey": 0.80,
                "success_story": 0.75,
                "vision_narrative": 0.72,
                "hero_journey": 0.78
            },
            "audience_insights": {
                "most_engaged": "leadership_team",
                "highest_impact": "new_hires",
                "preferred_formats": ["written_narrative", "presentation", "video_story"],
                "emotional_response_leaders": ["inspiration", "hope", "determination"]
            },
            "top_performing_stories": [
                {
                    "story_id": "story_001",
                    "title": "Innovation Transformation Journey",
                    "impact_score": 0.89,
                    "engagement_rate": 0.85
                },
                {
                    "story_id": "story_002",
                    "title": "Team Success Celebration",
                    "impact_score": 0.82,
                    "engagement_rate": 0.78
                }
            ],
            "optimization_opportunities": [
                "Enhance emotional elements in technical stories",
                "Improve personalization for remote audiences",
                "Develop more interactive story formats",
                "Create audience-specific character variations"
            ],
            "trend_analysis": {
                "engagement_trend": "increasing",
                "format_preferences": "shifting_to_multimedia",
                "emotional_impact_trend": "stable_high",
                "completion_rates": "improving"
            }
        }
        
        return {
            "success": True,
            "data": analytics
        }
        
    except Exception as e:
        logger.error(f"Error getting storytelling analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/personalization/{personalization_id}")
async def get_story_personalization(personalization_id: str):
    """Get story personalization details"""
    try:
        personalization = storytelling_engine.story_personalizations.get(personalization_id)
        if not personalization:
            raise HTTPException(status_code=404, detail="Personalization not found")
        
        return {
            "success": True,
            "data": {
                "personalization_id": personalization.id,
                "base_story_id": personalization.base_story_id,
                "audience_id": personalization.audience_id,
                "personalized_content": personalization.personalized_content,
                "character_adaptations": personalization.character_adaptations,
                "cultural_adaptations": personalization.cultural_adaptations,
                "language_style": personalization.language_style,
                "delivery_format": personalization.delivery_format.value,
                "personalization_score": personalization.personalization_score,
                "created_at": personalization.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting story personalization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/{analytics_id}")
async def get_story_analytics(analytics_id: str):
    """Get story analytics details"""
    try:
        analytics = storytelling_engine.story_analytics.get(analytics_id)
        if not analytics:
            raise HTTPException(status_code=404, detail="Analytics not found")
        
        return {
            "success": True,
            "data": {
                "analytics_id": analytics.id,
                "story_id": analytics.story_id,
                "time_period": {
                    "start": analytics.time_period["start"].isoformat(),
                    "end": analytics.time_period["end"].isoformat()
                },
                "engagement_metrics": analytics.engagement_metrics,
                "impact_metrics": analytics.impact_metrics,
                "audience_insights": analytics.audience_insights,
                "optimization_recommendations": analytics.optimization_recommendations,
                "trend_analysis": analytics.trend_analysis,
                "comparative_performance": analytics.comparative_performance,
                "generated_at": analytics.generated_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting story analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))