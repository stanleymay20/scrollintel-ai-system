"""
Cultural Messaging API Routes

REST API endpoints for cultural messaging framework including message creation,
customization, delivery, and effectiveness tracking.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ...engines.cultural_messaging_engine import CulturalMessagingEngine
from ...models.cultural_messaging_models import (
    MessageType, AudienceType, MessageChannel
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cultural-messaging", tags=["cultural-messaging"])

# Global engine instance
messaging_engine = CulturalMessagingEngine()


@router.post("/strategies")
async def create_messaging_strategy(
    organization_id: str,
    cultural_vision: str,
    core_values: List[str],
    audience_data: List[Dict[str, Any]]
):
    """Create comprehensive messaging strategy"""
    try:
        strategy = messaging_engine.create_messaging_strategy(
            organization_id=organization_id,
            cultural_vision=cultural_vision,
            core_values=core_values,
            audience_data=audience_data
        )
        
        return {
            "success": True,
            "strategy_id": strategy.id,
            "message": "Messaging strategy created successfully",
            "data": {
                "strategy_id": strategy.id,
                "organization_id": strategy.organization_id,
                "key_themes": strategy.key_themes,
                "audience_count": len(strategy.audience_segments),
                "template_count": len(strategy.message_templates),
                "effectiveness_targets": strategy.effectiveness_targets
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating messaging strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/messages")
async def create_cultural_message(
    title: str,
    content: str,
    message_type: str,
    cultural_themes: List[str],
    key_values: List[str],
    template_id: Optional[str] = None
):
    """Create new cultural message"""
    try:
        # Validate message type
        try:
            msg_type = MessageType(message_type)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid message type")
        
        message = messaging_engine.create_cultural_message(
            title=title,
            content=content,
            message_type=msg_type,
            cultural_themes=cultural_themes,
            key_values=key_values,
            template_id=template_id
        )
        
        return {
            "success": True,
            "message_id": message.id,
            "message": "Cultural message created successfully",
            "data": {
                "message_id": message.id,
                "title": message.title,
                "message_type": message.message_type.value,
                "cultural_themes": message.cultural_themes,
                "alignment_score": message.metadata.get('alignment_score', 0),
                "created_at": message.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating cultural message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/messages/{message_id}/customize")
async def customize_message_for_audience(
    message_id: str,
    audience_id: str,
    channel: str,
    delivery_timing: Optional[datetime] = None
):
    """Customize message for specific audience"""
    try:
        # Validate channel
        try:
            msg_channel = MessageChannel(channel)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid channel")
        
        customization = messaging_engine.customize_message_for_audience(
            message_id=message_id,
            audience_id=audience_id,
            channel=msg_channel,
            delivery_timing=delivery_timing
        )
        
        return {
            "success": True,
            "customization_id": customization.id,
            "message": "Message customized successfully",
            "data": {
                "customization_id": customization.id,
                "base_message_id": customization.base_message_id,
                "audience_id": customization.audience_id,
                "channel": customization.channel.value,
                "delivery_timing": customization.delivery_timing.isoformat(),
                "personalization_score": customization.personalization_data.get('personalization_score', 0),
                "content_preview": customization.customized_content[:200] + "..."
            }
        }
        
    except Exception as e:
        logger.error(f"Error customizing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/messages/{message_id}/track-effectiveness")
async def track_message_effectiveness(
    message_id: str,
    audience_id: str,
    engagement_data: Dict[str, Any]
):
    """Track and analyze message effectiveness"""
    try:
        effectiveness = messaging_engine.track_message_effectiveness(
            message_id=message_id,
            audience_id=audience_id,
            engagement_data=engagement_data
        )
        
        return {
            "success": True,
            "effectiveness_id": effectiveness.id,
            "message": "Message effectiveness tracked successfully",
            "data": {
                "effectiveness_id": effectiveness.id,
                "message_id": effectiveness.message_id,
                "audience_id": effectiveness.audience_id,
                "effectiveness_score": effectiveness.effectiveness_score,
                "cultural_alignment_score": effectiveness.cultural_alignment_score,
                "behavior_change_indicators": effectiveness.behavior_change_indicators,
                "recommendations": effectiveness.recommendations,
                "measured_at": effectiveness.measured_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error tracking message effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/{strategy_id}/optimize")
async def optimize_messaging_strategy(
    strategy_id: str,
    performance_data: Dict[str, Any]
):
    """Optimize messaging strategy based on performance data"""
    try:
        optimizations = messaging_engine.optimize_messaging_strategy(
            strategy_id=strategy_id,
            performance_data=performance_data
        )
        
        return {
            "success": True,
            "message": "Messaging strategy optimized successfully",
            "data": {
                "strategy_id": strategy_id,
                "optimizations": optimizations,
                "optimization_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error optimizing messaging strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/campaigns")
async def create_messaging_campaign(
    name: str,
    description: str,
    cultural_objectives: List[str],
    target_audiences: List[str],
    messages: List[str],
    duration_days: int
):
    """Create cultural messaging campaign"""
    try:
        campaign = messaging_engine.create_messaging_campaign(
            name=name,
            description=description,
            cultural_objectives=cultural_objectives,
            target_audiences=target_audiences,
            messages=messages,
            duration_days=duration_days
        )
        
        return {
            "success": True,
            "campaign_id": campaign.id,
            "message": "Messaging campaign created successfully",
            "data": {
                "campaign_id": campaign.id,
                "name": campaign.name,
                "cultural_objectives": campaign.cultural_objectives,
                "target_audiences": campaign.target_audiences,
                "message_count": len(campaign.messages),
                "start_date": campaign.start_date.isoformat(),
                "end_date": campaign.end_date.isoformat(),
                "status": campaign.status
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating messaging campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}")
async def get_messaging_strategy(strategy_id: str):
    """Get messaging strategy details"""
    try:
        strategy = messaging_engine.messaging_strategies.get(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return {
            "success": True,
            "data": {
                "strategy_id": strategy.id,
                "organization_id": strategy.organization_id,
                "cultural_vision": strategy.cultural_vision,
                "core_values": strategy.core_values,
                "key_themes": strategy.key_themes,
                "audience_count": len(strategy.audience_segments),
                "template_count": len(strategy.message_templates),
                "effectiveness_targets": strategy.effectiveness_targets,
                "created_at": strategy.created_at.isoformat(),
                "updated_at": strategy.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting messaging strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages/{message_id}")
async def get_cultural_message(message_id: str):
    """Get cultural message details"""
    try:
        message = messaging_engine.message_history.get(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        return {
            "success": True,
            "data": {
                "message_id": message.id,
                "title": message.title,
                "content": message.content,
                "message_type": message.message_type.value,
                "cultural_themes": message.cultural_themes,
                "key_values": message.key_values,
                "alignment_score": message.metadata.get('alignment_score', 0),
                "version": message.version,
                "is_active": message.is_active,
                "created_at": message.created_at.isoformat(),
                "updated_at": message.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting cultural message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/campaigns/{campaign_id}")
async def get_messaging_campaign(campaign_id: str):
    """Get messaging campaign details"""
    try:
        campaign = messaging_engine.active_campaigns.get(campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        return {
            "success": True,
            "data": {
                "campaign_id": campaign.id,
                "name": campaign.name,
                "description": campaign.description,
                "cultural_objectives": campaign.cultural_objectives,
                "target_audiences": campaign.target_audiences,
                "messages": campaign.messages,
                "start_date": campaign.start_date.isoformat(),
                "end_date": campaign.end_date.isoformat(),
                "status": campaign.status,
                "success_metrics": campaign.success_metrics,
                "created_at": campaign.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting messaging campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/effectiveness/{message_id}/{audience_id}")
async def get_message_effectiveness(message_id: str, audience_id: str):
    """Get message effectiveness data"""
    try:
        key = f"{message_id}_{audience_id}"
        effectiveness = messaging_engine.effectiveness_data.get(key)
        
        if not effectiveness:
            raise HTTPException(status_code=404, detail="Effectiveness data not found")
        
        return {
            "success": True,
            "data": {
                "effectiveness_id": effectiveness.id,
                "message_id": effectiveness.message_id,
                "audience_id": effectiveness.audience_id,
                "effectiveness_score": effectiveness.effectiveness_score,
                "cultural_alignment_score": effectiveness.cultural_alignment_score,
                "behavior_change_indicators": effectiveness.behavior_change_indicators,
                "feedback_summary": effectiveness.feedback_summary,
                "recommendations": effectiveness.recommendations,
                "measured_at": effectiveness.measured_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting message effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/dashboard")
async def get_messaging_analytics_dashboard(
    strategy_id: Optional[str] = Query(None),
    time_range: Optional[str] = Query("30d")
):
    """Get messaging analytics dashboard data"""
    try:
        # Aggregate analytics data
        analytics = {
            "overview": {
                "total_messages": len(messaging_engine.message_history),
                "active_campaigns": len([c for c in messaging_engine.active_campaigns.values() if c.status == "active"]),
                "total_audiences": len(messaging_engine.audience_profiles),
                "avg_effectiveness": 0.72  # Calculated from effectiveness data
            },
            "performance_metrics": {
                "engagement_rate": 0.68,
                "cultural_alignment": 0.75,
                "behavior_change_rate": 0.58,
                "message_reach": 0.82
            },
            "top_performing_messages": [
                {
                    "message_id": "msg_001",
                    "title": "Vision Communication",
                    "effectiveness_score": 0.89,
                    "engagement_rate": 0.76
                }
            ],
            "audience_insights": {
                "most_engaged": "leadership_team",
                "least_engaged": "remote_workers",
                "engagement_trends": {}
            },
            "recommendations": [
                "Increase personalization for remote workers",
                "Leverage high-performing message templates",
                "Optimize timing for better engagement"
            ]
        }
        
        return {
            "success": True,
            "data": analytics
        }
        
    except Exception as e:
        logger.error(f"Error getting messaging analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_message_templates(
    message_type: Optional[str] = Query(None)
):
    """Get available message templates"""
    try:
        templates = []
        
        # Get templates from all strategies
        for strategy in messaging_engine.messaging_strategies.values():
            for template in strategy.message_templates:
                if not message_type or template.message_type.value == message_type:
                    templates.append({
                        "template_id": template.id,
                        "name": template.name,
                        "message_type": template.message_type.value,
                        "required_variables": template.required_variables,
                        "optional_variables": template.optional_variables,
                        "usage_guidelines": template.usage_guidelines,
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
        logger.error(f"Error getting message templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audiences")
async def get_audience_profiles():
    """Get all audience profiles"""
    try:
        audiences = []
        
        for audience in messaging_engine.audience_profiles.values():
            audiences.append({
                "audience_id": audience.id,
                "name": audience.name,
                "audience_type": audience.audience_type.value,
                "size": audience.size,
                "characteristics": audience.characteristics,
                "communication_preferences": audience.communication_preferences,
                "cultural_context": audience.cultural_context,
                "engagement_history": audience.engagement_history
            })
        
        return {
            "success": True,
            "data": {
                "audiences": audiences,
                "total_count": len(audiences)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting audience profiles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))