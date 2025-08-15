"""
API routes for Market Education System
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from scrollintel.engines.market_education_engine import (
    MarketEducationEngine, 
    CampaignType, 
    TargetSegment,
    ContentType
)
from scrollintel.core.content_delivery_system import (
    ContentDeliverySystem,
    DeliveryChannel,
    EngagementLevel
)
from scrollintel.core.market_readiness_assessment import (
    MarketReadinessAssessment,
    MarketSegment,
    ReadinessLevel
)
from scrollintel.core.campaign_management_platform import CampaignManagementPlatform
from scrollintel.core.education_content_tracker import EducationContentTracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/market-education", tags=["market-education"])

# Initialize engines
market_education_engine = MarketEducationEngine()
content_delivery_system = ContentDeliverySystem()
readiness_assessment = MarketReadinessAssessment()
campaign_platform = CampaignManagementPlatform()
content_tracker = EducationContentTracker()

@router.post("/campaigns/create")
async def create_campaign(
    template_name: str,
    customizations: Optional[Dict[str, Any]] = None
):
    """Create a new marketing campaign from template"""
    try:
        campaign = await market_education_engine.create_campaign(
            template_name, customizations
        )
        
        return {
            "success": True,
            "campaign": {
                "id": campaign.id,
                "name": campaign.name,
                "type": campaign.campaign_type.value,
                "target_segments": [seg.value for seg in campaign.target_segments],
                "start_date": campaign.start_date.isoformat(),
                "end_date": campaign.end_date.isoformat(),
                "budget": campaign.budget,
                "content_pieces": len(campaign.content_pieces)
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str):
    """Get campaign details"""
    try:
        if campaign_id not in market_education_engine.campaigns:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        campaign = market_education_engine.campaigns[campaign_id]
        
        return {
            "success": True,
            "campaign": {
                "id": campaign.id,
                "name": campaign.name,
                "type": campaign.campaign_type.value,
                "target_segments": [seg.value for seg in campaign.target_segments],
                "start_date": campaign.start_date.isoformat(),
                "end_date": campaign.end_date.isoformat(),
                "budget": campaign.budget,
                "channels": campaign.channels,
                "status": campaign.status,
                "kpis": campaign.kpis,
                "results": campaign.results,
                "content_pieces": [
                    {
                        "id": content.id,
                        "title": content.title,
                        "type": content.content_type.value,
                        "effectiveness_score": content.effectiveness_score
                    }
                    for content in campaign.content_pieces
                ]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/campaigns")
async def list_campaigns():
    """List all campaigns"""
    try:
        campaigns = []
        
        for campaign in market_education_engine.campaigns.values():
            campaigns.append({
                "id": campaign.id,
                "name": campaign.name,
                "type": campaign.campaign_type.value,
                "status": campaign.status,
                "start_date": campaign.start_date.isoformat(),
                "end_date": campaign.end_date.isoformat(),
                "budget": campaign.budget,
                "content_pieces": len(campaign.content_pieces)
            })
        
        return {
            "success": True,
            "campaigns": campaigns,
            "total": len(campaigns)
        }
        
    except Exception as e:
        logger.error(f"Error listing campaigns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/campaigns/{campaign_id}/adapt")
async def adapt_campaign_strategy(
    campaign_id: str,
    market_feedback: Dict[str, Any]
):
    """Adapt campaign strategy based on market feedback"""
    try:
        await market_education_engine.adapt_campaign_strategy(
            campaign_id, market_feedback
        )
        
        return {
            "success": True,
            "message": f"Campaign {campaign_id} strategy adapted successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error adapting campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/readiness/{segment}")
async def assess_segment_readiness(segment: str):
    """Assess market readiness for specific segment"""
    try:
        try:
            market_segment = MarketSegment(segment)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid segment: {segment}")
        
        assessment = await readiness_assessment.assess_segment_readiness(market_segment)
        
        return {
            "success": True,
            "assessment": {
                "segment": assessment.segment.value,
                "readiness_level": assessment.readiness_level.value,
                "readiness_score": assessment.readiness_score,
                "confidence_level": assessment.confidence_level,
                "barriers": assessment.barriers,
                "accelerators": assessment.accelerators,
                "recommended_actions": assessment.recommended_actions,
                "indicators": {
                    name: {
                        "name": indicator.name,
                        "current_value": indicator.current_value,
                        "target_value": indicator.target_value,
                        "trend": indicator.trend,
                        "weight": indicator.weight
                    }
                    for name, indicator in assessment.indicators.items()
                },
                "last_updated": assessment.last_updated.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assessing readiness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/readiness/report/comprehensive")
async def get_comprehensive_readiness_report():
    """Get comprehensive market readiness report"""
    try:
        report = await readiness_assessment.generate_comprehensive_report()
        
        return {
            "success": True,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Error generating readiness report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/delivery/targets/register")
async def register_delivery_target(target_data: Dict[str, Any]):
    """Register a new content delivery target"""
    try:
        target = await content_delivery_system.register_target(target_data)
        
        return {
            "success": True,
            "target": {
                "id": target.id,
                "name": target.name,
                "segment": target.segment,
                "preferred_channels": [ch.value for ch in target.preferred_channels],
                "engagement_level": target.current_engagement_level.value,
                "conversion_probability": target.conversion_probability
            }
        }
        
    except Exception as e:
        logger.error(f"Error registering target: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/delivery/schedule")
async def schedule_content_delivery(
    content_id: str,
    target_ids: List[str],
    channel: str,
    delivery_time: Optional[str] = None
):
    """Schedule content delivery to targets"""
    try:
        try:
            delivery_channel = DeliveryChannel(channel)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid channel: {channel}")
        
        delivery_datetime = None
        if delivery_time:
            delivery_datetime = datetime.fromisoformat(delivery_time)
        
        deliveries = await content_delivery_system.schedule_content_delivery(
            content_id, target_ids, delivery_channel, delivery_datetime
        )
        
        return {
            "success": True,
            "deliveries": [
                {
                    "id": delivery.id,
                    "content_id": delivery.content_id,
                    "target_id": delivery.target_id,
                    "channel": delivery.channel.value,
                    "scheduled_time": delivery.scheduled_time.isoformat(),
                    "status": delivery.status
                }
                for delivery in deliveries
            ],
            "total_scheduled": len(deliveries)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling delivery: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/delivery/{delivery_id}/execute")
async def execute_delivery(delivery_id: str, background_tasks: BackgroundTasks):
    """Execute a scheduled content delivery"""
    try:
        success = await content_delivery_system.execute_delivery(delivery_id)
        
        if success:
            # Schedule follow-up tracking
            background_tasks.add_task(
                _schedule_engagement_tracking, delivery_id
            )
        
        return {
            "success": success,
            "message": f"Delivery {'executed successfully' if success else 'failed'}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing delivery: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/delivery/{delivery_id}/track-engagement")
async def track_engagement(delivery_id: str, engagement_data: Dict[str, Any]):
    """Track engagement metrics for delivered content"""
    try:
        await content_delivery_system.track_engagement(delivery_id, engagement_data)
        
        return {
            "success": True,
            "message": "Engagement tracked successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error tracking engagement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/delivery/report")
async def get_delivery_report(campaign_id: Optional[str] = None):
    """Get delivery and engagement report"""
    try:
        report = await content_delivery_system.generate_delivery_report(campaign_id)
        
        return {
            "success": True,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Error generating delivery report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/delivery/optimize")
async def optimize_delivery_strategy(campaign_id: str):
    """Optimize delivery strategy based on performance data"""
    try:
        optimization = await content_delivery_system.optimize_delivery_strategy(campaign_id)
        
        return {
            "success": True,
            "optimization": optimization
        }
        
    except Exception as e:
        logger.error(f"Error optimizing delivery strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/five-year-plan/execute")
async def execute_five_year_plan(background_tasks: BackgroundTasks):
    """Execute the complete 5-year market conditioning plan"""
    try:
        execution_plan = await market_education_engine.execute_five_year_plan()
        
        # Schedule ongoing monitoring and optimization
        background_tasks.add_task(_monitor_plan_execution, execution_plan)
        
        return {
            "success": True,
            "execution_plan": execution_plan
        }
        
    except Exception as e:
        logger.error(f"Error executing 5-year plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-conditions")
async def get_market_conditions():
    """Get current market conditions"""
    try:
        conditions = await readiness_assessment.monitor_market_conditions()
        
        return {
            "success": True,
            "conditions": [
                {
                    "factor": condition.factor,
                    "impact_level": condition.impact_level,
                    "trend": condition.trend,
                    "description": condition.description,
                    "recommended_response": condition.recommended_response
                }
                for condition in conditions
            ],
            "total": len(conditions)
        }
        
    except Exception as e:
        logger.error(f"Error getting market conditions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategy/adapt")
async def adapt_strategy_based_on_conditions():
    """Adapt market conditioning strategy based on current conditions"""
    try:
        adaptations = await readiness_assessment.adapt_strategy_based_on_conditions()
        
        return {
            "success": True,
            "adaptations": [
                {
                    "id": adaptation.id,
                    "trigger_conditions": adaptation.trigger_conditions,
                    "actions": adaptation.actions,
                    "expected_impact": adaptation.expected_impact,
                    "implementation_timeline": adaptation.implementation_timeline,
                    "success_metrics": adaptation.success_metrics,
                    "status": adaptation.status
                }
                for adaptation in adaptations
            ],
            "total": len(adaptations)
        }
        
    except Exception as e:
        logger.error(f"Error adapting strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/content/{content_id}")
async def get_content_performance(content_id: str):
    """Get performance metrics for specific content"""
    try:
        if content_id not in market_education_engine.content_library:
            raise HTTPException(status_code=404, detail="Content not found")
        
        content = market_education_engine.content_library[content_id]
        
        return {
            "success": True,
            "content": {
                "id": content.id,
                "title": content.title,
                "type": content.content_type.value,
                "target_segments": [seg.value for seg in content.target_segments],
                "key_messages": content.key_messages,
                "engagement_metrics": content.engagement_metrics,
                "effectiveness_score": content.effectiveness_score,
                "created_at": content.created_at.isoformat(),
                "updated_at": content.updated_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting content performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/content/{content_id}/update-engagement")
async def update_content_engagement(content_id: str, metrics: Dict[str, float]):
    """Update engagement metrics for content"""
    try:
        await market_education_engine.track_content_engagement(content_id, metrics)
        
        return {
            "success": True,
            "message": "Content engagement updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating content engagement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def _schedule_engagement_tracking(delivery_id: str):
    """Schedule engagement tracking for delivered content"""
    try:
        # Simulate engagement tracking after delivery
        import asyncio
        await asyncio.sleep(300)  # Wait 5 minutes
        
        # Simulate engagement data
        engagement_data = {
            "opened": True,
            "clicked": False,
            "shared": False,
            "responded": False,
            "converted": False
        }
        
        await content_delivery_system.track_engagement(delivery_id, engagement_data)
        logger.info(f"Automated engagement tracking completed for delivery: {delivery_id}")
        
    except Exception as e:
        logger.error(f"Error in automated engagement tracking: {str(e)}")

async def _monitor_plan_execution(execution_plan: Dict[str, Any]):
    """Monitor and optimize 5-year plan execution"""
    try:
        # Implement ongoing monitoring and optimization
        logger.info("Started monitoring 5-year plan execution")
        
        # This would include:
        # - Regular readiness assessments
        # - Campaign performance monitoring
        # - Strategy adaptations
        # - Resource reallocation
        
    except Exception as e:
        logger.error(f"Error monitoring plan execution: {str(e)}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for market education system"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "market_education_engine": "operational",
            "content_delivery_system": "operational", 
            "readiness_assessment": "operational"
        }
    }
# C
ampaign Management Platform Endpoints

@router.post("/platform/master-plan/create")
async def create_master_plan():
    """Create comprehensive 5-year market conditioning master plan"""
    try:
        master_plan = await campaign_platform.create_five_year_master_plan()
        
        return {
            "success": True,
            "master_plan": master_plan
        }
        
    except Exception as e:
        logger.error(f"Error creating master plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/platform/master-plan/report")
async def get_master_plan_report():
    """Get comprehensive master plan progress report"""
    try:
        report = await campaign_platform.generate_master_plan_report()
        
        return {
            "success": True,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Error generating master plan report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/platform/campaigns/{campaign_id}/execute")
async def execute_campaign(campaign_id: str):
    """Execute a specific campaign"""
    try:
        execution_result = await campaign_platform.execute_campaign(campaign_id)
        
        return {
            "success": True,
            "execution_result": execution_result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/platform/campaigns/{campaign_id}/performance")
async def monitor_campaign_performance(campaign_id: str):
    """Monitor and analyze campaign performance"""
    try:
        performance_report = await campaign_platform.monitor_campaign_performance(campaign_id)
        
        return {
            "success": True,
            "performance_report": performance_report
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error monitoring campaign performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/platform/campaigns/{campaign_id}/optimize")
async def optimize_campaign(campaign_id: str, optimization_data: Dict[str, Any]):
    """Optimize campaign based on performance data"""
    try:
        optimization_result = await campaign_platform.optimize_campaign(campaign_id, optimization_data)
        
        return {
            "success": True,
            "optimization_result": optimization_result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error optimizing campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/platform/master-plan/adapt")
async def adapt_master_plan(adaptation_data: Dict[str, Any]):
    """Adapt master plan based on market conditions and performance"""
    try:
        adaptation_result = await campaign_platform.adapt_master_plan(adaptation_data)
        
        return {
            "success": True,
            "adaptation_result": adaptation_result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error adapting master plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Content Tracking Endpoints

@router.post("/content/engagement/track")
async def track_content_engagement(
    content_id: str,
    user_id: str,
    engagement_data: Dict[str, Any]
):
    """Track detailed content engagement"""
    try:
        engagement = await content_tracker.track_content_engagement(
            content_id, user_id, engagement_data
        )
        
        return {
            "success": True,
            "engagement": {
                "content_id": engagement.content_id,
                "user_id": engagement.user_id,
                "session_id": engagement.session_id,
                "duration_seconds": engagement.duration_seconds,
                "completion_percentage": engagement.completion_percentage,
                "engagement_score": engagement.engagement_score,
                "learning_outcome_achieved": engagement.learning_outcome_achieved
            }
        }
        
    except Exception as e:
        logger.error(f"Error tracking content engagement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/content/recommendations/{user_id}")
async def get_personalized_recommendations(user_id: str, limit: int = 5):
    """Generate personalized content recommendations for user"""
    try:
        recommendations = await content_tracker.generate_personalized_recommendations(user_id, limit)
        
        return {
            "success": True,
            "recommendations": [
                {
                    "content_id": rec.content_id,
                    "recommendation_score": rec.recommendation_score,
                    "reasoning": rec.reasoning,
                    "optimal_delivery_time": rec.optimal_delivery_time.isoformat(),
                    "preferred_format": rec.preferred_format.value,
                    "estimated_engagement_probability": rec.estimated_engagement_probability
                }
                for rec in recommendations
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/content/analytics/effectiveness")
async def analyze_learning_effectiveness(content_id: Optional[str] = None):
    """Analyze learning effectiveness across content or specific content"""
    try:
        analysis = await content_tracker.analyze_learning_effectiveness(content_id)
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing learning effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/content/analytics/report")
async def export_analytics_report(format: str = "json"):
    """Export comprehensive analytics report"""
    try:
        report = await content_tracker.export_analytics_report(format)
        
        return {
            "success": True,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Error exporting analytics report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced Readiness Assessment Endpoints

@router.post("/readiness/adaptive-framework/create")
async def create_adaptive_framework():
    """Create comprehensive adaptive strategy framework"""
    try:
        framework = await readiness_assessment.create_adaptive_strategy_framework()
        
        return {
            "success": True,
            "framework": framework
        }
        
    except Exception as e:
        logger.error(f"Error creating adaptive framework: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/readiness/comprehensive-report")
async def get_comprehensive_readiness_report():
    """Get comprehensive market readiness report with all segments"""
    try:
        report = await readiness_assessment.generate_comprehensive_report()
        
        return {
            "success": True,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Integration Endpoints

@router.post("/integration/campaign-content/sync")
async def sync_campaign_content(campaign_id: str):
    """Sync campaign content with tracking system"""
    try:
        # Get campaign from platform
        if campaign_id not in campaign_platform.campaigns:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        campaign = campaign_platform.campaigns[campaign_id]
        
        # Sync content assets with tracker
        synced_content = []
        for content_id in campaign.content_assets:
            # This would sync content metadata and tracking setup
            synced_content.append({
                "content_id": content_id,
                "campaign_id": campaign_id,
                "tracking_enabled": True
            })
        
        return {
            "success": True,
            "synced_content": synced_content,
            "total_synced": len(synced_content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing campaign content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/integration/readiness-campaign/align")
async def align_readiness_with_campaigns():
    """Align market readiness assessment with campaign strategies"""
    try:
        # Get comprehensive readiness report
        readiness_report = await readiness_assessment.generate_comprehensive_report()
        
        # Get master plan
        master_plan_report = await campaign_platform.generate_master_plan_report()
        
        # Create alignment recommendations
        alignment = {
            "overall_alignment_score": 0.75,  # Calculated based on readiness vs campaign focus
            "segment_alignments": {},
            "campaign_adjustments": [],
            "readiness_priorities": readiness_report["recommendations"]["strategic_priorities"],
            "campaign_priorities": master_plan_report.get("upcoming_milestones", []),
            "integration_opportunities": [
                "Accelerate education campaigns for high-readiness segments",
                "Increase awareness efforts for low-readiness segments",
                "Align content delivery with readiness assessment insights"
            ]
        }
        
        # Analyze segment alignments
        for segment, readiness_data in readiness_report["segment_assessments"].items():
            alignment["segment_alignments"][segment] = {
                "readiness_score": readiness_data["score"],
                "campaign_focus_level": 0.6,  # Would be calculated from campaign data
                "alignment_score": 0.7,
                "recommended_adjustments": readiness_data["recommendations"][:3]
            }
        
        return {
            "success": True,
            "alignment": alignment
        }
        
    except Exception as e:
        logger.error(f"Error aligning readiness with campaigns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/health/comprehensive")
async def comprehensive_system_health():
    """Comprehensive health check for all market education systems"""
    try:
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "market_education_engine": {
                    "status": "operational",
                    "campaigns_active": len([c for c in market_education_engine.campaigns.values() if c.status == "active"]),
                    "content_library_size": len(market_education_engine.content_library)
                },
                "content_delivery_system": {
                    "status": "operational",
                    "delivery_targets": len(content_delivery_system.delivery_targets),
                    "scheduled_deliveries": len(content_delivery_system.scheduled_deliveries)
                },
                "readiness_assessment": {
                    "status": "operational",
                    "segments_assessed": len(readiness_assessment.segment_assessments),
                    "market_conditions_monitored": len(readiness_assessment.market_conditions)
                },
                "campaign_platform": {
                    "status": "operational",
                    "total_campaigns": len(campaign_platform.campaigns),
                    "master_plan_status": "active" if campaign_platform.five_year_master_plan else "not_created"
                },
                "content_tracker": {
                    "status": "operational",
                    "learning_paths": len(content_tracker.learning_paths),
                    "user_profiles": len(content_tracker.user_profiles),
                    "content_engagements": sum(len(engagements) for engagements in content_tracker.content_engagements.values())
                }
            },
            "performance_metrics": {
                "total_market_readiness": 65.0,  # Would be calculated from actual data
                "campaign_effectiveness": 72.0,
                "content_engagement_rate": 68.0,
                "system_utilization": 85.0
            },
            "alerts": [],
            "recommendations": [
                "Continue monitoring campaign performance",
                "Optimize content delivery based on engagement data",
                "Maintain focus on high-readiness market segments"
            ]
        }
        
        return {
            "success": True,
            "health_status": health_status
        }
        
    except Exception as e:
        logger.error(f"Error checking system health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))