"""
Content Delivery and Tracking System - Targeted education content delivery
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)

class DeliveryChannel(Enum):
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    WEBINAR = "webinar"
    CONFERENCE = "conference"
    WEBSITE = "website"
    DIRECT_OUTREACH = "direct_outreach"
    PARTNER_NETWORK = "partner_network"
    MEDIA_PLACEMENT = "media_placement"

class EngagementLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class DeliveryTarget:
    id: str
    name: str
    segment: str
    contact_info: Dict[str, str]
    preferred_channels: List[DeliveryChannel]
    engagement_history: List[Dict] = field(default_factory=list)
    current_engagement_level: EngagementLevel = EngagementLevel.LOW
    last_contacted: Optional[datetime] = None
    conversion_probability: float = 0.0

@dataclass
class ContentDelivery:
    id: str
    content_id: str
    target_id: str
    channel: DeliveryChannel
    scheduled_time: datetime
    delivered_time: Optional[datetime] = None
    opened_time: Optional[datetime] = None
    engagement_metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "scheduled"
    follow_up_required: bool = False

@dataclass
class CampaignMetrics:
    campaign_id: str
    total_deliveries: int = 0
    successful_deliveries: int = 0
    total_opens: int = 0
    total_clicks: int = 0
    total_conversions: int = 0
    engagement_rate: float = 0.0
    conversion_rate: float = 0.0
    roi: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class ContentDeliverySystem:
    """
    Manages targeted delivery of education content across multiple channels
    with comprehensive tracking and optimization.
    """
    
    def __init__(self):
        self.delivery_targets: Dict[str, DeliveryTarget] = {}
        self.scheduled_deliveries: Dict[str, ContentDelivery] = {}
        self.campaign_metrics: Dict[str, CampaignMetrics] = {}
        self.channel_performance: Dict[DeliveryChannel, Dict[str, float]] = {}
        self.content_performance: Dict[str, Dict[str, float]] = {}
        
    async def register_target(self, target_data: Dict[str, Any]) -> DeliveryTarget:
        """Register a new delivery target"""
        try:
            target_id = f"target_{len(self.delivery_targets) + 1}"
            
            target = DeliveryTarget(
                id=target_id,
                name=target_data["name"],
                segment=target_data["segment"],
                contact_info=target_data.get("contact_info", {}),
                preferred_channels=[
                    DeliveryChannel(ch) for ch in target_data.get("preferred_channels", ["email"])
                ]
            )
            
            # Calculate initial conversion probability based on segment
            segment_probabilities = {
                "enterprise_ctos": 0.15,
                "tech_leaders": 0.25,
                "board_members": 0.08,
                "investors": 0.35,
                "industry_analysts": 0.45,
                "media": 0.30,
                "developers": 0.40
            }
            
            target.conversion_probability = segment_probabilities.get(
                target_data["segment"], 0.20
            )
            
            self.delivery_targets[target_id] = target
            logger.info(f"Registered delivery target: {target.name} ({target_id})")
            
            return target
            
        except Exception as e:
            logger.error(f"Error registering target: {str(e)}")
            raise
    
    async def schedule_content_delivery(
        self, 
        content_id: str, 
        target_ids: List[str], 
        channel: DeliveryChannel,
        delivery_time: Optional[datetime] = None
    ) -> List[ContentDelivery]:
        """Schedule content delivery to multiple targets"""
        try:
            if delivery_time is None:
                delivery_time = datetime.now() + timedelta(minutes=30)
            
            deliveries = []
            
            for target_id in target_ids:
                if target_id not in self.delivery_targets:
                    logger.warning(f"Target not found: {target_id}")
                    continue
                
                target = self.delivery_targets[target_id]
                
                # Check if channel is preferred for this target
                if channel not in target.preferred_channels:
                    # Find alternative channel
                    channel = target.preferred_channels[0] if target.preferred_channels else DeliveryChannel.EMAIL
                
                delivery_id = f"delivery_{len(self.scheduled_deliveries) + 1}"
                
                delivery = ContentDelivery(
                    id=delivery_id,
                    content_id=content_id,
                    target_id=target_id,
                    channel=channel,
                    scheduled_time=delivery_time
                )
                
                self.scheduled_deliveries[delivery_id] = delivery
                deliveries.append(delivery)
            
            logger.info(f"Scheduled {len(deliveries)} content deliveries")
            return deliveries
            
        except Exception as e:
            logger.error(f"Error scheduling deliveries: {str(e)}")
            raise
    
    async def execute_delivery(self, delivery_id: str) -> bool:
        """Execute a scheduled content delivery"""
        try:
            if delivery_id not in self.scheduled_deliveries:
                raise ValueError(f"Delivery not found: {delivery_id}")
            
            delivery = self.scheduled_deliveries[delivery_id]
            target = self.delivery_targets[delivery.target_id]
            
            # Simulate delivery execution based on channel
            success = await self._execute_channel_delivery(delivery, target)
            
            if success:
                delivery.delivered_time = datetime.now()
                delivery.status = "delivered"
                
                # Update target engagement history
                target.engagement_history.append({
                    "delivery_id": delivery_id,
                    "content_id": delivery.content_id,
                    "channel": delivery.channel.value,
                    "delivered_at": delivery.delivered_time.isoformat()
                })
                
                target.last_contacted = delivery.delivered_time
                
                logger.info(f"Successfully delivered content to {target.name}")
                return True
            else:
                delivery.status = "failed"
                delivery.follow_up_required = True
                logger.warning(f"Failed to deliver content to {target.name}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing delivery: {str(e)}")
            return False
    
    async def _execute_channel_delivery(self, delivery: ContentDelivery, target: DeliveryTarget) -> bool:
        """Execute delivery through specific channel"""
        channel_success_rates = {
            DeliveryChannel.EMAIL: 0.95,
            DeliveryChannel.SOCIAL_MEDIA: 0.85,
            DeliveryChannel.WEBINAR: 0.90,
            DeliveryChannel.CONFERENCE: 0.98,
            DeliveryChannel.WEBSITE: 0.99,
            DeliveryChannel.DIRECT_OUTREACH: 0.92,
            DeliveryChannel.PARTNER_NETWORK: 0.88,
            DeliveryChannel.MEDIA_PLACEMENT: 0.93
        }
        
        success_rate = channel_success_rates.get(delivery.channel, 0.90)
        
        # Factor in target engagement level
        engagement_multipliers = {
            EngagementLevel.LOW: 0.8,
            EngagementLevel.MEDIUM: 1.0,
            EngagementLevel.HIGH: 1.2,
            EngagementLevel.VERY_HIGH: 1.4
        }
        
        multiplier = engagement_multipliers.get(target.current_engagement_level, 1.0)
        final_success_rate = min(0.99, success_rate * multiplier)
        
        # Simulate delivery
        import random
        return random.random() < final_success_rate
    
    async def track_engagement(self, delivery_id: str, engagement_data: Dict[str, Any]):
        """Track engagement metrics for delivered content"""
        try:
            if delivery_id not in self.scheduled_deliveries:
                raise ValueError(f"Delivery not found: {delivery_id}")
            
            delivery = self.scheduled_deliveries[delivery_id]
            target = self.delivery_targets[delivery.target_id]
            
            # Update delivery metrics
            delivery.engagement_metrics.update(engagement_data)
            
            # Track opening
            if engagement_data.get("opened", False) and not delivery.opened_time:
                delivery.opened_time = datetime.now()
            
            # Update target engagement level
            await self._update_target_engagement_level(target, engagement_data)
            
            # Update content performance
            await self._update_content_performance(delivery.content_id, engagement_data)
            
            # Update channel performance
            await self._update_channel_performance(delivery.channel, engagement_data)
            
            logger.info(f"Tracked engagement for delivery: {delivery_id}")
            
        except Exception as e:
            logger.error(f"Error tracking engagement: {str(e)}")
            raise
    
    async def _update_target_engagement_level(self, target: DeliveryTarget, engagement_data: Dict[str, Any]):
        """Update target's engagement level based on interaction"""
        engagement_score = 0
        
        if engagement_data.get("opened", False):
            engagement_score += 1
        if engagement_data.get("clicked", False):
            engagement_score += 2
        if engagement_data.get("shared", False):
            engagement_score += 3
        if engagement_data.get("responded", False):
            engagement_score += 4
        if engagement_data.get("converted", False):
            engagement_score += 5
        
        # Calculate average engagement from history
        recent_engagements = [
            eng.get("score", 0) for eng in target.engagement_history[-10:]
        ]
        recent_engagements.append(engagement_score)
        
        avg_engagement = sum(recent_engagements) / len(recent_engagements)
        
        # Update engagement level
        if avg_engagement >= 3.5:
            target.current_engagement_level = EngagementLevel.VERY_HIGH
        elif avg_engagement >= 2.5:
            target.current_engagement_level = EngagementLevel.HIGH
        elif avg_engagement >= 1.5:
            target.current_engagement_level = EngagementLevel.MEDIUM
        else:
            target.current_engagement_level = EngagementLevel.LOW
        
        # Update conversion probability
        engagement_multipliers = {
            EngagementLevel.LOW: 0.8,
            EngagementLevel.MEDIUM: 1.2,
            EngagementLevel.HIGH: 1.8,
            EngagementLevel.VERY_HIGH: 2.5
        }
        
        base_probability = 0.20  # Base conversion probability
        multiplier = engagement_multipliers.get(target.current_engagement_level, 1.0)
        target.conversion_probability = min(0.95, base_probability * multiplier)
    
    async def _update_content_performance(self, content_id: str, engagement_data: Dict[str, Any]):
        """Update performance metrics for content"""
        if content_id not in self.content_performance:
            self.content_performance[content_id] = {
                "total_deliveries": 0,
                "total_opens": 0,
                "total_clicks": 0,
                "total_shares": 0,
                "total_conversions": 0
            }
        
        metrics = self.content_performance[content_id]
        metrics["total_deliveries"] += 1
        
        if engagement_data.get("opened", False):
            metrics["total_opens"] += 1
        if engagement_data.get("clicked", False):
            metrics["total_clicks"] += 1
        if engagement_data.get("shared", False):
            metrics["total_shares"] += 1
        if engagement_data.get("converted", False):
            metrics["total_conversions"] += 1
    
    async def _update_channel_performance(self, channel: DeliveryChannel, engagement_data: Dict[str, Any]):
        """Update performance metrics for delivery channel"""
        if channel not in self.channel_performance:
            self.channel_performance[channel] = {
                "total_deliveries": 0,
                "total_opens": 0,
                "total_clicks": 0,
                "total_conversions": 0,
                "avg_engagement_score": 0.0
            }
        
        metrics = self.channel_performance[channel]
        metrics["total_deliveries"] += 1
        
        if engagement_data.get("opened", False):
            metrics["total_opens"] += 1
        if engagement_data.get("clicked", False):
            metrics["total_clicks"] += 1
        if engagement_data.get("converted", False):
            metrics["total_conversions"] += 1
        
        # Calculate engagement score
        engagement_score = sum([
            1 if engagement_data.get("opened", False) else 0,
            2 if engagement_data.get("clicked", False) else 0,
            3 if engagement_data.get("shared", False) else 0,
            5 if engagement_data.get("converted", False) else 0
        ])
        
        # Update average engagement score
        current_avg = metrics["avg_engagement_score"]
        total_deliveries = metrics["total_deliveries"]
        metrics["avg_engagement_score"] = (
            (current_avg * (total_deliveries - 1) + engagement_score) / total_deliveries
        )
    
    async def optimize_delivery_strategy(self, campaign_id: str) -> Dict[str, Any]:
        """Optimize delivery strategy based on performance data"""
        try:
            # Analyze channel performance
            best_channels = sorted(
                self.channel_performance.items(),
                key=lambda x: x[1].get("avg_engagement_score", 0),
                reverse=True
            )[:3]
            
            # Analyze target segments
            segment_performance = {}
            for target in self.delivery_targets.values():
                segment = target.segment
                if segment not in segment_performance:
                    segment_performance[segment] = {
                        "total_targets": 0,
                        "high_engagement_targets": 0,
                        "avg_conversion_probability": 0.0
                    }
                
                segment_performance[segment]["total_targets"] += 1
                segment_performance[segment]["avg_conversion_probability"] += target.conversion_probability
                
                if target.current_engagement_level in [EngagementLevel.HIGH, EngagementLevel.VERY_HIGH]:
                    segment_performance[segment]["high_engagement_targets"] += 1
            
            # Calculate averages
            for segment_data in segment_performance.values():
                if segment_data["total_targets"] > 0:
                    segment_data["avg_conversion_probability"] /= segment_data["total_targets"]
                    segment_data["engagement_rate"] = (
                        segment_data["high_engagement_targets"] / segment_data["total_targets"]
                    )
            
            optimization_recommendations = {
                "best_channels": [
                    {"channel": ch.value, "score": metrics["avg_engagement_score"]}
                    for ch, metrics in best_channels
                ],
                "segment_performance": segment_performance,
                "recommendations": [],
                "generated_at": datetime.now().isoformat()
            }
            
            # Generate specific recommendations
            if best_channels:
                top_channel = best_channels[0][0].value
                optimization_recommendations["recommendations"].append(
                    f"Focus on {top_channel} channel for highest engagement"
                )
            
            high_performing_segments = [
                segment for segment, data in segment_performance.items()
                if data.get("engagement_rate", 0) > 0.3
            ]
            
            if high_performing_segments:
                optimization_recommendations["recommendations"].append(
                    f"Prioritize segments: {', '.join(high_performing_segments)}"
                )
            
            return optimization_recommendations
            
        except Exception as e:
            logger.error(f"Error optimizing delivery strategy: {str(e)}")
            raise
    
    async def generate_delivery_report(self, campaign_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive delivery and engagement report"""
        try:
            total_deliveries = len(self.scheduled_deliveries)
            successful_deliveries = sum(
                1 for d in self.scheduled_deliveries.values() 
                if d.status == "delivered"
            )
            
            total_opens = sum(
                1 for d in self.scheduled_deliveries.values() 
                if d.opened_time is not None
            )
            
            report = {
                "summary": {
                    "total_deliveries": total_deliveries,
                    "successful_deliveries": successful_deliveries,
                    "delivery_rate": successful_deliveries / total_deliveries if total_deliveries > 0 else 0,
                    "total_opens": total_opens,
                    "open_rate": total_opens / successful_deliveries if successful_deliveries > 0 else 0
                },
                "channel_performance": {
                    channel.value: metrics for channel, metrics in self.channel_performance.items()
                },
                "content_performance": self.content_performance,
                "target_engagement": {
                    "total_targets": len(self.delivery_targets),
                    "high_engagement_targets": sum(
                        1 for t in self.delivery_targets.values()
                        if t.current_engagement_level in [EngagementLevel.HIGH, EngagementLevel.VERY_HIGH]
                    ),
                    "avg_conversion_probability": sum(
                        t.conversion_probability for t in self.delivery_targets.values()
                    ) / len(self.delivery_targets) if self.delivery_targets else 0
                },
                "generated_at": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating delivery report: {str(e)}")
            raise
    
    async def execute_automated_follow_up(self):
        """Execute automated follow-up for deliveries requiring attention"""
        try:
            follow_up_deliveries = [
                d for d in self.scheduled_deliveries.values()
                if d.follow_up_required and d.status != "completed"
            ]
            
            for delivery in follow_up_deliveries:
                target = self.delivery_targets[delivery.target_id]
                
                # Determine follow-up strategy based on engagement level
                if target.current_engagement_level == EngagementLevel.LOW:
                    # Try different channel
                    alternative_channels = [
                        ch for ch in target.preferred_channels 
                        if ch != delivery.channel
                    ]
                    
                    if alternative_channels:
                        # Schedule follow-up with different channel
                        await self.schedule_content_delivery(
                            delivery.content_id,
                            [target.id],
                            alternative_channels[0],
                            datetime.now() + timedelta(days=3)
                        )
                
                elif target.current_engagement_level in [EngagementLevel.MEDIUM, EngagementLevel.HIGH]:
                    # Schedule personalized follow-up
                    await self.schedule_content_delivery(
                        delivery.content_id,
                        [target.id],
                        DeliveryChannel.DIRECT_OUTREACH,
                        datetime.now() + timedelta(days=1)
                    )
                
                delivery.follow_up_required = False
            
            logger.info(f"Executed automated follow-up for {len(follow_up_deliveries)} deliveries")
            
        except Exception as e:
            logger.error(f"Error executing automated follow-up: {str(e)}")
            raise