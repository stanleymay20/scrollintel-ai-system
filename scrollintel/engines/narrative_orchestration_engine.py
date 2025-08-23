"""
Narrative Orchestration Engine for Global Influence Network System

This engine provides cross-platform messaging coordination, narrative consistency,
and timing optimization for influence campaigns.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json

from ..models.influence_network_models import (
    InfluenceCampaign, NarrativeTheme, ContentPiece, ThoughtLeaderProfile,
    CampaignStatus
)


class MessageType(Enum):
    """Types of messages in narrative orchestration"""
    ANNOUNCEMENT = "announcement"
    THOUGHT_LEADERSHIP = "thought_leadership"
    RESPONSE = "response"
    AMPLIFICATION = "amplification"
    CORRECTION = "correction"


class ChannelType(Enum):
    """Types of communication channels"""
    SOCIAL_MEDIA = "social_media"
    TRADITIONAL_MEDIA = "traditional_media"
    INDUSTRY_EVENTS = "industry_events"
    DIRECT_COMMUNICATION = "direct_communication"
    CONTENT_PLATFORMS = "content_platforms"


@dataclass
class Message:
    """Represents a message in the narrative orchestration"""
    message_id: str
    campaign_id: str
    narrative_theme_id: str
    message_type: MessageType
    content: str
    target_audience: List[str]
    channels: List[ChannelType]
    scheduled_time: datetime
    priority: str  # 'high', 'medium', 'low'
    approval_status: str  # 'draft', 'pending', 'approved', 'published'
    performance_metrics: Dict[str, float]
    created_at: datetime
    created_by: str


@dataclass
class ChannelCoordination:
    """Represents coordination across multiple channels"""
    coordination_id: str
    campaign_id: str
    channels: List[ChannelType]
    coordination_strategy: str
    timing_sequence: List[Dict[str, Any]]
    consistency_rules: List[str]
    cross_channel_metrics: Dict[str, float]
    created_at: datetime


@dataclass
class NarrativeConsistency:
    """Represents narrative consistency tracking"""
    consistency_id: str
    campaign_id: str
    core_messages: List[str]
    consistency_score: float
    deviations: List[Dict[str, Any]]
    correction_actions: List[str]
    last_updated: datetime


class NarrativeOrchestrationEngine:
    """Engine for orchestrating narrative campaigns across multiple channels"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_campaigns = {}
        self.message_queue = {}
        self.channel_coordinators = {}
        self.narrative_consistency_tracker = {}
        self.thought_leaders = {}
        
        # Configuration
        self.config = {
            'max_messages_per_hour': 10,
            'consistency_threshold': 0.8,
            'approval_timeout_hours': 24,
            'cross_channel_delay_minutes': 15
        }
    
    async def create_narrative_campaign(
        self,
        campaign_data: Dict[str, Any],
        narrative_themes: List[NarrativeTheme],
        target_channels: List[ChannelType]
    ) -> InfluenceCampaign:
        """Create a new narrative influence campaign"""
        try:
            campaign = InfluenceCampaign(
                campaign_id=f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=campaign_data['name'],
                description=campaign_data['description'],
                objectives=campaign_data['objectives'],
                target_network=campaign_data.get('target_network', ''),
                target_nodes=campaign_data.get('target_nodes', []),
                narrative_themes=narrative_themes,
                channels=[channel.value for channel in target_channels],
                timeline=campaign_data.get('timeline', {}),
                budget=campaign_data.get('budget', 0.0),
                status=CampaignStatus.PLANNING,
                performance_metrics={},
                content_assets=[],
                team_members=campaign_data.get('team_members', []),
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            # Initialize campaign tracking
            self.active_campaigns[campaign.campaign_id] = campaign
            self.message_queue[campaign.campaign_id] = []
            
            # Create channel coordination
            coordination = await self._create_channel_coordination(campaign, target_channels)
            self.channel_coordinators[campaign.campaign_id] = coordination
            
            # Initialize narrative consistency tracking
            consistency = await self._initialize_narrative_consistency(campaign)
            self.narrative_consistency_tracker[campaign.campaign_id] = consistency
            
            self.logger.info(f"Created narrative campaign: {campaign.name}")
            return campaign
            
        except Exception as e:
            self.logger.error(f"Error creating narrative campaign: {str(e)}")
            raise
    
    async def orchestrate_cross_platform_messaging(
        self,
        campaign_id: str,
        messages: List[Message],
        coordination_strategy: str = "sequential"
    ) -> Dict[str, Any]:
        """Orchestrate messaging across multiple platforms"""
        try:
            if campaign_id not in self.active_campaigns:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            campaign = self.active_campaigns[campaign_id]
            coordination = self.channel_coordinators[campaign_id]
            
            orchestration_result = {
                'campaign_id': campaign_id,
                'total_messages': len(messages),
                'scheduled_messages': [],
                'coordination_timeline': [],
                'estimated_reach': 0,
                'orchestration_strategy': coordination_strategy
            }
            
            # Sort messages by priority and timing
            sorted_messages = sorted(messages, key=lambda m: (m.priority, m.scheduled_time))
            
            if coordination_strategy == "sequential":
                orchestration_result = await self._orchestrate_sequential(
                    campaign, sorted_messages, coordination
                )
            elif coordination_strategy == "simultaneous":
                orchestration_result = await self._orchestrate_simultaneous(
                    campaign, sorted_messages, coordination
                )
            elif coordination_strategy == "cascading":
                orchestration_result = await self._orchestrate_cascading(
                    campaign, sorted_messages, coordination
                )
            
            # Update message queue
            self.message_queue[campaign_id].extend(messages)
            
            self.logger.info(f"Orchestrated {len(messages)} messages for campaign {campaign_id}")
            return orchestration_result
            
        except Exception as e:
            self.logger.error(f"Error orchestrating cross-platform messaging: {str(e)}")
            raise
    
    async def ensure_narrative_consistency(
        self,
        campaign_id: str,
        new_content: List[ContentPiece]
    ) -> NarrativeConsistency:
        """Ensure narrative consistency across all campaign content"""
        try:
            if campaign_id not in self.active_campaigns:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            campaign = self.active_campaigns[campaign_id]
            consistency_tracker = self.narrative_consistency_tracker[campaign_id]
            
            # Extract core messages from narrative themes
            core_messages = []
            for theme in campaign.narrative_themes:
                core_messages.extend(theme.key_messages)
            
            # Analyze consistency of new content
            consistency_analysis = await self._analyze_content_consistency(
                core_messages, new_content
            )
            
            # Update consistency tracker
            consistency_tracker.core_messages = core_messages
            consistency_tracker.consistency_score = consistency_analysis['score']
            consistency_tracker.deviations = consistency_analysis['deviations']
            consistency_tracker.correction_actions = consistency_analysis['corrections']
            consistency_tracker.last_updated = datetime.now()
            
            # Generate recommendations if consistency is low
            if consistency_tracker.consistency_score < self.config['consistency_threshold']:
                await self._generate_consistency_corrections(campaign_id, consistency_tracker)
            
            self.logger.info(f"Analyzed narrative consistency for campaign {campaign_id}: {consistency_tracker.consistency_score:.2f}")
            return consistency_tracker
            
        except Exception as e:
            self.logger.error(f"Error ensuring narrative consistency: {str(e)}")
            raise
    
    async def optimize_campaign_timing(
        self,
        campaign_id: str,
        target_audience_data: Dict[str, Any],
        channel_analytics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize timing for campaign messages across channels"""
        try:
            if campaign_id not in self.active_campaigns:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            campaign = self.active_campaigns[campaign_id]
            messages = self.message_queue[campaign_id]
            
            timing_optimization = {
                'campaign_id': campaign_id,
                'optimized_schedule': [],
                'channel_timing': {},
                'audience_engagement_windows': {},
                'cross_channel_coordination': [],
                'expected_performance_lift': 0.0
            }
            
            # Analyze optimal timing for each channel
            for channel in campaign.channels:
                channel_timing = await self._analyze_channel_timing(
                    channel, target_audience_data, channel_analytics
                )
                timing_optimization['channel_timing'][channel] = channel_timing
            
            # Optimize message scheduling
            optimized_messages = await self._optimize_message_schedule(
                messages, timing_optimization['channel_timing']
            )
            timing_optimization['optimized_schedule'] = optimized_messages
            
            # Calculate cross-channel coordination
            coordination_plan = await self._plan_cross_channel_coordination(
                optimized_messages, campaign.channels
            )
            timing_optimization['cross_channel_coordination'] = coordination_plan
            
            # Estimate performance improvement
            timing_optimization['expected_performance_lift'] = await self._estimate_timing_impact(
                optimized_messages, channel_analytics
            )
            
            self.logger.info(f"Optimized timing for campaign {campaign_id}")
            return timing_optimization
            
        except Exception as e:
            self.logger.error(f"Error optimizing campaign timing: {str(e)}")
            raise    
  
  async def activate_thought_leaders(
        self,
        campaign_id: str,
        thought_leaders: List[ThoughtLeaderProfile],
        activation_strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Activate thought leaders for campaign amplification"""
        try:
            if campaign_id not in self.active_campaigns:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            campaign = self.active_campaigns[campaign_id]
            
            activation_result = {
                'campaign_id': campaign_id,
                'activated_leaders': [],
                'activation_timeline': [],
                'expected_amplification': 0.0,
                'engagement_strategy': activation_strategy
            }
            
            for leader in thought_leaders:
                # Analyze leader fit for campaign
                fit_score = await self._analyze_leader_campaign_fit(leader, campaign)
                
                if fit_score > 0.6:  # Minimum fit threshold
                    # Create activation plan
                    activation_plan = await self._create_leader_activation_plan(
                        leader, campaign, activation_strategy
                    )
                    
                    activation_result['activated_leaders'].append({
                        'leader_id': leader.leader_id,
                        'name': leader.name,
                        'fit_score': fit_score,
                        'activation_plan': activation_plan,
                        'expected_reach': sum(leader.follower_count.values()),
                        'engagement_rate': leader.engagement_rate
                    })
                    
                    # Store thought leader
                    self.thought_leaders[leader.leader_id] = leader
            
            # Calculate total expected amplification
            total_reach = sum(
                leader['expected_reach'] * leader['engagement_rate']
                for leader in activation_result['activated_leaders']
            )
            activation_result['expected_amplification'] = total_reach
            
            self.logger.info(f"Activated {len(activation_result['activated_leaders'])} thought leaders for campaign {campaign_id}")
            return activation_result
            
        except Exception as e:
            self.logger.error(f"Error activating thought leaders: {str(e)}")
            raise
    
    async def track_campaign_performance(
        self,
        campaign_id: str,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Track and analyze campaign performance across channels"""
        try:
            if campaign_id not in self.active_campaigns:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            campaign = self.active_campaigns[campaign_id]
            
            # Update campaign performance metrics
            campaign.performance_metrics.update(performance_data)
            campaign.last_updated = datetime.now()
            
            # Analyze performance by channel
            channel_performance = {}
            for channel in campaign.channels:
                channel_data = performance_data.get(channel, {})
                channel_performance[channel] = await self._analyze_channel_performance(
                    channel, channel_data
                )
            
            # Analyze narrative theme performance
            theme_performance = {}
            for theme in campaign.narrative_themes:
                theme_data = performance_data.get(f"theme_{theme.theme_id}", {})
                theme_performance[theme.theme_id] = await self._analyze_theme_performance(
                    theme, theme_data
                )
            
            # Calculate overall campaign effectiveness
            overall_effectiveness = await self._calculate_campaign_effectiveness(
                campaign, channel_performance, theme_performance
            )
            
            performance_summary = {
                'campaign_id': campaign_id,
                'overall_effectiveness': overall_effectiveness,
                'channel_performance': channel_performance,
                'theme_performance': theme_performance,
                'key_insights': await self._generate_performance_insights(
                    campaign, channel_performance, theme_performance
                ),
                'optimization_recommendations': await self._generate_optimization_recommendations(
                    campaign, channel_performance
                ),
                'last_updated': datetime.now().isoformat()
            }
            
            self.logger.info(f"Updated performance tracking for campaign {campaign_id}")
            return performance_summary
            
        except Exception as e:
            self.logger.error(f"Error tracking campaign performance: {str(e)}")
            raise
    
    # Helper methods
    
    async def _create_channel_coordination(
        self,
        campaign: InfluenceCampaign,
        channels: List[ChannelType]
    ) -> ChannelCoordination:
        """Create channel coordination strategy"""
        coordination = ChannelCoordination(
            coordination_id=f"coord_{campaign.campaign_id}",
            campaign_id=campaign.campaign_id,
            channels=channels,
            coordination_strategy="sequential",
            timing_sequence=[],
            consistency_rules=[
                "Maintain core message consistency",
                "Adapt tone for channel audience",
                "Coordinate timing across channels"
            ],
            cross_channel_metrics={},
            created_at=datetime.now()
        )
        return coordination
    
    async def _initialize_narrative_consistency(
        self,
        campaign: InfluenceCampaign
    ) -> NarrativeConsistency:
        """Initialize narrative consistency tracking"""
        consistency = NarrativeConsistency(
            consistency_id=f"consistency_{campaign.campaign_id}",
            campaign_id=campaign.campaign_id,
            core_messages=[],
            consistency_score=1.0,
            deviations=[],
            correction_actions=[],
            last_updated=datetime.now()
        )
        return consistency
    
    async def _orchestrate_sequential(
        self,
        campaign: InfluenceCampaign,
        messages: List[Message],
        coordination: ChannelCoordination
    ) -> Dict[str, Any]:
        """Orchestrate messages sequentially across channels"""
        result = {
            'strategy': 'sequential',
            'timeline': [],
            'estimated_reach': 0
        }
        
        current_time = datetime.now()
        delay_minutes = self.config['cross_channel_delay_minutes']
        
        for i, message in enumerate(messages):
            scheduled_time = current_time + timedelta(minutes=i * delay_minutes)
            result['timeline'].append({
                'message_id': message.message_id,
                'scheduled_time': scheduled_time.isoformat(),
                'channels': [ch.value for ch in message.channels],
                'priority': message.priority
            })
        
        return result
    
    async def _orchestrate_simultaneous(
        self,
        campaign: InfluenceCampaign,
        messages: List[Message],
        coordination: ChannelCoordination
    ) -> Dict[str, Any]:
        """Orchestrate messages simultaneously across channels"""
        result = {
            'strategy': 'simultaneous',
            'timeline': [],
            'estimated_reach': 0
        }
        
        base_time = datetime.now()
        
        for message in messages:
            result['timeline'].append({
                'message_id': message.message_id,
                'scheduled_time': base_time.isoformat(),
                'channels': [ch.value for ch in message.channels],
                'priority': message.priority
            })
        
        return result
    
    async def _orchestrate_cascading(
        self,
        campaign: InfluenceCampaign,
        messages: List[Message],
        coordination: ChannelCoordination
    ) -> Dict[str, Any]:
        """Orchestrate messages in cascading pattern"""
        result = {
            'strategy': 'cascading',
            'timeline': [],
            'estimated_reach': 0
        }
        
        # Group messages by priority
        high_priority = [m for m in messages if m.priority == 'high']
        medium_priority = [m for m in messages if m.priority == 'medium']
        low_priority = [m for m in messages if m.priority == 'low']
        
        current_time = datetime.now()
        
        # Schedule high priority first
        for i, message in enumerate(high_priority):
            scheduled_time = current_time + timedelta(minutes=i * 5)
            result['timeline'].append({
                'message_id': message.message_id,
                'scheduled_time': scheduled_time.isoformat(),
                'channels': [ch.value for ch in message.channels],
                'priority': message.priority
            })
        
        # Then medium priority
        base_time = current_time + timedelta(minutes=len(high_priority) * 5 + 30)
        for i, message in enumerate(medium_priority):
            scheduled_time = base_time + timedelta(minutes=i * 10)
            result['timeline'].append({
                'message_id': message.message_id,
                'scheduled_time': scheduled_time.isoformat(),
                'channels': [ch.value for ch in message.channels],
                'priority': message.priority
            })
        
        return result
    
    async def _analyze_content_consistency(
        self,
        core_messages: List[str],
        content_pieces: List[ContentPiece]
    ) -> Dict[str, Any]:
        """Analyze consistency of content with core messages"""
        analysis = {
            'score': 0.0,
            'deviations': [],
            'corrections': []
        }
        
        if not core_messages or not content_pieces:
            analysis['score'] = 1.0
            return analysis
        
        # Simplified consistency analysis
        consistent_pieces = 0
        total_pieces = len(content_pieces)
        
        for piece in content_pieces:
            # Check if content aligns with core messages
            piece_consistency = 0.0
            for core_message in core_messages:
                # Simple keyword matching (would use NLP in practice)
                if any(word in piece.title.lower() or word in piece.key_messages 
                       for word in core_message.lower().split()):
                    piece_consistency += 1.0
            
            piece_consistency = piece_consistency / len(core_messages)
            
            if piece_consistency > 0.7:
                consistent_pieces += 1
            else:
                analysis['deviations'].append({
                    'content_id': piece.content_id,
                    'consistency_score': piece_consistency,
                    'issue': 'Low alignment with core messages'
                })
                analysis['corrections'].append(f"Review content {piece.content_id} for message alignment")
        
        analysis['score'] = consistent_pieces / total_pieces if total_pieces > 0 else 1.0
        return analysis
    
    async def _analyze_channel_timing(
        self,
        channel: str,
        audience_data: Dict[str, Any],
        analytics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze optimal timing for a specific channel"""
        # Default timing patterns (would use real analytics in practice)
        timing_patterns = {
            'social_media': {
                'peak_hours': [9, 12, 17, 20],
                'peak_days': ['tuesday', 'wednesday', 'thursday'],
                'engagement_multiplier': 1.5
            },
            'traditional_media': {
                'peak_hours': [6, 8, 18],
                'peak_days': ['monday', 'tuesday', 'wednesday'],
                'engagement_multiplier': 2.0
            },
            'industry_events': {
                'peak_hours': [10, 14, 16],
                'peak_days': ['tuesday', 'wednesday', 'thursday'],
                'engagement_multiplier': 3.0
            }
        }
        
        return timing_patterns.get(channel, {
            'peak_hours': [12, 15, 18],
            'peak_days': ['monday', 'wednesday', 'friday'],
            'engagement_multiplier': 1.0
        })
    
    async def _optimize_message_schedule(
        self,
        messages: List[Message],
        channel_timing: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Optimize message scheduling based on channel timing"""
        optimized_schedule = []
        
        for message in messages:
            best_time = message.scheduled_time
            best_score = 1.0
            
            # Find optimal time for each channel
            for channel in message.channels:
                if channel.value in channel_timing:
                    timing_data = channel_timing[channel.value]
                    # Simple optimization (would be more sophisticated in practice)
                    if message.scheduled_time.hour in timing_data.get('peak_hours', []):
                        best_score *= timing_data.get('engagement_multiplier', 1.0)
            
            optimized_schedule.append({
                'message_id': message.message_id,
                'original_time': message.scheduled_time.isoformat(),
                'optimized_time': best_time.isoformat(),
                'optimization_score': best_score,
                'channels': [ch.value for ch in message.channels]
            })
        
        return optimized_schedule
    
    async def _analyze_leader_campaign_fit(
        self,
        leader: ThoughtLeaderProfile,
        campaign: InfluenceCampaign
    ) -> float:
        """Analyze how well a thought leader fits the campaign"""
        fit_score = 0.0
        
        # Check expertise alignment
        campaign_topics = set()
        for theme in campaign.narrative_themes:
            campaign_topics.update(theme.key_messages)
        
        leader_topics = set(leader.expertise_areas + leader.content_themes)
        
        topic_overlap = len(campaign_topics & leader_topics)
        total_topics = len(campaign_topics | leader_topics)
        
        if total_topics > 0:
            fit_score += (topic_overlap / total_topics) * 0.6
        
        # Check influence level
        total_followers = sum(leader.follower_count.values())
        if total_followers > 10000:
            fit_score += 0.3
        elif total_followers > 1000:
            fit_score += 0.2
        
        # Check engagement rate
        if leader.engagement_rate > 0.05:
            fit_score += 0.1
        
        return min(fit_score, 1.0)
    
    async def _create_leader_activation_plan(
        self,
        leader: ThoughtLeaderProfile,
        campaign: InfluenceCampaign,
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create activation plan for a thought leader"""
        plan = {
            'leader_id': leader.leader_id,
            'activation_type': strategy.get('type', 'content_collaboration'),
            'timeline': strategy.get('timeline', '2-4 weeks'),
            'deliverables': [],
            'compensation': strategy.get('compensation', 'exposure'),
            'success_metrics': ['engagement_rate', 'reach', 'message_amplification']
        }
        
        # Define deliverables based on leader's strengths
        if 'speaking' in leader.speaking_topics:
            plan['deliverables'].append('Conference presentation')
        
        if leader.engagement_rate > 0.03:
            plan['deliverables'].append('Social media content series')
        
        plan['deliverables'].append('Thought leadership article')
        
        return plan
    
    def get_campaign_status(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a campaign"""
        if campaign_id not in self.active_campaigns:
            return None
        
        campaign = self.active_campaigns[campaign_id]
        
        return {
            'campaign_id': campaign_id,
            'name': campaign.name,
            'status': campaign.status.value,
            'created_at': campaign.created_at.isoformat(),
            'last_updated': campaign.last_updated.isoformat(),
            'total_messages': len(self.message_queue.get(campaign_id, [])),
            'channels': campaign.channels,
            'performance_metrics': campaign.performance_metrics
        }