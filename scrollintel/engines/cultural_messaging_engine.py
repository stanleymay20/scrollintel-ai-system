"""
Cultural Messaging Engine

Implements consistent cultural vision and values communication system with
message customization for different audiences and effectiveness tracking.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import uuid
import re
from dataclasses import asdict

from ..models.cultural_messaging_models import (
    CulturalMessage, MessageTemplate, AudienceProfile, MessageCustomization,
    MessageDelivery, MessageEngagement, MessageEffectiveness,
    CulturalMessagingCampaign, MessagingStrategy,
    MessageType, AudienceType, MessageChannel
)

logger = logging.getLogger(__name__)


class CulturalMessagingEngine:
    """
    Engine for creating, customizing, and tracking cultural messages
    """
    
    def __init__(self):
        self.message_templates = {}
        self.audience_profiles = {}
        self.messaging_strategies = {}
        self.active_campaigns = {}
        self.message_history = {}
        self.effectiveness_data = {}
        
    def create_messaging_strategy(
        self,
        organization_id: str,
        cultural_vision: str,
        core_values: List[str],
        audience_data: List[Dict[str, Any]]
    ) -> MessagingStrategy:
        """Create comprehensive messaging strategy"""
        try:
            # Create audience profiles
            audience_profiles = []
            for audience_info in audience_data:
                profile = AudienceProfile(
                    id=str(uuid.uuid4()),
                    name=audience_info['name'],
                    audience_type=AudienceType(audience_info['type']),
                    characteristics=audience_info.get('characteristics', {}),
                    communication_preferences=audience_info.get('preferences', {}),
                    cultural_context=audience_info.get('cultural_context', {}),
                    size=audience_info.get('size', 0)
                )
                audience_profiles.append(profile)
                self.audience_profiles[profile.id] = profile
            
            # Extract key themes from vision and values
            key_themes = self._extract_cultural_themes(cultural_vision, core_values)
            
            # Create message templates for different types
            message_templates = self._create_default_templates(core_values, key_themes)
            
            # Create messaging strategy
            strategy = MessagingStrategy(
                id=str(uuid.uuid4()),
                organization_id=organization_id,
                cultural_vision=cultural_vision,
                core_values=core_values,
                key_themes=key_themes,
                audience_segments=audience_profiles,
                message_templates=message_templates,
                communication_calendar={},
                effectiveness_targets={
                    'engagement_rate': 0.75,
                    'cultural_alignment': 0.80,
                    'behavior_change': 0.60
                }
            )
            
            self.messaging_strategies[strategy.id] = strategy
            
            logger.info(f"Created messaging strategy for organization {organization_id}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating messaging strategy: {str(e)}")
            raise
    
    def create_cultural_message(
        self,
        title: str,
        content: str,
        message_type: MessageType,
        cultural_themes: List[str],
        key_values: List[str],
        template_id: Optional[str] = None
    ) -> CulturalMessage:
        """Create new cultural message"""
        try:
            message = CulturalMessage(
                id=str(uuid.uuid4()),
                title=title,
                content=content,
                message_type=message_type,
                cultural_themes=cultural_themes,
                key_values=key_values,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Validate message alignment with cultural themes
            alignment_score = self._validate_cultural_alignment(message)
            message.metadata['alignment_score'] = alignment_score
            
            self.message_history[message.id] = message
            
            logger.info(f"Created cultural message: {title}")
            return message
            
        except Exception as e:
            logger.error(f"Error creating cultural message: {str(e)}")
            raise
    
    def customize_message_for_audience(
        self,
        message_id: str,
        audience_id: str,
        channel: MessageChannel,
        delivery_timing: Optional[datetime] = None
    ) -> MessageCustomization:
        """Customize message for specific audience"""
        try:
            message = self.message_history.get(message_id)
            audience = self.audience_profiles.get(audience_id)
            
            if not message or not audience:
                raise ValueError("Message or audience not found")
            
            # Customize content based on audience characteristics
            customized_content = self._customize_content(
                message.content,
                audience,
                channel
            )
            
            # Add personalization data
            personalization_data = self._generate_personalization_data(
                message,
                audience,
                channel
            )
            
            customization = MessageCustomization(
                id=str(uuid.uuid4()),
                base_message_id=message_id,
                audience_id=audience_id,
                customized_content=customized_content,
                personalization_data=personalization_data,
                channel=channel,
                delivery_timing=delivery_timing or datetime.now()
            )
            
            logger.info(f"Customized message {message_id} for audience {audience_id}")
            return customization
            
        except Exception as e:
            logger.error(f"Error customizing message: {str(e)}")
            raise
    
    def track_message_effectiveness(
        self,
        message_id: str,
        audience_id: str,
        engagement_data: Dict[str, Any]
    ) -> MessageEffectiveness:
        """Track and analyze message effectiveness"""
        try:
            message = self.message_history.get(message_id)
            audience = self.audience_profiles.get(audience_id)
            
            if not message or not audience:
                raise ValueError("Message or audience not found")
            
            # Calculate engagement metrics
            engagement = MessageEngagement(
                id=str(uuid.uuid4()),
                message_id=message_id,
                audience_id=audience_id,
                channel=MessageChannel(engagement_data.get('channel', 'email')),
                views=engagement_data.get('views', 0),
                clicks=engagement_data.get('clicks', 0),
                shares=engagement_data.get('shares', 0),
                responses=engagement_data.get('responses', 0),
                sentiment_score=engagement_data.get('sentiment_score', 0.0)
            )
            
            # Calculate engagement rate
            if engagement.views > 0:
                engagement.engagement_rate = (
                    engagement.clicks + engagement.shares + engagement.responses
                ) / engagement.views
            
            # Analyze effectiveness
            effectiveness_score = self._calculate_effectiveness_score(
                message, audience, engagement
            )
            
            cultural_alignment_score = self._assess_cultural_alignment(
                message, engagement_data.get('feedback', {})
            )
            
            behavior_indicators = self._analyze_behavior_change_indicators(
                message, audience, engagement_data
            )
            
            recommendations = self._generate_improvement_recommendations(
                message, audience, engagement, effectiveness_score
            )
            
            effectiveness = MessageEffectiveness(
                id=str(uuid.uuid4()),
                message_id=message_id,
                audience_id=audience_id,
                effectiveness_score=effectiveness_score,
                cultural_alignment_score=cultural_alignment_score,
                behavior_change_indicators=behavior_indicators,
                feedback_summary=engagement_data.get('feedback', {}),
                recommendations=recommendations
            )
            
            # Store effectiveness data
            key = f"{message_id}_{audience_id}"
            self.effectiveness_data[key] = effectiveness
            
            logger.info(f"Tracked effectiveness for message {message_id}")
            return effectiveness
            
        except Exception as e:
            logger.error(f"Error tracking message effectiveness: {str(e)}")
            raise
    
    def optimize_messaging_strategy(
        self,
        strategy_id: str,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize messaging strategy based on performance data"""
        try:
            strategy = self.messaging_strategies.get(strategy_id)
            if not strategy:
                raise ValueError("Strategy not found")
            
            # Analyze performance across all messages
            performance_analysis = self._analyze_strategy_performance(
                strategy, performance_data
            )
            
            # Generate optimization recommendations
            optimizations = {
                'content_improvements': self._suggest_content_improvements(
                    performance_analysis
                ),
                'audience_targeting': self._optimize_audience_targeting(
                    performance_analysis
                ),
                'channel_optimization': self._optimize_channel_selection(
                    performance_analysis
                ),
                'timing_recommendations': self._optimize_timing(
                    performance_analysis
                ),
                'template_updates': self._suggest_template_updates(
                    performance_analysis
                )
            }
            
            # Update strategy with optimizations
            strategy.updated_at = datetime.now()
            strategy.effectiveness_targets.update(
                performance_analysis.get('updated_targets', {})
            )
            
            logger.info(f"Optimized messaging strategy {strategy_id}")
            return optimizations
            
        except Exception as e:
            logger.error(f"Error optimizing messaging strategy: {str(e)}")
            raise
    
    def create_messaging_campaign(
        self,
        name: str,
        description: str,
        cultural_objectives: List[str],
        target_audiences: List[str],
        messages: List[str],
        duration_days: int
    ) -> CulturalMessagingCampaign:
        """Create cultural messaging campaign"""
        try:
            start_date = datetime.now()
            end_date = start_date + timedelta(days=duration_days)
            
            campaign = CulturalMessagingCampaign(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                cultural_objectives=cultural_objectives,
                target_audiences=target_audiences,
                messages=messages,
                start_date=start_date,
                end_date=end_date,
                status="planned"
            )
            
            self.active_campaigns[campaign.id] = campaign
            
            logger.info(f"Created messaging campaign: {name}")
            return campaign
            
        except Exception as e:
            logger.error(f"Error creating messaging campaign: {str(e)}")
            raise
    
    def _extract_cultural_themes(
        self,
        vision: str,
        values: List[str]
    ) -> List[str]:
        """Extract key cultural themes from vision and values"""
        themes = []
        
        # Common cultural themes
        theme_keywords = {
            'innovation': ['innovation', 'creative', 'breakthrough', 'pioneering'],
            'collaboration': ['collaboration', 'teamwork', 'together', 'partnership'],
            'excellence': ['excellence', 'quality', 'best', 'superior'],
            'integrity': ['integrity', 'honest', 'ethical', 'trust'],
            'growth': ['growth', 'development', 'learning', 'improvement'],
            'customer_focus': ['customer', 'client', 'service', 'satisfaction'],
            'diversity': ['diversity', 'inclusion', 'diverse', 'inclusive'],
            'agility': ['agile', 'flexible', 'adaptive', 'responsive']
        }
        
        text_to_analyze = f"{vision} {' '.join(values)}".lower()
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_to_analyze for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def _create_default_templates(
        self,
        core_values: List[str],
        themes: List[str]
    ) -> List[MessageTemplate]:
        """Create default message templates"""
        templates = []
        
        # Vision communication template
        vision_template = MessageTemplate(
            id=str(uuid.uuid4()),
            name="Vision Communication",
            template_content="""
            Dear {audience_name},
            
            I want to share our cultural vision with you: {vision_statement}
            
            This vision reflects our core values of {values_list} and guides us toward {cultural_objectives}.
            
            Your role in bringing this vision to life is {role_specific_message}.
            
            Together, we will {collective_action}.
            
            Best regards,
            {sender_name}
            """,
            message_type=MessageType.VISION_COMMUNICATION,
            required_variables=['audience_name', 'vision_statement', 'values_list', 'cultural_objectives', 'role_specific_message', 'collective_action', 'sender_name']
        )
        templates.append(vision_template)
        
        # Values reinforcement template
        values_template = MessageTemplate(
            id=str(uuid.uuid4()),
            name="Values Reinforcement",
            template_content="""
            Team,
            
            Today I want to highlight how our value of {featured_value} is making a difference.
            
            {success_example}
            
            This demonstrates {value_impact} and shows how living our values creates {positive_outcome}.
            
            I encourage you to {call_to_action}.
            
            {closing_message}
            """,
            message_type=MessageType.VALUES_REINFORCEMENT,
            required_variables=['featured_value', 'success_example', 'value_impact', 'positive_outcome', 'call_to_action', 'closing_message']
        )
        templates.append(values_template)
        
        return templates
    
    def _customize_content(
        self,
        base_content: str,
        audience: AudienceProfile,
        channel: MessageChannel
    ) -> str:
        """Customize message content for specific audience and channel"""
        customized = base_content
        
        # Adjust tone based on audience type
        if audience.audience_type == AudienceType.LEADERSHIP_TEAM:
            customized = self._adjust_tone_for_leadership(customized)
        elif audience.audience_type == AudienceType.NEW_HIRES:
            customized = self._adjust_tone_for_new_hires(customized)
        
        # Adjust format based on channel
        if channel == MessageChannel.SLACK:
            customized = self._format_for_slack(customized)
        elif channel == MessageChannel.EMAIL:
            customized = self._format_for_email(customized)
        
        # Add audience-specific context
        if audience.cultural_context:
            customized = self._add_cultural_context(customized, audience.cultural_context)
        
        return customized
    
    def _generate_personalization_data(
        self,
        message: CulturalMessage,
        audience: AudienceProfile,
        channel: MessageChannel
    ) -> Dict[str, Any]:
        """Generate personalization data for message"""
        return {
            'audience_preferences': audience.communication_preferences,
            'cultural_context': audience.cultural_context,
            'channel_optimization': {
                'optimal_length': self._get_optimal_length(channel),
                'preferred_format': self._get_preferred_format(channel),
                'timing_preference': audience.communication_preferences.get('timing', 'morning')
            },
            'engagement_history': audience.engagement_history,
            'personalization_score': self._calculate_personalization_score(message, audience)
        }
    
    def _calculate_effectiveness_score(
        self,
        message: CulturalMessage,
        audience: AudienceProfile,
        engagement: MessageEngagement
    ) -> float:
        """Calculate overall message effectiveness score"""
        # Weight different factors
        engagement_weight = 0.4
        alignment_weight = 0.3
        reach_weight = 0.3
        
        # Normalize engagement rate (0-1)
        engagement_score = min(engagement.engagement_rate * 2, 1.0)
        
        # Cultural alignment score from message metadata
        alignment_score = message.metadata.get('alignment_score', 0.5)
        
        # Reach score based on audience size and views
        reach_score = min(engagement.views / max(audience.size, 1), 1.0)
        
        effectiveness = (
            engagement_score * engagement_weight +
            alignment_score * alignment_weight +
            reach_score * reach_weight
        )
        
        return round(effectiveness, 3)
    
    def _validate_cultural_alignment(self, message: CulturalMessage) -> float:
        """Validate message alignment with cultural themes and values"""
        content_lower = message.content.lower()
        
        # Check for theme presence
        theme_score = 0
        for theme in message.cultural_themes:
            if theme.lower() in content_lower:
                theme_score += 1
        theme_score = min(theme_score / max(len(message.cultural_themes), 1), 1.0)
        
        # Check for values presence
        values_score = 0
        for value in message.key_values:
            if value.lower() in content_lower:
                values_score += 1
        values_score = min(values_score / max(len(message.key_values), 1), 1.0)
        
        return (theme_score + values_score) / 2
    
    def _assess_cultural_alignment(
        self,
        message: CulturalMessage,
        feedback: Dict[str, Any]
    ) -> float:
        """Assess cultural alignment based on feedback"""
        alignment_indicators = feedback.get('alignment_indicators', {})
        
        # Default alignment score from message validation
        base_score = message.metadata.get('alignment_score', 0.5)
        
        # Adjust based on feedback
        feedback_score = alignment_indicators.get('cultural_resonance', 0.5)
        clarity_score = alignment_indicators.get('message_clarity', 0.5)
        relevance_score = alignment_indicators.get('relevance', 0.5)
        
        # Weighted average
        final_score = (
            base_score * 0.4 +
            feedback_score * 0.3 +
            clarity_score * 0.15 +
            relevance_score * 0.15
        )
        
        return round(final_score, 3)
    
    def _analyze_behavior_change_indicators(
        self,
        message: CulturalMessage,
        audience: AudienceProfile,
        engagement_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze indicators of behavior change"""
        indicators = {}
        
        # Engagement depth
        if engagement_data.get('views', 0) > 0:
            indicators['engagement_depth'] = (
                engagement_data.get('time_spent', 0) / 
                max(engagement_data.get('views', 1), 1)
            )
        
        # Action taking
        indicators['action_rate'] = (
            engagement_data.get('actions_taken', 0) /
            max(engagement_data.get('views', 1), 1)
        )
        
        # Discussion generation
        indicators['discussion_rate'] = (
            engagement_data.get('discussions_started', 0) /
            max(engagement_data.get('views', 1), 1)
        )
        
        # Follow-up engagement
        indicators['follow_up_rate'] = (
            engagement_data.get('follow_up_actions', 0) /
            max(engagement_data.get('views', 1), 1)
        )
        
        return indicators
    
    def _generate_improvement_recommendations(
        self,
        message: CulturalMessage,
        audience: AudienceProfile,
        engagement: MessageEngagement,
        effectiveness_score: float
    ) -> List[str]:
        """Generate recommendations for message improvement"""
        recommendations = []
        
        if effectiveness_score < 0.6:
            recommendations.append("Consider revising message content for better cultural alignment")
        
        if engagement.engagement_rate < 0.3:
            recommendations.append("Improve call-to-action to increase engagement")
        
        if engagement.sentiment_score < 0.5:
            recommendations.append("Adjust tone to be more positive and inspiring")
        
        if engagement.views < audience.size * 0.5:
            recommendations.append("Optimize delivery timing and channel selection")
        
        return recommendations
    
    def _analyze_strategy_performance(
        self,
        strategy: MessagingStrategy,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze overall strategy performance"""
        return {
            'overall_effectiveness': performance_data.get('avg_effectiveness', 0.5),
            'audience_performance': performance_data.get('audience_breakdown', {}),
            'channel_performance': performance_data.get('channel_breakdown', {}),
            'theme_performance': performance_data.get('theme_breakdown', {}),
            'improvement_areas': performance_data.get('improvement_areas', []),
            'updated_targets': {
                'engagement_rate': min(strategy.effectiveness_targets['engagement_rate'] + 0.05, 1.0),
                'cultural_alignment': min(strategy.effectiveness_targets['cultural_alignment'] + 0.03, 1.0)
            }
        }
    
    def _suggest_content_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest content improvements based on analysis"""
        improvements = []
        
        if analysis['overall_effectiveness'] < 0.6:
            improvements.append("Strengthen cultural theme integration in messages")
            improvements.append("Use more concrete examples and success stories")
        
        if 'clarity' in analysis.get('improvement_areas', []):
            improvements.append("Simplify language and improve message clarity")
        
        return improvements
    
    def _optimize_audience_targeting(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize audience targeting based on performance"""
        return {
            'high_performing_segments': analysis.get('audience_performance', {}),
            'targeting_adjustments': "Focus on segments with >70% engagement",
            'new_segment_opportunities': "Consider creating micro-segments for better personalization"
        }
    
    def _optimize_channel_selection(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize communication channel selection"""
        return {
            'channel_effectiveness': analysis.get('channel_performance', {}),
            'recommendations': "Prioritize channels with highest engagement rates",
            'multi_channel_strategy': "Use complementary channels for reinforcement"
        }
    
    def _optimize_timing(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Optimize message timing"""
        return {
            'optimal_times': "Tuesday-Thursday, 9-11 AM for highest engagement",
            'audience_specific': "Adjust timing based on audience work patterns",
            'campaign_spacing': "Space messages 3-5 days apart for optimal impact"
        }
    
    def _suggest_template_updates(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest template updates based on performance"""
        return [
            "Update templates with high-performing language patterns",
            "Add more personalization variables for better customization",
            "Create audience-specific template variations"
        ]
    
    # Helper methods for content customization
    def _adjust_tone_for_leadership(self, content: str) -> str:
        """Adjust tone for leadership audience"""
        # Add strategic context and business impact
        return content.replace("we will", "we are strategically positioned to")
    
    def _adjust_tone_for_new_hires(self, content: str) -> str:
        """Adjust tone for new hires"""
        # Add welcoming and explanatory context
        return f"Welcome to our culture! {content}"
    
    def _format_for_slack(self, content: str) -> str:
        """Format content for Slack"""
        # Shorter, more casual format
        return content[:500] + "..." if len(content) > 500 else content
    
    def _format_for_email(self, content: str) -> str:
        """Format content for email"""
        # Add proper email structure
        return content
    
    def _add_cultural_context(self, content: str, context: Dict[str, str]) -> str:
        """Add cultural context to message"""
        if 'location' in context:
            content = f"[{context['location']}] {content}"
        return content
    
    def _get_optimal_length(self, channel: MessageChannel) -> int:
        """Get optimal message length for channel"""
        length_map = {
            MessageChannel.SLACK: 300,
            MessageChannel.EMAIL: 800,
            MessageChannel.INTRANET: 1200
        }
        return length_map.get(channel, 500)
    
    def _get_preferred_format(self, channel: MessageChannel) -> str:
        """Get preferred format for channel"""
        format_map = {
            MessageChannel.SLACK: "casual",
            MessageChannel.EMAIL: "professional",
            MessageChannel.TOWN_HALL: "formal"
        }
        return format_map.get(channel, "professional")
    
    def _calculate_personalization_score(
        self,
        message: CulturalMessage,
        audience: AudienceProfile
    ) -> float:
        """Calculate personalization score"""
        # Simple scoring based on audience characteristics match
        score = 0.5  # Base score
        
        # Adjust based on audience type alignment
        if message.message_type == MessageType.VISION_COMMUNICATION:
            if audience.audience_type == AudienceType.LEADERSHIP_TEAM:
                score += 0.2
        
        return min(score, 1.0)