"""
Tests for Cultural Messaging Engine

Comprehensive tests for cultural messaging framework including message creation,
customization, effectiveness tracking, and optimization.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import uuid

from scrollintel.engines.cultural_messaging_engine import CulturalMessagingEngine
from scrollintel.models.cultural_messaging_models import (
    MessageType, AudienceType, MessageChannel
)


def test_simple_engine_creation():
    """Simple test to verify engine creation"""
    engine = CulturalMessagingEngine()
    assert engine is not None


class TestCulturalMessagingEngine:
    """Test suite for Cultural Messaging Engine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = CulturalMessagingEngine()
        
        # Sample data
        self.sample_vision = "To be the most innovative and customer-focused company in the world"
        self.sample_values = ["Innovation", "Customer Focus", "Integrity", "Excellence"]
        self.sample_audience_data = [
            {
                "name": "Leadership Team",
                "type": "leadership_team",
                "size": 25,
                "characteristics": {"seniority": "high", "decision_making": "strategic"},
                "preferences": {"channel": "email", "timing": "morning"},
                "cultural_context": {"location": "headquarters"}
            },
            {
                "name": "Engineering Team",
                "type": "department_specific",
                "size": 150,
                "characteristics": {"technical": True, "collaborative": True},
                "preferences": {"channel": "slack", "timing": "afternoon"},
                "cultural_context": {"department": "engineering"}
            }
        ]
    
    def test_create_messaging_strategy(self):
        """Test creating comprehensive messaging strategy"""
        strategy = self.engine.create_messaging_strategy(
            organization_id="org_123",
            cultural_vision=self.sample_vision,
            core_values=self.sample_values,
            audience_data=self.sample_audience_data
        )
        
        assert strategy.id is not None
        assert strategy.organization_id == "org_123"
        assert strategy.cultural_vision == self.sample_vision
        assert strategy.core_values == self.sample_values
        assert len(strategy.audience_segments) == 2
        assert len(strategy.message_templates) > 0
        assert "innovation" in strategy.key_themes
        assert "customer_focus" in strategy.key_themes
        assert strategy.effectiveness_targets["engagement_rate"] == 0.75
        
        # Verify audience profiles are created
        assert len(self.engine.audience_profiles) == 2
        
        # Verify strategy is stored
        assert strategy.id in self.engine.messaging_strategies
    
    def test_create_cultural_message(self):
        """Test creating cultural message"""
        message = self.engine.create_cultural_message(
            title="Vision Announcement",
            content="Our vision is to transform the industry through innovation and excellence.",
            message_type=MessageType.VISION_COMMUNICATION,
            cultural_themes=["innovation", "excellence"],
            key_values=["Innovation", "Excellence"]
        )
        
        assert message.id is not None
        assert message.title == "Vision Announcement"
        assert message.message_type == MessageType.VISION_COMMUNICATION
        assert "innovation" in message.cultural_themes
        assert "Excellence" in message.key_values
        assert message.version == 1
        assert message.is_active is True
        assert "alignment_score" in message.metadata
        
        # Verify message is stored
        assert message.id in self.engine.message_history
    
    def test_customize_message_for_audience(self):
        """Test message customization for specific audience"""
        # Create strategy and message first
        strategy = self.engine.create_messaging_strategy(
            organization_id="org_123",
            cultural_vision=self.sample_vision,
            core_values=self.sample_values,
            audience_data=self.sample_audience_data
        )
        
        message = self.engine.create_cultural_message(
            title="Values Update",
            content="We are committed to our core values of innovation and excellence.",
            message_type=MessageType.VALUES_REINFORCEMENT,
            cultural_themes=["innovation"],
            key_values=["Innovation"]
        )
        
        # Get audience ID
        audience_id = list(self.engine.audience_profiles.keys())[0]
        
        customization = self.engine.customize_message_for_audience(
            message_id=message.id,
            audience_id=audience_id,
            channel=MessageChannel.EMAIL,
            delivery_timing=datetime.now() + timedelta(hours=1)
        )
        
        assert customization.id is not None
        assert customization.base_message_id == message.id
        assert customization.audience_id == audience_id
        assert customization.channel == MessageChannel.EMAIL
        assert customization.customized_content is not None
        assert "audience_preferences" in customization.personalization_data
        assert "personalization_score" in customization.personalization_data
    
    def test_track_message_effectiveness(self):
        """Test message effectiveness tracking"""
        # Create strategy and message first
        strategy = self.engine.create_messaging_strategy(
            organization_id="org_123",
            cultural_vision=self.sample_vision,
            core_values=self.sample_values,
            audience_data=self.sample_audience_data
        )
        
        message = self.engine.create_cultural_message(
            title="Team Update",
            content="Great progress on our innovation initiatives this quarter.",
            message_type=MessageType.SUCCESS_CELEBRATION,
            cultural_themes=["innovation"],
            key_values=["Innovation"]
        )
        
        audience_id = list(self.engine.audience_profiles.keys())[0]
        
        engagement_data = {
            "channel": "email",
            "views": 100,
            "clicks": 25,
            "shares": 5,
            "responses": 10,
            "sentiment_score": 0.8,
            "time_spent": 120,
            "actions_taken": 15,
            "discussions_started": 3,
            "follow_up_actions": 8,
            "feedback": {
                "alignment_indicators": {
                    "cultural_resonance": 0.85,
                    "message_clarity": 0.90,
                    "relevance": 0.80
                }
            }
        }
        
        effectiveness = self.engine.track_message_effectiveness(
            message_id=message.id,
            audience_id=audience_id,
            engagement_data=engagement_data
        )
        
        assert effectiveness.id is not None
        assert effectiveness.message_id == message.id
        assert effectiveness.audience_id == audience_id
        assert effectiveness.effectiveness_score > 0
        assert effectiveness.cultural_alignment_score > 0
        assert "engagement_depth" in effectiveness.behavior_change_indicators
        assert "action_rate" in effectiveness.behavior_change_indicators
        assert len(effectiveness.recommendations) >= 0
        
        # Verify effectiveness data is stored
        key = f"{message.id}_{audience_id}"
        assert key in self.engine.effectiveness_data
    
    def test_optimize_messaging_strategy(self):
        """Test messaging strategy optimization"""
        # Create strategy first
        strategy = self.engine.create_messaging_strategy(
            organization_id="org_123",
            cultural_vision=self.sample_vision,
            core_values=self.sample_values,
            audience_data=self.sample_audience_data
        )
        
        performance_data = {
            "avg_effectiveness": 0.65,
            "audience_breakdown": {
                "leadership_team": 0.80,
                "engineering_team": 0.50
            },
            "channel_breakdown": {
                "email": 0.70,
                "slack": 0.60
            },
            "theme_breakdown": {
                "innovation": 0.75,
                "excellence": 0.55
            },
            "improvement_areas": ["clarity", "engagement"]
        }
        
        optimizations = self.engine.optimize_messaging_strategy(
            strategy_id=strategy.id,
            performance_data=performance_data
        )
        
        assert "content_improvements" in optimizations
        assert "audience_targeting" in optimizations
        assert "channel_optimization" in optimizations
        assert "timing_recommendations" in optimizations
        assert "template_updates" in optimizations
        
        # Verify strategy is updated
        updated_strategy = self.engine.messaging_strategies[strategy.id]
        assert updated_strategy.updated_at > strategy.created_at
    
    def test_create_messaging_campaign(self):
        """Test creating messaging campaign"""
        campaign = self.engine.create_messaging_campaign(
            name="Q1 Culture Initiative",
            description="Reinforce our cultural values for the new quarter",
            cultural_objectives=["Increase innovation mindset", "Strengthen collaboration"],
            target_audiences=["all_employees", "leadership_team"],
            messages=["msg_001", "msg_002"],
            duration_days=30
        )
        
        assert campaign.id is not None
        assert campaign.name == "Q1 Culture Initiative"
        assert len(campaign.cultural_objectives) == 2
        assert len(campaign.target_audiences) == 2
        assert len(campaign.messages) == 2
        assert campaign.status == "planned"
        assert campaign.end_date > campaign.start_date
        
        # Verify campaign is stored
        assert campaign.id in self.engine.active_campaigns
    
    def test_extract_cultural_themes(self):
        """Test cultural theme extraction"""
        vision = "We innovate through collaboration and maintain excellence with integrity"
        values = ["Innovation", "Collaboration", "Excellence", "Integrity"]
        
        themes = self.engine._extract_cultural_themes(vision, values)
        
        assert "innovation" in themes
        assert "collaboration" in themes
        assert "excellence" in themes
        assert "integrity" in themes
    
    def test_validate_cultural_alignment(self):
        """Test cultural alignment validation"""
        message = self.engine.create_cultural_message(
            title="Innovation Focus",
            content="Our commitment to innovation and excellence drives breakthrough results",
            message_type=MessageType.VISION_COMMUNICATION,
            cultural_themes=["innovation", "excellence"],
            key_values=["Innovation", "Excellence"]
        )
        
        alignment_score = self.engine._validate_cultural_alignment(message)
        
        assert 0 <= alignment_score <= 1
        assert alignment_score > 0.5  # Should be high due to theme/value presence
    
    def test_calculate_effectiveness_score(self):
        """Test effectiveness score calculation"""
        # Create mock objects
        message = Mock()
        message.metadata = {"alignment_score": 0.8}
        
        audience = Mock()
        audience.size = 100
        
        engagement = Mock()
        engagement.engagement_rate = 0.6
        engagement.views = 80
        
        score = self.engine._calculate_effectiveness_score(message, audience, engagement)
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
    
    def test_customize_content_for_leadership(self):
        """Test content customization for leadership audience"""
        base_content = "We will achieve our goals through innovation"
        
        # Create mock audience
        audience = Mock()
        audience.audience_type = AudienceType.LEADERSHIP_TEAM
        audience.communication_preferences = {}
        audience.cultural_context = {}
        
        customized = self.engine._customize_content(
            base_content, audience, MessageChannel.EMAIL
        )
        
        assert "strategically positioned" in customized
    
    def test_customize_content_for_new_hires(self):
        """Test content customization for new hires"""
        base_content = "Our values guide everything we do"
        
        # Create mock audience
        audience = Mock()
        audience.audience_type = AudienceType.NEW_HIRES
        audience.communication_preferences = {}
        audience.cultural_context = {}
        
        customized = self.engine._customize_content(
            base_content, audience, MessageChannel.EMAIL
        )
        
        assert "Welcome to our culture!" in customized
    
    def test_format_for_slack(self):
        """Test Slack formatting"""
        long_content = "This is a very long message " * 50  # > 500 chars
        
        formatted = self.engine._format_for_slack(long_content)
        
        assert len(formatted) <= 503  # 500 + "..."
        assert formatted.endswith("...")
    
    def test_generate_improvement_recommendations(self):
        """Test improvement recommendation generation"""
        message = Mock()
        audience = Mock()
        engagement = Mock()
        engagement.engagement_rate = 0.2  # Low engagement
        engagement.sentiment_score = 0.3  # Low sentiment
        engagement.views = 10
        audience.size = 100  # Low reach
        
        recommendations = self.engine._generate_improvement_recommendations(
            message, audience, engagement, 0.4  # Low effectiveness
        )
        
        assert len(recommendations) > 0
        assert any("cultural alignment" in rec for rec in recommendations)
        assert any("call-to-action" in rec for rec in recommendations)
        assert any("positive and inspiring" in rec for rec in recommendations)
        assert any("delivery timing" in rec for rec in recommendations)
    
    def test_analyze_behavior_change_indicators(self):
        """Test behavior change indicator analysis"""
        message = Mock()
        audience = Mock()
        
        engagement_data = {
            "views": 100,
            "time_spent": 300,
            "actions_taken": 20,
            "discussions_started": 5,
            "follow_up_actions": 15
        }
        
        indicators = self.engine._analyze_behavior_change_indicators(
            message, audience, engagement_data
        )
        
        assert "engagement_depth" in indicators
        assert "action_rate" in indicators
        assert "discussion_rate" in indicators
        assert "follow_up_rate" in indicators
        
        assert indicators["engagement_depth"] == 3.0  # 300/100
        assert indicators["action_rate"] == 0.2  # 20/100
        assert indicators["discussion_rate"] == 0.05  # 5/100
        assert indicators["follow_up_rate"] == 0.15  # 15/100
    
    def test_error_handling_invalid_message_id(self):
        """Test error handling for invalid message ID"""
        with pytest.raises(ValueError, match="Message or audience not found"):
            self.engine.customize_message_for_audience(
                message_id="invalid_id",
                audience_id="invalid_audience",
                channel=MessageChannel.EMAIL
            )
    
    def test_error_handling_invalid_strategy_id(self):
        """Test error handling for invalid strategy ID"""
        with pytest.raises(ValueError, match="Strategy not found"):
            self.engine.optimize_messaging_strategy(
                strategy_id="invalid_id",
                performance_data={}
            )
    
    def test_message_template_creation(self):
        """Test message template creation"""
        templates = self.engine._create_default_templates(
            core_values=["Innovation", "Excellence"],
            themes=["innovation", "excellence"]
        )
        
        assert len(templates) >= 2
        
        # Check vision communication template
        vision_template = next(
            (t for t in templates if t.message_type == MessageType.VISION_COMMUNICATION),
            None
        )
        assert vision_template is not None
        assert "vision_statement" in vision_template.required_variables
        assert "values_list" in vision_template.required_variables
        
        # Check values reinforcement template
        values_template = next(
            (t for t in templates if t.message_type == MessageType.VALUES_REINFORCEMENT),
            None
        )
        assert values_template is not None
        assert "featured_value" in values_template.required_variables
        assert "success_example" in values_template.required_variables
    
    def test_personalization_data_generation(self):
        """Test personalization data generation"""
        message = Mock()
        
        audience = Mock()
        audience.communication_preferences = {"timing": "morning", "format": "brief"}
        audience.cultural_context = {"location": "remote", "department": "engineering"}
        audience.engagement_history = {"email": 0.7, "slack": 0.8}
        
        personalization_data = self.engine._generate_personalization_data(
            message, audience, MessageChannel.EMAIL
        )
        
        assert "audience_preferences" in personalization_data
        assert "cultural_context" in personalization_data
        assert "channel_optimization" in personalization_data
        assert "engagement_history" in personalization_data
        assert "personalization_score" in personalization_data
        
        assert personalization_data["audience_preferences"]["timing"] == "morning"
        assert personalization_data["cultural_context"]["location"] == "remote"
        assert personalization_data["engagement_history"]["email"] == 0.7
    
    def test_optimal_channel_settings(self):
        """Test optimal channel settings"""
        # Test optimal length
        slack_length = self.engine._get_optimal_length(MessageChannel.SLACK)
        email_length = self.engine._get_optimal_length(MessageChannel.EMAIL)
        
        assert slack_length == 300
        assert email_length == 800
        assert email_length > slack_length
        
        # Test preferred format
        slack_format = self.engine._get_preferred_format(MessageChannel.SLACK)
        email_format = self.engine._get_preferred_format(MessageChannel.EMAIL)
        
        assert slack_format == "casual"
        assert email_format == "professional"
    
    def test_cultural_context_addition(self):
        """Test cultural context addition to messages"""
        content = "This is a test message"
        context = {"location": "New York Office"}
        
        enhanced_content = self.engine._add_cultural_context(content, context)
        
        assert "[New York Office]" in enhanced_content
        assert content in enhanced_content
    
    def test_assess_cultural_alignment_with_feedback(self):
        """Test cultural alignment assessment with feedback"""
        message = Mock()
        message.metadata = {"alignment_score": 0.7}
        
        feedback = {
            "alignment_indicators": {
                "cultural_resonance": 0.8,
                "message_clarity": 0.9,
                "relevance": 0.75
            }
        }
        
        alignment_score = self.engine._assess_cultural_alignment(message, feedback)
        
        assert 0 <= alignment_score <= 1
        assert alignment_score > 0.7  # Should be higher than base score due to good feedback
    
    def test_strategy_performance_analysis(self):
        """Test strategy performance analysis"""
        strategy = Mock()
        strategy.effectiveness_targets = {"engagement_rate": 0.7, "cultural_alignment": 0.8}
        
        performance_data = {
            "avg_effectiveness": 0.65,
            "audience_breakdown": {"leadership": 0.8, "staff": 0.5},
            "channel_breakdown": {"email": 0.7, "slack": 0.6},
            "theme_breakdown": {"innovation": 0.75},
            "improvement_areas": ["clarity"]
        }
        
        analysis = self.engine._analyze_strategy_performance(strategy, performance_data)
        
        assert "overall_effectiveness" in analysis
        assert "audience_performance" in analysis
        assert "channel_performance" in analysis
        assert "theme_performance" in analysis
        assert "improvement_areas" in analysis
        assert "updated_targets" in analysis
        
        assert analysis["overall_effectiveness"] == 0.65
        assert analysis["updated_targets"]["engagement_rate"] == 0.75  # Increased by 0.05