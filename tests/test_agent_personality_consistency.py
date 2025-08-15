"""
Tests for Agent Personality Consistency
Tests requirements 2.3, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6 for agent personality consistency.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.core.agent_personality import (
    AgentPersonalityEngine, PersonalityProfile, PersonalityTrait, 
    CommunicationStyle, EmotionalState, ConversationContext
)
from scrollintel.core.conversational_memory import ConversationalMemoryEngine
from scrollintel.core.response_streaming import ResponseStreamingEngine
from scrollintel.core.agent_response_templates import (
    AgentResponseTemplateEngine, ResponseType, ResponseTone
)
from scrollintel.core.enhanced_agent_system import EnhancedAgentSystem


class TestAgentPersonalityConsistency:
    """Test agent personality consistency across interactions."""
    
    @pytest.fixture
    def personality_engine(self):
        return AgentPersonalityEngine()
    
    @pytest.fixture
    def enhanced_system(self):
        return EnhancedAgentSystem()
    
    def test_personality_profiles_loaded(self, personality_engine):
        """Test that all agent personality profiles are loaded correctly."""
        expected_agents = [
            "scroll-cto-agent",
            "scroll-data-scientist", 
            "scroll-ml-engineer",
            "scroll-bi-agent"
        ]
        
        for agent_id in expected_agents:
            profile = personality_engine.get_personality_profile(agent_id)
            assert profile is not None, f"Profile not found for {agent_id}"
            assert profile.agent_id == agent_id
            assert profile.name is not None
            assert len(profile.primary_traits) > 0
            assert profile.communication_style is not None
            assert profile.emotional_baseline is not None
    
    def test_cto_agent_personality_consistency(self, personality_engine):
        """Test CTO agent personality consistency."""
        profile = personality_engine.get_personality_profile("scroll-cto-agent")
        
        # Check personality traits
        assert PersonalityTrait.ANALYTICAL in profile.primary_traits
        assert PersonalityTrait.ASSERTIVE in profile.primary_traits
        assert PersonalityTrait.INNOVATIVE in profile.primary_traits
        
        # Check communication style
        assert profile.communication_style == CommunicationStyle.PROFESSIONAL
        
        # Check emotional baseline
        assert profile.emotional_baseline == EmotionalState.CONFIDENT
        
        # Check expertise confidence is high
        assert profile.expertise_confidence >= 0.8
        
        # Check avatar configuration
        avatar_config = personality_engine.get_avatar_config("scroll-cto-agent")
        assert avatar_config["color_scheme"] == "blue"
        assert "glasses" in avatar_config.get("accessories", [])
    
    def test_data_scientist_personality_consistency(self, personality_engine):
        """Test Data Scientist agent personality consistency."""
        profile = personality_engine.get_personality_profile("scroll-data-scientist")
        
        # Check personality traits
        assert PersonalityTrait.ANALYTICAL in profile.primary_traits
        assert PersonalityTrait.CURIOUS in profile.primary_traits
        assert PersonalityTrait.METHODICAL in profile.primary_traits
        
        # Check communication style
        assert profile.communication_style == CommunicationStyle.DETAILED
        
        # Check emotional baseline
        assert profile.emotional_baseline == EmotionalState.CURIOUS
        
        # Check preferred emojis include data-related ones
        assert "📊" in profile.preferred_emojis
        assert "🔍" in profile.preferred_emojis
    
    def test_ml_engineer_personality_consistency(self, personality_engine):
        """Test ML Engineer agent personality consistency."""
        profile = personality_engine.get_personality_profile("scroll-ml-engineer")
        
        # Check personality traits
        assert PersonalityTrait.INNOVATIVE in profile.primary_traits
        assert PersonalityTrait.PATIENT in profile.primary_traits
        assert PersonalityTrait.METHODICAL in profile.primary_traits
        
        # Check communication style
        assert profile.communication_style == CommunicationStyle.TECHNICAL
        
        # Check emotional baseline
        assert profile.emotional_baseline == EmotionalState.FOCUSED
        
        # Check preferred emojis include ML-related ones
        assert "🤖" in profile.preferred_emojis
        assert "🧠" in profile.preferred_emojis
    
    def test_bi_agent_personality_consistency(self, personality_engine):
        """Test BI agent personality consistency."""
        profile = personality_engine.get_personality_profile("scroll-bi-agent")
        
        # Check personality traits
        assert PersonalityTrait.SUPPORTIVE in profile.primary_traits
        assert PersonalityTrait.ENTHUSIASTIC in profile.primary_traits
        assert PersonalityTrait.DIRECT in profile.primary_traits
        
        # Check communication style
        assert profile.communication_style == CommunicationStyle.FRIENDLY
        
        # Check emotional baseline
        assert profile.emotional_baseline == EmotionalState.EXCITED
        
        # Check preferred emojis include business-related ones
        assert "💼" in profile.preferred_emojis
        assert "📈" in profile.preferred_emojis
    
    def test_response_formatting_consistency(self, personality_engine):
        """Test that response formatting is consistent with personality."""
        # Test CTO agent response formatting
        cto_response = personality_engine.format_response_with_personality(
            "scroll-cto-agent",
            "Here's my technical recommendation for your architecture.",
            "recommendation"
        )
        
        assert "Alex" in cto_response or "CTO" in cto_response
        assert len(cto_response) > 50  # Should be enhanced with personality
        
        # Test Data Scientist response formatting
        ds_response = personality_engine.format_response_with_personality(
            "scroll-data-scientist",
            "The data analysis shows interesting patterns.",
            "analysis"
        )
        
        assert "Sarah" in ds_response or "data" in ds_response.lower()
        assert len(ds_response) > 40
    
    def test_thinking_messages_consistency(self, personality_engine):
        """Test that thinking messages are consistent with agent personalities."""
        # Test CTO thinking messages
        cto_message = personality_engine.get_typing_indicator_message("scroll-cto-agent")
        assert any(word in cto_message.lower() for word in [
            "analyzing", "architecture", "technical", "evaluating", "reviewing"
        ])
        
        # Test Data Scientist thinking messages
        ds_message = personality_engine.get_typing_indicator_message("scroll-data-scientist")
        assert any(word in ds_message.lower() for word in [
            "analyzing", "data", "statistical", "patterns", "crunching"
        ])
        
        # Test ML Engineer thinking messages
        ml_message = personality_engine.get_typing_indicator_message("scroll-ml-engineer")
        assert any(word in ml_message.lower() for word in [
            "training", "model", "optimizing", "hyperparameters", "evaluating"
        ])
    
    def test_conversation_context_consistency(self, personality_engine):
        """Test that conversation context maintains personality consistency."""
        conversation_id = "test_conv_123"
        user_id = "test_user"
        agent_id = "scroll-cto-agent"
        
        # Create conversation context
        context = personality_engine.create_conversation_context(
            conversation_id, user_id, agent_id
        )
        
        assert context.conversation_id == conversation_id
        assert context.user_id == user_id
        assert context.agent_id == agent_id
        assert context.interaction_count == 0
        
        # Update context with message
        personality_engine.update_conversation_context(conversation_id, {
            "role": "user",
            "content": "I need help with system architecture",
            "timestamp": datetime.now().isoformat()
        })
        
        # Check that context was updated
        updated_context = personality_engine.conversation_contexts[conversation_id]
        assert updated_context.interaction_count == 1
        assert len(updated_context.message_history) == 1
        assert "system_architecture" in updated_context.conversation_topics
    
    def test_enthusiasm_consistency(self, personality_engine):
        """Test that enthusiasm levels are consistent with personality."""
        # BI agent should show more enthusiasm
        bi_enthusiasm = personality_engine.should_show_enthusiasm("scroll-bi-agent")
        assert bi_enthusiasm  # BI agent has ENTHUSIASTIC trait
        
        # CTO agent should show less enthusiasm (more professional)
        cto_enthusiasm = personality_engine.should_show_enthusiasm("scroll-cto-agent")
        # CTO agent doesn't have ENTHUSIASTIC trait, but has CONFIDENT baseline
        
        # ML engineer should show moderate enthusiasm
        ml_enthusiasm = personality_engine.should_show_enthusiasm("scroll-ml-engineer")
    
    def test_avatar_config_consistency(self, personality_engine):
        """Test that avatar configurations are consistent with personalities."""
        # Test CTO avatar
        cto_avatar = personality_engine.get_avatar_config("scroll-cto-agent")
        assert cto_avatar["style"] == "professional"
        assert cto_avatar["color_scheme"] == "blue"
        
        # Test Data Scientist avatar
        ds_avatar = personality_engine.get_avatar_config("scroll-data-scientist")
        assert ds_avatar["style"] == "academic"
        assert ds_avatar["color_scheme"] == "purple"
        
        # Test ML Engineer avatar
        ml_avatar = personality_engine.get_avatar_config("scroll-ml-engineer")
        assert ml_avatar["style"] == "casual_tech"
        assert ml_avatar["color_scheme"] == "green"
        
        # Test BI agent avatar
        bi_avatar = personality_engine.get_avatar_config("scroll-bi-agent")
        assert bi_avatar["style"] == "business_casual"
        assert bi_avatar["color_scheme"] == "orange"


class TestResponseTemplateConsistency:
    """Test response template consistency."""
    
    @pytest.fixture
    def template_engine(self):
        return AgentResponseTemplateEngine()
    
    def test_template_loading(self, template_engine):
        """Test that templates are loaded for all agents."""
        expected_agents = [
            "scroll-cto-agent",
            "scroll-data-scientist",
            "scroll-ml-engineer", 
            "scroll-bi-agent"
        ]
        
        for agent_id in expected_agents:
            templates = template_engine.templates.get(agent_id, [])
            assert len(templates) > 0, f"No templates found for {agent_id}"
            
            # Check that greeting templates exist
            greeting_templates = [t for t in templates if t.response_type == ResponseType.GREETING]
            assert len(greeting_templates) > 0, f"No greeting templates for {agent_id}"
    
    def test_cto_template_consistency(self, template_engine):
        """Test CTO agent template consistency."""
        cto_templates = template_engine.templates["scroll-cto-agent"]
        
        # Check greeting template
        greeting_template = template_engine.get_template(
            "scroll-cto-agent", ResponseType.GREETING, ResponseTone.PROFESSIONAL
        )
        assert greeting_template is not None
        assert "Alex" in greeting_template.template_text
        assert "CTO" in greeting_template.template_text
        
        # Check analysis template
        analysis_template = template_engine.get_template(
            "scroll-cto-agent", ResponseType.ANALYSIS, ResponseTone.ANALYTICAL
        )
        assert analysis_template is not None
        assert "Technical Analysis" in analysis_template.template_text
    
    def test_template_variable_consistency(self, template_engine):
        """Test that template variables are consistent."""
        # Get CTO analysis template
        template = template_engine.get_template(
            "scroll-cto-agent", ResponseType.ANALYSIS, ResponseTone.ANALYTICAL
        )
        
        if template and template.variables:
            # Test formatting with variables
            test_variables = {var: f"test_{var}" for var in template.variables}
            
            try:
                formatted = template_engine.format_response(
                    "scroll-cto-agent", template, test_variables
                )
                assert len(formatted) > 0
                # Check that variables were replaced
                for var in template.variables:
                    assert f"test_{var}" in formatted or f"{{{var}}}" not in formatted
            except KeyError as e:
                pytest.fail(f"Missing required variable: {e}")
    
    def test_style_consistency(self, template_engine):
        """Test that agent styles are consistent."""
        # Test CTO style
        cto_style = template_engine.get_agent_style("scroll-cto-agent")
        assert cto_style is not None
        assert cto_style.formatting_preferences["use_headers"]
        assert cto_style.emoji_usage
        assert "🏗️" in cto_style.preferred_emojis
        
        # Test Data Scientist style
        ds_style = template_engine.get_agent_style("scroll-data-scientist")
        assert ds_style is not None
        assert ds_style.formatting_preferences["use_math_notation"]
        assert "📊" in ds_style.preferred_emojis


class TestConversationalMemoryConsistency:
    """Test conversational memory consistency."""
    
    @pytest.fixture
    async def memory_engine(self):
        engine = ConversationalMemoryEngine(db_path=":memory:")
        return engine
    
    @pytest.mark.asyncio
    async def test_conversation_memory_persistence(self, memory_engine):
        """Test that conversation memory persists personality context."""
        user_id = "test_user"
        agent_id = "scroll-cto-agent"
        
        # Start conversation
        conversation_id = await memory_engine.start_conversation(user_id, agent_id)
        assert conversation_id is not None
        
        # Add conversation turn
        turn_id = await memory_engine.add_conversation_turn(
            conversation_id,
            "I need help with system architecture",
            "I'll help you design a scalable architecture. Let me analyze your requirements..."
        )
        assert turn_id is not None
        
        # Get conversation context
        context = await memory_engine.get_conversation_context(conversation_id)
        assert context["agent_id"] == agent_id
        assert context["turn_count"] == 1
        assert "system_architecture" in context["recent_topics"]
    
    @pytest.mark.asyncio
    async def test_agent_memory_consistency(self, memory_engine):
        """Test that agent memory maintains consistency."""
        agent_id = "scroll-cto-agent"
        user_id = "test_user"
        
        # Get agent memory
        memory = await memory_engine.get_agent_memory_for_user(agent_id, user_id)
        assert isinstance(memory, dict)
        
        # Memory should be empty initially
        assert len(memory) == 0


class TestEnhancedSystemIntegration:
    """Test enhanced system integration and consistency."""
    
    @pytest.fixture
    def enhanced_system(self):
        return EnhancedAgentSystem()
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, enhanced_system):
        """Test that enhanced system initializes correctly."""
        # Check that all engines are initialized
        assert enhanced_system.personality_engine is not None
        assert enhanced_system.memory_engine is not None
        assert enhanced_system.streaming_engine is not None
        assert enhanced_system.template_engine is not None
        
        # Check that agents are loaded
        assert len(enhanced_system.agents) > 0
        assert "scroll-cto-agent" in enhanced_system.agents
    
    @pytest.mark.asyncio
    async def test_agent_status_consistency(self, enhanced_system):
        """Test that agent status includes personality information."""
        agent_status = await enhanced_system.get_agent_status("scroll-cto-agent")
        
        assert agent_status["agent_id"] == "scroll-cto-agent"
        assert "personality" in agent_status
        assert agent_status["personality"]["name"] == "Alex Chen"
        assert len(agent_status["personality"]["traits"]) > 0
        assert agent_status["personality"]["communication_style"] == "professional"
        assert "avatar_config" in agent_status
    
    def test_available_agents_consistency(self, enhanced_system):
        """Test that available agents list is consistent."""
        agents = enhanced_system.get_available_agents()
        
        assert len(agents) > 0
        
        for agent in agents:
            assert "id" in agent
            assert "name" in agent
            assert "personality" in agent
            assert "avatar_config" in agent
            assert "capabilities" in agent
            
            # Check personality consistency
            personality = agent["personality"]
            assert len(personality["traits"]) > 0
            assert personality["communication_style"] in [
                "professional", "friendly", "technical", "detailed", "casual"
            ]
            assert 0.0 <= personality["expertise_confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_system_stats_consistency(self, enhanced_system):
        """Test that system stats are consistent."""
        stats = await enhanced_system.get_system_stats()
        
        assert "agents" in stats
        assert "conversations" in stats
        assert "templates" in stats
        assert "memory" in stats
        
        # Check agent stats
        agent_stats = stats["agents"]
        assert agent_stats["total"] > 0
        assert agent_stats["personalities_loaded"] > 0
        
        # Check template stats
        template_stats = stats["templates"]
        assert template_stats["total"] > 0
        assert template_stats["agents_with_templates"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])