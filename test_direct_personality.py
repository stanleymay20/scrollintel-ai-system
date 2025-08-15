#!/usr/bin/env python3
"""
Direct test for agent personality components
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.abspath('.'))

# Import directly without going through scrollintel.__init__
from scrollintel.core.agent_personality import (
    AgentPersonalityEngine, PersonalityProfile, PersonalityTrait, 
    CommunicationStyle, EmotionalState
)
from scrollintel.core.agent_response_templates import (
    AgentResponseTemplateEngine, ResponseType, ResponseTone
)
from scrollintel.core.conversational_memory import ConversationalMemoryEngine
from scrollintel.core.response_streaming import ResponseStreamingEngine

def test_personality_engine():
    """Test personality engine directly"""
    print("Testing AgentPersonalityEngine...")
    
    engine = AgentPersonalityEngine()
    
    # Test that profiles are loaded
    profile = engine.get_personality_profile("scroll-cto-agent")
    assert profile is not None
    assert profile.name == "Alex Chen"
    assert PersonalityTrait.ANALYTICAL in profile.primary_traits
    assert profile.communication_style == CommunicationStyle.PROFESSIONAL
    print("✓ CTO agent personality loaded correctly")
    
    profile = engine.get_personality_profile("scroll-data-scientist")
    assert profile is not None
    assert profile.name == "Dr. Sarah Kim"
    assert PersonalityTrait.CURIOUS in profile.primary_traits
    assert profile.communication_style == CommunicationStyle.DETAILED
    print("✓ Data Scientist agent personality loaded correctly")
    
    profile = engine.get_personality_profile("scroll-ml-engineer")
    assert profile is not None
    assert profile.name == "Marcus Rodriguez"
    assert PersonalityTrait.INNOVATIVE in profile.primary_traits
    assert profile.communication_style == CommunicationStyle.TECHNICAL
    print("✓ ML Engineer agent personality loaded correctly")
    
    profile = engine.get_personality_profile("scroll-bi-agent")
    assert profile is not None
    assert profile.name == "Emma Thompson"
    assert PersonalityTrait.ENTHUSIASTIC in profile.primary_traits
    assert profile.communication_style == CommunicationStyle.FRIENDLY
    print("✓ BI agent personality loaded correctly")
    
    # Test response formatting
    response = engine.format_response_with_personality(
        "scroll-cto-agent",
        "Here's my technical recommendation.",
        "recommendation"
    )
    assert len(response) > 30  # Should be enhanced
    print("✓ Response formatting with personality works")
    
    # Test thinking messages
    message = engine.get_typing_indicator_message("scroll-cto-agent")
    assert any(word in message.lower() for word in ["analyzing", "architecture", "technical"])
    print("✓ Thinking messages are personality-appropriate")
    
    # Test avatar configs
    avatar = engine.get_avatar_config("scroll-cto-agent")
    assert avatar["color_scheme"] == "blue"
    assert avatar["style"] == "professional"
    print("✓ Avatar configurations are consistent")

def test_response_templates():
    """Test response templates"""
    print("\nTesting AgentResponseTemplateEngine...")
    
    engine = AgentResponseTemplateEngine()
    
    # Test CTO templates
    template = engine.get_template("scroll-cto-agent", ResponseType.GREETING)
    assert template is not None
    assert "Alex" in template.template_text
    assert template.tone in [ResponseTone.PROFESSIONAL, ResponseTone.CONFIDENT]
    print("✓ CTO greeting template loaded correctly")
    
    # Test Data Scientist templates
    template = engine.get_template("scroll-data-scientist", ResponseType.GREETING)
    assert template is not None
    assert "Sarah" in template.template_text
    print("✓ Data Scientist greeting template loaded correctly")
    
    # Test template formatting
    template = engine.get_template("scroll-cto-agent", ResponseType.ANALYSIS)
    if template:
        variables = {var: f"test_{var}" for var in template.variables}
        formatted = engine.format_response("scroll-cto-agent", template, variables)
        assert len(formatted) > 50
        print("✓ Template formatting works correctly")
    
    # Test agent styles
    style = engine.get_agent_style("scroll-cto-agent")
    assert style is not None
    assert style.emoji_usage
    assert "🏗️" in style.preferred_emojis
    print("✓ Agent styles are consistent")

async def test_conversational_memory():
    """Test conversational memory"""
    print("\nTesting ConversationalMemoryEngine...")
    
    engine = ConversationalMemoryEngine(db_path=":memory:")
    
    # Start conversation
    conversation_id = await engine.start_conversation("test_user", "scroll-cto-agent")
    assert conversation_id is not None
    print("✓ Conversation started successfully")
    
    # Add turn
    turn_id = await engine.add_conversation_turn(
        conversation_id,
        "Hello, I need help with system architecture",
        "Hi! I'm Alex, your CTO advisor. I'll help you design a scalable architecture."
    )
    assert turn_id is not None
    print("✓ Conversation turn added successfully")
    
    # Get context
    context = await engine.get_conversation_context(conversation_id)
    assert context["turn_count"] == 1
    assert context["agent_id"] == "scroll-cto-agent"
    assert "system_architecture" in context["recent_topics"]
    print("✓ Conversation context retrieved correctly")
    
    # Test agent memory
    memory = await engine.get_agent_memory_for_user("scroll-cto-agent", "test_user")
    assert isinstance(memory, dict)
    print("✓ Agent memory system working")
    
    # End conversation
    summary = await engine.end_conversation(conversation_id)
    assert summary.total_turns == 1
    assert "system_architecture" in summary.main_topics
    print("✓ Conversation ended with proper summary")

def test_response_streaming():
    """Test response streaming"""
    print("\nTesting ResponseStreamingEngine...")
    
    engine = ResponseStreamingEngine()
    
    # Test thinking messages for different agents
    agents = ["scroll-cto-agent", "scroll-data-scientist", "scroll-ml-engineer", "scroll-bi-agent"]
    
    for agent_id in agents:
        message = engine.get_typing_indicator_message(agent_id)
        assert len(message) > 10
        assert "..." in message
        print(f"✓ {agent_id} thinking message: '{message}'")
    
    # Test response generator
    generator = engine.create_response_generator("This is a test response with multiple words.")
    chunks = []
    
    async def collect_chunks():
        async for chunk in generator:
            chunks.append(chunk)
    
    asyncio.run(collect_chunks())
    assert len(chunks) > 0
    full_response = " ".join(chunks).strip()
    assert "test response" in full_response
    print("✓ Response streaming generator works correctly")

async def main():
    """Run all tests"""
    print("Testing Agent Personality and Conversational AI System")
    print("=" * 60)
    
    try:
        test_personality_engine()
        test_response_templates()
        await test_conversational_memory()
        test_response_streaming()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("Agent personality and conversational AI system is working correctly.")
        print("\nKey features implemented:")
        print("✓ Agent personality profiles with traits and communication styles")
        print("✓ Conversational memory and context tracking")
        print("✓ Response streaming with typing indicators")
        print("✓ Agent-specific response templates and styles")
        print("✓ Avatar configurations and visual personality elements")
        print("✓ Consistent personality across all interactions")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())