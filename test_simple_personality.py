#!/usr/bin/env python3
"""
Simple test for agent personality system without full imports
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_personality_engine():
    """Test personality engine directly"""
    from scrollintel.core.agent_personality import AgentPersonalityEngine
    
    engine = AgentPersonalityEngine()
    
    # Test that profiles are loaded
    profile = engine.get_personality_profile("scroll-cto-agent")
    assert profile is not None
    assert profile.name == "Alex Chen"
    print("✓ CTO agent personality loaded")
    
    profile = engine.get_personality_profile("scroll-data-scientist")
    assert profile is not None
    assert profile.name == "Dr. Sarah Kim"
    print("✓ Data Scientist agent personality loaded")
    
    profile = engine.get_personality_profile("scroll-ml-engineer")
    assert profile is not None
    assert profile.name == "Marcus Rodriguez"
    print("✓ ML Engineer agent personality loaded")
    
    profile = engine.get_personality_profile("scroll-bi-agent")
    assert profile is not None
    assert profile.name == "Emma Thompson"
    print("✓ BI agent personality loaded")
    
    print("All personality profiles loaded successfully!")

def test_response_templates():
    """Test response templates"""
    from scrollintel.core.agent_response_templates import AgentResponseTemplateEngine, ResponseType
    
    engine = AgentResponseTemplateEngine()
    
    # Test CTO templates
    template = engine.get_template("scroll-cto-agent", ResponseType.GREETING)
    assert template is not None
    assert "Alex" in template.template_text
    print("✓ CTO greeting template loaded")
    
    # Test Data Scientist templates
    template = engine.get_template("scroll-data-scientist", ResponseType.GREETING)
    assert template is not None
    assert "Sarah" in template.template_text
    print("✓ Data Scientist greeting template loaded")
    
    print("All response templates loaded successfully!")

def test_conversational_memory():
    """Test conversational memory"""
    import asyncio
    from scrollintel.core.conversational_memory import ConversationalMemoryEngine
    
    async def run_test():
        engine = ConversationalMemoryEngine(db_path=":memory:")
        
        # Start conversation
        conversation_id = await engine.start_conversation("test_user", "scroll-cto-agent")
        assert conversation_id is not None
        print("✓ Conversation started")
        
        # Add turn
        turn_id = await engine.add_conversation_turn(
            conversation_id,
            "Hello, I need help with architecture",
            "Hi! I'm Alex, ready to help with your architecture needs."
        )
        assert turn_id is not None
        print("✓ Conversation turn added")
        
        # Get context
        context = await engine.get_conversation_context(conversation_id)
        assert context["turn_count"] == 1
        print("✓ Conversation context retrieved")
        
        print("Conversational memory working successfully!")
    
    asyncio.run(run_test())

def test_response_streaming():
    """Test response streaming"""
    from scrollintel.core.response_streaming import ResponseStreamingEngine
    
    engine = ResponseStreamingEngine()
    
    # Test thinking messages
    message = engine.get_typing_indicator_message("scroll-cto-agent")
    assert "analyzing" in message.lower() or "architecture" in message.lower()
    print("✓ CTO thinking message generated")
    
    message = engine.get_typing_indicator_message("scroll-data-scientist")
    assert "data" in message.lower() or "analyzing" in message.lower()
    print("✓ Data Scientist thinking message generated")
    
    print("Response streaming working successfully!")

if __name__ == "__main__":
    print("Testing Agent Personality System...")
    print("=" * 50)
    
    try:
        test_personality_engine()
        print()
        
        test_response_templates()
        print()
        
        test_conversational_memory()
        print()
        
        test_response_streaming()
        print()
        
        print("=" * 50)
        print("🎉 All tests passed! Agent personality system is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)