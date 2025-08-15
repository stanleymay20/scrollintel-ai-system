"""
Simple tests for Cultural Messaging Engine
"""

import pytest
from datetime import datetime
from scrollintel.engines.cultural_messaging_engine import CulturalMessagingEngine
from scrollintel.models.cultural_messaging_models import MessageType, AudienceType, MessageChannel


def test_engine_creation():
    """Test engine creation"""
    engine = CulturalMessagingEngine()
    assert engine is not None
    assert hasattr(engine, 'message_templates')
    assert hasattr(engine, 'audience_profiles')


def test_create_messaging_strategy():
    """Test creating messaging strategy"""
    engine = CulturalMessagingEngine()
    
    audience_data = [
        {
            "name": "Leadership Team",
            "type": "leadership_team",
            "size": 25,
            "characteristics": {"seniority": "high"},
            "preferences": {"channel": "email"},
            "cultural_context": {"location": "headquarters"}
        }
    ]
    
    strategy = engine.create_messaging_strategy(
        organization_id="org_123",
        cultural_vision="To be innovative and customer-focused",
        core_values=["Innovation", "Customer Focus"],
        audience_data=audience_data
    )
    
    assert strategy.id is not None
    assert strategy.organization_id == "org_123"
    assert len(strategy.audience_segments) == 1
    assert "innovation" in strategy.key_themes


def test_create_cultural_message():
    """Test creating cultural message"""
    engine = CulturalMessagingEngine()
    
    message = engine.create_cultural_message(
        title="Vision Announcement",
        content="Our vision is to transform through innovation.",
        message_type=MessageType.VISION_COMMUNICATION,
        cultural_themes=["innovation"],
        key_values=["Innovation"]
    )
    
    assert message.id is not None
    assert message.title == "Vision Announcement"
    assert message.message_type == MessageType.VISION_COMMUNICATION
    assert "innovation" in message.cultural_themes


def test_message_customization():
    """Test message customization"""
    engine = CulturalMessagingEngine()
    
    # Create strategy and audience first
    audience_data = [
        {
            "name": "Engineering Team",
            "type": "department_specific",
            "size": 100,
            "characteristics": {"technical": True},
            "preferences": {"channel": "slack"},
            "cultural_context": {"department": "engineering"}
        }
    ]
    
    strategy = engine.create_messaging_strategy(
        organization_id="org_123",
        cultural_vision="Innovation through technology",
        core_values=["Innovation"],
        audience_data=audience_data
    )
    
    message = engine.create_cultural_message(
        title="Tech Update",
        content="We are advancing our technical capabilities.",
        message_type=MessageType.VALUES_REINFORCEMENT,
        cultural_themes=["innovation"],
        key_values=["Innovation"]
    )
    
    audience_id = list(engine.audience_profiles.keys())[0]
    
    customization = engine.customize_message_for_audience(
        message_id=message.id,
        audience_id=audience_id,
        channel=MessageChannel.SLACK
    )
    
    assert customization.id is not None
    assert customization.base_message_id == message.id
    assert customization.channel == MessageChannel.SLACK


def test_effectiveness_tracking():
    """Test message effectiveness tracking"""
    engine = CulturalMessagingEngine()
    
    # Setup
    audience_data = [
        {
            "name": "All Staff",
            "type": "all_employees",
            "size": 200,
            "characteristics": {},
            "preferences": {},
            "cultural_context": {}
        }
    ]
    
    strategy = engine.create_messaging_strategy(
        organization_id="org_123",
        cultural_vision="Excellence in everything",
        core_values=["Excellence"],
        audience_data=audience_data
    )
    
    message = engine.create_cultural_message(
        title="Excellence Update",
        content="We strive for excellence in all our work.",
        message_type=MessageType.SUCCESS_CELEBRATION,
        cultural_themes=["excellence"],
        key_values=["Excellence"]
    )
    
    audience_id = list(engine.audience_profiles.keys())[0]
    
    engagement_data = {
        "channel": "email",
        "views": 150,
        "clicks": 30,
        "shares": 5,
        "responses": 10,
        "sentiment_score": 0.8,
        "feedback": {}
    }
    
    effectiveness = engine.track_message_effectiveness(
        message_id=message.id,
        audience_id=audience_id,
        engagement_data=engagement_data
    )
    
    assert effectiveness.id is not None
    assert effectiveness.effectiveness_score >= 0
    assert effectiveness.cultural_alignment_score >= 0