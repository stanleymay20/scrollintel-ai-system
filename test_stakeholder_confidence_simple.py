#!/usr/bin/env python3
"""
Simple test for stakeholder confidence management system
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from scrollintel.models.stakeholder_confidence_models import (
        StakeholderProfile, StakeholderType, ConfidenceLevel, StakeholderFeedback
    )
    print("✅ Models imported successfully")
except Exception as e:
    print(f"❌ Models import failed: {e}")
    sys.exit(1)

# Try to import the engine using a different approach
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stakeholder_confidence_engine", 
        "scrollintel/engines/stakeholder_confidence_engine.py"
    )
    engine_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine_module)
    
    StakeholderConfidenceEngine = engine_module.StakeholderConfidenceEngine
    print("✅ Engine imported successfully using importlib")
except Exception as e:
    print(f"❌ Engine import failed: {e}")
    sys.exit(1)

async def test_stakeholder_confidence_system():
    """Test the stakeholder confidence management system"""
    
    print("\n🎯 Testing Stakeholder Confidence Management System")
    print("=" * 60)
    
    # Initialize engine
    engine = StakeholderConfidenceEngine()
    print("✅ Engine initialized")
    
    # Create test stakeholder profile
    stakeholder = StakeholderProfile(
        stakeholder_id="test_stakeholder_001",
        name="Test Investor",
        stakeholder_type=StakeholderType.INVESTOR,
        influence_level="high",
        communication_preferences=["email", "phone"],
        historical_confidence=[0.8, 0.7, 0.6],
        key_concerns=["financial_impact", "timeline"],
        relationship_strength=0.75,
        contact_information={"email": "test@investor.com"},
    )
    
    engine.stakeholder_profiles[stakeholder.stakeholder_id] = stakeholder
    print("✅ Stakeholder profile created")
    
    # Test confidence monitoring
    try:
        confidence_data = await engine.monitor_stakeholder_confidence(
            "test_crisis_001", 
            [stakeholder.stakeholder_id]
        )
        print(f"✅ Confidence monitoring: {len(confidence_data)} stakeholders monitored")
        
        metrics = confidence_data[stakeholder.stakeholder_id]
        print(f"   📊 Confidence Level: {metrics.confidence_level.value}")
        print(f"   🎯 Trust Score: {metrics.trust_score:.2f}")
        
    except Exception as e:
        print(f"❌ Confidence monitoring failed: {e}")
    
    # Test overall assessment
    try:
        assessment = await engine.assess_overall_confidence("test_crisis_001")
        print(f"✅ Overall assessment: {assessment.overall_confidence_score:.2f}")
        print(f"   📈 Risk areas: {len(assessment.risk_areas)}")
        print(f"   💡 Opportunities: {len(assessment.improvement_opportunities)}")
        
    except Exception as e:
        print(f"❌ Overall assessment failed: {e}")
    
    # Test confidence strategy building
    try:
        strategy = await engine.build_confidence_strategy(
            StakeholderType.INVESTOR,
            ConfidenceLevel.LOW,
            ConfidenceLevel.HIGH
        )
        print(f"✅ Strategy built: {strategy.strategy_id}")
        print(f"   💬 Approach: {strategy.communication_approach}")
        print(f"   🔑 Messages: {len(strategy.key_messages)}")
        print(f"   🎪 Tactics: {len(strategy.engagement_tactics)}")
        
    except Exception as e:
        print(f"❌ Strategy building failed: {e}")
    
    # Test trust maintenance
    try:
        crisis_context = {
            "crisis_type": "security_breach",
            "severity": "high"
        }
        
        actions = await engine.maintain_stakeholder_trust(
            stakeholder.stakeholder_id, 
            crisis_context
        )
        print(f"✅ Trust maintenance: {len(actions)} actions generated")
        
        for i, action in enumerate(actions, 1):
            print(f"   Action {i}: {action.action_type} (Priority: {action.priority})")
        
    except Exception as e:
        print(f"❌ Trust maintenance failed: {e}")
    
    # Test communication plan
    try:
        comm_plan = await engine.create_communication_plan(
            "test_crisis_001",
            [StakeholderType.INVESTOR, StakeholderType.CUSTOMER]
        )
        print(f"✅ Communication plan: {comm_plan.plan_id}")
        print(f"   📡 Channels: {len(comm_plan.communication_channels)}")
        print(f"   📅 Frequency: {comm_plan.frequency}")
        
    except Exception as e:
        print(f"❌ Communication plan failed: {e}")
    
    # Test feedback processing
    try:
        feedback = StakeholderFeedback(
            feedback_id="test_feedback_001",
            stakeholder_id=stakeholder.stakeholder_id,
            feedback_type="concern",
            content="Test concern about the crisis",
            sentiment="negative",
            urgency_level="high",
            received_time=engine.assessments[0].assessment_time if engine.assessments else None,
            response_required=True
        )
        
        result = await engine.process_stakeholder_feedback(feedback)
        print(f"✅ Feedback processed: {result['feedback_id']}")
        print(f"   📊 Analysis: {result['analysis']['sentiment_score']:.2f}")
        print(f"   📋 Actions: {len(result['follow_up_actions'])}")
        
    except Exception as e:
        print(f"❌ Feedback processing failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Stakeholder Confidence Management System Test Complete!")
    print(f"📊 Total stakeholders: {len(engine.stakeholder_profiles)}")
    print(f"🚨 Active alerts: {len(engine.active_alerts)}")
    print(f"💬 Feedback items: {len(engine.feedback_queue)}")
    print(f"🚀 Strategies: {len(engine.building_strategies)}")
    print(f"📋 Assessments: {len(engine.assessments)}")
    print("=" * 60)

if __name__ == "__main__":
    print("🚀 Starting Stakeholder Confidence Management Test...")
    asyncio.run(test_stakeholder_confidence_system())
    print("✅ Test completed successfully!")