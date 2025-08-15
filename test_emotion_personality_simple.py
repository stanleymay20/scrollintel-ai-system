#!/usr/bin/env python3
"""
Simple test script for emotion and personality engines
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from the modules
from scrollintel.engines.emotion_simulator import EmotionSimulator
from scrollintel.engines.personality_engine import PersonalityEngine
from scrollintel.models.emotion_models import (
    EmotionType, PersonalityTrait, PersonalityProfile, SocialContext
)

def test_emotion_simulator():
    """Test basic emotion simulator functionality"""
    print("Testing EmotionSimulator...")
    
    # Initialize emotion simulator
    emotion_sim = EmotionSimulator()
    print(f"✓ EmotionSimulator initialized with emotion: {emotion_sim.current_emotional_state.primary_emotion}")
    
    # Test emotional stimulus processing
    stimulus = "Great success with the project implementation!"
    response = emotion_sim.process_emotional_stimulus(stimulus)
    print(f"✓ Processed stimulus: {response.emotional_state.primary_emotion} (intensity: {response.emotional_state.intensity:.2f})")
    
    # Test empathy assessment
    empathy = emotion_sim.assess_empathy("colleague", "Person seems excited and energetic")
    print(f"✓ Empathy assessment: {empathy.perceived_emotion} (confidence: {empathy.confidence:.2f})")
    
    print("EmotionSimulator tests passed!\n")

def test_personality_engine():
    """Test basic personality engine functionality"""
    print("Testing PersonalityEngine...")
    
    # Initialize personality engine
    personality_engine = PersonalityEngine()
    print(f"✓ PersonalityEngine initialized with {len(personality_engine.personality_profile.traits)} traits")
    
    # Test decision influence
    decision_context = "Choose between innovative approach vs proven method"
    options = ["innovative_solution", "proven_method"]
    influence = personality_engine.influence_decision(decision_context, options)
    print(f"✓ Decision influence: {influence.personality_bias} (risk tolerance: {influence.risk_tolerance:.2f})")
    
    # Test communication style adaptation
    social_context = SocialContext(social_setting="professional")
    comm_style = personality_engine.adapt_communication_style("technical_team", social_context)
    print(f"✓ Communication style: {comm_style['tone']} (directness: {comm_style['directness']:.2f})")
    
    # Test personality insights
    insights = personality_engine.get_personality_insights()
    print(f"✓ Personality insights: {len(insights['strengths'])} strengths, {len(insights['growth_areas'])} growth areas")
    
    print("PersonalityEngine tests passed!\n")

def test_integration():
    """Test integration between emotion and personality systems"""
    print("Testing Integration...")
    
    emotion_sim = EmotionSimulator()
    personality_engine = PersonalityEngine()
    
    # Test emotional response with personality influence
    stimulus = "Complex technical challenge requires immediate solution"
    emotional_response = emotion_sim.process_emotional_stimulus(stimulus)
    
    behavioral_response = personality_engine.generate_behavioral_response(
        stimulus, 
        SocialContext(social_setting="professional"),
        emotional_response.emotional_state
    )
    
    print(f"✓ Integrated response: {emotional_response.emotional_state.primary_emotion} -> {behavioral_response['behavioral_response'][:50]}...")
    print("Integration tests passed!\n")

if __name__ == "__main__":
    print("Running Emotion and Personality Engine Tests\n")
    
    try:
        test_emotion_simulator()
        test_personality_engine()
        test_integration()
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)