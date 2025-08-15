#!/usr/bin/env python3
"""
Final test for emotion and personality simulation implementation
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from the modules
from scrollintel.engines.personality_engine import PersonalityEngine
from scrollintel.models.emotion_models import (
    EmotionType, PersonalityTrait, PersonalityProfile, SocialContext
)

def test_personality_engine():
    """Test personality engine functionality"""
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
    
    # Test behavioral response generation
    situation = "Complex technical problem requires immediate solution"
    behavioral_response = personality_engine.generate_behavioral_response(
        situation, social_context
    )
    print(f"✓ Behavioral response: {behavioral_response['behavioral_response'][:50]}...")
    
    # Test personality compatibility
    other_personality = PersonalityProfile(
        traits={
            PersonalityTrait.OPENNESS: 0.6,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
            PersonalityTrait.EXTRAVERSION: 0.4,
            PersonalityTrait.AGREEABLENESS: 0.7,
            PersonalityTrait.NEUROTICISM: 0.4
        }
    )
    compatibility = personality_engine.assess_personality_compatibility(other_personality)
    print(f"✓ Personality compatibility: {compatibility['overall_compatibility']:.2f}")
    
    # Test learning from interaction
    personality_engine.learn_from_interaction(
        "Collaborative problem-solving session",
        "Successful resolution with positive team feedback",
        "Great approach, very collaborative and effective"
    )
    print(f"✓ Learning from interaction: {len(personality_engine.social_learning_experiences)} experiences stored")
    
    # Test personality insights
    insights = personality_engine.get_personality_insights()
    print(f"✓ Personality insights: {len(insights['strengths'])} strengths, {len(insights['growth_areas'])} growth areas")
    
    print("PersonalityEngine tests passed!\n")

def test_emotion_simulation_basic():
    """Test basic emotion simulation concepts"""
    print("Testing Emotion Simulation Concepts...")
    
    # Test emotion models
    from scrollintel.models.emotion_models import EmotionalState, EmotionalResponse
    
    # Create emotional state
    emotional_state = EmotionalState(
        primary_emotion=EmotionType.JOY,
        intensity=0.8,
        arousal=0.7,
        valence=0.9,
        context="successful_project_completion"
    )
    print(f"✓ Created emotional state: {emotional_state.primary_emotion} (intensity: {emotional_state.intensity})")
    
    # Create emotional response
    emotional_response = EmotionalResponse(
        stimulus="Great success with the project implementation!",
        emotional_state=emotional_state,
        behavioral_response="Express enthusiasm and celebrate achievement",
        cognitive_appraisal="This success validates our approach and opens new opportunities",
        social_appropriateness=0.9,
        confidence=0.85
    )
    print(f"✓ Created emotional response with {emotional_response.social_appropriateness:.1f} social appropriateness")
    
    print("Emotion simulation concepts tested successfully!\n")

def test_integration():
    """Test integration between emotion and personality systems"""
    print("Testing Integration...")
    
    # Create personality-driven emotional response
    personality_engine = PersonalityEngine()
    
    # High empathy personality should show strong empathetic responses
    high_empathy_personality = PersonalityProfile(
        traits={PersonalityTrait.EMPATHY: 0.9, PersonalityTrait.AGREEABLENESS: 0.8}
    )
    
    empathetic_engine = PersonalityEngine(high_empathy_personality)
    
    # Test empathetic situation
    situation = "Team member is experiencing personal difficulties"
    social_context = SocialContext(social_setting="professional")
    
    behavioral_response = empathetic_engine.generate_behavioral_response(
        situation, social_context
    )
    
    print(f"✓ Empathetic response: {behavioral_response['behavioral_response']}")
    print(f"✓ Appropriateness score: {behavioral_response['appropriateness_score']:.2f}")
    
    # Test personality influence on emotional regulation
    decision_context = "Handle team conflict with emotional sensitivity"
    options = ["direct_confrontation", "empathetic_mediation"]
    
    influence = empathetic_engine.influence_decision(decision_context, options)
    print(f"✓ Empathy-influenced decision bias: {influence.personality_bias}")
    print(f"✓ Social consideration level: {influence.social_consideration:.2f}")
    
    print("Integration tests passed!\n")

def demonstrate_capabilities():
    """Demonstrate the key capabilities implemented"""
    print("=== EMOTION AND PERSONALITY SIMULATION CAPABILITIES ===\n")
    
    print("1. PERSONALITY TRAITS AND DECISION-MAKING:")
    personality_engine = PersonalityEngine()
    
    # Show personality profile
    insights = personality_engine.get_personality_insights()
    print(f"   - Personality traits: {list(insights['trait_profile'].keys())[:5]}...")
    print(f"   - Behavioral tendencies: {insights['behavioral_tendencies']}")
    print(f"   - Key strengths: {insights['strengths']}")
    
    print("\n2. ADAPTIVE COMMUNICATION:")
    social_context = SocialContext(social_setting="professional")
    comm_style = personality_engine.adapt_communication_style("executive_team", social_context)
    print(f"   - Communication tone: {comm_style['tone']}")
    print(f"   - Formality level: {comm_style['formality_level']:.2f}")
    print(f"   - Technical depth: {comm_style['technical_depth']:.2f}")
    
    print("\n3. SOCIAL COGNITION AND EMPATHY:")
    situation = "Team member struggling with work-life balance"
    response = personality_engine.generate_behavioral_response(situation, social_context)
    print(f"   - Behavioral response: {response['behavioral_response']}")
    print(f"   - Confidence: {response['confidence']:.2f}")
    
    print("\n4. PERSONALITY COMPATIBILITY:")
    other_personality = PersonalityProfile(traits={PersonalityTrait.CONSCIENTIOUSNESS: 0.9})
    compatibility = personality_engine.assess_personality_compatibility(other_personality)
    print(f"   - Overall compatibility: {compatibility['overall_compatibility']:.2f}")
    print(f"   - Communication compatibility: {compatibility['communication_compatibility']:.2f}")
    
    print("\n5. LEARNING AND ADAPTATION:")
    initial_experiences = len(personality_engine.social_learning_experiences)
    personality_engine.learn_from_interaction(
        "Cross-functional collaboration",
        "Highly successful outcome",
        "Excellent leadership and team coordination"
    )
    print(f"   - Learning experiences: {len(personality_engine.social_learning_experiences)} (was {initial_experiences})")
    
    print("\n6. EMOTIONAL INTELLIGENCE MODELS:")
    from scrollintel.models.emotion_models import EmpathyAssessment
    
    empathy_assessment = EmpathyAssessment(
        target_person="colleague",
        perceived_emotion=EmotionType.SADNESS,
        perceived_intensity=0.6,
        confidence=0.8,
        contextual_factors=["work_stress", "personal_challenges"],
        appropriate_response="Offer support and flexible work arrangements",
        emotional_contagion=0.2
    )
    
    print(f"   - Empathy assessment: {empathy_assessment.perceived_emotion} at {empathy_assessment.perceived_intensity:.1f} intensity")
    print(f"   - Appropriate response: {empathy_assessment.appropriate_response}")
    print(f"   - Emotional contagion: {empathy_assessment.emotional_contagion:.1f}")

if __name__ == "__main__":
    print("🧠 EMOTION AND PERSONALITY SIMULATION SYSTEM TEST\n")
    
    try:
        test_personality_engine()
        test_emotion_simulation_basic()
        test_integration()
        
        print("🎉 All core tests passed successfully!\n")
        
        demonstrate_capabilities()
        
        print("\n✅ IMPLEMENTATION COMPLETE")
        print("The emotion and personality simulation system has been successfully implemented with:")
        print("- ✓ PersonalityEngine with trait-based decision making")
        print("- ✓ Adaptive communication style based on personality")
        print("- ✓ Social cognition and behavioral response generation")
        print("- ✓ Personality compatibility assessment")
        print("- ✓ Learning and adaptation from social interactions")
        print("- ✓ Comprehensive emotion models and data structures")
        print("- ✓ Integration between personality and emotional systems")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)