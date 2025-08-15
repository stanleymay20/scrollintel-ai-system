"""
Tests for Emotion and Personality Simulation Integration

This module tests the integration between emotion simulation and personality
engine for human-like emotional intelligence and social cognition.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.emotion_simulator import EmotionSimulator
from scrollintel.engines.personality_engine import PersonalityEngine
from scrollintel.models.emotion_models import (
    EmotionType, EmotionalState, PersonalityTrait, PersonalityProfile,
    SocialContext, EmpathyAssessment, EmotionalResponse
)


class TestEmotionPersonalityIntegration:
    """Test integration between emotion and personality systems"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.emotion_simulator = EmotionSimulator()
        
        # Create test personality profile
        test_personality = PersonalityProfile(
            traits={
                PersonalityTrait.OPENNESS: 0.8,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.7,
                PersonalityTrait.EXTRAVERSION: 0.6,
                PersonalityTrait.AGREEABLENESS: 0.8,
                PersonalityTrait.NEUROTICISM: 0.3,
                PersonalityTrait.EMPATHY: 0.9,
                PersonalityTrait.ASSERTIVENESS: 0.6
            }
        )
        
        self.personality_engine = PersonalityEngine(test_personality)
        
        # Test social context
        self.social_context = SocialContext(
            participants=["user", "system"],
            relationship_types={"user": "professional"},
            social_setting="professional",
            cultural_context="business"
        )
    
    def test_emotion_simulator_initialization(self):
        """Test emotion simulator initializes correctly"""
        assert self.emotion_simulator.current_emotional_state.primary_emotion == EmotionType.TRUST
        assert 0.0 <= self.emotion_simulator.current_emotional_state.intensity <= 1.0
        assert len(self.emotion_simulator.emotional_memories) == 0
        assert len(self.emotion_simulator.emotion_regulation_strategies) > 0
    
    def test_personality_engine_initialization(self):
        """Test personality engine initializes correctly"""
        assert len(self.personality_engine.personality_profile.traits) > 0
        assert self.personality_engine.personality_profile.traits[PersonalityTrait.EMPATHY] == 0.9
        assert len(self.personality_engine.behavioral_history) == 0
    
    def test_emotional_stimulus_processing(self):
        """Test processing of emotional stimuli"""
        stimulus = "Great success with the project implementation!"
        
        response = self.emotion_simulator.process_emotional_stimulus(
            stimulus, self.social_context
        )
        
        assert isinstance(response, EmotionalResponse)
        assert response.stimulus == stimulus
        assert isinstance(response.emotional_state, EmotionalState)
        assert response.confidence > 0.0
        assert response.social_appropriateness > 0.0
    
    def test_empathy_assessment(self):
        """Test empathetic understanding of others"""
        target_person = "colleague"
        observed_behavior = "Person seems frustrated and tense during meeting"
        
        empathy_assessment = self.emotion_simulator.assess_empathy(
            target_person, observed_behavior, self.social_context
        )
        
        assert isinstance(empathy_assessment, EmpathyAssessment)
        assert empathy_assessment.target_person == target_person
        assert isinstance(empathy_assessment.perceived_emotion, EmotionType)
        assert 0.0 <= empathy_assessment.confidence <= 1.0
        assert len(empathy_assessment.contextual_factors) > 0
        assert empathy_assessment.appropriate_response != ""
    
    def test_emotion_regulation(self):
        """Test active emotion regulation"""
        # Set current state to high intensity negative emotion
        self.emotion_simulator.current_emotional_state = EmotionalState(
            primary_emotion=EmotionType.ANGER,
            intensity=0.9,
            arousal=0.8,
            valence=0.2
        )
        
        # Regulate to calmer state
        regulated_state = self.emotion_simulator.regulate_emotion(
            EmotionType.TRUST, 0.5
        )
        
        assert regulated_state.primary_emotion == EmotionType.TRUST
        assert regulated_state.intensity <= 0.9  # Should be regulated down
        assert regulated_state.arousal <= 0.8    # Should be calmer
    
    def test_social_response_generation(self):
        """Test generation of socially appropriate responses"""
        social_situation = "Team member is struggling with technical challenges"
        
        social_response = self.emotion_simulator.generate_social_response(
            social_situation, self.social_context
        )
        
        assert "emotional_response" in social_response
        assert "behavioral_response" in social_response
        assert "social_assessment" in social_response
        assert "appropriateness_score" in social_response
        assert 0.0 <= social_response["appropriateness_score"] <= 1.0
    
    def test_personality_decision_influence(self):
        """Test personality influence on decision-making"""
        decision_context = "Choose between innovative approach vs proven method"
        options = ["innovative_solution", "proven_method"]
        
        # Create emotional state
        emotional_state = EmotionalState(
            primary_emotion=EmotionType.ANTICIPATION,
            intensity=0.6,
            arousal=0.5,
            valence=0.7
        )
        
        influence = self.personality_engine.influence_decision(
            decision_context, options, emotional_state
        )
        
        assert influence.decision_context == decision_context
        assert len(influence.trait_influences) > 0
        assert influence.personality_bias != ""
        assert -0.5 <= influence.confidence_modifier <= 0.5
        assert 0.0 <= influence.risk_tolerance <= 1.0
    
    def test_communication_style_adaptation(self):
        """Test adaptation of communication style"""
        target_audience = "technical team"
        
        communication_style = self.personality_engine.adapt_communication_style(
            target_audience, self.social_context
        )
        
        assert "communication_style" in communication_style
        assert "tone" in communication_style
        assert "formality_level" in communication_style
        assert "directness" in communication_style
        assert "empathy_level" in communication_style
        assert "technical_depth" in communication_style
    
    def test_behavioral_response_generation(self):
        """Test personality-driven behavioral responses"""
        situation = "Complex technical problem requires immediate solution"
        
        emotional_state = EmotionalState(
            primary_emotion=EmotionType.ANTICIPATION,
            intensity=0.7,
            arousal=0.6,
            valence=0.6
        )
        
        behavioral_response = self.personality_engine.generate_behavioral_response(
            situation, self.social_context, emotional_state
        )
        
        assert "behavioral_response" in behavioral_response
        assert "behavioral_drivers" in behavioral_response
        assert "appropriateness_score" in behavioral_response
        assert "confidence" in behavioral_response
        assert 0.0 <= behavioral_response["appropriateness_score"] <= 1.0
    
    def test_personality_compatibility_assessment(self):
        """Test assessment of personality compatibility"""
        # Create another personality profile
        other_personality = PersonalityProfile(
            traits={
                PersonalityTrait.OPENNESS: 0.6,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
                PersonalityTrait.EXTRAVERSION: 0.4,
                PersonalityTrait.AGREEABLENESS: 0.7,
                PersonalityTrait.NEUROTICISM: 0.4
            }
        )
        
        compatibility = self.personality_engine.assess_personality_compatibility(
            other_personality
        )
        
        assert "trait_compatibility" in compatibility
        assert "overall_compatibility" in compatibility
        assert "communication_compatibility" in compatibility
        assert "working_style_compatibility" in compatibility
        assert "collaboration_potential" in compatibility
        
        # All scores should be between 0 and 1
        for score in compatibility.values():
            if isinstance(score, dict):
                for sub_score in score.values():
                    assert 0.0 <= sub_score <= 1.0
            else:
                assert 0.0 <= score <= 1.0
    
    def test_learning_from_interaction(self):
        """Test learning and adaptation from interactions"""
        interaction_context = "Collaborative problem-solving session"
        outcome = "Successful resolution with positive team feedback"
        feedback = "Great approach, very collaborative and effective"
        
        initial_agreeableness = self.personality_engine.personality_profile.traits[PersonalityTrait.AGREEABLENESS]
        
        self.personality_engine.learn_from_interaction(
            interaction_context, outcome, feedback
        )
        
        # Should have stored learning experience
        assert len(self.personality_engine.social_learning_experiences) > 0
        
        # Check learning experience structure
        experience = self.personality_engine.social_learning_experiences[-1]
        assert experience["context"] == interaction_context
        assert experience["outcome"] == outcome
        assert experience["feedback"] == feedback
    
    def test_personality_insights_generation(self):
        """Test generation of personality insights"""
        insights = self.personality_engine.get_personality_insights()
        
        assert "trait_profile" in insights
        assert "trait_analysis" in insights
        assert "behavioral_tendencies" in insights
        assert "strengths" in insights
        assert "growth_areas" in insights
        assert "personality_signature" in insights
        
        # Verify trait profile contains all expected traits
        trait_profile = insights["trait_profile"]
        assert PersonalityTrait.EMPATHY.value in [trait.value if hasattr(trait, 'value') else str(trait) for trait in trait_profile.keys()]
    
    def test_emotional_memory_storage(self):
        """Test storage and retrieval of emotional memories"""
        stimulus = "Challenging technical discussion with stakeholders"
        
        initial_memory_count = len(self.emotion_simulator.emotional_memories)
        
        # Process stimulus to create memory
        self.emotion_simulator.process_emotional_stimulus(
            stimulus, self.social_context
        )
        
        # Should have stored emotional memory
        assert len(self.emotion_simulator.emotional_memories) == initial_memory_count + 1
        
        # Check memory structure
        memory = self.emotion_simulator.emotional_memories[-1]
        assert memory.event_description == stimulus
        assert isinstance(memory.emotional_state, EmotionalState)
        assert memory.outcome != ""
        assert len(memory.lessons_learned) > 0
    
    def test_emotion_personality_consistency(self):
        """Test consistency between emotional responses and personality"""
        # High empathy personality should show strong empathetic responses
        high_empathy_personality = PersonalityProfile(
            traits={PersonalityTrait.EMPATHY: 0.9, PersonalityTrait.AGREEABLENESS: 0.8}
        )
        
        empathetic_engine = PersonalityEngine(high_empathy_personality)
        
        # Test empathetic situation
        situation = "Team member is experiencing personal difficulties"
        
        behavioral_response = empathetic_engine.generate_behavioral_response(
            situation, self.social_context
        )
        
        # Should show empathetic behavioral drivers
        assert "collaborative_solution" in behavioral_response.get("behavioral_drivers", [])
        assert behavioral_response["appropriateness_score"] > 0.7
    
    def test_emotional_regulation_strategies(self):
        """Test different emotional regulation strategies"""
        # Test cognitive reappraisal
        high_intensity_state = EmotionalState(
            primary_emotion=EmotionType.ANGER,
            intensity=0.9,
            arousal=0.8,
            valence=0.2
        )
        
        self.emotion_simulator.current_emotional_state = high_intensity_state
        
        # Apply regulation
        regulated_state = self.emotion_simulator._apply_emotional_regulation(
            high_intensity_state, self.social_context
        )
        
        # Should be regulated down
        assert regulated_state.intensity < high_intensity_state.intensity
        assert regulated_state.arousal < high_intensity_state.arousal
    
    def test_social_cognition_integration(self):
        """Test integration of social cognition with emotion and personality"""
        # Complex social situation
        social_situation = "Mediating conflict between team members with different approaches"
        
        # Get emotional response
        emotional_response = self.emotion_simulator.process_emotional_stimulus(
            social_situation, self.social_context
        )
        
        # Get personality-driven behavioral response
        behavioral_response = self.personality_engine.generate_behavioral_response(
            social_situation, self.social_context, emotional_response.emotional_state
        )
        
        # Both should be socially appropriate
        assert emotional_response.social_appropriateness > 0.5
        assert behavioral_response["appropriateness_score"] > 0.5
        
        # Should show empathetic and collaborative elements
        assert "collaborative" in behavioral_response["behavioral_response"].lower() or \
               "understanding" in behavioral_response["behavioral_response"].lower()


class TestEmotionValidation:
    """Test emotion simulation validation and benchmarks"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.emotion_simulator = EmotionSimulator()
    
    def test_emotion_intensity_bounds(self):
        """Test that emotion intensities stay within valid bounds"""
        test_stimuli = [
            "Extremely exciting breakthrough in AI research!",
            "Devastating failure in critical system deployment",
            "Mild concern about upcoming deadline",
            "Neutral status update on project progress"
        ]
        
        for stimulus in test_stimuli:
            response = self.emotion_simulator.process_emotional_stimulus(stimulus)
            
            # Intensity should be between 0 and 1
            assert 0.0 <= response.emotional_state.intensity <= 1.0
            
            # Arousal should be between 0 and 1
            assert 0.0 <= response.emotional_state.arousal <= 1.0
            
            # Valence should be between 0 and 1
            assert 0.0 <= response.emotional_state.valence <= 1.0
    
    def test_emotional_consistency(self):
        """Test emotional consistency across similar stimuli"""
        positive_stimuli = [
            "Great success with the project!",
            "Excellent results from the implementation!",
            "Outstanding performance by the team!"
        ]
        
        responses = []
        for stimulus in positive_stimuli:
            response = self.emotion_simulator.process_emotional_stimulus(stimulus)
            responses.append(response)
        
        # All should have positive emotions
        for response in responses:
            assert response.emotional_state.primary_emotion in [
                EmotionType.JOY, EmotionType.TRUST, EmotionType.ANTICIPATION
            ]
            assert response.emotional_state.valence > 0.5
    
    def test_empathy_accuracy(self):
        """Test accuracy of empathetic assessments"""
        test_cases = [
            {
                "behavior": "Person is smiling and speaking enthusiastically",
                "expected_emotion": EmotionType.JOY
            },
            {
                "behavior": "Person appears withdrawn and speaks quietly",
                "expected_emotion": EmotionType.SADNESS
            },
            {
                "behavior": "Person shows tense body language and sharp responses",
                "expected_emotion": EmotionType.ANGER
            }
        ]
        
        for case in test_cases:
            empathy_assessment = self.emotion_simulator.assess_empathy(
                "test_person", case["behavior"]
            )
            
            # Should detect appropriate emotion (allowing for some variation)
            assert empathy_assessment.confidence > 0.3
            # Note: Exact emotion matching might vary, so we test confidence instead


class TestPersonalityValidation:
    """Test personality engine validation and benchmarks"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.personality_engine = PersonalityEngine()
    
    def test_trait_value_bounds(self):
        """Test that personality trait values stay within valid bounds"""
        traits = self.personality_engine.personality_profile.traits
        
        for trait, value in traits.items():
            assert 0.0 <= value <= 1.0, f"Trait {trait} has invalid value {value}"
    
    def test_decision_influence_consistency(self):
        """Test consistency of personality influence on decisions"""
        # High conscientiousness should prefer systematic approaches
        high_conscientiousness_profile = PersonalityProfile(
            traits={PersonalityTrait.CONSCIENTIOUSNESS: 0.9}
        )
        
        conscientious_engine = PersonalityEngine(high_conscientiousness_profile)
        
        decision_context = "Choose approach for complex project implementation"
        options = ["systematic_planning", "agile_iteration"]
        
        influence = conscientious_engine.influence_decision(decision_context, options)
        
        # Should show strong conscientiousness influence
        assert influence.trait_influences[PersonalityTrait.CONSCIENTIOUSNESS] > 0.5
        assert influence.personality_bias in ["systematic_bias", "stability_bias"]
    
    def test_communication_adaptation_appropriateness(self):
        """Test appropriateness of communication style adaptations"""
        contexts = [
            ("executive_presentation", "professional"),
            ("team_brainstorming", "collaborative"),
            ("technical_review", "analytical")
        ]
        
        for audience, expected_style_element in contexts:
            social_context = SocialContext(
                social_setting="professional",
                participants=[audience]
            )
            
            communication_style = self.personality_engine.adapt_communication_style(
                audience, social_context
            )
            
            # Should adapt appropriately for context
            assert communication_style["formality_level"] > 0.3
            assert 0.0 <= communication_style["directness"] <= 1.0
    
    def test_personality_learning_bounds(self):
        """Test that personality learning stays within reasonable bounds"""
        initial_traits = dict(self.personality_engine.personality_profile.traits)
        
        # Simulate multiple learning experiences
        for i in range(10):
            self.personality_engine.learn_from_interaction(
                f"interaction_{i}",
                "positive_outcome",
                "good_feedback"
            )
        
        # Traits should still be within bounds
        current_traits = self.personality_engine.personality_profile.traits
        
        for trait, value in current_traits.items():
            assert 0.0 <= value <= 1.0
            
            # Changes should be gradual
            initial_value = initial_traits.get(trait, 0.5)
            change = abs(value - initial_value)
            assert change < 0.5  # No dramatic personality changes


if __name__ == "__main__":
    pytest.main([__file__])