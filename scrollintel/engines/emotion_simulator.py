"""
Emotion Simulator Engine for AGI Cognitive Architecture

This module implements emotional intelligence, emotional regulation,
and appropriate emotional responses for human-like interaction.
"""

import random
import math
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from scrollintel.models.emotion_models import (
    EmotionType, EmotionalState, EmotionalResponse, EmotionalMemory,
    SocialContext, EmpathyAssessment
)


class EmotionSimulator:
    """
    Simulates human-like emotional responses and emotional intelligence
    """
    
    def __init__(self):
        self.current_emotional_state = EmotionalState(
            primary_emotion=EmotionType.TRUST,
            intensity=0.3,
            arousal=0.4,
            valence=0.7
        )
        self.emotional_memories: List[EmotionalMemory] = []
        self.emotion_regulation_strategies = {
            "cognitive_reappraisal": 0.8,
            "suppression": 0.3,
            "distraction": 0.6,
            "mindfulness": 0.7,
            "problem_solving": 0.9
        }
        self.baseline_emotions = {
            EmotionType.JOY: 0.4,
            EmotionType.TRUST: 0.6,
            EmotionType.ANTICIPATION: 0.5,
            EmotionType.SURPRISE: 0.2,
            EmotionType.FEAR: 0.1,
            EmotionType.ANGER: 0.1,
            EmotionType.SADNESS: 0.1,
            EmotionType.DISGUST: 0.1
        }
    
    def process_emotional_stimulus(self, stimulus: str, context: Optional[SocialContext] = None) -> EmotionalResponse:
        """
        Process an emotional stimulus and generate appropriate emotional response
        """
        # Analyze stimulus for emotional content
        emotion_triggers = self._analyze_stimulus(stimulus)
        
        # Generate emotional state based on stimulus and context
        new_emotional_state = self._generate_emotional_state(
            emotion_triggers, context, stimulus
        )
        
        # Apply emotional regulation if needed
        regulated_state = self._apply_emotional_regulation(new_emotional_state, context)
        
        # Generate behavioral response
        behavioral_response = self._generate_behavioral_response(
            regulated_state, context, stimulus
        )
        
        # Assess social appropriateness
        social_appropriateness = self._assess_social_appropriateness(
            regulated_state, behavioral_response, context
        )
        
        # Create cognitive appraisal
        cognitive_appraisal = self._generate_cognitive_appraisal(
            stimulus, regulated_state, context
        )
        
        # Update current emotional state
        self.current_emotional_state = regulated_state
        
        # Store emotional memory
        self._store_emotional_memory(stimulus, regulated_state, behavioral_response)
        
        return EmotionalResponse(
            stimulus=stimulus,
            emotional_state=regulated_state,
            behavioral_response=behavioral_response,
            cognitive_appraisal=cognitive_appraisal,
            social_appropriateness=social_appropriateness,
            regulation_strategy=self._select_regulation_strategy(regulated_state),
            confidence=0.85
        )
    
    def assess_empathy(self, target_person: str, observed_behavior: str, 
                      context: Optional[SocialContext] = None) -> EmpathyAssessment:
        """
        Assess and understand another person's emotional state through empathy
        """
        # Analyze observed behavior for emotional cues
        emotional_cues = self._extract_emotional_cues(observed_behavior)
        
        # Infer emotional state based on cues and context
        perceived_emotion, perceived_intensity = self._infer_emotional_state(
            emotional_cues, context
        )
        
        # Assess confidence in empathetic understanding
        confidence = self._calculate_empathy_confidence(
            emotional_cues, context, observed_behavior
        )
        
        # Determine contextual factors affecting the person
        contextual_factors = self._identify_contextual_factors(
            target_person, context, observed_behavior
        )
        
        # Generate appropriate empathetic response
        appropriate_response = self._generate_empathetic_response(
            perceived_emotion, perceived_intensity, contextual_factors
        )
        
        # Calculate emotional contagion effect
        emotional_contagion = self._calculate_emotional_contagion(
            perceived_emotion, perceived_intensity
        )
        
        return EmpathyAssessment(
            target_person=target_person,
            perceived_emotion=perceived_emotion,
            perceived_intensity=perceived_intensity,
            confidence=confidence,
            contextual_factors=contextual_factors,
            appropriate_response=appropriate_response,
            emotional_contagion=emotional_contagion
        )
    
    def regulate_emotion(self, target_emotion: EmotionType, 
                        target_intensity: float = 0.5) -> EmotionalState:
        """
        Actively regulate emotions to achieve desired emotional state
        """
        current_emotion = self.current_emotional_state.primary_emotion
        current_intensity = self.current_emotional_state.intensity
        
        # Select appropriate regulation strategy
        strategy = self._select_optimal_regulation_strategy(
            current_emotion, target_emotion, current_intensity, target_intensity
        )
        
        # Apply regulation technique
        regulated_state = self._apply_regulation_technique(
            strategy, target_emotion, target_intensity
        )
        
        # Update current state
        self.current_emotional_state = regulated_state
        
        return regulated_state
    
    def generate_social_response(self, social_situation: str, 
                               context: SocialContext) -> Dict[str, Any]:
        """
        Generate socially appropriate emotional and behavioral responses
        """
        # Assess social situation
        social_assessment = self._assess_social_situation(social_situation, context)
        
        # Determine appropriate emotional response
        appropriate_emotion = self._determine_social_emotion(
            social_assessment, context
        )
        
        # Generate behavioral response
        behavioral_response = self._generate_social_behavior(
            appropriate_emotion, social_assessment, context
        )
        
        # Assess social appropriateness
        appropriateness_score = self._assess_social_appropriateness(
            appropriate_emotion, behavioral_response, context
        )
        
        return {
            "emotional_response": appropriate_emotion,
            "behavioral_response": behavioral_response,
            "social_assessment": social_assessment,
            "appropriateness_score": appropriateness_score,
            "confidence": 0.85
        } 
   
    def _analyze_stimulus(self, stimulus: str) -> Dict[str, float]:
        """Analyze stimulus for emotional triggers"""
        emotion_triggers = {}
        
        # Simple keyword-based emotion detection (can be enhanced with NLP)
        positive_words = ["success", "achievement", "joy", "happy", "excellent", "great"]
        negative_words = ["failure", "problem", "error", "sad", "angry", "frustrated"]
        fear_words = ["danger", "risk", "threat", "uncertain", "worried"]
        
        stimulus_lower = stimulus.lower()
        
        for word in positive_words:
            if word in stimulus_lower:
                emotion_triggers[EmotionType.JOY.value] = emotion_triggers.get(EmotionType.JOY.value, 0) + 0.3
        
        for word in negative_words:
            if word in stimulus_lower:
                emotion_triggers[EmotionType.SADNESS.value] = emotion_triggers.get(EmotionType.SADNESS.value, 0) + 0.3
        
        for word in fear_words:
            if word in stimulus_lower:
                emotion_triggers[EmotionType.FEAR.value] = emotion_triggers.get(EmotionType.FEAR.value, 0) + 0.3
        
        return emotion_triggers
    
    def _generate_emotional_state(self, emotion_triggers: Dict[str, float], 
                                 context: Optional[SocialContext], 
                                 stimulus: str) -> EmotionalState:
        """Generate new emotional state based on triggers and context"""
        if not emotion_triggers:
            # Default to current state with slight decay
            return EmotionalState(
                primary_emotion=self.current_emotional_state.primary_emotion,
                intensity=max(0.1, self.current_emotional_state.intensity * 0.9),
                arousal=self.current_emotional_state.arousal * 0.95,
                valence=self.current_emotional_state.valence,
                context=stimulus,
                triggers=[stimulus]
            )
        
        # Find dominant emotion
        dominant_emotion_str = max(emotion_triggers.keys(), key=emotion_triggers.get)
        dominant_emotion = EmotionType(dominant_emotion_str)
        intensity = min(1.0, emotion_triggers[dominant_emotion_str])
        
        # Calculate arousal and valence
        arousal = self._calculate_arousal(dominant_emotion, intensity)
        valence = self._calculate_valence(dominant_emotion, intensity)
        
        # Create secondary emotions
        secondary_emotions = {}
        for emotion_str, trigger_strength in emotion_triggers.items():
            if emotion_str != dominant_emotion_str and trigger_strength > 0.1:
                secondary_emotions[EmotionType(emotion_str)] = trigger_strength
        
        return EmotionalState(
            primary_emotion=dominant_emotion,
            intensity=intensity,
            secondary_emotions=secondary_emotions,
            arousal=arousal,
            valence=valence,
            context=stimulus,
            triggers=list(emotion_triggers.keys())
        )
    
    def _apply_emotional_regulation(self, emotional_state: EmotionalState, 
                                   context: Optional[SocialContext]) -> EmotionalState:
        """Apply emotional regulation strategies"""
        # Check if regulation is needed
        if emotional_state.intensity < 0.7 and emotional_state.arousal < 0.8:
            return emotional_state
        
        # Select regulation strategy
        strategy = self._select_regulation_strategy(emotional_state)
        
        # Apply regulation
        regulated_intensity = emotional_state.intensity * self.emotion_regulation_strategies.get(strategy, 0.8)
        regulated_arousal = emotional_state.arousal * 0.8
        
        return EmotionalState(
            primary_emotion=emotional_state.primary_emotion,
            intensity=regulated_intensity,
            secondary_emotions=emotional_state.secondary_emotions,
            arousal=regulated_arousal,
            valence=emotional_state.valence,
            context=emotional_state.context,
            triggers=emotional_state.triggers
        )
    
    def _generate_behavioral_response(self, emotional_state: EmotionalState, 
                                    context: Optional[SocialContext], 
                                    stimulus: str) -> str:
        """Generate appropriate behavioral response"""
        emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        
        if emotion == EmotionType.JOY:
            if intensity > 0.7:
                return "Express enthusiasm and positive engagement"
            else:
                return "Show mild satisfaction and encouragement"
        elif emotion == EmotionType.SADNESS:
            if intensity > 0.7:
                return "Express concern and offer support"
            else:
                return "Show understanding and empathy"
        elif emotion == EmotionType.ANGER:
            if intensity > 0.7:
                return "Express firm disagreement while maintaining professionalism"
            else:
                return "Show mild frustration but remain constructive"
        elif emotion == EmotionType.FEAR:
            if intensity > 0.7:
                return "Express caution and request more information"
            else:
                return "Show mild concern and seek clarification"
        else:
            return "Maintain neutral, professional demeanor"
    
    def _assess_social_appropriateness(self, emotional_state: EmotionalState, 
                                     behavioral_response: str, 
                                     context: Optional[SocialContext]) -> float:
        """Assess social appropriateness of emotional response"""
        base_appropriateness = 0.8
        
        # Adjust based on context
        if context and context.social_setting == "professional":
            if emotional_state.intensity > 0.8:
                base_appropriateness -= 0.2
            if emotional_state.primary_emotion in [EmotionType.ANGER, EmotionType.DISGUST]:
                base_appropriateness -= 0.3
        
        return max(0.0, min(1.0, base_appropriateness))
    
    def _generate_cognitive_appraisal(self, stimulus: str, emotional_state: EmotionalState, 
                                    context: Optional[SocialContext]) -> str:
        """Generate cognitive appraisal of the situation"""
        emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        
        if emotion == EmotionType.JOY:
            return f"This situation presents positive opportunities with high confidence (intensity: {intensity:.2f})"
        elif emotion == EmotionType.FEAR:
            return f"This situation requires careful analysis due to potential risks (intensity: {intensity:.2f})"
        elif emotion == EmotionType.ANGER:
            return f"This situation involves challenges that need to be addressed constructively (intensity: {intensity:.2f})"
        else:
            return f"This situation requires balanced consideration of multiple factors (intensity: {intensity:.2f})"
    
    def _select_regulation_strategy(self, emotional_state: EmotionalState) -> str:
        """Select appropriate emotion regulation strategy"""
        if emotional_state.intensity > 0.8:
            return "cognitive_reappraisal"
        elif emotional_state.arousal > 0.7:
            return "mindfulness"
        else:
            return "problem_solving"
    
    def _store_emotional_memory(self, stimulus: str, emotional_state: EmotionalState, 
                               behavioral_response: str):
        """Store emotional memory for future reference"""
        memory = EmotionalMemory(
            event_description=stimulus,
            emotional_state=emotional_state,
            outcome=behavioral_response,
            emotional_significance=emotional_state.intensity,
            lessons_learned=[f"Responded to {emotional_state.primary_emotion.value} with {behavioral_response}"]
        )
        
        self.emotional_memories.append(memory)
        
        # Keep only recent memories (last 100)
        if len(self.emotional_memories) > 100:
            self.emotional_memories = self.emotional_memories[-100:]
    
    def _extract_emotional_cues(self, observed_behavior: str) -> List[str]:
        """Extract emotional cues from observed behavior"""
        cues = []
        behavior_lower = observed_behavior.lower()
        
        if any(word in behavior_lower for word in ["smile", "laugh", "excited"]):
            cues.append("positive_expression")
        if any(word in behavior_lower for word in ["frown", "cry", "sad"]):
            cues.append("negative_expression")
        if any(word in behavior_lower for word in ["tense", "rigid", "angry"]):
            cues.append("tension")
        if any(word in behavior_lower for word in ["withdrawn", "quiet", "distant"]):
            cues.append("withdrawal")
        
        return cues
    
    def _infer_emotional_state(self, emotional_cues: List[str], 
                              context: Optional[SocialContext]) -> Tuple[EmotionType, float]:
        """Infer emotional state from cues and context"""
        if "positive_expression" in emotional_cues:
            return EmotionType.JOY, 0.7
        elif "negative_expression" in emotional_cues:
            return EmotionType.SADNESS, 0.6
        elif "tension" in emotional_cues:
            return EmotionType.ANGER, 0.5
        elif "withdrawal" in emotional_cues:
            return EmotionType.FEAR, 0.4
        else:
            return EmotionType.TRUST, 0.3
    
    def _calculate_empathy_confidence(self, emotional_cues: List[str], 
                                    context: Optional[SocialContext], 
                                    observed_behavior: str) -> float:
        """Calculate confidence in empathetic assessment"""
        base_confidence = 0.6
        
        # More cues = higher confidence
        base_confidence += len(emotional_cues) * 0.1
        
        # Context helps confidence
        if context:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _identify_contextual_factors(self, target_person: str, 
                                   context: Optional[SocialContext], 
                                   observed_behavior: str) -> List[str]:
        """Identify contextual factors affecting the person"""
        factors = []
        
        if context:
            if context.social_setting == "professional":
                factors.append("work_environment")
            if target_person in context.power_dynamics:
                factors.append("power_dynamics")
        
        factors.append("behavioral_observation")
        return factors
    
    def _generate_empathetic_response(self, perceived_emotion: EmotionType, 
                                    perceived_intensity: float, 
                                    contextual_factors: List[str]) -> str:
        """Generate appropriate empathetic response"""
        if perceived_emotion == EmotionType.JOY:
            return "Share in their positive experience and offer encouragement"
        elif perceived_emotion == EmotionType.SADNESS:
            return "Offer support and understanding without being intrusive"
        elif perceived_emotion == EmotionType.ANGER:
            return "Acknowledge their frustration and help find constructive solutions"
        elif perceived_emotion == EmotionType.FEAR:
            return "Provide reassurance and help address their concerns"
        else:
            return "Show understanding and offer appropriate support"
    
    def _calculate_emotional_contagion(self, perceived_emotion: EmotionType, 
                                     perceived_intensity: float) -> float:
        """Calculate how much the perceived emotion affects the system"""
        # Empathetic systems are affected by others' emotions
        base_contagion = perceived_intensity * 0.3
        
        # Some emotions are more contagious
        if perceived_emotion in [EmotionType.JOY, EmotionType.FEAR]:
            base_contagion *= 1.2
        
        return min(1.0, base_contagion)
    
    def _select_optimal_regulation_strategy(self, current_emotion: EmotionType, 
                                          target_emotion: EmotionType, 
                                          current_intensity: float, 
                                          target_intensity: float) -> str:
        """Select optimal regulation strategy for emotion change"""
        intensity_diff = abs(current_intensity - target_intensity)
        
        if intensity_diff > 0.5:
            return "cognitive_reappraisal"
        elif current_emotion != target_emotion:
            return "problem_solving"
        else:
            return "mindfulness"
    
    def _apply_regulation_technique(self, strategy: str, target_emotion: EmotionType, 
                                  target_intensity: float) -> EmotionalState:
        """Apply specific regulation technique"""
        effectiveness = self.emotion_regulation_strategies.get(strategy, 0.7)
        
        # Move toward target emotion and intensity
        new_intensity = (self.current_emotional_state.intensity + target_intensity * effectiveness) / 2
        
        return EmotionalState(
            primary_emotion=target_emotion,
            intensity=new_intensity,
            arousal=self.current_emotional_state.arousal * 0.9,
            valence=self._calculate_valence(target_emotion, new_intensity),
            context="emotion_regulation"
        )
    
    def _assess_social_situation(self, social_situation: str, 
                               context: SocialContext) -> Dict[str, Any]:
        """Assess social situation for appropriate response"""
        return {
            "situation_type": "professional_interaction",
            "complexity": 0.6,
            "social_norms": ["maintain_professionalism", "show_respect"],
            "expected_behavior": "collaborative_engagement"
        }
    
    def _determine_social_emotion(self, social_assessment: Dict[str, Any], 
                                context: SocialContext) -> EmotionalState:
        """Determine appropriate emotion for social context"""
        return EmotionalState(
            primary_emotion=EmotionType.TRUST,
            intensity=0.6,
            arousal=0.5,
            valence=0.7,
            context="social_interaction"
        )
    
    def _generate_social_behavior(self, appropriate_emotion: EmotionalState, 
                                social_assessment: Dict[str, Any], 
                                context: SocialContext) -> str:
        """Generate socially appropriate behavior"""
        return "Engage professionally with appropriate emotional tone"
    
    def _calculate_arousal(self, emotion: EmotionType, intensity: float) -> float:
        """Calculate arousal level for emotion"""
        high_arousal_emotions = [EmotionType.ANGER, EmotionType.FEAR, EmotionType.SURPRISE, EmotionType.JOY]
        
        if emotion in high_arousal_emotions:
            return min(1.0, 0.5 + intensity * 0.5)
        else:
            return min(1.0, 0.3 + intensity * 0.3)
    
    def _calculate_valence(self, emotion: EmotionType, intensity: float) -> float:
        """Calculate valence (positive/negative) for emotion"""
        positive_emotions = [EmotionType.JOY, EmotionType.TRUST, EmotionType.ANTICIPATION]
        negative_emotions = [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR, EmotionType.DISGUST]
        
        if emotion in positive_emotions:
            return min(1.0, 0.6 + intensity * 0.4)
        elif emotion in negative_emotions:
            return max(0.0, 0.4 - intensity * 0.4)
        else:
            return 0.5