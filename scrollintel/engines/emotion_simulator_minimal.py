"""
Minimal Emotion Simulator for testing
"""

from scrollintel.models.emotion_models import (
    EmotionType, EmotionalState, EmotionalResponse, EmotionalMemory,
    SocialContext, EmpathyAssessment
)


class EmotionSimulator:
    """Minimal emotion simulator for testing"""
    
    def __init__(self):
        self.current_emotional_state = EmotionalState(
            primary_emotion=EmotionType.TRUST,
            intensity=0.3,
            arousal=0.4,
            valence=0.7
        )
    
    def process_emotional_stimulus(self, stimulus: str, context=None):
        """Process emotional stimulus"""
        return EmotionalResponse(
            stimulus=stimulus,
            emotional_state=self.current_emotional_state,
            behavioral_response="Test response",
            cognitive_appraisal="Test appraisal",
            social_appropriateness=0.8,
            confidence=0.85
        )