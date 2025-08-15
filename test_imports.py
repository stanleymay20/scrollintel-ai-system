#!/usr/bin/env python3

print("Testing imports...")

try:
    from scrollintel.models.emotion_models import (
        EmotionType, EmotionalState, EmotionalResponse, EmotionalMemory,
        SocialContext, EmpathyAssessment
    )
    print("✓ All imports successful")
    
    # Test creating an EmotionalState
    state = EmotionalState(
        primary_emotion=EmotionType.TRUST,
        intensity=0.3,
        arousal=0.4,
        valence=0.7
    )
    print("✓ EmotionalState created successfully")
    
    # Now test the class definition manually
    exec("""
class EmotionSimulator:
    def __init__(self):
        self.current_emotional_state = EmotionalState(
            primary_emotion=EmotionType.TRUST,
            intensity=0.3,
            arousal=0.4,
            valence=0.7
        )
        print("EmotionSimulator initialized")
""")
    
    print("✓ Class definition executed")
    print("EmotionSimulator in locals:", 'EmotionSimulator' in locals())
    
    if 'EmotionSimulator' in locals():
        simulator = EmotionSimulator()
        print("✓ EmotionSimulator instantiated")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()