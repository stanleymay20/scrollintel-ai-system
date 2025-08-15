"""
Emotion and Personality Models for AGI Cognitive Architecture

This module defines the data models for emotion simulation, personality traits,
and social cognition capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time
from datetime import datetime


class EmotionType(Enum):
    """Core emotion types based on psychological research"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


class PersonalityTrait(Enum):
    """Big Five personality traits and additional cognitive traits"""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"
    CURIOSITY = "curiosity"
    EMPATHY = "empathy"
    ASSERTIVENESS = "assertiveness"


@dataclass
class EmotionalState:
    """Represents the current emotional state of the system"""
    primary_emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    secondary_emotions: Dict[EmotionType, float] = field(default_factory=dict)
    arousal: float = 0.5  # 0.0 (calm) to 1.0 (excited)
    valence: float = 0.5  # 0.0 (negative) to 1.0 (positive)
    timestamp: datetime = field(default_factory=datetime.now)
    context: Optional[str] = None
    triggers: List[str] = field(default_factory=list)


@dataclass
class PersonalityProfile:
    """Represents the personality configuration of the system"""
    traits: Dict[PersonalityTrait, float] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    behavioral_patterns: Dict[str, float] = field(default_factory=dict)
    social_style: str = "balanced"
    communication_style: str = "professional"
    decision_making_style: str = "analytical"
    
    def __post_init__(self):
        # Initialize default trait values if not provided
        if not self.traits:
            self.traits = {
                PersonalityTrait.OPENNESS: 0.8,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
                PersonalityTrait.EXTRAVERSION: 0.6,
                PersonalityTrait.AGREEABLENESS: 0.7,
                PersonalityTrait.NEUROTICISM: 0.2,
                PersonalityTrait.CURIOSITY: 0.9,
                PersonalityTrait.EMPATHY: 0.8,
                PersonalityTrait.ASSERTIVENESS: 0.7
            }


@dataclass
class SocialContext:
    """Represents the social context for emotional and personality responses"""
    participants: List[str] = field(default_factory=list)
    relationship_types: Dict[str, str] = field(default_factory=dict)
    social_setting: str = "professional"
    cultural_context: str = "business"
    power_dynamics: Dict[str, float] = field(default_factory=dict)
    group_dynamics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmotionalResponse:
    """Represents an emotional response to a stimulus"""
    stimulus: str
    emotional_state: EmotionalState
    behavioral_response: str
    cognitive_appraisal: str
    social_appropriateness: float  # 0.0 to 1.0
    regulation_strategy: Optional[str] = None
    confidence: float = 0.8


@dataclass
class EmpathyAssessment:
    """Represents empathetic understanding of others' emotional states"""
    target_person: str
    perceived_emotion: EmotionType
    perceived_intensity: float
    confidence: float
    contextual_factors: List[str] = field(default_factory=list)
    appropriate_response: str = ""
    emotional_contagion: float = 0.0  # How much the emotion affects the system


@dataclass
class SocialCognition:
    """Represents social understanding and theory of mind capabilities"""
    situation_assessment: str
    social_norms: List[str] = field(default_factory=list)
    relationship_dynamics: Dict[str, Any] = field(default_factory=dict)
    communication_strategy: str = ""
    predicted_reactions: Dict[str, str] = field(default_factory=dict)
    social_goals: List[str] = field(default_factory=list)


@dataclass
class EmotionalMemory:
    """Represents emotional memories that influence future responses"""
    event_description: str
    emotional_state: EmotionalState
    outcome: str
    lessons_learned: List[str] = field(default_factory=list)
    emotional_significance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PersonalityInfluence:
    """Represents how personality traits influence decision-making"""
    decision_context: str
    trait_influences: Dict[PersonalityTrait, float] = field(default_factory=dict)
    personality_bias: str = ""
    confidence_modifier: float = 0.0
    risk_tolerance: float = 0.5
    social_consideration: float = 0.5