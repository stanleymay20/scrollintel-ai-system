"""
Consciousness simulation models for AGI cognitive architecture.
Implements consciousness state, awareness, and meta-cognitive structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import uuid


class ConsciousnessLevel(Enum):
    """Levels of consciousness simulation"""
    UNCONSCIOUS = "unconscious"
    SUBCONSCIOUS = "subconscious"
    CONSCIOUS = "conscious"
    SELF_AWARE = "self_aware"
    META_CONSCIOUS = "meta_conscious"


class AwarenessType(Enum):
    """Types of awareness in the system"""
    SELF_AWARENESS = "self_awareness"
    SITUATIONAL_AWARENESS = "situational_awareness"
    TEMPORAL_AWARENESS = "temporal_awareness"
    GOAL_AWARENESS = "goal_awareness"
    EMOTIONAL_AWARENESS = "emotional_awareness"


@dataclass
class Thought:
    """Represents a single thought or cognitive process"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    thought_type: str = "general"
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    related_thoughts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaCognitiveInsight:
    """Represents insights about thinking processes"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    insight_type: str = ""
    description: str = ""
    thought_pattern: str = ""
    effectiveness_score: float = 0.0
    improvement_suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Goal:
    """Represents a goal or intention"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: float = 0.0
    status: str = "active"
    sub_goals: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class IntentionalState:
    """Represents the intentional state of the system"""
    primary_goal: Optional[Goal] = None
    active_goals: List[Goal] = field(default_factory=list)
    goal_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    intention_strength: float = 0.0
    focus_direction: str = ""
    commitment_level: float = 0.0


@dataclass
class Experience:
    """Represents an experience or event"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    experience_type: str = "general"
    emotional_valence: float = 0.0
    significance: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    outcomes: List[str] = field(default_factory=list)


@dataclass
class SelfReflection:
    """Represents self-reflective analysis"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reflection_type: str = ""
    insights: List[str] = field(default_factory=list)
    self_assessment: Dict[str, float] = field(default_factory=dict)
    areas_for_improvement: List[str] = field(default_factory=list)
    strengths_identified: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AwarenessState:
    """Represents the current state of awareness"""
    level: ConsciousnessLevel = ConsciousnessLevel.CONSCIOUS
    awareness_types: List[AwarenessType] = field(default_factory=list)
    attention_focus: str = ""
    awareness_intensity: float = 0.0
    context_understanding: Dict[str, Any] = field(default_factory=dict)
    self_model: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsciousnessState:
    """Main consciousness state representation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: ConsciousnessLevel = ConsciousnessLevel.CONSCIOUS
    awareness: AwarenessState = field(default_factory=AwarenessState)
    active_thoughts: List[Thought] = field(default_factory=list)
    intentional_state: IntentionalState = field(default_factory=IntentionalState)
    meta_cognitive_state: Dict[str, Any] = field(default_factory=dict)
    consciousness_coherence: float = 0.0
    self_monitoring_active: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_thought(self, thought: Thought) -> None:
        """Add a thought to active consciousness"""
        self.active_thoughts.append(thought)
        # Keep only recent thoughts to prevent memory overflow
        if len(self.active_thoughts) > 100:
            self.active_thoughts = self.active_thoughts[-50:]
    
    def update_awareness(self, awareness_type: AwarenessType, intensity: float) -> None:
        """Update awareness state"""
        if awareness_type not in self.awareness.awareness_types:
            self.awareness.awareness_types.append(awareness_type)
        self.awareness.awareness_intensity = max(self.awareness.awareness_intensity, intensity)
    
    def set_intention(self, goal: Goal) -> None:
        """Set primary intention"""
        self.intentional_state.primary_goal = goal
        if goal not in self.intentional_state.active_goals:
            self.intentional_state.active_goals.append(goal)


@dataclass
class CognitiveContext:
    """Context for cognitive processing"""
    situation: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)
    available_resources: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    time_pressure: float = 0.0
    complexity_level: float = 0.0
    stakeholders: List[str] = field(default_factory=list)