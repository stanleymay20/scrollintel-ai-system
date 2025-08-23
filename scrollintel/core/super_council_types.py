"""
Shared types and enums for the Superintelligent Council of Models
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List
from datetime import datetime


class DebateRole(str, Enum):
    RED_TEAM = "red_team"
    BLUE_TEAM = "blue_team"
    MODERATOR = "moderator"
    JUROR = "juror"
    SOCRATIC_QUESTIONER = "socratic_questioner"


class ArgumentationDepth(str, Enum):
    SURFACE = "surface"
    INTERMEDIATE = "intermediate"
    DEEP = "deep"
    INFINITE = "infinite"


@dataclass
class ModelCapability:
    """Represents the capabilities of a frontier AI model"""
    model_name: str
    reasoning_strength: float
    creativity_score: float
    factual_accuracy: float
    philosophical_depth: float
    adversarial_robustness: float
    specializations: List[str] = field(default_factory=list)


@dataclass
class DebateArgument:
    """Represents a single argument in the adversarial debate"""
    model_id: str
    role: DebateRole
    content: str
    confidence: float
    reasoning_chain: List[str]
    evidence: List[str]
    counterarguments_addressed: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SocraticQuestion:
    """Represents a Socratic question for deep philosophical inquiry"""
    question: str
    philosophical_domain: str
    depth_level: int
    target_assumptions: List[str]
    expected_insight_categories: List[str]