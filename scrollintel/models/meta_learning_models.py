"""
Meta-learning and adaptation models for AGI cognitive architecture.
Implements learning-to-learn algorithms and adaptive capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np


class LearningStrategy(Enum):
    """Types of learning strategies for meta-learning."""
    GRADIENT_BASED = "gradient_based"
    MODEL_AGNOSTIC = "model_agnostic"
    MEMORY_AUGMENTED = "memory_augmented"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"


class AdaptationType(Enum):
    """Types of adaptation mechanisms."""
    PARAMETER_ADAPTATION = "parameter_adaptation"
    ARCHITECTURE_ADAPTATION = "architecture_adaptation"
    STRATEGY_ADAPTATION = "strategy_adaptation"
    ENVIRONMENT_ADAPTATION = "environment_adaptation"


@dataclass
class Task:
    """Represents a learning task."""
    task_id: str
    domain: str
    description: str
    input_format: Dict[str, Any]
    output_format: Dict[str, Any]
    complexity_level: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class LearningExperience:
    """Captures learning experience from a task."""
    task: Task
    performance_metrics: Dict[str, float]
    learning_trajectory: List[float]
    adaptation_steps: List[Dict[str, Any]]
    time_to_learn: float
    final_accuracy: float
    transfer_potential: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaKnowledge:
    """Represents meta-knowledge about learning."""
    domain: str
    learning_patterns: Dict[str, Any]
    optimal_strategies: List[LearningStrategy]
    adaptation_rules: Dict[str, Any]
    transfer_mappings: Dict[str, List[str]]
    performance_predictors: Dict[str, Any]
    confidence_score: float
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptationState:
    """Current state of adaptation process."""
    current_environment: Dict[str, Any]
    active_adaptations: List[AdaptationType]
    adaptation_history: List[Dict[str, Any]]
    performance_trend: List[float]
    adaptation_confidence: float
    next_adaptation_time: Optional[datetime] = None


@dataclass
class SkillAcquisition:
    """Tracks rapid skill acquisition progress."""
    skill_name: str
    domain: str
    acquisition_strategy: LearningStrategy
    learning_curve: List[float]
    milestones: List[Dict[str, Any]]
    transfer_sources: List[str]
    mastery_level: float
    acquisition_time: float
    retention_score: float


@dataclass
class TransferLearningMap:
    """Maps knowledge transfer between domains."""
    source_domain: str
    target_domain: str
    transferable_features: List[str]
    transfer_efficiency: float
    adaptation_requirements: Dict[str, Any]
    success_probability: float
    transfer_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SelfImprovementPlan:
    """Plan for self-improving algorithms."""
    improvement_target: str
    current_capability: float
    target_capability: float
    improvement_strategy: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    timeline: Dict[str, datetime]
    risk_assessment: Dict[str, float]
    success_metrics: List[str]


@dataclass
class EnvironmentalChallenge:
    """Represents environmental challenges requiring adaptation."""
    challenge_id: str
    environment_type: str
    challenge_description: str
    difficulty_level: float
    required_adaptations: List[AdaptationType]
    success_criteria: Dict[str, float]
    time_constraints: Optional[float] = None
    resource_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaLearningState:
    """Overall state of the meta-learning system."""
    active_tasks: List[Task]
    learning_experiences: List[LearningExperience]
    meta_knowledge_base: Dict[str, MetaKnowledge]
    adaptation_state: AdaptationState
    skill_inventory: List[SkillAcquisition]
    transfer_maps: List[TransferLearningMap]
    improvement_plans: List[SelfImprovementPlan]
    performance_history: Dict[str, List[float]]
    last_update: datetime = field(default_factory=datetime.now)