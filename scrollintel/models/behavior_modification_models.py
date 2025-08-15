"""
Behavior Modification Models for Cultural Transformation Leadership

This module defines data models for behavior modification strategies,
techniques, and progress tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class ModificationTechnique(Enum):
    """Types of behavior modification techniques"""
    POSITIVE_REINFORCEMENT = "positive_reinforcement"
    NEGATIVE_REINFORCEMENT = "negative_reinforcement"
    MODELING = "modeling"
    COACHING = "coaching"
    TRAINING = "training"
    FEEDBACK = "feedback"
    GOAL_SETTING = "goal_setting"
    ENVIRONMENTAL_DESIGN = "environmental_design"
    PEER_INFLUENCE = "peer_influence"
    INCENTIVE_SYSTEMS = "incentive_systems"


class ModificationStatus(Enum):
    """Status of behavior modification efforts"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ProgressLevel(Enum):
    """Levels of behavior change progress"""
    NO_CHANGE = "no_change"
    MINIMAL_PROGRESS = "minimal_progress"
    MODERATE_PROGRESS = "moderate_progress"
    SIGNIFICANT_PROGRESS = "significant_progress"
    COMPLETE_CHANGE = "complete_change"


@dataclass
class BehaviorModificationStrategy:
    """Represents a behavior modification strategy"""
    id: str
    name: str
    description: str
    target_behavior: str
    desired_outcome: str
    techniques: List[ModificationTechnique]
    timeline_weeks: int
    success_criteria: List[str]
    resources_required: List[str]
    stakeholders: List[str]
    risk_factors: List[str]
    mitigation_strategies: List[str]
    created_date: datetime
    created_by: str
    
    def __post_init__(self):
        if self.timeline_weeks <= 0:
            raise ValueError("Timeline must be positive")


@dataclass
class ModificationIntervention:
    """Represents a specific behavior modification intervention"""
    id: str
    strategy_id: str
    technique: ModificationTechnique
    intervention_name: str
    description: str
    target_participants: List[str]
    implementation_steps: List[str]
    duration_days: int
    frequency: str  # e.g., "daily", "weekly", "as_needed"
    resources_needed: List[str]
    success_metrics: List[str]
    status: ModificationStatus
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    assigned_facilitator: Optional[str] = None
    
    def __post_init__(self):
        if self.duration_days <= 0:
            raise ValueError("Duration must be positive")


@dataclass
class BehaviorChangeProgress:
    """Tracks progress of behavior change efforts"""
    id: str
    strategy_id: str
    participant_id: str
    baseline_measurement: float  # 0.0 to 1.0
    current_measurement: float  # 0.0 to 1.0
    target_measurement: float  # 0.0 to 1.0
    progress_level: ProgressLevel
    improvement_rate: float  # Change per week
    milestones_achieved: List[str]
    challenges_encountered: List[str]
    adjustments_made: List[str]
    last_updated: datetime
    next_review_date: datetime
    
    def __post_init__(self):
        for measurement in [self.baseline_measurement, self.current_measurement, self.target_measurement]:
            if not 0.0 <= measurement <= 1.0:
                raise ValueError("Measurements must be between 0.0 and 1.0")


@dataclass
class ModificationPlan:
    """Complete behavior modification plan"""
    id: str
    organization_id: str
    plan_name: str
    description: str
    strategies: List[BehaviorModificationStrategy]
    interventions: List[ModificationIntervention]
    overall_timeline_weeks: int
    success_criteria: List[str]
    budget_allocated: float
    roi_expectations: Dict[str, float]
    risk_assessment: Dict[str, Any]
    approval_status: str
    created_date: datetime
    created_by: str
    approved_by: Optional[str] = None
    approved_date: Optional[datetime] = None


@dataclass
class TechniqueEffectiveness:
    """Tracks effectiveness of modification techniques"""
    technique: ModificationTechnique
    organization_id: str
    behavior_type: str
    success_rate: float  # 0.0 to 1.0
    average_improvement: float  # 0.0 to 1.0
    time_to_change_weeks: float
    participant_satisfaction: float  # 0.0 to 1.0
    cost_effectiveness: float  # 0.0 to 1.0
    sustainability_score: float  # 0.0 to 1.0
    sample_size: int
    last_updated: datetime
    
    def __post_init__(self):
        for score in [self.success_rate, self.average_improvement, self.participant_satisfaction, 
                     self.cost_effectiveness, self.sustainability_score]:
            if not 0.0 <= score <= 1.0:
                raise ValueError("Scores must be between 0.0 and 1.0")


@dataclass
class ModificationOptimization:
    """Optimization recommendations for behavior modification"""
    strategy_id: str
    current_effectiveness: float
    optimization_opportunities: List[str]
    recommended_adjustments: List[str]
    expected_improvement: float
    implementation_effort: str  # "low", "medium", "high"
    risk_level: str  # "low", "medium", "high"
    priority_score: float  # 0.0 to 1.0
    analysis_date: datetime
    
    def __post_init__(self):
        if not 0.0 <= self.priority_score <= 1.0:
            raise ValueError("Priority score must be between 0.0 and 1.0")


@dataclass
class BehaviorChangeMetrics:
    """Comprehensive metrics for behavior change tracking"""
    organization_id: str
    total_strategies: int
    active_interventions: int
    participants_engaged: int
    overall_success_rate: float
    average_improvement_rate: float
    time_to_change_average: float
    participant_satisfaction_average: float
    cost_per_participant: float
    roi_achieved: float
    sustainability_index: float
    calculated_date: datetime