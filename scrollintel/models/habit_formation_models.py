"""
Habit Formation Models for Cultural Transformation Leadership

This module defines data models for organizational habit formation,
including habit design, implementation strategies, and sustainability mechanisms.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum


class HabitType(Enum):
    """Types of organizational habits"""
    COMMUNICATION = "communication"
    COLLABORATION = "collaboration"
    LEARNING = "learning"
    INNOVATION = "innovation"
    QUALITY = "quality"
    FEEDBACK = "feedback"
    RECOGNITION = "recognition"
    PLANNING = "planning"
    REFLECTION = "reflection"
    WELLNESS = "wellness"


class HabitFrequency(Enum):
    """Frequency of habit execution"""
    DAILY = "daily"
    WEEKLY = "weekly"
    BI_WEEKLY = "bi_weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    EVENT_BASED = "event_based"


class HabitStage(Enum):
    """Stages of habit formation"""
    DESIGN = "design"
    INITIATION = "initiation"
    FORMATION = "formation"
    MAINTENANCE = "maintenance"
    OPTIMIZATION = "optimization"
    INSTITUTIONALIZED = "institutionalized"


class SustainabilityLevel(Enum):
    """Levels of habit sustainability"""
    FRAGILE = "fragile"
    DEVELOPING = "developing"
    STABLE = "stable"
    ROBUST = "robust"
    SELF_REINFORCING = "self_reinforcing"


@dataclass
class OrganizationalHabit:
    """Represents a positive organizational habit"""
    id: str
    name: str
    description: str
    habit_type: HabitType
    target_behavior: str
    trigger_conditions: List[str]
    execution_steps: List[str]
    success_indicators: List[str]
    frequency: HabitFrequency
    duration_minutes: int
    participants: List[str]
    facilitators: List[str]
    resources_required: List[str]
    cultural_alignment: float  # 0.0 to 1.0
    business_impact: str
    stage: HabitStage
    created_date: datetime
    created_by: str
    
    def __post_init__(self):
        if not 0.0 <= self.cultural_alignment <= 1.0:
            raise ValueError("Cultural alignment must be between 0.0 and 1.0")
        if self.duration_minutes <= 0:
            raise ValueError("Duration must be positive")


@dataclass
class HabitFormationStrategy:
    """Strategy for forming organizational habits"""
    id: str
    habit_id: str
    strategy_name: str
    description: str
    formation_phases: List[str]
    timeline_weeks: int
    key_milestones: List[str]
    success_metrics: List[str]
    reinforcement_mechanisms: List[str]
    barrier_mitigation: List[str]
    stakeholder_engagement: Dict[str, List[str]]
    resource_allocation: Dict[str, Any]
    risk_assessment: Dict[str, str]
    created_date: datetime
    
    def __post_init__(self):
        if self.timeline_weeks <= 0:
            raise ValueError("Timeline must be positive")


@dataclass
class HabitImplementation:
    """Implementation details for habit formation"""
    id: str
    strategy_id: str
    implementation_name: str
    phase: str
    start_date: datetime
    end_date: datetime
    activities: List[str]
    deliverables: List[str]
    responsible_parties: List[str]
    success_criteria: List[str]
    progress_indicators: List[str]
    status: str
    completion_percentage: float  # 0.0 to 1.0
    challenges_encountered: List[str]
    lessons_learned: List[str]
    next_steps: List[str]
    
    def __post_init__(self):
        if not 0.0 <= self.completion_percentage <= 1.0:
            raise ValueError("Completion percentage must be between 0.0 and 1.0")


@dataclass
class HabitSustainability:
    """Sustainability mechanisms for organizational habits"""
    id: str
    habit_id: str
    sustainability_level: SustainabilityLevel
    reinforcement_systems: List[str]
    monitoring_mechanisms: List[str]
    feedback_loops: List[str]
    adaptation_triggers: List[str]
    renewal_strategies: List[str]
    institutional_support: List[str]
    cultural_integration: float  # 0.0 to 1.0
    resilience_factors: List[str]
    vulnerability_points: List[str]
    mitigation_plans: List[str]
    sustainability_score: float  # 0.0 to 1.0
    last_assessment: datetime
    next_review: datetime
    
    def __post_init__(self):
        for score in [self.cultural_integration, self.sustainability_score]:
            if not 0.0 <= score <= 1.0:
                raise ValueError("Scores must be between 0.0 and 1.0")


@dataclass
class HabitProgress:
    """Progress tracking for habit formation"""
    id: str
    habit_id: str
    participant_id: str
    tracking_period: str  # e.g., "2024-W01", "2024-01"
    execution_count: int
    target_count: int
    consistency_rate: float  # 0.0 to 1.0
    quality_score: float  # 0.0 to 1.0
    engagement_level: float  # 0.0 to 1.0
    barriers_encountered: List[str]
    support_received: List[str]
    improvements_noted: List[str]
    feedback_provided: str
    next_period_goals: List[str]
    recorded_date: datetime
    
    def __post_init__(self):
        for score in [self.consistency_rate, self.quality_score, self.engagement_level]:
            if not 0.0 <= score <= 1.0:
                raise ValueError("Scores must be between 0.0 and 1.0")
        if self.execution_count < 0 or self.target_count < 0:
            raise ValueError("Counts must be non-negative")


@dataclass
class HabitReinforcementMechanism:
    """Reinforcement mechanism for habit sustainability"""
    id: str
    habit_id: str
    mechanism_type: str  # e.g., "recognition", "reward", "feedback", "social"
    mechanism_name: str
    description: str
    trigger_conditions: List[str]
    implementation_steps: List[str]
    frequency: HabitFrequency
    effectiveness_score: float  # 0.0 to 1.0
    cost_effectiveness: float  # 0.0 to 1.0
    participant_satisfaction: float  # 0.0 to 1.0
    sustainability_impact: float  # 0.0 to 1.0
    active: bool
    created_date: datetime
    last_used: Optional[datetime] = None
    
    def __post_init__(self):
        for score in [self.effectiveness_score, self.cost_effectiveness, 
                     self.participant_satisfaction, self.sustainability_impact]:
            if not 0.0 <= score <= 1.0:
                raise ValueError("Scores must be between 0.0 and 1.0")


@dataclass
class HabitFormationPlan:
    """Complete habit formation plan"""
    id: str
    organization_id: str
    plan_name: str
    description: str
    target_habits: List[OrganizationalHabit]
    formation_strategies: List[HabitFormationStrategy]
    implementation_timeline: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    success_metrics: List[str]
    roi_projections: Dict[str, float]
    risk_mitigation: Dict[str, List[str]]
    stakeholder_commitments: Dict[str, List[str]]
    approval_status: str
    created_date: datetime
    created_by: str
    approved_by: Optional[str] = None
    approved_date: Optional[datetime] = None


@dataclass
class HabitFormationMetrics:
    """Comprehensive metrics for habit formation tracking"""
    organization_id: str
    total_habits_designed: int
    habits_in_formation: int
    habits_established: int
    habits_institutionalized: int
    average_formation_time_weeks: float
    overall_success_rate: float
    participant_engagement_average: float
    sustainability_index: float
    cultural_integration_score: float
    business_impact_score: float
    roi_achieved: float
    calculated_date: datetime


@dataclass
class HabitOptimization:
    """Optimization recommendations for habit formation"""
    habit_id: str
    current_effectiveness: float
    optimization_areas: List[str]
    recommended_improvements: List[str]
    expected_impact: float
    implementation_complexity: str  # "low", "medium", "high"
    resource_requirements: List[str]
    timeline_estimate: int  # weeks
    priority_level: str  # "low", "medium", "high", "critical"
    analysis_date: datetime