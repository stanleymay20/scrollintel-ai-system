"""
Intervention Design Models

Data models for strategic intervention identification, design, effectiveness prediction,
and sequencing coordination.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum


class InterventionType(Enum):
    """Type of cultural intervention"""
    COMMUNICATION = "communication"
    TRAINING = "training"
    PROCESS_CHANGE = "process_change"
    SYSTEM_CHANGE = "system_change"
    POLICY_CHANGE = "policy_change"
    STRUCTURAL_CHANGE = "structural_change"
    BEHAVIORAL_NUDGE = "behavioral_nudge"
    RECOGNITION_REWARD = "recognition_reward"
    LEADERSHIP_MODELING = "leadership_modeling"
    ENVIRONMENTAL_CHANGE = "environmental_change"


class InterventionScope(Enum):
    """Scope of intervention impact"""
    INDIVIDUAL = "individual"
    TEAM = "team"
    DEPARTMENT = "department"
    DIVISION = "division"
    ORGANIZATION = "organization"
    ECOSYSTEM = "ecosystem"


class InterventionStatus(Enum):
    """Status of intervention"""
    DESIGNED = "designed"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class EffectivenessLevel(Enum):
    """Level of intervention effectiveness"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class InterventionTarget:
    """Target for cultural intervention"""
    target_type: str  # behavior, value, norm, process, etc.
    current_state: str
    desired_state: str
    gap_size: float  # 0-1 scale
    priority: int
    measurable_indicators: List[str]
    
    def __post_init__(self):
        if not 0 <= self.gap_size <= 1:
            raise ValueError("Gap size must be between 0 and 1")


@dataclass
class InterventionDesign:
    """Design specification for a cultural intervention"""
    id: str
    name: str
    description: str
    intervention_type: InterventionType
    scope: InterventionScope
    targets: List[InterventionTarget]
    objectives: List[str]
    activities: List[str]
    resources_required: Dict[str, Any]
    duration: timedelta
    participants: List[str]
    facilitators: List[str]
    materials: List[str]
    success_criteria: List[str]
    measurement_methods: List[str]
    predicted_effectiveness: float = 0.0
    implementation_complexity: float = 0.0
    resource_intensity: float = 0.0
    risk_level: float = 0.0
    created_date: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        for score in [self.predicted_effectiveness, self.implementation_complexity, 
                     self.resource_intensity, self.risk_level]:
            if not 0 <= score <= 1:
                raise ValueError("All scores must be between 0 and 1")


@dataclass
class InterventionSequence:
    """Sequence of coordinated interventions"""
    id: str
    name: str
    description: str
    interventions: List[str]  # Intervention IDs in sequence
    sequencing_rationale: str
    dependencies: Dict[str, List[str]]  # intervention_id -> list of prerequisite IDs
    parallel_groups: List[List[str]]  # Groups of interventions that can run in parallel
    total_duration: timedelta
    coordination_requirements: List[str]


@dataclass
class EffectivenessPrediction:
    """Prediction of intervention effectiveness"""
    intervention_id: str
    predicted_effectiveness: EffectivenessLevel
    confidence_score: float
    contributing_factors: List[str]
    risk_factors: List[str]
    success_probability: float
    expected_outcomes: List[str]
    measurement_timeline: Dict[str, datetime]  # outcome -> measurement_date
    
    def __post_init__(self):
        if not 0 <= self.confidence_score <= 1:
            raise ValueError("Confidence score must be between 0 and 1")
        if not 0 <= self.success_probability <= 1:
            raise ValueError("Success probability must be between 0 and 1")


@dataclass
class InterventionOptimization:
    """Optimization recommendations for intervention"""
    intervention_id: str
    optimization_type: str
    current_limitation: str
    recommended_improvement: str
    expected_benefit: str
    implementation_effort: str
    priority: int
    estimated_impact: float
    
    def __post_init__(self):
        if not 0 <= self.estimated_impact <= 1:
            raise ValueError("Estimated impact must be between 0 and 1")


@dataclass
class InterventionDesignRequest:
    """Request for intervention design"""
    organization_id: str
    roadmap_id: str
    cultural_gaps: List[Dict[str, Any]]
    target_behaviors: List[str]
    available_resources: Dict[str, Any]
    constraints: List[str]
    timeline: timedelta
    stakeholder_preferences: Dict[str, Any]
    risk_tolerance: float = 0.5
    effectiveness_threshold: float = 0.7
    
    def __post_init__(self):
        if not 0 <= self.risk_tolerance <= 1:
            raise ValueError("Risk tolerance must be between 0 and 1")
        if not 0 <= self.effectiveness_threshold <= 1:
            raise ValueError("Effectiveness threshold must be between 0 and 1")


@dataclass
class InterventionDesignResult:
    """Result of intervention design process"""
    interventions: List[InterventionDesign]
    sequence: InterventionSequence
    effectiveness_predictions: List[EffectivenessPrediction]
    resource_requirements: Dict[str, Any]
    implementation_plan: Dict[str, Any]
    risk_assessment: List[str]
    optimization_recommendations: List[InterventionOptimization]
    success_probability: float
    
    def __post_init__(self):
        if not 0 <= self.success_probability <= 1:
            raise ValueError("Success probability must be between 0 and 1")


@dataclass
class InterventionTemplate:
    """Template for common intervention types"""
    id: str
    name: str
    description: str
    intervention_type: InterventionType
    typical_scope: InterventionScope
    template_activities: List[str]
    typical_duration: timedelta
    resource_requirements: Dict[str, Any]
    success_factors: List[str]
    common_challenges: List[str]
    effectiveness_indicators: List[str]
    customization_options: Dict[str, Any]


@dataclass
class InterventionCoordination:
    """Coordination requirements between interventions"""
    id: str
    intervention_ids: List[str]
    coordination_type: str  # sequential, parallel, conditional, etc.
    coordination_requirements: List[str]
    synchronization_points: List[str]
    communication_needs: List[str]
    resource_sharing: Dict[str, Any]
    risk_mitigation: List[str]


@dataclass
class InterventionFeedback:
    """Feedback on intervention effectiveness"""
    intervention_id: str
    feedback_date: datetime
    effectiveness_rating: float
    participant_satisfaction: float
    objective_achievement: float
    unexpected_outcomes: List[str]
    improvement_suggestions: List[str]
    continuation_recommendation: str
    lessons_learned: List[str]
    
    def __post_init__(self):
        for rating in [self.effectiveness_rating, self.participant_satisfaction, 
                      self.objective_achievement]:
            if not 0 <= rating <= 1:
                raise ValueError("All ratings must be between 0 and 1")