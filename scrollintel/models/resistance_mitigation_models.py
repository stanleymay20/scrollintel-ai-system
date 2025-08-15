"""
Models for Cultural Change Resistance Mitigation Framework
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum

from .resistance_detection_models import ResistanceType, ResistanceSeverity


class MitigationStrategy(Enum):
    """Types of resistance mitigation strategies"""
    COMMUNICATION_ENHANCEMENT = "communication_enhancement"
    STAKEHOLDER_ENGAGEMENT = "stakeholder_engagement"
    TRAINING_SUPPORT = "training_support"
    INCENTIVE_ALIGNMENT = "incentive_alignment"
    PROCESS_MODIFICATION = "process_modification"
    LEADERSHIP_INTERVENTION = "leadership_intervention"
    PEER_INFLUENCE = "peer_influence"
    GRADUAL_IMPLEMENTATION = "gradual_implementation"
    RESOURCE_PROVISION = "resource_provision"
    FEEDBACK_INTEGRATION = "feedback_integration"


class InterventionType(Enum):
    """Types of resistance mitigation interventions"""
    INDIVIDUAL_COACHING = "individual_coaching"
    TEAM_WORKSHOP = "team_workshop"
    TOWN_HALL_MEETING = "town_hall_meeting"
    TRAINING_SESSION = "training_session"
    MENTORING_PROGRAM = "mentoring_program"
    COMMUNICATION_CAMPAIGN = "communication_campaign"
    POLICY_ADJUSTMENT = "policy_adjustment"
    PROCESS_SIMPLIFICATION = "process_simplification"
    RESOURCE_ALLOCATION = "resource_allocation"
    RECOGNITION_PROGRAM = "recognition_program"


class MitigationStatus(Enum):
    """Status of mitigation efforts"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class MitigationPlan:
    """Plan for mitigating resistance"""
    id: str
    detection_id: str
    organization_id: str
    transformation_id: str
    resistance_type: ResistanceType
    severity: ResistanceSeverity
    strategies: List[MitigationStrategy]
    interventions: List['MitigationIntervention']
    target_stakeholders: List[str]
    success_criteria: Dict[str, Any]
    timeline: Dict[str, datetime]
    resource_requirements: Dict[str, Any]
    risk_factors: List[str]
    contingency_plans: List[str]
    created_at: datetime
    created_by: str


@dataclass
class MitigationIntervention:
    """Individual intervention for resistance mitigation"""
    id: str
    plan_id: str
    intervention_type: InterventionType
    strategy: MitigationStrategy
    title: str
    description: str
    target_audience: List[str]
    facilitators: List[str]
    duration_hours: float
    scheduled_date: datetime
    completion_date: Optional[datetime]
    status: MitigationStatus
    success_metrics: Dict[str, Any]
    actual_results: Dict[str, Any]
    participant_feedback: List[Dict[str, Any]]
    effectiveness_score: Optional[float]
    lessons_learned: List[str]
    follow_up_actions: List[str]


@dataclass
class ResistanceAddressingStrategy:
    """Strategy for addressing specific resistance patterns"""
    id: str
    resistance_type: ResistanceType
    strategy_name: str
    description: str
    approach_steps: List[str]
    required_resources: Dict[str, Any]
    typical_duration: int
    success_rate: float
    best_practices: List[str]
    common_pitfalls: List[str]
    effectiveness_factors: Dict[str, float]
    stakeholder_considerations: Dict[str, List[str]]


@dataclass
class MitigationExecution:
    """Execution details for mitigation plan"""
    id: str
    plan_id: str
    execution_phase: str
    start_date: datetime
    end_date: Optional[datetime]
    status: MitigationStatus
    progress_percentage: float
    completed_interventions: List[str]
    active_interventions: List[str]
    pending_interventions: List[str]
    resource_utilization: Dict[str, float]
    stakeholder_engagement: Dict[str, float]
    interim_results: Dict[str, Any]
    challenges_encountered: List[str]
    adjustments_made: List[str]
    next_steps: List[str]


@dataclass
class ResistanceResolution:
    """Resolution tracking for resistance instances"""
    id: str
    detection_id: str
    plan_id: str
    resolution_date: datetime
    resolution_method: str
    final_status: str
    effectiveness_rating: float
    stakeholder_satisfaction: Dict[str, float]
    behavioral_changes: List[str]
    cultural_impact: Dict[str, float]
    lessons_learned: List[str]
    recommendations: List[str]
    follow_up_required: bool
    follow_up_schedule: Optional[datetime]


@dataclass
class MitigationValidation:
    """Validation of mitigation effectiveness"""
    id: str
    plan_id: str
    validation_date: datetime
    validation_method: str
    success_criteria_met: Dict[str, bool]
    quantitative_results: Dict[str, float]
    qualitative_feedback: List[str]
    stakeholder_assessments: Dict[str, Dict[str, Any]]
    behavioral_indicators: Dict[str, float]
    cultural_metrics: Dict[str, float]
    sustainability_assessment: float
    improvement_recommendations: List[str]
    validation_confidence: float


@dataclass
class MitigationTemplate:
    """Template for common mitigation scenarios"""
    id: str
    template_name: str
    resistance_types: List[ResistanceType]
    severity_levels: List[ResistanceSeverity]
    template_strategies: List[MitigationStrategy]
    template_interventions: List[Dict[str, Any]]
    customization_points: List[str]
    success_factors: List[str]
    implementation_guide: List[str]
    resource_estimates: Dict[str, Any]
    timeline_template: Dict[str, int]
    validation_criteria: Dict[str, Any]


@dataclass
class MitigationMetrics:
    """Metrics for tracking mitigation performance"""
    id: str
    plan_id: str
    measurement_date: datetime
    resistance_reduction: float
    engagement_improvement: float
    sentiment_change: float
    behavioral_compliance: float
    stakeholder_satisfaction: float
    intervention_effectiveness: Dict[str, float]
    resource_efficiency: float
    timeline_adherence: float
    cost_effectiveness: float
    sustainability_indicators: Dict[str, float]