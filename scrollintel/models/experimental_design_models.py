"""
Data models for experimental design system.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum


class ExperimentType(Enum):
    """Types of experiments that can be conducted."""
    CONTROLLED = "controlled"
    OBSERVATIONAL = "observational"
    COMPARATIVE = "comparative"
    LONGITUDINAL = "longitudinal"
    CROSS_SECTIONAL = "cross_sectional"
    FACTORIAL = "factorial"
    RANDOMIZED = "randomized"


class ExperimentStatus(Enum):
    """Status of experiment execution."""
    PLANNED = "planned"
    DESIGNED = "designed"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MethodologyType(Enum):
    """Types of experimental methodologies."""
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    MIXED_METHODS = "mixed_methods"
    COMPUTATIONAL = "computational"
    SIMULATION = "simulation"


@dataclass
class ExperimentalVariable:
    """Represents a variable in an experiment."""
    name: str
    variable_type: str  # independent, dependent, control
    data_type: str  # continuous, categorical, binary
    measurement_unit: Optional[str] = None
    expected_range: Optional[tuple] = None
    description: Optional[str] = None


@dataclass
class ExperimentalCondition:
    """Represents a specific experimental condition."""
    condition_id: str
    name: str
    variables: Dict[str, Any]
    sample_size: int
    description: Optional[str] = None


@dataclass
class Hypothesis:
    """Represents a research hypothesis."""
    hypothesis_id: str
    statement: str
    null_hypothesis: str
    alternative_hypothesis: str
    variables_involved: List[str]
    expected_outcome: Optional[str] = None
    confidence_level: float = 0.95


@dataclass
class ExperimentalProtocol:
    """Detailed experimental protocol and procedures."""
    protocol_id: str
    title: str
    objective: str
    methodology: MethodologyType
    procedures: List[str]
    materials_required: List[str]
    safety_considerations: List[str]
    quality_controls: List[str]
    data_collection_methods: List[str]
    analysis_plan: str
    estimated_duration: timedelta
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceRequirement:
    """Resource requirements for experiments."""
    resource_type: str  # personnel, equipment, materials, computational
    resource_name: str
    quantity_needed: int
    duration_needed: timedelta
    cost_estimate: Optional[float] = None
    availability_constraint: Optional[str] = None


@dataclass
class ExperimentMilestone:
    """Milestone in experiment timeline."""
    milestone_id: str
    name: str
    description: str
    target_date: datetime
    dependencies: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    completion_criteria: List[str] = field(default_factory=list)


@dataclass
class ExperimentPlan:
    """Comprehensive experiment plan."""
    plan_id: str
    title: str
    research_question: str
    hypotheses: List[Hypothesis]
    experiment_type: ExperimentType
    methodology: MethodologyType
    variables: List[ExperimentalVariable]
    conditions: List[ExperimentalCondition]
    protocol: ExperimentalProtocol
    resource_requirements: List[ResourceRequirement]
    timeline: List[ExperimentMilestone]
    success_criteria: List[str]
    risk_factors: List[str]
    mitigation_strategies: List[str]
    status: ExperimentStatus = ExperimentStatus.PLANNED
    created_at: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None


@dataclass
class ValidationStudy:
    """Validation study for experimental design."""
    study_id: str
    experiment_plan_id: str
    validation_type: str  # internal, external, construct, statistical
    validation_methods: List[str]
    validation_criteria: List[str]
    expected_outcomes: List[str]
    validation_timeline: List[ExperimentMilestone]
    resource_requirements: List[ResourceRequirement]
    status: ExperimentStatus = ExperimentStatus.PLANNED


@dataclass
class MethodologyRecommendation:
    """Recommendation for experimental methodology."""
    methodology: MethodologyType
    experiment_type: ExperimentType
    suitability_score: float
    advantages: List[str]
    disadvantages: List[str]
    resource_requirements: List[ResourceRequirement]
    estimated_duration: timedelta
    confidence_level: float


@dataclass
class ExperimentOptimization:
    """Optimization suggestions for experiment design."""
    optimization_id: str
    experiment_plan_id: str
    optimization_type: str  # cost, time, accuracy, reliability
    current_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    optimization_strategies: List[str]
    trade_offs: List[str]
    implementation_steps: List[str]
    confidence_score: float