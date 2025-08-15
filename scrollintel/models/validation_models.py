"""
Data models for Innovation Validation Engine.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid


class ValidationType(str, Enum):
    """Types of validation that can be performed."""
    TECHNICAL_FEASIBILITY = "technical_feasibility"
    MARKET_VIABILITY = "market_viability"
    RESOURCE_AVAILABILITY = "resource_availability"
    RISK_ASSESSMENT = "risk_assessment"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    SCALABILITY_ANALYSIS = "scalability_analysis"
    COST_BENEFIT_ANALYSIS = "cost_benefit_analysis"


class ValidationStatus(str, Enum):
    """Status of validation process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


class ValidationResult(str, Enum):
    """Result of validation."""
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL_APPROVAL = "conditional_approval"
    NEEDS_MODIFICATION = "needs_modification"
    INSUFFICIENT_DATA = "insufficient_data"


class ImpactLevel(str, Enum):
    """Level of impact assessment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    TRANSFORMATIONAL = "transformational"


class SuccessProbability(str, Enum):
    """Success probability categories."""
    VERY_LOW = "very_low"      # 0-20%
    LOW = "low"                # 20-40%
    MEDIUM = "medium"          # 40-60%
    HIGH = "high"              # 60-80%
    VERY_HIGH = "very_high"    # 80-100%


@dataclass
class ValidationCriteria:
    """Criteria for validation assessment."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    weight: float = 1.0  # Importance weight (0.0 to 1.0)
    threshold: float = 0.5  # Minimum score to pass
    validation_type: ValidationType = ValidationType.TECHNICAL_FEASIBILITY
    required: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationScore:
    """Score for a specific validation criteria."""
    criteria_id: str = ""
    score: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.0  # 0.0 to 1.0
    reasoning: str = ""
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Innovation:
    """Innovation concept to be validated."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    category: str = ""
    domain: str = ""
    technology_stack: List[str] = field(default_factory=list)
    target_market: str = ""
    problem_statement: str = ""
    proposed_solution: str = ""
    unique_value_proposition: str = ""
    competitive_advantages: List[str] = field(default_factory=list)
    required_resources: Dict[str, Any] = field(default_factory=dict)
    estimated_timeline: str = ""
    estimated_cost: float = 0.0
    potential_revenue: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationRequest:
    """Request for innovation validation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    innovation_id: str = ""
    validation_types: List[ValidationType] = field(default_factory=list)
    priority: str = "medium"  # low, medium, high, urgent
    deadline: Optional[datetime] = None
    requester: str = ""
    additional_context: Dict[str, Any] = field(default_factory=dict)
    status: ValidationStatus = ValidationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    innovation_id: str = ""
    request_id: str = ""
    overall_score: float = 0.0  # 0.0 to 1.0
    overall_result: ValidationResult = ValidationResult.INSUFFICIENT_DATA
    confidence_level: float = 0.0  # 0.0 to 1.0
    validation_scores: List[ValidationScore] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    risk_mitigation: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    validation_methodology: str = ""
    data_sources: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


@dataclass
class ImpactAssessment:
    """Assessment of innovation impact."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    innovation_id: str = ""
    market_impact: ImpactLevel = ImpactLevel.MEDIUM
    technical_impact: ImpactLevel = ImpactLevel.MEDIUM
    business_impact: ImpactLevel = ImpactLevel.MEDIUM
    social_impact: ImpactLevel = ImpactLevel.MEDIUM
    environmental_impact: ImpactLevel = ImpactLevel.MEDIUM
    economic_impact: ImpactLevel = ImpactLevel.MEDIUM
    market_size: float = 0.0
    addressable_market: float = 0.0
    market_penetration_potential: float = 0.0  # 0.0 to 1.0
    revenue_potential: float = 0.0
    cost_savings_potential: float = 0.0
    job_creation_potential: int = 0
    disruption_potential: float = 0.0  # 0.0 to 1.0
    scalability_factor: float = 0.0  # 0.0 to 1.0
    time_to_market: int = 0  # months
    competitive_advantage_duration: int = 0  # months
    impact_timeline: Dict[str, Any] = field(default_factory=dict)
    quantitative_metrics: Dict[str, float] = field(default_factory=dict)
    qualitative_factors: List[str] = field(default_factory=list)
    stakeholder_impact: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SuccessPrediction:
    """Prediction of innovation success."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    innovation_id: str = ""
    overall_probability: float = 0.0  # 0.0 to 1.0
    probability_category: SuccessProbability = SuccessProbability.MEDIUM
    technical_success_probability: float = 0.0
    market_success_probability: float = 0.0
    financial_success_probability: float = 0.0
    timeline_success_probability: float = 0.0
    key_success_factors: List[str] = field(default_factory=list)
    critical_risks: List[str] = field(default_factory=list)
    success_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    failure_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    confidence_intervals: Dict[str, tuple] = field(default_factory=dict)
    model_accuracy: float = 0.0
    prediction_methodology: str = ""
    data_quality_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


@dataclass
class ValidationMethodology:
    """Methodology used for validation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    validation_types: List[ValidationType] = field(default_factory=list)
    criteria: List[ValidationCriteria] = field(default_factory=list)
    process_steps: List[str] = field(default_factory=list)
    required_data: List[str] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)
    estimated_duration: int = 0  # hours
    accuracy_rate: float = 0.0  # 0.0 to 1.0
    confidence_level: float = 0.0  # 0.0 to 1.0
    applicable_domains: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationContext:
    """Context for validation process."""
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    competitive_landscape: Dict[str, Any] = field(default_factory=dict)
    regulatory_environment: Dict[str, Any] = field(default_factory=dict)
    technology_trends: List[str] = field(default_factory=list)
    economic_indicators: Dict[str, float] = field(default_factory=dict)
    industry_benchmarks: Dict[str, float] = field(default_factory=dict)
    historical_data: Dict[str, Any] = field(default_factory=dict)
    expert_opinions: List[Dict[str, Any]] = field(default_factory=list)
    similar_innovations: List[str] = field(default_factory=list)
    success_patterns: List[str] = field(default_factory=list)
    failure_patterns: List[str] = field(default_factory=list)