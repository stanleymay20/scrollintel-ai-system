"""
Risk-Benefit Analysis Models for Crisis Leadership Excellence

This module defines data models for risk assessment, benefit analysis,
and trade-off evaluation during crisis situations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any
from uuid import uuid4


class RiskLevel(Enum):
    """Risk severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class BenefitType(Enum):
    """Types of benefits that can be achieved"""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    REPUTATIONAL = "reputational"
    STAKEHOLDER = "stakeholder"
    COMPETITIVE = "competitive"


class UncertaintyLevel(Enum):
    """Levels of uncertainty in analysis"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class RiskFactor:
    """Individual risk factor in crisis response"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    category: str = ""
    probability: float = 0.0  # 0.0 to 1.0
    impact_severity: RiskLevel = RiskLevel.MEDIUM
    potential_impact: str = ""
    time_horizon: str = ""  # immediate, short_term, long_term
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.MODERATE
    mitigation_strategies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BenefitFactor:
    """Individual benefit factor in crisis response"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    benefit_type: BenefitType = BenefitType.OPERATIONAL
    expected_value: float = 0.0
    probability_of_realization: float = 0.0  # 0.0 to 1.0
    time_to_realization: str = ""  # immediate, short_term, long_term
    sustainability: str = ""  # temporary, medium_term, permanent
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.MODERATE
    optimization_strategies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ResponseOption:
    """Crisis response option for evaluation"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    category: str = ""
    implementation_complexity: str = ""  # low, medium, high
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    time_to_implement: str = ""
    risks: List[RiskFactor] = field(default_factory=list)
    benefits: List[BenefitFactor] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MitigationStrategy:
    """Risk mitigation strategy"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    target_risks: List[str] = field(default_factory=list)  # Risk IDs
    effectiveness_score: float = 0.0  # 0.0 to 1.0
    implementation_cost: float = 0.0
    implementation_time: str = ""
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)
    success_probability: float = 0.0  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TradeOffAnalysis:
    """Trade-off analysis between competing factors"""
    id: str = field(default_factory=lambda: str(uuid4()))
    option_a_id: str = ""
    option_b_id: str = ""
    comparison_criteria: List[str] = field(default_factory=list)
    trade_off_factors: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    confidence_level: float = 0.0  # 0.0 to 1.0
    decision_rationale: str = ""
    sensitivity_analysis: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RiskBenefitEvaluation:
    """Complete risk-benefit evaluation of response options"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    response_options: List[ResponseOption] = field(default_factory=list)
    evaluation_criteria: Dict[str, float] = field(default_factory=dict)
    risk_tolerance: RiskLevel = RiskLevel.MEDIUM
    time_pressure: str = ""  # immediate, urgent, moderate, low
    stakeholder_priorities: Dict[str, float] = field(default_factory=dict)
    recommended_option: Optional[str] = None
    confidence_score: float = 0.0  # 0.0 to 1.0
    uncertainty_factors: List[str] = field(default_factory=list)
    sensitivity_analysis: Dict[str, Any] = field(default_factory=dict)
    trade_off_analyses: List[TradeOffAnalysis] = field(default_factory=list)
    mitigation_plan: List[MitigationStrategy] = field(default_factory=list)
    monitoring_requirements: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationResult:
    """Result of benefit optimization analysis"""
    id: str = field(default_factory=lambda: str(uuid4()))
    evaluation_id: str = ""
    optimization_objective: str = ""
    optimized_benefits: List[BenefitFactor] = field(default_factory=list)
    optimization_strategies: List[str] = field(default_factory=list)
    expected_improvement: float = 0.0
    implementation_requirements: Dict[str, Any] = field(default_factory=dict)
    success_probability: float = 0.0  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.now)