"""
Cultural Strategic Integration Models

Data models for integrating cultural transformation with strategic planning systems.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class StrategicObjectiveType(Enum):
    """Types of strategic objectives"""
    GROWTH = "growth"
    INNOVATION = "innovation"
    EFFICIENCY = "efficiency"
    MARKET_EXPANSION = "market_expansion"
    DIGITAL_TRANSFORMATION = "digital_transformation"
    SUSTAINABILITY = "sustainability"
    TALENT_ACQUISITION = "talent_acquisition"
    CUSTOMER_EXPERIENCE = "customer_experience"


class CulturalAlignment(Enum):
    """Levels of cultural alignment with strategic objectives"""
    FULLY_ALIGNED = "fully_aligned"
    MOSTLY_ALIGNED = "mostly_aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    MISALIGNED = "misaligned"
    CONFLICTING = "conflicting"


class ImpactLevel(Enum):
    """Levels of cultural impact on strategic initiatives"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class StrategicObjective:
    """Strategic objective model"""
    id: str
    name: str
    description: str
    objective_type: StrategicObjectiveType
    target_date: datetime
    success_metrics: List[str]
    priority_level: int
    owner: str
    cultural_requirements: List[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class CulturalStrategicAlignment:
    """Cultural alignment with strategic objectives"""
    id: str
    objective_id: str
    cultural_dimension: str
    alignment_level: CulturalAlignment
    alignment_score: float  # 0.0 to 1.0
    gap_analysis: Dict[str, Any]
    recommendations: List[str]
    assessed_by: str
    assessment_date: datetime


@dataclass
class StrategicInitiative:
    """Strategic initiative model"""
    id: str
    name: str
    description: str
    objective_ids: List[str]
    start_date: datetime
    end_date: datetime
    budget: float
    team_size: int
    cultural_impact_level: ImpactLevel
    cultural_requirements: List[str]
    success_criteria: List[str]
    status: str


@dataclass
class CulturalImpactAssessment:
    """Assessment of cultural impact on strategic initiatives"""
    id: str
    initiative_id: str
    impact_level: ImpactLevel
    impact_score: float  # 0.0 to 1.0
    cultural_enablers: List[str]
    cultural_barriers: List[str]
    mitigation_strategies: List[str]
    success_probability: float
    assessment_date: datetime
    assessor: str


@dataclass
class CultureAwareDecision:
    """Culture-aware strategic decision"""
    id: str
    decision_context: str
    strategic_options: List[Dict[str, Any]]
    cultural_considerations: Dict[str, Any]
    recommended_option: str
    cultural_rationale: str
    risk_assessment: Dict[str, float]
    implementation_plan: List[str]
    decision_date: datetime
    decision_maker: str


@dataclass
class StrategicCulturalMetric:
    """Metrics for strategic cultural integration"""
    id: str
    metric_name: str
    metric_type: str
    current_value: float
    target_value: float
    measurement_date: datetime
    trend_direction: str
    strategic_relevance: float
    cultural_impact: float


@dataclass
class IntegrationReport:
    """Report on cultural-strategic integration"""
    id: str
    report_type: str
    reporting_period: str
    alignment_summary: Dict[str, Any]
    impact_assessments: List[CulturalImpactAssessment]
    recommendations: List[str]
    success_metrics: Dict[str, float]
    generated_at: datetime
    generated_by: str