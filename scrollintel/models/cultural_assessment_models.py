"""
Cultural Assessment Models for Cultural Transformation Leadership System
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class CulturalDimension(Enum):
    """Key cultural dimensions for assessment"""
    POWER_DISTANCE = "power_distance"
    INDIVIDUALISM_COLLECTIVISM = "individualism_collectivism"
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"
    LONG_TERM_ORIENTATION = "long_term_orientation"
    MASCULINITY_FEMININITY = "masculinity_femininity"
    INDULGENCE_RESTRAINT = "indulgence_restraint"
    INNOVATION_ORIENTATION = "innovation_orientation"
    COLLABORATION_STYLE = "collaboration_style"
    COMMUNICATION_DIRECTNESS = "communication_directness"
    RISK_TOLERANCE = "risk_tolerance"


class SubcultureType(Enum):
    """Types of organizational subcultures"""
    DEPARTMENTAL = "departmental"
    HIERARCHICAL = "hierarchical"
    GENERATIONAL = "generational"
    FUNCTIONAL = "functional"
    GEOGRAPHIC = "geographic"
    PROJECT_BASED = "project_based"


@dataclass
class CulturalValue:
    """Represents a cultural value within the organization"""
    name: str
    description: str
    importance_score: float  # 0.0 to 1.0
    alignment_score: float  # How well organization aligns with this value
    evidence: List[str] = field(default_factory=list)


@dataclass
class CulturalBehavior:
    """Represents observed cultural behaviors"""
    behavior_id: str
    description: str
    frequency: float  # How often this behavior is observed
    impact_score: float  # Impact on culture (positive/negative)
    context: str
    examples: List[str] = field(default_factory=list)


@dataclass
class CulturalNorm:
    """Represents cultural norms and expectations"""
    norm_id: str
    description: str
    enforcement_level: float  # How strictly enforced
    acceptance_level: float  # How widely accepted
    category: str
    violations: List[str] = field(default_factory=list)


@dataclass
class Subculture:
    """Represents a distinct subculture within the organization"""
    subculture_id: str
    name: str
    type: SubcultureType
    members: List[str]  # Employee IDs or identifiers
    characteristics: Dict[str, Any]
    values: List[CulturalValue]
    behaviors: List[CulturalBehavior]
    strength: float  # How distinct/strong this subculture is
    influence: float  # Influence on overall culture


@dataclass
class CulturalHealthMetric:
    """Represents a quantitative cultural health measurement"""
    metric_id: str
    name: str
    value: float
    target_value: Optional[float]
    trend: str  # "improving", "declining", "stable"
    measurement_date: datetime
    data_sources: List[str]
    confidence_level: float


@dataclass
class CultureMap:
    """Comprehensive mapping of organizational culture"""
    organization_id: str
    assessment_date: datetime
    cultural_dimensions: Dict[CulturalDimension, float]
    values: List[CulturalValue]
    behaviors: List[CulturalBehavior]
    norms: List[CulturalNorm]
    subcultures: List[Subculture]
    health_metrics: List[CulturalHealthMetric]
    overall_health_score: float
    assessment_confidence: float
    data_sources: List[str]
    assessor_notes: str = ""


@dataclass
class DimensionAnalysis:
    """Analysis of cultural dimensions"""
    dimension: CulturalDimension
    current_score: float
    ideal_score: Optional[float]
    gap_analysis: str
    contributing_factors: List[str]
    improvement_recommendations: List[str]
    measurement_confidence: float


@dataclass
class CultureData:
    """Raw cultural data for analysis"""
    organization_id: str
    data_type: str  # "survey", "observation", "interview", "document"
    data_points: List[Dict[str, Any]]
    collection_date: datetime
    source: str
    reliability_score: float
    sample_size: Optional[int]


@dataclass
class CulturalAssessmentRequest:
    """Request for cultural assessment"""
    organization_id: str
    assessment_type: str  # "comprehensive", "focused", "quick"
    focus_areas: List[CulturalDimension]
    data_sources: List[str]
    timeline: str
    stakeholders: List[str]
    special_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CulturalAssessmentResult:
    """Result of cultural assessment"""
    request_id: str
    organization_id: str
    culture_map: CultureMap
    dimension_analyses: List[DimensionAnalysis]
    key_findings: List[str]
    recommendations: List[str]
    assessment_summary: str
    confidence_score: float
    completion_date: datetime


@dataclass
class Culture:
    """Represents organizational culture state"""
    organization_id: str
    cultural_dimensions: Dict[str, float]
    values: List[CulturalValue]
    behaviors: List[CulturalBehavior]
    norms: List[CulturalNorm]
    subcultures: List[Subculture]
    health_score: float
    assessment_date: datetime


@dataclass
class Organization:
    """Represents an organization for cultural assessment"""
    id: str
    name: str
    size: str  # startup, small, medium, large, enterprise
    industry: str
    employee_count: int
    departments: List[str]
    locations: List[str]
    founded_year: Optional[int] = None
    culture_maturity: str = "developing"
    complexity_level: str = "medium"
    current_culture: Optional[Culture] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CulturalTransformation:
    """Represents a cultural transformation initiative"""
    id: str
    organization_id: str
    current_culture: Dict[str, Any]
    target_culture: Dict[str, Any]
    vision: Dict[str, Any]
    roadmap: Dict[str, Any]
    interventions: List[Dict[str, Any]]
    progress: float
    start_date: datetime
    target_completion: datetime
    status: str = "active"
    created_by: Optional[str] = None
    last_updated: Optional[datetime] = None