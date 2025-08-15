"""
Models for Cultural Change Resistance Detection System
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class ResistanceType(Enum):
    """Types of cultural resistance"""
    ACTIVE_OPPOSITION = "active_opposition"
    PASSIVE_RESISTANCE = "passive_resistance"
    SKEPTICISM = "skepticism"
    FEAR_BASED = "fear_based"
    RESOURCE_BASED = "resource_based"
    COMPETENCY_BASED = "competency_based"
    IDENTITY_BASED = "identity_based"
    SYSTEMIC = "systemic"


class ResistanceSource(Enum):
    """Sources of resistance"""
    INDIVIDUAL = "individual"
    TEAM = "team"
    DEPARTMENT = "department"
    LEADERSHIP = "leadership"
    ORGANIZATIONAL = "organizational"
    EXTERNAL = "external"


class ResistanceSeverity(Enum):
    """Severity levels of resistance"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ResistancePattern:
    """Pattern of cultural resistance"""
    id: str
    pattern_type: ResistanceType
    description: str
    indicators: List[str]
    typical_sources: List[ResistanceSource]
    severity_factors: Dict[str, float]
    detection_methods: List[str]
    created_at: datetime


@dataclass
class ResistanceIndicator:
    """Individual indicator of resistance"""
    id: str
    indicator_type: str
    description: str
    measurement_method: str
    threshold_values: Dict[str, float]
    weight: float
    reliability_score: float


@dataclass
class ResistanceDetection:
    """Detected instance of resistance"""
    id: str
    organization_id: str
    transformation_id: str
    resistance_type: ResistanceType
    source: ResistanceSource
    severity: ResistanceSeverity
    confidence_score: float
    detected_at: datetime
    indicators_triggered: List[str]
    affected_areas: List[str]
    potential_impact: Dict[str, Any]
    detection_method: str
    raw_data: Dict[str, Any]


@dataclass
class ResistanceSourceAnalysis:
    """Source analysis of resistance"""
    id: str
    detection_id: str
    source_type: ResistanceSource
    source_identifier: str
    influence_level: float
    resistance_reasons: List[str]
    historical_patterns: List[str]
    stakeholder_connections: List[str]
    mitigation_receptivity: float


@dataclass
class ResistanceImpactAssessment:
    """Assessment of resistance impact"""
    id: str
    detection_id: str
    transformation_impact: Dict[str, float]
    timeline_impact: Dict[str, int]
    resource_impact: Dict[str, float]
    stakeholder_impact: Dict[str, float]
    success_probability_reduction: float
    cascading_effects: List[str]
    critical_path_disruption: bool
    assessment_confidence: float


@dataclass
class ResistancePrediction:
    """Prediction of future resistance"""
    id: str
    organization_id: str
    transformation_id: str
    predicted_resistance_type: ResistanceType
    predicted_source: ResistanceSource
    probability: float
    expected_timeframe: Dict[str, datetime]
    triggering_factors: List[str]
    prevention_opportunities: List[str]
    prediction_confidence: float
    created_at: datetime


@dataclass
class ResistanceMonitoringConfig:
    """Configuration for resistance monitoring"""
    id: str
    organization_id: str
    monitoring_frequency: str
    detection_sensitivity: float
    alert_thresholds: Dict[ResistanceSeverity, float]
    monitoring_channels: List[str]
    stakeholder_groups: List[str]
    escalation_rules: Dict[str, Any]
    reporting_schedule: str