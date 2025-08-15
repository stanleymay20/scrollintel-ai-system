"""
Culture Maintenance Models

Data models for cultural sustainability assessment and maintenance framework.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class MaintenanceStatus(Enum):
    STABLE = "stable"
    AT_RISK = "at_risk"
    DECLINING = "declining"
    CRITICAL = "critical"


class SustainabilityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"


@dataclass
class CultureHealthIndicator:
    """Individual culture health metric"""
    indicator_id: str
    name: str
    current_value: float
    target_value: float
    trend: str  # "improving", "stable", "declining"
    importance_weight: float
    measurement_date: datetime


@dataclass
class SustainabilityAssessment:
    """Assessment of cultural change sustainability"""
    assessment_id: str
    organization_id: str
    transformation_id: str
    sustainability_level: SustainabilityLevel
    risk_factors: List[str]
    protective_factors: List[str]
    health_indicators: List[CultureHealthIndicator]
    overall_score: float
    assessment_date: datetime
    next_assessment_due: datetime


@dataclass
class MaintenanceStrategy:
    """Strategy for maintaining cultural changes"""
    strategy_id: str
    organization_id: str
    target_culture_elements: List[str]
    maintenance_activities: List[Dict[str, Any]]
    monitoring_schedule: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    success_metrics: List[str]
    review_frequency: str
    created_date: datetime


@dataclass
class CultureMaintenancePlan:
    """Comprehensive culture maintenance plan"""
    plan_id: str
    organization_id: str
    sustainability_assessment: SustainabilityAssessment
    maintenance_strategies: List[MaintenanceStrategy]
    monitoring_framework: Dict[str, Any]
    intervention_triggers: List[Dict[str, Any]]
    resource_allocation: Dict[str, Any]
    timeline: Dict[str, Any]
    status: MaintenanceStatus
    created_date: datetime
    last_updated: datetime


@dataclass
class MaintenanceIntervention:
    """Intervention to address culture maintenance issues"""
    intervention_id: str
    plan_id: str
    trigger_event: str
    intervention_type: str
    target_areas: List[str]
    actions: List[Dict[str, Any]]
    expected_outcomes: List[str]
    timeline: Dict[str, Any]
    resources_needed: Dict[str, Any]
    status: str
    created_date: datetime


@dataclass
class LongTermMonitoringResult:
    """Results from long-term culture health monitoring"""
    monitoring_id: str
    organization_id: str
    monitoring_period: Dict[str, datetime]
    health_trends: Dict[str, List[float]]
    sustainability_metrics: Dict[str, float]
    risk_indicators: List[str]
    recommendations: List[str]
    next_actions: List[Dict[str, Any]]
    monitoring_date: datetime