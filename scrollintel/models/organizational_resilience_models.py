"""
Organizational Resilience Models

Data models for organizational resilience assessment, enhancement, and monitoring.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class ResilienceLevel(Enum):
    """Organizational resilience levels"""
    FRAGILE = "fragile"
    BASIC = "basic"
    ROBUST = "robust"
    ANTIFRAGILE = "antifragile"


class ResilienceCategory(Enum):
    """Categories of organizational resilience"""
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    TECHNOLOGICAL = "technological"
    HUMAN_CAPITAL = "human_capital"
    STRATEGIC = "strategic"
    CULTURAL = "cultural"


class ResilienceMetricType(Enum):
    """Types of resilience metrics"""
    RECOVERY_TIME = "recovery_time"
    ADAPTATION_SPEED = "adaptation_speed"
    STRESS_TOLERANCE = "stress_tolerance"
    LEARNING_CAPACITY = "learning_capacity"
    REDUNDANCY_LEVEL = "redundancy_level"


@dataclass
class ResilienceAssessment:
    """Organizational resilience assessment"""
    id: str
    organization_id: str
    assessment_date: datetime
    overall_resilience_level: ResilienceLevel
    category_scores: Dict[ResilienceCategory, float]
    strengths: List[str]
    vulnerabilities: List[str]
    improvement_areas: List[str]
    assessment_methodology: str
    confidence_score: float


@dataclass
class ResilienceMetric:
    """Individual resilience metric"""
    id: str
    metric_type: ResilienceMetricType
    category: ResilienceCategory
    current_value: float
    target_value: float
    measurement_unit: str
    measurement_date: datetime
    trend_direction: str
    benchmark_comparison: float


@dataclass
class ResilienceStrategy:
    """Resilience building strategy"""
    id: str
    strategy_name: str
    target_categories: List[ResilienceCategory]
    objectives: List[str]
    initiatives: List[str]
    timeline: Dict[str, datetime]
    resource_requirements: Dict[str, Any]
    success_metrics: List[str]
    risk_factors: List[str]
    expected_impact: Dict[ResilienceCategory, float]


@dataclass
class ResilienceInitiative:
    """Individual resilience building initiative"""
    id: str
    strategy_id: str
    initiative_name: str
    description: str
    category: ResilienceCategory
    priority_level: str
    start_date: datetime
    target_completion: datetime
    actual_completion: Optional[datetime]
    status: str
    progress_percentage: float
    resource_allocation: Dict[str, Any]
    success_indicators: List[str]
    challenges: List[str]


@dataclass
class ResilienceMonitoringData:
    """Resilience monitoring data point"""
    id: str
    monitoring_date: datetime
    category: ResilienceCategory
    metric_values: Dict[ResilienceMetricType, float]
    alert_triggers: List[str]
    trend_analysis: Dict[str, Any]
    anomaly_detection: List[str]
    recommendations: List[str]


@dataclass
class ResilienceImprovement:
    """Resilience improvement recommendation"""
    id: str
    assessment_id: str
    improvement_type: str
    category: ResilienceCategory
    priority: str
    description: str
    implementation_steps: List[str]
    estimated_timeline: str
    resource_requirements: Dict[str, Any]
    expected_benefits: List[str]
    success_metrics: List[str]
    dependencies: List[str]


@dataclass
class ResilienceCapability:
    """Organizational resilience capability"""
    id: str
    capability_name: str
    category: ResilienceCategory
    current_maturity: float
    target_maturity: float
    assessment_criteria: List[str]
    development_path: List[str]
    required_resources: Dict[str, Any]
    timeline: Dict[str, datetime]
    success_indicators: List[str]


@dataclass
class ResilienceReport:
    """Comprehensive resilience report"""
    id: str
    report_date: datetime
    organization_id: str
    executive_summary: str
    overall_resilience_score: float
    category_breakdown: Dict[ResilienceCategory, Dict[str, Any]]
    trend_analysis: Dict[str, Any]
    benchmark_comparison: Dict[str, float]
    key_findings: List[str]
    recommendations: List[ResilienceImprovement]
    action_plan: List[str]
    next_assessment_date: datetime