"""
Transformation Outcome Models

Data models for cultural transformation outcome testing and validation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class TransformationStatus(Enum):
    """Status of transformation"""
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VALIDATED = "validated"
    FAILED = "failed"


class SuccessLevel(Enum):
    """Level of transformation success"""
    EXCEPTIONAL = "exceptional"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    FAILED = "failed"


class SustainabilityLevel(Enum):
    """Level of cultural sustainability"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    CONCERNING = "concerning"
    POOR = "poor"


@dataclass
class TransformationOutcome:
    """Complete transformation outcome data"""
    transformation_id: str
    organization_id: str
    start_date: datetime
    end_date: datetime
    status: TransformationStatus
    target_metrics: Dict[str, float]
    achieved_metrics: Dict[str, float]
    success_criteria: List[Dict[str, Any]]
    investment_data: Dict[str, Any]
    benefit_data: Dict[str, Any]
    sustainability_data: Dict[str, Any]
    validation_results: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


@dataclass
class SuccessMetric:
    """Individual success metric"""
    metric_id: str
    metric_name: str
    target_value: float
    achieved_value: float
    threshold: float
    weight: float
    success_margin: float
    achievement_percentage: float
    success_status: str


@dataclass
class TransformationGoal:
    """Transformation goal definition"""
    goal_id: str
    goal_name: str
    description: str
    target_metrics: List[str]
    success_criteria: Dict[str, Any]
    priority: str
    weight: float
    achievement_status: str
    achievement_score: float


@dataclass
class ImpactMeasurement:
    """Impact measurement data"""
    measurement_id: str
    transformation_id: str
    metric_name: str
    baseline_value: float
    post_transformation_value: float
    improvement_amount: float
    improvement_percentage: float
    statistical_significance: float
    confidence_interval: tuple
    attribution_score: float
    measurement_date: datetime


@dataclass
class BenchmarkData:
    """Benchmark comparison data"""
    benchmark_id: str
    industry: str
    organization_size: str
    transformation_type: str
    benchmark_metrics: Dict[str, float]
    percentile_rankings: Dict[str, float]
    best_practices: Dict[str, Any]
    data_source: str
    last_updated: datetime


@dataclass
class QualityIndicator:
    """Quality indicator for transformation"""
    indicator_id: str
    indicator_name: str
    category: str  # depth, integration, sustainability
    measurement_method: str
    score: float
    confidence_level: float
    evidence: List[str]
    measurement_date: datetime


@dataclass
class SustainabilityMetrics:
    """Cultural sustainability metrics"""
    metrics_id: str
    transformation_id: str
    stability_score: float
    trend_consistency: float
    reinforcement_strength: float
    decay_resistance: float
    sustainability_level: SustainabilityLevel
    risk_factors: List[str]
    measurement_period: int  # days
    confidence_level: float
    last_measured: datetime


@dataclass
class ReinforcementMechanism:
    """Cultural reinforcement mechanism"""
    mechanism_id: str
    mechanism_type: str
    description: str
    strength_score: float
    consistency_score: float
    effectiveness_score: float
    implementation_date: datetime
    last_assessment: datetime
    status: str


@dataclass
class CulturalDriftIndicator:
    """Cultural drift indicator"""
    indicator_id: str
    transformation_id: str
    cultural_dimension: str
    baseline_value: float
    current_value: float
    drift_magnitude: float
    drift_direction: str  # positive, negative, stable
    drift_rate: float  # per month
    detection_date: datetime
    severity_level: str


@dataclass
class ROICalculation:
    """ROI calculation data"""
    calculation_id: str
    transformation_id: str
    total_investment: float
    total_benefits: float
    roi_percentage: float
    payback_period: float  # months
    net_present_value: float
    benefit_cost_ratio: float
    calculation_method: str
    assumptions: Dict[str, Any]
    confidence_level: float
    calculated_date: datetime


@dataclass
class ValueDriver:
    """Value creation driver"""
    driver_id: str
    driver_name: str
    driver_type: str  # tangible, intangible
    annual_value: float
    contribution_percentage: float
    sustainability_score: float
    measurement_confidence: float
    evidence_sources: List[str]


@dataclass
class RiskFactor:
    """Risk factor for transformation outcomes"""
    risk_id: str
    risk_name: str
    risk_category: str
    probability: float
    impact_score: float
    risk_score: float
    mitigation_strategies: List[str]
    monitoring_indicators: List[str]
    status: str
    last_assessed: datetime


@dataclass
class ValidationTest:
    """Validation test result"""
    test_id: str
    test_name: str
    test_category: str
    test_description: str
    passed: bool
    score: float
    confidence: float
    details: Dict[str, Any]
    evidence: List[str]
    recommendations: List[str]
    executed_date: datetime


@dataclass
class OutcomeReport:
    """Comprehensive outcome report"""
    report_id: str
    transformation_id: str
    report_type: str
    success_assessment: Dict[str, Any]
    sustainability_assessment: Dict[str, Any]
    roi_assessment: Dict[str, Any]
    benchmark_comparison: Dict[str, Any]
    validation_results: List[ValidationTest]
    recommendations: List[str]
    executive_summary: str
    generated_date: datetime
    report_version: str


@dataclass
class LongTermProjection:
    """Long-term value projection"""
    projection_id: str
    transformation_id: str
    projection_period: int  # months
    projected_benefits: List[float]
    cumulative_value: float
    sustainability_forecast: float
    risk_adjusted_value: float
    confidence_intervals: Dict[str, tuple]
    key_assumptions: Dict[str, Any]
    milestone_projections: List[Dict[str, Any]]
    created_date: datetime


@dataclass
class CompetitiveAnalysis:
    """Competitive analysis data"""
    analysis_id: str
    organization_id: str
    competitor_benchmarks: Dict[str, float]
    market_position: str
    competitive_advantages: List[str]
    improvement_gaps: List[str]
    strategic_recommendations: List[str]
    analysis_date: datetime


@dataclass
class StakeholderImpact:
    """Stakeholder impact assessment"""
    impact_id: str
    transformation_id: str
    stakeholder_group: str
    impact_category: str
    impact_score: float
    satisfaction_change: float
    engagement_change: float
    behavioral_changes: List[str]
    feedback_summary: str
    measurement_date: datetime


@dataclass
class CulturalHealthMetrics:
    """Cultural health metrics"""
    metrics_id: str
    organization_id: str
    overall_health_score: float
    dimension_scores: Dict[str, float]
    engagement_index: float
    alignment_score: float
    innovation_index: float
    collaboration_score: float
    accountability_score: float
    adaptability_score: float
    measurement_date: datetime
    confidence_level: float


@dataclass
class TransformationTimeline:
    """Transformation timeline and milestones"""
    timeline_id: str
    transformation_id: str
    planned_milestones: List[Dict[str, Any]]
    achieved_milestones: List[Dict[str, Any]]
    timeline_adherence: float
    critical_path_items: List[str]
    delays_and_reasons: List[Dict[str, Any]]
    acceleration_opportunities: List[str]
    last_updated: datetime


# Database table models (if using SQLAlchemy or similar)
class TransformationOutcomeTable:
    """Database table structure for transformation outcomes"""
    
    def __init__(self):
        self.table_name = "transformation_outcomes"
        self.columns = {
            "id": "VARCHAR(50) PRIMARY KEY",
            "transformation_id": "VARCHAR(50) NOT NULL",
            "organization_id": "VARCHAR(50) NOT NULL",
            "start_date": "TIMESTAMP NOT NULL",
            "end_date": "TIMESTAMP",
            "status": "VARCHAR(20) NOT NULL",
            "target_metrics": "JSON",
            "achieved_metrics": "JSON",
            "success_criteria": "JSON",
            "investment_data": "JSON",
            "benefit_data": "JSON",
            "sustainability_data": "JSON",
            "validation_results": "JSON",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
        }
        self.indexes = [
            "INDEX idx_transformation_id (transformation_id)",
            "INDEX idx_organization_id (organization_id)",
            "INDEX idx_status (status)",
            "INDEX idx_end_date (end_date)"
        ]


class ValidationTestTable:
    """Database table structure for validation tests"""
    
    def __init__(self):
        self.table_name = "validation_tests"
        self.columns = {
            "id": "VARCHAR(50) PRIMARY KEY",
            "test_id": "VARCHAR(50) NOT NULL",
            "transformation_id": "VARCHAR(50) NOT NULL",
            "test_name": "VARCHAR(100) NOT NULL",
            "test_category": "VARCHAR(50) NOT NULL",
            "passed": "BOOLEAN NOT NULL",
            "score": "DECIMAL(5,4)",
            "confidence": "DECIMAL(5,4)",
            "details": "JSON",
            "evidence": "JSON",
            "recommendations": "JSON",
            "executed_date": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        }
        self.indexes = [
            "INDEX idx_transformation_id (transformation_id)",
            "INDEX idx_test_category (test_category)",
            "INDEX idx_passed (passed)",
            "INDEX idx_executed_date (executed_date)"
        ]


class ROICalculationTable:
    """Database table structure for ROI calculations"""
    
    def __init__(self):
        self.table_name = "roi_calculations"
        self.columns = {
            "id": "VARCHAR(50) PRIMARY KEY",
            "calculation_id": "VARCHAR(50) NOT NULL",
            "transformation_id": "VARCHAR(50) NOT NULL",
            "total_investment": "DECIMAL(15,2)",
            "total_benefits": "DECIMAL(15,2)",
            "roi_percentage": "DECIMAL(8,4)",
            "payback_period": "DECIMAL(8,2)",
            "net_present_value": "DECIMAL(15,2)",
            "benefit_cost_ratio": "DECIMAL(8,4)",
            "calculation_method": "VARCHAR(50)",
            "assumptions": "JSON",
            "confidence_level": "DECIMAL(5,4)",
            "calculated_date": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        }
        self.indexes = [
            "INDEX idx_transformation_id (transformation_id)",
            "INDEX idx_roi_percentage (roi_percentage)",
            "INDEX idx_calculated_date (calculated_date)"
        ]


# Utility functions for model operations
def create_transformation_outcome(transformation_data: Dict[str, Any]) -> TransformationOutcome:
    """Create a TransformationOutcome instance from dictionary data"""
    return TransformationOutcome(
        transformation_id=transformation_data.get("transformation_id", ""),
        organization_id=transformation_data.get("organization_id", ""),
        start_date=transformation_data.get("start_date", datetime.now()),
        end_date=transformation_data.get("end_date", datetime.now()),
        status=TransformationStatus(transformation_data.get("status", "planning")),
        target_metrics=transformation_data.get("target_metrics", {}),
        achieved_metrics=transformation_data.get("achieved_metrics", {}),
        success_criteria=transformation_data.get("success_criteria", []),
        investment_data=transformation_data.get("investment_data", {}),
        benefit_data=transformation_data.get("benefit_data", {}),
        sustainability_data=transformation_data.get("sustainability_data", {}),
        validation_results=transformation_data.get("validation_results", []),
        created_at=transformation_data.get("created_at", datetime.now()),
        updated_at=transformation_data.get("updated_at", datetime.now())
    )


def create_validation_test(test_data: Dict[str, Any]) -> ValidationTest:
    """Create a ValidationTest instance from dictionary data"""
    return ValidationTest(
        test_id=test_data.get("test_id", ""),
        test_name=test_data.get("test_name", ""),
        test_category=test_data.get("test_category", ""),
        test_description=test_data.get("test_description", ""),
        passed=test_data.get("passed", False),
        score=test_data.get("score", 0.0),
        confidence=test_data.get("confidence", 0.0),
        details=test_data.get("details", {}),
        evidence=test_data.get("evidence", []),
        recommendations=test_data.get("recommendations", []),
        executed_date=test_data.get("executed_date", datetime.now())
    )


def create_roi_calculation(roi_data: Dict[str, Any]) -> ROICalculation:
    """Create an ROICalculation instance from dictionary data"""
    return ROICalculation(
        calculation_id=roi_data.get("calculation_id", ""),
        transformation_id=roi_data.get("transformation_id", ""),
        total_investment=roi_data.get("total_investment", 0.0),
        total_benefits=roi_data.get("total_benefits", 0.0),
        roi_percentage=roi_data.get("roi_percentage", 0.0),
        payback_period=roi_data.get("payback_period", 0.0),
        net_present_value=roi_data.get("net_present_value", 0.0),
        benefit_cost_ratio=roi_data.get("benefit_cost_ratio", 0.0),
        calculation_method=roi_data.get("calculation_method", ""),
        assumptions=roi_data.get("assumptions", {}),
        confidence_level=roi_data.get("confidence_level", 0.0),
        calculated_date=roi_data.get("calculated_date", datetime.now())
    )