"""
Impact Assessment Models for Cultural Transformation Leadership

This module defines data models for assessing the impact of cultural changes
on organizational performance and outcomes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class ImpactCategory(Enum):
    """Categories of organizational impact"""
    PERFORMANCE = "performance"
    ENGAGEMENT = "engagement"
    PRODUCTIVITY = "productivity"
    INNOVATION = "innovation"
    RETENTION = "retention"
    SATISFACTION = "satisfaction"
    COLLABORATION = "collaboration"
    LEADERSHIP = "leadership"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"


class ImpactType(Enum):
    """Types of impact measurement"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    LEADING = "leading"
    LAGGING = "lagging"
    QUALITATIVE = "qualitative"
    QUANTITATIVE = "quantitative"


class CorrelationStrength(Enum):
    """Strength of correlation between cultural change and impact"""
    VERY_STRONG = "very_strong"  # 0.8-1.0
    STRONG = "strong"  # 0.6-0.8
    MODERATE = "moderate"  # 0.4-0.6
    WEAK = "weak"  # 0.2-0.4
    VERY_WEAK = "very_weak"  # 0.0-0.2
    NONE = "none"  # No correlation


@dataclass
class ImpactMetric:
    """Individual impact metric"""
    id: str
    name: str
    description: str
    category: ImpactCategory
    impact_type: ImpactType
    baseline_value: float
    current_value: float
    target_value: Optional[float]
    unit: str
    measurement_date: datetime
    data_source: str
    confidence_level: float  # 0.0 to 1.0
    
    @property
    def change_from_baseline(self) -> float:
        """Calculate change from baseline"""
        if self.baseline_value == 0:
            return 0.0
        return ((self.current_value - self.baseline_value) / self.baseline_value) * 100
    
    @property
    def target_progress(self) -> Optional[float]:
        """Calculate progress toward target"""
        if self.target_value is None or self.baseline_value == self.target_value:
            return None
        
        total_change_needed = self.target_value - self.baseline_value
        current_change = self.current_value - self.baseline_value
        
        if total_change_needed == 0:
            return 100.0
        
        return (current_change / total_change_needed) * 100


@dataclass
class CulturalChange:
    """Represents a specific cultural change"""
    change_id: str
    name: str
    description: str
    change_type: str  # "value_shift", "behavior_change", "norm_evolution"
    implementation_date: datetime
    affected_population: str
    change_magnitude: float  # 0.0 to 1.0
    measurement_method: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class ImpactCorrelation:
    """Correlation between cultural change and organizational impact"""
    correlation_id: str
    cultural_change: CulturalChange
    impact_metric: ImpactMetric
    correlation_coefficient: float  # -1.0 to 1.0
    correlation_strength: CorrelationStrength
    statistical_significance: float  # p-value
    confidence_interval: tuple[float, float]
    analysis_method: str
    sample_size: int
    analysis_date: datetime
    
    @property
    def is_statistically_significant(self) -> bool:
        """Check if correlation is statistically significant"""
        return self.statistical_significance < 0.05


@dataclass
class ImpactAnalysis:
    """Comprehensive impact analysis"""
    analysis_id: str
    transformation_id: str
    analysis_date: datetime
    time_period: str  # "1_month", "3_months", "6_months", "1_year"
    cultural_changes: List[CulturalChange]
    impact_metrics: List[ImpactMetric]
    correlations: List[ImpactCorrelation]
    overall_impact_score: float  # 0.0 to 100.0
    key_findings: List[str]
    unexpected_impacts: List[str]
    analysis_confidence: float
    
    @property
    def positive_impacts(self) -> List[ImpactMetric]:
        """Get metrics showing positive impact"""
        return [m for m in self.impact_metrics if m.change_from_baseline > 0]
    
    @property
    def negative_impacts(self) -> List[ImpactMetric]:
        """Get metrics showing negative impact"""
        return [m for m in self.impact_metrics if m.change_from_baseline < 0]
    
    @property
    def strong_correlations(self) -> List[ImpactCorrelation]:
        """Get strong correlations"""
        return [c for c in self.correlations 
                if c.correlation_strength in [CorrelationStrength.STRONG, CorrelationStrength.VERY_STRONG]]


@dataclass
class ImpactPrediction:
    """Prediction of future impact based on current trends"""
    prediction_id: str
    transformation_id: str
    prediction_date: datetime
    forecast_period: str  # "3_months", "6_months", "1_year"
    predicted_metrics: Dict[str, float]  # metric_id -> predicted_value
    prediction_confidence: Dict[str, float]  # metric_id -> confidence
    assumptions: List[str]
    risk_factors: List[str]
    model_used: str
    
    @property
    def high_confidence_predictions(self) -> Dict[str, float]:
        """Get predictions with high confidence (>0.7)"""
        return {
            metric_id: value for metric_id, value in self.predicted_metrics.items()
            if self.prediction_confidence.get(metric_id, 0) > 0.7
        }


@dataclass
class ImpactOptimization:
    """Optimization recommendations based on impact analysis"""
    optimization_id: str
    transformation_id: str
    analysis_date: datetime
    target_improvements: Dict[str, float]  # metric_id -> target_improvement
    recommended_actions: List[Dict[str, Any]]
    expected_outcomes: Dict[str, float]  # metric_id -> expected_value
    implementation_priority: List[str]  # action_ids in priority order
    resource_requirements: Dict[str, Any]
    timeline: Dict[str, str]  # action_id -> timeline
    success_probability: Dict[str, float]  # action_id -> probability


@dataclass
class ImpactReport:
    """Comprehensive impact assessment report"""
    report_id: str
    transformation_id: str
    report_date: datetime
    reporting_period: str
    executive_summary: str
    impact_analysis: ImpactAnalysis
    predictions: List[ImpactPrediction]
    optimizations: List[ImpactOptimization]
    roi_analysis: Dict[str, float]
    recommendations: List[str]
    next_assessment_date: datetime
    
    @property
    def overall_success_indicator(self) -> str:
        """Get overall success indicator"""
        if self.impact_analysis.overall_impact_score >= 80:
            return "excellent"
        elif self.impact_analysis.overall_impact_score >= 60:
            return "good"
        elif self.impact_analysis.overall_impact_score >= 40:
            return "moderate"
        elif self.impact_analysis.overall_impact_score >= 20:
            return "poor"
        else:
            return "very_poor"


@dataclass
class ImpactDashboard:
    """Dashboard data for impact visualization"""
    dashboard_id: str
    transformation_id: str
    dashboard_date: datetime
    impact_summary: Dict[str, Any]
    metric_trends: Dict[str, List[float]]
    correlation_matrix: Dict[str, Dict[str, float]]
    impact_heatmap: Dict[str, Dict[str, float]]
    key_insights: List[str]
    alert_indicators: List[Dict[str, Any]]
    performance_indicators: Dict[str, float]


@dataclass
class ImpactAlert:
    """Alert for significant impact changes"""
    alert_id: str
    transformation_id: str
    alert_type: str  # "significant_improvement", "concerning_decline", "unexpected_impact"
    severity: str  # "low", "medium", "high", "critical"
    metric_affected: str
    current_value: float
    threshold_value: float
    change_magnitude: float
    message: str
    created_date: datetime
    resolved_date: Optional[datetime] = None
    action_required: bool = True
    assigned_to: Optional[str] = None