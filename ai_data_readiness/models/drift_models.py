"""Data models for drift monitoring and detection."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class DriftType(Enum):
    """Types of data drift."""
    COVARIATE_SHIFT = "covariate_shift"
    PRIOR_PROBABILITY_SHIFT = "prior_probability_shift"
    CONCEPT_DRIFT = "concept_drift"
    FEATURE_DRIFT = "feature_drift"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StatisticalTest:
    """Statistical test result for drift detection."""
    test_name: str
    statistic: float
    p_value: float
    threshold: float
    is_significant: bool
    interpretation: str


@dataclass
class DriftAlert:
    """Drift detection alert."""
    id: str
    dataset_id: str
    drift_type: DriftType
    severity: AlertSeverity
    message: str
    affected_features: List[str]
    drift_score: float
    threshold: float
    detected_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class DriftRecommendation:
    """Recommendation for handling detected drift."""
    type: str  # retrain, collect_data, investigate, etc.
    priority: str
    description: str
    action_items: List[str]
    estimated_effort: str
    expected_impact: str


@dataclass
class DriftMetrics:
    """Comprehensive drift metrics."""
    overall_drift_score: float
    feature_drift_scores: Dict[str, float]
    distribution_distances: Dict[str, float]
    statistical_tests: Dict[str, StatisticalTest]
    drift_velocity: float  # rate of change
    drift_magnitude: float  # size of change
    confidence_interval: tuple  # (lower, upper)


@dataclass
class DriftThresholds:
    """Configurable thresholds for drift detection."""
    low_threshold: float = 0.1
    medium_threshold: float = 0.3
    high_threshold: float = 0.5
    critical_threshold: float = 0.7
    statistical_significance: float = 0.05
    minimum_samples: int = 100


@dataclass
class DriftReport:
    """Comprehensive drift analysis report."""
    dataset_id: str
    reference_dataset_id: str
    drift_score: float
    feature_drift_scores: Dict[str, float]
    statistical_tests: Dict[str, StatisticalTest]
    alerts: List[DriftAlert] = field(default_factory=list)
    recommendations: List[DriftRecommendation] = field(default_factory=list)
    metrics: Optional[DriftMetrics] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_severity_level(self) -> AlertSeverity:
        """Determine overall severity level based on drift score."""
        if self.drift_score >= 0.7:
            return AlertSeverity.CRITICAL
        elif self.drift_score >= 0.5:
            return AlertSeverity.HIGH
        elif self.drift_score >= 0.3:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def has_significant_drift(self, threshold: float = 0.3) -> bool:
        """Check if drift exceeds significance threshold."""
        return self.drift_score >= threshold