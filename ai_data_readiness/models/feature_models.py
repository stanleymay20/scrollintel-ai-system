"""Data models for feature engineering and recommendations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum


class FeatureType(Enum):
    """Types of features."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    BINARY = "binary"
    ORDINAL = "ordinal"


class TransformationType(Enum):
    """Types of feature transformations."""
    SCALING = "scaling"
    NORMALIZATION = "normalization"
    ENCODING = "encoding"
    BINNING = "binning"
    POLYNOMIAL = "polynomial"
    INTERACTION = "interaction"
    AGGREGATION = "aggregation"
    EXTRACTION = "extraction"


class ModelType(Enum):
    """Types of ML models for feature optimization."""
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"


@dataclass
class FeatureInfo:
    """Information about a feature."""
    name: str
    type: FeatureType
    data_type: str  # int, float, string, etc.
    missing_rate: float
    unique_values: int
    cardinality: Optional[int] = None
    distribution_stats: Dict[str, Any] = field(default_factory=dict)
    correlation_with_target: Optional[float] = None
    importance_score: Optional[float] = None


@dataclass
class TransformationStep:
    """A single transformation step."""
    transformation_type: TransformationType
    parameters: Dict[str, Any]
    input_features: List[str]
    output_features: List[str]
    description: str
    rationale: str


@dataclass
class EncodingStrategy:
    """Strategy for encoding categorical variables."""
    feature_name: str
    encoding_type: str  # one_hot, label, target, binary, etc.
    parameters: Dict[str, Any]
    expected_dimensions: int
    handle_unknown: str = "ignore"
    drop_first: bool = False


@dataclass
class TemporalFeatures:
    """Temporal feature engineering recommendations."""
    time_column: str
    features_to_create: List[str]
    aggregation_windows: List[str]  # 1h, 1d, 1w, etc.
    lag_features: List[int]
    seasonal_features: bool = True
    trend_features: bool = True


@dataclass
class FeatureRecommendation:
    """Feature engineering recommendation."""
    feature_name: str
    recommendation_type: str
    transformation: TransformationStep
    expected_impact: float  # 0-1 scale
    confidence: float  # 0-1 scale
    rationale: str
    implementation_complexity: str  # low, medium, high


@dataclass
class FeatureRecommendations:
    """Collection of feature engineering recommendations."""
    dataset_id: str
    model_type: ModelType
    target_column: Optional[str]
    recommendations: List[FeatureRecommendation] = field(default_factory=list)
    encoding_strategies: List[EncodingStrategy] = field(default_factory=list)
    temporal_features: Optional[TemporalFeatures] = None
    feature_selection_recommendations: List[str] = field(default_factory=list)
    dimensionality_reduction_recommendation: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_high_impact_recommendations(self, threshold: float = 0.7) -> List[FeatureRecommendation]:
        """Get recommendations with high expected impact."""
        return [rec for rec in self.recommendations if rec.expected_impact >= threshold]
    
    def get_low_complexity_recommendations(self) -> List[FeatureRecommendation]:
        """Get recommendations with low implementation complexity."""
        return [rec for rec in self.recommendations if rec.implementation_complexity == "low"]


@dataclass
class TransformedDataset:
    """Result of applying feature transformations."""
    original_dataset_id: str
    transformed_dataset_id: str
    transformations_applied: List[TransformationStep]
    feature_mapping: Dict[str, List[str]]  # original -> transformed features
    transformation_metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_feature_count_change(self) -> Dict[str, int]:
        """Get the change in feature count."""
        original_count = len(self.feature_mapping)
        transformed_count = sum(len(features) for features in self.feature_mapping.values())
        return {
            "original": original_count,
            "transformed": transformed_count,
            "change": transformed_count - original_count
        }