"""Modern GraphQL types with enhanced features."""

import strawberry
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


@strawberry.enum
class DatasetStatus(Enum):
    """Dataset processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@strawberry.enum
class QualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@strawberry.type
class Dataset:
    """Enhanced dataset type with modern features."""
    id: str
    name: str
    description: Optional[str] = None
    status: DatasetStatus
    created_at: datetime
    updated_at: datetime
    size_bytes: int
    row_count: int
    column_count: int
    file_format: str
    tags: List[str] = strawberry.field(default_factory=list)
    metadata: strawberry.scalars.JSON = strawberry.field(default_factory=dict)
    
    @strawberry.field
    def quality_score(self) -> float:
        """Calculate overall quality score."""
        # Implementation would go here
        return 0.85
    
    @strawberry.field
    def is_ai_ready(self) -> bool:
        """Check if dataset is AI-ready."""
        return self.quality_score > 0.8


@strawberry.type
class QualityReport:
    """Enhanced quality report with detailed metrics."""
    id: str
    dataset_id: str
    overall_score: float
    quality_level: QualityLevel
    completeness_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    accuracy_score: float
    timeliness_score: float
    issues: List[str] = strawberry.field(default_factory=list)
    recommendations: List[str] = strawberry.field(default_factory=list)
    created_at: datetime
    
    @strawberry.field
    def detailed_metrics(self) -> strawberry.scalars.JSON:
        """Get detailed quality metrics."""
        return {
            "completeness": self.completeness_score,
            "consistency": self.consistency_score,
            "validity": self.validity_score,
            "uniqueness": self.uniqueness_score,
            "accuracy": self.accuracy_score,
            "timeliness": self.timeliness_score
        }


@strawberry.type
class BiasReport:
    """Enhanced bias analysis report."""
    id: str
    dataset_id: str
    overall_bias_score: float
    demographic_parity: float
    equalized_odds: float
    statistical_parity: float
    individual_fairness: float
    protected_attributes: List[str] = strawberry.field(default_factory=list)
    bias_sources: List[str] = strawberry.field(default_factory=list)
    mitigation_strategies: List[str] = strawberry.field(default_factory=list)
    created_at: datetime


@strawberry.type
class AIReadinessScore:
    """AI readiness assessment."""
    dataset_id: str
    overall_score: float
    data_quality_score: float
    bias_score: float
    completeness_score: float
    feature_quality_score: float
    readiness_level: str
    blocking_issues: List[str] = strawberry.field(default_factory=list)
    recommendations: List[str] = strawberry.field(default_factory=list)
    estimated_preparation_time: int  # in hours


@strawberry.type
class FeatureRecommendations:
    """Feature engineering recommendations."""
    dataset_id: str
    recommended_features: List[str] = strawberry.field(default_factory=list)
    feature_importance: strawberry.scalars.JSON = strawberry.field(default_factory=dict)
    transformation_suggestions: List[str] = strawberry.field(default_factory=list)
    encoding_recommendations: strawberry.scalars.JSON = strawberry.field(default_factory=dict)


@strawberry.type
class ComplianceReport:
    """Data compliance assessment."""
    id: str
    dataset_id: str
    gdpr_compliant: bool
    ccpa_compliant: bool
    hipaa_compliant: bool
    pii_detected: bool
    sensitive_data_types: List[str] = strawberry.field(default_factory=list)
    compliance_score: float
    violations: List[str] = strawberry.field(default_factory=list)
    remediation_steps: List[str] = strawberry.field(default_factory=list)


@strawberry.type
class LineageInfo:
    """Data lineage information."""
    dataset_id: str
    source_systems: List[str] = strawberry.field(default_factory=list)
    transformation_steps: List[str] = strawberry.field(default_factory=list)
    dependencies: List[str] = strawberry.field(default_factory=list)
    downstream_consumers: List[str] = strawberry.field(default_factory=list)
    lineage_graph: strawberry.scalars.JSON = strawberry.field(default_factory=dict)


@strawberry.type
class DriftReport:
    """Data drift detection report."""
    id: str
    dataset_id: str
    drift_detected: bool
    drift_score: float
    affected_features: List[str] = strawberry.field(default_factory=list)
    drift_type: str  # statistical, concept, etc.
    detection_method: str
    confidence_level: float
    created_at: datetime


@strawberry.type
class ProcessingJob:
    """Data processing job status."""
    id: str
    dataset_id: str
    job_type: str
    status: DatasetStatus
    progress: float
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_data: strawberry.scalars.JSON = strawberry.field(default_factory=dict)


# Input types for mutations
@strawberry.input
class DatasetInput:
    """Input for creating/updating datasets."""
    name: str
    description: Optional[str] = None
    file_path: Optional[str] = None
    tags: List[str] = strawberry.field(default_factory=list)
    metadata: strawberry.scalars.JSON = strawberry.field(default_factory=dict)


@strawberry.input
class QualityAssessmentInput:
    """Input for quality assessment."""
    dataset_id: str
    include_detailed_analysis: bool = True
    custom_rules: strawberry.scalars.JSON = strawberry.field(default_factory=dict)


@strawberry.input
class BiasAnalysisInput:
    """Input for bias analysis."""
    dataset_id: str
    protected_attributes: List[str]
    fairness_metrics: List[str] = strawberry.field(default_factory=list)
    target_column: Optional[str] = None
