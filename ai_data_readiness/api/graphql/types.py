"""GraphQL type definitions."""

import strawberry
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


@strawberry.enum
class DatasetStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"


@strawberry.enum
class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@strawberry.enum
class QualityDimension(Enum):
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"


@strawberry.type
class Schema:
    """Dataset schema type."""
    columns: Dict[str, str]
    primary_key: Optional[str] = None
    foreign_keys: Dict[str, str] = strawberry.field(default_factory=dict)
    constraints: List[str] = strawberry.field(default_factory=list)
    version: str = "1.0"


@strawberry.type
class DatasetMetadata:
    """Dataset metadata type."""
    id: str
    name: str
    description: str
    source: str
    format: str
    size_bytes: int
    row_count: int
    column_count: int
    tags: List[str]
    owner: str
    version: str
    created_at: datetime
    updated_at: datetime


@strawberry.type
class Dataset:
    """Dataset GraphQL type."""
    id: str
    name: str
    description: str
    schema: Optional[Schema] = None
    metadata: Optional[DatasetMetadata] = None
    quality_score: float
    ai_readiness_score: float
    status: DatasetStatus
    created_at: datetime
    updated_at: datetime
    version: str
    
    # Computed fields
    @strawberry.field
    def is_ai_ready(self, threshold: float = 0.8) -> bool:
        """Check if dataset meets AI readiness threshold."""
        return self.ai_readiness_score >= threshold
    
    @strawberry.field
    def quality_grade(self) -> str:
        """Get quality grade based on score."""
        if self.quality_score >= 0.9:
            return "A"
        elif self.quality_score >= 0.8:
            return "B"
        elif self.quality_score >= 0.7:
            return "C"
        elif self.quality_score >= 0.6:
            return "D"
        else:
            return "F"


@strawberry.type
class QualityIssue:
    """Quality issue type."""
    dimension: QualityDimension
    severity: str
    description: str
    affected_columns: List[str]
    affected_rows: int
    recommendation: str
    detected_at: datetime


@strawberry.type
class Recommendation:
    """Improvement recommendation type."""
    id: str
    type: str
    priority: str
    title: str
    description: str
    action_items: List[str]
    estimated_impact: str
    implementation_effort: str
    affected_columns: List[str]


@strawberry.type
class QualityReport:
    """Quality assessment report type."""
    dataset_id: str
    overall_score: float
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    timeliness_score: float
    issues: List[QualityIssue]
    recommendations: List[Recommendation]
    generated_at: datetime
    
    @strawberry.field
    def dimension_scores(self) -> Dict[str, float]:
        """Get all dimension scores as a dictionary."""
        return {
            "completeness": self.completeness_score,
            "accuracy": self.accuracy_score,
            "consistency": self.consistency_score,
            "validity": self.validity_score,
            "uniqueness": self.uniqueness_score,
            "timeliness": self.timeliness_score
        }


@strawberry.type
class FairnessViolation:
    """Fairness violation type."""
    bias_type: str
    protected_attribute: str
    severity: str
    description: str
    metric_value: float
    threshold: float
    affected_groups: List[str]


@strawberry.type
class MitigationStrategy:
    """Bias mitigation strategy type."""
    strategy_type: str
    description: str
    implementation_steps: List[str]
    expected_impact: float
    complexity: str


@strawberry.type
class BiasReport:
    """Bias analysis report type."""
    dataset_id: str
    protected_attributes: List[str]
    bias_metrics: Dict[str, float]
    fairness_violations: List[FairnessViolation]
    mitigation_strategies: List[MitigationStrategy]
    generated_at: datetime


@strawberry.type
class DimensionScore:
    """AI readiness dimension score type."""
    dimension: str
    score: float
    weight: float
    details: Dict[str, Any]


@strawberry.type
class ImprovementArea:
    """Improvement area type."""
    area: str
    current_score: float
    target_score: float
    priority: str
    estimated_effort: str
    actions: List[str]


@strawberry.type
class AIReadinessScore:
    """AI readiness assessment type."""
    overall_score: float
    data_quality_score: float
    feature_quality_score: float
    bias_score: float
    compliance_score: float
    scalability_score: float
    dimensions: Dict[str, DimensionScore]
    improvement_areas: List[ImprovementArea]
    generated_at: datetime


@strawberry.type
class FeatureRecommendation:
    """Feature engineering recommendation type."""
    feature_name: str
    transformation: str
    description: str
    priority: str
    expected_impact: float


@strawberry.type
class FeatureRecommendations:
    """Feature engineering recommendations type."""
    dataset_id: str
    model_type: str
    recommendations: List[FeatureRecommendation]
    transformations: List[Dict[str, Any]]
    encoding_strategies: Dict[str, str]
    generated_at: datetime


@strawberry.type
class ComplianceViolation:
    """Compliance violation type."""
    regulation: str
    violation_type: str
    severity: str
    description: str
    affected_columns: List[str]


@strawberry.type
class SensitiveData:
    """Sensitive data detection type."""
    column: str
    data_type: str
    confidence: float
    sample_values: List[str]


@strawberry.type
class ComplianceReport:
    """Compliance check report type."""
    dataset_id: str
    regulations: List[str]
    compliance_score: float
    violations: List[ComplianceViolation]
    recommendations: List[Recommendation]
    sensitive_data_detected: List[SensitiveData]
    generated_at: datetime


@strawberry.type
class Transformation:
    """Data transformation type."""
    type: str
    description: str
    parameters: Dict[str, Any]
    timestamp: datetime


@strawberry.type
class LineageInfo:
    """Data lineage information type."""
    dataset_id: str
    source_datasets: List[str]
    transformations: List[Transformation]
    downstream_datasets: List[str]
    models_trained: List[str]
    created_by: str
    created_at: datetime


@strawberry.type
class DriftAlert:
    """Drift alert type."""
    feature: str
    drift_score: float
    threshold: float
    severity: str
    description: str
    timestamp: datetime


@strawberry.type
class DriftRecommendation:
    """Drift recommendation type."""
    type: str
    description: str
    priority: str
    action_items: List[str]


@strawberry.type
class DriftReport:
    """Drift monitoring report type."""
    dataset_id: str
    reference_dataset_id: str
    drift_score: float
    feature_drift_scores: Dict[str, float]
    statistical_tests: Dict[str, Dict[str, Any]]
    alerts: List[DriftAlert]
    recommendations: List[DriftRecommendation]
    generated_at: datetime


@strawberry.type
class ProcessingJob:
    """Processing job type."""
    job_id: str
    dataset_id: str
    job_type: str
    status: JobStatus
    progress: float
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @strawberry.field
    def duration_seconds(self) -> Optional[int]:
        """Calculate job duration in seconds."""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return None
    
    @strawberry.field
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == JobStatus.RUNNING


# Input types for mutations
@strawberry.input
class DatasetCreateInput:
    """Input for creating a dataset."""
    name: str
    description: str = ""
    source: str
    format: str
    tags: List[str] = strawberry.field(default_factory=list)


@strawberry.input
class DatasetUpdateInput:
    """Input for updating a dataset."""
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


@strawberry.input
class QualityAssessmentInput:
    """Input for quality assessment."""
    dataset_id: str
    dimensions: List[QualityDimension] = strawberry.field(default_factory=list)
    generate_recommendations: bool = True


@strawberry.input
class BiasAnalysisInput:
    """Input for bias analysis."""
    dataset_id: str
    protected_attributes: List[str]
    target_column: Optional[str] = None
    bias_types: List[str] = strawberry.field(default_factory=list)


@strawberry.input
class FeatureEngineeringInput:
    """Input for feature engineering."""
    dataset_id: str
    model_type: str = "classification"
    target_column: Optional[str] = None
    categorical_columns: Optional[List[str]] = None
    numerical_columns: Optional[List[str]] = None


@strawberry.input
class ComplianceCheckInput:
    """Input for compliance checking."""
    dataset_id: str
    regulations: List[str] = strawberry.field(default_factory=lambda: ["GDPR", "CCPA"])
    sensitive_data_types: Optional[List[str]] = None


@strawberry.input
class DriftMonitoringInput:
    """Input for drift monitoring setup."""
    dataset_id: str
    reference_dataset_id: str
    monitoring_frequency: str = "daily"
    drift_threshold: float = 0.1
    alert_email: Optional[str] = None


@strawberry.input
class ProcessingJobInput:
    """Input for creating a processing job."""
    dataset_id: str
    job_type: str
    parameters: Dict[str, Any] = strawberry.field(default_factory=dict)
    priority: str = "normal"