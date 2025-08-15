"""Base data models for the AI Data Readiness Platform."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import uuid


class DatasetStatus(Enum):
    """Dataset processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"


class QualityDimension(Enum):
    """Data quality dimensions."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"


class BiasType(Enum):
    """Types of bias that can be detected."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    STATISTICAL_PARITY = "statistical_parity"
    INDIVIDUAL_FAIRNESS = "individual_fairness"


class RecommendationType(Enum):
    """Types of recommendations that can be generated."""
    DATA_COLLECTION = "data_collection"
    DATA_PREPROCESSING = "data_preprocessing"
    DATA_VALIDATION = "data_validation"
    DATA_STANDARDIZATION = "data_standardization"
    DATA_CLEANING = "data_cleaning"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_BALANCING = "data_balancing"


class RecommendationPriority(Enum):
    """Priority levels for recommendations."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Schema:
    """Dataset schema definition."""
    columns: Dict[str, str]  # column_name -> data_type
    primary_key: Optional[str] = None
    foreign_keys: Dict[str, str] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    version: str = "1.0"
    
    def validate_data_types(self) -> bool:
        """Validate that all data types are supported."""
        supported_types = {
            'integer', 'float', 'string', 'boolean', 'datetime', 
            'categorical', 'text', 'binary', 'json'
        }
        return all(dtype in supported_types for dtype in self.columns.values())


@dataclass
class DatasetMetadata:
    """Metadata for a dataset."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    source: str = ""
    format: str = ""  # csv, json, parquet, etc.
    size_bytes: int = 0
    row_count: int = 0
    column_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    owner: str = ""
    version: str = "1.0"


@dataclass
class QualityIssue:
    """Represents a data quality issue."""
    dimension: QualityDimension
    severity: str  # low, medium, high, critical
    description: str
    affected_columns: List[str]
    affected_rows: int
    recommendation: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    # Additional fields needed by engines
    column: Optional[str] = None
    category: str = ""
    issue_type: str = ""
    affected_percentage: float = 0.0


@dataclass
class Recommendation:
    """Improvement recommendation."""
    id: str
    type: RecommendationType
    priority: RecommendationPriority
    title: str
    description: str
    action_items: List[str]
    estimated_impact: str
    implementation_effort: str
    affected_columns: List[str]
    category: str
    priority_score: Optional[float] = None


@dataclass
class QualityReport:
    """Data quality assessment report."""
    dataset_id: str
    overall_score: float
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    timeliness_score: float
    issues: List[QualityIssue] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_dimension_score(self, dimension: QualityDimension) -> float:
        """Get score for a specific quality dimension."""
        dimension_map = {
            QualityDimension.COMPLETENESS: self.completeness_score,
            QualityDimension.ACCURACY: self.accuracy_score,
            QualityDimension.CONSISTENCY: self.consistency_score,
            QualityDimension.VALIDITY: self.validity_score,
            QualityDimension.UNIQUENESS: self.uniqueness_score,
            QualityDimension.TIMELINESS: self.timeliness_score
        }
        return dimension_map.get(dimension, 0.0)


@dataclass
class DimensionScore:
    """Score for a specific AI readiness dimension."""
    dimension: str
    score: float
    weight: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementArea:
    """Area for improvement in AI readiness."""
    area: str
    current_score: float
    target_score: float
    priority: str
    estimated_effort: str = "Medium"
    actions: List[str] = field(default_factory=list)


@dataclass
class AIReadinessScore:
    """AI readiness assessment score."""
    overall_score: float
    data_quality_score: float
    feature_quality_score: float
    bias_score: float
    compliance_score: float
    scalability_score: float
    dimensions: Dict[str, DimensionScore] = field(default_factory=dict)
    improvement_areas: List[ImprovementArea] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FairnessViolation:
    """Represents a fairness/bias violation."""
    bias_type: BiasType
    protected_attribute: str
    severity: str
    description: str
    metric_value: float
    threshold: float
    affected_groups: List[str]


@dataclass
class MitigationStrategy:
    """Strategy for mitigating bias."""
    strategy_type: str
    description: str
    implementation_steps: List[str]
    expected_impact: float
    complexity: str  # low, medium, high


@dataclass
class BiasReport:
    """Bias analysis report."""
    dataset_id: str
    protected_attributes: List[str]
    bias_metrics: Dict[str, float]
    fairness_violations: List[FairnessViolation] = field(default_factory=list)
    mitigation_strategies: List[MitigationStrategy] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataLineage:
    """Data lineage tracking."""
    dataset_id: str
    source_datasets: List[str] = field(default_factory=list)
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_transformation(self, transformation_type: str, details: Dict[str, Any]):
        """Add a transformation to the lineage."""
        self.transformations.append({
            'type': transformation_type,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        })


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_error(self, error: str):
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a validation warning."""
        self.warnings.append(warning)


@dataclass
class Dataset:
    """Main dataset model."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    schema: Optional[Schema] = None
    metadata: Optional[DatasetMetadata] = None
    quality_score: float = 0.0
    ai_readiness_score: float = 0.0
    status: DatasetStatus = DatasetStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"
    lineage: Optional[DataLineage] = None
    
    def update_status(self, new_status: DatasetStatus):
        """Update dataset status and timestamp."""
        self.status = new_status
        self.updated_at = datetime.utcnow()
    
    def is_ai_ready(self, threshold: float = 0.8) -> bool:
        """Check if dataset meets AI readiness threshold."""
        return self.ai_readiness_score >= threshold