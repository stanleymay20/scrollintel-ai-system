"""
Pydantic schemas for data validation and API serialization.
Contains request/response models and validation logic.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from uuid import UUID
from pydantic import BaseModel, Field, EmailStr, field_validator, model_validator, validator
from enum import Enum

from ..core.interfaces import AgentType, AgentStatus, ResponseStatus, UserRole


# Base schemas
class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = {
        "from_attributes": True,
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    }


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""
    
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=20, ge=1, le=100, description="Page size")
    sort_by: Optional[str] = Field(default=None, description="Sort field")
    sort_order: Optional[str] = Field(default="asc", pattern="^(asc|desc)$", description="Sort order")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int


# User schemas
class UserBase(BaseSchema):
    """Base user schema."""
    
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER
    permissions: List[str] = Field(default_factory=list)
    is_active: bool = True


class UserCreate(UserBase):
    """Schema for creating a new user."""
    
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserUpdate(BaseSchema):
    """Schema for updating a user."""
    
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    permissions: Optional[List[str]] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=8)
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        """Validate password strength if provided."""
        if v is not None:
            if len(v) < 8:
                raise ValueError('Password must be at least 8 characters long')
            if not any(c.isupper() for c in v):
                raise ValueError('Password must contain at least one uppercase letter')
            if not any(c.islower() for c in v):
                raise ValueError('Password must contain at least one lowercase letter')
            if not any(c.isdigit() for c in v):
                raise ValueError('Password must contain at least one digit')
        return v


class UserResponse(UserBase):
    """Schema for user responses."""
    
    id: UUID
    is_verified: bool
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class UserLogin(BaseSchema):
    """Schema for user login."""
    
    email: EmailStr
    password: str


# Agent schemas
class AgentCapabilitySchema(BaseSchema):
    """Schema for agent capabilities."""
    
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]


class AgentBase(BaseSchema):
    """Base agent schema."""
    
    name: str = Field(..., min_length=1, max_length=255)
    agent_type: AgentType
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0.0"
    endpoint_url: Optional[str] = None
    health_check_url: Optional[str] = None
    is_enabled: bool = True


class AgentCreate(AgentBase):
    """Schema for creating a new agent."""
    pass


class AgentUpdate(BaseSchema):
    """Schema for updating an agent."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    configuration: Optional[Dict[str, Any]] = None
    status: Optional[AgentStatus] = None
    version: Optional[str] = None
    endpoint_url: Optional[str] = None
    health_check_url: Optional[str] = None
    is_enabled: Optional[bool] = None


class AgentResponse(AgentBase):
    """Schema for agent responses."""
    
    id: UUID
    status: AgentStatus
    last_health_check: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# Dataset schemas
class DatasetBase(BaseSchema):
    """Base dataset schema."""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    source_type: str = Field(..., pattern="^(csv|xlsx|sql|json|api|database)$")
    data_schema: Dict[str, Any] = Field(default_factory=dict)
    dataset_metadata: Dict[str, Any] = Field(default_factory=dict)
    file_path: Optional[str] = None
    connection_string: Optional[str] = None
    table_name: Optional[str] = None
    query: Optional[str] = None
    refresh_interval_minutes: Optional[int] = Field(None, ge=1)
    is_active: bool = True


class DatasetCreate(DatasetBase):
    """Schema for creating a new dataset."""
    
    @model_validator(mode='before')
    @classmethod
    def validate_source_requirements(cls, values):
        """Validate source-specific requirements."""
        if isinstance(values, dict):
            source_type = values.get('source_type')
            file_path = values.get('file_path')
            connection_string = values.get('connection_string')
            table_name = values.get('table_name')
            query = values.get('query')
            
            if source_type in ['csv', 'xlsx', 'json']:
                if not file_path:
                    raise ValueError(f'file_path is required for {source_type} datasets')
            elif source_type == 'sql':
                if not connection_string:
                    raise ValueError('connection_string is required for sql datasets')
                if not (table_name or query):
                    raise ValueError('Either table_name or query is required for sql datasets')
            elif source_type == 'database':
                if not connection_string:
                    raise ValueError('connection_string is required for database datasets')
        
        return values


class DatasetUpdate(BaseSchema):
    """Schema for updating a dataset."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    data_schema: Optional[Dict[str, Any]] = None
    dataset_metadata: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    connection_string: Optional[str] = None
    table_name: Optional[str] = None
    query: Optional[str] = None
    refresh_interval_minutes: Optional[int] = Field(None, ge=1)
    is_active: Optional[bool] = None


class DatasetResponse(DatasetBase):
    """Schema for dataset responses."""
    
    id: UUID
    row_count: Optional[int] = None
    last_refreshed: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# ML Model schemas
class MLModelBase(BaseSchema):
    """Base ML model schema."""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    algorithm: str = Field(..., min_length=1, max_length=100)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    feature_columns: List[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    version: str = "1.0.0"
    is_active: bool = True


class MLModelCreate(MLModelBase):
    """Schema for creating a new ML model."""
    
    dataset_id: UUID
    model_path: str = Field(..., min_length=1)


class MLModelUpdate(BaseSchema):
    """Schema for updating an ML model."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    feature_columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    model_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    version: Optional[str] = None
    is_deployed: Optional[bool] = None
    is_active: Optional[bool] = None


class MLModelResponse(MLModelBase):
    """Schema for ML model responses."""
    
    id: UUID
    dataset_id: UUID
    metrics: Dict[str, Any]
    model_path: str
    model_size_bytes: Optional[int] = None
    training_duration_seconds: Optional[float] = None
    api_endpoint: Optional[str] = None
    is_deployed: bool
    created_at: datetime
    updated_at: datetime


# Dashboard schemas
class DashboardBase(BaseSchema):
    """Base dashboard schema."""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    layout: Dict[str, Any] = Field(default_factory=dict)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    refresh_interval_minutes: Optional[int] = Field(60, ge=1)
    is_public: bool = False
    is_active: bool = True
    tags: List[str] = Field(default_factory=list)


class DashboardCreate(DashboardBase):
    """Schema for creating a new dashboard."""
    
    dataset_ids: List[UUID] = Field(default_factory=list)


class DashboardUpdate(BaseSchema):
    """Schema for updating a dashboard."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    layout: Optional[Dict[str, Any]] = None
    charts: Optional[List[Dict[str, Any]]] = None
    filters: Optional[Dict[str, Any]] = None
    refresh_interval_minutes: Optional[int] = Field(None, ge=1)
    is_public: Optional[bool] = None
    is_active: Optional[bool] = None
    tags: Optional[List[str]] = None
    dataset_ids: Optional[List[UUID]] = None


class DashboardResponse(DashboardBase):
    """Schema for dashboard responses."""
    
    id: UUID
    user_id: UUID
    last_refreshed: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    datasets: List[DatasetResponse] = Field(default_factory=list)


# Agent Request/Response schemas
class AgentRequestBase(BaseSchema):
    """Base agent request schema."""
    
    prompt: str = Field(..., min_length=1)
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(1, ge=1, le=10)


class AgentRequestCreate(AgentRequestBase):
    """Schema for creating an agent request."""
    
    agent_id: UUID


class AgentRequestResponse(AgentRequestBase):
    """Schema for agent request responses."""
    
    id: UUID
    user_id: UUID
    agent_id: UUID
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AgentResponseBase(BaseSchema):
    """Base agent response schema."""
    
    content: str = Field(..., min_length=1)
    artifacts: List[str] = Field(default_factory=list)
    response_metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time_seconds: Optional[float] = Field(None, ge=0)
    status: ResponseStatus = ResponseStatus.SUCCESS
    error_message: Optional[str] = None


class AgentResponseCreate(AgentResponseBase):
    """Schema for creating an agent response."""
    
    request_id: UUID
    agent_id: UUID


class AgentResponseResponse(AgentResponseBase):
    """Schema for agent response responses."""
    
    id: UUID
    request_id: UUID
    agent_id: UUID
    created_at: datetime


# Audit Log schemas
class AuditLogBase(BaseSchema):
    """Base audit log schema."""
    
    action: str = Field(..., min_length=1, max_length=100)
    resource_type: str = Field(..., min_length=1, max_length=100)
    resource_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class AuditLogCreate(AuditLogBase):
    """Schema for creating an audit log entry."""
    
    user_id: Optional[UUID] = None


class AuditLogResponse(AuditLogBase):
    """Schema for audit log responses."""
    
    id: UUID
    user_id: Optional[UUID] = None
    timestamp: datetime


# Authentication schemas
class Token(BaseSchema):
    """Schema for authentication tokens."""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseSchema):
    """Schema for token data."""
    
    user_id: Optional[UUID] = None
    email: Optional[str] = None
    role: Optional[UserRole] = None
    permissions: List[str] = Field(default_factory=list)


# File upload schemas
class FileUploadRequest(BaseSchema):
    """Schema for file upload requests."""
    
    name: Optional[str] = None
    description: Optional[str] = None
    auto_detect_schema: bool = True
    generate_preview: bool = True


class FileUploadResponse(BaseSchema):
    """Schema for file upload responses."""
    
    upload_id: str
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    content_type: str
    detected_type: str
    schema_info: Dict[str, Any] = Field(default_factory=dict)
    preview_data: Optional[Dict[str, Any]] = None
    quality_report: Optional[Dict[str, Any]] = None
    dataset_id: Optional[UUID] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FileProcessingStatus(BaseSchema):
    """Schema for file processing status."""
    
    upload_id: str
    status: str  # pending, processing, completed, failed
    progress: float = 0.0
    message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DataPreviewResponse(BaseSchema):
    """Schema for data preview responses."""
    
    columns: List[Dict[str, Any]]
    sample_data: List[Dict[str, Any]]
    total_rows: int
    data_types: Dict[str, str]
    statistics: Dict[str, Any] = Field(default_factory=dict)


class DataQualityReport(BaseSchema):
    """Schema for data quality reports."""
    
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    duplicate_rows: int
    data_type_issues: List[Dict[str, Any]]
    outliers: Dict[str, List[Any]] = Field(default_factory=dict)
    quality_score: float
    recommendations: List[str]


# Vault schemas
class VaultInsightBase(BaseSchema):
    """Base vault insight schema."""
    
    title: str = Field(..., min_length=1, max_length=500)
    insight_type: str = Field(..., pattern="^(analysis_result|model_explanation|prediction|report|visualization|recommendation|audit_log|research_finding)$")
    access_level: str = Field(..., pattern="^(public|internal|confidential|restricted|top_secret)$")
    retention_policy: str = Field(..., pattern="^(permanent|long_term|medium_term|short_term|temporary)$")
    organization_id: str = Field(default="default", min_length=1, max_length=255)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VaultInsightCreate(VaultInsightBase):
    """Schema for creating a vault insight."""
    
    content: Dict[str, Any] = Field(..., description="Insight content to be encrypted")
    parent_id: Optional[UUID] = None


class VaultInsightUpdate(BaseSchema):
    """Schema for updating a vault insight."""
    
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    access_level: Optional[str] = Field(None, pattern="^(public|internal|confidential|restricted|top_secret)$")
    retention_policy: Optional[str] = Field(None, pattern="^(permanent|long_term|medium_term|short_term|temporary)$")


class VaultInsightResponse(VaultInsightBase):
    """Schema for vault insight responses."""
    
    id: UUID
    creator_id: UUID
    version: int
    parent_id: Optional[UUID] = None
    access_count: int
    last_accessed: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    content: Optional[Dict[str, Any]] = None  # Only included when decrypted


class VaultInsightSummary(BaseSchema):
    """Schema for vault insight summary (without content)."""
    
    id: UUID
    title: str
    insight_type: str
    access_level: str
    creator_id: UUID
    tags: List[str]
    version: int
    access_count: int
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None


class VaultSearchQuery(BaseSchema):
    """Schema for vault search queries."""
    
    query: str = Field(default="", description="Search query text")
    filters: Dict[str, Any] = Field(default_factory=dict)
    access_levels: Optional[List[str]] = Field(None, description="Filter by access levels")
    insight_types: Optional[List[str]] = Field(None, description="Filter by insight types")
    date_range: Optional[Tuple[datetime, datetime]] = Field(None, description="Filter by date range")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    creator_id: Optional[UUID] = Field(None, description="Filter by creator")
    limit: int = Field(50, ge=1, le=100, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Number of results to skip")


class VaultSearchResponse(BaseSchema):
    """Schema for vault search responses."""
    
    results: List[VaultInsightSummary]
    total_count: int
    query: str
    filters_applied: Dict[str, Any]
    offset: int
    limit: int


class VaultAccessLogResponse(BaseSchema):
    """Schema for vault access log responses."""
    
    id: UUID
    insight_id: UUID
    user_id: UUID
    action: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime


class VaultStatsResponse(BaseSchema):
    """Schema for vault statistics responses."""
    
    total_insights: int
    insights_by_type: Dict[str, int]
    insights_by_access_level: Dict[str, int]
    total_access_count: int
    recent_activity: List[VaultAccessLogResponse]
    storage_usage: Dict[str, Any]


# Health check schema
class HealthCheck(BaseSchema):
    """Schema for health check responses."""
    
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    database: bool = True
    redis: bool = True
    ai_services: Dict[str, bool] = Field(default_factory=dict)


# Error response schema
class ErrorResponse(BaseSchema):
    """Schema for error responses."""
    
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


# Validation helpers
def validate_json_field(value: Any, field_name: str) -> Dict[str, Any]:
    """Validate that a field is a valid JSON object."""
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary")
    return value


def validate_json_array_field(value: Any, field_name: str) -> List[Any]:
    """Validate that a field is a valid JSON array."""
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


# ScrollInsightRadar schemas
class PatternDetectionConfig(BaseSchema):
    """Configuration for pattern detection analysis."""
    
    correlation_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum correlation threshold")
    anomaly_contamination: float = Field(0.1, ge=0.01, le=0.5, description="Expected proportion of anomalies")
    seasonal_period: Optional[int] = Field(None, ge=2, description="Seasonal period for decomposition")
    clustering_eps: float = Field(0.5, ge=0.1, le=2.0, description="DBSCAN epsilon parameter")
    clustering_min_samples: int = Field(5, ge=2, le=50, description="DBSCAN minimum samples")
    significance_level: float = Field(0.05, ge=0.001, le=0.1, description="Statistical significance level")
    max_features: Optional[int] = Field(None, ge=1, description="Maximum features to analyze")


class TrendAnalysis(BaseSchema):
    """Schema for trend analysis results."""
    
    datetime_column: str
    numeric_column: str
    slope: float
    r_squared: float
    p_value: float
    trend_direction: str = Field(..., pattern="^(increasing|decreasing|stable)$")
    is_significant: bool
    trend_strength: float = Field(..., ge=0.0, le=1.0)
    confidence_level: str = Field(..., pattern="^(high|medium|low)$")


class AnomalyDetection(BaseSchema):
    """Schema for anomaly detection results."""
    
    index: int
    anomaly_score: float
    values: Dict[str, Any]
    severity: str = Field(..., pattern="^(high|medium|low)$")
    detection_method: str = Field(default="isolation_forest")


class InsightRadarResult(BaseSchema):
    """Schema for ScrollInsightRadar analysis results."""
    
    timestamp: datetime
    dataset_info: Dict[str, Any]
    patterns: Dict[str, Any] = Field(default_factory=dict)
    trends: Dict[str, Any] = Field(default_factory=dict)
    anomalies: Dict[str, Any] = Field(default_factory=dict)
    insights: List[Dict[str, Any]] = Field(default_factory=list)
    statistical_tests: Dict[str, Any] = Field(default_factory=dict)
    business_impact_score: float = Field(..., ge=0.0, le=1.0)


class InsightNotification(BaseSchema):
    """Schema for insight notifications."""
    
    user_id: str
    insight_type: str
    title: str
    description: str
    priority: str = Field(..., pattern="^(high|medium|low)$")
    impact_score: float = Field(..., ge=0.0, le=1.0)
    actionable: bool = True
    notification_sent: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PatternDetectionRequest(BaseSchema):
    """Schema for pattern detection requests."""
    
    file_name: str
    config: Optional[PatternDetectionConfig] = None
    send_notifications: bool = False
    analysis_type: str = Field(default="comprehensive", pattern="^(comprehensive|trends_only|anomalies_only|patterns_only)$")


class PatternDetectionResponse(BaseSchema):
    """Schema for pattern detection responses."""
    
    success: bool
    message: str
    file_name: str
    analysis_timestamp: datetime
    user_id: str
    results: InsightRadarResult
    notifications_sent: bool = False


class BatchAnalysisRequest(BaseSchema):
    """Schema for batch analysis requests."""
    
    file_names: List[str]
    config: Optional[PatternDetectionConfig] = None
    max_files: int = Field(10, ge=1, le=20)


class BatchAnalysisResponse(BaseSchema):
    """Schema for batch analysis responses."""
    
    success: bool
    message: str
    analysis_timestamp: datetime
    user_id: str
    files_processed: int
    batch_results: List[Dict[str, Any]]
    aggregated_insights: List[Dict[str, Any]]
    total_insights: int
    average_impact_score: float


class InsightRankingCriteria(BaseSchema):
    """Schema for insight ranking criteria."""
    
    impact_weight: float = Field(0.4, ge=0.0, le=1.0)
    significance_weight: float = Field(0.3, ge=0.0, le=1.0)
    actionability_weight: float = Field(0.2, ge=0.0, le=1.0)
    novelty_weight: float = Field(0.1, ge=0.0, le=1.0)


class InsightRadarHealth(BaseSchema):
    """Schema for ScrollInsightRadar health status."""
    
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    engine: str
    version: str
    capabilities: List[str]
    last_check: datetime
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)


# Custom validators for common patterns
class EmailValidator:
    """Email validation utilities."""
    
    @staticmethod
    def validate_email_format(email: str) -> str:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise ValueError("Invalid email format")
        return email.lower()


class PasswordValidator:
    """Password validation utilities."""
    
    @staticmethod
    def validate_password_strength(password: str) -> str:
        """Validate password strength."""
        if len(password) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in password):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in password):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in password):
            raise ValueError('Password must contain at least one digit')
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
            raise ValueError('Password must contain at least one special character')
        return password


# Prompt Engineering schemas
class PromptTemplateBase(BaseSchema):
    """Base prompt template schema."""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    category: str = Field(..., pattern="^(data_analysis|code_generation|creative_writing|business_intelligence|technical_documentation|customer_service|research_analysis|strategic_planning)$")
    industry: Optional[str] = Field(None, max_length=100)
    use_case: Optional[str] = Field(None, max_length=255)
    template_content: str = Field(..., min_length=1)
    variables: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    is_public: bool = False
    is_active: bool = True


class PromptTemplateCreate(PromptTemplateBase):
    """Schema for creating a prompt template."""
    pass


class PromptTemplateUpdate(BaseSchema):
    """Schema for updating a prompt template."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    template_content: Optional[str] = Field(None, min_length=1)
    variables: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    is_active: Optional[bool] = None


class PromptTemplateResponse(PromptTemplateBase):
    """Schema for prompt template responses."""
    
    id: UUID
    creator_id: UUID
    usage_count: int
    performance_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class PromptHistoryBase(BaseSchema):
    """Base prompt history schema."""
    
    original_prompt: str = Field(..., min_length=1)
    optimized_prompt: str = Field(..., min_length=1)
    optimization_strategy: str = Field(..., pattern="^(a_b_testing|genetic_algorithm|reinforcement_learning|semantic_similarity|performance_based)$")
    performance_improvement: Optional[float] = None
    success_rate_before: Optional[float] = Field(None, ge=0.0, le=1.0)
    success_rate_after: Optional[float] = Field(None, ge=0.0, le=1.0)
    response_time_before: Optional[float] = Field(None, ge=0.0)
    response_time_after: Optional[float] = Field(None, ge=0.0)
    test_cases_count: Optional[int] = Field(None, ge=0)
    optimization_metadata: Dict[str, Any] = Field(default_factory=dict)
    feedback_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    is_favorite: bool = False


class PromptHistoryCreate(PromptHistoryBase):
    """Schema for creating prompt history."""
    pass


class PromptHistoryUpdate(BaseSchema):
    """Schema for updating prompt history."""
    
    feedback_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    is_favorite: Optional[bool] = None


class PromptHistoryResponse(PromptHistoryBase):
    """Schema for prompt history responses."""
    
    id: UUID
    user_id: UUID
    created_at: datetime


class PromptTestBase(BaseSchema):
    """Base prompt test schema."""
    
    test_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    test_type: str = Field(..., pattern="^(a_b_test|performance_test|variation_test)$")
    prompt_variations: List[Dict[str, Any]] = Field(default_factory=list)
    test_cases: List[Dict[str, Any]] = Field(default_factory=list)


class PromptTestCreate(PromptTestBase):
    """Schema for creating a prompt test."""
    
    template_id: Optional[UUID] = None
    history_id: Optional[UUID] = None


class PromptTestUpdate(BaseSchema):
    """Schema for updating a prompt test."""
    
    status: Optional[str] = Field(None, pattern="^(pending|running|completed|failed)$")
    test_results: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    statistical_analysis: Optional[Dict[str, Any]] = None
    winner_variation_id: Optional[str] = None
    confidence_level: Optional[float] = Field(None, ge=0.0, le=1.0)


class PromptTestResponse(PromptTestBase):
    """Schema for prompt test responses."""
    
    id: UUID
    user_id: UUID
    template_id: Optional[UUID] = None
    history_id: Optional[UUID] = None
    status: str
    test_results: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    statistical_analysis: Dict[str, Any]
    winner_variation_id: Optional[str] = None
    confidence_level: Optional[float] = None
    total_test_runs: int
    successful_runs: int
    average_response_time: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class PromptOptimizationRequest(BaseSchema):
    """Schema for prompt optimization requests."""
    
    original_prompt: str = Field(..., min_length=1)
    strategy: str = Field(default="a_b_testing", pattern="^(a_b_testing|genetic_algorithm|reinforcement_learning|semantic_similarity|performance_based)$")
    test_data: Optional[List[str]] = Field(default_factory=list)
    target_metric: str = Field(default="performance_score", pattern="^(performance_score|success_rate|response_time|user_satisfaction)$")
    max_variations: int = Field(default=5, ge=1, le=10)
    test_iterations: int = Field(default=10, ge=1, le=100)


class PromptOptimizationResponse(BaseSchema):
    """Schema for prompt optimization responses."""
    
    original_prompt: str
    optimized_prompt: str
    strategy: str
    performance_improvement: float
    success_rate_improvement: Optional[float] = None
    response_time_improvement: Optional[float] = None
    variations_tested: int
    test_results: Dict[str, Any]
    recommendations: List[str]
    optimization_explanation: str


class PromptVariationTestRequest(BaseSchema):
    """Schema for prompt variation testing requests."""
    
    variations: List[str] = Field(..., min_items=2, max_items=10)
    test_cases: List[str] = Field(..., min_items=1)
    evaluation_criteria: List[str] = Field(default_factory=lambda: ["accuracy", "relevance", "clarity"])
    statistical_significance: float = Field(default=0.05, ge=0.001, le=0.1)


class PromptVariationTestResponse(BaseSchema):
    """Schema for prompt variation testing responses."""
    
    test_id: str
    variations_tested: int
    test_cases_count: int
    winner_variation: str
    confidence_level: float
    performance_comparison: Dict[str, Any]
    statistical_analysis: Dict[str, Any]
    recommendations: List[str]


class PromptChainRequest(BaseSchema):
    """Schema for prompt chain requests."""
    
    chain_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    prompts: List[Dict[str, Any]] = Field(..., min_items=1)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    execution_context: Dict[str, Any] = Field(default_factory=dict)


class PromptChainResponse(BaseSchema):
    """Schema for prompt chain responses."""
    
    chain_id: str
    chain_name: str
    execution_results: List[Dict[str, Any]]
    execution_flow: List[str]
    performance_metrics: Dict[str, Any]
    optimization_suggestions: List[str]


class PromptTemplateGenerationRequest(BaseSchema):
    """Schema for prompt template generation requests."""
    
    industry: str = Field(..., min_length=1, max_length=100)
    use_case: str = Field(..., min_length=1, max_length=255)
    requirements: List[str] = Field(default_factory=list)
    target_audience: Optional[str] = Field(None, max_length=255)
    output_format: Optional[str] = Field(None, max_length=100)
    complexity_level: str = Field(default="intermediate", pattern="^(beginner|intermediate|advanced|expert)$")


class PromptTemplateGenerationResponse(BaseSchema):
    """Schema for prompt template generation responses."""
    
    templates: List[Dict[str, Any]]
    usage_guidelines: List[str]
    customization_options: List[str]
    optimization_tips: List[str]
    industry_best_practices: List[str]


class URLValidator:
    """URL validation utilities."""
    
    @staticmethod
    def validate_url_format(url: str) -> str:
        """Validate URL format."""
        import re
        pattern = r'^https?://(?:[-\w.])+(?::[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
        if not re.match(pattern, url):
            raise ValueError("Invalid URL format")
        return url