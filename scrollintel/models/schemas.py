"""
Pydantic schemas for data validation and API serialization.
Contains request/response models and validation logic.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from uuid import UUID
from pydantic import BaseModel, Field, EmailStr, validator, root_validator
from enum import Enum

from ..core.interfaces import AgentType, AgentStatus, ResponseStatus, UserRole


# Base schemas
class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    class Config:
        from_attributes = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""
    
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=20, ge=1, le=100, description="Page size")
    sort_by: Optional[str] = Field(default=None, description="Sort field")
    sort_order: Optional[str] = Field(default="asc", regex="^(asc|desc)$", description="Sort order")


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
    
    @validator('password')
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
    
    @validator('password')
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
    type: AgentType
    description: Optional[str] = None
    capabilities: List[Dict[str, Any]] = Field(default_factory=list)
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
    capabilities: Optional[List[Dict[str, Any]]] = None
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
    source_type: str = Field(..., regex="^(csv|xlsx|sql|json|api|database)$")
    schema: Dict[str, Any] = Field(default_factory=dict)
    dataset_metadata: Dict[str, Any] = Field(default_factory=dict)
    file_path: Optional[str] = None
    connection_string: Optional[str] = None
    table_name: Optional[str] = None
    query: Optional[str] = None
    refresh_interval_minutes: Optional[int] = Field(None, ge=1)
    is_active: bool = True


class DatasetCreate(DatasetBase):
    """Schema for creating a new dataset."""
    
    @root_validator
    def validate_source_requirements(cls, values):
        """Validate source-specific requirements."""
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
    schema: Optional[Dict[str, Any]] = None
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
    dataset_ids: Optional[List[UUID]] = None


class DashboardResponse(DashboardBase):
    """Schema for dashboard responses."""
    
    id: UUID
    user_id: UUID
    last_refreshed: Optional[datetime] = None
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
    metadata: Dict[str, Any] = Field(default_factory=dict)
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
class FileUploadResponse(BaseSchema):
    """Schema for file upload responses."""
    
    filename: str
    file_path: str
    file_size: int
    content_type: str
    dataset_id: Optional[UUID] = None
    upload_id: str


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