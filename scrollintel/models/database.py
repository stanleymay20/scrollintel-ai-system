"""
SQLAlchemy database models for ScrollIntel system.
Contains all database table definitions and relationships.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
import json
from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, Text, JSON, 
    ForeignKey, Float, Enum as SQLEnum, Index, UniqueConstraint
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.types import TypeDecorator
import uuid

from ..core.interfaces import AgentType, AgentStatus, ResponseStatus, UserRole

Base = declarative_base()


class JSONType(TypeDecorator):
    """JSON type that works with both PostgreSQL (JSONB) and SQLite (JSON)."""
    
    impl = JSON
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(JSON())
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        return value
    
    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return value


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.VIEWER)
    permissions = Column(JSONType, nullable=False, default=list)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    agent_requests = relationship("AgentRequest", back_populates="user", cascade="all, delete-orphan")
    dashboards = relationship("Dashboard", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    # user_audit_logs = relationship("UserAuditLog", back_populates="user", cascade="all, delete-orphan")  # Defined in user_management_models
    
    # Indexes
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_role', 'role'),
        Index('idx_user_active', 'is_active'),
    )
    
    @validates('email')
    def validate_email(self, key, email):
        """Validate email format."""
        import re
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            raise ValueError("Invalid email format")
        return email.lower()
    
    @validates('permissions')
    def validate_permissions(self, key, permissions):
        """Validate permissions is a list."""
        if not isinstance(permissions, list):
            raise ValueError("Permissions must be a list")
        return permissions
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"


class Agent(Base):
    """Agent model for AI agent registry and management."""
    
    __tablename__ = "agents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    agent_type = Column(SQLEnum(AgentType), nullable=False)
    description = Column(Text, nullable=True)
    capabilities = Column(JSONType, nullable=False, default=list)
    configuration = Column(JSONType, nullable=False, default=dict)
    status = Column(SQLEnum(AgentStatus), nullable=False, default=AgentStatus.INACTIVE)
    version = Column(String(50), nullable=False, default="1.0.0")
    endpoint_url = Column(String(500), nullable=True)
    health_check_url = Column(String(500), nullable=True)
    last_health_check = Column(DateTime, nullable=True)
    is_enabled = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    agent_requests = relationship("AgentRequest", back_populates="agent", cascade="all, delete-orphan")
    agent_responses = relationship("AgentResponse", back_populates="agent", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_agent_type', 'agent_type'),
        Index('idx_agent_status', 'status'),
        Index('idx_agent_enabled', 'is_enabled'),
        UniqueConstraint('name', 'agent_type', name='uq_agent_name_type'),
    )
    
    @validates('capabilities')
    def validate_capabilities(self, key, capabilities):
        """Validate capabilities is a list."""
        if not isinstance(capabilities, list):
            raise ValueError("Capabilities must be a list")
        return capabilities
    
    @validates('configuration')
    def validate_configuration(self, key, configuration):
        """Validate configuration is a dict."""
        if not isinstance(configuration, dict):
            raise ValueError("Configuration must be a dictionary")
        return configuration
    
    def __repr__(self):
        return f"<Agent(id={self.id}, name={self.name}, type={self.agent_type}, status={self.status})>"


class Dataset(Base):
    """Dataset model for data source management."""
    
    __tablename__ = "datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    source_type = Column(String(50), nullable=False)  # csv, xlsx, sql, json, api
    data_schema = Column(JSONType, nullable=False, default=dict)
    dataset_metadata = Column(JSONType, nullable=False, default=dict)
    row_count = Column(Integer, nullable=True)
    file_path = Column(String(1000), nullable=True)
    connection_string = Column(String(1000), nullable=True)
    table_name = Column(String(255), nullable=True)
    query = Column(Text, nullable=True)
    refresh_interval_minutes = Column(Integer, nullable=True)
    last_refreshed = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    ml_models = relationship("MLModel", back_populates="dataset", cascade="all, delete-orphan")
    dashboards = relationship("Dashboard", secondary="dashboard_datasets", back_populates="datasets")
    
    # Indexes
    __table_args__ = (
        Index('idx_dataset_source_type', 'source_type'),
        Index('idx_dataset_active', 'is_active'),
        Index('idx_dataset_name', 'name'),
    )
    
    @validates('data_schema')
    def validate_data_schema(self, key, data_schema):
        """Validate data_schema is a dict."""
        if not isinstance(data_schema, dict):
            raise ValueError("Data schema must be a dictionary")
        return data_schema
    
    @validates('dataset_metadata')
    def validate_dataset_metadata(self, key, dataset_metadata):
        """Validate dataset_metadata is a dict."""
        if not isinstance(dataset_metadata, dict):
            raise ValueError("Dataset metadata must be a dictionary")
        return dataset_metadata
    
    @validates('source_type')
    def validate_source_type(self, key, source_type):
        """Validate source type is supported."""
        allowed_types = ['csv', 'xlsx', 'sql', 'json', 'api', 'database']
        if source_type not in allowed_types:
            raise ValueError(f"Source type must be one of: {allowed_types}")
        return source_type
    
    def __repr__(self):
        return f"<Dataset(id={self.id}, name={self.name}, source_type={self.source_type})>"


class MLModel(Base):
    """ML Model model for trained model management."""
    
    __tablename__ = "ml_models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    algorithm = Column(String(100), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    parameters = Column(JSONType, nullable=False, default=dict)
    hyperparameters = Column(JSONType, nullable=False, default=dict)
    metrics = Column(JSONType, nullable=False, default=dict)
    feature_columns = Column(JSONType, nullable=False, default=list)
    target_column = Column(String(255), nullable=True)
    model_path = Column(String(1000), nullable=False)
    model_size_bytes = Column(Integer, nullable=True)
    training_duration_seconds = Column(Float, nullable=True)
    api_endpoint = Column(String(500), nullable=True)
    version = Column(String(50), nullable=False, default="1.0.0")
    is_deployed = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="ml_models")
    
    # Indexes
    __table_args__ = (
        Index('idx_mlmodel_algorithm', 'algorithm'),
        Index('idx_mlmodel_dataset', 'dataset_id'),
        Index('idx_mlmodel_deployed', 'is_deployed'),
        Index('idx_mlmodel_active', 'is_active'),
    )
    
    @validates('parameters')
    def validate_parameters(self, key, parameters):
        """Validate parameters is a dict."""
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary")
        return parameters
    
    @validates('hyperparameters')
    def validate_hyperparameters(self, key, hyperparameters):
        """Validate hyperparameters is a dict."""
        if not isinstance(hyperparameters, dict):
            raise ValueError("Hyperparameters must be a dictionary")
        return hyperparameters
    
    @validates('metrics')
    def validate_metrics(self, key, metrics):
        """Validate metrics is a dict."""
        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be a dictionary")
        return metrics
    
    @validates('feature_columns')
    def validate_feature_columns(self, key, feature_columns):
        """Validate feature_columns is a list."""
        if not isinstance(feature_columns, list):
            raise ValueError("Feature columns must be a list")
        return feature_columns
    
    def __repr__(self):
        return f"<MLModel(id={self.id}, name={self.name}, algorithm={self.algorithm})>"


# Association table for many-to-many relationship between dashboards and datasets
from sqlalchemy import Table
dashboard_datasets = Table(
    'dashboard_datasets',
    Base.metadata,
    Column('dashboard_id', UUID(as_uuid=True), ForeignKey('dashboards.id'), primary_key=True),
    Column('dataset_id', UUID(as_uuid=True), ForeignKey('datasets.id'), primary_key=True),
    Index('idx_dashboard_datasets_dashboard', 'dashboard_id'),
    Index('idx_dashboard_datasets_dataset', 'dataset_id'),
)


class Dashboard(Base):
    """Dashboard model for BI dashboard management."""
    
    __tablename__ = "dashboards"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    config = Column(JSONType, nullable=False, default=dict)
    layout = Column(JSONType, nullable=False, default=dict)
    charts = Column(JSONType, nullable=False, default=list)
    filters = Column(JSONType, nullable=False, default=dict)
    refresh_interval_minutes = Column(Integer, nullable=True, default=60)
    last_refreshed = Column(DateTime, nullable=True)
    is_public = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    tags = Column(JSONType, nullable=False, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="dashboards")
    datasets = relationship("Dataset", secondary=dashboard_datasets, back_populates="dashboards")
    
    # Indexes
    __table_args__ = (
        Index('idx_dashboard_user', 'user_id'),
        Index('idx_dashboard_public', 'is_public'),
        Index('idx_dashboard_active', 'is_active'),
        Index('idx_dashboard_name', 'name'),
    )
    
    @validates('config')
    def validate_config(self, key, config):
        """Validate config is a dict."""
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        return config
    
    @validates('layout')
    def validate_layout(self, key, layout):
        """Validate layout is a dict."""
        if not isinstance(layout, dict):
            raise ValueError("Layout must be a dictionary")
        return layout
    
    @validates('charts')
    def validate_charts(self, key, charts):
        """Validate charts is a list."""
        if not isinstance(charts, list):
            raise ValueError("Charts must be a list")
        return charts
    
    @validates('filters')
    def validate_filters(self, key, filters):
        """Validate filters is a dict."""
        if not isinstance(filters, dict):
            raise ValueError("Filters must be a dictionary")
        return filters
    
    @validates('tags')
    def validate_tags(self, key, tags):
        """Validate tags is a list."""
        if not isinstance(tags, list):
            raise ValueError("Tags must be a list")
        return tags
    
    def __repr__(self):
        return f"<Dashboard(id={self.id}, name={self.name}, user_id={self.user_id})>"


class AgentRequest(Base):
    """Agent request model for tracking user requests to agents."""
    
    __tablename__ = "agent_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)
    prompt = Column(Text, nullable=False)
    context = Column(JSONType, nullable=False, default=dict)
    priority = Column(Integer, nullable=False, default=1)
    status = Column(String(50), nullable=False, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="agent_requests")
    agent = relationship("Agent", back_populates="agent_requests")
    responses = relationship("AgentResponse", back_populates="request", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_agent_request_user', 'user_id'),
        Index('idx_agent_request_agent', 'agent_id'),
        Index('idx_agent_request_status', 'status'),
        Index('idx_agent_request_created', 'created_at'),
        Index('idx_agent_request_priority', 'priority'),
    )
    
    @validates('context')
    def validate_context(self, key, context):
        """Validate context is a dict."""
        if not isinstance(context, dict):
            raise ValueError("Context must be a dictionary")
        return context
    
    @validates('priority')
    def validate_priority(self, key, priority):
        """Validate priority is between 1 and 10."""
        if not isinstance(priority, int) or priority < 1 or priority > 10:
            raise ValueError("Priority must be an integer between 1 and 10")
        return priority
    
    def __repr__(self):
        return f"<AgentRequest(id={self.id}, user_id={self.user_id}, agent_id={self.agent_id}, status={self.status})>"


class AgentResponse(Base):
    """Agent response model for tracking agent responses."""
    
    __tablename__ = "agent_responses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(UUID(as_uuid=True), ForeignKey("agent_requests.id"), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)
    content = Column(Text, nullable=False)
    artifacts = Column(JSONType, nullable=False, default=list)
    response_metadata = Column(JSONType, nullable=False, default=dict)
    execution_time_seconds = Column(Float, nullable=True)
    status = Column(SQLEnum(ResponseStatus), nullable=False, default=ResponseStatus.SUCCESS)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    request = relationship("AgentRequest", back_populates="responses")
    agent = relationship("Agent", back_populates="agent_responses")
    
    # Indexes
    __table_args__ = (
        Index('idx_agent_response_request', 'request_id'),
        Index('idx_agent_response_agent', 'agent_id'),
        Index('idx_agent_response_status', 'status'),
        Index('idx_agent_response_created', 'created_at'),
    )
    
    @validates('artifacts')
    def validate_artifacts(self, key, artifacts):
        """Validate artifacts is a list."""
        if not isinstance(artifacts, list):
            raise ValueError("Artifacts must be a list")
        return artifacts
    
    @validates('response_metadata')
    def validate_response_metadata(self, key, response_metadata):
        """Validate response_metadata is a dict."""
        if not isinstance(response_metadata, dict):
            raise ValueError("Response metadata must be a dictionary")
        return response_metadata
    
    def __repr__(self):
        return f"<AgentResponse(id={self.id}, request_id={self.request_id}, status={self.status})>"


class FileUpload(Base):
    """File upload model for tracking uploaded files and processing status."""
    
    __tablename__ = "file_uploads"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    upload_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(100), nullable=False)
    detected_type = Column(String(50), nullable=False)  # csv, xlsx, json, sql
    schema_info = Column(JSONType, nullable=False, default=dict)
    preview_data = Column(JSONType, nullable=True)
    quality_report = Column(JSONType, nullable=True)
    processing_status = Column(String(50), nullable=False, default="pending")  # pending, processing, completed, failed
    processing_progress = Column(Float, nullable=False, default=0.0)
    processing_message = Column(Text, nullable=True)
    error_details = Column(JSONType, nullable=True)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User")
    dataset = relationship("Dataset")
    
    # Indexes
    __table_args__ = (
        Index('idx_file_upload_user', 'user_id'),
        Index('idx_file_upload_status', 'processing_status'),
        Index('idx_file_upload_type', 'detected_type'),
        Index('idx_file_upload_created', 'created_at'),
    )
    
    @validates('schema_info')
    def validate_schema_info(self, key, schema_info):
        """Validate schema_info is a dict."""
        if not isinstance(schema_info, dict):
            raise ValueError("Schema info must be a dictionary")
        return schema_info
    
    @validates('preview_data')
    def validate_preview_data(self, key, preview_data):
        """Validate preview_data is a dict or None."""
        if preview_data is not None and not isinstance(preview_data, dict):
            raise ValueError("Preview data must be a dictionary or None")
        return preview_data
    
    @validates('quality_report')
    def validate_quality_report(self, key, quality_report):
        """Validate quality_report is a dict or None."""
        if quality_report is not None and not isinstance(quality_report, dict):
            raise ValueError("Quality report must be a dictionary or None")
        return quality_report
    
    @validates('error_details')
    def validate_error_details(self, key, error_details):
        """Validate error_details is a dict or None."""
        if error_details is not None and not isinstance(error_details, dict):
            raise ValueError("Error details must be a dictionary or None")
        return error_details
    
    @validates('detected_type')
    def validate_detected_type(self, key, detected_type):
        """Validate detected type is supported."""
        allowed_types = ['csv', 'xlsx', 'json', 'sql']
        if detected_type not in allowed_types:
            raise ValueError(f"Detected type must be one of: {allowed_types}")
        return detected_type
    
    def __repr__(self):
        return f"<FileUpload(id={self.id}, filename={self.filename}, detected_type={self.detected_type}, status={self.processing_status})>"


class VaultInsight(Base):
    """Vault insight model for secure storage of AI-generated insights."""
    
    __tablename__ = "vault_insights"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)  # Encrypted content
    insight_type = Column(String(50), nullable=False)  # analysis_result, model_explanation, etc.
    access_level = Column(String(50), nullable=False)  # public, internal, confidential, etc.
    retention_policy = Column(String(50), nullable=False)  # permanent, long_term, etc.
    creator_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    organization_id = Column(String(255), nullable=False, default="default")
    tags = Column(JSONType, nullable=False, default=list)
    insight_metadata = Column(JSONType, nullable=False, default=dict)
    version = Column(Integer, nullable=False, default=1)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("vault_insights.id"), nullable=True)
    encryption_key_id = Column(String(255), nullable=True)
    content_hash = Column(String(64), nullable=True)  # SHA-256 hash
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    creator = relationship("User")
    parent = relationship("VaultInsight", remote_side=[id])
    children = relationship("VaultInsight", back_populates="parent")
    access_logs = relationship("VaultAccessLog", back_populates="insight", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_vault_insight_creator', 'creator_id'),
        Index('idx_vault_insight_type', 'insight_type'),
        Index('idx_vault_insight_access_level', 'access_level'),
        Index('idx_vault_insight_organization', 'organization_id'),
        Index('idx_vault_insight_created', 'created_at'),
        Index('idx_vault_insight_expires', 'expires_at'),
        Index('idx_vault_insight_parent', 'parent_id'),
        Index('idx_vault_insight_hash', 'content_hash'),
    )
    
    @validates('tags')
    def validate_tags(self, key, tags):
        """Validate tags is a list."""
        if not isinstance(tags, list):
            raise ValueError("Tags must be a list")
        return tags
    
    @validates('insight_metadata')
    def validate_insight_metadata(self, key, insight_metadata):
        """Validate insight_metadata is a dict."""
        if not isinstance(insight_metadata, dict):
            raise ValueError("Insight metadata must be a dictionary")
        return insight_metadata
    
    @validates('insight_type')
    def validate_insight_type(self, key, insight_type):
        """Validate insight type is supported."""
        allowed_types = [
            'analysis_result', 'model_explanation', 'prediction', 'report',
            'visualization', 'recommendation', 'audit_log', 'research_finding'
        ]
        if insight_type not in allowed_types:
            raise ValueError(f"Insight type must be one of: {allowed_types}")
        return insight_type
    
    @validates('access_level')
    def validate_access_level(self, key, access_level):
        """Validate access level is supported."""
        allowed_levels = ['public', 'internal', 'confidential', 'restricted', 'top_secret']
        if access_level not in allowed_levels:
            raise ValueError(f"Access level must be one of: {allowed_levels}")
        return access_level
    
    @validates('retention_policy')
    def validate_retention_policy(self, key, retention_policy):
        """Validate retention policy is supported."""
        allowed_policies = ['permanent', 'long_term', 'medium_term', 'short_term', 'temporary']
        if retention_policy not in allowed_policies:
            raise ValueError(f"Retention policy must be one of: {allowed_policies}")
        return retention_policy
    
    def __repr__(self):
        return f"<VaultInsight(id={self.id}, title={self.title}, type={self.insight_type}, access_level={self.access_level})>"


class VaultAccessLog(Base):
    """Vault access log model for audit trails of insight access."""
    
    __tablename__ = "vault_access_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    insight_id = Column(UUID(as_uuid=True), ForeignKey("vault_insights.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    action = Column(String(50), nullable=False)  # read, write, delete, share
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    user_agent = Column(String(500), nullable=True)
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    insight = relationship("VaultInsight", back_populates="access_logs")
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_vault_access_insight', 'insight_id'),
        Index('idx_vault_access_user', 'user_id'),
        Index('idx_vault_access_action', 'action'),
        Index('idx_vault_access_timestamp', 'timestamp'),
        Index('idx_vault_access_success', 'success'),
    )
    
    @validates('action')
    def validate_action(self, key, action):
        """Validate action is supported."""
        allowed_actions = ['read', 'write', 'delete', 'share', 'update']
        if action not in allowed_actions:
            raise ValueError(f"Action must be one of: {allowed_actions}")
        return action
    
    def __repr__(self):
        return f"<VaultAccessLog(id={self.id}, insight_id={self.insight_id}, action={self.action}, timestamp={self.timestamp})>"


class PromptTemplate(Base):
    """Prompt template model for industry-specific prompt templates."""
    
    __tablename__ = "prompt_templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=False)  # data_analysis, code_generation, etc.
    industry = Column(String(100), nullable=True)
    use_case = Column(String(255), nullable=True)
    template_content = Column(Text, nullable=False)
    variables = Column(JSONType, nullable=False, default=list)  # List of template variables
    tags = Column(JSONType, nullable=False, default=list)
    usage_count = Column(Integer, nullable=False, default=0)
    performance_score = Column(Float, nullable=True)
    creator_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    creator = relationship("User")
    prompt_tests = relationship("PromptTest", back_populates="template", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_prompt_template_category', 'category'),
        Index('idx_prompt_template_industry', 'industry'),
        Index('idx_prompt_template_creator', 'creator_id'),
        Index('idx_prompt_template_public', 'is_public'),
        Index('idx_prompt_template_active', 'is_active'),
        Index('idx_prompt_template_usage', 'usage_count'),
    )
    
    @validates('variables')
    def validate_variables(self, key, variables):
        """Validate variables is a list."""
        if not isinstance(variables, list):
            raise ValueError("Variables must be a list")
        return variables
    
    @validates('tags')
    def validate_tags(self, key, tags):
        """Validate tags is a list."""
        if not isinstance(tags, list):
            raise ValueError("Tags must be a list")
        return tags
    
    def __repr__(self):
        return f"<PromptTemplate(id={self.id}, name={self.name}, category={self.category})>"


class PromptHistory(Base):
    """Prompt history model for tracking prompt optimization history."""
    
    __tablename__ = "prompt_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    original_prompt = Column(Text, nullable=False)
    optimized_prompt = Column(Text, nullable=False)
    optimization_strategy = Column(String(100), nullable=False)
    performance_improvement = Column(Float, nullable=True)
    success_rate_before = Column(Float, nullable=True)
    success_rate_after = Column(Float, nullable=True)
    response_time_before = Column(Float, nullable=True)
    response_time_after = Column(Float, nullable=True)
    test_cases_count = Column(Integer, nullable=True)
    optimization_metadata = Column(JSONType, nullable=False, default=dict)
    feedback_score = Column(Float, nullable=True)  # User feedback on optimization
    is_favorite = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User")
    prompt_tests = relationship("PromptTest", back_populates="history", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_prompt_history_user', 'user_id'),
        Index('idx_prompt_history_strategy', 'optimization_strategy'),
        Index('idx_prompt_history_created', 'created_at'),
        Index('idx_prompt_history_favorite', 'is_favorite'),
        Index('idx_prompt_history_performance', 'performance_improvement'),
    )
    
    @validates('optimization_metadata')
    def validate_optimization_metadata(self, key, optimization_metadata):
        """Validate optimization_metadata is a dict."""
        if not isinstance(optimization_metadata, dict):
            raise ValueError("Optimization metadata must be a dictionary")
        return optimization_metadata
    
    def __repr__(self):
        return f"<PromptHistory(id={self.id}, user_id={self.user_id}, strategy={self.optimization_strategy})>"


class PromptTest(Base):
    """Prompt test model for A/B testing and performance evaluation."""
    
    __tablename__ = "prompt_tests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    test_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    template_id = Column(UUID(as_uuid=True), ForeignKey("prompt_templates.id"), nullable=True)
    history_id = Column(UUID(as_uuid=True), ForeignKey("prompt_history.id"), nullable=True)
    test_type = Column(String(50), nullable=False)  # a_b_test, performance_test, variation_test
    status = Column(String(50), nullable=False, default="pending")  # pending, running, completed, failed
    prompt_variations = Column(JSONType, nullable=False, default=list)
    test_cases = Column(JSONType, nullable=False, default=list)
    test_results = Column(JSONType, nullable=False, default=dict)
    performance_metrics = Column(JSONType, nullable=False, default=dict)
    statistical_analysis = Column(JSONType, nullable=False, default=dict)
    winner_variation_id = Column(String(255), nullable=True)
    confidence_level = Column(Float, nullable=True)
    total_test_runs = Column(Integer, nullable=False, default=0)
    successful_runs = Column(Integer, nullable=False, default=0)
    average_response_time = Column(Float, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User")
    template = relationship("PromptTemplate", back_populates="prompt_tests")
    history = relationship("PromptHistory", back_populates="prompt_tests")
    
    # Indexes
    __table_args__ = (
        Index('idx_prompt_test_user', 'user_id'),
        Index('idx_prompt_test_template', 'template_id'),
        Index('idx_prompt_test_history', 'history_id'),
        Index('idx_prompt_test_type', 'test_type'),
        Index('idx_prompt_test_status', 'status'),
        Index('idx_prompt_test_created', 'created_at'),
        Index('idx_prompt_test_completed', 'completed_at'),
    )
    
    @validates('prompt_variations')
    def validate_prompt_variations(self, key, prompt_variations):
        """Validate prompt_variations is a list."""
        if not isinstance(prompt_variations, list):
            raise ValueError("Prompt variations must be a list")
        return prompt_variations
    
    @validates('test_cases')
    def validate_test_cases(self, key, test_cases):
        """Validate test_cases is a list."""
        if not isinstance(test_cases, list):
            raise ValueError("Test cases must be a list")
        return test_cases
    
    @validates('test_results')
    def validate_test_results(self, key, test_results):
        """Validate test_results is a dict."""
        if not isinstance(test_results, dict):
            raise ValueError("Test results must be a dictionary")
        return test_results
    
    @validates('performance_metrics')
    def validate_performance_metrics(self, key, performance_metrics):
        """Validate performance_metrics is a dict."""
        if not isinstance(performance_metrics, dict):
            raise ValueError("Performance metrics must be a dictionary")
        return performance_metrics
    
    @validates('statistical_analysis')
    def validate_statistical_analysis(self, key, statistical_analysis):
        """Validate statistical_analysis is a dict."""
        if not isinstance(statistical_analysis, dict):
            raise ValueError("Statistical analysis must be a dictionary")
        return statistical_analysis
    
    def __repr__(self):
        return f"<PromptTest(id={self.id}, name={self.test_name}, type={self.test_type}, status={self.status})>"


class AuditLog(Base):
    """Audit log model for EXOUSIA security tracking."""
    
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(255), nullable=True)
    details = Column(JSONType, nullable=False, default=dict)
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    user_agent = Column(String(500), nullable=True)
    session_id = Column(String(255), nullable=True)
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_log_user', 'user_id'),
        Index('idx_audit_log_action', 'action'),
        Index('idx_audit_log_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_log_timestamp', 'timestamp'),
        Index('idx_audit_log_success', 'success'),
        Index('idx_audit_log_session', 'session_id'),
    )
    
    @validates('details')
    def validate_details(self, key, details):
        """Validate details is a dict."""
        if not isinstance(details, dict):
            raise ValueError("Details must be a dictionary")
        return details
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, resource_type={self.resource_type}, timestamp={self.timestamp})>"

# Database session management functions
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import Generator

# Global session factory
_SessionLocal = None
_engine = None

def init_database_session(database_url: str = None):
    """Initialize database session factory"""
    global _SessionLocal, _engine
    
    if database_url is None:
        # Try to get from environment or use default
        database_url = os.getenv('DATABASE_URL', 'sqlite:///./scrollintel.db')
    
    # Create engine
    if database_url.startswith('sqlite'):
        _engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            echo=False
        )
    else:
        _engine = create_engine(database_url, echo=False)
    
    # Create session factory
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    
    # Create tables
    Base.metadata.create_all(bind=_engine)

@contextmanager
def get_db_session() -> Generator:
    """Get database session with automatic cleanup"""
    global _SessionLocal
    
    if _SessionLocal is None:
        init_database_session()
    
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def get_engine():
    """Get database engine"""
    global _engine
    if _engine is None:
        init_database_session()
    return _engine
