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
import uuid

from ..core.interfaces import AgentType, AgentStatus, ResponseStatus, UserRole

Base = declarative_base()


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.VIEWER)
    permissions = Column(JSONB, nullable=False, default=list)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    agent_requests = relationship("AgentRequest", back_populates="user", cascade="all, delete-orphan")
    dashboards = relationship("Dashboard", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    
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
    type = Column(SQLEnum(AgentType), nullable=False)
    description = Column(Text, nullable=True)
    capabilities = Column(JSONB, nullable=False, default=list)
    configuration = Column(JSONB, nullable=False, default=dict)
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
        Index('idx_agent_type', 'type'),
        Index('idx_agent_status', 'status'),
        Index('idx_agent_enabled', 'is_enabled'),
        UniqueConstraint('name', 'type', name='uq_agent_name_type'),
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
        return f"<Agent(id={self.id}, name={self.name}, type={self.type}, status={self.status})>"


class Dataset(Base):
    """Dataset model for data source management."""
    
    __tablename__ = "datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    source_type = Column(String(50), nullable=False)  # csv, xlsx, sql, json, api
    schema = Column(JSONB, nullable=False, default=dict)
    dataset_metadata = Column(JSONB, nullable=False, default=dict)
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
    
    @validates('schema')
    def validate_schema(self, key, schema):
        """Validate schema is a dict."""
        if not isinstance(schema, dict):
            raise ValueError("Schema must be a dictionary")
        return schema
    
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
    parameters = Column(JSONB, nullable=False, default=dict)
    hyperparameters = Column(JSONB, nullable=False, default=dict)
    metrics = Column(JSONB, nullable=False, default=dict)
    feature_columns = Column(JSONB, nullable=False, default=list)
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
    config = Column(JSONB, nullable=False, default=dict)
    layout = Column(JSONB, nullable=False, default=dict)
    charts = Column(JSONB, nullable=False, default=list)
    filters = Column(JSONB, nullable=False, default=dict)
    refresh_interval_minutes = Column(Integer, nullable=True, default=60)
    last_refreshed = Column(DateTime, nullable=True)
    is_public = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
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
    
    def __repr__(self):
        return f"<Dashboard(id={self.id}, name={self.name}, user_id={self.user_id})>"


class AgentRequest(Base):
    """Agent request model for tracking user requests to agents."""
    
    __tablename__ = "agent_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)
    prompt = Column(Text, nullable=False)
    context = Column(JSONB, nullable=False, default=dict)
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
    artifacts = Column(JSONB, nullable=False, default=list)
    metadata = Column(JSONB, nullable=False, default=dict)
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
    
    @validates('metadata')
    def validate_metadata(self, key, metadata):
        """Validate metadata is a dict."""
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")
        return metadata
    
    def __repr__(self):
        return f"<AgentResponse(id={self.id}, request_id={self.request_id}, status={self.status})>"


class AuditLog(Base):
    """Audit log model for EXOUSIA security tracking."""
    
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(255), nullable=True)
    details = Column(JSONB, nullable=False, default=dict)
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