"""
Database models for ScrollIntel Core
Simplified schema focused on core functionality
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from .database import Base


class User(Base):
    """User accounts"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="user")  # admin, user, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_active = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    workspaces = relationship("WorkspaceMember", back_populates="user")
    datasets = relationship("Dataset", back_populates="owner")
    analyses = relationship("Analysis", back_populates="user")


class Workspace(Base):
    """Project workspaces for organizing data and analyses"""
    __tablename__ = "workspaces"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    settings = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    members = relationship("WorkspaceMember", back_populates="workspace")
    datasets = relationship("Dataset", back_populates="workspace")
    dashboards = relationship("Dashboard", back_populates="workspace")


class WorkspaceMember(Base):
    """Workspace membership with roles"""
    __tablename__ = "workspace_members"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(String(50), default="member")  # owner, admin, member, viewer
    joined_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    workspace = relationship("Workspace", back_populates="members")
    user = relationship("User", back_populates="workspaces")


class Dataset(Base):
    """Uploaded datasets"""
    __tablename__ = "datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    file_type = Column(String(50))  # csv, excel, json, parquet
    schema_info = Column(JSON)  # Column names, types, sample data
    row_count = Column(Integer)
    column_count = Column(Integer)
    
    # Relationships
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    workspace = relationship("Workspace", back_populates="datasets")
    owner = relationship("User", back_populates="datasets")
    analyses = relationship("Analysis", back_populates="dataset")
    models = relationship("Model", back_populates="dataset")


class Analysis(Base):
    """Agent analysis results"""
    __tablename__ = "analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    agent_type = Column(String(50), nullable=False)  # cto, data_scientist, ml_engineer, etc.
    query = Column(Text, nullable=False)
    results = Column(JSON)
    metadata = Column(JSON, default={})
    processing_time = Column(Float)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    dataset = relationship("Dataset", back_populates="analyses")
    user = relationship("User", back_populates="analyses")


class Model(Base):
    """ML models created by agents"""
    __tablename__ = "models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    model_type = Column(String(100))  # classification, regression, clustering, etc.
    algorithm = Column(String(100))  # random_forest, linear_regression, etc.
    
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    
    # Model artifacts
    model_path = Column(String(500))  # Path to serialized model
    performance_metrics = Column(JSON)  # Accuracy, precision, recall, etc.
    feature_importance = Column(JSON)
    training_config = Column(JSON)
    
    # Deployment
    status = Column(String(50), default="trained")  # trained, deployed, archived
    endpoint_url = Column(String(500))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    dataset = relationship("Dataset", back_populates="models")


class Dashboard(Base):
    """BI dashboards"""
    __tablename__ = "dashboards"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    
    # Dashboard configuration
    config = Column(JSON)  # Chart configurations, layout, etc.
    charts = Column(JSON)  # Chart definitions
    data_sources = Column(JSON)  # Connected datasets
    
    # Settings
    auto_refresh = Column(Boolean, default=False)
    refresh_interval = Column(Integer, default=300)  # seconds
    is_public = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    workspace = relationship("Workspace", back_populates="dashboards")


class AgentSession(Base):
    """Agent conversation sessions with NL processing"""
    __tablename__ = "agent_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_data = Column(JSON, default={})  # Conversation history, context
    last_agent = Column(String(50))
    last_intent = Column(String(50))  # Last detected intent
    context_cache = Column(JSON, default={})  # Cached context for multi-turn
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ConversationTurn(Base):
    """Individual conversation turns for detailed tracking"""
    __tablename__ = "conversation_turns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("agent_sessions.id"), nullable=False)
    user_query = Column(Text, nullable=False)
    detected_intent = Column(String(50))
    intent_confidence = Column(Float)
    extracted_entities = Column(JSON, default=[])  # List of entities
    agent_used = Column(String(50))
    agent_response = Column(JSON)
    processing_time = Column(Float)
    success = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    """Audit trail for security and compliance"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(100))
    details = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())