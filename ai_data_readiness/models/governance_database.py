"""Database models for data governance."""

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, Boolean, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from .database import Base
from .governance_models import AccessLevel, PolicyType, DataClassification, AuditEventType, PolicyStatus


# Association tables for many-to-many relationships
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id'), primary_key=True),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id'), primary_key=True)
)

role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id'), primary_key=True),
    Column('permission_id', UUID(as_uuid=True), ForeignKey('permissions.id'), primary_key=True)
)


class UserModel(Base):
    """SQLAlchemy model for users."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    full_name = Column(String(255))
    department = Column(String(255))
    role = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    roles = relationship("RoleModel", secondary=user_roles, back_populates="users")
    audit_events = relationship("AuditEventModel", back_populates="user")


class RoleModel(Base):
    """SQLAlchemy model for roles."""
    __tablename__ = "roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    is_system_role = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    users = relationship("UserModel", secondary=user_roles, back_populates="roles")
    permissions = relationship("PermissionModel", secondary=role_permissions, back_populates="roles")


class PermissionModel(Base):
    """SQLAlchemy model for permissions."""
    __tablename__ = "permissions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    resource_type = Column(String(100))
    action = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    roles = relationship("RoleModel", secondary=role_permissions, back_populates="permissions")


class DataCatalogEntryModel(Base):
    """SQLAlchemy model for data catalog entries."""
    __tablename__ = "data_catalog"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey('datasets.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    classification = Column(String(50), default=DataClassification.INTERNAL.value)
    owner = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    steward = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    business_glossary_terms = Column(JSON)
    tags = Column(JSON)
    schema_info = Column(JSON)
    lineage_info = Column(JSON)
    quality_metrics = Column(JSON)
    usage_statistics = Column(JSON)
    retention_policy = Column(String(255))
    compliance_requirements = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed = Column(DateTime)
    
    # Relationships
    owner_user = relationship("UserModel", foreign_keys=[owner])
    steward_user = relationship("UserModel", foreign_keys=[steward])


class GovernancePolicyModel(Base):
    """SQLAlchemy model for governance policies."""
    __tablename__ = "governance_policies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    policy_type = Column(String(50), nullable=False)
    status = Column(String(50), default=PolicyStatus.DRAFT.value)
    rules = Column(JSON)
    conditions = Column(JSON)
    enforcement_level = Column(String(50), default="strict")
    applicable_resources = Column(JSON)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    approved_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    effective_date = Column(DateTime)
    expiry_date = Column(DateTime)
    version = Column(String(50), default="1.0")
    
    # Relationships
    creator = relationship("UserModel", foreign_keys=[created_by])
    approver = relationship("UserModel", foreign_keys=[approved_by])


class AccessControlEntryModel(Base):
    """SQLAlchemy model for access control entries."""
    __tablename__ = "access_control_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_id = Column(String(255), nullable=False)
    resource_type = Column(String(100), nullable=False)
    principal_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    principal_type = Column(String(50), default="user")
    access_level = Column(String(50), nullable=False)
    granted_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    granted_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    conditions = Column(JSON)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("UserModel", foreign_keys=[principal_id])
    granter = relationship("UserModel", foreign_keys=[granted_by])


class AuditEventModel(Base):
    """SQLAlchemy model for audit events."""
    __tablename__ = "audit_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(50), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    resource_id = Column(String(255))
    resource_type = Column(String(100))
    action = Column(String(255), nullable=False)
    details = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    session_id = Column(String(255))
    timestamp = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # Relationships
    user = relationship("UserModel", back_populates="audit_events")


class UsageMetricsModel(Base):
    """SQLAlchemy model for usage metrics."""
    __tablename__ = "usage_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_id = Column(String(255), nullable=False)
    resource_type = Column(String(100), nullable=False)
    access_count = Column(Integer, default=0)
    unique_users = Column(Integer, default=0)
    last_accessed = Column(DateTime)
    most_frequent_user = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    access_patterns = Column(JSON)
    performance_metrics = Column(JSON)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Relationships
    frequent_user = relationship("UserModel")


class ComplianceReportModel(Base):
    """SQLAlchemy model for compliance reports."""
    __tablename__ = "compliance_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_type = Column(String(100), nullable=False)
    scope = Column(JSON)
    compliance_score = Column(Float, default=0.0)
    violations = Column(JSON)
    recommendations = Column(JSON)
    assessment_criteria = Column(JSON)
    generated_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    generated_at = Column(DateTime, default=datetime.utcnow)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Relationships
    generator = relationship("UserModel")


class DataLineageNodeModel(Base):
    """SQLAlchemy model for data lineage nodes."""
    __tablename__ = "data_lineage_nodes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_id = Column(String(255), nullable=False)
    resource_type = Column(String(100), nullable=False)
    name = Column(String(255), nullable=False)
    node_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class DataLineageEdgeModel(Base):
    """SQLAlchemy model for data lineage edges."""
    __tablename__ = "data_lineage_edges"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_node_id = Column(UUID(as_uuid=True), ForeignKey('data_lineage_nodes.id'))
    target_node_id = Column(UUID(as_uuid=True), ForeignKey('data_lineage_nodes.id'))
    relationship_type = Column(String(100), nullable=False)
    transformation_details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    source_node = relationship("DataLineageNodeModel", foreign_keys=[source_node_id])
    target_node = relationship("DataLineageNodeModel", foreign_keys=[target_node_id])


class GovernanceMetricsModel(Base):
    """SQLAlchemy model for governance metrics."""
    __tablename__ = "governance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    total_datasets = Column(Integer, default=0)
    classified_datasets = Column(Integer, default=0)
    policy_violations = Column(Integer, default=0)
    compliance_score = Column(Float, default=0.0)
    data_quality_score = Column(Float, default=0.0)
    access_requests_pending = Column(Integer, default=0)
    audit_events_count = Column(Integer, default=0)
    active_users = Column(Integer, default=0)
    data_stewards = Column(Integer, default=0)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    calculated_at = Column(DateTime, default=datetime.utcnow)