"""
Data Product Registry Models

Core database models for the Data Product Registry system including
data products, schemas, metadata, and versioning support.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from sqlalchemy import Column, String, DateTime, Text, JSON, Float, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import TypeDecorator, String as SQLString
import uuid

Base = declarative_base()


class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses String(36).
    """
    impl = SQLString
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(SQLString(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(value)
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value


class AccessLevel(Enum):
    """Access levels for data products"""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class VerificationStatus(Enum):
    """Verification status for data products"""
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class DataProduct(Base):
    """Core data product entity with comprehensive metadata"""
    __tablename__ = "data_products"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    description = Column(Text)
    schema_definition = Column(JSON, nullable=False)
    product_metadata = Column(JSON, default={})
    
    # Quality and compliance metrics
    quality_score = Column(Float, default=0.0)
    bias_score = Column(Float, default=0.0)
    compliance_tags = Column(JSON, default=[])
    
    # Access control and governance
    owner = Column(String(255), nullable=False, index=True)
    access_level = Column(String(20), nullable=False, default=AccessLevel.INTERNAL.value)
    verification_status = Column(String(20), nullable=False, default=VerificationStatus.PENDING.value)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    freshness_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    provenance = relationship("DataProvenance", back_populates="data_product", uselist=False)
    quality_metrics = relationship("QualityMetrics", back_populates="data_product")
    bias_assessments = relationship("BiasAssessment", back_populates="data_product")
    versions = relationship("DataProductVersion", back_populates="data_product")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_data_product_name_version', 'name', 'version'),
        Index('idx_data_product_owner', 'owner'),
        Index('idx_data_product_created_at', 'created_at'),
        Index('idx_data_product_verification_status', 'verification_status'),
    )


class DataProductVersion(Base):
    """Version control for data products with hash-based tracking"""
    __tablename__ = "data_product_versions"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    data_product_id = Column(GUID(), ForeignKey('data_products.id'), nullable=False)
    version_hash = Column(String(64), nullable=False, unique=True, index=True)
    version_number = Column(String(50), nullable=False)
    schema_hash = Column(String(64), nullable=False)
    
    # Version metadata
    change_description = Column(Text)
    change_type = Column(String(50))  # major, minor, patch, hotfix
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(255), nullable=False)
    
    # Relationships
    data_product = relationship("DataProduct", back_populates="versions")
    
    __table_args__ = (
        Index('idx_version_data_product_id', 'data_product_id'),
        Index('idx_version_hash', 'version_hash'),
        Index('idx_version_created_at', 'created_at'),
    )


class DataProvenance(Base):
    """Data lineage and provenance tracking"""
    __tablename__ = "data_provenance"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    data_product_id = Column(GUID(), ForeignKey('data_products.id'), nullable=False)
    
    # Source information
    source_systems = Column(JSON, default=[])
    transformations = Column(JSON, default=[])
    lineage_graph = Column(JSON, default={})
    
    # Provenance metadata
    creation_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_modified = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    modification_history = Column(JSON, default=[])
    
    # Verification
    provenance_hash = Column(String(64), nullable=False)
    is_verified = Column(Boolean, default=False)
    
    # Relationships
    data_product = relationship("DataProduct", back_populates="provenance")
    
    __table_args__ = (
        Index('idx_provenance_data_product_id', 'data_product_id'),
        Index('idx_provenance_hash', 'provenance_hash'),
    )


class QualityMetrics(Base):
    """Data quality assessment metrics"""
    __tablename__ = "quality_metrics"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    data_product_id = Column(GUID(), ForeignKey('data_products.id'), nullable=False)
    
    # Quality dimensions
    completeness_score = Column(Float, default=0.0)
    accuracy_score = Column(Float, default=0.0)
    consistency_score = Column(Float, default=0.0)
    timeliness_score = Column(Float, default=0.0)
    overall_score = Column(Float, default=0.0)
    
    # Quality issues and recommendations
    issues = Column(JSON, default=[])
    recommendations = Column(JSON, default=[])
    
    # Assessment metadata
    assessed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    assessed_by = Column(String(255))
    assessment_method = Column(String(100))
    
    # Relationships
    data_product = relationship("DataProduct", back_populates="quality_metrics")
    
    __table_args__ = (
        Index('idx_quality_data_product_id', 'data_product_id'),
        Index('idx_quality_assessed_at', 'assessed_at'),
        Index('idx_quality_overall_score', 'overall_score'),
    )


class BiasAssessment(Base):
    """Bias and fairness assessment for data products"""
    __tablename__ = "bias_assessments"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    data_product_id = Column(GUID(), ForeignKey('data_products.id'), nullable=False)
    
    # Protected attributes and fairness metrics
    protected_attributes = Column(JSON, default=[])
    statistical_parity = Column(Float, default=0.0)
    equalized_odds = Column(Float, default=0.0)
    demographic_parity = Column(Float, default=0.0)
    individual_fairness = Column(Float, default=0.0)
    
    # Bias issues and mitigation
    bias_issues = Column(JSON, default=[])
    mitigation_strategies = Column(JSON, default=[])
    
    # Assessment metadata
    assessed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    assessed_by = Column(String(255))
    assessment_version = Column(String(50))
    
    # Relationships
    data_product = relationship("DataProduct", back_populates="bias_assessments")
    
    __table_args__ = (
        Index('idx_bias_data_product_id', 'data_product_id'),
        Index('idx_bias_assessed_at', 'assessed_at'),
    )


class DataSchema(Base):
    """Schema definitions and validation rules"""
    __tablename__ = "data_schemas"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    schema_definition = Column(JSON, nullable=False)
    
    # Schema metadata
    description = Column(Text)
    schema_type = Column(String(50))  # json_schema, avro, protobuf, etc.
    validation_rules = Column(JSON, default=[])
    
    # Governance
    owner = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index('idx_schema_name_version', 'name', 'version'),
        Index('idx_schema_owner', 'owner'),
        Index('idx_schema_type', 'schema_type'),
    )


class GovernancePolicy(Base):
    """Governance policies and rules"""
    __tablename__ = "governance_policies"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    policy_type = Column(String(100), nullable=False)  # access, quality, compliance, etc.
    
    # Policy definition
    policy_rules = Column(JSON, nullable=False)
    conditions = Column(JSON, default=[])
    actions = Column(JSON, default=[])
    
    # Policy metadata
    description = Column(Text)
    priority = Column(String(20), default="medium")  # low, medium, high, critical
    is_active = Column(Boolean, default=True)
    
    # Governance
    owner = Column(String(255), nullable=False)
    approved_by = Column(String(255))
    approval_date = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    effective_from = Column(DateTime, default=datetime.utcnow)
    effective_until = Column(DateTime)
    
    __table_args__ = (
        Index('idx_policy_type', 'policy_type'),
        Index('idx_policy_owner', 'owner'),
        Index('idx_policy_active', 'is_active'),
        Index('idx_policy_effective', 'effective_from', 'effective_until'),
    )


class ComplianceTag(Base):
    """Compliance tags for regulatory requirements"""
    __tablename__ = "compliance_tags"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    tag_name = Column(String(100), nullable=False, unique=True, index=True)
    regulation = Column(String(100), nullable=False)  # GDPR, CCPA, HIPAA, etc.
    
    # Tag definition
    description = Column(Text)
    requirements = Column(JSON, default=[])
    validation_rules = Column(JSON, default=[])
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    
    __table_args__ = (
        Index('idx_compliance_regulation', 'regulation'),
        Index('idx_compliance_active', 'is_active'),
    )