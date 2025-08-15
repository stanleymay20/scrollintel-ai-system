"""
Schema catalog models for the AI Data Readiness Platform.

This module provides models for schema versioning, cataloging, and metadata management.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

from .database import Base


class SchemaChangeType(Enum):
    """Types of schema changes."""
    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    COLUMN_TYPE_CHANGED = "column_type_changed"
    COLUMN_RENAMED = "column_renamed"
    CONSTRAINT_ADDED = "constraint_added"
    CONSTRAINT_REMOVED = "constraint_removed"
    INDEX_ADDED = "index_added"
    INDEX_REMOVED = "index_removed"


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""
    name: str
    data_type: str
    nullable: bool = True
    unique: bool = False
    primary_key: bool = False
    foreign_key: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    description: Optional[str] = None
    default_value: Optional[Any] = None
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None


@dataclass
class Schema:
    """Complete schema definition for a dataset."""
    dataset_id: str
    columns: List[ColumnSchema]
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: Dict[str, str] = field(default_factory=dict)
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class SchemaChange:
    """Represents a change between schema versions."""
    change_type: SchemaChangeType
    column_name: Optional[str] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    description: str = ""


class SchemaCatalogModel(Base):
    """SQLAlchemy model for schema catalog entries."""
    __tablename__ = "schema_catalog"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    version_id = Column(String(255), nullable=False, unique=True)
    version_number = Column(Integer, nullable=False)
    schema_definition = Column(JSON, nullable=False)
    schema_hash = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(255), nullable=False)
    change_summary = Column(Text)
    parent_version_id = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    tags = Column(JSON)
    
    # Relationships
    metadata_profiles = relationship("DatasetProfileModel", back_populates="schema_version")


class DatasetProfileModel(Base):
    """SQLAlchemy model for dataset profiles."""
    __tablename__ = "dataset_profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    schema_version_id = Column(UUID(as_uuid=True), ForeignKey('schema_catalog.id'), nullable=True)
    profile_level = Column(String(50), nullable=False)
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    memory_usage = Column(Integer, nullable=False)
    missing_values_total = Column(Integer, nullable=False)
    missing_values_percentage = Column(Float, nullable=False)
    duplicate_rows = Column(Integer, nullable=False)
    duplicate_rows_percentage = Column(Float, nullable=False)
    data_types_distribution = Column(JSON, nullable=False)
    column_profiles = Column(JSON, nullable=False)
    correlations = Column(JSON)
    statistical_summary = Column(JSON)
    data_quality_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    schema_version = relationship("SchemaCatalogModel", back_populates="metadata_profiles")


class SchemaChangeLogModel(Base):
    """SQLAlchemy model for tracking schema changes."""
    __tablename__ = "schema_change_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    from_version_id = Column(String(255), nullable=True)
    to_version_id = Column(String(255), nullable=False)
    change_type = Column(String(50), nullable=False)
    column_name = Column(String(255))
    old_value = Column(JSON)
    new_value = Column(JSON)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(255), nullable=False)


class DatasetCatalogModel(Base):
    """SQLAlchemy model for dataset catalog metadata."""
    __tablename__ = "dataset_catalog"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    source = Column(String(255))
    format = Column(String(50))
    size_bytes = Column(Integer, default=0)
    owner = Column(String(255))
    tags = Column(JSON)
    business_glossary = Column(JSON)  # Business terms and definitions
    data_classification = Column(String(50))  # public, internal, confidential, restricted
    retention_policy = Column(JSON)
    access_permissions = Column(JSON)
    usage_statistics = Column(JSON)
    lineage_upstream = Column(JSON)  # Source datasets
    lineage_downstream = Column(JSON)  # Derived datasets
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed_at = Column(DateTime)
    access_count = Column(Integer, default=0)
    
    def increment_access(self):
        """Increment access count and update last accessed time."""
        self.access_count += 1
        self.last_accessed_at = datetime.utcnow()


class DatasetUsageModel(Base):
    """SQLAlchemy model for tracking dataset usage."""
    __tablename__ = "dataset_usage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(String(255), nullable=False)
    operation = Column(String(100), nullable=False)  # read, write, transform, analyze
    timestamp = Column(DateTime, default=datetime.utcnow)
    duration_seconds = Column(Float)
    rows_processed = Column(Integer)
    bytes_processed = Column(Integer)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    metadata = Column(JSON)  # Additional operation-specific metadata


@dataclass
class CatalogEntry:
    """Catalog entry for a dataset."""
    dataset_id: str
    name: str
    description: str
    current_schema_version: str
    schema_versions: List[str]
    profile_summary: Dict[str, Any]
    usage_statistics: Dict[str, Any]
    lineage: Dict[str, Any]
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0


@dataclass
class SchemaEvolution:
    """Schema evolution tracking."""
    dataset_id: str
    versions: List[Dict[str, Any]]
    changes: List[SchemaChange]
    evolution_timeline: List[Dict[str, Any]]
    compatibility_matrix: Dict[str, Dict[str, bool]]  # version compatibility