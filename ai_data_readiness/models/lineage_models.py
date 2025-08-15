"""
Data models for lineage tracking in AI Data Readiness Platform
"""

from sqlalchemy import Column, String, DateTime, Text, JSON, ForeignKey, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class LineageNodeModel(Base):
    """Database model for lineage nodes"""
    __tablename__ = 'lineage_nodes'
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    node_type = Column(String(50), nullable=False)  # dataset, transformation, model, feature, source
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(String(50), nullable=False, default="1.0.0")
    checksum = Column(String(255), nullable=True)
    
    # Relationships
    source_edges = relationship("LineageEdgeModel", foreign_keys="LineageEdgeModel.source_node_id", back_populates="source_node")
    target_edges = relationship("LineageEdgeModel", foreign_keys="LineageEdgeModel.target_node_id", back_populates="target_node")
    dataset_versions = relationship("DatasetVersionModel", back_populates="dataset_node")


class LineageEdgeModel(Base):
    """Database model for lineage edges (transformations)"""
    __tablename__ = 'lineage_edges'
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_node_id = Column(String(255), ForeignKey('lineage_nodes.id'), nullable=False)
    target_node_id = Column(String(255), ForeignKey('lineage_nodes.id'), nullable=False)
    transformation_type = Column(String(50), nullable=False)
    transformation_details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(255), nullable=False)
    
    # Relationships
    source_node = relationship("LineageNodeModel", foreign_keys=[source_node_id], back_populates="source_edges")
    target_node = relationship("LineageNodeModel", foreign_keys=[target_node_id], back_populates="target_edges")


class DatasetVersionModel(Base):
    """Database model for dataset versions"""
    __tablename__ = 'dataset_versions'
    
    version_id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String(255), ForeignKey('lineage_nodes.id'), nullable=False)
    version_number = Column(String(50), nullable=False)
    parent_version_id = Column(String(255), ForeignKey('dataset_versions.version_id'), nullable=True)
    changes = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(255), nullable=False)
    commit_message = Column(Text, nullable=False)
    schema_hash = Column(String(255), nullable=False)
    data_hash = Column(String(255), nullable=False)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    dataset_node = relationship("LineageNodeModel", back_populates="dataset_versions")
    parent_version = relationship("DatasetVersionModel", remote_side=[version_id])
    child_versions = relationship("DatasetVersionModel", remote_side=[parent_version_id])


class ModelDatasetLinkModel(Base):
    """Database model for model-dataset links"""
    __tablename__ = 'model_dataset_links'
    
    link_id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String(255), nullable=False)
    model_version = Column(String(50), nullable=False)
    dataset_id = Column(String(255), ForeignKey('lineage_nodes.id'), nullable=False)
    dataset_version = Column(String(50), nullable=False)
    usage_type = Column(String(50), nullable=False)  # training, validation, test, inference
    created_at = Column(DateTime, default=datetime.utcnow)
    performance_metrics = Column(JSON, nullable=True)
    
    # Relationships
    dataset_node = relationship("LineageNodeModel")


class LineageQueryModel(Base):
    """Database model for storing lineage queries and their results"""
    __tablename__ = 'lineage_queries'
    
    query_id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    query_type = Column(String(50), nullable=False)  # upstream, downstream, full_lineage, impact_analysis
    target_node_id = Column(String(255), nullable=False)
    query_parameters = Column(JSON, nullable=True)
    result_cache = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    cache_expires_at = Column(DateTime, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)


class LineageMetricsModel(Base):
    """Database model for lineage metrics and statistics"""
    __tablename__ = 'lineage_metrics'
    
    metric_id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    metric_type = Column(String(50), nullable=False)  # complexity, depth, breadth, etc.
    node_id = Column(String(255), ForeignKey('lineage_nodes.id'), nullable=True)
    metric_value = Column(Float, nullable=False)
    metric_metadata = Column(JSON, nullable=True)
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    node = relationship("LineageNodeModel")