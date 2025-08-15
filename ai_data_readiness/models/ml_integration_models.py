"""
Data models for ML platform integration
"""

from sqlalchemy import Column, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, Any, Optional

from .base_models import Base


class MLPlatform(Base):
    """Model for registered ML platforms"""
    __tablename__ = 'ml_platforms'
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    platform_type = Column(String(50), nullable=False)  # mlflow, kubeflow, generic
    endpoint_url = Column(String(500), nullable=False)
    credentials = Column(JSON)  # Encrypted credentials
    metadata = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    model_deployments = relationship("ModelDeployment", back_populates="platform")
    quality_correlations = relationship("DataQualityCorrelation", back_populates="platform")


class ModelDeployment(Base):
    """Model for tracking model deployments"""
    __tablename__ = 'model_deployments'
    
    id = Column(String(50), primary_key=True)
    platform_id = Column(String(50), ForeignKey('ml_platforms.id'), nullable=False)
    model_id = Column(String(100), nullable=False)  # ID in the ML platform
    model_name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False)
    deployment_status = Column(String(50), nullable=False)  # deployed, failed, stopped
    endpoint_url = Column(String(500))
    performance_metrics = Column(JSON)
    deployment_config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    platform = relationship("MLPlatform", back_populates="model_deployments")
    quality_correlations = relationship("DataQualityCorrelation", back_populates="model_deployment")


class DataQualityCorrelation(Base):
    """Model for storing data quality and model performance correlations"""
    __tablename__ = 'data_quality_correlations'
    
    id = Column(String(50), primary_key=True)
    dataset_id = Column(String(50), nullable=False)
    platform_id = Column(String(50), ForeignKey('ml_platforms.id'), nullable=False)
    model_deployment_id = Column(String(50), ForeignKey('model_deployments.id'), nullable=False)
    quality_score = Column(Float, nullable=False)
    performance_score = Column(Float, nullable=False)
    correlation_coefficient = Column(Float, nullable=False)
    quality_dimensions = Column(JSON)  # Detailed quality metrics
    performance_metrics = Column(JSON)  # Detailed performance metrics
    analysis_metadata = Column(JSON)  # Additional analysis information
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    platform = relationship("MLPlatform", back_populates="quality_correlations")
    model_deployment = relationship("ModelDeployment", back_populates="quality_correlations")


class MLPlatformSync(Base):
    """Model for tracking synchronization status with ML platforms"""
    __tablename__ = 'ml_platform_syncs'
    
    id = Column(String(50), primary_key=True)
    platform_id = Column(String(50), ForeignKey('ml_platforms.id'), nullable=False)
    sync_type = Column(String(50), nullable=False)  # models, deployments, metrics
    last_sync_at = Column(DateTime)
    sync_status = Column(String(50), nullable=False)  # success, failed, in_progress
    sync_details = Column(JSON)  # Details about what was synced
    error_message = Column(Text)
    records_processed = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelPerformanceHistory(Base):
    """Model for tracking model performance over time"""
    __tablename__ = 'model_performance_history'
    
    id = Column(String(50), primary_key=True)
    model_deployment_id = Column(String(50), ForeignKey('model_deployments.id'), nullable=False)
    dataset_id = Column(String(50))  # Associated dataset if available
    performance_metrics = Column(JSON, nullable=False)
    data_quality_score = Column(Float)
    correlation_score = Column(Float)
    measurement_timestamp = Column(DateTime, nullable=False)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class IntegrationEvent(Base):
    """Model for tracking integration events and activities"""
    __tablename__ = 'integration_events'
    
    id = Column(String(50), primary_key=True)
    platform_id = Column(String(50), ForeignKey('ml_platforms.id'), nullable=False)
    event_type = Column(String(50), nullable=False)  # deployment, sync, error, etc.
    event_data = Column(JSON)
    status = Column(String(50), nullable=False)  # success, failed, pending
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)