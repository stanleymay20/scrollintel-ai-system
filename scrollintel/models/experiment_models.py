"""
Data models for A/B Testing Engine in the Advanced Prompt Management System.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey, Boolean, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from enum import Enum

from .database import Base


class ExperimentStatus(Enum):
    """Status of an A/B test experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantType(Enum):
    """Type of experiment variant."""
    CONTROL = "control"
    TREATMENT = "treatment"


class StatisticalSignificance(Enum):
    """Statistical significance levels."""
    NOT_SIGNIFICANT = "not_significant"
    SIGNIFICANT = "significant"
    HIGHLY_SIGNIFICANT = "highly_significant"


class Experiment(Base):
    """A/B test experiment model."""
    __tablename__ = "experiments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    prompt_id = Column(String, ForeignKey("advanced_prompt_templates.id"), nullable=False)
    hypothesis = Column(Text, nullable=False)
    success_metrics = Column(JSON, default=list)  # List of metric names to track
    target_sample_size = Column(Integer, nullable=False, default=1000)
    confidence_level = Column(Float, nullable=False, default=0.95)
    minimum_effect_size = Column(Float, nullable=False, default=0.05)
    status = Column(String(20), nullable=False, default=ExperimentStatus.DRAFT.value)
    traffic_allocation = Column(Float, nullable=False, default=1.0)  # Percentage of traffic to include
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    variants = relationship("ExperimentVariant", back_populates="experiment", cascade="all, delete-orphan")
    results = relationship("ExperimentResult", back_populates="experiment", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "prompt_id": self.prompt_id,
            "hypothesis": self.hypothesis,
            "success_metrics": self.success_metrics or [],
            "target_sample_size": self.target_sample_size,
            "confidence_level": self.confidence_level,
            "minimum_effect_size": self.minimum_effect_size,
            "status": self.status,
            "traffic_allocation": self.traffic_allocation,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class ExperimentVariant(Base):
    """Variant in an A/B test experiment."""
    __tablename__ = "experiment_variants"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    prompt_content = Column(Text, nullable=False)
    prompt_variables = Column(JSON, default=dict)  # Variable values for this variant
    variant_type = Column(String(20), nullable=False, default=VariantType.TREATMENT.value)
    traffic_weight = Column(Float, nullable=False, default=0.5)  # Percentage of experiment traffic
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="variants")
    metrics = relationship("VariantMetric", back_populates="variant", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "prompt_content": self.prompt_content,
            "prompt_variables": self.prompt_variables or {},
            "variant_type": self.variant_type,
            "traffic_weight": self.traffic_weight,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class VariantMetric(Base):
    """Metrics collected for experiment variants."""
    __tablename__ = "variant_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    variant_id = Column(String, ForeignKey("experiment_variants.id"), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    sample_size = Column(Integer, nullable=False, default=1)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String(255))  # For tracking individual sessions
    user_feedback = Column(JSON)  # Optional user feedback data
    
    # Relationships
    variant = relationship("ExperimentVariant", back_populates="metrics")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "variant_id": self.variant_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "sample_size": self.sample_size,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "session_id": self.session_id,
            "user_feedback": self.user_feedback
        }


class ExperimentResult(Base):
    """Statistical results of A/B test experiments."""
    __tablename__ = "experiment_results"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    metric_name = Column(String(100), nullable=False)
    control_variant_id = Column(String, ForeignKey("experiment_variants.id"), nullable=False)
    treatment_variant_id = Column(String, ForeignKey("experiment_variants.id"), nullable=False)
    control_mean = Column(Float, nullable=False)
    treatment_mean = Column(Float, nullable=False)
    control_std = Column(Float, nullable=False)
    treatment_std = Column(Float, nullable=False)
    control_sample_size = Column(Integer, nullable=False)
    treatment_sample_size = Column(Integer, nullable=False)
    effect_size = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    confidence_interval_lower = Column(Float, nullable=False)
    confidence_interval_upper = Column(Float, nullable=False)
    statistical_significance = Column(String(20), nullable=False)
    statistical_power = Column(Float)
    winner_variant_id = Column(String, ForeignKey("experiment_variants.id"))
    confidence_level = Column(Float, nullable=False, default=0.95)
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="results")
    control_variant = relationship("ExperimentVariant", foreign_keys=[control_variant_id])
    treatment_variant = relationship("ExperimentVariant", foreign_keys=[treatment_variant_id])
    winner_variant = relationship("ExperimentVariant", foreign_keys=[winner_variant_id])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "metric_name": self.metric_name,
            "control_variant_id": self.control_variant_id,
            "treatment_variant_id": self.treatment_variant_id,
            "control_mean": self.control_mean,
            "treatment_mean": self.treatment_mean,
            "control_std": self.control_std,
            "treatment_std": self.treatment_std,
            "control_sample_size": self.control_sample_size,
            "treatment_sample_size": self.treatment_sample_size,
            "effect_size": self.effect_size,
            "p_value": self.p_value,
            "confidence_interval_lower": self.confidence_interval_lower,
            "confidence_interval_upper": self.confidence_interval_upper,
            "statistical_significance": self.statistical_significance,
            "statistical_power": self.statistical_power,
            "winner_variant_id": self.winner_variant_id,
            "confidence_level": self.confidence_level,
            "calculated_at": self.calculated_at.isoformat() if self.calculated_at else None
        }


class ExperimentSchedule(Base):
    """Scheduled experiment automation."""
    __tablename__ = "experiment_schedules"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    schedule_type = Column(String(50), nullable=False)  # daily, weekly, monthly, custom
    cron_expression = Column(String(100))  # For custom schedules
    auto_start = Column(Boolean, default=False)
    auto_stop = Column(Boolean, default=False)
    auto_promote_winner = Column(Boolean, default=False)
    promotion_threshold = Column(Float, default=0.05)  # Minimum effect size for auto-promotion
    max_duration_hours = Column(Integer)  # Maximum experiment duration
    is_active = Column(Boolean, default=True)
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_run = Column(DateTime)
    next_run = Column(DateTime)
    
    # Relationships
    experiment = relationship("Experiment")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "schedule_type": self.schedule_type,
            "cron_expression": self.cron_expression,
            "auto_start": self.auto_start,
            "auto_stop": self.auto_stop,
            "auto_promote_winner": self.auto_promote_winner,
            "promotion_threshold": self.promotion_threshold,
            "max_duration_hours": self.max_duration_hours,
            "is_active": self.is_active,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None
        }