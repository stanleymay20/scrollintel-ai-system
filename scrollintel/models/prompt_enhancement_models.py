"""
Database models for prompt enhancement system.
Stores successful prompt patterns, templates, variations, and A/B testing data.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

from .database import Base

class VisualPromptCategory(Base):
    """Categories for organizing prompt templates."""
    __tablename__ = "visual_prompt_categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    templates = relationship("VisualPromptTemplate", back_populates="category")

class VisualPromptTemplate(Base):
    """Successful prompt templates with parameters."""
    __tablename__ = "visual_prompt_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    template = Column(Text, nullable=False)
    description = Column(Text)
    category_id = Column(Integer, ForeignKey("visual_prompt_categories.id"))
    parameters = Column(JSON)  # List of parameter names
    success_rate = Column(Float, default=0.0)
    usage_count = Column(Integer, default=0)
    average_quality_score = Column(Float, default=0.0)
    tags = Column(JSON)  # List of tags
    is_active = Column(Boolean, default=True)
    created_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    category = relationship("VisualPromptCategory", back_populates="templates")
    variations = relationship("VisualPromptVariation", back_populates="template")
    ab_experiments = relationship("VisualABTestExperiment", back_populates="template")

class VisualPromptPattern(Base):
    """Successful prompt patterns and phrases."""
    __tablename__ = "visual_prompt_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    pattern_text = Column(Text, nullable=False, index=True)
    pattern_type = Column(String(100))  # e.g., "quality_enhancer", "style_modifier", "composition_enhancer"
    success_rate = Column(Float, default=0.0)
    usage_count = Column(Integer, default=0)
    context = Column(String(200))  # Context where this pattern works well
    effectiveness_score = Column(Float, default=0.0)
    tags = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class VisualPromptVariation(Base):
    """Variations of prompt templates for A/B testing."""
    __tablename__ = "visual_prompt_variations"
    
    id = Column(Integer, primary_key=True, index=True)
    template_id = Column(Integer, ForeignKey("visual_prompt_templates.id"))
    variation_name = Column(String(200), nullable=False)
    variation_text = Column(Text, nullable=False)
    description = Column(Text)
    success_rate = Column(Float, default=0.0)
    usage_count = Column(Integer, default=0)
    average_quality_score = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    template = relationship("VisualPromptTemplate", back_populates="variations")
    ab_results = relationship("VisualABTestResult", back_populates="variation")

class VisualABTestExperiment(Base):
    """A/B testing experiments for prompt optimization."""
    __tablename__ = "visual_ab_test_experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    template_id = Column(Integer, ForeignKey("visual_prompt_templates.id"))
    status = Column(String(50), default="active")  # active, completed, paused
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime)
    target_sample_size = Column(Integer, default=100)
    current_sample_size = Column(Integer, default=0)
    confidence_level = Column(Float, default=0.95)
    statistical_significance = Column(Float, default=0.0)
    winning_variation_id = Column(Integer)
    experiment_config = Column(JSON)  # Configuration parameters
    created_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    template = relationship("VisualPromptTemplate", back_populates="ab_experiments")
    results = relationship("VisualABTestResult", back_populates="experiment")

class VisualABTestResult(Base):
    """Individual results from A/B testing experiments."""
    __tablename__ = "visual_ab_test_results"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("visual_ab_test_experiments.id"))
    variation_id = Column(Integer, ForeignKey("visual_prompt_variations.id"))
    user_id = Column(String(100))
    quality_score = Column(Float)
    user_rating = Column(Integer)  # 1-5 rating
    generation_time = Column(Float)
    success = Column(Boolean)
    feedback = Column(Text)
    result_metadata = Column(JSON)  # Additional metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment = relationship("VisualABTestExperiment", back_populates="results")
    variation = relationship("VisualPromptVariation", back_populates="ab_results")

class VisualPromptUsageLog(Base):
    """Log of prompt usage for analytics."""
    __tablename__ = "visual_prompt_usage_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    template_id = Column(Integer, ForeignKey("visual_prompt_templates.id"))
    user_id = Column(String(100))
    prompt_text = Column(Text)
    parameters_used = Column(JSON)
    quality_score = Column(Float)
    user_rating = Column(Integer)
    generation_time = Column(Float)
    success = Column(Boolean)
    model_used = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class VisualPromptOptimizationSuggestion(Base):
    """AI-generated suggestions for prompt optimization."""
    __tablename__ = "visual_prompt_optimization_suggestions"
    
    id = Column(Integer, primary_key=True, index=True)
    original_prompt = Column(Text, nullable=False)
    suggested_prompt = Column(Text, nullable=False)
    suggestion_type = Column(String(100))  # e.g., "quality_improvement", "style_enhancement"
    confidence_score = Column(Float)
    reasoning = Column(Text)
    applied = Column(Boolean, default=False)
    user_feedback = Column(String(50))  # "accepted", "rejected", "modified"
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)