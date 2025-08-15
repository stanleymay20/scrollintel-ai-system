"""
Data models for Strategy Optimization Engine
"""

from sqlalchemy import Column, String, Float, DateTime, Text, JSON, Boolean, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

Base = declarative_base()


class OptimizationType(str, Enum):
    """Types of optimization strategies"""
    PERFORMANCE_BASED = "performance_based"
    TIMELINE_BASED = "timeline_based"
    RESOURCE_BASED = "resource_based"
    RESISTANCE_BASED = "resistance_based"
    ENGAGEMENT_BASED = "engagement_based"


class OptimizationPriority(str, Enum):
    """Priority levels for optimization recommendations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AdjustmentStatus(str, Enum):
    """Status of strategy adjustments"""
    PENDING = "pending"
    APPROVED = "approved"
    IMPLEMENTING = "implementing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationContext(Base):
    """Context information for strategy optimization"""
    __tablename__ = "optimization_contexts"
    
    id = Column(String, primary_key=True)
    transformation_id = Column(String, nullable=False, index=True)
    current_progress = Column(Float, default=0.0)
    timeline_status = Column(String, default="unknown")
    budget_utilization = Column(Float, default=0.0)
    resistance_level = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.0)
    performance_metrics = Column(JSON, default=dict)
    external_factors = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    metrics = relationship("OptimizationMetric", back_populates="context")
    recommendations = relationship("OptimizationRecommendation", back_populates="context")


class OptimizationMetric(Base):
    """Metric used for strategy optimization"""
    __tablename__ = "optimization_metrics"
    
    id = Column(String, primary_key=True)
    context_id = Column(String, ForeignKey("optimization_contexts.id"))
    name = Column(String, nullable=False)
    current_value = Column(Float, nullable=False)
    target_value = Column(Float, nullable=False)
    weight = Column(Float, default=1.0)
    trend = Column(String, default="stable")  # "improving", "declining", "stable"
    measurement_unit = Column(String)
    description = Column(Text)
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    context = relationship("OptimizationContext", back_populates="metrics")


class OptimizationRecommendation(Base):
    """Recommendation for strategy optimization"""
    __tablename__ = "optimization_recommendations"
    
    id = Column(String, primary_key=True)
    context_id = Column(String, ForeignKey("optimization_contexts.id"))
    optimization_type = Column(String, nullable=False)  # OptimizationType enum
    priority = Column(String, nullable=False)  # OptimizationPriority enum
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    expected_impact = Column(Float, nullable=False)
    implementation_effort = Column(String)  # "Low", "Medium", "High"
    timeline = Column(String)
    success_probability = Column(Float, default=0.5)
    dependencies = Column(JSON, default=list)
    risks = Column(JSON, default=list)
    rationale = Column(Text)
    implementation_notes = Column(Text)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    context = relationship("OptimizationContext", back_populates="recommendations")
    adjustments = relationship("StrategyAdjustment", back_populates="recommendation")
    feedback = relationship("RecommendationFeedback", back_populates="recommendation")


class StrategyAdjustment(Base):
    """Adjustment to transformation strategy"""
    __tablename__ = "strategy_adjustments"
    
    id = Column(String, primary_key=True)
    transformation_id = Column(String, nullable=False, index=True)
    recommendation_id = Column(String, ForeignKey("optimization_recommendations.id"))
    adjustment_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    original_strategy = Column(JSON, nullable=False)
    adjusted_strategy = Column(JSON, nullable=False)
    rationale = Column(Text, nullable=False)
    expected_outcomes = Column(JSON, default=list)
    implementation_date = Column(DateTime)
    completion_date = Column(DateTime)
    status = Column(String, default=AdjustmentStatus.PENDING)  # AdjustmentStatus enum
    effectiveness_score = Column(Float)
    implementation_notes = Column(Text)
    rollback_plan = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    recommendation = relationship("OptimizationRecommendation", back_populates="adjustments")
    effectiveness_reports = relationship("AdjustmentEffectiveness", back_populates="adjustment")


class AdjustmentEffectiveness(Base):
    """Effectiveness tracking for strategy adjustments"""
    __tablename__ = "adjustment_effectiveness"
    
    id = Column(String, primary_key=True)
    adjustment_id = Column(String, ForeignKey("strategy_adjustments.id"))
    measurement_date = Column(DateTime, nullable=False)
    effectiveness_score = Column(Float, nullable=False)
    performance_impact = Column(JSON, default=dict)
    behavioral_impact = Column(JSON, default=dict)
    engagement_impact = Column(Float)
    resistance_impact = Column(Float)
    timeline_impact = Column(Float)
    cost_impact = Column(Float)
    positive_outcomes = Column(JSON, default=list)
    negative_outcomes = Column(JSON, default=list)
    unexpected_outcomes = Column(JSON, default=list)
    lessons_learned = Column(Text)
    recommendations_for_future = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    adjustment = relationship("StrategyAdjustment", back_populates="effectiveness_reports")


class OptimizationHistory(Base):
    """Historical record of optimization activities"""
    __tablename__ = "optimization_history"
    
    id = Column(String, primary_key=True)
    transformation_id = Column(String, nullable=False, index=True)
    optimization_session_id = Column(String, nullable=False)
    optimization_type = Column(String, nullable=False)
    context_snapshot = Column(JSON, nullable=False)
    metrics_snapshot = Column(JSON, nullable=False)
    recommendations_generated = Column(Integer, default=0)
    recommendations_implemented = Column(Integer, default=0)
    overall_effectiveness = Column(Float)
    session_duration = Column(Float)  # in minutes
    user_satisfaction = Column(Float)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class RecommendationFeedback(Base):
    """User feedback on optimization recommendations"""
    __tablename__ = "recommendation_feedback"
    
    id = Column(String, primary_key=True)
    recommendation_id = Column(String, ForeignKey("optimization_recommendations.id"))
    user_id = Column(String)
    usefulness_score = Column(Float, nullable=False)  # 1-5 scale
    implementation_difficulty = Column(String)  # "easy", "medium", "hard"
    actual_impact = Column(Float)  # Actual impact vs expected
    accuracy_score = Column(Float)  # How accurate was the recommendation
    would_recommend = Column(Boolean, default=False)
    comments = Column(Text)
    improvement_suggestions = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    recommendation = relationship("OptimizationRecommendation", back_populates="feedback")


class ContinuousImprovementPlan(Base):
    """Plans for continuous improvement based on optimization patterns"""
    __tablename__ = "continuous_improvement_plans"
    
    id = Column(String, primary_key=True)
    transformation_id = Column(String, nullable=False, index=True)
    plan_version = Column(String, nullable=False)
    patterns_identified = Column(JSON, nullable=False)
    improvement_recommendations = Column(JSON, nullable=False)
    learning_integration_plan = Column(JSON, nullable=False)
    success_metrics = Column(JSON, default=list)
    implementation_timeline = Column(JSON, default=dict)
    resource_requirements = Column(JSON, default=dict)
    risk_assessment = Column(JSON, default=dict)
    status = Column(String, default="draft")
    approved_by = Column(String)
    approved_at = Column(DateTime)
    implementation_started_at = Column(DateTime)
    completion_target = Column(DateTime)
    actual_completion = Column(DateTime)
    effectiveness_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class OptimizationAlgorithmConfig(Base):
    """Configuration for optimization algorithms"""
    __tablename__ = "optimization_algorithm_configs"
    
    id = Column(String, primary_key=True)
    algorithm_name = Column(String, nullable=False, unique=True)
    algorithm_type = Column(String, nullable=False)  # OptimizationType
    version = Column(String, nullable=False)
    parameters = Column(JSON, nullable=False)
    weights = Column(JSON, default=dict)
    thresholds = Column(JSON, default=dict)
    enabled = Column(Boolean, default=True)
    performance_metrics = Column(JSON, default=dict)
    last_tuned = Column(DateTime)
    tuning_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class OptimizationBenchmark(Base):
    """Benchmarks for optimization performance"""
    __tablename__ = "optimization_benchmarks"
    
    id = Column(String, primary_key=True)
    benchmark_name = Column(String, nullable=False)
    optimization_type = Column(String, nullable=False)
    baseline_metrics = Column(JSON, nullable=False)
    target_metrics = Column(JSON, nullable=False)
    achieved_metrics = Column(JSON, default=dict)
    benchmark_date = Column(DateTime, nullable=False)
    transformation_context = Column(JSON, default=dict)
    success_criteria = Column(JSON, default=list)
    actual_outcomes = Column(JSON, default=list)
    lessons_learned = Column(Text)
    recommendations = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# Utility functions for model operations
def create_optimization_context(transformation_id: str, context_data: Dict[str, Any]) -> OptimizationContext:
    """Create a new optimization context"""
    return OptimizationContext(
        id=f"ctx_{transformation_id}_{datetime.now().timestamp()}",
        transformation_id=transformation_id,
        current_progress=context_data.get("current_progress", 0.0),
        timeline_status=context_data.get("timeline_status", "unknown"),
        budget_utilization=context_data.get("budget_utilization", 0.0),
        resistance_level=context_data.get("resistance_level", 0.0),
        engagement_score=context_data.get("engagement_score", 0.0),
        performance_metrics=context_data.get("performance_metrics", {}),
        external_factors=context_data.get("external_factors", [])
    )


def create_optimization_metric(context_id: str, metric_data: Dict[str, Any]) -> OptimizationMetric:
    """Create a new optimization metric"""
    return OptimizationMetric(
        id=f"metric_{context_id}_{metric_data['name']}_{datetime.now().timestamp()}",
        context_id=context_id,
        name=metric_data["name"],
        current_value=metric_data["current_value"],
        target_value=metric_data["target_value"],
        weight=metric_data.get("weight", 1.0),
        trend=metric_data.get("trend", "stable"),
        measurement_unit=metric_data.get("measurement_unit"),
        description=metric_data.get("description")
    )


def create_optimization_recommendation(context_id: str, rec_data: Dict[str, Any]) -> OptimizationRecommendation:
    """Create a new optimization recommendation"""
    return OptimizationRecommendation(
        id=f"rec_{context_id}_{datetime.now().timestamp()}",
        context_id=context_id,
        optimization_type=rec_data["optimization_type"],
        priority=rec_data["priority"],
        title=rec_data["title"],
        description=rec_data["description"],
        expected_impact=rec_data["expected_impact"],
        implementation_effort=rec_data.get("implementation_effort", "Medium"),
        timeline=rec_data.get("timeline", "1-3 months"),
        success_probability=rec_data.get("success_probability", 0.5),
        dependencies=rec_data.get("dependencies", []),
        risks=rec_data.get("risks", []),
        rationale=rec_data.get("rationale", ""),
        implementation_notes=rec_data.get("implementation_notes", "")
    )


def create_strategy_adjustment(transformation_id: str, recommendation_id: str, 
                             adjustment_data: Dict[str, Any]) -> StrategyAdjustment:
    """Create a new strategy adjustment"""
    return StrategyAdjustment(
        id=f"adj_{transformation_id}_{datetime.now().timestamp()}",
        transformation_id=transformation_id,
        recommendation_id=recommendation_id,
        adjustment_type=adjustment_data["adjustment_type"],
        title=adjustment_data["title"],
        description=adjustment_data.get("description", ""),
        original_strategy=adjustment_data["original_strategy"],
        adjusted_strategy=adjustment_data["adjusted_strategy"],
        rationale=adjustment_data["rationale"],
        expected_outcomes=adjustment_data.get("expected_outcomes", []),
        implementation_date=adjustment_data.get("implementation_date"),
        rollback_plan=adjustment_data.get("rollback_plan", {})
    )


# Database initialization
def init_optimization_db(engine):
    """Initialize optimization database tables"""
    Base.metadata.create_all(bind=engine)


# Example usage
if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create in-memory database for testing
    engine = create_engine("sqlite:///:memory:")
    init_optimization_db(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create sample data
    context = create_optimization_context("trans_001", {
        "current_progress": 0.6,
        "timeline_status": "on_track",
        "resistance_level": 0.3,
        "engagement_score": 0.7
    })
    
    session.add(context)
    session.commit()
    
    print(f"Created optimization context: {context.id}")
    
    session.close()