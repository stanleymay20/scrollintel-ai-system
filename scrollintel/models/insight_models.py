"""
Insight and Pattern data models for the AI Insight Generation Engine.
"""
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum
import uuid

from .database import Base


class InsightType(Enum):
    TREND = "trend"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"
    OPPORTUNITY = "opportunity"
    RISK = "risk"
    RECOMMENDATION = "recommendation"


class PatternType(Enum):
    INCREASING_TREND = "increasing_trend"
    DECREASING_TREND = "decreasing_trend"
    SEASONAL_PATTERN = "seasonal_pattern"
    CYCLICAL_PATTERN = "cyclical_pattern"
    OUTLIER = "outlier"
    CORRELATION = "correlation"
    THRESHOLD_BREACH = "threshold_breach"


class SignificanceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Pattern(Base):
    __tablename__ = "patterns"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    type = Column(String(50), nullable=False)  # PatternType enum
    metric_name = Column(String(255), nullable=False)
    metric_category = Column(String(100))
    description = Column(Text)
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    significance = Column(String(20), nullable=False)  # SignificanceLevel enum
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    data_points = Column(JSON)  # Raw data points that form the pattern
    statistical_measures = Column(JSON)  # Statistical analysis results
    context = Column(JSON)  # Business context and metadata
    detected_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    insights = relationship("Insight", back_populates="pattern")


class Insight(Base):
    __tablename__ = "insights"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    pattern_id = Column(String, ForeignKey("patterns.id"), nullable=False)
    type = Column(String(50), nullable=False)  # InsightType enum
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    explanation = Column(Text)  # Natural language explanation
    business_impact = Column(Text)  # Business significance explanation
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    significance = Column(Float, nullable=False)  # Business impact score
    priority = Column(String(20), default="medium")  # ActionPriority enum
    tags = Column(JSON)  # Categorization tags
    affected_metrics = Column(JSON)  # List of affected business metrics
    related_insights = Column(JSON)  # IDs of related insights
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    pattern = relationship("Pattern", back_populates="insights")
    recommendations = relationship("ActionRecommendation", back_populates="insight", cascade="all, delete-orphan")


class ActionRecommendation(Base):
    __tablename__ = "action_recommendations"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    insight_id = Column(String, ForeignKey("insights.id"), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    action_type = Column(String(100))  # Type of recommended action
    priority = Column(String(20), nullable=False)  # ActionPriority enum
    estimated_impact = Column(Float)  # Expected business impact
    effort_required = Column(String(20))  # low, medium, high
    timeline = Column(String(100))  # Recommended timeline
    responsible_role = Column(String(100))  # Who should take action
    success_metrics = Column(JSON)  # How to measure success
    implementation_steps = Column(JSON)  # Step-by-step implementation
    is_implemented = Column(Boolean, default=False)
    implemented_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    insight = relationship("Insight", back_populates="recommendations")


class BusinessContext(Base):
    __tablename__ = "business_contexts"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    context_type = Column(String(100), nullable=False)  # industry, department, project, etc.
    name = Column(String(255), nullable=False)
    description = Column(Text)
    context_metadata = Column(JSON)  # Context-specific metadata (renamed from metadata)
    thresholds = Column(JSON)  # Business thresholds and targets
    kpis = Column(JSON)  # Key performance indicators
    benchmarks = Column(JSON)  # Industry or internal benchmarks
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Anomaly(Base):
    __tablename__ = "anomalies"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Float, nullable=False)
    expected_value = Column(Float)
    deviation_score = Column(Float, nullable=False)  # How far from normal
    anomaly_type = Column(String(50))  # spike, drop, drift, etc.
    severity = Column(String(20), nullable=False)  # SignificanceLevel enum
    detected_at = Column(DateTime, default=datetime.utcnow)
    context = Column(JSON)  # Context when anomaly was detected
    root_causes = Column(JSON)  # Potential root causes
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)