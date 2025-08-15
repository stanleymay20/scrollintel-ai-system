"""
Data models for Quality Control Automation system
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class QualityStandardType(Enum):
    """Types of quality standards"""
    RESEARCH_RIGOR = "research_rigor"
    EXPERIMENTAL_VALIDITY = "experimental_validity"
    PROTOTYPE_FUNCTIONALITY = "prototype_functionality"
    VALIDATION_COMPLETENESS = "validation_completeness"
    KNOWLEDGE_ACCURACY = "knowledge_accuracy"

class QualityLevelType(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"

class ProcessStatus(Enum):
    """Process status based on quality assessment"""
    APPROVED = "approved"
    CONDITIONAL = "conditional"
    BLOCKED = "blocked"
    UNDER_REVIEW = "under_review"

@dataclass
class QualityMetricData:
    """Quality metric data structure"""
    name: str
    value: float
    threshold: float
    weight: float
    description: str
    measurement_time: datetime = field(default_factory=datetime.now)
    
    def is_passing(self) -> bool:
        """Check if metric passes threshold"""
        return self.value >= self.threshold
    
    def get_score_percentage(self) -> float:
        """Get score as percentage"""
        return min(100.0, self.value * 100)

@dataclass
class QualityAssessmentData:
    """Quality assessment data structure"""
    process_id: str
    process_type: str
    overall_score: float
    quality_level: QualityLevelType
    metrics: List[QualityMetricData]
    issues: List[str]
    recommendations: List[str]
    assessment_time: datetime = field(default_factory=datetime.now)
    
    def get_failing_metrics(self) -> List[QualityMetricData]:
        """Get metrics that are failing thresholds"""
        return [metric for metric in self.metrics if not metric.is_passing()]
    
    def get_critical_issues(self) -> List[str]:
        """Get critical quality issues"""
        return [issue for issue in self.issues if "critical" in issue.lower()]

@dataclass
class QualityStandardConfig:
    """Quality standard configuration"""
    standard_type: QualityStandardType
    metrics: List[str]
    thresholds: Dict[str, float]
    weights: Dict[str, float]
    validation_rules: List[str]
    description: str
    
    def get_weighted_score(self, metric_values: Dict[str, float]) -> float:
        """Calculate weighted score from metric values"""
        total_score = 0.0
        total_weight = 0.0
        
        for metric, value in metric_values.items():
            if metric in self.weights:
                weight = self.weights[metric]
                total_score += value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

@dataclass
class QualityTrend:
    """Quality trend analysis data"""
    process_type: str
    metric_name: str
    trend_direction: str  # "improving", "declining", "stable"
    trend_strength: float  # 0.0 to 1.0
    data_points: List[float]
    time_period: str
    analysis_time: datetime = field(default_factory=datetime.now)

@dataclass
class QualityOptimization:
    """Quality optimization recommendation"""
    optimization_type: str
    target_metric: str
    current_value: float
    recommended_value: float
    expected_improvement: float
    implementation_effort: str  # "low", "medium", "high"
    priority: str  # "low", "medium", "high", "critical"

# SQLAlchemy Models

class QualityAssessment(Base):
    """Quality assessment database model"""
    __tablename__ = "quality_assessments"
    
    id = Column(Integer, primary_key=True)
    process_id = Column(String(255), nullable=False, index=True)
    process_type = Column(String(100), nullable=False)
    overall_score = Column(Float, nullable=False)
    quality_level = Column(String(50), nullable=False)
    metrics_data = Column(JSON, nullable=False)
    issues = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    assessment_time = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "process_id": self.process_id,
            "process_type": self.process_type,
            "overall_score": self.overall_score,
            "quality_level": self.quality_level,
            "metrics_data": self.metrics_data,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "assessment_time": self.assessment_time.isoformat() if self.assessment_time else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class QualityMetric(Base):
    """Quality metric database model"""
    __tablename__ = "quality_metrics"
    
    id = Column(Integer, primary_key=True)
    assessment_id = Column(Integer, nullable=False, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    weight = Column(Float, nullable=False)
    description = Column(Text, nullable=True)
    is_passing = Column(Boolean, nullable=False)
    measurement_time = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "assessment_id": self.assessment_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "weight": self.weight,
            "description": self.description,
            "is_passing": self.is_passing,
            "measurement_time": self.measurement_time.isoformat() if self.measurement_time else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class QualityStandard(Base):
    """Quality standard database model"""
    __tablename__ = "quality_standards"
    
    id = Column(Integer, primary_key=True)
    standard_type = Column(String(100), nullable=False, unique=True)
    metrics = Column(JSON, nullable=False)
    thresholds = Column(JSON, nullable=False)
    weights = Column(JSON, nullable=False)
    validation_rules = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    version = Column(String(20), default="1.0")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "standard_type": self.standard_type,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "weights": self.weights,
            "validation_rules": self.validation_rules,
            "description": self.description,
            "is_active": self.is_active,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class QualityTrendAnalysis(Base):
    """Quality trend analysis database model"""
    __tablename__ = "quality_trends"
    
    id = Column(Integer, primary_key=True)
    process_type = Column(String(100), nullable=False)
    metric_name = Column(String(100), nullable=False)
    trend_direction = Column(String(50), nullable=False)
    trend_strength = Column(Float, nullable=False)
    data_points = Column(JSON, nullable=False)
    time_period = Column(String(50), nullable=False)
    analysis_time = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "process_type": self.process_type,
            "metric_name": self.metric_name,
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "data_points": self.data_points,
            "time_period": self.time_period,
            "analysis_time": self.analysis_time.isoformat() if self.analysis_time else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class QualityOptimizationRecommendation(Base):
    """Quality optimization recommendation database model"""
    __tablename__ = "quality_optimizations"
    
    id = Column(Integer, primary_key=True)
    optimization_type = Column(String(100), nullable=False)
    target_metric = Column(String(100), nullable=False)
    current_value = Column(Float, nullable=False)
    recommended_value = Column(Float, nullable=False)
    expected_improvement = Column(Float, nullable=False)
    implementation_effort = Column(String(50), nullable=False)
    priority = Column(String(50), nullable=False)
    status = Column(String(50), default="pending")
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "optimization_type": self.optimization_type,
            "target_metric": self.target_metric,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "expected_improvement": self.expected_improvement,
            "implementation_effort": self.implementation_effort,
            "priority": self.priority,
            "status": self.status,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class QualityAlert(Base):
    """Quality alert database model"""
    __tablename__ = "quality_alerts"
    
    id = Column(Integer, primary_key=True)
    process_id = Column(String(255), nullable=False, index=True)
    alert_type = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)  # "low", "medium", "high", "critical"
    message = Column(Text, nullable=False)
    metric_name = Column(String(100), nullable=True)
    metric_value = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "process_id": self.process_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "is_resolved": self.is_resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }