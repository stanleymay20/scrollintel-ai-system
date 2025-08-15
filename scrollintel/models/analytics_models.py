"""
Analytics models for prompt management system.
Provides data models for tracking prompt performance, usage analytics, and reporting.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel
import uuid

Base = declarative_base()

class PromptMetrics(Base):
    """Model for tracking individual prompt performance metrics."""
    __tablename__ = "prompt_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt_id = Column(String, ForeignKey("prompt_templates.id"), nullable=False)
    version_id = Column(String, ForeignKey("prompt_versions.id"), nullable=True)
    
    # Performance metrics
    accuracy_score = Column(Float, nullable=True)
    relevance_score = Column(Float, nullable=True)
    efficiency_score = Column(Float, nullable=True)
    user_satisfaction = Column(Float, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    token_usage = Column(Integer, nullable=True)
    cost_per_request = Column(Float, nullable=True)
    
    # Usage context
    use_case = Column(String, nullable=True)
    model_used = Column(String, nullable=True)
    user_id = Column(String, nullable=True)
    team_id = Column(String, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    usage_analytics = relationship("UsageAnalytics", back_populates="prompt_metrics")

class UsageAnalytics(Base):
    """Model for tracking prompt usage patterns and analytics."""
    __tablename__ = "usage_analytics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt_id = Column(String, ForeignKey("prompt_templates.id"), nullable=False)
    metrics_id = Column(String, ForeignKey("prompt_metrics.id"), nullable=True)
    
    # Usage statistics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    avg_response_time = Column(Float, nullable=True)
    peak_usage_hour = Column(Integer, nullable=True)
    
    # Performance trends
    performance_trend = Column(String, nullable=True)  # 'improving', 'declining', 'stable'
    usage_trend = Column(String, nullable=True)
    cost_trend = Column(String, nullable=True)
    
    # Time-based analytics
    daily_usage = Column(JSON, nullable=True)  # {"2024-01-01": 150, "2024-01-02": 200}
    hourly_patterns = Column(JSON, nullable=True)  # {"0": 10, "1": 5, ..., "23": 25}
    weekly_patterns = Column(JSON, nullable=True)
    
    # Team analytics
    team_usage = Column(JSON, nullable=True)  # {"team_a": 100, "team_b": 50}
    user_adoption = Column(JSON, nullable=True)
    
    # Metadata
    analysis_period_start = Column(DateTime, nullable=False)
    analysis_period_end = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    prompt_metrics = relationship("PromptMetrics", back_populates="usage_analytics")

class AnalyticsReport(Base):
    """Model for storing generated analytics reports."""
    __tablename__ = "analytics_reports"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Report metadata
    report_type = Column(String, nullable=False)  # 'performance', 'usage', 'team', 'trend'
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # Report scope
    prompt_ids = Column(JSON, nullable=True)  # List of prompt IDs included
    team_ids = Column(JSON, nullable=True)
    date_range_start = Column(DateTime, nullable=False)
    date_range_end = Column(DateTime, nullable=False)
    
    # Report content
    summary = Column(JSON, nullable=False)  # Key insights and metrics
    detailed_data = Column(JSON, nullable=False)  # Full report data
    visualizations = Column(JSON, nullable=True)  # Chart configurations
    recommendations = Column(JSON, nullable=True)  # AI-generated recommendations
    
    # Report status
    status = Column(String, default="generated")  # 'generating', 'generated', 'failed'
    generated_by = Column(String, nullable=True)
    scheduled = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AlertRule(Base):
    """Model for defining analytics-based alert rules."""
    __tablename__ = "alert_rules"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Rule definition
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    rule_type = Column(String, nullable=False)  # 'threshold', 'trend', 'anomaly'
    
    # Rule conditions
    metric_name = Column(String, nullable=False)  # 'accuracy_score', 'usage_count', etc.
    condition = Column(String, nullable=False)  # 'greater_than', 'less_than', 'equals'
    threshold_value = Column(Float, nullable=True)
    trend_direction = Column(String, nullable=True)  # 'increasing', 'decreasing'
    
    # Rule scope
    prompt_ids = Column(JSON, nullable=True)
    team_ids = Column(JSON, nullable=True)
    
    # Alert configuration
    severity = Column(String, default="medium")  # 'low', 'medium', 'high', 'critical'
    notification_channels = Column(JSON, nullable=False)  # ['email', 'slack', 'webhook']
    recipients = Column(JSON, nullable=False)
    
    # Rule status
    active = Column(Boolean, default=True)
    last_triggered = Column(DateTime, nullable=True)
    trigger_count = Column(Integer, default=0)
    
    # Metadata
    created_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Pydantic models for API responses
class PromptMetricsResponse(BaseModel):
    """Response model for prompt metrics."""
    id: str
    prompt_id: str
    version_id: Optional[str]
    accuracy_score: Optional[float]
    relevance_score: Optional[float]
    efficiency_score: Optional[float]
    user_satisfaction: Optional[float]
    response_time_ms: Optional[int]
    token_usage: Optional[int]
    cost_per_request: Optional[float]
    use_case: Optional[str]
    model_used: Optional[str]
    created_at: datetime

class UsageAnalyticsResponse(BaseModel):
    """Response model for usage analytics."""
    id: str
    prompt_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: Optional[float]
    performance_trend: Optional[str]
    usage_trend: Optional[str]
    daily_usage: Optional[Dict[str, int]]
    hourly_patterns: Optional[Dict[str, int]]
    team_usage: Optional[Dict[str, int]]
    analysis_period_start: datetime
    analysis_period_end: datetime

class AnalyticsReportResponse(BaseModel):
    """Response model for analytics reports."""
    id: str
    report_type: str
    title: str
    description: Optional[str]
    summary: Dict[str, Any]
    date_range_start: datetime
    date_range_end: datetime
    status: str
    created_at: datetime

class TrendAnalysis(BaseModel):
    """Model for trend analysis results."""
    metric_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0.0 to 1.0
    confidence_level: float  # 0.0 to 1.0
    data_points: List[Dict[str, Any]]
    forecast: Optional[List[Dict[str, Any]]]

class PatternRecognition(BaseModel):
    """Model for pattern recognition results."""
    pattern_type: str  # 'seasonal', 'cyclical', 'anomaly'
    pattern_description: str
    confidence_score: float
    affected_prompts: List[str]
    recommendations: List[str]