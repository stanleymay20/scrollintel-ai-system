"""
Database models for competitive analysis and market intelligence.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

from scrollintel.models.database import Base


class CompetitorPlatform(Base):
    """Model for storing competitor platform metrics."""
    
    __tablename__ = "competitor_platforms"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform_name = Column(String(100), nullable=False, index=True)
    generation_speed = Column(Float, nullable=False)  # seconds
    quality_score = Column(Float, nullable=False)  # 0-1
    cost_per_generation = Column(Float, nullable=False)  # USD
    uptime_percentage = Column(Float, nullable=False)  # 0-100
    feature_count = Column(Integer, nullable=False)
    user_satisfaction = Column(Float, nullable=False)  # 0-5
    market_share = Column(Float, nullable=False)  # 0-100
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<CompetitorPlatform(platform='{self.platform_name}', quality={self.quality_score})>"


class QualityComparison(Base):
    """Model for storing automated quality comparison results."""
    
    __tablename__ = "quality_comparisons"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    our_score = Column(Float, nullable=False)
    competitor_scores = Column(JSON, nullable=False)  # Dict of platform: score
    advantage_percentage = Column(Float, nullable=False)
    test_prompt = Column(Text, nullable=False)
    comparison_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    detailed_metrics = Column(JSON, nullable=True)  # Additional quality metrics
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<QualityComparison(our_score={self.our_score}, advantage={self.advantage_percentage}%)>"


class MarketIntelligence(Base):
    """Model for storing market intelligence reports."""
    
    __tablename__ = "market_intelligence"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    industry_trends = Column(JSON, nullable=False)  # List of trends
    emerging_technologies = Column(JSON, nullable=False)  # List of technologies
    competitor_updates = Column(JSON, nullable=False)  # List of competitor updates
    market_opportunities = Column(JSON, nullable=False)  # List of opportunities
    threat_assessment = Column(JSON, nullable=False)  # Dict of threat: score
    recommendation_priority = Column(String(50), nullable=False)
    report_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<MarketIntelligence(timestamp={self.report_timestamp}, priority={self.recommendation_priority})>"


class PerformanceMetric(Base):
    """Model for storing our performance metrics over time."""
    
    __tablename__ = "performance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50), nullable=True)
    benchmark_value = Column(Float, nullable=True)
    advantage_percentage = Column(Float, nullable=True)
    measurement_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<PerformanceMetric(name='{self.metric_name}', value={self.metric_value})>"


class CompetitiveAdvantage(Base):
    """Model for storing competitive advantage analysis."""
    
    __tablename__ = "competitive_advantages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    advantage_category = Column(String(100), nullable=False, index=True)
    advantage_description = Column(Text, nullable=False)
    advantage_percentage = Column(Float, nullable=False)
    supporting_metrics = Column(JSON, nullable=True)
    competitive_moat_strength = Column(String(50), nullable=False)
    sustainability_score = Column(Float, nullable=False)  # 0-1
    analysis_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<CompetitiveAdvantage(category='{self.advantage_category}', percentage={self.advantage_percentage}%)>"


class MarketPosition(Base):
    """Model for tracking our market position over time."""
    
    __tablename__ = "market_positions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    market_segment = Column(String(100), nullable=False, index=True)
    position_rank = Column(Integer, nullable=False)  # 1 = leader
    market_share_percentage = Column(Float, nullable=False)
    growth_rate = Column(Float, nullable=False)  # percentage
    competitive_threats = Column(JSON, nullable=True)
    strategic_advantages = Column(JSON, nullable=True)
    position_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<MarketPosition(segment='{self.market_segment}', rank={self.position_rank})>"


class CompetitorUpdate(Base):
    """Model for tracking competitor updates and changes."""
    
    __tablename__ = "competitor_updates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    competitor_name = Column(String(100), nullable=False, index=True)
    update_type = Column(String(50), nullable=False)  # feature, pricing, partnership, etc.
    update_description = Column(Text, nullable=False)
    impact_assessment = Column(String(50), nullable=False)  # low, medium, high
    threat_level = Column(Float, nullable=False)  # 0-1
    response_required = Column(Boolean, nullable=False, default=False)
    response_strategy = Column(Text, nullable=True)
    update_date = Column(DateTime, nullable=False)
    detected_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<CompetitorUpdate(competitor='{self.competitor_name}', type='{self.update_type}')>"


class StrategicRecommendation(Base):
    """Model for storing strategic recommendations based on competitive analysis."""
    
    __tablename__ = "strategic_recommendations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recommendation_type = Column(String(100), nullable=False, index=True)
    priority_level = Column(String(20), nullable=False)  # low, medium, high, critical
    recommendation_title = Column(String(200), nullable=False)
    recommendation_description = Column(Text, nullable=False)
    supporting_analysis = Column(JSON, nullable=True)
    expected_impact = Column(Text, nullable=True)
    implementation_effort = Column(String(20), nullable=True)  # low, medium, high
    timeline_estimate = Column(String(50), nullable=True)
    status = Column(String(20), nullable=False, default="pending")  # pending, approved, implemented, rejected
    generated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<StrategicRecommendation(type='{self.recommendation_type}', priority='{self.priority_level}')>"


class CompetitiveAlert(Base):
    """Model for storing competitive intelligence alerts."""
    
    __tablename__ = "competitive_alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_type = Column(String(50), nullable=False, index=True)
    severity_level = Column(String(20), nullable=False)  # info, warning, critical
    alert_title = Column(String(200), nullable=False)
    alert_description = Column(Text, nullable=False)
    affected_competitors = Column(JSON, nullable=True)  # List of competitor names
    impact_assessment = Column(Text, nullable=True)
    recommended_actions = Column(JSON, nullable=True)  # List of recommended actions
    alert_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    acknowledged = Column(Boolean, nullable=False, default=False)
    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<CompetitiveAlert(type='{self.alert_type}', severity='{self.severity_level}')>"