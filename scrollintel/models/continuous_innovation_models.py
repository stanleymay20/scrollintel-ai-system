"""
Data models for the Continuous Innovation Engine.

This module defines the database models and schemas for tracking research breakthroughs,
patent opportunities, competitive intelligence, and innovation metrics.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

Base = declarative_base()


class InnovationPriorityEnum(str, Enum):
    """Priority levels for innovation opportunities."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PatentStatusEnum(str, Enum):
    """Status of patent applications."""
    PENDING = "pending"
    FILED = "filed"
    APPROVED = "approved"
    REJECTED = "rejected"


class ResearchBreakthroughModel(Base):
    """Database model for research breakthroughs."""
    __tablename__ = "research_breakthroughs"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    source = Column(String, nullable=False)
    relevance_score = Column(Float, nullable=False)
    potential_impact = Column(String, nullable=False)
    discovered_at = Column(DateTime, default=datetime.utcnow)
    keywords = Column(JSON)
    priority = Column(SQLEnum(InnovationPriorityEnum), nullable=False)
    implementation_complexity = Column(String, nullable=False)
    estimated_timeline = Column(String, nullable=False)
    competitive_advantage = Column(Float, nullable=False)
    implemented = Column(Boolean, default=False)
    implementation_date = Column(DateTime, nullable=True)
    roi_achieved = Column(Float, nullable=True)
    
    # Relationships
    patent_opportunities = relationship("PatentOpportunityModel", back_populates="breakthrough")


class PatentOpportunityModel(Base):
    """Database model for patent opportunities."""
    __tablename__ = "patent_opportunities"
    
    id = Column(String, primary_key=True)
    innovation_id = Column(String, ForeignKey("research_breakthroughs.id"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    technical_details = Column(Text, nullable=False)
    novelty_score = Column(Float, nullable=False)
    commercial_potential = Column(Float, nullable=False)
    filing_priority = Column(SQLEnum(InnovationPriorityEnum), nullable=False)
    estimated_cost = Column(Float, nullable=False)
    status = Column(SQLEnum(PatentStatusEnum), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    filed_at = Column(DateTime, nullable=True)
    approved_at = Column(DateTime, nullable=True)
    patent_number = Column(String, nullable=True)
    
    # Relationships
    breakthrough = relationship("ResearchBreakthroughModel", back_populates="patent_opportunities")


class CompetitorIntelligenceModel(Base):
    """Database model for competitive intelligence."""
    __tablename__ = "competitor_intelligence"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    competitor_name = Column(String, nullable=False, unique=True)
    technology_area = Column(String, nullable=False)
    recent_developments = Column(JSON)
    patent_filings = Column(JSON)
    market_position = Column(String, nullable=False)
    threat_level = Column(Float, nullable=False)
    opportunities = Column(JSON)
    last_updated = Column(DateTime, default=datetime.utcnow)
    market_share = Column(Float, nullable=True)
    funding_raised = Column(Float, nullable=True)
    employee_count = Column(Integer, nullable=True)
    key_personnel = Column(JSON, nullable=True)


class InnovationMetricsModel(Base):
    """Database model for innovation metrics tracking."""
    __tablename__ = "innovation_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    total_breakthroughs = Column(Integer, default=0)
    patents_filed = Column(Integer, default=0)
    patents_approved = Column(Integer, default=0)
    competitive_advantages_gained = Column(Integer, default=0)
    implementation_success_rate = Column(Float, default=0.0)
    roi_on_innovation = Column(Float, default=0.0)
    time_to_market_average = Column(Float, default=0.0)
    breakthrough_prediction_accuracy = Column(Float, default=0.0)
    research_investment = Column(Float, default=0.0)
    revenue_from_innovation = Column(Float, default=0.0)


class ResearchInitiativeModel(Base):
    """Database model for research initiatives."""
    __tablename__ = "research_initiatives"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    objective = Column(Text, nullable=False)
    priority = Column(SQLEnum(InnovationPriorityEnum), nullable=False)
    budget_allocated = Column(Float, nullable=False)
    budget_spent = Column(Float, default=0.0)
    start_date = Column(DateTime, nullable=False)
    target_completion_date = Column(DateTime, nullable=False)
    actual_completion_date = Column(DateTime, nullable=True)
    status = Column(String, nullable=False, default="active")
    success_metrics = Column(JSON)
    team_members = Column(JSON)
    milestones = Column(JSON)
    risks = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


# Pydantic models for API serialization

class ResearchBreakthroughSchema(BaseModel):
    """Pydantic schema for research breakthroughs."""
    id: str
    title: str
    description: str
    source: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    potential_impact: str
    discovered_at: datetime
    keywords: List[str]
    priority: InnovationPriorityEnum
    implementation_complexity: str
    estimated_timeline: str
    competitive_advantage: float = Field(..., ge=0.0, le=1.0)
    implemented: bool = False
    implementation_date: Optional[datetime] = None
    roi_achieved: Optional[float] = None
    
    class Config:
        from_attributes = True


class PatentOpportunitySchema(BaseModel):
    """Pydantic schema for patent opportunities."""
    id: str
    innovation_id: str
    title: str
    description: str
    technical_details: str
    novelty_score: float = Field(..., ge=0.0, le=1.0)
    commercial_potential: float = Field(..., ge=0.0, le=1.0)
    filing_priority: InnovationPriorityEnum
    estimated_cost: float = Field(..., ge=0.0)
    status: PatentStatusEnum
    created_at: datetime
    filed_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    patent_number: Optional[str] = None
    
    class Config:
        from_attributes = True


class CompetitorIntelligenceSchema(BaseModel):
    """Pydantic schema for competitive intelligence."""
    id: int
    competitor_name: str
    technology_area: str
    recent_developments: List[str]
    patent_filings: List[str]
    market_position: str
    threat_level: float = Field(..., ge=0.0, le=1.0)
    opportunities: List[str]
    last_updated: datetime
    market_share: Optional[float] = None
    funding_raised: Optional[float] = None
    employee_count: Optional[int] = None
    key_personnel: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        from_attributes = True


class InnovationMetricsSchema(BaseModel):
    """Pydantic schema for innovation metrics."""
    id: int
    recorded_at: datetime
    total_breakthroughs: int = Field(..., ge=0)
    patents_filed: int = Field(..., ge=0)
    patents_approved: int = Field(..., ge=0)
    competitive_advantages_gained: int = Field(..., ge=0)
    implementation_success_rate: float = Field(..., ge=0.0, le=1.0)
    roi_on_innovation: float
    time_to_market_average: float = Field(..., ge=0.0)
    breakthrough_prediction_accuracy: float = Field(..., ge=0.0, le=1.0)
    research_investment: float = Field(..., ge=0.0)
    revenue_from_innovation: float = Field(..., ge=0.0)
    
    class Config:
        from_attributes = True


class ResearchInitiativeSchema(BaseModel):
    """Pydantic schema for research initiatives."""
    id: str
    title: str
    description: str
    objective: str
    priority: InnovationPriorityEnum
    budget_allocated: float = Field(..., ge=0.0)
    budget_spent: float = Field(..., ge=0.0)
    start_date: datetime
    target_completion_date: datetime
    actual_completion_date: Optional[datetime] = None
    status: str
    success_metrics: Dict[str, Any]
    team_members: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]
    risks: List[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True


# Request/Response models for API endpoints

class CreateResearchBreakthroughRequest(BaseModel):
    """Request model for creating research breakthroughs."""
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1, max_length=200)
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    potential_impact: str = Field(..., min_length=1, max_length=100)
    keywords: List[str] = Field(..., min_items=1)
    priority: InnovationPriorityEnum
    implementation_complexity: str = Field(..., min_length=1, max_length=50)
    estimated_timeline: str = Field(..., min_length=1, max_length=50)
    competitive_advantage: float = Field(..., ge=0.0, le=1.0)


class CreatePatentOpportunityRequest(BaseModel):
    """Request model for creating patent opportunities."""
    innovation_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=1)
    technical_details: str = Field(..., min_length=1)
    novelty_score: float = Field(..., ge=0.0, le=1.0)
    commercial_potential: float = Field(..., ge=0.0, le=1.0)
    filing_priority: InnovationPriorityEnum
    estimated_cost: float = Field(..., ge=0.0)


class UpdateCompetitorIntelligenceRequest(BaseModel):
    """Request model for updating competitive intelligence."""
    competitor_name: str = Field(..., min_length=1, max_length=200)
    technology_area: str = Field(..., min_length=1, max_length=200)
    recent_developments: List[str] = Field(default_factory=list)
    patent_filings: List[str] = Field(default_factory=list)
    market_position: str = Field(..., min_length=1, max_length=100)
    threat_level: float = Field(..., ge=0.0, le=1.0)
    opportunities: List[str] = Field(default_factory=list)
    market_share: Optional[float] = Field(None, ge=0.0, le=1.0)
    funding_raised: Optional[float] = Field(None, ge=0.0)
    employee_count: Optional[int] = Field(None, ge=0)
    key_personnel: Optional[List[Dict[str, Any]]] = None


class CreateResearchInitiativeRequest(BaseModel):
    """Request model for creating research initiatives."""
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=1)
    objective: str = Field(..., min_length=1)
    priority: InnovationPriorityEnum
    budget_allocated: float = Field(..., ge=0.0)
    start_date: datetime
    target_completion_date: datetime
    success_metrics: Dict[str, Any] = Field(default_factory=dict)
    team_members: List[Dict[str, Any]] = Field(default_factory=list)
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    risks: List[Dict[str, Any]] = Field(default_factory=list)


class InnovationSummaryResponse(BaseModel):
    """Response model for innovation summary."""
    metrics: InnovationMetricsSchema
    recent_breakthroughs: List[ResearchBreakthroughSchema]
    patent_opportunities: List[PatentOpportunitySchema]
    competitive_intelligence: List[CompetitorIntelligenceSchema]
    active_initiatives: List[ResearchInitiativeSchema]
    summary_generated_at: datetime


class BreakthroughPredictionResponse(BaseModel):
    """Response model for breakthrough predictions."""
    predicted_breakthroughs: List[Dict[str, Any]]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    prediction_horizon: str
    key_trends: List[str]
    recommended_actions: List[str]
    generated_at: datetime


class CompetitiveThreatAnalysisResponse(BaseModel):
    """Response model for competitive threat analysis."""
    threat_level: float = Field(..., ge=0.0, le=1.0)
    key_threats: List[Dict[str, Any]]
    opportunities: List[Dict[str, Any]]
    recommended_responses: List[str]
    market_positioning: Dict[str, Any]
    analysis_date: datetime