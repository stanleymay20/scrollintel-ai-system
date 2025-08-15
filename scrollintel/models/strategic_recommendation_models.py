"""
Strategic Recommendation Models

Data models for strategic recommendation system including board priorities,
recommendations, financial impact, and validation results.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, Any, List

Base = declarative_base()

class BoardPriorityModel(Base):
    """Board priority data model"""
    __tablename__ = 'board_priorities'
    
    id = Column(String, primary_key=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    priority_level = Column(String(20), nullable=False)  # critical, high, medium, low
    impact_areas = Column(JSON)  # List of impact areas
    target_timeline = Column(String(100))
    success_metrics = Column(JSON)  # List of success metrics
    stakeholders = Column(JSON)  # List of stakeholders
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    recommendations = relationship("StrategicRecommendationModel", back_populates="priority_refs")

class StrategicRecommendationModel(Base):
    """Strategic recommendation data model"""
    __tablename__ = 'strategic_recommendations'
    
    id = Column(String, primary_key=True)
    title = Column(String(200), nullable=False)
    recommendation_type = Column(String(50), nullable=False)
    board_priorities = Column(JSON)  # List of priority IDs
    strategic_rationale = Column(Text)
    quality_score = Column(Float, default=0.0)
    impact_prediction = Column(JSON)  # Dict of impact predictions
    validation_status = Column(String(20), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    financial_impact = relationship("FinancialImpactModel", back_populates="recommendation", uselist=False)
    risk_assessment = relationship("RiskAssessmentModel", back_populates="recommendation", uselist=False)
    implementation_plan = relationship("ImplementationPlanModel", back_populates="recommendation", uselist=False)
    priority_refs = relationship("BoardPriorityModel", back_populates="recommendations")

class FinancialImpactModel(Base):
    """Financial impact data model"""
    __tablename__ = 'financial_impacts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    recommendation_id = Column(String, ForeignKey('strategic_recommendations.id'), nullable=False)
    revenue_impact = Column(Float, default=0.0)
    cost_impact = Column(Float, default=0.0)
    roi_projection = Column(Float, default=0.0)
    payback_period = Column(Integer, default=12)  # months
    confidence_level = Column(Float, default=0.8)
    assumptions = Column(JSON)  # List of assumptions
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    recommendation = relationship("StrategicRecommendationModel", back_populates="financial_impact")

class RiskAssessmentModel(Base):
    """Risk assessment data model"""
    __tablename__ = 'risk_assessments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    recommendation_id = Column(String, ForeignKey('strategic_recommendations.id'), nullable=False)
    risk_level = Column(String(20), nullable=False)
    key_risks = Column(JSON)  # List of key risks
    mitigation_strategies = Column(JSON)  # List of mitigation strategies
    success_probability = Column(Float, default=0.7)
    contingency_plans = Column(JSON)  # List of contingency plans
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    recommendation = relationship("StrategicRecommendationModel", back_populates="risk_assessment")

class ImplementationPlanModel(Base):
    """Implementation plan data model"""
    __tablename__ = 'implementation_plans'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    recommendation_id = Column(String, ForeignKey('strategic_recommendations.id'), nullable=False)
    phases = Column(JSON)  # List of implementation phases
    timeline = Column(String(50))
    resource_requirements = Column(JSON)  # Dict of resource requirements
    dependencies = Column(JSON)  # List of dependencies
    milestones = Column(JSON)  # List of milestones
    success_criteria = Column(JSON)  # List of success criteria
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    recommendation = relationship("StrategicRecommendationModel", back_populates="implementation_plan")

class RecommendationValidationModel(Base):
    """Recommendation validation results data model"""
    __tablename__ = 'recommendation_validations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    recommendation_id = Column(String, ForeignKey('strategic_recommendations.id'), nullable=False)
    validation_status = Column(String(20), nullable=False)  # approved, needs_improvement, rejected
    quality_score = Column(Float, nullable=False)
    meets_threshold = Column(Boolean, default=False)
    validation_details = Column(JSON)  # Detailed validation results
    improvement_recommendations = Column(JSON)  # List of improvement suggestions
    validated_by = Column(String(100))
    validated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    recommendation = relationship("StrategicRecommendationModel")

class RecommendationFeedbackModel(Base):
    """Recommendation feedback data model"""
    __tablename__ = 'recommendation_feedback'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    recommendation_id = Column(String, ForeignKey('strategic_recommendations.id'), nullable=False)
    feedback_type = Column(String(20), nullable=False)  # board, executive, stakeholder
    feedback_source = Column(String(100))  # Who provided feedback
    feedback_content = Column(Text)
    rating = Column(Integer)  # 1-5 rating
    suggestions = Column(JSON)  # List of suggestions
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    recommendation = relationship("StrategicRecommendationModel")

class RecommendationMetricsModel(Base):
    """Recommendation performance metrics data model"""
    __tablename__ = 'recommendation_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    recommendation_id = Column(String, ForeignKey('strategic_recommendations.id'), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float)
    metric_target = Column(Float)
    measurement_date = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    recommendation = relationship("StrategicRecommendationModel")

# Pydantic models for API serialization
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class RecommendationTypeEnum(str, Enum):
    STRATEGIC_INITIATIVE = "strategic_initiative"
    OPERATIONAL_IMPROVEMENT = "operational_improvement"
    TECHNOLOGY_INVESTMENT = "technology_investment"
    MARKET_EXPANSION = "market_expansion"
    RISK_MITIGATION = "risk_mitigation"
    COST_OPTIMIZATION = "cost_optimization"
    PARTNERSHIP = "partnership"
    ACQUISITION = "acquisition"

class PriorityLevelEnum(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ValidationStatusEnum(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    NEEDS_IMPROVEMENT = "needs_improvement"
    REJECTED = "rejected"

class BoardPriorityCreate(BaseModel):
    title: str = Field(..., max_length=200)
    description: str
    priority_level: PriorityLevelEnum
    impact_areas: List[str]
    target_timeline: str
    success_metrics: List[str]
    stakeholders: List[str]

class BoardPriorityResponse(BaseModel):
    id: str
    title: str
    description: str
    priority_level: str
    impact_areas: List[str]
    target_timeline: str
    success_metrics: List[str]
    stakeholders: List[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class FinancialImpactCreate(BaseModel):
    revenue_impact: float = 0.0
    cost_impact: float = 0.0
    roi_projection: float = 0.0
    payback_period: int = 12
    confidence_level: float = 0.8
    assumptions: List[str] = []

class FinancialImpactResponse(BaseModel):
    revenue_impact: float
    cost_impact: float
    roi_projection: float
    payback_period: int
    confidence_level: float
    assumptions: List[str]
    
    class Config:
        from_attributes = True

class RiskAssessmentCreate(BaseModel):
    risk_level: str
    key_risks: List[str]
    mitigation_strategies: List[str]
    success_probability: float = 0.7
    contingency_plans: List[str] = []

class RiskAssessmentResponse(BaseModel):
    risk_level: str
    key_risks: List[str]
    mitigation_strategies: List[str]
    success_probability: float
    contingency_plans: List[str]
    
    class Config:
        from_attributes = True

class ImplementationPlanCreate(BaseModel):
    phases: List[Dict[str, Any]]
    timeline: str
    resource_requirements: Dict[str, Any]
    dependencies: List[str]
    milestones: List[Dict[str, Any]]
    success_criteria: List[str]

class ImplementationPlanResponse(BaseModel):
    phases: List[Dict[str, Any]]
    timeline: str
    resource_requirements: Dict[str, Any]
    dependencies: List[str]
    milestones: List[Dict[str, Any]]
    success_criteria: List[str]
    
    class Config:
        from_attributes = True

class StrategicRecommendationCreate(BaseModel):
    title: str = Field(..., max_length=200)
    recommendation_type: RecommendationTypeEnum
    board_priorities: List[str]
    strategic_context: Dict[str, Any] = {}

class StrategicRecommendationResponse(BaseModel):
    id: str
    title: str
    recommendation_type: str
    board_priorities: List[str]
    strategic_rationale: str
    quality_score: float
    impact_prediction: Dict[str, float]
    validation_status: str
    financial_impact: Optional[FinancialImpactResponse] = None
    risk_assessment: Optional[RiskAssessmentResponse] = None
    implementation_plan: Optional[ImplementationPlanResponse] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class RecommendationValidationCreate(BaseModel):
    recommendation_id: str
    validated_by: str

class RecommendationValidationResponse(BaseModel):
    recommendation_id: str
    validation_status: str
    quality_score: float
    meets_threshold: bool
    validation_details: Dict[str, Any]
    improvement_recommendations: List[str]
    validated_by: str
    validated_at: datetime
    
    class Config:
        from_attributes = True

class RecommendationFeedbackCreate(BaseModel):
    recommendation_id: str
    feedback_type: str = Field(..., regex="^(board|executive|stakeholder)$")
    feedback_source: str
    feedback_content: str
    rating: Optional[int] = Field(None, ge=1, le=5)
    suggestions: List[str] = []

class RecommendationFeedbackResponse(BaseModel):
    id: int
    recommendation_id: str
    feedback_type: str
    feedback_source: str
    feedback_content: str
    rating: Optional[int]
    suggestions: List[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class RecommendationSummaryResponse(BaseModel):
    title: str
    type: str
    quality_score: float
    strategic_alignment: int
    financial_impact: Dict[str, str]
    risk_assessment: Dict[str, str]
    implementation_timeline: str
    validation_status: str
    key_benefits: List[str]
    next_steps: List[str]

class RecommendationOptimizationRequest(BaseModel):
    recommendation_id: str
    optimization_areas: List[str] = []  # specific areas to focus optimization on

class RecommendationOptimizationResponse(BaseModel):
    recommendation_id: str
    original_score: float
    optimized_score: float
    improvement: float
    optimization_details: Dict[str, Any]
    updated_at: datetime