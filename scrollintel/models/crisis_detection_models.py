"""
Data models for Crisis Detection and Assessment Engine
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, Any, List, Optional

Base = declarative_base()


class CrisisRecord(Base):
    """Database model for crisis records"""
    __tablename__ = "crisis_records"
    
    id = Column(String, primary_key=True)
    crisis_type = Column(String, nullable=False)
    severity_level = Column(Integer, nullable=False)
    status = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    resolution_time = Column(DateTime, nullable=True)
    affected_areas = Column(JSON, nullable=False, default=list)
    stakeholders_impacted = Column(JSON, nullable=False, default=list)
    
    # Classification data
    classification_confidence = Column(Float, nullable=True)
    classification_sub_categories = Column(JSON, nullable=True, default=list)
    classification_related_crises = Column(JSON, nullable=True, default=list)
    classification_rationale = Column(Text, nullable=True)
    
    # Impact assessment data
    financial_impact = Column(JSON, nullable=True, default=dict)
    operational_impact = Column(JSON, nullable=True, default=dict)
    reputation_impact = Column(JSON, nullable=True, default=dict)
    stakeholder_impact = Column(JSON, nullable=True, default=dict)
    timeline_impact = Column(JSON, nullable=True, default=dict)
    recovery_estimate_seconds = Column(Integer, nullable=True)
    cascading_risks = Column(JSON, nullable=True, default=list)
    mitigation_urgency = Column(Integer, nullable=True)
    
    # Escalation data
    escalation_history = Column(JSON, nullable=False, default=list)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    signals = relationship("CrisisSignal", back_populates="crisis", cascade="all, delete-orphan")


class CrisisSignal(Base):
    """Database model for crisis signals"""
    __tablename__ = "crisis_signals"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    crisis_id = Column(String, ForeignKey("crisis_records.id"), nullable=False)
    source = Column(String, nullable=False)
    signal_type = Column(String, nullable=False)
    value = Column(String, nullable=False)  # Store as string to handle various types
    timestamp = Column(DateTime, nullable=False)
    confidence = Column(Float, nullable=False)
    signal_metadata = Column(JSON, nullable=False, default=dict)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    crisis = relationship("CrisisRecord", back_populates="signals")


class EscalationRecord(Base):
    """Database model for escalation records"""
    __tablename__ = "escalation_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    crisis_id = Column(String, ForeignKey("crisis_records.id"), nullable=False)
    escalation_level = Column(Integer, nullable=False)
    escalation_reason = Column(Text, nullable=False)
    escalation_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    notifications_sent = Column(JSON, nullable=False, default=list)
    
    # Status tracking
    is_resolved = Column(Boolean, nullable=False, default=False)
    resolution_time = Column(DateTime, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class CrisisMetrics(Base):
    """Database model for crisis metrics and analytics"""
    __tablename__ = "crisis_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Detection metrics
    total_signals_processed = Column(Integer, nullable=False, default=0)
    potential_crises_identified = Column(Integer, nullable=False, default=0)
    actual_crises_created = Column(Integer, nullable=False, default=0)
    false_positive_rate = Column(Float, nullable=False, default=0.0)
    
    # Response metrics
    average_detection_time_seconds = Column(Float, nullable=False, default=0.0)
    average_classification_time_seconds = Column(Float, nullable=False, default=0.0)
    average_assessment_time_seconds = Column(Float, nullable=False, default=0.0)
    average_escalation_time_seconds = Column(Float, nullable=False, default=0.0)
    
    # Resolution metrics
    total_crises_resolved = Column(Integer, nullable=False, default=0)
    average_resolution_time_seconds = Column(Float, nullable=False, default=0.0)
    escalation_rate = Column(Float, nullable=False, default=0.0)
    
    # Crisis type distribution
    crisis_type_distribution = Column(JSON, nullable=False, default=dict)
    severity_distribution = Column(JSON, nullable=False, default=dict)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class CrisisTemplate(Base):
    """Database model for crisis response templates"""
    __tablename__ = "crisis_templates"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    crisis_type = Column(String, nullable=False)
    severity_level = Column(Integer, nullable=False)
    
    # Template content
    response_procedures = Column(JSON, nullable=False, default=list)
    escalation_rules = Column(JSON, nullable=False, default=dict)
    communication_templates = Column(JSON, nullable=False, default=dict)
    resource_requirements = Column(JSON, nullable=False, default=dict)
    
    # Template metadata
    is_active = Column(Boolean, nullable=False, default=True)
    version = Column(String, nullable=False, default="1.0")
    created_by = Column(String, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class CrisisLearning(Base):
    """Database model for crisis learning and improvement"""
    __tablename__ = "crisis_learning"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    crisis_id = Column(String, ForeignKey("crisis_records.id"), nullable=False)
    
    # Learning data
    lessons_learned = Column(JSON, nullable=False, default=list)
    improvement_recommendations = Column(JSON, nullable=False, default=list)
    process_gaps_identified = Column(JSON, nullable=False, default=list)
    
    # Effectiveness metrics
    response_effectiveness_score = Column(Float, nullable=True)
    communication_effectiveness_score = Column(Float, nullable=True)
    resolution_effectiveness_score = Column(Float, nullable=True)
    
    # Follow-up actions
    action_items = Column(JSON, nullable=False, default=list)
    responsible_parties = Column(JSON, nullable=False, default=list)
    implementation_status = Column(String, nullable=False, default="pending")
    
    # Metadata
    analysis_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    analyzed_by = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic models for API serialization
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any, Optional


class SignalModel(BaseModel):
    """Pydantic model for crisis signals"""
    source: str
    signal_type: str
    value: Any
    timestamp: datetime
    confidence: float
    metadata: Dict[str, Any] = {}


class CrisisClassificationModel(BaseModel):
    """Pydantic model for crisis classification"""
    crisis_type: str
    severity_level: str
    confidence: float
    sub_categories: List[str] = []
    related_crises: List[str] = []
    classification_rationale: str


class ImpactAssessmentModel(BaseModel):
    """Pydantic model for impact assessment"""
    financial_impact: Dict[str, float]
    operational_impact: Dict[str, str]
    reputation_impact: Dict[str, float]
    stakeholder_impact: Dict[str, List[str]]
    timeline_impact: Dict[str, float]  # Converted to seconds
    recovery_estimate_seconds: float
    cascading_risks: List[str]
    mitigation_urgency: str


class CrisisModel(BaseModel):
    """Pydantic model for crisis records"""
    id: str
    crisis_type: str
    severity_level: str
    status: str
    start_time: datetime
    resolution_time: Optional[datetime] = None
    affected_areas: List[str]
    stakeholders_impacted: List[str]
    signals: List[SignalModel] = []
    classification: Optional[CrisisClassificationModel] = None
    impact_assessment: Optional[ImpactAssessmentModel] = None
    escalation_history: List[Dict[str, Any]] = []


class CrisisMetricsModel(BaseModel):
    """Pydantic model for crisis metrics"""
    active_crises: int
    total_resolved: int
    average_resolution_time_seconds: float
    crisis_type_distribution: Dict[str, int]
    escalation_rate: float


class CrisisCreateRequest(BaseModel):
    """Pydantic model for crisis creation requests"""
    crisis_type: str
    affected_areas: List[str] = []
    stakeholders_impacted: List[str] = []
    initial_signals: List[SignalModel] = []
    metadata: Dict[str, Any] = {}


class SignalSubmissionRequest(BaseModel):
    """Pydantic model for signal submission"""
    source: str
    signal_type: str
    value: Any
    confidence: float
    metadata: Dict[str, Any] = {}


class CrisisSimulationRequest(BaseModel):
    """Pydantic model for crisis simulation"""
    crisis_type: str
    severity_level: Optional[str] = None
    affected_areas: List[str] = []
    custom_signals: List[SignalModel] = []
    metadata: Dict[str, Any] = {}


class EscalationNotificationModel(BaseModel):
    """Pydantic model for escalation notifications"""
    recipient: str
    channel: str
    message: str
    sent_time: datetime
    status: str


class CrisisTemplateModel(BaseModel):
    """Pydantic model for crisis templates"""
    name: str
    crisis_type: str
    severity_level: int
    response_procedures: List[str]
    escalation_rules: Dict[str, Any]
    communication_templates: Dict[str, str]
    resource_requirements: Dict[str, Any]
    is_active: bool = True
    version: str = "1.0"


class CrisisLearningModel(BaseModel):
    """Pydantic model for crisis learning records"""
    crisis_id: str
    lessons_learned: List[str]
    improvement_recommendations: List[str]
    process_gaps_identified: List[str]
    response_effectiveness_score: Optional[float] = None
    communication_effectiveness_score: Optional[float] = None
    resolution_effectiveness_score: Optional[float] = None
    action_items: List[Dict[str, Any]] = []
    responsible_parties: List[str] = []
    implementation_status: str = "pending"