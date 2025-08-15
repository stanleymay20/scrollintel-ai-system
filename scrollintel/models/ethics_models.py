"""
Data models for EthicsEngine - AI bias detection and fairness evaluation
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

from .database import Base

class BiasDetectionResult(Base):
    """Model for storing bias detection results"""
    __tablename__ = "bias_detection_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    model_name = Column(String, index=True)
    dataset_name = Column(String)
    protected_attributes = Column(JSON)  # List of protected attribute names
    total_samples = Column(Integer)
    bias_detected = Column(Boolean, default=False)
    fairness_metrics = Column(JSON)  # Detailed fairness metrics
    group_statistics = Column(JSON)  # Group-wise statistics
    recommendations = Column(JSON)  # List of recommendations
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TransparencyReport(Base):
    """Model for storing AI transparency reports"""
    __tablename__ = "transparency_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String, unique=True, index=True)
    user_id = Column(String, index=True)
    model_name = Column(String, index=True)
    model_type = Column(String)
    model_version = Column(String)
    training_date = Column(DateTime)
    model_information = Column(JSON)  # Detailed model info
    fairness_assessment = Column(JSON)  # Bias detection results
    performance_metrics = Column(JSON)  # Model performance data
    ethical_compliance = Column(JSON)  # Ethical principle compliance
    risk_assessment = Column(JSON)  # Risk analysis
    recommendations = Column(JSON)  # List of recommendations
    limitations = Column(JSON)  # Model limitations
    monitoring_plan = Column(JSON)  # Ongoing monitoring plan
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ComplianceCheck(Base):
    """Model for storing regulatory compliance checks"""
    __tablename__ = "compliance_checks"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    model_name = Column(String, index=True)
    framework = Column(String, index=True)  # GDPR, NIST, etc.
    compliant = Column(Boolean, default=False)
    issues = Column(JSON)  # List of compliance issues
    recommendations = Column(JSON)  # List of recommendations
    assessment_details = Column(JSON)  # Detailed assessment results
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class EthicsAuditLog(Base):
    """Model for storing ethics engine audit trail"""
    __tablename__ = "ethics_audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    event_type = Column(String, index=True)  # bias_detection, compliance_check, etc.
    event_details = Column(JSON)  # Event-specific details
    model_name = Column(String, index=True)
    protected_attributes = Column(JSON)  # Attributes analyzed
    bias_detected = Column(Boolean)
    compliance_framework = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class FairnessThreshold(Base):
    """Model for storing fairness thresholds configuration"""
    __tablename__ = "fairness_thresholds"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    metric_name = Column(String, index=True)  # demographic_parity_difference, etc.
    threshold_value = Column(Float)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class EthicalPrinciple(Base):
    """Model for storing ethical principles and guidelines"""
    __tablename__ = "ethical_principles"
    
    id = Column(Integer, primary_key=True, index=True)
    principle_name = Column(String, unique=True, index=True)
    description = Column(Text)
    guidelines = Column(JSON)  # Detailed guidelines
    compliance_criteria = Column(JSON)  # Criteria for compliance
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Pydantic models for API requests/responses

class BiasDetectionRequest(BaseModel):
    """Request model for bias detection"""
    data: List[Dict[str, Any]] = Field(..., description="Dataset for bias analysis")
    predictions: List[float] = Field(..., description="Model predictions")
    protected_attributes: List[str] = Field(..., description="Protected attribute column names")
    true_labels: Optional[List[int]] = Field(None, description="Ground truth labels")
    prediction_probabilities: Optional[List[float]] = Field(None, description="Prediction probabilities")
    model_name: Optional[str] = Field(None, description="Model name for tracking")
    dataset_name: Optional[str] = Field(None, description="Dataset name for tracking")

class BiasDetectionResponse(BaseModel):
    """Response model for bias detection"""
    status: str
    bias_analysis: Dict[str, Any]
    timestamp: str

class FairnessMetrics(BaseModel):
    """Model for fairness metrics"""
    demographic_parity: Optional[Dict[str, Any]] = None
    equalized_odds: Optional[Dict[str, Any]] = None
    equal_opportunity: Optional[Dict[str, Any]] = None
    calibration: Optional[Dict[str, Any]] = None

class GroupStatistics(BaseModel):
    """Model for group statistics"""
    sample_size: int
    percentage: float
    positive_prediction_rate: float
    feature_statistics: Dict[str, Dict[str, Optional[float]]]

class TransparencyReportRequest(BaseModel):
    """Request model for transparency report generation"""
    model_info: Dict[str, Any] = Field(..., description="Model information")
    bias_results: Dict[str, Any] = Field(..., description="Bias detection results")
    performance_metrics: Dict[str, Any] = Field(..., description="Model performance metrics")

class TransparencyReportResponse(BaseModel):
    """Response model for transparency report"""
    status: str
    transparency_report: Dict[str, Any]
    timestamp: str

class ComplianceFrameworkEnum(str, Enum):
    """Enum for compliance frameworks"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO_27001 = "iso_27001"
    NIST_AI_RMF = "nist_ai_rmf"
    EU_AI_ACT = "eu_ai_act"

class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking"""
    framework: ComplianceFrameworkEnum = Field(..., description="Compliance framework")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    bias_results: Dict[str, Any] = Field(..., description="Bias detection results")

class ComplianceCheckResponse(BaseModel):
    """Response model for compliance check"""
    status: str
    compliance_check: Dict[str, Any]
    framework: str
    timestamp: str

class AuditTrailRequest(BaseModel):
    """Request model for audit trail retrieval"""
    start_date: Optional[str] = Field(None, description="Start date filter (ISO format)")
    end_date: Optional[str] = Field(None, description="End date filter (ISO format)")
    event_type: Optional[str] = Field(None, description="Event type filter")
    model_name: Optional[str] = Field(None, description="Model name filter")

class AuditTrailResponse(BaseModel):
    """Response model for audit trail"""
    status: str
    audit_trail: List[Dict[str, Any]]
    total_entries: int
    filters: Dict[str, Optional[str]]
    timestamp: str

class FairnessThresholdUpdate(BaseModel):
    """Request model for updating fairness thresholds"""
    thresholds: Dict[str, float] = Field(..., description="New fairness thresholds")

class FairnessThresholdResponse(BaseModel):
    """Response model for fairness threshold update"""
    status: str
    message: str
    updated_thresholds: Dict[str, float]
    timestamp: str

class EthicalGuidelinesResponse(BaseModel):
    """Response model for ethical guidelines"""
    status: str
    guidelines: Dict[str, Any]
    timestamp: str

class BiasMetricType(str, Enum):
    """Enum for bias metric types"""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"

class FairnessMetricType(str, Enum):
    """Enum for fairness metric types"""
    DEMOGRAPHIC_PARITY_DIFFERENCE = "demographic_parity_difference"
    DEMOGRAPHIC_PARITY_RATIO = "demographic_parity_ratio"
    EQUALIZED_ODDS_DIFFERENCE = "equalized_odds_difference"
    EQUALIZED_ODDS_RATIO = "equalized_odds_ratio"
    EQUAL_OPPORTUNITY_DIFFERENCE = "equal_opportunity_difference"
    CALIBRATION_ERROR = "calibration_error"
    STATISTICAL_PARITY = "statistical_parity"

class RiskLevel(str, Enum):
    """Enum for risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EthicalPrincipleType(str, Enum):
    """Enum for ethical principles"""
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    PRIVACY = "privacy"
    BENEFICENCE = "beneficence"
    NON_MALEFICENCE = "non_maleficence"
    AUTONOMY = "autonomy"
    JUSTICE = "justice"

class BiasRecommendation(BaseModel):
    """Model for bias mitigation recommendations"""
    recommendation_type: str
    description: str
    priority: str  # high, medium, low
    implementation_effort: str  # high, medium, low
    expected_impact: str  # high, medium, low

class RiskAssessment(BaseModel):
    """Model for risk assessment"""
    risk_type: str
    risk_level: RiskLevel
    description: str
    mitigation_strategies: List[str]
    monitoring_requirements: List[str]

class MonitoringPlan(BaseModel):
    """Model for monitoring plan"""
    bias_monitoring: Dict[str, Any]
    performance_monitoring: Dict[str, Any]
    compliance_review: Dict[str, Any]
    stakeholder_review: Dict[str, Any]

class EthicsEngineStatus(BaseModel):
    """Model for ethics engine status"""
    engine_id: str
    name: str
    status: str
    version: str
    audit_entries: int
    protected_attributes: int
    supported_metrics: List[str]
    compliance_frameworks: List[str]
    healthy: bool

class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    engine: str
    timestamp: str
    error: Optional[str] = None