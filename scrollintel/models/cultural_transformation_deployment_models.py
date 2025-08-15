"""
Data models for Cultural Transformation Leadership System Deployment
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

Base = declarative_base()

class DeploymentStatus(str, Enum):
    """Deployment status enumeration"""
    INITIALIZING = "initializing"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

class ValidationStatus(str, Enum):
    """Validation status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class OrganizationSize(str, Enum):
    """Organization size enumeration"""
    STARTUP = "startup"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"

class CultureMaturity(str, Enum):
    """Culture maturity level enumeration"""
    EMERGING = "emerging"
    DEVELOPING = "developing"
    MATURE = "mature"
    ADVANCED = "advanced"

class ComplexityLevel(str, Enum):
    """Complexity level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class CulturalTransformationDeployment(Base):
    """Cultural transformation deployment tracking"""
    __tablename__ = "cultural_transformation_deployments"
    
    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(String(100), unique=True, index=True)
    status = Column(String(50), default=DeploymentStatus.INITIALIZING)
    
    # Deployment configuration
    components_count = Column(Integer, default=0)
    integration_count = Column(Integer, default=0)
    
    # Deployment metrics
    deployment_start_time = Column(DateTime, default=func.now())
    deployment_end_time = Column(DateTime, nullable=True)
    deployment_duration = Column(Float, nullable=True)  # in minutes
    
    # System health metrics
    system_health_score = Column(Float, default=0.0)
    component_health_status = Column(JSON, nullable=True)
    integration_health_status = Column(JSON, nullable=True)
    performance_metrics = Column(JSON, nullable=True)
    
    # Deployment results
    deployment_success = Column(Boolean, default=False)
    deployment_errors = Column(JSON, nullable=True)
    deployment_warnings = Column(JSON, nullable=True)
    
    # Configuration details
    production_config = Column(JSON, nullable=True)
    security_config = Column(JSON, nullable=True)
    monitoring_config = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class OrganizationValidation(Base):
    """Organization type validation tracking"""
    __tablename__ = "organization_validations"
    
    id = Column(Integer, primary_key=True, index=True)
    validation_id = Column(String(100), unique=True, index=True)
    deployment_id = Column(String(100), index=True)
    
    # Organization characteristics
    organization_name = Column(String(200))
    organization_size = Column(String(50))
    industry = Column(String(100))
    culture_maturity = Column(String(50))
    complexity_level = Column(String(50))
    
    # Validation metrics
    assessment_accuracy = Column(Float, default=0.0)
    transformation_effectiveness = Column(Float, default=0.0)
    behavioral_change_success = Column(Float, default=0.0)
    engagement_improvement = Column(Float, default=0.0)
    sustainability_score = Column(Float, default=0.0)
    overall_success = Column(Float, default=0.0)
    
    # Additional metrics
    complexity_handling_score = Column(Float, default=0.0)
    scalability_score = Column(Float, default=0.0)
    adaptability_score = Column(Float, default=0.0)
    
    # Validation details
    validation_status = Column(String(50), default=ValidationStatus.PENDING)
    validation_start_time = Column(DateTime, default=func.now())
    validation_end_time = Column(DateTime, nullable=True)
    validation_duration = Column(Float, nullable=True)  # in minutes
    
    # Detailed results
    simulation_results = Column(JSON, nullable=True)
    detailed_metrics = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class ContinuousLearningSystem(Base):
    """Continuous learning system configuration"""
    __tablename__ = "continuous_learning_systems"
    
    id = Column(Integer, primary_key=True, index=True)
    system_id = Column(String(100), unique=True, index=True)
    deployment_id = Column(String(100), index=True)
    
    # Learning system status
    status = Column(String(50), default="initializing")
    setup_start_time = Column(DateTime, default=func.now())
    setup_end_time = Column(DateTime, nullable=True)
    
    # Learning components
    feedback_collection_config = Column(JSON, nullable=True)
    performance_monitoring_config = Column(JSON, nullable=True)
    adaptation_engine_config = Column(JSON, nullable=True)
    knowledge_base_config = Column(JSON, nullable=True)
    improvement_pipeline_config = Column(JSON, nullable=True)
    
    # Learning metrics
    feedback_processing_rate = Column(Float, default=0.0)
    adaptation_frequency = Column(Float, default=0.0)
    knowledge_growth_rate = Column(Float, default=0.0)
    improvement_implementation_rate = Column(Float, default=0.0)
    
    # Learning effectiveness
    learning_effectiveness_score = Column(Float, default=0.0)
    adaptation_success_rate = Column(Float, default=0.0)
    improvement_impact_score = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class SystemHealthMetrics(Base):
    """System health metrics tracking"""
    __tablename__ = "system_health_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(String(100), index=True)
    
    # Timestamp
    measurement_timestamp = Column(DateTime, default=func.now())
    
    # Overall health
    overall_health_score = Column(Float, default=0.0)
    system_status = Column(String(50))
    
    # Component health
    component_health_scores = Column(JSON, nullable=True)
    component_status_details = Column(JSON, nullable=True)
    
    # Integration health
    integration_health_scores = Column(JSON, nullable=True)
    integration_status_details = Column(JSON, nullable=True)
    
    # Performance metrics
    response_time = Column(Float, default=0.0)  # in seconds
    throughput = Column(Float, default=0.0)  # requests per minute
    cpu_usage = Column(Float, default=0.0)  # percentage
    memory_usage = Column(Float, default=0.0)  # percentage
    error_rate = Column(Float, default=0.0)  # percentage
    
    # Resource availability
    cpu_available = Column(Float, default=0.0)
    memory_available = Column(Float, default=0.0)
    storage_available = Column(Float, default=0.0)
    network_bandwidth = Column(Float, default=0.0)
    
    # Security status
    security_status = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=func.now())

class ValidationReport(Base):
    """Comprehensive validation reports"""
    __tablename__ = "validation_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(100), unique=True, index=True)
    deployment_id = Column(String(100), index=True)
    
    # Report metadata
    report_type = Column(String(50))  # comprehensive, summary, detailed
    generation_timestamp = Column(DateTime, default=func.now())
    
    # Overall validation results
    total_organizations_tested = Column(Integer, default=0)
    overall_validation_success = Column(Float, default=0.0)
    
    # Success by categories
    success_by_size = Column(JSON, nullable=True)
    success_by_industry = Column(JSON, nullable=True)
    success_by_complexity = Column(JSON, nullable=True)
    success_by_maturity = Column(JSON, nullable=True)
    
    # Detailed analysis
    performance_analysis = Column(JSON, nullable=True)
    effectiveness_analysis = Column(JSON, nullable=True)
    scalability_analysis = Column(JSON, nullable=True)
    
    # Recommendations
    system_recommendations = Column(JSON, nullable=True)
    improvement_suggestions = Column(JSON, nullable=True)
    optimization_opportunities = Column(JSON, nullable=True)
    
    # Report content
    executive_summary = Column(Text, nullable=True)
    detailed_findings = Column(JSON, nullable=True)
    appendices = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=func.now())

@dataclass
class DeploymentRequest:
    """Deployment request data model"""
    deployment_type: str = "full_system"
    configuration: Optional[Dict[str, Any]] = None
    environment: str = "production"
    validation_required: bool = True
    monitoring_enabled: bool = True

@dataclass
class ValidationRequest:
    """Validation request data model"""
    validation_type: str = "comprehensive"
    organization_types: Optional[List[str]] = None
    custom_organizations: Optional[List[Dict[str, Any]]] = None
    validation_depth: str = "full"
    generate_report: bool = True

@dataclass
class LearningSystemRequest:
    """Learning system setup request"""
    learning_components: List[str]
    configuration: Optional[Dict[str, Any]] = None
    integration_mode: str = "full"
    monitoring_enabled: bool = True

@dataclass
class HealthCheckRequest:
    """Health check request data model"""
    check_type: str = "comprehensive"
    include_performance: bool = True
    include_security: bool = True
    include_integrations: bool = True

@dataclass
class DeploymentResponse:
    """Deployment response data model"""
    deployment_id: str
    status: str
    message: str
    estimated_completion: Optional[str] = None
    components_count: Optional[int] = None
    timestamp: str = datetime.now().isoformat()

@dataclass
class ValidationResponse:
    """Validation response data model"""
    validation_id: str
    status: str
    message: str
    organizations_count: Optional[int] = None
    estimated_completion: Optional[str] = None
    timestamp: str = datetime.now().isoformat()

@dataclass
class HealthCheckResponse:
    """Health check response data model"""
    overall_status: str
    component_health: Dict[str, str]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    timestamp: str = datetime.now().isoformat()

@dataclass
class SystemCapabilities:
    """System capabilities data model"""
    cultural_transformation_capabilities: Dict[str, List[str]]
    integration_capabilities: List[str]
    validation_coverage: List[str]
    continuous_improvement: List[str]
    deployment_readiness: Dict[str, Any]

@dataclass
class MetricsSummary:
    """Metrics summary data model"""
    deployment_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    capability_metrics: Dict[str, Any]
    integration_metrics: Dict[str, Any]
    overall_readiness: str
    recommendation: str

# Pydantic models for API validation
from pydantic import BaseModel
from typing import Union

class DeploymentStatusModel(BaseModel):
    """Pydantic model for deployment status"""
    deployment_id: str
    status: str
    components_count: int
    system_health_score: float
    deployment_success: bool
    created_at: datetime
    updated_at: datetime

class ValidationResultModel(BaseModel):
    """Pydantic model for validation results"""
    validation_id: str
    organization_name: str
    organization_size: str
    industry: str
    assessment_accuracy: float
    transformation_effectiveness: float
    behavioral_change_success: float
    engagement_improvement: float
    sustainability_score: float
    overall_success: float
    validation_timestamp: datetime

class SystemHealthModel(BaseModel):
    """Pydantic model for system health"""
    overall_health_score: float
    system_status: str
    component_health_scores: Dict[str, float]
    performance_metrics: Dict[str, float]
    measurement_timestamp: datetime

class LearningSystemModel(BaseModel):
    """Pydantic model for learning system"""
    system_id: str
    status: str
    learning_effectiveness_score: float
    adaptation_success_rate: float
    improvement_impact_score: float
    created_at: datetime