"""
Data models for the Rapid Prototyping System

This module defines the data structures used by the autonomous innovation lab's
rapid prototyping capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class PrototypeType(Enum):
    """Types of prototypes that can be created"""
    WEB_APP = "web_app"
    MOBILE_APP = "mobile_app"
    API_SERVICE = "api_service"
    ML_MODEL = "ml_model"
    IOT_DEVICE = "iot_device"
    DESKTOP_APP = "desktop_app"
    PROOF_OF_CONCEPT = "proof_of_concept"


class PrototypeStatus(Enum):
    """Status of prototype development"""
    PLANNED = "planned"
    GENERATED = "generated"
    IN_DEVELOPMENT = "in_development"
    FUNCTIONAL = "functional"
    TESTING = "testing"
    VALIDATED = "validated"
    OPTIMIZED = "optimized"
    FAILED = "failed"
    ARCHIVED = "archived"


class ConceptCategory(Enum):
    """Categories of innovation concepts"""
    TECHNOLOGY = "technology"
    PRODUCT = "product"
    SERVICE = "service"
    PROCESS = "process"
    BUSINESS_MODEL = "business_model"
    RESEARCH = "research"


@dataclass
class Concept:
    """Innovation concept for prototyping"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: ConceptCategory = ConceptCategory.TECHNOLOGY
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    target_users: List[str] = field(default_factory=list)
    business_value: str = ""
    technical_complexity: float = 0.5  # 0-1 scale
    innovation_potential: float = 0.5  # 0-1 scale
    market_readiness: float = 0.5  # 0-1 scale
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TechnologyStack:
    """Technology stack for prototype development"""
    primary_technology: str = ""
    framework: str = ""
    language: str = ""
    supporting_tools: List[str] = field(default_factory=list)
    deployment_target: str = ""
    version_requirements: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Quality metrics for prototype evaluation"""
    code_coverage: float = 0.0
    performance_score: float = 0.0
    usability_score: float = 0.0
    reliability_score: float = 0.0
    security_score: float = 0.0
    maintainability_score: float = 0.0
    scalability_score: float = 0.0
    documentation_score: float = 0.0


@dataclass
class TestResult:
    """Result of prototype testing"""
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    test_name: str = ""
    test_type: str = ""  # unit, integration, performance, usability
    status: str = ""  # passed, failed, skipped
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationResult:
    """Result of prototype validation"""
    prototype_id: str = ""
    overall_score: float = 0.0
    category_scores: Dict[str, float] = field(default_factory=dict)
    passes_validation: bool = False
    validation_timestamp: datetime = field(default_factory=datetime.utcnow)
    recommendations: List[str] = field(default_factory=list)
    detailed_feedback: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prototype:
    """Prototype created by the rapid prototyping system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    concept_id: str = ""
    name: str = ""
    description: str = ""
    prototype_type: PrototypeType = PrototypeType.PROOF_OF_CONCEPT
    status: PrototypeStatus = PrototypeStatus.PLANNED
    
    # Technical details
    technology_stack: Optional[TechnologyStack] = None
    generated_code: Dict[str, str] = field(default_factory=dict)
    file_structure: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    # Development tracking
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)
    completion_timestamp: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    development_progress: float = 0.0  # 0-1 scale
    estimated_completion_time: int = 0  # hours
    actual_development_time: int = 0  # hours
    
    # Quality and validation
    quality_metrics: Optional[QualityMetrics] = None
    validation_result: Optional[ValidationResult] = None
    test_results: List[TestResult] = field(default_factory=list)
    
    # Features and capabilities
    implemented_features: List[str] = field(default_factory=list)
    pending_features: List[str] = field(default_factory=list)
    known_issues: List[str] = field(default_factory=list)
    
    # Documentation and resources
    documentation: str = ""
    user_guide: str = ""
    api_documentation: str = ""
    deployment_instructions: str = ""
    
    # Performance and monitoring
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    error_logs: List[str] = field(default_factory=list)
    
    # Configuration and settings
    configuration: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Business and impact metrics
    business_value_delivered: float = 0.0
    user_feedback: List[str] = field(default_factory=list)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Prototype-specific flags
    is_deployable: bool = False
    is_scalable: bool = False
    error_handling_implemented: bool = False
    security_implemented: bool = False
    monitoring_implemented: bool = False
    
    # Metadata and tags
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"


@dataclass
class PrototypingSession:
    """Session for managing multiple related prototypes"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    concept_ids: List[str] = field(default_factory=list)
    prototype_ids: List[str] = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.utcnow)
    session_end: Optional[datetime] = None
    status: str = "active"  # active, completed, paused, cancelled
    objectives: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    resource_allocation: Dict[str, Any] = field(default_factory=dict)
    progress_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PrototypingPipeline:
    """Pipeline for automated prototype development"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    stages: List[str] = field(default_factory=list)
    current_stage: str = ""
    stage_configurations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    automation_rules: List[Dict[str, Any]] = field(default_factory=list)
    quality_gates: List[Dict[str, Any]] = field(default_factory=list)
    notification_settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PrototypeTemplate:
    """Template for rapid prototype generation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    prototype_type: PrototypeType = PrototypeType.PROOF_OF_CONCEPT
    technology_stack: TechnologyStack = field(default_factory=TechnologyStack)
    code_templates: Dict[str, str] = field(default_factory=dict)
    configuration_templates: Dict[str, Any] = field(default_factory=dict)
    default_features: List[str] = field(default_factory=list)
    customization_options: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    success_rate: float = 0.0
    average_development_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PrototypeAnalytics:
    """Analytics data for prototype performance"""
    prototype_id: str = ""
    metrics_collected_at: datetime = field(default_factory=datetime.utcnow)
    
    # Development metrics
    development_velocity: float = 0.0
    code_quality_trend: List[float] = field(default_factory=list)
    feature_completion_rate: float = 0.0
    bug_discovery_rate: float = 0.0
    
    # Performance metrics
    response_times: List[float] = field(default_factory=list)
    throughput_metrics: Dict[str, float] = field(default_factory=dict)
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)
    
    # User interaction metrics
    user_engagement_score: float = 0.0
    feature_usage_stats: Dict[str, int] = field(default_factory=dict)
    user_satisfaction_score: float = 0.0
    conversion_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Business metrics
    business_value_score: float = 0.0
    roi_estimate: float = 0.0
    market_validation_score: float = 0.0
    competitive_advantage_score: float = 0.0


# Utility functions for working with prototype models

def create_concept_from_description(name: str, description: str, 
                                  category: ConceptCategory = ConceptCategory.TECHNOLOGY) -> Concept:
    """Create a concept from basic description"""
    return Concept(
        name=name,
        description=description,
        category=category,
        technical_complexity=0.5,
        innovation_potential=0.7,
        market_readiness=0.4
    )


def create_default_technology_stack(primary_tech: str) -> TechnologyStack:
    """Create a default technology stack for a given primary technology"""
    defaults = {
        "web_frontend": TechnologyStack(
            primary_technology="web_frontend",
            framework="React",
            language="TypeScript",
            supporting_tools=["Webpack", "Jest", "ESLint"],
            deployment_target="cdn"
        ),
        "api_service": TechnologyStack(
            primary_technology="api_service",
            framework="FastAPI",
            language="Python",
            supporting_tools=["Docker", "PostgreSQL", "Redis"],
            deployment_target="cloud"
        ),
        "ml_model": TechnologyStack(
            primary_technology="ml_model",
            framework="PyTorch",
            language="Python",
            supporting_tools=["Jupyter", "MLflow", "Docker"],
            deployment_target="ml_platform"
        )
    }
    
    return defaults.get(primary_tech, TechnologyStack(primary_technology=primary_tech))


def calculate_prototype_score(prototype: Prototype) -> float:
    """Calculate overall prototype score based on various metrics"""
    if not prototype.quality_metrics:
        return 0.0
    
    metrics = prototype.quality_metrics
    weights = {
        "code_coverage": 0.15,
        "performance_score": 0.20,
        "usability_score": 0.20,
        "reliability_score": 0.15,
        "security_score": 0.10,
        "maintainability_score": 0.10,
        "scalability_score": 0.10
    }
    
    total_score = 0.0
    for metric, weight in weights.items():
        value = getattr(metrics, metric, 0.0)
        total_score += value * weight
    
    return min(total_score, 1.0)


def is_prototype_production_ready(prototype: Prototype) -> bool:
    """Check if prototype is ready for production deployment"""
    if not prototype.validation_result:
        return False
    
    requirements = [
        prototype.validation_result.passes_validation,
        prototype.is_deployable,
        prototype.error_handling_implemented,
        prototype.security_implemented,
        prototype.status in [PrototypeStatus.VALIDATED, PrototypeStatus.OPTIMIZED]
    ]
    
    return all(requirements)