"""
Innovation Lab Integration Data Models

This module defines the data models for the Innovation Lab Integration system,
including database schemas and data structures for system integrations.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

Base = declarative_base()


class IntegrationType(Enum):
    """Types of system integrations"""
    BREAKTHROUGH_INNOVATION = "breakthrough_innovation"
    QUANTUM_AI_RESEARCH = "quantum_ai_research"
    STRATEGIC_PLANNING = "strategic_planning"
    MARKET_INTELLIGENCE = "market_intelligence"
    COGNITIVE_ARCHITECTURE = "cognitive_architecture"


class SynergyLevel(Enum):
    """Levels of innovation synergy"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BREAKTHROUGH = "breakthrough"


class SystemIntegration(Base):
    """Database model for system integrations"""
    __tablename__ = "system_integrations"
    
    id = Column(String, primary_key=True)
    system_name = Column(String, nullable=False)
    integration_type = Column(String, nullable=False)
    api_endpoint = Column(String, nullable=False)
    data_format = Column(String, default="json")
    authentication_method = Column(String, default="api_key")
    rate_limits = Column(JSON)
    capabilities = Column(JSON)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    cross_pollinations = relationship("CrossPollination", back_populates="integration")
    quantum_enhancements = relationship("QuantumEnhancement", back_populates="integration")


class CrossPollination(Base):
    """Database model for innovation cross-pollination"""
    __tablename__ = "cross_pollinations"
    
    id = Column(String, primary_key=True)
    integration_id = Column(String, ForeignKey("system_integrations.id"))
    source_system = Column(String, nullable=False)
    target_system = Column(String, nullable=False)
    innovation_concept = Column(Text, nullable=False)
    synergy_level = Column(String, nullable=False)
    enhancement_potential = Column(Float, nullable=False)
    implementation_complexity = Column(Float, nullable=False)
    expected_impact = Column(Float, nullable=False)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Relationships
    integration = relationship("SystemIntegration", back_populates="cross_pollinations")
    synergies = relationship("InnovationSynergy", back_populates="cross_pollination")


class InnovationSynergy(Base):
    """Database model for innovation synergies"""
    __tablename__ = "innovation_synergies"
    
    id = Column(String, primary_key=True)
    cross_pollination_id = Column(String, ForeignKey("cross_pollinations.id"))
    innovation_ids = Column(JSON, nullable=False)
    synergy_type = Column(String, nullable=False)
    synergy_description = Column(Text, nullable=False)
    combined_potential = Column(Float, nullable=False)
    exploitation_strategy = Column(Text, nullable=False)
    resource_requirements = Column(JSON)
    timeline = Column(JSON)
    status = Column(String, default="identified")
    created_at = Column(DateTime, default=datetime.utcnow)
    exploited_at = Column(DateTime)
    
    # Relationships
    cross_pollination = relationship("CrossPollination", back_populates="synergies")


class QuantumEnhancement(Base):
    """Database model for quantum AI enhancements"""
    __tablename__ = "quantum_enhancements"
    
    id = Column(String, primary_key=True)
    integration_id = Column(String, ForeignKey("system_integrations.id"))
    innovation_id = Column(String, nullable=False)
    enhancement_score = Column(Float, nullable=False)
    quantum_algorithms = Column(JSON)
    expected_speedup = Column(Float, nullable=False)
    implementation_complexity = Column(Float, nullable=False)
    quantum_advantage = Column(Boolean, default=False)
    acceleration_factor = Column(Float, default=1.0)
    quantum_solutions = Column(JSON)
    status = Column(String, default="enhanced")
    created_at = Column(DateTime, default=datetime.utcnow)
    deployed_at = Column(DateTime)
    
    # Relationships
    integration = relationship("SystemIntegration", back_populates="quantum_enhancements")


class IntegrationMetrics(Base):
    """Database model for integration performance metrics"""
    __tablename__ = "integration_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    integration_id = Column(String, ForeignKey("system_integrations.id"))
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String)
    measurement_time = Column(DateTime, default=datetime.utcnow)
    context = Column(JSON)


class IntegrationEvent(Base):
    """Database model for integration events and logs"""
    __tablename__ = "integration_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    integration_id = Column(String, ForeignKey("system_integrations.id"))
    event_type = Column(String, nullable=False)
    event_description = Column(Text, nullable=False)
    event_data = Column(JSON)
    severity = Column(String, default="info")
    timestamp = Column(DateTime, default=datetime.utcnow)


# Data classes for in-memory operations
@dataclass
class InnovationProject:
    """Data class for innovation projects"""
    id: str
    title: str
    domain: str
    research_topic: Any
    hypotheses: List[Any]
    experiment_plans: List[Any]
    status: str
    created_at: datetime
    successful_experiments: List[Any] = None
    prototypes: List[Any] = None
    validated_innovations: List[Any] = None
    
    def __post_init__(self):
        if self.successful_experiments is None:
            self.successful_experiments = []
        if self.prototypes is None:
            self.prototypes = []
        if self.validated_innovations is None:
            self.validated_innovations = []


class LabStatus(Enum):
    """Lab status enumeration"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class InnovationMetrics:
    """Data class for innovation metrics"""
    total_innovations: int = 0
    active_projects: int = 0
    completed_projects: int = 0
    success_rate: float = 0.0


class ResearchDomain(Enum):
    """Research domain enumeration"""
    AI = "artificial_intelligence"
    QUANTUM = "quantum_computing"
    BIOTECH = "biotechnology"
    NANOTECH = "nanotechnology"
    ENERGY = "renewable_energy"
    SPACE = "space_technology"


@dataclass
class LabStatusResponse:
    """Response model for lab status"""
    status: str
    is_running: bool
    active_projects: int
    metrics: Dict[str, Any]
    last_validation: Optional[str] = None
    research_domains: List[str] = None
    message: Optional[str] = None


@dataclass
class ValidationRequest:
    """Request model for validation"""
    domain: Optional[str] = None


@dataclass
class ValidationResponse:
    """Response model for validation"""
    overall_success: bool
    domain_results: Dict[str, Any]
    validation_timestamp: str
    lab_status: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class InnovationProjectResponse:
    """Response model for innovation project"""
    id: str
    title: str
    domain: str
    status: str
    created_at: str
    experiment_count: int
    prototype_count: int
    innovation_count: int


@dataclass
class LabMetricsResponse:
    """Response model for lab metrics"""
    total_innovations: int
    active_projects: int
    completed_projects: int
    total_projects: int
    success_rate: float
    research_domains: List[str]
    last_updated: Optional[str] = None


@dataclass
class InnovationCrossPollinationData:
    """Data class for innovation cross-pollination"""
    id: str
    source_system: str
    target_system: str
    innovation_concept: str
    synergy_level: SynergyLevel
    enhancement_potential: float
    implementation_complexity: float
    expected_impact: float
    created_at: datetime


@dataclass
class SystemIntegrationPointData:
    """Data class for system integration points"""
    id: str
    system_name: str
    integration_type: IntegrationType
    api_endpoint: str
    data_format: str
    authentication_method: str
    rate_limits: Dict[str, int]
    capabilities: List[str]
    status: str = "active"


@dataclass
class InnovationSynergyData:
    """Data class for innovation synergies"""
    id: str
    innovation_ids: List[str]
    synergy_type: str
    synergy_description: str
    combined_potential: float
    exploitation_strategy: str
    resource_requirements: Dict[str, Any]
    timeline: Dict[str, datetime]


@dataclass
class QuantumEnhancementData:
    """Data class for quantum enhancements"""
    id: str
    innovation_id: str
    enhancement_score: float
    quantum_algorithms: List[str]
    expected_speedup: float
    implementation_complexity: float
    quantum_advantage: bool = False
    acceleration_factor: float = 1.0
    quantum_solutions: List[str] = None


@dataclass
class IntegrationStatusData:
    """Data class for integration status"""
    integration_status: Dict[str, str]
    breakthrough_integrations: int
    quantum_integrations: int
    active_cross_pollinations: int
    quantum_enhanced_innovations: int
    total_synergies_identified: int
    total_enhancements_applied: int


@dataclass
class InnovationData:
    """Data class for innovation information"""
    id: str
    name: str
    description: str
    domain: str
    complexity: float
    novelty: float
    potential: float
    domains: List[str]
    capabilities: List[str]
    computational_bottlenecks: List[str]
    cross_pollinated: bool = False
    quantum_enhanced: bool = False
    quantum_accelerated: bool = False


class IntegrationConfig:
    """Configuration class for system integrations"""
    
    def __init__(self):
        self.breakthrough_innovation_config = {
            "enabled": False,
            "api_endpoint": "/api/v1/breakthrough",
            "authentication": "api_key",
            "rate_limits": {
                "requests_per_minute": 100,
                "concurrent_requests": 10
            },
            "capabilities": [
                "creative_intelligence",
                "cross_domain_synthesis",
                "innovation_opportunity_detection",
                "breakthrough_validation",
                "innovation_acceleration"
            ]
        }
        
        self.quantum_ai_config = {
            "enabled": False,
            "api_endpoint": "/api/v1/quantum",
            "authentication": "quantum_key",
            "rate_limits": {
                "requests_per_minute": 50,
                "concurrent_requests": 5
            },
            "capabilities": [
                "quantum_algorithm_development",
                "quantum_machine_learning",
                "quantum_optimization",
                "quantum_classical_integration",
                "quantum_advantage_validation"
            ]
        }
    
    def get_breakthrough_config(self) -> Dict[str, Any]:
        """Get breakthrough innovation configuration"""
        return self.breakthrough_innovation_config.copy()
    
    def get_quantum_config(self) -> Dict[str, Any]:
        """Get quantum AI configuration"""
        return self.quantum_ai_config.copy()
    
    def update_breakthrough_config(self, config: Dict[str, Any]):
        """Update breakthrough innovation configuration"""
        self.breakthrough_innovation_config.update(config)
    
    def update_quantum_config(self, config: Dict[str, Any]):
        """Update quantum AI configuration"""
        self.quantum_ai_config.update(config)


class IntegrationMetricsCalculator:
    """Calculator for integration performance metrics"""
    
    @staticmethod
    def calculate_cross_pollination_effectiveness(cross_pollinations: List[InnovationCrossPollinationData]) -> float:
        """Calculate effectiveness of cross-pollination"""
        if not cross_pollinations:
            return 0.0
        
        total_impact = sum(cp.expected_impact for cp in cross_pollinations)
        return total_impact / len(cross_pollinations)
    
    @staticmethod
    def calculate_quantum_enhancement_benefit(enhancements: List[QuantumEnhancementData]) -> float:
        """Calculate benefit of quantum enhancements"""
        if not enhancements:
            return 0.0
        
        total_speedup = sum(qe.expected_speedup for qe in enhancements)
        return total_speedup / len(enhancements)
    
    @staticmethod
    def calculate_synergy_exploitation_rate(synergies: List[InnovationSynergyData]) -> float:
        """Calculate rate of synergy exploitation"""
        if not synergies:
            return 0.0
        
        high_potential_synergies = [s for s in synergies if s.combined_potential > 0.7]
        return len(high_potential_synergies) / len(synergies)
    
    @staticmethod
    def calculate_integration_roi(
        cross_pollinations: List[InnovationCrossPollinationData],
        quantum_enhancements: List[QuantumEnhancementData],
        resource_investment: float
    ) -> float:
        """Calculate return on investment for integrations"""
        if resource_investment <= 0:
            return 0.0
        
        # Calculate total value from cross-pollinations
        cross_pollination_value = sum(
            cp.expected_impact * cp.enhancement_potential 
            for cp in cross_pollinations
        )
        
        # Calculate total value from quantum enhancements
        quantum_value = sum(
            qe.expected_speedup * qe.enhancement_score 
            for qe in quantum_enhancements
        )
        
        total_value = cross_pollination_value + quantum_value
        return total_value / resource_investment


# Utility functions for data conversion
def cross_pollination_to_data(cp: CrossPollination) -> InnovationCrossPollinationData:
    """Convert database model to data class"""
    return InnovationCrossPollinationData(
        id=cp.id,
        source_system=cp.source_system,
        target_system=cp.target_system,
        innovation_concept=cp.innovation_concept,
        synergy_level=SynergyLevel(cp.synergy_level),
        enhancement_potential=cp.enhancement_potential,
        implementation_complexity=cp.implementation_complexity,
        expected_impact=cp.expected_impact,
        created_at=cp.created_at
    )


def quantum_enhancement_to_data(qe: QuantumEnhancement) -> QuantumEnhancementData:
    """Convert database model to data class"""
    return QuantumEnhancementData(
        id=qe.id,
        innovation_id=qe.innovation_id,
        enhancement_score=qe.enhancement_score,
        quantum_algorithms=qe.quantum_algorithms or [],
        expected_speedup=qe.expected_speedup,
        implementation_complexity=qe.implementation_complexity,
        quantum_advantage=qe.quantum_advantage,
        acceleration_factor=qe.acceleration_factor,
        quantum_solutions=qe.quantum_solutions or []
    )


def integration_to_data(integration: SystemIntegration) -> SystemIntegrationPointData:
    """Convert database model to data class"""
    return SystemIntegrationPointData(
        id=integration.id,
        system_name=integration.system_name,
        integration_type=IntegrationType(integration.integration_type),
        api_endpoint=integration.api_endpoint,
        data_format=integration.data_format,
        authentication_method=integration.authentication_method,
        rate_limits=integration.rate_limits or {},
        capabilities=integration.capabilities or [],
        status=integration.status
    )