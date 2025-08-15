"""
Models for Breakthrough Innovation Integration System
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime

class InnovationType(Enum):
    PARADIGM_SHIFT = "paradigm_shift"
    DISRUPTIVE_TECHNOLOGY = "disruptive_technology"
    MARKET_CREATION = "market_creation"
    CROSS_DOMAIN_FUSION = "cross_domain_fusion"
    EMERGENT_SOLUTION = "emergent_solution"

class BreakthroughPotential(Enum):
    REVOLUTIONARY = "revolutionary"
    TRANSFORMATIVE = "transformative"
    DISRUPTIVE = "disruptive"
    INNOVATIVE = "innovative"
    INCREMENTAL = "incremental"

class SynergyType(Enum):
    RESEARCH_ACCELERATION = "research_acceleration"
    INNOVATION_AMPLIFICATION = "innovation_amplification"
    CROSS_POLLINATION = "cross_pollination"
    VALIDATION_ENHANCEMENT = "validation_enhancement"
    IMPLEMENTATION_OPTIMIZATION = "implementation_optimization"

@dataclass
class BreakthroughInnovation:
    id: str
    title: str
    description: str
    innovation_type: InnovationType
    breakthrough_potential: BreakthroughPotential
    domains_involved: List[str]
    key_concepts: List[str]
    feasibility_score: float
    impact_score: float
    novelty_score: float
    implementation_complexity: float
    resource_requirements: Dict[str, Any]
    timeline_estimate: int  # days
    success_probability: float
    created_at: datetime
    updated_at: datetime

@dataclass
class InnovationSynergy:
    id: str
    innovation_lab_component: str
    breakthrough_innovation_id: str
    synergy_type: SynergyType
    synergy_strength: float
    enhancement_potential: float
    implementation_effort: float
    expected_benefits: List[str]
    integration_requirements: List[str]
    validation_metrics: Dict[str, float]
    created_at: datetime

@dataclass
class CrossPollinationOpportunity:
    id: str
    source_innovation: str
    target_research_area: str
    pollination_type: str
    enhancement_potential: float
    feasibility_score: float
    expected_outcomes: List[str]
    integration_pathway: List[str]
    resource_requirements: Dict[str, Any]
    timeline_estimate: int
    success_indicators: List[str]

@dataclass
class InnovationAccelerationPlan:
    id: str
    target_innovation: str
    acceleration_strategies: List[str]
    resource_optimization: Dict[str, Any]
    timeline_compression: float
    risk_mitigation: List[str]
    success_metrics: Dict[str, float]
    implementation_steps: List[str]
    monitoring_framework: Dict[str, Any]

@dataclass
class BreakthroughValidationResult:
    innovation_id: str
    validation_score: float
    feasibility_assessment: Dict[str, float]
    impact_prediction: Dict[str, Any]
    risk_analysis: Dict[str, float]
    implementation_pathway: List[str]
    resource_requirements: Dict[str, Any]
    success_probability: float
    recommendations: List[str]
    validation_timestamp: datetime

@dataclass
class IntegrationMetrics:
    total_synergies_identified: int
    active_cross_pollinations: int
    acceleration_projects: int
    breakthrough_validations: int
    average_enhancement_potential: float
    integration_success_rate: float
    innovation_velocity_improvement: float
    resource_efficiency_gain: float
    last_updated: datetime