"""
Cultural Evolution Models

Data models for continuous cultural evolution and adaptation framework.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class EvolutionStage(Enum):
    EMERGING = "emerging"
    DEVELOPING = "developing"
    MATURING = "maturing"
    OPTIMIZING = "optimizing"
    TRANSFORMING = "transforming"


class AdaptabilityLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXCEPTIONAL = "exceptional"


class InnovationType(Enum):
    INCREMENTAL = "incremental"
    RADICAL = "radical"
    DISRUPTIVE = "disruptive"
    ARCHITECTURAL = "architectural"


@dataclass
class CulturalInnovation:
    """Individual cultural innovation or improvement"""
    innovation_id: str
    name: str
    description: str
    innovation_type: InnovationType
    target_areas: List[str]
    expected_impact: Dict[str, float]
    implementation_complexity: str  # "low", "medium", "high"
    resource_requirements: Dict[str, Any]
    success_metrics: List[str]
    status: str
    created_date: datetime


@dataclass
class EvolutionTrigger:
    """Trigger for cultural evolution"""
    trigger_id: str
    trigger_type: str  # "internal", "external", "strategic", "crisis"
    description: str
    urgency_level: str  # "low", "medium", "high", "critical"
    impact_areas: List[str]
    required_response: str
    detected_date: datetime


@dataclass
class AdaptationMechanism:
    """Mechanism for cultural adaptation"""
    mechanism_id: str
    name: str
    mechanism_type: str  # "feedback_loop", "learning_system", "innovation_process"
    description: str
    activation_conditions: List[str]
    adaptation_speed: str  # "slow", "moderate", "fast", "rapid"
    effectiveness_score: float
    resource_cost: str
    implementation_date: datetime


@dataclass
class CulturalEvolutionPlan:
    """Plan for continuous cultural evolution"""
    plan_id: str
    organization_id: str
    current_evolution_stage: EvolutionStage
    target_evolution_stage: EvolutionStage
    evolution_timeline: Dict[str, Any]
    cultural_innovations: List[CulturalInnovation]
    adaptation_mechanisms: List[AdaptationMechanism]
    evolution_triggers: List[EvolutionTrigger]
    success_criteria: List[str]
    monitoring_framework: Dict[str, Any]
    created_date: datetime
    last_updated: datetime


@dataclass
class ResilienceCapability:
    """Cultural resilience capability"""
    capability_id: str
    name: str
    capability_type: str  # "recovery", "adaptation", "transformation", "anticipation"
    description: str
    strength_level: float  # 0.0 to 1.0
    development_areas: List[str]
    supporting_mechanisms: List[str]
    effectiveness_metrics: List[str]
    last_assessed: datetime


@dataclass
class CulturalResilience:
    """Overall cultural resilience assessment"""
    resilience_id: str
    organization_id: str
    overall_resilience_score: float
    adaptability_level: AdaptabilityLevel
    resilience_capabilities: List[ResilienceCapability]
    vulnerability_areas: List[str]
    strength_areas: List[str]
    improvement_recommendations: List[str]
    assessment_date: datetime


@dataclass
class EvolutionOutcome:
    """Outcome of cultural evolution process"""
    outcome_id: str
    evolution_plan_id: str
    outcome_type: str  # "innovation_adoption", "adaptation_success", "resilience_improvement"
    description: str
    impact_metrics: Dict[str, float]
    success_indicators: List[str]
    lessons_learned: List[str]
    next_evolution_opportunities: List[str]
    outcome_date: datetime


@dataclass
class ContinuousImprovementCycle:
    """Continuous improvement cycle for culture"""
    cycle_id: str
    organization_id: str
    cycle_phase: str  # "assess", "plan", "implement", "evaluate", "adapt"
    current_focus_areas: List[str]
    improvement_initiatives: List[Dict[str, Any]]
    feedback_mechanisms: List[str]
    learning_outcomes: List[str]
    cycle_metrics: Dict[str, float]
    cycle_start_date: datetime
    next_cycle_date: datetime