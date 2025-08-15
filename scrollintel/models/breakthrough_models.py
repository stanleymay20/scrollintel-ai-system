"""
Data models for breakthrough innovation engine
"""
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class TechnologyDomain(Enum):
    """Technology domains for breakthrough concepts"""
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    QUANTUM_COMPUTING = "quantum_computing"
    BIOTECHNOLOGY = "biotechnology"
    NANOTECHNOLOGY = "nanotechnology"
    ROBOTICS = "robotics"
    BLOCKCHAIN = "blockchain"
    AUGMENTED_REALITY = "augmented_reality"
    SPACE_TECHNOLOGY = "space_technology"
    ENERGY_STORAGE = "energy_storage"
    NEURAL_INTERFACES = "neural_interfaces"


class InnovationStage(Enum):
    """Stages of innovation development"""
    CONCEPT = "concept"
    RESEARCH = "research"
    PROTOTYPE = "prototype"
    DEVELOPMENT = "development"
    MARKET_READY = "market_ready"


class DisruptionLevel(Enum):
    """Levels of market disruption potential"""
    INCREMENTAL = "incremental"
    SIGNIFICANT = "significant"
    TRANSFORMATIVE = "transformative"
    REVOLUTIONARY = "revolutionary"


@dataclass
class Capability:
    """Required capability for breakthrough development"""
    name: str
    description: str
    current_level: float  # 0.0 to 1.0
    required_level: float  # 0.0 to 1.0
    development_time_months: int
    cost_estimate: float


@dataclass
class MarketOpportunity:
    """Market opportunity assessment"""
    market_size_billions: float
    growth_rate_percent: float
    time_to_market_years: int
    competitive_landscape: str
    barriers_to_entry: List[str]
    key_success_factors: List[str]


@dataclass
class TechnologyTrend:
    """Technology trend analysis"""
    trend_name: str
    domain: TechnologyDomain
    momentum_score: float  # 0.0 to 1.0
    patent_activity: int
    research_papers: int
    investment_millions: float
    key_players: List[str]
    predicted_breakthrough_timeline: int  # years


@dataclass
class InnovationPotential:
    """Assessment of innovation potential for a breakthrough concept"""
    concept_id: str
    novelty_score: float  # 0.0 to 1.0
    feasibility_score: float  # 0.0 to 1.0
    market_impact_score: float  # 0.0 to 1.0
    competitive_advantage_score: float  # 0.0 to 1.0
    risk_score: float  # 0.0 to 1.0
    overall_potential: float  # 0.0 to 1.0
    confidence_level: float  # 0.0 to 1.0
    
    # Detailed assessments
    technical_risks: List[str]
    market_risks: List[str]
    regulatory_risks: List[str]
    success_probability: float
    expected_roi: float
    
    # Timeline predictions
    research_phase_months: int
    development_phase_months: int
    market_entry_months: int
    
    created_at: datetime
    updated_at: datetime


@dataclass
class BreakthroughConcept:
    """Core breakthrough innovation concept"""
    id: str
    name: str
    description: str
    detailed_specification: str
    
    # Classification
    technology_domain: TechnologyDomain
    innovation_stage: InnovationStage
    disruption_level: DisruptionLevel
    
    # Core metrics
    innovation_potential: Optional[InnovationPotential]
    market_opportunity: MarketOpportunity
    required_capabilities: List[Capability]
    
    # Technology analysis
    underlying_technologies: List[str]
    breakthrough_mechanisms: List[str]
    scientific_principles: List[str]
    
    # Competitive analysis
    existing_solutions: List[str]
    competitive_advantages: List[str]
    differentiation_factors: List[str]
    
    # Development roadmap
    research_milestones: List[Dict[str, Any]]
    development_phases: List[Dict[str, Any]]
    success_metrics: List[Dict[str, Any]]
    
    # Metadata
    created_by: str
    created_at: datetime
    updated_at: datetime
    version: str
    tags: List[str]
    
    # AI-generated insights
    ai_confidence_score: float
    generated_hypotheses: List[str]
    recommended_experiments: List[str]
    potential_partnerships: List[str]


@dataclass
class DisruptionPrediction:
    """Prediction of market disruption from technology"""
    technology_name: str
    target_industry: str
    disruption_timeline_years: int
    disruption_probability: float
    
    # Impact assessment
    market_size_affected_billions: float
    jobs_displaced: int
    jobs_created: int
    productivity_gain_percent: float
    
    # Disruption mechanisms
    cost_reduction_percent: float
    performance_improvement_percent: float
    new_capabilities: List[str]
    obsoleted_technologies: List[str]
    
    # Strategic implications
    first_mover_advantages: List[str]
    defensive_strategies: List[str]
    investment_requirements_millions: float
    regulatory_challenges: List[str]
    
    created_at: datetime


@dataclass
class ResearchDirection:
    """Recommended research direction"""
    title: str
    description: str
    domain: TechnologyDomain
    priority_score: float
    
    # Research parameters
    hypothesis: str
    methodology: str
    required_resources: Dict[str, Any]
    expected_duration_months: int
    
    # Expected outcomes
    breakthrough_probability: float
    potential_applications: List[str]
    commercial_value: float
    scientific_impact: float
    
    # Dependencies
    prerequisite_research: List[str]
    required_collaborations: List[str]
    critical_resources: List[str]
    
    created_at: datetime