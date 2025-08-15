"""
Strategic Planning Models for Big Tech CTO Capabilities

This module defines data models for 10+ year strategic planning, technology roadmaps,
and long-term technology investment optimization.
"""

from datetime import datetime, date
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field


class TechnologyDomain(str, Enum):
    """Technology domains for strategic planning"""
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    QUANTUM_COMPUTING = "quantum_computing"
    BIOTECHNOLOGY = "biotechnology"
    ROBOTICS = "robotics"
    BLOCKCHAIN = "blockchain"
    AUGMENTED_REALITY = "augmented_reality"
    INTERNET_OF_THINGS = "internet_of_things"
    EDGE_COMPUTING = "edge_computing"
    CYBERSECURITY = "cybersecurity"
    RENEWABLE_ENERGY = "renewable_energy"


class InvestmentRisk(str, Enum):
    """Risk levels for technology investments"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class MarketImpact(str, Enum):
    """Potential market impact levels"""
    INCREMENTAL = "incremental"
    SIGNIFICANT = "significant"
    TRANSFORMATIVE = "transformative"
    REVOLUTIONARY = "revolutionary"


class CompetitivePosition(str, Enum):
    """Competitive positioning strategies"""
    LEADER = "leader"
    FAST_FOLLOWER = "fast_follower"
    DISRUPTOR = "disruptor"
    NICHE_PLAYER = "niche_player"


@dataclass
class TechnologyBet:
    """Represents a long-term technology investment bet"""
    id: str
    name: str
    description: str
    domain: TechnologyDomain
    investment_amount: float
    time_horizon: int  # years
    risk_level: InvestmentRisk
    expected_roi: float
    market_impact: MarketImpact
    competitive_advantage: float
    technical_feasibility: float
    market_readiness: float
    regulatory_risk: float
    talent_requirements: Dict[str, int]
    key_milestones: List[Dict[str, Any]]
    success_metrics: List[str]
    dependencies: List[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class StrategicMilestone:
    """Represents a milestone in the strategic roadmap"""
    id: str
    name: str
    description: str
    target_date: date
    completion_criteria: List[str]
    success_metrics: List[str]
    dependencies: List[str]
    risk_factors: List[str]
    resource_requirements: Dict[str, Any]
    status: str = "planned"


@dataclass
class RiskAssessment:
    """Risk assessment for strategic initiatives"""
    id: str
    risk_type: str
    description: str
    probability: float  # 0-1
    impact: float  # 0-1
    mitigation_strategies: List[str]
    contingency_plans: List[str]
    monitoring_indicators: List[str]


@dataclass
class SuccessMetric:
    """Success metrics for strategic initiatives"""
    id: str
    name: str
    description: str
    target_value: float
    current_value: float
    measurement_unit: str
    measurement_frequency: str
    data_source: str


@dataclass
class TechnologyVision:
    """Long-term technology vision"""
    id: str
    title: str
    description: str
    time_horizon: int
    key_principles: List[str]
    strategic_objectives: List[str]
    success_criteria: List[str]
    market_assumptions: List[str]


@dataclass
class StrategicRoadmap:
    """Comprehensive strategic roadmap for 10+ year planning"""
    id: str
    name: str
    description: str
    vision: TechnologyVision
    time_horizon: int
    milestones: List[StrategicMilestone]
    technology_bets: List[TechnologyBet]
    risk_assessments: List[RiskAssessment]
    success_metrics: List[SuccessMetric]
    competitive_positioning: CompetitivePosition
    market_assumptions: List[str]
    resource_allocation: Dict[str, float]
    scenario_plans: List[Dict[str, Any]]
    review_schedule: List[date]
    stakeholders: List[str]
    created_at: datetime
    updated_at: datetime


class DisruptionPrediction(BaseModel):
    """Prediction of industry disruption"""
    industry: str
    disruption_type: str
    probability: float = Field(ge=0, le=1)
    time_horizon: int
    impact_magnitude: float = Field(ge=0, le=10)
    key_drivers: List[str]
    affected_sectors: List[str]
    opportunities: List[str]
    threats: List[str]
    recommended_actions: List[str]


class IndustryForecast(BaseModel):
    """Long-term industry evolution forecast"""
    industry: str
    time_horizon: int
    growth_projections: Dict[str, float]
    technology_trends: List[str]
    market_dynamics: Dict[str, Any]
    competitive_landscape: Dict[str, Any]
    regulatory_changes: List[str]
    disruption_risks: List[DisruptionPrediction]
    investment_opportunities: List[str]


class StrategicPivot(BaseModel):
    """Strategic pivot recommendation"""
    id: str
    name: str
    description: str
    trigger_conditions: List[str]
    implementation_timeline: int
    resource_requirements: Dict[str, Any]
    expected_outcomes: List[str]
    risk_factors: List[str]
    success_probability: float = Field(ge=0, le=1)
    roi_projection: float


class InvestmentAnalysis(BaseModel):
    """Analysis of technology investment portfolio"""
    total_investment: float
    portfolio_risk: float
    expected_return: float
    diversification_score: float
    technology_coverage: Dict[TechnologyDomain, float]
    time_horizon_distribution: Dict[str, float]
    risk_return_profile: Dict[str, float]
    recommendations: List[str]
    optimization_opportunities: List[str]


class CompetitiveIntelligence(BaseModel):
    """Competitive intelligence analysis"""
    competitor_name: str
    market_position: str
    technology_capabilities: Dict[str, float]
    investment_patterns: Dict[str, float]
    strategic_moves: List[str]
    strengths: List[str]
    weaknesses: List[str]
    threats: List[str]
    opportunities: List[str]
    predicted_actions: List[str]
    counter_strategies: List[str]


class MarketChange(BaseModel):
    """Market change event"""
    change_type: str
    description: str
    impact_magnitude: float
    affected_markets: List[str]
    time_horizon: int
    probability: float = Field(ge=0, le=1)
    strategic_implications: List[str]