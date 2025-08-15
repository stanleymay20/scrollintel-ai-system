"""
Decision Analysis Models for Board Executive Mastery System

This module defines data models for executive decision analysis and recommendation systems.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class DecisionType(Enum):
    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    TECHNOLOGY = "technology"
    PERSONNEL = "personnel"
    RISK_MANAGEMENT = "risk_management"


class DecisionUrgency(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


class ImpactLevel(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DecisionCriteria:
    """Criteria for evaluating decisions"""
    id: str
    name: str
    weight: float  # 0.0 to 1.0
    description: str
    measurement_method: str


@dataclass
class DecisionOption:
    """Individual decision option with analysis"""
    id: str
    title: str
    description: str
    pros: List[str]
    cons: List[str]
    estimated_cost: Optional[float]
    estimated_timeline: Optional[str]
    risk_level: ImpactLevel
    expected_outcome: str
    success_probability: float  # 0.0 to 1.0
    criteria_scores: Dict[str, float]  # criteria_id -> score


@dataclass
class StakeholderImpact:
    """Impact analysis on stakeholders"""
    stakeholder_id: str
    stakeholder_name: str
    impact_level: ImpactLevel
    impact_description: str
    support_likelihood: float  # 0.0 to 1.0
    concerns: List[str]
    mitigation_strategies: List[str]


@dataclass
class RiskAssessment:
    """Risk assessment for decision options"""
    id: str
    risk_category: str
    probability: float  # 0.0 to 1.0
    impact: ImpactLevel
    description: str
    mitigation_strategies: List[str]
    contingency_plans: List[str]


@dataclass
class DecisionAnalysis:
    """Comprehensive decision analysis"""
    id: str
    title: str
    description: str
    decision_type: DecisionType
    urgency: DecisionUrgency
    complexity: DecisionComplexity
    created_at: datetime
    deadline: Optional[datetime]
    
    # Context
    background: str
    current_situation: str
    decision_drivers: List[str]
    constraints: List[str]
    
    # Analysis components
    criteria: List[DecisionCriteria]
    options: List[DecisionOption]
    stakeholder_impacts: List[StakeholderImpact]
    risk_assessments: List[RiskAssessment]
    
    # Recommendations
    recommended_option_id: Optional[str]
    recommendation_rationale: str
    implementation_plan: List[str]
    success_metrics: List[str]
    
    # Metadata
    analyst_id: str
    confidence_level: float  # 0.0 to 1.0
    analysis_quality_score: float  # 0.0 to 1.0


@dataclass
class DecisionVisualization:
    """Visualization configuration for decision analysis"""
    id: str
    decision_analysis_id: str
    visualization_type: str  # "comparison_matrix", "risk_impact", "stakeholder_map", etc.
    title: str
    description: str
    chart_config: Dict[str, Any]
    executive_summary: str


@dataclass
class DecisionRecommendation:
    """Executive decision recommendation"""
    id: str
    decision_analysis_id: str
    title: str
    executive_summary: str
    recommended_action: str
    key_benefits: List[str]
    critical_risks: List[str]
    resource_requirements: Dict[str, Any]
    timeline: str
    success_probability: float
    confidence_level: float
    next_steps: List[str]
    approval_requirements: List[str]


@dataclass
class DecisionImpactAssessment:
    """Assessment of decision impact across dimensions"""
    id: str
    decision_analysis_id: str
    
    # Financial impact
    financial_impact: Dict[str, float]  # "cost", "revenue", "savings", etc.
    roi_projection: Optional[float]
    payback_period: Optional[str]
    
    # Strategic impact
    strategic_alignment: float  # 0.0 to 1.0
    competitive_advantage: float  # 0.0 to 1.0
    market_position_impact: str
    
    # Operational impact
    operational_complexity: DecisionComplexity
    resource_requirements: Dict[str, Any]
    implementation_difficulty: float  # 0.0 to 1.0
    
    # Risk impact
    overall_risk_level: ImpactLevel
    risk_mitigation_cost: Optional[float]
    regulatory_compliance_impact: str
    
    # Stakeholder impact
    board_support_likelihood: float  # 0.0 to 1.0
    employee_impact: str
    customer_impact: str
    investor_impact: str


@dataclass
class DecisionOptimization:
    """Optimization recommendations for decision options"""
    id: str
    decision_analysis_id: str
    option_id: str
    
    optimization_suggestions: List[str]
    enhanced_benefits: List[str]
    risk_reductions: List[str]
    cost_optimizations: List[str]
    timeline_improvements: List[str]
    
    optimized_success_probability: float
    optimization_confidence: float
    implementation_complexity: DecisionComplexity