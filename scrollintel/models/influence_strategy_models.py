"""
Influence Strategy Models for Board Executive Mastery System
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class InfluenceType(Enum):
    """Types of influence strategies"""
    RATIONAL_PERSUASION = "rational_persuasion"
    INSPIRATIONAL_APPEALS = "inspirational_appeals"
    CONSULTATION = "consultation"
    COALITION_TACTICS = "coalition_tactics"
    PERSONAL_APPEALS = "personal_appeals"
    EXCHANGE_TACTICS = "exchange_tactics"
    LEGITIMATING_TACTICS = "legitimating_tactics"
    PRESSURE_TACTICS = "pressure_tactics"
    INGRATIATION = "ingratiation"


class InfluenceContext(Enum):
    """Context for influence strategies"""
    BOARD_MEETING = "board_meeting"
    ONE_ON_ONE = "one_on_one"
    COMMITTEE_MEETING = "committee_meeting"
    INFORMAL_SETTING = "informal_setting"
    CRISIS_SITUATION = "crisis_situation"
    STRATEGIC_PLANNING = "strategic_planning"


class InfluenceObjective(Enum):
    """Objectives for influence strategies"""
    BUILD_SUPPORT = "build_support"
    GAIN_CONSENSUS = "gain_consensus"
    CHANGE_OPINION = "change_opinion"
    SECURE_APPROVAL = "secure_approval"
    PREVENT_OPPOSITION = "prevent_opposition"
    BUILD_COALITION = "build_coalition"


@dataclass
class InfluenceTactic:
    """Individual influence tactic"""
    id: str
    name: str
    influence_type: InfluenceType
    description: str
    effectiveness_score: float
    context_suitability: List[InfluenceContext]
    target_personality_types: List[str]
    required_preparation: List[str]
    success_indicators: List[str]
    risk_factors: List[str]
    
    def __post_init__(self):
        if not 0 <= self.effectiveness_score <= 1:
            raise ValueError("Effectiveness score must be between 0 and 1")


@dataclass
class InfluenceTarget:
    """Target for influence strategy"""
    stakeholder_id: str
    name: str
    role: str
    influence_level: float
    decision_making_style: str
    communication_preferences: List[str]
    key_motivators: List[str]
    concerns: List[str]
    relationship_strength: float
    historical_responses: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfluenceStrategy:
    """Complete influence strategy"""
    id: str
    name: str
    objective: InfluenceObjective
    target_stakeholders: List[InfluenceTarget]
    primary_tactics: List[InfluenceTactic]
    secondary_tactics: List[InfluenceTactic]
    context: InfluenceContext
    timeline: Dict[str, datetime]
    success_metrics: List[str]
    risk_mitigation: Dict[str, str]
    resource_requirements: List[str]
    expected_effectiveness: float
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class InfluenceExecution:
    """Execution of influence strategy"""
    id: str
    strategy_id: str
    execution_date: datetime
    context_details: Dict[str, Any]
    tactics_used: List[str]
    target_responses: Dict[str, str]
    immediate_outcomes: List[str]
    effectiveness_rating: float
    lessons_learned: List[str]
    follow_up_actions: List[str]


@dataclass
class InfluenceEffectivenessMetrics:
    """Metrics for measuring influence effectiveness"""
    strategy_id: str
    execution_id: str
    objective_achievement: float
    stakeholder_satisfaction: Dict[str, float]
    relationship_impact: Dict[str, float]
    consensus_level: float
    support_gained: float
    opposition_reduced: float
    long_term_relationship_health: float
    measured_at: datetime = field(default_factory=datetime.now)


@dataclass
class InfluenceOptimization:
    """Optimization recommendations for influence strategies"""
    strategy_id: str
    current_effectiveness: float
    optimization_opportunities: List[str]
    recommended_tactic_changes: List[Dict[str, Any]]
    timing_adjustments: List[str]
    context_modifications: List[str]
    target_approach_refinements: Dict[str, List[str]]
    expected_improvement: float
    confidence_level: float