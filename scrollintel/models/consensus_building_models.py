"""
Consensus Building Models for Board Executive Mastery System

This module defines data models for board consensus building strategy development,
tracking, and facilitation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class ConsensusStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PARTIAL_CONSENSUS = "partial_consensus"
    STRONG_CONSENSUS = "strong_consensus"
    UNANIMOUS = "unanimous"
    BLOCKED = "blocked"


class StakeholderPosition(Enum):
    STRONGLY_SUPPORT = "strongly_support"
    SUPPORT = "support"
    NEUTRAL = "neutral"
    OPPOSE = "oppose"
    STRONGLY_OPPOSE = "strongly_oppose"
    UNDECIDED = "undecided"


class InfluenceLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ConsensusStrategyType(Enum):
    DIRECT_PERSUASION = "direct_persuasion"
    COALITION_BUILDING = "coalition_building"
    COMPROMISE_SEEKING = "compromise_seeking"
    INFORMATION_SHARING = "information_sharing"
    STAKEHOLDER_ENGAGEMENT = "stakeholder_engagement"
    INCREMENTAL_AGREEMENT = "incremental_agreement"


@dataclass
class BoardMemberProfile:
    """Profile of board member for consensus building"""
    id: str
    name: str
    role: str
    influence_level: InfluenceLevel
    decision_making_style: str  # "analytical", "intuitive", "collaborative", etc.
    key_concerns: List[str]
    motivations: List[str]
    communication_preferences: List[str]
    historical_positions: Dict[str, str]  # topic -> typical position
    relationship_network: List[str]  # IDs of closely aligned members


@dataclass
class ConsensusPosition:
    """Individual stakeholder position on a decision"""
    stakeholder_id: str
    stakeholder_name: str
    current_position: StakeholderPosition
    confidence_level: float  # 0.0 to 1.0 - how confident they are in their position
    key_concerns: List[str]
    requirements_for_support: List[str]
    deal_breakers: List[str]
    influence_on_others: List[str]  # IDs of members they can influence
    last_updated: datetime


@dataclass
class ConsensusBarrier:
    """Barrier to achieving consensus"""
    id: str
    description: str
    barrier_type: str  # "information", "trust", "interests", "values", "process"
    affected_stakeholders: List[str]
    severity: float  # 0.0 to 1.0
    mitigation_strategies: List[str]
    estimated_resolution_time: Optional[str]


@dataclass
class ConsensusStrategy:
    """Strategy for building consensus"""
    id: str
    strategy_type: ConsensusStrategyType
    target_stakeholders: List[str]
    description: str
    tactics: List[str]
    expected_outcomes: List[str]
    success_probability: float  # 0.0 to 1.0
    estimated_timeline: str
    resource_requirements: List[str]
    risks: List[str]


@dataclass
class ConsensusAction:
    """Specific action to build consensus"""
    id: str
    title: str
    description: str
    action_type: str  # "meeting", "presentation", "negotiation", "information_sharing"
    target_stakeholders: List[str]
    responsible_party: str
    deadline: datetime
    status: str  # "planned", "in_progress", "completed", "cancelled"
    expected_impact: str
    actual_impact: Optional[str]
    follow_up_required: bool


@dataclass
class ConsensusBuilding:
    """Main consensus building framework"""
    id: str
    title: str
    description: str
    decision_topic: str
    created_at: datetime
    target_consensus_level: ConsensusStatus
    current_consensus_level: ConsensusStatus
    deadline: Optional[datetime]
    
    # Stakeholder analysis
    board_members: List[BoardMemberProfile]
    stakeholder_positions: List[ConsensusPosition]
    
    # Consensus analysis
    barriers: List[ConsensusBarrier]
    strategies: List[ConsensusStrategy]
    actions: List[ConsensusAction]
    
    # Progress tracking
    consensus_score: float  # 0.0 to 1.0
    momentum: float  # -1.0 to 1.0 (negative = losing consensus, positive = gaining)
    key_influencers: List[str]  # Stakeholder IDs with highest influence
    coalition_map: Dict[str, List[str]]  # Position -> list of stakeholder IDs
    
    # Metadata
    facilitator_id: str
    success_probability: float
    last_updated: datetime


@dataclass
class ConsensusMetrics:
    """Metrics for tracking consensus building progress"""
    id: str
    consensus_building_id: str
    measurement_date: datetime
    
    # Quantitative metrics
    support_percentage: float
    opposition_percentage: float
    neutral_percentage: float
    weighted_support_score: float  # Weighted by influence level
    
    # Qualitative assessments
    momentum_direction: str  # "positive", "negative", "stable"
    key_concerns_addressed: int
    barriers_resolved: int
    new_barriers_identified: int
    
    # Engagement metrics
    stakeholder_engagement_level: float  # 0.0 to 1.0
    communication_effectiveness: float  # 0.0 to 1.0
    trust_level: float  # 0.0 to 1.0


@dataclass
class ConsensusRecommendation:
    """Recommendation for achieving consensus"""
    id: str
    consensus_building_id: str
    title: str
    description: str
    
    # Strategic recommendations
    recommended_approach: ConsensusStrategyType
    priority_actions: List[str]
    key_stakeholders_to_focus: List[str]
    timeline_recommendation: str
    
    # Tactical recommendations
    communication_strategy: str
    meeting_recommendations: List[str]
    negotiation_points: List[str]
    compromise_options: List[str]
    
    # Risk mitigation
    potential_risks: List[str]
    mitigation_strategies: List[str]
    contingency_plans: List[str]
    
    # Success factors
    success_probability: float
    critical_success_factors: List[str]
    early_warning_indicators: List[str]


@dataclass
class ConsensusOptimization:
    """Optimization recommendations for consensus building"""
    id: str
    consensus_building_id: str
    
    # Process optimizations
    process_improvements: List[str]
    communication_enhancements: List[str]
    engagement_strategies: List[str]
    
    # Stakeholder-specific optimizations
    stakeholder_specific_approaches: Dict[str, List[str]]  # stakeholder_id -> approaches
    coalition_building_opportunities: List[str]
    influence_leverage_points: List[str]
    
    # Timeline optimizations
    accelerated_timeline_options: List[str]
    parallel_workstream_opportunities: List[str]
    quick_wins: List[str]
    
    # Quality improvements
    decision_quality_enhancements: List[str]
    information_gaps_to_address: List[str]
    expertise_to_bring_in: List[str]


@dataclass
class ConsensusVisualization:
    """Visualization configuration for consensus building"""
    id: str
    consensus_building_id: str
    visualization_type: str  # "stakeholder_map", "consensus_timeline", "influence_network"
    title: str
    description: str
    chart_config: Dict[str, Any]
    executive_summary: str


@dataclass
class ConsensusAchievement:
    """Record of achieved consensus"""
    id: str
    consensus_building_id: str
    achieved_at: datetime
    final_consensus_level: ConsensusStatus
    
    # Achievement details
    final_support_percentage: float
    key_success_factors: List[str]
    critical_moments: List[str]
    lessons_learned: List[str]
    
    # Stakeholder outcomes
    final_stakeholder_positions: List[ConsensusPosition]
    coalition_composition: Dict[str, List[str]]
    holdout_stakeholders: List[str]
    
    # Process effectiveness
    total_time_taken: str
    strategies_used: List[ConsensusStrategy]
    actions_completed: int
    barriers_overcome: int
    
    # Quality assessment
    consensus_quality_score: float  # 0.0 to 1.0
    sustainability_assessment: str
    implementation_readiness: float  # 0.0 to 1.0