"""
Stakeholder Influence Models for Board Executive Mastery System
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class StakeholderType(Enum):
    BOARD_MEMBER = "board_member"
    EXECUTIVE = "executive"
    INVESTOR = "investor"
    ADVISOR = "advisor"
    PARTNER = "partner"


class InfluenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RelationshipStatus(Enum):
    STRONG = "strong"
    NEUTRAL = "neutral"
    WEAK = "weak"
    ADVERSARIAL = "adversarial"


class CommunicationStyle(Enum):
    ANALYTICAL = "analytical"
    RELATIONSHIP_FOCUSED = "relationship_focused"
    RESULTS_ORIENTED = "results_oriented"
    VISIONARY = "visionary"
    DETAIL_ORIENTED = "detail_oriented"


@dataclass
class Background:
    industry_experience: List[str]
    functional_expertise: List[str]
    education: List[str]
    previous_roles: List[str]
    achievements: List[str]


@dataclass
class Priority:
    name: str
    description: str
    importance: float  # 0.0 to 1.0
    category: str


@dataclass
class Relationship:
    stakeholder_id: str
    relationship_type: str
    strength: float  # 0.0 to 1.0
    history: List[str]
    last_interaction: Optional[datetime] = None


@dataclass
class DecisionPattern:
    decision_style: str
    key_factors: List[str]
    typical_concerns: List[str]
    influence_tactics: List[str]


@dataclass
class Stakeholder:
    id: str
    name: str
    title: str
    organization: str
    stakeholder_type: StakeholderType
    background: Background
    influence_level: InfluenceLevel
    communication_style: CommunicationStyle
    decision_pattern: DecisionPattern
    priorities: List[Priority]
    relationships: List[Relationship]
    contact_preferences: Dict[str, Any]
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class InfluenceNetwork:
    id: str
    name: str
    stakeholders: List[str]  # stakeholder IDs
    influence_flows: Dict[str, Dict[str, float]]  # from -> to -> strength
    power_centers: List[str]  # stakeholder IDs with high influence
    coalition_potential: Dict[str, List[str]]  # issue -> stakeholder IDs


@dataclass
class StakeholderMap:
    id: str
    organization_id: str
    stakeholders: List[Stakeholder]
    influence_networks: List[InfluenceNetwork]
    key_relationships: List[Relationship]
    power_dynamics: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class InfluenceAssessment:
    stakeholder_id: str
    formal_authority: float  # 0.0 to 1.0
    informal_influence: float  # 0.0 to 1.0
    network_centrality: float  # 0.0 to 1.0
    expertise_credibility: float  # 0.0 to 1.0
    resource_control: float  # 0.0 to 1.0
    overall_influence: float  # calculated composite score
    assessment_date: datetime = field(default_factory=datetime.now)


@dataclass
class RelationshipOptimization:
    stakeholder_id: str
    current_relationship_strength: float
    target_relationship_strength: float
    optimization_strategies: List[str]
    action_items: List[str]
    timeline: Dict[str, datetime]
    success_metrics: List[str]


@dataclass
class StakeholderAnalysis:
    stakeholder_id: str
    influence_assessment: InfluenceAssessment
    relationship_optimization: RelationshipOptimization
    engagement_history: List[Dict[str, Any]]
    predicted_positions: Dict[str, str]  # issue -> predicted stance
    engagement_recommendations: List[str]
    analysis_date: datetime = field(default_factory=datetime.now)