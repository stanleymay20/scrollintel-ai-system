"""
Data models for board and executive relationship management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class RelationshipType(Enum):
    """Types of executive relationships."""
    BOARD_MEMBER = "board_member"
    EXECUTIVE = "executive"
    INVESTOR = "investor"
    STRATEGIC_PARTNER = "strategic_partner"
    REGULATORY_OFFICIAL = "regulatory_official"
    MEDIA_CONTACT = "media_contact"


class RelationshipStatus(Enum):
    """Status of relationship development."""
    INITIAL = "initial"
    DEVELOPING = "developing"
    ESTABLISHED = "established"
    STRONG = "strong"
    STRAINED = "strained"
    DAMAGED = "damaged"


class InteractionType(Enum):
    """Types of interactions with stakeholders."""
    MEETING = "meeting"
    EMAIL = "email"
    PHONE_CALL = "phone_call"
    PRESENTATION = "presentation"
    SOCIAL_EVENT = "social_event"
    CRISIS_COMMUNICATION = "crisis_communication"


class CommunicationStyle(Enum):
    """Communication preferences of stakeholders."""
    DIRECT = "direct"
    DIPLOMATIC = "diplomatic"
    DATA_DRIVEN = "data_driven"
    RELATIONSHIP_FOCUSED = "relationship_focused"
    RESULTS_ORIENTED = "results_oriented"


@dataclass
class PersonalityProfile:
    """Personality and communication profile of a stakeholder."""
    communication_style: CommunicationStyle
    decision_making_style: str
    key_motivators: List[str]
    concerns: List[str]
    preferred_interaction_frequency: str
    optimal_meeting_times: List[str]
    communication_preferences: Dict[str, Any]


@dataclass
class RelationshipHistory:
    """Historical record of relationship interactions."""
    interaction_id: str
    date: datetime
    interaction_type: InteractionType
    duration_minutes: Optional[int]
    topics_discussed: List[str]
    sentiment_score: float  # -1.0 to 1.0
    outcomes: List[str]
    follow_up_required: bool
    notes: str


@dataclass
class RelationshipGoal:
    """Specific goals for relationship development."""
    goal_id: str
    description: str
    target_date: datetime
    priority: str  # high, medium, low
    success_metrics: List[str]
    current_progress: float  # 0.0 to 1.0
    action_items: List[str]


@dataclass
class TrustMetrics:
    """Quantified trust and credibility measurements."""
    overall_trust_score: float  # 0.0 to 1.0
    competence_trust: float
    benevolence_trust: float
    integrity_trust: float
    predictability_trust: float
    transparency_score: float
    reliability_score: float
    last_updated: datetime


@dataclass
class RelationshipProfile:
    """Comprehensive profile of a board/executive relationship."""
    stakeholder_id: str
    name: str
    title: str
    organization: str
    relationship_type: RelationshipType
    relationship_status: RelationshipStatus
    
    # Profile information
    personality_profile: PersonalityProfile
    influence_level: float  # 0.0 to 1.0
    decision_making_power: float  # 0.0 to 1.0
    network_connections: List[str]
    
    # Relationship metrics
    trust_metrics: TrustMetrics
    relationship_strength: float  # 0.0 to 1.0
    engagement_frequency: float
    response_rate: float
    
    # Historical data
    relationship_start_date: datetime
    last_interaction_date: Optional[datetime]
    interaction_history: List[RelationshipHistory]
    
    # Development planning
    relationship_goals: List[RelationshipGoal]
    development_strategy: str
    next_planned_interaction: Optional[datetime]
    
    # Context and preferences
    key_interests: List[str]
    business_priorities: List[str]
    personal_interests: List[str]
    communication_cadence: str
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class RelationshipAction:
    """Specific action to take for relationship building."""
    action_id: str
    stakeholder_id: str
    action_type: str
    description: str
    scheduled_date: datetime
    priority: str
    expected_outcome: str
    preparation_required: List[str]
    success_criteria: List[str]
    status: str  # planned, in_progress, completed, cancelled


@dataclass
class RelationshipInsight:
    """AI-generated insights about relationship dynamics."""
    insight_id: str
    stakeholder_id: str
    insight_type: str  # opportunity, risk, recommendation
    description: str
    confidence_score: float
    supporting_evidence: List[str]
    recommended_actions: List[str]
    generated_at: datetime


@dataclass
class RelationshipNetwork:
    """Network analysis of stakeholder relationships."""
    network_id: str
    stakeholders: List[str]
    connections: Dict[str, List[str]]
    influence_paths: Dict[str, List[str]]
    coalition_opportunities: List[Dict[str, Any]]
    potential_conflicts: List[Dict[str, Any]]
    network_health_score: float


@dataclass
class RelationshipMaintenancePlan:
    """Systematic plan for maintaining relationships."""
    plan_id: str
    stakeholder_id: str
    maintenance_frequency: str
    touch_point_types: List[str]
    content_themes: List[str]
    seasonal_considerations: List[str]
    escalation_triggers: List[str]
    success_indicators: List[str]
    next_review_date: datetime