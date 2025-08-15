"""
Board Executive Mastery Models
Data models for board and executive engagement mastery
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class BoardMemberRole(Enum):
    """Board member roles"""
    CHAIR = "chair"
    CEO = "ceo"
    INDEPENDENT_DIRECTOR = "independent_director"
    EXECUTIVE_DIRECTOR = "executive_director"
    AUDIT_COMMITTEE_CHAIR = "audit_committee_chair"
    COMPENSATION_COMMITTEE_CHAIR = "compensation_committee_chair"
    NOMINATING_COMMITTEE_CHAIR = "nominating_committee_chair"

class CommunicationStyle(Enum):
    """Communication styles"""
    ANALYTICAL = "analytical"
    RELATIONSHIP_FOCUSED = "relationship_focused"
    RESULTS_ORIENTED = "results_oriented"
    VISIONARY = "visionary"
    DETAIL_ORIENTED = "detail_oriented"
    BIG_PICTURE = "big_picture"

class EngagementType(Enum):
    """Types of board engagements"""
    BOARD_MEETING = "board_meeting"
    COMMITTEE_MEETING = "committee_meeting"
    STRATEGIC_SESSION = "strategic_session"
    INVESTOR_PRESENTATION = "investor_presentation"
    CRISIS_COMMUNICATION = "crisis_communication"
    PERFORMANCE_REVIEW = "performance_review"

@dataclass
class BoardMemberProfile:
    """Individual board member profile"""
    id: str
    name: str
    role: BoardMemberRole
    background: str
    expertise_areas: List[str]
    communication_style: CommunicationStyle
    influence_level: float
    decision_making_pattern: str
    key_concerns: List[str]
    relationship_dynamics: Dict[str, float]
    preferred_information_format: str
    trust_level: float

@dataclass
class BoardInfo:
    """Board information and composition"""
    id: str
    company_name: str
    board_size: int
    members: List[BoardMemberProfile]
    governance_structure: Dict[str, Any]
    meeting_frequency: str
    decision_making_process: str
    current_priorities: List[str]
    recent_challenges: List[str]
    performance_metrics: Dict[str, float]

@dataclass
class ExecutiveProfile:
    """Executive profile for stakeholder mapping"""
    id: str
    name: str
    title: str
    department: str
    influence_level: float
    communication_style: CommunicationStyle
    key_relationships: List[str]
    strategic_priorities: List[str]
    trust_level: float

@dataclass
class CommunicationContext:
    """Context for executive communication"""
    engagement_type: EngagementType
    audience_profiles: List[BoardMemberProfile]
    key_messages: List[str]
    sensitive_topics: List[str]
    desired_outcomes: List[str]
    time_constraints: Dict[str, Any]
    cultural_considerations: List[str]

@dataclass
class PresentationRequirements:
    """Requirements for board presentations"""
    presentation_type: str
    duration_minutes: int
    audience_size: int
    key_topics: List[str]
    data_requirements: List[str]
    visual_preferences: Dict[str, Any]
    interaction_level: str
    follow_up_requirements: List[str]

@dataclass
class StrategicContext:
    """Strategic context for recommendations"""
    current_strategy: Dict[str, Any]
    market_conditions: Dict[str, Any]
    competitive_landscape: Dict[str, Any]
    financial_position: Dict[str, Any]
    risk_factors: List[str]
    growth_opportunities: List[str]
    stakeholder_expectations: Dict[str, Any]

@dataclass
class MeetingContext:
    """Context for meeting preparation"""
    meeting_type: str
    agenda_items: List[str]
    expected_attendees: List[str]
    decision_points: List[str]
    preparation_time: int
    follow_up_requirements: List[str]
    success_criteria: List[str]

@dataclass
class CredibilityContext:
    """Context for credibility building"""
    current_credibility_level: float
    credibility_challenges: List[str]
    trust_building_opportunities: List[str]
    reputation_factors: Dict[str, float]
    stakeholder_perceptions: Dict[str, Any]
    improvement_areas: List[str]

@dataclass
class BoardExecutiveMasteryRequest:
    """Request for board executive mastery services"""
    id: str
    board_info: BoardInfo
    executives: List[ExecutiveProfile]
    communication_context: CommunicationContext
    presentation_requirements: PresentationRequirements
    strategic_context: StrategicContext
    meeting_context: MeetingContext
    credibility_context: CredibilityContext
    success_criteria: Dict[str, float]
    timeline: Dict[str, datetime]
    created_at: datetime

@dataclass
class BoardAnalysis:
    """Analysis of board dynamics and composition"""
    board_id: str
    composition_analysis: Dict[str, Any]
    power_structure_map: Dict[str, float]
    decision_patterns: Dict[str, Any]
    communication_preferences: Dict[str, CommunicationStyle]
    influence_networks: Dict[str, List[str]]
    priorities: List[str]
    potential_challenges: List[str]
    engagement_opportunities: List[str]
    confidence_score: float

@dataclass
class StakeholderMap:
    """Stakeholder influence mapping"""
    stakeholders: List[ExecutiveProfile]
    influence_matrix: Dict[str, Dict[str, float]]
    coalition_opportunities: List[Dict[str, Any]]
    resistance_points: List[Dict[str, Any]]
    key_relationships: Dict[str, List[str]]
    influence_strategies: Dict[str, str]

@dataclass
class CommunicationStrategy:
    """Executive communication strategy"""
    strategy_id: str
    target_audience: List[str]
    key_messages: List[str]
    communication_channels: List[str]
    messaging_framework: Dict[str, Any]
    tone_and_style: Dict[str, str]
    timing_strategy: Dict[str, datetime]
    feedback_mechanisms: List[str]
    effectiveness_score: float

@dataclass
class PresentationPlan:
    """Board presentation plan"""
    plan_id: str
    presentation_structure: List[Dict[str, Any]]
    visual_design_guidelines: Dict[str, Any]
    data_visualization_plan: List[Dict[str, Any]]
    interaction_strategy: Dict[str, Any]
    qa_preparation: Dict[str, List[str]]
    backup_materials: List[str]
    success_metrics: Dict[str, float]

@dataclass
class StrategicPlan:
    """Strategic recommendations plan"""
    plan_id: str
    strategic_recommendations: List[Dict[str, Any]]
    implementation_roadmap: Dict[str, Any]
    risk_mitigation_strategies: List[Dict[str, Any]]
    success_metrics: Dict[str, float]
    stakeholder_alignment: Dict[str, float]
    resource_requirements: Dict[str, Any]
    timeline: Dict[str, datetime]
    quality_score: float

@dataclass
class MeetingStrategy:
    """Meeting preparation and facilitation strategy"""
    strategy_id: str
    preparation_checklist: List[str]
    agenda_optimization: Dict[str, Any]
    facilitation_guidelines: List[str]
    decision_support_materials: List[Dict[str, Any]]
    conflict_resolution_strategies: List[str]
    follow_up_plan: Dict[str, Any]

@dataclass
class CredibilityPlan:
    """Credibility building plan"""
    plan_id: str
    credibility_building_strategies: List[Dict[str, Any]]
    trust_enhancement_activities: List[Dict[str, Any]]
    reputation_management_plan: Dict[str, Any]
    stakeholder_engagement_plan: Dict[str, Any]
    measurement_framework: Dict[str, float]
    timeline: Dict[str, datetime]

@dataclass
class BoardEngagementPlan:
    """Comprehensive board engagement plan"""
    id: str
    board_id: str
    board_analysis: BoardAnalysis
    stakeholder_map: StakeholderMap
    communication_strategy: CommunicationStrategy
    presentation_plan: PresentationPlan
    strategic_plan: StrategicPlan
    meeting_strategy: MeetingStrategy
    credibility_plan: CredibilityPlan
    success_metrics: Dict[str, float]
    created_at: datetime
    last_updated: Optional[datetime] = None

@dataclass
class ExecutiveInteractionStrategy:
    """Real-time executive interaction strategy"""
    engagement_id: str
    interaction_context: Dict[str, Any]
    adapted_communication: Dict[str, Any]
    strategic_responses: List[Dict[str, Any]]
    decision_support: Dict[str, Any]
    confidence_level: float
    timestamp: datetime

@dataclass
class BoardMasteryMetrics:
    """Metrics for board executive mastery validation"""
    engagement_id: str
    board_confidence_score: float
    executive_trust_score: float
    strategic_alignment_score: float
    communication_effectiveness_score: float
    stakeholder_influence_score: float
    overall_mastery_score: float
    validation_timestamp: datetime
    meets_success_criteria: bool
    improvement_recommendations: Optional[List[str]] = None

@dataclass
class BoardExecutiveMasteryResponse:
    """Response from board executive mastery system"""
    request_id: str
    engagement_plan: Optional[BoardEngagementPlan] = None
    interaction_strategy: Optional[ExecutiveInteractionStrategy] = None
    mastery_metrics: Optional[BoardMasteryMetrics] = None
    system_status: Dict[str, Any] = None
    success: bool = True
    message: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class BoardMasteryOptimization:
    """Optimization recommendations for board mastery"""
    engagement_id: str
    optimization_areas: List[str]
    recommended_actions: List[Dict[str, Any]]
    expected_improvements: Dict[str, float]
    implementation_priority: str
    timeline: Dict[str, datetime]
    resource_requirements: Dict[str, Any]

@dataclass
class ContinuousLearningData:
    """Data for continuous learning and improvement"""
    interaction_id: str
    engagement_context: Dict[str, Any]
    outcomes: Dict[str, Any]
    feedback: Dict[str, Any]
    lessons_learned: List[str]
    improvement_opportunities: List[str]
    timestamp: datetime

# Validation schemas for API requests
BOARD_MASTERY_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["board_info", "communication_context", "strategic_context"],
    "properties": {
        "board_info": {
            "type": "object",
            "required": ["id", "company_name", "members"],
            "properties": {
                "id": {"type": "string"},
                "company_name": {"type": "string"},
                "members": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "name", "role"],
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "role": {"type": "string"}
                        }
                    }
                }
            }
        },
        "communication_context": {
            "type": "object",
            "required": ["engagement_type", "key_messages"],
            "properties": {
                "engagement_type": {"type": "string"},
                "key_messages": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },
        "strategic_context": {
            "type": "object",
            "required": ["current_strategy"],
            "properties": {
                "current_strategy": {"type": "object"}
            }
        }
    }
}

INTERACTION_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["engagement_id", "interaction_context"],
    "properties": {
        "engagement_id": {"type": "string"},
        "interaction_context": {"type": "object"}
    }
}

VALIDATION_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["engagement_id", "validation_context"],
    "properties": {
        "engagement_id": {"type": "string"},
        "validation_context": {"type": "object"}
    }
}