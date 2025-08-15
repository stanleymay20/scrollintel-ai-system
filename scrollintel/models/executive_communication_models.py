"""
Executive Communication Models for Board Executive Mastery System
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class ExecutiveLevel(Enum):
    """Executive hierarchy levels"""
    BOARD_CHAIR = "board_chair"
    CEO = "ceo"
    CTO = "cto"
    CFO = "cfo"
    COO = "coo"
    BOARD_MEMBER = "board_member"
    INVESTOR = "investor"
    REGULATORY = "regulatory"


class CommunicationStyle(Enum):
    """Communication style preferences"""
    ANALYTICAL = "analytical"
    STRATEGIC = "strategic"
    DIPLOMATIC = "diplomatic"
    DIRECT = "direct"
    COLLABORATIVE = "collaborative"
    AUTHORITATIVE = "authoritative"


class MessageType(Enum):
    """Types of executive messages"""
    STRATEGIC_UPDATE = "strategic_update"
    RISK_ASSESSMENT = "risk_assessment"
    PERFORMANCE_REPORT = "performance_report"
    RECOMMENDATION = "recommendation"
    CRISIS_COMMUNICATION = "crisis_communication"
    BOARD_PRESENTATION = "board_presentation"


@dataclass
class ExecutiveAudience:
    """Executive audience profile"""
    id: str
    name: str
    title: str
    executive_level: ExecutiveLevel
    communication_style: CommunicationStyle
    expertise_areas: List[str]
    decision_making_pattern: str
    influence_level: float
    preferred_communication_format: str
    attention_span: int  # minutes
    detail_preference: str  # high, medium, low
    risk_tolerance: str  # high, medium, low
    created_at: datetime


@dataclass
class Message:
    """Original message to be adapted"""
    id: str
    content: str
    message_type: MessageType
    technical_complexity: float
    urgency_level: str
    key_points: List[str]
    supporting_data: Dict[str, Any]
    created_at: datetime


@dataclass
class AdaptedMessage:
    """Message adapted for executive audience"""
    id: str
    original_message_id: str
    audience_id: str
    adapted_content: str
    executive_summary: str
    key_recommendations: List[str]
    tone: str
    language_complexity: str
    estimated_reading_time: int
    effectiveness_score: float
    adaptation_rationale: str
    created_at: datetime


@dataclass
class CommunicationEffectiveness:
    """Measurement of communication effectiveness"""
    id: str
    message_id: str
    audience_id: str
    engagement_score: float
    comprehension_score: float
    action_taken: bool
    feedback_received: Optional[str]
    response_time: Optional[int]  # minutes
    follow_up_questions: int
    decision_influenced: bool
    measured_at: datetime


@dataclass
class LanguageAdaptationRule:
    """Rules for adapting language to executive audiences"""
    id: str
    executive_level: ExecutiveLevel
    communication_style: CommunicationStyle
    vocabulary_adjustments: Dict[str, str]
    tone_guidelines: str
    structure_preferences: str
    emphasis_patterns: List[str]
    avoid_patterns: List[str]
    created_at: datetime