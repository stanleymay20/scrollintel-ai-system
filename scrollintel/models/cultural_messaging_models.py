"""
Cultural Messaging Models

Data models for cultural messaging framework including message templates,
audience targeting, and effectiveness tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class MessageType(Enum):
    """Types of cultural messages"""
    VISION_COMMUNICATION = "vision_communication"
    VALUES_REINFORCEMENT = "values_reinforcement"
    CHANGE_ANNOUNCEMENT = "change_announcement"
    SUCCESS_CELEBRATION = "success_celebration"
    FEEDBACK_REQUEST = "feedback_request"
    TRAINING_CONTENT = "training_content"


class AudienceType(Enum):
    """Target audience types"""
    ALL_EMPLOYEES = "all_employees"
    LEADERSHIP_TEAM = "leadership_team"
    DEPARTMENT_SPECIFIC = "department_specific"
    NEW_HIRES = "new_hires"
    REMOTE_WORKERS = "remote_workers"
    CHANGE_CHAMPIONS = "change_champions"


class MessageChannel(Enum):
    """Communication channels"""
    EMAIL = "email"
    SLACK = "slack"
    INTRANET = "intranet"
    TOWN_HALL = "town_hall"
    TEAM_MEETING = "team_meeting"
    DIGITAL_SIGNAGE = "digital_signage"


@dataclass
class CulturalMessage:
    """Core cultural message structure"""
    id: str
    title: str
    content: str
    message_type: MessageType
    cultural_themes: List[str]
    key_values: List[str]
    created_at: datetime
    updated_at: datetime
    version: int = 1
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageTemplate:
    """Reusable message template"""
    id: str
    name: str
    template_content: str
    message_type: MessageType
    required_variables: List[str]
    optional_variables: List[str] = field(default_factory=list)
    usage_guidelines: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AudienceProfile:
    """Target audience profile"""
    id: str
    name: str
    audience_type: AudienceType
    characteristics: Dict[str, Any]
    communication_preferences: Dict[str, Any]
    cultural_context: Dict[str, str]
    size: int
    engagement_history: Dict[str, float] = field(default_factory=dict)


@dataclass
class MessageCustomization:
    """Message customization for specific audience"""
    id: str
    base_message_id: str
    audience_id: str
    customized_content: str
    personalization_data: Dict[str, Any]
    channel: MessageChannel
    delivery_timing: datetime
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MessageDelivery:
    """Message delivery tracking"""
    id: str
    message_id: str
    audience_id: str
    channel: MessageChannel
    delivered_at: datetime
    delivery_status: str
    recipient_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageEngagement:
    """Message engagement metrics"""
    id: str
    message_id: str
    audience_id: str
    channel: MessageChannel
    views: int = 0
    clicks: int = 0
    shares: int = 0
    responses: int = 0
    sentiment_score: float = 0.0
    engagement_rate: float = 0.0
    measured_at: datetime = field(default_factory=datetime.now)


@dataclass
class MessageEffectiveness:
    """Message effectiveness analysis"""
    id: str
    message_id: str
    audience_id: str
    effectiveness_score: float
    cultural_alignment_score: float
    behavior_change_indicators: Dict[str, float]
    feedback_summary: Dict[str, Any]
    recommendations: List[str]
    measured_at: datetime = field(default_factory=datetime.now)


@dataclass
class CulturalMessagingCampaign:
    """Cultural messaging campaign"""
    id: str
    name: str
    description: str
    cultural_objectives: List[str]
    target_audiences: List[str]
    messages: List[str]
    start_date: datetime
    end_date: datetime
    status: str
    success_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MessagingStrategy:
    """Overall messaging strategy"""
    id: str
    organization_id: str
    cultural_vision: str
    core_values: List[str]
    key_themes: List[str]
    audience_segments: List[AudienceProfile]
    message_templates: List[MessageTemplate]
    communication_calendar: Dict[str, Any]
    effectiveness_targets: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)