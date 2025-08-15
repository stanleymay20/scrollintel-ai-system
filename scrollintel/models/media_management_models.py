"""
Media Management Models for Crisis Leadership Excellence

This module defines data models for media management during crisis situations,
including media inquiries, PR strategies, and sentiment monitoring.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any
from uuid import uuid4


class MediaInquiryType(Enum):
    """Types of media inquiries"""
    BREAKING_NEWS = "breaking_news"
    INVESTIGATIVE = "investigative"
    FOLLOW_UP = "follow_up"
    STATEMENT_REQUEST = "statement_request"
    INTERVIEW_REQUEST = "interview_request"
    FACT_CHECK = "fact_check"


class MediaOutletType(Enum):
    """Types of media outlets"""
    NEWSPAPER = "newspaper"
    TELEVISION = "television"
    RADIO = "radio"
    ONLINE_NEWS = "online_news"
    BLOG = "blog"
    SOCIAL_MEDIA = "social_media"
    TRADE_PUBLICATION = "trade_publication"
    PODCAST = "podcast"


class InquiryPriority(Enum):
    """Priority levels for media inquiries"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResponseStatus(Enum):
    """Status of media response"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    SENT = "sent"
    DECLINED = "declined"


class SentimentScore(Enum):
    """Sentiment analysis scores"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class MediaOutlet:
    """Represents a media outlet"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    outlet_type: MediaOutletType = MediaOutletType.ONLINE_NEWS
    reach: int = 0  # Estimated audience size
    influence_score: float = 0.0  # 0-100 influence rating
    contact_info: Dict[str, str] = field(default_factory=dict)
    beat_reporters: List[str] = field(default_factory=list)
    typical_response_time: int = 60  # minutes
    relationship_quality: str = "neutral"  # good, neutral, poor
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MediaInquiry:
    """Represents a media inquiry during crisis"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    outlet: MediaOutlet = field(default_factory=MediaOutlet)
    reporter_name: str = ""
    reporter_contact: str = ""
    inquiry_type: MediaInquiryType = MediaInquiryType.STATEMENT_REQUEST
    subject: str = ""
    questions: List[str] = field(default_factory=list)
    deadline: datetime = field(default_factory=datetime.now)
    priority: InquiryPriority = InquiryPriority.MEDIUM
    received_at: datetime = field(default_factory=datetime.now)
    context: str = ""
    potential_impact: str = ""
    recommended_response: str = ""
    response_status: ResponseStatus = ResponseStatus.PENDING
    assigned_spokesperson: str = ""


@dataclass
class MediaResponse:
    """Represents a response to media inquiry"""
    id: str = field(default_factory=lambda: str(uuid4()))
    inquiry_id: str = ""
    response_type: str = "statement"  # statement, interview, no_comment, decline
    content: str = ""
    key_messages: List[str] = field(default_factory=list)
    approved_by: List[str] = field(default_factory=list)
    sent_at: Optional[datetime] = None
    follow_up_required: bool = False
    effectiveness_score: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PRStrategy:
    """Public relations strategy for crisis management"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    strategy_name: str = ""
    objectives: List[str] = field(default_factory=list)
    target_audiences: List[str] = field(default_factory=list)
    key_messages: List[str] = field(default_factory=list)
    communication_channels: List[str] = field(default_factory=list)
    timeline: Dict[str, datetime] = field(default_factory=dict)
    success_metrics: List[str] = field(default_factory=list)
    risk_mitigation: List[str] = field(default_factory=list)
    spokesperson_assignments: Dict[str, str] = field(default_factory=dict)
    budget_allocation: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class MediaMention:
    """Represents a media mention or coverage"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    outlet: MediaOutlet = field(default_factory=MediaOutlet)
    headline: str = ""
    content: str = ""
    url: str = ""
    author: str = ""
    published_at: datetime = field(default_factory=datetime.now)
    reach: int = 0
    engagement_metrics: Dict[str, int] = field(default_factory=dict)
    sentiment_score: SentimentScore = SentimentScore.NEUTRAL
    sentiment_confidence: float = 0.0
    key_topics: List[str] = field(default_factory=list)
    mentions_company: bool = False
    mentions_executives: List[str] = field(default_factory=list)
    tone: str = "neutral"  # positive, neutral, negative, critical
    accuracy_rating: str = "accurate"  # accurate, misleading, false


@dataclass
class SentimentAnalysis:
    """Sentiment analysis results for media coverage"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    analysis_period: Dict[str, datetime] = field(default_factory=dict)
    overall_sentiment: SentimentScore = SentimentScore.NEUTRAL
    sentiment_trend: str = "stable"  # improving, stable, declining
    mention_volume: int = 0
    positive_mentions: int = 0
    negative_mentions: int = 0
    neutral_mentions: int = 0
    key_sentiment_drivers: List[str] = field(default_factory=list)
    outlet_breakdown: Dict[str, Dict] = field(default_factory=dict)
    geographic_breakdown: Dict[str, Dict] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MediaMonitoringAlert:
    """Alert for significant media activity"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    alert_type: str = "volume_spike"  # volume_spike, sentiment_drop, major_outlet
    severity: str = "medium"  # low, medium, high, critical
    description: str = ""
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    affected_metrics: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    stakeholders_to_notify: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


@dataclass
class MediaManagementMetrics:
    """Metrics for media management effectiveness"""
    crisis_id: str = ""
    total_inquiries: int = 0
    response_rate: float = 0.0
    average_response_time: float = 0.0  # minutes
    positive_coverage_percentage: float = 0.0
    media_reach: int = 0
    message_consistency_score: float = 0.0
    spokesperson_effectiveness: Dict[str, float] = field(default_factory=dict)
    crisis_narrative_control: float = 0.0  # 0-100 scale
    reputation_impact_score: float = 0.0
    calculated_at: datetime = field(default_factory=datetime.now)