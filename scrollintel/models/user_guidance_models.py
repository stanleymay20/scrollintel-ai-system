"""
Data models for the User Guidance and Support System
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid

class GuidanceType(Enum):
    """Types of guidance that can be provided"""
    CONTEXTUAL_HELP = "contextual_help"
    ERROR_EXPLANATION = "error_explanation"
    PROACTIVE_SYSTEM = "proactive_system"
    PROACTIVE_HELP = "proactive_help"
    FEATURE_DISCOVERY = "feature_discovery"
    FALLBACK = "fallback"

class SeverityLevel(Enum):
    """Severity levels for errors and issues"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TicketStatus(Enum):
    """Status values for support tickets"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

@dataclass
class GuidanceContext:
    """Context information for providing guidance"""
    user_id: str
    session_id: str
    current_page: str
    user_action: Optional[str] = None
    system_state: Dict[str, Any] = field(default_factory=dict)
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    additional_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HelpRequest:
    """Request for contextual help"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    context: Optional[GuidanceContext] = None
    help_topic: Optional[str] = None
    urgency: SeverityLevel = SeverityLevel.MEDIUM
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ErrorExplanation:
    """Intelligent explanation of an error"""
    error_id: str
    error_type: str
    error_message: str
    user_friendly_explanation: str
    actionable_solutions: List[Dict[str, Any]]
    severity: SeverityLevel
    context: GuidanceContext
    timestamp: datetime
    resolution_confidence: float
    related_errors: List[str] = field(default_factory=list)
    auto_resolved: bool = False

@dataclass
class ProactiveGuidance:
    """Proactive guidance provided to users"""
    guidance_id: str
    user_id: str
    type: GuidanceType
    title: str
    message: str
    actions: List[Dict[str, Any]] = field(default_factory=list)
    priority: str = "medium"
    shown: bool = False
    dismissed: bool = False
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SupportTicket:
    """Automated support ticket"""
    ticket_id: str
    user_id: str
    title: str
    description: str
    detailed_context: Dict[str, Any]
    priority: str
    status: TicketStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    resolution_notes: Optional[str] = None
    auto_created: bool = True

@dataclass
class UserBehaviorPattern:
    """User behavior patterns for proactive guidance"""
    user_id: str
    common_actions: List[str] = field(default_factory=list)
    struggle_points: List[str] = field(default_factory=list)
    expertise_level: str = "beginner"
    preferred_help_format: str = "text"
    session_duration_avg: float = 0.0
    error_frequency: float = 0.0
    feature_usage: Dict[str, int] = field(default_factory=dict)
    last_active: datetime = field(default_factory=datetime.utcnow)
    guidance_effectiveness: Dict[str, float] = field(default_factory=dict)

@dataclass
class GuidanceMetrics:
    """Metrics for guidance system effectiveness"""
    total_help_requests: int = 0
    successful_resolutions: int = 0
    average_resolution_time: float = 0.0
    user_satisfaction_score: float = 0.0
    proactive_guidance_acceptance: float = 0.0
    error_explanation_clarity: float = 0.0
    support_ticket_auto_resolution: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class HelpContent:
    """Structured help content"""
    content_id: str
    title: str
    content: str
    content_type: str = "text"  # text, video, interactive
    tags: List[str] = field(default_factory=list)
    difficulty_level: str = "beginner"
    estimated_read_time: int = 0  # in minutes
    related_topics: List[str] = field(default_factory=list)
    effectiveness_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

@dataclass
class GuidanceResponse:
    """Response from guidance system"""
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    guidance_type: GuidanceType = GuidanceType.CONTEXTUAL_HELP
    title: str = ""
    content: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    estimated_resolution_time: Optional[int] = None  # in minutes
    follow_up_actions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class UserFeedback:
    """User feedback on guidance provided"""
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    guidance_id: str = ""
    rating: int = 0  # 1-5 scale
    helpful: bool = False
    comments: Optional[str] = None
    resolution_achieved: bool = False
    time_to_resolution: Optional[int] = None  # in minutes
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ContextualHint:
    """Contextual hints for user interface elements"""
    hint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    element_selector: str = ""
    hint_text: str = ""
    trigger_condition: str = ""
    priority: int = 1
    shown_count: int = 0
    max_shows: int = 3
    user_dismissed: bool = False
    effectiveness_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AutomatedResolution:
    """Automated resolution attempt"""
    resolution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error_id: str = ""
    resolution_type: str = ""
    resolution_steps: List[str] = field(default_factory=list)
    success: bool = False
    execution_time: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)