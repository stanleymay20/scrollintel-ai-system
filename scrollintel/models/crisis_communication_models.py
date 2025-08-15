"""
Crisis Communication Models

Data models for crisis communication system including stakeholder management,
notification tracking, and message coordination.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any
from uuid import uuid4


class StakeholderType(Enum):
    """Types of stakeholders in crisis communication"""
    BOARD_MEMBER = "board_member"
    EXECUTIVE = "executive"
    EMPLOYEE = "employee"
    CUSTOMER = "customer"
    INVESTOR = "investor"
    MEDIA = "media"
    REGULATOR = "regulator"
    PARTNER = "partner"
    VENDOR = "vendor"
    PUBLIC = "public"


class NotificationPriority(Enum):
    """Priority levels for notifications"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SMS = "sms"
    PHONE = "phone"
    SLACK = "slack"
    TEAMS = "teams"
    PUSH = "push"
    PORTAL = "portal"
    MEDIA_RELEASE = "media_release"


class NotificationStatus(Enum):
    """Status of notification delivery"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    BOUNCED = "bounced"


@dataclass
class Stakeholder:
    """Represents a stakeholder in crisis communication"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    stakeholder_type: StakeholderType = StakeholderType.EMPLOYEE
    contact_info: Dict[str, str] = field(default_factory=dict)
    preferred_channels: List[NotificationChannel] = field(default_factory=list)
    priority_level: NotificationPriority = NotificationPriority.MEDIUM
    role: str = ""
    department: str = ""
    influence_level: int = 1  # 1-10 scale
    crisis_relevance: Dict[str, int] = field(default_factory=dict)  # crisis_type -> relevance score
    timezone: str = "UTC"
    availability_hours: Dict[str, str] = field(default_factory=dict)
    escalation_path: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class NotificationTemplate:
    """Template for crisis notifications"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    crisis_type: str = ""
    stakeholder_type: StakeholderType = StakeholderType.EMPLOYEE
    channel: NotificationChannel = NotificationChannel.EMAIL
    subject_template: str = ""
    body_template: str = ""
    variables: List[str] = field(default_factory=list)
    approval_required: bool = False
    auto_send: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class NotificationMessage:
    """Individual notification message"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    stakeholder_id: str = ""
    template_id: Optional[str] = None
    channel: NotificationChannel = NotificationChannel.EMAIL
    priority: NotificationPriority = NotificationPriority.MEDIUM
    subject: str = ""
    content: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)
    status: NotificationStatus = NotificationStatus.PENDING
    scheduled_time: Optional[datetime] = None
    sent_time: Optional[datetime] = None
    delivered_time: Optional[datetime] = None
    read_time: Optional[datetime] = None
    failure_reason: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class NotificationBatch:
    """Batch of notifications for coordinated sending"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    name: str = ""
    description: str = ""
    stakeholder_groups: List[StakeholderType] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)  # message IDs
    status: str = "pending"
    scheduled_time: Optional[datetime] = None
    sent_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StakeholderGroup:
    """Group of stakeholders for targeted communication"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    stakeholder_ids: List[str] = field(default_factory=list)
    criteria: Dict[str, Any] = field(default_factory=dict)
    auto_update: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CommunicationPlan:
    """Communication plan for crisis response"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    crisis_type: str = ""
    phases: List[Dict[str, Any]] = field(default_factory=list)
    stakeholder_matrix: Dict[str, List[str]] = field(default_factory=dict)
    message_sequences: List[Dict[str, Any]] = field(default_factory=list)
    escalation_triggers: List[Dict[str, Any]] = field(default_factory=list)
    approval_workflow: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class NotificationMetrics:
    """Metrics for notification effectiveness"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_id: str = ""
    total_sent: int = 0
    total_delivered: int = 0
    total_read: int = 0
    total_failed: int = 0
    delivery_rate: float = 0.0
    read_rate: float = 0.0
    average_delivery_time: float = 0.0
    channel_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stakeholder_engagement: Dict[str, Dict[str, float]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)