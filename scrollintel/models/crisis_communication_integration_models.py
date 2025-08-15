"""
Data Models for Crisis Communication Integration

Defines the data structures and models used for integrating crisis leadership
capabilities with communication systems.
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

Base = declarative_base()


class CommunicationChannelType(str, Enum):
    """Types of communication channels"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    SMS = "sms"
    PHONE = "phone"
    SOCIAL_MEDIA = "social_media"
    INTERNAL_CHAT = "internal_chat"
    EXTERNAL_API = "external_api"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"


class MessagePriority(str, Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class CommunicationStatus(str, Enum):
    """Communication status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"


# Database Models

class CrisisCommunicationSession(Base):
    """Database model for crisis communication sessions"""
    __tablename__ = "crisis_communication_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    crisis_id = Column(String(255), nullable=False, index=True)
    session_start = Column(DateTime, default=datetime.utcnow)
    session_end = Column(DateTime, nullable=True)
    crisis_type = Column(String(100), nullable=False)
    severity_level = Column(Integer, nullable=False)
    escalation_level = Column(Integer, default=1)
    status = Column(String(50), default="active")
    
    # Configuration
    enabled_channels = Column(JSON, nullable=False)
    communication_protocols = Column(JSON, nullable=False)
    
    # Relationships
    messages = relationship("CrisisCommunicationMessage", back_populates="session")
    broadcasts = relationship("CrisisBroadcast", back_populates="session")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CrisisCommunicationMessage(Base):
    """Database model for crisis communication messages"""
    __tablename__ = "crisis_communication_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("crisis_communication_sessions.id"), nullable=False)
    
    # Message details
    channel_type = Column(String(50), nullable=False)
    message_content = Column(Text, nullable=False)
    original_message = Column(Text, nullable=True)
    sender = Column(String(255), nullable=False)
    recipient = Column(String(255), nullable=True)
    
    # Crisis context
    crisis_aware = Column(Boolean, default=False)
    crisis_context = Column(JSON, nullable=True)
    
    # Processing details
    priority = Column(String(20), default="normal")
    status = Column(String(20), default="pending")
    processing_time = Column(DateTime, nullable=True)
    delivery_time = Column(DateTime, nullable=True)
    
    # Response details
    response_generated = Column(Text, nullable=True)
    protocols_applied = Column(JSON, nullable=True)
    routing_info = Column(JSON, nullable=True)
    
    # Relationships
    session = relationship("CrisisCommunicationSession", back_populates="messages")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CrisisBroadcast(Base):
    """Database model for crisis broadcasts"""
    __tablename__ = "crisis_broadcasts"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("crisis_communication_sessions.id"), nullable=False)
    
    # Broadcast details
    broadcast_message = Column(Text, nullable=False)
    target_channels = Column(JSON, nullable=False)
    broadcast_type = Column(String(50), default="update")
    
    # Status tracking
    status = Column(String(20), default="pending")
    initiated_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Results
    broadcast_results = Column(JSON, nullable=True)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    
    # Relationships
    session = relationship("CrisisCommunicationSession", back_populates="broadcasts")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CommunicationChannelConfig(Base):
    """Database model for communication channel configurations"""
    __tablename__ = "communication_channel_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    channel_type = Column(String(50), nullable=False, unique=True)
    
    # Configuration
    enabled = Column(Boolean, default=True)
    crisis_mode_enabled = Column(Boolean, default=True)
    auto_escalation = Column(Boolean, default=False)
    require_approval = Column(Boolean, default=False)
    
    # Channel-specific settings
    channel_settings = Column(JSON, nullable=True)
    message_templates = Column(JSON, nullable=True)
    escalation_rules = Column(JSON, nullable=True)
    
    # Rate limiting
    rate_limit_enabled = Column(Boolean, default=False)
    max_messages_per_minute = Column(Integer, default=60)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CommunicationMetrics(Base):
    """Database model for communication metrics"""
    __tablename__ = "communication_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    crisis_id = Column(String(255), nullable=False, index=True)
    
    # Metrics
    total_messages = Column(Integer, default=0)
    crisis_aware_messages = Column(Integer, default=0)
    broadcasts_sent = Column(Integer, default=0)
    successful_deliveries = Column(Integer, default=0)
    failed_deliveries = Column(Integer, default=0)
    
    # Performance metrics
    avg_response_time = Column(Integer, default=0)  # milliseconds
    avg_processing_time = Column(Integer, default=0)  # milliseconds
    escalations_triggered = Column(Integer, default=0)
    
    # Channel breakdown
    channel_usage = Column(JSON, nullable=True)
    channel_success_rates = Column(JSON, nullable=True)
    
    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic Models for API

class CrisisContextModel(BaseModel):
    """Pydantic model for crisis context"""
    crisis_id: str
    crisis_type: str
    severity_level: int
    status: str
    affected_systems: List[str]
    stakeholders: List[str]
    communication_protocols: Dict[str, Any]
    escalation_level: int
    start_time: datetime
    last_update: datetime


class CommunicationMessageModel(BaseModel):
    """Pydantic model for communication messages"""
    id: Optional[int] = None
    channel_type: CommunicationChannelType
    message_content: str
    original_message: Optional[str] = None
    sender: str
    recipient: Optional[str] = None
    crisis_aware: bool = False
    crisis_context: Optional[Dict[str, Any]] = None
    priority: MessagePriority = MessagePriority.NORMAL
    status: CommunicationStatus = CommunicationStatus.PENDING
    processing_time: Optional[datetime] = None
    delivery_time: Optional[datetime] = None
    response_generated: Optional[str] = None
    protocols_applied: Optional[Dict[str, Any]] = None
    routing_info: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


class BroadcastModel(BaseModel):
    """Pydantic model for crisis broadcasts"""
    id: Optional[int] = None
    crisis_id: str
    broadcast_message: str
    target_channels: List[CommunicationChannelType]
    broadcast_type: str = "update"
    status: str = "pending"
    initiated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    broadcast_results: Optional[Dict[str, Any]] = None
    success_count: int = 0
    failure_count: int = 0


class ChannelConfigModel(BaseModel):
    """Pydantic model for channel configuration"""
    channel_type: CommunicationChannelType
    enabled: bool = True
    crisis_mode_enabled: bool = True
    auto_escalation: bool = False
    require_approval: bool = False
    channel_settings: Optional[Dict[str, Any]] = None
    message_templates: Optional[Dict[str, str]] = None
    escalation_rules: Optional[Dict[str, Any]] = None
    rate_limit_enabled: bool = False
    max_messages_per_minute: int = 60


class CommunicationMetricsModel(BaseModel):
    """Pydantic model for communication metrics"""
    crisis_id: str
    total_messages: int = 0
    crisis_aware_messages: int = 0
    broadcasts_sent: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    avg_response_time: int = 0
    avg_processing_time: int = 0
    escalations_triggered: int = 0
    channel_usage: Optional[Dict[str, int]] = None
    channel_success_rates: Optional[Dict[str, float]] = None
    period_start: datetime
    period_end: datetime


class CommunicationIntegrationStatus(BaseModel):
    """Pydantic model for integration status"""
    active_crises: int
    enabled_channels: List[CommunicationChannelType]
    total_messages_today: int
    crisis_aware_messages_today: int
    broadcasts_today: int
    avg_response_time_today: int
    system_health: str
    last_updated: datetime


class CrisisRegistrationRequest(BaseModel):
    """Request model for crisis registration"""
    crisis_id: str = Field(..., description="Unique crisis identifier")
    crisis_type: str = Field(..., description="Type of crisis")
    severity_level: int = Field(..., ge=1, le=5, description="Severity level (1-5)")
    affected_areas: List[str] = Field(..., description="Affected systems/areas")
    stakeholders_impacted: List[str] = Field(..., description="Impacted stakeholders")
    communication_protocols: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Custom communication protocols"
    )


class CommunicationProcessingRequest(BaseModel):
    """Request model for processing communication"""
    channel: CommunicationChannelType = Field(..., description="Communication channel")
    message: str = Field(..., description="Message content")
    sender: str = Field(..., description="Message sender")
    recipient: Optional[str] = Field(default=None, description="Message recipient")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="Message priority")


class BroadcastRequest(BaseModel):
    """Request model for broadcasting messages"""
    crisis_id: str = Field(..., description="Crisis identifier")
    message: str = Field(..., description="Broadcast message")
    target_channels: Optional[List[CommunicationChannelType]] = Field(
        default=None, 
        description="Target channels (all if not specified)"
    )
    broadcast_type: str = Field(default="update", description="Type of broadcast")
    priority: MessagePriority = Field(default=MessagePriority.HIGH, description="Broadcast priority")


class CommunicationResponse(BaseModel):
    """Response model for communication processing"""
    success: bool
    message_id: Optional[str] = None
    response_content: Optional[str] = None
    crisis_context_applied: bool = False
    protocols_applied: Optional[List[str]] = None
    routing_info: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[int] = None
    timestamp: datetime
    error: Optional[str] = None


class BroadcastResponse(BaseModel):
    """Response model for broadcast operations"""
    success: bool
    broadcast_id: Optional[str] = None
    crisis_id: str
    channels_targeted: List[CommunicationChannelType]
    channels_successful: List[CommunicationChannelType]
    channels_failed: List[CommunicationChannelType]
    total_recipients: int
    timestamp: datetime
    error: Optional[str] = None