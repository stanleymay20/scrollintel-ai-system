"""
Data models for stakeholder confidence management system.
Supports confidence monitoring, assessment, and trust maintenance during crisis.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class StakeholderType(Enum):
    """Types of stakeholders in crisis situations"""
    BOARD_MEMBER = "board_member"
    INVESTOR = "investor"
    CUSTOMER = "customer"
    EMPLOYEE = "employee"
    PARTNER = "partner"
    REGULATOR = "regulator"
    MEDIA = "media"
    PUBLIC = "public"


class ConfidenceLevel(Enum):
    """Confidence levels for stakeholder assessment"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"
    CRITICAL = "critical"


class TrustIndicator(Enum):
    """Indicators of stakeholder trust"""
    COMMUNICATION_RESPONSE = "communication_response"
    ENGAGEMENT_LEVEL = "engagement_level"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    BEHAVIORAL_PATTERNS = "behavioral_patterns"
    FEEDBACK_QUALITY = "feedback_quality"
    RETENTION_METRICS = "retention_metrics"


@dataclass
class StakeholderProfile:
    """Profile information for stakeholder confidence tracking"""
    stakeholder_id: str
    name: str
    stakeholder_type: StakeholderType
    influence_level: str  # high, medium, low
    communication_preferences: List[str]
    historical_confidence: List[float]
    key_concerns: List[str]
    relationship_strength: float
    contact_information: Dict[str, str]
    last_interaction: Optional[datetime] = None


@dataclass
class ConfidenceMetrics:
    """Metrics for measuring stakeholder confidence"""
    stakeholder_id: str
    confidence_level: ConfidenceLevel
    trust_score: float  # 0.0 to 1.0
    engagement_score: float
    sentiment_score: float
    response_rate: float
    satisfaction_rating: Optional[float]
    risk_indicators: List[str]
    measurement_time: datetime
    data_sources: List[str]


@dataclass
class ConfidenceBuildingStrategy:
    """Strategy for building and maintaining stakeholder confidence"""
    strategy_id: str
    stakeholder_type: StakeholderType
    target_confidence_level: ConfidenceLevel
    communication_approach: str
    key_messages: List[str]
    engagement_tactics: List[str]
    timeline: Dict[str, datetime]
    success_metrics: List[str]
    resource_requirements: List[str]
    risk_mitigation: List[str]


@dataclass
class TrustMaintenanceAction:
    """Actions for maintaining stakeholder trust during crisis"""
    action_id: str
    stakeholder_id: str
    action_type: str
    description: str
    priority: str  # high, medium, low
    expected_impact: str
    implementation_steps: List[str]
    required_resources: List[str]
    timeline: datetime
    success_criteria: List[str]
    status: str = "planned"


@dataclass
class CommunicationPlan:
    """Communication plan for stakeholder confidence management"""
    plan_id: str
    stakeholder_segments: List[StakeholderType]
    key_messages: Dict[str, str]
    communication_channels: List[str]
    frequency: str
    tone_and_style: str
    approval_workflow: List[str]
    feedback_mechanisms: List[str]
    escalation_triggers: List[str]
    effectiveness_metrics: List[str]


@dataclass
class ConfidenceAssessment:
    """Assessment of overall stakeholder confidence"""
    assessment_id: str
    crisis_id: str
    assessment_time: datetime
    overall_confidence_score: float
    stakeholder_breakdown: Dict[StakeholderType, float]
    risk_areas: List[str]
    improvement_opportunities: List[str]
    recommended_actions: List[str]
    trend_analysis: Dict[str, Any]
    next_assessment_date: datetime


@dataclass
class StakeholderFeedback:
    """Feedback from stakeholders during crisis"""
    feedback_id: str
    stakeholder_id: str
    feedback_type: str  # concern, suggestion, complaint, praise
    content: str
    sentiment: str
    urgency_level: str
    received_time: datetime
    response_required: bool
    assigned_to: Optional[str] = None
    resolution_status: str = "open"
    follow_up_actions: List[str] = None


@dataclass
class ConfidenceAlert:
    """Alert for confidence-related issues"""
    alert_id: str
    stakeholder_id: str
    alert_type: str
    severity: str
    description: str
    triggered_by: str
    trigger_time: datetime
    recommended_response: str
    escalation_path: List[str]
    auto_actions: List[str]
    manual_review_required: bool = False