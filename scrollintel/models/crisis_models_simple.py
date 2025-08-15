"""
Simple Crisis Models for Crisis Communication System

Dataclass-based models that don't depend on SQLAlchemy for the crisis communication system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any
from uuid import uuid4


class CrisisType(Enum):
    """Types of crisis situations"""
    SECURITY_BREACH = "security_breach"
    SYSTEM_OUTAGE = "system_outage"
    DATA_LOSS = "data_loss"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    FINANCIAL_CRISIS = "financial_crisis"
    REGULATORY_ISSUE = "regulatory_issue"
    REPUTATION_DAMAGE = "reputation_damage"
    NATURAL_DISASTER = "natural_disaster"
    CYBER_ATTACK = "cyber_attack"
    COMPLIANCE_VIOLATION = "compliance_violation"


class SeverityLevel(Enum):
    """Crisis severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CrisisStatus(Enum):
    """Crisis status states"""
    DETECTED = "detected"
    ACTIVE = "active"
    ESCALATED = "escalated"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class Signal:
    """Crisis detection signal"""
    id: str = field(default_factory=lambda: str(uuid4()))
    source: str = ""
    signal_type: str = ""
    value: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseAction:
    """Crisis response action"""
    id: str = field(default_factory=lambda: str(uuid4()))
    action_type: str = ""
    description: str = ""
    assigned_to: str = ""
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


@dataclass
class Crisis:
    """Crisis information model"""
    id: str = field(default_factory=lambda: str(uuid4()))
    crisis_type: CrisisType = CrisisType.SYSTEM_OUTAGE
    severity_level: SeverityLevel = SeverityLevel.MEDIUM
    start_time: datetime = field(default_factory=datetime.utcnow)
    affected_areas: List[str] = field(default_factory=list)
    stakeholders_impacted: List[str] = field(default_factory=list)
    current_status: CrisisStatus = CrisisStatus.DETECTED
    response_actions: List[ResponseAction] = field(default_factory=list)
    resolution_time: Optional[datetime] = None
    description: str = ""
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CrisisClassification:
    """Crisis classification result"""
    crisis_type: CrisisType = CrisisType.SYSTEM_OUTAGE
    severity_level: SeverityLevel = SeverityLevel.MEDIUM
    confidence: float = 0.0
    sub_categories: List[str] = field(default_factory=list)
    related_crises: List[str] = field(default_factory=list)
    classification_rationale: str = ""
    classified_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ImpactAssessment:
    """Crisis impact assessment"""
    financial_impact: Dict[str, float] = field(default_factory=dict)
    operational_impact: Dict[str, str] = field(default_factory=dict)
    reputation_impact: Dict[str, float] = field(default_factory=dict)
    stakeholder_impact: Dict[str, List[str]] = field(default_factory=dict)
    timeline_impact: Dict[str, float] = field(default_factory=dict)
    recovery_estimate_seconds: float = 0.0
    cascading_risks: List[str] = field(default_factory=list)
    mitigation_urgency: str = "medium"
    assessed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PotentialCrisis:
    """Potential crisis identified by early warning system"""
    id: str = field(default_factory=lambda: str(uuid4()))
    signals: List[Signal] = field(default_factory=list)
    probability: float = 0.0
    potential_type: CrisisType = CrisisType.SYSTEM_OUTAGE
    potential_severity: SeverityLevel = SeverityLevel.MEDIUM
    confidence: float = 0.0
    recommendation: str = ""
    detected_at: datetime = field(default_factory=datetime.utcnow)