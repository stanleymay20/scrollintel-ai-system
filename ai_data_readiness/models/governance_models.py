"""Data governance models for the AI Data Readiness Platform."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import uuid


class AccessLevel(Enum):
    """Access levels for data governance."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"


class PolicyType(Enum):
    """Types of governance policies."""
    DATA_CLASSIFICATION = "data_classification"
    ACCESS_CONTROL = "access_control"
    DATA_RETENTION = "data_retention"
    DATA_QUALITY = "data_quality"
    PRIVACY_PROTECTION = "privacy_protection"
    COMPLIANCE = "compliance"


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class AuditEventType(Enum):
    """Types of audit events."""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    POLICY_CHANGE = "policy_change"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    COMPLIANCE_CHECK = "compliance_check"


class PolicyStatus(Enum):
    """Status of governance policies."""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"


@dataclass
class User:
    """User model for governance."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    full_name: str = ""
    department: str = ""
    role: str = ""
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    permissions: Set[str] = field(default_factory=set)


@dataclass
class Role:
    """Role model for RBAC."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    permissions: Set[str] = field(default_factory=set)
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""


@dataclass
class Permission:
    """Permission model."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    resource_type: str = ""  # dataset, report, model, etc.
    action: str = ""  # read, write, delete, execute, etc.
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataCatalogEntry:
    """Data catalog entry with governance metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_id: str = ""
    name: str = ""
    description: str = ""
    classification: DataClassification = DataClassification.INTERNAL
    owner: str = ""
    steward: str = ""
    business_glossary_terms: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    schema_info: Dict[str, Any] = field(default_factory=dict)
    lineage_info: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    usage_statistics: Dict[str, Any] = field(default_factory=dict)
    retention_policy: Optional[str] = None
    compliance_requirements: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None


@dataclass
class GovernancePolicy:
    """Governance policy definition."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    policy_type: PolicyType = PolicyType.ACCESS_CONTROL
    status: PolicyStatus = PolicyStatus.DRAFT
    rules: List[Dict[str, Any]] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    enforcement_level: str = "strict"  # strict, warning, advisory
    applicable_resources: List[str] = field(default_factory=list)
    created_by: str = ""
    approved_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    effective_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    version: str = "1.0"


@dataclass
class AccessControlEntry:
    """Access control entry for resources."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_id: str = ""
    resource_type: str = ""  # dataset, report, model, etc.
    principal_id: str = ""  # user or role ID
    principal_type: str = ""  # user, role
    access_level: AccessLevel = AccessLevel.READ
    granted_by: str = ""
    granted_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


@dataclass
class AuditEvent:
    """Audit event for tracking activities."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.USER_ACTION
    user_id: str = ""
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class UsageMetrics:
    """Usage metrics for data assets."""
    resource_id: str = ""
    resource_type: str = ""
    access_count: int = 0
    unique_users: int = 0
    last_accessed: Optional[datetime] = None
    most_frequent_user: Optional[str] = None
    access_patterns: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplianceReport:
    """Compliance assessment report."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    report_type: str = ""  # GDPR, CCPA, SOX, etc.
    scope: List[str] = field(default_factory=list)  # resource IDs
    compliance_score: float = 0.0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    assessment_criteria: Dict[str, Any] = field(default_factory=dict)
    generated_by: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataLineageNode:
    """Node in data lineage graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_id: str = ""
    resource_type: str = ""  # dataset, transformation, model, etc.
    name: str = ""
    node_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataLineageEdge:
    """Edge in data lineage graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = ""
    target_node_id: str = ""
    relationship_type: str = ""  # derives_from, transforms_to, etc.
    transformation_details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GovernanceMetrics:
    """Governance metrics and KPIs."""
    total_datasets: int = 0
    classified_datasets: int = 0
    policy_violations: int = 0
    compliance_score: float = 0.0
    data_quality_score: float = 0.0
    access_requests_pending: int = 0
    audit_events_count: int = 0
    active_users: int = 0
    data_stewards: int = 0
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    calculated_at: datetime = field(default_factory=datetime.utcnow)