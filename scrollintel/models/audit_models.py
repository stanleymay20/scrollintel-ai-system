"""
Audit and compliance models for prompt management system.
"""
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, DateTime, Text, JSON, Boolean, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel

Base = declarative_base()


class AuditAction(str, Enum):
    """Audit action types."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    VIEW = "view"
    EXPORT = "export"
    IMPORT = "import"
    APPROVE = "approve"
    REJECT = "reject"
    ROLLBACK = "rollback"


class ComplianceStatus(str, Enum):
    """Compliance status types."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    REQUIRES_APPROVAL = "requires_approval"


class AuditLog(Base):
    """Audit log entry for tracking all prompt operations."""
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_id = Column(String, nullable=False)
    user_email = Column(String, nullable=False)
    action = Column(String, nullable=False)  # AuditAction enum
    resource_type = Column(String, nullable=False)  # prompt, template, experiment, etc.
    resource_id = Column(String, nullable=False)
    resource_name = Column(String)
    old_values = Column(JSON)  # Previous state
    new_values = Column(JSON)  # New state
    changes = Column(JSON)  # Specific changes made
    ip_address = Column(String)
    user_agent = Column(String)
    session_id = Column(String)
    compliance_tags = Column(JSON)  # Compliance-related metadata
    risk_level = Column(String, default="low")  # low, medium, high, critical
    approval_required = Column(Boolean, default=False)
    approved_by = Column(String)
    approved_at = Column(DateTime)
    audit_metadata = Column(JSON)  # Additional context


class ComplianceRule(Base):
    """Compliance rules for prompt management."""
    __tablename__ = "compliance_rules"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    rule_type = Column(String, nullable=False)  # content, access, approval, retention
    conditions = Column(JSON, nullable=False)  # Rule conditions
    actions = Column(JSON, nullable=False)  # Actions to take
    severity = Column(String, default="medium")  # low, medium, high, critical
    enabled = Column(Boolean, default=True)
    created_by = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ComplianceViolation(Base):
    """Compliance violations detected by the system."""
    __tablename__ = "compliance_violations"
    
    id = Column(String, primary_key=True)
    rule_id = Column(String, ForeignKey("compliance_rules.id"), nullable=False)
    resource_type = Column(String, nullable=False)
    resource_id = Column(String, nullable=False)
    violation_type = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(String, nullable=False)
    status = Column(String, default="open")  # open, investigating, resolved, false_positive
    detected_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    resolved_by = Column(String)
    resolution_notes = Column(Text)
    violation_metadata = Column(JSON)
    
    # Relationship
    rule = relationship("ComplianceRule")


class AccessControl(Base):
    """Access control for prompt resources."""
    __tablename__ = "access_controls"
    
    id = Column(String, primary_key=True)
    resource_type = Column(String, nullable=False)
    resource_id = Column(String, nullable=False)
    user_id = Column(String)
    role = Column(String)
    team_id = Column(String)
    permissions = Column(JSON, nullable=False)  # read, write, delete, approve, etc.
    conditions = Column(JSON)  # Time-based, IP-based, etc.
    granted_by = Column(String, nullable=False)
    granted_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    revoked_at = Column(DateTime)
    revoked_by = Column(String)


class ChangeApproval(Base):
    """Change approval workflow for sensitive prompts."""
    __tablename__ = "change_approvals"
    
    id = Column(String, primary_key=True)
    resource_type = Column(String, nullable=False)
    resource_id = Column(String, nullable=False)
    change_description = Column(Text, nullable=False)
    proposed_changes = Column(JSON, nullable=False)
    requested_by = Column(String, nullable=False)
    requested_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")  # pending, approved, rejected, cancelled
    approver_id = Column(String)
    approved_at = Column(DateTime)
    rejection_reason = Column(Text)
    approval_notes = Column(Text)
    priority = Column(String, default="normal")  # low, normal, high, urgent
    deadline = Column(DateTime)
    approval_metadata = Column(JSON)


# Pydantic models for API
class AuditLogCreate(BaseModel):
    """Create audit log entry."""
    user_id: str
    user_email: str
    action: AuditAction
    resource_type: str
    resource_id: str
    resource_name: Optional[str] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    changes: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    compliance_tags: Optional[List[str]] = None
    risk_level: str = "low"
    audit_metadata: Optional[Dict[str, Any]] = None


class AuditLogResponse(BaseModel):
    """Audit log response."""
    id: str
    timestamp: datetime
    user_id: str
    user_email: str
    action: str
    resource_type: str
    resource_id: str
    resource_name: Optional[str]
    changes: Optional[Dict[str, Any]]
    risk_level: str
    compliance_tags: Optional[List[str]]
    
    model_config = {"from_attributes": True}


class ComplianceRuleCreate(BaseModel):
    """Create compliance rule."""
    name: str
    description: Optional[str] = None
    rule_type: str
    conditions: Dict[str, Any]
    actions: Dict[str, Any]
    severity: str = "medium"
    enabled: bool = True


class ComplianceRuleResponse(BaseModel):
    """Compliance rule response."""
    id: str
    name: str
    description: Optional[str]
    rule_type: str
    conditions: Dict[str, Any]
    actions: Dict[str, Any]
    severity: str
    enabled: bool
    created_at: datetime
    
    model_config = {"from_attributes": True}


class ComplianceViolationResponse(BaseModel):
    """Compliance violation response."""
    id: str
    rule_id: str
    resource_type: str
    resource_id: str
    violation_type: str
    description: str
    severity: str
    status: str
    detected_at: datetime
    
    model_config = {"from_attributes": True}


class AccessControlCreate(BaseModel):
    """Create access control."""
    resource_type: str
    resource_id: str
    user_id: Optional[str] = None
    role: Optional[str] = None
    team_id: Optional[str] = None
    permissions: List[str]
    conditions: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None


class ChangeApprovalCreate(BaseModel):
    """Create change approval request."""
    resource_type: str
    resource_id: str
    change_description: str
    proposed_changes: Dict[str, Any]
    priority: str = "normal"
    deadline: Optional[datetime] = None


class ChangeApprovalResponse(BaseModel):
    """Change approval response."""
    id: str
    resource_type: str
    resource_id: str
    change_description: str
    requested_by: str
    requested_at: datetime
    status: str
    priority: str
    deadline: Optional[datetime]
    
    model_config = {"from_attributes": True}


class ComplianceReport(BaseModel):
    """Compliance report."""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    total_actions: int
    violations_count: int
    pending_approvals: int
    compliance_score: float
    risk_summary: Dict[str, int]
    top_violations: List[ComplianceViolationResponse]
    recommendations: List[str]