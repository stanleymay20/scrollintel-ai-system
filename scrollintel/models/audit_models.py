"""
Audit and Compliance Models for Advanced Prompt Management System

This module defines the data models for audit logging, compliance tracking,
access control, and change approval workflows.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from sqlalchemy import Column, String, DateTime, Text, JSON, Boolean, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

Base = declarative_base()


class AuditAction(str, Enum):
    """Enumeration of auditable actions"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    VIEW = "view"
    EXPORT = "export"
    IMPORT = "import"
    APPROVE = "approve"
    REJECT = "reject"
    PROMOTE = "promote"
    ROLLBACK = "rollback"


class ComplianceStatus(str, Enum):
    """Compliance check status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    REQUIRES_APPROVAL = "requires_approval"


class ApprovalStatus(str, Enum):
    """Change approval workflow status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class RiskLevel(str, Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# SQLAlchemy Models
class AuditLog(Base):
    """Comprehensive audit log for all prompt operations"""
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_id = Column(String, nullable=False)
    user_email = Column(String, nullable=False)
    session_id = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    
    # Action details
    action = Column(String, nullable=False)  # AuditAction enum
    resource_type = Column(String, nullable=False)  # prompt, experiment, etc.
    resource_id = Column(String, nullable=False)
    resource_name = Column(String, nullable=True)
    
    # Change details
    old_values = Column(JSON, nullable=True)
    new_values = Column(JSON, nullable=True)
    changes_summary = Column(Text, nullable=True)
    
    # Context and metadata
    context = Column(JSON, nullable=True)
    audit_metadata = Column(JSON, nullable=True)
    
    # Compliance and risk
    compliance_status = Column(String, default=ComplianceStatus.PENDING_REVIEW)
    risk_level = Column(String, default=RiskLevel.LOW)
    
    # Relationships
    compliance_checks = relationship("ComplianceCheck", back_populates="audit_log")


class ComplianceCheck(Base):
    """Automated compliance validation results"""
    __tablename__ = "compliance_checks"
    
    id = Column(String, primary_key=True)
    audit_log_id = Column(String, ForeignKey("audit_logs.id"), nullable=False)
    check_name = Column(String, nullable=False)
    check_type = Column(String, nullable=False)  # policy, regulation, standard
    status = Column(String, nullable=False)  # ComplianceStatus enum
    
    # Check details
    rule_definition = Column(JSON, nullable=False)
    check_result = Column(JSON, nullable=False)
    violation_details = Column(Text, nullable=True)
    remediation_suggestions = Column(Text, nullable=True)
    
    # Timestamps
    checked_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    resolved_at = Column(DateTime, nullable=True)
    
    # Relationships
    audit_log = relationship("AuditLog", back_populates="compliance_checks")


class ChangeApproval(Base):
    """Change approval workflow tracking"""
    __tablename__ = "change_approvals"
    
    id = Column(String, primary_key=True)
    prompt_id = Column(String, nullable=False)
    requester_id = Column(String, nullable=False)
    requester_email = Column(String, nullable=False)
    
    # Change details
    change_type = Column(String, nullable=False)
    change_description = Column(Text, nullable=False)
    change_justification = Column(Text, nullable=False)
    proposed_changes = Column(JSON, nullable=False)
    
    # Approval workflow
    status = Column(String, default=ApprovalStatus.PENDING)
    approver_id = Column(String, nullable=True)
    approver_email = Column(String, nullable=True)
    approval_comments = Column(Text, nullable=True)
    
    # Risk assessment
    risk_level = Column(String, default=RiskLevel.LOW)
    risk_assessment = Column(JSON, nullable=True)
    
    # Timestamps
    requested_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reviewed_at = Column(DateTime, nullable=True)
    approved_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)


class AccessControl(Base):
    """User access permissions and roles"""
    __tablename__ = "access_controls"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    user_email = Column(String, nullable=False)
    
    # Role and permissions
    role = Column(String, nullable=False)
    permissions = Column(JSON, nullable=False)
    resource_restrictions = Column(JSON, nullable=True)
    
    # Access metadata
    granted_by = Column(String, nullable=False)
    granted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Compliance tracking
    last_access = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0, nullable=False)


class ComplianceReport(Base):
    """Generated compliance reports"""
    __tablename__ = "compliance_reports"
    
    id = Column(String, primary_key=True)
    report_type = Column(String, nullable=False)
    report_name = Column(String, nullable=False)
    
    # Report parameters
    date_range_start = Column(DateTime, nullable=False)
    date_range_end = Column(DateTime, nullable=False)
    filters = Column(JSON, nullable=True)
    
    # Report content
    summary = Column(JSON, nullable=False)
    detailed_findings = Column(JSON, nullable=False)
    recommendations = Column(JSON, nullable=True)
    
    # Report metadata
    generated_by = Column(String, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    file_path = Column(String, nullable=True)
    file_format = Column(String, nullable=False)


# Pydantic Models for API
class AuditLogCreate(BaseModel):
    """Create audit log entry"""
    user_id: str
    user_email: str
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    action: AuditAction
    resource_type: str
    resource_id: str
    resource_name: Optional[str] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    changes_summary: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class AuditLogResponse(BaseModel):
    """Audit log response model"""
    id: str
    timestamp: datetime
    user_id: str
    user_email: str
    action: str
    resource_type: str
    resource_id: str
    resource_name: Optional[str]
    changes_summary: Optional[str]
    compliance_status: str
    risk_level: str
    
    class Config:
        from_attributes = True


class ComplianceCheckCreate(BaseModel):
    """Create compliance check"""
    audit_log_id: str
    check_name: str
    check_type: str
    rule_definition: Dict[str, Any]
    check_result: Dict[str, Any]
    violation_details: Optional[str] = None
    remediation_suggestions: Optional[str] = None


class ComplianceCheckResponse(BaseModel):
    """Compliance check response model"""
    id: str
    check_name: str
    check_type: str
    status: str
    violation_details: Optional[str]
    remediation_suggestions: Optional[str]
    checked_at: datetime
    
    class Config:
        from_attributes = True


class ChangeApprovalCreate(BaseModel):
    """Create change approval request"""
    prompt_id: str
    requester_id: str
    requester_email: str
    change_type: str
    change_description: str
    change_justification: str
    proposed_changes: Dict[str, Any]
    risk_level: RiskLevel = RiskLevel.LOW
    risk_assessment: Optional[Dict[str, Any]] = None


class ChangeApprovalResponse(BaseModel):
    """Change approval response model"""
    id: str
    prompt_id: str
    requester_email: str
    change_type: str
    change_description: str
    status: str
    risk_level: str
    requested_at: datetime
    reviewed_at: Optional[datetime]
    approver_email: Optional[str]
    approval_comments: Optional[str]
    
    class Config:
        from_attributes = True


class AccessControlCreate(BaseModel):
    """Create access control entry"""
    user_id: str
    user_email: str
    role: str
    permissions: Dict[str, Any]
    resource_restrictions: Optional[Dict[str, Any]] = None
    granted_by: str
    expires_at: Optional[datetime] = None


class AccessControlResponse(BaseModel):
    """Access control response model"""
    id: str
    user_id: str
    user_email: str
    role: str
    permissions: Dict[str, Any]
    is_active: bool
    granted_at: datetime
    expires_at: Optional[datetime]
    last_access: Optional[datetime]
    access_count: int
    
    class Config:
        from_attributes = True


class ComplianceReportCreate(BaseModel):
    """Create compliance report"""
    report_type: str
    report_name: str
    date_range_start: datetime
    date_range_end: datetime
    filters: Optional[Dict[str, Any]] = None
    generated_by: str
    file_format: str = "json"


class ComplianceReportResponse(BaseModel):
    """Compliance report response model"""
    id: str
    report_type: str
    report_name: str
    date_range_start: datetime
    date_range_end: datetime
    summary: Dict[str, Any]
    generated_by: str
    generated_at: datetime
    file_path: Optional[str]
    file_format: str
    
    class Config:
        from_attributes = True