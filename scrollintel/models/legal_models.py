"""
Legal compliance data models for ScrollIntel.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class LegalDocument(Base):
    """Legal document storage model."""
    __tablename__ = "legal_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    document_type = Column(String(50), nullable=False)  # 'terms', 'privacy', 'cookies', etc.
    version = Column(String(20), nullable=False)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    effective_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    document_metadata = Column(JSON, default={})

class UserConsent(Base):
    """User consent tracking model."""
    __tablename__ = "user_consents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    consent_type = Column(String(50), nullable=False)  # 'cookies', 'marketing', 'analytics'
    consent_given = Column(Boolean, nullable=False)
    consent_date = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    document_version = Column(String(20))
    withdrawal_date = Column(DateTime, nullable=True)

class DataExportRequest(Base):
    """GDPR data export request tracking."""
    __tablename__ = "data_export_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    request_type = Column(String(20), nullable=False)  # 'export', 'delete'
    status = Column(String(20), default='pending')  # 'pending', 'processing', 'completed', 'failed'
    requested_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    export_file_path = Column(String(500), nullable=True)
    verification_token = Column(String(100), nullable=True)

class ComplianceAudit(Base):
    """Compliance audit log model."""
    __tablename__ = "compliance_audits"
    
    id = Column(Integer, primary_key=True, index=True)
    audit_type = Column(String(50), nullable=False)
    user_id = Column(String(100), nullable=True, index=True)
    action = Column(String(100), nullable=False)
    details = Column(JSON, default={})
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45))
    user_agent = Column(String(500))

# Pydantic models for API
class LegalDocumentResponse(BaseModel):
    """Legal document response model."""
    id: int
    document_type: str
    version: str
    title: str
    content: str
    effective_date: datetime
    is_active: bool
    document_metadata: Dict[str, Any] = {}

class ConsentRequest(BaseModel):
    """User consent request model."""
    consent_type: str = Field(..., description="Type of consent (cookies, marketing, analytics)")
    consent_given: bool = Field(..., description="Whether consent is given")
    document_version: Optional[str] = Field(None, description="Version of document consented to")

class ConsentResponse(BaseModel):
    """User consent response model."""
    id: int
    consent_type: str
    consent_given: bool
    consent_date: datetime
    document_version: Optional[str]
    withdrawal_date: Optional[datetime]

class DataExportRequestModel(BaseModel):
    """Data export request model."""
    request_type: str = Field(..., description="Type of request (export, delete)")
    verification_email: str = Field(..., description="Email for verification")

class DataExportResponse(BaseModel):
    """Data export response model."""
    id: int
    request_type: str
    status: str
    requested_at: datetime
    completed_at: Optional[datetime]
    download_url: Optional[str]

class CookieSettings(BaseModel):
    """Cookie consent settings model."""
    necessary: bool = True  # Always true, cannot be disabled
    analytics: bool = False
    marketing: bool = False
    preferences: bool = False

class PrivacySettings(BaseModel):
    """User privacy settings model."""
    data_processing_consent: bool
    marketing_emails: bool = False
    analytics_tracking: bool = False
    third_party_sharing: bool = False
    data_retention_period: str = "2_years"  # Options: 1_year, 2_years, 5_years

class ComplianceReport(BaseModel):
    """Compliance report model."""
    report_type: str
    period_start: datetime
    period_end: datetime
    total_users: int
    consent_stats: Dict[str, int]
    export_requests: int
    deletion_requests: int
    compliance_score: float