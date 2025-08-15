"""
Legal compliance management system for ScrollIntel.
Handles GDPR compliance, data export, deletion, and legal document management.
"""

import os
import json
import zipfile
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..models.legal_models import (
    LegalDocument, UserConsent, DataExportRequest, ComplianceAudit,
    CookieSettings, PrivacySettings, ComplianceReport
)
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from ..core.config import get_settings

class LegalComplianceManager:
    """Manages legal compliance features including GDPR, data export, and consent."""
    
    def __init__(self):
        self.settings = get_settings()
        self.export_directory = Path("data/exports")
        self.export_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection - use SQLite for demo
        database_url = 'sqlite:///legal_compliance.db'
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they don't exist
        from ..models.legal_models import Base
        Base.metadata.create_all(bind=self.engine)
    
    @contextmanager
    def get_db(self):
        """Get database session context manager."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    async def get_legal_document(self, document_type: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get legal document by type and version."""
        try:
            with self.get_db() as db:
                query = db.query(LegalDocument).filter(
                    LegalDocument.document_type == document_type,
                    LegalDocument.is_active == True
                )
                
                if version:
                    query = query.filter(LegalDocument.version == version)
                else:
                    query = query.order_by(LegalDocument.effective_date.desc())
                
                document = query.first()
                
                if document:
                    return {
                        "id": document.id,
                        "document_type": document.document_type,
                        "version": document.version,
                        "title": document.title,
                        "content": document.content,
                        "effective_date": document.effective_date,
                        "metadata": document.document_metadata
                    }
                return None
        except Exception as e:
            print(f"Error getting legal document: {e}")
            return None
    
    async def create_legal_document(self, document_type: str, title: str, content: str, 
                                  version: str, effective_date: datetime, 
                                  metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new legal document."""
        try:
            with self.get_db() as db:
                # Deactivate previous versions
                db.query(LegalDocument).filter(
                    LegalDocument.document_type == document_type
                ).update({"is_active": False})
                
                # Create new document
                document = LegalDocument(
                    document_type=document_type,
                    version=version,
                    title=title,
                    content=content,
                    effective_date=effective_date,
                    document_metadata=metadata or {}
                )
                
                db.add(document)
                db.commit()
                db.refresh(document)
                
                # Log compliance audit
                await self._log_compliance_audit(
                    audit_type="document_management",
                    action=f"created_{document_type}_v{version}",
                    details={"document_id": document.id, "title": title}
                )
                
                return {
                    "id": document.id,
                    "document_type": document.document_type,
                    "version": document.version,
                    "title": document.title,
                    "effective_date": document.effective_date,
                    "status": "created"
                }
        except Exception as e:
            print(f"Error creating legal document: {e}")
            raise
    
    async def record_user_consent(self, user_id: str, consent_type: str, 
                                consent_given: bool, ip_address: str = None,
                                user_agent: str = None, document_version: str = None) -> Dict[str, Any]:
        """Record user consent for various purposes."""
        try:
            with self.get_db() as db:
                # Check if consent already exists
                existing_consent = db.query(UserConsent).filter(
                    and_(
                        UserConsent.user_id == user_id,
                        UserConsent.consent_type == consent_type,
                        UserConsent.withdrawal_date.is_(None)
                    )
                ).first()
                
                if existing_consent:
                    # Update existing consent
                    existing_consent.consent_given = consent_given
                    existing_consent.consent_date = datetime.utcnow()
                    existing_consent.ip_address = ip_address
                    existing_consent.user_agent = user_agent
                    existing_consent.document_version = document_version
                    
                    if not consent_given:
                        existing_consent.withdrawal_date = datetime.utcnow()
                else:
                    # Create new consent record
                    consent = UserConsent(
                        user_id=user_id,
                        consent_type=consent_type,
                        consent_given=consent_given,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        document_version=document_version
                    )
                    db.add(consent)
                    existing_consent = consent
                
                db.commit()
                db.refresh(existing_consent)
                
                # Log compliance audit
                await self._log_compliance_audit(
                    audit_type="consent_management",
                    user_id=user_id,
                    action=f"consent_{consent_type}_{'given' if consent_given else 'withdrawn'}",
                    details={
                        "consent_type": consent_type,
                        "consent_given": consent_given,
                        "document_version": document_version
                    },
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
                return {
                    "id": existing_consent.id,
                    "consent_type": consent_type,
                    "consent_given": consent_given,
                    "consent_date": existing_consent.consent_date,
                    "status": "recorded"
                }
        except Exception as e:
            print(f"Error recording user consent: {e}")
            raise
    
    async def get_user_consents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active consents for a user."""
        try:
            with self.get_db() as db:
                consents = db.query(UserConsent).filter(
                    and_(
                        UserConsent.user_id == user_id,
                        UserConsent.withdrawal_date.is_(None)
                    )
                ).all()
                
                return [
                    {
                        "id": consent.id,
                        "consent_type": consent.consent_type,
                        "consent_given": consent.consent_given,
                        "consent_date": consent.consent_date,
                        "document_version": consent.document_version
                    }
                    for consent in consents
                ]
        except Exception as e:
            print(f"Error getting user consents: {e}")
            return []
    
    async def request_data_export(self, user_id: str, request_type: str = "export") -> Dict[str, Any]:
        """Create a data export request for GDPR compliance."""
        try:
            with self.get_db() as db:
                # Check for existing pending requests
                existing_request = db.query(DataExportRequest).filter(
                    and_(
                        DataExportRequest.user_id == user_id,
                        DataExportRequest.request_type == request_type,
                        DataExportRequest.status.in_(["pending", "processing"])
                    )
                ).first()
                
                if existing_request:
                    return {
                        "id": existing_request.id,
                        "status": "existing_request_pending",
                        "message": "A request is already being processed"
                    }
                
                # Create verification token
                verification_token = hashlib.sha256(
                    f"{user_id}_{request_type}_{datetime.utcnow().isoformat()}".encode()
                ).hexdigest()[:32]
                
                # Create new request
                export_request = DataExportRequest(
                    user_id=user_id,
                    request_type=request_type,
                    verification_token=verification_token
                )
                
                db.add(export_request)
                db.commit()
                db.refresh(export_request)
                
                # Log compliance audit
                await self._log_compliance_audit(
                    audit_type="data_rights",
                    user_id=user_id,
                    action=f"requested_{request_type}",
                    details={
                        "request_id": export_request.id,
                        "request_type": request_type
                    }
                )
                
                # Start processing in background
                asyncio.create_task(self._process_data_request(export_request.id))
                
                return {
                    "id": export_request.id,
                    "request_type": request_type,
                    "status": "pending",
                    "verification_token": verification_token,
                    "message": "Request submitted successfully"
                }
        except Exception as e:
            print(f"Error creating data export request: {e}")
            raise
    
    async def _process_data_request(self, request_id: int):
        """Process data export or deletion request."""
        try:
            with self.get_db() as db:
                request = db.query(DataExportRequest).filter(
                    DataExportRequest.id == request_id
                ).first()
                
                if not request:
                    return
                
                # Update status to processing
                request.status = "processing"
                db.commit()
                
                if request.request_type == "export":
                    # Export user data
                    export_path = await self._export_user_data(request.user_id)
                    request.export_file_path = str(export_path)
                elif request.request_type == "delete":
                    # Delete user data
                    await self._delete_user_data(request.user_id)
                
                # Update status to completed
                request.status = "completed"
                request.completed_at = datetime.utcnow()
                db.commit()
                
                # Log completion
                await self._log_compliance_audit(
                    audit_type="data_rights",
                    user_id=request.user_id,
                    action=f"completed_{request.request_type}",
                    details={
                        "request_id": request_id,
                        "completed_at": request.completed_at.isoformat()
                    }
                )
                
        except Exception as e:
            print(f"Error processing data request {request_id}: {e}")
            # Update status to failed
            with self.get_db() as db:
                request = db.query(DataExportRequest).filter(
                    DataExportRequest.id == request_id
                ).first()
                if request:
                    request.status = "failed"
                    db.commit()
    
    async def _export_user_data(self, user_id: str) -> Path:
        """Export all user data to a ZIP file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        export_filename = f"user_data_export_{user_id}_{timestamp}.zip"
        export_path = self.export_directory / export_filename
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Export user profile data
            user_data = await self._get_user_profile_data(user_id)
            zipf.writestr("profile.json", json.dumps(user_data, indent=2, default=str))
            
            # Export user consents
            consents = await self.get_user_consents(user_id)
            zipf.writestr("consents.json", json.dumps(consents, indent=2, default=str))
            
            # Export user activity logs
            activity_logs = await self._get_user_activity_logs(user_id)
            zipf.writestr("activity_logs.json", json.dumps(activity_logs, indent=2, default=str))
            
            # Export user files and data
            user_files = await self._get_user_files(user_id)
            zipf.writestr("files_metadata.json", json.dumps(user_files, indent=2, default=str))
            
            # Add README with export information
            readme_content = f"""
ScrollIntel Data Export
======================

Export Date: {datetime.utcnow().isoformat()}
User ID: {user_id}

This archive contains all personal data associated with your ScrollIntel account:

- profile.json: Your account profile information
- consents.json: Your consent preferences and history
- activity_logs.json: Your platform activity logs
- files_metadata.json: Metadata about files you've uploaded

This export complies with GDPR Article 20 (Right to Data Portability).

For questions about this export, please contact support@scrollintel.com
"""
            zipf.writestr("README.txt", readme_content)
        
        return export_path
    
    async def _delete_user_data(self, user_id: str):
        """Delete all user data (Right to be Forgotten)."""
        try:
            with self.get_db() as db:
                # Delete user consents
                db.query(UserConsent).filter(UserConsent.user_id == user_id).delete()
                
                # Anonymize audit logs (keep for compliance but remove PII)
                audit_logs = db.query(ComplianceAudit).filter(
                    ComplianceAudit.user_id == user_id
                ).all()
                
                for log in audit_logs:
                    log.user_id = "DELETED_USER"
                    log.details = {"action": "user_data_deleted"}
                
                # Delete user files and data
                await self._delete_user_files(user_id)
                
                db.commit()
                
        except Exception as e:
            print(f"Error deleting user data: {e}")
            raise
    
    async def _get_user_profile_data(self, user_id: str) -> Dict[str, Any]:
        """Get user profile data for export."""
        # This would integrate with your user management system
        return {
            "user_id": user_id,
            "export_note": "User profile data would be retrieved from user management system"
        }
    
    async def _get_user_activity_logs(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user activity logs for export."""
        try:
            with self.get_db() as db:
                logs = db.query(ComplianceAudit).filter(
                    ComplianceAudit.user_id == user_id
                ).order_by(ComplianceAudit.timestamp.desc()).all()
                
                return [
                    {
                        "timestamp": log.timestamp.isoformat(),
                        "audit_type": log.audit_type,
                        "action": log.action,
                        "details": log.details
                    }
                    for log in logs
                ]
        except Exception as e:
            print(f"Error getting user activity logs: {e}")
            return []
    
    async def _get_user_files(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user files metadata for export."""
        # This would integrate with your file management system
        return [
            {
                "note": "User files metadata would be retrieved from file management system",
                "user_id": user_id
            }
        ]
    
    async def _delete_user_files(self, user_id: str):
        """Delete user files from storage."""
        # This would integrate with your file management system
        pass
    
    async def _log_compliance_audit(self, audit_type: str, action: str, 
                                  details: Dict[str, Any] = None,
                                  user_id: str = None, ip_address: str = None,
                                  user_agent: str = None):
        """Log compliance audit event."""
        try:
            with self.get_db() as db:
                audit = ComplianceAudit(
                    audit_type=audit_type,
                    user_id=user_id,
                    action=action,
                    details=details or {},
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
                db.add(audit)
                db.commit()
        except Exception as e:
            print(f"Error logging compliance audit: {e}")
    
    async def generate_compliance_report(self, start_date: datetime, 
                                       end_date: datetime) -> ComplianceReport:
        """Generate compliance report for a given period."""
        try:
            with self.get_db() as db:
                # Get consent statistics
                consent_stats = {}
                consent_types = ["cookies", "analytics", "marketing", "preferences"]
                
                for consent_type in consent_types:
                    given_count = db.query(UserConsent).filter(
                        and_(
                            UserConsent.consent_type == consent_type,
                            UserConsent.consent_given == True,
                            UserConsent.consent_date.between(start_date, end_date)
                        )
                    ).count()
                    
                    consent_stats[f"{consent_type}_given"] = given_count
                
                # Get export/deletion requests
                export_requests = db.query(DataExportRequest).filter(
                    and_(
                        DataExportRequest.request_type == "export",
                        DataExportRequest.requested_at.between(start_date, end_date)
                    )
                ).count()
                
                deletion_requests = db.query(DataExportRequest).filter(
                    and_(
                        DataExportRequest.request_type == "delete",
                        DataExportRequest.requested_at.between(start_date, end_date)
                    )
                ).count()
                
                # Calculate compliance score (simplified)
                total_requests = export_requests + deletion_requests
                completed_requests = db.query(DataExportRequest).filter(
                    and_(
                        DataExportRequest.requested_at.between(start_date, end_date),
                        DataExportRequest.status == "completed"
                    )
                ).count()
                
                compliance_score = (completed_requests / max(total_requests, 1)) * 100
                
                return ComplianceReport(
                    report_type="gdpr_compliance",
                    period_start=start_date,
                    period_end=end_date,
                    total_users=0,  # Would be calculated from user management system
                    consent_stats=consent_stats,
                    export_requests=export_requests,
                    deletion_requests=deletion_requests,
                    compliance_score=compliance_score
                )
                
        except Exception as e:
            print(f"Error generating compliance report: {e}")
            raise
    
    async def cleanup_expired_exports(self, retention_days: int = 30):
        """Clean up expired export files."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            with self.get_db() as db:
                expired_requests = db.query(DataExportRequest).filter(
                    and_(
                        DataExportRequest.completed_at < cutoff_date,
                        DataExportRequest.export_file_path.isnot(None)
                    )
                ).all()
                
                for request in expired_requests:
                    # Delete file
                    if request.export_file_path and os.path.exists(request.export_file_path):
                        os.remove(request.export_file_path)
                    
                    # Clear file path
                    request.export_file_path = None
                
                db.commit()
                
                await self._log_compliance_audit(
                    audit_type="data_retention",
                    action="cleanup_expired_exports",
                    details={"cleaned_count": len(expired_requests)}
                )
                
        except Exception as e:
            print(f"Error cleaning up expired exports: {e}")
