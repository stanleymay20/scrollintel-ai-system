"""
Legal compliance API routes for ScrollIntel.
Handles legal documents, consent management, and GDPR compliance.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...models.legal_models import (
    LegalDocumentResponse, ConsentRequest, ConsentResponse,
    DataExportRequestModel, DataExportResponse, CookieSettings,
    PrivacySettings, ComplianceReport
)
from ...core.legal_compliance_manager import LegalComplianceManager
from ...security.auth import get_current_user

router = APIRouter(prefix="/api/legal", tags=["legal"])

# Initialize compliance manager
compliance_manager = LegalComplianceManager()

class LegalDocumentCreate(BaseModel):
    """Model for creating legal documents."""
    document_type: str
    title: str
    content: str
    version: str
    effective_date: datetime
    metadata: Optional[Dict[str, Any]] = {}

@router.get("/documents/{document_type}", response_model=LegalDocumentResponse)
async def get_legal_document(
    document_type: str,
    version: Optional[str] = None
):
    """Get legal document by type and optional version."""
    try:
        document = await compliance_manager.get_legal_document(document_type, version)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Legal document '{document_type}' not found"
            )
        
        return LegalDocumentResponse(**document)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents", response_model=Dict[str, Any])
async def create_legal_document(
    document: LegalDocumentCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new legal document (admin only)."""
    try:
        # Check if user has admin permissions
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        result = await compliance_manager.create_legal_document(
            document_type=document.document_type,
            title=document.title,
            content=document.content,
            version=document.version,
            effective_date=document.effective_date,
            metadata=document.metadata
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/terms-of-service", response_model=LegalDocumentResponse)
async def get_terms_of_service(version: Optional[str] = None):
    """Get current terms of service."""
    return await get_legal_document("terms_of_service", version)

@router.get("/privacy-policy", response_model=LegalDocumentResponse)
async def get_privacy_policy(version: Optional[str] = None):
    """Get current privacy policy."""
    return await get_legal_document("privacy_policy", version)

@router.get("/cookie-policy", response_model=LegalDocumentResponse)
async def get_cookie_policy(version: Optional[str] = None):
    """Get current cookie policy."""
    return await get_legal_document("cookie_policy", version)

@router.post("/consent", response_model=ConsentResponse)
async def record_consent(
    consent: ConsentRequest,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Record user consent."""
    try:
        user_id = current_user.get("user_id") or current_user.get("id")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found")
        
        # Get client information
        ip_address = request.client.host
        user_agent = request.headers.get("user-agent")
        
        result = await compliance_manager.record_user_consent(
            user_id=str(user_id),
            consent_type=consent.consent_type,
            consent_given=consent.consent_given,
            ip_address=ip_address,
            user_agent=user_agent,
            document_version=consent.document_version
        )
        
        return ConsentResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/consent", response_model=List[ConsentResponse])
async def get_user_consents(current_user: dict = Depends(get_current_user)):
    """Get all user consents."""
    try:
        user_id = current_user.get("user_id") or current_user.get("id")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found")
        
        consents = await compliance_manager.get_user_consents(str(user_id))
        return [ConsentResponse(**consent) for consent in consents]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cookie-consent")
async def set_cookie_consent(
    settings: CookieSettings,
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Set cookie consent preferences."""
    try:
        user_id = None
        if current_user:
            user_id = str(current_user.get("user_id") or current_user.get("id"))
        
        # Get client information
        ip_address = request.client.host
        user_agent = request.headers.get("user-agent")
        
        # Record each consent type
        consent_types = {
            "analytics": settings.analytics,
            "marketing": settings.marketing,
            "preferences": settings.preferences
        }
        
        results = []
        for consent_type, consent_given in consent_types.items():
            if user_id:
                result = await compliance_manager.record_user_consent(
                    user_id=user_id,
                    consent_type=f"cookies_{consent_type}",
                    consent_given=consent_given,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                results.append(result)
        
        return {
            "status": "success",
            "message": "Cookie preferences saved",
            "consents_recorded": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/privacy-settings")
async def update_privacy_settings(
    settings: PrivacySettings,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Update user privacy settings."""
    try:
        user_id = str(current_user.get("user_id") or current_user.get("id"))
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found")
        
        # Get client information
        ip_address = request.client.host
        user_agent = request.headers.get("user-agent")
        
        # Record privacy consents
        privacy_consents = {
            "data_processing": settings.data_processing_consent,
            "marketing_emails": settings.marketing_emails,
            "analytics_tracking": settings.analytics_tracking,
            "third_party_sharing": settings.third_party_sharing
        }
        
        results = []
        for consent_type, consent_given in privacy_consents.items():
            result = await compliance_manager.record_user_consent(
                user_id=user_id,
                consent_type=f"privacy_{consent_type}",
                consent_given=consent_given,
                ip_address=ip_address,
                user_agent=user_agent
            )
            results.append(result)
        
        return {
            "status": "success",
            "message": "Privacy settings updated",
            "consents_recorded": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data-export", response_model=DataExportResponse)
async def request_data_export(
    export_request: DataExportRequestModel,
    current_user: dict = Depends(get_current_user)
):
    """Request data export (GDPR Article 20)."""
    try:
        user_id = str(current_user.get("user_id") or current_user.get("id"))
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found")
        
        result = await compliance_manager.request_data_export(
            user_id=user_id,
            request_type=export_request.request_type
        )
        
        return DataExportResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data-deletion", response_model=DataExportResponse)
async def request_data_deletion(
    current_user: dict = Depends(get_current_user)
):
    """Request data deletion (GDPR Article 17 - Right to be Forgotten)."""
    try:
        user_id = str(current_user.get("user_id") or current_user.get("id"))
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found")
        
        result = await compliance_manager.request_data_export(
            user_id=user_id,
            request_type="delete"
        )
        
        return DataExportResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data-export/{request_id}/download")
async def download_data_export(
    request_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Download data export file."""
    try:
        user_id = str(current_user.get("user_id") or current_user.get("id"))
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found")
        
        # Get export request
        from ...models.legal_models import DataExportRequest
        
        with compliance_manager.get_db() as db:
            export_request = db.query(DataExportRequest).filter(
                DataExportRequest.id == request_id,
                DataExportRequest.user_id == user_id,
                DataExportRequest.status == "completed"
            ).first()
            
            if not export_request or not export_request.export_file_path:
                raise HTTPException(status_code=404, detail="Export file not found")
            
            return FileResponse(
                path=export_request.export_file_path,
                filename=f"scrollintel_data_export_{user_id}.zip",
                media_type="application/zip"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/compliance-report", response_model=ComplianceReport)
async def get_compliance_report(
    start_date: datetime,
    end_date: datetime,
    current_user: dict = Depends(get_current_user)
):
    """Get compliance report (admin only)."""
    try:
        # Check if user has admin permissions
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        report = await compliance_manager.generate_compliance_report(
            start_date=start_date,
            end_date=end_date
        )
        
        return report
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup-exports")
async def cleanup_expired_exports(
    background_tasks: BackgroundTasks,
    retention_days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    """Clean up expired export files (admin only)."""
    try:
        # Check if user has admin permissions
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        background_tasks.add_task(
            compliance_manager.cleanup_expired_exports,
            retention_days
        )
        
        return {
            "status": "success",
            "message": f"Cleanup scheduled for exports older than {retention_days} days"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def legal_health_check():
    """Health check for legal compliance system."""
    try:
        # Basic health check
        test_document = await compliance_manager.get_legal_document("terms_of_service")
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "legal_documents_available": test_document is not None,
            "compliance_system": "operational"
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }