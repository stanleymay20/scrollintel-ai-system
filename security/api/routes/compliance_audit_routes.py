"""
API Routes for Compliance and Audit Framework
Provides REST endpoints for compliance reporting, audit logging, and privacy controls
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
import uuid

from security.compliance.immutable_audit_logger import ImmutableAuditLogger, AuditEvent
from security.compliance.compliance_reporting import ComplianceReportingEngine, ComplianceFramework, ControlStatus
from security.compliance.evidence_generator import EvidenceGenerator
from security.compliance.violation_detector import ComplianceViolationDetector, ViolationSeverity, ViolationStatus
from security.compliance.data_privacy_controls import DataPrivacyControls, RequestType, DataCategory


router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])
security = HTTPBearer()

# Initialize components
audit_logger = ImmutableAuditLogger()
compliance_engine = ComplianceReportingEngine()
evidence_generator = EvidenceGenerator()
violation_detector = ComplianceViolationDetector()
privacy_controls = DataPrivacyControls()

# Start background services
violation_detector.start_monitoring()
privacy_controls.start_processing()


# Pydantic models for API
class AuditEventRequest(BaseModel):
    event_type: str
    user_id: Optional[str] = None
    resource: str
    action: str
    outcome: str
    details: Dict[str, Any] = {}
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None


class ComplianceReportRequest(BaseModel):
    framework: str
    assessor: str = "API User"
    period_days: int = 90


class EvidencePackageRequest(BaseModel):
    framework: str
    control_ids: Optional[List[str]] = None
    period_days: int = 90


class DataSubjectRequestModel(BaseModel):
    request_type: str
    data_subject_email: EmailStr
    data_subject_name: Optional[str] = None
    requested_categories: Optional[List[str]] = None
    specific_items: Optional[List[str]] = None
    reason: Optional[str] = None
    evidence_of_identity: Optional[Dict[str, Any]] = None


class ConsentRecordModel(BaseModel):
    data_subject_email: EmailStr
    purpose: str
    legal_basis: str
    consent_given: bool
    consent_method: str
    data_categories: List[str]
    retention_period_days: Optional[int] = None


# Audit Logging Endpoints
@router.post("/audit/log-event")
async def log_audit_event(event: AuditEventRequest, token: str = Depends(security)):
    """Log an immutable audit event"""
    try:
        audit_event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().timestamp(),
            event_type=event.event_type,
            user_id=event.user_id,
            resource=event.resource,
            action=event.action,
            outcome=event.outcome,
            details=event.details,
            source_ip=event.source_ip,
            user_agent=event.user_agent,
            session_id=event.session_id
        )
        
        event_id = audit_logger.log_event(audit_event)
        
        return {
            "success": True,
            "event_id": event_id,
            "message": "Audit event logged successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log audit event: {str(e)}")


@router.get("/audit/verify-integrity")
async def verify_audit_integrity(token: str = Depends(security)):
    """Verify blockchain integrity of audit trail"""
    try:
        verification_result = audit_logger.verify_integrity()
        return verification_result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify integrity: {str(e)}")


@router.get("/audit/trail")
async def get_audit_trail(
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    user_id: Optional[str] = None,
    resource: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 1000,
    token: str = Depends(security)
):
    """Retrieve audit trail with filtering options"""
    try:
        trail = audit_logger.get_audit_trail(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            resource=resource,
            event_type=event_type,
            limit=limit
        )
        
        return {
            "success": True,
            "audit_trail": trail,
            "count": len(trail)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit trail: {str(e)}")


@router.get("/audit/stats")
async def get_audit_stats(token: str = Depends(security)):
    """Get audit blockchain statistics"""
    try:
        stats = audit_logger.get_blockchain_stats()
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get audit stats: {str(e)}")


# Compliance Reporting Endpoints
@router.post("/reports/generate")
async def generate_compliance_report(
    request: ComplianceReportRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Generate comprehensive compliance report"""
    try:
        # Validate framework
        try:
            framework = ComplianceFramework(request.framework.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unsupported framework: {request.framework}")
        
        # Generate report in background
        def generate_report():
            return compliance_engine.generate_compliance_report(
                framework=framework,
                assessor=request.assessor,
                period_days=request.period_days
            )
        
        background_tasks.add_task(generate_report)
        
        return {
            "success": True,
            "message": f"Compliance report generation started for {framework.value}",
            "framework": framework.value,
            "estimated_completion": "5-10 minutes"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.get("/reports/{framework}")
async def get_compliance_report(framework: str, token: str = Depends(security)):
    """Get latest compliance report for framework"""
    try:
        # This would retrieve the latest report from database
        # For now, generate a new one
        try:
            framework_enum = ComplianceFramework(framework.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unsupported framework: {framework}")
        
        report = compliance_engine.generate_compliance_report(framework_enum)
        
        return {
            "success": True,
            "report": report.to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")


@router.get("/dashboard")
async def get_compliance_dashboard(token: str = Depends(security)):
    """Get compliance dashboard data"""
    try:
        dashboard = compliance_engine.get_compliance_dashboard()
        return dashboard
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard: {str(e)}")


@router.put("/controls/{control_id}/status")
async def update_control_status(
    control_id: str,
    status: str,
    evidence: Optional[List[str]] = None,
    notes: Optional[str] = None,
    token: str = Depends(security)
):
    """Update compliance control status"""
    try:
        try:
            status_enum = ControlStatus(status.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        compliance_engine.update_control_status(
            control_id=control_id,
            status=status_enum,
            evidence=evidence,
            notes=notes
        )
        
        return {
            "success": True,
            "message": f"Control {control_id} status updated to {status}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update control status: {str(e)}")


# Evidence Generation Endpoints
@router.post("/evidence/generate-package")
async def generate_evidence_package(
    request: EvidencePackageRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Generate evidence package for audit"""
    try:
        def generate_package():
            return evidence_generator.generate_evidence_package(
                framework=request.framework,
                control_ids=request.control_ids,
                period_days=request.period_days
            )
        
        background_tasks.add_task(generate_package)
        
        return {
            "success": True,
            "message": f"Evidence package generation started for {request.framework}",
            "estimated_completion": "10-15 minutes"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate evidence package: {str(e)}")


# Violation Detection Endpoints
@router.get("/violations")
async def get_violations(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    token: str = Depends(security)
):
    """Get compliance violations"""
    try:
        # This would query violations from database with filters
        dashboard = violation_detector.get_violations_dashboard()
        
        return {
            "success": True,
            "violations": dashboard["recent_violations"][:limit],
            "summary": dashboard["metrics"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get violations: {str(e)}")


@router.get("/violations/dashboard")
async def get_violations_dashboard(token: str = Depends(security)):
    """Get violations dashboard data"""
    try:
        dashboard = violation_detector.get_violations_dashboard()
        return dashboard
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get violations dashboard: {str(e)}")


# Data Privacy Endpoints
@router.post("/privacy/data-subject-request")
async def submit_data_subject_request(request: DataSubjectRequestModel):
    """Submit a data subject request (GDPR, CCPA, etc.)"""
    try:
        # Convert string enums
        try:
            request_type = RequestType(request.request_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid request type: {request.request_type}")
        
        requested_categories = []
        if request.requested_categories:
            for cat in request.requested_categories:
                try:
                    requested_categories.append(DataCategory(cat.lower()))
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid data category: {cat}")
        
        request_id = privacy_controls.submit_data_subject_request(
            request_type=request_type,
            data_subject_email=request.data_subject_email,
            data_subject_name=request.data_subject_name,
            requested_categories=requested_categories or None,
            specific_items=request.specific_items,
            reason=request.reason,
            evidence_of_identity=request.evidence_of_identity
        )
        
        return {
            "success": True,
            "request_id": request_id,
            "message": "Data subject request submitted successfully",
            "next_steps": "You will receive a confirmation email shortly"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit request: {str(e)}")


@router.get("/privacy/requests/{request_id}")
async def get_data_subject_request(request_id: str, token: str = Depends(security)):
    """Get data subject request status"""
    try:
        # This would query the request from database
        # For now, return a mock response
        return {
            "success": True,
            "request_id": request_id,
            "status": "in_progress",
            "message": "Request is being processed"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get request: {str(e)}")


@router.get("/privacy/dashboard")
async def get_privacy_dashboard(token: str = Depends(security)):
    """Get privacy controls dashboard"""
    try:
        dashboard = privacy_controls.get_privacy_dashboard()
        return dashboard
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get privacy dashboard: {str(e)}")


@router.post("/privacy/consent")
async def record_consent(consent: ConsentRecordModel):
    """Record consent for data processing"""
    try:
        # Convert string enums
        data_categories = []
        for cat in consent.data_categories:
            try:
                data_categories.append(DataCategory(cat.lower()))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid data category: {cat}")
        
        consent_id = privacy_controls.record_consent(
            data_subject_email=consent.data_subject_email,
            purpose=consent.purpose,
            legal_basis=consent.legal_basis,
            consent_given=consent.consent_given,
            consent_method=consent.consent_method,
            data_categories=data_categories,
            retention_period_days=consent.retention_period_days
        )
        
        return {
            "success": True,
            "consent_id": consent_id,
            "message": "Consent recorded successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record consent: {str(e)}")


@router.delete("/privacy/consent/{consent_id}")
async def withdraw_consent(consent_id: str, withdrawal_method: str = "api"):
    """Withdraw consent"""
    try:
        success = privacy_controls.withdraw_consent(consent_id, withdrawal_method)
        
        if success:
            return {
                "success": True,
                "message": "Consent withdrawn successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Consent record not found or already withdrawn")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to withdraw consent: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for compliance services"""
    try:
        # Check if all services are running
        audit_stats = audit_logger.get_blockchain_stats()
        compliance_dashboard = compliance_engine.get_compliance_dashboard()
        violations_dashboard = violation_detector.get_violations_dashboard()
        privacy_dashboard = privacy_controls.get_privacy_dashboard()
        
        return {
            "status": "healthy",
            "services": {
                "audit_logger": "operational",
                "compliance_engine": "operational",
                "violation_detector": "operational",
                "privacy_controls": "operational"
            },
            "metrics": {
                "total_audit_events": audit_stats.get("total_events", 0),
                "total_compliance_controls": compliance_dashboard["overall_metrics"]["total_controls"],
                "open_violations": violations_dashboard["metrics"]["open_violations"],
                "pending_privacy_requests": privacy_dashboard["metrics"]["pending_requests"]
            }
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }