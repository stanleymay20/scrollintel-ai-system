"""
API Routes for Audit and Compliance System

This module provides REST API endpoints for audit logging, compliance checking,
access control, and change approval workflows.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field

from ...core.audit_logger import audit_logger, AuditAction
from ...core.compliance_manager import compliance_manager
from ...core.access_control import access_control_manager, Permission
from ...core.change_approval import change_approval_manager
from ...models.audit_models import (
    AuditLogResponse, ComplianceCheckResponse, ComplianceReportResponse,
    AccessControlResponse, ChangeApprovalResponse, RiskLevel, ApprovalStatus,
    ComplianceStatus
)
from ...security.auth import get_current_user


router = APIRouter(prefix="/api/v1/audit-compliance", tags=["audit-compliance"])
security = HTTPBearer()


# Request/Response Models
class AuditLogRequest(BaseModel):
    action: str
    resource_type: str
    resource_id: str
    resource_name: Optional[str] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    changes_summary: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    risk_level: RiskLevel = RiskLevel.LOW


class ComplianceCheckRequest(BaseModel):
    audit_log_ids: List[str]
    rules: Optional[List[str]] = None


class AccessControlRequest(BaseModel):
    user_id: str
    user_email: str
    role: str
    custom_permissions: Optional[List[str]] = None
    resource_restrictions: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None


class ChangeApprovalRequest(BaseModel):
    prompt_id: str
    change_type: str
    change_description: str
    change_justification: str
    proposed_changes: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class ApprovalDecisionRequest(BaseModel):
    decision: str = Field(..., regex="^(approve|reject)$")
    comments: Optional[str] = None


class ComplianceReportRequest(BaseModel):
    report_type: str = Field(..., regex="^(audit_summary|violations|compliance_status)$")
    report_name: str
    start_date: datetime
    end_date: datetime
    filters: Optional[Dict[str, Any]] = None
    file_format: str = Field(default="json", regex="^(json|csv)$")


# Audit Logging Endpoints
@router.post("/audit-logs", response_model=Dict[str, str])
async def create_audit_log(
    request: AuditLogRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Create a new audit log entry"""
    try:
        audit_id = audit_logger.log_action(
            user_id=current_user["user_id"],
            user_email=current_user["email"],
            action=AuditAction(request.action),
            resource_type=request.resource_type,
            resource_id=request.resource_id,
            resource_name=request.resource_name,
            old_values=request.old_values,
            new_values=request.new_values,
            changes_summary=request.changes_summary,
            context=request.context,
            metadata=request.metadata,
            risk_level=request.risk_level
        )
        
        return {"audit_id": audit_id}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/audit-logs", response_model=List[AuditLogResponse])
async def get_audit_logs(
    resource_id: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    current_user: Dict = Depends(get_current_user)
):
    """Get audit logs with filtering options"""
    # Check permission
    if not access_control_manager.check_permission(
        current_user["user_id"], Permission.ADMIN_AUDIT_READ
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        audit_action_enum = AuditAction(action) if action else None
        
        audit_logs = audit_logger.get_audit_trail(
            resource_id=resource_id,
            resource_type=resource_type,
            user_id=user_id,
            action=audit_action_enum,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )
        
        return audit_logs
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/audit-logs/search", response_model=List[AuditLogResponse])
async def search_audit_logs(
    query: str = Query(..., min_length=3),
    filters: Optional[Dict[str, Any]] = Body(None),
    limit: int = Query(100, le=1000),
    current_user: Dict = Depends(get_current_user)
):
    """Search audit logs with text search"""
    # Check permission
    if not access_control_manager.check_permission(
        current_user["user_id"], Permission.ADMIN_AUDIT_READ
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        audit_logs = audit_logger.search_audit_logs(
            search_query=query,
            filters=filters,
            limit=limit
        )
        
        return audit_logs
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/audit-logs/user-activity/{user_id}", response_model=Dict[str, Any])
async def get_user_activity(
    user_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """Get comprehensive user activity summary"""
    # Check permission (users can view their own activity, admins can view any)
    if (current_user["user_id"] != user_id and 
        not access_control_manager.check_permission(
            current_user["user_id"], Permission.ADMIN_AUDIT_READ
        )):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        activity = audit_logger.get_user_activity(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return activity
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Compliance Management Endpoints
@router.post("/compliance-checks", response_model=Dict[str, List[str]])
async def run_compliance_checks(
    request: ComplianceCheckRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Run compliance checks on audit log entries"""
    # Check permission
    if not access_control_manager.check_permission(
        current_user["user_id"], Permission.ADMIN_COMPLIANCE_MANAGE
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        all_check_ids = []
        
        for audit_log_id in request.audit_log_ids:
            check_ids = compliance_manager.run_compliance_check(
                audit_log_id=audit_log_id,
                rules=request.rules
            )
            all_check_ids.extend(check_ids)
        
        return {"compliance_check_ids": all_check_ids}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/compliance-checks/bulk", response_model=Dict[str, Any])
async def run_bulk_compliance_checks(
    start_date: Optional[datetime] = Body(None),
    end_date: Optional[datetime] = Body(None),
    batch_size: int = Body(100, le=1000),
    current_user: Dict = Depends(get_current_user)
):
    """Run compliance checks on multiple audit log entries"""
    # Check permission
    if not access_control_manager.check_permission(
        current_user["user_id"], Permission.ADMIN_COMPLIANCE_MANAGE
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        results = compliance_manager.run_bulk_compliance_check(
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/compliance-status", response_model=Dict[str, Any])
async def get_compliance_status(
    resource_id: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """Get current compliance status"""
    # Check permission
    if not access_control_manager.check_permission(
        current_user["user_id"], Permission.ADMIN_COMPLIANCE_MANAGE
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        status = compliance_manager.get_compliance_status(
            resource_id=resource_id,
            resource_type=resource_type
        )
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/compliance-reports", response_model=Dict[str, str])
async def generate_compliance_report(
    request: ComplianceReportRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Generate comprehensive compliance report"""
    # Check permission
    if not access_control_manager.check_permission(
        current_user["user_id"], Permission.ADMIN_COMPLIANCE_MANAGE
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        report_id = compliance_manager.generate_compliance_report(
            report_type=request.report_type,
            report_name=request.report_name,
            start_date=request.start_date,
            end_date=request.end_date,
            filters=request.filters,
            generated_by=current_user["user_id"],
            file_format=request.file_format
        )
        
        return {"report_id": report_id}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Access Control Endpoints
@router.post("/access-control", response_model=Dict[str, str])
async def grant_user_access(
    request: AccessControlRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Grant access permissions to a user"""
    # Check permission
    if not access_control_manager.check_permission(
        current_user["user_id"], Permission.ADMIN_USER_MANAGE
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        access_id = access_control_manager.grant_access(
            user_id=request.user_id,
            user_email=request.user_email,
            role=request.role,
            granted_by=current_user["user_id"],
            custom_permissions=set(request.custom_permissions) if request.custom_permissions else None,
            resource_restrictions=request.resource_restrictions,
            expires_at=request.expires_at
        )
        
        return {"access_id": access_id}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/access-control/{user_id}", response_model=Dict[str, bool])
async def revoke_user_access(
    user_id: str,
    reason: Optional[str] = Body(None),
    current_user: Dict = Depends(get_current_user)
):
    """Revoke user access permissions"""
    # Check permission
    if not access_control_manager.check_permission(
        current_user["user_id"], Permission.ADMIN_USER_MANAGE
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        success = access_control_manager.revoke_access(
            user_id=user_id,
            revoked_by=current_user["user_id"],
            reason=reason
        )
        
        return {"revoked": success}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/access-control", response_model=List[AccessControlResponse])
async def list_user_access(
    include_inactive: bool = Query(False),
    role_filter: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    current_user: Dict = Depends(get_current_user)
):
    """List all user access entries"""
    # Check permission
    if not access_control_manager.check_permission(
        current_user["user_id"], Permission.ADMIN_USER_MANAGE
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        access_entries = access_control_manager.list_user_access(
            include_inactive=include_inactive,
            role_filter=role_filter,
            limit=limit,
            offset=offset
        )
        
        return access_entries
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/access-control/permissions/{user_id}", response_model=Dict[str, Any])
async def get_user_permissions(
    user_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get comprehensive user permissions and access info"""
    # Users can view their own permissions, admins can view any
    if (current_user["user_id"] != user_id and 
        not access_control_manager.check_permission(
            current_user["user_id"], Permission.ADMIN_USER_MANAGE
        )):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        permissions = access_control_manager.get_user_permissions(user_id)
        return permissions
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/access-control/roles", response_model=List[Dict[str, Any]])
async def list_available_roles(
    current_user: Dict = Depends(get_current_user)
):
    """List all available roles"""
    try:
        roles = access_control_manager.list_available_roles()
        return roles
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Change Approval Endpoints
@router.post("/change-approvals", response_model=Dict[str, str])
async def request_change_approval(
    request: ChangeApprovalRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Request approval for a prompt change"""
    try:
        approval_id = change_approval_manager.request_approval(
            prompt_id=request.prompt_id,
            requester_id=current_user["user_id"],
            requester_email=current_user["email"],
            change_type=request.change_type,
            change_description=request.change_description,
            change_justification=request.change_justification,
            proposed_changes=request.proposed_changes,
            context=request.context
        )
        
        return {"approval_id": approval_id}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/change-approvals/{approval_id}/decision", response_model=Dict[str, bool])
async def make_approval_decision(
    approval_id: str,
    request: ApprovalDecisionRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Approve or reject a change request"""
    try:
        if request.decision == "approve":
            success = change_approval_manager.approve_change(
                approval_id=approval_id,
                approver_id=current_user["user_id"],
                approver_email=current_user["email"],
                comments=request.comments
            )
        else:  # reject
            if not request.comments:
                raise HTTPException(
                    status_code=400, 
                    detail="Comments are required for rejection"
                )
            success = change_approval_manager.reject_change(
                approval_id=approval_id,
                approver_id=current_user["user_id"],
                approver_email=current_user["email"],
                comments=request.comments
            )
        
        return {"success": success}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/change-approvals/pending", response_model=List[ChangeApprovalResponse])
async def get_pending_approvals(
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    current_user: Dict = Depends(get_current_user)
):
    """Get pending approval requests"""
    # Check permission
    if not access_control_manager.check_permission(
        current_user["user_id"], Permission.ADMIN_COMPLIANCE_MANAGE
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        approvals = change_approval_manager.get_pending_approvals(
            approver_id=current_user["user_id"],
            limit=limit,
            offset=offset
        )
        
        return approvals
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/change-approvals/history", response_model=List[ChangeApprovalResponse])
async def get_approval_history(
    prompt_id: Optional[str] = Query(None),
    requester_id: Optional[str] = Query(None),
    approver_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    current_user: Dict = Depends(get_current_user)
):
    """Get approval request history"""
    # Check permission
    if not access_control_manager.check_permission(
        current_user["user_id"], Permission.ADMIN_COMPLIANCE_MANAGE
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        status_enum = ApprovalStatus(status) if status else None
        
        approvals = change_approval_manager.get_approval_history(
            prompt_id=prompt_id,
            requester_id=requester_id,
            approver_id=approver_id,
            status=status_enum,
            limit=limit,
            offset=offset
        )
        
        return approvals
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Maintenance Endpoints
@router.post("/maintenance/cleanup-expired", response_model=Dict[str, int])
async def cleanup_expired_entries(
    current_user: Dict = Depends(get_current_user)
):
    """Clean up expired access entries and approval requests"""
    # Check permission
    if not access_control_manager.check_permission(
        current_user["user_id"], Permission.ADMIN_SYSTEM_CONFIG
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        expired_access = access_control_manager.cleanup_expired_access()
        expired_approvals = change_approval_manager.cleanup_expired_requests()
        
        return {
            "expired_access_entries": expired_access,
            "expired_approval_requests": expired_approvals
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))