"""
API routes for audit and compliance system.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel

from scrollintel.models.audit_models import (
    AuditLogResponse, ComplianceRuleCreate, ComplianceRuleResponse,
    ComplianceViolationResponse, AccessControlCreate, ChangeApprovalCreate,
    ChangeApprovalResponse, ComplianceReport, AuditAction
)
from scrollintel.core.audit_logger import AuditLogger
from scrollintel.core.compliance_manager import ComplianceManager
from scrollintel.core.access_control import AccessControlManager
from scrollintel.core.change_approval import ChangeApprovalManager
from scrollintel.models.database import get_db


router = APIRouter(prefix="/audit-compliance", tags=["audit", "compliance"])


# Request/Response models
class ComplianceCheckRequest(BaseModel):
    resource_type: str
    resource_data: Dict[str, Any]
    action: str
    user_context: Dict[str, Any]


class ComplianceCheckResponse(BaseModel):
    compliant: bool
    violations: List[Dict[str, Any]]
    actions_required: List[str]
    risk_level: str


class PermissionCheckRequest(BaseModel):
    user_id: str
    resource_type: str
    resource_id: str
    permission: str
    user_roles: Optional[List[str]] = None
    user_teams: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None


class PermissionCheckResponse(BaseModel):
    allowed: bool
    reason: Optional[str] = None
    access_control_id: Optional[str] = None
    granted_by: Optional[str] = None
    permissions: Optional[List[str]] = None


# Audit Log Routes
@router.get("/audit-logs", response_model=List[AuditLogResponse])
async def get_audit_logs(
    resource_type: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """Get audit logs with filters."""
    
    audit_logger = AuditLogger(db)
    
    action_enum = None
    if action:
        try:
            action_enum = AuditAction(action)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}")
    
    logs = audit_logger.get_audit_trail(
        resource_type=resource_type,
        resource_id=resource_id,
        user_id=user_id,
        action=action_enum,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )
    
    return logs


@router.get("/audit-logs/resource/{resource_type}/{resource_id}", response_model=List[AuditLogResponse])
async def get_resource_audit_history(
    resource_type: str,
    resource_id: str,
    db: Session = Depends(get_db)
):
    """Get complete audit history for a specific resource."""
    
    audit_logger = AuditLogger(db)
    logs = audit_logger.get_resource_history(resource_type, resource_id)
    
    return logs


@router.get("/audit-logs/search", response_model=List[AuditLogResponse])
async def search_audit_logs(
    q: str = Query(..., description="Search term"),
    risk_level: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db)
):
    """Search audit logs by content."""
    
    audit_logger = AuditLogger(db)
    
    filters = {}
    if risk_level:
        filters["risk_level"] = risk_level
    if action:
        filters["action"] = action
    
    logs = audit_logger.search_audit_logs(q, filters, limit)
    
    return logs


@router.get("/audit-logs/statistics")
async def get_audit_statistics(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db)
):
    """Get audit statistics for reporting."""
    
    audit_logger = AuditLogger(db)
    stats = audit_logger.get_audit_statistics(start_date, end_date)
    
    return stats


# Compliance Rules Routes
@router.post("/compliance-rules", response_model=ComplianceRuleResponse)
async def create_compliance_rule(
    rule: ComplianceRuleCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """Create a new compliance rule."""
    
    compliance_manager = ComplianceManager(db)
    
    # Get user from request context (would be from auth middleware)
    created_by = request.headers.get("X-User-ID", "unknown")
    
    rule_response = compliance_manager.create_rule(rule, created_by)
    
    return rule_response


@router.get("/compliance-rules", response_model=List[ComplianceRuleResponse])
async def get_compliance_rules(
    rule_type: Optional[str] = Query(None),
    enabled_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get compliance rules."""
    
    compliance_manager = ComplianceManager(db)
    rules = compliance_manager.get_rules(rule_type, enabled_only)
    
    return rules


@router.put("/compliance-rules/{rule_id}", response_model=ComplianceRuleResponse)
async def update_compliance_rule(
    rule_id: str,
    updates: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Update a compliance rule."""
    
    compliance_manager = ComplianceManager(db)
    rule = compliance_manager.update_rule(rule_id, updates)
    
    if not rule:
        raise HTTPException(status_code=404, detail="Compliance rule not found")
    
    return rule


@router.post("/compliance-rules/check", response_model=ComplianceCheckResponse)
async def check_compliance(
    request: ComplianceCheckRequest,
    db: Session = Depends(get_db)
):
    """Check compliance for a resource operation."""
    
    compliance_manager = ComplianceManager(db)
    
    result = compliance_manager.check_compliance(
        resource_type=request.resource_type,
        resource_data=request.resource_data,
        action=request.action,
        user_context=request.user_context
    )
    
    return ComplianceCheckResponse(**result)


# Compliance Violations Routes
@router.get("/compliance-violations", response_model=List[ComplianceViolationResponse])
async def get_compliance_violations(
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db)
):
    """Get compliance violations."""
    
    compliance_manager = ComplianceManager(db)
    violations = compliance_manager.get_violations(status, severity, resource_type, limit)
    
    return violations


@router.get("/compliance-violations/report", response_model=ComplianceReport)
async def generate_compliance_report(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    db: Session = Depends(get_db)
):
    """Generate compliance report for a date range."""
    
    compliance_manager = ComplianceManager(db)
    report = compliance_manager.generate_compliance_report(start_date, end_date)
    
    return report


# Access Control Routes
@router.post("/access-control/grant")
async def grant_access(
    access_config: AccessControlCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """Grant access to a resource."""
    
    access_manager = AccessControlManager(db)
    
    # Get user from request context
    granted_by = request.headers.get("X-User-ID", "unknown")
    
    access_id = access_manager.grant_access(
        resource_type=access_config.resource_type,
        resource_id=access_config.resource_id,
        granted_by=granted_by,
        user_id=access_config.user_id,
        role=access_config.role,
        team_id=access_config.team_id,
        permissions=access_config.permissions,
        conditions=access_config.conditions,
        expires_at=access_config.expires_at
    )
    
    return {"access_id": access_id, "message": "Access granted successfully"}


@router.delete("/access-control/{access_id}")
async def revoke_access(
    access_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Revoke access control."""
    
    access_manager = AccessControlManager(db)
    
    # Get user from request context
    revoked_by = request.headers.get("X-User-ID", "unknown")
    
    success = access_manager.revoke_access(access_id, revoked_by)
    
    if not success:
        raise HTTPException(status_code=404, detail="Access control not found")
    
    return {"message": "Access revoked successfully"}


@router.post("/access-control/check", response_model=PermissionCheckResponse)
async def check_permission(
    request: PermissionCheckRequest,
    db: Session = Depends(get_db)
):
    """Check if user has permission for a resource."""
    
    access_manager = AccessControlManager(db)
    
    result = access_manager.check_permission(
        user_id=request.user_id,
        resource_type=request.resource_type,
        resource_id=request.resource_id,
        permission=request.permission,
        user_roles=request.user_roles,
        user_teams=request.user_teams,
        context=request.context
    )
    
    return PermissionCheckResponse(**result)


@router.get("/access-control/user/{user_id}")
async def get_user_permissions(
    user_id: str,
    resource_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get all permissions for a user."""
    
    access_manager = AccessControlManager(db)
    permissions = access_manager.get_user_permissions(user_id, resource_type)
    
    return {"user_id": user_id, "permissions": permissions}


@router.get("/access-control/resource/{resource_type}/{resource_id}")
async def get_resource_permissions(
    resource_type: str,
    resource_id: str,
    db: Session = Depends(get_db)
):
    """Get all permissions for a resource."""
    
    access_manager = AccessControlManager(db)
    permissions = access_manager.get_resource_permissions(resource_type, resource_id)
    
    return {"resource_type": resource_type, "resource_id": resource_id, "permissions": permissions}


# Change Approval Routes
@router.post("/change-approvals", response_model=str)
async def request_change_approval(
    approval_request: ChangeApprovalCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """Request approval for a change."""
    
    approval_manager = ChangeApprovalManager(db)
    
    # Get user from request context
    requested_by = request.headers.get("X-User-ID", "unknown")
    
    approval_id = approval_manager.request_approval(
        resource_type=approval_request.resource_type,
        resource_id=approval_request.resource_id,
        change_description=approval_request.change_description,
        proposed_changes=approval_request.proposed_changes,
        requested_by=requested_by,
        priority=approval_request.priority,
        deadline=approval_request.deadline
    )
    
    if not approval_id:
        return {"message": "No approval required for this change"}
    
    return {"approval_id": approval_id, "message": "Approval request created"}


@router.post("/change-approvals/{approval_id}/approve")
async def approve_change(
    approval_id: str,
    approval_notes: Optional[str] = None,
    request: Request = None,
    db: Session = Depends(get_db)
):
    """Approve a change request."""
    
    approval_manager = ChangeApprovalManager(db)
    
    # Get user from request context
    approver_id = request.headers.get("X-User-ID", "unknown")
    
    result = approval_manager.approve_change(approval_id, approver_id, approval_notes)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/change-approvals/{approval_id}/reject")
async def reject_change(
    approval_id: str,
    rejection_reason: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Reject a change request."""
    
    approval_manager = ChangeApprovalManager(db)
    
    # Get user from request context
    approver_id = request.headers.get("X-User-ID", "unknown")
    
    result = approval_manager.reject_change(approval_id, approver_id, rejection_reason)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.get("/change-approvals/pending", response_model=List[ChangeApprovalResponse])
async def get_pending_approvals(
    resource_type: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """Get pending approval requests."""
    
    approval_manager = ChangeApprovalManager(db)
    
    # Get user from request context for filtering
    approver_id = request.headers.get("X-User-ID")
    
    approvals = approval_manager.get_pending_approvals(
        approver_id=approver_id,
        resource_type=resource_type,
        priority=priority,
        limit=limit
    )
    
    return approvals


@router.get("/change-approvals/history", response_model=List[ChangeApprovalResponse])
async def get_approval_history(
    resource_type: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    requested_by: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db)
):
    """Get approval history."""
    
    approval_manager = ChangeApprovalManager(db)
    
    approvals = approval_manager.get_approval_history(
        resource_type=resource_type,
        resource_id=resource_id,
        requested_by=requested_by,
        limit=limit
    )
    
    return approvals


@router.get("/change-approvals/status/{resource_type}/{resource_id}")
async def check_approval_status(
    resource_type: str,
    resource_id: str,
    db: Session = Depends(get_db)
):
    """Check approval status for a resource."""
    
    approval_manager = ChangeApprovalManager(db)
    status = approval_manager.check_approval_status(resource_type, resource_id)
    
    return status


# Maintenance Routes
@router.post("/maintenance/cleanup-expired-access")
async def cleanup_expired_access(db: Session = Depends(get_db)):
    """Clean up expired access controls."""
    
    access_manager = AccessControlManager(db)
    count = access_manager.cleanup_expired_access()
    
    return {"message": f"Cleaned up {count} expired access controls"}


@router.post("/maintenance/expire-old-approvals")
async def expire_old_approvals(db: Session = Depends(get_db)):
    """Expire old pending approval requests."""
    
    approval_manager = ChangeApprovalManager(db)
    count = approval_manager.expire_old_approvals()
    
    return {"message": f"Expired {count} old approval requests"}


# Statistics Routes
@router.get("/statistics/access-control")
async def get_access_control_statistics(db: Session = Depends(get_db)):
    """Get access control statistics."""
    
    access_manager = AccessControlManager(db)
    stats = access_manager.get_access_statistics()
    
    return stats


@router.get("/statistics/approvals")
async def get_approval_statistics(db: Session = Depends(get_db)):
    """Get approval workflow statistics."""
    
    approval_manager = ChangeApprovalManager(db)
    stats = approval_manager.get_approval_statistics()
    
    return stats