"""
Change approval workflow system for prompt management.
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from enum import Enum

from scrollintel.models.audit_models import (
    ChangeApproval, ChangeApprovalCreate, ChangeApprovalResponse
)


class ApprovalStatus(str, Enum):
    """Approval status types."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ChangeApprovalManager:
    """Service for managing change approval workflows."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.approval_rules = {
            "production_prompt": {
                "required_approvers": 2,
                "approval_timeout_hours": 24,
                "auto_approve_minor": False
            },
            "sensitive_template": {
                "required_approvers": 1,
                "approval_timeout_hours": 48,
                "auto_approve_minor": True
            },
            "experiment": {
                "required_approvers": 1,
                "approval_timeout_hours": 12,
                "auto_approve_minor": True
            }
        }
    
    def request_approval(
        self,
        resource_type: str,
        resource_id: str,
        change_description: str,
        proposed_changes: Dict[str, Any],
        requested_by: str,
        priority: str = "normal",
        deadline: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Request approval for a change."""
        
        # Check if approval is required
        if not self._requires_approval(resource_type, proposed_changes):
            return None
        
        # Calculate deadline if not provided
        if not deadline:
            rules = self.approval_rules.get(resource_type, {})
            timeout_hours = rules.get("approval_timeout_hours", 24)
            deadline = datetime.utcnow() + timedelta(hours=timeout_hours)
        
        approval_request = ChangeApproval(
            id=str(uuid.uuid4()),
            resource_type=resource_type,
            resource_id=resource_id,
            change_description=change_description,
            proposed_changes=proposed_changes,
            requested_by=requested_by,
            requested_at=datetime.utcnow(),
            status=ApprovalStatus.PENDING.value,
            priority=priority,
            deadline=deadline,
            approval_metadata=metadata or {}
        )
        
        self.db.add(approval_request)
        self.db.commit()
        
        # Notify approvers
        self._notify_approvers(approval_request)
        
        return approval_request.id
    
    def approve_change(
        self,
        approval_id: str,
        approver_id: str,
        approval_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Approve a change request."""
        
        approval = self.db.query(ChangeApproval).filter(
            ChangeApproval.id == approval_id
        ).first()
        
        if not approval:
            return {"success": False, "error": "Approval request not found"}
        
        if approval.status != ApprovalStatus.PENDING.value:
            return {"success": False, "error": f"Approval request is {approval.status}"}
        
        if approval.deadline and datetime.utcnow() > approval.deadline:
            approval.status = ApprovalStatus.EXPIRED.value
            self.db.commit()
            return {"success": False, "error": "Approval request has expired"}
        
        # Check if approver has permission
        if not self._can_approve(approver_id, approval.resource_type, approval.resource_id):
            return {"success": False, "error": "Insufficient permissions to approve"}
        
        approval.status = ApprovalStatus.APPROVED.value
        approval.approver_id = approver_id
        approval.approved_at = datetime.utcnow()
        approval.approval_notes = approval_notes
        
        self.db.commit()
        
        # Log approval
        self._log_approval_action(approval, approver_id, "approved")
        
        # Notify requester
        self._notify_approval_decision(approval, "approved")
        
        return {
            "success": True,
            "approval_id": approval_id,
            "status": "approved",
            "approved_at": approval.approved_at
        }
    
    def reject_change(
        self,
        approval_id: str,
        approver_id: str,
        rejection_reason: str
    ) -> Dict[str, Any]:
        """Reject a change request."""
        
        approval = self.db.query(ChangeApproval).filter(
            ChangeApproval.id == approval_id
        ).first()
        
        if not approval:
            return {"success": False, "error": "Approval request not found"}
        
        if approval.status != ApprovalStatus.PENDING.value:
            return {"success": False, "error": f"Approval request is {approval.status}"}
        
        # Check if approver has permission
        if not self._can_approve(approver_id, approval.resource_type, approval.resource_id):
            return {"success": False, "error": "Insufficient permissions to reject"}
        
        approval.status = ApprovalStatus.REJECTED.value
        approval.approver_id = approver_id
        approval.approved_at = datetime.utcnow()
        approval.rejection_reason = rejection_reason
        
        self.db.commit()
        
        # Log rejection
        self._log_approval_action(approval, approver_id, "rejected")
        
        # Notify requester
        self._notify_approval_decision(approval, "rejected")
        
        return {
            "success": True,
            "approval_id": approval_id,
            "status": "rejected",
            "rejected_at": approval.approved_at,
            "reason": rejection_reason
        }
    
    def cancel_approval(self, approval_id: str, cancelled_by: str) -> Dict[str, Any]:
        """Cancel a pending approval request."""
        
        approval = self.db.query(ChangeApproval).filter(
            ChangeApproval.id == approval_id
        ).first()
        
        if not approval:
            return {"success": False, "error": "Approval request not found"}
        
        if approval.status != ApprovalStatus.PENDING.value:
            return {"success": False, "error": f"Cannot cancel {approval.status} approval"}
        
        # Check if user can cancel (requester or admin)
        if approval.requested_by != cancelled_by and not self._is_admin(cancelled_by):
            return {"success": False, "error": "Insufficient permissions to cancel"}
        
        approval.status = ApprovalStatus.CANCELLED.value
        approval.approval_metadata["cancelled_by"] = cancelled_by
        approval.approval_metadata["cancelled_at"] = datetime.utcnow().isoformat()
        
        self.db.commit()
        
        return {"success": True, "approval_id": approval_id, "status": "cancelled"}
    
    def get_pending_approvals(
        self,
        approver_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 100
    ) -> List[ChangeApprovalResponse]:
        """Get pending approval requests."""
        
        query = self.db.query(ChangeApproval).filter(
            ChangeApproval.status == ApprovalStatus.PENDING.value
        )
        
        if resource_type:
            query = query.filter(ChangeApproval.resource_type == resource_type)
        
        if priority:
            query = query.filter(ChangeApproval.priority == priority)
        
        # Filter by approver permissions if specified
        if approver_id:
            # This would need integration with access control system
            pass
        
        # Order by priority (urgent > high > normal > low) then by requested time
        priority_order = {"urgent": 4, "high": 3, "normal": 2, "low": 1}
        
        approvals = query.all()
        
        # Sort in Python since SQLAlchemy ordering by enum values can be tricky
        approvals.sort(key=lambda x: (
            priority_order.get(x.priority, 0),
            x.requested_at
        ), reverse=True)
        
        approvals = approvals[:limit]
        
        return [ChangeApprovalResponse.model_validate(approval) for approval in approvals]
    
    def get_approval_history(
        self,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        requested_by: Optional[str] = None,
        limit: int = 100
    ) -> List[ChangeApprovalResponse]:
        """Get approval history."""
        
        query = self.db.query(ChangeApproval)
        
        if resource_type:
            query = query.filter(ChangeApproval.resource_type == resource_type)
        
        if resource_id:
            query = query.filter(ChangeApproval.resource_id == resource_id)
        
        if requested_by:
            query = query.filter(ChangeApproval.requested_by == requested_by)
        
        approvals = query.order_by(desc(ChangeApproval.requested_at)).limit(limit).all()
        
        return [ChangeApprovalResponse.model_validate(approval) for approval in approvals]
    
    def check_approval_status(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        """Check if a resource has pending or recent approvals."""
        
        # Check for pending approvals
        pending = self.db.query(ChangeApproval).filter(
            and_(
                ChangeApproval.resource_type == resource_type,
                ChangeApproval.resource_id == resource_id,
                ChangeApproval.status == ApprovalStatus.PENDING.value
            )
        ).first()
        
        if pending:
            return {
                "has_pending": True,
                "approval_id": pending.id,
                "requested_at": pending.requested_at,
                "deadline": pending.deadline,
                "priority": pending.priority
            }
        
        # Check for recent approvals
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent = self.db.query(ChangeApproval).filter(
            and_(
                ChangeApproval.resource_type == resource_type,
                ChangeApproval.resource_id == resource_id,
                ChangeApproval.approved_at >= recent_cutoff,
                ChangeApproval.status == ApprovalStatus.APPROVED.value
            )
        ).first()
        
        if recent:
            return {
                "has_pending": False,
                "recently_approved": True,
                "approval_id": recent.id,
                "approved_at": recent.approved_at,
                "approved_by": recent.approver_id
            }
        
        return {"has_pending": False, "recently_approved": False}
    
    def _requires_approval(self, resource_type: str, proposed_changes: Dict[str, Any]) -> bool:
        """Check if changes require approval."""
        
        rules = self.approval_rules.get(resource_type)
        if not rules:
            return False
        
        # Check for minor changes that can be auto-approved
        if rules.get("auto_approve_minor", False):
            if self._is_minor_change(proposed_changes):
                return False
        
        return True
    
    def _is_minor_change(self, proposed_changes: Dict[str, Any]) -> bool:
        """Determine if changes are minor and can be auto-approved."""
        
        minor_fields = ["description", "tags", "metadata", "examples"]
        
        # If only minor fields are changed
        changed_fields = set(proposed_changes.keys())
        if changed_fields.issubset(set(minor_fields)):
            return True
        
        # If content changes are small
        if "content" in proposed_changes:
            old_content = proposed_changes["content"].get("old", "")
            new_content = proposed_changes["content"].get("new", "")
            
            # Simple heuristic: less than 10% change
            if len(old_content) > 0:
                change_ratio = abs(len(new_content) - len(old_content)) / len(old_content)
                if change_ratio < 0.1:
                    return True
        
        return False
    
    def _can_approve(self, approver_id: str, resource_type: str, resource_id: str) -> bool:
        """Check if user can approve changes for resource."""
        
        # This would integrate with the access control system
        from scrollintel.core.access_control import AccessControlManager
        
        access_manager = AccessControlManager(self.db)
        result = access_manager.check_permission(
            user_id=approver_id,
            resource_type=resource_type,
            resource_id=resource_id,
            permission="approve"
        )
        
        return result.get("allowed", False)
    
    def _is_admin(self, user_id: str) -> bool:
        """Check if user is an admin."""
        
        # This would integrate with user management system
        # For now, return False as placeholder
        return False
    
    def _notify_approvers(self, approval: ChangeApproval):
        """Notify potential approvers of pending request."""
        
        # This would integrate with notification system
        # For now, just log the notification
        print(f"Notification: Approval request {approval.id} for {approval.resource_type} {approval.resource_id}")
    
    def _notify_approval_decision(self, approval: ChangeApproval, decision: str):
        """Notify requester of approval decision."""
        
        # This would integrate with notification system
        print(f"Notification: Approval request {approval.id} was {decision}")
    
    def _log_approval_action(self, approval: ChangeApproval, user_id: str, action: str):
        """Log approval action to audit trail."""
        
        from scrollintel.core.audit_logger import AuditLogger
        
        audit_logger = AuditLogger(self.db)
        audit_logger.log_action(
            user_id=user_id,
            user_email="unknown",  # Would get from user service
            action=f"approval_{action}",
            resource_type="change_approval",
            resource_id=approval.id,
            resource_name=f"{approval.resource_type}:{approval.resource_id}",
            metadata={
                "original_resource_type": approval.resource_type,
                "original_resource_id": approval.resource_id,
                "change_description": approval.change_description,
                "priority": approval.priority
            }
        )
    
    def expire_old_approvals(self) -> int:
        """Expire old pending approval requests."""
        
        expired_count = self.db.query(ChangeApproval).filter(
            and_(
                ChangeApproval.status == ApprovalStatus.PENDING.value,
                ChangeApproval.deadline < datetime.utcnow()
            )
        ).update({"status": ApprovalStatus.EXPIRED.value})
        
        self.db.commit()
        
        return expired_count
    
    def get_approval_statistics(self) -> Dict[str, Any]:
        """Get approval workflow statistics."""
        
        total_requests = self.db.query(ChangeApproval).count()
        
        status_counts = {}
        for status in ApprovalStatus:
            count = self.db.query(ChangeApproval).filter(
                ChangeApproval.status == status.value
            ).count()
            status_counts[status.value] = count
        
        # Average approval time
        approved_requests = self.db.query(ChangeApproval).filter(
            ChangeApproval.status == ApprovalStatus.APPROVED.value
        ).all()
        
        if approved_requests:
            approval_times = []
            for request in approved_requests:
                if request.approved_at and request.requested_at:
                    time_diff = request.approved_at - request.requested_at
                    approval_times.append(time_diff.total_seconds() / 3600)  # hours
            
            avg_approval_time = sum(approval_times) / len(approval_times) if approval_times else 0
        else:
            avg_approval_time = 0
        
        return {
            "total_requests": total_requests,
            "status_distribution": status_counts,
            "average_approval_time_hours": avg_approval_time,
            "approval_rate": status_counts.get("approved", 0) / max(total_requests, 1) * 100
        }