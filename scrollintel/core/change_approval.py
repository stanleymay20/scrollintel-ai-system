"""
Change Approval Workflow System

This module provides comprehensive change approval workflows for sensitive
prompt operations, ensuring proper review and authorization processes.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from ..models.audit_models import (
    ChangeApproval, ApprovalStatus, RiskLevel, AuditAction,
    ChangeApprovalCreate, ChangeApprovalResponse
)
from ..models.database import get_sync_db
from ..core.audit_logger import audit_logger, audit_action
from ..core.access_control import access_control_manager, Permission
from ..core.config import get_settings


class ApprovalRule:
    """Base class for approval rules"""
    
    def __init__(self, name: str, description: str, condition: Callable):
        self.name = name
        self.description = description
        self.condition = condition
    
    def applies_to(self, change_request: Dict[str, Any]) -> bool:
        """Check if this rule applies to the change request"""
        return self.condition(change_request)


class ChangeApprovalManager:
    """Comprehensive change approval workflow management"""
    
    def __init__(self):
        self.settings = get_settings()
        self.approval_rules = self._initialize_approval_rules()
        self.notification_handlers = []
    
    def _initialize_approval_rules(self) -> List[ApprovalRule]:
        """Initialize default approval rules"""
        rules = []
        
        # High-risk changes require approval
        rules.append(ApprovalRule(
            name="high_risk_changes",
            description="Changes with high or critical risk level require approval",
            condition=lambda req: req.get("risk_level") in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ))
        
        # Production prompt changes require approval
        rules.append(ApprovalRule(
            name="production_prompts",
            description="Changes to production prompts require approval",
            condition=lambda req: req.get("prompt_environment") == "production"
        ))
        
        # Sensitive category prompts require approval
        rules.append(ApprovalRule(
            name="sensitive_categories",
            description="Changes to sensitive prompt categories require approval",
            condition=lambda req: req.get("prompt_category") in [
                "financial", "medical", "legal", "personal_data", "security"
            ]
        ))
        
        # Large content changes require approval
        rules.append(ApprovalRule(
            name="large_content_changes",
            description="Large content changes require approval",
            condition=lambda req: self._is_large_content_change(req)
        ))
        
        # Deletion operations require approval
        rules.append(ApprovalRule(
            name="deletion_operations",
            description="Prompt deletion operations require approval",
            condition=lambda req: req.get("change_type") == "delete"
        ))
        
        return rules
    
    def _is_large_content_change(self, change_request: Dict[str, Any]) -> bool:
        """Check if the change involves large content modifications"""
        proposed_changes = change_request.get("proposed_changes", {})
        old_content = proposed_changes.get("old_content", "")
        new_content = proposed_changes.get("new_content", "")
        
        # Consider it large if content changes by more than 50% or 500 characters
        if len(old_content) == 0:
            return len(new_content) > 500
        
        change_ratio = abs(len(new_content) - len(old_content)) / len(old_content)
        return change_ratio > 0.5 or abs(len(new_content) - len(old_content)) > 500
    
    @audit_action(AuditAction.CREATE, "change_approval", RiskLevel.MEDIUM)
    def request_approval(
        self,
        prompt_id: str,
        requester_id: str,
        requester_email: str,
        change_type: str,
        change_description: str,
        change_justification: str,
        proposed_changes: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Request approval for a prompt change
        
        Args:
            prompt_id: ID of the prompt to be changed
            requester_id: ID of the user requesting the change
            requester_email: Email of the requester
            change_type: Type of change (create, update, delete)
            change_description: Description of the change
            change_justification: Business justification for the change
            proposed_changes: Detailed proposed changes
            context: Additional context information
            
        Returns:
            str: Change approval request ID
        """
        with get_sync_db() as db:
            # Assess risk level
            risk_assessment = self._assess_change_risk(
                change_type, proposed_changes, context or {}
            )
            
            # Check if approval is required
            change_request = {
                "prompt_id": prompt_id,
                "change_type": change_type,
                "proposed_changes": proposed_changes,
                "risk_level": risk_assessment["risk_level"],
                "prompt_environment": context.get("environment", "development"),
                "prompt_category": context.get("category", "general")
            }
            
            requires_approval = self._requires_approval(change_request)
            
            if not requires_approval:
                # Auto-approve low-risk changes
                return self._auto_approve_change(
                    prompt_id, requester_id, requester_email,
                    change_type, change_description, proposed_changes
                )
            
            # Create approval request
            approval_request = ChangeApproval(
                id=str(uuid.uuid4()),
                prompt_id=prompt_id,
                requester_id=requester_id,
                requester_email=requester_email,
                change_type=change_type,
                change_description=change_description,
                change_justification=change_justification,
                proposed_changes=proposed_changes,
                status=ApprovalStatus.PENDING.value,
                risk_level=risk_assessment["risk_level"].value,
                risk_assessment=risk_assessment,
                requested_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=7)  # 7-day expiration
            )
            
            db.add(approval_request)
            db.commit()
            
            # Send notifications
            self._send_approval_notifications(approval_request)
            
            return approval_request.id
    
    def _assess_change_risk(
        self,
        change_type: str,
        proposed_changes: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess the risk level of a proposed change"""
        risk_factors = []
        risk_score = 0
        
        # Change type risk
        if change_type == "delete":
            risk_score += 30
            risk_factors.append("Deletion operation")
        elif change_type == "update":
            risk_score += 10
            risk_factors.append("Update operation")
        
        # Environment risk
        environment = context.get("environment", "development")
        if environment == "production":
            risk_score += 25
            risk_factors.append("Production environment")
        elif environment == "staging":
            risk_score += 15
            risk_factors.append("Staging environment")
        
        # Category risk
        category = context.get("category", "general")
        sensitive_categories = ["financial", "medical", "legal", "personal_data", "security"]
        if category in sensitive_categories:
            risk_score += 20
            risk_factors.append(f"Sensitive category: {category}")
        
        # Content change size risk
        if self._is_large_content_change({"proposed_changes": proposed_changes}):
            risk_score += 15
            risk_factors.append("Large content change")
        
        # Usage frequency risk
        usage_frequency = context.get("usage_frequency", 0)
        if usage_frequency > 1000:
            risk_score += 10
            risk_factors.append("High usage frequency")
        
        # Determine risk level
        if risk_score >= 60:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 40:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 20:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "assessment_details": {
                "change_type_score": 30 if change_type == "delete" else 10,
                "environment_score": 25 if environment == "production" else 0,
                "category_score": 20 if category in sensitive_categories else 0,
                "content_size_score": 15 if self._is_large_content_change({"proposed_changes": proposed_changes}) else 0,
                "usage_frequency_score": 10 if usage_frequency > 1000 else 0
            }
        }
    
    def _requires_approval(self, change_request: Dict[str, Any]) -> bool:
        """Check if the change request requires approval"""
        for rule in self.approval_rules:
            if rule.applies_to(change_request):
                return True
        return False
    
    def _auto_approve_change(
        self,
        prompt_id: str,
        requester_id: str,
        requester_email: str,
        change_type: str,
        change_description: str,
        proposed_changes: Dict[str, Any]
    ) -> str:
        """Auto-approve low-risk changes"""
        with get_sync_db() as db:
            approval_request = ChangeApproval(
                id=str(uuid.uuid4()),
                prompt_id=prompt_id,
                requester_id=requester_id,
                requester_email=requester_email,
                change_type=change_type,
                change_description=change_description,
                change_justification="Auto-approved low-risk change",
                proposed_changes=proposed_changes,
                status=ApprovalStatus.APPROVED.value,
                approver_id="system",
                approver_email="system@auto-approval",
                approval_comments="Automatically approved based on low risk assessment",
                risk_level=RiskLevel.LOW.value,
                requested_at=datetime.utcnow(),
                reviewed_at=datetime.utcnow(),
                approved_at=datetime.utcnow()
            )
            
            db.add(approval_request)
            db.commit()
            
            return approval_request.id
    
    @audit_action(AuditAction.APPROVE, "change_approval", RiskLevel.MEDIUM)
    def approve_change(
        self,
        approval_id: str,
        approver_id: str,
        approver_email: str,
        comments: Optional[str] = None
    ) -> bool:
        """
        Approve a change request
        
        Args:
            approval_id: ID of the approval request
            approver_id: ID of the approver
            approver_email: Email of the approver
            comments: Approval comments
            
        Returns:
            bool: True if approval was successful
        """
        with get_sync_db() as db:
            approval_request = db.query(ChangeApproval).filter(
                ChangeApproval.id == approval_id
            ).first()
            
            if not approval_request:
                return False
            
            if approval_request.status != ApprovalStatus.PENDING.value:
                return False
            
            # Check if approver has permission
            if not access_control_manager.check_permission(
                approver_id, Permission.ADMIN_COMPLIANCE_MANAGE
            ):
                raise PermissionError("Insufficient permissions to approve changes")
            
            # Update approval request
            approval_request.status = ApprovalStatus.APPROVED.value
            approval_request.approver_id = approver_id
            approval_request.approver_email = approver_email
            approval_request.approval_comments = comments
            approval_request.reviewed_at = datetime.utcnow()
            approval_request.approved_at = datetime.utcnow()
            
            db.commit()
            
            # Send notifications
            self._send_approval_decision_notifications(approval_request, "approved")
            
            return True
    
    @audit_action(AuditAction.REJECT, "change_approval", RiskLevel.MEDIUM)
    def reject_change(
        self,
        approval_id: str,
        approver_id: str,
        approver_email: str,
        comments: str
    ) -> bool:
        """
        Reject a change request
        
        Args:
            approval_id: ID of the approval request
            approver_id: ID of the approver
            approver_email: Email of the approver
            comments: Rejection comments (required)
            
        Returns:
            bool: True if rejection was successful
        """
        with get_sync_db() as db:
            approval_request = db.query(ChangeApproval).filter(
                ChangeApproval.id == approval_id
            ).first()
            
            if not approval_request:
                return False
            
            if approval_request.status != ApprovalStatus.PENDING.value:
                return False
            
            # Check if approver has permission
            if not access_control_manager.check_permission(
                approver_id, Permission.ADMIN_COMPLIANCE_MANAGE
            ):
                raise PermissionError("Insufficient permissions to reject changes")
            
            # Update approval request
            approval_request.status = ApprovalStatus.REJECTED.value
            approval_request.approver_id = approver_id
            approval_request.approver_email = approver_email
            approval_request.approval_comments = comments
            approval_request.reviewed_at = datetime.utcnow()
            
            db.commit()
            
            # Send notifications
            self._send_approval_decision_notifications(approval_request, "rejected")
            
            return True
    
    def get_pending_approvals(
        self,
        approver_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ChangeApprovalResponse]:
        """
        Get pending approval requests
        
        Args:
            approver_id: Filter by specific approver (if None, returns all)
            limit: Maximum number of requests to return
            offset: Number of requests to skip
            
        Returns:
            List[ChangeApprovalResponse]: List of pending approval requests
        """
        with get_sync_db() as db:
            query = db.query(ChangeApproval).filter(
                ChangeApproval.status == ApprovalStatus.PENDING.value
            )
            
            # Filter expired requests
            query = query.filter(
                or_(
                    ChangeApproval.expires_at.is_(None),
                    ChangeApproval.expires_at > datetime.utcnow()
                )
            )
            
            approval_requests = query.order_by(
                desc(ChangeApproval.requested_at)
            ).offset(offset).limit(limit).all()
            
            return [ChangeApprovalResponse.from_orm(req) for req in approval_requests]
    
    def get_approval_history(
        self,
        prompt_id: Optional[str] = None,
        requester_id: Optional[str] = None,
        approver_id: Optional[str] = None,
        status: Optional[ApprovalStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ChangeApprovalResponse]:
        """
        Get approval request history
        
        Args:
            prompt_id: Filter by prompt ID
            requester_id: Filter by requester ID
            approver_id: Filter by approver ID
            status: Filter by approval status
            limit: Maximum number of requests to return
            offset: Number of requests to skip
            
        Returns:
            List[ChangeApprovalResponse]: List of approval requests
        """
        with get_sync_db() as db:
            query = db.query(ChangeApproval)
            
            if prompt_id:
                query = query.filter(ChangeApproval.prompt_id == prompt_id)
            if requester_id:
                query = query.filter(ChangeApproval.requester_id == requester_id)
            if approver_id:
                query = query.filter(ChangeApproval.approver_id == approver_id)
            if status:
                query = query.filter(ChangeApproval.status == status.value)
            
            approval_requests = query.order_by(
                desc(ChangeApproval.requested_at)
            ).offset(offset).limit(limit).all()
            
            return [ChangeApprovalResponse.from_orm(req) for req in approval_requests]
    
    def cleanup_expired_requests(self) -> int:
        """
        Clean up expired approval requests
        
        Returns:
            int: Number of requests cleaned up
        """
        with get_sync_db() as db:
            expired_requests = db.query(ChangeApproval).filter(
                and_(
                    ChangeApproval.status == ApprovalStatus.PENDING.value,
                    ChangeApproval.expires_at < datetime.utcnow()
                )
            ).all()
            
            count = len(expired_requests)
            
            for request in expired_requests:
                request.status = ApprovalStatus.CANCELLED.value
                request.approval_comments = "Request expired"
            
            db.commit()
            
            return count
    
    def add_notification_handler(self, handler: Callable):
        """Add a notification handler for approval events"""
        self.notification_handlers.append(handler)
    
    def _send_approval_notifications(self, approval_request: ChangeApproval):
        """Send notifications for new approval requests"""
        notification_data = {
            "type": "approval_requested",
            "approval_id": approval_request.id,
            "prompt_id": approval_request.prompt_id,
            "requester_email": approval_request.requester_email,
            "change_type": approval_request.change_type,
            "change_description": approval_request.change_description,
            "risk_level": approval_request.risk_level,
            "requested_at": approval_request.requested_at.isoformat()
        }
        
        for handler in self.notification_handlers:
            try:
                handler(notification_data)
            except Exception as e:
                # Log notification failure but don't fail the approval process
                audit_logger.log_action(
                    user_id="system",
                    user_email="system@notifications",
                    action=AuditAction.UPDATE,
                    resource_type="notification",
                    resource_id=approval_request.id,
                    context={"error": str(e), "notification_type": "approval_requested"}
                )
    
    def _send_approval_decision_notifications(
        self,
        approval_request: ChangeApproval,
        decision: str
    ):
        """Send notifications for approval decisions"""
        notification_data = {
            "type": f"approval_{decision}",
            "approval_id": approval_request.id,
            "prompt_id": approval_request.prompt_id,
            "requester_email": approval_request.requester_email,
            "approver_email": approval_request.approver_email,
            "change_type": approval_request.change_type,
            "approval_comments": approval_request.approval_comments,
            "reviewed_at": approval_request.reviewed_at.isoformat()
        }
        
        for handler in self.notification_handlers:
            try:
                handler(notification_data)
            except Exception as e:
                # Log notification failure but don't fail the approval process
                audit_logger.log_action(
                    user_id="system",
                    user_email="system@notifications",
                    action=AuditAction.UPDATE,
                    resource_type="notification",
                    resource_id=approval_request.id,
                    context={"error": str(e), "notification_type": f"approval_{decision}"}
                )


# Global change approval manager instance
change_approval_manager = ChangeApprovalManager()
