"""
Just-in-Time (JIT) Access Provisioning System
Implements automated approval workflows and temporary access management
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import secrets
import json

logger = logging.getLogger(__name__)

class AccessRequestStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    REVOKED = "revoked"

class ApprovalType(Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    ESCALATED = "escalated"

@dataclass
class AccessRequest:
    request_id: str
    user_id: str
    resource_id: str
    permissions: List[str]
    justification: str
    requested_at: datetime
    requested_duration: timedelta
    status: AccessRequestStatus = AccessRequestStatus.PENDING
    approval_type: Optional[ApprovalType] = None
    approver_id: Optional[str] = None
    approved_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ApprovalRule:
    rule_id: str
    resource_pattern: str
    permission_patterns: List[str]
    auto_approve_conditions: Dict[str, Any]
    required_approvers: List[str]
    escalation_rules: Dict[str, Any]
    max_duration: timedelta
    risk_score_threshold: float

@dataclass
class TemporaryAccess:
    access_id: str
    user_id: str
    resource_id: str
    permissions: List[str]
    granted_at: datetime
    expires_at: datetime
    request_id: str
    is_active: bool = True

class JITAccessSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.access_requests: Dict[str, AccessRequest] = {}
        self.approval_rules: Dict[str, ApprovalRule] = {}
        self.temporary_accesses: Dict[str, TemporaryAccess] = {}
        self.user_risk_scores: Dict[str, float] = {}
        self.pending_approvals: Dict[str, List[str]] = {}  # approver_id -> request_ids
        
        # Load default approval rules
        self._load_default_rules()
    
    def request_access(self, user_id: str, resource_id: str, permissions: List[str],
                      justification: str, duration_hours: int = 8) -> str:
        """Submit access request"""
        try:
            request_id = secrets.token_urlsafe(32)
            duration = timedelta(hours=min(duration_hours, 24))  # Max 24 hours
            
            request = AccessRequest(
                request_id=request_id,
                user_id=user_id,
                resource_id=resource_id,
                permissions=permissions,
                justification=justification,
                requested_at=datetime.utcnow(),
                requested_duration=duration,
                metadata={
                    "user_risk_score": self.user_risk_scores.get(user_id, 0.5),
                    "resource_sensitivity": self._get_resource_sensitivity(resource_id),
                    "permission_risk": self._calculate_permission_risk(permissions)
                }
            )
            
            self.access_requests[request_id] = request
            
            # Process request through approval workflow
            self._process_access_request(request)
            
            logger.info(f"Access request submitted: {request_id} for user {user_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"Access request submission failed: {str(e)}")
            raise
    
    def _process_access_request(self, request: AccessRequest):
        """Process access request through approval workflow"""
        try:
            # Find matching approval rule
            matching_rule = self._find_matching_rule(request.resource_id, request.permissions)
            
            if not matching_rule:
                # Default to manual approval
                request.approval_type = ApprovalType.MANUAL
                self._assign_to_approvers(request, ["admin"])
                return
            
            # Check auto-approval conditions
            if self._check_auto_approval(request, matching_rule):
                self._auto_approve_request(request)
            else:
                # Assign to manual approvers
                request.approval_type = ApprovalType.MANUAL
                self._assign_to_approvers(request, matching_rule.required_approvers)
            
        except Exception as e:
            logger.error(f"Access request processing failed: {str(e)}")
            request.status = AccessRequestStatus.DENIED
    
    def _check_auto_approval(self, request: AccessRequest, rule: ApprovalRule) -> bool:
        """Check if request meets auto-approval conditions"""
        try:
            conditions = rule.auto_approve_conditions
            
            # Check user risk score
            user_risk = self.user_risk_scores.get(request.user_id, 0.5)
            if user_risk > conditions.get("max_user_risk", 0.3):
                return False
            
            # Check request duration
            if request.requested_duration > rule.max_duration:
                return False
            
            # Check time-based conditions
            current_hour = datetime.utcnow().hour
            allowed_hours = conditions.get("allowed_hours", list(range(24)))
            if current_hour not in allowed_hours:
                return False
            
            # Check permission risk
            permission_risk = request.metadata.get("permission_risk", 1.0)
            if permission_risk > conditions.get("max_permission_risk", 0.5):
                return False
            
            # Check resource sensitivity
            resource_sensitivity = request.metadata.get("resource_sensitivity", 1.0)
            if resource_sensitivity > conditions.get("max_resource_sensitivity", 0.5):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Auto-approval check failed: {str(e)}")
            return False
    
    def _auto_approve_request(self, request: AccessRequest):
        """Automatically approve request"""
        try:
            request.status = AccessRequestStatus.APPROVED
            request.approval_type = ApprovalType.AUTOMATIC
            request.approved_at = datetime.utcnow()
            request.expires_at = request.approved_at + request.requested_duration
            
            # Grant temporary access
            self._grant_temporary_access(request)
            
            logger.info(f"Access request auto-approved: {request.request_id}")
            
        except Exception as e:
            logger.error(f"Auto-approval failed: {str(e)}")
            request.status = AccessRequestStatus.DENIED
    
    def _assign_to_approvers(self, request: AccessRequest, approvers: List[str]):
        """Assign request to approvers"""
        for approver_id in approvers:
            if approver_id not in self.pending_approvals:
                self.pending_approvals[approver_id] = []
            self.pending_approvals[approver_id].append(request.request_id)
        
        logger.info(f"Access request assigned to approvers: {request.request_id}")
    
    def approve_request(self, request_id: str, approver_id: str, 
                       comments: Optional[str] = None) -> bool:
        """Approve access request"""
        try:
            if request_id not in self.access_requests:
                logger.warning(f"Access request not found: {request_id}")
                return False
            
            request = self.access_requests[request_id]
            
            if request.status != AccessRequestStatus.PENDING:
                logger.warning(f"Request not in pending status: {request_id}")
                return False
            
            # Verify approver authorization
            if not self._is_authorized_approver(approver_id, request):
                logger.warning(f"Unauthorized approver: {approver_id} for request {request_id}")
                return False
            
            # Approve request
            request.status = AccessRequestStatus.APPROVED
            request.approver_id = approver_id
            request.approved_at = datetime.utcnow()
            request.expires_at = request.approved_at + request.requested_duration
            
            if comments:
                request.metadata["approval_comments"] = comments
            
            # Grant temporary access
            self._grant_temporary_access(request)
            
            # Remove from pending approvals
            self._remove_from_pending_approvals(request_id)
            
            logger.info(f"Access request approved: {request_id} by {approver_id}")
            return True
            
        except Exception as e:
            logger.error(f"Request approval failed: {str(e)}")
            return False
    
    def deny_request(self, request_id: str, approver_id: str, 
                    reason: str) -> bool:
        """Deny access request"""
        try:
            if request_id not in self.access_requests:
                logger.warning(f"Access request not found: {request_id}")
                return False
            
            request = self.access_requests[request_id]
            
            if request.status != AccessRequestStatus.PENDING:
                logger.warning(f"Request not in pending status: {request_id}")
                return False
            
            # Verify approver authorization
            if not self._is_authorized_approver(approver_id, request):
                logger.warning(f"Unauthorized approver: {approver_id} for request {request_id}")
                return False
            
            # Deny request
            request.status = AccessRequestStatus.DENIED
            request.approver_id = approver_id
            request.metadata["denial_reason"] = reason
            
            # Remove from pending approvals
            self._remove_from_pending_approvals(request_id)
            
            logger.info(f"Access request denied: {request_id} by {approver_id}")
            return True
            
        except Exception as e:
            logger.error(f"Request denial failed: {str(e)}")
            return False
    
    def _grant_temporary_access(self, request: AccessRequest):
        """Grant temporary access based on approved request"""
        try:
            access_id = secrets.token_urlsafe(32)
            
            temporary_access = TemporaryAccess(
                access_id=access_id,
                user_id=request.user_id,
                resource_id=request.resource_id,
                permissions=request.permissions,
                granted_at=request.approved_at,
                expires_at=request.expires_at,
                request_id=request.request_id
            )
            
            self.temporary_accesses[access_id] = temporary_access
            
            logger.info(f"Temporary access granted: {access_id} for user {request.user_id}")
            
        except Exception as e:
            logger.error(f"Temporary access grant failed: {str(e)}")
            raise
    
    def revoke_access(self, access_id: str, revoker_id: str, reason: str) -> bool:
        """Revoke temporary access"""
        try:
            if access_id not in self.temporary_accesses:
                logger.warning(f"Temporary access not found: {access_id}")
                return False
            
            access = self.temporary_accesses[access_id]
            
            if not access.is_active:
                logger.warning(f"Access already inactive: {access_id}")
                return False
            
            # Revoke access
            access.is_active = False
            
            # Update original request
            if access.request_id in self.access_requests:
                request = self.access_requests[access.request_id]
                request.status = AccessRequestStatus.REVOKED
                request.metadata["revocation_reason"] = reason
                request.metadata["revoked_by"] = revoker_id
                request.metadata["revoked_at"] = datetime.utcnow().isoformat()
            
            logger.info(f"Temporary access revoked: {access_id} by {revoker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Access revocation failed: {str(e)}")
            return False
    
    def check_access(self, user_id: str, resource_id: str, permission: str) -> bool:
        """Check if user has temporary access to resource with permission"""
        try:
            current_time = datetime.utcnow()
            
            for access in self.temporary_accesses.values():
                if (access.user_id == user_id and 
                    access.resource_id == resource_id and
                    permission in access.permissions and
                    access.is_active and
                    current_time < access.expires_at):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Access check failed: {str(e)}")
            return False
    
    def get_user_active_accesses(self, user_id: str) -> List[TemporaryAccess]:
        """Get all active temporary accesses for user"""
        current_time = datetime.utcnow()
        
        return [
            access for access in self.temporary_accesses.values()
            if (access.user_id == user_id and 
                access.is_active and 
                current_time < access.expires_at)
        ]
    
    def get_pending_approvals(self, approver_id: str) -> List[AccessRequest]:
        """Get pending approval requests for approver"""
        if approver_id not in self.pending_approvals:
            return []
        
        request_ids = self.pending_approvals[approver_id]
        return [
            self.access_requests[req_id] for req_id in request_ids
            if req_id in self.access_requests and 
            self.access_requests[req_id].status == AccessRequestStatus.PENDING
        ]
    
    def cleanup_expired_accesses(self):
        """Remove expired temporary accesses"""
        current_time = datetime.utcnow()
        expired_accesses = []
        
        for access_id, access in self.temporary_accesses.items():
            if current_time > access.expires_at and access.is_active:
                access.is_active = False
                expired_accesses.append(access_id)
                
                # Update request status
                if access.request_id in self.access_requests:
                    self.access_requests[access.request_id].status = AccessRequestStatus.EXPIRED
        
        if expired_accesses:
            logger.info(f"Expired {len(expired_accesses)} temporary accesses")
    
    def _find_matching_rule(self, resource_id: str, permissions: List[str]) -> Optional[ApprovalRule]:
        """Find matching approval rule for resource and permissions"""
        # Simple pattern matching - in production, use more sophisticated matching
        for rule in self.approval_rules.values():
            if resource_id.startswith(rule.resource_pattern.rstrip('*')):
                return rule
        return None
    
    def _is_authorized_approver(self, approver_id: str, request: AccessRequest) -> bool:
        """Check if user is authorized to approve request"""
        # In production, integrate with RBAC system
        return approver_id in self.pending_approvals and request.request_id in self.pending_approvals[approver_id]
    
    def _remove_from_pending_approvals(self, request_id: str):
        """Remove request from all pending approval lists"""
        for approver_id, request_ids in self.pending_approvals.items():
            if request_id in request_ids:
                request_ids.remove(request_id)
    
    def _get_resource_sensitivity(self, resource_id: str) -> float:
        """Get resource sensitivity score (0.0 to 1.0)"""
        # Simple implementation - in production, use resource classification system
        if "admin" in resource_id.lower():
            return 1.0
        elif "sensitive" in resource_id.lower():
            return 0.8
        elif "internal" in resource_id.lower():
            return 0.6
        else:
            return 0.3
    
    def _calculate_permission_risk(self, permissions: List[str]) -> float:
        """Calculate risk score for permissions"""
        risk_weights = {
            "admin": 1.0,
            "write": 0.7,
            "delete": 0.9,
            "read": 0.2,
            "execute": 0.8
        }
        
        if not permissions:
            return 0.0
        
        total_risk = sum(
            max([risk_weights.get(keyword, 0.5) for keyword in risk_weights.keys() 
                 if keyword in perm.lower()] + [0.3])
            for perm in permissions
        )
        
        return min(total_risk / len(permissions), 1.0)
    
    def _load_default_rules(self):
        """Load default approval rules"""
        # Admin resources require manual approval
        admin_rule = ApprovalRule(
            rule_id="admin_resources",
            resource_pattern="admin/*",
            permission_patterns=["*"],
            auto_approve_conditions={},
            required_approvers=["admin", "security_admin"],
            escalation_rules={},
            max_duration=timedelta(hours=4),
            risk_score_threshold=0.9
        )
        
        # Low-risk resources can be auto-approved
        low_risk_rule = ApprovalRule(
            rule_id="low_risk_resources",
            resource_pattern="public/*",
            permission_patterns=["read"],
            auto_approve_conditions={
                "max_user_risk": 0.3,
                "max_permission_risk": 0.3,
                "max_resource_sensitivity": 0.3,
                "allowed_hours": list(range(6, 22))  # 6 AM to 10 PM
            },
            required_approvers=["manager"],
            escalation_rules={},
            max_duration=timedelta(hours=8),
            risk_score_threshold=0.3
        )
        
        self.approval_rules[admin_rule.rule_id] = admin_rule
        self.approval_rules[low_risk_rule.rule_id] = low_risk_rule