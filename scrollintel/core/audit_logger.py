"""
Audit logging service for prompt management system.
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
import json
import hashlib

from scrollintel.models.audit_models import (
    AuditLog, AuditAction, AuditLogCreate, AuditLogResponse
)


class AuditLogger:
    """Service for logging audit events."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def log_action(
        self,
        user_id: str,
        user_email: str,
        action: AuditAction,
        resource_type: str,
        resource_id: str,
        resource_name: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        compliance_tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an audit action."""
        
        # Calculate changes if both old and new values provided
        changes = None
        if old_values and new_values:
            changes = self._calculate_changes(old_values, new_values)
        
        # Determine risk level
        risk_level = self._assess_risk_level(action, resource_type, changes)
        
        # Create audit log entry
        audit_log = AuditLog(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            user_id=user_id,
            user_email=user_email,
            action=action.value if hasattr(action, 'value') else str(action),
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            old_values=old_values,
            new_values=new_values,
            changes=changes,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            compliance_tags=compliance_tags or [],
            risk_level=risk_level,
            audit_metadata=metadata or {}
        )
        
        self.db.add(audit_log)
        self.db.commit()
        
        return audit_log.id
    
    def _calculate_changes(self, old_values: Dict[str, Any], new_values: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate specific changes between old and new values."""
        changes = {}
        
        # Find added, modified, and removed fields
        all_keys = set(old_values.keys()) | set(new_values.keys())
        
        for key in all_keys:
            old_val = old_values.get(key)
            new_val = new_values.get(key)
            
            if key not in old_values:
                changes[key] = {"action": "added", "new_value": new_val}
            elif key not in new_values:
                changes[key] = {"action": "removed", "old_value": old_val}
            elif old_val != new_val:
                changes[key] = {
                    "action": "modified",
                    "old_value": old_val,
                    "new_value": new_val
                }
        
        return changes
    
    def _assess_risk_level(self, action: AuditAction, resource_type: str, changes: Optional[Dict[str, Any]]) -> str:
        """Assess risk level of the action."""
        
        # High risk actions
        if action in [AuditAction.DELETE, AuditAction.ROLLBACK]:
            return "high"
        
        # Critical resources
        if resource_type in ["production_prompt", "approved_template"]:
            if action in [AuditAction.UPDATE, AuditAction.APPROVE]:
                return "high"
        
        # Sensitive field changes
        if changes:
            sensitive_fields = ["content", "variables", "permissions", "approval_status"]
            if any(field in changes for field in sensitive_fields):
                return "medium"
        
        # Default to low risk
        return "low"
    
    def get_audit_trail(
        self,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditLogResponse]:
        """Get audit trail with filters."""
        
        query = self.db.query(AuditLog)
        
        # Apply filters
        if resource_type:
            query = query.filter(AuditLog.resource_type == resource_type)
        
        if resource_id:
            query = query.filter(AuditLog.resource_id == resource_id)
        
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        
        if action:
            query = query.filter(AuditLog.action == action.value)
        
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        # Order by timestamp descending
        query = query.order_by(desc(AuditLog.timestamp))
        
        # Apply pagination
        audit_logs = query.offset(offset).limit(limit).all()
        
        return [AuditLogResponse.model_validate(log) for log in audit_logs]
    
    def get_resource_history(self, resource_type: str, resource_id: str) -> List[AuditLogResponse]:
        """Get complete history for a specific resource."""
        
        audit_logs = self.db.query(AuditLog).filter(
            and_(
                AuditLog.resource_type == resource_type,
                AuditLog.resource_id == resource_id
            )
        ).order_by(desc(AuditLog.timestamp)).all()
        
        return [AuditLogResponse.model_validate(log) for log in audit_logs]
    
    def search_audit_logs(
        self,
        search_term: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[AuditLogResponse]:
        """Search audit logs by content."""
        
        query = self.db.query(AuditLog)
        
        # Search in multiple fields
        search_conditions = []
        
        if AuditLog.resource_name:
            search_conditions.append(AuditLog.resource_name.ilike(f"%{search_term}%"))
        
        search_conditions.append(AuditLog.user_email.ilike(f"%{search_term}%"))
        
        # For JSON fields, use different approach based on database
        try:
            search_conditions.append(AuditLog.changes.astext.ilike(f"%{search_term}%"))
        except:
            # Fallback for databases that don't support astext
            pass
        
        query = query.filter(or_(*search_conditions))
        
        # Apply additional filters
        if filters:
            if "risk_level" in filters:
                query = query.filter(AuditLog.risk_level == filters["risk_level"])
            
            if "action" in filters:
                query = query.filter(AuditLog.action == filters["action"])
        
        audit_logs = query.order_by(desc(AuditLog.timestamp)).limit(limit).all()
        
        return [AuditLogResponse.model_validate(log) for log in audit_logs]
    
    def get_audit_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get audit statistics for reporting."""
        
        query = self.db.query(AuditLog)
        
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        total_actions = query.count()
        
        # Actions by type
        action_counts = {}
        for action in AuditAction:
            count = query.filter(AuditLog.action == action.value).count()
            action_counts[action.value] = count
        
        # Risk level distribution
        risk_counts = {}
        for risk_level in ["low", "medium", "high", "critical"]:
            count = query.filter(AuditLog.risk_level == risk_level).count()
            risk_counts[risk_level] = count
        
        # Top users by activity - use SQLAlchemy ORM instead of raw SQL
        from sqlalchemy import func
        
        user_query = self.db.query(
            AuditLog.user_email,
            func.count(AuditLog.id).label('action_count')
        )
        
        if start_date:
            user_query = user_query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            user_query = user_query.filter(AuditLog.timestamp <= end_date)
        
        user_activity = user_query.group_by(AuditLog.user_email).order_by(
            func.count(AuditLog.id).desc()
        ).limit(10).all()
        
        return {
            "total_actions": total_actions,
            "action_counts": action_counts,
            "risk_counts": risk_counts,
            "top_users": [{"email": row.user_email, "count": row.action_count} for row in user_activity]
        }
    
    def create_audit_hash(self, audit_log: AuditLog) -> str:
        """Create tamper-proof hash for audit log."""
        
        # Create hash from critical fields
        hash_data = {
            "id": audit_log.id,
            "timestamp": audit_log.timestamp.isoformat(),
            "user_id": audit_log.user_id,
            "action": audit_log.action,
            "resource_type": audit_log.resource_type,
            "resource_id": audit_log.resource_id,
            "changes": audit_log.changes
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def verify_audit_integrity(self, audit_log_id: str) -> bool:
        """Verify audit log integrity."""
        
        audit_log = self.db.query(AuditLog).filter(AuditLog.id == audit_log_id).first()
        if not audit_log:
            return False
        
        # Recalculate hash and compare
        expected_hash = self.create_audit_hash(audit_log)
        stored_hash = audit_log.audit_metadata.get("integrity_hash") if audit_log.audit_metadata else None
        
        return expected_hash == stored_hash