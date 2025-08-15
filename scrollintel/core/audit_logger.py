"""
Comprehensive Audit Logging System

This module provides comprehensive audit logging for all prompt operations,
ensuring complete traceability and compliance with regulatory requirements.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from ..models.audit_models import (
    AuditLog, AuditAction, ComplianceStatus, RiskLevel,
    AuditLogCreate, AuditLogResponse
)
from ..models.database_utils import get_sync_db
from ..core.config import get_settings


class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self):
        self.settings = get_settings()
        self._session_context = {}
    
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
        changes_summary: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        risk_level: RiskLevel = RiskLevel.LOW
    ) -> str:
        """
        Log an auditable action
        
        Args:
            user_id: ID of the user performing the action
            user_email: Email of the user performing the action
            action: Type of action being performed
            resource_type: Type of resource being acted upon
            resource_id: ID of the resource being acted upon
            resource_name: Human-readable name of the resource
            old_values: Previous values before change
            new_values: New values after change
            changes_summary: Human-readable summary of changes
            context: Additional context information
            metadata: Additional metadata
            session_id: User session ID
            ip_address: User IP address
            user_agent: User agent string
            risk_level: Risk level of the action
            
        Returns:
            str: Audit log entry ID
        """
        with get_sync_db() as db:
            # Generate changes summary if not provided
            if not changes_summary and old_values and new_values:
                changes_summary = self._generate_changes_summary(old_values, new_values)
            
            # Create audit log entry
            audit_log = AuditLog(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                user_id=user_id,
                user_email=user_email,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                action=action.value,
                resource_type=resource_type,
                resource_id=resource_id,
                resource_name=resource_name,
                old_values=old_values,
                new_values=new_values,
                changes_summary=changes_summary,
                context=context or {},
                audit_metadata=metadata or {},
                compliance_status=ComplianceStatus.PENDING_REVIEW.value,
                risk_level=risk_level.value
            )
            
            db.add(audit_log)
            db.commit()
            
            return audit_log.id
    
    def get_audit_trail(
        self,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditLogResponse]:
        """
        Retrieve audit trail with filtering options
        
        Args:
            resource_id: Filter by resource ID
            resource_type: Filter by resource type
            user_id: Filter by user ID
            action: Filter by action type
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List[AuditLogResponse]: List of audit log entries
        """
        with get_sync_db() as db:
            query = db.query(AuditLog)
            
            # Apply filters
            if resource_id:
                query = query.filter(AuditLog.resource_id == resource_id)
            if resource_type:
                query = query.filter(AuditLog.resource_type == resource_type)
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
            
            return [AuditLogResponse.from_orm(log) for log in audit_logs]
    
    def get_user_activity(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive user activity summary
        
        Args:
            user_id: User ID to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dict[str, Any]: User activity summary
        """
        with get_sync_db() as db:
            query = db.query(AuditLog).filter(AuditLog.user_id == user_id)
            
            if start_date:
                query = query.filter(AuditLog.timestamp >= start_date)
            if end_date:
                query = query.filter(AuditLog.timestamp <= end_date)
            
            audit_logs = query.all()
            
            # Analyze activity patterns
            activity_summary = {
                "total_actions": len(audit_logs),
                "actions_by_type": {},
                "resources_accessed": set(),
                "risk_distribution": {},
                "compliance_status": {},
                "timeline": [],
                "suspicious_patterns": []
            }
            
            for log in audit_logs:
                # Count actions by type
                action = log.action
                activity_summary["actions_by_type"][action] = \
                    activity_summary["actions_by_type"].get(action, 0) + 1
                
                # Track resources accessed
                activity_summary["resources_accessed"].add(
                    f"{log.resource_type}:{log.resource_id}"
                )
                
                # Risk distribution
                risk = log.risk_level
                activity_summary["risk_distribution"][risk] = \
                    activity_summary["risk_distribution"].get(risk, 0) + 1
                
                # Compliance status
                compliance = log.compliance_status
                activity_summary["compliance_status"][compliance] = \
                    activity_summary["compliance_status"].get(compliance, 0) + 1
                
                # Timeline entry
                activity_summary["timeline"].append({
                    "timestamp": log.timestamp.isoformat(),
                    "action": log.action,
                    "resource": f"{log.resource_type}:{log.resource_id}",
                    "risk_level": log.risk_level
                })
            
            # Convert set to list for JSON serialization
            activity_summary["resources_accessed"] = list(
                activity_summary["resources_accessed"]
            )
            
            # Detect suspicious patterns
            activity_summary["suspicious_patterns"] = self._detect_suspicious_patterns(
                audit_logs
            )
            
            return activity_summary
    
    def search_audit_logs(
        self,
        search_query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[AuditLogResponse]:
        """
        Search audit logs with text search and filters
        
        Args:
            search_query: Text to search for
            filters: Additional filters to apply
            limit: Maximum number of results
            
        Returns:
            List[AuditLogResponse]: Matching audit log entries
        """
        with get_sync_db() as db:
            query = db.query(AuditLog)
            
            # Text search across multiple fields
            search_conditions = [
                AuditLog.resource_name.ilike(f"%{search_query}%"),
                AuditLog.changes_summary.ilike(f"%{search_query}%"),
                AuditLog.user_email.ilike(f"%{search_query}%"),
                AuditLog.resource_id.ilike(f"%{search_query}%")
            ]
            
            query = query.filter(or_(*search_conditions))
            
            # Apply additional filters
            if filters:
                for key, value in filters.items():
                    if hasattr(AuditLog, key):
                        query = query.filter(getattr(AuditLog, key) == value)
            
            audit_logs = query.order_by(desc(AuditLog.timestamp)).limit(limit).all()
            
            return [AuditLogResponse.from_orm(log) for log in audit_logs]
    
    @contextmanager
    def audit_context(
        self,
        user_id: str,
        user_email: str,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Context manager for setting audit context
        
        Args:
            user_id: User ID
            user_email: User email
            session_id: Session ID
            ip_address: IP address
            user_agent: User agent
        """
        context_id = str(uuid.uuid4())
        self._session_context[context_id] = {
            "user_id": user_id,
            "user_email": user_email,
            "session_id": session_id,
            "ip_address": ip_address,
            "user_agent": user_agent
        }
        
        try:
            yield context_id
        finally:
            self._session_context.pop(context_id, None)
    
    def _generate_changes_summary(
        self,
        old_values: Dict[str, Any],
        new_values: Dict[str, Any]
    ) -> str:
        """Generate human-readable changes summary"""
        changes = []
        
        # Find changed fields
        all_keys = set(old_values.keys()) | set(new_values.keys())
        
        for key in all_keys:
            old_val = old_values.get(key)
            new_val = new_values.get(key)
            
            if old_val != new_val:
                if old_val is None:
                    changes.append(f"Added {key}: {new_val}")
                elif new_val is None:
                    changes.append(f"Removed {key}: {old_val}")
                else:
                    changes.append(f"Changed {key}: {old_val} → {new_val}")
        
        return "; ".join(changes) if changes else "No changes detected"
    
    def _detect_suspicious_patterns(self, audit_logs: List[AuditLog]) -> List[Dict[str, Any]]:
        """Detect suspicious activity patterns"""
        patterns = []
        
        if not audit_logs:
            return patterns
        
        # Check for rapid successive actions
        timestamps = [log.timestamp for log in audit_logs]
        timestamps.sort()
        
        rapid_actions = 0
        for i in range(1, len(timestamps)):
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            if time_diff < 1:  # Less than 1 second between actions
                rapid_actions += 1
        
        if rapid_actions > 5:
            patterns.append({
                "type": "rapid_actions",
                "description": f"Detected {rapid_actions} rapid successive actions",
                "severity": "medium"
            })
        
        # Check for unusual access patterns
        resource_types = [log.resource_type for log in audit_logs]
        unique_resources = len(set(resource_types))
        
        if unique_resources > 10:
            patterns.append({
                "type": "broad_access",
                "description": f"Accessed {unique_resources} different resource types",
                "severity": "low"
            })
        
        # Check for high-risk actions
        high_risk_actions = [
            log for log in audit_logs 
            if log.risk_level in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]
        ]
        
        if len(high_risk_actions) > 3:
            patterns.append({
                "type": "high_risk_actions",
                "description": f"Performed {len(high_risk_actions)} high-risk actions",
                "severity": "high"
            })
        
        return patterns


# Global audit logger instance
audit_logger = AuditLogger()


# Decorator for automatic audit logging
def audit_action(
    action: AuditAction,
    resource_type: str,
    risk_level: RiskLevel = RiskLevel.LOW
):
    """
    Decorator for automatic audit logging of function calls
    
    Args:
        action: Type of action being performed
        resource_type: Type of resource being acted upon
        risk_level: Risk level of the action
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract audit context from kwargs or function arguments
            user_id = kwargs.get('user_id') or getattr(args[0], 'user_id', None)
            user_email = kwargs.get('user_email') or getattr(args[0], 'user_email', None)
            resource_id = kwargs.get('resource_id') or kwargs.get('id')
            
            if not user_id or not user_email:
                # Execute function without audit logging if context is missing
                return func(*args, **kwargs)
            
            # Execute function and capture result
            try:
                result = func(*args, **kwargs)
                
                # Log successful action
                audit_logger.log_action(
                    user_id=user_id,
                    user_email=user_email,
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(resource_id) if resource_id else "unknown",
                    risk_level=risk_level,
                    context={"function": func.__name__, "success": True}
                )
                
                return result
                
            except Exception as e:
                # Log failed action
                audit_logger.log_action(
                    user_id=user_id,
                    user_email=user_email,
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(resource_id) if resource_id else "unknown",
                    risk_level=RiskLevel.HIGH,
                    context={
                        "function": func.__name__,
                        "success": False,
                        "error": str(e)
                    }
                )
                raise
        
        return wrapper
    return decorator