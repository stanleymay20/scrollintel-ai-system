"""Audit logging system for data governance."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..models.governance_models import (
    AuditEvent, AuditEventType, UsageMetrics
)
from ..models.governance_database import (
    AuditEventModel, UsageMetricsModel, UserModel
)
from ..models.database import get_db_session
from ..core.exceptions import AIDataReadinessError


logger = logging.getLogger(__name__)


class AuditLoggerError(AIDataReadinessError):
    """Exception raised for audit logger errors."""
    pass


class AuditLogger:
    """Audit logging system for tracking all data access and modifications."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def log_data_access(
        self,
        user_id: str,
        resource_id: str,
        resource_type: str,
        action: str = "read",
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> AuditEvent:
        """Log data access event."""
        return self._log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=user_id,
            resource_id=resource_id,
            resource_type=resource_type,
            action=action,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id
        )
    
    def log_data_modification(
        self,
        user_id: str,
        resource_id: str,
        resource_type: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> AuditEvent:
        """Log data modification event."""
        return self._log_event(
            event_type=AuditEventType.DATA_MODIFICATION,
            user_id=user_id,
            resource_id=resource_id,
            resource_type=resource_type,
            action=action,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id
        )
    
    def log_policy_change(
        self,
        user_id: str,
        policy_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> AuditEvent:
        """Log policy change event."""
        return self._log_event(
            event_type=AuditEventType.POLICY_CHANGE,
            user_id=user_id,
            resource_id=policy_id,
            resource_type="policy",
            action=action,
            details=details or {},
            ip_address=ip_address,
            session_id=session_id
        )
    
    def log_user_action(
        self,
        user_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditEvent:
        """Log general user action."""
        return self._log_event(
            event_type=AuditEventType.USER_ACTION,
            user_id=user_id,
            resource_id=resource_id,
            resource_type=resource_type,
            action=action,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            success=success,
            error_message=error_message
        )
    
    def log_system_event(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        user_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditEvent:
        """Log system event."""
        return self._log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            user_id=user_id or "system",
            resource_id=resource_id,
            resource_type=resource_type,
            action=action,
            details=details or {},
            success=success,
            error_message=error_message
        )
    
    def log_compliance_check(
        self,
        user_id: str,
        resource_id: str,
        resource_type: str,
        compliance_type: str,
        result: Dict[str, Any],
        success: bool = True
    ) -> AuditEvent:
        """Log compliance check event."""
        return self._log_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            user_id=user_id,
            resource_id=resource_id,
            resource_type=resource_type,
            action=f"compliance_check_{compliance_type}",
            details=result,
            success=success
        )
    
    def get_audit_trail(
        self,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Get audit trail with filters."""
        try:
            with get_db_session() as session:
                query = session.query(AuditEventModel)
                
                # Apply filters
                if resource_id:
                    query = query.filter(AuditEventModel.resource_id == resource_id)
                
                if resource_type:
                    query = query.filter(AuditEventModel.resource_type == resource_type)
                
                if user_id:
                    query = query.filter(AuditEventModel.user_id == user_id)
                
                if event_type:
                    query = query.filter(AuditEventModel.event_type == event_type.value)
                
                if start_date:
                    query = query.filter(AuditEventModel.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(AuditEventModel.timestamp <= end_date)
                
                # Order by timestamp descending
                query = query.order_by(AuditEventModel.timestamp.desc())
                
                events = query.limit(limit).all()
                
                return [self._model_to_dataclass(event) for event in events]
                
        except Exception as e:
            self.logger.error(f"Error retrieving audit trail: {str(e)}")
            raise AuditLoggerError(f"Failed to retrieve audit trail: {str(e)}")
    
    def get_user_activity_summary(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get user activity summary."""
        try:
            with get_db_session() as session:
                query = session.query(AuditEventModel).filter(
                    AuditEventModel.user_id == user_id
                )
                
                if start_date:
                    query = query.filter(AuditEventModel.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(AuditEventModel.timestamp <= end_date)
                
                events = query.all()
                
                # Calculate summary statistics
                total_events = len(events)
                successful_events = sum(1 for event in events if event.success)
                failed_events = total_events - successful_events
                
                # Group by event type
                event_types = {}
                for event in events:
                    event_type = event.event_type
                    if event_type not in event_types:
                        event_types[event_type] = 0
                    event_types[event_type] += 1
                
                # Group by action
                actions = {}
                for event in events:
                    action = event.action
                    if action not in actions:
                        actions[action] = 0
                    actions[action] += 1
                
                # Get unique resources accessed
                unique_resources = set()
                for event in events:
                    if event.resource_id:
                        unique_resources.add(f"{event.resource_type}:{event.resource_id}")
                
                return {
                    'user_id': user_id,
                    'period_start': start_date,
                    'period_end': end_date,
                    'total_events': total_events,
                    'successful_events': successful_events,
                    'failed_events': failed_events,
                    'success_rate': successful_events / total_events if total_events > 0 else 0,
                    'event_types': event_types,
                    'actions': actions,
                    'unique_resources_accessed': len(unique_resources),
                    'last_activity': events[0].timestamp if events else None
                }
                
        except Exception as e:
            self.logger.error(f"Error getting user activity summary: {str(e)}")
            raise AuditLoggerError(f"Failed to get user activity summary: {str(e)}")
    
    def get_resource_access_summary(
        self,
        resource_id: str,
        resource_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get resource access summary."""
        try:
            with get_db_session() as session:
                query = session.query(AuditEventModel).filter(
                    AuditEventModel.resource_id == resource_id,
                    AuditEventModel.resource_type == resource_type
                )
                
                if start_date:
                    query = query.filter(AuditEventModel.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(AuditEventModel.timestamp <= end_date)
                
                events = query.all()
                
                # Calculate summary statistics
                total_accesses = len(events)
                unique_users = len(set(event.user_id for event in events))
                
                # Group by user
                user_accesses = {}
                for event in events:
                    user_id = event.user_id
                    if user_id not in user_accesses:
                        user_accesses[user_id] = 0
                    user_accesses[user_id] += 1
                
                # Find most frequent user
                most_frequent_user = max(user_accesses.items(), key=lambda x: x[1])[0] if user_accesses else None
                
                # Group by action
                actions = {}
                for event in events:
                    action = event.action
                    if action not in actions:
                        actions[action] = 0
                    actions[action] += 1
                
                return {
                    'resource_id': resource_id,
                    'resource_type': resource_type,
                    'period_start': start_date,
                    'period_end': end_date,
                    'total_accesses': total_accesses,
                    'unique_users': unique_users,
                    'user_accesses': user_accesses,
                    'most_frequent_user': most_frequent_user,
                    'actions': actions,
                    'last_access': events[0].timestamp if events else None
                }
                
        except Exception as e:
            self.logger.error(f"Error getting resource access summary: {str(e)}")
            raise AuditLoggerError(f"Failed to get resource access summary: {str(e)}")
    
    def update_usage_metrics(
        self,
        resource_id: str,
        resource_type: str,
        period_start: datetime,
        period_end: datetime
    ) -> UsageMetrics:
        """Update usage metrics for a resource."""
        try:
            with get_db_session() as session:
                # Get access events for the period
                events = session.query(AuditEventModel).filter(
                    AuditEventModel.resource_id == resource_id,
                    AuditEventModel.resource_type == resource_type,
                    AuditEventModel.event_type == AuditEventType.DATA_ACCESS.value,
                    AuditEventModel.timestamp >= period_start,
                    AuditEventModel.timestamp <= period_end
                ).all()
                
                # Calculate metrics
                access_count = len(events)
                unique_users = len(set(event.user_id for event in events))
                last_accessed = max(event.timestamp for event in events) if events else None
                
                # Find most frequent user
                user_counts = {}
                for event in events:
                    user_id = event.user_id
                    user_counts[user_id] = user_counts.get(user_id, 0) + 1
                
                most_frequent_user = max(user_counts.items(), key=lambda x: x[1])[0] if user_counts else None
                
                # Calculate access patterns (hourly distribution)
                hourly_pattern = {}
                for event in events:
                    hour = event.timestamp.hour
                    hourly_pattern[hour] = hourly_pattern.get(hour, 0) + 1
                
                # Create or update usage metrics
                existing_metrics = session.query(UsageMetricsModel).filter(
                    UsageMetricsModel.resource_id == resource_id,
                    UsageMetricsModel.resource_type == resource_type,
                    UsageMetricsModel.period_start == period_start,
                    UsageMetricsModel.period_end == period_end
                ).first()
                
                if existing_metrics:
                    existing_metrics.access_count = access_count
                    existing_metrics.unique_users = unique_users
                    existing_metrics.last_accessed = last_accessed
                    existing_metrics.most_frequent_user = most_frequent_user
                    existing_metrics.access_patterns = {'hourly': hourly_pattern}
                    
                    session.commit()
                    session.refresh(existing_metrics)
                    
                    metrics_model = existing_metrics
                else:
                    metrics_model = UsageMetricsModel(
                        resource_id=resource_id,
                        resource_type=resource_type,
                        access_count=access_count,
                        unique_users=unique_users,
                        last_accessed=last_accessed,
                        most_frequent_user=most_frequent_user,
                        access_patterns={'hourly': hourly_pattern},
                        performance_metrics={},
                        period_start=period_start,
                        period_end=period_end
                    )
                    
                    session.add(metrics_model)
                    session.commit()
                    session.refresh(metrics_model)
                
                return self._usage_model_to_dataclass(metrics_model)
                
        except Exception as e:
            self.logger.error(f"Error updating usage metrics: {str(e)}")
            raise AuditLoggerError(f"Failed to update usage metrics: {str(e)}")
    
    def _log_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        action: str,
        details: Dict[str, Any],
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditEvent:
        """Internal method to log an audit event."""
        try:
            with get_db_session() as session:
                audit_event = AuditEventModel(
                    event_type=event_type.value,
                    user_id=user_id,
                    resource_id=resource_id,
                    resource_type=resource_type,
                    action=action,
                    details=details,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    session_id=session_id,
                    success=success,
                    error_message=error_message
                )
                
                session.add(audit_event)
                session.commit()
                session.refresh(audit_event)
                
                return self._model_to_dataclass(audit_event)
                
        except Exception as e:
            self.logger.error(f"Error logging audit event: {str(e)}")
            raise AuditLoggerError(f"Failed to log audit event: {str(e)}")
    
    def _model_to_dataclass(self, model: AuditEventModel) -> AuditEvent:
        """Convert database model to dataclass."""
        return AuditEvent(
            id=str(model.id),
            event_type=AuditEventType(model.event_type),
            user_id=model.user_id,
            resource_id=model.resource_id,
            resource_type=model.resource_type,
            action=model.action,
            details=model.details or {},
            ip_address=model.ip_address,
            user_agent=model.user_agent,
            session_id=model.session_id,
            timestamp=model.timestamp,
            success=model.success,
            error_message=model.error_message
        )
    
    def _usage_model_to_dataclass(self, model: UsageMetricsModel) -> UsageMetrics:
        """Convert usage metrics model to dataclass."""
        return UsageMetrics(
            resource_id=model.resource_id,
            resource_type=model.resource_type,
            access_count=model.access_count,
            unique_users=model.unique_users,
            last_accessed=model.last_accessed,
            most_frequent_user=str(model.most_frequent_user) if model.most_frequent_user else None,
            access_patterns=model.access_patterns or {},
            performance_metrics=model.performance_metrics or {},
            period_start=model.period_start,
            period_end=model.period_end
        )