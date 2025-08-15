"""
Audit logging system for EXOUSIA security.
Tracks all operations with user context for security and compliance.
"""

import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from uuid import uuid4, UUID
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc

from ..core.interfaces import AuditEvent, SecurityContext, SecurityError
from ..models.database import AuditLog, User
from ..models.database_utils import get_db


class AuditAction:
    """Standard audit actions."""
    
    # Authentication actions
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    TOKEN_REFRESH = "auth.token.refresh"
    PASSWORD_CHANGE = "auth.password.change"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_REGISTER = "user.register"
    
    # User management actions
    USER_CREATE = "user.create"
    USER_UPDATE = "user.update"
    USER_DELETE = "user.delete"
    USER_ROLE_CHANGE = "user.role.change"
    USER_PERMISSION_CHANGE = "user.permission.change"
    
    # Agent actions
    AGENT_CREATE = "agent.create"
    AGENT_UPDATE = "agent.update"
    AGENT_DELETE = "agent.delete"
    AGENT_EXECUTE = "agent.execute"
    AGENT_START = "agent.start"
    AGENT_STOP = "agent.stop"
    AGENT_RESTART = "agent.restart"
    AGENT_COMPLETE = "agent.complete"
    
    # Workflow actions
    WORKFLOW_START = "workflow.start"
    WORKFLOW_COMPLETE = "workflow.complete"
    WORKFLOW_ERROR = "workflow.error"
    
    # Dataset actions
    DATASET_CREATE = "dataset.create"
    DATASET_CREATE_COMPLETE = "dataset.create.complete"
    DATASET_UPDATE = "dataset.update"
    DATASET_DELETE = "dataset.delete"
    DATASET_UPLOAD = "dataset.upload"
    DATASET_DOWNLOAD = "dataset.download"
    
    # File upload actions
    FILE_UPLOAD = "file.upload"
    FILE_UPLOAD_COMPLETE = "file.upload.complete"
    FILE_DELETE = "file.delete"
    FILE_PROCESS = "file.process"
    
    # Model actions
    MODEL_CREATE = "model.create"
    MODEL_UPDATE = "model.update"
    MODEL_DELETE = "model.delete"
    MODEL_TRAIN = "model.train"
    MODEL_DEPLOY = "model.deploy"
    MODEL_PREDICT = "model.predict"
    
    # Dashboard actions
    DASHBOARD_CREATE = "dashboard.create"
    DASHBOARD_UPDATE = "dashboard.update"
    DASHBOARD_DELETE = "dashboard.delete"
    DASHBOARD_VIEW = "dashboard.view"
    DASHBOARD_SHARE = "dashboard.share"
    
    # System actions
    SYSTEM_CONFIG_CHANGE = "system.config.change"
    SYSTEM_BACKUP = "system.backup"
    SYSTEM_RESTORE = "system.restore"
    SYSTEM_HEALTH_CHECK = "system.health.check"
    
    # Security actions
    PERMISSION_DENIED = "security.permission.denied"
    SUSPICIOUS_ACTIVITY = "security.suspicious.activity"
    RATE_LIMIT_EXCEEDED = "security.rate_limit.exceeded"


class AuditLogger:
    """Handles audit logging operations."""
    
    def __init__(self):
        self._log_queue: asyncio.Queue = asyncio.Queue()
        self._background_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the background audit logging task."""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._process_audit_logs())
    
    async def stop(self) -> None:
        """Stop the background audit logging task."""
        self._shutdown_event.set()
        if self._background_task:
            await self._background_task
    
    async def log(self, action: str, resource_type: str, resource_id: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None, user_id: Optional[UUID] = None,
                  ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                  session_id: Optional[str] = None, success: bool = True,
                  error_message: Optional[str] = None) -> None:
        """Log an audit event."""
        
        audit_log = AuditLog(
            id=uuid4(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            success=success,
            error_message=error_message,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add to queue for background processing
        await self._log_queue.put(audit_log)
    
    async def log_with_context(self, context: SecurityContext, action: str, 
                             resource_type: str, resource_id: Optional[str] = None,
                             details: Optional[Dict[str, Any]] = None,
                             success: bool = True, error_message: Optional[str] = None) -> None:
        """Log an audit event with security context."""
        await self.log(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            user_id=UUID(context.user_id) if context.user_id else None,
            ip_address=context.ip_address,
            session_id=context.session_id,
            success=success,
            error_message=error_message
        )
    
    async def log_authentication(self, action: str, email: str, ip_address: str,
                               user_agent: str, success: bool = True,
                               error_message: Optional[str] = None,
                               user_id: Optional[UUID] = None) -> None:
        """Log authentication events."""
        details = {
            "email": email,
            "user_agent": user_agent
        }
        
        await self.log(
            action=action,
            resource_type="authentication",
            details=details,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
    
    async def log_permission_denied(self, context: SecurityContext, action: str,
                                  resource_type: str, resource_id: Optional[str] = None) -> None:
        """Log permission denied events."""
        details = {
            "attempted_action": action,
            "user_role": context.role.value,
            "user_permissions": context.permissions
        }
        
        await self.log_with_context(
            context=context,
            action=AuditAction.PERMISSION_DENIED,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            success=False,
            error_message=f"Permission denied for {action} on {resource_type}"
        )
    
    async def log_suspicious_activity(self, context: SecurityContext, activity_type: str,
                                    details: Dict[str, Any]) -> None:
        """Log suspicious activity."""
        await self.log_with_context(
            context=context,
            action=AuditAction.SUSPICIOUS_ACTIVITY,
            resource_type="security",
            details={
                "activity_type": activity_type,
                **details
            },
            success=False,
            error_message=f"Suspicious activity detected: {activity_type}"
        )
    
    async def _process_audit_logs(self) -> None:
        """Background task to process audit logs."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for audit log or shutdown
                audit_log = await asyncio.wait_for(
                    self._log_queue.get(),
                    timeout=1.0
                )
                
                # Save to database
                await self._save_audit_log(audit_log)
                
            except asyncio.TimeoutError:
                # Continue loop to check shutdown event
                continue
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing audit log: {e}")
                continue
    
    async def _save_audit_log(self, audit_log: AuditLog) -> None:
        """Save audit log to database."""
        try:
            async for session in get_db():
                session.add(audit_log)
                await session.commit()
                break  # Only need one iteration
        except Exception as e:
            print(f"Failed to save audit log: {e}")
            # In production, you might want to write to a fallback log file
    
    async def get_audit_logs(self, user_id: Optional[UUID] = None,
                           action: Optional[str] = None,
                           resource_type: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           limit: int = 100,
                           offset: int = 0) -> List[AuditLog]:
        """Retrieve audit logs with filtering."""
        async for session in get_db():
            query = select(AuditLog)
            
            # Apply filters
            conditions = []
            if user_id:
                conditions.append(AuditLog.user_id == user_id)
            if action:
                conditions.append(AuditLog.action == action)
            if resource_type:
                conditions.append(AuditLog.resource_type == resource_type)
            if start_date:
                conditions.append(AuditLog.timestamp >= start_date)
            if end_date:
                conditions.append(AuditLog.timestamp <= end_date)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            # Order by timestamp descending
            query = query.order_by(desc(AuditLog.timestamp))
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()
            break  # Only need one iteration
    
    async def get_user_activity(self, user_id: UUID, days: int = 30) -> List[AuditLog]:
        """Get recent activity for a specific user."""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        return await self.get_audit_logs(
            user_id=user_id,
            start_date=start_date,
            limit=1000
        )
    
    async def get_failed_logins(self, hours: int = 24) -> List[AuditLog]:
        """Get failed login attempts in the last N hours."""
        start_date = datetime.now(timezone.utc) - timedelta(hours=hours)
        return await self.get_audit_logs(
            action=AuditAction.LOGIN_FAILURE,
            start_date=start_date,
            limit=1000
        )
    
    async def get_security_events(self, hours: int = 24) -> List[AuditLog]:
        """Get security-related events in the last N hours."""
        start_date = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        async for session in get_db():
            query = select(AuditLog).where(
                and_(
                    AuditLog.timestamp >= start_date,
                    or_(
                        AuditLog.action == AuditAction.PERMISSION_DENIED,
                        AuditLog.action == AuditAction.SUSPICIOUS_ACTIVITY,
                        AuditLog.action == AuditAction.RATE_LIMIT_EXCEEDED,
                        AuditLog.action == AuditAction.LOGIN_FAILURE
                    )
                )
            ).order_by(desc(AuditLog.timestamp)).limit(1000)
            
            result = await session.execute(query)
            return result.scalars().all()
            break  # Only need one iteration
    
    async def export_audit_logs(self, start_date: datetime, end_date: datetime,
                              format: str = "json") -> str:
        """Export audit logs for compliance purposes."""
        logs = await self.get_audit_logs(
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        if format == "json":
            return json.dumps([
                {
                    "id": str(log.id),
                    "user_id": str(log.user_id) if log.user_id else None,
                    "action": log.action,
                    "resource_type": log.resource_type,
                    "resource_id": log.resource_id,
                    "details": log.details,
                    "ip_address": log.ip_address,
                    "user_agent": log.user_agent,
                    "session_id": log.session_id,
                    "success": log.success,
                    "error_message": log.error_message,
                    "timestamp": log.timestamp.isoformat()
                }
                for log in logs
            ], indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


@asynccontextmanager
async def audit_context(logger: AuditLogger):
    """Context manager for audit logging lifecycle."""
    await logger.start()
    try:
        yield logger
    finally:
        await logger.stop()


# Global audit logger instance
audit_logger = AuditLogger()