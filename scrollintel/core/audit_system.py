"""
Comprehensive Audit Logging and Compliance System for ScrollIntel Launch MVP

This module provides detailed audit logging for all user actions, compliance reporting,
data export features, data retention policies, and audit trail export capabilities.
"""

import json
import csv
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from uuid import uuid4, UUID
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func, text
from sqlalchemy.orm import selectinload

from ..models.database import AuditLog, User
from ..models.database_utils import get_db
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class AuditAction(str, Enum):
    """Comprehensive audit actions for all user operations"""
    
    # Authentication & Authorization
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    PASSWORD_CHANGE = "auth.password.change"
    TOKEN_REFRESH = "auth.token.refresh"
    PERMISSION_DENIED = "auth.permission.denied"
    
    # User Management
    USER_CREATE = "user.create"
    USER_UPDATE = "user.update"
    USER_DELETE = "user.delete"
    USER_ROLE_CHANGE = "user.role.change"
    USER_ACTIVATE = "user.activate"
    USER_DEACTIVATE = "user.deactivate"
    
    # Organization Management
    ORG_CREATE = "organization.create"
    ORG_UPDATE = "organization.update"
    ORG_DELETE = "organization.delete"
    ORG_MEMBER_ADD = "organization.member.add"
    ORG_MEMBER_REMOVE = "organization.member.remove"
    
    # Workspace Management
    WORKSPACE_CREATE = "workspace.create"
    WORKSPACE_UPDATE = "workspace.update"
    WORKSPACE_DELETE = "workspace.delete"
    WORKSPACE_SHARE = "workspace.share"
    WORKSPACE_ACCESS = "workspace.access"
    
    # Data Operations
    DATA_UPLOAD = "data.upload"
    DATA_DOWNLOAD = "data.download"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"
    DATA_VIEW = "data.view"
    
    # Agent Operations
    AGENT_EXECUTE = "agent.execute"
    AGENT_CREATE = "agent.create"
    AGENT_UPDATE = "agent.update"
    AGENT_DELETE = "agent.delete"
    AGENT_DEPLOY = "agent.deploy"
    
    # Dashboard Operations
    DASHBOARD_CREATE = "dashboard.create"
    DASHBOARD_UPDATE = "dashboard.update"
    DASHBOARD_DELETE = "dashboard.delete"
    DASHBOARD_VIEW = "dashboard.view"
    DASHBOARD_SHARE = "dashboard.share"
    DASHBOARD_EXPORT = "dashboard.export"
    
    # Model Operations
    MODEL_CREATE = "model.create"
    MODEL_TRAIN = "model.train"
    MODEL_DEPLOY = "model.deploy"
    MODEL_PREDICT = "model.predict"
    MODEL_DELETE = "model.delete"
    
    # API Operations
    API_KEY_CREATE = "api.key.create"
    API_KEY_DELETE = "api.key.delete"
    API_REQUEST = "api.request"
    API_RATE_LIMIT = "api.rate_limit"
    
    # System Operations
    SYSTEM_CONFIG_CHANGE = "system.config.change"
    SYSTEM_BACKUP = "system.backup"
    SYSTEM_RESTORE = "system.restore"
    SYSTEM_MAINTENANCE = "system.maintenance"
    
    # Compliance Operations
    COMPLIANCE_REPORT_GENERATE = "compliance.report.generate"
    COMPLIANCE_EXPORT = "compliance.export"
    AUDIT_LOG_EXPORT = "audit.log.export"
    DATA_RETENTION_CLEANUP = "data.retention.cleanup"


class ComplianceLevel(str, Enum):
    """Compliance requirement levels"""
    BASIC = "basic"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    ENTERPRISE = "enterprise"


class RetentionPolicy(str, Enum):
    """Data retention policies"""
    DAYS_30 = "30_days"
    DAYS_90 = "90_days"
    MONTHS_6 = "6_months"
    YEAR_1 = "1_year"
    YEARS_3 = "3_years"
    YEARS_7 = "7_years"
    PERMANENT = "permanent"


@dataclass
class AuditEvent:
    """Structured audit event data"""
    id: str
    timestamp: datetime
    user_id: Optional[str]
    user_email: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    action: AuditAction
    resource_type: str
    resource_id: Optional[str]
    resource_name: Optional[str]
    details: Dict[str, Any]
    success: bool
    error_message: Optional[str]
    compliance_level: ComplianceLevel
    retention_policy: RetentionPolicy


@dataclass
class ComplianceReport:
    """Compliance report structure"""
    id: str
    report_type: str
    generated_at: datetime
    date_range_start: datetime
    date_range_end: datetime
    total_events: int
    successful_events: int
    failed_events: int
    security_events: int
    compliance_violations: List[Dict[str, Any]]
    user_activity_summary: Dict[str, Any]
    resource_access_summary: Dict[str, Any]
    recommendations: List[str]


class AuditSystem:
    """Comprehensive audit logging and compliance system"""
    
    def __init__(self):
        self._log_queue: asyncio.Queue = asyncio.Queue()
        self._background_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._retention_policies = self._load_retention_policies()
    
    def _load_retention_policies(self) -> Dict[str, timedelta]:
        """Load data retention policies"""
        return {
            RetentionPolicy.DAYS_30: timedelta(days=30),
            RetentionPolicy.DAYS_90: timedelta(days=90),
            RetentionPolicy.MONTHS_6: timedelta(days=180),
            RetentionPolicy.YEAR_1: timedelta(days=365),
            RetentionPolicy.YEARS_3: timedelta(days=1095),
            RetentionPolicy.YEARS_7: timedelta(days=2555),
            RetentionPolicy.PERMANENT: timedelta(days=36500)  # 100 years
        }
    
    async def start(self) -> None:
        """Start the audit system background tasks"""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._process_audit_logs())
            logger.info("Audit system started")
    
    async def stop(self) -> None:
        """Stop the audit system"""
        self._shutdown_event.set()
        if self._background_task:
            await self._background_task
        logger.info("Audit system stopped")
    
    async def log_event(
        self,
        action: AuditAction,
        resource_type: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        compliance_level: ComplianceLevel = ComplianceLevel.STANDARD,
        retention_policy: RetentionPolicy = RetentionPolicy.YEAR_1
    ) -> str:
        """Log an audit event"""
        
        event_id = str(uuid4())
        event = AuditEvent(
            id=event_id,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            user_email=user_email,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            details=details or {},
            success=success,
            error_message=error_message,
            compliance_level=compliance_level,
            retention_policy=retention_policy
        )
        
        # Add to processing queue
        await self._log_queue.put(event)
        
        # Log to structured logger for immediate visibility
        logger.info(
            f"Audit event: {action.value} on {resource_type}",
            user_id=user_id,
            action=action.value,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
            compliance_level=compliance_level.value
        )
        
        return event_id
    
    async def log_user_action(
        self,
        user_id: str,
        user_email: str,
        action: AuditAction,
        resource_type: str,
        resource_id: Optional[str] = None,
        resource_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> str:
        """Log a user action with enhanced context"""
        
        # Determine compliance level based on action
        compliance_level = self._determine_compliance_level(action)
        retention_policy = self._determine_retention_policy(action)
        
        return await self.log_event(
            action=action,
            resource_type=resource_type,
            user_id=user_id,
            user_email=user_email,
            resource_id=resource_id,
            resource_name=resource_name,
            details=details,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            compliance_level=compliance_level,
            retention_policy=retention_policy
        )
    
    def _determine_compliance_level(self, action: AuditAction) -> ComplianceLevel:
        """Determine compliance level based on action type"""
        high_risk_actions = {
            AuditAction.USER_DELETE,
            AuditAction.ORG_DELETE,
            AuditAction.WORKSPACE_DELETE,
            AuditAction.DATA_DELETE,
            AuditAction.SYSTEM_CONFIG_CHANGE,
            AuditAction.API_KEY_CREATE,
            AuditAction.USER_ROLE_CHANGE
        }
        
        if action in high_risk_actions:
            return ComplianceLevel.ENTERPRISE
        elif action.value.startswith(('auth.', 'user.', 'org.')):
            return ComplianceLevel.ENHANCED
        else:
            return ComplianceLevel.STANDARD
    
    def _determine_retention_policy(self, action: AuditAction) -> RetentionPolicy:
        """Determine retention policy based on action type"""
        permanent_actions = {
            AuditAction.USER_DELETE,
            AuditAction.ORG_DELETE,
            AuditAction.DATA_DELETE,
            AuditAction.SYSTEM_CONFIG_CHANGE
        }
        
        long_term_actions = {
            AuditAction.LOGIN_SUCCESS,
            AuditAction.LOGIN_FAILURE,
            AuditAction.USER_CREATE,
            AuditAction.USER_ROLE_CHANGE,
            AuditAction.PERMISSION_DENIED
        }
        
        if action in permanent_actions:
            return RetentionPolicy.PERMANENT
        elif action in long_term_actions:
            return RetentionPolicy.YEARS_7
        else:
            return RetentionPolicy.YEAR_1
    
    async def _process_audit_logs(self) -> None:
        """Background task to process audit log queue"""
        while not self._shutdown_event.is_set():
            try:
                # Wait for audit event or timeout
                event = await asyncio.wait_for(
                    self._log_queue.get(),
                    timeout=1.0
                )
                
                # Save to database
                await self._save_audit_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing audit log: {e}")
                continue
    
    async def _save_audit_event(self, event: AuditEvent) -> None:
        """Save audit event to database"""
        try:
            async for session in get_db():
                audit_log = AuditLog(
                    id=UUID(event.id),
                    user_id=UUID(event.user_id) if event.user_id else None,
                    action=event.action.value,
                    resource_type=event.resource_type,
                    resource_id=event.resource_id,
                    details=event.details,
                    ip_address=event.ip_address,
                    user_agent=event.user_agent,
                    session_id=event.session_id,
                    success=event.success,
                    error_message=event.error_message,
                    timestamp=event.timestamp
                )
                
                session.add(audit_log)
                await session.commit()
                break
                
        except Exception as e:
            logger.error(f"Failed to save audit event: {e}")
    
    async def search_audit_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        success: Optional[bool] = None,
        ip_address: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Search audit logs with comprehensive filtering"""
        
        async for session in get_db():
            query = select(AuditLog).options(selectinload(AuditLog.user))
            
            # Build filter conditions
            conditions = []
            
            if user_id:
                conditions.append(AuditLog.user_id == UUID(user_id))
            if action:
                conditions.append(AuditLog.action == action)
            if resource_type:
                conditions.append(AuditLog.resource_type == resource_type)
            if resource_id:
                conditions.append(AuditLog.resource_id == resource_id)
            if start_date:
                conditions.append(AuditLog.timestamp >= start_date)
            if end_date:
                conditions.append(AuditLog.timestamp <= end_date)
            if success is not None:
                conditions.append(AuditLog.success == success)
            if ip_address:
                conditions.append(AuditLog.ip_address == ip_address)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            # Order by timestamp descending
            query = query.order_by(desc(AuditLog.timestamp))
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            result = await session.execute(query)
            logs = result.scalars().all()
            
            # Convert to dict format
            return [
                {
                    "id": str(log.id),
                    "timestamp": log.timestamp.isoformat(),
                    "user_id": str(log.user_id) if log.user_id else None,
                    "user_email": log.user.email if log.user else None,
                    "action": log.action,
                    "resource_type": log.resource_type,
                    "resource_id": log.resource_id,
                    "details": log.details,
                    "ip_address": log.ip_address,
                    "user_agent": log.user_agent,
                    "session_id": log.session_id,
                    "success": log.success,
                    "error_message": log.error_message
                }
                for log in logs
            ]
    
    async def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str = "comprehensive"
    ) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        
        async for session in get_db():
            # Get total event counts
            total_query = select(func.count(AuditLog.id)).where(
                and_(
                    AuditLog.timestamp >= start_date,
                    AuditLog.timestamp <= end_date
                )
            )
            total_result = await session.execute(total_query)
            total_events = total_result.scalar()
            
            # Get success/failure counts
            success_query = select(func.count(AuditLog.id)).where(
                and_(
                    AuditLog.timestamp >= start_date,
                    AuditLog.timestamp <= end_date,
                    AuditLog.success == True
                )
            )
            success_result = await session.execute(success_query)
            successful_events = success_result.scalar()
            
            failed_events = total_events - successful_events
            
            # Get security events
            security_actions = [
                AuditAction.LOGIN_FAILURE.value,
                AuditAction.PERMISSION_DENIED.value,
                AuditAction.API_RATE_LIMIT.value
            ]
            
            security_query = select(func.count(AuditLog.id)).where(
                and_(
                    AuditLog.timestamp >= start_date,
                    AuditLog.timestamp <= end_date,
                    AuditLog.action.in_(security_actions)
                )
            )
            security_result = await session.execute(security_query)
            security_events = security_result.scalar()
            
            # Get compliance violations (failed security events)
            violations_query = select(AuditLog).where(
                and_(
                    AuditLog.timestamp >= start_date,
                    AuditLog.timestamp <= end_date,
                    AuditLog.action.in_(security_actions),
                    AuditLog.success == False
                )
            ).limit(100)
            
            violations_result = await session.execute(violations_query)
            violations = violations_result.scalars().all()
            
            compliance_violations = [
                {
                    "timestamp": v.timestamp.isoformat(),
                    "action": v.action,
                    "user_id": str(v.user_id) if v.user_id else None,
                    "ip_address": v.ip_address,
                    "error_message": v.error_message,
                    "details": v.details
                }
                for v in violations
            ]
            
            # Get user activity summary
            user_activity_query = select(
                AuditLog.user_id,
                func.count(AuditLog.id).label('activity_count')
            ).where(
                and_(
                    AuditLog.timestamp >= start_date,
                    AuditLog.timestamp <= end_date,
                    AuditLog.user_id.isnot(None)
                )
            ).group_by(AuditLog.user_id).limit(50)
            
            user_activity_result = await session.execute(user_activity_query)
            user_activity = user_activity_result.all()
            
            user_activity_summary = {
                str(row.user_id): row.activity_count
                for row in user_activity
            }
            
            # Get resource access summary
            resource_query = select(
                AuditLog.resource_type,
                func.count(AuditLog.id).label('access_count')
            ).where(
                and_(
                    AuditLog.timestamp >= start_date,
                    AuditLog.timestamp <= end_date
                )
            ).group_by(AuditLog.resource_type)
            
            resource_result = await session.execute(resource_query)
            resource_access = resource_result.all()
            
            resource_access_summary = {
                row.resource_type: row.access_count
                for row in resource_access
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                total_events, failed_events, security_events, len(compliance_violations)
            )
            
            return ComplianceReport(
                id=str(uuid4()),
                report_type=report_type,
                generated_at=datetime.utcnow(),
                date_range_start=start_date,
                date_range_end=end_date,
                total_events=total_events,
                successful_events=successful_events,
                failed_events=failed_events,
                security_events=security_events,
                compliance_violations=compliance_violations,
                user_activity_summary=user_activity_summary,
                resource_access_summary=resource_access_summary,
                recommendations=recommendations
            )
    
    def _generate_recommendations(
        self,
        total_events: int,
        failed_events: int,
        security_events: int,
        violations_count: int
    ) -> List[str]:
        """Generate compliance recommendations based on audit data"""
        
        recommendations = []
        
        if failed_events > 0:
            failure_rate = (failed_events / total_events) * 100
            if failure_rate > 5:
                recommendations.append(
                    f"High failure rate detected ({failure_rate:.1f}%). "
                    "Review error handling and user training."
                )
        
        if security_events > 0:
            recommendations.append(
                f"Monitor {security_events} security events. "
                "Consider implementing additional security measures."
            )
        
        if violations_count > 10:
            recommendations.append(
                f"Multiple compliance violations detected ({violations_count}). "
                "Review access controls and user permissions."
            )
        
        if total_events > 10000:
            recommendations.append(
                "High activity volume detected. Consider implementing "
                "automated monitoring and alerting."
            )
        
        if not recommendations:
            recommendations.append("No significant compliance issues detected.")
        
        return recommendations
    
    async def export_audit_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json",
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None
    ) -> str:
        """Export audit logs for compliance purposes"""
        
        # Get audit logs
        logs = await self.search_audit_logs(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        # Create export directory
        export_dir = Path("exports/audit_logs")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_logs_{timestamp}.{format}"
        filepath = export_dir / filename
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump({
                    "export_metadata": {
                        "generated_at": datetime.utcnow().isoformat(),
                        "date_range_start": start_date.isoformat(),
                        "date_range_end": end_date.isoformat(),
                        "total_records": len(logs),
                        "filters": {
                            "user_id": user_id,
                            "action": action,
                            "resource_type": resource_type
                        }
                    },
                    "audit_logs": logs
                }, f, indent=2, default=str)
        
        elif format == "csv":
            with open(filepath, 'w', newline='') as f:
                if logs:
                    writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                    writer.writeheader()
                    writer.writerows(logs)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Log the export action
        await self.log_event(
            action=AuditAction.AUDIT_LOG_EXPORT,
            resource_type="audit_logs",
            details={
                "export_format": format,
                "date_range_start": start_date.isoformat(),
                "date_range_end": end_date.isoformat(),
                "record_count": len(logs),
                "file_path": str(filepath)
            }
        )
        
        logger.info(f"Audit logs exported to {filepath}")
        return str(filepath)
    
    async def cleanup_expired_logs(self) -> int:
        """Clean up audit logs based on retention policies"""
        
        cleaned_count = 0
        
        async for session in get_db():
            for policy, retention_period in self._retention_policies.items():
                if policy == RetentionPolicy.PERMANENT:
                    continue
                
                cutoff_date = datetime.utcnow() - retention_period
                
                # Find logs to delete (this is a simplified approach)
                # In practice, you'd need to store retention policy with each log
                delete_query = text("""
                    DELETE FROM audit_logs 
                    WHERE timestamp < :cutoff_date 
                    AND action NOT IN (
                        'user.delete', 'org.delete', 'data.delete', 'system.config.change'
                    )
                """)
                
                result = await session.execute(delete_query, {"cutoff_date": cutoff_date})
                deleted_count = result.rowcount
                cleaned_count += deleted_count
                
                await session.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} audit logs older than {retention_period}")
        
        # Log the cleanup action
        await self.log_event(
            action=AuditAction.DATA_RETENTION_CLEANUP,
            resource_type="audit_logs",
            details={
                "cleaned_count": cleaned_count,
                "cleanup_date": datetime.utcnow().isoformat()
            }
        )
        
        return cleaned_count


# Global audit system instance
audit_system = AuditSystem()


@asynccontextmanager
async def audit_context():
    """Context manager for audit system lifecycle"""
    await audit_system.start()
    try:
        yield audit_system
    finally:
        await audit_system.stop()