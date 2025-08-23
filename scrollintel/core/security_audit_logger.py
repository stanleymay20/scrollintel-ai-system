"""
Comprehensive Security Audit Logger
Handles all security events and audit logging for enterprise integrations
"""
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from ..models.security_audit_models import (
    SecurityAuditLog, SecurityEventType, SeverityLevel, 
    AuditEventStatus, SecurityAuditLogCreate, SecurityMetrics
)
from ..core.database_connection_manager import get_sync_session
from ..core.logging_config import get_logger

logger = get_logger(__name__)

class SecurityAuditLogger:
    """Comprehensive security audit logging system"""
    
    def __init__(self):
        self.correlation_cache = {}
        self.risk_thresholds = {
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 5,
            SeverityLevel.HIGH: 8,
            SeverityLevel.CRITICAL: 10
        }
    
    async def log_security_event(
        self,
        event_type: SecurityEventType,
        action: str,
        outcome: str,
        severity: SeverityLevel = SeverityLevel.LOW,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        integration_id: Optional[str] = None
    ) -> str:
        """Log a security event with comprehensive details"""
        try:
            event_id = str(uuid.uuid4())
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(
                event_type, severity, details, user_id, source_ip
            )
            
            # Generate correlation ID if not provided
            if not correlation_id:
                correlation_id = self._generate_correlation_id(
                    user_id, session_id, source_ip, action
                )
            
            audit_log = SecurityAuditLog(
                id=event_id,
                event_type=event_type.value,
                severity=severity.value,
                user_id=user_id,
                session_id=session_id,
                source_ip=source_ip,
                user_agent=user_agent,
                resource=resource,
                action=action,
                outcome=outcome,
                details=details or {},
                risk_score=risk_score,
                correlation_id=correlation_id,
                integration_id=integration_id,
                status=AuditEventStatus.PENDING.value
            )
            
            with get_sync_session() as db:
                db.add(audit_log)
                db.commit()
                
            logger.info(f"Security event logged: {event_id} - {event_type.value}")
            
            # Trigger real-time analysis for high-risk events
            if risk_score >= self.risk_thresholds[SeverityLevel.HIGH]:
                await self._trigger_real_time_analysis(event_id, audit_log)
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log security event: {str(e)}")
            raise
    
    async def log_authentication_event(
        self,
        user_id: str,
        action: str,
        outcome: str,
        source_ip: str,
        user_agent: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log authentication-specific events"""
        severity = SeverityLevel.MEDIUM if outcome == "failure" else SeverityLevel.LOW
        
        auth_details = {
            "authentication_method": details.get("method", "unknown") if details else "unknown",
            "failure_reason": details.get("failure_reason") if details and outcome == "failure" else None,
            "mfa_used": details.get("mfa_used", False) if details else False,
            **(details or {})
        }
        
        return await self.log_security_event(
            event_type=SecurityEventType.AUTHENTICATION,
            action=action,
            outcome=outcome,
            severity=severity,
            user_id=user_id,
            source_ip=source_ip,
            user_agent=user_agent,
            details=auth_details
        )
    
    async def log_data_access_event(
        self,
        user_id: str,
        resource: str,
        action: str,
        outcome: str,
        integration_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log data access events"""
        severity = SeverityLevel.HIGH if "sensitive" in resource.lower() else SeverityLevel.MEDIUM
        
        access_details = {
            "data_classification": details.get("classification", "unknown") if details else "unknown",
            "record_count": details.get("record_count", 0) if details else 0,
            "query_type": details.get("query_type") if details else None,
            **(details or {})
        }
        
        return await self.log_security_event(
            event_type=SecurityEventType.DATA_ACCESS,
            action=action,
            outcome=outcome,
            severity=severity,
            user_id=user_id,
            resource=resource,
            integration_id=integration_id,
            details=access_details
        )
    
    async def log_configuration_change(
        self,
        user_id: str,
        resource: str,
        action: str,
        outcome: str,
        old_config: Optional[Dict[str, Any]] = None,
        new_config: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log configuration change events"""
        config_details = {
            "old_configuration": old_config,
            "new_configuration": new_config,
            "change_type": details.get("change_type", "modification") if details else "modification",
            **(details or {})
        }
        
        return await self.log_security_event(
            event_type=SecurityEventType.CONFIGURATION_CHANGE,
            action=action,
            outcome=outcome,
            severity=SeverityLevel.MEDIUM,
            user_id=user_id,
            resource=resource,
            details=config_details
        )
    
    async def log_threat_detection(
        self,
        threat_type: str,
        severity: SeverityLevel,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log threat detection events"""
        threat_details = {
            "threat_type": threat_type,
            "detection_method": details.get("detection_method", "automated") if details else "automated",
            "confidence_score": details.get("confidence_score", 0.5) if details else 0.5,
            "indicators": details.get("indicators", []) if details else [],
            **(details or {})
        }
        
        return await self.log_security_event(
            event_type=SecurityEventType.THREAT_DETECTED,
            action="threat_detection",
            outcome="detected",
            severity=severity,
            user_id=user_id,
            source_ip=source_ip,
            details=threat_details
        )
    
    def get_security_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[SecurityEventType]] = None,
        severity_levels: Optional[List[SeverityLevel]] = None,
        user_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[SecurityAuditLog]:
        """Retrieve security events with filtering"""
        try:
            with get_sync_session() as db:
                query = db.query(SecurityAuditLog)
                
                if start_time:
                    query = query.filter(SecurityAuditLog.timestamp >= start_time)
                if end_time:
                    query = query.filter(SecurityAuditLog.timestamp <= end_time)
                if event_types:
                    query = query.filter(SecurityAuditLog.event_type.in_([et.value for et in event_types]))
                if severity_levels:
                    query = query.filter(SecurityAuditLog.severity.in_([sl.value for sl in severity_levels]))
                if user_id:
                    query = query.filter(SecurityAuditLog.user_id == user_id)
                
                return query.order_by(desc(SecurityAuditLog.timestamp)).limit(limit).all()
                
        except Exception as e:
            logger.error(f"Failed to retrieve security events: {str(e)}")
            return []
    
    def get_security_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> SecurityMetrics:
        """Calculate security metrics and KPIs"""
        try:
            if not start_time:
                start_time = datetime.utcnow() - timedelta(days=30)
            if not end_time:
                end_time = datetime.utcnow()
            
            with get_sync_session() as db:
                # Total events
                total_events = db.query(func.count(SecurityAuditLog.id)).filter(
                    and_(
                        SecurityAuditLog.timestamp >= start_time,
                        SecurityAuditLog.timestamp <= end_time
                    )
                ).scalar() or 0
                
                # Critical events
                critical_events = db.query(func.count(SecurityAuditLog.id)).filter(
                    and_(
                        SecurityAuditLog.timestamp >= start_time,
                        SecurityAuditLog.timestamp <= end_time,
                        SecurityAuditLog.severity == SeverityLevel.CRITICAL.value
                    )
                ).scalar() or 0
                
                # High severity events
                high_severity_events = db.query(func.count(SecurityAuditLog.id)).filter(
                    and_(
                        SecurityAuditLog.timestamp >= start_time,
                        SecurityAuditLog.timestamp <= end_time,
                        SecurityAuditLog.severity == SeverityLevel.HIGH.value
                    )
                ).scalar() or 0
                
                return SecurityMetrics(
                    total_events=total_events,
                    critical_events=critical_events,
                    high_severity_events=high_severity_events,
                    incidents_open=0,  # Will be calculated from incidents table
                    incidents_resolved=0,  # Will be calculated from incidents table
                    average_response_time=0.0,  # Will be calculated from incidents
                    compliance_score=95.0,  # Placeholder - calculate based on violations
                    threat_detection_rate=0.85,  # Placeholder - calculate based on detections
                    false_positive_rate=0.05  # Placeholder - calculate based on validations
                )
                
        except Exception as e:
            logger.error(f"Failed to calculate security metrics: {str(e)}")
            return SecurityMetrics(
                total_events=0, critical_events=0, high_severity_events=0,
                incidents_open=0, incidents_resolved=0, average_response_time=0.0,
                compliance_score=0.0, threat_detection_rate=0.0, false_positive_rate=0.0
            )
    
    def _calculate_risk_score(
        self,
        event_type: SecurityEventType,
        severity: SeverityLevel,
        details: Optional[Dict[str, Any]],
        user_id: Optional[str],
        source_ip: Optional[str]
    ) -> int:
        """Calculate risk score for security event"""
        base_score = self.risk_thresholds[severity]
        
        # Adjust based on event type
        type_multipliers = {
            SecurityEventType.SYSTEM_BREACH: 2.0,
            SecurityEventType.THREAT_DETECTED: 1.8,
            SecurityEventType.COMPLIANCE_VIOLATION: 1.5,
            SecurityEventType.SUSPICIOUS_ACTIVITY: 1.3,
            SecurityEventType.DATA_ACCESS: 1.2,
            SecurityEventType.CONFIGURATION_CHANGE: 1.1,
            SecurityEventType.AUTHORIZATION: 1.0,
            SecurityEventType.AUTHENTICATION: 0.8
        }
        
        score = base_score * type_multipliers.get(event_type, 1.0)
        
        # Additional risk factors
        if details:
            if details.get("privileged_access"):
                score *= 1.5
            if details.get("external_source"):
                score *= 1.3
            if details.get("after_hours"):
                score *= 1.2
        
        return min(int(score), 10)  # Cap at 10
    
    def _generate_correlation_id(
        self,
        user_id: Optional[str],
        session_id: Optional[str],
        source_ip: Optional[str],
        action: str
    ) -> str:
        """Generate correlation ID for related events"""
        key = f"{user_id}:{session_id}:{source_ip}:{action}"
        
        if key in self.correlation_cache:
            return self.correlation_cache[key]
        
        correlation_id = str(uuid.uuid4())
        self.correlation_cache[key] = correlation_id
        
        # Clean cache periodically (simple implementation)
        if len(self.correlation_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(self.correlation_cache.keys())[:100]
            for k in keys_to_remove:
                del self.correlation_cache[k]
        
        return correlation_id
    
    async def _trigger_real_time_analysis(
        self,
        event_id: str,
        audit_log: SecurityAuditLog
    ):
        """Trigger real-time analysis for high-risk events"""
        try:
            # This would integrate with threat detection engine
            logger.warning(f"High-risk security event detected: {event_id}")
            
            # Update status to escalated
            with get_sync_session() as db:
                db.query(SecurityAuditLog).filter(
                    SecurityAuditLog.id == event_id
                ).update({"status": AuditEventStatus.ESCALATED.value})
                db.commit()
                
        except Exception as e:
            logger.error(f"Failed to trigger real-time analysis: {str(e)}")

# Global audit logger instance
audit_logger = SecurityAuditLogger()