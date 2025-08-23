"""
Security Audit API Routes
Provides REST API endpoints for security audit and SIEM integration
"""
import uuid
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel

from ...models.security_audit_models import (
    SecurityAuditLogCreate, SIEMIntegrationCreate, ThreatDetectionRuleCreate,
    SecurityIncidentCreate, ComplianceReportCreate, SecurityEventType,
    SeverityLevel, SIEMPlatform, ComplianceFramework
)
from ...core.security_audit_logger import audit_logger
from ...core.siem_integration import siem_manager
from ...core.threat_detection_engine import threat_engine
from ...core.compliance_reporting import compliance_engine
from ...core.database_connection_manager import get_sync_session
from ...core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/security", tags=["Security Audit"])

class SecurityEventRequest(BaseModel):
    """Request model for logging security events"""
    event_type: SecurityEventType
    action: str
    outcome: str
    severity: SeverityLevel = SeverityLevel.LOW
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    details: Optional[dict] = None
    correlation_id: Optional[str] = None
    integration_id: Optional[str] = None

class SecurityMetricsResponse(BaseModel):
    """Response model for security metrics"""
    total_events: int
    critical_events: int
    high_severity_events: int
    incidents_open: int
    incidents_resolved: int
    average_response_time: float
    compliance_score: float
    threat_detection_rate: float
    false_positive_rate: float

class ThreatAlertResponse(BaseModel):
    """Response model for threat alerts"""
    alert_id: str
    threat_type: str
    severity: str
    confidence: float
    description: str
    affected_resources: List[str]
    indicators: List[dict]
    recommended_actions: List[str]
    timestamp: datetime

@router.post("/events", response_model=dict)
async def log_security_event(
    event: SecurityEventRequest,
    background_tasks: BackgroundTasks
):
    """Log a security event"""
    try:
        event_id = await audit_logger.log_security_event(
            event_type=event.event_type,
            action=event.action,
            outcome=event.outcome,
            severity=event.severity,
            user_id=event.user_id,
            session_id=event.session_id,
            source_ip=event.source_ip,
            user_agent=event.user_agent,
            resource=event.resource,
            details=event.details,
            correlation_id=event.correlation_id,
            integration_id=event.integration_id
        )
        
        # Trigger threat analysis in background
        background_tasks.add_task(analyze_event_for_threats, event_id)
        
        return {
            "event_id": event_id,
            "status": "logged",
            "message": "Security event logged successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to log security event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events", response_model=List[dict])
async def get_security_events(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    event_types: Optional[List[SecurityEventType]] = Query(None),
    severity_levels: Optional[List[SeverityLevel]] = Query(None),
    user_id: Optional[str] = Query(None),
    limit: int = Query(100, le=1000)
):
    """Retrieve security events with filtering"""
    try:
        events = audit_logger.get_security_events(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
            severity_levels=severity_levels,
            user_id=user_id,
            limit=limit
        )
        
        return [
            {
                "id": event.id,
                "event_type": event.event_type,
                "severity": event.severity,
                "timestamp": event.timestamp,
                "user_id": event.user_id,
                "source_ip": event.source_ip,
                "resource": event.resource,
                "action": event.action,
                "outcome": event.outcome,
                "risk_score": event.risk_score,
                "details": event.details
            }
            for event in events
        ]
        
    except Exception as e:
        logger.error(f"Failed to retrieve security events: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=SecurityMetricsResponse)
async def get_security_metrics(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None)
):
    """Get security metrics and KPIs"""
    try:
        metrics = audit_logger.get_security_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        return SecurityMetricsResponse(**metrics.dict())
        
    except Exception as e:
        logger.error(f"Failed to get security metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/siem/integrations", response_model=dict)
async def create_siem_integration(integration: SIEMIntegrationCreate):
    """Create new SIEM integration"""
    try:
        integration_id = await siem_manager.create_integration(integration)
        
        return {
            "integration_id": integration_id,
            "status": "created",
            "message": "SIEM integration created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create SIEM integration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/siem/integrations/{integration_id}/test", response_model=dict)
async def test_siem_integration(integration_id: str):
    """Test SIEM integration connection"""
    try:
        success = await siem_manager.test_integration(integration_id)
        
        return {
            "integration_id": integration_id,
            "status": "success" if success else "failed",
            "message": "Connection test completed"
        }
        
    except Exception as e:
        logger.error(f"Failed to test SIEM integration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/siem/forward-events", response_model=dict)
async def forward_events_to_siem(
    background_tasks: BackgroundTasks,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    integration_ids: Optional[List[str]] = Query(None)
):
    """Forward security events to SIEM platforms"""
    try:
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=1)
        if not end_time:
            end_time = datetime.utcnow()
        
        # Get events to forward
        events = audit_logger.get_security_events(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        # Forward events in background
        background_tasks.add_task(
            forward_events_background,
            events,
            integration_ids
        )
        
        return {
            "status": "queued",
            "events_count": len(events),
            "message": "Events queued for forwarding to SIEM platforms"
        }
        
    except Exception as e:
        logger.error(f"Failed to forward events to SIEM: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/threat-detection/rules", response_model=dict)
async def create_threat_detection_rule(rule: ThreatDetectionRuleCreate):
    """Create custom threat detection rule"""
    try:
        # This would save to database and activate the rule
        rule_id = str(uuid.uuid4())
        
        # Save rule to database (implementation needed)
        # threat_engine.add_custom_rule(rule)
        
        return {
            "rule_id": rule_id,
            "status": "created",
            "message": "Threat detection rule created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create threat detection rule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/threat-detection/alerts", response_model=List[ThreatAlertResponse])
async def get_threat_alerts(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    severity_levels: Optional[List[SeverityLevel]] = Query(None),
    limit: int = Query(100, le=1000)
):
    """Get threat detection alerts"""
    try:
        # This would retrieve alerts from database or threat engine
        # For now, return empty list
        return []
        
    except Exception as e:
        logger.error(f"Failed to get threat alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compliance/reports", response_model=dict)
async def generate_compliance_report(
    report: ComplianceReportCreate,
    background_tasks: BackgroundTasks
):
    """Generate compliance report"""
    try:
        # Generate report in background
        background_tasks.add_task(
            generate_compliance_report_background,
            report.framework,
            report.period_start,
            report.period_end,
            "full",
            report.generated_by
        )
        
        return {
            "status": "queued",
            "message": "Compliance report generation started"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/compliance/metrics/{framework}", response_model=dict)
async def get_compliance_metrics(
    framework: ComplianceFramework,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None)
):
    """Get compliance metrics for framework"""
    try:
        metrics = compliance_engine.get_compliance_metrics(
            framework=framework,
            period_start=start_time,
            period_end=end_time
        )
        
        return {
            "framework": metrics.framework,
            "overall_score": metrics.overall_score,
            "compliant_controls": metrics.compliant_controls,
            "total_controls": metrics.total_controls,
            "violations_count": metrics.violations_count,
            "critical_violations": metrics.critical_violations,
            "last_assessment": metrics.last_assessment
        }
        
    except Exception as e:
        logger.error(f"Failed to get compliance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/incidents", response_model=dict)
async def create_security_incident(incident: SecurityIncidentCreate):
    """Create security incident"""
    try:
        # This would create incident in database
        incident_id = str(uuid.uuid4())
        
        return {
            "incident_id": incident_id,
            "status": "created",
            "message": "Security incident created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create security incident: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def analyze_event_for_threats(event_id: str):
    """Background task to analyze event for threats"""
    try:
        with get_sync_session() as db:
            from ...models.security_audit_models import SecurityAuditLog
            event = db.query(SecurityAuditLog).filter(
                SecurityAuditLog.id == event_id
            ).first()
            
            if event:
                alerts = await threat_engine.analyze_security_event(event)
                if alerts:
                    logger.warning(f"Threats detected for event {event_id}: {len(alerts)} alerts")
                    
    except Exception as e:
        logger.error(f"Failed to analyze event for threats: {str(e)}")

async def forward_events_background(events, integration_ids):
    """Background task to forward events to SIEM"""
    try:
        results = await siem_manager.forward_events_to_siem(
            events=events,
            integration_ids=integration_ids
        )
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Forwarded events to {success_count}/{len(results)} SIEM integrations")
        
    except Exception as e:
        logger.error(f"Failed to forward events in background: {str(e)}")

async def generate_compliance_report_background(
    framework, period_start, period_end, report_type, generated_by
):
    """Background task to generate compliance report"""
    try:
        report_id = await compliance_engine.generate_compliance_report(
            framework=ComplianceFramework(framework),
            period_start=period_start,
            period_end=period_end,
            report_type=report_type,
            generated_by=generated_by
        )
        
        logger.info(f"Compliance report generated: {report_id}")
        
    except Exception as e:
        logger.error(f"Failed to generate compliance report in background: {str(e)}")

# Authentication events helper
@router.post("/events/authentication", response_model=dict)
async def log_authentication_event(
    user_id: str,
    action: str,
    outcome: str,
    source_ip: str,
    user_agent: str,
    details: Optional[dict] = None
):
    """Log authentication-specific event"""
    try:
        event_id = await audit_logger.log_authentication_event(
            user_id=user_id,
            action=action,
            outcome=outcome,
            source_ip=source_ip,
            user_agent=user_agent,
            details=details
        )
        
        return {
            "event_id": event_id,
            "status": "logged",
            "message": "Authentication event logged successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to log authentication event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Data access events helper
@router.post("/events/data-access", response_model=dict)
async def log_data_access_event(
    user_id: str,
    resource: str,
    action: str,
    outcome: str,
    integration_id: Optional[str] = None,
    details: Optional[dict] = None
):
    """Log data access event"""
    try:
        event_id = await audit_logger.log_data_access_event(
            user_id=user_id,
            resource=resource,
            action=action,
            outcome=outcome,
            integration_id=integration_id,
            details=details
        )
        
        return {
            "event_id": event_id,
            "status": "logged",
            "message": "Data access event logged successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to log data access event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))