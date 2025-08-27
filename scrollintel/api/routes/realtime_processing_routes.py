"""
API Routes for Real-time Data Processing and Alerts
Provides REST endpoints for managing real-time processing, alerts, and notifications
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json
import asyncio
from enum import Enum

from ...core.realtime_data_processor import (
    RealTimeDataProcessor, StreamMessage, StreamType
)
from ...core.intelligent_alerting_system import (
    IntelligentAlertingSystem, ThresholdRule, AlertRule, AlertSeverity, AlertStatus
)
from ...core.notification_system import (
    NotificationSystem, NotificationTemplate, NotificationRecipient, 
    NotificationRule, NotificationPriority, NotificationChannel
)
from ...core.data_quality_monitoring import (
    DataQualityMonitor, QualityRule, DataQualityDimension, QualityCheckType
)

router = APIRouter(prefix="/api/v1/realtime", tags=["Real-time Processing"])

# Pydantic Models for API

class StreamMessageRequest(BaseModel):
    stream_type: str = Field(..., description="Type of stream (metrics, events, alerts, insights)")
    source: str = Field(..., description="Source of the message")
    data: Dict[str, Any] = Field(..., description="Message data")
    priority: int = Field(default=1, ge=1, le=5, description="Message priority (1-5)")

class ThresholdRuleRequest(BaseModel):
    name: str = Field(..., description="Rule name")
    metric_name: str = Field(..., description="Metric name to monitor")
    operator: str = Field(..., description="Comparison operator (>, <, >=, <=, ==, !=)")
    value: float = Field(..., description="Threshold value")
    severity: str = Field(..., description="Alert severity (low, medium, high, critical)")
    description: str = Field(..., description="Rule description")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    cooldown_minutes: int = Field(default=5, ge=1, description="Cooldown period in minutes")
    consecutive_breaches: int = Field(default=1, ge=1, description="Required consecutive breaches")

class AlertRuleRequest(BaseModel):
    name: str = Field(..., description="Alert rule name")
    description: str = Field(..., description="Alert rule description")
    conditions: List[Dict[str, Any]] = Field(..., description="Alert conditions")
    severity: str = Field(..., description="Alert severity")
    notification_channels: List[str] = Field(..., description="Notification channels")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    cooldown_minutes: int = Field(default=15, ge=1, description="Cooldown period in minutes")
    auto_resolve_minutes: Optional[int] = Field(None, description="Auto-resolve time in minutes")

class NotificationTemplateRequest(BaseModel):
    name: str = Field(..., description="Template name")
    channel: str = Field(..., description="Notification channel")
    subject_template: str = Field(..., description="Subject template")
    body_template: str = Field(..., description="Body template")
    priority: str = Field(..., description="Template priority")
    enabled: bool = Field(default=True, description="Whether template is enabled")

class NotificationRecipientRequest(BaseModel):
    name: str = Field(..., description="Recipient name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    slack_user_id: Optional[str] = Field(None, description="Slack user ID")
    webhook_url: Optional[str] = Field(None, description="Webhook URL")
    push_token: Optional[str] = Field(None, description="Push notification token")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")

class QualityRuleRequest(BaseModel):
    name: str = Field(..., description="Quality rule name")
    description: str = Field(..., description="Quality rule description")
    dimension: str = Field(..., description="Quality dimension")
    check_type: str = Field(..., description="Quality check type")
    table_name: str = Field(..., description="Table name to check")
    column_name: Optional[str] = Field(None, description="Column name to check")
    rule_config: Dict[str, Any] = Field(default_factory=dict, description="Rule configuration")
    threshold_warning: float = Field(default=0.85, ge=0, le=1, description="Warning threshold")
    threshold_critical: float = Field(default=0.70, ge=0, le=1, description="Critical threshold")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    schedule_minutes: int = Field(default=60, ge=1, description="Check schedule in minutes")

# Dependency injection
async def get_data_processor() -> RealTimeDataProcessor:
    # This would be injected from the application context
    # For now, return a mock or raise an error
    raise HTTPException(status_code=503, detail="Data processor not available")

async def get_alerting_system() -> IntelligentAlertingSystem:
    # This would be injected from the application context
    raise HTTPException(status_code=503, detail="Alerting system not available")

async def get_notification_system() -> NotificationSystem:
    # This would be injected from the application context
    raise HTTPException(status_code=503, detail="Notification system not available")

async def get_quality_monitor() -> DataQualityMonitor:
    # This would be injected from the application context
    raise HTTPException(status_code=503, detail="Quality monitor not available")

# Real-time Data Processing Endpoints

@router.post("/messages/ingest")
async def ingest_message(
    message_request: StreamMessageRequest,
    processor: RealTimeDataProcessor = Depends(get_data_processor)
):
    """Ingest a real-time message into the processing pipeline"""
    try:
        # Convert string stream type to enum
        try:
            stream_type = StreamType(message_request.stream_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid stream type: {message_request.stream_type}"
            )
        
        # Create stream message
        message = StreamMessage(
            id=f"api_{datetime.now().timestamp()}",
            stream_type=stream_type,
            timestamp=datetime.now(),
            source=message_request.source,
            data=message_request.data,
            priority=message_request.priority
        )
        
        # Ingest message
        success = await processor.ingest_data(message)
        
        if success:
            return {
                "status": "success",
                "message_id": message.id,
                "timestamp": message.timestamp.isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to ingest message")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/messages/stats")
async def get_processing_stats(
    processor: RealTimeDataProcessor = Depends(get_data_processor)
):
    """Get real-time processing statistics"""
    try:
        stats = await processor.get_stream_stats()
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/messages/stream")
async def stream_messages(
    stream_type: Optional[str] = Query(None, description="Filter by stream type"),
    priority: Optional[int] = Query(None, ge=1, le=5, description="Filter by priority")
):
    """Stream real-time messages (Server-Sent Events)"""
    async def event_generator():
        try:
            # This would connect to the actual message stream
            # For now, simulate streaming
            counter = 0
            while counter < 100:  # Limit for demo
                event_data = {
                    "id": f"stream_event_{counter}",
                    "type": "message",
                    "data": {
                        "counter": counter,
                        "timestamp": datetime.now().isoformat(),
                        "stream_type": stream_type or "metrics",
                        "priority": priority or 1
                    }
                }
                
                yield f"data: {json.dumps(event_data)}\n\n"
                await asyncio.sleep(1)  # 1 second interval
                counter += 1
                
        except Exception as e:
            error_event = {
                "type": "error",
                "data": {"error": str(e)}
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

# Alerting System Endpoints

@router.post("/alerts/threshold-rules")
async def create_threshold_rule(
    rule_request: ThresholdRuleRequest,
    alerting_system: IntelligentAlertingSystem = Depends(get_alerting_system)
):
    """Create a new threshold monitoring rule"""
    try:
        # Convert severity string to enum
        try:
            severity = AlertSeverity(rule_request.severity.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid severity: {rule_request.severity}"
            )
        
        # Create threshold rule
        rule = ThresholdRule(
            id=f"threshold_{datetime.now().timestamp()}",
            metric_name=rule_request.metric_name,
            operator=rule_request.operator,
            value=rule_request.value,
            severity=severity,
            description=rule_request.description,
            enabled=rule_request.enabled,
            cooldown_minutes=rule_request.cooldown_minutes,
            consecutive_breaches=rule_request.consecutive_breaches
        )
        
        success = await alerting_system.add_threshold_rule(rule)
        
        if success:
            return {
                "status": "success",
                "rule_id": rule.id,
                "message": "Threshold rule created successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create threshold rule")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/threshold-rules")
async def get_threshold_rules(
    alerting_system: IntelligentAlertingSystem = Depends(get_alerting_system)
):
    """Get all threshold monitoring rules"""
    try:
        rules = await alerting_system.get_threshold_rules()
        return {
            "status": "success",
            "data": rules,
            "count": len(rules)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/alerts/threshold-rules/{rule_id}")
async def delete_threshold_rule(
    rule_id: str,
    alerting_system: IntelligentAlertingSystem = Depends(get_alerting_system)
):
    """Delete a threshold monitoring rule"""
    try:
        success = await alerting_system.remove_threshold_rule(rule_id)
        
        if success:
            return {
                "status": "success",
                "message": "Threshold rule deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Threshold rule not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/rules")
async def create_alert_rule(
    rule_request: AlertRuleRequest,
    alerting_system: IntelligentAlertingSystem = Depends(get_alerting_system)
):
    """Create a new alert rule"""
    try:
        # Convert severity and channels
        try:
            severity = AlertSeverity(rule_request.severity.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid severity: {rule_request.severity}"
            )
        
        try:
            channels = [NotificationChannel(ch.lower()) for ch in rule_request.notification_channels]
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid notification channel: {str(e)}"
            )
        
        # Create alert rule
        rule = AlertRule(
            id=f"alert_rule_{datetime.now().timestamp()}",
            name=rule_request.name,
            description=rule_request.description,
            conditions=rule_request.conditions,
            severity=severity,
            notification_channels=channels,
            enabled=rule_request.enabled,
            cooldown_minutes=rule_request.cooldown_minutes,
            auto_resolve_minutes=rule_request.auto_resolve_minutes
        )
        
        success = await alerting_system.add_alert_rule(rule)
        
        if success:
            return {
                "status": "success",
                "rule_id": rule.id,
                "message": "Alert rule created successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create alert rule")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/active")
async def get_active_alerts(
    alerting_system: IntelligentAlertingSystem = Depends(get_alerting_system)
):
    """Get all active alerts"""
    try:
        alerts = await alerting_system.get_active_alerts()
        return {
            "status": "success",
            "data": alerts,
            "count": len(alerts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str = Query(..., description="User acknowledging the alert"),
    alerting_system: IntelligentAlertingSystem = Depends(get_alerting_system)
):
    """Acknowledge an active alert"""
    try:
        success = await alerting_system.acknowledge_alert(alert_id, acknowledged_by)
        
        if success:
            return {
                "status": "success",
                "message": "Alert acknowledged successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found or already acknowledged")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolved_by: Optional[str] = Query(None, description="User resolving the alert"),
    alerting_system: IntelligentAlertingSystem = Depends(get_alerting_system)
):
    """Resolve an active alert"""
    try:
        success = await alerting_system.resolve_alert(alert_id, resolved_by)
        
        if success:
            return {
                "status": "success",
                "message": "Alert resolved successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found or already resolved")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/statistics")
async def get_alert_statistics(
    alerting_system: IntelligentAlertingSystem = Depends(get_alerting_system)
):
    """Get alerting system statistics"""
    try:
        stats = await alerting_system.get_alert_statistics()
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Notification System Endpoints

@router.post("/notifications/templates")
async def create_notification_template(
    template_request: NotificationTemplateRequest,
    notification_system: NotificationSystem = Depends(get_notification_system)
):
    """Create a new notification template"""
    try:
        # Convert enums
        try:
            channel = NotificationChannel(template_request.channel.lower())
            priority = NotificationPriority(template_request.priority.lower())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
        
        # Create template
        template = NotificationTemplate(
            id=f"template_{datetime.now().timestamp()}",
            name=template_request.name,
            channel=channel,
            subject_template=template_request.subject_template,
            body_template=template_request.body_template,
            priority=priority,
            enabled=template_request.enabled
        )
        
        success = await notification_system.add_template(template)
        
        if success:
            return {
                "status": "success",
                "template_id": template.id,
                "message": "Notification template created successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create notification template")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/notifications/recipients")
async def create_notification_recipient(
    recipient_request: NotificationRecipientRequest,
    notification_system: NotificationSystem = Depends(get_notification_system)
):
    """Create a new notification recipient"""
    try:
        # Create recipient
        recipient = NotificationRecipient(
            id=f"recipient_{datetime.now().timestamp()}",
            name=recipient_request.name,
            email=recipient_request.email,
            phone=recipient_request.phone,
            slack_user_id=recipient_request.slack_user_id,
            webhook_url=recipient_request.webhook_url,
            push_token=recipient_request.push_token,
            preferences=recipient_request.preferences
        )
        
        success = await notification_system.add_recipient(recipient)
        
        if success:
            return {
                "status": "success",
                "recipient_id": recipient.id,
                "message": "Notification recipient created successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create notification recipient")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/notifications/send")
async def send_notification(
    event_type: str = Query(..., description="Event type"),
    data: Dict[str, Any] = {},
    priority: str = Query("medium", description="Notification priority"),
    notification_system: NotificationSystem = Depends(get_notification_system)
):
    """Send a notification"""
    try:
        # Convert priority
        try:
            notification_priority = NotificationPriority(priority.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")
        
        # Send notification
        notification_ids = await notification_system.send_notification(
            event_type=event_type,
            data=data,
            priority=notification_priority
        )
        
        return {
            "status": "success",
            "notification_ids": notification_ids,
            "count": len(notification_ids),
            "message": "Notifications sent successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/notifications/statistics")
async def get_notification_statistics(
    notification_system: NotificationSystem = Depends(get_notification_system)
):
    """Get notification system statistics"""
    try:
        stats = await notification_system.get_notification_statistics()
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Data Quality Monitoring Endpoints

@router.post("/quality/rules")
async def create_quality_rule(
    rule_request: QualityRuleRequest,
    quality_monitor: DataQualityMonitor = Depends(get_quality_monitor)
):
    """Create a new data quality rule"""
    try:
        # Convert enums
        try:
            dimension = DataQualityDimension(rule_request.dimension.lower())
            check_type = QualityCheckType(rule_request.check_type.lower())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
        
        # Create quality rule
        rule = QualityRule(
            id=f"quality_rule_{datetime.now().timestamp()}",
            name=rule_request.name,
            description=rule_request.description,
            dimension=dimension,
            check_type=check_type,
            table_name=rule_request.table_name,
            column_name=rule_request.column_name,
            rule_config=rule_request.rule_config,
            threshold_warning=rule_request.threshold_warning,
            threshold_critical=rule_request.threshold_critical,
            enabled=rule_request.enabled,
            schedule_minutes=rule_request.schedule_minutes
        )
        
        success = await quality_monitor.add_quality_rule(rule)
        
        if success:
            return {
                "status": "success",
                "rule_id": rule.id,
                "message": "Data quality rule created successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create data quality rule")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quality/rules/{rule_id}/run")
async def run_quality_check(
    rule_id: str,
    background_tasks: BackgroundTasks,
    quality_monitor: DataQualityMonitor = Depends(get_quality_monitor)
):
    """Run a specific quality check"""
    try:
        # Run quality check in background
        background_tasks.add_task(quality_monitor.run_quality_check, rule_id)
        
        return {
            "status": "success",
            "message": "Quality check started",
            "rule_id": rule_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality/summary")
async def get_quality_summary(
    quality_monitor: DataQualityMonitor = Depends(get_quality_monitor)
):
    """Get overall data quality summary"""
    try:
        summary = await quality_monitor.get_quality_summary()
        return {
            "status": "success",
            "data": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality/metrics")
async def get_quality_metrics(
    rule_id: Optional[str] = Query(None, description="Filter by rule ID"),
    hours: int = Query(24, ge=1, le=168, description="Time range in hours"),
    quality_monitor: DataQualityMonitor = Depends(get_quality_monitor)
):
    """Get quality metrics"""
    try:
        metrics = await quality_monitor.get_quality_metrics(rule_id, hours)
        return {
            "status": "success",
            "data": metrics,
            "count": len(metrics),
            "filters": {
                "rule_id": rule_id,
                "hours": hours
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality/issues")
async def get_quality_issues(
    resolved: bool = Query(False, description="Include resolved issues"),
    quality_monitor: DataQualityMonitor = Depends(get_quality_monitor)
):
    """Get data quality issues"""
    try:
        issues = await quality_monitor.get_quality_issues(resolved)
        return {
            "status": "success",
            "data": issues,
            "count": len(issues),
            "filters": {
                "resolved": resolved
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quality/issues/{issue_id}/resolve")
async def resolve_quality_issue(
    issue_id: str,
    resolved_by: Optional[str] = Query(None, description="User resolving the issue"),
    quality_monitor: DataQualityMonitor = Depends(get_quality_monitor)
):
    """Resolve a data quality issue"""
    try:
        success = await quality_monitor.resolve_quality_issue(issue_id, resolved_by)
        
        if success:
            return {
                "status": "success",
                "message": "Quality issue resolved successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Quality issue not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health and Status Endpoints

@router.get("/health")
async def health_check():
    """Health check endpoint for real-time processing system"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "data_processor": "running",
            "alerting_system": "running",
            "notification_system": "running",
            "quality_monitor": "running"
        }
    }

@router.get("/status")
async def get_system_status(
    processor: RealTimeDataProcessor = Depends(get_data_processor),
    alerting_system: IntelligentAlertingSystem = Depends(get_alerting_system),
    notification_system: NotificationSystem = Depends(get_notification_system),
    quality_monitor: DataQualityMonitor = Depends(get_quality_monitor)
):
    """Get comprehensive system status"""
    try:
        # Gather status from all components
        processor_stats = await processor.get_stream_stats()
        alert_stats = await alerting_system.get_alert_statistics()
        notification_stats = await notification_system.get_notification_statistics()
        quality_summary = await quality_monitor.get_quality_summary()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "data_processor": {
                    "status": "running" if processor_stats.get('running') else "stopped",
                    "stats": processor_stats
                },
                "alerting_system": {
                    "status": "running" if alert_stats.get('system_status') == 'running' else "stopped",
                    "stats": alert_stats
                },
                "notification_system": {
                    "status": "running" if notification_stats.get('system_status') == 'running' else "stopped",
                    "stats": notification_stats
                },
                "quality_monitor": {
                    "status": "running",
                    "summary": quality_summary
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))