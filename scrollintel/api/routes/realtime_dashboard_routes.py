"""
API Routes for Real-time Dashboard Updates and Alerts
Provides REST and WebSocket endpoints for real-time analytics dashboard
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import json
import asyncio
from datetime import datetime, timedelta
import logging

from scrollintel.core.realtime_data_processor import RealtimeDataProcessor, StreamType, StreamMessage, ProcessingRule
from scrollintel.core.intelligent_alerting_system import IntelligentAlertingSystem, ThresholdRule, AnomalyRule, AlertSeverity, Alert
from scrollintel.core.realtime_websocket_manager import RealtimeWebSocketManager, MessageType, WebSocketMessage
from scrollintel.core.notification_system import NotificationSystem, NotificationRule, NotificationChannel, NotificationPriority
from scrollintel.core.data_quality_monitor import DataQualityMonitor, DataQualityRule, DataQualityIssueType, QualityCheckSeverity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/realtime", tags=["realtime"])

# Global instances (in production, use dependency injection)
realtime_processor = None
alerting_system = None
websocket_manager = None
notification_system = None
quality_monitor = None

async def get_realtime_processor():
    """Get real-time processor instance"""
    global realtime_processor
    if realtime_processor is None:
        realtime_processor = RealtimeDataProcessor()
        await realtime_processor.initialize()
    return realtime_processor

async def get_alerting_system():
    """Get alerting system instance"""
    global alerting_system
    if alerting_system is None:
        alerting_system = IntelligentAlertingSystem()
        await alerting_system.initialize()
    return alerting_system

async def get_websocket_manager():
    """Get WebSocket manager instance"""
    global websocket_manager
    if websocket_manager is None:
        websocket_manager = RealtimeWebSocketManager()
        await websocket_manager.initialize()
    return websocket_manager

async def get_notification_system():
    """Get notification system instance"""
    global notification_system
    if notification_system is None:
        notification_system = NotificationSystem()
        await notification_system.initialize()
    return notification_system

async def get_quality_monitor():
    """Get data quality monitor instance"""
    global quality_monitor
    if quality_monitor is None:
        quality_monitor = DataQualityMonitor()
        await quality_monitor.initialize()
    return quality_monitor

# Data Processing Endpoints

@router.post("/streams/publish")
async def publish_stream_message(
    stream_type: str,
    data: Dict[str, Any],
    source: str = "api",
    priority: int = 1,
    processor: RealtimeDataProcessor = Depends(get_realtime_processor)
):
    """Publish message to real-time stream"""
    try:
        stream_enum = StreamType(stream_type)
        await processor.publish_message(stream_enum, data, source, priority)
        
        return {"status": "success", "message": "Message published successfully"}
    
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid stream type: {stream_type}")
    except Exception as e:
        logger.error(f"Error publishing message: {e}")
        raise HTTPException(status_code=500, detail="Failed to publish message")

@router.get("/streams/{stream_type}/stats")
async def get_stream_stats(
    stream_type: str,
    processor: RealtimeDataProcessor = Depends(get_realtime_processor)
):
    """Get stream processing statistics"""
    try:
        # Get stats from Redis
        redis_client = processor.redis_client
        stream_key = f"stream:{stream_type}"
        
        length = await redis_client.xlen(stream_key)
        info = await redis_client.xinfo_stream(stream_key)
        
        return {
            "stream_type": stream_type,
            "length": length,
            "first_entry": info.get("first-entry"),
            "last_entry": info.get("last-entry"),
            "groups": info.get("groups", 0)
        }
    
    except Exception as e:
        logger.error(f"Error getting stream stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stream statistics")

@router.post("/processing/rules")
async def add_processing_rule(
    rule_data: Dict[str, Any],
    processor: RealtimeDataProcessor = Depends(get_realtime_processor)
):
    """Add data processing rule"""
    try:
        rule = ProcessingRule(
            name=rule_data["name"],
            condition=rule_data["condition"],
            action=rule_data["action"],
            parameters=rule_data.get("parameters", {}),
            enabled=rule_data.get("enabled", True)
        )
        
        processor.add_processing_rule(rule)
        
        return {"status": "success", "message": "Processing rule added successfully"}
    
    except Exception as e:
        logger.error(f"Error adding processing rule: {e}")
        raise HTTPException(status_code=500, detail="Failed to add processing rule")

# Alerting Endpoints

@router.post("/alerts/threshold-rules")
async def add_threshold_rule(
    rule_data: Dict[str, Any],
    alerting: IntelligentAlertingSystem = Depends(get_alerting_system)
):
    """Add threshold monitoring rule"""
    try:
        rule = ThresholdRule(
            name=rule_data["name"],
            metric_name=rule_data["metric_name"],
            operator=rule_data["operator"],
            threshold_value=float(rule_data["threshold_value"]),
            severity=AlertSeverity(rule_data["severity"]),
            enabled=rule_data.get("enabled", True),
            cooldown_minutes=rule_data.get("cooldown_minutes", 5),
            description=rule_data.get("description", "")
        )
        
        alerting.add_threshold_rule(rule)
        
        return {"status": "success", "message": "Threshold rule added successfully"}
    
    except Exception as e:
        logger.error(f"Error adding threshold rule: {e}")
        raise HTTPException(status_code=500, detail="Failed to add threshold rule")

@router.post("/alerts/anomaly-rules")
async def add_anomaly_rule(
    rule_data: Dict[str, Any],
    alerting: IntelligentAlertingSystem = Depends(get_alerting_system)
):
    """Add anomaly detection rule"""
    try:
        rule = AnomalyRule(
            name=rule_data["name"],
            metric_name=rule_data["metric_name"],
            sensitivity=float(rule_data.get("sensitivity", 0.1)),
            min_samples=int(rule_data.get("min_samples", 100)),
            enabled=rule_data.get("enabled", True),
            description=rule_data.get("description", "")
        )
        
        alerting.add_anomaly_rule(rule)
        
        return {"status": "success", "message": "Anomaly rule added successfully"}
    
    except Exception as e:
        logger.error(f"Error adding anomaly rule: {e}")
        raise HTTPException(status_code=500, detail="Failed to add anomaly rule")

@router.get("/alerts/active")
async def get_active_alerts(
    alerting: IntelligentAlertingSystem = Depends(get_alerting_system)
):
    """Get active alerts"""
    try:
        alerts = alerting.get_active_alerts()
        
        return {
            "alerts": [
                {
                    "id": alert.id,
                    "rule_name": alert.rule_name,
                    "metric_name": alert.metric_name,
                    "severity": alert.severity.value,
                    "status": alert.status.value,
                    "title": alert.title,
                    "description": alert.description,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "created_at": alert.created_at.isoformat(),
                    "updated_at": alert.updated_at.isoformat()
                }
                for alert in alerts
            ]
        }
    
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active alerts")

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str,
    alerting: IntelligentAlertingSystem = Depends(get_alerting_system)
):
    """Acknowledge an alert"""
    try:
        await alerting.acknowledge_alert(alert_id, acknowledged_by)
        return {"status": "success", "message": "Alert acknowledged successfully"}
    
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolved_by: str,
    alerting: IntelligentAlertingSystem = Depends(get_alerting_system)
):
    """Resolve an alert"""
    try:
        await alerting.resolve_alert(alert_id, resolved_by)
        return {"status": "success", "message": "Alert resolved successfully"}
    
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")

# Dashboard Update Endpoints

@router.post("/dashboard/{dashboard_id}/update")
async def update_dashboard(
    dashboard_id: str,
    data: Dict[str, Any],
    websocket_mgr: RealtimeWebSocketManager = Depends(get_websocket_manager)
):
    """Update dashboard data and broadcast to subscribers"""
    try:
        await websocket_mgr.broadcast_dashboard_update(dashboard_id, data)
        return {"status": "success", "message": "Dashboard updated successfully"}
    
    except Exception as e:
        logger.error(f"Error updating dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to update dashboard")

@router.post("/metrics/{metric_name}/update")
async def update_metric(
    metric_name: str,
    value: float,
    metadata: Optional[Dict[str, Any]] = None,
    websocket_mgr: RealtimeWebSocketManager = Depends(get_websocket_manager)
):
    """Update metric value and broadcast to subscribers"""
    try:
        await websocket_mgr.broadcast_metric_update(metric_name, value, metadata or {})
        return {"status": "success", "message": "Metric updated successfully"}
    
    except Exception as e:
        logger.error(f"Error updating metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to update metric")

@router.get("/dashboard/connections/stats")
async def get_connection_stats(
    websocket_mgr: RealtimeWebSocketManager = Depends(get_websocket_manager)
):
    """Get WebSocket connection statistics"""
    try:
        stats = websocket_mgr.get_connection_stats()
        return stats
    
    except Exception as e:
        logger.error(f"Error getting connection stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get connection statistics")

# Notification Endpoints

@router.post("/notifications/rules")
async def add_notification_rule(
    rule_data: Dict[str, Any],
    notification_sys: NotificationSystem = Depends(get_notification_system)
):
    """Add notification rule"""
    try:
        rule = NotificationRule(
            name=rule_data["name"],
            conditions=rule_data["conditions"],
            channels=[NotificationChannel(ch) for ch in rule_data["channels"]],
            priority=NotificationPriority(rule_data["priority"]),
            template_name=rule_data["template_name"],
            recipients=rule_data["recipients"],
            enabled=rule_data.get("enabled", True),
            rate_limit=rule_data.get("rate_limit"),
            quiet_hours=rule_data.get("quiet_hours")
        )
        
        notification_sys.add_rule(rule)
        
        return {"status": "success", "message": "Notification rule added successfully"}
    
    except Exception as e:
        logger.error(f"Error adding notification rule: {e}")
        raise HTTPException(status_code=500, detail="Failed to add notification rule")

@router.post("/notifications/send")
async def send_notification(
    rule_name: str,
    data: Dict[str, Any],
    recipients: Optional[List[str]] = None,
    notification_sys: NotificationSystem = Depends(get_notification_system)
):
    """Send notification based on rule"""
    try:
        await notification_sys.send_notification(rule_name, data, recipients)
        return {"status": "success", "message": "Notification sent successfully"}
    
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail="Failed to send notification")

@router.post("/notifications/channels/{channel}/configure")
async def configure_notification_channel(
    channel: str,
    config: Dict[str, Any],
    notification_sys: NotificationSystem = Depends(get_notification_system)
):
    """Configure notification channel"""
    try:
        channel_enum = NotificationChannel(channel)
        notification_sys.configure_channel(channel_enum, config)
        
        return {"status": "success", "message": "Notification channel configured successfully"}
    
    except Exception as e:
        logger.error(f"Error configuring notification channel: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure notification channel")

# Data Quality Endpoints

@router.post("/quality/rules")
async def add_quality_rule(
    table_name: str,
    rule_data: Dict[str, Any],
    quality_mon: DataQualityMonitor = Depends(get_quality_monitor)
):
    """Add data quality rule"""
    try:
        rule = DataQualityRule(
            name=rule_data["name"],
            description=rule_data["description"],
            rule_type=DataQualityIssueType(rule_data["rule_type"]),
            severity=QualityCheckSeverity(rule_data["severity"]),
            threshold=float(rule_data["threshold"]),
            enabled=rule_data.get("enabled", True),
            metadata=rule_data.get("metadata", {})
        )
        
        quality_mon.add_quality_rule(table_name, rule)
        
        return {"status": "success", "message": "Quality rule added successfully"}
    
    except Exception as e:
        logger.error(f"Error adding quality rule: {e}")
        raise HTTPException(status_code=500, detail="Failed to add quality rule")

@router.get("/quality/metrics/{table_name}")
async def get_quality_metrics(
    table_name: str,
    quality_mon: DataQualityMonitor = Depends(get_quality_monitor)
):
    """Get data quality metrics for table"""
    try:
        metrics = await quality_mon.get_quality_metrics(table_name)
        
        if metrics:
            return {
                "source_name": metrics.source_name,
                "table_name": metrics.table_name,
                "total_records": metrics.total_records,
                "completeness_score": metrics.completeness_score,
                "accuracy_score": metrics.accuracy_score,
                "consistency_score": metrics.consistency_score,
                "validity_score": metrics.validity_score,
                "freshness_score": metrics.freshness_score,
                "overall_score": metrics.overall_score,
                "issues_count": metrics.issues_count,
                "critical_issues_count": metrics.critical_issues_count,
                "measured_at": metrics.measured_at.isoformat()
            }
        else:
            return {"message": "No quality metrics found for table"}
    
    except Exception as e:
        logger.error(f"Error getting quality metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get quality metrics")

@router.get("/quality/issues")
async def get_quality_issues(
    table_name: Optional[str] = None,
    quality_mon: DataQualityMonitor = Depends(get_quality_monitor)
):
    """Get active data quality issues"""
    try:
        issues = await quality_mon.get_active_issues(table_name)
        
        return {
            "issues": [
                {
                    "id": issue.id,
                    "rule_name": issue.rule_name,
                    "issue_type": issue.issue_type.value,
                    "severity": issue.severity.value,
                    "description": issue.description,
                    "affected_records": issue.affected_records,
                    "total_records": issue.total_records,
                    "quality_score": issue.quality_score,
                    "detected_at": issue.detected_at.isoformat(),
                    "source_table": issue.source_table,
                    "source_column": issue.source_column,
                    "metadata": issue.metadata
                }
                for issue in issues
            ]
        }
    
    except Exception as e:
        logger.error(f"Error getting quality issues: {e}")
        raise HTTPException(status_code=500, detail="Failed to get quality issues")

@router.post("/quality/issues/{issue_id}/resolve")
async def resolve_quality_issue(
    issue_id: str,
    quality_mon: DataQualityMonitor = Depends(get_quality_monitor)
):
    """Resolve data quality issue"""
    try:
        await quality_mon.resolve_issue(issue_id)
        return {"status": "success", "message": "Quality issue resolved successfully"}
    
    except Exception as e:
        logger.error(f"Error resolving quality issue: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve quality issue")

# System Control Endpoints

@router.post("/system/start")
async def start_realtime_systems(background_tasks: BackgroundTasks):
    """Start all real-time systems"""
    try:
        processor = await get_realtime_processor()
        alerting = await get_alerting_system()
        notification_sys = await get_notification_system()
        quality_mon = await get_quality_monitor()
        
        # Start systems in background
        background_tasks.add_task(processor.start_processing)
        background_tasks.add_task(alerting.start_monitoring)
        background_tasks.add_task(notification_sys.start_processing)
        background_tasks.add_task(quality_mon.start_monitoring)
        
        return {"status": "success", "message": "Real-time systems started successfully"}
    
    except Exception as e:
        logger.error(f"Error starting real-time systems: {e}")
        raise HTTPException(status_code=500, detail="Failed to start real-time systems")

@router.post("/system/stop")
async def stop_realtime_systems():
    """Stop all real-time systems"""
    try:
        if realtime_processor:
            await realtime_processor.stop_processing()
        if alerting_system:
            await alerting_system.stop_monitoring()
        if notification_system:
            await notification_system.stop_processing()
        if quality_monitor:
            await quality_monitor.stop_monitoring()
        
        return {"status": "success", "message": "Real-time systems stopped successfully"}
    
    except Exception as e:
        logger.error(f"Error stopping real-time systems: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop real-time systems")

@router.get("/system/health")
async def get_system_health():
    """Get health status of all real-time systems"""
    try:
        health_status = {
            "processor": {"status": "running" if realtime_processor and realtime_processor.running else "stopped"},
            "alerting": {"status": "running" if alerting_system and alerting_system.running else "stopped"},
            "notifications": {"status": "running" if notification_system and notification_system.running else "stopped"},
            "quality_monitor": {"status": "running" if quality_monitor and quality_monitor.running else "stopped"},
            "timestamp": datetime.now().isoformat()
        }
        
        # Add connection stats if WebSocket manager is available
        if websocket_manager:
            health_status["websocket"] = websocket_manager.get_connection_stats()
        
        return health_status
    
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system health")

# WebSocket Endpoint

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await websocket.accept()
    
    websocket_mgr = await get_websocket_manager()
    websocket_mgr.add_websocket_connection(websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            
            # Process incoming WebSocket messages
            try:
                message_data = json.loads(data)
                message_type = MessageType(message_data.get('type'))
                
                # Handle different message types
                if message_type == MessageType.HEARTBEAT:
                    await websocket.send_text(json.dumps({
                        'type': 'heartbeat',
                        'timestamp': datetime.now().isoformat()
                    }))
                
                elif message_type == MessageType.USER_ACTION:
                    action = message_data.get('data', {}).get('action')
                    
                    if action == 'subscribe_dashboard':
                        dashboard_id = message_data.get('data', {}).get('dashboard_id')
                        if dashboard_id:
                            # Add to dashboard subscribers
                            client_id = f"ws_{id(websocket)}"
                            if dashboard_id not in websocket_mgr.dashboard_subscribers:
                                websocket_mgr.dashboard_subscribers[dashboard_id] = set()
                            websocket_mgr.dashboard_subscribers[dashboard_id].add(client_id)
                            
                            await websocket.send_text(json.dumps({
                                'type': 'system_status',
                                'data': {'status': 'subscribed', 'dashboard_id': dashboard_id}
                            }))
                    
                    elif action == 'subscribe_metric':
                        metric_name = message_data.get('data', {}).get('metric_name')
                        if metric_name:
                            # Add to metric subscribers
                            client_id = f"ws_{id(websocket)}"
                            if metric_name not in websocket_mgr.metric_subscribers:
                                websocket_mgr.metric_subscribers[metric_name] = set()
                            websocket_mgr.metric_subscribers[metric_name].add(client_id)
                            
                            await websocket.send_text(json.dumps({
                                'type': 'system_status',
                                'data': {'status': 'subscribed', 'metric_name': metric_name}
                            }))
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'data': {'error': 'Invalid JSON message'}
                }))
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'data': {'error': 'Message processing failed'}
                }))
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up connection
        websocket_mgr.remove_websocket_connection(websocket)

# Utility function to initialize all systems
async def initialize_realtime_systems():
    """Initialize all real-time systems"""
    try:
        await get_realtime_processor()
        await get_alerting_system()
        await get_websocket_manager()
        await get_notification_system()
        await get_quality_monitor()
        
        logger.info("All real-time systems initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing real-time systems: {e}")
        raise