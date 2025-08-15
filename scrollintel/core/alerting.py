"""
ScrollIntel Alerting System
Comprehensive alerting for system health and performance issues
"""

import asyncio
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
import aiohttp
import logging

from ..core.config import get_settings
from ..core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals"
    threshold: float
    severity: AlertSeverity
    duration: int  # seconds the condition must be true
    description: str
    enabled: bool = True
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: List[AlertRule] = []
        self.notification_channels: List['NotificationChannel'] = []
        self.alert_history: List[Alert] = []
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="High CPU Usage",
                metric_name="cpu_percent",
                condition="greater_than",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                duration=300,  # 5 minutes
                description="CPU usage is above 80% for 5 minutes"
            ),
            AlertRule(
                name="Critical CPU Usage",
                metric_name="cpu_percent",
                condition="greater_than",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                duration=60,  # 1 minute
                description="CPU usage is above 95% for 1 minute"
            ),
            AlertRule(
                name="High Memory Usage",
                metric_name="memory_percent",
                condition="greater_than",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration=300,
                description="Memory usage is above 85% for 5 minutes"
            ),
            AlertRule(
                name="Critical Memory Usage",
                metric_name="memory_percent",
                condition="greater_than",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                duration=60,
                description="Memory usage is above 95% for 1 minute"
            ),
            AlertRule(
                name="High Disk Usage",
                metric_name="disk_percent",
                condition="greater_than",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                duration=0,  # Immediate
                description="Disk usage is above 90%"
            ),
            AlertRule(
                name="High Error Rate",
                metric_name="error_rate",
                condition="greater_than",
                threshold=5.0,
                severity=AlertSeverity.WARNING,
                duration=180,
                description="Error rate is above 5% for 3 minutes"
            ),
            AlertRule(
                name="High Response Time",
                metric_name="avg_response_time",
                condition="greater_than",
                threshold=5.0,
                severity=AlertSeverity.WARNING,
                duration=300,
                description="Average response time is above 5 seconds for 5 minutes"
            ),
            AlertRule(
                name="Database Connection Issues",
                metric_name="db_connections",
                condition="greater_than",
                threshold=100,
                severity=AlertSeverity.WARNING,
                duration=60,
                description="Database connections exceed 100 for 1 minute"
            ),
            AlertRule(
                name="Agent Failure Rate",
                metric_name="agent_failure_rate",
                condition="greater_than",
                threshold=10.0,
                severity=AlertSeverity.CRITICAL,
                duration=120,
                description="Agent failure rate is above 10% for 2 minutes"
            )
        ]
        
        self.alert_rules.extend(default_rules)
        
    def add_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.name}")
        
    def remove_rule(self, rule_name: str):
        """Remove an alert rule"""
        self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
        self.logger.info(f"Removed alert rule: {rule_name}")
        
    def add_notification_channel(self, channel: 'NotificationChannel'):
        """Add a notification channel"""
        self.notification_channels.append(channel)
        self.logger.info(f"Added notification channel: {channel.__class__.__name__}")
        
    def evaluate_metrics(self, metrics: Dict[str, float]):
        """Evaluate metrics against alert rules"""
        current_time = datetime.utcnow()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            if rule.metric_name not in metrics:
                continue
                
            current_value = metrics[rule.metric_name]
            should_alert = self._evaluate_condition(current_value, rule.condition, rule.threshold)
            
            alert_id = f"{rule.name}_{rule.metric_name}"
            
            if should_alert:
                if alert_id not in self.active_alerts:
                    # Create new alert
                    alert = Alert(
                        id=alert_id,
                        name=rule.name,
                        description=rule.description,
                        severity=rule.severity,
                        status=AlertStatus.ACTIVE,
                        metric_name=rule.metric_name,
                        current_value=current_value,
                        threshold=rule.threshold,
                        timestamp=current_time,
                        tags=rule.tags.copy()
                    )
                    
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    
                    # Send notifications
                    asyncio.create_task(self._send_notifications(alert))
                    
                    self.logger.warning(
                        f"Alert triggered: {rule.name}",
                        alert_id=alert_id,
                        metric_name=rule.metric_name,
                        current_value=current_value,
                        threshold=rule.threshold,
                        severity=rule.severity.value
                    )
                else:
                    # Update existing alert
                    self.active_alerts[alert_id].current_value = current_value
                    
            else:
                # Resolve alert if it exists
                if alert_id in self.active_alerts:
                    alert = self.active_alerts[alert_id]
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = current_time
                    
                    # Send resolution notification
                    asyncio.create_task(self._send_resolution_notification(alert))
                    
                    self.logger.info(
                        f"Alert resolved: {alert.name}",
                        alert_id=alert_id,
                        duration=(current_time - alert.timestamp).total_seconds()
                    )
                    
                    del self.active_alerts[alert_id]
                    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return abs(value - threshold) < 0.001
        else:
            return False
            
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications to all channels"""
        for channel in self.notification_channels:
            try:
                await channel.send_alert(alert)
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel.__class__.__name__}: {e}")
                
    async def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notifications"""
        for channel in self.notification_channels:
            try:
                await channel.send_resolution(alert)
            except Exception as e:
                self.logger.error(f"Failed to send resolution via {channel.__class__.__name__}: {e}")
                
    def acknowledge_alert(self, alert_id: str, user_id: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = user_id
            
            self.logger.info(f"Alert acknowledged: {alert_id} by {user_id}")
            
    def suppress_alert(self, alert_id: str, duration_minutes: int):
        """Suppress an alert for a specified duration"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            
            # Schedule unsuppression
            asyncio.create_task(self._unsuppress_alert(alert_id, duration_minutes))
            
            self.logger.info(f"Alert suppressed: {alert_id} for {duration_minutes} minutes")
            
    async def _unsuppress_alert(self, alert_id: str, duration_minutes: int):
        """Unsuppress an alert after duration"""
        await asyncio.sleep(duration_minutes * 60)
        
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].status = AlertStatus.ACTIVE
            self.logger.info(f"Alert unsuppressed: {alert_id}")
            
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
        
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff]

class NotificationChannel:
    """Base class for notification channels"""
    
    async def send_alert(self, alert: Alert):
        """Send alert notification"""
        raise NotImplementedError
        
    async def send_resolution(self, alert: Alert):
        """Send alert resolution notification"""
        raise NotImplementedError

class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str, 
                 from_email: str, to_emails: List[str]):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.logger = get_logger(__name__)
        
    async def send_alert(self, alert: Alert):
        """Send alert email"""
        subject = f"[ScrollIntel Alert] {alert.severity.value.upper()}: {alert.name}"
        
        body = f"""
        Alert Details:
        - Name: {alert.name}
        - Description: {alert.description}
        - Severity: {alert.severity.value.upper()}
        - Metric: {alert.metric_name}
        - Current Value: {alert.current_value}
        - Threshold: {alert.threshold}
        - Timestamp: {alert.timestamp}
        
        Please investigate this issue immediately.
        """
        
        await self._send_email(subject, body)
        
    async def send_resolution(self, alert: Alert):
        """Send resolution email"""
        subject = f"[ScrollIntel Resolved] {alert.name}"
        
        duration = (alert.resolved_at - alert.timestamp).total_seconds() / 60
        
        body = f"""
        Alert Resolved:
        - Name: {alert.name}
        - Duration: {duration:.1f} minutes
        - Resolved At: {alert.resolved_at}
        
        The issue has been resolved.
        """
        
        await self._send_email(subject, body)
        
    async def _send_email(self, subject: str, body: str):
        """Send email using SMTP"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Alert email sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")

class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.logger = get_logger(__name__)
        
    async def send_alert(self, alert: Alert):
        """Send alert to Slack"""
        color = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning", 
            AlertSeverity.CRITICAL: "danger",
            AlertSeverity.EMERGENCY: "danger"
        }.get(alert.severity, "warning")
        
        payload = {
            "attachments": [{
                "color": color,
                "title": f"🚨 {alert.name}",
                "text": alert.description,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Metric", "value": alert.metric_name, "short": True},
                    {"title": "Current Value", "value": str(alert.current_value), "short": True},
                    {"title": "Threshold", "value": str(alert.threshold), "short": True}
                ],
                "timestamp": int(alert.timestamp.timestamp())
            }]
        }
        
        await self._send_webhook(payload)
        
    async def send_resolution(self, alert: Alert):
        """Send resolution to Slack"""
        duration = (alert.resolved_at - alert.timestamp).total_seconds() / 60
        
        payload = {
            "attachments": [{
                "color": "good",
                "title": f"✅ Alert Resolved: {alert.name}",
                "text": f"Alert was active for {duration:.1f} minutes",
                "timestamp": int(alert.resolved_at.timestamp())
            }]
        }
        
        await self._send_webhook(payload)
        
    async def _send_webhook(self, payload: Dict[str, Any]):
        """Send webhook to Slack"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("Slack notification sent successfully")
                    else:
                        self.logger.error(f"Failed to send Slack notification: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to send Slack webhook: {e}")

# Global alert manager instance
alert_manager = AlertManager()