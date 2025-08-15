"""
Alerting and notification system for visual generation performance issues.
"""

import asyncio
import logging
import smtplib
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import aiohttp
import redis.asyncio as redis

from ..config import InfrastructureConfig
from ..exceptions import AlertingError


@dataclass
class NotificationChannel:
    """Configuration for a notification channel."""
    name: str
    type: str  # 'email', 'slack', 'webhook', 'sms'
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[str] = field(default_factory=lambda: ['critical', 'warning', 'info'])


@dataclass
class AlertNotification:
    """Alert notification data."""
    alert_id: str
    rule_name: str
    severity: str
    metric_path: str
    current_value: float
    threshold: float
    timestamp: float
    message: str
    additional_data: Dict[str, Any] = field(default_factory=dict)


class AlertingSystem:
    """Manages alerting and notifications for performance issues."""
    
    def __init__(self, config: InfrastructureConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Notification channels
        self.notification_channels: List[NotificationChannel] = []
        
        # Alert history and deduplication
        self.alert_history: List[AlertNotification] = []
        self.alert_deduplication: Dict[str, float] = {}
        
        # Redis for distributed alerting
        self.redis_client: Optional[redis.Redis] = None
        
        # HTTP session for webhooks
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        self._initialize_redis()
        self._setup_default_channels()
    
    def _initialize_redis(self):
        """Initialize Redis connection."""
        if hasattr(self.config, 'redis_url') and self.config.redis_url:
            try:
                self.redis_client = redis.from_url(
                    self.config.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize Redis for alerting: {e}")
    
    def _setup_default_channels(self):
        """Setup default notification channels."""
        # Email channel
        if hasattr(self.config, 'smtp_config'):
            self.notification_channels.append(
                NotificationChannel(
                    name='email',
                    type='email',
                    config=self.config.smtp_config,
                    severity_filter=['critical', 'warning']
                )
            )
        
        # Slack channel
        if hasattr(self.config, 'slack_webhook_url'):
            self.notification_channels.append(
                NotificationChannel(
                    name='slack',
                    type='slack',
                    config={'webhook_url': self.config.slack_webhook_url},
                    severity_filter=['critical', 'warning']
                )
            )
        
        # Generic webhook
        if hasattr(self.config, 'alert_webhook_url'):
            self.notification_channels.append(
                NotificationChannel(
                    name='webhook',
                    type='webhook',
                    config={'webhook_url': self.config.alert_webhook_url},
                    severity_filter=['critical']
                )
            )
    
    async def initialize(self):
        """Initialize the alerting system."""
        self.http_session = aiohttp.ClientSession()
        self.logger.info("Alerting system initialized")
    
    async def cleanup(self):
        """Clean up alerting system resources."""
        if self.http_session:
            await self.http_session.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        self.logger.info("Alerting system cleaned up")
    
    async def send_alert(self, alert_data: Dict[str, Any]):
        """Send an alert through configured notification channels."""
        try:
            # Create alert notification
            notification = AlertNotification(
                alert_id=f"{alert_data['rule_name']}_{int(alert_data['timestamp'])}",
                rule_name=alert_data['rule_name'],
                severity=alert_data['severity'],
                metric_path=alert_data['metric_path'],
                current_value=alert_data['current_value'],
                threshold=alert_data['threshold'],
                timestamp=alert_data['timestamp'],
                message=alert_data['message'],
                additional_data=alert_data.get('additional_data', {})
            )
            
            # Check for deduplication
            if await self._should_deduplicate_alert(notification):
                self.logger.debug(f"Alert deduplicated: {notification.alert_id}")
                return
            
            # Send through all applicable channels
            for channel in self.notification_channels:
                if (channel.enabled and 
                    notification.severity in channel.severity_filter):
                    await self._send_through_channel(notification, channel)
            
            # Store alert history
            self.alert_history.append(notification)
            await self._store_alert_in_redis(notification)
            
            # Update deduplication tracking
            self.alert_deduplication[notification.rule_name] = notification.timestamp
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
            raise AlertingError(f"Alert sending failed: {str(e)}")
    
    async def _should_deduplicate_alert(self, notification: AlertNotification) -> bool:
        """Check if alert should be deduplicated."""
        # Don't send same alert more than once per hour
        last_sent = self.alert_deduplication.get(notification.rule_name, 0)
        return notification.timestamp - last_sent < 3600  # 1 hour
    
    async def _send_through_channel(self, notification: AlertNotification, channel: NotificationChannel):
        """Send notification through a specific channel."""
        try:
            if channel.type == 'email':
                await self._send_email_notification(notification, channel)
            elif channel.type == 'slack':
                await self._send_slack_notification(notification, channel)
            elif channel.type == 'webhook':
                await self._send_webhook_notification(notification, channel)
            elif channel.type == 'sms':
                await self._send_sms_notification(notification, channel)
            else:
                self.logger.warning(f"Unknown notification channel type: {channel.type}")
                
        except Exception as e:
            self.logger.error(f"Failed to send notification through {channel.name}: {e}")
    
    async def _send_email_notification(self, notification: AlertNotification, channel: NotificationChannel):
        """Send email notification."""
        try:
            config = channel.config
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"[{notification.severity.upper()}] ScrollIntel Alert: {notification.rule_name}"
            
            # Create email body
            body = self._create_email_body(notification)
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(config['smtp_server'], config.get('smtp_port', 587)) as server:
                if config.get('use_tls', True):
                    server.starttls()
                
                if config.get('username') and config.get('password'):
                    server.login(config['username'], config['password'])
                
                server.send_message(msg)
            
            self.logger.info(f"Email alert sent: {notification.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            raise
    
    def _create_email_body(self, notification: AlertNotification) -> str:
        """Create HTML email body for alert notification."""
        severity_colors = {
            'critical': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8'
        }
        
        color = severity_colors.get(notification.severity, '#6c757d')
        
        return f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
                    <h2 style="margin: 0;">ScrollIntel Alert</h2>
                    <p style="margin: 5px 0 0 0; font-size: 18px;">{notification.rule_name}</p>
                </div>
                
                <div style="background-color: #f8f9fa; padding: 20px; border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 5px 5px;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Severity:</td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6; color: {color}; font-weight: bold;">{notification.severity.upper()}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Metric:</td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{notification.metric_path}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Current Value:</td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{notification.current_value}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Threshold:</td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{notification.threshold}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Time:</td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{datetime.fromtimestamp(notification.timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
                        </tr>
                    </table>
                    
                    <div style="margin-top: 20px; padding: 15px; background-color: white; border-radius: 5px; border-left: 4px solid {color};">
                        <h4 style="margin: 0 0 10px 0;">Message:</h4>
                        <p style="margin: 0;">{notification.message}</p>
                    </div>
                    
                    <div style="margin-top: 20px; text-align: center;">
                        <a href="https://scrollintel.com/monitoring/dashboard" 
                           style="background-color: {color}; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">
                            View Dashboard
                        </a>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    
    async def _send_slack_notification(self, notification: AlertNotification, channel: NotificationChannel):
        """Send Slack notification."""
        try:
            webhook_url = channel.config['webhook_url']
            
            # Create Slack message
            severity_colors = {
                'critical': 'danger',
                'warning': 'warning',
                'info': 'good'
            }
            
            color = severity_colors.get(notification.severity, 'good')
            
            payload = {
                "text": f"ScrollIntel Alert: {notification.rule_name}",
                "attachments": [
                    {
                        "color": color,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": notification.severity.upper(),
                                "short": True
                            },
                            {
                                "title": "Metric",
                                "value": notification.metric_path,
                                "short": True
                            },
                            {
                                "title": "Current Value",
                                "value": str(notification.current_value),
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": str(notification.threshold),
                                "short": True
                            },
                            {
                                "title": "Message",
                                "value": notification.message,
                                "short": False
                            }
                        ],
                        "footer": "ScrollIntel Monitoring",
                        "ts": int(notification.timestamp)
                    }
                ]
            }
            
            # Send to Slack
            async with self.http_session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    raise AlertingError(f"Slack webhook returned status {response.status}")
            
            self.logger.info(f"Slack alert sent: {notification.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            raise
    
    async def _send_webhook_notification(self, notification: AlertNotification, channel: NotificationChannel):
        """Send webhook notification."""
        try:
            webhook_url = channel.config['webhook_url']
            
            # Create webhook payload
            payload = {
                "alert_id": notification.alert_id,
                "rule_name": notification.rule_name,
                "severity": notification.severity,
                "metric_path": notification.metric_path,
                "current_value": notification.current_value,
                "threshold": notification.threshold,
                "timestamp": notification.timestamp,
                "message": notification.message,
                "additional_data": notification.additional_data
            }
            
            # Add custom headers if configured
            headers = channel.config.get('headers', {})
            headers['Content-Type'] = 'application/json'
            
            # Send webhook
            async with self.http_session.post(
                webhook_url, 
                json=payload, 
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status not in [200, 201, 202]:
                    raise AlertingError(f"Webhook returned status {response.status}")
            
            self.logger.info(f"Webhook alert sent: {notification.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            raise
    
    async def _send_sms_notification(self, notification: AlertNotification, channel: NotificationChannel):
        """Send SMS notification."""
        try:
            # This would integrate with SMS providers like Twilio, AWS SNS, etc.
            # For now, just log the SMS that would be sent
            
            sms_message = f"ScrollIntel Alert: {notification.rule_name} - {notification.severity.upper()} - {notification.message}"
            
            self.logger.info(f"SMS alert (simulated): {sms_message}")
            
        except Exception as e:
            self.logger.error(f"Failed to send SMS notification: {e}")
            raise
    
    async def _store_alert_in_redis(self, notification: AlertNotification):
        """Store alert notification in Redis."""
        if not self.redis_client:
            return
        
        try:
            alert_key = f"alert_notifications:{notification.alert_id}"
            alert_data = {
                'alert_id': notification.alert_id,
                'rule_name': notification.rule_name,
                'severity': notification.severity,
                'metric_path': notification.metric_path,
                'current_value': notification.current_value,
                'threshold': notification.threshold,
                'timestamp': notification.timestamp,
                'message': notification.message,
                'additional_data': json.dumps(notification.additional_data)
            }
            
            await self.redis_client.hset(alert_key, mapping=alert_data)
            await self.redis_client.expire(alert_key, 604800)  # 7 days
            
            # Add to sorted set for easy retrieval
            await self.redis_client.zadd(
                'alert_notifications_by_time',
                {notification.alert_id: notification.timestamp}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store alert in Redis: {e}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add a new notification channel."""
        self.notification_channels.append(channel)
        self.logger.info(f"Added notification channel: {channel.name}")
    
    def remove_notification_channel(self, channel_name: str):
        """Remove a notification channel by name."""
        self.notification_channels = [
            c for c in self.notification_channels 
            if c.name != channel_name
        ]
        self.logger.info(f"Removed notification channel: {channel_name}")
    
    def enable_channel(self, channel_name: str):
        """Enable a notification channel."""
        for channel in self.notification_channels:
            if channel.name == channel_name:
                channel.enabled = True
                self.logger.info(f"Enabled notification channel: {channel_name}")
                break
    
    def disable_channel(self, channel_name: str):
        """Disable a notification channel."""
        for channel in self.notification_channels:
            if channel.name == channel_name:
                channel.enabled = False
                self.logger.info(f"Disabled notification channel: {channel_name}")
                break
    
    async def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts from the specified number of hours."""
        if not self.redis_client:
            # Return from local history
            cutoff_time = notification.timestamp - (hours * 3600)
            return [
                {
                    'alert_id': alert.alert_id,
                    'rule_name': alert.rule_name,
                    'severity': alert.severity,
                    'metric_path': alert.metric_path,
                    'current_value': alert.current_value,
                    'threshold': alert.threshold,
                    'timestamp': alert.timestamp,
                    'message': alert.message
                }
                for alert in self.alert_history
                if alert.timestamp > cutoff_time
            ]
        
        try:
            # Get from Redis
            cutoff_time = notification.timestamp - (hours * 3600)
            alert_ids = await self.redis_client.zrangebyscore(
                'alert_notifications_by_time',
                cutoff_time,
                '+inf'
            )
            
            alerts = []
            for alert_id in alert_ids:
                alert_data = await self.redis_client.hgetall(f"alert_notifications:{alert_id}")
                if alert_data:
                    alert_data['additional_data'] = json.loads(alert_data.get('additional_data', '{}'))
                    alerts.append(alert_data)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to get recent alerts from Redis: {e}")
            return []
    
    async def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for the specified time period."""
        recent_alerts = await self.get_recent_alerts(hours)
        
        if not recent_alerts:
            return {
                'total_alerts': 0,
                'by_severity': {},
                'by_rule': {},
                'alert_rate': 0.0
            }
        
        # Count by severity
        by_severity = {}
        for alert in recent_alerts:
            severity = alert['severity']
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Count by rule
        by_rule = {}
        for alert in recent_alerts:
            rule_name = alert['rule_name']
            by_rule[rule_name] = by_rule.get(rule_name, 0) + 1
        
        # Calculate alert rate (alerts per hour)
        alert_rate = len(recent_alerts) / hours
        
        return {
            'total_alerts': len(recent_alerts),
            'by_severity': by_severity,
            'by_rule': by_rule,
            'alert_rate': alert_rate,
            'time_period_hours': hours
        }
    
    async def test_notification_channels(self) -> Dict[str, bool]:
        """Test all notification channels."""
        results = {}
        
        # Create test notification
        test_notification = AlertNotification(
            alert_id='test_alert',
            rule_name='Test Alert',
            severity='info',
            metric_path='test.metric',
            current_value=100.0,
            threshold=90.0,
            timestamp=time.time(),
            message='This is a test alert to verify notification channels are working correctly.'
        )
        
        for channel in self.notification_channels:
            if not channel.enabled:
                results[channel.name] = False
                continue
            
            try:
                await self._send_through_channel(test_notification, channel)
                results[channel.name] = True
                self.logger.info(f"Test notification sent successfully through {channel.name}")
            except Exception as e:
                results[channel.name] = False
                self.logger.error(f"Test notification failed for {channel.name}: {e}")
        
        return results