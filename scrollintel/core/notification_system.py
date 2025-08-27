"""
Notification System for Critical Insights and Anomalies
Handles multi-channel notifications with intelligent routing and delivery
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp

from ..core.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

class NotificationPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class NotificationStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRY = "retry"

class NotificationChannel(Enum):
    WEBSOCKET = "websocket"
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PUSH = "push"
    DASHBOARD = "dashboard"

@dataclass
class NotificationTemplate:
    """Notification template configuration"""
    id: str
    name: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    priority: NotificationPriority
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'channel': self.channel.value,
            'priority': self.priority.value
        }

@dataclass
class NotificationRecipient:
    """Notification recipient configuration"""
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    slack_user_id: Optional[str] = None
    webhook_url: Optional[str] = None
    push_token: Optional[str] = None
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}

@dataclass
class NotificationRule:
    """Notification routing rule"""
    id: str
    name: str
    conditions: List[Dict[str, Any]]
    recipients: List[str]  # Recipient IDs
    channels: List[NotificationChannel]
    template_id: str
    priority: NotificationPriority
    enabled: bool = True
    cooldown_minutes: int = 15
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'channels': [ch.value for ch in self.channels],
            'priority': self.priority.value
        }

@dataclass
class NotificationInstance:
    """Individual notification instance"""
    id: str
    rule_id: str
    recipient_id: str
    channel: NotificationChannel
    priority: NotificationPriority
    status: NotificationStatus
    subject: str
    body: str
    created_at: datetime
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'channel': self.channel.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'delivered_at': self.delivered_at.isoformat() if self.delivered_at else None,
            'failed_at': self.failed_at.isoformat() if self.failed_at else None
        }

class NotificationSystem:
    """
    Comprehensive notification system for critical insights and anomalies
    """
    
    def __init__(self, redis_client: redis.Redis, websocket_manager: WebSocketManager,
                 config: Dict[str, Any] = None):
        self.redis_client = redis_client
        self.websocket_manager = websocket_manager
        self.config = config or {}
        
        # Storage
        self.templates: Dict[str, NotificationTemplate] = {}
        self.recipients: Dict[str, NotificationRecipient] = {}
        self.rules: Dict[str, NotificationRule] = {}
        self.pending_notifications: Dict[str, NotificationInstance] = {}
        
        # Channel handlers
        self.channel_handlers: Dict[NotificationChannel, Callable] = {}
        
        # Delivery tracking
        self.delivery_stats = {
            'total_sent': 0,
            'total_delivered': 0,
            'total_failed': 0,
            'by_channel': {},
            'by_priority': {}
        }
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, datetime]] = {}
        self.cooldown_tracker: Dict[str, datetime] = {}
        
        # Background tasks
        self._running = False
        self._tasks = []
        
        # Initialize default handlers
        self._setup_default_handlers()
    
    async def start(self):
        """Start the notification system"""
        if self._running:
            return
            
        self._running = True
        logger.info("Starting notification system")
        
        # Load configuration from Redis
        await self._load_configuration()
        
        # Start background tasks
        processor_task = asyncio.create_task(self._process_notifications())
        self._tasks.append(processor_task)
        
        retry_task = asyncio.create_task(self._retry_failed_notifications())
        self._tasks.append(retry_task)
        
        cleanup_task = asyncio.create_task(self._cleanup_old_notifications())
        self._tasks.append(cleanup_task)
        
        stats_task = asyncio.create_task(self._update_statistics())
        self._tasks.append(stats_task)
    
    async def stop(self):
        """Stop the notification system"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info("Notification system stopped")
    
    # Configuration Management
    
    async def add_template(self, template: NotificationTemplate) -> bool:
        """Add a notification template"""
        try:
            self.templates[template.id] = template
            
            # Store in Redis
            await self.redis_client.hset(
                "notifications:templates",
                template.id,
                json.dumps(template.to_dict())
            )
            
            logger.info(f"Added notification template: {template.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding template: {str(e)}")
            return False
    
    async def add_recipient(self, recipient: NotificationRecipient) -> bool:
        """Add a notification recipient"""
        try:
            self.recipients[recipient.id] = recipient
            
            # Store in Redis
            await self.redis_client.hset(
                "notifications:recipients",
                recipient.id,
                json.dumps(asdict(recipient))
            )
            
            logger.info(f"Added notification recipient: {recipient.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding recipient: {str(e)}")
            return False
    
    async def add_rule(self, rule: NotificationRule) -> bool:
        """Add a notification rule"""
        try:
            self.rules[rule.id] = rule
            
            # Store in Redis
            await self.redis_client.hset(
                "notifications:rules",
                rule.id,
                json.dumps(rule.to_dict())
            )
            
            logger.info(f"Added notification rule: {rule.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding rule: {str(e)}")
            return False
    
    # Notification Processing
    
    async def send_notification(self, event_type: str, data: Dict[str, Any],
                              priority: NotificationPriority = NotificationPriority.MEDIUM) -> List[str]:
        """
        Send notifications based on event type and data
        
        Args:
            event_type: Type of event triggering notification
            data: Event data for template rendering
            priority: Notification priority
            
        Returns:
            List of notification IDs created
        """
        notification_ids = []
        
        try:
            # Find applicable rules
            applicable_rules = await self._find_applicable_rules(event_type, data, priority)
            
            for rule in applicable_rules:
                # Check cooldown
                if not self._check_rule_cooldown(rule.id, rule.cooldown_minutes):
                    continue
                
                # Get template
                template = self.templates.get(rule.template_id)
                if not template or not template.enabled:
                    continue
                
                # Create notifications for each recipient and channel
                for recipient_id in rule.recipients:
                    recipient = self.recipients.get(recipient_id)
                    if not recipient:
                        continue
                    
                    for channel in rule.channels:
                        # Check if recipient supports this channel
                        if not self._recipient_supports_channel(recipient, channel):
                            continue
                        
                        # Render notification content
                        subject, body = await self._render_template(template, data)
                        
                        # Create notification instance
                        notification = NotificationInstance(
                            id=f"notif_{datetime.now().timestamp()}_{rule.id}_{recipient_id}_{channel.value}",
                            rule_id=rule.id,
                            recipient_id=recipient_id,
                            channel=channel,
                            priority=priority,
                            status=NotificationStatus.PENDING,
                            subject=subject,
                            body=body,
                            created_at=datetime.now(),
                            context=data
                        )
                        
                        # Queue notification
                        self.pending_notifications[notification.id] = notification
                        
                        # Store in Redis
                        await self.redis_client.hset(
                            "notifications:pending",
                            notification.id,
                            json.dumps(notification.to_dict())
                        )
                        
                        notification_ids.append(notification.id)
                
                # Update cooldown
                self.cooldown_tracker[rule.id] = datetime.now()
            
            logger.info(f"Created {len(notification_ids)} notifications for event: {event_type}")
            return notification_ids
            
        except Exception as e:
            logger.error(f"Error sending notifications: {str(e)}")
            return []
    
    async def send_critical_alert(self, title: str, message: str, 
                                context: Dict[str, Any] = None) -> List[str]:
        """Send critical alert to all configured recipients"""
        alert_data = {
            'title': title,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }
        
        return await self.send_notification(
            'critical_alert',
            alert_data,
            NotificationPriority.CRITICAL
        )
    
    async def send_insight_notification(self, insight: Dict[str, Any]) -> List[str]:
        """Send notification for new insights"""
        return await self.send_notification(
            'new_insight',
            insight,
            NotificationPriority.MEDIUM
        )
    
    async def send_anomaly_alert(self, anomaly: Dict[str, Any]) -> List[str]:
        """Send notification for detected anomalies"""
        priority = NotificationPriority.HIGH if anomaly.get('severity') == 'high' else NotificationPriority.MEDIUM
        
        return await self.send_notification(
            'anomaly_detected',
            anomaly,
            priority
        )
    
    # Channel Handlers
    
    def register_channel_handler(self, channel: NotificationChannel, 
                                handler: Callable[[NotificationInstance], bool]):
        """Register a custom channel handler"""
        self.channel_handlers[channel] = handler
    
    async def _send_websocket_notification(self, notification: NotificationInstance) -> bool:
        """Send WebSocket notification"""
        try:
            recipient = self.recipients.get(notification.recipient_id)
            if not recipient:
                return False
            
            notification_data = {
                'type': 'notification',
                'id': notification.id,
                'priority': notification.priority.value,
                'subject': notification.subject,
                'body': notification.body,
                'timestamp': notification.created_at.isoformat(),
                'context': notification.context
            }
            
            # Send to specific user or broadcast
            if recipient.id == 'all':
                await self.websocket_manager.broadcast_to_dashboards(notification_data)
            else:
                await self.websocket_manager.send_to_user(recipient.id, notification_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending WebSocket notification: {str(e)}")
            return False
    
    async def _send_email_notification(self, notification: NotificationInstance) -> bool:
        """Send email notification"""
        try:
            recipient = self.recipients.get(notification.recipient_id)
            if not recipient or not recipient.email:
                return False
            
            # Get email configuration
            smtp_config = self.config.get('email', {})
            if not smtp_config:
                logger.warning("Email configuration not found")
                return False
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from_address')
            msg['To'] = recipient.email
            msg['Subject'] = notification.subject
            
            # Add body
            msg.attach(MIMEText(notification.body, 'html' if '<' in notification.body else 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_config.get('host'), smtp_config.get('port', 587)) as server:
                if smtp_config.get('use_tls'):
                    server.starttls()
                
                if smtp_config.get('username'):
                    server.login(smtp_config.get('username'), smtp_config.get('password'))
                
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            return False
    
    async def _send_slack_notification(self, notification: NotificationInstance) -> bool:
        """Send Slack notification"""
        try:
            recipient = self.recipients.get(notification.recipient_id)
            if not recipient or not recipient.slack_user_id:
                return False
            
            slack_config = self.config.get('slack', {})
            if not slack_config.get('webhook_url'):
                logger.warning("Slack configuration not found")
                return False
            
            # Prepare Slack message
            slack_message = {
                'text': notification.subject,
                'attachments': [{
                    'color': self._get_priority_color(notification.priority),
                    'fields': [{
                        'title': 'Message',
                        'value': notification.body,
                        'short': False
                    }],
                    'timestamp': int(notification.created_at.timestamp())
                }]
            }
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    slack_config['webhook_url'],
                    json=slack_message
                ) as response:
                    return response.status == 200
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
            return False
    
    async def _send_webhook_notification(self, notification: NotificationInstance) -> bool:
        """Send webhook notification"""
        try:
            recipient = self.recipients.get(notification.recipient_id)
            if not recipient or not recipient.webhook_url:
                return False
            
            # Prepare webhook payload
            webhook_payload = {
                'notification_id': notification.id,
                'priority': notification.priority.value,
                'subject': notification.subject,
                'body': notification.body,
                'timestamp': notification.created_at.isoformat(),
                'context': notification.context
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    recipient.webhook_url,
                    json=webhook_payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    return response.status in [200, 201, 202]
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {str(e)}")
            return False
    
    # Helper Methods
    
    async def _find_applicable_rules(self, event_type: str, data: Dict[str, Any],
                                   priority: NotificationPriority) -> List[NotificationRule]:
        """Find notification rules applicable to the event"""
        applicable_rules = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check if rule applies to this event
            if await self._evaluate_rule_conditions(rule, event_type, data, priority):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    async def _evaluate_rule_conditions(self, rule: NotificationRule, event_type: str,
                                      data: Dict[str, Any], priority: NotificationPriority) -> bool:
        """Evaluate if a rule's conditions are met"""
        try:
            for condition in rule.conditions:
                condition_type = condition.get('type')
                
                if condition_type == 'event_type':
                    if condition.get('value') != event_type:
                        return False
                
                elif condition_type == 'priority':
                    min_priority = NotificationPriority(condition.get('min_priority', 1))
                    if priority.value < min_priority.value:
                        return False
                
                elif condition_type == 'data_field':
                    field_name = condition.get('field')
                    expected_value = condition.get('value')
                    operator = condition.get('operator', '==')
                    
                    actual_value = data.get(field_name)
                    
                    if not self._evaluate_condition(actual_value, expected_value, operator):
                        return False
                
                elif condition_type == 'time_window':
                    # Check if current time is within specified window
                    if not self._check_time_window(condition):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating rule conditions: {str(e)}")
            return False
    
    def _evaluate_condition(self, actual: Any, expected: Any, operator: str) -> bool:
        """Evaluate a condition with the given operator"""
        try:
            if operator == '==':
                return actual == expected
            elif operator == '!=':
                return actual != expected
            elif operator == '>':
                return actual > expected
            elif operator == '<':
                return actual < expected
            elif operator == '>=':
                return actual >= expected
            elif operator == '<=':
                return actual <= expected
            elif operator == 'contains':
                return expected in str(actual)
            elif operator == 'not_contains':
                return expected not in str(actual)
            else:
                return False
        except:
            return False
    
    def _check_time_window(self, condition: Dict[str, Any]) -> bool:
        """Check if current time is within specified window"""
        # Implementation for time window checking
        return True
    
    def _check_rule_cooldown(self, rule_id: str, cooldown_minutes: int) -> bool:
        """Check if rule cooldown period has passed"""
        if rule_id not in self.cooldown_tracker:
            return True
        
        last_notification = self.cooldown_tracker[rule_id]
        cooldown_period = timedelta(minutes=cooldown_minutes)
        
        return datetime.now() - last_notification >= cooldown_period
    
    def _recipient_supports_channel(self, recipient: NotificationRecipient, 
                                  channel: NotificationChannel) -> bool:
        """Check if recipient supports the notification channel"""
        if channel == NotificationChannel.EMAIL:
            return bool(recipient.email)
        elif channel == NotificationChannel.SMS:
            return bool(recipient.phone)
        elif channel == NotificationChannel.SLACK:
            return bool(recipient.slack_user_id)
        elif channel == NotificationChannel.WEBHOOK:
            return bool(recipient.webhook_url)
        elif channel == NotificationChannel.PUSH:
            return bool(recipient.push_token)
        else:
            return True  # WebSocket and Dashboard are always supported
    
    async def _render_template(self, template: NotificationTemplate, 
                             data: Dict[str, Any]) -> tuple[str, str]:
        """Render notification template with data"""
        try:
            # Simple template rendering - can be enhanced with Jinja2
            subject = template.subject_template
            body = template.body_template
            
            # Replace placeholders
            for key, value in data.items():
                placeholder = f"{{{key}}}"
                subject = subject.replace(placeholder, str(value))
                body = body.replace(placeholder, str(value))
            
            return subject, body
            
        except Exception as e:
            logger.error(f"Error rendering template: {str(e)}")
            return template.subject_template, template.body_template
    
    def _get_priority_color(self, priority: NotificationPriority) -> str:
        """Get color code for notification priority"""
        colors = {
            NotificationPriority.LOW: '#36a64f',      # Green
            NotificationPriority.MEDIUM: '#ffaa00',   # Orange
            NotificationPriority.HIGH: '#ff6600',     # Red-Orange
            NotificationPriority.CRITICAL: '#ff0000', # Red
            NotificationPriority.EMERGENCY: '#8b0000' # Dark Red
        }
        return colors.get(priority, '#36a64f')
    
    def _setup_default_handlers(self):
        """Setup default notification channel handlers"""
        self.channel_handlers[NotificationChannel.WEBSOCKET] = self._send_websocket_notification
        self.channel_handlers[NotificationChannel.EMAIL] = self._send_email_notification
        self.channel_handlers[NotificationChannel.SLACK] = self._send_slack_notification
        self.channel_handlers[NotificationChannel.WEBHOOK] = self._send_webhook_notification
    
    # Background Tasks
    
    async def _process_notifications(self):
        """Process pending notifications"""
        while self._running:
            try:
                # Process notifications by priority
                notifications_to_process = sorted(
                    self.pending_notifications.values(),
                    key=lambda n: (n.priority.value, n.created_at),
                    reverse=True
                )
                
                for notification in notifications_to_process[:10]:  # Process up to 10 at a time
                    success = await self._send_notification(notification)
                    
                    if success:
                        notification.status = NotificationStatus.SENT
                        notification.sent_at = datetime.now()
                        
                        # Remove from pending
                        del self.pending_notifications[notification.id]
                        await self.redis_client.hdel("notifications:pending", notification.id)
                        
                        # Add to sent
                        await self.redis_client.hset(
                            "notifications:sent",
                            notification.id,
                            json.dumps(notification.to_dict())
                        )
                        
                        # Update statistics
                        self.delivery_stats['total_sent'] += 1
                        
                    else:
                        notification.status = NotificationStatus.FAILED
                        notification.failed_at = datetime.now()
                        notification.retry_count += 1
                        
                        # Update in Redis
                        await self.redis_client.hset(
                            "notifications:pending",
                            notification.id,
                            json.dumps(notification.to_dict())
                        )
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Error processing notifications: {str(e)}")
                await asyncio.sleep(5)
    
    async def _send_notification(self, notification: NotificationInstance) -> bool:
        """Send individual notification"""
        try:
            handler = self.channel_handlers.get(notification.channel)
            if handler:
                return await handler(notification)
            else:
                logger.warning(f"No handler for channel: {notification.channel}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending notification {notification.id}: {str(e)}")
            notification.error_message = str(e)
            return False
    
    async def _retry_failed_notifications(self):
        """Retry failed notifications"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = datetime.now()
                
                for notification in list(self.pending_notifications.values()):
                    if (notification.status == NotificationStatus.FAILED and
                        notification.retry_count < 3):
                        
                        # Wait before retry (exponential backoff)
                        retry_delay = 2 ** notification.retry_count * 60  # minutes
                        if notification.failed_at:
                            time_since_failure = (current_time - notification.failed_at).total_seconds() / 60
                            
                            if time_since_failure >= retry_delay:
                                notification.status = NotificationStatus.RETRY
                                logger.info(f"Retrying notification: {notification.id}")
                
            except Exception as e:
                logger.error(f"Error in retry task: {str(e)}")
    
    async def _cleanup_old_notifications(self):
        """Clean up old notification records"""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up sent notifications older than 7 days
                cutoff_time = datetime.now() - timedelta(days=7)
                
                sent_notifications = await self.redis_client.hgetall("notifications:sent")
                
                for notif_id, notif_data in sent_notifications.items():
                    try:
                        notif_dict = json.loads(notif_data)
                        sent_at = datetime.fromisoformat(notif_dict.get('sent_at', ''))
                        
                        if sent_at < cutoff_time:
                            await self.redis_client.hdel("notifications:sent", notif_id)
                            
                    except Exception as e:
                        logger.error(f"Error processing notification {notif_id}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
    
    async def _update_statistics(self):
        """Update notification statistics"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Update statistics
                stats = {
                    **self.delivery_stats,
                    'pending_count': len(self.pending_notifications),
                    'templates_count': len(self.templates),
                    'recipients_count': len(self.recipients),
                    'rules_count': len(self.rules),
                    'last_updated': datetime.now().isoformat()
                }
                
                # Store in Redis
                await self.redis_client.setex(
                    "notifications:statistics",
                    300,  # 5 minutes TTL
                    json.dumps(stats)
                )
                
            except Exception as e:
                logger.error(f"Error updating statistics: {str(e)}")
    
    async def _load_configuration(self):
        """Load configuration from Redis"""
        try:
            # Load templates
            templates = await self.redis_client.hgetall("notifications:templates")
            for template_id, template_data in templates.items():
                try:
                    template_dict = json.loads(template_data)
                    template_dict['channel'] = NotificationChannel(template_dict['channel'])
                    template_dict['priority'] = NotificationPriority(template_dict['priority'])
                    template = NotificationTemplate(**template_dict)
                    self.templates[template_id] = template
                except Exception as e:
                    logger.error(f"Error loading template {template_id}: {str(e)}")
            
            # Load recipients
            recipients = await self.redis_client.hgetall("notifications:recipients")
            for recipient_id, recipient_data in recipients.items():
                try:
                    recipient_dict = json.loads(recipient_data)
                    recipient = NotificationRecipient(**recipient_dict)
                    self.recipients[recipient_id] = recipient
                except Exception as e:
                    logger.error(f"Error loading recipient {recipient_id}: {str(e)}")
            
            # Load rules
            rules = await self.redis_client.hgetall("notifications:rules")
            for rule_id, rule_data in rules.items():
                try:
                    rule_dict = json.loads(rule_data)
                    rule_dict['channels'] = [NotificationChannel(ch) for ch in rule_dict['channels']]
                    rule_dict['priority'] = NotificationPriority(rule_dict['priority'])
                    rule = NotificationRule(**rule_dict)
                    self.rules[rule_id] = rule
                except Exception as e:
                    logger.error(f"Error loading rule {rule_id}: {str(e)}")
            
            logger.info(f"Loaded {len(self.templates)} templates, {len(self.recipients)} recipients, {len(self.rules)} rules")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    # Public API Methods
    
    async def get_notification_statistics(self) -> Dict[str, Any]:
        """Get notification system statistics"""
        return {
            **self.delivery_stats,
            'pending_count': len(self.pending_notifications),
            'system_status': 'running' if self._running else 'stopped'
        }