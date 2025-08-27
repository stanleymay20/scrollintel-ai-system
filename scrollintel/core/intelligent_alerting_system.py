"""
Intelligent Alerting System for Advanced Analytics Dashboard
Provides threshold monitoring, smart notifications, and alert management
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from sqlalchemy.orm import Session

from ..models.dashboard_models import Alert, BusinessMetric, Dashboard
from ..core.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class NotificationChannel(Enum):
    WEBSOCKET = "websocket"
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"

@dataclass
class ThresholdRule:
    """Threshold monitoring rule configuration"""
    id: str
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    value: float
    severity: AlertSeverity
    description: str
    enabled: bool = True
    cooldown_minutes: int = 5
    consecutive_breaches: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'severity': self.severity.value
        }

@dataclass
class AlertRule:
    """Alert rule configuration"""
    id: str
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    severity: AlertSeverity
    notification_channels: List[NotificationChannel]
    enabled: bool = True
    cooldown_minutes: int = 15
    auto_resolve_minutes: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'severity': self.severity.value,
            'notification_channels': [ch.value for ch in self.notification_channels]
        }

@dataclass
class AlertInstance:
    """Active alert instance"""
    id: str
    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    source: str
    context: Dict[str, Any]
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'severity': self.severity.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

class IntelligentAlertingSystem:
    """
    Intelligent alerting system with threshold monitoring and smart notifications
    """
    
    def __init__(self, redis_client: redis.Redis, websocket_manager: WebSocketManager):
        self.redis_client = redis_client
        self.websocket_manager = websocket_manager
        
        # Alert storage
        self.threshold_rules: Dict[str, ThresholdRule] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, AlertInstance] = {}
        
        # Breach tracking for consecutive threshold violations
        self.breach_counters: Dict[str, Dict[str, int]] = {}
        self.last_breach_times: Dict[str, Dict[str, datetime]] = {}
        
        # Cooldown tracking
        self.cooldown_tracker: Dict[str, datetime] = {}
        
        # Notification handlers
        self.notification_handlers: Dict[NotificationChannel, Callable] = {}
        
        # Background tasks
        self._running = False
        self._tasks = []
        
        # Alert statistics
        self.stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'suppressed_alerts': 0
        }
    
    async def start(self):
        """Start the intelligent alerting system"""
        if self._running:
            return
            
        self._running = True
        logger.info("Starting intelligent alerting system")
        
        # Load existing rules and alerts
        await self._load_rules_from_redis()
        await self._load_active_alerts()
        
        # Start background tasks
        cleanup_task = asyncio.create_task(self._cleanup_resolved_alerts())
        self._tasks.append(cleanup_task)
        
        auto_resolve_task = asyncio.create_task(self._auto_resolve_alerts())
        self._tasks.append(auto_resolve_task)
        
        stats_task = asyncio.create_task(self._update_statistics())
        self._tasks.append(stats_task)
    
    async def stop(self):
        """Stop the alerting system"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info("Intelligent alerting system stopped")
    
    # Threshold Rule Management
    
    async def add_threshold_rule(self, rule: ThresholdRule) -> bool:
        """Add a new threshold monitoring rule"""
        try:
            self.threshold_rules[rule.id] = rule
            
            # Store in Redis
            await self.redis_client.hset(
                "alerting:threshold_rules",
                rule.id,
                json.dumps(rule.to_dict())
            )
            
            logger.info(f"Added threshold rule: {rule.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding threshold rule: {str(e)}")
            return False
    
    async def remove_threshold_rule(self, rule_id: str) -> bool:
        """Remove a threshold monitoring rule"""
        try:
            if rule_id in self.threshold_rules:
                del self.threshold_rules[rule_id]
                
                # Remove from Redis
                await self.redis_client.hdel("alerting:threshold_rules", rule_id)
                
                # Clean up breach tracking
                if rule_id in self.breach_counters:
                    del self.breach_counters[rule_id]
                if rule_id in self.last_breach_times:
                    del self.last_breach_times[rule_id]
                
                logger.info(f"Removed threshold rule: {rule_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing threshold rule: {str(e)}")
            return False
    
    async def update_threshold_rule(self, rule: ThresholdRule) -> bool:
        """Update an existing threshold rule"""
        try:
            if rule.id in self.threshold_rules:
                self.threshold_rules[rule.id] = rule
                
                # Update in Redis
                await self.redis_client.hset(
                    "alerting:threshold_rules",
                    rule.id,
                    json.dumps(rule.to_dict())
                )
                
                logger.info(f"Updated threshold rule: {rule.id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating threshold rule: {str(e)}")
            return False
    
    # Alert Rule Management
    
    async def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add a new alert rule"""
        try:
            self.alert_rules[rule.id] = rule
            
            # Store in Redis
            await self.redis_client.hset(
                "alerting:alert_rules",
                rule.id,
                json.dumps(rule.to_dict())
            )
            
            logger.info(f"Added alert rule: {rule.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding alert rule: {str(e)}")
            return False
    
    async def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        try:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                
                # Remove from Redis
                await self.redis_client.hdel("alerting:alert_rules", rule_id)
                
                logger.info(f"Removed alert rule: {rule_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing alert rule: {str(e)}")
            return False
    
    # Threshold Monitoring
    
    async def check_thresholds(self, metric_name: str, value: float, 
                             timestamp: datetime, context: Dict[str, Any] = None) -> List[AlertInstance]:
        """
        Check metric value against threshold rules
        
        Args:
            metric_name: Name of the metric
            value: Metric value to check
            timestamp: Timestamp of the metric
            context: Additional context data
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        try:
            # Find applicable threshold rules
            applicable_rules = [
                rule for rule in self.threshold_rules.values()
                if rule.metric_name == metric_name and rule.enabled
            ]
            
            for rule in applicable_rules:
                # Check if threshold is breached
                if self._evaluate_threshold(rule, value):
                    # Track consecutive breaches
                    breach_count = self._track_breach(rule.id, timestamp)
                    
                    # Check if we should trigger an alert
                    if (breach_count >= rule.consecutive_breaches and
                        self._check_cooldown(rule.id, rule.cooldown_minutes)):
                        
                        alert = await self._create_threshold_alert(
                            rule, value, timestamp, context or {}
                        )
                        
                        if alert:
                            triggered_alerts.append(alert)
                            
                            # Update cooldown
                            self.cooldown_tracker[rule.id] = timestamp
                else:
                    # Reset breach counter if threshold is not breached
                    self._reset_breach_counter(rule.id)
            
            return triggered_alerts
            
        except Exception as e:
            logger.error(f"Error checking thresholds: {str(e)}")
            return []
    
    async def evaluate_alert_conditions(self, conditions: List[Dict[str, Any]], 
                                      context: Dict[str, Any]) -> bool:
        """
        Evaluate complex alert conditions
        
        Args:
            conditions: List of condition dictionaries
            context: Context data for evaluation
            
        Returns:
            bool: True if conditions are met
        """
        try:
            for condition in conditions:
                condition_type = condition.get('type')
                
                if condition_type == 'metric_threshold':
                    if not self._evaluate_metric_condition(condition, context):
                        return False
                elif condition_type == 'pattern_match':
                    if not self._evaluate_pattern_condition(condition, context):
                        return False
                elif condition_type == 'time_window':
                    if not self._evaluate_time_condition(condition, context):
                        return False
                elif condition_type == 'composite':
                    if not await self._evaluate_composite_condition(condition, context):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating alert conditions: {str(e)}")
            return False
    
    # Alert Management
    
    async def create_alert(self, rule_id: str, title: str, message: str,
                          severity: AlertSeverity, source: str,
                          context: Dict[str, Any] = None) -> Optional[AlertInstance]:
        """Create a new alert instance"""
        try:
            alert_id = f"alert_{datetime.now().timestamp()}_{rule_id}"
            
            alert = AlertInstance(
                id=alert_id,
                rule_id=rule_id,
                title=title,
                message=message,
                severity=severity,
                status=AlertStatus.ACTIVE,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source=source,
                context=context or {}
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            
            # Persist to Redis
            await self.redis_client.hset(
                "alerting:active_alerts",
                alert_id,
                json.dumps(alert.to_dict())
            )
            
            # Send notifications
            await self._send_notifications(alert)
            
            # Update statistics
            self.stats['total_alerts'] += 1
            self.stats['active_alerts'] += 1
            
            logger.info(f"Created alert: {alert_id}")
            return alert
            
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            return None
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                
                if alert.status == AlertStatus.ACTIVE:
                    alert.status = AlertStatus.ACKNOWLEDGED
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = datetime.now()
                    alert.updated_at = datetime.now()
                    
                    # Update in Redis
                    await self.redis_client.hset(
                        "alerting:active_alerts",
                        alert_id,
                        json.dumps(alert.to_dict())
                    )
                    
                    # Broadcast acknowledgment
                    await self.websocket_manager.broadcast_to_dashboards({
                        'type': 'alert_acknowledged',
                        'alert_id': alert_id,
                        'acknowledged_by': acknowledged_by,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {str(e)}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = None) -> bool:
        """Resolve an active alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                
                if alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = datetime.now()
                    alert.updated_at = datetime.now()
                    
                    # Move to resolved alerts
                    await self.redis_client.hset(
                        "alerting:resolved_alerts",
                        alert_id,
                        json.dumps(alert.to_dict())
                    )
                    
                    # Remove from active alerts
                    await self.redis_client.hdel("alerting:active_alerts", alert_id)
                    del self.active_alerts[alert_id]
                    
                    # Update statistics
                    self.stats['active_alerts'] -= 1
                    self.stats['resolved_alerts'] += 1
                    
                    # Broadcast resolution
                    await self.websocket_manager.broadcast_to_dashboards({
                        'type': 'alert_resolved',
                        'alert_id': alert_id,
                        'resolved_by': resolved_by,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    logger.info(f"Alert resolved: {alert_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert: {str(e)}")
            return False
    
    async def suppress_alert(self, alert_id: str, suppressed_by: str, 
                           duration_minutes: int = 60) -> bool:
        """Suppress an alert for a specified duration"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                
                alert.status = AlertStatus.SUPPRESSED
                alert.updated_at = datetime.now()
                
                # Set suppression expiry
                suppression_key = f"alerting:suppressed:{alert_id}"
                await self.redis_client.setex(
                    suppression_key,
                    duration_minutes * 60,
                    json.dumps({
                        'suppressed_by': suppressed_by,
                        'suppressed_at': datetime.now().isoformat(),
                        'expires_at': (datetime.now() + timedelta(minutes=duration_minutes)).isoformat()
                    })
                )
                
                # Update alert in Redis
                await self.redis_client.hset(
                    "alerting:active_alerts",
                    alert_id,
                    json.dumps(alert.to_dict())
                )
                
                # Update statistics
                self.stats['suppressed_alerts'] += 1
                
                logger.info(f"Alert suppressed: {alert_id} for {duration_minutes} minutes")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error suppressing alert: {str(e)}")
            return False
    
    # Notification System
    
    def register_notification_handler(self, channel: NotificationChannel, 
                                    handler: Callable[[AlertInstance], None]):
        """Register a notification handler for a specific channel"""
        self.notification_handlers[channel] = handler
    
    async def _send_notifications(self, alert: AlertInstance):
        """Send notifications for an alert"""
        try:
            # Get alert rule to determine notification channels
            rule = self.alert_rules.get(alert.rule_id)
            if not rule:
                # Default to WebSocket for threshold alerts
                channels = [NotificationChannel.WEBSOCKET]
            else:
                channels = rule.notification_channels
            
            # Send notifications through each channel
            for channel in channels:
                handler = self.notification_handlers.get(channel)
                if handler:
                    try:
                        await handler(alert)
                    except Exception as e:
                        logger.error(f"Error sending {channel.value} notification: {str(e)}")
                else:
                    # Default WebSocket notification
                    if channel == NotificationChannel.WEBSOCKET:
                        await self._send_websocket_notification(alert)
            
        except Exception as e:
            logger.error(f"Error sending notifications: {str(e)}")
    
    async def _send_websocket_notification(self, alert: AlertInstance):
        """Send WebSocket notification"""
        try:
            notification_data = {
                'type': 'new_alert',
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            await self.websocket_manager.broadcast_to_dashboards(notification_data)
            
        except Exception as e:
            logger.error(f"Error sending WebSocket notification: {str(e)}")
    
    # Helper Methods
    
    def _evaluate_threshold(self, rule: ThresholdRule, value: float) -> bool:
        """Evaluate if a value breaches a threshold rule"""
        try:
            if rule.operator == '>':
                return value > rule.value
            elif rule.operator == '<':
                return value < rule.value
            elif rule.operator == '>=':
                return value >= rule.value
            elif rule.operator == '<=':
                return value <= rule.value
            elif rule.operator == '==':
                return value == rule.value
            elif rule.operator == '!=':
                return value != rule.value
            else:
                logger.warning(f"Unknown operator: {rule.operator}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating threshold: {str(e)}")
            return False
    
    def _track_breach(self, rule_id: str, timestamp: datetime) -> int:
        """Track consecutive threshold breaches"""
        if rule_id not in self.breach_counters:
            self.breach_counters[rule_id] = {}
        
        if rule_id not in self.last_breach_times:
            self.last_breach_times[rule_id] = {}
        
        # Check if this is a consecutive breach (within 5 minutes of last breach)
        last_breach = self.last_breach_times[rule_id].get('last')
        if last_breach and (timestamp - last_breach).total_seconds() <= 300:
            self.breach_counters[rule_id]['count'] = self.breach_counters[rule_id].get('count', 0) + 1
        else:
            self.breach_counters[rule_id]['count'] = 1
        
        self.last_breach_times[rule_id]['last'] = timestamp
        
        return self.breach_counters[rule_id]['count']
    
    def _reset_breach_counter(self, rule_id: str):
        """Reset breach counter for a rule"""
        if rule_id in self.breach_counters:
            self.breach_counters[rule_id]['count'] = 0
    
    def _check_cooldown(self, rule_id: str, cooldown_minutes: int) -> bool:
        """Check if cooldown period has passed"""
        if rule_id not in self.cooldown_tracker:
            return True
        
        last_alert = self.cooldown_tracker[rule_id]
        cooldown_period = timedelta(minutes=cooldown_minutes)
        
        return datetime.now() - last_alert >= cooldown_period
    
    async def _create_threshold_alert(self, rule: ThresholdRule, value: float,
                                    timestamp: datetime, context: Dict[str, Any]) -> Optional[AlertInstance]:
        """Create an alert for a threshold breach"""
        try:
            title = f"Threshold Alert: {rule.metric_name}"
            message = f"{rule.description}. Current value: {value}, Threshold: {rule.operator} {rule.value}"
            
            alert_context = {
                **context,
                'threshold_rule': rule.to_dict(),
                'current_value': value,
                'breach_time': timestamp.isoformat()
            }
            
            return await self.create_alert(
                rule_id=rule.id,
                title=title,
                message=message,
                severity=rule.severity,
                source="threshold_monitor",
                context=alert_context
            )
            
        except Exception as e:
            logger.error(f"Error creating threshold alert: {str(e)}")
            return None
    
    def _evaluate_metric_condition(self, condition: Dict[str, Any], 
                                 context: Dict[str, Any]) -> bool:
        """Evaluate metric-based condition"""
        # Implementation for metric conditions
        return True
    
    def _evaluate_pattern_condition(self, condition: Dict[str, Any], 
                                  context: Dict[str, Any]) -> bool:
        """Evaluate pattern-based condition"""
        # Implementation for pattern conditions
        return True
    
    def _evaluate_time_condition(self, condition: Dict[str, Any], 
                               context: Dict[str, Any]) -> bool:
        """Evaluate time-based condition"""
        # Implementation for time conditions
        return True
    
    async def _evaluate_composite_condition(self, condition: Dict[str, Any], 
                                          context: Dict[str, Any]) -> bool:
        """Evaluate composite condition"""
        # Implementation for composite conditions
        return True
    
    # Background Tasks
    
    async def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts"""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up resolved alerts older than 7 days
                cutoff_time = datetime.now() - timedelta(days=7)
                
                # Get all resolved alerts
                resolved_alerts = await self.redis_client.hgetall("alerting:resolved_alerts")
                
                for alert_id, alert_data in resolved_alerts.items():
                    try:
                        alert_dict = json.loads(alert_data)
                        resolved_at = datetime.fromisoformat(alert_dict.get('resolved_at', ''))
                        
                        if resolved_at < cutoff_time:
                            await self.redis_client.hdel("alerting:resolved_alerts", alert_id)
                            
                    except Exception as e:
                        logger.error(f"Error processing resolved alert {alert_id}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
    
    async def _auto_resolve_alerts(self):
        """Auto-resolve alerts based on rules"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = datetime.now()
                
                for alert_id, alert in list(self.active_alerts.items()):
                    rule = self.alert_rules.get(alert.rule_id)
                    
                    if rule and rule.auto_resolve_minutes:
                        alert_age = (current_time - alert.created_at).total_seconds() / 60
                        
                        if alert_age >= rule.auto_resolve_minutes:
                            await self.resolve_alert(alert_id, "auto_resolve")
                
            except Exception as e:
                logger.error(f"Error in auto-resolve task: {str(e)}")
    
    async def _update_statistics(self):
        """Update alerting statistics"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Update active alerts count
                self.stats['active_alerts'] = len(self.active_alerts)
                
                # Store statistics in Redis
                await self.redis_client.setex(
                    "alerting:statistics",
                    300,  # 5 minutes TTL
                    json.dumps(self.stats)
                )
                
            except Exception as e:
                logger.error(f"Error updating statistics: {str(e)}")
    
    async def _load_rules_from_redis(self):
        """Load rules from Redis on startup"""
        try:
            # Load threshold rules
            threshold_rules = await self.redis_client.hgetall("alerting:threshold_rules")
            for rule_id, rule_data in threshold_rules.items():
                try:
                    rule_dict = json.loads(rule_data)
                    rule_dict['severity'] = AlertSeverity(rule_dict['severity'])
                    rule = ThresholdRule(**rule_dict)
                    self.threshold_rules[rule_id] = rule
                except Exception as e:
                    logger.error(f"Error loading threshold rule {rule_id}: {str(e)}")
            
            # Load alert rules
            alert_rules = await self.redis_client.hgetall("alerting:alert_rules")
            for rule_id, rule_data in alert_rules.items():
                try:
                    rule_dict = json.loads(rule_data)
                    rule_dict['severity'] = AlertSeverity(rule_dict['severity'])
                    rule_dict['notification_channels'] = [
                        NotificationChannel(ch) for ch in rule_dict['notification_channels']
                    ]
                    rule = AlertRule(**rule_dict)
                    self.alert_rules[rule_id] = rule
                except Exception as e:
                    logger.error(f"Error loading alert rule {rule_id}: {str(e)}")
            
            logger.info(f"Loaded {len(self.threshold_rules)} threshold rules and {len(self.alert_rules)} alert rules")
            
        except Exception as e:
            logger.error(f"Error loading rules from Redis: {str(e)}")
    
    async def _load_active_alerts(self):
        """Load active alerts from Redis on startup"""
        try:
            active_alerts = await self.redis_client.hgetall("alerting:active_alerts")
            
            for alert_id, alert_data in active_alerts.items():
                try:
                    alert_dict = json.loads(alert_data)
                    
                    # Convert string dates back to datetime objects
                    alert_dict['created_at'] = datetime.fromisoformat(alert_dict['created_at'])
                    alert_dict['updated_at'] = datetime.fromisoformat(alert_dict['updated_at'])
                    
                    if alert_dict.get('acknowledged_at'):
                        alert_dict['acknowledged_at'] = datetime.fromisoformat(alert_dict['acknowledged_at'])
                    if alert_dict.get('resolved_at'):
                        alert_dict['resolved_at'] = datetime.fromisoformat(alert_dict['resolved_at'])
                    
                    # Convert enum strings back to enums
                    alert_dict['severity'] = AlertSeverity(alert_dict['severity'])
                    alert_dict['status'] = AlertStatus(alert_dict['status'])
                    
                    alert = AlertInstance(**alert_dict)
                    self.active_alerts[alert_id] = alert
                    
                except Exception as e:
                    logger.error(f"Error loading active alert {alert_id}: {str(e)}")
            
            logger.info(f"Loaded {len(self.active_alerts)} active alerts")
            
        except Exception as e:
            logger.error(f"Error loading active alerts from Redis: {str(e)}")
    
    # Public API Methods
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alerting system statistics"""
        return {
            **self.stats,
            'threshold_rules_count': len(self.threshold_rules),
            'alert_rules_count': len(self.alert_rules),
            'system_status': 'running' if self._running else 'stopped'
        }
    
    async def get_threshold_rules(self) -> List[Dict[str, Any]]:
        """Get all threshold rules"""
        return [rule.to_dict() for rule in self.threshold_rules.values()]
    
    async def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get all alert rules"""
        return [rule.to_dict() for rule in self.alert_rules.values()]