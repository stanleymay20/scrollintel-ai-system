"""
Error monitoring and alerting system for ScrollIntel.
Provides real-time error tracking, rate monitoring, and automated alerting.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .error_handling import ErrorCategory, ErrorSeverity, ErrorContext


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"


@dataclass
class AlertRule:
    """Configuration for error alerting rules."""
    name: str
    condition: str  # e.g., "error_rate > 0.05"
    threshold: float
    window_minutes: int = 5
    alert_level: AlertLevel = AlertLevel.WARNING
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.EMAIL])
    cooldown_minutes: int = 15
    enabled: bool = True
    components: Optional[List[str]] = None  # None means all components
    categories: Optional[List[ErrorCategory]] = None  # None means all categories


@dataclass
class Alert:
    """Alert instance."""
    id: str
    rule_name: str
    level: AlertLevel
    title: str
    message: str
    component: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None


class ErrorMetrics:
    """Tracks error metrics for monitoring."""
    
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.window_seconds = window_minutes * 60
        
        # Error tracking
        self.error_timestamps: Dict[str, deque] = defaultdict(lambda: deque())
        self.error_counts: Dict[str, Dict[ErrorCategory, int]] = defaultdict(lambda: defaultdict(int))
        self.error_severities: Dict[str, Dict[ErrorSeverity, int]] = defaultdict(lambda: defaultdict(int))
        
        # Performance tracking
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque())
        self.success_counts: Dict[str, int] = defaultdict(int)
        self.total_requests: Dict[str, int] = defaultdict(int)
        
        # System health
        self.component_health: Dict[str, str] = {}  # healthy, degraded, unhealthy
        self.last_health_check: Dict[str, float] = {}
    
    def record_error(self, context: ErrorContext):
        """Record an error occurrence."""
        current_time = time.time()
        component = context.component
        
        # Add to error timestamps
        self.error_timestamps[component].append(current_time)
        
        # Update error counts
        self.error_counts[component][context.category] += 1
        self.error_severities[component][context.severity] += 1
        
        # Clean old entries
        self._clean_old_entries(component, current_time)
        
        # Update component health
        self._update_component_health(component)
    
    def record_success(self, component: str, response_time: float):
        """Record a successful operation."""
        current_time = time.time()
        
        # Record response time
        self.response_times[component].append((current_time, response_time))
        
        # Update success count
        self.success_counts[component] += 1
        self.total_requests[component] += 1
        
        # Clean old entries
        self._clean_old_response_times(component, current_time)
        
        # Update component health
        self._update_component_health(component)
    
    def get_error_rate(self, component: str) -> float:
        """Get current error rate for component (errors per minute)."""
        current_time = time.time()
        self._clean_old_entries(component, current_time)
        
        error_count = len(self.error_timestamps[component])
        return error_count / self.window_minutes
    
    def get_success_rate(self, component: str) -> float:
        """Get success rate for component (0.0 to 1.0)."""
        total = self.total_requests[component]
        if total == 0:
            return 1.0
        
        errors = len(self.error_timestamps[component])
        successes = total - errors
        return successes / total
    
    def get_average_response_time(self, component: str) -> float:
        """Get average response time for component."""
        current_time = time.time()
        self._clean_old_response_times(component, current_time)
        
        response_times = [rt for _, rt in self.response_times[component]]
        if not response_times:
            return 0.0
        
        return sum(response_times) / len(response_times)
    
    def get_component_health(self, component: str) -> str:
        """Get component health status."""
        return self.component_health.get(component, "unknown")
    
    def get_error_breakdown(self, component: str) -> Dict[str, Any]:
        """Get detailed error breakdown for component."""
        current_time = time.time()
        self._clean_old_entries(component, current_time)
        
        total_errors = len(self.error_timestamps[component])
        
        return {
            "total_errors": total_errors,
            "error_rate": self.get_error_rate(component),
            "success_rate": self.get_success_rate(component),
            "avg_response_time": self.get_average_response_time(component),
            "health_status": self.get_component_health(component),
            "errors_by_category": dict(self.error_counts[component]),
            "errors_by_severity": dict(self.error_severities[component]),
            "window_minutes": self.window_minutes
        }
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide error overview."""
        components = set(self.error_timestamps.keys()) | set(self.response_times.keys())
        
        overview = {
            "total_components": len(components),
            "healthy_components": 0,
            "degraded_components": 0,
            "unhealthy_components": 0,
            "total_errors": 0,
            "total_requests": sum(self.total_requests.values()),
            "overall_success_rate": 0.0,
            "components": {}
        }
        
        total_successes = 0
        total_requests = 0
        
        for component in components:
            breakdown = self.get_error_breakdown(component)
            overview["components"][component] = breakdown
            overview["total_errors"] += breakdown["total_errors"]
            
            # Count component health
            health = breakdown["health_status"]
            if health == "healthy":
                overview["healthy_components"] += 1
            elif health == "degraded":
                overview["degraded_components"] += 1
            elif health == "unhealthy":
                overview["unhealthy_components"] += 1
            
            # Calculate overall success rate
            component_requests = self.total_requests[component]
            component_successes = int(component_requests * breakdown["success_rate"])
            total_successes += component_successes
            total_requests += component_requests
        
        if total_requests > 0:
            overview["overall_success_rate"] = total_successes / total_requests
        
        return overview
    
    def _clean_old_entries(self, component: str, current_time: float):
        """Remove entries outside the time window."""
        cutoff_time = current_time - self.window_seconds
        
        # Clean error timestamps
        while (self.error_timestamps[component] and 
               self.error_timestamps[component][0] < cutoff_time):
            self.error_timestamps[component].popleft()
    
    def _clean_old_response_times(self, component: str, current_time: float):
        """Remove old response time entries."""
        cutoff_time = current_time - self.window_seconds
        
        while (self.response_times[component] and 
               self.response_times[component][0][0] < cutoff_time):
            self.response_times[component].popleft()
    
    def _update_component_health(self, component: str):
        """Update component health based on current metrics."""
        error_rate = self.get_error_rate(component)
        success_rate = self.get_success_rate(component)
        
        if error_rate > 10:  # More than 10 errors per minute
            health = "unhealthy"
        elif error_rate > 5 or success_rate < 0.95:  # More than 5 errors/min or <95% success
            health = "degraded"
        else:
            health = "healthy"
        
        self.component_health[component] = health
        self.last_health_check[component] = time.time()


class AlertManager:
    """Manages error alerts and notifications."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.cooldown_tracker: Dict[str, float] = {}
        self.notification_handlers: Dict[AlertChannel, Callable] = {}
        
        # Initialize default rules
        self._initialize_default_rules()
        
        # Initialize notification handlers
        self._initialize_notification_handlers()
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
    
    def check_alerts(self, metrics: ErrorMetrics):
        """Check all alert rules against current metrics."""
        current_time = time.time()
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if self._is_in_cooldown(rule_name, current_time):
                continue
            
            # Check rule condition
            if self._evaluate_rule(rule, metrics):
                self._trigger_alert(rule, metrics, current_time)
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Resolved alert: {alert.title}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]
    
    def _initialize_default_rules(self):
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                condition="error_rate > 10",
                threshold=10.0,
                window_minutes=5,
                alert_level=AlertLevel.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD]
            ),
            AlertRule(
                name="medium_error_rate",
                condition="error_rate > 5",
                threshold=5.0,
                window_minutes=10,
                alert_level=AlertLevel.WARNING,
                channels=[AlertChannel.DASHBOARD]
            ),
            AlertRule(
                name="low_success_rate",
                condition="success_rate < 0.9",
                threshold=0.9,
                window_minutes=15,
                alert_level=AlertLevel.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD]
            ),
            AlertRule(
                name="slow_response_time",
                condition="avg_response_time > 5.0",
                threshold=5.0,
                window_minutes=10,
                alert_level=AlertLevel.WARNING,
                channels=[AlertChannel.DASHBOARD]
            ),
            AlertRule(
                name="component_unhealthy",
                condition="health_status == 'unhealthy'",
                threshold=0,
                window_minutes=5,
                alert_level=AlertLevel.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD]
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def _initialize_notification_handlers(self):
        """Initialize notification handlers for different channels."""
        self.notification_handlers = {
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.DASHBOARD: self._send_dashboard_alert,
            AlertChannel.WEBHOOK: self._send_webhook_alert,
            AlertChannel.SLACK: self._send_slack_alert
        }
    
    def _is_in_cooldown(self, rule_name: str, current_time: float) -> bool:
        """Check if rule is in cooldown period."""
        if rule_name not in self.cooldown_tracker:
            return False
        
        rule = self.rules[rule_name]
        last_alert_time = self.cooldown_tracker[rule_name]
        cooldown_seconds = rule.cooldown_minutes * 60
        
        return (current_time - last_alert_time) < cooldown_seconds
    
    def _evaluate_rule(self, rule: AlertRule, metrics: ErrorMetrics) -> bool:
        """Evaluate if rule condition is met."""
        # Get components to check
        components = rule.components or list(metrics.component_health.keys())
        if not components:
            return False
        
        # Check condition for each component
        for component in components:
            if self._evaluate_component_rule(rule, component, metrics):
                return True
        
        return False
    
    def _evaluate_component_rule(self, rule: AlertRule, component: str, metrics: ErrorMetrics) -> bool:
        """Evaluate rule for a specific component."""
        breakdown = metrics.get_error_breakdown(component)
        
        # Parse condition
        if "error_rate >" in rule.condition:
            return breakdown["error_rate"] > rule.threshold
        elif "success_rate <" in rule.condition:
            return breakdown["success_rate"] < rule.threshold
        elif "avg_response_time >" in rule.condition:
            return breakdown["avg_response_time"] > rule.threshold
        elif "health_status ==" in rule.condition:
            expected_status = rule.condition.split("==")[1].strip().strip("'\"")
            return breakdown["health_status"] == expected_status
        
        return False
    
    def _trigger_alert(self, rule: AlertRule, metrics: ErrorMetrics, current_time: float):
        """Trigger an alert."""
        import uuid
        
        alert_id = str(uuid.uuid4())
        
        # Find the component that triggered the alert
        components = rule.components or list(metrics.component_health.keys())
        triggered_component = None
        
        for component in components:
            if self._evaluate_component_rule(rule, component, metrics):
                triggered_component = component
                break
        
        if not triggered_component:
            return
        
        # Get component metrics
        breakdown = metrics.get_error_breakdown(triggered_component)
        
        # Create alert
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            level=rule.alert_level,
            title=self._generate_alert_title(rule, triggered_component, breakdown),
            message=self._generate_alert_message(rule, triggered_component, breakdown),
            component=triggered_component,
            timestamp=current_time,
            metadata={
                "rule": rule.name,
                "threshold": rule.threshold,
                "actual_value": self._get_actual_value(rule, breakdown),
                "component_metrics": breakdown
            }
        )
        
        # Add to active alerts
        self.active_alerts[alert_id] = alert
        
        # Update cooldown
        self.cooldown_tracker[rule.name] = current_time
        
        # Send notifications
        self._send_alert_notifications(alert, rule)
        
        self.logger.warning(f"Alert triggered: {alert.title}")
    
    def _generate_alert_title(self, rule: AlertRule, component: str, breakdown: Dict[str, Any]) -> str:
        """Generate alert title."""
        if "error_rate" in rule.condition:
            return f"High Error Rate: {component} ({breakdown['error_rate']:.1f} errors/min)"
        elif "success_rate" in rule.condition:
            return f"Low Success Rate: {component} ({breakdown['success_rate']:.1%})"
        elif "response_time" in rule.condition:
            return f"Slow Response Time: {component} ({breakdown['avg_response_time']:.2f}s)"
        elif "health_status" in rule.condition:
            return f"Component Unhealthy: {component} ({breakdown['health_status']})"
        else:
            return f"Alert: {rule.name} - {component}"
    
    def _generate_alert_message(self, rule: AlertRule, component: str, breakdown: Dict[str, Any]) -> str:
        """Generate detailed alert message."""
        message = f"Component: {component}\n"
        message += f"Alert Rule: {rule.name}\n"
        message += f"Threshold: {rule.threshold}\n\n"
        
        message += "Current Metrics:\n"
        message += f"- Error Rate: {breakdown['error_rate']:.1f} errors/min\n"
        message += f"- Success Rate: {breakdown['success_rate']:.1%}\n"
        message += f"- Avg Response Time: {breakdown['avg_response_time']:.2f}s\n"
        message += f"- Health Status: {breakdown['health_status']}\n"
        message += f"- Total Errors: {breakdown['total_errors']}\n\n"
        
        if breakdown['errors_by_category']:
            message += "Errors by Category:\n"
            for category, count in breakdown['errors_by_category'].items():
                message += f"- {category.value}: {count}\n"
        
        return message
    
    def _get_actual_value(self, rule: AlertRule, breakdown: Dict[str, Any]) -> float:
        """Get the actual value that triggered the alert."""
        if "error_rate" in rule.condition:
            return breakdown['error_rate']
        elif "success_rate" in rule.condition:
            return breakdown['success_rate']
        elif "response_time" in rule.condition:
            return breakdown['avg_response_time']
        else:
            return 0.0
    
    def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send alert notifications through configured channels."""
        for channel in rule.channels:
            handler = self.notification_handlers.get(channel)
            if handler:
                try:
                    asyncio.create_task(handler(alert, rule))
                except Exception as e:
                    self.logger.error(f"Failed to send alert via {channel.value}: {e}")
    
    async def _send_email_alert(self, alert: Alert, rule: AlertRule):
        """Send email alert notification."""
        # This would integrate with actual email service
        self.logger.info(f"Email alert sent: {alert.title}")
    
    async def _send_dashboard_alert(self, alert: Alert, rule: AlertRule):
        """Send dashboard alert notification."""
        # This would update the dashboard alert system
        self.logger.info(f"Dashboard alert sent: {alert.title}")
    
    async def _send_webhook_alert(self, alert: Alert, rule: AlertRule):
        """Send webhook alert notification."""
        # This would send HTTP webhook
        self.logger.info(f"Webhook alert sent: {alert.title}")
    
    async def _send_slack_alert(self, alert: Alert, rule: AlertRule):
        """Send Slack alert notification."""
        # This would integrate with Slack API
        self.logger.info(f"Slack alert sent: {alert.title}")


class ErrorMonitor:
    """Main error monitoring system."""
    
    def __init__(self, metrics_window_minutes: int = 60):
        self.logger = logging.getLogger(__name__)
        self.metrics = ErrorMetrics(metrics_window_minutes)
        self.alert_manager = AlertManager()
        self.monitoring_enabled = True
        self.check_interval = 60  # Check alerts every minute
        self._monitoring_task = None
    
    async def start_monitoring(self):
        """Start the monitoring system."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Error monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            self.logger.info("Error monitoring stopped")
    
    def record_error(self, context: ErrorContext):
        """Record an error for monitoring."""
        if self.monitoring_enabled:
            self.metrics.record_error(context)
    
    def record_success(self, component: str, response_time: float):
        """Record a successful operation."""
        if self.monitoring_enabled:
            self.metrics.record_success(component, response_time)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return self.metrics.get_system_overview()
    
    def get_component_metrics(self, component: str) -> Dict[str, Any]:
        """Get metrics for a specific component."""
        return self.metrics.get_error_breakdown(component)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return self.alert_manager.get_active_alerts()
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        self.alert_manager.add_rule(rule)
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        self.alert_manager.resolve_alert(alert_id)
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                # Check alert conditions
                self.alert_manager.check_alerts(self.metrics)
                
                # Auto-resolve alerts if conditions are no longer met
                await self._auto_resolve_alerts()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _auto_resolve_alerts(self):
        """Auto-resolve alerts when conditions are no longer met."""
        current_time = time.time()
        alerts_to_resolve = []
        
        for alert_id, alert in self.alert_manager.active_alerts.items():
            # Check if alert condition is still met
            rule = self.alert_manager.rules.get(alert.rule_name)
            if rule and not self.alert_manager._evaluate_component_rule(
                rule, alert.component, self.metrics
            ):
                # Condition no longer met, resolve after grace period
                if current_time - alert.timestamp > 300:  # 5 minute grace period
                    alerts_to_resolve.append(alert_id)
        
        for alert_id in alerts_to_resolve:
            self.alert_manager.resolve_alert(alert_id)


# Global error monitor instance
error_monitor = ErrorMonitor()


# Convenience functions
async def start_error_monitoring():
    """Start global error monitoring."""
    await error_monitor.start_monitoring()


async def stop_error_monitoring():
    """Stop global error monitoring."""
    await error_monitor.stop_monitoring()


def record_error_for_monitoring(context: ErrorContext):
    """Record error for monitoring."""
    error_monitor.record_error(context)


def record_success_for_monitoring(component: str, response_time: float):
    """Record success for monitoring."""
    error_monitor.record_success(component, response_time)


def get_system_health() -> Dict[str, Any]:
    """Get current system health metrics."""
    return error_monitor.get_metrics()


def get_component_health(component: str) -> Dict[str, Any]:
    """Get health metrics for a specific component."""
    return error_monitor.get_component_metrics(component)