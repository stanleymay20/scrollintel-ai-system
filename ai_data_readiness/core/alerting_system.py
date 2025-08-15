"""Advanced alerting system for AI Data Readiness Platform."""

import asyncio
import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
import threading
import time
from collections import defaultdict, deque

from .platform_monitor import get_platform_monitor
from .resource_optimizer import get_resource_optimizer
from ..models.monitoring_models import Alert, AlertSeverity, HealthStatus
from .config import get_settings


class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    SMS = "sms"
    DASHBOARD = "dashboard"


class EscalationLevel(Enum):
    """Alert escalation levels."""
    L1 = "level_1"  # First responder
    L2 = "level_2"  # Team lead
    L3 = "level_3"  # Manager
    L4 = "level_4"  # Executive


@dataclass
class AlertRule:
    """Alert rule configuration."""
    id: str
    name: str
    description: str
    metric_name: str
    condition: str  # >, <, >=, <=, ==, !=
    threshold: float
    severity: AlertSeverity
    duration_minutes: int = 5  # How long condition must persist
    cooldown_minutes: int = 30  # Minimum time between alerts
    channels: List[AlertChannel] = field(default_factory=list)
    escalation_rules: Dict[EscalationLevel, int] = field(default_factory=dict)  # minutes to escalate
    tags: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'metric_name': self.metric_name,
            'condition': self.condition,
            'threshold': self.threshold,
            'severity': self.severity.value,
            'duration_minutes': self.duration_minutes,
            'cooldown_minutes': self.cooldown_minutes,
            'channels': [c.value for c in self.channels],
            'escalation_rules': {k.value: v for k, v in self.escalation_rules.items()},
            'tags': self.tags,
            'enabled': self.enabled
        }


@dataclass
class AlertContact:
    """Alert contact information."""
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    slack_user_id: Optional[str] = None
    escalation_level: EscalationLevel = EscalationLevel.L1
    channels: List[AlertChannel] = field(default_factory=list)
    schedule: Dict[str, Any] = field(default_factory=dict)  # On-call schedule
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'slack_user_id': self.slack_user_id,
            'escalation_level': self.escalation_level.value,
            'channels': [c.value for c in self.channels],
            'schedule': self.schedule,
            'enabled': self.enabled
        }


@dataclass
class AlertIncident:
    """Alert incident tracking."""
    id: str
    alert_rule_id: str
    alert: Alert
    created_at: datetime = field(default_factory=datetime.utcnow)
    escalation_level: EscalationLevel = EscalationLevel.L1
    escalated_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def is_active(self) -> bool:
        """Check if incident is still active."""
        return self.resolved_at is None
    
    @property
    def is_acknowledged(self) -> bool:
        """Check if incident has been acknowledged."""
        return self.acknowledged_at is not None
    
    @property
    def duration_minutes(self) -> float:
        """Get incident duration in minutes."""
        end_time = self.resolved_at or datetime.utcnow()
        return (end_time - self.created_at).total_seconds() / 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'alert_rule_id': self.alert_rule_id,
            'alert': self.alert.to_dict(),
            'created_at': self.created_at.isoformat(),
            'escalation_level': self.escalation_level.value,
            'escalated_at': self.escalated_at.isoformat() if self.escalated_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved_by': self.resolved_by,
            'is_active': self.is_active,
            'duration_minutes': self.duration_minutes,
            'notifications_sent': self.notifications_sent
        }


class AlertingSystem:
    """Comprehensive alerting and escalation system."""
    
    def __init__(self):
        self.config = get_settings()
        self.logger = logging.getLogger(__name__)
        self.monitor = get_platform_monitor()
        self.optimizer = get_resource_optimizer()
        
        # Alert configuration
        self.alert_rules: Dict[str, AlertRule] = {}
        self.contacts: Dict[str, AlertContact] = {}
        self.active_incidents: Dict[str, AlertIncident] = {}
        self.incident_history: deque = deque(maxlen=10000)
        
        # Alert state tracking
        self.metric_states: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Notification handlers
        self.notification_handlers: Dict[AlertChannel, Callable] = {
            AlertChannel.EMAIL: self._send_email_notification,
            AlertChannel.WEBHOOK: self._send_webhook_notification,
            AlertChannel.SLACK: self._send_slack_notification,
            AlertChannel.SMS: self._send_sms_notification,
            AlertChannel.DASHBOARD: self._send_dashboard_notification
        }
        
        # Background processing
        self.alerting_active = False
        self.alerting_thread: Optional[threading.Thread] = None
        
        # Load default alert rules
        self._load_default_alert_rules()
        self._load_default_contacts()
    
    def start_alerting(self, check_interval_seconds: int = 30):
        """Start the alerting system."""
        if self.alerting_active:
            return
        
        self.alerting_active = True
        self.alerting_thread = threading.Thread(
            target=self._alerting_loop,
            args=(check_interval_seconds,),
            daemon=True
        )
        self.alerting_thread.start()
        self.logger.info("Alerting system started")
    
    def stop_alerting(self):
        """Stop the alerting system."""
        self.alerting_active = False
        if self.alerting_thread:
            self.alerting_thread.join(timeout=5)
        self.logger.info("Alerting system stopped")
    
    def _alerting_loop(self, check_interval_seconds: int):
        """Main alerting loop."""
        while self.alerting_active:
            try:
                self._check_alert_conditions()
                self._process_escalations()
                self._cleanup_old_incidents()
                time.sleep(check_interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in alerting loop: {e}")
                time.sleep(check_interval_seconds)
    
    def _check_alert_conditions(self):
        """Check all alert rule conditions."""
        current_system = self.monitor.get_current_system_metrics()
        current_platform = self.monitor.get_current_platform_metrics()
        
        if not current_system or not current_platform:
            return
        
        # Combine all metrics
        all_metrics = {
            **current_system.to_dict(),
            **current_platform.to_dict()
        }
        
        # Check each alert rule
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                self._evaluate_alert_rule(rule, all_metrics)
            except Exception as e:
                self.logger.error(f"Error evaluating alert rule {rule_id}: {e}")
    
    def _evaluate_alert_rule(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Evaluate a single alert rule."""
        if rule.metric_name not in metrics:
            return
        
        current_value = metrics[rule.metric_name]
        timestamp = datetime.utcnow()
        
        # Store metric state
        self.metric_states[rule.id].append({
            'timestamp': timestamp,
            'value': current_value,
            'threshold': rule.threshold
        })
        
        # Check condition
        condition_met = self._check_condition(current_value, rule.condition, rule.threshold)
        
        # Check if condition has been met for required duration
        if condition_met:
            duration_met = self._check_duration_requirement(rule)
            
            if duration_met and self._check_cooldown(rule):
                self._trigger_alert(rule, current_value, metrics)
        else:
            # Condition not met, resolve any active incidents for this rule
            self._resolve_incidents_for_rule(rule.id)
    
    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if alert condition is met."""
        if condition == '>':
            return value > threshold
        elif condition == '<':
            return value < threshold
        elif condition == '>=':
            return value >= threshold
        elif condition == '<=':
            return value <= threshold
        elif condition == '==':
            return value == threshold
        elif condition == '!=':
            return value != threshold
        else:
            self.logger.warning(f"Unknown condition: {condition}")
            return False
    
    def _check_duration_requirement(self, rule: AlertRule) -> bool:
        """Check if condition has been met for required duration."""
        if rule.duration_minutes <= 0:
            return True
        
        states = list(self.metric_states[rule.id])
        if len(states) < 2:
            return False
        
        # Check if condition has been consistently met for the duration
        cutoff_time = datetime.utcnow() - timedelta(minutes=rule.duration_minutes)
        
        for state in reversed(states):
            if state['timestamp'] < cutoff_time:
                break
            
            condition_met = self._check_condition(
                state['value'], rule.condition, state['threshold']
            )
            if not condition_met:
                return False
        
        return True
    
    def _check_cooldown(self, rule: AlertRule) -> bool:
        """Check if cooldown period has passed."""
        if rule.id not in self.last_alert_times:
            return True
        
        last_alert = self.last_alert_times[rule.id]
        cooldown_end = last_alert + timedelta(minutes=rule.cooldown_minutes)
        
        return datetime.utcnow() > cooldown_end
    
    def _trigger_alert(self, rule: AlertRule, current_value: float, metrics: Dict[str, Any]):
        """Trigger an alert."""
        # Create alert
        alert = Alert(
            severity=rule.severity,
            title=f"Alert: {rule.name}",
            description=f"{rule.description}. Current value: {current_value}, Threshold: {rule.threshold}",
            metric_name=rule.metric_name,
            threshold_value=rule.threshold,
            actual_value=current_value,
            tags=rule.tags,
            metadata={'rule_id': rule.id, 'metrics': metrics}
        )
        
        # Create incident
        incident = AlertIncident(
            id=f"incident_{rule.id}_{int(time.time())}",
            alert_rule_id=rule.id,
            alert=alert
        )
        
        self.active_incidents[incident.id] = incident
        self.last_alert_times[rule.id] = datetime.utcnow()
        
        # Send notifications
        self._send_alert_notifications(incident, rule)
        
        self.logger.warning(f"Alert triggered: {rule.name} - {alert.description}")
    
    def _send_alert_notifications(self, incident: AlertIncident, rule: AlertRule):
        """Send alert notifications through configured channels."""
        for channel in rule.channels:
            try:
                handler = self.notification_handlers.get(channel)
                if handler:
                    success = handler(incident, rule)
                    
                    incident.notifications_sent.append({
                        'channel': channel.value,
                        'timestamp': datetime.utcnow().isoformat(),
                        'success': success
                    })
                else:
                    self.logger.warning(f"No handler for channel: {channel}")
            except Exception as e:
                self.logger.error(f"Error sending notification via {channel}: {e}")
    
    def _send_email_notification(self, incident: AlertIncident, rule: AlertRule) -> bool:
        """Send email notification."""
        try:
            # Get contacts for this escalation level
            contacts = self._get_contacts_for_escalation(incident.escalation_level)
            email_contacts = [c for c in contacts if c.email and AlertChannel.EMAIL in c.channels]
            
            if not email_contacts:
                return False
            
            # Create email content
            subject = f"[{incident.alert.severity.value.upper()}] {incident.alert.title}"
            
            body = f"""
            Alert Details:
            - Rule: {rule.name}
            - Severity: {incident.alert.severity.value}
            - Description: {incident.alert.description}
            - Metric: {incident.alert.metric_name}
            - Current Value: {incident.alert.actual_value}
            - Threshold: {incident.alert.threshold_value}
            - Time: {incident.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
            
            Incident ID: {incident.id}
            
            Please acknowledge this alert in the monitoring dashboard.
            """
            
            # Send to all email contacts
            for contact in email_contacts:
                self._send_email(contact.email, subject, body)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email notification: {e}")
            return False
    
    def _send_email(self, to_email: str, subject: str, body: str):
        """Send individual email."""
        # This would use actual SMTP configuration
        # For now, just log the email
        self.logger.info(f"EMAIL TO {to_email}: {subject}")
        self.logger.info(f"BODY: {body}")
    
    def _send_webhook_notification(self, incident: AlertIncident, rule: AlertRule) -> bool:
        """Send webhook notification."""
        # This would send HTTP POST to configured webhook URL
        self.logger.info(f"WEBHOOK: {incident.alert.title}")
        return True
    
    def _send_slack_notification(self, incident: AlertIncident, rule: AlertRule) -> bool:
        """Send Slack notification."""
        # This would use Slack API
        self.logger.info(f"SLACK: {incident.alert.title}")
        return True
    
    def _send_sms_notification(self, incident: AlertIncident, rule: AlertRule) -> bool:
        """Send SMS notification."""
        # This would use SMS service
        self.logger.info(f"SMS: {incident.alert.title}")
        return True
    
    def _send_dashboard_notification(self, incident: AlertIncident, rule: AlertRule) -> bool:
        """Send dashboard notification."""
        # This would update dashboard alerts
        self.logger.info(f"DASHBOARD: {incident.alert.title}")
        return True
    
    def _process_escalations(self):
        """Process alert escalations."""
        for incident in self.active_incidents.values():
            if not incident.is_active or incident.acknowledged_at:
                continue
            
            rule = self.alert_rules.get(incident.alert_rule_id)
            if not rule:
                continue
            
            # Check if escalation is needed
            for level, minutes in rule.escalation_rules.items():
                escalation_time = incident.created_at + timedelta(minutes=minutes)
                
                if (datetime.utcnow() > escalation_time and 
                    incident.escalation_level.value < level.value):
                    
                    self._escalate_incident(incident, level, rule)
                    break
    
    def _escalate_incident(self, incident: AlertIncident, new_level: EscalationLevel, rule: AlertRule):
        """Escalate an incident to higher level."""
        incident.escalation_level = new_level
        incident.escalated_at = datetime.utcnow()
        
        # Send escalation notifications
        self._send_alert_notifications(incident, rule)
        
        self.logger.warning(f"Incident {incident.id} escalated to {new_level.value}")
    
    def _resolve_incidents_for_rule(self, rule_id: str):
        """Resolve all active incidents for a rule."""
        for incident in list(self.active_incidents.values()):
            if incident.alert_rule_id == rule_id and incident.is_active:
                self._resolve_incident(incident.id, "system")
    
    def _resolve_incident(self, incident_id: str, resolved_by: str):
        """Resolve an incident."""
        if incident_id not in self.active_incidents:
            return
        
        incident = self.active_incidents[incident_id]
        incident.resolved_at = datetime.utcnow()
        incident.resolved_by = resolved_by
        incident.alert.resolve()
        
        # Move to history
        self.incident_history.append(incident)
        del self.active_incidents[incident_id]
        
        self.logger.info(f"Incident {incident_id} resolved by {resolved_by}")
    
    def _cleanup_old_incidents(self):
        """Clean up old incident data."""
        # Remove old metric states
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for rule_id in list(self.metric_states.keys()):
            states = self.metric_states[rule_id]
            while states and states[0]['timestamp'] < cutoff_time:
                states.popleft()
    
    def _get_contacts_for_escalation(self, level: EscalationLevel) -> List[AlertContact]:
        """Get contacts for escalation level."""
        return [
            contact for contact in self.contacts.values()
            if contact.escalation_level == level and contact.enabled
        ]
    
    def _load_default_alert_rules(self):
        """Load default alert rules."""
        default_rules = [
            AlertRule(
                id="high_cpu_usage",
                name="High CPU Usage",
                description="CPU usage is above threshold",
                metric_name="cpu_percent",
                condition=">",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                duration_minutes=5,
                cooldown_minutes=30,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                escalation_rules={EscalationLevel.L2: 15, EscalationLevel.L3: 60}
            ),
            AlertRule(
                id="critical_cpu_usage",
                name="Critical CPU Usage",
                description="CPU usage is critically high",
                metric_name="cpu_percent",
                condition=">",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                duration_minutes=2,
                cooldown_minutes=15,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DASHBOARD],
                escalation_rules={EscalationLevel.L2: 5, EscalationLevel.L3: 15, EscalationLevel.L4: 30}
            ),
            AlertRule(
                id="high_memory_usage",
                name="High Memory Usage",
                description="Memory usage is above threshold",
                metric_name="memory_percent",
                condition=">",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration_minutes=5,
                cooldown_minutes=30,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD]
            ),
            AlertRule(
                id="high_error_rate",
                name="High Error Rate",
                description="Processing error rate is above threshold",
                metric_name="error_rate_percent",
                condition=">",
                threshold=5.0,
                severity=AlertSeverity.CRITICAL,
                duration_minutes=3,
                cooldown_minutes=20,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DASHBOARD],
                escalation_rules={EscalationLevel.L2: 10, EscalationLevel.L3: 30}
            ),
            AlertRule(
                id="slow_processing",
                name="Slow Processing",
                description="Average processing time is above threshold",
                metric_name="avg_processing_time_seconds",
                condition=">",
                threshold=300.0,
                severity=AlertSeverity.WARNING,
                duration_minutes=10,
                cooldown_minutes=60,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD]
            ),
            AlertRule(
                id="disk_space_low",
                name="Low Disk Space",
                description="Disk usage is above threshold",
                metric_name="disk_usage_percent",
                condition=">",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                duration_minutes=5,
                cooldown_minutes=120,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DASHBOARD],
                escalation_rules={EscalationLevel.L2: 30, EscalationLevel.L3: 120}
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
    
    def _load_default_contacts(self):
        """Load default alert contacts."""
        default_contacts = [
            AlertContact(
                id="ops_team_l1",
                name="Operations Team L1",
                email="ops-l1@company.com",
                escalation_level=EscalationLevel.L1,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
            ),
            AlertContact(
                id="ops_lead_l2",
                name="Operations Lead L2",
                email="ops-lead@company.com",
                escalation_level=EscalationLevel.L2,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.SMS]
            ),
            AlertContact(
                id="engineering_manager_l3",
                name="Engineering Manager L3",
                email="eng-manager@company.com",
                escalation_level=EscalationLevel.L3,
                channels=[AlertChannel.EMAIL, AlertChannel.SMS]
            ),
            AlertContact(
                id="cto_l4",
                name="CTO L4",
                email="cto@company.com",
                escalation_level=EscalationLevel.L4,
                channels=[AlertChannel.EMAIL, AlertChannel.SMS]
            )
        ]
        
        for contact in default_contacts:
            self.contacts[contact.id] = contact
    
    # Public API methods
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.alert_rules[rule.id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")
    
    def add_contact(self, contact: AlertContact):
        """Add a new alert contact."""
        self.contacts[contact.id] = contact
        self.logger.info(f"Added alert contact: {contact.name}")
    
    def remove_contact(self, contact_id: str):
        """Remove an alert contact."""
        if contact_id in self.contacts:
            del self.contacts[contact_id]
            self.logger.info(f"Removed alert contact: {contact_id}")
    
    def acknowledge_incident(self, incident_id: str, acknowledged_by: str):
        """Acknowledge an incident."""
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            incident.acknowledged_at = datetime.utcnow()
            incident.acknowledged_by = acknowledged_by
            incident.alert.acknowledge(acknowledged_by)
            
            self.logger.info(f"Incident {incident_id} acknowledged by {acknowledged_by}")
    
    def resolve_incident(self, incident_id: str, resolved_by: str):
        """Manually resolve an incident."""
        self._resolve_incident(incident_id, resolved_by)
    
    def get_active_incidents(self) -> List[Dict[str, Any]]:
        """Get all active incidents."""
        return [incident.to_dict() for incident in self.active_incidents.values()]
    
    def get_incident_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get incident history."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            incident.to_dict() for incident in self.incident_history
            if incident.created_at > cutoff_time
        ]
    
    def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get all alert rules."""
        return [rule.to_dict() for rule in self.alert_rules.values()]
    
    def get_contacts(self) -> List[Dict[str, Any]]:
        """Get all alert contacts."""
        return [contact.to_dict() for contact in self.contacts.values()]
    
    def get_alerting_statistics(self) -> Dict[str, Any]:
        """Get alerting system statistics."""
        total_incidents = len(self.active_incidents) + len(self.incident_history)
        active_incidents = len(self.active_incidents)
        
        # Calculate MTTR (Mean Time To Resolution)
        resolved_incidents = [i for i in self.incident_history if i.resolved_at]
        mttr_minutes = 0
        if resolved_incidents:
            total_resolution_time = sum(i.duration_minutes for i in resolved_incidents)
            mttr_minutes = total_resolution_time / len(resolved_incidents)
        
        # Alert rate
        recent_incidents = [
            i for i in self.incident_history
            if i.created_at > datetime.utcnow() - timedelta(hours=24)
        ]
        alerts_per_hour = len(recent_incidents) / 24
        
        return {
            'total_incidents': total_incidents,
            'active_incidents': active_incidents,
            'resolved_incidents': len(resolved_incidents),
            'mttr_minutes': mttr_minutes,
            'alerts_per_hour': alerts_per_hour,
            'alert_rules_count': len(self.alert_rules),
            'contacts_count': len(self.contacts),
            'alerting_active': self.alerting_active
        }


# Global alerting system instance
_alerting_system = None


def get_alerting_system() -> AlertingSystem:
    """Get global alerting system instance."""
    global _alerting_system
    if _alerting_system is None:
        _alerting_system = AlertingSystem()
    return _alerting_system