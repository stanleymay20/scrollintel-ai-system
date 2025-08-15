"""
Alert management and notification system for drift monitoring.

This module implements automated drift alerting with configurable thresholds,
alert management workflows, and integration with external notification systems.
"""

import asyncio
import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import requests
from abc import ABC, abstractmethod

from ..models.drift_models import (
    DriftAlert, AlertSeverity, DriftReport, DriftThresholds
)


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"


class EscalationLevel(Enum):
    """Alert escalation levels."""
    LEVEL_1 = "level_1"  # Team lead
    LEVEL_2 = "level_2"  # Manager
    LEVEL_3 = "level_3"  # Director
    LEVEL_4 = "level_4"  # Executive


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    retry_attempts: int = 3
    retry_delay: int = 60  # seconds
    timeout: int = 30  # seconds


@dataclass
class EscalationRule:
    """Rules for alert escalation."""
    severity: AlertSeverity
    escalation_delay: int  # minutes
    escalation_levels: List[EscalationLevel]
    notification_channels: List[NotificationChannel]
    auto_acknowledge: bool = False
    max_escalations: int = 3


@dataclass
class AlertRule:
    """Rules for alert generation and handling."""
    name: str
    description: str
    conditions: Dict[str, Any]
    severity: AlertSeverity
    notification_channels: List[NotificationChannel]
    escalation_rule: Optional[EscalationRule] = None
    cooldown_period: int = 300  # seconds
    enabled: bool = True


@dataclass
class NotificationTemplate:
    """Template for notifications."""
    channel: NotificationChannel
    subject_template: str
    body_template: str
    format_type: str = "text"  # text, html, markdown


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""
    
    @abstractmethod
    async def send_notification(self, alert: DriftAlert, config: NotificationConfig,
                              template: NotificationTemplate) -> bool:
        """Send notification through this provider."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate provider configuration."""
        pass


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider."""
    
    async def send_notification(self, alert: DriftAlert, config: NotificationConfig,
                              template: NotificationTemplate) -> bool:
        """Send email notification."""
        try:
            smtp_config = config.config
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = ', '.join(smtp_config['to_emails'])
            msg['Subject'] = self._format_template(template.subject_template, alert)
            
            body = self._format_template(template.body_template, alert)
            msg.attach(MIMEText(body, template.format_type))
            
            # Send email
            with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
                if smtp_config.get('use_tls', True):
                    server.starttls()
                if smtp_config.get('username') and smtp_config.get('password'):
                    server.login(smtp_config['username'], smtp_config['password'])
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email notification: {str(e)}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate email configuration."""
        required_fields = ['smtp_server', 'smtp_port', 'from_email', 'to_emails']
        return all(field in config for field in required_fields)
    
    def _format_template(self, template: str, alert: DriftAlert) -> str:
        """Format template with alert data."""
        return template.format(
            alert_id=alert.id,
            dataset_id=alert.dataset_id,
            severity=alert.severity.value,
            message=alert.message,
            drift_score=alert.drift_score,
            threshold=alert.threshold,
            detected_at=alert.detected_at.strftime('%Y-%m-%d %H:%M:%S'),
            affected_features=', '.join(alert.affected_features)
        )


class SlackNotificationProvider(NotificationProvider):
    """Slack notification provider."""
    
    async def send_notification(self, alert: DriftAlert, config: NotificationConfig,
                              template: NotificationTemplate) -> bool:
        """Send Slack notification."""
        try:
            webhook_url = config.config['webhook_url']
            
            # Create Slack message
            message = {
                "text": self._format_template(template.subject_template, alert),
                "attachments": [
                    {
                        "color": self._get_color_for_severity(alert.severity),
                        "fields": [
                            {"title": "Dataset", "value": alert.dataset_id, "short": True},
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Drift Score", "value": f"{alert.drift_score:.3f}", "short": True},
                            {"title": "Threshold", "value": f"{alert.threshold:.3f}", "short": True},
                            {"title": "Affected Features", "value": ', '.join(alert.affected_features), "short": False},
                            {"title": "Message", "value": alert.message, "short": False}
                        ],
                        "footer": "AI Data Readiness Platform",
                        "ts": int(alert.detected_at.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(
                webhook_url,
                json=message,
                timeout=config.timeout
            )
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send Slack notification: {str(e)}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Slack configuration."""
        return 'webhook_url' in config
    
    def _format_template(self, template: str, alert: DriftAlert) -> str:
        """Format template with alert data."""
        return template.format(
            alert_id=alert.id,
            dataset_id=alert.dataset_id,
            severity=alert.severity.value,
            message=alert.message,
            drift_score=alert.drift_score,
            threshold=alert.threshold,
            detected_at=alert.detected_at.strftime('%Y-%m-%d %H:%M:%S'),
            affected_features=', '.join(alert.affected_features)
        )
    
    def _get_color_for_severity(self, severity: AlertSeverity) -> str:
        """Get color code for severity level."""
        color_map = {
            AlertSeverity.LOW: "good",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.HIGH: "danger",
            AlertSeverity.CRITICAL: "#ff0000"
        }
        return color_map.get(severity, "good")


class WebhookNotificationProvider(NotificationProvider):
    """Generic webhook notification provider."""
    
    async def send_notification(self, alert: DriftAlert, config: NotificationConfig,
                              template: NotificationTemplate) -> bool:
        """Send webhook notification."""
        try:
            webhook_url = config.config['webhook_url']
            
            # Create payload
            payload = {
                "alert_id": alert.id,
                "dataset_id": alert.dataset_id,
                "severity": alert.severity.value,
                "message": alert.message,
                "drift_score": alert.drift_score,
                "threshold": alert.threshold,
                "detected_at": alert.detected_at.isoformat(),
                "affected_features": alert.affected_features,
                "drift_type": alert.drift_type.value
            }
            
            # Add custom fields if specified
            if 'custom_fields' in config.config:
                payload.update(config.config['custom_fields'])
            
            # Send webhook
            headers = config.config.get('headers', {'Content-Type': 'application/json'})
            response = requests.post(
                webhook_url,
                json=payload,
                headers=headers,
                timeout=config.timeout
            )
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send webhook notification: {str(e)}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate webhook configuration."""
        return 'webhook_url' in config


class AlertManager:
    """
    Comprehensive alert management and notification system.
    
    Features:
    - Configurable alert rules and thresholds
    - Multiple notification channels
    - Alert escalation workflows
    - Alert acknowledgment and resolution
    - Alert history and analytics
    """
    
    def __init__(self):
        """Initialize alert manager."""
        self.logger = logging.getLogger(__name__)
        self.notification_providers = {
            NotificationChannel.EMAIL: EmailNotificationProvider(),
            NotificationChannel.SLACK: SlackNotificationProvider(),
            NotificationChannel.WEBHOOK: WebhookNotificationProvider()
        }
        self.notification_configs: Dict[NotificationChannel, NotificationConfig] = {}
        self.alert_rules: List[AlertRule] = []
        self.escalation_rules: Dict[AlertSeverity, EscalationRule] = {}
        self.notification_templates: Dict[NotificationChannel, NotificationTemplate] = {}
        self.active_alerts: Dict[str, DriftAlert] = {}
        self.alert_history: List[DriftAlert] = []
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Initialize default templates
        self._initialize_default_templates()
    
    def configure_notification_channel(self, channel: NotificationChannel,
                                     config: NotificationConfig) -> None:
        """Configure a notification channel."""
        try:
            # Validate configuration
            provider = self.notification_providers.get(channel)
            if not provider:
                raise ValueError(f"Unsupported notification channel: {channel}")
            
            if not provider.validate_config(config.config):
                raise ValueError(f"Invalid configuration for {channel}")
            
            self.notification_configs[channel] = config
            self.logger.info(f"Configured notification channel: {channel.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to configure notification channel {channel}: {str(e)}")
            raise
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def add_escalation_rule(self, severity: AlertSeverity, rule: EscalationRule) -> None:
        """Add an escalation rule for a severity level."""
        self.escalation_rules[severity] = rule
        self.logger.info(f"Added escalation rule for {severity.value}")
    
    def set_notification_template(self, channel: NotificationChannel,
                                template: NotificationTemplate) -> None:
        """Set notification template for a channel."""
        self.notification_templates[channel] = template
        self.logger.info(f"Set notification template for {channel.value}")
    
    async def process_drift_report(self, report: DriftReport) -> List[DriftAlert]:
        """Process drift report and generate alerts."""
        try:
            generated_alerts = []
            
            # Process existing alerts from report
            for alert in report.alerts:
                if self._should_send_alert(alert):
                    await self._send_alert_notifications(alert)
                    self._track_alert(alert)
                    generated_alerts.append(alert)
            
            # Check for custom alert rules
            custom_alerts = self._evaluate_custom_rules(report)
            for alert in custom_alerts:
                if self._should_send_alert(alert):
                    await self._send_alert_notifications(alert)
                    self._track_alert(alert)
                    generated_alerts.append(alert)
            
            return generated_alerts
            
        except Exception as e:
            self.logger.error(f"Error processing drift report: {str(e)}")
            raise
    
    async def send_alert(self, alert: DriftAlert) -> bool:
        """Send alert through configured notification channels."""
        try:
            if not self._should_send_alert(alert):
                return False
            
            success = await self._send_alert_notifications(alert)
            if success:
                self._track_alert(alert)
                
                # Start escalation if configured
                escalation_rule = self.escalation_rules.get(alert.severity)
                if escalation_rule:
                    asyncio.create_task(self._handle_escalation(alert, escalation_rule))
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending alert {alert.id}: {str(e)}")
            return False
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow()
                
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error acknowledging alert {alert_id}: {str(e)}")
            return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts.pop(alert_id)
                alert.acknowledged = True
                alert.acknowledged_by = resolved_by
                alert.acknowledged_at = datetime.utcnow()
                
                # Move to history
                self.alert_history.append(alert)
                
                self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error resolving alert {alert_id}: {str(e)}")
            return False
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[DriftAlert]:
        """Get active alerts with optional severity filtering."""
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        return sorted(alerts, key=lambda x: x.detected_at, reverse=True)
    
    def get_alert_history(self, hours: int = 24) -> List[DriftAlert]:
        """Get alert history for specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.detected_at >= cutoff_time
        ]
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alert_history + list(self.active_alerts.values())
            if alert.detected_at >= cutoff_time
        ]
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                alert for alert in recent_alerts if alert.severity == severity
            ])
        
        return {
            "total_alerts": len(recent_alerts),
            "active_alerts": len(self.active_alerts),
            "resolved_alerts": len([a for a in self.alert_history if a.detected_at >= cutoff_time]),
            "severity_breakdown": severity_counts,
            "acknowledgment_rate": len([a for a in recent_alerts if a.acknowledged]) / max(len(recent_alerts), 1),
            "time_period_hours": hours
        }
    
    async def _send_alert_notifications(self, alert: DriftAlert) -> bool:
        """Send alert through all configured notification channels."""
        success_count = 0
        total_channels = 0
        
        # Determine which channels to use
        channels_to_use = self._get_channels_for_alert(alert)
        
        for channel in channels_to_use:
            if channel not in self.notification_configs:
                continue
            
            config = self.notification_configs[channel]
            if not config.enabled:
                continue
            
            provider = self.notification_providers.get(channel)
            template = self.notification_templates.get(channel)
            
            if not provider or not template:
                continue
            
            total_channels += 1
            
            # Retry logic
            for attempt in range(config.retry_attempts):
                try:
                    success = await provider.send_notification(alert, config, template)
                    if success:
                        success_count += 1
                        self.logger.info(f"Alert {alert.id} sent via {channel.value}")
                        break
                    else:
                        if attempt < config.retry_attempts - 1:
                            await asyncio.sleep(config.retry_delay)
                
                except Exception as e:
                    self.logger.error(f"Attempt {attempt + 1} failed for {channel.value}: {str(e)}")
                    if attempt < config.retry_attempts - 1:
                        await asyncio.sleep(config.retry_delay)
        
        return success_count > 0
    
    def _should_send_alert(self, alert: DriftAlert) -> bool:
        """Check if alert should be sent based on cooldown and other rules."""
        # Check cooldown
        cooldown_key = f"{alert.dataset_id}_{alert.drift_type.value}_{alert.severity.value}"
        if cooldown_key in self.alert_cooldowns:
            if datetime.utcnow() < self.alert_cooldowns[cooldown_key]:
                return False
        
        # Check if alert already exists
        if alert.id in self.active_alerts:
            return False
        
        return True
    
    def _track_alert(self, alert: DriftAlert) -> None:
        """Track alert and set cooldown."""
        self.active_alerts[alert.id] = alert
        
        # Set cooldown
        cooldown_key = f"{alert.dataset_id}_{alert.drift_type.value}_{alert.severity.value}"
        cooldown_period = self._get_cooldown_period(alert)
        self.alert_cooldowns[cooldown_key] = datetime.utcnow() + timedelta(seconds=cooldown_period)
    
    def _get_channels_for_alert(self, alert: DriftAlert) -> List[NotificationChannel]:
        """Determine which notification channels to use for an alert."""
        # Check alert rules
        for rule in self.alert_rules:
            if self._alert_matches_rule(alert, rule):
                return rule.notification_channels
        
        # Default channels based on severity
        if alert.severity == AlertSeverity.CRITICAL:
            return [NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.WEBHOOK]
        elif alert.severity == AlertSeverity.HIGH:
            return [NotificationChannel.EMAIL, NotificationChannel.SLACK]
        elif alert.severity == AlertSeverity.MEDIUM:
            return [NotificationChannel.SLACK]
        else:
            return [NotificationChannel.WEBHOOK]
    
    def _get_cooldown_period(self, alert: DriftAlert) -> int:
        """Get cooldown period for alert type."""
        # Check alert rules
        for rule in self.alert_rules:
            if self._alert_matches_rule(alert, rule):
                return rule.cooldown_period
        
        # Default cooldown based on severity
        cooldown_map = {
            AlertSeverity.CRITICAL: 300,   # 5 minutes
            AlertSeverity.HIGH: 600,       # 10 minutes
            AlertSeverity.MEDIUM: 1800,    # 30 minutes
            AlertSeverity.LOW: 3600        # 1 hour
        }
        return cooldown_map.get(alert.severity, 300)
    
    def _alert_matches_rule(self, alert: DriftAlert, rule: AlertRule) -> bool:
        """Check if alert matches a rule."""
        # Simple matching based on severity and conditions
        if rule.severity != alert.severity:
            return False
        
        # Check conditions
        conditions = rule.conditions
        if 'dataset_pattern' in conditions:
            import re
            if not re.match(conditions['dataset_pattern'], alert.dataset_id):
                return False
        
        if 'drift_score_threshold' in conditions:
            if alert.drift_score < conditions['drift_score_threshold']:
                return False
        
        return True
    
    def _evaluate_custom_rules(self, report: DriftReport) -> List[DriftAlert]:
        """Evaluate custom alert rules against drift report."""
        custom_alerts = []
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Check if rule conditions are met
            if self._rule_conditions_met(report, rule):
                alert = DriftAlert(
                    id=f"custom_{rule.name}_{report.dataset_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    dataset_id=report.dataset_id,
                    drift_type=report.alerts[0].drift_type if report.alerts else None,
                    severity=rule.severity,
                    message=f"Custom rule '{rule.name}' triggered: {rule.description}",
                    affected_features=list(report.feature_drift_scores.keys()),
                    drift_score=report.drift_score,
                    threshold=0.0  # Custom rules don't have thresholds
                )
                custom_alerts.append(alert)
        
        return custom_alerts
    
    def _rule_conditions_met(self, report: DriftReport, rule: AlertRule) -> bool:
        """Check if rule conditions are met by the drift report."""
        conditions = rule.conditions
        
        if 'min_drift_score' in conditions:
            if report.drift_score < conditions['min_drift_score']:
                return False
        
        if 'max_drift_score' in conditions:
            if report.drift_score > conditions['max_drift_score']:
                return False
        
        if 'min_affected_features' in conditions:
            high_drift_features = [
                feature for feature, score in report.feature_drift_scores.items()
                if score >= conditions.get('feature_drift_threshold', 0.3)
            ]
            if len(high_drift_features) < conditions['min_affected_features']:
                return False
        
        return True
    
    async def _handle_escalation(self, alert: DriftAlert, escalation_rule: EscalationRule) -> None:
        """Handle alert escalation."""
        try:
            await asyncio.sleep(escalation_rule.escalation_delay * 60)  # Convert to seconds
            
            # Check if alert is still active and not acknowledged
            if alert.id in self.active_alerts and not self.active_alerts[alert.id].acknowledged:
                self.logger.info(f"Escalating alert {alert.id}")
                
                # Create escalated alert
                escalated_alert = DriftAlert(
                    id=f"escalated_{alert.id}",
                    dataset_id=alert.dataset_id,
                    drift_type=alert.drift_type,
                    severity=alert.severity,
                    message=f"ESCALATED: {alert.message}",
                    affected_features=alert.affected_features,
                    drift_score=alert.drift_score,
                    threshold=alert.threshold
                )
                
                # Send escalated notification
                await self._send_alert_notifications(escalated_alert)
        
        except Exception as e:
            self.logger.error(f"Error handling escalation for alert {alert.id}: {str(e)}")
    
    def _initialize_default_templates(self) -> None:
        """Initialize default notification templates."""
        # Email template
        email_template = NotificationTemplate(
            channel=NotificationChannel.EMAIL,
            subject_template="[{severity}] Data Drift Alert - {dataset_id}",
            body_template="""
Data Drift Alert

Dataset: {dataset_id}
Severity: {severity}
Drift Score: {drift_score:.3f}
Threshold: {threshold:.3f}
Detected At: {detected_at}

Message: {message}

Affected Features: {affected_features}

Alert ID: {alert_id}

Please investigate this drift detection and take appropriate action.
            """.strip(),
            format_type="text"
        )
        self.notification_templates[NotificationChannel.EMAIL] = email_template
        
        # Slack template
        slack_template = NotificationTemplate(
            channel=NotificationChannel.SLACK,
            subject_template="🚨 Data Drift Alert: {dataset_id} ({severity})",
            body_template="Data drift detected in dataset {dataset_id} with score {drift_score:.3f}",
            format_type="text"
        )
        self.notification_templates[NotificationChannel.SLACK] = slack_template
        
        # Webhook template
        webhook_template = NotificationTemplate(
            channel=NotificationChannel.WEBHOOK,
            subject_template="Data Drift Alert",
            body_template="Alert for dataset {dataset_id}",
            format_type="json"
        )
        self.notification_templates[NotificationChannel.WEBHOOK] = webhook_template