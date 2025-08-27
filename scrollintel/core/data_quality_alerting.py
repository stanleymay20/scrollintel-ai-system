"""
Data Quality Alerting System for Advanced Analytics Dashboard

Provides automated alerting and notification capabilities for data quality issues
with configurable thresholds and multiple notification channels.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
from collections import defaultdict

from .data_quality_monitor import QualityReport, QualityIssue, QualitySeverity

logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"


class AlertFrequency(Enum):
    IMMEDIATE = "immediate"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class AlertRule:
    """Defines when and how to send quality alerts"""
    id: str
    name: str
    description: str
    dataset_patterns: List[str]  # Regex patterns for dataset names
    severity_threshold: QualitySeverity
    score_threshold: float
    channels: List[AlertChannel]
    frequency: AlertFrequency
    enabled: bool = True
    recipients: List[str] = field(default_factory=list)
    custom_conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AlertNotification:
    """Represents a quality alert notification"""
    id: str
    rule_id: str
    dataset_name: str
    alert_type: str
    severity: str
    message: str
    details: Dict[str, Any]
    channels_sent: List[AlertChannel] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    sent_at: Optional[datetime] = None
    status: str = "pending"  # pending, sent, failed


@dataclass
class NotificationConfig:
    """Configuration for notification channels"""
    email_config: Dict[str, Any] = field(default_factory=dict)
    slack_config: Dict[str, Any] = field(default_factory=dict)
    webhook_config: Dict[str, Any] = field(default_factory=dict)
    sms_config: Dict[str, Any] = field(default_factory=dict)


class DataQualityAlerting:
    """
    Comprehensive data quality alerting and notification system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_config = NotificationConfig()
        self.alert_history: List[AlertNotification] = []
        self.pending_alerts: List[AlertNotification] = []
        self.notification_handlers = self._initialize_handlers()
        
        # Load configuration
        self._load_notification_config()
    
    def register_alert_rule(self, rule: AlertRule) -> bool:
        """Register a new alert rule"""
        try:
            self.alert_rules[rule.id] = rule
            logger.info(f"Registered alert rule: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register alert rule {rule.name}: {str(e)}")
            return False
    
    def process_quality_report(self, report: QualityReport) -> List[AlertNotification]:
        """Process a quality report and generate alerts if needed"""
        try:
            generated_alerts = []
            
            # Check each alert rule
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                # Check if rule applies to this dataset
                if not self._rule_applies_to_dataset(rule, report.dataset_name):
                    continue
                
                # Check if alert conditions are met
                alert_triggered = self._check_alert_conditions(rule, report)
                
                if alert_triggered:
                    # Check frequency constraints
                    if self._should_send_alert(rule, report.dataset_name):
                        alert = self._create_alert_notification(rule, report, alert_triggered)
                        generated_alerts.append(alert)
                        self.pending_alerts.append(alert)
            
            # Process pending alerts
            self._process_pending_alerts()
            
            return generated_alerts
            
        except Exception as e:
            logger.error(f"Failed to process quality report: {str(e)}")
            return []
    
    def _rule_applies_to_dataset(self, rule: AlertRule, dataset_name: str) -> bool:
        """Check if alert rule applies to the given dataset"""
        try:
            import re
            
            for pattern in rule.dataset_patterns:
                if re.match(pattern, dataset_name):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check rule applicability: {str(e)}")
            return False
    
    def _check_alert_conditions(self, rule: AlertRule, report: QualityReport) -> Optional[Dict[str, Any]]:
        """Check if alert conditions are met for the given report"""
        try:
            alert_reasons = []
            
            # Check overall score threshold
            if report.overall_score < rule.score_threshold:
                alert_reasons.append({
                    "type": "low_score",
                    "message": f"Overall quality score ({report.overall_score:.1f}%) below threshold ({rule.score_threshold}%)",
                    "current_value": report.overall_score,
                    "threshold": rule.score_threshold
                })
            
            # Check severity threshold
            severe_issues = [
                issue for issue in report.issues
                if self._severity_level(issue.severity) >= self._severity_level(rule.severity_threshold)
            ]
            
            if severe_issues:
                alert_reasons.append({
                    "type": "severe_issues",
                    "message": f"Found {len(severe_issues)} issues at or above {rule.severity_threshold.value} severity",
                    "issue_count": len(severe_issues),
                    "severity_threshold": rule.severity_threshold.value,
                    "issues": [
                        {
                            "field": issue.field_name,
                            "description": issue.issue_description,
                            "severity": issue.severity.value,
                            "affected_records": issue.affected_records
                        }
                        for issue in severe_issues[:5]  # Limit to first 5 issues
                    ]
                })
            
            # Check custom conditions
            custom_alerts = self._check_custom_conditions(rule, report)
            alert_reasons.extend(custom_alerts)
            
            if alert_reasons:
                return {
                    "reasons": alert_reasons,
                    "report_summary": {
                        "dataset_name": report.dataset_name,
                        "overall_score": report.overall_score,
                        "total_issues": len(report.issues),
                        "assessment_timestamp": report.assessment_timestamp.isoformat()
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check alert conditions: {str(e)}")
            return None
    
    def _severity_level(self, severity: QualitySeverity) -> int:
        """Convert severity to numeric level for comparison"""
        severity_levels = {
            QualitySeverity.LOW: 1,
            QualitySeverity.MEDIUM: 2,
            QualitySeverity.HIGH: 3,
            QualitySeverity.CRITICAL: 4
        }
        return severity_levels.get(severity, 0)
    
    def _check_custom_conditions(self, rule: AlertRule, report: QualityReport) -> List[Dict[str, Any]]:
        """Check custom alert conditions"""
        custom_alerts = []
        
        try:
            custom_conditions = rule.custom_conditions
            
            # Check dimension score thresholds
            dimension_thresholds = custom_conditions.get("dimension_thresholds", {})
            for dimension, threshold in dimension_thresholds.items():
                if dimension in report.dimension_scores:
                    if report.dimension_scores[dimension] < threshold:
                        custom_alerts.append({
                            "type": "dimension_threshold",
                            "message": f"{dimension} score ({report.dimension_scores[dimension]:.1f}%) below threshold ({threshold}%)",
                            "dimension": dimension,
                            "current_value": report.dimension_scores[dimension],
                            "threshold": threshold
                        })
            
            # Check issue count thresholds
            issue_count_threshold = custom_conditions.get("max_issues", float('inf'))
            if len(report.issues) > issue_count_threshold:
                custom_alerts.append({
                    "type": "issue_count_threshold",
                    "message": f"Total issues ({len(report.issues)}) exceeds threshold ({issue_count_threshold})",
                    "current_value": len(report.issues),
                    "threshold": issue_count_threshold
                })
            
            # Check affected records threshold
            total_affected = sum(issue.affected_records for issue in report.issues)
            affected_threshold = custom_conditions.get("max_affected_records", float('inf'))
            if total_affected > affected_threshold:
                custom_alerts.append({
                    "type": "affected_records_threshold",
                    "message": f"Total affected records ({total_affected:,}) exceeds threshold ({affected_threshold:,})",
                    "current_value": total_affected,
                    "threshold": affected_threshold
                })
            
        except Exception as e:
            logger.error(f"Failed to check custom conditions: {str(e)}")
        
        return custom_alerts
    
    def _should_send_alert(self, rule: AlertRule, dataset_name: str) -> bool:
        """Check if alert should be sent based on frequency constraints"""
        try:
            if rule.frequency == AlertFrequency.IMMEDIATE:
                return True
            
            # Check recent alerts for this rule and dataset
            now = datetime.utcnow()
            
            if rule.frequency == AlertFrequency.HOURLY:
                cutoff = now - timedelta(hours=1)
            elif rule.frequency == AlertFrequency.DAILY:
                cutoff = now - timedelta(days=1)
            elif rule.frequency == AlertFrequency.WEEKLY:
                cutoff = now - timedelta(weeks=1)
            else:
                return True
            
            # Check if we've sent an alert for this rule and dataset recently
            recent_alerts = [
                alert for alert in self.alert_history
                if (alert.rule_id == rule.id and 
                    alert.dataset_name == dataset_name and
                    alert.created_at >= cutoff)
            ]
            
            return len(recent_alerts) == 0
            
        except Exception as e:
            logger.error(f"Failed to check alert frequency: {str(e)}")
            return True
    
    def _create_alert_notification(self, rule: AlertRule, report: QualityReport, 
                                 alert_details: Dict[str, Any]) -> AlertNotification:
        """Create an alert notification"""
        try:
            alert_id = f"alert_{rule.id}_{report.dataset_name}_{datetime.utcnow().timestamp()}"
            
            # Determine alert severity
            max_severity = "low"
            for reason in alert_details["reasons"]:
                if reason["type"] == "severe_issues":
                    max_severity = "critical"
                    break
                elif reason["type"] in ["low_score", "dimension_threshold"]:
                    max_severity = "high"
            
            # Create summary message
            reason_messages = [reason["message"] for reason in alert_details["reasons"]]
            summary_message = f"Data quality alert for {report.dataset_name}: " + "; ".join(reason_messages)
            
            notification = AlertNotification(
                id=alert_id,
                rule_id=rule.id,
                dataset_name=report.dataset_name,
                alert_type="quality_degradation",
                severity=max_severity,
                message=summary_message,
                details=alert_details
            )
            
            return notification
            
        except Exception as e:
            logger.error(f"Failed to create alert notification: {str(e)}")
            raise
    
    def _process_pending_alerts(self) -> None:
        """Process all pending alert notifications"""
        try:
            for alert in self.pending_alerts[:]:  # Copy list to avoid modification during iteration
                try:
                    rule = self.alert_rules.get(alert.rule_id)
                    if not rule:
                        continue
                    
                    success = True
                    channels_sent = []
                    
                    # Send to each configured channel
                    for channel in rule.channels:
                        try:
                            handler = self.notification_handlers.get(channel)
                            if handler:
                                if handler(alert, rule):
                                    channels_sent.append(channel)
                                else:
                                    success = False
                            else:
                                logger.warning(f"No handler for channel: {channel}")
                                success = False
                        except Exception as e:
                            logger.error(f"Failed to send alert via {channel}: {str(e)}")
                            success = False
                    
                    # Update alert status
                    alert.channels_sent = channels_sent
                    alert.sent_at = datetime.utcnow()
                    alert.status = "sent" if success else "failed"
                    
                    # Move to history
                    self.alert_history.append(alert)
                    self.pending_alerts.remove(alert)
                    
                except Exception as e:
                    logger.error(f"Failed to process alert {alert.id}: {str(e)}")
                    alert.status = "failed"
                    self.alert_history.append(alert)
                    self.pending_alerts.remove(alert)
            
        except Exception as e:
            logger.error(f"Failed to process pending alerts: {str(e)}")
    
    def _initialize_handlers(self) -> Dict[AlertChannel, Callable]:
        """Initialize notification channel handlers"""
        return {
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.SLACK: self._send_slack_alert,
            AlertChannel.WEBHOOK: self._send_webhook_alert,
            AlertChannel.SMS: self._send_sms_alert,
            AlertChannel.DASHBOARD: self._send_dashboard_alert
        }
    
    def _load_notification_config(self) -> None:
        """Load notification configuration from config"""
        try:
            config = self.config.get("notifications", {})
            
            self.notification_config.email_config = config.get("email", {})
            self.notification_config.slack_config = config.get("slack", {})
            self.notification_config.webhook_config = config.get("webhook", {})
            self.notification_config.sms_config = config.get("sms", {})
            
        except Exception as e:
            logger.error(f"Failed to load notification config: {str(e)}")
    
    def _send_email_alert(self, alert: AlertNotification, rule: AlertRule) -> bool:
        """Send alert via email"""
        try:
            email_config = self.notification_config.email_config
            
            if not email_config or not email_config.get("enabled", False):
                return False
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = email_config.get("from_address", "noreply@scrollintel.com")
            msg['To'] = ", ".join(rule.recipients)
            msg['Subject'] = f"Data Quality Alert: {alert.dataset_name}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            smtp_server = email_config.get("smtp_server", "localhost")
            smtp_port = email_config.get("smtp_port", 587)
            username = email_config.get("username")
            password = email_config.get("password")
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if email_config.get("use_tls", True):
                    server.starttls()
                
                if username and password:
                    server.login(username, password)
                
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False
    
    def _send_slack_alert(self, alert: AlertNotification, rule: AlertRule) -> bool:
        """Send alert via Slack"""
        try:
            slack_config = self.notification_config.slack_config
            
            if not slack_config or not slack_config.get("enabled", False):
                return False
            
            webhook_url = slack_config.get("webhook_url")
            if not webhook_url:
                return False
            
            # Create Slack message
            color = "#ff0000" if alert.severity == "critical" else "#ff9900" if alert.severity == "high" else "#ffcc00"
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"Data Quality Alert: {alert.dataset_name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.upper(),
                                "short": True
                            },
                            {
                                "title": "Dataset",
                                "value": alert.dataset_name,
                                "short": True
                            },
                            {
                                "title": "Alert Time",
                                "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "short": True
                            }
                        ],
                        "footer": "ScrollIntel Data Quality Monitor"
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent for {alert.dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            return False
    
    def _send_webhook_alert(self, alert: AlertNotification, rule: AlertRule) -> bool:
        """Send alert via webhook"""
        try:
            webhook_config = self.notification_config.webhook_config
            
            if not webhook_config or not webhook_config.get("enabled", False):
                return False
            
            webhook_url = webhook_config.get("url")
            if not webhook_url:
                return False
            
            # Create webhook payload
            payload = {
                "alert_id": alert.id,
                "rule_id": alert.rule_id,
                "dataset_name": alert.dataset_name,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "details": alert.details,
                "timestamp": alert.created_at.isoformat()
            }
            
            headers = {"Content-Type": "application/json"}
            auth_token = webhook_config.get("auth_token")
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
            
            response = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent for {alert.dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {str(e)}")
            return False
    
    def _send_sms_alert(self, alert: AlertNotification, rule: AlertRule) -> bool:
        """Send alert via SMS (placeholder implementation)"""
        try:
            sms_config = self.notification_config.sms_config
            
            if not sms_config or not sms_config.get("enabled", False):
                return False
            
            # This would integrate with SMS service like Twilio
            logger.info(f"SMS alert would be sent for {alert.dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {str(e)}")
            return False
    
    def _send_dashboard_alert(self, alert: AlertNotification, rule: AlertRule) -> bool:
        """Send alert to dashboard (in-app notification)"""
        try:
            # This would integrate with the dashboard notification system
            logger.info(f"Dashboard alert sent for {alert.dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send dashboard alert: {str(e)}")
            return False
    
    def _create_email_body(self, alert: AlertNotification) -> str:
        """Create HTML email body for alert"""
        try:
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: {bg_color}; color: white; padding: 20px; border-radius: 5px; }}
                    .content {{ margin: 20px 0; }}
                    .details {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                    .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Data Quality Alert</h1>
                    <h2>{dataset_name}</h2>
                    <p>Severity: {severity}</p>
                </div>
                
                <div class="content">
                    <h3>Alert Details</h3>
                    <p>{message}</p>
                    
                    <div class="details">
                        <h4>Summary</h4>
                        <ul>
                            <li>Dataset: {dataset_name}</li>
                            <li>Alert Time: {alert_time}</li>
                            <li>Alert ID: {alert_id}</li>
                        </ul>
                        
                        {reasons_html}
                    </div>
                </div>
                
                <div class="footer">
                    <p>This alert was generated by ScrollIntel Data Quality Monitor</p>
                    <p>Please review your data quality dashboard for more details</p>
                </div>
            </body>
            </html>
            """
            
            # Determine background color based on severity
            bg_colors = {
                "critical": "#dc3545",
                "high": "#fd7e14",
                "medium": "#ffc107",
                "low": "#28a745"
            }
            bg_color = bg_colors.get(alert.severity, "#6c757d")
            
            # Generate reasons HTML
            reasons_html = ""
            if "reasons" in alert.details:
                reasons_html = "<h4>Issues Found</h4><ul>"
                for reason in alert.details["reasons"]:
                    reasons_html += f"<li>{reason['message']}</li>"
                reasons_html += "</ul>"
            
            return html_template.format(
                dataset_name=alert.dataset_name,
                severity=alert.severity.upper(),
                message=alert.message,
                alert_time=alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                alert_id=alert.id,
                bg_color=bg_color,
                reasons_html=reasons_html
            )
            
        except Exception as e:
            logger.error(f"Failed to create email body: {str(e)}")
            return f"<html><body><h1>Data Quality Alert</h1><p>{alert.message}</p></body></html>"
    
    def get_alert_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get alert statistics for the specified period"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.created_at >= cutoff_date
            ]
            
            if not recent_alerts:
                return {"message": "No alerts found in specified period"}
            
            # Calculate statistics
            stats = {
                "period_days": days,
                "total_alerts": len(recent_alerts),
                "alerts_by_severity": defaultdict(int),
                "alerts_by_dataset": defaultdict(int),
                "alerts_by_rule": defaultdict(int),
                "success_rate": 0,
                "most_frequent_issues": [],
                "alert_frequency": {
                    "daily_average": len(recent_alerts) / days,
                    "peak_day": None,
                    "quiet_day": None
                }
            }
            
            # Count by categories
            for alert in recent_alerts:
                stats["alerts_by_severity"][alert.severity] += 1
                stats["alerts_by_dataset"][alert.dataset_name] += 1
                stats["alerts_by_rule"][alert.rule_id] += 1
            
            # Calculate success rate
            successful_alerts = sum(1 for alert in recent_alerts if alert.status == "sent")
            stats["success_rate"] = successful_alerts / len(recent_alerts) if recent_alerts else 0
            
            # Convert defaultdicts to regular dicts
            stats["alerts_by_severity"] = dict(stats["alerts_by_severity"])
            stats["alerts_by_dataset"] = dict(stats["alerts_by_dataset"])
            stats["alerts_by_rule"] = dict(stats["alerts_by_rule"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get alert statistics: {str(e)}")
            return {"error": str(e)}