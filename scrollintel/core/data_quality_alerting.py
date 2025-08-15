"""
Automated Data Quality Reporting and Alerting System

Provides automated quality monitoring, alerting, and reporting
capabilities for the Advanced Analytics Dashboard System.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
from jinja2 import Template

from .data_quality_monitor import DataQualityMonitor, QualityReport, QualityIssue, QualitySeverity

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
    MONTHLY = "monthly"


class ReportFormat(Enum):
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"


@dataclass
class AlertRule:
    """Defines when and how to send quality alerts"""
    id: str
    name: str
    description: str
    trigger_conditions: Dict[str, Any]
    severity_threshold: QualitySeverity
    channels: List[AlertChannel]
    recipients: List[str]
    frequency: AlertFrequency
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportSchedule:
    """Defines automated report generation schedule"""
    id: str
    name: str
    description: str
    datasets: List[str]
    report_format: ReportFormat
    frequency: AlertFrequency
    recipients: List[str]
    template_id: Optional[str] = None
    enabled: bool = True
    last_generated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Represents a quality alert"""
    id: str
    rule_id: str
    dataset_name: str
    severity: QualitySeverity
    title: str
    message: str
    issues: List[QualityIssue]
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class DataQualityAlertingSystem:
    """
    Automated data quality alerting and reporting system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.quality_monitor = DataQualityMonitor()
        self.alert_rules: Dict[str, AlertRule] = {}
        self.report_schedules: Dict[str, ReportSchedule] = {}
        self.alert_history: List[Alert] = []
        self.report_templates = self._initialize_templates()
        self.notification_handlers = self._initialize_notification_handlers()
        
    def register_alert_rule(self, rule: AlertRule) -> bool:
        """Register an alert rule"""
        try:
            self.alert_rules[rule.id] = rule
            logger.info(f"Registered alert rule: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register alert rule {rule.name}: {str(e)}")
            return False
    
    def register_report_schedule(self, schedule: ReportSchedule) -> bool:
        """Register a report schedule"""
        try:
            self.report_schedules[schedule.id] = schedule
            logger.info(f"Registered report schedule: {schedule.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register report schedule {schedule.name}: {str(e)}")
            return False
    
    async def monitor_and_alert(self, dataset_name: str, data: Any) -> Dict[str, Any]:
        """Monitor data quality and trigger alerts if needed"""
        try:
            # Perform quality assessment
            quality_report = self.quality_monitor.assess_data_quality(data, dataset_name)
            
            # Check alert rules
            triggered_alerts = []
            
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Check if rule should be triggered
                should_trigger = self._evaluate_alert_rule(rule, quality_report)
                
                if should_trigger:
                    # Check frequency constraints
                    if self._can_trigger_alert(rule):
                        alert = await self._create_and_send_alert(rule, quality_report)
                        if alert:
                            triggered_alerts.append(alert)
                            rule.last_triggered = datetime.utcnow()
            
            return {
                "quality_report": quality_report,
                "alerts_triggered": len(triggered_alerts),
                "alert_details": [
                    {
                        "id": alert.id,
                        "severity": alert.severity.value,
                        "title": alert.title
                    }
                    for alert in triggered_alerts
                ]
            }
            
        except Exception as e:
            logger.error(f"Monitoring and alerting failed: {str(e)}")
            return {"error": str(e)}
    
    def _evaluate_alert_rule(self, rule: AlertRule, quality_report: QualityReport) -> bool:
        """Evaluate if an alert rule should be triggered"""
        try:
            conditions = rule.trigger_conditions
            
            # Check overall quality score threshold
            if "min_quality_score" in conditions:
                if quality_report.overall_score < conditions["min_quality_score"]:
                    return True
            
            # Check dimension score thresholds
            if "dimension_thresholds" in conditions:
                for dimension, threshold in conditions["dimension_thresholds"].items():
                    if dimension in quality_report.dimension_scores:
                        if quality_report.dimension_scores[dimension] < threshold:
                            return True
            
            # Check issue count thresholds
            if "max_issues" in conditions:
                if len(quality_report.issues) > conditions["max_issues"]:
                    return True
            
            # Check severity-specific issue counts
            if "max_critical_issues" in conditions:
                critical_issues = [
                    issue for issue in quality_report.issues
                    if issue.severity == QualitySeverity.CRITICAL
                ]
                if len(critical_issues) > conditions["max_critical_issues"]:
                    return True
            
            # Check specific field issues
            if "field_issues" in conditions:
                for field_name, max_issues in conditions["field_issues"].items():
                    field_issues = [
                        issue for issue in quality_report.issues
                        if issue.field_name == field_name
                    ]
                    if len(field_issues) > max_issues:
                        return True
            
            # Check data completeness
            if "min_completeness" in conditions:
                # Calculate overall completeness from quality metrics
                completeness_metrics = quality_report.metrics.get("completeness", {})
                if completeness_metrics:
                    avg_completeness = sum(completeness_metrics.values()) / len(completeness_metrics)
                    if avg_completeness < conditions["min_completeness"]:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate alert rule {rule.name}: {str(e)}")
            return False
    
    def _can_trigger_alert(self, rule: AlertRule) -> bool:
        """Check if alert can be triggered based on frequency constraints"""
        if not rule.last_triggered:
            return True
        
        current_time = datetime.utcnow()
        time_since_last = current_time - rule.last_triggered
        
        if rule.frequency == AlertFrequency.IMMEDIATE:
            return True
        elif rule.frequency == AlertFrequency.HOURLY:
            return time_since_last >= timedelta(hours=1)
        elif rule.frequency == AlertFrequency.DAILY:
            return time_since_last >= timedelta(days=1)
        elif rule.frequency == AlertFrequency.WEEKLY:
            return time_since_last >= timedelta(weeks=1)
        elif rule.frequency == AlertFrequency.MONTHLY:
            return time_since_last >= timedelta(days=30)
        
        return False
    
    async def _create_and_send_alert(self, rule: AlertRule, 
                                   quality_report: QualityReport) -> Optional[Alert]:
        """Create and send an alert"""
        try:
            # Filter issues by severity
            relevant_issues = [
                issue for issue in quality_report.issues
                if self._issue_meets_severity_threshold(issue.severity, rule.severity_threshold)
            ]
            
            if not relevant_issues:
                return None
            
            # Create alert
            alert = Alert(
                id=f"alert_{datetime.utcnow().timestamp()}",
                rule_id=rule.id,
                dataset_name=quality_report.dataset_name,
                severity=max(issue.severity for issue in relevant_issues),
                title=f"Data Quality Alert: {rule.name}",
                message=self._generate_alert_message(rule, quality_report, relevant_issues),
                issues=relevant_issues
            )
            
            # Send alert through configured channels
            for channel in rule.channels:
                try:
                    handler = self.notification_handlers.get(channel)
                    if handler:
                        await handler(alert, rule.recipients)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel.value}: {str(e)}")
            
            # Store alert in history
            self.alert_history.append(alert)
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create and send alert: {str(e)}")
            return None
    
    def _issue_meets_severity_threshold(self, issue_severity: QualitySeverity, 
                                      threshold: QualitySeverity) -> bool:
        """Check if issue severity meets the threshold"""
        severity_levels = {
            QualitySeverity.LOW: 1,
            QualitySeverity.MEDIUM: 2,
            QualitySeverity.HIGH: 3,
            QualitySeverity.CRITICAL: 4
        }
        
        return severity_levels.get(issue_severity, 0) >= severity_levels.get(threshold, 0)
    
    def _generate_alert_message(self, rule: AlertRule, quality_report: QualityReport,
                              issues: List[QualityIssue]) -> str:
        """Generate alert message"""
        try:
            message_parts = [
                f"Data quality alert triggered for dataset: {quality_report.dataset_name}",
                f"Overall quality score: {quality_report.overall_score:.1f}%",
                f"Issues found: {len(issues)}",
                "",
                "Issue Summary:"
            ]
            
            # Group issues by severity
            issues_by_severity = {}
            for issue in issues:
                if issue.severity not in issues_by_severity:
                    issues_by_severity[issue.severity] = []
                issues_by_severity[issue.severity].append(issue)
            
            for severity in [QualitySeverity.CRITICAL, QualitySeverity.HIGH, 
                           QualitySeverity.MEDIUM, QualitySeverity.LOW]:
                if severity in issues_by_severity:
                    message_parts.append(f"  {severity.value.upper()}: {len(issues_by_severity[severity])} issues")
                    
                    # Add top 3 issues for this severity
                    for issue in issues_by_severity[severity][:3]:
                        message_parts.append(f"    - {issue.field_name}: {issue.issue_description}")
            
            # Add recommendations if available
            if quality_report.recommendations:
                message_parts.extend([
                    "",
                    "Recommendations:",
                    *[f"  - {rec}" for rec in quality_report.recommendations[:5]]
                ])
            
            return "\n".join(message_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate alert message: {str(e)}")
            return f"Data quality alert for {quality_report.dataset_name} - {len(issues)} issues found"
    
    async def generate_scheduled_reports(self) -> Dict[str, Any]:
        """Generate and send scheduled reports"""
        try:
            current_time = datetime.utcnow()
            reports_generated = []
            
            for schedule_id, schedule in self.report_schedules.items():
                if not schedule.enabled:
                    continue
                
                # Check if report should be generated
                if self._should_generate_report(schedule, current_time):
                    try:
                        report_result = await self._generate_and_send_report(schedule)
                        if report_result["success"]:
                            reports_generated.append({
                                "schedule_id": schedule_id,
                                "schedule_name": schedule.name,
                                "datasets": schedule.datasets,
                                "recipients": len(schedule.recipients)
                            })
                            schedule.last_generated = current_time
                    except Exception as e:
                        logger.error(f"Failed to generate report for schedule {schedule.name}: {str(e)}")
            
            return {
                "reports_generated": len(reports_generated),
                "report_details": reports_generated,
                "timestamp": current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Scheduled report generation failed: {str(e)}")
            return {"error": str(e)}
    
    def _should_generate_report(self, schedule: ReportSchedule, current_time: datetime) -> bool:
        """Check if a report should be generated based on schedule"""
        if not schedule.last_generated:
            return True
        
        time_since_last = current_time - schedule.last_generated
        
        if schedule.frequency == AlertFrequency.HOURLY:
            return time_since_last >= timedelta(hours=1)
        elif schedule.frequency == AlertFrequency.DAILY:
            return time_since_last >= timedelta(days=1)
        elif schedule.frequency == AlertFrequency.WEEKLY:
            return time_since_last >= timedelta(weeks=1)
        elif schedule.frequency == AlertFrequency.MONTHLY:
            return time_since_last >= timedelta(days=30)
        
        return False
    
    async def _generate_and_send_report(self, schedule: ReportSchedule) -> Dict[str, Any]:
        """Generate and send a quality report"""
        try:
            # Collect quality reports for all datasets
            dataset_reports = {}
            
            for dataset_name in schedule.datasets:
                # This would typically fetch recent data for the dataset
                # For now, we'll use the most recent report from history
                recent_reports = [
                    report for report in self.quality_monitor.quality_history
                    if report.dataset_name == dataset_name
                ]
                
                if recent_reports:
                    # Get the most recent report
                    latest_report = max(recent_reports, key=lambda r: r.assessment_timestamp)
                    dataset_reports[dataset_name] = latest_report
            
            if not dataset_reports:
                return {"success": False, "error": "No quality reports available for datasets"}
            
            # Generate report content
            report_content = self._generate_report_content(
                schedule, dataset_reports
            )
            
            # Send report to recipients
            for recipient in schedule.recipients:
                try:
                    await self._send_report(report_content, schedule, recipient)
                except Exception as e:
                    logger.error(f"Failed to send report to {recipient}: {str(e)}")
            
            return {
                "success": True,
                "datasets_included": len(dataset_reports),
                "recipients_notified": len(schedule.recipients)
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _generate_report_content(self, schedule: ReportSchedule, 
                               dataset_reports: Dict[str, QualityReport]) -> str:
        """Generate report content based on format and template"""
        try:
            if schedule.report_format == ReportFormat.HTML:
                return self._generate_html_report(schedule, dataset_reports)
            elif schedule.report_format == ReportFormat.JSON:
                return self._generate_json_report(schedule, dataset_reports)
            elif schedule.report_format == ReportFormat.CSV:
                return self._generate_csv_report(schedule, dataset_reports)
            else:
                # Default to JSON
                return self._generate_json_report(schedule, dataset_reports)
                
        except Exception as e:
            logger.error(f"Failed to generate report content: {str(e)}")
            return f"Report generation failed: {str(e)}"
    
    def _generate_html_report(self, schedule: ReportSchedule, 
                            dataset_reports: Dict[str, QualityReport]) -> str:
        """Generate HTML report"""
        template = self.report_templates.get("html_template", "")
        
        if not template:
            # Basic HTML template
            html_parts = [
                "<html><head><title>Data Quality Report</title></head><body>",
                f"<h1>{schedule.name}</h1>",
                f"<p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</p>",
                "<h2>Dataset Quality Summary</h2>"
            ]
            
            for dataset_name, report in dataset_reports.items():
                html_parts.extend([
                    f"<h3>{dataset_name}</h3>",
                    f"<p>Overall Score: {report.overall_score:.1f}%</p>",
                    f"<p>Total Records: {report.total_records:,}</p>",
                    f"<p>Issues Found: {len(report.issues)}</p>",
                    "<h4>Dimension Scores</h4>",
                    "<ul>"
                ])
                
                for dimension, score in report.dimension_scores.items():
                    html_parts.append(f"<li>{dimension}: {score:.1f}%</li>")
                
                html_parts.append("</ul>")
                
                if report.issues:
                    html_parts.extend([
                        "<h4>Top Issues</h4>",
                        "<ul>"
                    ])
                    
                    for issue in report.issues[:5]:
                        html_parts.append(
                            f"<li><strong>{issue.severity.value.upper()}</strong> - "
                            f"{issue.field_name}: {issue.issue_description}</li>"
                        )
                    
                    html_parts.append("</ul>")
            
            html_parts.append("</body></html>")
            return "\n".join(html_parts)
        
        else:
            # Use Jinja2 template
            template_obj = Template(template)
            return template_obj.render(
                schedule=schedule,
                dataset_reports=dataset_reports,
                generated_at=datetime.utcnow()
            )
    
    def _generate_json_report(self, schedule: ReportSchedule, 
                            dataset_reports: Dict[str, QualityReport]) -> str:
        """Generate JSON report"""
        report_data = {
            "report_name": schedule.name,
            "generated_at": datetime.utcnow().isoformat(),
            "datasets": {}
        }
        
        for dataset_name, report in dataset_reports.items():
            report_data["datasets"][dataset_name] = {
                "overall_score": report.overall_score,
                "total_records": report.total_records,
                "dimension_scores": report.dimension_scores,
                "issues_count": len(report.issues),
                "issues": [
                    {
                        "severity": issue.severity.value,
                        "field_name": issue.field_name,
                        "description": issue.issue_description,
                        "affected_records": issue.affected_records
                    }
                    for issue in report.issues
                ],
                "recommendations": report.recommendations
            }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_csv_report(self, schedule: ReportSchedule, 
                           dataset_reports: Dict[str, QualityReport]) -> str:
        """Generate CSV report"""
        csv_lines = [
            "Dataset,Overall_Score,Total_Records,Issues_Count,Top_Issue_Severity,Top_Issue_Description"
        ]
        
        for dataset_name, report in dataset_reports.items():
            top_issue = report.issues[0] if report.issues else None
            
            csv_lines.append(
                f'"{dataset_name}",'
                f'{report.overall_score:.1f},'
                f'{report.total_records},'
                f'{len(report.issues)},'
                f'"{top_issue.severity.value if top_issue else ""}",'
                f'"{top_issue.issue_description if top_issue else ""}"'
            )
        
        return "\n".join(csv_lines)
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize report templates"""
        return {
            "html_template": """
            <html>
            <head>
                <title>{{ schedule.name }}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background-color: #f0f0f0; padding: 10px; }
                    .dataset { margin: 20px 0; border: 1px solid #ddd; padding: 15px; }
                    .score { font-size: 24px; font-weight: bold; }
                    .critical { color: #d32f2f; }
                    .high { color: #f57c00; }
                    .medium { color: #fbc02d; }
                    .low { color: #388e3c; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{{ schedule.name }}</h1>
                    <p>Generated: {{ generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                </div>
                
                {% for dataset_name, report in dataset_reports.items() %}
                <div class="dataset">
                    <h2>{{ dataset_name }}</h2>
                    <div class="score">Overall Score: {{ "%.1f"|format(report.overall_score) }}%</div>
                    <p>Total Records: {{ "{:,}".format(report.total_records) }}</p>
                    <p>Issues Found: {{ report.issues|length }}</p>
                    
                    <h3>Dimension Scores</h3>
                    <ul>
                    {% for dimension, score in report.dimension_scores.items() %}
                        <li>{{ dimension }}: {{ "%.1f"|format(score) }}%</li>
                    {% endfor %}
                    </ul>
                    
                    {% if report.issues %}
                    <h3>Top Issues</h3>
                    <ul>
                    {% for issue in report.issues[:5] %}
                        <li class="{{ issue.severity.value }}">
                            <strong>{{ issue.severity.value.upper() }}</strong> - 
                            {{ issue.field_name }}: {{ issue.issue_description }}
                        </li>
                    {% endfor %}
                    </ul>
                    {% endif %}
                </div>
                {% endfor %}
            </body>
            </html>
            """
        }
    
    def _initialize_notification_handlers(self) -> Dict[AlertChannel, Callable]:
        """Initialize notification handlers for different channels"""
        return {
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.SLACK: self._send_slack_alert,
            AlertChannel.WEBHOOK: self._send_webhook_alert,
            AlertChannel.DASHBOARD: self._send_dashboard_alert
        }
    
    async def _send_email_alert(self, alert: Alert, recipients: List[str]) -> None:
        """Send email alert"""
        try:
            smtp_config = self.config.get("smtp", {})
            if not smtp_config:
                logger.warning("SMTP configuration not found, skipping email alert")
                return
            
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get("from_address", "noreply@scrollintel.com")
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = alert.title
            
            body = alert.message
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_config.get("host", "localhost"), 
                                smtp_config.get("port", 587))
            if smtp_config.get("use_tls", True):
                server.starttls()
            
            if smtp_config.get("username") and smtp_config.get("password"):
                server.login(smtp_config["username"], smtp_config["password"])
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent to {len(recipients)} recipients")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
    
    async def _send_slack_alert(self, alert: Alert, recipients: List[str]) -> None:
        """Send Slack alert"""
        try:
            slack_config = self.config.get("slack", {})
            webhook_url = slack_config.get("webhook_url")
            
            if not webhook_url:
                logger.warning("Slack webhook URL not configured, skipping Slack alert")
                return
            
            # Format message for Slack
            slack_message = {
                "text": alert.title,
                "attachments": [
                    {
                        "color": self._get_slack_color(alert.severity),
                        "fields": [
                            {
                                "title": "Dataset",
                                "value": alert.dataset_name,
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Issues Found",
                                "value": str(len(alert.issues)),
                                "short": True
                            }
                        ],
                        "text": alert.message[:500] + "..." if len(alert.message) > 500 else alert.message
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(webhook_url, json=slack_message)
            response.raise_for_status()
            
            logger.info("Slack alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
    
    async def _send_webhook_alert(self, alert: Alert, recipients: List[str]) -> None:
        """Send webhook alert"""
        try:
            webhook_config = self.config.get("webhook", {})
            webhook_url = webhook_config.get("url")
            
            if not webhook_url:
                logger.warning("Webhook URL not configured, skipping webhook alert")
                return
            
            # Prepare webhook payload
            payload = {
                "alert_id": alert.id,
                "title": alert.title,
                "message": alert.message,
                "dataset": alert.dataset_name,
                "severity": alert.severity.value,
                "issues_count": len(alert.issues),
                "triggered_at": alert.triggered_at.isoformat(),
                "recipients": recipients
            }
            
            # Send webhook
            headers = {"Content-Type": "application/json"}
            if webhook_config.get("auth_header"):
                headers["Authorization"] = webhook_config["auth_header"]
            
            response = requests.post(webhook_url, json=payload, headers=headers)
            response.raise_for_status()
            
            logger.info("Webhook alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {str(e)}")
    
    async def _send_dashboard_alert(self, alert: Alert, recipients: List[str]) -> None:
        """Send dashboard alert (store for dashboard display)"""
        try:
            # This would typically store the alert in a database or cache
            # for display in the dashboard UI
            logger.info(f"Dashboard alert created: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to create dashboard alert: {str(e)}")
    
    async def _send_report(self, report_content: str, schedule: ReportSchedule, 
                         recipient: str) -> None:
        """Send report to recipient"""
        try:
            if schedule.report_format == ReportFormat.EMAIL or "@" in recipient:
                await self._send_report_email(report_content, schedule, recipient)
            else:
                logger.warning(f"Unsupported recipient format: {recipient}")
                
        except Exception as e:
            logger.error(f"Failed to send report to {recipient}: {str(e)}")
    
    async def _send_report_email(self, report_content: str, schedule: ReportSchedule, 
                               recipient: str) -> None:
        """Send report via email"""
        try:
            smtp_config = self.config.get("smtp", {})
            if not smtp_config:
                logger.warning("SMTP configuration not found, skipping report email")
                return
            
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get("from_address", "noreply@scrollintel.com")
            msg['To'] = recipient
            msg['Subject'] = f"Data Quality Report: {schedule.name}"
            
            # Attach report content
            if schedule.report_format == ReportFormat.HTML:
                msg.attach(MIMEText(report_content, 'html'))
            else:
                # Attach as file
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(report_content.encode())
                encoders.encode_base64(attachment)
                
                filename = f"quality_report_{schedule.name}_{datetime.utcnow().strftime('%Y%m%d')}"
                if schedule.report_format == ReportFormat.JSON:
                    filename += ".json"
                elif schedule.report_format == ReportFormat.CSV:
                    filename += ".csv"
                
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename}'
                )
                msg.attach(attachment)
            
            # Send email
            server = smtplib.SMTP(smtp_config.get("host", "localhost"), 
                                smtp_config.get("port", 587))
            if smtp_config.get("use_tls", True):
                server.starttls()
            
            if smtp_config.get("username") and smtp_config.get("password"):
                server.login(smtp_config["username"], smtp_config["password"])
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Report email sent to {recipient}")
            
        except Exception as e:
            logger.error(f"Failed to send report email: {str(e)}")
    
    def _get_slack_color(self, severity: QualitySeverity) -> str:
        """Get Slack color for severity level"""
        color_map = {
            QualitySeverity.LOW: "good",
            QualitySeverity.MEDIUM: "warning",
            QualitySeverity.HIGH: "danger",
            QualitySeverity.CRITICAL: "danger"
        }
        return color_map.get(severity, "warning")
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            for alert in self.alert_history:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = user_id
                    alert.acknowledged_at = datetime.utcnow()
                    logger.info(f"Alert {alert_id} acknowledged by {user_id}")
                    return True
            
            logger.warning(f"Alert {alert_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {str(e)}")
            return False
    
    def get_alert_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get alert summary for specified period"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.triggered_at >= cutoff_date
            ]
            
            if not recent_alerts:
                return {"message": "No alerts in specified period"}
            
            # Analyze alerts
            severity_counts = {severity.value: 0 for severity in QualitySeverity}
            dataset_counts = {}
            acknowledged_count = 0
            
            for alert in recent_alerts:
                severity_counts[alert.severity.value] += 1
                dataset_counts[alert.dataset_name] = dataset_counts.get(alert.dataset_name, 0) + 1
                if alert.acknowledged:
                    acknowledged_count += 1
            
            return {
                "period_days": days,
                "total_alerts": len(recent_alerts),
                "alerts_by_severity": severity_counts,
                "alerts_by_dataset": dataset_counts,
                "acknowledgment_rate": acknowledged_count / len(recent_alerts) if recent_alerts else 0,
                "unacknowledged_alerts": len(recent_alerts) - acknowledged_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get alert summary: {str(e)}")
            return {"error": str(e)}