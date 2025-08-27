"""
Automated reporting and alerting system for prompt analytics.
Provides scheduled reports, real-time alerts, and notification management.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from ..core.config import get_settings
from ..core.logging_config import get_logger
from .prompt_analytics import prompt_performance_tracker
from .analytics_dashboard import team_analytics_dashboard

settings = get_settings()
logger = get_logger(__name__)

class ReportFrequency(Enum):
    """Report frequency options."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ReportSchedule:
    """Scheduled report configuration."""
    schedule_id: str
    name: str
    report_type: str
    frequency: ReportFrequency
    recipients: List[str]
    team_ids: List[str]
    prompt_ids: Optional[List[str]]
    next_run: datetime
    last_run: Optional[datetime]
    active: bool
    created_by: str
    created_at: datetime

@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: Optional[str]
    rule_type: str  # 'threshold', 'trend', 'anomaly'
    metric_name: str
    condition: str  # 'greater_than', 'less_than', 'equals'
    threshold_value: Optional[float]
    trend_direction: Optional[str]
    severity: AlertSeverity
    notification_channels: List[str]
    recipients: List[str]
    prompt_ids: Optional[List[str]]
    team_ids: Optional[List[str]]
    active: bool
    last_triggered: Optional[datetime]
    trigger_count: int
    created_by: str
    created_at: datetime

@dataclass
class Alert:
    """Active alert instance."""
    alert_id: str
    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    triggered_at: datetime
    resolved_at: Optional[datetime]
    status: str  # 'active', 'resolved', 'acknowledged'
    metadata: Dict[str, Any]

class AutomatedReportingSystem:
    """Comprehensive automated reporting and alerting system."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.report_schedules: Dict[str, ReportSchedule] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.running = False
        self.scheduler_task = None
        
    async def create_report_schedule(
        self,
        name: str,
        report_type: str,
        frequency: ReportFrequency,
        recipients: List[str],
        team_ids: List[str],
        prompt_ids: Optional[List[str]] = None,
        created_by: str = "system"
    ) -> str:
        """Create a new report schedule."""
        try:
            schedule_id = str(uuid.uuid4())
            
            # Calculate next run time
            now = datetime.utcnow()
            if frequency == ReportFrequency.HOURLY:
                next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            elif frequency == ReportFrequency.DAILY:
                next_run = now.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=1)
            elif frequency == ReportFrequency.WEEKLY:
                days_ahead = 0 - now.weekday()  # Monday
                if days_ahead <= 0:
                    days_ahead += 7
                next_run = now.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)
            else:  # MONTHLY
                if now.month == 12:
                    next_run = now.replace(year=now.year + 1, month=1, day=1, hour=8, minute=0, second=0, microsecond=0)
                else:
                    next_run = now.replace(month=now.month + 1, day=1, hour=8, minute=0, second=0, microsecond=0)
            
            schedule = ReportSchedule(
                schedule_id=schedule_id,
                name=name,
                report_type=report_type,
                frequency=frequency,
                recipients=recipients,
                team_ids=team_ids,
                prompt_ids=prompt_ids,
                next_run=next_run,
                last_run=None,
                active=True,
                created_by=created_by,
                created_at=now
            )
            
            self.report_schedules[schedule_id] = schedule
            
            self.logger.info(
                f"Created report schedule: {name}",
                schedule_id=schedule_id,
                frequency=frequency.value,
                next_run=next_run.isoformat()
            )
            
            return schedule_id
            
        except Exception as e:
            self.logger.error(f"Error creating report schedule: {e}")
            raise
    
    async def create_alert_rule(
        self,
        name: str,
        rule_type: str,
        metric_name: str,
        condition: str,
        threshold_value: Optional[float] = None,
        trend_direction: Optional[str] = None,
        severity: str = "medium",
        notification_channels: List[str] = None,
        recipients: List[str] = None,
        prompt_ids: Optional[List[str]] = None,
        team_ids: Optional[List[str]] = None,
        created_by: str = "system"
    ) -> str:
        """Create a new alert rule."""
        try:
            rule_id = str(uuid.uuid4())
            
            alert_rule = AlertRule(
                rule_id=rule_id,
                name=name,
                description=None,
                rule_type=rule_type,
                metric_name=metric_name,
                condition=condition,
                threshold_value=threshold_value,
                trend_direction=trend_direction,
                severity=AlertSeverity(severity),
                notification_channels=notification_channels or ["email"],
                recipients=recipients or [],
                prompt_ids=prompt_ids,
                team_ids=team_ids,
                active=True,
                last_triggered=None,
                trigger_count=0,
                created_by=created_by,
                created_at=datetime.utcnow()
            )
            
            self.alert_rules[rule_id] = alert_rule
            
            self.logger.info(
                f"Created alert rule: {name}",
                rule_id=rule_id,
                metric_name=metric_name,
                condition=condition,
                threshold_value=threshold_value
            )
            
            return rule_id
            
        except Exception as e:
            self.logger.error(f"Error creating alert rule: {e}")
            raise
    
    async def generate_report(
        self,
        report_type: str,
        team_ids: List[str],
        date_range: Tuple[datetime, datetime],
        prompt_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive report."""
        try:
            start_date, end_date = date_range
            
            report = {
                'report_id': str(uuid.uuid4()),
                'report_type': report_type,
                'generated_at': datetime.utcnow(),
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'teams': [],
                'summary': {},
                'insights': [],
                'recommendations': []
            }
            
            # Generate report for each team
            for team_id in team_ids:
                team_data = await team_analytics_dashboard.get_team_dashboard_data(
                    team_id=team_id,
                    date_range=date_range
                )
                
                if 'error' not in team_data:
                    report['teams'].append({
                        'team_id': team_id,
                        'data': team_data
                    })
            
            # Generate overall summary
            if report['teams']:
                report['summary'] = await self._generate_report_summary(report['teams'])
                report['insights'] = await self._generate_report_insights(report['teams'])
                report['recommendations'] = await self._generate_report_recommendations(report['teams'])
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return {'error': str(e)}
    
    async def _generate_report_summary(self, teams_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall summary from team data."""
        try:
            total_requests = 0
            total_successful = 0
            total_cost = 0
            accuracy_scores = []
            response_times = []
            
            for team in teams_data:
                team_summary = team['data'].get('summary', {})
                total_requests += team_summary.get('total_requests', 0)
                total_successful += team_summary.get('successful_requests', 0)
                total_cost += team_summary.get('total_cost', 0) or 0
                
                if team_summary.get('avg_accuracy_score'):
                    accuracy_scores.append(team_summary['avg_accuracy_score'])
                
                if team_summary.get('avg_response_time_ms'):
                    response_times.append(team_summary['avg_response_time_ms'])
            
            summary = {
                'total_teams': len(teams_data),
                'total_requests': total_requests,
                'total_successful_requests': total_successful,
                'overall_success_rate': (total_successful / total_requests * 100) if total_requests > 0 else 0,
                'total_cost': total_cost,
                'avg_cost_per_request': total_cost / total_requests if total_requests > 0 else 0,
                'avg_accuracy_score': sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else None,
                'avg_response_time_ms': sum(response_times) / len(response_times) if response_times else None
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating report summary: {e}")
            return {}
    
    async def _generate_report_insights(self, teams_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights from team data."""
        try:
            insights = []
            
            # Analyze performance across teams
            success_rates = []
            for team in teams_data:
                team_summary = team['data'].get('summary', {})
                success_rate = team_summary.get('success_rate', 0)
                success_rates.append(success_rate)
            
            if success_rates:
                avg_success_rate = sum(success_rates) / len(success_rates)
                if avg_success_rate > 95:
                    insights.append({
                        'type': 'performance',
                        'category': 'positive',
                        'title': 'Excellent Overall Performance',
                        'description': f'Average success rate across teams is {avg_success_rate:.1f}%'
                    })
                elif avg_success_rate < 90:
                    insights.append({
                        'type': 'performance',
                        'category': 'concern',
                        'title': 'Performance Issues Detected',
                        'description': f'Average success rate of {avg_success_rate:.1f}% needs attention'
                    })
            
            # Analyze cost efficiency
            total_cost = sum(team['data'].get('summary', {}).get('total_cost', 0) or 0 for team in teams_data)
            if total_cost > 1000:
                insights.append({
                    'type': 'cost',
                    'category': 'optimization',
                    'title': 'Cost Optimization Opportunity',
                    'description': f'Total cost of ${total_cost:.2f} suggests review of usage patterns'
                })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating report insights: {e}")
            return []
    
    async def _generate_report_recommendations(self, teams_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations from team data."""
        try:
            recommendations = []
            
            # Analyze team performance variations
            success_rates = []
            for team in teams_data:
                team_summary = team['data'].get('summary', {})
                success_rates.append(team_summary.get('success_rate', 0))
            
            if len(success_rates) > 1:
                min_rate = min(success_rates)
                max_rate = max(success_rates)
                
                if max_rate - min_rate > 10:  # Significant variation
                    recommendations.append({
                        'type': 'standardization',
                        'priority': 'medium',
                        'title': 'Standardize Best Practices',
                        'description': 'Significant performance variation between teams detected',
                        'action_items': [
                            'Share best practices from high-performing teams',
                            'Standardize prompt templates and guidelines',
                            'Implement cross-team knowledge sharing sessions'
                        ]
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating report recommendations: {e}")
            return []
    
    async def generate_ad_hoc_report(
        self,
        team_id: str,
        report_type: str,
        date_range: Tuple[datetime, datetime],
        recipients: List[str]
    ):
        """Generate and send an ad-hoc report."""
        try:
            report = await self.generate_report(
                report_type=report_type,
                team_ids=[team_id],
                date_range=date_range
            )
            
            if 'error' not in report:
                # Send report to recipients
                await self._send_report(report, recipients)
                
                self.logger.info(
                    f"Generated and sent ad-hoc report",
                    team_id=team_id,
                    report_type=report_type,
                    recipients=len(recipients)
                )
            
        except Exception as e:
            self.logger.error(f"Error generating ad-hoc report: {e}")
    
    async def _send_report(self, report: Dict[str, Any], recipients: List[str]):
        """Send report to recipients."""
        try:
            # Format report as HTML email
            html_content = self._format_report_html(report)
            
            # Send email (simplified implementation)
            for recipient in recipients:
                await self._send_email(
                    to_email=recipient,
                    subject=f"Prompt Analytics Report - {report['report_type'].title()}",
                    html_content=html_content
                )
            
        except Exception as e:
            self.logger.error(f"Error sending report: {e}")
    
    def _format_report_html(self, report: Dict[str, Any]) -> str:
        """Format report as HTML."""
        try:
            html = f"""
            <html>
            <head>
                <title>Prompt Analytics Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .summary {{ margin: 20px 0; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e8f4f8; border-radius: 5px; }}
                    .insights {{ margin: 20px 0; }}
                    .insight {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007acc; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Prompt Analytics Report</h1>
                    <p>Report Type: {report['report_type'].title()}</p>
                    <p>Generated: {report['generated_at'].isoformat()}</p>
                    <p>Period: {report['date_range']['start']} to {report['date_range']['end']}</p>
                </div>
                
                <div class="summary">
                    <h2>Summary</h2>
            """
            
            summary = report.get('summary', {})
            for key, value in summary.items():
                if value is not None:
                    html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
            
            html += """
                </div>
                
                <div class="insights">
                    <h2>Key Insights</h2>
            """
            
            for insight in report.get('insights', []):
                html += f"""
                <div class="insight">
                    <h3>{insight['title']}</h3>
                    <p>{insight['description']}</p>
                </div>
                """
            
            html += """
                </div>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error formatting report HTML: {e}")
            return "<html><body><p>Error formatting report</p></body></html>"
    
    async def _send_email(self, to_email: str, subject: str, html_content: str):
        """Send email notification."""
        try:
            # This is a simplified email implementation
            # In production, you would use a proper email service
            self.logger.info(
                f"Email sent (simulated)",
                to_email=to_email,
                subject=subject
            )
            
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
    
    async def check_alert_rules(self):
        """Check all active alert rules and trigger alerts if conditions are met."""
        try:
            for rule_id, rule in self.alert_rules.items():
                if not rule.active:
                    continue
                
                # Check rule conditions
                should_trigger = await self._evaluate_alert_rule(rule)
                
                if should_trigger:
                    await self._trigger_alert(rule)
            
        except Exception as e:
            self.logger.error(f"Error checking alert rules: {e}")
    
    async def _evaluate_alert_rule(self, rule: AlertRule) -> bool:
        """Evaluate if an alert rule should trigger."""
        try:
            # Get relevant data based on rule scope
            if rule.team_ids:
                for team_id in rule.team_ids:
                    team_analytics = await prompt_performance_tracker.get_team_analytics(
                        team_id=team_id,
                        days=1  # Check recent data
                    )
                    
                    if 'error' in team_analytics:
                        continue
                    
                    summary = team_analytics.get('summary', {})
                    metric_value = summary.get(rule.metric_name)
                    
                    if metric_value is not None:
                        if self._check_threshold_condition(metric_value, rule):
                            return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating alert rule: {e}")
            return False
    
    def _check_threshold_condition(self, value: float, rule: AlertRule) -> bool:
        """Check if a value meets the threshold condition."""
        if rule.threshold_value is None:
            return False
        
        if rule.condition == 'greater_than':
            return value > rule.threshold_value
        elif rule.condition == 'less_than':
            return value < rule.threshold_value
        elif rule.condition == 'equals':
            return abs(value - rule.threshold_value) < 0.001
        
        return False
    
    async def _trigger_alert(self, rule: AlertRule):
        """Trigger an alert based on a rule."""
        try:
            alert_id = str(uuid.uuid4())
            
            alert = Alert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                title=f"Alert: {rule.name}",
                message=f"Alert rule '{rule.name}' has been triggered",
                severity=rule.severity,
                triggered_at=datetime.utcnow(),
                resolved_at=None,
                status='active',
                metadata={
                    'metric_name': rule.metric_name,
                    'condition': rule.condition,
                    'threshold_value': rule.threshold_value
                }
            )
            
            self.active_alerts[alert_id] = alert
            
            # Update rule statistics
            rule.last_triggered = datetime.utcnow()
            rule.trigger_count += 1
            
            # Send notifications
            await self._send_alert_notifications(alert, rule)
            
            self.logger.warning(
                f"Alert triggered: {rule.name}",
                alert_id=alert_id,
                rule_id=rule.rule_id,
                severity=rule.severity.value
            )
            
        except Exception as e:
            self.logger.error(f"Error triggering alert: {e}")
    
    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send alert notifications through configured channels."""
        try:
            for channel in rule.notification_channels:
                if channel == 'email':
                    for recipient in rule.recipients:
                        await self._send_email(
                            to_email=recipient,
                            subject=f"[{alert.severity.value.upper()}] {alert.title}",
                            html_content=f"""
                            <html>
                            <body>
                                <h2>Alert Notification</h2>
                                <p><strong>Alert:</strong> {alert.title}</p>
                                <p><strong>Severity:</strong> {alert.severity.value}</p>
                                <p><strong>Message:</strong> {alert.message}</p>
                                <p><strong>Triggered:</strong> {alert.triggered_at.isoformat()}</p>
                            </body>
                            </html>
                            """
                        )
            
        except Exception as e:
            self.logger.error(f"Error sending alert notifications: {e}")
    
    async def start_system(self):
        """Start the automated reporting system."""
        if self.running:
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        self.logger.info("Automated reporting system started")
    
    async def stop_system(self):
        """Stop the automated reporting system."""
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Automated reporting system stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                now = datetime.utcnow()
                
                # Check scheduled reports
                for schedule_id, schedule in self.report_schedules.items():
                    if schedule.active and schedule.next_run <= now:
                        await self._execute_scheduled_report(schedule)
                
                # Check alert rules
                await self.check_alert_rules()
                
                # Sleep for 1 minute before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)
    
    async def _execute_scheduled_report(self, schedule: ReportSchedule):
        """Execute a scheduled report."""
        try:
            # Calculate date range based on frequency
            end_date = datetime.utcnow()
            if schedule.frequency == ReportFrequency.HOURLY:
                start_date = end_date - timedelta(hours=1)
                next_run = end_date + timedelta(hours=1)
            elif schedule.frequency == ReportFrequency.DAILY:
                start_date = end_date - timedelta(days=1)
                next_run = end_date + timedelta(days=1)
            elif schedule.frequency == ReportFrequency.WEEKLY:
                start_date = end_date - timedelta(weeks=1)
                next_run = end_date + timedelta(weeks=1)
            else:  # MONTHLY
                start_date = end_date - timedelta(days=30)
                next_run = end_date + timedelta(days=30)
            
            # Generate and send report
            report = await self.generate_report(
                report_type=schedule.report_type,
                team_ids=schedule.team_ids,
                date_range=(start_date, end_date),
                prompt_ids=schedule.prompt_ids
            )
            
            if 'error' not in report:
                await self._send_report(report, schedule.recipients)
            
            # Update schedule
            schedule.last_run = end_date
            schedule.next_run = next_run
            
            self.logger.info(
                f"Executed scheduled report: {schedule.name}",
                schedule_id=schedule.schedule_id,
                next_run=next_run.isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Error executing scheduled report: {e}")

# Global instance
automated_reporting_system = AutomatedReportingSystem()