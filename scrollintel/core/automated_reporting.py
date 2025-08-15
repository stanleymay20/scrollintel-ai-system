"""
Automated reporting and alerting system for prompt analytics.
Provides scheduled reports, real-time alerts, and notification management.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from dataclasses import dataclass
from enum import Enum

from ..models.analytics_models import (
    AnalyticsReport, AlertRule, PromptMetrics, UsageAnalytics
)
from ..models.database import get_db_session
from .analytics_dashboard import TeamAnalyticsDashboard, InsightsGenerator
from .prompt_analytics import PromptPerformanceTracker

logger = logging.getLogger(__name__)

class ReportFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ReportSchedule:
    """Configuration for scheduled reports."""
    id: str
    name: str
    report_type: str
    frequency: ReportFrequency
    recipients: List[str]
    team_ids: List[str]
    prompt_ids: Optional[List[str]] = None
    active: bool = True
    last_generated: Optional[datetime] = None
    next_generation: Optional[datetime] = None

@dataclass
class NotificationChannel:
    """Configuration for notification channels."""
    type: str  # 'email', 'slack', 'webhook', 'sms'
    config: Dict[str, Any]
    active: bool = True

class AutomatedReportingSystem:
    """Main system for automated reporting and alerting."""
    
    def __init__(self):
        self.dashboard = TeamAnalyticsDashboard()
        self.insights_generator = InsightsGenerator()
        self.performance_tracker = PromptPerformanceTracker()
        self.notification_channels = {}
        self.report_schedules = {}
        self.alert_rules = {}
        self.running = False
    
    async def start_system(self):
        """Start the automated reporting system."""
        if self.running:
            logger.warning("Automated reporting system is already running")
            return
        
        self.running = True
        logger.info("Starting automated reporting system")
        
        # Load configurations
        await self._load_configurations()
        
        # Start background tasks
        asyncio.create_task(self._report_scheduler())
        asyncio.create_task(self._alert_monitor())
        
        logger.info("Automated reporting system started successfully")
    
    async def stop_system(self):
        """Stop the automated reporting system."""
        self.running = False
        logger.info("Automated reporting system stopped")
    
    async def _load_configurations(self):
        """Load report schedules and alert rules from database."""
        try:
            with get_db_session() as db:
                # Load alert rules
                alert_rules = db.query(AlertRule).filter(AlertRule.active == True).all()
                
                for rule in alert_rules:
                    self.alert_rules[rule.id] = {
                        "id": rule.id,
                        "name": rule.name,
                        "rule_type": rule.rule_type,
                        "metric_name": rule.metric_name,
                        "condition": rule.condition,
                        "threshold_value": rule.threshold_value,
                        "severity": rule.severity,
                        "notification_channels": rule.notification_channels,
                        "recipients": rule.recipients,
                        "prompt_ids": rule.prompt_ids,
                        "team_ids": rule.team_ids,
                        "last_triggered": rule.last_triggered,
                        "trigger_count": rule.trigger_count
                    }
                
                logger.info(f"Loaded {len(self.alert_rules)} alert rules")
                
        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
    
    async def _report_scheduler(self):
        """Background task for scheduled report generation."""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # Check for due reports
                for schedule_id, schedule in self.report_schedules.items():
                    if (schedule.active and 
                        schedule.next_generation and 
                        current_time >= schedule.next_generation):
                        
                        await self._generate_scheduled_report(schedule)
                
                # Sleep for 1 minute before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in report scheduler: {str(e)}")
                await asyncio.sleep(60)
    
    async def _alert_monitor(self):
        """Background task for monitoring alert conditions."""
        while self.running:
            try:
                # Check all active alert rules
                for rule_id, rule in self.alert_rules.items():
                    await self._check_alert_rule(rule)
                
                # Sleep for 5 minutes before next check
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in alert monitor: {str(e)}")
                await asyncio.sleep(300)
    
    async def _generate_scheduled_report(self, schedule: ReportSchedule):
        """Generate a scheduled report."""
        try:
            logger.info(f"Generating scheduled report: {schedule.name}")
            
            # Determine date range based on frequency
            end_date = datetime.utcnow()
            if schedule.frequency == ReportFrequency.DAILY:
                start_date = end_date - timedelta(days=1)
            elif schedule.frequency == ReportFrequency.WEEKLY:
                start_date = end_date - timedelta(weeks=1)
            elif schedule.frequency == ReportFrequency.MONTHLY:
                start_date = end_date - timedelta(days=30)
            else:  # QUARTERLY
                start_date = end_date - timedelta(days=90)
            
            # Generate report for each team
            for team_id in schedule.team_ids:
                report_data = await self._create_comprehensive_report(
                    team_id=team_id,
                    report_type=schedule.report_type,
                    date_range=(start_date, end_date),
                    prompt_ids=schedule.prompt_ids
                )
                
                # Save report to database
                report_id = await self._save_report(report_data)
                
                # Send report to recipients
                await self._send_report_notification(
                    report_id=report_id,
                    recipients=schedule.recipients,
                    report_data=report_data
                )
            
            # Update schedule
            schedule.last_generated = datetime.utcnow()
            schedule.next_generation = self._calculate_next_generation_time(
                schedule.frequency, schedule.last_generated
            )
            
            logger.info(f"Successfully generated scheduled report: {schedule.name}")
            
        except Exception as e:
            logger.error(f"Error generating scheduled report {schedule.name}: {str(e)}")
    
    async def _create_comprehensive_report(
        self,
        team_id: str,
        report_type: str,
        date_range: tuple,
        prompt_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create comprehensive report data."""
        try:
            # Get dashboard data
            dashboard_data = await self.dashboard.get_team_dashboard_data(team_id, date_range)
            
            # Generate insights
            insights = await self.insights_generator.generate_insights(team_id, date_range)
            
            # Create report structure
            report_data = {
                "report_type": report_type,
                "team_id": team_id,
                "date_range": {
                    "start": date_range[0].isoformat(),
                    "end": date_range[1].isoformat()
                },
                "generated_at": datetime.utcnow().isoformat(),
                "executive_summary": await self._create_executive_summary(dashboard_data, insights),
                "detailed_metrics": dashboard_data,
                "key_insights": insights,
                "recommendations": await self._generate_report_recommendations(dashboard_data, insights),
                "appendix": {
                    "methodology": "Analytics based on prompt usage metrics and performance data",
                    "data_sources": ["prompt_metrics", "usage_analytics", "user_feedback"]
                }
            }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error creating comprehensive report: {str(e)}")
            return {}
    
    async def _create_executive_summary(
        self,
        dashboard_data: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create executive summary for report."""
        try:
            overview = dashboard_data.get("overview", {})
            
            # Key metrics
            key_metrics = {
                "total_requests": overview.get("total_requests", 0),
                "unique_prompts": overview.get("unique_prompts", 0),
                "avg_accuracy": overview.get("avg_accuracy"),
                "avg_satisfaction": overview.get("avg_satisfaction"),
                "total_cost": overview.get("total_cost", 0)
            }
            
            # Performance status
            performance_status = "good"
            if overview.get("avg_accuracy", 1.0) < 0.7:
                performance_status = "needs_improvement"
            elif overview.get("avg_accuracy", 1.0) < 0.8:
                performance_status = "fair"
            
            # Top insights
            high_priority_insights = [i for i in insights if i.get("priority") == "high"]
            
            summary = {
                "key_metrics": key_metrics,
                "performance_status": performance_status,
                "total_insights": len(insights),
                "high_priority_issues": len(high_priority_insights),
                "cost_efficiency": self._calculate_cost_efficiency(overview),
                "trend_summary": self._summarize_trends(dashboard_data.get("trends", {}))
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating executive summary: {str(e)}")
            return {}
    
    async def _generate_report_recommendations(
        self,
        dashboard_data: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for the report."""
        recommendations = []
        
        # Based on insights
        for insight in insights[:5]:  # Top 5 insights
            if insight.get("actionable", False):
                recommendations.append({
                    "category": insight.get("type", "general"),
                    "priority": insight.get("priority", "medium"),
                    "title": f"Address {insight.get('title', 'Issue')}",
                    "description": insight.get("description", ""),
                    "expected_impact": insight.get("impact", "medium"),
                    "effort_required": "medium"  # Could be calculated based on complexity
                })
        
        # Based on performance metrics
        performance_metrics = dashboard_data.get("performance_metrics", {})
        
        for metric_name, metric_data in performance_metrics.items():
            if metric_data.get("trend") == "declining":
                recommendations.append({
                    "category": "performance",
                    "priority": "high",
                    "title": f"Improve {metric_name.replace('_', ' ').title()}",
                    "description": f"This metric has been declining and needs attention",
                    "expected_impact": "high",
                    "effort_required": "medium"
                })
        
        return recommendations[:10]  # Limit to top 10
    
    async def _save_report(self, report_data: Dict[str, Any]) -> str:
        """Save report to database."""
        try:
            with get_db_session() as db:
                report = AnalyticsReport(
                    report_type=report_data["report_type"],
                    title=f"{report_data['report_type'].title()} Report - {report_data['team_id']}",
                    description=f"Automated {report_data['report_type']} report",
                    team_ids=[report_data["team_id"]],
                    date_range_start=datetime.fromisoformat(report_data["date_range"]["start"]),
                    date_range_end=datetime.fromisoformat(report_data["date_range"]["end"]),
                    summary=report_data["executive_summary"],
                    detailed_data=report_data,
                    recommendations=report_data["recommendations"],
                    status="generated",
                    scheduled=True
                )
                
                db.add(report)
                db.commit()
                
                logger.info(f"Saved report to database: {report.id}")
                return report.id
                
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            raise
    
    async def _send_report_notification(
        self,
        report_id: str,
        recipients: List[str],
        report_data: Dict[str, Any]
    ):
        """Send report notification to recipients."""
        try:
            # Create email content
            subject = f"Prompt Analytics Report - {report_data['team_id']}"
            
            # Create HTML email body
            html_body = await self._create_report_email_html(report_data)
            
            # Send to each recipient
            for recipient in recipients:
                await self._send_email_notification(
                    recipient=recipient,
                    subject=subject,
                    html_body=html_body,
                    report_id=report_id
                )
            
            logger.info(f"Sent report notifications to {len(recipients)} recipients")
            
        except Exception as e:
            logger.error(f"Error sending report notification: {str(e)}")
    
    async def _create_report_email_html(self, report_data: Dict[str, Any]) -> str:
        """Create HTML email body for report."""
        executive_summary = report_data.get("executive_summary", {})
        key_insights = report_data.get("key_insights", [])[:3]  # Top 3 insights
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
                .insight {{ margin: 10px 0; padding: 10px; border-left: 3px solid #007acc; background-color: #f9f9f9; }}
                .priority-high {{ border-left-color: #ff4444; }}
                .priority-medium {{ border-left-color: #ffaa00; }}
                .priority-low {{ border-left-color: #44ff44; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Prompt Analytics Report</h1>
                <p><strong>Team:</strong> {report_data.get('team_id', 'N/A')}</p>
                <p><strong>Period:</strong> {report_data['date_range']['start'][:10]} to {report_data['date_range']['end'][:10]}</p>
                <p><strong>Generated:</strong> {report_data.get('generated_at', '')[:19]}</p>
            </div>
            
            <h2>Executive Summary</h2>
            <div class="metric">
                <strong>Total Requests:</strong> {executive_summary.get('key_metrics', {}).get('total_requests', 0):,}
            </div>
            <div class="metric">
                <strong>Unique Prompts:</strong> {executive_summary.get('key_metrics', {}).get('unique_prompts', 0)}
            </div>
            <div class="metric">
                <strong>Avg Accuracy:</strong> {executive_summary.get('key_metrics', {}).get('avg_accuracy', 0):.1%}
            </div>
            <div class="metric">
                <strong>Total Cost:</strong> ${executive_summary.get('key_metrics', {}).get('total_cost', 0):.2f}
            </div>
            
            <h2>Key Insights</h2>
        """
        
        for insight in key_insights:
            priority_class = f"priority-{insight.get('priority', 'low')}"
            html += f"""
            <div class="insight {priority_class}">
                <h3>{insight.get('title', 'Insight')}</h3>
                <p>{insight.get('description', 'No description available')}</p>
                <p><strong>Priority:</strong> {insight.get('priority', 'low').title()}</p>
            </div>
            """
        
        html += """
            <h2>Next Steps</h2>
            <p>Review the full report in your analytics dashboard for detailed metrics and recommendations.</p>
            
            <p><em>This is an automated report. For questions, contact your analytics team.</em></p>
        </body>
        </html>
        """
        
        return html
    
    async def _check_alert_rule(self, rule: Dict[str, Any]):
        """Check if an alert rule condition is met."""
        try:
            # Get current metric values
            current_values = await self._get_current_metric_values(
                rule["metric_name"],
                rule.get("prompt_ids"),
                rule.get("team_ids")
            )
            
            if not current_values:
                return
            
            # Check condition
            should_trigger = False
            
            if rule["rule_type"] == "threshold":
                should_trigger = self._check_threshold_condition(
                    current_values,
                    rule["condition"],
                    rule["threshold_value"]
                )
            elif rule["rule_type"] == "trend":
                should_trigger = await self._check_trend_condition(
                    rule["metric_name"],
                    rule["trend_direction"],
                    rule.get("prompt_ids"),
                    rule.get("team_ids")
                )
            elif rule["rule_type"] == "anomaly":
                should_trigger = await self._check_anomaly_condition(
                    rule["metric_name"],
                    rule.get("prompt_ids"),
                    rule.get("team_ids")
                )
            
            if should_trigger:
                await self._trigger_alert(rule, current_values)
                
        except Exception as e:
            logger.error(f"Error checking alert rule {rule['id']}: {str(e)}")
    
    async def _get_current_metric_values(
        self,
        metric_name: str,
        prompt_ids: Optional[List[str]] = None,
        team_ids: Optional[List[str]] = None
    ) -> List[float]:
        """Get current values for a metric."""
        try:
            with get_db_session() as db:
                # Get recent metrics (last hour)
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                
                query = db.query(PromptMetrics).filter(
                    PromptMetrics.created_at >= cutoff_time
                )
                
                if prompt_ids:
                    query = query.filter(PromptMetrics.prompt_id.in_(prompt_ids))
                
                if team_ids:
                    query = query.filter(PromptMetrics.team_id.in_(team_ids))
                
                metrics = query.all()
                
                values = [getattr(m, metric_name) for m in metrics if getattr(m, metric_name) is not None]
                return values
                
        except Exception as e:
            logger.error(f"Error getting current metric values: {str(e)}")
            return []
    
    def _check_threshold_condition(
        self,
        values: List[float],
        condition: str,
        threshold: float
    ) -> bool:
        """Check threshold condition."""
        if not values:
            return False
        
        avg_value = sum(values) / len(values)
        
        if condition == "greater_than":
            return avg_value > threshold
        elif condition == "less_than":
            return avg_value < threshold
        elif condition == "equals":
            return abs(avg_value - threshold) < 0.01
        
        return False
    
    async def _check_trend_condition(
        self,
        metric_name: str,
        expected_direction: str,
        prompt_ids: Optional[List[str]] = None,
        team_ids: Optional[List[str]] = None
    ) -> bool:
        """Check trend condition."""
        try:
            # Get historical data for trend analysis
            with get_db_session() as db:
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                
                query = db.query(PromptMetrics).filter(
                    PromptMetrics.created_at >= cutoff_time
                ).order_by(PromptMetrics.created_at)
                
                if prompt_ids:
                    query = query.filter(PromptMetrics.prompt_id.in_(prompt_ids))
                
                if team_ids:
                    query = query.filter(PromptMetrics.team_id.in_(team_ids))
                
                metrics = query.all()
                values = [getattr(m, metric_name) for m in metrics if getattr(m, metric_name) is not None]
                
                if len(values) < 5:  # Need minimum data points
                    return False
                
                # Simple trend detection
                recent_avg = sum(values[-len(values)//3:]) / (len(values)//3)
                older_avg = sum(values[:len(values)//3]) / (len(values)//3)
                
                if expected_direction == "increasing":
                    return recent_avg > older_avg * 1.1  # 10% increase
                elif expected_direction == "decreasing":
                    return recent_avg < older_avg * 0.9  # 10% decrease
                
                return False
                
        except Exception as e:
            logger.error(f"Error checking trend condition: {str(e)}")
            return False
    
    async def _check_anomaly_condition(
        self,
        metric_name: str,
        prompt_ids: Optional[List[str]] = None,
        team_ids: Optional[List[str]] = None
    ) -> bool:
        """Check anomaly condition using statistical methods."""
        try:
            # Get recent data for anomaly detection
            with get_db_session() as db:
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                
                query = db.query(PromptMetrics).filter(
                    PromptMetrics.created_at >= cutoff_time
                )
                
                if prompt_ids:
                    query = query.filter(PromptMetrics.prompt_id.in_(prompt_ids))
                
                if team_ids:
                    query = query.filter(PromptMetrics.team_id.in_(team_ids))
                
                metrics = query.all()
                values = [getattr(m, metric_name) for m in metrics if getattr(m, metric_name) is not None]
                
                if len(values) < 10:  # Need sufficient data
                    return False
                
                # Calculate z-score for recent values
                import numpy as np
                
                mean_val = np.mean(values[:-5])  # Exclude recent values from baseline
                std_val = np.std(values[:-5])
                
                if std_val == 0:
                    return False
                
                recent_values = values[-5:]  # Check last 5 values
                z_scores = [(val - mean_val) / std_val for val in recent_values]
                
                # Check if any recent value is anomalous (z-score > 2)
                return any(abs(z) > 2.0 for z in z_scores)
                
        except Exception as e:
            logger.error(f"Error checking anomaly condition: {str(e)}")
            return False
    
    async def _trigger_alert(self, rule: Dict[str, Any], current_values: List[float]):
        """Trigger an alert."""
        try:
            logger.warning(f"Triggering alert: {rule['name']}")
            
            # Create alert message
            alert_message = {
                "alert_id": rule["id"],
                "alert_name": rule["name"],
                "severity": rule["severity"],
                "metric_name": rule["metric_name"],
                "current_value": sum(current_values) / len(current_values) if current_values else None,
                "threshold": rule.get("threshold_value"),
                "triggered_at": datetime.utcnow().isoformat(),
                "description": f"Alert triggered for {rule['metric_name']}"
            }
            
            # Send notifications through configured channels
            for channel_type in rule["notification_channels"]:
                if channel_type == "email":
                    await self._send_alert_email(rule, alert_message)
                elif channel_type == "slack":
                    await self._send_slack_alert(rule, alert_message)
                elif channel_type == "webhook":
                    await self._send_webhook_alert(rule, alert_message)
            
            # Update alert rule in database
            await self._update_alert_trigger_count(rule["id"])
            
            logger.info(f"Alert {rule['name']} triggered successfully")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {str(e)}")
    
    async def _send_alert_email(self, rule: Dict[str, Any], alert_message: Dict[str, Any]):
        """Send alert via email."""
        try:
            subject = f"🚨 Alert: {rule['name']} - {rule['severity'].upper()}"
            
            html_body = f"""
            <html>
            <body>
                <h2 style="color: {'red' if rule['severity'] == 'critical' else 'orange'};">
                    Alert Triggered: {rule['name']}
                </h2>
                <p><strong>Severity:</strong> {rule['severity'].upper()}</p>
                <p><strong>Metric:</strong> {rule['metric_name']}</p>
                <p><strong>Current Value:</strong> {alert_message.get('current_value', 'N/A')}</p>
                <p><strong>Threshold:</strong> {alert_message.get('threshold', 'N/A')}</p>
                <p><strong>Triggered At:</strong> {alert_message['triggered_at']}</p>
                
                <h3>Description</h3>
                <p>{alert_message['description']}</p>
                
                <p><em>Please review your analytics dashboard for more details.</em></p>
            </body>
            </html>
            """
            
            for recipient in rule["recipients"]:
                await self._send_email_notification(
                    recipient=recipient,
                    subject=subject,
                    html_body=html_body
                )
                
        except Exception as e:
            logger.error(f"Error sending alert email: {str(e)}")
    
    async def _send_email_notification(
        self,
        recipient: str,
        subject: str,
        html_body: str,
        report_id: Optional[str] = None
    ):
        """Send email notification."""
        try:
            # This is a simplified implementation
            # In production, you'd use a proper email service like SendGrid, SES, etc.
            logger.info(f"Would send email to {recipient}: {subject}")
            
            # Placeholder for actual email sending logic
            # msg = MIMEMultipart('alternative')
            # msg['Subject'] = subject
            # msg['From'] = "alerts@yourcompany.com"
            # msg['To'] = recipient
            # 
            # html_part = MIMEText(html_body, 'html')
            # msg.attach(html_part)
            # 
            # with smtplib.SMTP('localhost') as server:
            #     server.send_message(msg)
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
    
    async def _send_slack_alert(self, rule: Dict[str, Any], alert_message: Dict[str, Any]):
        """Send alert to Slack."""
        try:
            # Placeholder for Slack integration
            logger.info(f"Would send Slack alert: {rule['name']}")
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {str(e)}")
    
    async def _send_webhook_alert(self, rule: Dict[str, Any], alert_message: Dict[str, Any]):
        """Send alert via webhook."""
        try:
            # Placeholder for webhook integration
            logger.info(f"Would send webhook alert: {rule['name']}")
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {str(e)}")
    
    async def _update_alert_trigger_count(self, rule_id: str):
        """Update alert trigger count in database."""
        try:
            with get_db_session() as db:
                alert_rule = db.query(AlertRule).filter(AlertRule.id == rule_id).first()
                if alert_rule:
                    alert_rule.last_triggered = datetime.utcnow()
                    alert_rule.trigger_count += 1
                    db.commit()
                    
        except Exception as e:
            logger.error(f"Error updating alert trigger count: {str(e)}")
    
    def _calculate_next_generation_time(
        self,
        frequency: ReportFrequency,
        last_generated: datetime
    ) -> datetime:
        """Calculate next report generation time."""
        if frequency == ReportFrequency.DAILY:
            return last_generated + timedelta(days=1)
        elif frequency == ReportFrequency.WEEKLY:
            return last_generated + timedelta(weeks=1)
        elif frequency == ReportFrequency.MONTHLY:
            return last_generated + timedelta(days=30)
        else:  # QUARTERLY
            return last_generated + timedelta(days=90)
    
    def _calculate_cost_efficiency(self, overview: Dict[str, Any]) -> str:
        """Calculate cost efficiency rating."""
        cost_per_request = overview.get("cost_per_request", 0)
        
        if cost_per_request < 0.001:
            return "excellent"
        elif cost_per_request < 0.005:
            return "good"
        elif cost_per_request < 0.01:
            return "fair"
        else:
            return "needs_improvement"
    
    def _summarize_trends(self, trends: Dict[str, Any]) -> str:
        """Summarize trend data."""
        if not trends:
            return "insufficient_data"
        
        improving_count = sum(1 for trend in trends.values() 
                            if trend.get("trend_direction") == "improving")
        declining_count = sum(1 for trend in trends.values() 
                            if trend.get("trend_direction") == "declining")
        
        if improving_count > declining_count:
            return "mostly_improving"
        elif declining_count > improving_count:
            return "mostly_declining"
        else:
            return "mixed"
    
    # Public API methods
    
    async def create_report_schedule(
        self,
        name: str,
        report_type: str,
        frequency: ReportFrequency,
        recipients: List[str],
        team_ids: List[str],
        prompt_ids: Optional[List[str]] = None
    ) -> str:
        """Create a new report schedule."""
        schedule_id = f"schedule_{datetime.utcnow().timestamp()}"
        
        schedule = ReportSchedule(
            id=schedule_id,
            name=name,
            report_type=report_type,
            frequency=frequency,
            recipients=recipients,
            team_ids=team_ids,
            prompt_ids=prompt_ids,
            next_generation=datetime.utcnow() + timedelta(hours=1)  # Start in 1 hour
        )
        
        self.report_schedules[schedule_id] = schedule
        
        logger.info(f"Created report schedule: {name}")
        return schedule_id
    
    async def create_alert_rule(
        self,
        name: str,
        rule_type: str,
        metric_name: str,
        condition: str,
        threshold_value: Optional[float] = None,
        severity: str = "medium",
        notification_channels: List[str] = None,
        recipients: List[str] = None,
        prompt_ids: Optional[List[str]] = None,
        team_ids: Optional[List[str]] = None
    ) -> str:
        """Create a new alert rule."""
        try:
            with get_db_session() as db:
                alert_rule = AlertRule(
                    name=name,
                    rule_type=rule_type,
                    metric_name=metric_name,
                    condition=condition,
                    threshold_value=threshold_value,
                    severity=severity,
                    notification_channels=notification_channels or ["email"],
                    recipients=recipients or [],
                    prompt_ids=prompt_ids,
                    team_ids=team_ids
                )
                
                db.add(alert_rule)
                db.commit()
                
                # Add to runtime cache
                self.alert_rules[alert_rule.id] = {
                    "id": alert_rule.id,
                    "name": name,
                    "rule_type": rule_type,
                    "metric_name": metric_name,
                    "condition": condition,
                    "threshold_value": threshold_value,
                    "severity": severity,
                    "notification_channels": notification_channels or ["email"],
                    "recipients": recipients or [],
                    "prompt_ids": prompt_ids,
                    "team_ids": team_ids,
                    "last_triggered": None,
                    "trigger_count": 0
                }
                
                logger.info(f"Created alert rule: {name}")
                return alert_rule.id
                
        except Exception as e:
            logger.error(f"Error creating alert rule: {str(e)}")
            raise
    
    async def generate_ad_hoc_report(
        self,
        team_id: str,
        report_type: str,
        date_range: tuple,
        recipients: List[str]
    ) -> str:
        """Generate an ad-hoc report."""
        try:
            report_data = await self._create_comprehensive_report(
                team_id=team_id,
                report_type=report_type,
                date_range=date_range
            )
            
            report_id = await self._save_report(report_data)
            
            await self._send_report_notification(
                report_id=report_id,
                recipients=recipients,
                report_data=report_data
            )
            
            logger.info(f"Generated ad-hoc report: {report_id}")
            return report_id
            
        except Exception as e:
            logger.error(f"Error generating ad-hoc report: {str(e)}")
            raise