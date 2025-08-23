"""
Real-time Quality Monitoring and Alerting System
Provides continuous monitoring and immediate alerting for data quality issues
"""
import asyncio
import json
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import logging
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from ..models.data_quality_models import (
    QualityRule, QualityReport, QualityAlert, DataAnomaly,
    QualityStatus, Severity
)

logger = logging.getLogger(__name__)

class AlertChannel(Enum):
    """Available alert channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"

@dataclass
class AlertConfig:
    """Alert configuration"""
    channels: List[AlertChannel]
    recipients: List[str]
    severity_threshold: Severity
    cooldown_minutes: int = 15
    escalation_minutes: int = 60

@dataclass
class MonitoringMetrics:
    """Real-time monitoring metrics"""
    total_checks: int
    passed_checks: int
    failed_checks: int
    critical_alerts: int
    active_anomalies: int
    avg_quality_score: float
    last_check_time: datetime

class QualityAlertManager:
    """Manages quality alerts and notifications"""
    
    def __init__(self, db_session: Session, alert_config: AlertConfig = None):
        self.db_session = db_session
        self.alert_config = alert_config or AlertConfig(
            channels=[AlertChannel.DASHBOARD],
            recipients=[],
            severity_threshold=Severity.MEDIUM
        )
        self.logger = logging.getLogger(__name__)
        self._alert_handlers = {
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.SLACK: self._send_slack_alert,
            AlertChannel.WEBHOOK: self._send_webhook_alert,
            AlertChannel.SMS: self._send_sms_alert,
            AlertChannel.DASHBOARD: self._send_dashboard_alert
        }
        
    async def process_quality_report(self, report: QualityReport):
        """Process a quality report and trigger alerts if needed"""
        try:
            # Check if alert is needed
            if self._should_alert(report):
                alert = await self._create_alert(report)
                await self._send_alert(alert)
                
            # Update monitoring metrics
            await self._update_metrics(report)
            
        except Exception as e:
            self.logger.error(f"Error processing quality report: {str(e)}")
    
    def _should_alert(self, report: QualityReport) -> bool:
        """Determine if an alert should be sent for this report"""
        # Don't alert on passed checks
        if report.status == QualityStatus.PASSED:
            return False
        
        # Get the rule to check severity
        rule = self.db_session.query(QualityRule).filter_by(id=report.rule_id).first()
        if not rule:
            return False
        
        # Check severity threshold
        severity_levels = {
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4
        }
        
        if severity_levels.get(rule.severity, 0) < severity_levels.get(self.alert_config.severity_threshold, 0):
            return False
        
        # Check cooldown period
        if self._is_in_cooldown(rule.id):
            return False
        
        return True
    
    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period"""
        cooldown_time = datetime.utcnow() - timedelta(minutes=self.alert_config.cooldown_minutes)
        
        recent_alert = self.db_session.query(QualityAlert).filter(
            and_(
                QualityAlert.rule_id == rule_id,
                QualityAlert.created_at > cooldown_time
            )
        ).first()
        
        return recent_alert is not None
    
    async def _create_alert(self, report: QualityReport) -> QualityAlert:
        """Create a quality alert from a report"""
        rule = self.db_session.query(QualityRule).filter_by(id=report.rule_id).first()
        
        alert = QualityAlert(
            rule_id=report.rule_id,
            quality_report_id=report.id,
            alert_type="quality_failure",
            severity=rule.severity,
            message=self._generate_alert_message(report, rule),
            pipeline_id=rule.target_pipeline_id,
            table_name=rule.target_table,
            column_name=rule.target_column
        )
        
        self.db_session.add(alert)
        self.db_session.commit()
        
        return alert
    
    def _generate_alert_message(self, report: QualityReport, rule: QualityRule) -> str:
        """Generate a human-readable alert message"""
        base_message = f"Quality rule '{rule.name}' failed"
        
        details = []
        if report.score is not None:
            details.append(f"Score: {report.score:.1f}%")
        if report.records_failed:
            details.append(f"Failed records: {report.records_failed}/{report.records_checked}")
        if report.error_message:
            details.append(f"Error: {report.error_message}")
        
        if details:
            return f"{base_message} - {', '.join(details)}"
        
        return base_message
    
    async def _send_alert(self, alert: QualityAlert):
        """Send alert through configured channels"""
        for channel in self.alert_config.channels:
            try:
                handler = self._alert_handlers.get(channel)
                if handler:
                    await handler(alert)
                else:
                    self.logger.warning(f"No handler for alert channel: {channel}")
            except Exception as e:
                self.logger.error(f"Error sending alert via {channel}: {str(e)}")
    
    async def _send_email_alert(self, alert: QualityAlert):
        """Send email alert"""
        if not self.alert_config.recipients:
            return
        
        # This would integrate with your email service
        self.logger.info(f"EMAIL ALERT: {alert.message}")
        
        # Example email implementation (would need SMTP configuration)
        # msg = MimeMultipart()
        # msg['From'] = "alerts@yourcompany.com"
        # msg['To'] = ", ".join(self.alert_config.recipients)
        # msg['Subject'] = f"Data Quality Alert - {alert.severity.value.upper()}"
        # msg.attach(MimeText(alert.message, 'plain'))
    
    async def _send_slack_alert(self, alert: QualityAlert):
        """Send Slack alert"""
        self.logger.info(f"SLACK ALERT: {alert.message}")
        # Would integrate with Slack API
    
    async def _send_webhook_alert(self, alert: QualityAlert):
        """Send webhook alert"""
        self.logger.info(f"WEBHOOK ALERT: {alert.message}")
        # Would send HTTP POST to configured webhook URL
    
    async def _send_sms_alert(self, alert: QualityAlert):
        """Send SMS alert"""
        self.logger.info(f"SMS ALERT: {alert.message}")
        # Would integrate with SMS service (Twilio, etc.)
    
    async def _send_dashboard_alert(self, alert: QualityAlert):
        """Send dashboard alert (real-time notification)"""
        self.logger.info(f"DASHBOARD ALERT: {alert.message}")
        # Would push to real-time dashboard via WebSocket
    
    async def _update_metrics(self, report: QualityReport):
        """Update real-time monitoring metrics"""
        # This would update a metrics cache/dashboard
        pass

class RealTimeQualityMonitor:
    """Real-time quality monitoring orchestrator"""
    
    def __init__(self, db_session: Session, alert_manager: QualityAlertManager):
        self.db_session = db_session
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)
        self._monitoring_tasks = {}
        self._is_running = False
    
    async def start_monitoring(self, pipeline_ids: List[str] = None):
        """Start real-time monitoring for specified pipelines"""
        self._is_running = True
        self.logger.info("Starting real-time quality monitoring")
        
        # Start monitoring tasks
        if pipeline_ids:
            for pipeline_id in pipeline_ids:
                task = asyncio.create_task(self._monitor_pipeline(pipeline_id))
                self._monitoring_tasks[pipeline_id] = task
        else:
            # Monitor all active pipelines
            task = asyncio.create_task(self._monitor_all_pipelines())
            self._monitoring_tasks['all'] = task
        
        # Start alert processing task
        alert_task = asyncio.create_task(self._process_alerts())
        self._monitoring_tasks['alerts'] = alert_task
    
    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        self._is_running = False
        self.logger.info("Stopping real-time quality monitoring")
        
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)
        self._monitoring_tasks.clear()
    
    async def _monitor_pipeline(self, pipeline_id: str):
        """Monitor a specific pipeline for quality issues"""
        while self._is_running:
            try:
                # Check for new quality reports
                recent_reports = self._get_recent_reports(pipeline_id)
                
                for report in recent_reports:
                    await self.alert_manager.process_quality_report(report)
                
                # Check for anomalies
                await self._check_anomalies(pipeline_id)
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring pipeline {pipeline_id}: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _monitor_all_pipelines(self):
        """Monitor all active pipelines"""
        while self._is_running:
            try:
                # Get all active pipelines
                active_pipelines = self._get_active_pipelines()
                
                for pipeline_id in active_pipelines:
                    recent_reports = self._get_recent_reports(pipeline_id)
                    
                    for report in recent_reports:
                        await self.alert_manager.process_quality_report(report)
                
                await asyncio.sleep(60)  # Check every minute for all pipelines
                
            except Exception as e:
                self.logger.error(f"Error monitoring all pipelines: {str(e)}")
                await asyncio.sleep(120)
    
    def _get_recent_reports(self, pipeline_id: str) -> List[QualityReport]:
        """Get recent quality reports for a pipeline"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)
        
        reports = self.db_session.query(QualityReport).join(QualityRule).filter(
            and_(
                QualityRule.target_pipeline_id == pipeline_id,
                QualityReport.check_timestamp > cutoff_time
            )
        ).all()
        
        return reports
    
    def _get_active_pipelines(self) -> List[str]:
        """Get list of active pipeline IDs"""
        # This would query your pipeline system for active pipelines
        # For now, return pipelines that have recent quality rules
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        pipeline_ids = self.db_session.query(QualityRule.target_pipeline_id).filter(
            and_(
                QualityRule.is_active == True,
                QualityRule.updated_at > cutoff_time
            )
        ).distinct().all()
        
        return [pid[0] for pid in pipeline_ids if pid[0]]
    
    async def _check_anomalies(self, pipeline_id: str):
        """Check for new anomalies in pipeline data"""
        # Get recent anomalies
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)
        
        recent_anomalies = self.db_session.query(DataAnomaly).filter(
            and_(
                DataAnomaly.detected_at > cutoff_time,
                DataAnomaly.severity.in_([Severity.HIGH, Severity.CRITICAL])
            )
        ).all()
        
        for anomaly in recent_anomalies:
            await self._handle_anomaly_alert(anomaly)
    
    async def _handle_anomaly_alert(self, anomaly: DataAnomaly):
        """Handle alerts for detected anomalies"""
        alert = QualityAlert(
            alert_type="anomaly_detected",
            severity=anomaly.severity,
            message=f"Anomaly detected in {anomaly.table_name}.{anomaly.column_name}: {anomaly.anomaly_type}",
            table_name=anomaly.table_name,
            column_name=anomaly.column_name
        )
        
        self.db_session.add(alert)
        self.db_session.commit()
        
        await self.alert_manager._send_alert(alert)
    
    async def _process_alerts(self):
        """Process and escalate unacknowledged alerts"""
        while self._is_running:
            try:
                # Find unacknowledged critical alerts older than escalation time
                escalation_time = datetime.utcnow() - timedelta(
                    minutes=self.alert_manager.alert_config.escalation_minutes
                )
                
                unacknowledged_alerts = self.db_session.query(QualityAlert).filter(
                    and_(
                        QualityAlert.is_acknowledged == False,
                        QualityAlert.severity == Severity.CRITICAL,
                        QualityAlert.created_at < escalation_time
                    )
                ).all()
                
                for alert in unacknowledged_alerts:
                    await self._escalate_alert(alert)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error processing alerts: {str(e)}")
                await asyncio.sleep(600)
    
    async def _escalate_alert(self, alert: QualityAlert):
        """Escalate an unacknowledged critical alert"""
        escalated_alert = QualityAlert(
            rule_id=alert.rule_id,
            quality_report_id=alert.quality_report_id,
            alert_type="escalated_alert",
            severity=Severity.CRITICAL,
            message=f"ESCALATED: {alert.message}",
            pipeline_id=alert.pipeline_id,
            table_name=alert.table_name,
            column_name=alert.column_name
        )
        
        self.db_session.add(escalated_alert)
        self.db_session.commit()
        
        await self.alert_manager._send_alert(escalated_alert)
    
    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status"""
        return {
            "is_running": self._is_running,
            "active_tasks": len(self._monitoring_tasks),
            "monitored_pipelines": list(self._monitoring_tasks.keys())
        }
    
    def get_real_time_metrics(self) -> MonitoringMetrics:
        """Get real-time monitoring metrics"""
        # Calculate metrics from recent data
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        recent_reports = self.db_session.query(QualityReport).filter(
            QualityReport.check_timestamp > cutoff_time
        ).all()
        
        total_checks = len(recent_reports)
        passed_checks = sum(1 for r in recent_reports if r.status == QualityStatus.PASSED)
        failed_checks = total_checks - passed_checks
        
        avg_score = sum(r.score or 0 for r in recent_reports) / total_checks if total_checks > 0 else 0
        
        critical_alerts = self.db_session.query(QualityAlert).filter(
            and_(
                QualityAlert.created_at > cutoff_time,
                QualityAlert.severity == Severity.CRITICAL
            )
        ).count()
        
        active_anomalies = self.db_session.query(DataAnomaly).filter(
            and_(
                DataAnomaly.detected_at > cutoff_time,
                DataAnomaly.is_resolved == False
            )
        ).count()
        
        last_check = max([r.check_timestamp for r in recent_reports]) if recent_reports else datetime.utcnow()
        
        return MonitoringMetrics(
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            critical_alerts=critical_alerts,
            active_anomalies=active_anomalies,
            avg_quality_score=avg_score,
            last_check_time=last_check
        )