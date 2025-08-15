"""
Automated Report Generation and Scheduling System
Handles scheduled report generation, distribution, and management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import schedule
import threading
import time
from croniter import croniter

from scrollintel.engines.reporting_engine import ReportingEngine, ReportConfig, ReportFormat, ReportType

logger = logging.getLogger(__name__)

class ScheduleFrequency(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"

class DeliveryMethod(Enum):
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    FILE_SYSTEM = "file_system"

@dataclass
class ScheduleConfig:
    frequency: ScheduleFrequency
    time: str  # HH:MM format or cron expression for custom
    timezone: str = "UTC"
    days_of_week: Optional[List[int]] = None  # 0=Monday, 6=Sunday
    day_of_month: Optional[int] = None
    custom_cron: Optional[str] = None

@dataclass
class DeliveryConfig:
    method: DeliveryMethod
    recipients: List[str]
    subject_template: Optional[str] = None
    body_template: Optional[str] = None
    webhook_url: Optional[str] = None
    channel: Optional[str] = None  # For Slack/Teams
    file_path: Optional[str] = None

@dataclass
class ScheduledReport:
    id: str
    name: str
    description: str
    report_config: ReportConfig
    schedule_config: ScheduleConfig
    delivery_config: DeliveryConfig
    is_active: bool = True
    created_at: datetime = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None

class ReportScheduler:
    """Automated report generation and scheduling system."""
    
    def __init__(self, reporting_engine: ReportingEngine):
        self.reporting_engine = reporting_engine
        self.scheduled_reports: Dict[str, ScheduledReport] = {}
        self.scheduler_thread = None
        self.is_running = False
        self.email_config = self._load_email_config()
        
    def _load_email_config(self) -> Dict[str, str]:
        """Load email configuration."""
        return {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'reports@scrollintel.com',
            'password': 'your_app_password',
            'from_email': 'reports@scrollintel.com'
        }
    
    def create_scheduled_report(self, scheduled_report: ScheduledReport) -> str:
        """Create a new scheduled report."""
        try:
            if not scheduled_report.created_at:
                scheduled_report.created_at = datetime.now()
            
            # Calculate next run time
            scheduled_report.next_run = self._calculate_next_run(scheduled_report.schedule_config)
            
            # Store scheduled report
            self.scheduled_reports[scheduled_report.id] = scheduled_report
            
            logger.info(f"Created scheduled report: {scheduled_report.name}")
            return scheduled_report.id
            
        except Exception as e:
            logger.error(f"Error creating scheduled report: {str(e)}")
            raise
    
    def update_scheduled_report(self, report_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing scheduled report."""
        try:
            if report_id not in self.scheduled_reports:
                raise ValueError(f"Scheduled report not found: {report_id}")
            
            scheduled_report = self.scheduled_reports[report_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(scheduled_report, key):
                    setattr(scheduled_report, key, value)
            
            # Recalculate next run if schedule changed
            if 'schedule_config' in updates:
                scheduled_report.next_run = self._calculate_next_run(scheduled_report.schedule_config)
            
            logger.info(f"Updated scheduled report: {report_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating scheduled report: {str(e)}")
            return False
    
    def delete_scheduled_report(self, report_id: str) -> bool:
        """Delete a scheduled report."""
        try:
            if report_id in self.scheduled_reports:
                del self.scheduled_reports[report_id]
                logger.info(f"Deleted scheduled report: {report_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting scheduled report: {str(e)}")
            return False
    
    def get_scheduled_reports(self) -> List[ScheduledReport]:
        """Get all scheduled reports."""
        return list(self.scheduled_reports.values())
    
    def get_scheduled_report(self, report_id: str) -> Optional[ScheduledReport]:
        """Get a specific scheduled report."""
        return self.scheduled_reports.get(report_id)
    
    def start_scheduler(self):
        """Start the report scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info("Report scheduler started")
    
    def stop_scheduler(self):
        """Stop the report scheduler."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Report scheduler stopped")
    
    def _run_scheduler(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                for scheduled_report in self.scheduled_reports.values():
                    if (scheduled_report.is_active and 
                        scheduled_report.next_run and 
                        current_time >= scheduled_report.next_run):
                        
                        # Run report in background
                        asyncio.create_task(self._execute_scheduled_report(scheduled_report))
                
                # Sleep for 1 minute before checking again
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(60)
    
    async def _execute_scheduled_report(self, scheduled_report: ScheduledReport):
        """Execute a scheduled report."""
        try:
            logger.info(f"Executing scheduled report: {scheduled_report.name}")
            
            # Update run tracking
            scheduled_report.last_run = datetime.now()
            scheduled_report.run_count += 1
            
            # Generate report
            report = await self.reporting_engine.generate_report(scheduled_report.report_config)
            
            # Deliver report
            await self._deliver_report(report, scheduled_report.delivery_config)
            
            # Calculate next run
            scheduled_report.next_run = self._calculate_next_run(scheduled_report.schedule_config)
            
            # Clear any previous errors
            scheduled_report.last_error = None
            
            logger.info(f"Successfully executed scheduled report: {scheduled_report.name}")
            
        except Exception as e:
            error_msg = f"Error executing scheduled report {scheduled_report.name}: {str(e)}"
            logger.error(error_msg)
            
            scheduled_report.error_count += 1
            scheduled_report.last_error = error_msg
            
            # Calculate next run even on error
            scheduled_report.next_run = self._calculate_next_run(scheduled_report.schedule_config)
    
    def _calculate_next_run(self, schedule_config: ScheduleConfig) -> datetime:
        """Calculate the next run time for a scheduled report."""
        now = datetime.now()
        
        if schedule_config.frequency == ScheduleFrequency.HOURLY:
            return now + timedelta(hours=1)
        
        elif schedule_config.frequency == ScheduleFrequency.DAILY:
            # Parse time (HH:MM)
            hour, minute = map(int, schedule_config.time.split(':'))
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If time has passed today, schedule for tomorrow
            if next_run <= now:
                next_run += timedelta(days=1)
            
            return next_run
        
        elif schedule_config.frequency == ScheduleFrequency.WEEKLY:
            # Parse time and day of week
            hour, minute = map(int, schedule_config.time.split(':'))
            target_day = schedule_config.days_of_week[0] if schedule_config.days_of_week else 0
            
            # Calculate days until target day
            days_ahead = target_day - now.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            
            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            return next_run
        
        elif schedule_config.frequency == ScheduleFrequency.MONTHLY:
            # Parse time and day of month
            hour, minute = map(int, schedule_config.time.split(':'))
            target_day = schedule_config.day_of_month or 1
            
            # Start with current month
            next_run = now.replace(day=target_day, hour=hour, minute=minute, second=0, microsecond=0)
            
            # If date has passed this month, move to next month
            if next_run <= now:
                if now.month == 12:
                    next_run = next_run.replace(year=now.year + 1, month=1)
                else:
                    next_run = next_run.replace(month=now.month + 1)
            
            return next_run
        
        elif schedule_config.frequency == ScheduleFrequency.CUSTOM:
            # Use cron expression
            if schedule_config.custom_cron:
                cron = croniter(schedule_config.custom_cron, now)
                return cron.get_next(datetime)
        
        # Default: run in 1 hour
        return now + timedelta(hours=1)
    
    async def _deliver_report(self, report, delivery_config: DeliveryConfig):
        """Deliver a generated report."""
        try:
            if delivery_config.method == DeliveryMethod.EMAIL:
                await self._deliver_via_email(report, delivery_config)
            elif delivery_config.method == DeliveryMethod.SLACK:
                await self._deliver_via_slack(report, delivery_config)
            elif delivery_config.method == DeliveryMethod.TEAMS:
                await self._deliver_via_teams(report, delivery_config)
            elif delivery_config.method == DeliveryMethod.WEBHOOK:
                await self._deliver_via_webhook(report, delivery_config)
            elif delivery_config.method == DeliveryMethod.FILE_SYSTEM:
                await self._deliver_via_filesystem(report, delivery_config)
            
        except Exception as e:
            logger.error(f"Error delivering report: {str(e)}")
            raise
    
    async def _deliver_via_email(self, report, delivery_config: DeliveryConfig):
        """Deliver report via email."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(delivery_config.recipients)
            
            # Subject
            subject = delivery_config.subject_template or f"Automated Report: {report.config.title}"
            msg['Subject'] = subject
            
            # Body
            body = delivery_config.body_template or f"""
            Please find attached the automated report: {report.config.title}
            
            Generated at: {report.generated_at}
            Report ID: {report.id}
            
            Best regards,
            ScrollIntel Analytics Team
            """
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach report file
            if report.file_path:
                with open(report.file_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {report.file_path.split("/")[-1]}'
                )
                msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], delivery_config.recipients, text)
            server.quit()
            
            logger.info(f"Report delivered via email to {len(delivery_config.recipients)} recipients")
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            raise
    
    async def _deliver_via_slack(self, report, delivery_config: DeliveryConfig):
        """Deliver report via Slack."""
        # Placeholder for Slack integration
        logger.info("Slack delivery not implemented yet")
    
    async def _deliver_via_teams(self, report, delivery_config: DeliveryConfig):
        """Deliver report via Microsoft Teams."""
        # Placeholder for Teams integration
        logger.info("Teams delivery not implemented yet")
    
    async def _deliver_via_webhook(self, report, delivery_config: DeliveryConfig):
        """Deliver report via webhook."""
        # Placeholder for webhook delivery
        logger.info("Webhook delivery not implemented yet")
    
    async def _deliver_via_filesystem(self, report, delivery_config: DeliveryConfig):
        """Deliver report to file system."""
        try:
            import shutil
            
            if delivery_config.file_path and report.file_path:
                # Copy report to specified location
                destination = f"{delivery_config.file_path}/{report.file_path.split('/')[-1]}"
                shutil.copy2(report.file_path, destination)
                logger.info(f"Report delivered to file system: {destination}")
            
        except Exception as e:
            logger.error(f"Error delivering to file system: {str(e)}")
            raise
    
    async def run_report_now(self, report_id: str) -> bool:
        """Manually trigger a scheduled report."""
        try:
            scheduled_report = self.scheduled_reports.get(report_id)
            if not scheduled_report:
                raise ValueError(f"Scheduled report not found: {report_id}")
            
            await self._execute_scheduled_report(scheduled_report)
            return True
            
        except Exception as e:
            logger.error(f"Error running report manually: {str(e)}")
            return False
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status and statistics."""
        active_reports = sum(1 for r in self.scheduled_reports.values() if r.is_active)
        total_runs = sum(r.run_count for r in self.scheduled_reports.values())
        total_errors = sum(r.error_count for r in self.scheduled_reports.values())
        
        return {
            'is_running': self.is_running,
            'total_scheduled_reports': len(self.scheduled_reports),
            'active_reports': active_reports,
            'total_runs': total_runs,
            'total_errors': total_errors,
            'error_rate': total_errors / max(total_runs, 1) * 100
        }