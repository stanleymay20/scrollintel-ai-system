"""
Automated Report Scheduler for Advanced Analytics Dashboard
Handles scheduled report generation and distribution
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import schedule
import threading
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import logging

from scrollintel.engines.comprehensive_reporting_engine import (
    ComprehensiveReportingEngine, ReportConfig, ReportFormat, ReportType
)

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
    CLOUD_STORAGE = "cloud_storage"

@dataclass
class ScheduledReportConfig:
    id: str
    name: str
    description: str
    report_config: ReportConfig
    frequency: ScheduleFrequency
    schedule_time: str  # e.g., "09:00" for daily, "monday" for weekly
    delivery_methods: List[DeliveryMethod]
    delivery_config: Dict[str, Any]
    data_source_config: Dict[str, Any]
    is_active: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class ReportDelivery:
    id: str
    scheduled_report_id: str
    report_id: str
    delivery_method: DeliveryMethod
    status: str
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    recipients: List[str] = None

class AutomatedReportScheduler:
    """Automated report scheduling and delivery system"""
    
    def __init__(self, reporting_engine: ComprehensiveReportingEngine):
        self.reporting_engine = reporting_engine
        self.scheduled_reports: Dict[str, ScheduledReportConfig] = {}
        self.delivery_history: List[ReportDelivery] = []
        self.is_running = False
        self.scheduler_thread = None
        self.data_fetchers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
        
        # Email configuration
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': '',
            'password': '',
            'use_tls': True
        }
    
    def add_scheduled_report(self, config: ScheduledReportConfig) -> bool:
        """Add a new scheduled report"""
        try:
            config.created_at = datetime.now()
            config.updated_at = datetime.now()
            config.next_run = self._calculate_next_run(config)
            
            self.scheduled_reports[config.id] = config
            self._schedule_report(config)
            
            self.logger.info(f"Added scheduled report: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add scheduled report: {str(e)}")
            return False
    
    def update_scheduled_report(self, report_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing scheduled report"""
        try:
            if report_id not in self.scheduled_reports:
                return False
            
            config = self.scheduled_reports[report_id]
            
            # Update configuration
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            config.updated_at = datetime.now()
            config.next_run = self._calculate_next_run(config)
            
            # Reschedule
            self._unschedule_report(report_id)
            self._schedule_report(config)
            
            self.logger.info(f"Updated scheduled report: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update scheduled report: {str(e)}")
            return False
    
    def remove_scheduled_report(self, report_id: str) -> bool:
        """Remove a scheduled report"""
        try:
            if report_id in self.scheduled_reports:
                self._unschedule_report(report_id)
                del self.scheduled_reports[report_id]
                self.logger.info(f"Removed scheduled report: {report_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove scheduled report: {str(e)}")
            return False
    
    def start_scheduler(self):
        """Start the report scheduler"""
        if not self.is_running:
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            self.logger.info("Report scheduler started")
    
    def stop_scheduler(self):
        """Stop the report scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        self.logger.info("Report scheduler stopped")
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Scheduler error: {str(e)}")
                time.sleep(60)
    
    def _schedule_report(self, config: ScheduledReportConfig):
        """Schedule a report based on its configuration"""
        if not config.is_active:
            return
        
        job_func = lambda: self._generate_and_deliver_report(config.id)
        
        if config.frequency == ScheduleFrequency.HOURLY:
            schedule.every().hour.at(config.schedule_time).do(job_func).tag(config.id)
        elif config.frequency == ScheduleFrequency.DAILY:
            schedule.every().day.at(config.schedule_time).do(job_func).tag(config.id)
        elif config.frequency == ScheduleFrequency.WEEKLY:
            getattr(schedule.every(), config.schedule_time.lower()).at("09:00").do(job_func).tag(config.id)
        elif config.frequency == ScheduleFrequency.MONTHLY:
            # For monthly, we'll check daily and run on the first day of the month
            schedule.every().day.at("09:00").do(
                lambda: self._check_monthly_schedule(config.id)
            ).tag(config.id)
        elif config.frequency == ScheduleFrequency.QUARTERLY:
            # For quarterly, we'll check daily and run on the first day of the quarter
            schedule.every().day.at("09:00").do(
                lambda: self._check_quarterly_schedule(config.id)
            ).tag(config.id)
    
    def _unschedule_report(self, report_id: str):
        """Remove scheduled jobs for a report"""
        schedule.clear(report_id)
    
    def _calculate_next_run(self, config: ScheduledReportConfig) -> datetime:
        """Calculate the next run time for a scheduled report"""
        now = datetime.now()
        
        if config.frequency == ScheduleFrequency.HOURLY:
            return now.replace(minute=int(config.schedule_time.split(':')[1]), second=0, microsecond=0) + timedelta(hours=1)
        elif config.frequency == ScheduleFrequency.DAILY:
            time_parts = config.schedule_time.split(':')
            next_run = now.replace(hour=int(time_parts[0]), minute=int(time_parts[1]), second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        elif config.frequency == ScheduleFrequency.WEEKLY:
            # Calculate next occurrence of the specified day
            days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            target_day = days.index(config.schedule_time.lower())
            days_ahead = target_day - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return now + timedelta(days=days_ahead)
        elif config.frequency == ScheduleFrequency.MONTHLY:
            # First day of next month
            if now.month == 12:
                return datetime(now.year + 1, 1, 1, 9, 0)
            else:
                return datetime(now.year, now.month + 1, 1, 9, 0)
        elif config.frequency == ScheduleFrequency.QUARTERLY:
            # First day of next quarter
            quarter_months = [1, 4, 7, 10]
            current_quarter = (now.month - 1) // 3
            next_quarter_month = quarter_months[(current_quarter + 1) % 4]
            if next_quarter_month <= now.month:
                return datetime(now.year + 1, next_quarter_month, 1, 9, 0)
            else:
                return datetime(now.year, next_quarter_month, 1, 9, 0)
        
        return now + timedelta(days=1)
    
    def _check_monthly_schedule(self, report_id: str):
        """Check if it's time to run a monthly report"""
        if datetime.now().day == 1:
            self._generate_and_deliver_report(report_id)
    
    def _check_quarterly_schedule(self, report_id: str):
        """Check if it's time to run a quarterly report"""
        now = datetime.now()
        if now.day == 1 and now.month in [1, 4, 7, 10]:
            self._generate_and_deliver_report(report_id)
    
    def _generate_and_deliver_report(self, report_id: str):
        """Generate and deliver a scheduled report"""
        try:
            if report_id not in self.scheduled_reports:
                return
            
            config = self.scheduled_reports[report_id]
            config.last_run = datetime.now()
            config.next_run = self._calculate_next_run(config)
            
            # Fetch data
            data = self._fetch_report_data(config.data_source_config)
            
            # Generate report
            report = self.reporting_engine.generate_report(config.report_config, data)
            
            # Deliver report
            for delivery_method in config.delivery_methods:
                delivery = ReportDelivery(
                    id=f"delivery_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    scheduled_report_id=report_id,
                    report_id=report.id,
                    delivery_method=delivery_method,
                    status="pending",
                    recipients=config.delivery_config.get('recipients', [])
                )
                
                success = self._deliver_report(report, delivery_method, config.delivery_config)
                
                delivery.status = "delivered" if success else "failed"
                delivery.delivered_at = datetime.now() if success else None
                
                self.delivery_history.append(delivery)
            
            self.logger.info(f"Generated and delivered report: {config.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate/deliver report {report_id}: {str(e)}")
    
    def _fetch_report_data(self, data_source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data for report generation"""
        data = {}
        
        for source_name, source_config in data_source_config.items():
            try:
                if source_name in self.data_fetchers:
                    data[source_name] = self.data_fetchers[source_name](source_config)
                else:
                    # Default data fetching logic
                    data[source_name] = self._default_data_fetch(source_config)
            except Exception as e:
                self.logger.error(f"Failed to fetch data from {source_name}: {str(e)}")
                data[source_name] = {}
        
        return data
    
    def _default_data_fetch(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Default data fetching implementation"""
        # This would typically connect to databases, APIs, etc.
        return {
            'metrics': {
                'total_users': 1000,
                'revenue': 50000,
                'conversion_rate': 0.15
            },
            'performance': {
                'uptime': '99.9%',
                'response_time': '150ms',
                'throughput': '1000 req/s'
            }
        }
    
    def _deliver_report(self, report, delivery_method: DeliveryMethod, delivery_config: Dict[str, Any]) -> bool:
        """Deliver report using specified method"""
        try:
            if delivery_method == DeliveryMethod.EMAIL:
                return self._deliver_via_email(report, delivery_config)
            elif delivery_method == DeliveryMethod.SLACK:
                return self._deliver_via_slack(report, delivery_config)
            elif delivery_method == DeliveryMethod.TEAMS:
                return self._deliver_via_teams(report, delivery_config)
            elif delivery_method == DeliveryMethod.WEBHOOK:
                return self._deliver_via_webhook(report, delivery_config)
            elif delivery_method == DeliveryMethod.FILE_SYSTEM:
                return self._deliver_to_file_system(report, delivery_config)
            elif delivery_method == DeliveryMethod.CLOUD_STORAGE:
                return self._deliver_to_cloud_storage(report, delivery_config)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Delivery failed via {delivery_method.value}: {str(e)}")
            return False
    
    def _deliver_via_email(self, report, config: Dict[str, Any]) -> bool:
        """Deliver report via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = ', '.join(config['recipients'])
            msg['Subject'] = f"Scheduled Report: {report.config.title}"
            
            # Email body
            body = f"""
            Dear Recipient,
            
            Please find attached the scheduled report: {report.config.title}
            
            Report Details:
            - Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
            - Type: {report.config.report_type.value}
            - Format: {report.config.format.value}
            
            Best regards,
            ScrollIntel Analytics Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach report file
            if report.file_data:
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(report.file_data)
                encoders.encode_base64(attachment)
                
                filename = f"{report.id}.{report.config.format.value}"
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename}'
                )
                msg.attach(attachment)
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            if self.email_config['use_tls']:
                server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Email delivery failed: {str(e)}")
            return False
    
    def _deliver_via_slack(self, report, config: Dict[str, Any]) -> bool:
        """Deliver report via Slack"""
        # Implementation would use Slack API
        self.logger.info("Slack delivery not implemented yet")
        return True
    
    def _deliver_via_teams(self, report, config: Dict[str, Any]) -> bool:
        """Deliver report via Microsoft Teams"""
        # Implementation would use Teams API
        self.logger.info("Teams delivery not implemented yet")
        return True
    
    def _deliver_via_webhook(self, report, config: Dict[str, Any]) -> bool:
        """Deliver report via webhook"""
        # Implementation would POST to webhook URL
        self.logger.info("Webhook delivery not implemented yet")
        return True
    
    def _deliver_to_file_system(self, report, config: Dict[str, Any]) -> bool:
        """Save report to file system"""
        try:
            file_path = config.get('path', '/tmp/reports')
            filename = f"{report.id}.{report.config.format.value}"
            full_path = f"{file_path}/{filename}"
            
            with open(full_path, 'wb') as f:
                f.write(report.file_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"File system delivery failed: {str(e)}")
            return False
    
    def _deliver_to_cloud_storage(self, report, config: Dict[str, Any]) -> bool:
        """Upload report to cloud storage"""
        # Implementation would use cloud storage APIs (AWS S3, Azure Blob, etc.)
        self.logger.info("Cloud storage delivery not implemented yet")
        return True
    
    def register_data_fetcher(self, source_name: str, fetcher_func: Callable):
        """Register a custom data fetcher function"""
        self.data_fetchers[source_name] = fetcher_func
    
    def get_scheduled_reports(self) -> List[ScheduledReportConfig]:
        """Get all scheduled reports"""
        return list(self.scheduled_reports.values())
    
    def get_delivery_history(self, report_id: Optional[str] = None) -> List[ReportDelivery]:
        """Get delivery history"""
        if report_id:
            return [d for d in self.delivery_history if d.scheduled_report_id == report_id]
        return self.delivery_history
    
    def get_report_status(self, report_id: str) -> Dict[str, Any]:
        """Get status of a scheduled report"""
        if report_id not in self.scheduled_reports:
            return {}
        
        config = self.scheduled_reports[report_id]
        recent_deliveries = [d for d in self.delivery_history if d.scheduled_report_id == report_id]
        
        return {
            'id': report_id,
            'name': config.name,
            'is_active': config.is_active,
            'frequency': config.frequency.value,
            'last_run': config.last_run,
            'next_run': config.next_run,
            'total_deliveries': len(recent_deliveries),
            'successful_deliveries': len([d for d in recent_deliveries if d.status == 'delivered']),
            'failed_deliveries': len([d for d in recent_deliveries if d.status == 'failed'])
        }
    
    def configure_email(self, smtp_server: str, smtp_port: int, username: str, password: str, use_tls: bool = True):
        """Configure email settings"""
        self.email_config = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'use_tls': use_tls
        }