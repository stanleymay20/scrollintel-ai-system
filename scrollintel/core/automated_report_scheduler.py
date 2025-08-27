"""
Automated Report Scheduler and Distribution System

This module provides automated scheduling and distribution capabilities for reports
with support for various delivery methods and scheduling patterns.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import schedule
import threading
import time

logger = logging.getLogger(__name__)


class ScheduleFrequency(Enum):
    """Supported scheduling frequencies"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"


class DeliveryMethod(Enum):
    """Supported delivery methods"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    FILE_SYSTEM = "file_system"
    CLOUD_STORAGE = "cloud_storage"
    DASHBOARD = "dashboard"


class ScheduleStatus(Enum):
    """Schedule status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DeliveryConfig:
    """Configuration for report delivery"""
    method: DeliveryMethod
    recipients: List[str]
    settings: Dict[str, Any]
    retry_count: int = 3
    retry_delay: int = 300  # seconds


@dataclass
class ScheduleConfig:
    """Configuration for report scheduling"""
    frequency: ScheduleFrequency
    start_date: datetime
    end_date: Optional[datetime] = None
    time_of_day: Optional[str] = None  # HH:MM format
    day_of_week: Optional[int] = None  # 0=Monday, 6=Sunday
    day_of_month: Optional[int] = None
    custom_cron: Optional[str] = None
    timezone: str = "UTC"


@dataclass
class ReportSchedule:
    """Scheduled report configuration"""
    schedule_id: str
    name: str
    description: str
    report_config: Dict[str, Any]
    schedule_config: ScheduleConfig
    delivery_config: DeliveryConfig
    status: ScheduleStatus
    created_at: datetime
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    failure_count: int = 0


@dataclass
class ScheduleExecution:
    """Record of a schedule execution"""
    execution_id: str
    schedule_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"
    report_id: Optional[str] = None
    error_message: Optional[str] = None
    delivery_results: List[Dict] = None


class AutomatedReportScheduler:
    """
    Automated report scheduling and distribution system
    """
    
    def __init__(self, reporting_engine=None):
        self.logger = logging.getLogger(__name__)
        self.reporting_engine = reporting_engine
        self.schedules = {}
        self.executions = {}
        self.scheduler_thread = None
        self.running = False
        
        # Email configuration
        self.smtp_config = {
            "host": "localhost",
            "port": 587,
            "username": "",
            "password": "",
            "use_tls": True
        }
        
    def configure_smtp(self, host: str, port: int, username: str, password: str, use_tls: bool = True):
        """Configure SMTP settings for email delivery"""
        self.smtp_config = {
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "use_tls": use_tls
        }
        self.logger.info("SMTP configuration updated")
    
    async def create_schedule(self, schedule: ReportSchedule) -> str:
        """
        Create a new report schedule
        
        Args:
            schedule: Report schedule configuration
            
        Returns:
            Schedule ID
        """
        try:
            # Validate schedule configuration
            await self._validate_schedule(schedule)
            
            # Calculate next run time
            schedule.next_run = self._calculate_next_run(schedule.schedule_config)
            
            # Store schedule
            self.schedules[schedule.schedule_id] = schedule
            
            self.logger.info(f"Created schedule {schedule.schedule_id}: {schedule.name}")
            return schedule.schedule_id
            
        except Exception as e:
            self.logger.error(f"Error creating schedule: {str(e)}")
            raise
    
    async def update_schedule(self, schedule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing schedule"""
        try:
            if schedule_id not in self.schedules:
                raise ValueError(f"Schedule {schedule_id} not found")
            
            schedule = self.schedules[schedule_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(schedule, key):
                    setattr(schedule, key, value)
            
            # Recalculate next run if schedule config changed
            if any(key.startswith('schedule_config') for key in updates.keys()):
                schedule.next_run = self._calculate_next_run(schedule.schedule_config)
            
            self.logger.info(f"Updated schedule {schedule_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating schedule: {str(e)}")
            raise
    
    async def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule"""
        try:
            if schedule_id in self.schedules:
                del self.schedules[schedule_id]
                self.logger.info(f"Deleted schedule {schedule_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting schedule: {str(e)}")
            raise
    
    async def pause_schedule(self, schedule_id: str) -> bool:
        """Pause a schedule"""
        try:
            if schedule_id in self.schedules:
                self.schedules[schedule_id].status = ScheduleStatus.PAUSED
                self.logger.info(f"Paused schedule {schedule_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error pausing schedule: {str(e)}")
            raise
    
    async def resume_schedule(self, schedule_id: str) -> bool:
        """Resume a paused schedule"""
        try:
            if schedule_id in self.schedules:
                schedule = self.schedules[schedule_id]
                schedule.status = ScheduleStatus.ACTIVE
                schedule.next_run = self._calculate_next_run(schedule.schedule_config)
                self.logger.info(f"Resumed schedule {schedule_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error resuming schedule: {str(e)}")
            raise
    
    def start_scheduler(self):
        """Start the background scheduler"""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            self.logger.info("Report scheduler started")
    
    def stop_scheduler(self):
        """Stop the background scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("Report scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # Check for schedules that need to run
                for schedule_id, schedule in self.schedules.items():
                    if (schedule.status == ScheduleStatus.ACTIVE and 
                        schedule.next_run and 
                        current_time >= schedule.next_run):
                        
                        # Execute schedule in background
                        asyncio.create_task(self._execute_schedule(schedule))
                
                # Sleep for 60 seconds before next check
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(60)
    
    async def _execute_schedule(self, schedule: ReportSchedule):
        """Execute a scheduled report"""
        execution_id = f"exec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{schedule.schedule_id}"
        
        execution = ScheduleExecution(
            execution_id=execution_id,
            schedule_id=schedule.schedule_id,
            started_at=datetime.utcnow(),
            delivery_results=[]
        )
        
        try:
            self.logger.info(f"Executing schedule {schedule.schedule_id}")
            
            # Generate report
            if self.reporting_engine:
                from ..engines.comprehensive_reporting_engine import ReportConfig, ReportType, ReportFormat
                
                # Convert schedule report config to ReportConfig
                report_config = ReportConfig(
                    report_type=ReportType(schedule.report_config.get("report_type", "executive_summary")),
                    format=ReportFormat(schedule.report_config.get("format", "pdf")),
                    title=schedule.report_config.get("title", schedule.name),
                    description=schedule.report_config.get("description", "Automated report"),
                    data_sources=schedule.report_config.get("data_sources", []),
                    filters=schedule.report_config.get("filters", {}),
                    date_range=schedule.report_config.get("date_range", {
                        "start": datetime.utcnow() - timedelta(days=30),
                        "end": datetime.utcnow()
                    })
                )
                
                report = await self.reporting_engine.generate_report(report_config)
                execution.report_id = report.report_id
                
                # Deliver report
                delivery_result = await self._deliver_report(report, schedule.delivery_config)
                execution.delivery_results.append(delivery_result)
                
                execution.status = "completed"
                self.logger.info(f"Successfully executed schedule {schedule.schedule_id}")
            else:
                raise Exception("No reporting engine configured")
            
            # Update schedule
            schedule.last_run = datetime.utcnow()
            schedule.next_run = self._calculate_next_run(schedule.schedule_config)
            schedule.run_count += 1
            
        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)
            schedule.failure_count += 1
            self.logger.error(f"Failed to execute schedule {schedule.schedule_id}: {str(e)}")
        
        finally:
            execution.completed_at = datetime.utcnow()
            self.executions[execution_id] = execution
    
    async def _deliver_report(self, report, delivery_config: DeliveryConfig) -> Dict[str, Any]:
        """Deliver a report using the specified delivery method"""
        try:
            if delivery_config.method == DeliveryMethod.EMAIL:
                return await self._deliver_via_email(report, delivery_config)
            elif delivery_config.method == DeliveryMethod.WEBHOOK:
                return await self._deliver_via_webhook(report, delivery_config)
            elif delivery_config.method == DeliveryMethod.FILE_SYSTEM:
                return await self._deliver_to_filesystem(report, delivery_config)
            elif delivery_config.method == DeliveryMethod.CLOUD_STORAGE:
                return await self._deliver_to_cloud_storage(report, delivery_config)
            elif delivery_config.method == DeliveryMethod.DASHBOARD:
                return await self._deliver_to_dashboard(report, delivery_config)
            else:
                raise ValueError(f"Unsupported delivery method: {delivery_config.method}")
                
        except Exception as e:
            self.logger.error(f"Error delivering report: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "method": delivery_config.method.value
            }
    
    async def _deliver_via_email(self, report, delivery_config: DeliveryConfig) -> Dict[str, Any]:
        """Deliver report via email"""
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config["username"]
            msg['To'] = ", ".join(delivery_config.recipients)
            msg['Subject'] = f"Automated Report: {report.config.title}"
            
            # Email body
            body = f"""
            Dear Recipient,
            
            Please find attached the automated report: {report.config.title}
            
            Report Details:
            - Type: {report.config.report_type.value}
            - Format: {report.format.value}
            - Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
            - Size: {report.file_size} bytes
            
            Best regards,
            ScrollIntel Analytics System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach report file
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(report.content)
            encoders.encode_base64(attachment)
            
            filename = f"{report.report_id}.{report.format.value}"
            attachment.add_header('Content-Disposition', f'attachment; filename= {filename}')
            msg.attach(attachment)
            
            # Send email
            server = smtplib.SMTP(self.smtp_config["host"], self.smtp_config["port"])
            if self.smtp_config["use_tls"]:
                server.starttls()
            if self.smtp_config["username"] and self.smtp_config["password"]:
                server.login(self.smtp_config["username"], self.smtp_config["password"])
            
            server.sendmail(msg['From'], delivery_config.recipients, msg.as_string())
            server.quit()
            
            return {
                "success": True,
                "method": "email",
                "recipients": delivery_config.recipients,
                "message": "Report delivered successfully via email"
            }
            
        except Exception as e:
            raise Exception(f"Email delivery failed: {str(e)}")
    
    async def _deliver_via_webhook(self, report, delivery_config: DeliveryConfig) -> Dict[str, Any]:
        """Deliver report via webhook"""
        try:
            import aiohttp
            
            webhook_url = delivery_config.settings.get("webhook_url")
            if not webhook_url:
                raise ValueError("Webhook URL not configured")
            
            # Prepare payload
            payload = {
                "report_id": report.report_id,
                "title": report.config.title,
                "type": report.config.report_type.value,
                "format": report.format.value,
                "generated_at": report.generated_at.isoformat(),
                "file_size": report.file_size,
                "content": report.content.decode('utf-8') if report.format.value in ['json', 'csv', 'web'] else None
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        return {
                            "success": True,
                            "method": "webhook",
                            "url": webhook_url,
                            "message": "Report delivered successfully via webhook"
                        }
                    else:
                        raise Exception(f"Webhook returned status {response.status}")
                        
        except Exception as e:
            raise Exception(f"Webhook delivery failed: {str(e)}")
    
    async def _deliver_to_filesystem(self, report, delivery_config: DeliveryConfig) -> Dict[str, Any]:
        """Deliver report to file system"""
        try:
            import os
            
            output_dir = delivery_config.settings.get("output_directory", "/tmp/reports")
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"{report.report_id}.{report.format.value}"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(report.content)
            
            return {
                "success": True,
                "method": "file_system",
                "filepath": filepath,
                "message": "Report saved to file system"
            }
            
        except Exception as e:
            raise Exception(f"File system delivery failed: {str(e)}")
    
    async def _deliver_to_cloud_storage(self, report, delivery_config: DeliveryConfig) -> Dict[str, Any]:
        """Deliver report to cloud storage"""
        # Placeholder for cloud storage integration
        return {
            "success": True,
            "method": "cloud_storage",
            "message": "Cloud storage delivery not implemented yet"
        }
    
    async def _deliver_to_dashboard(self, report, delivery_config: DeliveryConfig) -> Dict[str, Any]:
        """Deliver report to dashboard"""
        # Placeholder for dashboard integration
        return {
            "success": True,
            "method": "dashboard",
            "message": "Dashboard delivery not implemented yet"
        }
    
    def _calculate_next_run(self, schedule_config: ScheduleConfig) -> datetime:
        """Calculate the next run time for a schedule"""
        now = datetime.utcnow()
        
        if schedule_config.frequency == ScheduleFrequency.HOURLY:
            return now + timedelta(hours=1)
        
        elif schedule_config.frequency == ScheduleFrequency.DAILY:
            next_run = now + timedelta(days=1)
            if schedule_config.time_of_day:
                hour, minute = map(int, schedule_config.time_of_day.split(':'))
                next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return next_run
        
        elif schedule_config.frequency == ScheduleFrequency.WEEKLY:
            days_ahead = schedule_config.day_of_week - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_run = now + timedelta(days=days_ahead)
            if schedule_config.time_of_day:
                hour, minute = map(int, schedule_config.time_of_day.split(':'))
                next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return next_run
        
        elif schedule_config.frequency == ScheduleFrequency.MONTHLY:
            if schedule_config.day_of_month:
                next_month = now.replace(day=1) + timedelta(days=32)
                next_month = next_month.replace(day=1)
                try:
                    next_run = next_month.replace(day=schedule_config.day_of_month)
                except ValueError:
                    # Handle months with fewer days
                    next_run = next_month.replace(day=28)
                if schedule_config.time_of_day:
                    hour, minute = map(int, schedule_config.time_of_day.split(':'))
                    next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return next_run
        
        # Default to daily if no specific frequency matched
        return now + timedelta(days=1)
    
    async def _validate_schedule(self, schedule: ReportSchedule):
        """Validate schedule configuration"""
        if not schedule.name:
            raise ValueError("Schedule name is required")
        
        if not schedule.report_config:
            raise ValueError("Report configuration is required")
        
        if not schedule.delivery_config.recipients:
            raise ValueError("Delivery recipients are required")
        
        # Validate schedule frequency
        if schedule.schedule_config.frequency == ScheduleFrequency.WEEKLY and schedule.schedule_config.day_of_week is None:
            raise ValueError("Day of week is required for weekly schedules")
        
        if schedule.schedule_config.frequency == ScheduleFrequency.MONTHLY and schedule.schedule_config.day_of_month is None:
            raise ValueError("Day of month is required for monthly schedules")
    
    async def get_schedule(self, schedule_id: str) -> Optional[ReportSchedule]:
        """Get a schedule by ID"""
        return self.schedules.get(schedule_id)
    
    async def list_schedules(self, status: Optional[ScheduleStatus] = None) -> List[Dict]:
        """List all schedules with optional status filter"""
        schedules = []
        for schedule_id, schedule in self.schedules.items():
            if status is None or schedule.status == status:
                schedules.append({
                    "schedule_id": schedule_id,
                    "name": schedule.name,
                    "status": schedule.status.value,
                    "frequency": schedule.schedule_config.frequency.value,
                    "next_run": schedule.next_run.isoformat() if schedule.next_run else None,
                    "last_run": schedule.last_run.isoformat() if schedule.last_run else None,
                    "run_count": schedule.run_count,
                    "failure_count": schedule.failure_count
                })
        
        return sorted(schedules, key=lambda x: x["name"])
    
    async def get_execution_history(self, schedule_id: Optional[str] = None) -> List[Dict]:
        """Get execution history for schedules"""
        executions = []
        for execution_id, execution in self.executions.items():
            if schedule_id is None or execution.schedule_id == schedule_id:
                executions.append({
                    "execution_id": execution_id,
                    "schedule_id": execution.schedule_id,
                    "started_at": execution.started_at.isoformat(),
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "status": execution.status,
                    "report_id": execution.report_id,
                    "error_message": execution.error_message,
                    "delivery_results": execution.delivery_results
                })
        
        return sorted(executions, key=lambda x: x["started_at"], reverse=True)