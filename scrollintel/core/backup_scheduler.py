"""
Backup Scheduler
Handles automated daily backups and scheduling
"""

import os
import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Any

from scrollintel.core.backup_system import backup_system
from scrollintel.core.config import get_settings

# Optional APScheduler import
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_APSCHEDULER = True
except ImportError:
    AsyncIOScheduler = None
    CronTrigger = None
    HAS_APSCHEDULER = False

logger = logging.getLogger(__name__)

class BackupScheduler:
    """Automated backup scheduler"""
    
    def __init__(self):
        self.settings = get_settings()
        self.scheduler = AsyncIOScheduler() if HAS_APSCHEDULER else None
        self.is_running = False
    
    async def start(self):
        """Start the backup scheduler"""
        if self.is_running:
            return
        
        if not HAS_APSCHEDULER:
            logger.warning("Scheduler not available: APScheduler not installed")
            return
        
        try:
            # Schedule daily database backup at 2 AM
            self.scheduler.add_job(
                self.run_daily_database_backup,
                CronTrigger(hour=2, minute=0),
                id='daily_database_backup',
                name='Daily Database Backup',
                replace_existing=True
            )
            
            # Schedule daily file backup at 3 AM
            self.scheduler.add_job(
                self.run_daily_file_backup,
                CronTrigger(hour=3, minute=0),
                id='daily_file_backup',
                name='Daily File Backup',
                replace_existing=True
            )
            
            # Schedule weekly backup cleanup on Sundays at 4 AM
            self.scheduler.add_job(
                self.run_backup_cleanup,
                CronTrigger(day_of_week=6, hour=4, minute=0),
                id='weekly_backup_cleanup',
                name='Weekly Backup Cleanup',
                replace_existing=True
            )
            
            # Schedule backup verification every 6 hours
            self.scheduler.add_job(
                self.run_backup_verification,
                CronTrigger(hour='*/6'),
                id='backup_verification',
                name='Backup Verification',
                replace_existing=True
            )
            
            self.scheduler.start()
            self.is_running = True
            logger.info("Backup scheduler started")
            
        except Exception as e:
            logger.error(f"Failed to start backup scheduler: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the backup scheduler"""
        if not self.is_running:
            return
        
        try:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Backup scheduler stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop backup scheduler: {str(e)}")
    
    async def run_daily_database_backup(self):
        """Run daily database backup"""
        try:
            logger.info("Starting daily database backup")
            
            # Create database backup
            backup_metadata = await backup_system.create_database_backup()
            
            # Verify backup integrity
            if await backup_system.verify_backup_integrity(backup_metadata):
                logger.info("Database backup integrity verified")
                
                # Replicate to cloud if configured
                backup_metadata = await backup_system.replicate_to_cloud(backup_metadata)
                
                if backup_metadata.get("cloud_replicated"):
                    logger.info("Database backup replicated to cloud")
                
                logger.info("Daily database backup completed successfully")
            else:
                logger.error("Database backup integrity check failed")
                
        except Exception as e:
            logger.error(f"Daily database backup failed: {str(e)}")
    
    async def run_daily_file_backup(self):
        """Run daily file backup"""
        try:
            logger.info("Starting daily file backup")
            
            # Define directories to backup
            directories_to_backup = [
                "data",
                "logs",
                "generated_content",
                "generated_reports",
                "storage"
            ]
            
            # Filter existing directories
            existing_dirs = [d for d in directories_to_backup if os.path.exists(d)]
            
            if not existing_dirs:
                logger.warning("No directories found to backup")
                return
            
            # Create file backup
            backup_metadata = await backup_system.create_file_backup(existing_dirs)
            
            # Verify backup integrity
            if await backup_system.verify_backup_integrity(backup_metadata):
                logger.info("File backup integrity verified")
                
                # Replicate to cloud if configured
                backup_metadata = await backup_system.replicate_to_cloud(backup_metadata)
                
                if backup_metadata.get("cloud_replicated"):
                    logger.info("File backup replicated to cloud")
                
                logger.info("Daily file backup completed successfully")
            else:
                logger.error("File backup integrity check failed")
                
        except Exception as e:
            logger.error(f"Daily file backup failed: {str(e)}")
    
    async def run_backup_cleanup(self):
        """Run weekly backup cleanup"""
        try:
            logger.info("Starting backup cleanup")
            
            # Clean up backups older than 30 days
            retention_days = getattr(self.settings, 'backup_retention_days', 30)
            cleaned_count = await backup_system.cleanup_old_backups(retention_days)
            
            logger.info(f"Backup cleanup completed: {cleaned_count} old backups removed")
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {str(e)}")
    
    async def run_backup_verification(self):
        """Run backup verification"""
        try:
            logger.info("Starting backup verification")
            
            # Get recent backups (last 7 days)
            backups = await backup_system.list_backups()
            recent_backups = []
            
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for backup in backups:
                backup_time = datetime.fromisoformat(backup["timestamp"].replace("_", "T"))
                if backup_time >= cutoff_date:
                    recent_backups.append(backup)
            
            verified_count = 0
            failed_count = 0
            
            for backup in recent_backups:
                if await backup_system.verify_backup_integrity(backup):
                    verified_count += 1
                else:
                    failed_count += 1
                    logger.error(f"Backup verification failed: {backup['filename']}")
            
            logger.info(f"Backup verification completed: {verified_count} verified, {failed_count} failed")
            
        except Exception as e:
            logger.error(f"Backup verification failed: {str(e)}")
    
    async def trigger_manual_backup(self, backup_type: str = "both") -> Dict[str, Any]:
        """Trigger manual backup"""
        try:
            results = {}
            
            if backup_type in ["database", "both"]:
                logger.info("Starting manual database backup")
                db_backup = await backup_system.create_database_backup()
                
                if await backup_system.verify_backup_integrity(db_backup):
                    db_backup = await backup_system.replicate_to_cloud(db_backup)
                    results["database"] = db_backup
                    logger.info("Manual database backup completed")
                else:
                    results["database"] = {"status": "failed", "error": "Integrity check failed"}
            
            if backup_type in ["files", "both"]:
                logger.info("Starting manual file backup")
                directories = ["data", "logs", "generated_content", "generated_reports", "storage"]
                existing_dirs = [d for d in directories if os.path.exists(d)]
                
                if existing_dirs:
                    file_backup = await backup_system.create_file_backup(existing_dirs)
                    
                    if await backup_system.verify_backup_integrity(file_backup):
                        file_backup = await backup_system.replicate_to_cloud(file_backup)
                        results["files"] = file_backup
                        logger.info("Manual file backup completed")
                    else:
                        results["files"] = {"status": "failed", "error": "Integrity check failed"}
                else:
                    results["files"] = {"status": "skipped", "error": "No directories to backup"}
            
            return results
            
        except Exception as e:
            logger.error(f"Manual backup failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def get_next_backup_times(self) -> Dict[str, str]:
        """Get next scheduled backup times"""
        try:
            jobs = {}
            
            for job in self.scheduler.get_jobs():
                if job.next_run_time:
                    jobs[job.id] = job.next_run_time.isoformat()
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to get next backup times: {str(e)}")
            return {}

# Global backup scheduler instance
backup_scheduler = BackupScheduler()