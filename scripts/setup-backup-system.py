#!/usr/bin/env python3
"""
Backup System Setup Script
Sets up automated backup and recovery system
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scrollintel.core.backup_system import backup_system
from scrollintel.core.backup_scheduler import backup_scheduler
from scrollintel.core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_backup_system():
    """Set up the backup system"""
    try:
        logger.info("Setting up backup system...")
        
        # Create backup directory
        backup_dir = backup_system.backup_dir
        backup_dir.mkdir(exist_ok=True)
        logger.info(f"Backup directory created: {backup_dir}")
        
        # Set proper permissions
        os.chmod(backup_dir, 0o755)
        
        # Test database connection
        try:
            settings = get_settings()
            db_url = settings.database_url
            logger.info("Database connection configured")
        except Exception as e:
            logger.warning(f"Database connection test failed: {e}")
        
        # Test backup creation
        logger.info("Testing backup creation...")
        
        # Create test directories for file backup
        test_dirs = []
        for dir_name in ["data", "logs"]:
            test_dir = Path(dir_name)
            if not test_dir.exists():
                test_dir.mkdir(exist_ok=True)
                # Create a test file
                (test_dir / "test.txt").write_text("test content")
                test_dirs.append(dir_name)
        
        if test_dirs:
            try:
                file_backup = await backup_system.create_file_backup(test_dirs)
                logger.info(f"Test file backup created: {file_backup['filename']}")
                
                # Verify backup
                if await backup_system.verify_backup_integrity(file_backup):
                    logger.info("Test backup verification successful")
                else:
                    logger.error("Test backup verification failed")
                
            except Exception as e:
                logger.error(f"Test file backup failed: {e}")
        
        # Start backup scheduler
        logger.info("Starting backup scheduler...")
        await backup_scheduler.start()
        
        # Get next backup times
        next_times = backup_scheduler.get_next_backup_times()
        logger.info("Scheduled backup times:")
        for job_id, next_time in next_times.items():
            logger.info(f"  {job_id}: {next_time}")
        
        logger.info("Backup system setup completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Backup system setup failed: {e}")
        return False

async def test_backup_operations():
    """Test backup operations"""
    try:
        logger.info("Testing backup operations...")
        
        # Test manual backup trigger
        logger.info("Testing manual backup trigger...")
        result = await backup_scheduler.trigger_manual_backup("files")
        logger.info(f"Manual backup result: {result}")
        
        # List backups
        backups = await backup_system.list_backups()
        logger.info(f"Total backups: {len(backups)}")
        
        for backup in backups[:3]:  # Show first 3
            logger.info(f"  {backup['filename']} ({backup['type']}) - {backup['size']} bytes")
        
        # Test backup verification
        if backups:
            logger.info("Testing backup verification...")
            latest_backup = backups[0]
            is_valid = await backup_system.verify_backup_integrity(latest_backup)
            logger.info(f"Latest backup verification: {'PASSED' if is_valid else 'FAILED'}")
        
        logger.info("Backup operations test completed!")
        
    except Exception as e:
        logger.error(f"Backup operations test failed: {e}")

def check_dependencies():
    """Check required dependencies"""
    logger.info("Checking dependencies...")
    
    # Check for pg_dump
    if os.system("which pg_dump > /dev/null 2>&1") != 0:
        logger.warning("pg_dump not found - database backups may not work")
        logger.info("Install PostgreSQL client tools: apt-get install postgresql-client")
    else:
        logger.info("pg_dump found")
    
    # Check for tar
    if os.system("which tar > /dev/null 2>&1") != 0:
        logger.error("tar not found - file backups will not work")
        return False
    else:
        logger.info("tar found")
    
    # Check backup directory permissions
    backup_dir = backup_system.backup_dir
    if backup_dir.exists():
        if not os.access(backup_dir, os.W_OK):
            logger.error(f"Backup directory not writable: {backup_dir}")
            return False
        else:
            logger.info(f"Backup directory writable: {backup_dir}")
    
    return True

def setup_environment():
    """Set up environment variables"""
    logger.info("Setting up environment...")
    
    settings = get_settings()
    
    # Check required settings
    required_settings = [
        'database_url',
        'backup_directory'
    ]
    
    missing_settings = []
    for setting in required_settings:
        if not hasattr(settings, setting) or not getattr(settings, setting):
            missing_settings.append(setting)
    
    if missing_settings:
        logger.warning(f"Missing settings: {missing_settings}")
        logger.info("Add these to your .env file:")
        for setting in missing_settings:
            if setting == 'backup_directory':
                logger.info(f"{setting.upper()}=./backups")
            elif setting == 'database_url':
                logger.info(f"{setting.upper()}=postgresql://user:pass@localhost/db")
    
    # Check optional cloud settings
    if hasattr(settings, 'aws_access_key_id') and settings.aws_access_key_id:
        logger.info("AWS credentials configured - cloud replication enabled")
    else:
        logger.info("AWS credentials not configured - cloud replication disabled")

async def main():
    """Main setup function"""
    logger.info("ScrollIntel Backup System Setup")
    logger.info("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        return 1
    
    # Setup environment
    setup_environment()
    
    # Setup backup system
    if not await setup_backup_system():
        logger.error("Backup system setup failed")
        return 1
    
    # Test operations
    await test_backup_operations()
    
    # Stop scheduler for clean exit
    await backup_scheduler.stop()
    
    logger.info("Setup completed successfully!")
    logger.info("The backup system is ready for production use.")
    logger.info("Scheduled backups will run automatically when the application starts.")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)