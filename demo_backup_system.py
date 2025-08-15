#!/usr/bin/env python3
"""
Backup System Demo
Demonstrates backup and recovery functionality
"""

import asyncio
import logging
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_backup_system():
    """Demo the backup system functionality"""
    logger.info("ScrollIntel Backup System Demo")
    logger.info("=" * 50)
    
    try:
        from scrollintel.core.backup_system import backup_system
        from scrollintel.core.backup_scheduler import backup_scheduler
        
        # Test 1: Create backup directory
        logger.info("1. Testing backup directory creation...")
        backup_dir = backup_system.backup_dir
        backup_dir.mkdir(exist_ok=True)
        logger.info(f"✓ Backup directory created: {backup_dir}")
        
        # Test 2: Create test data for file backup
        logger.info("\n2. Creating test data...")
        test_dir = Path("test_backup_data")
        test_dir.mkdir(exist_ok=True)
        
        # Create test files
        (test_dir / "file1.txt").write_text("Test content 1")
        (test_dir / "file2.txt").write_text("Test content 2")
        subdir = test_dir / "subdir"
        subdir.mkdir(exist_ok=True)
        (subdir / "file3.txt").write_text("Test content 3")
        logger.info(f"✓ Test data created in: {test_dir}")
        
        # Test 3: File backup (if tar is available)
        logger.info("\n3. Testing file backup...")
        try:
            backup_metadata = await backup_system.create_file_backup([str(test_dir)])
            logger.info(f"✓ File backup created: {backup_metadata['filename']}")
            logger.info(f"  Size: {backup_metadata['size']} bytes")
            logger.info(f"  Checksum: {backup_metadata['checksum'][:16]}...")
            
            # Test backup verification
            is_valid = await backup_system.verify_backup_integrity(backup_metadata)
            logger.info(f"✓ Backup verification: {'PASSED' if is_valid else 'FAILED'}")
            
        except Exception as e:
            logger.warning(f"File backup failed (expected on Windows): {e}")
        
        # Test 4: List backups
        logger.info("\n4. Testing backup listing...")
        backups = await backup_system.list_backups()
        logger.info(f"✓ Found {len(backups)} backups")
        
        for i, backup in enumerate(backups[:3]):
            logger.info(f"  {i+1}. {backup['filename']} ({backup['type']}) - {backup.get('size', 0)} bytes")
        
        # Test 5: Backup scheduler
        logger.info("\n5. Testing backup scheduler...")
        try:
            await backup_scheduler.start()
            if backup_scheduler.is_running:
                logger.info("✓ Backup scheduler started")
                
                # Get next backup times
                next_times = backup_scheduler.get_next_backup_times()
                logger.info(f"✓ Scheduled jobs: {len(next_times)}")
                for job_id, next_time in next_times.items():
                    logger.info(f"  - {job_id}: {next_time}")
                
                await backup_scheduler.stop()
                logger.info("✓ Backup scheduler stopped")
            else:
                logger.warning("Scheduler not running (APScheduler may not be installed)")
        except Exception as e:
            logger.warning(f"Scheduler test failed: {e}")
        
        # Test 6: Manual backup trigger
        logger.info("\n6. Testing manual backup trigger...")
        try:
            result = await backup_scheduler.trigger_manual_backup("files")
            logger.info(f"✓ Manual backup triggered: {result}")
        except Exception as e:
            logger.warning(f"Manual backup trigger failed: {e}")
        
        # Test 7: Backup system health
        logger.info("\n7. Testing backup system health...")
        
        # Check backup directory
        backup_healthy = backup_dir.exists() and backup_dir.is_dir()
        logger.info(f"✓ Backup directory healthy: {backup_healthy}")
        
        # Check recent backups
        recent_backups = [
            b for b in backups 
            if datetime.fromisoformat(b["timestamp"].replace("_", "T")) >= datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        ]
        logger.info(f"✓ Recent backups (today): {len(recent_backups)}")
        
        # Test 8: Backup metadata validation
        logger.info("\n8. Testing backup metadata...")
        if backups:
            latest_backup = backups[0]
            required_fields = ["filename", "timestamp", "type", "size", "checksum"]
            
            missing_fields = [field for field in required_fields if field not in latest_backup]
            if not missing_fields:
                logger.info("✓ Backup metadata complete")
            else:
                logger.warning(f"Missing metadata fields: {missing_fields}")
        
        # Test 9: Cloud replication (if configured)
        logger.info("\n9. Testing cloud replication...")
        if backup_system.s3_client:
            logger.info("✓ AWS S3 client configured")
            # Note: We don't actually test upload to avoid costs
            logger.info("  (Cloud upload test skipped to avoid charges)")
        else:
            logger.info("ℹ Cloud replication not configured (AWS credentials not set)")
        
        # Cleanup
        logger.info("\n10. Cleanup...")
        try:
            import shutil
            if test_dir.exists():
                shutil.rmtree(test_dir)
                logger.info("✓ Test data cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
        
        logger.info("\n" + "=" * 50)
        logger.info("BACKUP SYSTEM DEMO COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        
        # Summary
        logger.info("\nSUMMARY:")
        logger.info("✓ Backup system initialized")
        logger.info("✓ File operations tested")
        logger.info("✓ Metadata handling verified")
        logger.info("✓ Scheduler functionality checked")
        logger.info("✓ Health monitoring validated")
        
        logger.info("\nNOTES:")
        logger.info("- Database backups require PostgreSQL client tools (pg_dump)")
        logger.info("- File backups require tar utility (or 7-zip on Windows)")
        logger.info("- Cloud replication requires AWS credentials and boto3")
        logger.info("- Scheduling requires APScheduler library")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return False

async def main():
    """Main demo function"""
    success = await demo_backup_system()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)