"""
Automated Backup and Data Recovery System
Provides comprehensive backup, verification, and recovery capabilities
"""

import os
import json
import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import subprocess
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from scrollintel.core.config import get_settings

# Optional boto3 import for cloud storage
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    boto3 = None
    HAS_BOTO3 = False

logger = logging.getLogger(__name__)

class BackupSystem:
    """Automated backup and recovery system"""
    
    def __init__(self):
        self.settings = get_settings()
        self.backup_dir = Path(self.settings.backup_directory or "backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize cloud storage clients
        self.s3_client = None
        if HAS_BOTO3 and self.settings.aws_access_key_id:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region or 'us-east-1'
            )
    
    async def create_database_backup(self) -> Dict[str, Any]:
        """Create a complete database backup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"scrollintel_db_backup_{timestamp}.sql"
            backup_path = self.backup_dir / backup_filename
            
            # Get database connection details
            db_url = self.settings.database_url
            
            # Create PostgreSQL dump
            cmd = [
                "pg_dump",
                db_url,
                "--no-password",
                "--verbose",
                "--clean",
                "--no-acl",
                "--no-owner",
                "-f", str(backup_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Database backup failed: {stderr.decode()}")
            
            # Calculate checksum
            checksum = await self._calculate_file_checksum(backup_path)
            
            # Create backup metadata
            metadata = {
                "filename": backup_filename,
                "path": str(backup_path),
                "timestamp": timestamp,
                "size": backup_path.stat().st_size,
                "checksum": checksum,
                "type": "database",
                "status": "completed"
            }
            
            # Save metadata
            metadata_path = backup_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Database backup created: {backup_filename}")
            return metadata
            
        except Exception as e:
            logger.error(f"Database backup failed: {str(e)}")
            raise
    
    async def create_file_backup(self, directories: List[str]) -> Dict[str, Any]:
        """Create backup of specified directories"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"scrollintel_files_backup_{timestamp}.tar.gz"
            backup_path = self.backup_dir / backup_filename
            
            # Create tar archive
            cmd = ["tar", "-czf", str(backup_path)] + directories
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"File backup failed: {stderr.decode()}")
            
            # Calculate checksum
            checksum = await self._calculate_file_checksum(backup_path)
            
            # Create backup metadata
            metadata = {
                "filename": backup_filename,
                "path": str(backup_path),
                "timestamp": timestamp,
                "size": backup_path.stat().st_size,
                "checksum": checksum,
                "type": "files",
                "directories": directories,
                "status": "completed"
            }
            
            # Save metadata
            metadata_path = backup_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"File backup created: {backup_filename}")
            return metadata
            
        except Exception as e:
            logger.error(f"File backup failed: {str(e)}")
            raise
    
    async def verify_backup_integrity(self, backup_metadata: Dict[str, Any]) -> bool:
        """Verify backup file integrity using checksums"""
        try:
            backup_path = Path(backup_metadata["path"])
            
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Recalculate checksum
            current_checksum = await self._calculate_file_checksum(backup_path)
            original_checksum = backup_metadata["checksum"]
            
            if current_checksum != original_checksum:
                logger.error(f"Backup integrity check failed: {backup_path}")
                return False
            
            logger.info(f"Backup integrity verified: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup verification failed: {str(e)}")
            return False
    
    async def replicate_to_cloud(self, backup_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Replicate backup to cloud storage"""
        if not HAS_BOTO3:
            logger.warning("Cloud replication skipped: boto3 not installed")
            backup_metadata["cloud_replicated"] = False
            backup_metadata["replication_error"] = "boto3 not available"
            return backup_metadata
        
        if not self.s3_client:
            logger.warning("Cloud replication skipped: AWS credentials not configured")
            backup_metadata["cloud_replicated"] = False
            backup_metadata["replication_error"] = "AWS credentials not configured"
            return backup_metadata
        
        try:
            backup_path = Path(backup_metadata["path"])
            s3_key = f"backups/{backup_metadata['filename']}"
            
            # Upload to S3
            self.s3_client.upload_file(
                str(backup_path),
                self.settings.backup_s3_bucket,
                s3_key
            )
            
            # Upload metadata
            metadata_s3_key = f"backups/{backup_metadata['filename']}.json"
            metadata_path = backup_path.with_suffix('.json')
            self.s3_client.upload_file(
                str(metadata_path),
                self.settings.backup_s3_bucket,
                metadata_s3_key
            )
            
            # Update metadata with cloud info
            backup_metadata["cloud_replicated"] = True
            backup_metadata["s3_bucket"] = self.settings.backup_s3_bucket
            backup_metadata["s3_key"] = s3_key
            backup_metadata["replicated_at"] = datetime.now().isoformat()
            
            logger.info(f"Backup replicated to cloud: {s3_key}")
            return backup_metadata
            
        except Exception as e:
            logger.error(f"Cloud replication failed: {str(e)}")
            backup_metadata["cloud_replicated"] = False
            backup_metadata["replication_error"] = str(e)
            return backup_metadata
    
    async def restore_database(self, backup_metadata: Dict[str, Any]) -> bool:
        """Restore database from backup"""
        try:
            backup_path = Path(backup_metadata["path"])
            
            # Verify backup integrity first
            if not await self.verify_backup_integrity(backup_metadata):
                raise Exception("Backup integrity check failed")
            
            # Get database connection details
            db_url = self.settings.database_url
            
            # Restore database
            cmd = [
                "psql",
                db_url,
                "--no-password",
                "-f", str(backup_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Database restore failed: {stderr.decode()}")
            
            logger.info(f"Database restored from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database restore failed: {str(e)}")
            return False
    
    async def restore_files(self, backup_metadata: Dict[str, Any], restore_path: str) -> bool:
        """Restore files from backup"""
        try:
            backup_path = Path(backup_metadata["path"])
            
            # Verify backup integrity first
            if not await self.verify_backup_integrity(backup_metadata):
                raise Exception("Backup integrity check failed")
            
            # Extract files
            cmd = ["tar", "-xzf", str(backup_path), "-C", restore_path]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"File restore failed: {stderr.decode()}")
            
            logger.info(f"Files restored from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"File restore failed: {str(e)}")
            return False
    
    async def point_in_time_recovery(self, target_time: datetime) -> bool:
        """Perform point-in-time recovery"""
        try:
            # Find the most recent backup before target time
            backups = await self.list_backups()
            suitable_backup = None
            
            for backup in backups:
                backup_time = datetime.fromisoformat(backup["timestamp"].replace("_", "T"))
                if backup_time <= target_time and backup["type"] == "database":
                    if not suitable_backup or backup_time > datetime.fromisoformat(suitable_backup["timestamp"].replace("_", "T")):
                        suitable_backup = backup
            
            if not suitable_backup:
                raise Exception(f"No suitable backup found for time: {target_time}")
            
            # Restore from backup
            success = await self.restore_database(suitable_backup)
            
            if success:
                logger.info(f"Point-in-time recovery completed to: {target_time}")
            
            return success
            
        except Exception as e:
            logger.error(f"Point-in-time recovery failed: {str(e)}")
            return False
    
    async def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        backups = []
        
        for metadata_file in self.backup_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    backups.append(metadata)
            except Exception as e:
                logger.error(f"Failed to read backup metadata: {metadata_file}, {str(e)}")
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        return backups
    
    async def cleanup_old_backups(self, retention_days: int = 30) -> int:
        """Clean up backups older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            backups = await self.list_backups()
            cleaned_count = 0
            
            for backup in backups:
                backup_time = datetime.fromisoformat(backup["timestamp"].replace("_", "T"))
                
                if backup_time < cutoff_date:
                    # Remove backup file
                    backup_path = Path(backup["path"])
                    if backup_path.exists():
                        backup_path.unlink()
                    
                    # Remove metadata file
                    metadata_path = backup_path.with_suffix('.json')
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    cleaned_count += 1
                    logger.info(f"Cleaned up old backup: {backup['filename']}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {str(e)}")
            return 0
    
    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()

# Global backup system instance
backup_system = BackupSystem()