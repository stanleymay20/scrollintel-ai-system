"""
Content Versioning and Backup System for Visual Generation
Handles version control, backup, and recovery of generated content
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)

class VersionAction(Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"

class BackupStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ContentVersion:
    version_id: str
    content_id: str
    version_number: int
    action: VersionAction
    storage_key: str
    file_hash: str
    file_size: int
    metadata: Dict[str, Any]
    created_at: datetime
    created_by: str
    parent_version: Optional[str] = None

@dataclass
class BackupRecord:
    backup_id: str
    content_ids: List[str]
    backup_location: str
    status: BackupStatus
    created_at: datetime
    completed_at: Optional[datetime]
    file_count: int
    total_size: int
    retention_until: datetime
    metadata: Dict[str, Any]

class ContentVersionManager:
    """Manages content versioning and history"""
    
    def __init__(self, storage_manager):
        self.storage_manager = storage_manager
        self.version_history: Dict[str, List[ContentVersion]] = {}
        self.max_versions_per_content = 10
        self.version_retention_days = 90
    
    async def create_version(self, content_id: str, storage_key: str, 
                           action: VersionAction, metadata: Dict[str, Any] = None,
                           created_by: str = "system") -> ContentVersion:
        """Create a new version record"""
        try:
            # Calculate file hash for integrity checking
            file_hash = await self._calculate_file_hash(storage_key)
            
            # Get file size
            file_size = await self._get_file_size(storage_key)
            
            # Generate version ID
            version_id = self._generate_version_id(content_id)
            
            # Get next version number
            version_number = await self._get_next_version_number(content_id)
            
            # Get parent version
            parent_version = None
            if content_id in self.version_history and self.version_history[content_id]:
                parent_version = self.version_history[content_id][-1].version_id
            
            # Create version record
            version = ContentVersion(
                version_id=version_id,
                content_id=content_id,
                version_number=version_number,
                action=action,
                storage_key=storage_key,
                file_hash=file_hash,
                file_size=file_size,
                metadata=metadata or {},
                created_at=datetime.utcnow(),
                created_by=created_by,
                parent_version=parent_version
            )
            
            # Add to version history
            if content_id not in self.version_history:
                self.version_history[content_id] = []
            
            self.version_history[content_id].append(version)
            
            # Enforce version limits
            await self._enforce_version_limits(content_id)
            
            # Persist version record
            await self._persist_version_record(version)
            
            logger.info(f"Created version {version_id} for content {content_id}")
            return version
            
        except Exception as e:
            logger.error(f"Failed to create version for content {content_id}: {str(e)}")
            raise
    
    async def get_version_history(self, content_id: str) -> List[ContentVersion]:
        """Get version history for content"""
        return self.version_history.get(content_id, [])
    
    async def get_version(self, version_id: str) -> Optional[ContentVersion]:
        """Get specific version by ID"""
        for content_versions in self.version_history.values():
            for version in content_versions:
                if version.version_id == version_id:
                    return version
        return None
    
    async def restore_version(self, version_id: str, new_storage_key: str) -> bool:
        """Restore content to a specific version"""
        try:
            # Get version record
            version = await self.get_version(version_id)
            if not version:
                logger.error(f"Version {version_id} not found")
                return False
            
            # Copy content from version storage to new location
            success = await self.storage_manager.copy_content(
                version.storage_key, 
                new_storage_key
            )
            
            if success:
                # Create restore version record
                await self.create_version(
                    version.content_id,
                    new_storage_key,
                    VersionAction.RESTORE,
                    {"restored_from": version_id},
                    "system"
                )
                
                logger.info(f"Successfully restored version {version_id}")
                return True
            else:
                logger.error(f"Failed to restore version {version_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error restoring version {version_id}: {str(e)}")
            return False
    
    async def delete_version(self, version_id: str) -> bool:
        """Delete a specific version"""
        try:
            version = await self.get_version(version_id)
            if not version:
                return False
            
            # Remove from storage
            await self.storage_manager.delete_content(version.storage_key)
            
            # Remove from version history
            content_versions = self.version_history.get(version.content_id, [])
            self.version_history[version.content_id] = [
                v for v in content_versions if v.version_id != version_id
            ]
            
            logger.info(f"Deleted version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {str(e)}")
            return False
    
    async def cleanup_old_versions(self) -> Dict[str, int]:
        """Clean up old versions based on retention policy"""
        try:
            cleanup_stats = {"deleted": 0, "errors": 0}
            cutoff_date = datetime.utcnow() - timedelta(days=self.version_retention_days)
            
            for content_id, versions in self.version_history.items():
                # Keep at least one version
                if len(versions) <= 1:
                    continue
                
                # Sort by creation date
                sorted_versions = sorted(versions, key=lambda v: v.created_at)
                
                # Delete old versions
                for version in sorted_versions[:-1]:  # Keep the latest version
                    if version.created_at < cutoff_date:
                        if await self.delete_version(version.version_id):
                            cleanup_stats["deleted"] += 1
                        else:
                            cleanup_stats["errors"] += 1
            
            logger.info(f"Version cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Version cleanup failed: {str(e)}")
            return {"deleted": 0, "errors": 1}
    
    def _generate_version_id(self, content_id: str) -> str:
        """Generate unique version ID"""
        timestamp = datetime.utcnow().isoformat()
        data = f"{content_id}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def _get_next_version_number(self, content_id: str) -> int:
        """Get next version number for content"""
        versions = self.version_history.get(content_id, [])
        if not versions:
            return 1
        return max(v.version_number for v in versions) + 1
    
    async def _calculate_file_hash(self, storage_key: str) -> str:
        """Calculate SHA256 hash of file content"""
        try:
            # For cloud storage, we'd need to download and hash
            # This is a simplified implementation
            return hashlib.sha256(storage_key.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {storage_key}: {str(e)}")
            return ""
    
    async def _get_file_size(self, storage_key: str) -> int:
        """Get file size from storage"""
        try:
            # This would query the storage provider for file size
            # Simplified implementation
            return 0
        except Exception as e:
            logger.error(f"Failed to get file size for {storage_key}: {str(e)}")
            return 0
    
    async def _enforce_version_limits(self, content_id: str):
        """Enforce maximum version limits per content"""
        versions = self.version_history.get(content_id, [])
        if len(versions) > self.max_versions_per_content:
            # Remove oldest versions
            sorted_versions = sorted(versions, key=lambda v: v.created_at)
            versions_to_remove = sorted_versions[:-self.max_versions_per_content]
            
            for version in versions_to_remove:
                await self.delete_version(version.version_id)
    
    async def _persist_version_record(self, version: ContentVersion):
        """Persist version record to storage"""
        try:
            # Convert to JSON and store
            version_data = asdict(version)
            version_data['created_at'] = version.created_at.isoformat()
            
            version_key = f"versions/{version.content_id}/{version.version_id}.json"
            
            # Store version metadata
            async with aiofiles.tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                await f.write(json.dumps(version_data, indent=2))
                temp_path = f.name
            
            try:
                await self.storage_manager.upload_content(
                    version.version_id,
                    temp_path,
                    "metadata"
                )
            finally:
                Path(temp_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Failed to persist version record: {str(e)}")

class BackupManager:
    """Manages automated backups and disaster recovery"""
    
    def __init__(self, storage_manager, backup_storage_manager=None):
        self.storage_manager = storage_manager
        self.backup_storage_manager = backup_storage_manager or storage_manager
        self.backup_records: Dict[str, BackupRecord] = {}
        self.backup_schedule = {
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "monthly": timedelta(days=30)
        }
        self.retention_policies = {
            "daily": timedelta(days=7),
            "weekly": timedelta(days=30),
            "monthly": timedelta(days=365)
        }
    
    async def create_backup(self, content_ids: List[str], backup_type: str = "manual",
                          metadata: Dict[str, Any] = None) -> BackupRecord:
        """Create backup of specified content"""
        try:
            backup_id = self._generate_backup_id()
            
            # Calculate retention period
            retention_period = self.retention_policies.get(backup_type, timedelta(days=30))
            retention_until = datetime.utcnow() + retention_period
            
            # Create backup record
            backup_record = BackupRecord(
                backup_id=backup_id,
                content_ids=content_ids,
                backup_location=f"backups/{backup_type}/{backup_id}",
                status=BackupStatus.PENDING,
                created_at=datetime.utcnow(),
                completed_at=None,
                file_count=0,
                total_size=0,
                retention_until=retention_until,
                metadata=metadata or {}
            )
            
            self.backup_records[backup_id] = backup_record
            
            # Start backup process
            asyncio.create_task(self._execute_backup(backup_record))
            
            logger.info(f"Started backup {backup_id} for {len(content_ids)} items")
            return backup_record
            
        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            raise
    
    async def _execute_backup(self, backup_record: BackupRecord):
        """Execute the backup process"""
        try:
            backup_record.status = BackupStatus.IN_PROGRESS
            
            total_size = 0
            file_count = 0
            
            for content_id in backup_record.content_ids:
                try:
                    # Get content info
                    content_list = await self.storage_manager.list_content(
                        prefix=f"*{content_id}*"
                    )
                    
                    for content in content_list:
                        # Copy to backup location
                        backup_key = f"{backup_record.backup_location}/{content['key']}"
                        
                        success = await self.storage_manager.copy_content(
                            content['key'],
                            backup_key
                        )
                        
                        if success:
                            total_size += content['size']
                            file_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to backup content {content_id}: {str(e)}")
            
            # Update backup record
            backup_record.status = BackupStatus.COMPLETED
            backup_record.completed_at = datetime.utcnow()
            backup_record.file_count = file_count
            backup_record.total_size = total_size
            
            # Persist backup record
            await self._persist_backup_record(backup_record)
            
            logger.info(f"Backup {backup_record.backup_id} completed: {file_count} files, {total_size} bytes")
            
        except Exception as e:
            backup_record.status = BackupStatus.FAILED
            logger.error(f"Backup {backup_record.backup_id} failed: {str(e)}")
    
    async def restore_backup(self, backup_id: str, target_location: str = None) -> bool:
        """Restore content from backup"""
        try:
            backup_record = self.backup_records.get(backup_id)
            if not backup_record:
                logger.error(f"Backup {backup_id} not found")
                return False
            
            if backup_record.status != BackupStatus.COMPLETED:
                logger.error(f"Backup {backup_id} is not completed")
                return False
            
            # List backup content
            backup_content = await self.backup_storage_manager.list_content(
                prefix=backup_record.backup_location
            )
            
            restored_count = 0
            
            for content in backup_content:
                try:
                    # Determine restore location
                    original_key = content['key'].replace(f"{backup_record.backup_location}/", "")
                    restore_key = f"{target_location}/{original_key}" if target_location else original_key
                    
                    # Restore content
                    success = await self.backup_storage_manager.copy_content(
                        content['key'],
                        restore_key
                    )
                    
                    if success:
                        restored_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to restore {content['key']}: {str(e)}")
            
            logger.info(f"Restored {restored_count} files from backup {backup_id}")
            return restored_count > 0
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_id}: {str(e)}")
            return False
    
    async def cleanup_expired_backups(self) -> Dict[str, int]:
        """Clean up expired backups"""
        try:
            cleanup_stats = {"deleted": 0, "errors": 0}
            current_time = datetime.utcnow()
            
            for backup_id, backup_record in list(self.backup_records.items()):
                if current_time > backup_record.retention_until:
                    try:
                        # Delete backup content
                        backup_content = await self.backup_storage_manager.list_content(
                            prefix=backup_record.backup_location
                        )
                        
                        for content in backup_content:
                            await self.backup_storage_manager.delete_content(content['key'])
                        
                        # Remove backup record
                        del self.backup_records[backup_id]
                        cleanup_stats["deleted"] += 1
                        
                        logger.info(f"Deleted expired backup {backup_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to delete backup {backup_id}: {str(e)}")
                        cleanup_stats["errors"] += 1
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {str(e)}")
            return {"deleted": 0, "errors": 1}
    
    async def get_backup_status(self, backup_id: str) -> Optional[BackupRecord]:
        """Get backup status"""
        return self.backup_records.get(backup_id)
    
    async def list_backups(self, backup_type: str = None) -> List[BackupRecord]:
        """List all backups, optionally filtered by type"""
        backups = list(self.backup_records.values())
        
        if backup_type:
            backups = [
                backup for backup in backups
                if backup.metadata.get("type") == backup_type
            ]
        
        return sorted(backups, key=lambda b: b.created_at, reverse=True)
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"backup_{timestamp}_{hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]}"
    
    async def _persist_backup_record(self, backup_record: BackupRecord):
        """Persist backup record to storage"""
        try:
            # Convert to JSON
            backup_data = asdict(backup_record)
            backup_data['created_at'] = backup_record.created_at.isoformat()
            if backup_record.completed_at:
                backup_data['completed_at'] = backup_record.completed_at.isoformat()
            backup_data['retention_until'] = backup_record.retention_until.isoformat()
            backup_data['status'] = backup_record.status.value
            
            # Store backup metadata
            backup_key = f"backup_records/{backup_record.backup_id}.json"
            
            async with aiofiles.tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                await f.write(json.dumps(backup_data, indent=2))
                temp_path = f.name
            
            try:
                await self.storage_manager.upload_content(
                    backup_record.backup_id,
                    temp_path,
                    "metadata"
                )
            finally:
                Path(temp_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Failed to persist backup record: {str(e)}")

class DisasterRecoveryManager:
    """Manages disaster recovery procedures"""
    
    def __init__(self, storage_manager, backup_manager, version_manager):
        self.storage_manager = storage_manager
        self.backup_manager = backup_manager
        self.version_manager = version_manager
        
    async def create_recovery_plan(self, content_ids: List[str]) -> Dict[str, Any]:
        """Create disaster recovery plan for content"""
        try:
            recovery_plan = {
                "plan_id": self._generate_plan_id(),
                "content_ids": content_ids,
                "created_at": datetime.utcnow().isoformat(),
                "recovery_steps": [],
                "estimated_time": 0,
                "backup_requirements": {}
            }
            
            # Analyze content and create recovery steps
            for content_id in content_ids:
                # Check existing backups
                backups = await self.backup_manager.list_backups()
                relevant_backups = [
                    b for b in backups 
                    if content_id in b.content_ids and b.status == BackupStatus.COMPLETED
                ]
                
                if relevant_backups:
                    # Use most recent backup
                    latest_backup = max(relevant_backups, key=lambda b: b.created_at)
                    recovery_plan["recovery_steps"].append({
                        "step": f"restore_from_backup",
                        "content_id": content_id,
                        "backup_id": latest_backup.backup_id,
                        "estimated_time_minutes": 5
                    })
                else:
                    # Check version history
                    versions = await self.version_manager.get_version_history(content_id)
                    if versions:
                        latest_version = versions[-1]
                        recovery_plan["recovery_steps"].append({
                            "step": "restore_from_version",
                            "content_id": content_id,
                            "version_id": latest_version.version_id,
                            "estimated_time_minutes": 2
                        })
                    else:
                        recovery_plan["recovery_steps"].append({
                            "step": "regenerate_content",
                            "content_id": content_id,
                            "estimated_time_minutes": 30
                        })
            
            # Calculate total estimated time
            recovery_plan["estimated_time"] = sum(
                step["estimated_time_minutes"] for step in recovery_plan["recovery_steps"]
            )
            
            return recovery_plan
            
        except Exception as e:
            logger.error(f"Failed to create recovery plan: {str(e)}")
            return {}
    
    async def execute_recovery_plan(self, recovery_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute disaster recovery plan"""
        try:
            execution_results = {
                "plan_id": recovery_plan["plan_id"],
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": None,
                "success_count": 0,
                "failure_count": 0,
                "step_results": []
            }
            
            for step in recovery_plan["recovery_steps"]:
                step_result = {
                    "step": step["step"],
                    "content_id": step["content_id"],
                    "success": False,
                    "error": None,
                    "completed_at": None
                }
                
                try:
                    if step["step"] == "restore_from_backup":
                        success = await self.backup_manager.restore_backup(
                            step["backup_id"],
                            f"recovered/{step['content_id']}"
                        )
                        step_result["success"] = success
                        
                    elif step["step"] == "restore_from_version":
                        success = await self.version_manager.restore_version(
                            step["version_id"],
                            f"recovered/{step['content_id']}"
                        )
                        step_result["success"] = success
                        
                    elif step["step"] == "regenerate_content":
                        # This would trigger content regeneration
                        # For now, we'll mark as requiring manual intervention
                        step_result["success"] = False
                        step_result["error"] = "Requires manual regeneration"
                    
                    if step_result["success"]:
                        execution_results["success_count"] += 1
                    else:
                        execution_results["failure_count"] += 1
                    
                    step_result["completed_at"] = datetime.utcnow().isoformat()
                    
                except Exception as e:
                    step_result["success"] = False
                    step_result["error"] = str(e)
                    execution_results["failure_count"] += 1
                
                execution_results["step_results"].append(step_result)
            
            execution_results["completed_at"] = datetime.utcnow().isoformat()
            
            logger.info(f"Recovery plan executed: {execution_results['success_count']} successes, {execution_results['failure_count']} failures")
            return execution_results
            
        except Exception as e:
            logger.error(f"Failed to execute recovery plan: {str(e)}")
            return {"error": str(e)}
    
    def _generate_plan_id(self) -> str:
        """Generate unique recovery plan ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"recovery_{timestamp}_{hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]}"

# Factory functions
def create_version_manager(storage_manager) -> ContentVersionManager:
    """Create content version manager"""
    return ContentVersionManager(storage_manager)

def create_backup_manager(storage_manager, backup_storage_manager=None) -> BackupManager:
    """Create backup manager"""
    return BackupManager(storage_manager, backup_storage_manager)

def create_disaster_recovery_manager(storage_manager, backup_manager, version_manager) -> DisasterRecoveryManager:
    """Create disaster recovery manager"""
    return DisasterRecoveryManager(storage_manager, backup_manager, version_manager)