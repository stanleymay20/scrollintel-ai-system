"""
Backup API Routes
Provides REST API endpoints for backup management
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import os

from scrollintel.core.backup_system import backup_system
from scrollintel.core.backup_scheduler import backup_scheduler
from scrollintel.security.auth import get_current_admin_user
from scrollintel.models.backup_models import (
    BackupResponse,
    BackupListResponse,
    BackupTriggerRequest,
    RestoreRequest,
    BackupStatusResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/backup", tags=["backup"])

@router.get("/list", response_model=BackupListResponse)
async def list_backups(
    current_user = Depends(get_current_admin_user)
) -> BackupListResponse:
    """List all available backups"""
    try:
        backups = await backup_system.list_backups()
        
        return BackupListResponse(
            success=True,
            backups=backups,
            total_count=len(backups)
        )
        
    except Exception as e:
        logger.error(f"Failed to list backups: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trigger", response_model=BackupResponse)
async def trigger_backup(
    request: BackupTriggerRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_admin_user)
) -> BackupResponse:
    """Trigger manual backup"""
    try:
        # Run backup in background
        background_tasks.add_task(
            backup_scheduler.trigger_manual_backup,
            request.backup_type
        )
        
        return BackupResponse(
            success=True,
            message=f"Manual {request.backup_type} backup triggered",
            backup_type=request.backup_type,
            triggered_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to trigger backup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify/{backup_id}")
async def verify_backup(
    backup_id: str,
    current_user = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """Verify backup integrity"""
    try:
        backups = await backup_system.list_backups()
        backup_metadata = None
        
        for backup in backups:
            if backup.get("filename") == backup_id or backup.get("timestamp") == backup_id:
                backup_metadata = backup
                break
        
        if not backup_metadata:
            raise HTTPException(status_code=404, detail="Backup not found")
        
        is_valid = await backup_system.verify_backup_integrity(backup_metadata)
        
        return {
            "success": True,
            "backup_id": backup_id,
            "is_valid": is_valid,
            "verified_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify backup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/restore/database")
async def restore_database(
    request: RestoreRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """Restore database from backup"""
    try:
        backups = await backup_system.list_backups()
        backup_metadata = None
        
        for backup in backups:
            if (backup.get("filename") == request.backup_id or 
                backup.get("timestamp") == request.backup_id) and backup.get("type") == "database":
                backup_metadata = backup
                break
        
        if not backup_metadata:
            raise HTTPException(status_code=404, detail="Database backup not found")
        
        # Run restore in background
        background_tasks.add_task(
            backup_system.restore_database,
            backup_metadata
        )
        
        return {
            "success": True,
            "message": "Database restore initiated",
            "backup_id": request.backup_id,
            "initiated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restore database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/restore/files")
async def restore_files(
    request: RestoreRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """Restore files from backup"""
    try:
        backups = await backup_system.list_backups()
        backup_metadata = None
        
        for backup in backups:
            if (backup.get("filename") == request.backup_id or 
                backup.get("timestamp") == request.backup_id) and backup.get("type") == "files":
                backup_metadata = backup
                break
        
        if not backup_metadata:
            raise HTTPException(status_code=404, detail="File backup not found")
        
        restore_path = request.restore_path or "/tmp/restore"
        
        # Run restore in background
        background_tasks.add_task(
            backup_system.restore_files,
            backup_metadata,
            restore_path
        )
        
        return {
            "success": True,
            "message": "File restore initiated",
            "backup_id": request.backup_id,
            "restore_path": restore_path,
            "initiated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restore files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/point-in-time-recovery")
async def point_in_time_recovery(
    target_time: str,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """Perform point-in-time recovery"""
    try:
        # Parse target time
        try:
            target_datetime = datetime.fromisoformat(target_time.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid datetime format")
        
        # Run point-in-time recovery in background
        background_tasks.add_task(
            backup_system.point_in_time_recovery,
            target_datetime
        )
        
        return {
            "success": True,
            "message": "Point-in-time recovery initiated",
            "target_time": target_time,
            "initiated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initiate point-in-time recovery: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=BackupStatusResponse)
async def get_backup_status(
    current_user = Depends(get_current_admin_user)
) -> BackupStatusResponse:
    """Get backup system status"""
    try:
        backups = await backup_system.list_backups()
        
        # Get recent backups (last 7 days)
        recent_backups = []
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for backup in backups:
            backup_time = datetime.fromisoformat(backup["timestamp"].replace("_", "T"))
            if backup_time >= cutoff_date:
                recent_backups.append(backup)
        
        # Get next scheduled backup times
        next_backups = backup_scheduler.get_next_backup_times()
        
        # Calculate statistics
        total_size = sum(backup.get("size", 0) for backup in backups)
        database_backups = [b for b in backups if b.get("type") == "database"]
        file_backups = [b for b in backups if b.get("type") == "files"]
        
        return BackupStatusResponse(
            success=True,
            total_backups=len(backups),
            recent_backups=len(recent_backups),
            database_backups=len(database_backups),
            file_backups=len(file_backups),
            total_size_bytes=total_size,
            scheduler_running=backup_scheduler.is_running,
            next_scheduled_backups=next_backups,
            last_backup=backups[0] if backups else None
        )
        
    except Exception as e:
        logger.error(f"Failed to get backup status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup")
async def cleanup_old_backups(
    retention_days: int = 30,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """Clean up old backups"""
    try:
        # Run cleanup in background
        background_tasks.add_task(
            backup_system.cleanup_old_backups,
            retention_days
        )
        
        return {
            "success": True,
            "message": f"Backup cleanup initiated (retention: {retention_days} days)",
            "retention_days": retention_days,
            "initiated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup backups: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def backup_health_check() -> Dict[str, Any]:
    """Health check for backup system"""
    try:
        # Check if backup directory exists and is writable
        backup_dir = backup_system.backup_dir
        is_healthy = backup_dir.exists() and backup_dir.is_dir()
        
        # Check scheduler status
        scheduler_healthy = backup_scheduler.is_running
        
        # Get recent backup count
        backups = await backup_system.list_backups()
        recent_backup_count = len([
            b for b in backups 
            if datetime.fromisoformat(b["timestamp"].replace("_", "T")) >= datetime.now() - timedelta(days=1)
        ])
        
        return {
            "success": True,
            "healthy": is_healthy and scheduler_healthy,
            "backup_directory_exists": backup_dir.exists(),
            "scheduler_running": scheduler_healthy,
            "recent_backups_24h": recent_backup_count,
            "total_backups": len(backups),
            "checked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Backup health check failed: {str(e)}")
        return {
            "success": False,
            "healthy": False,
            "error": str(e),
            "checked_at": datetime.now().isoformat()
        }