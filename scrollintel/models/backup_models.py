"""
Backup Data Models
Pydantic models for backup system API requests and responses
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime

class BackupMetadata(BaseModel):
    """Backup metadata model"""
    filename: str
    path: str
    timestamp: str
    size: int
    checksum: str
    type: str  # "database" or "files"
    status: str
    directories: Optional[List[str]] = None
    cloud_replicated: Optional[bool] = False
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None
    replicated_at: Optional[str] = None
    replication_error: Optional[str] = None

class BackupTriggerRequest(BaseModel):
    """Request model for triggering manual backup"""
    backup_type: str = Field(..., description="Type of backup: 'database', 'files', or 'both'")

class RestoreRequest(BaseModel):
    """Request model for restore operations"""
    backup_id: str = Field(..., description="Backup ID (filename or timestamp)")
    restore_path: Optional[str] = Field(None, description="Path for file restore (optional)")

class BackupResponse(BaseModel):
    """Response model for backup operations"""
    success: bool
    message: str
    backup_type: Optional[str] = None
    triggered_at: Optional[str] = None
    metadata: Optional[BackupMetadata] = None

class BackupListResponse(BaseModel):
    """Response model for listing backups"""
    success: bool
    backups: List[Dict[str, Any]]
    total_count: int

class BackupStatusResponse(BaseModel):
    """Response model for backup system status"""
    success: bool
    total_backups: int
    recent_backups: int
    database_backups: int
    file_backups: int
    total_size_bytes: int
    scheduler_running: bool
    next_scheduled_backups: Dict[str, str]
    last_backup: Optional[Dict[str, Any]] = None

class BackupVerificationResult(BaseModel):
    """Result model for backup verification"""
    backup_id: str
    is_valid: bool
    verified_at: str
    error: Optional[str] = None

class PointInTimeRecoveryRequest(BaseModel):
    """Request model for point-in-time recovery"""
    target_time: str = Field(..., description="Target recovery time in ISO format")

class BackupHealthStatus(BaseModel):
    """Health status model for backup system"""
    healthy: bool
    backup_directory_exists: bool
    scheduler_running: bool
    recent_backups_24h: int
    total_backups: int
    checked_at: str
    error: Optional[str] = None