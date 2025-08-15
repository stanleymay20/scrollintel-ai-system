"""
Data models for cross-device and cross-session continuity.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class DeviceType(str, Enum):
    """Device types for optimization."""
    DESKTOP = "desktop"
    TABLET = "tablet"
    MOBILE = "mobile"
    UNKNOWN = "unknown"


class SyncStatus(str, Enum):
    """Synchronization status."""
    SYNCED = "synced"
    PENDING = "pending"
    CONFLICT = "conflict"
    OFFLINE = "offline"
    ERROR = "error"


class ConflictResolutionStrategy(str, Enum):
    """Conflict resolution strategies."""
    LAST_WRITE_WINS = "last_write_wins"
    MERGE_CHANGES = "merge_changes"
    USER_CHOICE = "user_choice"
    KEEP_BOTH = "keep_both"
    AUTOMATIC_MERGE = "automatic_merge"


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    user_id: str = Field(..., description="User ID")
    device_id: str = Field(..., description="Device ID")
    device_type: DeviceType = Field(default=DeviceType.DESKTOP, description="Device type")
    initial_state: Optional[Dict[str, Any]] = Field(default=None, description="Initial session state")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "device_id": "device456",
                "device_type": "desktop",
                "initial_state": {
                    "current_page": "/dashboard",
                    "filters": {"date_range": "last_30_days"},
                    "preferences": {"theme": "dark"}
                }
            }
        }


class UpdateSessionRequest(BaseModel):
    """Request to update session state."""
    user_id: str = Field(..., description="User ID")
    device_id: str = Field(..., description="Device ID")
    state_updates: Dict[str, Any] = Field(..., description="State updates to apply")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "device_id": "device456",
                "state_updates": {
                    "current_page": "/analytics",
                    "filters": {"date_range": "last_7_days"},
                    "last_action": "view_chart"
                }
            }
        }


class SyncRequest(BaseModel):
    """Request to synchronize across devices."""
    user_id: str = Field(..., description="User ID")
    force_sync: bool = Field(default=False, description="Force synchronization even if no conflicts")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "force_sync": False
            }
        }


class OfflineModeRequest(BaseModel):
    """Request for offline mode operations."""
    user_id: str = Field(..., description="User ID")
    device_id: str = Field(..., description="Device ID")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "device_id": "device456"
            }
        }


class SessionResponse(BaseModel):
    """Response containing session information."""
    success: bool = Field(..., description="Whether the operation was successful")
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    device_id: str = Field(..., description="Device ID")
    created_at: datetime = Field(..., description="Session creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")
    state_data: Dict[str, Any] = Field(..., description="Session state data")
    version: int = Field(..., description="Session version number")
    message: str = Field(..., description="Response message")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "session_id": "user123_device456_1640995200",
                "user_id": "user123",
                "device_id": "device456",
                "created_at": "2023-12-31T12:00:00Z",
                "last_updated": "2023-12-31T12:30:00Z",
                "state_data": {
                    "current_page": "/dashboard",
                    "filters": {"date_range": "last_30_days"}
                },
                "version": 5,
                "message": "Session retrieved successfully"
            }
        }


class SyncStatusResponse(BaseModel):
    """Response containing synchronization status."""
    success: bool = Field(..., description="Whether the operation was successful")
    user_id: str = Field(..., description="User ID")
    total_sessions: int = Field(..., description="Total number of sessions")
    active_devices: int = Field(..., description="Number of active devices")
    pending_conflicts: int = Field(..., description="Number of pending conflicts")
    offline_mode: bool = Field(..., description="Whether offline mode is enabled")
    offline_queue_size: int = Field(..., description="Number of queued offline changes")
    last_sync: Optional[str] = Field(default=None, description="Last synchronization timestamp")
    websocket_connections: int = Field(..., description="Number of active WebSocket connections")
    message: str = Field(..., description="Response message")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "user_id": "user123",
                "total_sessions": 3,
                "active_devices": 2,
                "pending_conflicts": 0,
                "offline_mode": False,
                "offline_queue_size": 0,
                "last_sync": "2023-12-31T12:30:00Z",
                "websocket_connections": 2,
                "message": "Sync status retrieved successfully"
            }
        }


class DeviceInfo(BaseModel):
    """Device information model."""
    device_id: str = Field(..., description="Device ID")
    device_type: DeviceType = Field(..., description="Device type")
    user_agent: str = Field(..., description="User agent string")
    last_seen: datetime = Field(..., description="Last seen timestamp")
    capabilities: Dict[str, Any] = Field(..., description="Device capabilities")
    network_quality: str = Field(default="good", description="Network quality")
    is_online: bool = Field(default=True, description="Whether device is online")
    
    class Config:
        schema_extra = {
            "example": {
                "device_id": "device456",
                "device_type": "desktop",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "last_seen": "2023-12-31T12:30:00Z",
                "capabilities": {
                    "sync": True,
                    "offline": True,
                    "websocket": True
                },
                "network_quality": "good",
                "is_online": True
            }
        }


class StateChange(BaseModel):
    """State change model for synchronization."""
    change_id: str = Field(..., description="Change ID")
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    device_id: str = Field(..., description="Device ID")
    timestamp: datetime = Field(..., description="Change timestamp")
    path: str = Field(..., description="JSON path to changed data")
    old_value: Any = Field(..., description="Previous value")
    new_value: Any = Field(..., description="New value")
    operation: str = Field(..., description="Operation type (set, delete, merge)")
    conflict_resolution: Optional[ConflictResolutionStrategy] = Field(
        default=None, description="Conflict resolution strategy"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "change_id": "change789",
                "session_id": "user123_device456_1640995200",
                "user_id": "user123",
                "device_id": "device456",
                "timestamp": "2023-12-31T12:30:00Z",
                "path": "filters.date_range",
                "old_value": "last_30_days",
                "new_value": "last_7_days",
                "operation": "set",
                "conflict_resolution": None
            }
        }


class SyncConflict(BaseModel):
    """Synchronization conflict model."""
    conflict_id: str = Field(..., description="Conflict ID")
    session_id: str = Field(..., description="Session ID")
    path: str = Field(..., description="JSON path where conflict occurred")
    local_change: StateChange = Field(..., description="Local change")
    remote_change: StateChange = Field(..., description="Remote change")
    resolution_strategy: ConflictResolutionStrategy = Field(..., description="Resolution strategy")
    resolved: bool = Field(default=False, description="Whether conflict is resolved")
    resolution_data: Optional[Any] = Field(default=None, description="Resolution result data")
    
    class Config:
        schema_extra = {
            "example": {
                "conflict_id": "conflict123",
                "session_id": "user123_device456_1640995200",
                "path": "filters.date_range",
                "local_change": {
                    "change_id": "change789",
                    "session_id": "user123_device456_1640995200",
                    "user_id": "user123",
                    "device_id": "device456",
                    "timestamp": "2023-12-31T12:30:00Z",
                    "path": "filters.date_range",
                    "old_value": "last_30_days",
                    "new_value": "last_7_days",
                    "operation": "set"
                },
                "remote_change": {
                    "change_id": "change790",
                    "session_id": "user123_device789_1640995200",
                    "user_id": "user123",
                    "device_id": "device789",
                    "timestamp": "2023-12-31T12:31:00Z",
                    "path": "filters.date_range",
                    "old_value": "last_30_days",
                    "new_value": "last_14_days",
                    "operation": "set"
                },
                "resolution_strategy": "last_write_wins",
                "resolved": False,
                "resolution_data": None
            }
        }


class TabSyncRequest(BaseModel):
    """Request to synchronize between tabs."""
    user_id: str = Field(..., description="User ID")
    tab_id: str = Field(..., description="Tab ID")
    state_updates: Dict[str, Any] = Field(..., description="State updates from this tab")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "tab_id": "tab456",
                "state_updates": {
                    "current_page": "/analytics",
                    "selected_chart": "revenue_chart",
                    "zoom_level": 1.2
                }
            }
        }


class OfflineCapabilities(BaseModel):
    """Offline capabilities model."""
    read_cached_data: bool = Field(default=True, description="Can read cached data offline")
    create_new_items: bool = Field(default=True, description="Can create new items offline")
    edit_existing_items: bool = Field(default=True, description="Can edit existing items offline")
    delete_items: bool = Field(default=True, description="Can delete items offline")
    sync_when_online: bool = Field(default=True, description="Can sync when back online")
    
    class Config:
        schema_extra = {
            "example": {
                "read_cached_data": True,
                "create_new_items": True,
                "edit_existing_items": True,
                "delete_items": True,
                "sync_when_online": True
            }
        }


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str = Field(..., description="Message type")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "state_update",
                "data": {
                    "session_id": "user123_device456_1640995200",
                    "state_updates": {
                        "current_page": "/analytics"
                    }
                },
                "timestamp": "2023-12-31T12:30:00Z"
            }
        }


class SyncResult(BaseModel):
    """Synchronization result model."""
    status: str = Field(..., description="Sync status")
    source_session: Optional[str] = Field(default=None, description="Source session ID")
    synced_sessions: int = Field(default=0, description="Number of synced sessions")
    conflicts: int = Field(default=0, description="Number of conflicts encountered")
    errors: List[str] = Field(default_factory=list, description="List of errors")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "source_session": "user123_device456_1640995200",
                "synced_sessions": 2,
                "conflicts": 0,
                "errors": []
            }
        }


class ConflictResolutionRequest(BaseModel):
    """Request to resolve a conflict."""
    conflict_id: str = Field(..., description="Conflict ID")
    resolution_strategy: ConflictResolutionStrategy = Field(..., description="Resolution strategy")
    user_choice_data: Optional[Any] = Field(default=None, description="User choice data if applicable")
    
    class Config:
        schema_extra = {
            "example": {
                "conflict_id": "conflict123",
                "resolution_strategy": "last_write_wins",
                "user_choice_data": None
            }
        }


class SessionMetadata(BaseModel):
    """Session metadata model."""
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    device_id: str = Field(..., description="Device ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")
    version: int = Field(..., description="Version number")
    checksum: str = Field(..., description="State checksum")
    size_bytes: int = Field(..., description="State size in bytes")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "user123_device456_1640995200",
                "user_id": "user123",
                "device_id": "device456",
                "created_at": "2023-12-31T12:00:00Z",
                "last_updated": "2023-12-31T12:30:00Z",
                "version": 5,
                "checksum": "a1b2c3d4e5f6",
                "size_bytes": 1024
            }
        }


class HealthStatus(BaseModel):
    """Health status model for cross-device continuity service."""
    service: str = Field(default="cross_device_continuity", description="Service name")
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Component health status")
    metrics: Dict[str, Any] = Field(..., description="Service metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "service": "cross_device_continuity",
                "status": "healthy",
                "timestamp": "2023-12-31T12:30:00Z",
                "components": {
                    "state_manager": "healthy",
                    "conflict_resolver": "healthy",
                    "offline_manager": "healthy",
                    "websocket_manager": "healthy"
                },
                "metrics": {
                    "active_sessions": 15,
                    "registered_devices": 8,
                    "pending_conflicts": 0,
                    "offline_queue_size": 0
                }
            }
        }