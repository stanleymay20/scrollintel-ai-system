"""
API routes for cross-device and cross-session continuity.
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging

from ...core.cross_device_continuity import (
    continuity_manager, create_user_session, restore_user_session,
    sync_user_state, enable_offline_mode, sync_after_reconnect,
    get_sync_status, DeviceType, ConflictResolutionStrategy
)
from ...core.never_fail_decorators import never_fail_api_endpoint
from ...models.cross_device_models import (
    CreateSessionRequest, UpdateSessionRequest, SyncRequest,
    SessionResponse, SyncStatusResponse, OfflineModeRequest
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cross-device", tags=["cross-device"])


@router.post("/sessions", response_model=SessionResponse)
@never_fail_api_endpoint(
    fallback_response={"success": False, "message": "Failed to create session"},
    user_action="create_session"
)
async def create_session(request: CreateSessionRequest):
    """Create a new cross-device session."""
    try:
        session = await create_user_session(
            user_id=request.user_id,
            device_id=request.device_id,
            device_type=request.device_type,
            initial_state=request.initial_state
        )
        
        return SessionResponse(
            success=True,
            session_id=session.session_id,
            user_id=session.user_id,
            device_id=session.device_id,
            created_at=session.created_at,
            last_updated=session.last_updated,
            state_data=session.state_data,
            version=session.version,
            message="Session created successfully"
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=SessionResponse)
@never_fail_api_endpoint(
    fallback_response={"success": False, "message": "Session not found"},
    user_action="restore_session"
)
async def get_session(session_id: str):
    """Restore a cross-device session."""
    try:
        session = await restore_user_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionResponse(
            success=True,
            session_id=session.session_id,
            user_id=session.user_id,
            device_id=session.device_id,
            created_at=session.created_at,
            last_updated=session.last_updated,
            state_data=session.state_data,
            version=session.version,
            message="Session restored successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restore session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sessions/{session_id}", response_model=Dict[str, Any])
@never_fail_api_endpoint(
    fallback_response={"success": False, "message": "Failed to update session"},
    user_action="update_session"
)
async def update_session(session_id: str, request: UpdateSessionRequest):
    """Update session state and sync across devices."""
    try:
        success = await sync_user_state(
            user_id=request.user_id,
            session_id=session_id,
            state_updates=request.state_updates,
            device_id=request.device_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found or update failed")
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Session updated and synced successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync", response_model=Dict[str, Any])
@never_fail_api_endpoint(
    fallback_response={"success": False, "message": "Sync failed"},
    user_action="sync_devices"
)
async def sync_devices(request: SyncRequest):
    """Synchronize state across all user's devices."""
    try:
        sync_results = await continuity_manager.sync_across_devices(request.user_id)
        
        return {
            "success": True,
            "user_id": request.user_id,
            "sync_results": sync_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to sync devices for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sync/status/{user_id}", response_model=SyncStatusResponse)
@never_fail_api_endpoint(
    fallback_response={"success": False, "message": "Failed to get sync status"},
    user_action="get_sync_status"
)
async def get_sync_status_endpoint(user_id: str):
    """Get synchronization status for a user."""
    try:
        status = await continuity_manager.get_sync_status(user_id)
        
        return SyncStatusResponse(
            success=True,
            user_id=user_id,
            total_sessions=status["total_sessions"],
            active_devices=status["active_devices"],
            pending_conflicts=status["pending_conflicts"],
            offline_mode=status["offline_mode"],
            offline_queue_size=status["offline_queue_size"],
            last_sync=status["last_sync"],
            websocket_connections=status["websocket_connections"],
            message="Sync status retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Failed to get sync status for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/offline/enable", response_model=Dict[str, Any])
@never_fail_api_endpoint(
    fallback_response={"success": False, "message": "Failed to enable offline mode"},
    user_action="enable_offline"
)
async def enable_offline_mode_endpoint(request: OfflineModeRequest):
    """Enable offline mode for a device."""
    try:
        result = await enable_offline_mode(request.user_id, request.device_id)
        
        return {
            "success": True,
            "user_id": request.user_id,
            "device_id": request.device_id,
            "offline_result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to enable offline mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/offline/sync", response_model=Dict[str, Any])
@never_fail_api_endpoint(
    fallback_response={"success": False, "message": "Failed to sync after reconnect"},
    user_action="sync_after_reconnect"
)
async def sync_after_reconnect_endpoint(request: OfflineModeRequest):
    """Sync after reconnecting from offline mode."""
    try:
        result = await sync_after_reconnect(request.user_id, request.device_id)
        
        return {
            "success": True,
            "user_id": request.user_id,
            "device_id": request.device_id,
            "sync_result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to sync after reconnect: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tabs/sync", response_model=Dict[str, Any])
@never_fail_api_endpoint(
    fallback_response={"success": False, "message": "Failed to sync tabs"},
    user_action="sync_tabs"
)
async def sync_tabs(user_id: str, tab_id: str, state_updates: Dict[str, Any]):
    """Synchronize state between multiple tabs."""
    try:
        result = await continuity_manager.handle_multi_tab_sync(
            user_id=user_id,
            tab_id=tab_id,
            state_updates=state_updates
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "tab_id": tab_id,
            "sync_result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to sync tabs for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/devices/{user_id}", response_model=Dict[str, Any])
@never_fail_api_endpoint(
    fallback_response={"success": False, "devices": [], "message": "Failed to get devices"},
    user_action="get_user_devices"
)
async def get_user_devices(user_id: str):
    """Get all devices for a user."""
    try:
        user_sessions = await continuity_manager.state_manager.get_user_sessions(user_id)
        
        # Group by device
        devices = {}
        for session in user_sessions:
            device_id = session.device_id
            if device_id not in devices:
                device_info = continuity_manager.devices.get(device_id)
                devices[device_id] = {
                    "device_id": device_id,
                    "device_type": device_info.device_type.value if device_info else "unknown",
                    "last_seen": device_info.last_seen.isoformat() if device_info else None,
                    "is_online": device_info.is_online if device_info else False,
                    "sessions": []
                }
            
            devices[device_id]["sessions"].append({
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "last_updated": session.last_updated.isoformat(),
                "version": session.version
            })
        
        return {
            "success": True,
            "user_id": user_id,
            "devices": list(devices.values()),
            "total_devices": len(devices),
            "message": "Devices retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Failed to get devices for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=Dict[str, Any])
@never_fail_api_endpoint(
    fallback_response={"success": False, "message": "Failed to delete session"},
    user_action="delete_session"
)
async def delete_session(session_id: str):
    """Delete a session."""
    try:
        # Remove from active sessions
        if session_id in continuity_manager.active_sessions:
            del continuity_manager.active_sessions[session_id]
        
        # Remove from persistent storage
        import os
        file_path = os.path.join(continuity_manager.state_manager.storage_path, f"{session_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Session deleted successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conflicts/{user_id}", response_model=Dict[str, Any])
@never_fail_api_endpoint(
    fallback_response={"success": False, "conflicts": [], "message": "Failed to get conflicts"},
    user_action="get_conflicts"
)
async def get_sync_conflicts(user_id: str):
    """Get synchronization conflicts for a user."""
    try:
        user_conflicts = [
            {
                "conflict_id": conflict.conflict_id,
                "session_id": conflict.session_id,
                "path": conflict.path,
                "local_change": {
                    "timestamp": conflict.local_change.timestamp.isoformat(),
                    "device_id": conflict.local_change.device_id,
                    "value": conflict.local_change.new_value
                },
                "remote_change": {
                    "timestamp": conflict.remote_change.timestamp.isoformat(),
                    "device_id": conflict.remote_change.device_id,
                    "value": conflict.remote_change.new_value
                },
                "resolution_strategy": conflict.resolution_strategy.value,
                "resolved": conflict.resolved,
                "resolution_data": conflict.resolution_data
            }
            for conflict in continuity_manager.sync_conflicts.values()
            if conflict.local_change.user_id == user_id
        ]
        
        return {
            "success": True,
            "user_id": user_id,
            "conflicts": user_conflicts,
            "total_conflicts": len(user_conflicts),
            "pending_conflicts": len([c for c in user_conflicts if not c["resolved"]]),
            "message": "Conflicts retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Failed to get conflicts for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conflicts/{conflict_id}/resolve", response_model=Dict[str, Any])
@never_fail_api_endpoint(
    fallback_response={"success": False, "message": "Failed to resolve conflict"},
    user_action="resolve_conflict"
)
async def resolve_conflict(conflict_id: str, resolution_strategy: str = "automatic_merge"):
    """Resolve a synchronization conflict."""
    try:
        if conflict_id not in continuity_manager.sync_conflicts:
            raise HTTPException(status_code=404, detail="Conflict not found")
        
        conflict = continuity_manager.sync_conflicts[conflict_id]
        
        # Update resolution strategy if provided
        try:
            conflict.resolution_strategy = ConflictResolutionStrategy(resolution_strategy)
        except ValueError:
            conflict.resolution_strategy = ConflictResolutionStrategy.AUTOMATIC_MERGE
        
        # Resolve the conflict
        resolution_data = await continuity_manager.conflict_resolver.resolve_conflict(conflict)
        
        return {
            "success": True,
            "conflict_id": conflict_id,
            "resolution_strategy": conflict.resolution_strategy.value,
            "resolution_data": resolution_data,
            "resolved": conflict.resolved,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve conflict {conflict_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time synchronization
@router.websocket("/ws/{user_id}/{device_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, device_id: str):
    """WebSocket endpoint for real-time cross-device synchronization."""
    await websocket.accept()
    
    try:
        # Send authentication message
        auth_message = {
            "user_id": user_id,
            "device_id": device_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket.send_text(json.dumps(auth_message))
        
        # Handle WebSocket communication
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "state_update":
                    # Update session state
                    session_id = message.get("session_id")
                    state_updates = message.get("state_updates", {})
                    
                    if session_id and state_updates:
                        await sync_user_state(user_id, session_id, state_updates, device_id)
                        
                        # Send confirmation
                        await websocket.send_text(json.dumps({
                            "type": "state_update_confirmed",
                            "session_id": session_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }))
                
                elif message.get("type") == "sync_request":
                    # Perform full sync
                    sync_results = await continuity_manager.sync_across_devices(user_id)
                    
                    await websocket.send_text(json.dumps({
                        "type": "sync_response",
                        "sync_results": sync_results,
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
                elif message.get("type") == "heartbeat":
                    # Respond to heartbeat
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat_response",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for user {user_id}, device {device_id}")
                break
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from WebSocket: {user_id}")
                continue
            except Exception as e:
                logger.error(f"WebSocket error for user {user_id}: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for user {user_id}, device {device_id}")
    except Exception as e:
        logger.error(f"WebSocket endpoint error: {e}")
    finally:
        # Clean up connection
        try:
            await websocket.close()
        except:
            pass


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint for cross-device continuity service."""
    try:
        # Check various components
        health_status = {
            "service": "cross_device_continuity",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "state_manager": "healthy",
                "conflict_resolver": "healthy",
                "offline_manager": "healthy",
                "websocket_manager": "healthy"
            },
            "metrics": {
                "active_sessions": len(continuity_manager.active_sessions),
                "registered_devices": len(continuity_manager.devices),
                "pending_conflicts": len([
                    c for c in continuity_manager.sync_conflicts.values() 
                    if not c.resolved
                ]),
                "offline_queue_size": len(continuity_manager.offline_manager.offline_queue)
            }
        }
        
        # Check if any component is unhealthy
        if continuity_manager.offline_manager.is_offline:
            health_status["components"]["offline_manager"] = "offline_mode"
        
        if len(continuity_manager.sync_conflicts) > 100:  # Too many conflicts
            health_status["components"]["conflict_resolver"] = "degraded"
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "service": "cross_device_continuity",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }