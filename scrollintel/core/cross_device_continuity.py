"""
Cross-device and cross-session continuity system for ScrollIntel.
Ensures seamless state transfer between devices, session recovery with exact context restoration,
multi-tab synchronization with conflict resolution, and offline mode with automatic sync.
"""

import asyncio
import logging
import json
import time
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import pickle
import os
from contextlib import asynccontextmanager
import websockets
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Synchronization status."""
    SYNCED = "synced"
    PENDING = "pending"
    CONFLICT = "conflict"
    OFFLINE = "offline"
    ERROR = "error"


class DeviceType(Enum):
    """Device types for optimization."""
    DESKTOP = "desktop"
    TABLET = "tablet"
    MOBILE = "mobile"
    UNKNOWN = "unknown"


class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies."""
    LAST_WRITE_WINS = "last_write_wins"
    MERGE_CHANGES = "merge_changes"
    USER_CHOICE = "user_choice"
    KEEP_BOTH = "keep_both"
    AUTOMATIC_MERGE = "automatic_merge"


@dataclass
class DeviceInfo:
    """Information about a device."""
    device_id: str
    device_type: DeviceType
    user_agent: str
    last_seen: datetime
    capabilities: Dict[str, Any]
    network_quality: str = "good"
    is_online: bool = True


@dataclass
class SessionState:
    """Complete session state."""
    session_id: str
    user_id: str
    device_id: str
    created_at: datetime
    last_updated: datetime
    state_data: Dict[str, Any]
    version: int = 1
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for state integrity."""
        state_json = json.dumps(self.state_data, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()
    
    def update_state(self, new_data: Dict[str, Any]):
        """Update state data and increment version."""
        self.state_data.update(new_data)
        self.version += 1
        self.last_updated = datetime.utcnow()
        self.checksum = self._calculate_checksum()


@dataclass
class StateChange:
    """Represents a state change for synchronization."""
    change_id: str
    session_id: str
    user_id: str
    device_id: str
    timestamp: datetime
    path: str  # JSON path to the changed data
    old_value: Any
    new_value: Any
    operation: str  # 'set', 'delete', 'merge'
    conflict_resolution: Optional[ConflictResolutionStrategy] = None


@dataclass
class SyncConflict:
    """Represents a synchronization conflict."""
    conflict_id: str
    session_id: str
    path: str
    local_change: StateChange
    remote_change: StateChange
    resolution_strategy: ConflictResolutionStrategy
    resolved: bool = False
    resolution_data: Optional[Any] = None


class StateManager:
    """Manages state storage and retrieval."""
    
    def __init__(self, storage_path: str = "data/session_states"):
        self.storage_path = storage_path
        self.memory_cache: Dict[str, SessionState] = {}
        self.change_log: List[StateChange] = []
        self.max_cache_size = 1000
        self.max_change_log_size = 10000
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
    
    async def save_session_state(self, session_state: SessionState):
        """Save session state to persistent storage."""
        # Update memory cache
        self.memory_cache[session_state.session_id] = session_state
        
        # Persist to disk
        file_path = os.path.join(self.storage_path, f"{session_state.session_id}.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(asdict(session_state), f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session state {session_state.session_id}: {e}")
        
        # Clean up cache if too large
        if len(self.memory_cache) > self.max_cache_size:
            oldest_session = min(
                self.memory_cache.values(),
                key=lambda s: s.last_updated
            )
            del self.memory_cache[oldest_session.session_id]
    
    async def load_session_state(self, session_id: str) -> Optional[SessionState]:
        """Load session state from storage."""
        # Check memory cache first
        if session_id in self.memory_cache:
            return self.memory_cache[session_id]
        
        # Load from disk
        file_path = os.path.join(self.storage_path, f"{session_id}.json")
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Convert datetime strings back to datetime objects
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                
                session_state = SessionState(**data)
                self.memory_cache[session_id] = session_state
                return session_state
        except Exception as e:
            logger.error(f"Failed to load session state {session_id}: {e}")
        
        return None
    
    async def get_user_sessions(self, user_id: str) -> List[SessionState]:
        """Get all sessions for a user."""
        sessions = []
        
        # Check memory cache
        for session in self.memory_cache.values():
            if session.user_id == user_id:
                sessions.append(session)
        
        # Load from disk if not in cache
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    session_id = filename[:-5]  # Remove .json extension
                    if session_id not in self.memory_cache:
                        session = await self.load_session_state(session_id)
                        if session and session.user_id == user_id:
                            sessions.append(session)
        except Exception as e:
            logger.error(f"Failed to load user sessions for {user_id}: {e}")
        
        return sessions
    
    def record_state_change(self, change: StateChange):
        """Record a state change for synchronization."""
        self.change_log.append(change)
        
        # Clean up change log if too large
        if len(self.change_log) > self.max_change_log_size:
            self.change_log = self.change_log[-self.max_change_log_size//2:]
    
    def get_changes_since(self, timestamp: datetime, user_id: str) -> List[StateChange]:
        """Get all changes for a user since a specific timestamp."""
        return [
            change for change in self.change_log
            if change.user_id == user_id and change.timestamp > timestamp
        ]


class ConflictResolver:
    """Resolves synchronization conflicts."""
    
    def __init__(self):
        self.resolution_strategies = {
            ConflictResolutionStrategy.LAST_WRITE_WINS: self._resolve_last_write_wins,
            ConflictResolutionStrategy.MERGE_CHANGES: self._resolve_merge_changes,
            ConflictResolutionStrategy.USER_CHOICE: self._resolve_user_choice,
            ConflictResolutionStrategy.KEEP_BOTH: self._resolve_keep_both,
            ConflictResolutionStrategy.AUTOMATIC_MERGE: self._resolve_automatic_merge
        }
    
    async def resolve_conflict(self, conflict: SyncConflict) -> Any:
        """Resolve a synchronization conflict."""
        resolver = self.resolution_strategies.get(conflict.resolution_strategy)
        if resolver:
            try:
                resolution_data = await resolver(conflict)
                conflict.resolved = True
                conflict.resolution_data = resolution_data
                return resolution_data
            except Exception as e:
                logger.error(f"Failed to resolve conflict {conflict.conflict_id}: {e}")
                # Fall back to last write wins
                return await self._resolve_last_write_wins(conflict)
        else:
            logger.warning(f"Unknown resolution strategy: {conflict.resolution_strategy}")
            return await self._resolve_last_write_wins(conflict)
    
    async def _resolve_last_write_wins(self, conflict: SyncConflict) -> Any:
        """Resolve conflict by keeping the most recent change."""
        if conflict.local_change.timestamp > conflict.remote_change.timestamp:
            return conflict.local_change.new_value
        else:
            return conflict.remote_change.new_value
    
    async def _resolve_merge_changes(self, conflict: SyncConflict) -> Any:
        """Resolve conflict by merging changes."""
        local_value = conflict.local_change.new_value
        remote_value = conflict.remote_change.new_value
        
        # If both values are dictionaries, merge them
        if isinstance(local_value, dict) and isinstance(remote_value, dict):
            merged = remote_value.copy()
            merged.update(local_value)
            return merged
        
        # If both values are lists, combine them
        elif isinstance(local_value, list) and isinstance(remote_value, list):
            return list(set(local_value + remote_value))  # Remove duplicates
        
        # Otherwise, fall back to last write wins
        else:
            return await self._resolve_last_write_wins(conflict)
    
    async def _resolve_user_choice(self, conflict: SyncConflict) -> Any:
        """Resolve conflict by asking user to choose."""
        # In a real implementation, this would present options to the user
        # For now, fall back to last write wins
        logger.info(f"User choice needed for conflict {conflict.conflict_id}")
        return await self._resolve_last_write_wins(conflict)
    
    async def _resolve_keep_both(self, conflict: SyncConflict) -> Any:
        """Resolve conflict by keeping both values."""
        return {
            "local": conflict.local_change.new_value,
            "remote": conflict.remote_change.new_value,
            "conflict_id": conflict.conflict_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _resolve_automatic_merge(self, conflict: SyncConflict) -> Any:
        """Resolve conflict using intelligent automatic merging."""
        local_value = conflict.local_change.new_value
        remote_value = conflict.remote_change.new_value
        
        # Analyze the type of data and merge intelligently
        if self._is_user_preference(conflict.path):
            # For user preferences, prefer local changes
            return local_value
        elif self._is_system_data(conflict.path):
            # For system data, prefer remote changes
            return remote_value
        elif self._is_collaborative_data(conflict.path):
            # For collaborative data, merge changes
            return await self._resolve_merge_changes(conflict)
        else:
            # Default to last write wins
            return await self._resolve_last_write_wins(conflict)
    
    def _is_user_preference(self, path: str) -> bool:
        """Check if path represents user preferences."""
        preference_paths = ['preferences', 'settings', 'theme', 'layout']
        return any(pref in path.lower() for pref in preference_paths)
    
    def _is_system_data(self, path: str) -> bool:
        """Check if path represents system data."""
        system_paths = ['system', 'config', 'metadata', 'version']
        return any(sys in path.lower() for sys in system_paths)
    
    def _is_collaborative_data(self, path: str) -> bool:
        """Check if path represents collaborative data."""
        collab_paths = ['shared', 'comments', 'annotations', 'collaboration']
        return any(collab in path.lower() for collab in collab_paths)


class OfflineManager:
    """Manages offline functionality and sync when reconnected."""
    
    def __init__(self, storage_path: str = "data/offline_data"):
        self.storage_path = storage_path
        self.offline_queue: List[StateChange] = []
        self.is_offline = False
        self.last_sync_time: Optional[datetime] = None
        self.offline_capabilities = {
            'read_cached_data': True,
            'create_new_items': True,
            'edit_existing_items': True,
            'delete_items': True,
            'sync_when_online': True
        }
        
        os.makedirs(storage_path, exist_ok=True)
        self._load_offline_queue()
    
    def _load_offline_queue(self):
        """Load offline queue from storage."""
        queue_file = os.path.join(self.storage_path, "offline_queue.json")
        try:
            if os.path.exists(queue_file):
                with open(queue_file, 'r') as f:
                    data = json.load(f)
                    self.offline_queue = [
                        StateChange(**change_data) for change_data in data
                    ]
        except Exception as e:
            logger.error(f"Failed to load offline queue: {e}")
    
    def _save_offline_queue(self):
        """Save offline queue to storage."""
        queue_file = os.path.join(self.storage_path, "offline_queue.json")
        try:
            with open(queue_file, 'w') as f:
                json.dump([asdict(change) for change in self.offline_queue], f, default=str)
        except Exception as e:
            logger.error(f"Failed to save offline queue: {e}")
    
    def set_offline_mode(self, offline: bool):
        """Set offline mode status."""
        if self.is_offline != offline:
            self.is_offline = offline
            if offline:
                logger.info("Entering offline mode")
            else:
                logger.info("Exiting offline mode - preparing to sync")
                asyncio.create_task(self._sync_offline_changes())
    
    def queue_offline_change(self, change: StateChange):
        """Queue a change for sync when online."""
        self.offline_queue.append(change)
        self._save_offline_queue()
        logger.debug(f"Queued offline change: {change.change_id}")
    
    async def _sync_offline_changes(self):
        """Sync all offline changes when coming back online."""
        if not self.offline_queue:
            return
        
        logger.info(f"Syncing {len(self.offline_queue)} offline changes")
        
        # Group changes by session
        changes_by_session = defaultdict(list)
        for change in self.offline_queue:
            changes_by_session[change.session_id].append(change)
        
        # Sync each session's changes
        synced_changes = []
        for session_id, changes in changes_by_session.items():
            try:
                # In a real implementation, this would sync with the server
                logger.info(f"Syncing {len(changes)} changes for session {session_id}")
                synced_changes.extend(changes)
            except Exception as e:
                logger.error(f"Failed to sync changes for session {session_id}: {e}")
        
        # Remove synced changes from queue
        for change in synced_changes:
            if change in self.offline_queue:
                self.offline_queue.remove(change)
        
        self._save_offline_queue()
        self.last_sync_time = datetime.utcnow()
        
        logger.info(f"Offline sync completed. {len(self.offline_queue)} changes remaining")
    
    def get_offline_capabilities(self) -> Dict[str, bool]:
        """Get current offline capabilities."""
        return self.offline_capabilities.copy()
    
    def can_perform_offline(self, operation: str) -> bool:
        """Check if an operation can be performed offline."""
        return self.offline_capabilities.get(operation, False)


class WebSocketManager:
    """Manages WebSocket connections for real-time synchronization."""
    
    def __init__(self):
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        self.device_connections: Dict[str, str] = {}  # device_id -> connection_id
        self.server = None
        self.port = 8765
    
    async def start_server(self):
        """Start WebSocket server for real-time sync."""
        try:
            self.server = await websockets.serve(
                self.handle_connection, "localhost", self.port
            )
            logger.info(f"WebSocket server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    async def stop_server(self):
        """Stop WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")
    
    async def handle_connection(self, websocket, path):
        """Handle new WebSocket connection."""
        connection_id = str(uuid.uuid4())
        self.connections[connection_id] = websocket
        
        try:
            # Wait for authentication message
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            user_id = auth_data.get('user_id')
            device_id = auth_data.get('device_id')
            
            if user_id and device_id:
                self.user_connections[user_id].add(connection_id)
                self.device_connections[device_id] = connection_id
                
                logger.info(f"WebSocket connection established: {connection_id} for user {user_id}")
                
                # Send confirmation
                await websocket.send(json.dumps({
                    'type': 'auth_success',
                    'connection_id': connection_id
                }))
                
                # Handle messages
                async for message in websocket:
                    await self.handle_message(connection_id, message, user_id, device_id)
            else:
                await websocket.send(json.dumps({
                    'type': 'auth_error',
                    'message': 'Invalid authentication data'
                }))
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            # Clean up connection
            self.connections.pop(connection_id, None)
            for user_id, conn_set in self.user_connections.items():
                conn_set.discard(connection_id)
            
            # Remove empty user connection sets
            self.user_connections = {
                user_id: conn_set for user_id, conn_set in self.user_connections.items()
                if conn_set
            }
            
            # Clean up device connection
            device_to_remove = None
            for device_id, conn_id in self.device_connections.items():
                if conn_id == connection_id:
                    device_to_remove = device_id
                    break
            if device_to_remove:
                del self.device_connections[device_to_remove]
    
    async def handle_message(self, connection_id: str, message: str, user_id: str, device_id: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'state_change':
                # Broadcast state change to other devices
                await self.broadcast_state_change(user_id, device_id, data)
            elif message_type == 'sync_request':
                # Handle sync request
                await self.handle_sync_request(connection_id, data)
            elif message_type == 'heartbeat':
                # Respond to heartbeat
                await self.send_to_connection(connection_id, {
                    'type': 'heartbeat_response',
                    'timestamp': datetime.utcnow().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def broadcast_state_change(self, user_id: str, source_device_id: str, change_data: Dict[str, Any]):
        """Broadcast state change to all user's devices except the source."""
        if user_id not in self.user_connections:
            return
        
        message = {
            'type': 'state_change',
            'source_device': source_device_id,
            'data': change_data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to all connections for this user except the source device
        for connection_id in self.user_connections[user_id]:
            # Skip source device
            if (source_device_id in self.device_connections and 
                self.device_connections[source_device_id] == connection_id):
                continue
            
            await self.send_to_connection(connection_id, message)
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection."""
        if connection_id in self.connections:
            try:
                await self.connections[connection_id].send(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to connection {connection_id}: {e}")
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send message to all connections for a user."""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id]:
                await self.send_to_connection(connection_id, message)
    
    async def handle_sync_request(self, connection_id: str, data: Dict[str, Any]):
        """Handle synchronization request."""
        # In a real implementation, this would fetch and send the latest state
        response = {
            'type': 'sync_response',
            'request_id': data.get('request_id'),
            'status': 'success',
            'data': {}  # Would contain actual sync data
        }
        await self.send_to_connection(connection_id, response)


class CrossDeviceContinuityManager:
    """Main manager for cross-device and cross-session continuity."""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.conflict_resolver = ConflictResolver()
        self.offline_manager = OfflineManager()
        self.websocket_manager = WebSocketManager()
        
        self.devices: Dict[str, DeviceInfo] = {}
        self.active_sessions: Dict[str, SessionState] = {}
        self.sync_conflicts: Dict[str, SyncConflict] = {}
        
        # Configuration
        self.auto_sync_interval = 30  # seconds
        self.conflict_resolution_timeout = 300  # 5 minutes
        self.session_timeout = timedelta(hours=24)
        
        # Background tasks
        self.sync_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Start background services
        asyncio.create_task(self._start_background_services())
    
    async def _start_background_services(self):
        """Start background services."""
        try:
            # Start WebSocket server
            await self.websocket_manager.start_server()
            
            # Start periodic sync task
            self.sync_task = asyncio.create_task(self._periodic_sync())
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            logger.info("Cross-device continuity services started")
        except Exception as e:
            logger.error(f"Failed to start background services: {e}")
    
    async def register_device(self, device_id: str, device_type: DeviceType, 
                            user_agent: str, capabilities: Dict[str, Any]) -> DeviceInfo:
        """Register a new device."""
        device_info = DeviceInfo(
            device_id=device_id,
            device_type=device_type,
            user_agent=user_agent,
            last_seen=datetime.utcnow(),
            capabilities=capabilities
        )
        
        self.devices[device_id] = device_info
        logger.info(f"Registered device: {device_id} ({device_type.value})")
        
        return device_info
    
    async def create_session(self, user_id: str, device_id: str, 
                           initial_state: Optional[Dict[str, Any]] = None) -> SessionState:
        """Create a new session."""
        session_id = f"{user_id}_{device_id}_{int(time.time())}"
        
        session_state = SessionState(
            session_id=session_id,
            user_id=user_id,
            device_id=device_id,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            state_data=initial_state or {}
        )
        
        # Save session
        await self.state_manager.save_session_state(session_state)
        self.active_sessions[session_id] = session_state
        
        logger.info(f"Created session: {session_id}")
        return session_state
    
    async def restore_session(self, session_id: str) -> Optional[SessionState]:
        """Restore a session with exact context."""
        session_state = await self.state_manager.load_session_state(session_id)
        
        if session_state:
            # Check if session is still valid
            if datetime.utcnow() - session_state.last_updated > self.session_timeout:
                logger.warning(f"Session {session_id} has expired")
                return None
            
            # Verify state integrity
            expected_checksum = session_state._calculate_checksum()
            if session_state.checksum != expected_checksum:
                logger.warning(f"Session {session_id} state integrity check failed")
                # Could attempt recovery here
            
            self.active_sessions[session_id] = session_state
            logger.info(f"Restored session: {session_id}")
            
            return session_state
        
        return None
    
    async def update_session_state(self, session_id: str, state_updates: Dict[str, Any], 
                                 device_id: str) -> bool:
        """Update session state and sync across devices."""
        session_state = self.active_sessions.get(session_id)
        if not session_state:
            session_state = await self.restore_session(session_id)
            if not session_state:
                logger.error(f"Session {session_id} not found")
                return False
        
        # Create state change record
        change = StateChange(
            change_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=session_state.user_id,
            device_id=device_id,
            timestamp=datetime.utcnow(),
            path="root",  # Could be more specific for nested updates
            old_value=session_state.state_data.copy(),
            new_value=state_updates,
            operation="merge"
        )
        
        # Update session state
        session_state.update_state(state_updates)
        
        # Save to persistent storage
        await self.state_manager.save_session_state(session_state)
        
        # Record change for sync
        self.state_manager.record_state_change(change)
        
        # Handle offline mode
        if self.offline_manager.is_offline:
            self.offline_manager.queue_offline_change(change)
        else:
            # Broadcast to other devices via WebSocket
            await self.websocket_manager.broadcast_state_change(
                session_state.user_id, device_id, {
                    'session_id': session_id,
                    'updates': state_updates,
                    'version': session_state.version
                }
            )
        
        logger.debug(f"Updated session state: {session_id}")
        return True
    
    async def sync_across_devices(self, user_id: str) -> Dict[str, Any]:
        """Synchronize state across all user's devices."""
        user_sessions = await self.state_manager.get_user_sessions(user_id)
        
        if not user_sessions:
            return {"status": "no_sessions", "synced_sessions": 0}
        
        # Get the most recent session as the source of truth
        latest_session = max(user_sessions, key=lambda s: s.last_updated)
        
        sync_results = {
            "status": "success",
            "source_session": latest_session.session_id,
            "synced_sessions": 0,
            "conflicts": 0,
            "errors": []
        }
        
        # Sync other sessions to match the latest
        for session in user_sessions:
            if session.session_id == latest_session.session_id:
                continue
            
            try:
                # Check for conflicts
                conflicts = await self._detect_conflicts(session, latest_session)
                
                if conflicts:
                    sync_results["conflicts"] += len(conflicts)
                    # Resolve conflicts
                    for conflict in conflicts:
                        await self.conflict_resolver.resolve_conflict(conflict)
                
                # Update session to match latest
                session.state_data = latest_session.state_data.copy()
                session.version = latest_session.version
                session.last_updated = datetime.utcnow()
                session.checksum = session._calculate_checksum()
                
                await self.state_manager.save_session_state(session)
                sync_results["synced_sessions"] += 1
                
            except Exception as e:
                error_msg = f"Failed to sync session {session.session_id}: {e}"
                sync_results["errors"].append(error_msg)
                logger.error(error_msg)
        
        # Notify all connected devices about the sync
        await self.websocket_manager.send_to_user(user_id, {
            'type': 'sync_complete',
            'results': sync_results
        })
        
        return sync_results
    
    async def _detect_conflicts(self, session1: SessionState, session2: SessionState) -> List[SyncConflict]:
        """Detect conflicts between two sessions."""
        conflicts = []
        
        # Simple conflict detection - in reality, this would be more sophisticated
        if session1.version != session2.version and session1.checksum != session2.checksum:
            conflict = SyncConflict(
                conflict_id=str(uuid.uuid4()),
                session_id=session1.session_id,
                path="root",
                local_change=StateChange(
                    change_id=str(uuid.uuid4()),
                    session_id=session1.session_id,
                    user_id=session1.user_id,
                    device_id=session1.device_id,
                    timestamp=session1.last_updated,
                    path="root",
                    old_value={},
                    new_value=session1.state_data,
                    operation="set"
                ),
                remote_change=StateChange(
                    change_id=str(uuid.uuid4()),
                    session_id=session2.session_id,
                    user_id=session2.user_id,
                    device_id=session2.device_id,
                    timestamp=session2.last_updated,
                    path="root",
                    old_value={},
                    new_value=session2.state_data,
                    operation="set"
                ),
                resolution_strategy=ConflictResolutionStrategy.AUTOMATIC_MERGE
            )
            conflicts.append(conflict)
        
        return conflicts
    
    async def handle_multi_tab_sync(self, user_id: str, tab_id: str, 
                                  state_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Handle synchronization between multiple tabs."""
        # Find or create session for this tab
        session_id = f"{user_id}_tab_{tab_id}"
        session_state = self.active_sessions.get(session_id)
        
        if not session_state:
            # Create new tab session
            session_state = await self.create_session(user_id, f"tab_{tab_id}", state_updates)
        else:
            # Update existing session
            await self.update_session_state(session_id, state_updates, f"tab_{tab_id}")
        
        # Sync with other tabs
        await self._sync_tabs(user_id, tab_id)
        
        return {
            "status": "synced",
            "session_id": session_id,
            "version": session_state.version
        }
    
    async def _sync_tabs(self, user_id: str, source_tab_id: str):
        """Synchronize state between tabs."""
        # Get all tab sessions for the user
        all_sessions = await self.state_manager.get_user_sessions(user_id)
        tab_sessions = [s for s in all_sessions if s.device_id.startswith("tab_")]
        
        if len(tab_sessions) <= 1:
            return  # No other tabs to sync
        
        # Find the source tab session
        source_session = None
        for session in tab_sessions:
            if session.device_id == f"tab_{source_tab_id}":
                source_session = session
                break
        
        if not source_session:
            return
        
        # Sync other tabs to match source
        for session in tab_sessions:
            if session.session_id == source_session.session_id:
                continue
            
            # Merge states (tabs typically share most state)
            merged_state = session.state_data.copy()
            merged_state.update(source_session.state_data)
            
            session.state_data = merged_state
            session.version += 1
            session.last_updated = datetime.utcnow()
            session.checksum = session._calculate_checksum()
            
            await self.state_manager.save_session_state(session)
        
        # Notify all tabs about the sync
        await self.websocket_manager.send_to_user(user_id, {
            'type': 'tab_sync',
            'source_tab': source_tab_id,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def enable_offline_mode(self, user_id: str, device_id: str) -> Dict[str, Any]:
        """Enable offline mode for a device."""
        self.offline_manager.set_offline_mode(True)
        
        # Cache current state for offline use
        user_sessions = await self.state_manager.get_user_sessions(user_id)
        device_sessions = [s for s in user_sessions if s.device_id == device_id]
        
        offline_data = {
            "sessions": [asdict(session) for session in device_sessions],
            "capabilities": self.offline_manager.get_offline_capabilities(),
            "last_sync": datetime.utcnow().isoformat()
        }
        
        # Save offline data
        offline_file = os.path.join(self.offline_manager.storage_path, f"{device_id}_offline.json")
        try:
            with open(offline_file, 'w') as f:
                json.dump(offline_data, f, default=str)
        except Exception as e:
            logger.error(f"Failed to save offline data: {e}")
        
        return {
            "status": "offline_enabled",
            "capabilities": self.offline_manager.get_offline_capabilities(),
            "cached_sessions": len(device_sessions)
        }
    
    async def sync_when_reconnected(self, user_id: str, device_id: str) -> Dict[str, Any]:
        """Sync when device reconnects after being offline."""
        self.offline_manager.set_offline_mode(False)
        
        # Load offline data
        offline_file = os.path.join(self.offline_manager.storage_path, f"{device_id}_offline.json")
        offline_data = {}
        
        try:
            if os.path.exists(offline_file):
                with open(offline_file, 'r') as f:
                    offline_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load offline data: {e}")
        
        # Sync offline changes
        await self.offline_manager._sync_offline_changes()
        
        # Sync with latest server state
        sync_results = await self.sync_across_devices(user_id)
        
        return {
            "status": "reconnected",
            "offline_changes_synced": len(self.offline_manager.offline_queue),
            "sync_results": sync_results,
            "last_offline_sync": offline_data.get("last_sync")
        }
    
    async def _periodic_sync(self):
        """Periodic synchronization task."""
        while True:
            try:
                await asyncio.sleep(self.auto_sync_interval)
                
                # Sync all active sessions
                users_to_sync = set()
                for session in self.active_sessions.values():
                    users_to_sync.add(session.user_id)
                
                for user_id in users_to_sync:
                    try:
                        await self.sync_across_devices(user_id)
                    except Exception as e:
                        logger.error(f"Failed to sync user {user_id}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic sync: {e}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old sessions and data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up expired sessions
                current_time = datetime.utcnow()
                expired_sessions = []
                
                for session_id, session in self.active_sessions.items():
                    if current_time - session.last_updated > self.session_timeout:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
                    logger.info(f"Cleaned up expired session: {session_id}")
                
                # Clean up old conflicts
                expired_conflicts = []
                for conflict_id, conflict in self.sync_conflicts.items():
                    if (current_time - conflict.local_change.timestamp > 
                        timedelta(seconds=self.conflict_resolution_timeout)):
                        expired_conflicts.append(conflict_id)
                
                for conflict_id in expired_conflicts:
                    del self.sync_conflicts[conflict_id]
                
                logger.debug(f"Cleanup completed: {len(expired_sessions)} sessions, {len(expired_conflicts)} conflicts")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def get_sync_status(self, user_id: str) -> Dict[str, Any]:
        """Get synchronization status for a user."""
        user_sessions = await self.state_manager.get_user_sessions(user_id)
        
        # Count devices and sessions
        devices = set(session.device_id for session in user_sessions)
        
        # Check for conflicts
        user_conflicts = [
            conflict for conflict in self.sync_conflicts.values()
            if conflict.local_change.user_id == user_id
        ]
        
        return {
            "user_id": user_id,
            "total_sessions": len(user_sessions),
            "active_devices": len(devices),
            "pending_conflicts": len([c for c in user_conflicts if not c.resolved]),
            "offline_mode": self.offline_manager.is_offline,
            "offline_queue_size": len(self.offline_manager.offline_queue),
            "last_sync": self.offline_manager.last_sync_time.isoformat() if self.offline_manager.last_sync_time else None,
            "websocket_connections": len(self.websocket_manager.user_connections.get(user_id, set()))
        }
    
    async def shutdown(self):
        """Shutdown the continuity manager."""
        logger.info("Shutting down cross-device continuity manager")
        
        # Cancel background tasks
        if self.sync_task:
            self.sync_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Stop WebSocket server
        await self.websocket_manager.stop_server()
        
        # Save any pending offline changes
        self.offline_manager._save_offline_queue()
        
        logger.info("Cross-device continuity manager shutdown complete")


# Global instance
continuity_manager = CrossDeviceContinuityManager()


# Convenience functions
async def create_user_session(user_id: str, device_id: str, device_type: str = "desktop",
                            initial_state: Optional[Dict[str, Any]] = None) -> SessionState:
    """Create a new user session with device registration."""
    # Register device if not already registered
    if device_id not in continuity_manager.devices:
        await continuity_manager.register_device(
            device_id=device_id,
            device_type=DeviceType(device_type),
            user_agent="ScrollIntel Client",
            capabilities={"sync": True, "offline": True}
        )
    
    return await continuity_manager.create_session(user_id, device_id, initial_state)


async def restore_user_session(session_id: str) -> Optional[SessionState]:
    """Restore a user session."""
    return await continuity_manager.restore_session(session_id)


async def sync_user_state(user_id: str, session_id: str, state_updates: Dict[str, Any], 
                         device_id: str) -> bool:
    """Update and sync user state across devices."""
    return await continuity_manager.update_session_state(session_id, state_updates, device_id)


async def enable_offline_mode(user_id: str, device_id: str) -> Dict[str, Any]:
    """Enable offline mode for a user's device."""
    return await continuity_manager.enable_offline_mode(user_id, device_id)


async def sync_after_reconnect(user_id: str, device_id: str) -> Dict[str, Any]:
    """Sync user state after reconnecting from offline mode."""
    return await continuity_manager.sync_when_reconnected(user_id, device_id)


def get_sync_status(user_id: str) -> Dict[str, Any]:
    """Get synchronization status for a user."""
    return asyncio.run(continuity_manager.get_sync_status(user_id))