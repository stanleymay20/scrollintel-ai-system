"""
Comprehensive offline data management system for ScrollIntel.
Provides robust offline-first architecture with intelligent sync and conflict resolution.
"""

import asyncio
import logging
import json
import sqlite3
import hashlib
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from contextlib import asynccontextmanager
import pickle
import gzip
import base64

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Status of data synchronization."""
    SYNCED = "synced"
    PENDING_SYNC = "pending_sync"
    CONFLICT = "conflict"
    SYNC_FAILED = "sync_failed"
    OFFLINE_ONLY = "offline_only"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving sync conflicts."""
    CLIENT_WINS = "client_wins"
    SERVER_WINS = "server_wins"
    MERGE = "merge"
    USER_CHOICE = "user_choice"
    TIMESTAMP_BASED = "timestamp_based"
    INTELLIGENT_MERGE = "intelligent_merge"


class DataOperation(Enum):
    """Types of data operations."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    READ = "read"


@dataclass
class OfflineDataEntry:
    """Represents a data entry in offline storage."""
    id: str
    collection: str
    data: Dict[str, Any]
    operation: DataOperation
    timestamp: datetime
    sync_status: SyncStatus = SyncStatus.PENDING_SYNC
    version: int = 1
    checksum: str = ""
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for data integrity."""
        data_str = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def is_valid(self) -> bool:
        """Verify data integrity."""
        return self.checksum == self._calculate_checksum()


@dataclass
class SyncConflict:
    """Represents a synchronization conflict."""
    id: str
    collection: str
    local_entry: OfflineDataEntry
    remote_entry: Dict[str, Any]
    conflict_type: str
    detected_at: datetime
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    resolved: bool = False
    resolution_data: Optional[Dict[str, Any]] = None


@dataclass
class SyncOperation:
    """Represents a sync operation."""
    id: str
    operation_type: str
    collection: str
    entry_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class OfflineDataManager:
    """Comprehensive offline data management system."""
    
    def __init__(self, db_path: str = "data/offline_data.db", 
                 cache_size_mb: int = 100, auto_sync: bool = True):
        self.db_path = Path(db_path)
        self.cache_size_mb = cache_size_mb
        self.auto_sync = auto_sync
        
        # In-memory cache for fast access
        self.memory_cache: Dict[str, Dict[str, OfflineDataEntry]] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.max_cache_entries = 10000
        
        # Sync management
        self.sync_queue: List[SyncOperation] = []
        self.conflicts: List[SyncConflict] = []
        self.sync_callbacks: List[Callable] = []
        self.conflict_resolvers: Dict[str, Callable] = {}
        
        # Connection management
        self.db_connection: Optional[sqlite3.Connection] = None
        self.connection_lock = threading.Lock()
        
        # Background sync
        self.sync_task: Optional[asyncio.Task] = None
        self.sync_interval = 30  # seconds
        self.is_online = True
        self.last_sync_attempt = datetime.now()
        
        # Statistics
        self.stats = {
            'total_entries': 0,
            'pending_sync': 0,
            'conflicts': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Initialize database and start background tasks
        self._initialize_database()
        self._setup_conflict_resolvers()
        if auto_sync:
            self._start_background_sync()
    
    def _initialize_database(self):
        """Initialize SQLite database for offline storage."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            # Main data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS offline_data (
                    id TEXT PRIMARY KEY,
                    collection TEXT NOT NULL,
                    data TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    sync_status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    user_id TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Sync operations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_operations (
                    id TEXT PRIMARY KEY,
                    operation_type TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    entry_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0
                )
            """)
            
            # Conflicts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_conflicts (
                    id TEXT PRIMARY KEY,
                    collection TEXT NOT NULL,
                    local_data TEXT NOT NULL,
                    remote_data TEXT NOT NULL,
                    conflict_type TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    resolution_strategy TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_data TEXT
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_collection ON offline_data(collection)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sync_status ON offline_data(sync_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON offline_data(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON offline_data(user_id)")
            
            conn.commit()
        
        logger.info(f"Initialized offline database at {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper locking."""
        with self.connection_lock:
            if self.db_connection is None or self.db_connection.execute("SELECT 1").fetchone() is None:
                self.db_connection = sqlite3.connect(
                    str(self.db_path), 
                    check_same_thread=False,
                    timeout=30.0
                )
                self.db_connection.row_factory = sqlite3.Row
            return self.db_connection
    
    def _setup_conflict_resolvers(self):
        """Setup default conflict resolution strategies."""
        self.conflict_resolvers = {
            ConflictResolutionStrategy.CLIENT_WINS.value: self._resolve_client_wins,
            ConflictResolutionStrategy.SERVER_WINS.value: self._resolve_server_wins,
            ConflictResolutionStrategy.TIMESTAMP_BASED.value: self._resolve_timestamp_based,
            ConflictResolutionStrategy.INTELLIGENT_MERGE.value: self._resolve_intelligent_merge,
            ConflictResolutionStrategy.MERGE.value: self._resolve_merge
        }
    
    async def store_data(self, collection: str, data: Dict[str, Any], 
                        operation: DataOperation = DataOperation.CREATE,
                        user_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store data in offline storage."""
        entry_id = data.get('id', str(uuid.uuid4()))
        
        entry = OfflineDataEntry(
            id=entry_id,
            collection=collection,
            data=data,
            operation=operation,
            timestamp=datetime.now(),
            user_id=user_id,
            metadata=metadata or {}
        )
        
        # Store in database
        await self._store_entry_to_db(entry)
        
        # Update memory cache
        self._update_memory_cache(entry)
        
        # Queue for sync if online
        if self.is_online and self.auto_sync:
            await self._queue_for_sync(entry)
        
        self.stats['total_entries'] += 1
        if entry.sync_status == SyncStatus.PENDING_SYNC:
            self.stats['pending_sync'] += 1
        
        logger.debug(f"Stored data entry {entry_id} in collection {collection}")
        return entry_id
    
    async def get_data(self, collection: str, entry_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from offline storage."""
        # Check memory cache first
        cache_key = f"{collection}:{entry_id}"
        if cache_key in self.memory_cache.get(collection, {}):
            self.cache_access_times[cache_key] = time.time()
            self.stats['cache_hits'] += 1
            entry = self.memory_cache[collection][entry_id]
            return entry.data if entry.is_valid() else None
        
        # Load from database
        entry = await self._load_entry_from_db(collection, entry_id)
        if entry:
            self._update_memory_cache(entry)
            self.stats['cache_misses'] += 1
            return entry.data if entry.is_valid() else None
        
        return None
    
    async def query_data(self, collection: str, filters: Optional[Dict[str, Any]] = None,
                        limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Query data from offline storage with filters."""
        entries = await self._query_entries_from_db(collection, filters, limit, offset)
        
        results = []
        for entry in entries:
            if entry.is_valid():
                # Update cache
                self._update_memory_cache(entry)
                results.append({
                    **entry.data,
                    '_offline_metadata': {
                        'sync_status': entry.sync_status.value,
                        'timestamp': entry.timestamp.isoformat(),
                        'version': entry.version
                    }
                })
        
        return results
    
    async def update_data(self, collection: str, entry_id: str, 
                         updates: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """Update existing data in offline storage."""
        existing_entry = await self._load_entry_from_db(collection, entry_id)
        if not existing_entry:
            return False
        
        # Merge updates with existing data
        updated_data = {**existing_entry.data, **updates}
        
        # Create new entry with updated data
        updated_entry = OfflineDataEntry(
            id=entry_id,
            collection=collection,
            data=updated_data,
            operation=DataOperation.UPDATE,
            timestamp=datetime.now(),
            version=existing_entry.version + 1,
            user_id=user_id or existing_entry.user_id,
            metadata=existing_entry.metadata
        )
        
        # Store updated entry
        await self._store_entry_to_db(updated_entry)
        self._update_memory_cache(updated_entry)
        
        # Queue for sync
        if self.is_online and self.auto_sync:
            await self._queue_for_sync(updated_entry)
        
        return True
    
    async def delete_data(self, collection: str, entry_id: str, 
                         user_id: Optional[str] = None) -> bool:
        """Mark data as deleted in offline storage."""
        existing_entry = await self._load_entry_from_db(collection, entry_id)
        if not existing_entry:
            return False
        
        # Create deletion entry
        delete_entry = OfflineDataEntry(
            id=entry_id,
            collection=collection,
            data={'_deleted': True, 'original_data': existing_entry.data},
            operation=DataOperation.DELETE,
            timestamp=datetime.now(),
            version=existing_entry.version + 1,
            user_id=user_id or existing_entry.user_id,
            metadata=existing_entry.metadata
        )
        
        # Store deletion entry
        await self._store_entry_to_db(delete_entry)
        
        # Remove from memory cache
        if collection in self.memory_cache and entry_id in self.memory_cache[collection]:
            del self.memory_cache[collection][entry_id]
        
        # Queue for sync
        if self.is_online and self.auto_sync:
            await self._queue_for_sync(delete_entry)
        
        return True
    
    async def _store_entry_to_db(self, entry: OfflineDataEntry):
        """Store entry to SQLite database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO offline_data 
                (id, collection, data, operation, timestamp, sync_status, version, 
                 checksum, user_id, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                entry.id,
                entry.collection,
                json.dumps(entry.data),
                entry.operation.value,
                entry.timestamp.isoformat(),
                entry.sync_status.value,
                entry.version,
                entry.checksum,
                entry.user_id,
                json.dumps(entry.metadata)
            ))
            conn.commit()
    
    async def _load_entry_from_db(self, collection: str, entry_id: str) -> Optional[OfflineDataEntry]:
        """Load entry from SQLite database."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM offline_data 
                WHERE collection = ? AND id = ?
                ORDER BY version DESC LIMIT 1
            """, (collection, entry_id)).fetchone()
            
            if row:
                return OfflineDataEntry(
                    id=row['id'],
                    collection=row['collection'],
                    data=json.loads(row['data']),
                    operation=DataOperation(row['operation']),
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    sync_status=SyncStatus(row['sync_status']),
                    version=row['version'],
                    checksum=row['checksum'],
                    user_id=row['user_id'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
        
        return None
    
    async def _query_entries_from_db(self, collection: str, 
                                   filters: Optional[Dict[str, Any]] = None,
                                   limit: Optional[int] = None, 
                                   offset: int = 0) -> List[OfflineDataEntry]:
        """Query entries from SQLite database."""
        query = "SELECT * FROM offline_data WHERE collection = ?"
        params = [collection]
        
        # Add filters (basic implementation)
        if filters:
            for key, value in filters.items():
                if key == 'user_id':
                    query += " AND user_id = ?"
                    params.append(value)
                elif key == 'sync_status':
                    query += " AND sync_status = ?"
                    params.append(value)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        if offset:
            query += " OFFSET ?"
            params.append(offset)
        
        entries = []
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            
            for row in rows:
                entry = OfflineDataEntry(
                    id=row['id'],
                    collection=row['collection'],
                    data=json.loads(row['data']),
                    operation=DataOperation(row['operation']),
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    sync_status=SyncStatus(row['sync_status']),
                    version=row['version'],
                    checksum=row['checksum'],
                    user_id=row['user_id'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                entries.append(entry)
        
        return entries
    
    def _update_memory_cache(self, entry: OfflineDataEntry):
        """Update in-memory cache with entry."""
        if entry.collection not in self.memory_cache:
            self.memory_cache[entry.collection] = {}
        
        self.memory_cache[entry.collection][entry.id] = entry
        cache_key = f"{entry.collection}:{entry.id}"
        self.cache_access_times[cache_key] = time.time()
        
        # Manage cache size
        self._manage_cache_size()
    
    def _manage_cache_size(self):
        """Manage memory cache size by removing least recently used entries."""
        total_entries = sum(len(collection) for collection in self.memory_cache.values())
        
        if total_entries > self.max_cache_entries:
            # Sort by access time and remove oldest
            sorted_keys = sorted(self.cache_access_times.items(), key=lambda x: x[1])
            keys_to_remove = sorted_keys[:total_entries - self.max_cache_entries]
            
            for cache_key, _ in keys_to_remove:
                collection, entry_id = cache_key.split(':', 1)
                if collection in self.memory_cache and entry_id in self.memory_cache[collection]:
                    del self.memory_cache[collection][entry_id]
                del self.cache_access_times[cache_key]
    
    async def _queue_for_sync(self, entry: OfflineDataEntry):
        """Queue entry for synchronization."""
        sync_op = SyncOperation(
            id=str(uuid.uuid4()),
            operation_type=entry.operation.value,
            collection=entry.collection,
            entry_id=entry.id,
            status="queued",
            started_at=datetime.now()
        )
        
        self.sync_queue.append(sync_op)
        
        # Store sync operation in database
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO sync_operations 
                (id, operation_type, collection, entry_id, status, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                sync_op.id,
                sync_op.operation_type,
                sync_op.collection,
                sync_op.entry_id,
                sync_op.status,
                sync_op.started_at.isoformat()
            ))
            conn.commit()
    
    def _start_background_sync(self):
        """Start background synchronization task."""
        if self.sync_task is None or self.sync_task.done():
            self.sync_task = asyncio.create_task(self._background_sync_loop())
    
    async def _background_sync_loop(self):
        """Background loop for automatic synchronization."""
        while True:
            try:
                if self.is_online and self.sync_queue:
                    await self._process_sync_queue()
                
                # Check for conflicts to resolve
                await self._process_pending_conflicts()
                
                # Clean up old sync operations
                await self._cleanup_old_sync_operations()
                
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Background sync error: {e}")
                await asyncio.sleep(self.sync_interval * 2)  # Back off on error
    
    async def _process_sync_queue(self):
        """Process queued sync operations."""
        if not self.sync_queue:
            return
        
        # Process up to 10 operations at a time
        operations_to_process = self.sync_queue[:10]
        
        for sync_op in operations_to_process:
            try:
                await self._execute_sync_operation(sync_op)
                self.sync_queue.remove(sync_op)
                self.stats['successful_syncs'] += 1
                
            except Exception as e:
                logger.error(f"Sync operation failed: {e}")
                sync_op.retry_count += 1
                sync_op.error_message = str(e)
                
                # Remove from queue if too many retries
                if sync_op.retry_count >= 3:
                    self.sync_queue.remove(sync_op)
                    self.stats['failed_syncs'] += 1
                else:
                    # Move to end of queue for retry
                    self.sync_queue.remove(sync_op)
                    self.sync_queue.append(sync_op)
    
    async def _execute_sync_operation(self, sync_op: SyncOperation):
        """Execute a single sync operation."""
        # Load the entry to sync
        entry = await self._load_entry_from_db(sync_op.collection, sync_op.entry_id)
        if not entry:
            raise ValueError(f"Entry not found: {sync_op.entry_id}")
        
        # Simulate API call to server (replace with actual implementation)
        success = await self._sync_with_server(entry)
        
        if success:
            # Update sync status
            entry.sync_status = SyncStatus.SYNCED
            await self._store_entry_to_db(entry)
            self._update_memory_cache(entry)
            
            # Update sync operation
            sync_op.status = "completed"
            sync_op.completed_at = datetime.now()
            
            # Notify callbacks
            for callback in self.sync_callbacks:
                try:
                    await callback(entry, "synced")
                except Exception as e:
                    logger.error(f"Sync callback error: {e}")
        else:
            raise Exception("Server sync failed")
    
    async def _sync_with_server(self, entry: OfflineDataEntry) -> bool:
        """Sync entry with server (mock implementation)."""
        # This would be replaced with actual API calls
        # For now, simulate success/failure
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Simulate occasional conflicts
        if entry.collection == "test_conflicts" and entry.id == "conflict_test":
            await self._handle_sync_conflict(entry, {
                'id': entry.id,
                'data': {'conflicting': 'server_data'},
                'version': entry.version + 1,
                'timestamp': datetime.now().isoformat()
            })
            return False
        
        return True  # Simulate successful sync
    
    async def _handle_sync_conflict(self, local_entry: OfflineDataEntry, 
                                  remote_data: Dict[str, Any]):
        """Handle synchronization conflict."""
        conflict = SyncConflict(
            id=str(uuid.uuid4()),
            collection=local_entry.collection,
            local_entry=local_entry,
            remote_entry=remote_data,
            conflict_type="data_mismatch",
            detected_at=datetime.now()
        )
        
        self.conflicts.append(conflict)
        self.stats['conflicts'] += 1
        
        # Store conflict in database
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO sync_conflicts 
                (id, collection, local_data, remote_data, conflict_type, detected_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                conflict.id,
                conflict.collection,
                json.dumps(asdict(conflict.local_entry)),
                json.dumps(conflict.remote_entry),
                conflict.conflict_type,
                conflict.detected_at.isoformat()
            ))
            conn.commit()
        
        # Try automatic resolution
        await self._attempt_automatic_conflict_resolution(conflict)
        
        logger.warning(f"Sync conflict detected for {local_entry.collection}:{local_entry.id}")
    
    async def _attempt_automatic_conflict_resolution(self, conflict: SyncConflict):
        """Attempt to automatically resolve conflict."""
        # Use timestamp-based resolution by default
        strategy = ConflictResolutionStrategy.TIMESTAMP_BASED
        
        resolver = self.conflict_resolvers.get(strategy.value)
        if resolver:
            try:
                resolved_data = await resolver(conflict)
                if resolved_data:
                    await self._apply_conflict_resolution(conflict, resolved_data, strategy)
                    logger.info(f"Automatically resolved conflict {conflict.id}")
            except Exception as e:
                logger.error(f"Automatic conflict resolution failed: {e}")
    
    async def _apply_conflict_resolution(self, conflict: SyncConflict, 
                                       resolved_data: Dict[str, Any],
                                       strategy: ConflictResolutionStrategy):
        """Apply conflict resolution."""
        # Create resolved entry
        resolved_entry = OfflineDataEntry(
            id=conflict.local_entry.id,
            collection=conflict.collection,
            data=resolved_data,
            operation=DataOperation.UPDATE,
            timestamp=datetime.now(),
            sync_status=SyncStatus.SYNCED,
            version=max(conflict.local_entry.version, 
                       conflict.remote_entry.get('version', 0)) + 1,
            user_id=conflict.local_entry.user_id,
            metadata={
                **conflict.local_entry.metadata,
                'conflict_resolved': True,
                'resolution_strategy': strategy.value,
                'resolved_at': datetime.now().isoformat()
            }
        )
        
        # Store resolved entry
        await self._store_entry_to_db(resolved_entry)
        self._update_memory_cache(resolved_entry)
        
        # Mark conflict as resolved
        conflict.resolved = True
        conflict.resolution_strategy = strategy
        conflict.resolution_data = resolved_data
        
        # Update conflict in database
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE sync_conflicts 
                SET resolved = TRUE, resolution_strategy = ?, resolution_data = ?
                WHERE id = ?
            """, (strategy.value, json.dumps(resolved_data), conflict.id))
            conn.commit()
    
    # Conflict resolution strategies
    async def _resolve_client_wins(self, conflict: SyncConflict) -> Dict[str, Any]:
        """Client wins conflict resolution."""
        return conflict.local_entry.data
    
    async def _resolve_server_wins(self, conflict: SyncConflict) -> Dict[str, Any]:
        """Server wins conflict resolution."""
        return conflict.remote_entry.get('data', {})
    
    async def _resolve_timestamp_based(self, conflict: SyncConflict) -> Dict[str, Any]:
        """Timestamp-based conflict resolution."""
        local_time = conflict.local_entry.timestamp
        remote_time_str = conflict.remote_entry.get('timestamp')
        
        if remote_time_str:
            remote_time = datetime.fromisoformat(remote_time_str.replace('Z', '+00:00'))
            if remote_time > local_time:
                return conflict.remote_entry.get('data', {})
        
        return conflict.local_entry.data
    
    async def _resolve_intelligent_merge(self, conflict: SyncConflict) -> Dict[str, Any]:
        """Intelligent merge conflict resolution."""
        local_data = conflict.local_entry.data
        remote_data = conflict.remote_entry.get('data', {})
        
        # Simple merge strategy: combine non-conflicting fields
        merged_data = {}
        
        # Start with remote data
        merged_data.update(remote_data)
        
        # Add local changes that don't conflict
        for key, value in local_data.items():
            if key not in remote_data or remote_data[key] == value:
                merged_data[key] = value
            else:
                # For conflicting fields, prefer newer timestamp
                local_time = conflict.local_entry.timestamp
                remote_time_str = conflict.remote_entry.get('timestamp')
                
                if remote_time_str:
                    remote_time = datetime.fromisoformat(remote_time_str.replace('Z', '+00:00'))
                    if local_time > remote_time:
                        merged_data[key] = value
                    # else keep remote value
                else:
                    merged_data[key] = value
        
        return merged_data
    
    async def _resolve_merge(self, conflict: SyncConflict) -> Dict[str, Any]:
        """Simple merge conflict resolution."""
        local_data = conflict.local_entry.data
        remote_data = conflict.remote_entry.get('data', {})
        
        # Merge dictionaries, local takes precedence
        merged_data = {**remote_data, **local_data}
        return merged_data
    
    async def _process_pending_conflicts(self):
        """Process pending conflicts that need resolution."""
        unresolved_conflicts = [c for c in self.conflicts if not c.resolved]
        
        for conflict in unresolved_conflicts[:5]:  # Process up to 5 at a time
            if not conflict.resolution_strategy:
                # Try automatic resolution again
                await self._attempt_automatic_conflict_resolution(conflict)
    
    async def _cleanup_old_sync_operations(self):
        """Clean up old sync operations."""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        with self._get_connection() as conn:
            conn.execute("""
                DELETE FROM sync_operations 
                WHERE started_at < ? AND status = 'completed'
            """, (cutoff_time.isoformat(),))
            conn.commit()
    
    # Public API methods
    async def force_sync(self, collection: Optional[str] = None) -> Dict[str, Any]:
        """Force synchronization of pending data."""
        if not self.is_online:
            return {'success': False, 'error': 'Offline mode'}
        
        # Get pending entries
        filters = {'sync_status': SyncStatus.PENDING_SYNC.value}
        if collection:
            pending_entries = await self._query_entries_from_db(collection, filters)
        else:
            # Get from all collections
            pending_entries = []
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT * FROM offline_data WHERE sync_status = ?
                """, (SyncStatus.PENDING_SYNC.value,)).fetchall()
                
                for row in rows:
                    entry = OfflineDataEntry(
                        id=row['id'],
                        collection=row['collection'],
                        data=json.loads(row['data']),
                        operation=DataOperation(row['operation']),
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        sync_status=SyncStatus(row['sync_status']),
                        version=row['version'],
                        checksum=row['checksum'],
                        user_id=row['user_id'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else {}
                    )
                    pending_entries.append(entry)
        
        # Queue all for sync
        for entry in pending_entries:
            await self._queue_for_sync(entry)
        
        # Process sync queue immediately
        await self._process_sync_queue()
        
        return {
            'success': True,
            'synced_entries': len(pending_entries),
            'remaining_conflicts': len([c for c in self.conflicts if not c.resolved])
        }
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status."""
        return {
            'is_online': self.is_online,
            'last_sync_attempt': self.last_sync_attempt.isoformat(),
            'pending_sync_operations': len(self.sync_queue),
            'unresolved_conflicts': len([c for c in self.conflicts if not c.resolved]),
            'statistics': self.stats.copy()
        }
    
    def get_conflicts(self) -> List[Dict[str, Any]]:
        """Get list of unresolved conflicts."""
        return [
            {
                'id': conflict.id,
                'collection': conflict.collection,
                'conflict_type': conflict.conflict_type,
                'detected_at': conflict.detected_at.isoformat(),
                'local_data': conflict.local_entry.data,
                'remote_data': conflict.remote_entry,
                'resolved': conflict.resolved
            }
            for conflict in self.conflicts if not conflict.resolved
        ]
    
    async def resolve_conflict(self, conflict_id: str, 
                             strategy: ConflictResolutionStrategy,
                             custom_data: Optional[Dict[str, Any]] = None) -> bool:
        """Manually resolve a conflict."""
        conflict = next((c for c in self.conflicts if c.id == conflict_id), None)
        if not conflict or conflict.resolved:
            return False
        
        if strategy == ConflictResolutionStrategy.USER_CHOICE and custom_data:
            resolved_data = custom_data
        else:
            resolver = self.conflict_resolvers.get(strategy.value)
            if not resolver:
                return False
            resolved_data = await resolver(conflict)
        
        await self._apply_conflict_resolution(conflict, resolved_data, strategy)
        return True
    
    def set_online_status(self, is_online: bool):
        """Set online/offline status."""
        self.is_online = is_online
        if is_online and self.auto_sync:
            self._start_background_sync()
        
        logger.info(f"Set online status to: {is_online}")
    
    def register_sync_callback(self, callback: Callable):
        """Register callback for sync events."""
        self.sync_callbacks.append(callback)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information."""
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        with self._get_connection() as conn:
            total_entries = conn.execute("SELECT COUNT(*) FROM offline_data").fetchone()[0]
            collections = conn.execute("""
                SELECT collection, COUNT(*) as count 
                FROM offline_data 
                GROUP BY collection
            """).fetchall()
        
        return {
            'database_size_bytes': db_size,
            'database_size_mb': round(db_size / (1024 * 1024), 2),
            'total_entries': total_entries,
            'collections': {row[0]: row[1] for row in collections},
            'memory_cache_entries': sum(len(c) for c in self.memory_cache.values()),
            'max_cache_entries': self.max_cache_entries
        }
    
    async def cleanup_storage(self, days_old: int = 30) -> Dict[str, Any]:
        """Clean up old data from storage."""
        cutoff_time = datetime.now() - timedelta(days=days_old)
        
        with self._get_connection() as conn:
            # Count entries to be deleted
            old_entries = conn.execute("""
                SELECT COUNT(*) FROM offline_data 
                WHERE timestamp < ? AND sync_status = ?
            """, (cutoff_time.isoformat(), SyncStatus.SYNCED.value)).fetchone()[0]
            
            # Delete old synced entries
            conn.execute("""
                DELETE FROM offline_data 
                WHERE timestamp < ? AND sync_status = ?
            """, (cutoff_time.isoformat(), SyncStatus.SYNCED.value))
            
            # Clean up resolved conflicts
            old_conflicts = conn.execute("""
                SELECT COUNT(*) FROM sync_conflicts 
                WHERE detected_at < ? AND resolved = TRUE
            """, (cutoff_time.isoformat(),)).fetchone()[0]
            
            conn.execute("""
                DELETE FROM sync_conflicts 
                WHERE detected_at < ? AND resolved = TRUE
            """, (cutoff_time.isoformat(),))
            
            conn.commit()
        
        # Update statistics
        self.stats['total_entries'] -= old_entries
        
        return {
            'deleted_entries': old_entries,
            'deleted_conflicts': old_conflicts,
            'storage_info': self.get_storage_info()
        }
    
    def close(self):
        """Close the offline data manager."""
        if self.sync_task and not self.sync_task.done():
            self.sync_task.cancel()
        
        if self.db_connection:
            self.db_connection.close()
        
        logger.info("Offline data manager closed")


# Global instance
offline_data_manager = OfflineDataManager()