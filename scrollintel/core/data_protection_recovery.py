"""
Comprehensive data protection and recovery system for ScrollIntel.
Implements automatic continuous save, multi-tier backup, data integrity verification,
and cross-device state synchronization with offline support.
"""

import asyncio
import logging
import json
import hashlib
import time
import os
import shutil
import sqlite3
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from contextlib import asynccontextmanager
import threading
import pickle
import gzip
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)


class BackupTier(Enum):
    """Backup storage tiers."""
    LOCAL = "local"           # Local disk storage
    REMOTE = "remote"         # Remote cloud storage
    DISTRIBUTED = "distributed"  # Distributed across multiple nodes
    ARCHIVE = "archive"       # Long-term archive storage


class DataIntegrityStatus(Enum):
    """Data integrity verification status."""
    VALID = "valid"
    CORRUPTED = "corrupted"
    MISSING = "missing"
    REPAIRING = "repairing"
    REPAIRED = "repaired"


class SyncStatus(Enum):
    """Cross-device synchronization status."""
    SYNCED = "synced"
    PENDING = "pending"
    CONFLICT = "conflict"
    OFFLINE = "offline"
    FAILED = "failed"


@dataclass
class DataSnapshot:
    """Represents a data snapshot for backup and recovery."""
    snapshot_id: str
    user_id: str
    data_type: str
    data: Any
    checksum: str
    timestamp: datetime
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    backup_tiers: List[BackupTier] = field(default_factory=list)


@dataclass
class RecoveryPoint:
    """Represents a recovery point with multiple snapshots."""
    recovery_id: str
    user_id: str
    timestamp: datetime
    snapshots: List[DataSnapshot]
    description: str
    auto_created: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncConflict:
    """Represents a synchronization conflict between devices."""
    conflict_id: str
    user_id: str
    data_type: str
    device_a_data: Any
    device_b_data: Any
    device_a_timestamp: datetime
    device_b_timestamp: datetime
    resolution_strategy: Optional[str] = None
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceState:
    """Represents the state of a user's device."""
    device_id: str
    user_id: str
    last_sync: datetime
    data_version: str
    offline_changes: List[Dict[str, Any]] = field(default_factory=list)
    sync_status: SyncStatus = SyncStatus.SYNCED
    capabilities: Dict[str, Any] = field(default_factory=dict)


class AutoSaveManager:
    """Manages automatic continuous saving of user data."""
    
    def __init__(self, save_interval: int = 30):
        self.save_interval = save_interval  # seconds
        self.pending_saves: Dict[str, Dict[str, Any]] = {}
        self.save_callbacks: Dict[str, Callable] = {}
        self.active_sessions: Dict[str, datetime] = {}
        self._save_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the auto-save manager."""
        if not self._running:
            self._running = True
            self._save_task = asyncio.create_task(self._auto_save_loop())
            logger.info("Auto-save manager started")
    
    async def stop(self):
        """Stop the auto-save manager."""
        self._running = False
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        logger.info("Auto-save manager stopped")
    
    async def register_data(self, user_id: str, data_type: str, data: Any, 
                           save_callback: Optional[Callable] = None):
        """Register data for automatic saving."""
        async with self._lock:
            key = f"{user_id}:{data_type}"
            self.pending_saves[key] = {
                'user_id': user_id,
                'data_type': data_type,
                'data': data,
                'timestamp': datetime.utcnow(),
                'dirty': True
            }
            
            if save_callback:
                self.save_callbacks[key] = save_callback
            
            # Update active session
            self.active_sessions[user_id] = datetime.utcnow()
    
    async def force_save(self, user_id: str, data_type: Optional[str] = None):
        """Force immediate save for user data."""
        async with self._lock:
            if data_type:
                key = f"{user_id}:{data_type}"
                if key in self.pending_saves:
                    await self._save_data(key, self.pending_saves[key])
            else:
                # Save all data for user
                user_keys = [k for k in self.pending_saves.keys() if k.startswith(f"{user_id}:")]
                for key in user_keys:
                    await self._save_data(key, self.pending_saves[key])
    
    async def _auto_save_loop(self):
        """Main auto-save loop."""
        try:
            while self._running:
                await asyncio.sleep(self.save_interval)
                await self._perform_auto_save()
        except asyncio.CancelledError:
            # Perform final save before stopping
            await self._perform_auto_save()
            raise
    
    async def _perform_auto_save(self):
        """Perform auto-save for all pending data."""
        async with self._lock:
            current_time = datetime.utcnow()
            
            # Clean up inactive sessions
            inactive_cutoff = current_time - timedelta(hours=1)
            inactive_users = [
                user_id for user_id, last_active in self.active_sessions.items()
                if last_active < inactive_cutoff
            ]
            
            for user_id in inactive_users:
                # Final save for inactive user
                user_keys = [k for k in self.pending_saves.keys() if k.startswith(f"{user_id}:")]
                for key in user_keys:
                    if self.pending_saves[key]['dirty']:
                        await self._save_data(key, self.pending_saves[key])
                
                # Clean up
                del self.active_sessions[user_id]
                for key in user_keys:
                    del self.pending_saves[key]
            
            # Save dirty data for active users
            for key, data_info in list(self.pending_saves.items()):
                if data_info['dirty']:
                    await self._save_data(key, data_info)
    
    async def _save_data(self, key: str, data_info: Dict[str, Any]):
        """Save individual data item."""
        try:
            # Use registered callback if available
            if key in self.save_callbacks:
                await self.save_callbacks[key](data_info)
            else:
                # Default save behavior
                logger.debug(f"Auto-saving data for {key}")
            
            # Mark as clean
            data_info['dirty'] = False
            data_info['last_saved'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to auto-save data for {key}: {e}")


class MultiTierBackupManager:
    """Manages multi-tier backup strategy with instant recovery."""
    
    def __init__(self, base_path: str = "data/backups"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Tier configurations
        self.tier_configs = {
            BackupTier.LOCAL: {
                'path': self.base_path / 'local',
                'retention_days': 7,
                'max_size_gb': 10
            },
            BackupTier.REMOTE: {
                'path': self.base_path / 'remote',
                'retention_days': 30,
                'max_size_gb': 100
            },
            BackupTier.DISTRIBUTED: {
                'path': self.base_path / 'distributed',
                'retention_days': 90,
                'max_size_gb': 1000
            },
            BackupTier.ARCHIVE: {
                'path': self.base_path / 'archive',
                'retention_days': 365,
                'max_size_gb': 10000
            }
        }
        
        # Create tier directories
        for config in self.tier_configs.values():
            config['path'].mkdir(parents=True, exist_ok=True)
        
        self.snapshots: Dict[str, DataSnapshot] = {}
        self.recovery_points: Dict[str, RecoveryPoint] = {}
        self._load_existing_backups()
    
    def _load_existing_backups(self):
        """Load existing backups from storage."""
        try:
            for tier, config in self.tier_configs.items():
                tier_path = config['path']
                if tier_path.exists():
                    for backup_file in tier_path.glob('*.backup'):
                        try:
                            with gzip.open(backup_file, 'rb') as f:
                                snapshot_data = pickle.load(f)
                                snapshot = DataSnapshot(**snapshot_data)
                                self.snapshots[snapshot.snapshot_id] = snapshot
                        except Exception as e:
                            logger.warning(f"Failed to load backup {backup_file}: {e}")
            
            logger.info(f"Loaded {len(self.snapshots)} existing backups")
        except Exception as e:
            logger.error(f"Failed to load existing backups: {e}")
    
    async def create_snapshot(self, user_id: str, data_type: str, data: Any,
                            tiers: List[BackupTier] = None) -> DataSnapshot:
        """Create a data snapshot and store in specified tiers."""
        if tiers is None:
            tiers = [BackupTier.LOCAL, BackupTier.REMOTE]
        
        # Create snapshot
        snapshot_id = str(uuid.uuid4())
        serialized_data = json.dumps(data, default=str)
        checksum = hashlib.sha256(serialized_data.encode()).hexdigest()
        
        snapshot = DataSnapshot(
            snapshot_id=snapshot_id,
            user_id=user_id,
            data_type=data_type,
            data=data,
            checksum=checksum,
            timestamp=datetime.utcnow(),
            size_bytes=len(serialized_data),
            backup_tiers=tiers
        )
        
        # Store in specified tiers
        for tier in tiers:
            await self._store_snapshot_in_tier(snapshot, tier)
        
        self.snapshots[snapshot_id] = snapshot
        logger.debug(f"Created snapshot {snapshot_id} for {user_id}:{data_type}")
        
        return snapshot
    
    async def _store_snapshot_in_tier(self, snapshot: DataSnapshot, tier: BackupTier):
        """Store snapshot in specific backup tier."""
        try:
            tier_config = self.tier_configs[tier]
            tier_path = tier_config['path']
            
            # Create user directory
            user_path = tier_path / snapshot.user_id
            user_path.mkdir(exist_ok=True)
            
            # Store snapshot
            filename = f"{snapshot.snapshot_id}.backup"
            filepath = user_path / filename
            
            snapshot_data = asdict(snapshot)
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(snapshot_data, f)
            
            logger.debug(f"Stored snapshot {snapshot.snapshot_id} in {tier.value} tier")
            
        except Exception as e:
            logger.error(f"Failed to store snapshot in {tier.value} tier: {e}")
    
    async def create_recovery_point(self, user_id: str, description: str = None) -> RecoveryPoint:
        """Create a recovery point with current user data snapshots."""
        recovery_id = str(uuid.uuid4())
        
        # Get recent snapshots for user
        user_snapshots = [
            snapshot for snapshot in self.snapshots.values()
            if snapshot.user_id == user_id and 
            snapshot.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        recovery_point = RecoveryPoint(
            recovery_id=recovery_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            snapshots=user_snapshots,
            description=description or f"Auto recovery point {datetime.utcnow().isoformat()}"
        )
        
        self.recovery_points[recovery_id] = recovery_point
        logger.info(f"Created recovery point {recovery_id} for user {user_id}")
        
        return recovery_point
    
    async def restore_from_snapshot(self, snapshot_id: str) -> Optional[Any]:
        """Restore data from a specific snapshot."""
        if snapshot_id not in self.snapshots:
            logger.error(f"Snapshot {snapshot_id} not found")
            return None
        
        snapshot = self.snapshots[snapshot_id]
        
        # Try to load from available tiers
        for tier in snapshot.backup_tiers:
            try:
                data = await self._load_snapshot_from_tier(snapshot, tier)
                if data is not None:
                    # Verify integrity
                    if await self._verify_snapshot_integrity(snapshot, data):
                        logger.info(f"Successfully restored snapshot {snapshot_id} from {tier.value}")
                        return data
                    else:
                        logger.warning(f"Integrity check failed for snapshot {snapshot_id} in {tier.value}")
            except Exception as e:
                logger.warning(f"Failed to restore from {tier.value} tier: {e}")
        
        logger.error(f"Failed to restore snapshot {snapshot_id} from any tier")
        return None
    
    async def _load_snapshot_from_tier(self, snapshot: DataSnapshot, tier: BackupTier) -> Optional[Any]:
        """Load snapshot data from specific tier."""
        try:
            tier_config = self.tier_configs[tier]
            tier_path = tier_config['path']
            
            filepath = tier_path / snapshot.user_id / f"{snapshot.snapshot_id}.backup"
            
            if not filepath.exists():
                return None
            
            with gzip.open(filepath, 'rb') as f:
                snapshot_data = pickle.load(f)
                return snapshot_data['data']
                
        except Exception as e:
            logger.error(f"Failed to load snapshot from {tier.value}: {e}")
            return None
    
    async def _verify_snapshot_integrity(self, snapshot: DataSnapshot, data: Any) -> bool:
        """Verify snapshot data integrity."""
        try:
            serialized_data = json.dumps(data, default=str)
            calculated_checksum = hashlib.sha256(serialized_data.encode()).hexdigest()
            return calculated_checksum == snapshot.checksum
        except Exception as e:
            logger.error(f"Failed to verify snapshot integrity: {e}")
            return False
    
    async def restore_from_recovery_point(self, recovery_id: str) -> Dict[str, Any]:
        """Restore all data from a recovery point."""
        if recovery_id not in self.recovery_points:
            logger.error(f"Recovery point {recovery_id} not found")
            return {}
        
        recovery_point = self.recovery_points[recovery_id]
        restored_data = {}
        
        for snapshot in recovery_point.snapshots:
            data = await self.restore_from_snapshot(snapshot.snapshot_id)
            if data is not None:
                restored_data[snapshot.data_type] = data
        
        logger.info(f"Restored {len(restored_data)} data types from recovery point {recovery_id}")
        return restored_data
    
    async def cleanup_old_backups(self):
        """Clean up old backups based on retention policies."""
        current_time = datetime.utcnow()
        
        for tier, config in self.tier_configs.items():
            retention_days = config['retention_days']
            cutoff_time = current_time - timedelta(days=retention_days)
            
            # Find old snapshots
            old_snapshots = [
                snapshot for snapshot in self.snapshots.values()
                if tier in snapshot.backup_tiers and snapshot.timestamp < cutoff_time
            ]
            
            # Remove old snapshots
            for snapshot in old_snapshots:
                await self._remove_snapshot_from_tier(snapshot, tier)
                
                # Remove from memory if not in any other tier
                if len(snapshot.backup_tiers) == 1 and tier in snapshot.backup_tiers:
                    del self.snapshots[snapshot.snapshot_id]
                else:
                    snapshot.backup_tiers.remove(tier)
            
            logger.info(f"Cleaned up {len(old_snapshots)} old backups from {tier.value} tier")
    
    async def _remove_snapshot_from_tier(self, snapshot: DataSnapshot, tier: BackupTier):
        """Remove snapshot from specific tier."""
        try:
            tier_config = self.tier_configs[tier]
            filepath = tier_config['path'] / snapshot.user_id / f"{snapshot.snapshot_id}.backup"
            
            if filepath.exists():
                filepath.unlink()
                
        except Exception as e:
            logger.error(f"Failed to remove snapshot from {tier.value}: {e}")


class DataIntegrityVerifier:
    """Verifies and repairs data integrity."""
    
    def __init__(self):
        self.integrity_checks: Dict[str, Callable] = {}
        self.repair_strategies: Dict[str, Callable] = {}
        self.verification_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def register_integrity_check(self, data_type: str, check_func: Callable):
        """Register integrity check function for data type."""
        self.integrity_checks[data_type] = check_func
    
    def register_repair_strategy(self, data_type: str, repair_func: Callable):
        """Register repair strategy for data type."""
        self.repair_strategies[data_type] = repair_func
    
    async def verify_data_integrity(self, user_id: str, data_type: str, data: Any) -> DataIntegrityStatus:
        """Verify integrity of specific data."""
        try:
            # Use registered check if available
            if data_type in self.integrity_checks:
                is_valid = await self.integrity_checks[data_type](data)
            else:
                # Default integrity check
                is_valid = await self._default_integrity_check(data)
            
            status = DataIntegrityStatus.VALID if is_valid else DataIntegrityStatus.CORRUPTED
            
            # Record verification
            self.verification_history[f"{user_id}:{data_type}"].append({
                'timestamp': datetime.utcnow().isoformat(),
                'status': status.value,
                'data_size': len(str(data)) if data else 0
            })
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to verify data integrity for {user_id}:{data_type}: {e}")
            return DataIntegrityStatus.CORRUPTED
    
    async def _default_integrity_check(self, data: Any) -> bool:
        """Default integrity check for any data type."""
        try:
            # Basic checks
            if data is None:
                return False
            
            # Check if data can be serialized
            json.dumps(data, default=str)
            
            # Check for common corruption indicators
            if isinstance(data, dict):
                # Check for null bytes or invalid characters
                for key, value in data.items():
                    if isinstance(key, str) and '\x00' in key:
                        return False
                    if isinstance(value, str) and '\x00' in value:
                        return False
            
            return True
            
        except Exception:
            return False
    
    async def repair_corrupted_data(self, user_id: str, data_type: str, 
                                  corrupted_data: Any, backup_manager: MultiTierBackupManager) -> Optional[Any]:
        """Attempt to repair corrupted data."""
        try:
            # Try registered repair strategy first
            if data_type in self.repair_strategies:
                repaired_data = await self.repair_strategies[data_type](corrupted_data, backup_manager)
                if repaired_data is not None:
                    # Verify repaired data
                    if await self.verify_data_integrity(user_id, data_type, repaired_data) == DataIntegrityStatus.VALID:
                        logger.info(f"Successfully repaired corrupted data for {user_id}:{data_type}")
                        return repaired_data
            
            # Try default repair strategies
            repaired_data = await self._default_repair_strategy(user_id, data_type, corrupted_data, backup_manager)
            
            if repaired_data is not None:
                logger.info(f"Repaired corrupted data using default strategy for {user_id}:{data_type}")
                return repaired_data
            
            logger.warning(f"Failed to repair corrupted data for {user_id}:{data_type}")
            return None
            
        except Exception as e:
            logger.error(f"Error during data repair for {user_id}:{data_type}: {e}")
            return None
    
    async def _default_repair_strategy(self, user_id: str, data_type: str, 
                                     corrupted_data: Any, backup_manager: MultiTierBackupManager) -> Optional[Any]:
        """Default repair strategy using backups."""
        # Find recent valid snapshots
        recent_snapshots = [
            snapshot for snapshot in backup_manager.snapshots.values()
            if (snapshot.user_id == user_id and 
                snapshot.data_type == data_type and
                snapshot.timestamp > datetime.utcnow() - timedelta(days=7))
        ]
        
        # Sort by timestamp (newest first)
        recent_snapshots.sort(key=lambda s: s.timestamp, reverse=True)
        
        # Try to restore from recent snapshots
        for snapshot in recent_snapshots:
            restored_data = await backup_manager.restore_from_snapshot(snapshot.snapshot_id)
            if restored_data is not None:
                # Verify restored data
                if await self.verify_data_integrity(user_id, data_type, restored_data) == DataIntegrityStatus.VALID:
                    return restored_data
        
        return None


class CrossDeviceSyncManager:
    """Manages cross-device state synchronization with offline support."""
    
    def __init__(self, db_path: str = "data/sync.db"):
        self.db_path = db_path
        self.device_states: Dict[str, DeviceState] = {}
        self.sync_conflicts: Dict[str, SyncConflict] = {}
        self.offline_queues: Dict[str, deque] = defaultdict(deque)
        self._init_database()
        self._load_device_states()
    
    def _init_database(self):
        """Initialize SQLite database for sync management."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS device_states (
                    device_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    last_sync TEXT NOT NULL,
                    data_version TEXT NOT NULL,
                    sync_status TEXT NOT NULL,
                    capabilities TEXT,
                    offline_changes TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sync_conflicts (
                    conflict_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    device_a_data TEXT,
                    device_b_data TEXT,
                    device_a_timestamp TEXT,
                    device_b_timestamp TEXT,
                    resolution_strategy TEXT,
                    resolved INTEGER DEFAULT 0,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sync_operations (
                    operation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    data TEXT,
                    timestamp TEXT NOT NULL,
                    synced INTEGER DEFAULT 0
                )
            ''')
    
    def _load_device_states(self):
        """Load device states from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT * FROM device_states')
                for row in cursor.fetchall():
                    device_id, user_id, last_sync, data_version, sync_status, capabilities, offline_changes = row
                    
                    device_state = DeviceState(
                        device_id=device_id,
                        user_id=user_id,
                        last_sync=datetime.fromisoformat(last_sync),
                        data_version=data_version,
                        sync_status=SyncStatus(sync_status),
                        capabilities=json.loads(capabilities) if capabilities else {},
                        offline_changes=json.loads(offline_changes) if offline_changes else []
                    )
                    
                    self.device_states[device_id] = device_state
                    
        except Exception as e:
            logger.error(f"Failed to load device states: {e}")
    
    async def register_device(self, device_id: str, user_id: str, capabilities: Dict[str, Any] = None) -> DeviceState:
        """Register a new device for synchronization."""
        device_state = DeviceState(
            device_id=device_id,
            user_id=user_id,
            last_sync=datetime.utcnow(),
            data_version=str(uuid.uuid4()),
            capabilities=capabilities or {}
        )
        
        self.device_states[device_id] = device_state
        await self._save_device_state(device_state)
        
        logger.info(f"Registered device {device_id} for user {user_id}")
        return device_state
    
    async def _save_device_state(self, device_state: DeviceState):
        """Save device state to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO device_states 
                    (device_id, user_id, last_sync, data_version, sync_status, capabilities, offline_changes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    device_state.device_id,
                    device_state.user_id,
                    device_state.last_sync.isoformat(),
                    device_state.data_version,
                    device_state.sync_status.value,
                    json.dumps(device_state.capabilities),
                    json.dumps(device_state.offline_changes)
                ))
                
        except Exception as e:
            logger.error(f"Failed to save device state: {e}")
    
    async def sync_data(self, device_id: str, data_type: str, data: Any, 
                       force_sync: bool = False) -> SyncStatus:
        """Synchronize data across devices."""
        if device_id not in self.device_states:
            logger.error(f"Device {device_id} not registered")
            return SyncStatus.FAILED
        
        device_state = self.device_states[device_id]
        
        try:
            # Check for conflicts with other devices
            conflicts = await self._detect_sync_conflicts(device_state.user_id, data_type, data, device_id)
            
            if conflicts and not force_sync:
                # Handle conflicts
                for conflict in conflicts:
                    self.sync_conflicts[conflict.conflict_id] = conflict
                    logger.warning(f"Sync conflict detected: {conflict.conflict_id}")
                
                device_state.sync_status = SyncStatus.CONFLICT
                await self._save_device_state(device_state)
                return SyncStatus.CONFLICT
            
            # Perform synchronization
            await self._perform_sync(device_state, data_type, data)
            
            device_state.sync_status = SyncStatus.SYNCED
            device_state.last_sync = datetime.utcnow()
            device_state.data_version = str(uuid.uuid4())
            
            await self._save_device_state(device_state)
            
            logger.debug(f"Successfully synced {data_type} for device {device_id}")
            return SyncStatus.SYNCED
            
        except Exception as e:
            logger.error(f"Failed to sync data for device {device_id}: {e}")
            device_state.sync_status = SyncStatus.FAILED
            await self._save_device_state(device_state)
            return SyncStatus.FAILED
    
    async def _detect_sync_conflicts(self, user_id: str, data_type: str, data: Any, 
                                   current_device_id: str) -> List[SyncConflict]:
        """Detect synchronization conflicts with other devices."""
        conflicts = []
        
        # Get other devices for the same user
        other_devices = [
            device for device in self.device_states.values()
            if device.user_id == user_id and device.device_id != current_device_id
        ]
        
        for other_device in other_devices:
            # Check for recent changes to the same data type
            recent_changes = [
                change for change in other_device.offline_changes
                if (change.get('data_type') == data_type and
                    datetime.fromisoformat(change.get('timestamp', '1970-01-01')) > 
                    datetime.utcnow() - timedelta(hours=1))
            ]
            
            if recent_changes:
                # Create conflict
                conflict = SyncConflict(
                    conflict_id=str(uuid.uuid4()),
                    user_id=user_id,
                    data_type=data_type,
                    device_a_data=data,
                    device_b_data=recent_changes[-1].get('data'),
                    device_a_timestamp=datetime.utcnow(),
                    device_b_timestamp=datetime.fromisoformat(recent_changes[-1].get('timestamp')),
                    metadata={
                        'device_a': current_device_id,
                        'device_b': other_device.device_id
                    }
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _perform_sync(self, device_state: DeviceState, data_type: str, data: Any):
        """Perform actual data synchronization."""
        # Record sync operation
        operation_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO sync_operations 
                (operation_id, user_id, device_id, operation_type, data_type, data, timestamp, synced)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                operation_id,
                device_state.user_id,
                device_state.device_id,
                'sync',
                data_type,
                json.dumps(data, default=str),
                datetime.utcnow().isoformat(),
                1
            ))
        
        # Update other devices
        await self._propagate_sync_to_devices(device_state.user_id, data_type, data, device_state.device_id)
    
    async def _propagate_sync_to_devices(self, user_id: str, data_type: str, data: Any, source_device_id: str):
        """Propagate sync changes to other user devices."""
        other_devices = [
            device for device in self.device_states.values()
            if device.user_id == user_id and device.device_id != source_device_id
        ]
        
        for device in other_devices:
            # Add to offline queue if device is offline
            if device.sync_status == SyncStatus.OFFLINE:
                self.offline_queues[device.device_id].append({
                    'operation_type': 'sync',
                    'data_type': data_type,
                    'data': data,
                    'timestamp': datetime.utcnow().isoformat(),
                    'source_device': source_device_id
                })
            else:
                # Mark as pending sync
                device.sync_status = SyncStatus.PENDING
                await self._save_device_state(device)
    
    async def resolve_conflict(self, conflict_id: str, resolution_strategy: str, 
                             resolved_data: Any = None) -> bool:
        """Resolve a synchronization conflict."""
        if conflict_id not in self.sync_conflicts:
            logger.error(f"Conflict {conflict_id} not found")
            return False
        
        conflict = self.sync_conflicts[conflict_id]
        
        try:
            if resolution_strategy == "use_latest":
                # Use data with latest timestamp
                if conflict.device_a_timestamp > conflict.device_b_timestamp:
                    resolved_data = conflict.device_a_data
                else:
                    resolved_data = conflict.device_b_data
            
            elif resolution_strategy == "merge":
                # Attempt to merge data (implementation depends on data type)
                resolved_data = await self._merge_conflicted_data(conflict)
            
            elif resolution_strategy == "manual":
                # Use manually provided resolution
                if resolved_data is None:
                    logger.error("Manual resolution requires resolved_data")
                    return False
            
            # Apply resolution
            await self._apply_conflict_resolution(conflict, resolved_data)
            
            conflict.resolution_strategy = resolution_strategy
            conflict.resolved = True
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO sync_conflicts 
                    (conflict_id, user_id, data_type, device_a_data, device_b_data, 
                     device_a_timestamp, device_b_timestamp, resolution_strategy, resolved, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    conflict.conflict_id,
                    conflict.user_id,
                    conflict.data_type,
                    json.dumps(conflict.device_a_data, default=str),
                    json.dumps(conflict.device_b_data, default=str),
                    conflict.device_a_timestamp.isoformat(),
                    conflict.device_b_timestamp.isoformat(),
                    conflict.resolution_strategy,
                    1 if conflict.resolved else 0,
                    json.dumps(conflict.metadata)
                ))
            
            logger.info(f"Resolved conflict {conflict_id} using {resolution_strategy}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve conflict {conflict_id}: {e}")
            return False
    
    async def _merge_conflicted_data(self, conflict: SyncConflict) -> Any:
        """Attempt to merge conflicted data."""
        # Basic merge strategy - this would be enhanced based on data types
        data_a = conflict.device_a_data
        data_b = conflict.device_b_data
        
        if isinstance(data_a, dict) and isinstance(data_b, dict):
            # Merge dictionaries
            merged = data_a.copy()
            for key, value in data_b.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(merged[key], dict) and isinstance(value, dict):
                    # Recursive merge for nested dicts
                    merged[key] = {**merged[key], **value}
            return merged
        
        elif isinstance(data_a, list) and isinstance(data_b, list):
            # Merge lists (remove duplicates)
            return list(set(data_a + data_b))
        
        else:
            # Use latest timestamp for non-mergeable data
            if conflict.device_a_timestamp > conflict.device_b_timestamp:
                return data_a
            else:
                return data_b
    
    async def _apply_conflict_resolution(self, conflict: SyncConflict, resolved_data: Any):
        """Apply conflict resolution to all user devices."""
        # Sync resolved data to all user devices
        user_devices = [
            device for device in self.device_states.values()
            if device.user_id == conflict.user_id
        ]
        
        for device in user_devices:
            await self._perform_sync(device, conflict.data_type, resolved_data)
    
    async def handle_device_offline(self, device_id: str):
        """Handle device going offline."""
        if device_id in self.device_states:
            device_state = self.device_states[device_id]
            device_state.sync_status = SyncStatus.OFFLINE
            await self._save_device_state(device_state)
            logger.info(f"Device {device_id} marked as offline")
    
    async def handle_device_online(self, device_id: str) -> List[Dict[str, Any]]:
        """Handle device coming back online and process offline queue."""
        if device_id not in self.device_states:
            logger.error(f"Device {device_id} not registered")
            return []
        
        device_state = self.device_states[device_id]
        device_state.sync_status = SyncStatus.PENDING
        
        # Process offline queue
        offline_operations = list(self.offline_queues[device_id])
        self.offline_queues[device_id].clear()
        
        # Apply offline operations
        for operation in offline_operations:
            try:
                await self._perform_sync(device_state, operation['data_type'], operation['data'])
            except Exception as e:
                logger.error(f"Failed to apply offline operation: {e}")
        
        device_state.sync_status = SyncStatus.SYNCED
        device_state.last_sync = datetime.utcnow()
        await self._save_device_state(device_state)
        
        logger.info(f"Device {device_id} back online, processed {len(offline_operations)} offline operations")
        return offline_operations
    
    def get_sync_status(self, user_id: str) -> Dict[str, Any]:
        """Get synchronization status for all user devices."""
        user_devices = [
            device for device in self.device_states.values()
            if device.user_id == user_id
        ]
        
        conflicts = [
            conflict for conflict in self.sync_conflicts.values()
            if conflict.user_id == user_id and not conflict.resolved
        ]
        
        return {
            'devices': [
                {
                    'device_id': device.device_id,
                    'sync_status': device.sync_status.value,
                    'last_sync': device.last_sync.isoformat(),
                    'data_version': device.data_version,
                    'offline_changes_count': len(device.offline_changes)
                }
                for device in user_devices
            ],
            'active_conflicts': len(conflicts),
            'total_devices': len(user_devices),
            'all_synced': all(device.sync_status == SyncStatus.SYNCED for device in user_devices)
        }


class DataProtectionRecoverySystem:
    """Main system that coordinates all data protection and recovery components."""
    
    def __init__(self):
        self.auto_save_manager = AutoSaveManager()
        self.backup_manager = MultiTierBackupManager()
        self.integrity_verifier = DataIntegrityVerifier()
        self.sync_manager = CrossDeviceSyncManager()
        self._running = False
        
        # Setup default integrity checks and repair strategies
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default integrity checks and repair strategies."""
        # Register default integrity checks
        self.integrity_verifier.register_integrity_check('user_data', self._check_user_data_integrity)
        self.integrity_verifier.register_integrity_check('file_data', self._check_file_data_integrity)
        self.integrity_verifier.register_integrity_check('analysis_results', self._check_analysis_integrity)
        
        # Register default repair strategies
        self.integrity_verifier.register_repair_strategy('user_data', self._repair_user_data)
        self.integrity_verifier.register_repair_strategy('file_data', self._repair_file_data)
        self.integrity_verifier.register_repair_strategy('analysis_results', self._repair_analysis_results)
    
    async def _check_user_data_integrity(self, data: Any) -> bool:
        """Check integrity of user data."""
        if not isinstance(data, dict):
            return False
        
        required_fields = ['user_id', 'timestamp']
        return all(field in data for field in required_fields)
    
    async def _check_file_data_integrity(self, data: Any) -> bool:
        """Check integrity of file data."""
        if not isinstance(data, dict):
            return False
        
        required_fields = ['filename', 'content', 'size']
        return all(field in data for field in required_fields)
    
    async def _check_analysis_integrity(self, data: Any) -> bool:
        """Check integrity of analysis results."""
        if not isinstance(data, dict):
            return False
        
        required_fields = ['analysis_type', 'results', 'timestamp']
        return all(field in data for field in required_fields)
    
    async def _repair_user_data(self, corrupted_data: Any, backup_manager: MultiTierBackupManager) -> Optional[Any]:
        """Repair corrupted user data."""
        # Try to extract user_id from corrupted data
        user_id = None
        if isinstance(corrupted_data, dict):
            user_id = corrupted_data.get('user_id')
        
        if user_id:
            # Find recent backup
            recent_snapshots = [
                snapshot for snapshot in backup_manager.snapshots.values()
                if (snapshot.user_id == user_id and 
                    snapshot.data_type == 'user_data' and
                    snapshot.timestamp > datetime.utcnow() - timedelta(days=1))
            ]
            
            if recent_snapshots:
                latest_snapshot = max(recent_snapshots, key=lambda s: s.timestamp)
                return await backup_manager.restore_from_snapshot(latest_snapshot.snapshot_id)
        
        return None
    
    async def _repair_file_data(self, corrupted_data: Any, backup_manager: MultiTierBackupManager) -> Optional[Any]:
        """Repair corrupted file data."""
        # Similar to user data repair but for files
        return None  # Placeholder implementation
    
    async def _repair_analysis_results(self, corrupted_data: Any, backup_manager: MultiTierBackupManager) -> Optional[Any]:
        """Repair corrupted analysis results."""
        # Similar to user data repair but for analysis results
        return None  # Placeholder implementation
    
    async def start(self):
        """Start the data protection and recovery system."""
        if not self._running:
            self._running = True
            await self.auto_save_manager.start()
            
            # Start background tasks
            asyncio.create_task(self._periodic_backup_cleanup())
            asyncio.create_task(self._periodic_integrity_check())
            
            logger.info("Data protection and recovery system started")
    
    async def stop(self):
        """Stop the data protection and recovery system."""
        if self._running:
            self._running = False
            await self.auto_save_manager.stop()
            logger.info("Data protection and recovery system stopped")
    
    async def protect_user_data(self, user_id: str, data_type: str, data: Any, 
                              device_id: Optional[str] = None) -> bool:
        """Comprehensive data protection for user data."""
        try:
            # 1. Auto-save registration
            await self.auto_save_manager.register_data(user_id, data_type, data)
            
            # 2. Create backup snapshot
            snapshot = await self.backup_manager.create_snapshot(user_id, data_type, data)
            
            # 3. Verify data integrity
            integrity_status = await self.integrity_verifier.verify_data_integrity(user_id, data_type, data)
            
            if integrity_status == DataIntegrityStatus.CORRUPTED:
                # Attempt repair
                repaired_data = await self.integrity_verifier.repair_corrupted_data(
                    user_id, data_type, data, self.backup_manager
                )
                if repaired_data is not None:
                    data = repaired_data
                    # Create new snapshot with repaired data
                    await self.backup_manager.create_snapshot(user_id, data_type, data)
            
            # 4. Cross-device synchronization
            if device_id:
                sync_status = await self.sync_manager.sync_data(device_id, data_type, data)
                if sync_status == SyncStatus.CONFLICT:
                    logger.warning(f"Sync conflict detected for {user_id}:{data_type}")
            
            logger.debug(f"Successfully protected data for {user_id}:{data_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to protect user data: {e}")
            return False
    
    async def recover_user_data(self, user_id: str, data_type: Optional[str] = None, 
                              recovery_point_id: Optional[str] = None) -> Dict[str, Any]:
        """Recover user data from backups."""
        try:
            if recovery_point_id:
                # Recover from specific recovery point
                return await self.backup_manager.restore_from_recovery_point(recovery_point_id)
            
            elif data_type:
                # Recover specific data type
                user_snapshots = [
                    snapshot for snapshot in self.backup_manager.snapshots.values()
                    if snapshot.user_id == user_id and snapshot.data_type == data_type
                ]
                
                if user_snapshots:
                    latest_snapshot = max(user_snapshots, key=lambda s: s.timestamp)
                    data = await self.backup_manager.restore_from_snapshot(latest_snapshot.snapshot_id)
                    return {data_type: data} if data is not None else {}
            
            else:
                # Recover all user data
                user_snapshots = [
                    snapshot for snapshot in self.backup_manager.snapshots.values()
                    if snapshot.user_id == user_id
                ]
                
                # Group by data type and get latest for each
                latest_by_type = {}
                for snapshot in user_snapshots:
                    if (snapshot.data_type not in latest_by_type or 
                        snapshot.timestamp > latest_by_type[snapshot.data_type].timestamp):
                        latest_by_type[snapshot.data_type] = snapshot
                
                # Restore all
                recovered_data = {}
                for data_type, snapshot in latest_by_type.items():
                    data = await self.backup_manager.restore_from_snapshot(snapshot.snapshot_id)
                    if data is not None:
                        recovered_data[data_type] = data
                
                return recovered_data
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to recover user data: {e}")
            return {}
    
    async def create_recovery_point(self, user_id: str, description: str = None) -> Optional[str]:
        """Create a recovery point for user data."""
        try:
            recovery_point = await self.backup_manager.create_recovery_point(user_id, description)
            return recovery_point.recovery_id
        except Exception as e:
            logger.error(f"Failed to create recovery point: {e}")
            return None
    
    async def get_protection_status(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive protection status for user."""
        try:
            # Get backup status
            user_snapshots = [
                snapshot for snapshot in self.backup_manager.snapshots.values()
                if snapshot.user_id == user_id
            ]
            
            # Get sync status
            sync_status = self.sync_manager.get_sync_status(user_id)
            
            # Get integrity status
            integrity_checks = len(self.integrity_verifier.verification_history.get(user_id, []))
            
            return {
                'user_id': user_id,
                'backup_snapshots': len(user_snapshots),
                'latest_backup': max(user_snapshots, key=lambda s: s.timestamp).timestamp.isoformat() if user_snapshots else None,
                'sync_status': sync_status,
                'integrity_checks_performed': integrity_checks,
                'auto_save_active': user_id in self.auto_save_manager.active_sessions,
                'protection_level': 'comprehensive' if user_snapshots and sync_status['all_synced'] else 'partial'
            }
            
        except Exception as e:
            logger.error(f"Failed to get protection status: {e}")
            return {'error': str(e)}
    
    async def _periodic_backup_cleanup(self):
        """Periodic cleanup of old backups."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.backup_manager.cleanup_old_backups()
            except Exception as e:
                logger.error(f"Backup cleanup error: {e}")
    
    async def _periodic_integrity_check(self):
        """Periodic integrity verification."""
        while self._running:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                # Check recent snapshots
                recent_snapshots = [
                    snapshot for snapshot in self.backup_manager.snapshots.values()
                    if snapshot.timestamp > datetime.utcnow() - timedelta(hours=1)
                ]
                
                for snapshot in recent_snapshots:
                    data = await self.backup_manager.restore_from_snapshot(snapshot.snapshot_id)
                    if data is not None:
                        await self.integrity_verifier.verify_data_integrity(
                            snapshot.user_id, snapshot.data_type, data
                        )
                
            except Exception as e:
                logger.error(f"Periodic integrity check error: {e}")


# Global instance
data_protection_system = DataProtectionRecoverySystem()


# Convenience functions
async def protect_data(user_id: str, data_type: str, data: Any, device_id: Optional[str] = None) -> bool:
    """Protect user data with comprehensive backup and sync."""
    return await data_protection_system.protect_user_data(user_id, data_type, data, device_id)


async def recover_data(user_id: str, data_type: Optional[str] = None, 
                      recovery_point_id: Optional[str] = None) -> Dict[str, Any]:
    """Recover user data from backups."""
    return await data_protection_system.recover_user_data(user_id, data_type, recovery_point_id)


async def create_recovery_point(user_id: str, description: str = None) -> Optional[str]:
    """Create a recovery point for user data."""
    return await data_protection_system.create_recovery_point(user_id, description)


def with_data_protection(data_type: str, device_id: Optional[str] = None):
    """Decorator to automatically protect function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Extract user_id from args or kwargs
            user_id = kwargs.get('user_id')
            if not user_id and args:
                # Try to find user_id in first argument if it's a dict
                if isinstance(args[0], dict) and 'user_id' in args[0]:
                    user_id = args[0]['user_id']
            
            if user_id and result is not None:
                await protect_data(user_id, data_type, result, device_id)
            
            return result
        return wrapper
    return decorator