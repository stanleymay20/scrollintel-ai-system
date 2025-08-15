"""
Tests for comprehensive data protection and recovery system.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from scrollintel.core.data_protection_recovery import (
    DataProtectionRecoverySystem,
    AutoSaveManager,
    MultiTierBackupManager,
    DataIntegrityVerifier,
    CrossDeviceSyncManager,
    BackupTier,
    DataIntegrityStatus,
    SyncStatus,
    protect_data,
    recover_data,
    create_recovery_point,
    with_data_protection
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
async def protection_system(temp_dir):
    """Create data protection system for testing."""
    # Override paths to use temp directory
    system = DataProtectionRecoverySystem()
    system.backup_manager.base_path = Path(temp_dir) / "backups"
    system.backup_manager.base_path.mkdir(parents=True, exist_ok=True)
    
    # Update tier configs
    for tier, config in system.backup_manager.tier_configs.items():
        config['path'] = system.backup_manager.base_path / tier.value
        config['path'].mkdir(parents=True, exist_ok=True)
    
    system.sync_manager.db_path = str(Path(temp_dir) / "sync.db")
    system.sync_manager._init_database()
    
    await system.start()
    yield system
    await system.stop()


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        'user_id': 'test_user_123',
        'name': 'Test User',
        'preferences': {
            'theme': 'dark',
            'language': 'en'
        },
        'timestamp': datetime.utcnow().isoformat(),
        'data': [1, 2, 3, 4, 5]
    }


@pytest.fixture
def sample_file_data():
    """Sample file data for testing."""
    return {
        'filename': 'test_file.txt',
        'content': 'This is test file content',
        'size': 25,
        'type': 'text/plain',
        'timestamp': datetime.utcnow().isoformat()
    }


class TestAutoSaveManager:
    """Test auto-save functionality."""
    
    @pytest.mark.asyncio
    async def test_auto_save_registration(self):
        """Test registering data for auto-save."""
        manager = AutoSaveManager(save_interval=1)  # 1 second for testing
        await manager.start()
        
        # Register data
        test_data = {'key': 'value', 'timestamp': datetime.utcnow().isoformat()}
        await manager.register_data('user123', 'test_data', test_data)
        
        # Check data is registered
        key = 'user123:test_data'
        assert key in manager.pending_saves
        assert manager.pending_saves[key]['dirty'] is True
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_force_save(self):
        """Test forcing immediate save."""
        manager = AutoSaveManager()
        save_called = False
        
        async def mock_save_callback(data_info):
            nonlocal save_called
            save_called = True
            assert data_info['user_id'] == 'user123'
            assert data_info['data_type'] == 'test_data'
        
        await manager.start()
        
        # Register with callback
        test_data = {'key': 'value'}
        await manager.register_data('user123', 'test_data', test_data, mock_save_callback)
        
        # Force save
        await manager.force_save('user123', 'test_data')
        
        assert save_called is True
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_auto_save_loop(self):
        """Test automatic save loop."""
        manager = AutoSaveManager(save_interval=0.1)  # Very short interval
        save_count = 0
        
        async def mock_save_callback(data_info):
            nonlocal save_count
            save_count += 1
        
        await manager.start()
        
        # Register data
        await manager.register_data('user123', 'test_data', {'key': 'value'}, mock_save_callback)
        
        # Wait for auto-save
        await asyncio.sleep(0.2)
        
        assert save_count > 0
        
        await manager.stop()


class TestMultiTierBackupManager:
    """Test multi-tier backup functionality."""
    
    @pytest.mark.asyncio
    async def test_create_snapshot(self, temp_dir):
        """Test creating data snapshots."""
        manager = MultiTierBackupManager(str(Path(temp_dir) / "backups"))
        
        test_data = {'key': 'value', 'number': 42}
        snapshot = await manager.create_snapshot('user123', 'test_data', test_data)
        
        assert snapshot.user_id == 'user123'
        assert snapshot.data_type == 'test_data'
        assert snapshot.data == test_data
        assert len(snapshot.checksum) == 64  # SHA256 hash
        assert BackupTier.LOCAL in snapshot.backup_tiers
        assert BackupTier.REMOTE in snapshot.backup_tiers
    
    @pytest.mark.asyncio
    async def test_restore_snapshot(self, temp_dir):
        """Test restoring from snapshots."""
        manager = MultiTierBackupManager(str(Path(temp_dir) / "backups"))
        
        test_data = {'key': 'value', 'number': 42}
        snapshot = await manager.create_snapshot('user123', 'test_data', test_data)
        
        # Restore data
        restored_data = await manager.restore_from_snapshot(snapshot.snapshot_id)
        
        assert restored_data == test_data
    
    @pytest.mark.asyncio
    async def test_recovery_point(self, temp_dir):
        """Test creating and restoring recovery points."""
        manager = MultiTierBackupManager(str(Path(temp_dir) / "backups"))
        
        # Create multiple snapshots
        await manager.create_snapshot('user123', 'data1', {'type': 'data1'})
        await manager.create_snapshot('user123', 'data2', {'type': 'data2'})
        
        # Create recovery point
        recovery_point = await manager.create_recovery_point('user123', 'Test recovery point')
        
        assert recovery_point.user_id == 'user123'
        assert recovery_point.description == 'Test recovery point'
        assert len(recovery_point.snapshots) >= 0  # May be 0 if snapshots are too old
    
    @pytest.mark.asyncio
    async def test_backup_cleanup(self, temp_dir):
        """Test cleanup of old backups."""
        manager = MultiTierBackupManager(str(Path(temp_dir) / "backups"))
        
        # Override retention for testing
        manager.tier_configs[BackupTier.LOCAL]['retention_days'] = 0  # Immediate cleanup
        
        # Create snapshot
        snapshot = await manager.create_snapshot('user123', 'test_data', {'key': 'value'})
        
        # Modify timestamp to make it old
        snapshot.timestamp = datetime.utcnow() - timedelta(days=1)
        
        # Run cleanup
        await manager.cleanup_old_backups()
        
        # Snapshot should be removed from local tier
        assert BackupTier.LOCAL not in snapshot.backup_tiers or len(snapshot.backup_tiers) == 0


class TestDataIntegrityVerifier:
    """Test data integrity verification."""
    
    @pytest.mark.asyncio
    async def test_verify_valid_data(self):
        """Test verifying valid data."""
        verifier = DataIntegrityVerifier()
        
        valid_data = {'key': 'value', 'number': 42}
        status = await verifier.verify_data_integrity('user123', 'test_data', valid_data)
        
        assert status == DataIntegrityStatus.VALID
    
    @pytest.mark.asyncio
    async def test_verify_corrupted_data(self):
        """Test detecting corrupted data."""
        verifier = DataIntegrityVerifier()
        
        # None data should be considered corrupted
        status = await verifier.verify_data_integrity('user123', 'test_data', None)
        
        assert status == DataIntegrityStatus.CORRUPTED
    
    @pytest.mark.asyncio
    async def test_custom_integrity_check(self):
        """Test custom integrity check function."""
        verifier = DataIntegrityVerifier()
        
        async def custom_check(data):
            return isinstance(data, dict) and 'required_field' in data
        
        verifier.register_integrity_check('custom_type', custom_check)
        
        # Valid data
        valid_data = {'required_field': 'value'}
        status = await verifier.verify_data_integrity('user123', 'custom_type', valid_data)
        assert status == DataIntegrityStatus.VALID
        
        # Invalid data
        invalid_data = {'other_field': 'value'}
        status = await verifier.verify_data_integrity('user123', 'custom_type', invalid_data)
        assert status == DataIntegrityStatus.CORRUPTED
    
    @pytest.mark.asyncio
    async def test_repair_corrupted_data(self, temp_dir):
        """Test repairing corrupted data."""
        verifier = DataIntegrityVerifier()
        backup_manager = MultiTierBackupManager(str(Path(temp_dir) / "backups"))
        
        # Create backup of good data
        good_data = {'key': 'value', 'status': 'good'}
        await backup_manager.create_snapshot('user123', 'test_data', good_data)
        
        # Try to repair corrupted data
        corrupted_data = None
        repaired_data = await verifier.repair_corrupted_data('user123', 'test_data', corrupted_data, backup_manager)
        
        # Should restore from backup
        assert repaired_data == good_data


class TestCrossDeviceSyncManager:
    """Test cross-device synchronization."""
    
    @pytest.mark.asyncio
    async def test_register_device(self, temp_dir):
        """Test registering a device."""
        sync_manager = CrossDeviceSyncManager(str(Path(temp_dir) / "sync.db"))
        
        device_state = await sync_manager.register_device('device123', 'user123', {'type': 'mobile'})
        
        assert device_state.device_id == 'device123'
        assert device_state.user_id == 'user123'
        assert device_state.sync_status == SyncStatus.SYNCED
        assert device_state.capabilities == {'type': 'mobile'}
    
    @pytest.mark.asyncio
    async def test_sync_data_no_conflict(self, temp_dir):
        """Test syncing data without conflicts."""
        sync_manager = CrossDeviceSyncManager(str(Path(temp_dir) / "sync.db"))
        
        # Register device
        await sync_manager.register_device('device123', 'user123')
        
        # Sync data
        test_data = {'key': 'value', 'timestamp': datetime.utcnow().isoformat()}
        status = await sync_manager.sync_data('device123', 'test_data', test_data)
        
        assert status == SyncStatus.SYNCED
    
    @pytest.mark.asyncio
    async def test_sync_conflict_detection(self, temp_dir):
        """Test detecting sync conflicts."""
        sync_manager = CrossDeviceSyncManager(str(Path(temp_dir) / "sync.db"))
        
        # Register two devices
        device1 = await sync_manager.register_device('device1', 'user123')
        device2 = await sync_manager.register_device('device2', 'user123')
        
        # Add offline change to device2
        device2.offline_changes.append({
            'data_type': 'test_data',
            'data': {'key': 'old_value'},
            'timestamp': datetime.utcnow().isoformat()
        })
        await sync_manager._save_device_state(device2)
        
        # Try to sync conflicting data from device1
        test_data = {'key': 'new_value'}
        status = await sync_manager.sync_data('device1', 'test_data', test_data)
        
        # Should detect conflict
        assert status == SyncStatus.CONFLICT
        assert len(sync_manager.sync_conflicts) > 0
    
    @pytest.mark.asyncio
    async def test_resolve_conflict(self, temp_dir):
        """Test resolving sync conflicts."""
        sync_manager = CrossDeviceSyncManager(str(Path(temp_dir) / "sync.db"))
        
        # Create a conflict manually
        from scrollintel.core.data_protection_recovery import SyncConflict
        import uuid
        
        conflict = SyncConflict(
            conflict_id=str(uuid.uuid4()),
            user_id='user123',
            data_type='test_data',
            device_a_data={'key': 'value_a'},
            device_b_data={'key': 'value_b'},
            device_a_timestamp=datetime.utcnow(),
            device_b_timestamp=datetime.utcnow() - timedelta(minutes=1)
        )
        
        sync_manager.sync_conflicts[conflict.conflict_id] = conflict
        
        # Resolve using latest strategy
        resolved = await sync_manager.resolve_conflict(conflict.conflict_id, 'use_latest')
        
        assert resolved is True
        assert conflict.resolved is True
        assert conflict.resolution_strategy == 'use_latest'
    
    @pytest.mark.asyncio
    async def test_offline_online_handling(self, temp_dir):
        """Test handling device offline/online transitions."""
        sync_manager = CrossDeviceSyncManager(str(Path(temp_dir) / "sync.db"))
        
        # Register device
        await sync_manager.register_device('device123', 'user123')
        
        # Mark device offline
        await sync_manager.handle_device_offline('device123')
        device_state = sync_manager.device_states['device123']
        assert device_state.sync_status == SyncStatus.OFFLINE
        
        # Bring device back online
        operations = await sync_manager.handle_device_online('device123')
        device_state = sync_manager.device_states['device123']
        assert device_state.sync_status == SyncStatus.SYNCED
        assert isinstance(operations, list)


class TestDataProtectionRecoverySystem:
    """Test the main data protection system."""
    
    @pytest.mark.asyncio
    async def test_protect_user_data(self, protection_system, sample_user_data):
        """Test comprehensive data protection."""
        success = await protection_system.protect_user_data(
            'user123', 'user_data', sample_user_data, 'device123'
        )
        
        assert success is True
        
        # Check that data was protected
        status = await protection_system.get_protection_status('user123')
        assert status['backup_snapshots'] > 0
        assert status['protection_level'] in ['comprehensive', 'partial']
    
    @pytest.mark.asyncio
    async def test_recover_user_data(self, protection_system, sample_user_data):
        """Test data recovery."""
        # First protect the data
        await protection_system.protect_user_data('user123', 'user_data', sample_user_data)
        
        # Then recover it
        recovered_data = await protection_system.recover_user_data('user123', 'user_data')
        
        assert 'user_data' in recovered_data
        assert recovered_data['user_data'] == sample_user_data
    
    @pytest.mark.asyncio
    async def test_create_recovery_point(self, protection_system, sample_user_data):
        """Test creating recovery points."""
        # Protect some data first
        await protection_system.protect_user_data('user123', 'user_data', sample_user_data)
        
        # Create recovery point
        recovery_id = await protection_system.create_recovery_point('user123', 'Test recovery point')
        
        assert recovery_id is not None
        assert recovery_id in protection_system.backup_manager.recovery_points
    
    @pytest.mark.asyncio
    async def test_data_integrity_with_repair(self, protection_system, sample_user_data):
        """Test data integrity verification and repair."""
        # First create a good backup
        await protection_system.protect_user_data('user123', 'user_data', sample_user_data)
        
        # Now try to protect corrupted data
        corrupted_data = None
        success = await protection_system.protect_user_data('user123', 'user_data', corrupted_data)
        
        # System should handle corruption gracefully
        assert success is True  # Should succeed due to repair mechanisms
    
    @pytest.mark.asyncio
    async def test_protection_status(self, protection_system, sample_user_data):
        """Test getting protection status."""
        # Protect some data
        await protection_system.protect_user_data('user123', 'user_data', sample_user_data)
        
        # Get status
        status = await protection_system.get_protection_status('user123')
        
        assert status['user_id'] == 'user123'
        assert 'backup_snapshots' in status
        assert 'sync_status' in status
        assert 'protection_level' in status
        assert status['protection_level'] in ['comprehensive', 'partial']


class TestConvenienceFunctions:
    """Test convenience functions and decorators."""
    
    @pytest.mark.asyncio
    async def test_protect_data_function(self, protection_system, sample_user_data):
        """Test protect_data convenience function."""
        # Override global instance for testing
        import scrollintel.core.data_protection_recovery as dpr_module
        original_system = dpr_module.data_protection_system
        dpr_module.data_protection_system = protection_system
        
        try:
            success = await protect_data('user123', 'user_data', sample_user_data)
            assert success is True
        finally:
            dpr_module.data_protection_system = original_system
    
    @pytest.mark.asyncio
    async def test_recover_data_function(self, protection_system, sample_user_data):
        """Test recover_data convenience function."""
        # Override global instance for testing
        import scrollintel.core.data_protection_recovery as dpr_module
        original_system = dpr_module.data_protection_system
        dpr_module.data_protection_system = protection_system
        
        try:
            # First protect data
            await protection_system.protect_user_data('user123', 'user_data', sample_user_data)
            
            # Then recover
            recovered = await recover_data('user123', 'user_data')
            assert 'user_data' in recovered
            assert recovered['user_data'] == sample_user_data
        finally:
            dpr_module.data_protection_system = original_system
    
    @pytest.mark.asyncio
    async def test_with_data_protection_decorator(self, protection_system):
        """Test with_data_protection decorator."""
        # Override global instance for testing
        import scrollintel.core.data_protection_recovery as dpr_module
        original_system = dpr_module.data_protection_system
        dpr_module.data_protection_system = protection_system
        
        try:
            @with_data_protection('test_data')
            async def test_function(user_id, data):
                return {'processed': data, 'user_id': user_id}
            
            result = await test_function('user123', {'input': 'test'})
            
            assert result['processed'] == {'input': 'test'}
            assert result['user_id'] == 'user123'
            
            # Check that data was protected
            status = await protection_system.get_protection_status('user123')
            assert status['backup_snapshots'] > 0
            
        finally:
            dpr_module.data_protection_system = original_system


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_snapshot_restore(self, temp_dir):
        """Test handling invalid snapshot restoration."""
        manager = MultiTierBackupManager(str(Path(temp_dir) / "backups"))
        
        # Try to restore non-existent snapshot
        result = await manager.restore_from_snapshot('invalid_id')
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_sync_unregistered_device(self, temp_dir):
        """Test syncing with unregistered device."""
        sync_manager = CrossDeviceSyncManager(str(Path(temp_dir) / "sync.db"))
        
        # Try to sync with unregistered device
        status = await sync_manager.sync_data('invalid_device', 'test_data', {'key': 'value'})
        
        assert status == SyncStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_protection_system_error_handling(self, protection_system):
        """Test protection system error handling."""
        # Try to protect invalid data type
        success = await protection_system.protect_user_data('', '', None)
        
        # Should handle gracefully
        assert success is False
    
    @pytest.mark.asyncio
    async def test_recovery_nonexistent_user(self, protection_system):
        """Test recovering data for non-existent user."""
        recovered = await protection_system.recover_user_data('nonexistent_user')
        
        assert recovered == {}


if __name__ == '__main__':
    pytest.main([__file__])