"""
Tests for cross-device and cross-session continuity system.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.core.cross_device_continuity import (
    CrossDeviceContinuityManager, StateManager, ConflictResolver, OfflineManager,
    SessionState, StateChange, SyncConflict, DeviceInfo, DeviceType,
    ConflictResolutionStrategy, create_user_session, restore_user_session,
    sync_user_state, enable_offline_mode, sync_after_reconnect
)


class TestStateManager:
    """Test the StateManager class."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def state_manager(self, temp_storage):
        """Create StateManager instance with temporary storage."""
        return StateManager(temp_storage)
    
    @pytest.mark.asyncio
    async def test_save_and_load_session_state(self, state_manager):
        """Test saving and loading session state."""
        # Create test session state
        session_state = SessionState(
            session_id="test_session_123",
            user_id="user123",
            device_id="device456",
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            state_data={"key": "value", "number": 42}
        )
        
        # Save session state
        await state_manager.save_session_state(session_state)
        
        # Load session state
        loaded_state = await state_manager.load_session_state("test_session_123")
        
        assert loaded_state is not None
        assert loaded_state.session_id == session_state.session_id
        assert loaded_state.user_id == session_state.user_id
        assert loaded_state.device_id == session_state.device_id
        assert loaded_state.state_data == session_state.state_data
        assert loaded_state.checksum == session_state.checksum
    
    @pytest.mark.asyncio
    async def test_get_user_sessions(self, state_manager):
        """Test getting all sessions for a user."""
        user_id = "user123"
        
        # Create multiple sessions for the user
        sessions = []
        for i in range(3):
            session = SessionState(
                session_id=f"session_{i}",
                user_id=user_id,
                device_id=f"device_{i}",
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                state_data={"session_number": i}
            )
            sessions.append(session)
            await state_manager.save_session_state(session)
        
        # Get user sessions
        user_sessions = await state_manager.get_user_sessions(user_id)
        
        assert len(user_sessions) == 3
        session_ids = [s.session_id for s in user_sessions]
        assert "session_0" in session_ids
        assert "session_1" in session_ids
        assert "session_2" in session_ids
    
    def test_record_state_change(self, state_manager):
        """Test recording state changes."""
        change = StateChange(
            change_id="change123",
            session_id="session123",
            user_id="user123",
            device_id="device123",
            timestamp=datetime.utcnow(),
            path="test.path",
            old_value="old",
            new_value="new",
            operation="set"
        )
        
        state_manager.record_state_change(change)
        
        assert len(state_manager.change_log) == 1
        assert state_manager.change_log[0] == change
    
    def test_get_changes_since(self, state_manager):
        """Test getting changes since a timestamp."""
        user_id = "user123"
        base_time = datetime.utcnow()
        
        # Create changes at different times
        changes = []
        for i in range(3):
            change = StateChange(
                change_id=f"change_{i}",
                session_id="session123",
                user_id=user_id,
                device_id="device123",
                timestamp=base_time + timedelta(minutes=i),
                path=f"path_{i}",
                old_value=f"old_{i}",
                new_value=f"new_{i}",
                operation="set"
            )
            changes.append(change)
            state_manager.record_state_change(change)
        
        # Get changes since middle time
        since_time = base_time + timedelta(minutes=1)
        recent_changes = state_manager.get_changes_since(since_time, user_id)
        
        assert len(recent_changes) == 1
        assert recent_changes[0].change_id == "change_2"


class TestConflictResolver:
    """Test the ConflictResolver class."""
    
    @pytest.fixture
    def conflict_resolver(self):
        """Create ConflictResolver instance."""
        return ConflictResolver()
    
    @pytest.mark.asyncio
    async def test_resolve_last_write_wins(self, conflict_resolver):
        """Test last write wins conflict resolution."""
        now = datetime.utcnow()
        
        local_change = StateChange(
            change_id="local",
            session_id="session123",
            user_id="user123",
            device_id="device1",
            timestamp=now,
            path="test.value",
            old_value="original",
            new_value="local_value",
            operation="set"
        )
        
        remote_change = StateChange(
            change_id="remote",
            session_id="session123",
            user_id="user123",
            device_id="device2",
            timestamp=now + timedelta(seconds=1),  # Remote is newer
            path="test.value",
            old_value="original",
            new_value="remote_value",
            operation="set"
        )
        
        conflict = SyncConflict(
            conflict_id="conflict123",
            session_id="session123",
            path="test.value",
            local_change=local_change,
            remote_change=remote_change,
            resolution_strategy=ConflictResolutionStrategy.LAST_WRITE_WINS
        )
        
        result = await conflict_resolver.resolve_conflict(conflict)
        
        assert result == "remote_value"  # Remote is newer
        assert conflict.resolved is True
    
    @pytest.mark.asyncio
    async def test_resolve_merge_changes_dict(self, conflict_resolver):
        """Test merging dictionary changes."""
        local_change = StateChange(
            change_id="local",
            session_id="session123",
            user_id="user123",
            device_id="device1",
            timestamp=datetime.utcnow(),
            path="test.dict",
            old_value={"a": 1},
            new_value={"a": 1, "b": 2},
            operation="set"
        )
        
        remote_change = StateChange(
            change_id="remote",
            session_id="session123",
            user_id="user123",
            device_id="device2",
            timestamp=datetime.utcnow(),
            path="test.dict",
            old_value={"a": 1},
            new_value={"a": 1, "c": 3},
            operation="set"
        )
        
        conflict = SyncConflict(
            conflict_id="conflict123",
            session_id="session123",
            path="test.dict",
            local_change=local_change,
            remote_change=remote_change,
            resolution_strategy=ConflictResolutionStrategy.MERGE_CHANGES
        )
        
        result = await conflict_resolver.resolve_conflict(conflict)
        
        assert result == {"a": 1, "b": 2, "c": 3}  # Merged
        assert conflict.resolved is True
    
    @pytest.mark.asyncio
    async def test_resolve_keep_both(self, conflict_resolver):
        """Test keeping both values in conflict."""
        local_change = StateChange(
            change_id="local",
            session_id="session123",
            user_id="user123",
            device_id="device1",
            timestamp=datetime.utcnow(),
            path="test.value",
            old_value="original",
            new_value="local_value",
            operation="set"
        )
        
        remote_change = StateChange(
            change_id="remote",
            session_id="session123",
            user_id="user123",
            device_id="device2",
            timestamp=datetime.utcnow(),
            path="test.value",
            old_value="original",
            new_value="remote_value",
            operation="set"
        )
        
        conflict = SyncConflict(
            conflict_id="conflict123",
            session_id="session123",
            path="test.value",
            local_change=local_change,
            remote_change=remote_change,
            resolution_strategy=ConflictResolutionStrategy.KEEP_BOTH
        )
        
        result = await conflict_resolver.resolve_conflict(conflict)
        
        assert isinstance(result, dict)
        assert result["local"] == "local_value"
        assert result["remote"] == "remote_value"
        assert "conflict_id" in result
        assert conflict.resolved is True


class TestOfflineManager:
    """Test the OfflineManager class."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def offline_manager(self, temp_storage):
        """Create OfflineManager instance with temporary storage."""
        return OfflineManager(temp_storage)
    
    def test_set_offline_mode(self, offline_manager):
        """Test setting offline mode."""
        assert offline_manager.is_offline is False
        
        offline_manager.set_offline_mode(True)
        assert offline_manager.is_offline is True
        
        offline_manager.set_offline_mode(False)
        assert offline_manager.is_offline is False
    
    def test_queue_offline_change(self, offline_manager):
        """Test queuing offline changes."""
        change = StateChange(
            change_id="change123",
            session_id="session123",
            user_id="user123",
            device_id="device123",
            timestamp=datetime.utcnow(),
            path="test.path",
            old_value="old",
            new_value="new",
            operation="set"
        )
        
        offline_manager.queue_offline_change(change)
        
        assert len(offline_manager.offline_queue) == 1
        assert offline_manager.offline_queue[0] == change
    
    def test_get_offline_capabilities(self, offline_manager):
        """Test getting offline capabilities."""
        capabilities = offline_manager.get_offline_capabilities()
        
        assert isinstance(capabilities, dict)
        assert capabilities["read_cached_data"] is True
        assert capabilities["sync_when_online"] is True
    
    def test_can_perform_offline(self, offline_manager):
        """Test checking offline operation capabilities."""
        assert offline_manager.can_perform_offline("read_cached_data") is True
        assert offline_manager.can_perform_offline("create_new_items") is True
        assert offline_manager.can_perform_offline("nonexistent_operation") is False


class TestCrossDeviceContinuityManager:
    """Test the main CrossDeviceContinuityManager class."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def continuity_manager(self, temp_storage):
        """Create CrossDeviceContinuityManager instance."""
        # Mock the WebSocket server to avoid actual network operations
        with patch('scrollintel.core.cross_device_continuity.WebSocketManager.start_server'):
            manager = CrossDeviceContinuityManager()
            manager.state_manager.storage_path = temp_storage
            manager.offline_manager.storage_path = temp_storage
            yield manager
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_register_device(self, continuity_manager):
        """Test device registration."""
        device_info = await continuity_manager.register_device(
            device_id="device123",
            device_type=DeviceType.DESKTOP,
            user_agent="Test Agent",
            capabilities={"sync": True}
        )
        
        assert device_info.device_id == "device123"
        assert device_info.device_type == DeviceType.DESKTOP
        assert device_info.user_agent == "Test Agent"
        assert device_info.capabilities == {"sync": True}
        assert "device123" in continuity_manager.devices
    
    @pytest.mark.asyncio
    async def test_create_session(self, continuity_manager):
        """Test session creation."""
        session = await continuity_manager.create_session(
            user_id="user123",
            device_id="device456",
            initial_state={"key": "value"}
        )
        
        assert session.user_id == "user123"
        assert session.device_id == "device456"
        assert session.state_data == {"key": "value"}
        assert session.version == 1
        assert session.session_id in continuity_manager.active_sessions
    
    @pytest.mark.asyncio
    async def test_restore_session(self, continuity_manager):
        """Test session restoration."""
        # Create a session first
        original_session = await continuity_manager.create_session(
            user_id="user123",
            device_id="device456",
            initial_state={"key": "value"}
        )
        
        # Remove from active sessions to simulate restoration
        del continuity_manager.active_sessions[original_session.session_id]
        
        # Restore session
        restored_session = await continuity_manager.restore_session(original_session.session_id)
        
        assert restored_session is not None
        assert restored_session.session_id == original_session.session_id
        assert restored_session.user_id == original_session.user_id
        assert restored_session.state_data == original_session.state_data
    
    @pytest.mark.asyncio
    async def test_update_session_state(self, continuity_manager):
        """Test updating session state."""
        # Create a session
        session = await continuity_manager.create_session(
            user_id="user123",
            device_id="device456",
            initial_state={"key": "value"}
        )
        
        # Update session state
        success = await continuity_manager.update_session_state(
            session_id=session.session_id,
            state_updates={"key": "updated_value", "new_key": "new_value"},
            device_id="device456"
        )
        
        assert success is True
        
        # Check that state was updated
        updated_session = continuity_manager.active_sessions[session.session_id]
        assert updated_session.state_data["key"] == "updated_value"
        assert updated_session.state_data["new_key"] == "new_value"
        assert updated_session.version == 2
    
    @pytest.mark.asyncio
    async def test_sync_across_devices(self, continuity_manager):
        """Test synchronizing across devices."""
        user_id = "user123"
        
        # Create multiple sessions for the user
        session1 = await continuity_manager.create_session(
            user_id=user_id,
            device_id="device1",
            initial_state={"key": "value1"}
        )
        
        session2 = await continuity_manager.create_session(
            user_id=user_id,
            device_id="device2",
            initial_state={"key": "value2"}
        )
        
        # Update one session to be newer
        await continuity_manager.update_session_state(
            session_id=session1.session_id,
            state_updates={"key": "latest_value"},
            device_id="device1"
        )
        
        # Sync across devices
        sync_results = await continuity_manager.sync_across_devices(user_id)
        
        assert sync_results["status"] == "success"
        assert sync_results["synced_sessions"] >= 1
    
    @pytest.mark.asyncio
    async def test_handle_multi_tab_sync(self, continuity_manager):
        """Test multi-tab synchronization."""
        user_id = "user123"
        tab_id = "tab456"
        
        result = await continuity_manager.handle_multi_tab_sync(
            user_id=user_id,
            tab_id=tab_id,
            state_updates={"current_page": "/dashboard", "zoom": 1.2}
        )
        
        assert result["status"] == "synced"
        assert "session_id" in result
        assert result["version"] >= 1
    
    @pytest.mark.asyncio
    async def test_enable_offline_mode(self, continuity_manager):
        """Test enabling offline mode."""
        user_id = "user123"
        device_id = "device456"
        
        # Create a session first
        await continuity_manager.create_session(
            user_id=user_id,
            device_id=device_id,
            initial_state={"key": "value"}
        )
        
        result = await continuity_manager.enable_offline_mode(user_id, device_id)
        
        assert result["status"] == "offline_enabled"
        assert "capabilities" in result
        assert "cached_sessions" in result
        assert continuity_manager.offline_manager.is_offline is True
    
    @pytest.mark.asyncio
    async def test_sync_when_reconnected(self, continuity_manager):
        """Test syncing when reconnected after offline."""
        user_id = "user123"
        device_id = "device456"
        
        # Enable offline mode first
        await continuity_manager.enable_offline_mode(user_id, device_id)
        
        # Sync when reconnected
        result = await continuity_manager.sync_when_reconnected(user_id, device_id)
        
        assert result["status"] == "reconnected"
        assert "sync_results" in result
        assert continuity_manager.offline_manager.is_offline is False
    
    @pytest.mark.asyncio
    async def test_get_sync_status(self, continuity_manager):
        """Test getting sync status."""
        user_id = "user123"
        
        # Create a session
        await continuity_manager.create_session(
            user_id=user_id,
            device_id="device456",
            initial_state={"key": "value"}
        )
        
        status = await continuity_manager.get_sync_status(user_id)
        
        assert status["user_id"] == user_id
        assert status["total_sessions"] >= 1
        assert status["active_devices"] >= 1
        assert "pending_conflicts" in status
        assert "offline_mode" in status


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    async def test_create_user_session(self):
        """Test create_user_session convenience function."""
        with patch('scrollintel.core.cross_device_continuity.continuity_manager') as mock_manager:
            mock_manager.devices = {}
            mock_manager.register_device = AsyncMock()
            mock_manager.create_session = AsyncMock()
            
            mock_session = Mock()
            mock_manager.create_session.return_value = mock_session
            
            result = await create_user_session(
                user_id="user123",
                device_id="device456",
                device_type="desktop",
                initial_state={"key": "value"}
            )
            
            assert result == mock_session
            mock_manager.register_device.assert_called_once()
            mock_manager.create_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_restore_user_session(self):
        """Test restore_user_session convenience function."""
        with patch('scrollintel.core.cross_device_continuity.continuity_manager') as mock_manager:
            mock_session = Mock()
            mock_manager.restore_session = AsyncMock(return_value=mock_session)
            
            result = await restore_user_session("session123")
            
            assert result == mock_session
            mock_manager.restore_session.assert_called_once_with("session123")
    
    @pytest.mark.asyncio
    async def test_sync_user_state(self):
        """Test sync_user_state convenience function."""
        with patch('scrollintel.core.cross_device_continuity.continuity_manager') as mock_manager:
            mock_manager.update_session_state = AsyncMock(return_value=True)
            
            result = await sync_user_state(
                user_id="user123",
                session_id="session456",
                state_updates={"key": "value"},
                device_id="device789"
            )
            
            assert result is True
            mock_manager.update_session_state.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enable_offline_mode_function(self):
        """Test enable_offline_mode convenience function."""
        with patch('scrollintel.core.cross_device_continuity.continuity_manager') as mock_manager:
            mock_result = {"status": "offline_enabled"}
            mock_manager.enable_offline_mode = AsyncMock(return_value=mock_result)
            
            result = await enable_offline_mode("user123", "device456")
            
            assert result == mock_result
            mock_manager.enable_offline_mode.assert_called_once_with("user123", "device456")
    
    @pytest.mark.asyncio
    async def test_sync_after_reconnect_function(self):
        """Test sync_after_reconnect convenience function."""
        with patch('scrollintel.core.cross_device_continuity.continuity_manager') as mock_manager:
            mock_result = {"status": "reconnected"}
            mock_manager.sync_when_reconnected = AsyncMock(return_value=mock_result)
            
            result = await sync_after_reconnect("user123", "device456")
            
            assert result == mock_result
            mock_manager.sync_when_reconnected.assert_called_once_with("user123", "device456")


class TestSessionState:
    """Test SessionState class."""
    
    def test_session_state_creation(self):
        """Test creating a SessionState."""
        state_data = {"key": "value", "number": 42}
        session = SessionState(
            session_id="test123",
            user_id="user123",
            device_id="device456",
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            state_data=state_data
        )
        
        assert session.session_id == "test123"
        assert session.user_id == "user123"
        assert session.device_id == "device456"
        assert session.state_data == state_data
        assert session.version == 1
        assert session.checksum != ""
    
    def test_update_state(self):
        """Test updating session state."""
        session = SessionState(
            session_id="test123",
            user_id="user123",
            device_id="device456",
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            state_data={"key": "value"}
        )
        
        original_version = session.version
        original_checksum = session.checksum
        
        session.update_state({"key": "updated_value", "new_key": "new_value"})
        
        assert session.version == original_version + 1
        assert session.checksum != original_checksum
        assert session.state_data["key"] == "updated_value"
        assert session.state_data["new_key"] == "new_value"
    
    def test_checksum_calculation(self):
        """Test checksum calculation for state integrity."""
        state_data = {"key": "value", "number": 42}
        
        session1 = SessionState(
            session_id="test123",
            user_id="user123",
            device_id="device456",
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            state_data=state_data.copy()
        )
        
        session2 = SessionState(
            session_id="test456",
            user_id="user456",
            device_id="device789",
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            state_data=state_data.copy()
        )
        
        # Same state data should produce same checksum
        assert session1.checksum == session2.checksum
        
        # Different state data should produce different checksum
        session2.state_data["key"] = "different_value"
        session2.checksum = session2._calculate_checksum()
        assert session1.checksum != session2.checksum


if __name__ == "__main__":
    pytest.main([__file__])