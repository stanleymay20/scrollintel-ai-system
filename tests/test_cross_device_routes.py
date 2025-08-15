"""
Integration tests for cross-device continuity API routes.
"""

import pytest
import json
import tempfile
import shutil
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock

from scrollintel.api.routes.cross_device_routes import router
from scrollintel.models.cross_device_models import (
    CreateSessionRequest, UpdateSessionRequest, SyncRequest, OfflineModeRequest
)
from fastapi import FastAPI

# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestCrossDeviceRoutes:
    """Test cross-device continuity API routes."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_continuity_manager(self, temp_storage):
        """Mock the continuity manager."""
        with patch('scrollintel.api.routes.cross_device_routes.continuity_manager') as mock_manager:
            # Mock basic properties
            mock_manager.devices = {}
            mock_manager.active_sessions = {}
            mock_manager.sync_conflicts = {}
            mock_manager.offline_manager.is_offline = False
            mock_manager.offline_manager.offline_queue = []
            
            # Mock state manager
            mock_manager.state_manager.storage_path = temp_storage
            mock_manager.state_manager.get_user_sessions = AsyncMock(return_value=[])
            
            yield mock_manager
    
    def test_create_session_success(self, mock_continuity_manager):
        """Test successful session creation."""
        # Mock the create_user_session function
        with patch('scrollintel.api.routes.cross_device_routes.create_user_session') as mock_create:
            mock_session = Mock()
            mock_session.session_id = "test_session_123"
            mock_session.user_id = "user123"
            mock_session.device_id = "device456"
            mock_session.created_at = datetime.utcnow()
            mock_session.last_updated = datetime.utcnow()
            mock_session.state_data = {"key": "value"}
            mock_session.version = 1
            
            mock_create.return_value = mock_session
            
            # Make request
            response = client.post("/api/cross-device/sessions", json={
                "user_id": "user123",
                "device_id": "device456",
                "device_type": "desktop",
                "initial_state": {"key": "value"}
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["session_id"] == "test_session_123"
            assert data["user_id"] == "user123"
            assert data["device_id"] == "device456"
    
    def test_create_session_failure(self, mock_continuity_manager):
        """Test session creation failure."""
        with patch('scrollintel.api.routes.cross_device_routes.create_user_session') as mock_create:
            mock_create.side_effect = Exception("Database error")
            
            response = client.post("/api/cross-device/sessions", json={
                "user_id": "user123",
                "device_id": "device456",
                "device_type": "desktop"
            })
            
            assert response.status_code == 500
    
    def test_get_session_success(self, mock_continuity_manager):
        """Test successful session retrieval."""
        with patch('scrollintel.api.routes.cross_device_routes.restore_user_session') as mock_restore:
            mock_session = Mock()
            mock_session.session_id = "test_session_123"
            mock_session.user_id = "user123"
            mock_session.device_id = "device456"
            mock_session.created_at = datetime.utcnow()
            mock_session.last_updated = datetime.utcnow()
            mock_session.state_data = {"key": "value"}
            mock_session.version = 1
            
            mock_restore.return_value = mock_session
            
            response = client.get("/api/cross-device/sessions/test_session_123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["session_id"] == "test_session_123"
    
    def test_get_session_not_found(self, mock_continuity_manager):
        """Test session not found."""
        with patch('scrollintel.api.routes.cross_device_routes.restore_user_session') as mock_restore:
            mock_restore.return_value = None
            
            response = client.get("/api/cross-device/sessions/nonexistent")
            
            assert response.status_code == 404
    
    def test_update_session_success(self, mock_continuity_manager):
        """Test successful session update."""
        with patch('scrollintel.api.routes.cross_device_routes.sync_user_state') as mock_sync:
            mock_sync.return_value = True
            
            response = client.put("/api/cross-device/sessions/test_session_123", json={
                "user_id": "user123",
                "device_id": "device456",
                "state_updates": {"key": "updated_value"}
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["session_id"] == "test_session_123"
    
    def test_update_session_failure(self, mock_continuity_manager):
        """Test session update failure."""
        with patch('scrollintel.api.routes.cross_device_routes.sync_user_state') as mock_sync:
            mock_sync.return_value = False
            
            response = client.put("/api/cross-device/sessions/test_session_123", json={
                "user_id": "user123",
                "device_id": "device456",
                "state_updates": {"key": "updated_value"}
            })
            
            assert response.status_code == 404
    
    def test_sync_devices_success(self, mock_continuity_manager):
        """Test successful device synchronization."""
        mock_continuity_manager.sync_across_devices = AsyncMock(return_value={
            "status": "success",
            "synced_sessions": 2,
            "conflicts": 0
        })
        
        response = client.post("/api/cross-device/sync", json={
            "user_id": "user123"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "user123"
        assert "sync_results" in data
    
    def test_get_sync_status_success(self, mock_continuity_manager):
        """Test getting sync status."""
        mock_continuity_manager.get_sync_status = AsyncMock(return_value={
            "user_id": "user123",
            "total_sessions": 3,
            "active_devices": 2,
            "pending_conflicts": 0,
            "offline_mode": False,
            "offline_queue_size": 0,
            "last_sync": None,
            "websocket_connections": 1
        })
        
        response = client.get("/api/cross-device/sync/status/user123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "user123"
        assert data["total_sessions"] == 3
        assert data["active_devices"] == 2
    
    def test_enable_offline_mode_success(self, mock_continuity_manager):
        """Test enabling offline mode."""
        with patch('scrollintel.api.routes.cross_device_routes.enable_offline_mode') as mock_enable:
            mock_enable.return_value = {
                "status": "offline_enabled",
                "capabilities": {"read_cached_data": True},
                "cached_sessions": 2
            }
            
            response = client.post("/api/cross-device/offline/enable", json={
                "user_id": "user123",
                "device_id": "device456"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["user_id"] == "user123"
            assert data["device_id"] == "device456"
            assert "offline_result" in data
    
    def test_sync_after_reconnect_success(self, mock_continuity_manager):
        """Test syncing after reconnect."""
        with patch('scrollintel.api.routes.cross_device_routes.sync_after_reconnect') as mock_sync:
            mock_sync.return_value = {
                "status": "reconnected",
                "offline_changes_synced": 5,
                "sync_results": {"status": "success"}
            }
            
            response = client.post("/api/cross-device/offline/sync", json={
                "user_id": "user123",
                "device_id": "device456"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "sync_result" in data
    
    def test_sync_tabs_success(self, mock_continuity_manager):
        """Test tab synchronization."""
        mock_continuity_manager.handle_multi_tab_sync = AsyncMock(return_value={
            "status": "synced",
            "session_id": "tab_session_123",
            "version": 2
        })
        
        response = client.post("/api/cross-device/tabs/sync", params={
            "user_id": "user123",
            "tab_id": "tab456"
        }, json={
            "current_page": "/dashboard",
            "zoom": 1.2
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "user123"
        assert data["tab_id"] == "tab456"
    
    def test_get_user_devices_success(self, mock_continuity_manager):
        """Test getting user devices."""
        # Mock sessions
        mock_session1 = Mock()
        mock_session1.session_id = "session1"
        mock_session1.device_id = "device1"
        mock_session1.created_at = datetime.utcnow()
        mock_session1.last_updated = datetime.utcnow()
        mock_session1.version = 1
        
        mock_session2 = Mock()
        mock_session2.session_id = "session2"
        mock_session2.device_id = "device2"
        mock_session2.created_at = datetime.utcnow()
        mock_session2.last_updated = datetime.utcnow()
        mock_session2.version = 2
        
        mock_continuity_manager.state_manager.get_user_sessions.return_value = [
            mock_session1, mock_session2
        ]
        
        # Mock device info
        mock_device1 = Mock()
        mock_device1.device_type.value = "desktop"
        mock_device1.last_seen = datetime.utcnow()
        mock_device1.is_online = True
        
        mock_device2 = Mock()
        mock_device2.device_type.value = "mobile"
        mock_device2.last_seen = datetime.utcnow()
        mock_device2.is_online = False
        
        mock_continuity_manager.devices = {
            "device1": mock_device1,
            "device2": mock_device2
        }
        
        response = client.get("/api/cross-device/devices/user123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "user123"
        assert data["total_devices"] == 2
        assert len(data["devices"]) == 2
    
    def test_delete_session_success(self, mock_continuity_manager):
        """Test session deletion."""
        # Add session to active sessions
        mock_continuity_manager.active_sessions = {"test_session_123": Mock()}
        
        with patch('os.path.exists', return_value=True), \
             patch('os.remove') as mock_remove:
            
            response = client.delete("/api/cross-device/sessions/test_session_123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["session_id"] == "test_session_123"
            
            # Check that session was removed from active sessions
            assert "test_session_123" not in mock_continuity_manager.active_sessions
    
    def test_get_sync_conflicts_success(self, mock_continuity_manager):
        """Test getting sync conflicts."""
        # Mock conflicts
        mock_conflict = Mock()
        mock_conflict.conflict_id = "conflict123"
        mock_conflict.session_id = "session123"
        mock_conflict.path = "test.path"
        mock_conflict.resolved = False
        mock_conflict.resolution_strategy.value = "last_write_wins"
        mock_conflict.resolution_data = None
        
        # Mock local and remote changes
        mock_local_change = Mock()
        mock_local_change.user_id = "user123"
        mock_local_change.timestamp = datetime.utcnow()
        mock_local_change.device_id = "device1"
        mock_local_change.new_value = "local_value"
        
        mock_remote_change = Mock()
        mock_remote_change.timestamp = datetime.utcnow()
        mock_remote_change.device_id = "device2"
        mock_remote_change.new_value = "remote_value"
        
        mock_conflict.local_change = mock_local_change
        mock_conflict.remote_change = mock_remote_change
        
        mock_continuity_manager.sync_conflicts = {"conflict123": mock_conflict}
        
        response = client.get("/api/cross-device/conflicts/user123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "user123"
        assert data["total_conflicts"] == 1
        assert data["pending_conflicts"] == 1
        assert len(data["conflicts"]) == 1
    
    def test_resolve_conflict_success(self, mock_continuity_manager):
        """Test conflict resolution."""
        # Mock conflict
        mock_conflict = Mock()
        mock_conflict.conflict_id = "conflict123"
        mock_conflict.resolution_strategy = Mock()
        mock_conflict.resolved = False
        
        mock_continuity_manager.sync_conflicts = {"conflict123": mock_conflict}
        mock_continuity_manager.conflict_resolver.resolve_conflict = AsyncMock(
            return_value="resolved_value"
        )
        
        response = client.post("/api/cross-device/conflicts/conflict123/resolve", params={
            "resolution_strategy": "automatic_merge"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["conflict_id"] == "conflict123"
        assert data["resolution_data"] == "resolved_value"
    
    def test_resolve_conflict_not_found(self, mock_continuity_manager):
        """Test resolving non-existent conflict."""
        mock_continuity_manager.sync_conflicts = {}
        
        response = client.post("/api/cross-device/conflicts/nonexistent/resolve")
        
        assert response.status_code == 404
    
    def test_health_check_success(self, mock_continuity_manager):
        """Test health check endpoint."""
        response = client.get("/api/cross-device/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "cross_device_continuity"
        assert data["status"] == "healthy"
        assert "components" in data
        assert "metrics" in data
        assert "timestamp" in data
    
    def test_health_check_degraded(self, mock_continuity_manager):
        """Test health check with degraded status."""
        # Create many conflicts to trigger degraded status
        mock_conflicts = {f"conflict_{i}": Mock() for i in range(150)}
        mock_continuity_manager.sync_conflicts = mock_conflicts
        
        response = client.get("/api/cross-device/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "cross_device_continuity"
        assert data["status"] == "degraded"
        assert data["components"]["conflict_resolver"] == "degraded"


class TestWebSocketEndpoint:
    """Test WebSocket endpoint functionality."""
    
    def test_websocket_connection(self):
        """Test WebSocket connection establishment."""
        # Note: Testing WebSocket connections requires more complex setup
        # This is a placeholder for WebSocket testing
        # In a real implementation, you would use websocket test clients
        pass
    
    def test_websocket_state_update(self):
        """Test WebSocket state update message."""
        # Placeholder for WebSocket state update testing
        pass
    
    def test_websocket_sync_request(self):
        """Test WebSocket sync request message."""
        # Placeholder for WebSocket sync request testing
        pass
    
    def test_websocket_heartbeat(self):
        """Test WebSocket heartbeat message."""
        # Placeholder for WebSocket heartbeat testing
        pass


class TestErrorHandling:
    """Test error handling in API routes."""
    
    def test_invalid_json_request(self):
        """Test handling of invalid JSON requests."""
        response = client.post("/api/cross-device/sessions", data="invalid json")
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        response = client.post("/api/cross-device/sessions", json={
            "user_id": "user123"
            # Missing device_id
        })
        
        assert response.status_code == 422
    
    def test_invalid_device_type(self):
        """Test handling of invalid device type."""
        response = client.post("/api/cross-device/sessions", json={
            "user_id": "user123",
            "device_id": "device456",
            "device_type": "invalid_type"
        })
        
        assert response.status_code == 422
    
    def test_internal_server_error_handling(self, mock_continuity_manager):
        """Test handling of internal server errors."""
        with patch('scrollintel.api.routes.cross_device_routes.create_user_session') as mock_create:
            mock_create.side_effect = Exception("Unexpected error")
            
            response = client.post("/api/cross-device/sessions", json={
                "user_id": "user123",
                "device_id": "device456",
                "device_type": "desktop"
            })
            
            assert response.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__])