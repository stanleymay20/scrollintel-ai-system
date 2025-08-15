"""
Unit tests for WebSocket Manager functionality.
"""
import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from scrollintel.core.websocket_manager import (
    DashboardWebSocketManager, WebSocketMessage, websocket_manager
)
from scrollintel.models.dashboard_models import BusinessMetric


class TestWebSocketMessage:
    """Test cases for WebSocketMessage class."""
    
    def test_websocket_message_creation(self):
        """Test WebSocketMessage creation."""
        message = WebSocketMessage(
            type="test_message",
            data={"key": "value"}
        )
        
        assert message.type == "test_message"
        assert message.data == {"key": "value"}
        assert isinstance(message.timestamp, datetime)
    
    def test_websocket_message_to_json(self):
        """Test WebSocketMessage JSON serialization."""
        timestamp = datetime.utcnow()
        message = WebSocketMessage(
            type="test_message",
            data={"key": "value"},
            timestamp=timestamp
        )
        
        json_str = message.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["type"] == "test_message"
        assert parsed["data"] == {"key": "value"}
        assert parsed["timestamp"] == timestamp.isoformat()


class TestDashboardWebSocketManager:
    """Test cases for DashboardWebSocketManager class."""
    
    @pytest.fixture
    def ws_manager(self):
        """Create a WebSocket manager instance for testing."""
        return DashboardWebSocketManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection."""
        websocket = AsyncMock()
        websocket.send = AsyncMock()
        return websocket
    
    @pytest.fixture
    def mock_dashboard_manager(self):
        """Mock dashboard manager."""
        with patch('scrollintel.core.websocket_manager.DashboardManager') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            yield mock_instance
    
    @pytest.mark.asyncio
    async def test_register_connection(self, ws_manager, mock_websocket, mock_dashboard_manager):
        """Test registering a WebSocket connection."""
        connection_id = "conn_123"
        user_id = "user_456"
        dashboard_id = "dash_789"
        
        # Mock dashboard data
        mock_dashboard_data = Mock()
        mock_dashboard_data.dashboard.id = dashboard_id
        mock_dashboard_data.dashboard.name = "Test Dashboard"
        mock_dashboard_data.widgets_data = {}
        mock_dashboard_data.metrics = []
        mock_dashboard_data.last_updated = datetime.utcnow()
        mock_dashboard_manager.get_dashboard_data.return_value = mock_dashboard_data
        
        await ws_manager.register_connection(mock_websocket, connection_id, user_id, dashboard_id)
        
        # Verify connection was registered
        assert dashboard_id in ws_manager.connections
        assert mock_websocket in ws_manager.connections[dashboard_id]
        assert connection_id in ws_manager.dashboard_subscribers[dashboard_id]
        assert connection_id in ws_manager.connection_metadata
        
        # Verify initial data was sent
        mock_websocket.send.assert_called()
    
    @pytest.mark.asyncio
    async def test_unregister_connection(self, ws_manager, mock_websocket):
        """Test unregistering a WebSocket connection."""
        connection_id = "conn_123"
        dashboard_id = "dash_789"
        
        # Manually register connection first
        ws_manager.connections[dashboard_id] = {mock_websocket}
        ws_manager.dashboard_subscribers[dashboard_id] = {connection_id}
        ws_manager.connection_metadata[connection_id] = {
            "dashboard_id": dashboard_id,
            "websocket": mock_websocket
        }
        
        await ws_manager.unregister_connection(connection_id)
        
        # Verify connection was unregistered
        assert dashboard_id not in ws_manager.connections
        assert dashboard_id not in ws_manager.dashboard_subscribers
        assert connection_id not in ws_manager.connection_metadata
    
    @pytest.mark.asyncio
    async def test_broadcast_to_dashboard(self, ws_manager, mock_websocket):
        """Test broadcasting message to dashboard subscribers."""
        dashboard_id = "dash_789"
        message = WebSocketMessage(type="test", data={"test": "data"})
        
        # Register connection
        ws_manager.connections[dashboard_id] = {mock_websocket}
        
        await ws_manager.broadcast_to_dashboard(dashboard_id, message)
        
        # Verify message was sent
        mock_websocket.send.assert_called_once_with(message.to_json())
    
    @pytest.mark.asyncio
    async def test_broadcast_to_dashboard_handles_disconnected_clients(self, ws_manager):
        """Test broadcasting handles disconnected WebSocket clients."""
        dashboard_id = "dash_789"
        message = WebSocketMessage(type="test", data={"test": "data"})
        
        # Create mock websockets - one working, one disconnected
        working_ws = AsyncMock()
        disconnected_ws = AsyncMock()
        disconnected_ws.send.side_effect = Exception("Connection closed")
        
        ws_manager.connections[dashboard_id] = {working_ws, disconnected_ws}
        ws_manager.connection_metadata["conn1"] = {"websocket": working_ws, "dashboard_id": dashboard_id}
        ws_manager.connection_metadata["conn2"] = {"websocket": disconnected_ws, "dashboard_id": dashboard_id}
        
        await ws_manager.broadcast_to_dashboard(dashboard_id, message)
        
        # Verify working connection received message
        working_ws.send.assert_called_once()
        # Verify disconnected connection was removed
        assert disconnected_ws not in ws_manager.connections[dashboard_id]
    
    @pytest.mark.asyncio
    async def test_send_dashboard_update(self, ws_manager, mock_websocket, mock_dashboard_manager):
        """Test sending dashboard update."""
        dashboard_id = "dash_789"
        connection_id = "conn_123"
        
        # Mock dashboard data
        mock_dashboard = Mock()
        mock_dashboard.id = dashboard_id
        mock_dashboard.name = "Test Dashboard"
        mock_dashboard.type = "executive"
        mock_dashboard.config = {}
        mock_dashboard.updated_at = datetime.utcnow()
        
        mock_metric = Mock()
        mock_metric.id = "metric_1"
        mock_metric.name = "test_metric"
        mock_metric.category = "test"
        mock_metric.value = 100
        mock_metric.unit = "count"
        mock_metric.timestamp = datetime.utcnow()
        mock_metric.source = "test_source"
        mock_metric.context = {}
        
        mock_dashboard_data = Mock()
        mock_dashboard_data.dashboard = mock_dashboard
        mock_dashboard_data.widgets_data = {}
        mock_dashboard_data.metrics = [mock_metric]
        mock_dashboard_data.last_updated = datetime.utcnow()
        
        mock_dashboard_manager.get_dashboard_data.return_value = mock_dashboard_data
        
        # Register connection
        ws_manager.connection_metadata[connection_id] = {
            "websocket": mock_websocket,
            "dashboard_id": dashboard_id
        }
        
        await ws_manager.send_dashboard_update(dashboard_id, connection_id)
        
        # Verify update was sent
        mock_websocket.send.assert_called_once()
        call_args = mock_websocket.send.call_args[0][0]
        message_data = json.loads(call_args)
        
        assert message_data["type"] == "dashboard_update"
        assert message_data["data"]["dashboard"]["id"] == dashboard_id
        assert len(message_data["data"]["metrics"]) == 1
    
    @pytest.mark.asyncio
    async def test_send_metric_update(self, ws_manager, mock_websocket):
        """Test sending metric updates."""
        dashboard_id = "dash_789"
        
        # Create mock metrics
        mock_metric = Mock(spec=BusinessMetric)
        mock_metric.id = "metric_1"
        mock_metric.name = "test_metric"
        mock_metric.category = "test"
        mock_metric.value = 100
        mock_metric.unit = "count"
        mock_metric.timestamp = datetime.utcnow()
        mock_metric.source = "test_source"
        mock_metric.context = {}
        
        # Register connection
        ws_manager.connections[dashboard_id] = {mock_websocket}
        
        await ws_manager.send_metric_update(dashboard_id, [mock_metric])
        
        # Verify metric update was sent
        mock_websocket.send.assert_called_once()
        call_args = mock_websocket.send.call_args[0][0]
        message_data = json.loads(call_args)
        
        assert message_data["type"] == "metrics_update"
        assert len(message_data["data"]) == 1
        assert message_data["data"][0]["name"] == "test_metric"
    
    @pytest.mark.asyncio
    async def test_send_alert(self, ws_manager, mock_websocket):
        """Test sending alerts."""
        dashboard_id = "dash_789"
        alert_data = {
            "id": "alert_1",
            "type": "warning",
            "message": "Test alert"
        }
        
        # Register connection
        ws_manager.connections[dashboard_id] = {mock_websocket}
        
        await ws_manager.send_alert(dashboard_id, alert_data)
        
        # Verify alert was sent
        mock_websocket.send.assert_called_once()
        call_args = mock_websocket.send.call_args[0][0]
        message_data = json.loads(call_args)
        
        assert message_data["type"] == "alert"
        assert message_data["data"] == alert_data
    
    @pytest.mark.asyncio
    async def test_handle_client_message_subscribe(self, ws_manager, mock_websocket, mock_dashboard_manager):
        """Test handling client subscribe message."""
        connection_id = "conn_123"
        dashboard_id = "dash_789"
        
        # Mock dashboard data
        mock_dashboard_data = Mock()
        mock_dashboard_data.dashboard.id = dashboard_id
        mock_dashboard_data.widgets_data = {}
        mock_dashboard_data.metrics = []
        mock_dashboard_data.last_updated = datetime.utcnow()
        mock_dashboard_manager.get_dashboard_data.return_value = mock_dashboard_data
        
        # Register connection
        ws_manager.connection_metadata[connection_id] = {
            "websocket": mock_websocket,
            "dashboard_id": "old_dashboard"
        }
        
        message_data = {
            "type": "subscribe_dashboard",
            "data": {"dashboard_id": dashboard_id}
        }
        
        await ws_manager.handle_client_message(connection_id, message_data)
        
        # Verify subscription was updated
        assert ws_manager.connection_metadata[connection_id]["dashboard_id"] == dashboard_id
        mock_websocket.send.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_client_message_ping(self, ws_manager, mock_websocket):
        """Test handling client ping message."""
        connection_id = "conn_123"
        
        # Register connection
        ws_manager.connection_metadata[connection_id] = {
            "websocket": mock_websocket,
            "dashboard_id": "dash_789"
        }
        
        message_data = {"type": "ping"}
        
        await ws_manager.handle_client_message(connection_id, message_data)
        
        # Verify pong was sent
        mock_websocket.send.assert_called_once()
        call_args = mock_websocket.send.call_args[0][0]
        message_data = json.loads(call_args)
        
        assert message_data["type"] == "pong"
        assert "timestamp" in message_data["data"]
    
    @pytest.mark.asyncio
    async def test_start_stop_periodic_updates(self, ws_manager):
        """Test starting and stopping periodic updates."""
        # Start periodic updates
        await ws_manager.start_periodic_updates(interval=1)
        
        assert ws_manager._running is True
        assert ws_manager._update_task is not None
        
        # Stop periodic updates
        await ws_manager.stop_periodic_updates()
        
        assert ws_manager._running is False
    
    @pytest.mark.asyncio
    async def test_get_connection_stats(self, ws_manager, mock_websocket):
        """Test getting connection statistics."""
        # Register some connections
        ws_manager.connections["dash1"] = {mock_websocket}
        ws_manager.dashboard_subscribers["dash1"] = {"conn1"}
        ws_manager.connection_metadata["conn1"] = {
            "websocket": mock_websocket,
            "dashboard_id": "dash1"
        }
        
        stats = await ws_manager.get_connection_stats()
        
        assert stats["total_connections"] == 1
        assert stats["active_dashboards"] == 1
        assert stats["connections_by_dashboard"]["dash1"] == 1
        assert "uptime" in stats


class TestWebSocketHandler:
    """Test cases for WebSocket handler function."""
    
    @pytest.mark.asyncio
    async def test_websocket_handler_connect(self):
        """Test WebSocket handler connection flow."""
        mock_websocket = AsyncMock()
        
        # Mock incoming messages
        connect_message = json.dumps({
            "type": "connect",
            "connection_id": "conn_123",
            "user_id": "user_456",
            "dashboard_id": "dash_789"
        })
        
        mock_websocket.__aiter__.return_value = [connect_message]
        
        with patch('scrollintel.core.websocket_manager.websocket_manager') as mock_manager:
            mock_manager.register_connection = AsyncMock()
            mock_manager.unregister_connection = AsyncMock()
            
            from scrollintel.core.websocket_manager import websocket_handler
            
            await websocket_handler(mock_websocket, "/ws")
            
            # Verify connection was registered
            mock_manager.register_connection.assert_called_once_with(
                mock_websocket, "conn_123", "user_456", "dash_789"
            )
            
            # Verify confirmation was sent
            mock_websocket.send.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])