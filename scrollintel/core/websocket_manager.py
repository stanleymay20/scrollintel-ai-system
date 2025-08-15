"""
WebSocket manager for real-time dashboard updates.
"""
import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
from dataclasses import dataclass, asdict

from .dashboard_manager import DashboardManager
from ..models.dashboard_models import BusinessMetric


logger = logging.getLogger(__name__)


@dataclass
class WebSocketMessage:
    type: str
    data: Any
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        })


class DashboardWebSocketManager:
    """Manages WebSocket connections for real-time dashboard updates."""
    
    def __init__(self):
        self.connections: Dict[str, Set[WebSocketServerProtocol]] = {}
        self.dashboard_subscribers: Dict[str, Set[str]] = {}  # dashboard_id -> set of connection_ids
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.dashboard_manager = DashboardManager()
        self._running = False
        self._update_task = None
    
    async def register_connection(self, websocket: WebSocketServerProtocol, 
                                connection_id: str, user_id: str, dashboard_id: str):
        """Register a new WebSocket connection."""
        try:
            # Store connection
            if dashboard_id not in self.connections:
                self.connections[dashboard_id] = set()
            self.connections[dashboard_id].add(websocket)
            
            # Store subscriber mapping
            if dashboard_id not in self.dashboard_subscribers:
                self.dashboard_subscribers[dashboard_id] = set()
            self.dashboard_subscribers[dashboard_id].add(connection_id)
            
            # Store metadata
            self.connection_metadata[connection_id] = {
                "user_id": user_id,
                "dashboard_id": dashboard_id,
                "connected_at": datetime.utcnow(),
                "websocket": websocket
            }
            
            logger.info(f"WebSocket connection registered: {connection_id} for dashboard {dashboard_id}")
            
            # Send initial dashboard data
            await self.send_dashboard_update(dashboard_id, connection_id)
            
        except Exception as e:
            logger.error(f"Error registering WebSocket connection: {e}")
            await self.unregister_connection(connection_id)
    
    async def unregister_connection(self, connection_id: str):
        """Unregister a WebSocket connection."""
        try:
            if connection_id in self.connection_metadata:
                metadata = self.connection_metadata[connection_id]
                dashboard_id = metadata["dashboard_id"]
                websocket = metadata["websocket"]
                
                # Remove from connections
                if dashboard_id in self.connections:
                    self.connections[dashboard_id].discard(websocket)
                    if not self.connections[dashboard_id]:
                        del self.connections[dashboard_id]
                
                # Remove from subscribers
                if dashboard_id in self.dashboard_subscribers:
                    self.dashboard_subscribers[dashboard_id].discard(connection_id)
                    if not self.dashboard_subscribers[dashboard_id]:
                        del self.dashboard_subscribers[dashboard_id]
                
                # Remove metadata
                del self.connection_metadata[connection_id]
                
                logger.info(f"WebSocket connection unregistered: {connection_id}")
                
        except Exception as e:
            logger.error(f"Error unregistering WebSocket connection: {e}")
    
    async def broadcast_to_dashboard(self, dashboard_id: str, message: WebSocketMessage):
        """Broadcast a message to all connections subscribed to a dashboard."""
        if dashboard_id not in self.connections:
            return
        
        disconnected = set()
        message_json = message.to_json()
        
        for websocket in self.connections[dashboard_id].copy():
            try:
                await websocket.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected:
            self.connections[dashboard_id].discard(websocket)
            # Find and remove connection metadata
            for conn_id, metadata in list(self.connection_metadata.items()):
                if metadata["websocket"] == websocket:
                    await self.unregister_connection(conn_id)
                    break
    
    async def send_dashboard_update(self, dashboard_id: str, connection_id: str = None):
        """Send dashboard data update to specific connection or all subscribers."""
        try:
            # Get dashboard data
            dashboard_data = self.dashboard_manager.get_dashboard_data(dashboard_id)
            if not dashboard_data:
                return
            
            # Prepare update message
            update_data = {
                "dashboard": {
                    "id": dashboard_data.dashboard.id,
                    "name": dashboard_data.dashboard.name,
                    "type": dashboard_data.dashboard.type,
                    "config": dashboard_data.dashboard.config,
                    "updated_at": dashboard_data.dashboard.updated_at.isoformat()
                },
                "widgets": dashboard_data.widgets_data,
                "metrics": [
                    {
                        "id": metric.id,
                        "name": metric.name,
                        "category": metric.category,
                        "value": metric.value,
                        "unit": metric.unit,
                        "timestamp": metric.timestamp.isoformat(),
                        "source": metric.source,
                        "context": metric.context
                    }
                    for metric in dashboard_data.metrics[-50:]  # Last 50 metrics
                ],
                "last_updated": dashboard_data.last_updated.isoformat()
            }
            
            message = WebSocketMessage(type="dashboard_update", data=update_data)
            
            if connection_id:
                # Send to specific connection
                if connection_id in self.connection_metadata:
                    websocket = self.connection_metadata[connection_id]["websocket"]
                    try:
                        await websocket.send(message.to_json())
                    except Exception as e:
                        logger.error(f"Error sending update to connection {connection_id}: {e}")
                        await self.unregister_connection(connection_id)
            else:
                # Broadcast to all subscribers
                await self.broadcast_to_dashboard(dashboard_id, message)
                
        except Exception as e:
            logger.error(f"Error sending dashboard update: {e}")
    
    async def send_metric_update(self, dashboard_id: str, metrics: List[BusinessMetric]):
        """Send real-time metric updates."""
        try:
            metrics_data = [
                {
                    "id": metric.id,
                    "name": metric.name,
                    "category": metric.category,
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat(),
                    "source": metric.source,
                    "context": metric.context
                }
                for metric in metrics
            ]
            
            message = WebSocketMessage(type="metrics_update", data=metrics_data)
            await self.broadcast_to_dashboard(dashboard_id, message)
            
        except Exception as e:
            logger.error(f"Error sending metric update: {e}")
    
    async def send_alert(self, dashboard_id: str, alert_data: Dict[str, Any]):
        """Send real-time alerts to dashboard subscribers."""
        try:
            message = WebSocketMessage(type="alert", data=alert_data)
            await self.broadcast_to_dashboard(dashboard_id, message)
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    async def handle_client_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle incoming messages from WebSocket clients."""
        try:
            message_type = message_data.get("type")
            data = message_data.get("data", {})
            
            if message_type == "subscribe_dashboard":
                dashboard_id = data.get("dashboard_id")
                if dashboard_id and connection_id in self.connection_metadata:
                    # Update subscription
                    self.connection_metadata[connection_id]["dashboard_id"] = dashboard_id
                    await self.send_dashboard_update(dashboard_id, connection_id)
            
            elif message_type == "request_update":
                if connection_id in self.connection_metadata:
                    dashboard_id = self.connection_metadata[connection_id]["dashboard_id"]
                    await self.send_dashboard_update(dashboard_id, connection_id)
            
            elif message_type == "ping":
                # Respond with pong
                pong_message = WebSocketMessage(type="pong", data={"timestamp": datetime.utcnow().isoformat()})
                websocket = self.connection_metadata[connection_id]["websocket"]
                await websocket.send(pong_message.to_json())
            
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def start_periodic_updates(self, interval: int = 300):
        """Start periodic dashboard updates."""
        self._running = True
        
        async def update_loop():
            while self._running:
                try:
                    # Update all active dashboards
                    for dashboard_id in list(self.dashboard_subscribers.keys()):
                        if self.dashboard_subscribers[dashboard_id]:  # Has subscribers
                            await self.send_dashboard_update(dashboard_id)
                    
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error in periodic update loop: {e}")
                    await asyncio.sleep(interval)
        
        self._update_task = asyncio.create_task(update_loop())
        logger.info(f"Started periodic dashboard updates with {interval}s interval")
    
    async def stop_periodic_updates(self):
        """Stop periodic dashboard updates."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped periodic dashboard updates")
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            "total_connections": len(self.connection_metadata),
            "active_dashboards": len(self.dashboard_subscribers),
            "connections_by_dashboard": {
                dashboard_id: len(subscribers)
                for dashboard_id, subscribers in self.dashboard_subscribers.items()
            },
            "uptime": datetime.utcnow().isoformat()
        }


# Global WebSocket manager instance
websocket_manager = DashboardWebSocketManager()


async def websocket_handler(websocket: WebSocketServerProtocol, path: str):
    """WebSocket connection handler."""
    connection_id = None
    try:
        # Parse connection parameters from path or initial message
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data.get("type") == "connect":
                    connection_id = data.get("connection_id")
                    user_id = data.get("user_id")
                    dashboard_id = data.get("dashboard_id")
                    
                    if connection_id and user_id and dashboard_id:
                        await websocket_manager.register_connection(
                            websocket, connection_id, user_id, dashboard_id
                        )
                        
                        # Send connection confirmation
                        confirm_message = WebSocketMessage(
                            type="connected",
                            data={"connection_id": connection_id, "dashboard_id": dashboard_id}
                        )
                        await websocket.send(confirm_message.to_json())
                else:
                    # Handle other message types
                    if connection_id:
                        await websocket_manager.handle_client_message(connection_id, data)
                        
            except json.JSONDecodeError:
                logger.error("Invalid JSON received from WebSocket client")
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"WebSocket connection closed: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    finally:
        if connection_id:
            await websocket_manager.unregister_connection(connection_id)