"""
Real-time WebSocket Manager for Dashboard Updates
Handles WebSocket connections and real-time data streaming to dashboards
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import websockets
import aioredis
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException
import jwt
from functools import wraps

logger = logging.getLogger(__name__)

class MessageType(Enum):
    DASHBOARD_UPDATE = "dashboard_update"
    METRIC_UPDATE = "metric_update"
    ALERT = "alert"
    INSIGHT = "insight"
    SYSTEM_STATUS = "system_status"
    USER_ACTION = "user_action"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

class ConnectionStatus(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class WebSocketMessage:
    type: MessageType
    data: Dict[str, Any]
    timestamp: datetime = None
    client_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'client_id': self.client_id
        }

@dataclass
class ClientConnection:
    websocket: WebSocketServerProtocol
    client_id: str
    user_id: Optional[str]
    dashboard_ids: Set[str]
    subscriptions: Set[str]
    status: ConnectionStatus
    connected_at: datetime
    last_heartbeat: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class RealtimeWebSocketManager:
    """Manages WebSocket connections for real-time dashboard updates"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 jwt_secret: str = "your-secret-key"):
        self.redis_url = redis_url
        self.jwt_secret = jwt_secret
        self.redis_client = None
        
        # Connection management
        self.connections: Dict[str, ClientConnection] = {}
        self.dashboard_subscribers: Dict[str, Set[str]] = {}  # dashboard_id -> client_ids
        self.metric_subscribers: Dict[str, Set[str]] = {}    # metric_name -> client_ids
        
        # Message handlers
        self.message_handlers: Dict[MessageType, Callable] = {
            MessageType.USER_ACTION: self._handle_user_action,
            MessageType.HEARTBEAT: self._handle_heartbeat,
        }
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 300  # seconds
        
        self.running = False
    
    async def initialize(self):
        """Initialize the WebSocket manager"""
        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            logger.info("WebSocket manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {e}")
            raise
    
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start the WebSocket server"""
        self.running = True
        logger.info(f"Starting WebSocket server on {host}:{port}")
        
        # Start background tasks
        background_tasks = [
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._connection_cleanup()),
            asyncio.create_task(self._redis_message_processor())
        ]
        
        # Start WebSocket server
        async with websockets.serve(self._handle_connection, host, port):
            await asyncio.gather(*background_tasks)
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        self.running = False
        
        # Close all connections
        for connection in self.connections.values():
            try:
                await connection.websocket.close()
            except:
                pass
        
        self.connections.clear()
        logger.info("WebSocket server stopped")
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        client_id = None
        try:
            # Generate client ID
            client_id = f"client_{int(datetime.now().timestamp())}_{id(websocket)}"
            
            # Create connection object
            connection = ClientConnection(
                websocket=websocket,
                client_id=client_id,
                user_id=None,
                dashboard_ids=set(),
                subscriptions=set(),
                status=ConnectionStatus.CONNECTING,
                connected_at=datetime.now(),
                last_heartbeat=datetime.now()
            )
            
            self.connections[client_id] = connection
            logger.info(f"New WebSocket connection: {client_id}")
            
            # Send welcome message
            await self._send_message(client_id, WebSocketMessage(
                type=MessageType.SYSTEM_STATUS,
                data={
                    'status': 'connected',
                    'client_id': client_id,
                    'server_time': datetime.now().isoformat()
                }
            ))
            
            connection.status = ConnectionStatus.CONNECTED
            
            # Handle messages
            async for message in websocket:
                await self._process_client_message(client_id, message)
                
        except ConnectionClosed:
            logger.info(f"WebSocket connection closed: {client_id}")
        except WebSocketException as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {client_id}: {e}")
        finally:
            # Clean up connection
            if client_id:
                await self._cleanup_connection(client_id)
    
    async def _process_client_message(self, client_id: str, message: str):
        """Process message from client"""
        try:
            data = json.loads(message)
            message_type = MessageType(data.get('type'))
            message_data = data.get('data', {})
            
            # Create message object
            ws_message = WebSocketMessage(
                type=message_type,
                data=message_data,
                client_id=client_id
            )
            
            # Handle authentication
            if message_type == MessageType.USER_ACTION and message_data.get('action') == 'authenticate':
                await self._authenticate_client(client_id, message_data)
                return
            
            # Handle subscription requests
            if message_type == MessageType.USER_ACTION and message_data.get('action') == 'subscribe':
                await self._handle_subscription(client_id, message_data)
                return
            
            # Route to appropriate handler
            handler = self.message_handlers.get(message_type)
            if handler:
                await handler(client_id, ws_message)
            else:
                logger.warning(f"No handler for message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error processing client message from {client_id}: {e}")
            await self._send_error(client_id, f"Error processing message: {str(e)}")
    
    async def _authenticate_client(self, client_id: str, data: Dict[str, Any]):
        """Authenticate client connection"""
        try:
            token = data.get('token')
            if not token:
                await self._send_error(client_id, "Authentication token required")
                return
            
            # Verify JWT token
            try:
                payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
                user_id = payload.get('user_id')
                
                if not user_id:
                    await self._send_error(client_id, "Invalid token payload")
                    return
                
                # Update connection
                connection = self.connections[client_id]
                connection.user_id = user_id
                connection.status = ConnectionStatus.AUTHENTICATED
                
                # Send authentication success
                await self._send_message(client_id, WebSocketMessage(
                    type=MessageType.SYSTEM_STATUS,
                    data={
                        'status': 'authenticated',
                        'user_id': user_id
                    }
                ))
                
                logger.info(f"Client authenticated: {client_id} (user: {user_id})")
                
            except jwt.InvalidTokenError:
                await self._send_error(client_id, "Invalid authentication token")
                
        except Exception as e:
            logger.error(f"Error authenticating client {client_id}: {e}")
            await self._send_error(client_id, "Authentication failed")
    
    async def _handle_subscription(self, client_id: str, data: Dict[str, Any]):
        """Handle subscription requests"""
        try:
            subscription_type = data.get('subscription_type')
            subscription_id = data.get('subscription_id')
            
            if not subscription_type or not subscription_id:
                await self._send_error(client_id, "Invalid subscription request")
                return
            
            connection = self.connections[client_id]
            
            if subscription_type == 'dashboard':
                # Subscribe to dashboard updates
                connection.dashboard_ids.add(subscription_id)
                
                if subscription_id not in self.dashboard_subscribers:
                    self.dashboard_subscribers[subscription_id] = set()
                self.dashboard_subscribers[subscription_id].add(client_id)
                
            elif subscription_type == 'metric':
                # Subscribe to metric updates
                connection.subscriptions.add(f"metric:{subscription_id}")
                
                if subscription_id not in self.metric_subscribers:
                    self.metric_subscribers[subscription_id] = set()
                self.metric_subscribers[subscription_id].add(client_id)
            
            # Send subscription confirmation
            await self._send_message(client_id, WebSocketMessage(
                type=MessageType.SYSTEM_STATUS,
                data={
                    'status': 'subscribed',
                    'subscription_type': subscription_type,
                    'subscription_id': subscription_id
                }
            ))
            
            logger.info(f"Client {client_id} subscribed to {subscription_type}:{subscription_id}")
            
        except Exception as e:
            logger.error(f"Error handling subscription for {client_id}: {e}")
            await self._send_error(client_id, "Subscription failed")
    
    async def _handle_user_action(self, client_id: str, message: WebSocketMessage):
        """Handle user action messages"""
        try:
            action = message.data.get('action')
            
            if action == 'ping':
                # Respond to ping
                await self._send_message(client_id, WebSocketMessage(
                    type=MessageType.SYSTEM_STATUS,
                    data={'status': 'pong'}
                ))
            
            elif action == 'get_dashboard_data':
                # Send current dashboard data
                dashboard_id = message.data.get('dashboard_id')
                if dashboard_id:
                    await self._send_dashboard_data(client_id, dashboard_id)
            
            # Update last activity
            if client_id in self.connections:
                self.connections[client_id].last_heartbeat = datetime.now()
                
        except Exception as e:
            logger.error(f"Error handling user action from {client_id}: {e}")
    
    async def _handle_heartbeat(self, client_id: str, message: WebSocketMessage):
        """Handle heartbeat messages"""
        if client_id in self.connections:
            self.connections[client_id].last_heartbeat = datetime.now()
            
            # Send heartbeat response
            await self._send_message(client_id, WebSocketMessage(
                type=MessageType.HEARTBEAT,
                data={'timestamp': datetime.now().isoformat()}
            ))
    
    async def _send_message(self, client_id: str, message: WebSocketMessage):
        """Send message to specific client"""
        try:
            if client_id not in self.connections:
                return
            
            connection = self.connections[client_id]
            message_dict = message.to_dict()
            
            await connection.websocket.send(json.dumps(message_dict))
            
        except ConnectionClosed:
            logger.info(f"Connection closed while sending message to {client_id}")
            await self._cleanup_connection(client_id)
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {e}")
    
    async def _send_error(self, client_id: str, error_message: str):
        """Send error message to client"""
        await self._send_message(client_id, WebSocketMessage(
            type=MessageType.ERROR,
            data={'error': error_message}
        ))
    
    async def _broadcast_to_dashboard_subscribers(self, dashboard_id: str, message: WebSocketMessage):
        """Broadcast message to all dashboard subscribers"""
        if dashboard_id not in self.dashboard_subscribers:
            return
        
        subscribers = self.dashboard_subscribers[dashboard_id].copy()
        
        for client_id in subscribers:
            await self._send_message(client_id, message)
    
    async def _broadcast_to_metric_subscribers(self, metric_name: str, message: WebSocketMessage):
        """Broadcast message to all metric subscribers"""
        if metric_name not in self.metric_subscribers:
            return
        
        subscribers = self.metric_subscribers[metric_name].copy()
        
        for client_id in subscribers:
            await self._send_message(client_id, message)
    
    async def _send_dashboard_data(self, client_id: str, dashboard_id: str):
        """Send current dashboard data to client"""
        try:
            # Get dashboard data from Redis
            dashboard_data = await self.redis_client.hgetall(f"dashboard:{dashboard_id}")
            
            if dashboard_data:
                # Decode Redis data
                decoded_data = {k.decode(): v.decode() for k, v in dashboard_data.items()}
                
                # Parse JSON fields
                for key, value in decoded_data.items():
                    try:
                        decoded_data[key] = json.loads(value)
                    except:
                        pass  # Keep as string if not JSON
                
                await self._send_message(client_id, WebSocketMessage(
                    type=MessageType.DASHBOARD_UPDATE,
                    data={
                        'dashboard_id': dashboard_id,
                        'data': decoded_data
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error sending dashboard data to {client_id}: {e}")
    
    async def _heartbeat_monitor(self):
        """Monitor client heartbeats"""
        while self.running:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.connection_timeout)
                
                # Find timed out connections
                timed_out_clients = []
                for client_id, connection in self.connections.items():
                    if connection.last_heartbeat < timeout_threshold:
                        timed_out_clients.append(client_id)
                
                # Clean up timed out connections
                for client_id in timed_out_clients:
                    logger.info(f"Connection timed out: {client_id}")
                    await self._cleanup_connection(client_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(10)
    
    async def _connection_cleanup(self):
        """Clean up disconnected connections"""
        while self.running:
            try:
                # Check for closed connections
                closed_connections = []
                for client_id, connection in self.connections.items():
                    if connection.websocket.closed:
                        closed_connections.append(client_id)
                
                # Clean up closed connections
                for client_id in closed_connections:
                    await self._cleanup_connection(client_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")
                await asyncio.sleep(30)
    
    async def _redis_message_processor(self):
        """Process messages from Redis for broadcasting"""
        while self.running:
            try:
                # Listen for dashboard updates
                messages = await self.redis_client.xread(
                    {'dashboard_updates': '$'}, 
                    count=10, 
                    block=1000
                )
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        await self._process_redis_message(fields)
                        
            except Exception as e:
                logger.error(f"Error processing Redis messages: {e}")
                await asyncio.sleep(1)
    
    async def _process_redis_message(self, fields: Dict[bytes, bytes]):
        """Process message from Redis stream"""
        try:
            decoded_fields = {k.decode(): v.decode() for k, v in fields.items()}
            
            message_type = decoded_fields.get('type')
            dashboard_id = decoded_fields.get('dashboard_id')
            metric_name = decoded_fields.get('metric_name')
            data = json.loads(decoded_fields.get('data', '{}'))
            
            # Create WebSocket message
            ws_message = WebSocketMessage(
                type=MessageType(message_type),
                data=data
            )
            
            # Broadcast to appropriate subscribers
            if dashboard_id:
                await self._broadcast_to_dashboard_subscribers(dashboard_id, ws_message)
            
            if metric_name:
                await self._broadcast_to_metric_subscribers(metric_name, ws_message)
                
        except Exception as e:
            logger.error(f"Error processing Redis message: {e}")
    
    async def _cleanup_connection(self, client_id: str):
        """Clean up client connection"""
        try:
            if client_id not in self.connections:
                return
            
            connection = self.connections[client_id]
            
            # Remove from dashboard subscribers
            for dashboard_id in connection.dashboard_ids:
                if dashboard_id in self.dashboard_subscribers:
                    self.dashboard_subscribers[dashboard_id].discard(client_id)
                    if not self.dashboard_subscribers[dashboard_id]:
                        del self.dashboard_subscribers[dashboard_id]
            
            # Remove from metric subscribers
            for subscription in connection.subscriptions:
                if subscription.startswith('metric:'):
                    metric_name = subscription[7:]  # Remove 'metric:' prefix
                    if metric_name in self.metric_subscribers:
                        self.metric_subscribers[metric_name].discard(client_id)
                        if not self.metric_subscribers[metric_name]:
                            del self.metric_subscribers[metric_name]
            
            # Close WebSocket if still open
            if not connection.websocket.closed:
                await connection.websocket.close()
            
            # Remove from connections
            del self.connections[client_id]
            
            logger.info(f"Cleaned up connection: {client_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up connection {client_id}: {e}")
    
    # Public API methods
    
    async def broadcast_dashboard_update(self, dashboard_id: str, data: Dict[str, Any]):
        """Broadcast dashboard update to subscribers"""
        message = WebSocketMessage(
            type=MessageType.DASHBOARD_UPDATE,
            data={
                'dashboard_id': dashboard_id,
                'data': data
            }
        )
        
        await self._broadcast_to_dashboard_subscribers(dashboard_id, message)
        
        # Also publish to Redis for other instances
        await self.redis_client.xadd('dashboard_updates', {
            'type': MessageType.DASHBOARD_UPDATE.value,
            'dashboard_id': dashboard_id,
            'data': json.dumps(data)
        })
    
    async def broadcast_metric_update(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Broadcast metric update to subscribers"""
        message = WebSocketMessage(
            type=MessageType.METRIC_UPDATE,
            data={
                'metric_name': metric_name,
                'value': value,
                'metadata': metadata or {}
            }
        )
        
        await self._broadcast_to_metric_subscribers(metric_name, message)
        
        # Also publish to Redis
        await self.redis_client.xadd('dashboard_updates', {
            'type': MessageType.METRIC_UPDATE.value,
            'metric_name': metric_name,
            'data': json.dumps(message.data)
        })
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """Broadcast alert to all connected clients"""
        message = WebSocketMessage(
            type=MessageType.ALERT,
            data=alert_data
        )
        
        # Send to all authenticated clients
        for client_id, connection in self.connections.items():
            if connection.status == ConnectionStatus.AUTHENTICATED:
                await self._send_message(client_id, message)
    
    async def broadcast_insight(self, insight_data: Dict[str, Any]):
        """Broadcast insight to all connected clients"""
        message = WebSocketMessage(
            type=MessageType.INSIGHT,
            data=insight_data
        )
        
        # Send to all authenticated clients
        for client_id, connection in self.connections.items():
            if connection.status == ConnectionStatus.AUTHENTICATED:
                await self._send_message(client_id, message)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        total_connections = len(self.connections)
        authenticated_connections = sum(1 for conn in self.connections.values() 
                                      if conn.status == ConnectionStatus.AUTHENTICATED)
        
        return {
            'total_connections': total_connections,
            'authenticated_connections': authenticated_connections,
            'dashboard_subscriptions': len(self.dashboard_subscribers),
            'metric_subscriptions': len(self.metric_subscribers)
        }

# Example usage
async def main():
    """Example usage of WebSocket manager"""
    manager = RealtimeWebSocketManager()
    await manager.initialize()
    
    # Start server
    await manager.start_server(host="0.0.0.0", port=8765)

if __name__ == "__main__":
    asyncio.run(main())