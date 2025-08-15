"""
WebSocket Support for Data Product Registry

Provides real-time updates for data product changes, quality alerts,
and verification status updates.
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Set, Optional, Any, List
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
import logging
from collections import defaultdict
import uuid

from scrollintel.core.data_product_registry import DataProductRegistry
from scrollintel.models.data_product_models import DataProduct, VerificationStatus
from scrollintel.models.database import get_db

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for data product updates"""
    
    def __init__(self):
        # Active connections by connection ID
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Subscriptions by type
        self.product_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # product_id -> connection_ids
        self.quality_subscriptions: Set[str] = set()  # connection_ids subscribed to quality alerts
        self.verification_subscriptions: Set[str] = set()  # connection_ids subscribed to verification updates
        self.global_subscriptions: Set[str] = set()  # connection_ids subscribed to all updates
        
        # Connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str = None) -> str:
        """Accept WebSocket connection and return connection ID"""
        await websocket.accept()
        
        if not connection_id:
            connection_id = str(uuid.uuid4())
        
        self.active_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "subscriptions": []
        }
        
        logger.info(f"WebSocket connection established: {connection_id}")
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Remove connection and clean up subscriptions"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        # Clean up subscriptions
        for product_id, conn_ids in self.product_subscriptions.items():
            conn_ids.discard(connection_id)
        
        self.quality_subscriptions.discard(connection_id)
        self.verification_subscriptions.discard(connection_id)
        self.global_subscriptions.discard(connection_id)
        
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
        
        logger.info(f"WebSocket connection closed: {connection_id}")
    
    async def send_personal_message(self, message: dict, connection_id: str):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(json.dumps(message))
                
                # Update last activity
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["last_activity"] = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def broadcast_to_subscribers(self, message: dict, subscriber_set: Set[str]):
        """Broadcast message to a set of subscribers"""
        if not subscriber_set:
            return
        
        disconnected = []
        for connection_id in subscriber_set:
            try:
                await self.send_personal_message(message, connection_id)
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    def subscribe_to_product(self, connection_id: str, product_id: str):
        """Subscribe connection to specific product updates"""
        self.product_subscriptions[product_id].add(connection_id)
        
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscriptions"].append(f"product:{product_id}")
        
        logger.info(f"Connection {connection_id} subscribed to product {product_id}")
    
    def subscribe_to_quality_alerts(self, connection_id: str):
        """Subscribe connection to quality alerts"""
        self.quality_subscriptions.add(connection_id)
        
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscriptions"].append("quality_alerts")
        
        logger.info(f"Connection {connection_id} subscribed to quality alerts")
    
    def subscribe_to_verification_updates(self, connection_id: str):
        """Subscribe connection to verification status updates"""
        self.verification_subscriptions.add(connection_id)
        
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscriptions"].append("verification_updates")
        
        logger.info(f"Connection {connection_id} subscribed to verification updates")
    
    def subscribe_to_all_updates(self, connection_id: str):
        """Subscribe connection to all data product updates"""
        self.global_subscriptions.add(connection_id)
        
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscriptions"].append("all_updates")
        
        logger.info(f"Connection {connection_id} subscribed to all updates")
    
    def unsubscribe_from_product(self, connection_id: str, product_id: str):
        """Unsubscribe connection from specific product updates"""
        self.product_subscriptions[product_id].discard(connection_id)
        
        if connection_id in self.connection_metadata:
            subscriptions = self.connection_metadata[connection_id]["subscriptions"]
            if f"product:{product_id}" in subscriptions:
                subscriptions.remove(f"product:{product_id}")
        
        logger.info(f"Connection {connection_id} unsubscribed from product {product_id}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "product_subscriptions": {
                product_id: len(conn_ids) 
                for product_id, conn_ids in self.product_subscriptions.items()
            },
            "quality_subscribers": len(self.quality_subscriptions),
            "verification_subscribers": len(self.verification_subscriptions),
            "global_subscribers": len(self.global_subscriptions),
            "connections": [
                {
                    "id": conn_id,
                    "connected_at": metadata["connected_at"].isoformat(),
                    "last_activity": metadata["last_activity"].isoformat(),
                    "subscriptions": metadata["subscriptions"]
                }
                for conn_id, metadata in self.connection_metadata.items()
            ]
        }


# Global connection manager instance
manager = ConnectionManager()


class DataProductWebSocketHandler:
    """Handles WebSocket events for data product updates"""
    
    def __init__(self, db: Session):
        self.db = db
        self.registry = DataProductRegistry(db)
    
    async def handle_message(self, websocket: WebSocket, connection_id: str, message: dict):
        """Handle incoming WebSocket message"""
        try:
            message_type = message.get("type")
            data = message.get("data", {})
            
            if message_type == "subscribe_product":
                product_id = data.get("product_id")
                if product_id:
                    manager.subscribe_to_product(connection_id, product_id)
                    await self.send_response(websocket, "subscription_confirmed", {
                        "type": "product",
                        "product_id": product_id
                    })
            
            elif message_type == "unsubscribe_product":
                product_id = data.get("product_id")
                if product_id:
                    manager.unsubscribe_from_product(connection_id, product_id)
                    await self.send_response(websocket, "unsubscription_confirmed", {
                        "type": "product",
                        "product_id": product_id
                    })
            
            elif message_type == "subscribe_quality_alerts":
                manager.subscribe_to_quality_alerts(connection_id)
                await self.send_response(websocket, "subscription_confirmed", {
                    "type": "quality_alerts"
                })
            
            elif message_type == "subscribe_verification_updates":
                manager.subscribe_to_verification_updates(connection_id)
                await self.send_response(websocket, "subscription_confirmed", {
                    "type": "verification_updates"
                })
            
            elif message_type == "subscribe_all":
                manager.subscribe_to_all_updates(connection_id)
                await self.send_response(websocket, "subscription_confirmed", {
                    "type": "all_updates"
                })
            
            elif message_type == "get_product_status":
                product_id = data.get("product_id")
                if product_id:
                    product = self.registry.get_data_product(product_id)
                    if product:
                        await self.send_response(websocket, "product_status", {
                            "product_id": product_id,
                            "status": {
                                "verification_status": product.verification_status.value,
                                "quality_score": product.quality_score,
                                "bias_score": product.bias_score,
                                "updated_at": product.updated_at.isoformat()
                            }
                        })
                    else:
                        await self.send_response(websocket, "error", {
                            "message": f"Product {product_id} not found"
                        })
            
            elif message_type == "ping":
                await self.send_response(websocket, "pong", {"timestamp": datetime.now().isoformat()})
            
            else:
                await self.send_response(websocket, "error", {
                    "message": f"Unknown message type: {message_type}"
                })
        
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self.send_response(websocket, "error", {
                "message": "Internal server error"
            })
    
    async def send_response(self, websocket: WebSocket, message_type: str, data: dict):
        """Send response message"""
        response = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send_text(json.dumps(response))


class DataProductEventBroadcaster:
    """Broadcasts data product events to WebSocket subscribers"""
    
    @staticmethod
    async def broadcast_product_updated(product: DataProduct):
        """Broadcast product update event"""
        message = {
            "type": "product_updated",
            "data": {
                "product_id": str(product.id),
                "name": product.name,
                "version": product.version,
                "verification_status": product.verification_status.value,
                "quality_score": product.quality_score,
                "bias_score": product.bias_score,
                "updated_at": product.updated_at.isoformat(),
                "owner": product.owner
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to product-specific subscribers
        product_subscribers = manager.product_subscriptions.get(str(product.id), set())
        await manager.broadcast_to_subscribers(message, product_subscribers)
        
        # Send to global subscribers
        await manager.broadcast_to_subscribers(message, manager.global_subscriptions)
    
    @staticmethod
    async def broadcast_quality_alert(product_id: str, alert_data: dict):
        """Broadcast quality alert"""
        message = {
            "type": "quality_alert",
            "data": {
                "product_id": product_id,
                "alert": alert_data
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to quality alert subscribers
        await manager.broadcast_to_subscribers(message, manager.quality_subscriptions)
        
        # Send to global subscribers
        await manager.broadcast_to_subscribers(message, manager.global_subscriptions)
    
    @staticmethod
    async def broadcast_verification_status_changed(product: DataProduct, old_status: VerificationStatus):
        """Broadcast verification status change"""
        message = {
            "type": "verification_status_changed",
            "data": {
                "product_id": str(product.id),
                "name": product.name,
                "old_status": old_status.value,
                "new_status": product.verification_status.value,
                "updated_at": product.updated_at.isoformat(),
                "owner": product.owner
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to verification subscribers
        await manager.broadcast_to_subscribers(message, manager.verification_subscriptions)
        
        # Send to product-specific subscribers
        product_subscribers = manager.product_subscriptions.get(str(product.id), set())
        await manager.broadcast_to_subscribers(message, product_subscribers)
        
        # Send to global subscribers
        await manager.broadcast_to_subscribers(message, manager.global_subscriptions)
    
    @staticmethod
    async def broadcast_product_created(product: DataProduct):
        """Broadcast new product creation"""
        message = {
            "type": "product_created",
            "data": {
                "product_id": str(product.id),
                "name": product.name,
                "version": product.version,
                "owner": product.owner,
                "access_level": product.access_level.value,
                "created_at": product.created_at.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to global subscribers
        await manager.broadcast_to_subscribers(message, manager.global_subscriptions)
    
    @staticmethod
    async def broadcast_product_deleted(product_id: str, product_name: str):
        """Broadcast product deletion"""
        message = {
            "type": "product_deleted",
            "data": {
                "product_id": product_id,
                "name": product_name
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to product-specific subscribers
        product_subscribers = manager.product_subscriptions.get(product_id, set())
        await manager.broadcast_to_subscribers(message, product_subscribers)
        
        # Send to global subscribers
        await manager.broadcast_to_subscribers(message, manager.global_subscriptions)
        
        # Clean up subscriptions for deleted product
        if product_id in manager.product_subscriptions:
            del manager.product_subscriptions[product_id]


async def websocket_endpoint(websocket: WebSocket, db: Session = None):
    """Main WebSocket endpoint for data product updates"""
    connection_id = None
    
    try:
        connection_id = await manager.connect(websocket)
        handler = DataProductWebSocketHandler(db or next(get_db()))
        
        # Send welcome message
        await handler.send_response(websocket, "connected", {
            "connection_id": connection_id,
            "message": "Connected to Data Product Registry WebSocket"
        })
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle message
            await handler.handle_message(websocket, connection_id, message)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if connection_id:
            manager.disconnect(connection_id)


# Background task for periodic updates
async def periodic_quality_check():
    """Periodic task to check for quality issues and send alerts"""
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            # Get database session
            db = next(get_db())
            registry = DataProductRegistry(db)
            
            # Check for quality issues
            quality_alerts = registry.get_quality_alerts(threshold=0.8)
            
            for alert in quality_alerts:
                await DataProductEventBroadcaster.broadcast_quality_alert(
                    alert["product_id"],
                    alert
                )
        
        except Exception as e:
            logger.error(f"Error in periodic quality check: {e}")
            await asyncio.sleep(60)  # Wait before retrying


# Start background tasks
def start_background_tasks():
    """Start background tasks for WebSocket functionality"""
    asyncio.create_task(periodic_quality_check())