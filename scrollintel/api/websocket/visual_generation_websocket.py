"""
WebSocket handler for real-time visual generation progress updates
"""

import json
import asyncio
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.routing import APIRouter
import logging

from ...security.auth import get_current_user_websocket
from ...engines.visual_generation import get_engine

logger = logging.getLogger(__name__)

# Store active WebSocket connections
active_connections: Dict[str, Set[WebSocket]] = {}

router = APIRouter()


class VisualGenerationWebSocketManager:
    """Manage WebSocket connections for visual generation updates"""
    
    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Connect a new WebSocket client"""
        await websocket.accept()
        
        if user_id not in self.connections:
            self.connections[user_id] = set()
        
        self.connections[user_id].add(websocket)
        logger.info(f"WebSocket connected for user {user_id}")
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Disconnect a WebSocket client"""
        if user_id in self.connections:
            self.connections[user_id].discard(websocket)
            
            # Remove user entry if no connections left
            if not self.connections[user_id]:
                del self.connections[user_id]
        
        logger.info(f"WebSocket disconnected for user {user_id}")
    
    async def send_progress_update(self, user_id: str, progress_data: dict):
        """Send progress update to all connections for a user"""
        if user_id not in self.connections:
            return
        
        # Create a copy of connections to avoid modification during iteration
        connections = self.connections[user_id].copy()
        
        for websocket in connections:
            try:
                await websocket.send_text(json.dumps(progress_data))
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                # Remove failed connection
                self.connections[user_id].discard(websocket)
    
    async def broadcast_system_status(self, status_data: dict):
        """Broadcast system status to all connected users"""
        for user_connections in self.connections.values():
            connections = user_connections.copy()
            
            for websocket in connections:
                try:
                    await websocket.send_text(json.dumps({
                        "type": "system_status",
                        "data": status_data
                    }))
                except Exception as e:
                    logger.error(f"Error broadcasting system status: {e}")


# Global WebSocket manager instance
ws_manager = VisualGenerationWebSocketManager()


@router.websocket("/ws/visual-generation")
async def visual_generation_websocket(websocket: WebSocket, token: str = None):
    """
    WebSocket endpoint for real-time visual generation updates
    """
    # Authenticate user
    if not token:
        await websocket.close(code=4001, reason="Authentication required")
        return
    
    user = get_current_user_websocket(token)
    if not user:
        await websocket.close(code=4001, reason="Invalid authentication")
        return
    
    user_id = user.get("id", "anonymous")
    
    try:
        # Connect WebSocket
        await ws_manager.connect(websocket, user_id)
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "user_id": user_id,
            "message": "Connected to ScrollIntel Visual Generation"
        }))
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    }))
                
                elif message.get("type") == "subscribe_to_job":
                    job_id = message.get("job_id")
                    if job_id:
                        # Subscribe to job updates (implementation depends on your job system)
                        await websocket.send_text(json.dumps({
                            "type": "subscribed",
                            "job_id": job_id
                        }))
                
                elif message.get("type") == "get_status":
                    # Send current system status
                    try:
                        engine = get_engine()
                        status = engine.get_system_status()
                        await websocket.send_text(json.dumps({
                            "type": "system_status",
                            "data": status
                        }))
                    except Exception as e:
                        logger.error(f"Error getting system status: {e}")
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Internal server error"
                }))
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        ws_manager.disconnect(websocket, user_id)


async def send_generation_progress(user_id: str, result_id: str, progress_data: dict):
    """
    Send generation progress update to user's WebSocket connections
    
    Args:
        user_id: User ID to send update to
        result_id: Generation result ID
        progress_data: Progress information
    """
    message = {
        "type": "generation_progress",
        "result_id": result_id,
        "user_id": user_id,
        **progress_data
    }
    
    await ws_manager.send_progress_update(user_id, message)


async def broadcast_system_update(status_data: dict):
    """
    Broadcast system status update to all connected users
    
    Args:
        status_data: System status information
    """
    await ws_manager.broadcast_system_status(status_data)


# Export the manager for use in other modules
__all__ = ["router", "send_generation_progress", "broadcast_system_update", "ws_manager"]