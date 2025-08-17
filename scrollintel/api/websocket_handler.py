"""
WebSocket handler for real-time chat functionality.
Handles streaming responses, typing indicators, and live updates.
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect, Depends
from fastapi.routing import APIRouter
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.conversation_manager import ConversationManager
from ..core.enhanced_agent_system import EnhancedAgentSystem
from ..security.auth import get_current_user_websocket
from ..models.conversation_models import MessageRole

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time chat."""
    
    def __init__(self):
        # Active connections: {user_id: {connection_id: websocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # Conversation rooms: {conversation_id: {user_id: set of connection_ids}}
        self.conversation_rooms: Dict[str, Dict[str, Set[str]]] = {}
        # Typing indicators: {conversation_id: {user_id: timestamp}}
        self.typing_indicators: Dict[str, Dict[str, datetime]] = {}
        # Connection metadata: {connection_id: {user_id, conversation_id, etc.}}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, connection_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = {}
        
        self.active_connections[user_id][connection_id] = websocket
        self.connection_metadata[connection_id] = {
            'user_id': user_id,
            'connected_at': datetime.utcnow(),
            'conversation_id': None
        }
        
        logger.info(f"WebSocket connected: user={user_id}, connection={connection_id}")
    
    def disconnect(self, user_id: str, connection_id: str):
        """Remove a WebSocket connection."""
        # Remove from active connections
        if user_id in self.active_connections:
            self.active_connections[user_id].pop(connection_id, None)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        # Remove from conversation rooms
        conversation_id = self.connection_metadata.get(connection_id, {}).get('conversation_id')
        if conversation_id and conversation_id in self.conversation_rooms:
            if user_id in self.conversation_rooms[conversation_id]:
                self.conversation_rooms[conversation_id][user_id].discard(connection_id)
                if not self.conversation_rooms[conversation_id][user_id]:
                    del self.conversation_rooms[conversation_id][user_id]
            
            if not self.conversation_rooms[conversation_id]:
                del self.conversation_rooms[conversation_id]
        
        # Remove typing indicator
        if conversation_id and conversation_id in self.typing_indicators:
            self.typing_indicators[conversation_id].pop(user_id, None)
        
        # Remove metadata
        self.connection_metadata.pop(connection_id, None)
        
        logger.info(f"WebSocket disconnected: user={user_id}, connection={connection_id}")
    
    async def join_conversation(self, user_id: str, connection_id: str, conversation_id: str):
        """Join a user to a conversation room."""
        if conversation_id not in self.conversation_rooms:
            self.conversation_rooms[conversation_id] = {}
        
        if user_id not in self.conversation_rooms[conversation_id]:
            self.conversation_rooms[conversation_id][user_id] = set()
        
        self.conversation_rooms[conversation_id][user_id].add(connection_id)
        
        # Update connection metadata
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]['conversation_id'] = conversation_id
        
        logger.info(f"User {user_id} joined conversation {conversation_id}")
    
    async def leave_conversation(self, user_id: str, connection_id: str, conversation_id: str):
        """Remove a user from a conversation room."""
        if conversation_id in self.conversation_rooms:
            if user_id in self.conversation_rooms[conversation_id]:
                self.conversation_rooms[conversation_id][user_id].discard(connection_id)
                if not self.conversation_rooms[conversation_id][user_id]:
                    del self.conversation_rooms[conversation_id][user_id]
            
            if not self.conversation_rooms[conversation_id]:
                del self.conversation_rooms[conversation_id]
        
        # Remove typing indicator
        if conversation_id in self.typing_indicators:
            self.typing_indicators[conversation_id].pop(user_id, None)
        
        # Update connection metadata
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]['conversation_id'] = None
        
        logger.info(f"User {user_id} left conversation {conversation_id}")
    
    async def send_to_user(self, user_id: str, message: dict):
        """Send a message to all connections of a specific user."""
        if user_id in self.active_connections:
            disconnected_connections = []
            
            for connection_id, websocket in self.active_connections[user_id].items():
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending to user {user_id}, connection {connection_id}: {e}")
                    disconnected_connections.append(connection_id)
            
            # Clean up disconnected connections
            for connection_id in disconnected_connections:
                self.disconnect(user_id, connection_id)
    
    async def send_to_conversation(self, conversation_id: str, message: dict, exclude_user: Optional[str] = None):
        """Send a message to all users in a conversation."""
        if conversation_id not in self.conversation_rooms:
            return
        
        for user_id in self.conversation_rooms[conversation_id]:
            if exclude_user and user_id == exclude_user:
                continue
            
            await self.send_to_user(user_id, message)
    
    async def send_streaming_chunk(self, conversation_id: str, message_id: str, content: str, metadata: Optional[dict] = None):
        """Send a streaming message chunk to conversation participants."""
        message = {
            'type': 'message_chunk',
            'conversation_id': conversation_id,
            'message_id': message_id,
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.send_to_conversation(conversation_id, message)
    
    async def send_streaming_complete(self, conversation_id: str, message_id: str, final_content: str, metadata: Optional[dict] = None):
        """Send streaming completion notification."""
        message = {
            'type': 'message_complete',
            'conversation_id': conversation_id,
            'message_id': message_id,
            'content': final_content,
            'metadata': metadata or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.send_to_conversation(conversation_id, message)
    
    async def send_typing_indicator(self, conversation_id: str, user_id: str, is_typing: bool):
        """Send typing indicator to conversation participants."""
        if is_typing:
            if conversation_id not in self.typing_indicators:
                self.typing_indicators[conversation_id] = {}
            self.typing_indicators[conversation_id][user_id] = datetime.utcnow()
        else:
            if conversation_id in self.typing_indicators:
                self.typing_indicators[conversation_id].pop(user_id, None)
        
        message = {
            'type': 'user_typing',
            'conversation_id': conversation_id,
            'user_id': user_id,
            'is_typing': is_typing,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.send_to_conversation(conversation_id, message, exclude_user=user_id)
    
    def get_conversation_participants(self, conversation_id: str) -> Set[str]:
        """Get list of users currently in a conversation."""
        if conversation_id in self.conversation_rooms:
            return set(self.conversation_rooms[conversation_id].keys())
        return set()
    
    def get_typing_users(self, conversation_id: str) -> Set[str]:
        """Get list of users currently typing in a conversation."""
        if conversation_id not in self.typing_indicators:
            return set()
        
        # Remove stale typing indicators (older than 10 seconds)
        current_time = datetime.utcnow()
        stale_users = []
        
        for user_id, timestamp in self.typing_indicators[conversation_id].items():
            if (current_time - timestamp).total_seconds() > 10:
                stale_users.append(user_id)
        
        for user_id in stale_users:
            del self.typing_indicators[conversation_id][user_id]
        
        return set(self.typing_indicators[conversation_id].keys())

# Global connection manager
connection_manager = ConnectionManager()

# WebSocket router
websocket_router = APIRouter()

@websocket_router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    db: Session = Depends(get_db)
):
    """Main WebSocket endpoint for chat functionality."""
    connection_id = f"conn_{datetime.utcnow().timestamp()}"
    user_id = None
    
    try:
        # Accept connection first
        await websocket.accept()
        
        # Wait for authentication message
        auth_data = await websocket.receive_text()
        auth_message = json.loads(auth_data)
        
        if auth_message.get('type') != 'auth':
            await websocket.send_text(json.dumps({
                'type': 'error',
                'message': 'Authentication required'
            }))
            await websocket.close()
            return
        
        # Authenticate user (simplified - in production, validate JWT token)
        token = auth_message.get('token')
        if not token:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'message': 'Authentication token required'
            }))
            await websocket.close()
            return
        
        # TODO: Implement proper JWT validation
        # For now, we'll use a simple user_id from the token
        user_id = auth_message.get('user_id', 'anonymous')
        
        # Register connection
        await connection_manager.connect(websocket, user_id, connection_id)
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            'type': 'connected',
            'connection_id': connection_id,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat()
        }))
        
        # Initialize conversation manager and agent system
        conversation_manager = ConversationManager(db)
        agent_system = EnhancedAgentSystem()
        
        # Message handling loop
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                message_type = message.get('type')
                
                if message_type == 'join_conversation':
                    conversation_id = message.get('conversation_id')
                    if conversation_id:
                        await connection_manager.join_conversation(user_id, connection_id, conversation_id)
                        await websocket.send_text(json.dumps({
                            'type': 'joined_conversation',
                            'conversation_id': conversation_id,
                            'participants': list(connection_manager.get_conversation_participants(conversation_id))
                        }))
                
                elif message_type == 'leave_conversation':
                    conversation_id = message.get('conversation_id')
                    if conversation_id:
                        await connection_manager.leave_conversation(user_id, connection_id, conversation_id)
                        await websocket.send_text(json.dumps({
                            'type': 'left_conversation',
                            'conversation_id': conversation_id
                        }))
                
                elif message_type == 'typing':
                    conversation_id = message.get('conversation_id')
                    is_typing = message.get('is_typing', False)
                    if conversation_id:
                        await connection_manager.send_typing_indicator(conversation_id, user_id, is_typing)
                
                elif message_type == 'send_message':
                    # Handle message sending with streaming response
                    conversation_id = message.get('conversation_id')
                    content = message.get('content', '')
                    
                    if conversation_id and content:
                        try:
                            # Send user message
                            user_message = await conversation_manager.send_message(
                                conversation_id=conversation_id,
                                user_id=user_id,
                                content=content,
                                content_type=message.get('content_type', 'text')
                            )
                            
                            # Notify conversation participants of new user message
                            await connection_manager.send_to_conversation(conversation_id, {
                                'type': 'new_message',
                                'message': {
                                    'id': user_message.id,
                                    'role': user_message.role,
                                    'content': user_message.content,
                                    'created_at': user_message.created_at.isoformat(),
                                    'user_id': user_id
                                }
                            })
                            
                            # Generate AI response with streaming
                            if hasattr(conversation_manager, 'conversation') and conversation_manager.conversation.agent_id:
                                agent_id = conversation_manager.conversation.agent_id
                                
                                # Create placeholder for streaming response
                                response_message_id = f"response_{datetime.utcnow().timestamp()}"
                                
                                # Start streaming response
                                await connection_manager.send_to_conversation(conversation_id, {
                                    'type': 'message_start',
                                    'message_id': response_message_id,
                                    'role': 'assistant'
                                })
                                
                                # Generate response with streaming
                                full_response = ""
                                async for chunk in agent_system.generate_streaming_response(
                                    agent_id=agent_id,
                                    message=content,
                                    conversation_id=conversation_id,
                                    user_id=user_id
                                ):
                                    full_response += chunk
                                    await connection_manager.send_streaming_chunk(
                                        conversation_id, response_message_id, chunk
                                    )
                                
                                # Complete streaming response
                                await connection_manager.send_streaming_complete(
                                    conversation_id, response_message_id, full_response
                                )
                        
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            await websocket.send_text(json.dumps({
                                'type': 'error',
                                'message': f'Error processing message: {str(e)}'
                            }))
                
                elif message_type == 'ping':
                    # Handle ping/pong for connection health
                    await websocket.send_text(json.dumps({
                        'type': 'pong',
                        'timestamp': datetime.utcnow().isoformat()
                    }))
                
                else:
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': f'Unknown message type: {message_type}'
                    }))
            
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': 'Invalid JSON format'
                }))
            
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': 'Internal server error'
                }))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        if user_id:
            connection_manager.disconnect(user_id, connection_id)

# Function to get the WebSocket router
def get_websocket_router() -> APIRouter:
    """Get the WebSocket router for inclusion in the main app."""
    return websocket_router