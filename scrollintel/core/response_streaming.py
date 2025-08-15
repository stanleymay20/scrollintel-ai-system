"""
Response Streaming System - Implements typing indicators and real-time response streaming
Implements requirements 2.3, 6.3 for typing indicators and response streaming.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4
import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class StreamingEventType(str, Enum):
    """Types of streaming events."""
    TYPING_START = "typing_start"
    TYPING_STOP = "typing_stop"
    RESPONSE_CHUNK = "response_chunk"
    RESPONSE_COMPLETE = "response_complete"
    ERROR = "error"
    AGENT_STATUS = "agent_status"
    THINKING = "thinking"


@dataclass
class StreamingEvent:
    """Event in the streaming response."""
    event_type: StreamingEventType
    agent_id: str
    conversation_id: str
    data: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class TypingIndicator:
    """Typing indicator configuration."""
    agent_id: str
    message: str
    duration_ms: int = 2000
    animation_type: str = "dots"  # dots, pulse, wave
    show_agent_avatar: bool = True


class ResponseStreamingEngine:
    """Engine for managing response streaming and typing indicators."""
    
    def __init__(self):
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.typing_indicators: Dict[str, TypingIndicator] = {}
        self.agent_thinking_messages = self._initialize_thinking_messages()
    
    def _initialize_thinking_messages(self) -> Dict[str, List[str]]:
        """Initialize agent-specific thinking messages."""
        return {
            "scroll-cto-agent": [
                "Analyzing system architecture...",
                "Evaluating scalability options...",
                "Reviewing technical requirements...",
                "Considering best practices...",
                "Assessing performance implications...",
                "Designing solution approach..."
            ],
            "scroll-data-scientist": [
                "Analyzing data patterns...",
                "Running statistical tests...",
                "Examining correlations...",
                "Validating hypotheses...",
                "Processing data quality checks...",
                "Generating insights..."
            ],
            "scroll-ml-engineer": [
                "Training model parameters...",
                "Optimizing hyperparameters...",
                "Evaluating model performance...",
                "Analyzing feature importance...",
                "Validating model accuracy...",
                "Fine-tuning algorithms..."
            ],
            "scroll-bi-agent": [
                "Building visualizations...",
                "Calculating business metrics...",
                "Analyzing trends...",
                "Generating reports...",
                "Processing KPIs...",
                "Creating dashboards..."
            ],
            "scroll-analyst": [
                "Examining data trends...",
                "Performing analysis...",
                "Identifying patterns...",
                "Calculating statistics...",
                "Generating summaries...",
                "Creating recommendations..."
            ]
        }
    
    async def start_typing_indicator(
        self, 
        conversation_id: str, 
        agent_id: str,
        custom_message: Optional[str] = None
    ) -> str:
        """Start typing indicator for an agent."""
        
        # Get agent-specific thinking message
        thinking_messages = self.agent_thinking_messages.get(agent_id, ["Processing your request..."])
        import random
        message = custom_message or random.choice(thinking_messages)
        
        indicator = TypingIndicator(
            agent_id=agent_id,
            message=message,
            duration_ms=2000,
            animation_type="dots"
        )
        
        indicator_id = f"typing_{uuid4()}"
        self.typing_indicators[indicator_id] = indicator
        
        # Send typing start event
        event = StreamingEvent(
            event_type=StreamingEventType.TYPING_START,
            agent_id=agent_id,
            conversation_id=conversation_id,
            data={
                "indicator_id": indicator_id,
                "message": message,
                "animation_type": indicator.animation_type,
                "show_avatar": indicator.show_agent_avatar
            }
        )
        
        await self._broadcast_event(conversation_id, event)
        
        logger.debug(f"Started typing indicator {indicator_id} for agent {agent_id}")
        return indicator_id
    
    async def stop_typing_indicator(self, conversation_id: str, indicator_id: str):
        """Stop typing indicator."""
        if indicator_id in self.typing_indicators:
            indicator = self.typing_indicators[indicator_id]
            
            event = StreamingEvent(
                event_type=StreamingEventType.TYPING_STOP,
                agent_id=indicator.agent_id,
                conversation_id=conversation_id,
                data={"indicator_id": indicator_id}
            )
            
            await self._broadcast_event(conversation_id, event)
            del self.typing_indicators[indicator_id]
            
            logger.debug(f"Stopped typing indicator {indicator_id}")
    
    async def stream_response(
        self, 
        conversation_id: str, 
        agent_id: str,
        response_generator: AsyncGenerator[str, None],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Stream response chunks to clients."""
        
        stream_id = f"stream_{uuid4()}"
        
        # Initialize stream
        self.active_streams[stream_id] = {
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "start_time": datetime.now(),
            "chunks_sent": 0,
            "total_length": 0,
            "metadata": metadata or {}
        }
        
        try:
            full_response = ""
            chunk_count = 0
            
            async for chunk in response_generator:
                if chunk:
                    full_response += chunk
                    chunk_count += 1
                    
                    # Send chunk event
                    event = StreamingEvent(
                        event_type=StreamingEventType.RESPONSE_CHUNK,
                        agent_id=agent_id,
                        conversation_id=conversation_id,
                        data={
                            "stream_id": stream_id,
                            "chunk": chunk,
                            "chunk_index": chunk_count,
                            "partial_response": full_response
                        }
                    )
                    
                    await self._broadcast_event(conversation_id, event)
                    
                    # Update stream info
                    self.active_streams[stream_id]["chunks_sent"] = chunk_count
                    self.active_streams[stream_id]["total_length"] = len(full_response)
                    
                    # Small delay to make streaming visible
                    await asyncio.sleep(0.05)
            
            # Send completion event
            completion_event = StreamingEvent(
                event_type=StreamingEventType.RESPONSE_COMPLETE,
                agent_id=agent_id,
                conversation_id=conversation_id,
                data={
                    "stream_id": stream_id,
                    "full_response": full_response,
                    "total_chunks": chunk_count,
                    "total_length": len(full_response),
                    "duration_ms": (datetime.now() - self.active_streams[stream_id]["start_time"]).total_seconds() * 1000
                }
            )
            
            await self._broadcast_event(conversation_id, completion_event)
            
            logger.info(f"Completed response stream {stream_id} with {chunk_count} chunks")
            return full_response
            
        except Exception as e:
            # Send error event
            error_event = StreamingEvent(
                event_type=StreamingEventType.ERROR,
                agent_id=agent_id,
                conversation_id=conversation_id,
                data={
                    "stream_id": stream_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            await self._broadcast_event(conversation_id, error_event)
            logger.error(f"Error in response stream {stream_id}: {e}")
            raise
            
        finally:
            # Clean up stream
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    async def show_thinking_process(
        self, 
        conversation_id: str, 
        agent_id: str,
        thinking_steps: List[str],
        step_duration_ms: int = 1500
    ):
        """Show agent thinking process with multiple steps."""
        
        for i, step in enumerate(thinking_steps):
            event = StreamingEvent(
                event_type=StreamingEventType.THINKING,
                agent_id=agent_id,
                conversation_id=conversation_id,
                data={
                    "step": step,
                    "step_index": i,
                    "total_steps": len(thinking_steps),
                    "progress": (i + 1) / len(thinking_steps)
                }
            )
            
            await self._broadcast_event(conversation_id, event)
            await asyncio.sleep(step_duration_ms / 1000)
    
    async def update_agent_status(
        self, 
        conversation_id: str, 
        agent_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Update agent status (busy, available, processing, etc.)."""
        
        event = StreamingEvent(
            event_type=StreamingEventType.AGENT_STATUS,
            agent_id=agent_id,
            conversation_id=conversation_id,
            data={
                "status": status,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self._broadcast_event(conversation_id, event)
    
    async def register_websocket(self, conversation_id: str, websocket: websockets.WebSocketServerProtocol):
        """Register websocket connection for a conversation."""
        self.websocket_connections[conversation_id] = websocket
        logger.debug(f"Registered websocket for conversation {conversation_id}")
    
    async def unregister_websocket(self, conversation_id: str):
        """Unregister websocket connection."""
        if conversation_id in self.websocket_connections:
            del self.websocket_connections[conversation_id]
            logger.debug(f"Unregistered websocket for conversation {conversation_id}")
    
    async def _broadcast_event(self, conversation_id: str, event: StreamingEvent):
        """Broadcast event to connected clients."""
        
        # Convert event to JSON
        event_data = {
            "event_type": event.event_type.value,
            "agent_id": event.agent_id,
            "conversation_id": event.conversation_id,
            "data": event.data,
            "timestamp": event.timestamp.isoformat()
        }
        
        message = json.dumps(event_data)
        
        # Send to websocket if connected
        if conversation_id in self.websocket_connections:
            websocket = self.websocket_connections[conversation_id]
            try:
                await websocket.send(message)
            except ConnectionClosed:
                # Remove closed connection
                await self.unregister_websocket(conversation_id)
            except Exception as e:
                logger.error(f"Error sending websocket message: {e}")
        
        # Also log for debugging
        logger.debug(f"Broadcast event {event.event_type.value} for conversation {conversation_id}")
    
    def create_response_generator(self, full_response: str, chunk_size: int = 10) -> AsyncGenerator[str, None]:
        """Create a response generator that yields chunks of the response."""
        
        async def generator():
            words = full_response.split()
            current_chunk = ""
            word_count = 0
            
            for word in words:
                current_chunk += word + " "
                word_count += 1
                
                if word_count >= chunk_size or word.endswith(('.', '!', '?', ':')):
                    yield current_chunk.strip()
                    current_chunk = ""
                    word_count = 0
            
            # Yield remaining chunk if any
            if current_chunk.strip():
                yield current_chunk.strip()
        
        return generator()
    
    def create_code_response_generator(self, full_response: str) -> AsyncGenerator[str, None]:
        """Create a response generator optimized for code responses."""
        
        async def generator():
            lines = full_response.split('\n')
            current_chunk = ""
            
            for line in lines:
                current_chunk += line + '\n'
                
                # Yield chunk at logical breakpoints
                if (line.strip().endswith((':', '{', '}', ';')) or 
                    line.strip().startswith(('#', '//', '/*', '*')) or
                    len(current_chunk) > 100):
                    yield current_chunk
                    current_chunk = ""
            
            # Yield remaining chunk
            if current_chunk.strip():
                yield current_chunk
        
        return generator()
    
    async def simulate_agent_work(
        self, 
        conversation_id: str, 
        agent_id: str,
        work_description: str,
        duration_seconds: float = 3.0
    ):
        """Simulate agent working with progress updates."""
        
        steps = [
            f"Starting {work_description}...",
            f"Processing requirements...",
            f"Analyzing data...",
            f"Generating solution...",
            f"Finalizing {work_description}..."
        ]
        
        step_duration = duration_seconds / len(steps)
        
        for i, step in enumerate(steps):
            await self.update_agent_status(
                conversation_id, 
                agent_id, 
                "processing",
                {
                    "current_step": step,
                    "progress": (i + 1) / len(steps),
                    "estimated_remaining_seconds": (len(steps) - i - 1) * step_duration
                }
            )
            
            await asyncio.sleep(step_duration)
        
        await self.update_agent_status(conversation_id, agent_id, "available")
    
    def get_active_streams(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active streams."""
        return self.active_streams.copy()
    
    def get_typing_indicators(self) -> Dict[str, TypingIndicator]:
        """Get active typing indicators."""
        return self.typing_indicators.copy()


# WebSocket handler for real-time communication
class StreamingWebSocketHandler:
    """WebSocket handler for streaming communication."""
    
    def __init__(self, streaming_engine: ResponseStreamingEngine):
        self.streaming_engine = streaming_engine
    
    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle new websocket connection."""
        try:
            # Extract conversation_id from path
            conversation_id = path.split('/')[-1] if '/' in path else path
            
            # Register connection
            await self.streaming_engine.register_websocket(conversation_id, websocket)
            
            # Send connection confirmation
            await websocket.send(json.dumps({
                "event_type": "connection_established",
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat()
            }))
            
            # Keep connection alive
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(conversation_id, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from client: {message}")
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
        
        except ConnectionClosed:
            logger.info(f"WebSocket connection closed for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Unregister connection
            if 'conversation_id' in locals():
                await self.streaming_engine.unregister_websocket(conversation_id)
    
    async def _handle_client_message(self, conversation_id: str, data: Dict[str, Any]):
        """Handle message from client."""
        message_type = data.get("type")
        
        if message_type == "ping":
            # Respond to ping with pong
            await self.streaming_engine._broadcast_event(conversation_id, StreamingEvent(
                event_type=StreamingEventType.AGENT_STATUS,
                agent_id="system",
                conversation_id=conversation_id,
                data={"type": "pong"}
            ))
        
        elif message_type == "request_status":
            # Send current status
            active_streams = self.streaming_engine.get_active_streams()
            typing_indicators = self.streaming_engine.get_typing_indicators()
            
            await self.streaming_engine._broadcast_event(conversation_id, StreamingEvent(
                event_type=StreamingEventType.AGENT_STATUS,
                agent_id="system",
                conversation_id=conversation_id,
                data={
                    "type": "status_update",
                    "active_streams": len([s for s in active_streams.values() if s["conversation_id"] == conversation_id]),
                    "typing_indicators": len([t for t in typing_indicators.values()]),
                }
            ))