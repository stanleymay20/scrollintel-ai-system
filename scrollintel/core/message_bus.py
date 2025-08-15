"""
Message Bus System for ScrollIntel Inter-Agent Communication.
Provides reliable message passing, event handling, and coordination.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from uuid import uuid4
import asyncio
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of messages in the system."""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    BROADCAST = "broadcast"
    COORDINATION = "coordination"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class MessagePriority(int, Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Message:
    """Enhanced message for inter-agent communication."""
    id: str = field(default_factory=lambda: str(uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: MessageType = MessageType.REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    delivered: bool = False
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "delivered": self.delivered,
            "acknowledged": self.acknowledged,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        message = cls()
        message.id = data.get("id", str(uuid4()))
        message.sender_id = data.get("sender_id", "")
        message.recipient_id = data.get("recipient_id", "")
        message.message_type = MessageType(data.get("message_type", MessageType.REQUEST))
        message.priority = MessagePriority(data.get("priority", MessagePriority.NORMAL))
        message.payload = data.get("payload", {})
        message.correlation_id = data.get("correlation_id")
        message.reply_to = data.get("reply_to")
        message.timestamp = datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat()))
        expires_at_str = data.get("expires_at")
        message.expires_at = datetime.fromisoformat(expires_at_str) if expires_at_str else None
        message.retry_count = data.get("retry_count", 0)
        message.max_retries = data.get("max_retries", 3)
        message.delivered = data.get("delivered", False)
        message.acknowledged = data.get("acknowledged", False)
        message.metadata = data.get("metadata", {})
        return message
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return self.expires_at is not None and datetime.utcnow() > self.expires_at


class MessageHandler:
    """Base class for message handlers."""
    
    def __init__(self, handler_id: str, message_types: List[MessageType] = None):
        self.handler_id = handler_id
        self.message_types = message_types or [MessageType.REQUEST]
        self.is_active = True
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle a message and optionally return a response."""
        raise NotImplementedError
    
    def can_handle(self, message: Message) -> bool:
        """Check if this handler can handle the message."""
        return self.is_active and message.message_type in self.message_types


class MessageBus:
    """Central message bus for inter-agent communication."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[MessageHandler]] = {}  # agent_id -> handlers
        self._message_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._pending_messages: Dict[str, Message] = {}
        self._message_history: List[Message] = []
        self._event_handlers: Dict[str, List[Callable]] = {}  # event_type -> handlers
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "messages_expired": 0,
        }
    
    async def start(self) -> None:
        """Start the message bus."""
        if self._running:
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_messages())
        logger.info("Message bus started")
    
    async def stop(self) -> None:
        """Stop the message bus."""
        if not self._running:
            return
        
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Message bus stopped")
    
    def subscribe(self, agent_id: str, handler: MessageHandler) -> None:
        """Subscribe an agent to receive messages."""
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(handler)
        logger.info(f"Agent {agent_id} subscribed with handler {handler.handler_id}")
    
    def unsubscribe(self, agent_id: str, handler_id: Optional[str] = None) -> None:
        """Unsubscribe an agent or specific handler."""
        if agent_id not in self._subscribers:
            return
        
        if handler_id:
            self._subscribers[agent_id] = [
                h for h in self._subscribers[agent_id] 
                if h.handler_id != handler_id
            ]
        else:
            del self._subscribers[agent_id]
        
        logger.info(f"Agent {agent_id} unsubscribed (handler: {handler_id})")
    
    async def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        expires_in_seconds: Optional[int] = None,
        max_retries: int = 3,
    ) -> str:
        """Send a message to a specific recipient."""
        message = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            priority=priority,
            payload=payload,
            correlation_id=correlation_id,
            reply_to=reply_to,
            expires_at=datetime.utcnow() + timedelta(seconds=expires_in_seconds) if expires_in_seconds else None,
            max_retries=max_retries,
        )
        
        await self._enqueue_message(message)
        self._stats["messages_sent"] += 1
        
        logger.info(f"Message {message.id} sent from {sender_id} to {recipient_id}")
        return message.id
    
    async def broadcast_message(
        self,
        sender_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        exclude_agents: Optional[List[str]] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> List[str]:
        """Broadcast a message to all subscribed agents."""
        exclude_agents = exclude_agents or []
        message_ids = []
        
        for agent_id in self._subscribers.keys():
            if agent_id != sender_id and agent_id not in exclude_agents:
                message_id = await self.send_message(
                    sender_id=sender_id,
                    recipient_id=agent_id,
                    message_type=message_type,
                    payload=payload,
                    priority=priority,
                )
                message_ids.append(message_id)
        
        logger.info(f"Broadcast message sent to {len(message_ids)} agents")
        return message_ids
    
    async def send_request_and_wait(
        self,
        sender_id: str,
        recipient_id: str,
        payload: Dict[str, Any],
        timeout: float = 30.0,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> Message:
        """Send a request and wait for response."""
        correlation_id = str(uuid4())
        
        # Send request
        await self.send_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.REQUEST,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
            reply_to=sender_id,
        )
        
        # Wait for response
        response_future = asyncio.Future()
        
        def response_handler(message: Message):
            if (message.correlation_id == correlation_id and 
                message.message_type == MessageType.RESPONSE):
                if not response_future.done():
                    response_future.set_result(message)
        
        # Register temporary event handler
        self.register_event_handler("message_received", response_handler)
        
        try:
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request to {recipient_id} timed out after {timeout} seconds")
        finally:
            self.unregister_event_handler("message_received", response_handler)
    
    async def _enqueue_message(self, message: Message) -> None:
        """Enqueue a message for processing."""
        # Priority queue uses negative priority for correct ordering (higher priority first)
        priority = -message.priority.value
        await self._message_queue.put((priority, message.timestamp, message))
        self._pending_messages[message.id] = message
    
    async def _process_messages(self) -> None:
        """Process messages from the queue."""
        while self._running:
            try:
                # Get message from queue with timeout
                try:
                    _, _, message = await asyncio.wait_for(
                        self._message_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Clean up expired messages
                    await self._cleanup_expired_messages()
                    continue
                
                # Check if message has expired
                if message.is_expired():
                    self._stats["messages_expired"] += 1
                    logger.warning(f"Message {message.id} expired")
                    continue
                
                # Deliver message
                success = await self._deliver_message(message)
                
                if success:
                    message.delivered = True
                    self._stats["messages_delivered"] += 1
                    
                    # Emit event
                    await self._emit_event("message_delivered", message)
                else:
                    # Retry if possible
                    if message.retry_count < message.max_retries:
                        message.retry_count += 1
                        await asyncio.sleep(min(2 ** message.retry_count, 30))  # Exponential backoff
                        await self._enqueue_message(message)
                        logger.warning(f"Retrying message {message.id} (attempt {message.retry_count})")
                    else:
                        self._stats["messages_failed"] += 1
                        logger.error(f"Message {message.id} failed after {message.max_retries} retries")
                        
                        # Emit failure event
                        await self._emit_event("message_failed", message)
                
                # Remove from pending messages
                if message.id in self._pending_messages:
                    del self._pending_messages[message.id]
                
                # Add to history (keep last 1000 messages)
                self._message_history.append(message)
                if len(self._message_history) > 1000:
                    self._message_history.pop(0)
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _deliver_message(self, message: Message) -> bool:
        """Deliver a message to its recipient."""
        handlers = self._subscribers.get(message.recipient_id, [])
        
        if not handlers:
            logger.warning(f"No handlers found for recipient {message.recipient_id}")
            return False
        
        # Find suitable handlers
        suitable_handlers = [h for h in handlers if h.can_handle(message)]
        
        if not suitable_handlers:
            logger.warning(f"No suitable handlers for message type {message.message_type}")
            return False
        
        # Deliver to all suitable handlers
        delivery_success = False
        for handler in suitable_handlers:
            try:
                response = await handler.handle_message(message)
                delivery_success = True
                
                # Send response if provided
                if response and message.reply_to:
                    response.recipient_id = message.reply_to
                    response.correlation_id = message.correlation_id
                    response.message_type = MessageType.RESPONSE
                    await self._enqueue_message(response)
                
            except Exception as e:
                logger.error(f"Handler {handler.handler_id} failed to process message: {e}")
        
        # Emit event
        await self._emit_event("message_received", message)
        
        return delivery_success
    
    async def _cleanup_expired_messages(self) -> None:
        """Clean up expired messages."""
        current_time = datetime.utcnow()
        expired_messages = [
            msg_id for msg_id, msg in self._pending_messages.items()
            if msg.is_expired()
        ]
        
        for msg_id in expired_messages:
            del self._pending_messages[msg_id]
            self._stats["messages_expired"] += 1
        
        if expired_messages:
            logger.info(f"Cleaned up {len(expired_messages)} expired messages")
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def unregister_event_handler(self, event_type: str, handler: Callable) -> None:
        """Unregister an event handler."""
        if event_type in self._event_handlers:
            self._event_handlers[event_type] = [
                h for h in self._event_handlers[event_type] if h != handler
            ]
    
    async def _emit_event(self, event_type: str, data: Any) -> None:
        """Emit an event to all registered handlers."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            **self._stats,
            "pending_messages": len(self._pending_messages),
            "subscribers": len(self._subscribers),
            "total_handlers": sum(len(handlers) for handlers in self._subscribers.values()),
            "event_handlers": {
                event_type: len(handlers) 
                for event_type, handlers in self._event_handlers.items()
            },
            "is_running": self._running,
        }
    
    def get_message_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent message history."""
        return [msg.to_dict() for msg in self._message_history[-limit:]]
    
    def get_pending_messages(self) -> List[Dict[str, Any]]:
        """Get all pending messages."""
        return [msg.to_dict() for msg in self._pending_messages.values()]


# Global message bus instance
_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get the global message bus instance."""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus


async def initialize_message_bus() -> MessageBus:
    """Initialize and start the global message bus."""
    bus = get_message_bus()
    await bus.start()
    return bus


async def shutdown_message_bus() -> None:
    """Shutdown the global message bus."""
    global _message_bus
    if _message_bus:
        await _message_bus.stop()
        _message_bus = None