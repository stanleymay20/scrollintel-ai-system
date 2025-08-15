"""
Agent Proxy System for ScrollIntel.
Handles inter-agent communication and message passing.
"""

from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4
import asyncio
from datetime import datetime
import logging

from ..core.interfaces import (
    BaseAgent,
    AgentRequest,
    AgentResponse,
    AgentError,
    ResponseStatus,
)

logger = logging.getLogger(__name__)


class AgentMessage:
    """Message for inter-agent communication."""
    
    def __init__(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
    ):
        self.id = str(uuid4())
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.message_type = message_type
        self.payload = payload
        self.correlation_id = correlation_id or str(uuid4())
        self.reply_to = reply_to
        self.timestamp = datetime.utcnow()
        self.delivered = False


class AgentProxy:
    """Proxy for handling inter-agent communication."""
    
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_messages: List[AgentMessage] = []
        self.sent_messages: List[AgentMessage] = []
        self.message_queue = asyncio.Queue()
        self._running = False
        self._message_processor_task: Optional[asyncio.Task] = None
    
    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler
    
    async def send_message(
        self,
        recipient_id: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> str:
        """Send a message to another agent."""
        message = AgentMessage(
            sender_id=self.agent.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
            reply_to=reply_to,
        )
        
        self.sent_messages.append(message)
        
        # In a real implementation, this would go through a message broker
        # For now, we'll use the proxy manager to route messages
        from .proxy_manager import ProxyManager
        proxy_manager = ProxyManager.get_instance()
        await proxy_manager.route_message(message)
        
        logger.info(f"Agent {self.agent.agent_id} sent message {message.id} to {recipient_id}")
        return message.id
    
    async def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent."""
        self.pending_messages.append(message)
        await self.message_queue.put(message)
        logger.info(f"Agent {self.agent.agent_id} received message {message.id} from {message.sender_id}")
    
    async def start_message_processing(self) -> None:
        """Start processing incoming messages."""
        if self._running:
            return
        
        self._running = True
        self._message_processor_task = asyncio.create_task(self._process_messages())
        logger.info(f"Started message processing for agent {self.agent.agent_id}")
    
    async def stop_message_processing(self) -> None:
        """Stop processing incoming messages."""
        if not self._running:
            return
        
        self._running = False
        if self._message_processor_task:
            self._message_processor_task.cancel()
            try:
                await self._message_processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Stopped message processing for agent {self.agent.agent_id}")
    
    async def _process_messages(self) -> None:
        """Process incoming messages from the queue."""
        while self._running:
            try:
                # Wait for a message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                # Find and execute the appropriate handler
                handler = self.message_handlers.get(message.message_type)
                if handler:
                    try:
                        await handler(message)
                        message.delivered = True
                    except Exception as e:
                        logger.error(f"Error handling message {message.id}: {e}")
                        # Send error response if reply_to is specified
                        if message.reply_to:
                            await self.send_message(
                                recipient_id=message.sender_id,
                                message_type="error_response",
                                payload={
                                    "error": str(e),
                                    "original_message_id": message.id,
                                },
                                correlation_id=message.correlation_id,
                            )
                else:
                    logger.warning(f"No handler for message type {message.message_type}")
                
            except asyncio.TimeoutError:
                # No message received, continue loop
                continue
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
    
    async def send_request_to_agent(
        self,
        recipient_id: str,
        request: AgentRequest,
        timeout: float = 30.0,
    ) -> AgentResponse:
        """Send a request to another agent and wait for response."""
        correlation_id = str(uuid4())
        
        # Send the request message
        await self.send_message(
            recipient_id=recipient_id,
            message_type="agent_request",
            payload={
                "request": request.dict(),
            },
            correlation_id=correlation_id,
            reply_to=self.agent.agent_id,
        )
        
        # Wait for response
        response_future = asyncio.Future()
        
        def response_handler(message: AgentMessage):
            if message.correlation_id == correlation_id:
                response_data = message.payload.get("response")
                if response_data:
                    response = AgentResponse(**response_data)
                    if not response_future.done():
                        response_future.set_result(response)
        
        # Register temporary handler for response
        self.register_message_handler("agent_response", response_handler)
        
        try:
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            raise AgentError(f"Request to agent {recipient_id} timed out")
        finally:
            # Clean up handler
            if "agent_response" in self.message_handlers:
                del self.message_handlers["agent_response"]
    
    def get_message_stats(self) -> Dict[str, Any]:
        """Get statistics about message handling."""
        return {
            "pending_messages": len(self.pending_messages),
            "sent_messages": len(self.sent_messages),
            "registered_handlers": list(self.message_handlers.keys()),
            "is_processing": self._running,
        }


