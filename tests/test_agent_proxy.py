"""
Unit tests for the Agent Proxy System.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from scrollintel.core.interfaces import (
    BaseAgent,
    AgentType,
    AgentStatus,
    AgentRequest,
    AgentResponse,
    AgentCapability,
    ResponseStatus,
    AgentError,
)
from scrollintel.agents.proxy import AgentProxy, AgentMessage
from scrollintel.agents.proxy_manager import ProxyManager


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, agent_id: str, name: str, agent_type: AgentType):
        super().__init__(agent_id, name, agent_type)
        self.process_request_mock = AsyncMock()
        self.health_check_mock = AsyncMock(return_value=True)
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        return await self.process_request_mock(request)
    
    def get_capabilities(self) -> list[AgentCapability]:
        return []
    
    async def health_check(self) -> bool:
        return await self.health_check_mock()


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return MockAgent("test-agent-1", "Test Agent", AgentType.DATA_SCIENTIST)


@pytest.fixture
def mock_recipient_agent():
    """Create a mock recipient agent for testing."""
    return MockAgent("test-agent-2", "Recipient Agent", AgentType.CTO)


@pytest.fixture
def agent_proxy(mock_agent):
    """Create an agent proxy for testing."""
    return AgentProxy(mock_agent)


@pytest.fixture
def proxy_manager():
    """Create a fresh proxy manager for each test."""
    # Reset the singleton instance
    ProxyManager._instance = None
    return ProxyManager.get_instance()


class TestAgentMessage:
    """Test cases for AgentMessage."""
    
    def test_agent_message_creation(self):
        """Test creating an agent message."""
        message = AgentMessage(
            sender_id="agent-1",
            recipient_id="agent-2",
            message_type="test_message",
            payload={"data": "test"},
            correlation_id="corr-123",
            reply_to="agent-1"
        )
        
        assert message.sender_id == "agent-1"
        assert message.recipient_id == "agent-2"
        assert message.message_type == "test_message"
        assert message.payload == {"data": "test"}
        assert message.correlation_id == "corr-123"
        assert message.reply_to == "agent-1"
        assert message.delivered is False
        assert isinstance(message.timestamp, datetime)
        assert message.id is not None
    
    def test_agent_message_auto_correlation_id(self):
        """Test that correlation_id is auto-generated if not provided."""
        message = AgentMessage(
            sender_id="agent-1",
            recipient_id="agent-2",
            message_type="test_message",
            payload={"data": "test"}
        )
        
        assert message.correlation_id is not None
        assert len(message.correlation_id) > 0


class TestAgentProxy:
    """Test cases for AgentProxy."""
    
    def test_proxy_creation(self, agent_proxy, mock_agent):
        """Test creating an agent proxy."""
        assert agent_proxy.agent == mock_agent
        assert len(agent_proxy.message_handlers) == 0
        assert len(agent_proxy.pending_messages) == 0
        assert len(agent_proxy.sent_messages) == 0
        assert agent_proxy._running is False
    
    def test_register_message_handler(self, agent_proxy):
        """Test registering a message handler."""
        async def test_handler(message):
            pass
        
        agent_proxy.register_message_handler("test_type", test_handler)
        
        assert "test_type" in agent_proxy.message_handlers
        assert agent_proxy.message_handlers["test_type"] == test_handler
    
    @pytest.mark.asyncio
    async def test_send_message(self, agent_proxy, proxy_manager):
        """Test sending a message."""
        # Mock the proxy manager routing
        with patch.object(proxy_manager, 'route_message', new_callable=AsyncMock) as mock_route:
            message_id = await agent_proxy.send_message(
                recipient_id="agent-2",
                message_type="test_message",
                payload={"data": "test"},
                correlation_id="corr-123"
            )
            
            assert message_id is not None
            assert len(agent_proxy.sent_messages) == 1
            
            sent_message = agent_proxy.sent_messages[0]
            assert sent_message.sender_id == agent_proxy.agent.agent_id
            assert sent_message.recipient_id == "agent-2"
            assert sent_message.message_type == "test_message"
            assert sent_message.payload == {"data": "test"}
            assert sent_message.correlation_id == "corr-123"
            
            mock_route.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_receive_message(self, agent_proxy):
        """Test receiving a message."""
        message = AgentMessage(
            sender_id="agent-2",
            recipient_id=agent_proxy.agent.agent_id,
            message_type="test_message",
            payload={"data": "test"}
        )
        
        await agent_proxy.receive_message(message)
        
        assert len(agent_proxy.pending_messages) == 1
        assert agent_proxy.pending_messages[0] == message
        
        # Check that message was added to queue
        assert not agent_proxy.message_queue.empty()
    
    @pytest.mark.asyncio
    async def test_start_stop_message_processing(self, agent_proxy):
        """Test starting and stopping message processing."""
        # Start processing
        await agent_proxy.start_message_processing()
        assert agent_proxy._running is True
        assert agent_proxy._message_processor_task is not None
        
        # Stop processing
        await agent_proxy.stop_message_processing()
        assert agent_proxy._running is False
    
    @pytest.mark.asyncio
    async def test_message_processing_with_handler(self, agent_proxy):
        """Test message processing with registered handler."""
        handler_called = asyncio.Event()
        received_message = None
        
        async def test_handler(message):
            nonlocal received_message
            received_message = message
            handler_called.set()
        
        agent_proxy.register_message_handler("test_message", test_handler)
        
        # Start processing
        await agent_proxy.start_message_processing()
        
        # Send a message
        message = AgentMessage(
            sender_id="agent-2",
            recipient_id=agent_proxy.agent.agent_id,
            message_type="test_message",
            payload={"data": "test"}
        )
        
        await agent_proxy.receive_message(message)
        
        # Wait for handler to be called
        await asyncio.wait_for(handler_called.wait(), timeout=2.0)
        
        assert received_message == message
        assert message.delivered is True
        
        await agent_proxy.stop_message_processing()
    
    @pytest.mark.asyncio
    async def test_message_processing_no_handler(self, agent_proxy):
        """Test message processing without registered handler."""
        # Start processing
        await agent_proxy.start_message_processing()
        
        # Send a message with no handler
        message = AgentMessage(
            sender_id="agent-2",
            recipient_id=agent_proxy.agent.agent_id,
            message_type="unknown_message",
            payload={"data": "test"}
        )
        
        await agent_proxy.receive_message(message)
        
        # Give some time for processing
        await asyncio.sleep(0.1)
        
        # Message should not be marked as delivered
        assert message.delivered is False
        
        await agent_proxy.stop_message_processing()
    
    @pytest.mark.asyncio
    async def test_send_request_to_agent(self, agent_proxy, proxy_manager):
        """Test sending a request to another agent."""
        # Mock the proxy manager and response
        response_data = {
            "id": "response-1",
            "request_id": "request-1",
            "content": "Test response",
            "execution_time": 0.5,
            "status": "success"
        }
        
        # Start message processing
        await agent_proxy.start_message_processing()
        
        # Create a mock request
        request = AgentRequest(
            id="request-1",
            user_id="user-1",
            agent_id="agent-2",
            prompt="Test prompt",
            created_at=datetime.utcnow()
        )
        
        # Mock the routing and simulate response
        async def mock_route_and_respond(message):
            # Simulate receiving a response
            response_message = AgentMessage(
                sender_id="agent-2",
                recipient_id=agent_proxy.agent.agent_id,
                message_type="agent_response",
                payload={"response": response_data},
                correlation_id=message.correlation_id
            )
            await agent_proxy.receive_message(response_message)
        
        with patch.object(proxy_manager, 'route_message', side_effect=mock_route_and_respond):
            response = await agent_proxy.send_request_to_agent("agent-2", request, timeout=1.0)
            
            assert response.id == "response-1"
            assert response.content == "Test response"
            assert response.status == ResponseStatus.SUCCESS
        
        await agent_proxy.stop_message_processing()
    
    @pytest.mark.asyncio
    async def test_send_request_to_agent_timeout(self, agent_proxy, proxy_manager):
        """Test request timeout."""
        # Start message processing
        await agent_proxy.start_message_processing()
        
        request = AgentRequest(
            id="request-1",
            user_id="user-1",
            agent_id="agent-2",
            prompt="Test prompt",
            created_at=datetime.utcnow()
        )
        
        # Mock routing without response
        with patch.object(proxy_manager, 'route_message', new_callable=AsyncMock):
            with pytest.raises(AgentError, match="timed out"):
                await agent_proxy.send_request_to_agent("agent-2", request, timeout=0.1)
        
        await agent_proxy.stop_message_processing()
    
    def test_get_message_stats(self, agent_proxy):
        """Test getting message statistics."""
        # Add some test data
        agent_proxy.pending_messages.append(Mock())
        agent_proxy.sent_messages.append(Mock())
        agent_proxy.register_message_handler("test", Mock())
        
        stats = agent_proxy.get_message_stats()
        
        assert stats["pending_messages"] == 1
        assert stats["sent_messages"] == 1
        assert "test" in stats["registered_handlers"]
        assert stats["is_processing"] is False


class TestProxyManager:
    """Test cases for ProxyManager."""
    
    def test_singleton_pattern(self):
        """Test that ProxyManager follows singleton pattern."""
        manager1 = ProxyManager.get_instance()
        manager2 = ProxyManager.get_instance()
        
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_register_proxy(self, proxy_manager, agent_proxy):
        """Test registering a proxy."""
        await proxy_manager.register_proxy(agent_proxy)
        
        assert agent_proxy.agent.agent_id in proxy_manager.proxies
        assert proxy_manager.proxies[agent_proxy.agent.agent_id] == agent_proxy
        assert agent_proxy._running is True
    
    @pytest.mark.asyncio
    async def test_unregister_proxy(self, proxy_manager, agent_proxy):
        """Test unregistering a proxy."""
        await proxy_manager.register_proxy(agent_proxy)
        await proxy_manager.unregister_proxy(agent_proxy.agent.agent_id)
        
        assert agent_proxy.agent.agent_id not in proxy_manager.proxies
        assert agent_proxy._running is False
    
    @pytest.mark.asyncio
    async def test_route_message(self, proxy_manager, agent_proxy, mock_recipient_agent):
        """Test routing a message."""
        recipient_proxy = AgentProxy(mock_recipient_agent)
        
        await proxy_manager.register_proxy(agent_proxy)
        await proxy_manager.register_proxy(recipient_proxy)
        
        message = AgentMessage(
            sender_id=agent_proxy.agent.agent_id,
            recipient_id=mock_recipient_agent.agent_id,
            message_type="test_message",
            payload={"data": "test"}
        )
        
        await proxy_manager.route_message(message)
        
        # Check that message was received by recipient
        assert len(recipient_proxy.pending_messages) == 1
        assert recipient_proxy.pending_messages[0] == message
    
    @pytest.mark.asyncio
    async def test_route_message_agent_not_found(self, proxy_manager):
        """Test routing message to non-existent agent."""
        message = AgentMessage(
            sender_id="agent-1",
            recipient_id="nonexistent-agent",
            message_type="test_message",
            payload={"data": "test"}
        )
        
        with pytest.raises(AgentError, match="not found"):
            await proxy_manager.route_message(message)
    
    def test_get_proxy(self, proxy_manager, agent_proxy):
        """Test getting a proxy by agent ID."""
        asyncio.run(proxy_manager.register_proxy(agent_proxy))
        
        retrieved_proxy = proxy_manager.get_proxy(agent_proxy.agent.agent_id)
        assert retrieved_proxy == agent_proxy
        
        nonexistent_proxy = proxy_manager.get_proxy("nonexistent-agent")
        assert nonexistent_proxy is None
    
    def test_get_all_proxies(self, proxy_manager, agent_proxy, mock_recipient_agent):
        """Test getting all proxies."""
        recipient_proxy = AgentProxy(mock_recipient_agent)
        
        asyncio.run(proxy_manager.register_proxy(agent_proxy))
        asyncio.run(proxy_manager.register_proxy(recipient_proxy))
        
        all_proxies = proxy_manager.get_all_proxies()
        assert len(all_proxies) == 2
        assert agent_proxy in all_proxies
        assert recipient_proxy in all_proxies
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, proxy_manager, agent_proxy, mock_recipient_agent):
        """Test broadcasting a message."""
        recipient_proxy = AgentProxy(mock_recipient_agent)
        
        await proxy_manager.register_proxy(agent_proxy)
        await proxy_manager.register_proxy(recipient_proxy)
        
        # Mock the send_message method
        with patch.object(agent_proxy, 'send_message', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = "message-id-1"
            
            message_ids = await proxy_manager.broadcast_message(
                sender_id=agent_proxy.agent.agent_id,
                message_type="broadcast_message",
                payload={"data": "broadcast"}
            )
            
            assert len(message_ids) == 1
            assert message_ids[0] == "message-id-1"
            mock_send.assert_called_once_with(
                recipient_id=mock_recipient_agent.agent_id,
                message_type="broadcast_message",
                payload={"data": "broadcast"}
            )
    
    @pytest.mark.asyncio
    async def test_broadcast_message_with_exclusions(self, proxy_manager, agent_proxy, mock_recipient_agent):
        """Test broadcasting with excluded agents."""
        recipient_proxy = AgentProxy(mock_recipient_agent)
        
        await proxy_manager.register_proxy(agent_proxy)
        await proxy_manager.register_proxy(recipient_proxy)
        
        with patch.object(agent_proxy, 'send_message', new_callable=AsyncMock) as mock_send:
            message_ids = await proxy_manager.broadcast_message(
                sender_id=agent_proxy.agent.agent_id,
                message_type="broadcast_message",
                payload={"data": "broadcast"},
                exclude_agents=[mock_recipient_agent.agent_id]
            )
            
            assert len(message_ids) == 0
            mock_send.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_broadcast_message_sender_not_found(self, proxy_manager):
        """Test broadcasting from non-existent sender."""
        with pytest.raises(AgentError, match="Sender agent .* not found"):
            await proxy_manager.broadcast_message(
                sender_id="nonexistent-agent",
                message_type="broadcast_message",
                payload={"data": "broadcast"}
            )
    
    def test_get_manager_stats(self, proxy_manager, agent_proxy, mock_recipient_agent):
        """Test getting manager statistics."""
        recipient_proxy = AgentProxy(mock_recipient_agent)
        
        asyncio.run(proxy_manager.register_proxy(agent_proxy))
        asyncio.run(proxy_manager.register_proxy(recipient_proxy))
        
        stats = proxy_manager.get_manager_stats()
        
        assert stats["total_proxies"] == 2
        assert stats["active_proxies"] == 2  # Both should be running
        assert agent_proxy.agent.agent_id in stats["proxy_stats"]
        assert mock_recipient_agent.agent_id in stats["proxy_stats"]