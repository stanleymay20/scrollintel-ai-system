"""
Unit tests for the Agent Registry System.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
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
from scrollintel.core.registry import AgentRegistry, TaskOrchestrator


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, agent_id: str, name: str, agent_type: AgentType, capabilities: list = None):
        super().__init__(agent_id, name, agent_type)
        self._capabilities = capabilities or []
        self.process_request_mock = AsyncMock()
        self.health_check_mock = AsyncMock(return_value=True)
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        return await self.process_request_mock(request)
    
    def get_capabilities(self) -> list[AgentCapability]:
        return self._capabilities
    
    async def health_check(self) -> bool:
        return await self.health_check_mock()


@pytest.fixture
def registry():
    """Create a fresh agent registry for each test."""
    return AgentRegistry()


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    capabilities = [
        AgentCapability(
            name="data_analysis",
            description="Analyze data",
            input_types=["csv", "json"],
            output_types=["report", "chart"]
        )
    ]
    return MockAgent("test-agent-1", "Test Agent", AgentType.DATA_SCIENTIST, capabilities)


@pytest.fixture
def mock_cto_agent():
    """Create a mock CTO agent for testing."""
    capabilities = [
        AgentCapability(
            name="architecture_design",
            description="Design system architecture",
            input_types=["requirements"],
            output_types=["architecture_doc"]
        )
    ]
    return MockAgent("cto-agent-1", "CTO Agent", AgentType.CTO, capabilities)


class TestAgentRegistry:
    """Test cases for AgentRegistry."""
    
    @pytest.mark.asyncio
    async def test_register_agent(self, registry, mock_agent):
        """Test agent registration."""
        agent_id = await registry.register_agent(mock_agent)
        
        assert agent_id == mock_agent.agent_id
        assert mock_agent.status == AgentStatus.ACTIVE
        assert registry.get_agent(agent_id) == mock_agent
    
    @pytest.mark.asyncio
    async def test_register_duplicate_agent(self, registry, mock_agent):
        """Test registering an agent with duplicate ID raises error."""
        await registry.register_agent(mock_agent)
        
        with pytest.raises(AgentError, match="already registered"):
            await registry.register_agent(mock_agent)
    
    @pytest.mark.asyncio
    async def test_unregister_agent(self, registry, mock_agent):
        """Test agent unregistration."""
        agent_id = await registry.register_agent(mock_agent)
        await registry.unregister_agent(agent_id)
        
        assert mock_agent.status == AgentStatus.INACTIVE
        assert registry.get_agent(agent_id) is None
    
    @pytest.mark.asyncio
    async def test_unregister_nonexistent_agent(self, registry):
        """Test unregistering non-existent agent raises error."""
        with pytest.raises(AgentError, match="not found"):
            await registry.unregister_agent("nonexistent-agent")
    
    def test_get_agents_by_type(self, registry, mock_agent, mock_cto_agent):
        """Test getting agents by type."""
        asyncio.run(registry.register_agent(mock_agent))
        asyncio.run(registry.register_agent(mock_cto_agent))
        
        data_scientists = registry.get_agents_by_type(AgentType.DATA_SCIENTIST)
        cto_agents = registry.get_agents_by_type(AgentType.CTO)
        
        assert len(data_scientists) == 1
        assert data_scientists[0] == mock_agent
        assert len(cto_agents) == 1
        assert cto_agents[0] == mock_cto_agent
    
    def test_get_agents_by_capability(self, registry, mock_agent, mock_cto_agent):
        """Test getting agents by capability."""
        asyncio.run(registry.register_agent(mock_agent))
        asyncio.run(registry.register_agent(mock_cto_agent))
        
        data_analysis_agents = registry.get_agents_by_capability("data_analysis")
        architecture_agents = registry.get_agents_by_capability("architecture_design")
        
        assert len(data_analysis_agents) == 1
        assert data_analysis_agents[0] == mock_agent
        assert len(architecture_agents) == 1
        assert architecture_agents[0] == mock_cto_agent
    
    def test_get_all_agents(self, registry, mock_agent, mock_cto_agent):
        """Test getting all agents."""
        asyncio.run(registry.register_agent(mock_agent))
        asyncio.run(registry.register_agent(mock_cto_agent))
        
        all_agents = registry.get_all_agents()
        assert len(all_agents) == 2
        assert mock_agent in all_agents
        assert mock_cto_agent in all_agents
    
    def test_get_active_agents(self, registry, mock_agent, mock_cto_agent):
        """Test getting only active agents."""
        asyncio.run(registry.register_agent(mock_agent))
        asyncio.run(registry.register_agent(mock_cto_agent))
        
        # Stop one agent
        mock_cto_agent.stop()
        
        active_agents = registry.get_active_agents()
        assert len(active_agents) == 1
        assert active_agents[0] == mock_agent
    
    @pytest.mark.asyncio
    async def test_route_request_with_agent_id(self, registry, mock_agent):
        """Test routing request to specific agent."""
        await registry.register_agent(mock_agent)
        
        # Mock the response
        expected_response = AgentResponse(
            id="response-1",
            request_id="request-1",
            content="Test response",
            execution_time=0.5,
            status=ResponseStatus.SUCCESS
        )
        mock_agent.process_request_mock.return_value = expected_response
        
        request = AgentRequest(
            id="request-1",
            user_id="user-1",
            agent_id=mock_agent.agent_id,
            prompt="Test prompt",
            created_at=datetime.utcnow()
        )
        
        response = await registry.route_request(request)
        
        assert response == expected_response
        mock_agent.process_request_mock.assert_called_once_with(request)
    
    @pytest.mark.asyncio
    async def test_route_request_agent_not_found(self, registry):
        """Test routing request to non-existent agent."""
        request = AgentRequest(
            id="request-1",
            user_id="user-1",
            agent_id="nonexistent-agent",
            prompt="Test prompt",
            created_at=datetime.utcnow()
        )
        
        with pytest.raises(AgentError, match="not found"):
            await registry.route_request(request)
    
    @pytest.mark.asyncio
    async def test_route_request_agent_inactive(self, registry, mock_agent):
        """Test routing request to inactive agent."""
        await registry.register_agent(mock_agent)
        mock_agent.stop()  # Make agent inactive
        
        request = AgentRequest(
            id="request-1",
            user_id="user-1",
            agent_id=mock_agent.agent_id,
            prompt="Test prompt",
            created_at=datetime.utcnow()
        )
        
        with pytest.raises(AgentError, match="not active"):
            await registry.route_request(request)
    
    @pytest.mark.asyncio
    async def test_route_request_auto_routing(self, registry, mock_agent):
        """Test automatic routing based on agent type."""
        await registry.register_agent(mock_agent)
        
        expected_response = AgentResponse(
            id="response-1",
            request_id="request-1",
            content="Test response",
            execution_time=0.5,
            status=ResponseStatus.SUCCESS
        )
        mock_agent.process_request_mock.return_value = expected_response
        
        request = AgentRequest(
            id="request-1",
            user_id="user-1",
            agent_id="",  # No specific agent
            prompt="Test prompt",
            context={"agent_type": "data_scientist"},
            created_at=datetime.utcnow()
        )
        
        response = await registry.route_request(request)
        
        assert response == expected_response
        mock_agent.process_request_mock.assert_called_once_with(request)
    
    @pytest.mark.asyncio
    async def test_route_request_no_suitable_agent(self, registry):
        """Test routing when no suitable agent is found."""
        request = AgentRequest(
            id="request-1",
            user_id="user-1",
            agent_id="",
            prompt="Test prompt",
            context={"agent_type": "nonexistent_type"},
            created_at=datetime.utcnow()
        )
        
        with pytest.raises(AgentError, match="No suitable agent found"):
            await registry.route_request(request)
    
    def test_get_capabilities(self, registry, mock_agent, mock_cto_agent):
        """Test getting all capabilities."""
        asyncio.run(registry.register_agent(mock_agent))
        asyncio.run(registry.register_agent(mock_cto_agent))
        
        capabilities = registry.get_capabilities()
        capability_names = [cap.name for cap in capabilities]
        
        assert "data_analysis" in capability_names
        assert "architecture_design" in capability_names
    
    def test_get_registry_status(self, registry, mock_agent, mock_cto_agent):
        """Test getting registry status."""
        asyncio.run(registry.register_agent(mock_agent))
        asyncio.run(registry.register_agent(mock_cto_agent))
        
        # Stop one agent
        mock_cto_agent.stop()
        
        status = registry.get_registry_status()
        
        assert status["total_agents"] == 2
        assert status["agent_status"]["active"] == 1
        assert status["agent_status"]["inactive"] == 1
        assert "data_scientist" in status["agent_types"]
        assert "cto" in status["agent_types"]
        assert "data_analysis" in status["capabilities"]
        assert "architecture_design" in status["capabilities"]
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, registry, mock_agent, mock_cto_agent):
        """Test health check for all agents."""
        await registry.register_agent(mock_agent)
        await registry.register_agent(mock_cto_agent)
        
        # Set up health check responses
        mock_agent.health_check_mock.return_value = True
        mock_cto_agent.health_check_mock.return_value = False
        
        health_results = await registry.health_check_all()
        
        assert health_results[mock_agent.agent_id] is True
        assert health_results[mock_cto_agent.agent_id] is False
        assert mock_agent.status == AgentStatus.ACTIVE
        assert mock_cto_agent.status == AgentStatus.ERROR


class TestTaskOrchestrator:
    """Test cases for TaskOrchestrator."""
    
    @pytest.fixture
    def orchestrator(self, registry):
        """Create a task orchestrator with registry."""
        return TaskOrchestrator(registry)
    
    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, orchestrator, registry, mock_agent):
        """Test successful workflow execution."""
        await registry.register_agent(mock_agent)
        
        # Mock responses
        response1 = AgentResponse(
            id="response-1",
            request_id="request-1",
            content="Step 1 complete",
            execution_time=0.5,
            status=ResponseStatus.SUCCESS
        )
        response2 = AgentResponse(
            id="response-2",
            request_id="request-2",
            content="Step 2 complete",
            execution_time=0.3,
            status=ResponseStatus.SUCCESS
        )
        mock_agent.process_request_mock.side_effect = [response1, response2]
        
        workflow_steps = [
            {
                "agent_id": mock_agent.agent_id,
                "prompt": "Execute step 1",
                "priority": 1
            },
            {
                "agent_id": mock_agent.agent_id,
                "prompt": "Execute step 2",
                "priority": 1
            }
        ]
        
        results = await orchestrator.execute_workflow(
            workflow_steps,
            context={"user_id": "user-1"}
        )
        
        assert len(results) == 2
        assert results[0] == response1
        assert results[1] == response2
        assert mock_agent.process_request_mock.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_workflow_failure(self, orchestrator, registry, mock_agent):
        """Test workflow execution with failure."""
        await registry.register_agent(mock_agent)
        
        # Mock first response as success, second as failure
        response1 = AgentResponse(
            id="response-1",
            request_id="request-1",
            content="Step 1 complete",
            execution_time=0.5,
            status=ResponseStatus.SUCCESS
        )
        response2 = AgentResponse(
            id="response-2",
            request_id="request-2",
            content="Step 2 failed",
            execution_time=0.3,
            status=ResponseStatus.ERROR,
            error_message="Processing failed"
        )
        mock_agent.process_request_mock.side_effect = [response1, response2]
        
        workflow_steps = [
            {
                "agent_id": mock_agent.agent_id,
                "prompt": "Execute step 1",
                "priority": 1
            },
            {
                "agent_id": mock_agent.agent_id,
                "prompt": "Execute step 2",
                "priority": 1
            }
        ]
        
        with pytest.raises(AgentError, match="Workflow step 2 failed"):
            await orchestrator.execute_workflow(
                workflow_steps,
                context={"user_id": "user-1"}
            )
    
    @pytest.mark.asyncio
    async def test_execute_workflow_continue_on_error(self, orchestrator, registry, mock_agent):
        """Test workflow execution with continue_on_error flag."""
        await registry.register_agent(mock_agent)
        
        # Mock responses
        response1 = AgentResponse(
            id="response-1",
            request_id="request-1",
            content="Step 1 complete",
            execution_time=0.5,
            status=ResponseStatus.SUCCESS
        )
        response2 = AgentResponse(
            id="response-2",
            request_id="request-2",
            content="Step 2 failed",
            execution_time=0.3,
            status=ResponseStatus.ERROR,
            error_message="Processing failed"
        )
        response3 = AgentResponse(
            id="response-3",
            request_id="request-3",
            content="Step 3 complete",
            execution_time=0.4,
            status=ResponseStatus.SUCCESS
        )
        mock_agent.process_request_mock.side_effect = [response1, response2, response3]
        
        workflow_steps = [
            {
                "agent_id": mock_agent.agent_id,
                "prompt": "Execute step 1",
                "priority": 1
            },
            {
                "agent_id": mock_agent.agent_id,
                "prompt": "Execute step 2",
                "priority": 1,
                "continue_on_error": True
            },
            {
                "agent_id": mock_agent.agent_id,
                "prompt": "Execute step 3",
                "priority": 1
            }
        ]
        
        results = await orchestrator.execute_workflow(
            workflow_steps,
            context={"user_id": "user-1"}
        )
        
        assert len(results) == 3
        assert results[0] == response1
        assert results[1] == response2
        assert results[2] == response3
    
    def test_get_task_status(self, orchestrator):
        """Test getting task status."""
        # Since we can't easily test the async workflow execution,
        # we'll test that the method exists and returns None for non-existent tasks
        status = orchestrator.get_task_status("nonexistent-task")
        assert status is None
    
    def test_get_active_tasks(self, orchestrator):
        """Test getting active tasks."""
        active_tasks = orchestrator.get_active_tasks()
        assert isinstance(active_tasks, dict)
        assert len(active_tasks) == 0  # No active tasks initially