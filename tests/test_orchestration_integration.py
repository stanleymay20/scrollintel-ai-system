"""
Integration tests for ScrollIntel orchestration and task coordination.
Tests multi-agent workflows, dependency management, and message passing.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

from scrollintel.core.orchestrator import (
    TaskOrchestrator,
    Workflow,
    WorkflowTask,
    TaskDependency,
    TaskStatus,
    WorkflowStatus,
)
from scrollintel.core.workflow_templates import WorkflowTemplateLibrary
from scrollintel.core.message_bus import (
    MessageBus,
    Message,
    MessageType,
    MessagePriority,
    MessageHandler,
    get_message_bus,
    initialize_message_bus,
)
from scrollintel.core.registry import AgentRegistry
from scrollintel.core.interfaces import (
    BaseAgent,
    AgentType,
    AgentStatus,
    AgentRequest,
    AgentResponse,
    ResponseStatus,
    AgentCapability,
)


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, agent_id: str, name: str, agent_type: AgentType, response_delay: float = 0.1):
        super().__init__(agent_id, name, agent_type)
        self.response_delay = response_delay
        self.processed_requests = []
        self.should_fail = False
        self.failure_message = "Mock failure"
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process a request with optional delay and failure simulation."""
        self.processed_requests.append(request)
        
        # Simulate processing time
        await asyncio.sleep(self.response_delay)
        
        if self.should_fail:
            return AgentResponse(
                id=f"response_{request.id}",
                request_id=request.id,
                content="",
                execution_time=self.response_delay,
                status=ResponseStatus.ERROR,
                error_message=self.failure_message,
            )
        
        # Generate mock response based on agent type
        content = f"Mock response from {self.name} for: {request.prompt}"
        artifacts = [f"artifact_{self.agent_type.value}_{len(self.processed_requests)}"]
        
        return AgentResponse(
            id=f"response_{request.id}",
            request_id=request.id,
            content=content,
            artifacts=artifacts,
            execution_time=self.response_delay,
            status=ResponseStatus.SUCCESS,
        )
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return mock capabilities."""
        return [
            AgentCapability(
                name=f"{self.agent_type.value}_capability",
                description=f"Mock capability for {self.agent_type.value}",
                input_types=["text"],
                output_types=["text", "json"],
            )
        ]
    
    async def health_check(self) -> bool:
        """Mock health check."""
        return self.status == AgentStatus.ACTIVE


class MockMessageHandler(MessageHandler):
    """Mock message handler for testing."""
    
    def __init__(self, handler_id: str, agent_id: str):
        super().__init__(handler_id, [MessageType.REQUEST, MessageType.EVENT])
        self.agent_id = agent_id
        self.received_messages = []
        self.should_respond = True
        self.response_payload = {"status": "handled"}
    
    async def handle_message(self, message: Message) -> Message:
        """Handle a message and return response."""
        self.received_messages.append(message)
        
        if self.should_respond and message.reply_to:
            return Message(
                sender_id=self.agent_id,
                recipient_id=message.reply_to,
                message_type=MessageType.RESPONSE,
                payload=self.response_payload,
                correlation_id=message.correlation_id,
            )
        
        return None


@pytest.fixture
def message_bus():
    """Create and start a message bus for testing."""
    async def _create_message_bus():
        bus = MessageBus()
        await bus.start()
        return bus
    
    async def _cleanup_message_bus(bus):
        await bus.stop()
    
    return _create_message_bus, _cleanup_message_bus


@pytest.fixture
def agent_registry():
    """Create an agent registry with mock agents."""
    async def _create_registry():
        registry = AgentRegistry()
        
        # Create mock agents
        agents = [
            MockAgent("cto_agent", "ScrollCTO", AgentType.CTO),
            MockAgent("data_scientist", "ScrollDataScientist", AgentType.DATA_SCIENTIST),
            MockAgent("ml_engineer", "ScrollMLEngineer", AgentType.ML_ENGINEER),
            MockAgent("ai_engineer", "ScrollAIEngineer", AgentType.AI_ENGINEER),
            MockAgent("analyst", "ScrollAnalyst", AgentType.ANALYST),
            MockAgent("bi_developer", "ScrollBI", AgentType.BI_DEVELOPER),
        ]
        
        # Register agents
        for agent in agents:
            await registry.register_agent(agent)
        
        return registry, agents
    
    return _create_registry


@pytest.fixture
def orchestrator(agent_registry):
    """Create a task orchestrator."""
    async def _create_orchestrator():
        registry, agents = await agent_registry()
        orchestrator_instance = TaskOrchestrator(registry)
        return orchestrator_instance, registry, agents
    
    return _create_orchestrator


class TestMessageBus:
    """Test the message bus system."""
    
    @pytest.mark.asyncio
    async def test_message_bus_basic_functionality(self, message_bus):
        """Test basic message sending and receiving."""
        create_bus, cleanup_bus = message_bus
        bus = await create_bus()
        
        try:
            handler = MockMessageHandler("test_handler", "test_agent")
            bus.subscribe("test_agent", handler)
        
        # Send a message
        message_id = await message_bus.send_message(
            sender_id="sender",
            recipient_id="test_agent",
            message_type=MessageType.REQUEST,
            payload={"test": "data"},
        )
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check that message was received
        assert len(handler.received_messages) == 1
        assert handler.received_messages[0].payload["test"] == "data"
        assert message_bus.get_stats()["messages_delivered"] == 1
    
    @pytest.mark.asyncio
    async def test_message_bus_request_response(self, message_bus):
        """Test request-response pattern."""
        handler = MockMessageHandler("test_handler", "test_agent")
        message_bus.subscribe("test_agent", handler)
        
        # Send request and wait for response
        response = await message_bus.send_request_and_wait(
            sender_id="sender",
            recipient_id="test_agent",
            payload={"request": "test"},
            timeout=5.0,
        )
        
        assert response.message_type == MessageType.RESPONSE
        assert response.payload["status"] == "handled"
    
    @pytest.mark.asyncio
    async def test_message_bus_broadcast(self, message_bus):
        """Test message broadcasting."""
        handlers = [
            MockMessageHandler(f"handler_{i}", f"agent_{i}")
            for i in range(3)
        ]
        
        for i, handler in enumerate(handlers):
            message_bus.subscribe(f"agent_{i}", handler)
        
        # Broadcast message
        message_ids = await message_bus.broadcast_message(
            sender_id="broadcaster",
            message_type=MessageType.EVENT,
            payload={"event": "test_broadcast"},
        )
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check that all agents received the message
        assert len(message_ids) == 3
        for handler in handlers:
            assert len(handler.received_messages) == 1
            assert handler.received_messages[0].payload["event"] == "test_broadcast"
    
    @pytest.mark.asyncio
    async def test_message_priority_handling(self, message_bus):
        """Test message priority handling."""
        handler = MockMessageHandler("test_handler", "test_agent")
        message_bus.subscribe("test_agent", handler)
        
        # Send messages with different priorities
        await message_bus.send_message(
            sender_id="sender",
            recipient_id="test_agent",
            message_type=MessageType.REQUEST,
            payload={"priority": "low"},
            priority=MessagePriority.LOW,
        )
        
        await message_bus.send_message(
            sender_id="sender",
            recipient_id="test_agent",
            message_type=MessageType.REQUEST,
            payload={"priority": "critical"},
            priority=MessagePriority.CRITICAL,
        )
        
        await message_bus.send_message(
            sender_id="sender",
            recipient_id="test_agent",
            message_type=MessageType.REQUEST,
            payload={"priority": "normal"},
            priority=MessagePriority.NORMAL,
        )
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Check that messages were processed in priority order
        assert len(handler.received_messages) == 3
        assert handler.received_messages[0].payload["priority"] == "critical"
        assert handler.received_messages[1].payload["priority"] == "normal"
        assert handler.received_messages[2].payload["priority"] == "low"


class TestTaskOrchestrator:
    """Test the task orchestrator system."""
    
    @pytest.mark.asyncio
    async def test_workflow_creation_from_template(self, orchestrator):
        """Test creating workflow from template."""
        orchestrator_instance, registry, agents = orchestrator
        
        # Create workflow from template
        workflow_id = await orchestrator_instance.create_workflow_from_template(
            template_id="data_science_pipeline",
            name="Test Data Science Workflow",
            context={"dataset_id": "test_dataset"},
            created_by="test_user",
        )
        
        # Check workflow was created
        workflow_status = orchestrator_instance.get_workflow_status(workflow_id)
        assert workflow_status is not None
        assert workflow_status["name"] == "Test Data Science Workflow"
        assert workflow_status["status"] == WorkflowStatus.CREATED
        assert len(workflow_status["tasks"]) > 0
    
    @pytest.mark.asyncio
    async def test_custom_workflow_creation(self, orchestrator):
        """Test creating custom workflow."""
        orchestrator_instance, registry, agents = orchestrator
        
        tasks = [
            {
                "name": "Task 1",
                "agent_type": "data_scientist",
                "prompt": "Analyze the data",
                "context": {"task_type": "analysis"},
            },
            {
                "name": "Task 2",
                "agent_type": "ml_engineer",
                "prompt": "Train a model",
                "context": {"task_type": "training"},
                "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
            },
        ]
        
        workflow_id = await orchestrator_instance.create_custom_workflow(
            name="Custom Test Workflow",
            tasks=tasks,
            context={"test": "context"},
            description="Test workflow",
            created_by="test_user",
        )
        
        # Check workflow was created
        workflow_status = orchestrator_instance.get_workflow_status(workflow_id)
        assert workflow_status is not None
        assert workflow_status["name"] == "Custom Test Workflow"
        assert len(workflow_status["tasks"]) == 2
    
    @pytest.mark.asyncio
    async def test_workflow_execution_success(self, orchestrator):
        """Test successful workflow execution."""
        orchestrator_instance, registry, agents = orchestrator
        
        # Create simple workflow
        tasks = [
            {
                "name": "Task 1",
                "agent_type": "data_scientist",
                "prompt": "Analyze the data",
                "timeout": 5.0,
            },
            {
                "name": "Task 2",
                "agent_type": "ml_engineer",
                "prompt": "Train a model",
                "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
                "timeout": 5.0,
            },
        ]
        
        workflow_id = await orchestrator_instance.create_custom_workflow(
            name="Test Workflow",
            tasks=tasks,
        )
        
        # Execute workflow
        result = await orchestrator_instance.execute_workflow(workflow_id)
        
        # Check results
        assert result["status"] == WorkflowStatus.COMPLETED
        assert result["progress"] == 100.0
        assert len(result["results"]) == 2
        assert all(r is not None for r in result["results"])
        
        # Check workflow status
        workflow_status = orchestrator_instance.get_workflow_status(workflow_id)
        assert workflow_status["status"] == WorkflowStatus.COMPLETED
        assert workflow_status["progress"] == 100.0
    
    @pytest.mark.asyncio
    async def test_workflow_execution_with_failure(self, orchestrator):
        """Test workflow execution with task failure."""
        orchestrator_instance, registry, agents = orchestrator
        
        # Make one agent fail
        agents[1].should_fail = True
        agents[1].failure_message = "Simulated failure"
        
        tasks = [
            {
                "name": "Task 1",
                "agent_type": "data_scientist",
                "prompt": "This will fail",
                "timeout": 5.0,
            },
        ]
        
        workflow_id = await orchestrator_instance.create_custom_workflow(
            name="Failing Workflow",
            tasks=tasks,
        )
        
        # Execute workflow and expect failure
        with pytest.raises(Exception) as exc_info:
            await orchestrator_instance.execute_workflow(workflow_id)
        
        assert "Simulated failure" in str(exc_info.value)
        
        # Check workflow status
        workflow_status = orchestrator_instance.get_workflow_status(workflow_id)
        assert workflow_status["status"] == WorkflowStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_workflow_execution_with_continue_on_error(self, orchestrator):
        """Test workflow execution with continue_on_error flag."""
        orchestrator_instance, registry, agents = orchestrator
        
        # Make one agent fail
        agents[1].should_fail = True
        
        tasks = [
            {
                "name": "Failing Task",
                "agent_type": "data_scientist",
                "prompt": "This will fail",
                "continue_on_error": True,
                "timeout": 5.0,
            },
            {
                "name": "Success Task",
                "agent_type": "ml_engineer",
                "prompt": "This will succeed",
                "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
                "timeout": 5.0,
            },
        ]
        
        workflow_id = await orchestrator_instance.create_custom_workflow(
            name="Continue on Error Workflow",
            tasks=tasks,
        )
        
        # Execute workflow
        result = await orchestrator_instance.execute_workflow(workflow_id)
        
        # Check that workflow completed despite failure
        assert result["status"] == WorkflowStatus.COMPLETED
        assert result["progress"] == 100.0
        
        # Check that first task failed but second succeeded
        workflow_status = orchestrator_instance.get_workflow_status(workflow_id)
        assert workflow_status["tasks"][0]["status"] == TaskStatus.FAILED
        assert workflow_status["tasks"][1]["status"] == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_workflow_dependency_management(self, orchestrator):
        """Test complex dependency management."""
        orchestrator_instance, registry, agents = orchestrator
        
        tasks = [
            {
                "name": "Task A",
                "agent_type": "data_scientist",
                "prompt": "Task A",
                "timeout": 5.0,
            },
            {
                "name": "Task B",
                "agent_type": "ml_engineer",
                "prompt": "Task B",
                "timeout": 5.0,
            },
            {
                "name": "Task C",
                "agent_type": "analyst",
                "prompt": "Task C depends on A and B",
                "dependencies": [
                    {"task_id": "0", "dependency_type": "completion"},
                    {"task_id": "1", "dependency_type": "completion"},
                ],
                "timeout": 5.0,
            },
        ]
        
        workflow_id = await orchestrator_instance.create_custom_workflow(
            name="Dependency Test Workflow",
            tasks=tasks,
        )
        
        # Execute workflow
        result = await orchestrator_instance.execute_workflow(workflow_id)
        
        # Check that all tasks completed
        assert result["status"] == WorkflowStatus.COMPLETED
        assert len(result["results"]) == 3
        
        # Check that dependencies were respected
        workflow_status = orchestrator_instance.get_workflow_status(workflow_id)
        for task in workflow_status["tasks"]:
            assert task["status"] == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_workflow_pause_resume_cancel(self, orchestrator):
        """Test workflow pause, resume, and cancel operations."""
        orchestrator_instance, registry, agents = orchestrator
        
        # Create workflow with slow tasks
        for agent in agents:
            agent.response_delay = 2.0  # Make tasks slower
        
        tasks = [
            {
                "name": "Slow Task",
                "agent_type": "data_scientist",
                "prompt": "This is slow",
                "timeout": 10.0,
            },
        ]
        
        workflow_id = await orchestrator_instance.create_custom_workflow(
            name="Slow Workflow",
            tasks=tasks,
        )
        
        # Start workflow execution in background
        execution_task = asyncio.create_task(
            orchestrator_instance.execute_workflow(workflow_id)
        )
        
        # Wait a bit then pause
        await asyncio.sleep(0.5)
        await orchestrator_instance.pause_workflow(workflow_id)
        
        # Check workflow is paused
        workflow_status = orchestrator_instance.get_workflow_status(workflow_id)
        assert workflow_status["status"] == WorkflowStatus.PAUSED
        
        # Cancel the workflow
        await orchestrator_instance.cancel_workflow(workflow_id)
        
        # Check workflow is cancelled
        workflow_status = orchestrator_instance.get_workflow_status(workflow_id)
        assert workflow_status["status"] == WorkflowStatus.CANCELLED
        
        # Clean up
        execution_task.cancel()
        try:
            await execution_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_workflow_templates(self, orchestrator):
        """Test workflow template functionality."""
        orchestrator_instance, registry, agents = orchestrator
        
        # Get available templates
        templates = orchestrator_instance.get_workflow_templates()
        assert len(templates) > 0
        
        # Check that default templates are available
        template_ids = [t["id"] for t in templates]
        assert "data_science_pipeline" in template_ids
        assert "bi_report_generation" in template_ids
        assert "ai_model_deployment" in template_ids
        
        # Test creating workflow from each template
        for template in templates[:2]:  # Test first 2 templates
            workflow_id = await orchestrator_instance.create_workflow_from_template(
                template_id=template["id"],
                name=f"Test {template['name']}",
            )
            
            workflow_status = orchestrator_instance.get_workflow_status(workflow_id)
            assert workflow_status is not None
            assert workflow_status["status"] == WorkflowStatus.CREATED
    
    @pytest.mark.asyncio
    async def test_workflow_progress_tracking(self, orchestrator):
        """Test workflow progress tracking and callbacks."""
        orchestrator_instance, registry, agents = orchestrator
        
        progress_updates = []
        
        def progress_callback(workflow_id: str, progress: float):
            progress_updates.append((workflow_id, progress))
        
        tasks = [
            {
                "name": "Task 1",
                "agent_type": "data_scientist",
                "prompt": "Task 1",
                "timeout": 5.0,
            },
            {
                "name": "Task 2",
                "agent_type": "ml_engineer",
                "prompt": "Task 2",
                "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
                "timeout": 5.0,
            },
        ]
        
        workflow_id = await orchestrator_instance.create_custom_workflow(
            name="Progress Test Workflow",
            tasks=tasks,
        )
        
        # Register progress callback
        orchestrator_instance.register_progress_callback(workflow_id, progress_callback)
        
        # Execute workflow
        await orchestrator_instance.execute_workflow(workflow_id)
        
        # Check progress updates
        assert len(progress_updates) >= 2  # At least one update per task
        assert progress_updates[-1][1] == 100.0  # Final progress should be 100%
    
    @pytest.mark.asyncio
    async def test_workflow_cleanup(self, orchestrator):
        """Test workflow cleanup functionality."""
        orchestrator_instance, registry, agents = orchestrator
        
        # Create and execute a simple workflow
        tasks = [
            {
                "name": "Simple Task",
                "agent_type": "data_scientist",
                "prompt": "Simple task",
                "timeout": 5.0,
            },
        ]
        
        workflow_id = await orchestrator_instance.create_custom_workflow(
            name="Cleanup Test Workflow",
            tasks=tasks,
        )
        
        await orchestrator_instance.execute_workflow(workflow_id)
        
        # Verify workflow exists
        workflow_status = orchestrator_instance.get_workflow_status(workflow_id)
        assert workflow_status is not None
        
        # Clean up completed workflows (with 0 hours threshold for immediate cleanup)
        cleaned_count = await orchestrator_instance.cleanup_completed_workflows(older_than_hours=0)
        
        # Verify workflow was cleaned up
        assert cleaned_count >= 1
        workflow_status = orchestrator_instance.get_workflow_status(workflow_id)
        assert workflow_status is None


class TestWorkflowTemplates:
    """Test workflow template functionality."""
    
    def test_template_library_completeness(self):
        """Test that all templates are properly defined."""
        templates = WorkflowTemplateLibrary.get_all_templates()
        
        # Check that we have expected templates
        expected_templates = [
            "data_science_pipeline",
            "bi_report_generation",
            "ai_model_deployment",
            "data_quality_assessment",
            "competitive_analysis",
            "customer_segmentation",
        ]
        
        for template_id in expected_templates:
            assert template_id in templates
            template = templates[template_id]
            assert template.name
            assert template.description
            assert len(template.tasks) > 0
            assert template.estimated_duration > 0
    
    def test_template_task_structure(self):
        """Test that template tasks have proper structure."""
        template = WorkflowTemplateLibrary.get_data_science_pipeline()
        
        for i, task in enumerate(template.tasks):
            assert "name" in task
            assert "prompt" in task
            assert task["name"]
            assert task["prompt"]
            
            # Check dependencies reference valid task IDs
            if "dependencies" in task:
                for dep in task["dependencies"]:
                    task_id = int(dep["task_id"])
                    assert 0 <= task_id < i  # Dependencies must reference earlier tasks
    
    def test_template_dependency_validation(self):
        """Test that template dependencies are valid."""
        templates = WorkflowTemplateLibrary.get_all_templates()
        
        for template in templates.values():
            task_ids = set(str(i) for i in range(len(template.tasks)))
            
            for task in template.tasks:
                if "dependencies" in task:
                    for dep in task["dependencies"]:
                        assert dep["task_id"] in task_ids
                        assert dep["dependency_type"] in ["completion", "data", "condition"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])