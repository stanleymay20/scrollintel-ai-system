"""
Simple integration tests for ScrollIntel orchestration system.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, List, Any

from scrollintel.core.orchestrator import (
    TaskOrchestrator,
    WorkflowStatus,
    TaskStatus,
)
from scrollintel.core.workflow_templates import WorkflowTemplateLibrary
from scrollintel.core.registry import AgentRegistry
from scrollintel.core.message_bus import MessageBus, MessageType, MessagePriority
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
    
    def __init__(self, agent_id: str, name: str, agent_type: AgentType):
        super().__init__(agent_id, name, agent_type)
        self.processed_requests = []
        self.should_fail = False
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process a request."""
        self.processed_requests.append(request)
        
        if self.should_fail:
            return AgentResponse(
                id=f"response_{request.id}",
                request_id=request.id,
                content="",
                execution_time=0.1,
                status=ResponseStatus.ERROR,
                error_message="Mock failure",
            )
        
        return AgentResponse(
            id=f"response_{request.id}",
            request_id=request.id,
            content=f"Mock response from {self.name}",
            artifacts=[f"artifact_{self.agent_type.value}"],
            execution_time=0.1,
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


class TestOrchestrationSystem:
    """Test the orchestration system."""
    
    @pytest.mark.asyncio
    async def test_workflow_template_creation(self):
        """Test creating workflow from template."""
        # Create registry and orchestrator
        registry = AgentRegistry()
        orchestrator = TaskOrchestrator(registry)
        
        # Create mock agents
        agents = [
            MockAgent("data_scientist", "ScrollDataScientist", AgentType.DATA_SCIENTIST),
            MockAgent("ml_engineer", "ScrollMLEngineer", AgentType.ML_ENGINEER),
        ]
        
        # Register agents
        for agent in agents:
            await registry.register_agent(agent)
        
        # Create workflow from template
        workflow_id = await orchestrator.create_workflow_from_template(
            template_id="data_science_pipeline",
            name="Test Workflow",
            context={"dataset_id": "test_dataset"},
        )
        
        # Check workflow was created
        workflow_status = orchestrator.get_workflow_status(workflow_id)
        assert workflow_status is not None
        assert workflow_status["name"] == "Test Workflow"
        assert workflow_status["status"] == WorkflowStatus.CREATED
    
    @pytest.mark.asyncio
    async def test_custom_workflow_execution(self):
        """Test executing a custom workflow."""
        # Create registry and orchestrator
        registry = AgentRegistry()
        orchestrator = TaskOrchestrator(registry)
        
        # Create mock agents
        agents = [
            MockAgent("data_scientist", "ScrollDataScientist", AgentType.DATA_SCIENTIST),
            MockAgent("ml_engineer", "ScrollMLEngineer", AgentType.ML_ENGINEER),
        ]
        
        # Register agents
        for agent in agents:
            await registry.register_agent(agent)
        
        # Create simple workflow
        tasks = [
            {
                "name": "Data Analysis",
                "agent_type": "data_scientist",
                "prompt": "Analyze the data",
                "timeout": 5.0,
            },
            {
                "name": "Model Training",
                "agent_type": "ml_engineer",
                "prompt": "Train a model",
                "dependencies": [{"task_id": "0", "dependency_type": "completion"}],
                "timeout": 5.0,
            },
        ]
        
        workflow_id = await orchestrator.create_custom_workflow(
            name="Test Custom Workflow",
            tasks=tasks,
        )
        
        # Execute workflow
        result = await orchestrator.execute_workflow(workflow_id)
        
        # Check results
        assert result["status"] == WorkflowStatus.COMPLETED
        assert result["progress"] == 100.0
        assert len(result["results"]) == 2
        
        # Check that agents processed requests
        assert len(agents[0].processed_requests) == 1
        assert len(agents[1].processed_requests) == 1
    
    @pytest.mark.asyncio
    async def test_workflow_with_failure(self):
        """Test workflow execution with failure."""
        # Create registry and orchestrator
        registry = AgentRegistry()
        orchestrator = TaskOrchestrator(registry)
        
        # Create mock agent that will fail
        agent = MockAgent("data_scientist", "ScrollDataScientist", AgentType.DATA_SCIENTIST)
        agent.should_fail = True
        await registry.register_agent(agent)
        
        # Create workflow
        tasks = [
            {
                "name": "Failing Task",
                "agent_type": "data_scientist",
                "prompt": "This will fail",
                "timeout": 5.0,
            },
        ]
        
        workflow_id = await orchestrator.create_custom_workflow(
            name="Failing Workflow",
            tasks=tasks,
        )
        
        # Execute workflow and expect failure
        with pytest.raises(Exception):
            await orchestrator.execute_workflow(workflow_id)
        
        # Check workflow status
        workflow_status = orchestrator.get_workflow_status(workflow_id)
        assert workflow_status["status"] == WorkflowStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_message_bus_basic_functionality(self):
        """Test basic message bus functionality."""
        bus = MessageBus()
        await bus.start()
        
        try:
            # Test sending a message
            message_id = await bus.send_message(
                sender_id="test_sender",
                recipient_id="test_recipient",
                message_type=MessageType.REQUEST,
                payload={"test": "data"},
            )
            
            assert message_id is not None
            
            # Check stats
            stats = bus.get_stats()
            assert stats["messages_sent"] == 1
            
        finally:
            await bus.stop()
    
    def test_workflow_templates_available(self):
        """Test that workflow templates are available."""
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
    
    def test_workflow_template_structure(self):
        """Test workflow template structure."""
        template = WorkflowTemplateLibrary.get_data_science_pipeline()
        
        # Check basic structure
        assert template.id == "data_science_pipeline"
        assert template.name
        assert template.description
        assert len(template.tasks) > 0
        
        # Check task structure
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])