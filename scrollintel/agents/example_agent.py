"""
Example agent implementation to demonstrate the base agent architecture.
"""

import asyncio
from datetime import datetime
from typing import List

from ..core.interfaces import (
    BaseAgent,
    AgentType,
    AgentRequest,
    AgentResponse,
    AgentCapability,
    ResponseStatus,
)


class ExampleDataAgent(BaseAgent):
    """Example data analysis agent implementation."""
    
    def __init__(self):
        super().__init__(
            agent_id="example-data-agent",
            name="Example Data Agent",
            agent_type=AgentType.DATA_SCIENTIST
        )
        self.capabilities = [
            AgentCapability(
                name="data_analysis",
                description="Analyze CSV and JSON data files",
                input_types=["csv", "json"],
                output_types=["report", "chart"]
            ),
            AgentCapability(
                name="statistical_modeling",
                description="Create statistical models from data",
                input_types=["dataset"],
                output_types=["model", "metrics"]
            )
        ]
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process a data analysis request."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # Simple response based on prompt
            if "analyze" in request.prompt.lower():
                content = f"Data analysis complete for user {request.user_id}. Found 3 key insights."
                artifacts = ["analysis_report.pdf", "data_summary.json"]
            elif "model" in request.prompt.lower():
                content = f"Statistical model created with 85% accuracy."
                artifacts = ["model.pkl", "metrics.json"]
            else:
                content = f"Processed request: {request.prompt}"
                artifacts = []
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return AgentResponse(
                id=f"response-{request.id}",
                request_id=request.id,
                content=content,
                artifacts=artifacts,
                execution_time=execution_time,
                status=ResponseStatus.SUCCESS
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"response-{request.id}",
                request_id=request.id,
                content="Error processing request",
                execution_time=execution_time,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities."""
        return self.capabilities
    
    async def health_check(self) -> bool:
        """Check if agent is healthy."""
        try:
            # Simulate health check
            await asyncio.sleep(0.01)
            return True
        except Exception:
            return False


class ExampleCTOAgent(BaseAgent):
    """Example CTO agent implementation."""
    
    def __init__(self):
        super().__init__(
            agent_id="example-cto-agent",
            name="Example CTO Agent",
            agent_type=AgentType.CTO
        )
        self.capabilities = [
            AgentCapability(
                name="architecture_design",
                description="Design system architecture",
                input_types=["requirements"],
                output_types=["architecture_doc", "tech_stack"]
            ),
            AgentCapability(
                name="technology_selection",
                description="Select appropriate technologies",
                input_types=["project_requirements"],
                output_types=["tech_recommendations"]
            )
        ]
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process a CTO-related request."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            await asyncio.sleep(0.15)  # Simulate longer processing for complex decisions
            
            if "architecture" in request.prompt.lower():
                content = "Recommended microservices architecture with FastAPI, PostgreSQL, and Redis."
                artifacts = ["architecture_diagram.png", "tech_stack.md"]
            elif "scale" in request.prompt.lower():
                content = "Scaling strategy: Horizontal scaling with load balancers and container orchestration."
                artifacts = ["scaling_plan.pdf"]
            else:
                content = f"Technical guidance provided for: {request.prompt}"
                artifacts = ["technical_recommendations.md"]
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return AgentResponse(
                id=f"response-{request.id}",
                request_id=request.id,
                content=content,
                artifacts=artifacts,
                execution_time=execution_time,
                status=ResponseStatus.SUCCESS
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"response-{request.id}",
                request_id=request.id,
                content="Error processing technical request",
                execution_time=execution_time,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities."""
        return self.capabilities
    
    async def health_check(self) -> bool:
        """Check if agent is healthy."""
        try:
            await asyncio.sleep(0.01)
            return True
        except Exception:
            return False


# Example usage function
async def demonstrate_agent_system():
    """Demonstrate the agent system in action."""
    from ..core.registry import AgentRegistry, TaskOrchestrator
    from .proxy import AgentProxy
    from .proxy_manager import ProxyManager
    
    # Create registry and agents
    registry = AgentRegistry()
    data_agent = ExampleDataAgent()
    cto_agent = ExampleCTOAgent()
    
    # Register agents
    await registry.register_agent(data_agent)
    await registry.register_agent(cto_agent)
    
    # Create proxies for inter-agent communication
    proxy_manager = ProxyManager.get_instance()
    data_proxy = AgentProxy(data_agent)
    cto_proxy = AgentProxy(cto_agent)
    
    await proxy_manager.register_proxy(data_proxy)
    await proxy_manager.register_proxy(cto_proxy)
    
    print("=== ScrollIntel Agent System Demo ===")
    print(f"Registry Status: {registry.get_registry_status()}")
    
    # Test direct agent requests
    print("\n--- Direct Agent Requests ---")
    
    data_request = AgentRequest(
        id="req-1",
        user_id="demo-user",
        agent_id=data_agent.agent_id,
        prompt="Analyze the sales data for trends",
        created_at=datetime.utcnow()
    )
    
    response = await registry.route_request(data_request)
    print(f"Data Agent Response: {response.content}")
    print(f"Artifacts: {response.artifacts}")
    print(f"Execution Time: {response.execution_time:.3f}s")
    
    cto_request = AgentRequest(
        id="req-2",
        user_id="demo-user",
        agent_id=cto_agent.agent_id,
        prompt="Design architecture for a high-traffic web application",
        created_at=datetime.utcnow()
    )
    
    response = await registry.route_request(cto_request)
    print(f"\nCTO Agent Response: {response.content}")
    print(f"Artifacts: {response.artifacts}")
    print(f"Execution Time: {response.execution_time:.3f}s")
    
    # Test workflow orchestration
    print("\n--- Workflow Orchestration ---")
    
    orchestrator = TaskOrchestrator(registry)
    workflow_steps = [
        {
            "agent_id": cto_agent.agent_id,
            "prompt": "Design system architecture for data analytics platform",
            "priority": 1
        },
        {
            "agent_id": data_agent.agent_id,
            "prompt": "Create data model for the analytics platform",
            "priority": 2
        }
    ]
    
    workflow_results = await orchestrator.execute_workflow(
        workflow_steps,
        context={"user_id": "demo-user", "project": "analytics-platform"}
    )
    
    print("Workflow Results:")
    for i, result in enumerate(workflow_results, 1):
        print(f"  Step {i}: {result.content}")
    
    # Test health checks
    print("\n--- Health Checks ---")
    health_results = await registry.health_check_all()
    for agent_id, is_healthy in health_results.items():
        print(f"  {agent_id}: {'Healthy' if is_healthy else 'Unhealthy'}")
    
    # Cleanup
    await proxy_manager.unregister_proxy(data_agent.agent_id)
    await proxy_manager.unregister_proxy(cto_agent.agent_id)
    await registry.unregister_agent(data_agent.agent_id)
    await registry.unregister_agent(cto_agent.agent_id)
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(demonstrate_agent_system())