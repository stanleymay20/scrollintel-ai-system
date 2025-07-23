"""
Agent Registry System for ScrollIntel.
Manages agent discovery, routing, and coordination.
"""

from typing import Dict, List, Optional, Any
from uuid import uuid4
import asyncio
from datetime import datetime

from .interfaces import (
    BaseAgent,
    AgentType,
    AgentStatus,
    AgentRequest,
    AgentResponse,
    AgentCapability,
    AgentError,
)


class AgentRegistry:
    """Central registry for managing all ScrollIntel AI agents."""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._capabilities: Dict[str, List[str]] = {}  # capability_name -> [agent_ids]
        self._agent_types: Dict[AgentType, List[str]] = {}  # agent_type -> [agent_ids]
        self._lock = asyncio.Lock()
    
    async def register_agent(self, agent: BaseAgent) -> str:
        """Register an agent in the registry."""
        async with self._lock:
            if agent.agent_id in self._agents:
                raise AgentError(f"Agent with ID {agent.agent_id} already registered")
            
            # Register the agent
            self._agents[agent.agent_id] = agent
            
            # Index by agent type
            if agent.agent_type not in self._agent_types:
                self._agent_types[agent.agent_type] = []
            self._agent_types[agent.agent_type].append(agent.agent_id)
            
            # Index by capabilities
            capabilities = agent.get_capabilities()
            for capability in capabilities:
                if capability.name not in self._capabilities:
                    self._capabilities[capability.name] = []
                self._capabilities[capability.name].append(agent.agent_id)
            
            # Start the agent
            agent.start()
            
            return agent.agent_id
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the registry."""
        async with self._lock:
            if agent_id not in self._agents:
                raise AgentError(f"Agent with ID {agent_id} not found")
            
            agent = self._agents[agent_id]
            
            # Stop the agent
            agent.stop()
            
            # Remove from type index
            if agent.agent_type in self._agent_types:
                self._agent_types[agent.agent_type].remove(agent_id)
                if not self._agent_types[agent.agent_type]:
                    del self._agent_types[agent.agent_type]
            
            # Remove from capability index
            capabilities = agent.get_capabilities()
            for capability in capabilities:
                if capability.name in self._capabilities:
                    self._capabilities[capability.name].remove(agent_id)
                    if not self._capabilities[capability.name]:
                        del self._capabilities[capability.name]
            
            # Remove the agent
            del self._agents[agent_id]
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of a specific type."""
        agent_ids = self._agent_types.get(agent_type, [])
        return [self._agents[agent_id] for agent_id in agent_ids if agent_id in self._agents]
    
    def get_agents_by_capability(self, capability_name: str) -> List[BaseAgent]:
        """Get all agents that have a specific capability."""
        agent_ids = self._capabilities.get(capability_name, [])
        return [self._agents[agent_id] for agent_id in agent_ids if agent_id in self._agents]
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    def get_active_agents(self) -> List[BaseAgent]:
        """Get all active agents."""
        return [agent for agent in self._agents.values() if agent.status == AgentStatus.ACTIVE]
    
    async def route_request(self, request: AgentRequest) -> AgentResponse:
        """Route a request to the appropriate agent."""
        # If agent_id is specified, route directly
        if request.agent_id:
            agent = self.get_agent(request.agent_id)
            if not agent:
                raise AgentError(f"Agent with ID {request.agent_id} not found")
            if agent.status != AgentStatus.ACTIVE:
                raise AgentError(f"Agent {request.agent_id} is not active")
            return await agent.process_request(request)
        
        # Otherwise, find the best agent based on context or capabilities
        # This is a simple implementation - could be enhanced with ML-based routing
        suitable_agents = self._find_suitable_agents(request)
        
        if not suitable_agents:
            raise AgentError("No suitable agent found for the request")
        
        # Use the first available agent (could implement load balancing here)
        agent = suitable_agents[0]
        return await agent.process_request(request)
    
    def _find_suitable_agents(self, request: AgentRequest) -> List[BaseAgent]:
        """Find agents suitable for handling the request."""
        # Simple heuristic based on request context
        context = request.context
        suitable_agents = []
        
        # Check for specific agent type hints in context
        if "agent_type" in context:
            agent_type = AgentType(context["agent_type"])
            suitable_agents.extend(self.get_agents_by_type(agent_type))
        
        # Check for capability requirements
        if "required_capabilities" in context:
            for capability in context["required_capabilities"]:
                suitable_agents.extend(self.get_agents_by_capability(capability))
        
        # If no specific requirements, return all active agents
        if not suitable_agents:
            suitable_agents = self.get_active_agents()
        
        # Filter to only active agents
        return [agent for agent in suitable_agents if agent.status == AgentStatus.ACTIVE]
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Get all available capabilities across all agents."""
        all_capabilities = []
        for agent in self._agents.values():
            all_capabilities.extend(agent.get_capabilities())
        return all_capabilities
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get the current status of the registry."""
        status_counts = {}
        for status in AgentStatus:
            status_counts[status.value] = 0
        
        for agent in self._agents.values():
            status_counts[agent.status.value] += 1
        
        return {
            "total_agents": len(self._agents),
            "agent_status": status_counts,
            "agent_types": {
                agent_type.value: len(agent_ids) 
                for agent_type, agent_ids in self._agent_types.items()
            },
            "capabilities": list(self._capabilities.keys()),
        }
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all agents."""
        health_results = {}
        
        for agent_id, agent in self._agents.items():
            try:
                is_healthy = await agent.health_check()
                health_results[agent_id] = is_healthy
                
                # Update agent status based on health check
                if not is_healthy and agent.status == AgentStatus.ACTIVE:
                    agent.status = AgentStatus.ERROR
                elif is_healthy and agent.status == AgentStatus.ERROR:
                    agent.status = AgentStatus.ACTIVE
                    
            except Exception as e:
                health_results[agent_id] = False
                agent.status = AgentStatus.ERROR
        
        return health_results


class TaskOrchestrator:
    """Orchestrates complex tasks across multiple agents."""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._active_tasks: Dict[str, Dict[str, Any]] = {}
    
    async def execute_workflow(self, workflow_steps: List[Dict[str, Any]], context: Dict[str, Any] = None) -> List[AgentResponse]:
        """Execute a multi-step workflow across agents."""
        task_id = str(uuid4())
        context = context or {}
        
        self._active_tasks[task_id] = {
            "steps": workflow_steps,
            "context": context,
            "results": [],
            "started_at": datetime.utcnow(),
            "status": "running"
        }
        
        try:
            results = []
            
            for i, step in enumerate(workflow_steps):
                # Create request for this step
                request = AgentRequest(
                    id=str(uuid4()),
                    user_id=context.get("user_id", "system"),
                    agent_id=step.get("agent_id"),
                    prompt=step["prompt"],
                    context={
                        **context,
                        **step.get("context", {}),
                        "workflow_task_id": task_id,
                        "step_number": i + 1,
                        "previous_results": results,
                    },
                    priority=step.get("priority", 1),
                    created_at=datetime.utcnow()
                )
                
                # Execute the step
                response = await self.registry.route_request(request)
                results.append(response)
                
                # Update task progress
                self._active_tasks[task_id]["results"] = results
                
                # Check if step failed and handle accordingly
                if response.status != "success":
                    if step.get("continue_on_error", False):
                        continue
                    else:
                        raise AgentError(f"Workflow step {i + 1} failed: {response.error_message}")
            
            self._active_tasks[task_id]["status"] = "completed"
            return results
            
        except Exception as e:
            self._active_tasks[task_id]["status"] = "failed"
            self._active_tasks[task_id]["error"] = str(e)
            raise
        
        finally:
            self._active_tasks[task_id]["completed_at"] = datetime.utcnow()
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a running task."""
        return self._active_tasks.get(task_id)
    
    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all active tasks."""
        return {
            task_id: task_info 
            for task_id, task_info in self._active_tasks.items() 
            if task_info["status"] == "running"
        }