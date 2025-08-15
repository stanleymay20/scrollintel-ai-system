"""
Agent routes for ScrollIntel API.
Handles agent communication, execution, and management.
"""

import time
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, Field

from ...core.interfaces import (
    AgentRequest, AgentResponse, AgentCapability, AgentType, 
    SecurityContext, ResponseStatus
)
from ...core.registry import AgentRegistry, TaskOrchestrator
from ...security.middleware import require_agent_execution, require_permission
from ...security.permissions import Permission
from ...security.audit import audit_logger, AuditAction


# Request/Response models
class AgentExecutionRequest(BaseModel):
    """Request model for agent execution."""
    prompt: str = Field(..., min_length=1, max_length=10000)
    agent_id: Optional[str] = None
    agent_type: Optional[AgentType] = None
    context: Dict[str, Any] = {}
    priority: int = Field(default=1, ge=1, le=10)
    timeout_seconds: Optional[int] = Field(default=300, ge=1, le=3600)


class WorkflowExecutionRequest(BaseModel):
    """Request model for multi-agent workflow execution."""
    workflow_name: str
    steps: List[Dict[str, Any]]
    context: Dict[str, Any] = {}
    continue_on_error: bool = False


class AgentStatusResponse(BaseModel):
    """Response model for agent status."""
    agent_id: str
    name: str
    type: str
    status: str
    capabilities: List[str]
    last_activity: Optional[float] = None


class ExecutionResponse(BaseModel):
    """Response model for agent execution."""
    request_id: str
    agent_id: str
    status: str
    content: str
    artifacts: List[str] = []
    execution_time: float
    timestamp: float
    error_message: Optional[str] = None


def create_agent_router(
    agent_registry: AgentRegistry, 
    task_orchestrator: TaskOrchestrator
) -> APIRouter:
    """Create agent router with dependencies."""
    
    router = APIRouter()
    
    @router.get("/", response_model=List[AgentStatusResponse])
    async def list_agents(
        context: SecurityContext = Depends(require_permission(Permission.AGENT_LIST))
    ):
        """List all available agents."""
        try:
            agents = agent_registry.get_all_agents()
            
            return [
                AgentStatusResponse(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    type=agent.agent_type.value,
                    status=agent.status.value,
                    capabilities=[cap.name for cap in agent.get_capabilities()],
                    last_activity=time.time()  # TODO: Track actual last activity
                )
                for agent in agents
            ]
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list agents: {str(e)}"
            )
    
    @router.get("/active", response_model=List[AgentStatusResponse])
    async def list_active_agents(
        context: SecurityContext = Depends(require_permission(Permission.AGENT_LIST))
    ):
        """List all active agents."""
        try:
            agents = agent_registry.get_active_agents()
            
            return [
                AgentStatusResponse(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    type=agent.agent_type.value,
                    status=agent.status.value,
                    capabilities=[cap.name for cap in agent.get_capabilities()],
                    last_activity=time.time()
                )
                for agent in agents
            ]
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list active agents: {str(e)}"
            )
    
    @router.get("/capabilities", response_model=List[AgentCapability])
    async def list_capabilities(
        context: SecurityContext = Depends(require_permission(Permission.AGENT_LIST))
    ):
        """List all available agent capabilities."""
        try:
            capabilities = agent_registry.get_capabilities()
            return capabilities
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list capabilities: {str(e)}"
            )
    
    @router.get("/{agent_id}", response_model=AgentStatusResponse)
    async def get_agent(
        agent_id: str,
        context: SecurityContext = Depends(require_permission(Permission.AGENT_READ))
    ):
        """Get information about a specific agent."""
        try:
            agent = agent_registry.get_agent(agent_id)
            
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent with ID {agent_id} not found"
                )
            
            return AgentStatusResponse(
                agent_id=agent.agent_id,
                name=agent.name,
                type=agent.agent_type.value,
                status=agent.status.value,
                capabilities=[cap.name for cap in agent.get_capabilities()],
                last_activity=time.time()
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get agent information: {str(e)}"
            )
    
    @router.post("/execute", response_model=ExecutionResponse)
    async def execute_agent(
        request: AgentExecutionRequest,
        http_request: Request,
        context: SecurityContext = Depends(require_agent_execution)
    ):
        """Execute an agent with the given prompt."""
        try:
            # Create agent request
            agent_request = AgentRequest(
                id=str(uuid4()),
                user_id=context.user_id,
                agent_id=request.agent_id or "",
                prompt=request.prompt,
                context={
                    **request.context,
                    "agent_type": request.agent_type.value if request.agent_type else None,
                    "timeout_seconds": request.timeout_seconds,
                    "user_role": context.role.value,
                    "session_id": context.session_id
                },
                priority=request.priority,
                created_at=datetime.utcnow()
            )
            
            # Log agent execution request
            await audit_logger.log(
                action=AuditAction.AGENT_EXECUTE,
                resource_type="agent",
                resource_id=request.agent_id or "auto-routed",
                user_id=context.user_id,
                session_id=context.session_id,
                ip_address=context.ip_address,
                details={
                    "prompt_length": len(request.prompt),
                    "agent_type": request.agent_type.value if request.agent_type else None,
                    "priority": request.priority,
                    "has_context": bool(request.context)
                },
                success=True
            )
            
            # Route and execute request
            start_time = time.time()
            agent_response = await agent_registry.route_request(agent_request)
            execution_time = time.time() - start_time
            
            # Log successful execution
            await audit_logger.log(
                action=AuditAction.AGENT_COMPLETE,
                resource_type="agent",
                resource_id=agent_response.id,
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "request_id": agent_request.id,
                    "execution_time": execution_time,
                    "response_status": agent_response.status.value,
                    "artifacts_count": len(agent_response.artifacts)
                },
                success=agent_response.status == ResponseStatus.SUCCESS
            )
            
            return ExecutionResponse(
                request_id=agent_request.id,
                agent_id=agent_response.id,
                status=agent_response.status.value,
                content=agent_response.content,
                artifacts=agent_response.artifacts,
                execution_time=execution_time,
                timestamp=time.time(),
                error_message=agent_response.error_message
            )
            
        except HTTPException:
            raise
        except Exception as e:
            # Log execution error
            await audit_logger.log(
                action=AuditAction.AGENT_EXECUTE,
                resource_type="agent",
                resource_id=request.agent_id or "unknown",
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "prompt_length": len(request.prompt),
                    "error": str(e)
                },
                success=False,
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent execution failed: {str(e)}"
            )
    
    @router.post("/workflow", response_model=List[ExecutionResponse])
    async def execute_workflow(
        request: WorkflowExecutionRequest,
        context: SecurityContext = Depends(require_agent_execution)
    ):
        """Execute a multi-agent workflow."""
        try:
            # Log workflow execution start
            await audit_logger.log(
                action=AuditAction.WORKFLOW_START,
                resource_type="workflow",
                resource_id=request.workflow_name,
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "workflow_name": request.workflow_name,
                    "steps_count": len(request.steps),
                    "continue_on_error": request.continue_on_error
                },
                success=True
            )
            
            # Execute workflow
            start_time = time.time()
            workflow_context = {
                **request.context,
                "user_id": context.user_id,
                "session_id": context.session_id,
                "user_role": context.role.value
            }
            
            # Add continue_on_error to each step if specified
            workflow_steps = []
            for step in request.steps:
                step_config = step.copy()
                if request.continue_on_error:
                    step_config["continue_on_error"] = True
                workflow_steps.append(step_config)
            
            responses = await task_orchestrator.execute_workflow(
                workflow_steps, workflow_context
            )
            execution_time = time.time() - start_time
            
            # Convert responses to API format
            execution_responses = []
            for response in responses:
                execution_responses.append(
                    ExecutionResponse(
                        request_id=response.request_id,
                        agent_id=response.id,
                        status=response.status.value,
                        content=response.content,
                        artifacts=response.artifacts,
                        execution_time=response.execution_time,
                        timestamp=time.time(),
                        error_message=response.error_message
                    )
                )
            
            # Log workflow completion
            await audit_logger.log(
                action=AuditAction.WORKFLOW_COMPLETE,
                resource_type="workflow",
                resource_id=request.workflow_name,
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "workflow_name": request.workflow_name,
                    "total_execution_time": execution_time,
                    "steps_completed": len(responses),
                    "successful_steps": sum(1 for r in responses if r.status == ResponseStatus.SUCCESS)
                },
                success=True
            )
            
            return execution_responses
            
        except HTTPException:
            raise
        except Exception as e:
            # Log workflow error
            await audit_logger.log(
                action=AuditAction.WORKFLOW_ERROR,
                resource_type="workflow",
                resource_id=request.workflow_name,
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "workflow_name": request.workflow_name,
                    "error": str(e)
                },
                success=False,
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Workflow execution failed: {str(e)}"
            )
    
    @router.get("/types", response_model=List[str])
    async def list_agent_types(
        context: SecurityContext = Depends(require_permission(Permission.AGENT_LIST))
    ):
        """List all available agent types."""
        return [agent_type.value for agent_type in AgentType]
    
    @router.get("/registry/status")
    async def get_registry_status(
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_HEALTH))
    ):
        """Get agent registry status information."""
        try:
            status_info = agent_registry.get_registry_status()
            return {
                "timestamp": time.time(),
                "registry_status": status_info
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get registry status: {str(e)}"
            )
    
    @router.post("/{agent_id}/health")
    async def check_agent_health(
        agent_id: str,
        context: SecurityContext = Depends(require_permission(Permission.AGENT_READ))
    ):
        """Check health of a specific agent."""
        try:
            agent = agent_registry.get_agent(agent_id)
            
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent with ID {agent_id} not found"
                )
            
            is_healthy = await agent.health_check()
            
            return {
                "agent_id": agent_id,
                "healthy": is_healthy,
                "status": agent.status.value,
                "timestamp": time.time()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Health check failed: {str(e)}"
            )
    
    return router

# Create the router instance
from ...core.registry import get_agent_registry
from ...core.orchestrator import TaskOrchestrator

agent_registry = get_agent_registry()
task_orchestrator = TaskOrchestrator(agent_registry)
router = create_agent_router(agent_registry, task_orchestrator)