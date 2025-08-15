"""
API routes for ScrollIntel orchestration and workflow management.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from ...core.orchestrator import TaskOrchestrator, WorkflowStatus
from ...core.workflow_templates import WorkflowTemplateLibrary
from ...core.registry import AgentRegistry
from ...core.message_bus import get_message_bus, MessageType, MessagePriority
from ...security.auth import get_current_user
from ...security.permissions import require_permission

router = APIRouter(prefix="/orchestration", tags=["orchestration"])


# Pydantic models for API
class WorkflowTaskRequest(BaseModel):
    """Request model for workflow task."""
    name: str
    agent_type: Optional[str] = None
    agent_id: Optional[str] = None
    prompt: str
    context: Dict[str, Any] = {}
    dependencies: List[Dict[str, Any]] = []
    timeout: float = 300.0
    max_retries: int = 3
    continue_on_error: bool = False
    priority: int = 1


class CreateWorkflowRequest(BaseModel):
    """Request model for creating a workflow."""
    name: str
    description: str = ""
    tasks: List[WorkflowTaskRequest]
    context: Dict[str, Any] = {}


class CreateWorkflowFromTemplateRequest(BaseModel):
    """Request model for creating workflow from template."""
    template_id: str
    name: str
    context: Dict[str, Any] = {}


class WorkflowResponse(BaseModel):
    """Response model for workflow."""
    id: str
    name: str
    description: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_by: Optional[str] = None
    error: Optional[str] = None
    tasks: List[Dict[str, Any]]


class WorkflowTemplateResponse(BaseModel):
    """Response model for workflow template."""
    id: str
    name: str
    description: str
    estimated_duration: Optional[float] = None
    tags: List[str]
    task_count: int


class MessageRequest(BaseModel):
    """Request model for sending messages."""
    recipient_id: str
    message_type: str = "request"
    payload: Dict[str, Any]
    priority: str = "normal"
    expires_in_seconds: Optional[int] = None


class BroadcastRequest(BaseModel):
    """Request model for broadcasting messages."""
    message_type: str = "event"
    payload: Dict[str, Any]
    exclude_agents: List[str] = []
    priority: str = "normal"


# Global orchestrator instance (in production, this would be dependency injected)
_orchestrator: Optional[TaskOrchestrator] = None


def get_orchestrator() -> TaskOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        # This would be properly initialized in the application startup
        from ...core.registry import AgentRegistry
        registry = AgentRegistry()
        _orchestrator = TaskOrchestrator(registry)
    return _orchestrator


@router.get("/templates", response_model=List[WorkflowTemplateResponse])
async def get_workflow_templates(
    current_user: Dict = Depends(get_current_user)
):
    """Get all available workflow templates."""
    orchestrator = get_orchestrator()
    templates = orchestrator.get_workflow_templates()
    
    return [
        WorkflowTemplateResponse(
            id=template["id"],
            name=template["name"],
            description=template["description"],
            estimated_duration=template.get("estimated_duration"),
            tags=template.get("tags", []),
            task_count=template.get("task_count", 0),
        )
        for template in templates
    ]


@router.post("/workflows/from-template", response_model=Dict[str, str])
async def create_workflow_from_template(
    request: CreateWorkflowFromTemplateRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user),
):
    """Create a workflow from a template."""
    orchestrator = get_orchestrator()
    
    try:
        workflow_id = await orchestrator.create_workflow_from_template(
            template_id=request.template_id,
            name=request.name,
            context=request.context,
            created_by=current_user.get("user_id"),
        )
        
        return {"workflow_id": workflow_id, "status": "created"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/workflows/custom", response_model=Dict[str, str])
async def create_custom_workflow(
    request: CreateWorkflowRequest,
    current_user: Dict = Depends(get_current_user),
):
    """Create a custom workflow."""
    orchestrator = get_orchestrator()
    
    try:
        # Convert request tasks to dict format
        tasks = [
            {
                "name": task.name,
                "agent_type": task.agent_type,
                "agent_id": task.agent_id,
                "prompt": task.prompt,
                "context": task.context,
                "dependencies": task.dependencies,
                "timeout": task.timeout,
                "max_retries": task.max_retries,
                "continue_on_error": task.continue_on_error,
                "priority": task.priority,
            }
            for task in request.tasks
        ]
        
        workflow_id = await orchestrator.create_custom_workflow(
            name=request.name,
            tasks=tasks,
            context=request.context,
            description=request.description,
            created_by=current_user.get("user_id"),
        )
        
        return {"workflow_id": workflow_id, "status": "created"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user),
):
    """Execute a workflow."""
    orchestrator = get_orchestrator()
    
    # Check if workflow exists
    workflow_status = orchestrator.get_workflow_status(workflow_id)
    if not workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Execute workflow in background
    background_tasks.add_task(_execute_workflow_background, workflow_id)
    
    return {"workflow_id": workflow_id, "status": "execution_started"}


async def _execute_workflow_background(workflow_id: str):
    """Execute workflow in background."""
    orchestrator = get_orchestrator()
    try:
        await orchestrator.execute_workflow(workflow_id)
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Workflow {workflow_id} execution failed: {e}")


@router.get("/workflows", response_model=List[WorkflowResponse])
async def get_workflows(
    status: Optional[str] = None,
    current_user: Dict = Depends(get_current_user),
):
    """Get all workflows, optionally filtered by status."""
    orchestrator = get_orchestrator()
    
    if status == "active":
        workflows = orchestrator.get_active_workflows()
    else:
        workflows = orchestrator.get_all_workflows()
    
    # Filter by status if specified
    if status and status != "active":
        workflows = [w for w in workflows if w["status"] == status]
    
    return [
        WorkflowResponse(
            id=workflow["id"],
            name=workflow["name"],
            description=workflow["description"],
            status=workflow["status"],
            progress=workflow["progress"],
            created_at=workflow["created_at"],
            started_at=workflow["started_at"],
            completed_at=workflow["completed_at"],
            created_by=workflow["created_by"],
            error=workflow["error"],
            tasks=workflow["tasks"],
        )
        for workflow in workflows
    ]


@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow_status(
    workflow_id: str,
    current_user: Dict = Depends(get_current_user),
):
    """Get the status of a specific workflow."""
    orchestrator = get_orchestrator()
    workflow = orchestrator.get_workflow_status(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return WorkflowResponse(
        id=workflow["id"],
        name=workflow["name"],
        description=workflow["description"],
        status=workflow["status"],
        progress=workflow["progress"],
        created_at=workflow["created_at"],
        started_at=workflow["started_at"],
        completed_at=workflow["completed_at"],
        created_by=workflow["created_by"],
        error=workflow["error"],
        tasks=workflow["tasks"],
    )


@router.post("/workflows/{workflow_id}/pause")
async def pause_workflow(
    workflow_id: str,
    current_user: Dict = Depends(get_current_user),
):
    """Pause a running workflow."""
    orchestrator = get_orchestrator()
    
    try:
        await orchestrator.pause_workflow(workflow_id)
        return {"workflow_id": workflow_id, "status": "paused"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/workflows/{workflow_id}/resume")
async def resume_workflow(
    workflow_id: str,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user),
):
    """Resume a paused workflow."""
    orchestrator = get_orchestrator()
    
    try:
        # Resume workflow in background
        background_tasks.add_task(_resume_workflow_background, workflow_id)
        return {"workflow_id": workflow_id, "status": "resuming"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _resume_workflow_background(workflow_id: str):
    """Resume workflow in background."""
    orchestrator = get_orchestrator()
    try:
        await orchestrator.resume_workflow(workflow_id)
    except Exception as e:
        print(f"Workflow {workflow_id} resume failed: {e}")


@router.post("/workflows/{workflow_id}/cancel")
async def cancel_workflow(
    workflow_id: str,
    current_user: Dict = Depends(get_current_user),
):
    """Cancel a workflow."""
    orchestrator = get_orchestrator()
    
    try:
        await orchestrator.cancel_workflow(workflow_id)
        return {"workflow_id": workflow_id, "status": "cancelled"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/workflows/cleanup")
async def cleanup_workflows(
    older_than_hours: int = 24,
    current_user: Dict = Depends(get_current_user),
):
    """Clean up completed workflows older than specified hours."""
    # Require admin permission for cleanup
    require_permission(current_user, "admin", "cleanup")
    
    orchestrator = get_orchestrator()
    
    try:
        cleaned_count = await orchestrator.cleanup_completed_workflows(older_than_hours)
        return {"cleaned_workflows": cleaned_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/messages/send")
async def send_message(
    request: MessageRequest,
    current_user: Dict = Depends(get_current_user),
):
    """Send a message to an agent."""
    message_bus = get_message_bus()
    
    try:
        # Convert string priority to enum
        priority_map = {
            "low": MessagePriority.LOW,
            "normal": MessagePriority.NORMAL,
            "high": MessagePriority.HIGH,
            "critical": MessagePriority.CRITICAL,
        }
        priority = priority_map.get(request.priority.lower(), MessagePriority.NORMAL)
        
        # Convert string message type to enum
        message_type_map = {
            "request": MessageType.REQUEST,
            "event": MessageType.EVENT,
            "broadcast": MessageType.BROADCAST,
            "coordination": MessageType.COORDINATION,
        }
        message_type = message_type_map.get(request.message_type.lower(), MessageType.REQUEST)
        
        message_id = await message_bus.send_message(
            sender_id=current_user.get("user_id", "api_user"),
            recipient_id=request.recipient_id,
            message_type=message_type,
            payload=request.payload,
            priority=priority,
            expires_in_seconds=request.expires_in_seconds,
        )
        
        return {"message_id": message_id, "status": "sent"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/messages/broadcast")
async def broadcast_message(
    request: BroadcastRequest,
    current_user: Dict = Depends(get_current_user),
):
    """Broadcast a message to all agents."""
    message_bus = get_message_bus()
    
    try:
        # Convert string priority to enum
        priority_map = {
            "low": MessagePriority.LOW,
            "normal": MessagePriority.NORMAL,
            "high": MessagePriority.HIGH,
            "critical": MessagePriority.CRITICAL,
        }
        priority = priority_map.get(request.priority.lower(), MessagePriority.NORMAL)
        
        # Convert string message type to enum
        message_type_map = {
            "event": MessageType.EVENT,
            "broadcast": MessageType.BROADCAST,
            "coordination": MessageType.COORDINATION,
        }
        message_type = message_type_map.get(request.message_type.lower(), MessageType.EVENT)
        
        message_ids = await message_bus.broadcast_message(
            sender_id=current_user.get("user_id", "api_user"),
            message_type=message_type,
            payload=request.payload,
            exclude_agents=request.exclude_agents,
            priority=priority,
        )
        
        return {"message_ids": message_ids, "recipients": len(message_ids)}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/messages/stats")
async def get_message_stats(
    current_user: Dict = Depends(get_current_user),
):
    """Get message bus statistics."""
    message_bus = get_message_bus()
    return message_bus.get_stats()


@router.get("/messages/history")
async def get_message_history(
    limit: int = 100,
    current_user: Dict = Depends(get_current_user),
):
    """Get recent message history."""
    message_bus = get_message_bus()
    return message_bus.get_message_history(limit)


@router.get("/messages/pending")
async def get_pending_messages(
    current_user: Dict = Depends(get_current_user),
):
    """Get all pending messages."""
    # Require admin permission to view pending messages
    require_permission(current_user, "admin", "view_messages")
    
    message_bus = get_message_bus()
    return message_bus.get_pending_messages()


@router.get("/health")
async def orchestration_health():
    """Get orchestration system health status."""
    try:
        message_bus = get_message_bus()
        orchestrator = get_orchestrator()
        
        message_stats = message_bus.get_stats()
        active_workflows = len(orchestrator.get_active_workflows())
        
        return {
            "status": "healthy",
            "message_bus": {
                "running": message_stats["is_running"],
                "pending_messages": message_stats["pending_messages"],
                "subscribers": message_stats["subscribers"],
            },
            "orchestrator": {
                "active_workflows": active_workflows,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }