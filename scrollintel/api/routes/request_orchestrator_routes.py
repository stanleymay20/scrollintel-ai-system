"""
API routes for Request Orchestrator functionality.

Provides endpoints for workflow management, task submission, status tracking,
and system monitoring.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from scrollintel.core.request_orchestrator import (
    RequestOrchestrator,
    TaskDefinition,
    WorkflowDefinition,
    TaskPriority,
    ResourceType,
    ResourceRequirement,
    create_image_generation_workflow,
    create_video_generation_workflow
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/orchestrator", tags=["Request Orchestrator"])

# Global orchestrator instance (in production, this would be dependency injected)
orchestrator = RequestOrchestrator(max_concurrent_tasks=10)


# Pydantic models for API

class ResourceRequirementModel(BaseModel):
    """Resource requirement specification."""
    resource_type: str = Field(..., description="Type of resource (gpu, cpu, memory, storage, network)")
    amount: float = Field(..., gt=0, description="Amount of resource required")
    unit: str = Field(default="units", description="Unit of measurement")


class TaskDefinitionModel(BaseModel):
    """Task definition for API."""
    name: str = Field(..., description="Human-readable task name")
    handler_name: str = Field(..., description="Name of the handler function")
    args: List[Any] = Field(default_factory=list, description="Positional arguments for handler")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments for handler")
    priority: str = Field(default="NORMAL", description="Task priority (LOW, NORMAL, HIGH, URGENT)")
    resource_requirements: List[ResourceRequirementModel] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list, description="List of task IDs this task depends on")
    timeout: Optional[float] = Field(None, gt=0, description="Task timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retry attempts")


class WorkflowDefinitionModel(BaseModel):
    """Workflow definition for API."""
    name: str = Field(..., description="Human-readable workflow name")
    tasks: List[TaskDefinitionModel] = Field(..., description="List of tasks in the workflow")
    description: Optional[str] = Field(None, description="Workflow description")


class BatchImageGenerationRequest(BaseModel):
    """Request for batch image generation workflow."""
    prompts: List[str] = Field(..., min_items=1, description="List of text prompts")
    model_preferences: Optional[List[str]] = Field(None, description="Preferred models for each prompt")
    priority: str = Field(default="NORMAL", description="Task priority")


class VideoGenerationRequest(BaseModel):
    """Request for video generation workflow."""
    prompt: str = Field(..., description="Text prompt for video generation")
    duration: float = Field(default=5.0, gt=0, le=60, description="Video duration in seconds")
    resolution: List[int] = Field(default=[1280, 720], description="Video resolution [width, height]")
    priority: str = Field(default="HIGH", description="Task priority")


class TaskStatusResponse(BaseModel):
    """Task status response."""
    id: str
    name: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    duration: Optional[str]
    error: Optional[str]
    allocated_resources: Dict[str, float]


class WorkflowStatusResponse(BaseModel):
    """Workflow status response."""
    progress: float
    status: str
    total_tasks: Optional[int] = None
    completed_tasks: Optional[int] = None
    failed_tasks: Optional[int] = None
    running_tasks: Optional[int] = None
    tasks: Dict[str, Dict[str, Any]]


class SystemStatusResponse(BaseModel):
    """System status response."""
    running_tasks: int
    max_concurrent_tasks: int
    queue_status: Dict[str, int]
    resource_utilization: Dict[str, str]
    shutdown: bool


# Dependency functions

async def get_orchestrator() -> RequestOrchestrator:
    """Get the orchestrator instance."""
    return orchestrator


def parse_priority(priority_str: str) -> TaskPriority:
    """Parse priority string to TaskPriority enum."""
    try:
        return TaskPriority[priority_str.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid priority: {priority_str}")


def parse_resource_type(resource_str: str) -> ResourceType:
    """Parse resource type string to ResourceType enum."""
    try:
        return ResourceType[resource_str.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid resource type: {resource_str}")


# Handler registry (in production, this would be more sophisticated)
HANDLER_REGISTRY = {
    "image_generation": lambda prompt, model=None: {"type": "image", "prompt": prompt, "model": model},
    "video_generation": lambda prompt, duration=5.0: {"type": "video", "prompt": prompt, "duration": duration},
    "image_enhancement": lambda image_path: {"type": "enhancement", "image": image_path},
    "style_transfer": lambda image_path, style: {"type": "style_transfer", "image": image_path, "style": style}
}


def get_handler(handler_name: str):
    """Get handler function by name."""
    if handler_name not in HANDLER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown handler: {handler_name}")
    return HANDLER_REGISTRY[handler_name]


# API Routes

@router.on_event("startup")
async def startup_orchestrator():
    """Start the orchestrator on application startup."""
    await orchestrator.start()
    logger.info("Request orchestrator started")


@router.on_event("shutdown")
async def shutdown_orchestrator():
    """Stop the orchestrator on application shutdown."""
    await orchestrator.stop()
    logger.info("Request orchestrator stopped")


@router.post("/tasks", response_model=Dict[str, str])
async def submit_task(
    task_def: TaskDefinitionModel,
    orchestrator: RequestOrchestrator = Depends(get_orchestrator)
) -> Dict[str, str]:
    """Submit a single task for execution."""
    try:
        # Convert API model to internal model
        priority = parse_priority(task_def.priority)
        
        resource_requirements = []
        for req in task_def.resource_requirements:
            resource_type = parse_resource_type(req.resource_type)
            resource_requirements.append(ResourceRequirement(
                resource_type=resource_type,
                amount=req.amount,
                unit=req.unit
            ))
        
        handler = get_handler(task_def.handler_name)
        
        task = TaskDefinition(
            name=task_def.name,
            handler=handler,
            args=tuple(task_def.args),
            kwargs=task_def.kwargs,
            priority=priority,
            resource_requirements=resource_requirements,
            dependencies=task_def.dependencies,
            timeout=task_def.timeout,
            max_retries=task_def.max_retries
        )
        
        task_id = await orchestrator.submit_task(task)
        
        return {"task_id": task_id, "status": "submitted"}
        
    except Exception as e:
        logger.error(f"Error submitting task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflows", response_model=Dict[str, str])
async def submit_workflow(
    workflow_def: WorkflowDefinitionModel,
    orchestrator: RequestOrchestrator = Depends(get_orchestrator)
) -> Dict[str, str]:
    """Submit a complete workflow for execution."""
    try:
        # Convert API model to internal model
        tasks = []
        for task_def in workflow_def.tasks:
            priority = parse_priority(task_def.priority)
            
            resource_requirements = []
            for req in task_def.resource_requirements:
                resource_type = parse_resource_type(req.resource_type)
                resource_requirements.append(ResourceRequirement(
                    resource_type=resource_type,
                    amount=req.amount,
                    unit=req.unit
                ))
            
            handler = get_handler(task_def.handler_name)
            
            task = TaskDefinition(
                name=task_def.name,
                handler=handler,
                args=tuple(task_def.args),
                kwargs=task_def.kwargs,
                priority=priority,
                resource_requirements=resource_requirements,
                dependencies=task_def.dependencies,
                timeout=task_def.timeout,
                max_retries=task_def.max_retries
            )
            tasks.append(task)
        
        workflow = WorkflowDefinition(
            name=workflow_def.name,
            tasks=tasks,
            description=workflow_def.description
        )
        
        workflow_id = await orchestrator.submit_workflow(workflow)
        
        return {"workflow_id": workflow_id, "status": "submitted"}
        
    except Exception as e:
        logger.error(f"Error submitting workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflows/batch-images", response_model=Dict[str, str])
async def submit_batch_image_generation(
    request: BatchImageGenerationRequest,
    orchestrator: RequestOrchestrator = Depends(get_orchestrator)
) -> Dict[str, str]:
    """Submit a batch image generation workflow."""
    try:
        priority = parse_priority(request.priority)
        
        workflow = await create_image_generation_workflow(
            prompts=request.prompts,
            model_preferences=request.model_preferences,
            priority=priority
        )
        
        workflow_id = await orchestrator.submit_workflow(workflow)
        
        return {"workflow_id": workflow_id, "status": "submitted", "image_count": len(request.prompts)}
        
    except Exception as e:
        logger.error(f"Error submitting batch image generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflows/video-generation", response_model=Dict[str, str])
async def submit_video_generation(
    request: VideoGenerationRequest,
    orchestrator: RequestOrchestrator = Depends(get_orchestrator)
) -> Dict[str, str]:
    """Submit a video generation workflow."""
    try:
        priority = parse_priority(request.priority)
        
        workflow = await create_video_generation_workflow(
            prompt=request.prompt,
            duration=request.duration,
            resolution=tuple(request.resolution),
            priority=priority
        )
        
        workflow_id = await orchestrator.submit_workflow(workflow)
        
        return {"workflow_id": workflow_id, "status": "submitted"}
        
    except Exception as e:
        logger.error(f"Error submitting video generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    orchestrator: RequestOrchestrator = Depends(get_orchestrator)
) -> TaskStatusResponse:
    """Get the status of a specific task."""
    try:
        status = await orchestrator.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return TaskStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    workflow_id: str,
    orchestrator: RequestOrchestrator = Depends(get_orchestrator)
) -> WorkflowStatusResponse:
    """Get the status of a specific workflow."""
    try:
        status = await orchestrator.get_workflow_status(workflow_id)
        return WorkflowStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    orchestrator: RequestOrchestrator = Depends(get_orchestrator)
) -> Dict[str, str]:
    """Cancel a specific task."""
    try:
        success = await orchestrator.cancel_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
        
        return {"task_id": task_id, "status": "cancelled"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status(
    orchestrator: RequestOrchestrator = Depends(get_orchestrator)
) -> SystemStatusResponse:
    """Get overall system status."""
    try:
        status = await orchestrator.get_system_status()
        return SystemStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.get("/handlers")
async def list_handlers() -> Dict[str, List[str]]:
    """List available task handlers."""
    return {"handlers": list(HANDLER_REGISTRY.keys())}