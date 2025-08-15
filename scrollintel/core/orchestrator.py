"""
Task Orchestrator for ScrollIntel Multi-Agent Workflows.
Manages complex task coordination, dependency management, and workflow execution.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from uuid import uuid4
import asyncio
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass, field

from .interfaces import (
    BaseAgent,
    AgentRequest,
    AgentResponse,
    AgentError,
    ResponseStatus,
)
from .registry import AgentRegistry
from ..agents.proxy import AgentMessage

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a task in the orchestrator."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_DEPENDENCY = "waiting_dependency"


class WorkflowStatus(str, Enum):
    """Status of a workflow."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class TaskDependency:
    """Represents a dependency between tasks."""
    task_id: str
    dependency_type: str = "completion"  # completion, data, condition
    condition: Optional[Callable] = None
    data_key: Optional[str] = None


@dataclass
class WorkflowTask:
    """Represents a single task in a workflow."""
    id: str
    name: str
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    prompt: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[TaskDependency] = field(default_factory=list)
    timeout: float = 300.0  # 5 minutes default
    retry_count: int = 0
    max_retries: int = 3
    continue_on_error: bool = False
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[AgentResponse] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None


@dataclass
class WorkflowTemplate:
    """Template for common multi-agent workflows."""
    id: str
    name: str
    description: str
    tasks: List[Dict[str, Any]]
    default_context: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: Optional[float] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class Workflow:
    """Represents a complete workflow with multiple tasks."""
    id: str
    name: str
    description: str = ""
    tasks: List[WorkflowTask] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None


class TaskOrchestrator:
    """Advanced orchestrator for managing multi-agent workflows."""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._workflows: Dict[str, Workflow] = {}
        self._workflow_templates: Dict[str, WorkflowTemplate] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_results: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._progress_callbacks: Dict[str, List[Callable]] = {}
        
        # Initialize with common workflow templates
        self._initialize_templates()
    
    def _initialize_templates(self) -> None:
        """Initialize common workflow templates."""
        # Import here to avoid circular imports
        from .workflow_templates import WorkflowTemplateLibrary
        
        # Load all templates from the library
        all_templates = WorkflowTemplateLibrary.get_all_templates()
        self._workflow_templates = all_templates
    
    async def create_workflow_from_template(
        self,
        template_id: str,
        name: str,
        context: Dict[str, Any] = None,
        created_by: Optional[str] = None,
    ) -> str:
        """Create a workflow from a template."""
        template = self._workflow_templates.get(template_id)
        if not template:
            raise AgentError(f"Workflow template {template_id} not found")
        
        workflow_id = str(uuid4())
        context = context or {}
        
        # Merge template context with provided context
        merged_context = {**template.default_context, **context}
        
        # Create workflow tasks from template
        tasks = []
        for i, task_template in enumerate(template.tasks):
            task = WorkflowTask(
                id=str(i),
                name=task_template["name"],
                agent_type=task_template.get("agent_type"),
                agent_id=task_template.get("agent_id"),
                prompt=task_template["prompt"],
                context={**merged_context, **task_template.get("context", {})},
                dependencies=[
                    TaskDependency(**dep) for dep in task_template.get("dependencies", [])
                ],
                timeout=task_template.get("timeout", 300.0),
                max_retries=task_template.get("max_retries", 3),
                continue_on_error=task_template.get("continue_on_error", False),
                priority=task_template.get("priority", 1),
            )
            tasks.append(task)
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=template.description,
            tasks=tasks,
            context=merged_context,
            created_by=created_by,
        )
        
        async with self._lock:
            self._workflows[workflow_id] = workflow
        
        logger.info(f"Created workflow {workflow_id} from template {template_id}")
        return workflow_id
    
    async def create_custom_workflow(
        self,
        name: str,
        tasks: List[Dict[str, Any]],
        context: Dict[str, Any] = None,
        description: str = "",
        created_by: Optional[str] = None,
    ) -> str:
        """Create a custom workflow with specified tasks."""
        workflow_id = str(uuid4())
        context = context or {}
        
        # Create workflow tasks
        workflow_tasks = []
        for i, task_data in enumerate(tasks):
            task = WorkflowTask(
                id=str(i),
                name=task_data["name"],
                agent_type=task_data.get("agent_type"),
                agent_id=task_data.get("agent_id"),
                prompt=task_data["prompt"],
                context={**context, **task_data.get("context", {})},
                dependencies=[
                    TaskDependency(**dep) for dep in task_data.get("dependencies", [])
                ],
                timeout=task_data.get("timeout", 300.0),
                max_retries=task_data.get("max_retries", 3),
                continue_on_error=task_data.get("continue_on_error", False),
                priority=task_data.get("priority", 1),
            )
            workflow_tasks.append(task)
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            tasks=workflow_tasks,
            context=context,
            created_by=created_by,
        )
        
        async with self._lock:
            self._workflows[workflow_id] = workflow
        
        logger.info(f"Created custom workflow {workflow_id}")
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow with dependency management and progress tracking."""
        async with self._lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                raise AgentError(f"Workflow {workflow_id} not found")
            
            if workflow.status != WorkflowStatus.CREATED:
                raise AgentError(f"Workflow {workflow_id} is not in created state")
            
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.utcnow()
        
        try:
            # Execute tasks with dependency management
            await self._execute_workflow_tasks(workflow)
            
            # Update workflow status
            async with self._lock:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.completed_at = datetime.utcnow()
                workflow.progress = 100.0
            
            logger.info(f"Workflow {workflow_id} completed successfully")
            
            return {
                "workflow_id": workflow_id,
                "status": workflow.status,
                "progress": workflow.progress,
                "results": [task.result.dict() if task.result else None for task in workflow.tasks],
                "execution_time": (workflow.completed_at - workflow.started_at).total_seconds(),
            }
            
        except Exception as e:
            async with self._lock:
                workflow.status = WorkflowStatus.FAILED
                workflow.error = str(e)
                workflow.completed_at = datetime.utcnow()
            
            logger.error(f"Workflow {workflow_id} failed: {e}")
            raise
    
    async def _execute_workflow_tasks(self, workflow: Workflow) -> None:
        """Execute workflow tasks with dependency management."""
        completed_tasks = set()
        running_tasks = {}
        
        while len(completed_tasks) < len(workflow.tasks):
            # Find tasks that can be executed (dependencies satisfied)
            ready_tasks = []
            for task in workflow.tasks:
                if (task.id not in completed_tasks and 
                    task.id not in running_tasks and
                    task.status == TaskStatus.PENDING):
                    
                    if self._are_dependencies_satisfied(task, completed_tasks, workflow):
                        ready_tasks.append(task)
            
            if not ready_tasks and not running_tasks:
                # No tasks ready and none running - deadlock or all failed
                remaining_tasks = [t for t in workflow.tasks if t.id not in completed_tasks]
                raise AgentError(f"Workflow deadlock: cannot execute remaining tasks {[t.name for t in remaining_tasks]}")
            
            # Start ready tasks
            for task in ready_tasks:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                
                # Create and start task execution
                task_coroutine = self._execute_single_task(task, workflow)
                running_tasks[task.id] = asyncio.create_task(task_coroutine)
                
                logger.info(f"Started task {task.name} in workflow {workflow.id}")
            
            # Wait for at least one task to complete
            if running_tasks:
                done, pending = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task_future in done:
                    task_id = None
                    for tid, future in running_tasks.items():
                        if future == task_future:
                            task_id = tid
                            break
                    
                    if task_id:
                        task = next(t for t in workflow.tasks if t.id == task_id)
                        
                        try:
                            result = await task_future
                            task.result = result
                            task.status = TaskStatus.COMPLETED
                            completed_tasks.add(task_id)
                            
                            # Store task result for other tasks to use
                            self._task_results[f"{workflow.id}_{task_id}"] = result
                            
                            logger.info(f"Task {task.name} completed successfully")
                            
                        except Exception as e:
                            task.status = TaskStatus.FAILED
                            task.error = str(e)
                            
                            if not task.continue_on_error:
                                # Cancel all pending tasks
                                for pending_task in pending:
                                    pending_task.cancel()
                                raise AgentError(f"Task {task.name} failed: {e}")
                            else:
                                completed_tasks.add(task_id)
                                logger.warning(f"Task {task.name} failed but continuing: {e}")
                        
                        finally:
                            task.completed_at = datetime.utcnow()
                            if task.started_at:
                                task.execution_time = (task.completed_at - task.started_at).total_seconds()
                            del running_tasks[task_id]
                
                # Update workflow progress
                progress = len(completed_tasks) / len(workflow.tasks) * 100
                workflow.progress = progress
                
                # Notify progress callbacks
                await self._notify_progress_callbacks(workflow.id, progress)
    
    def _are_dependencies_satisfied(
        self,
        task: WorkflowTask,
        completed_tasks: set,
        workflow: Workflow,
    ) -> bool:
        """Check if all dependencies for a task are satisfied."""
        for dependency in task.dependencies:
            if dependency.dependency_type == "completion":
                if dependency.task_id not in completed_tasks:
                    return False
            
            elif dependency.dependency_type == "data":
                # Check if required data is available
                data_key = f"{workflow.id}_{dependency.task_id}"
                if data_key not in self._task_results:
                    return False
                
                if dependency.data_key:
                    result = self._task_results[data_key]
                    if not hasattr(result, dependency.data_key):
                        return False
            
            elif dependency.dependency_type == "condition":
                # Check custom condition
                if dependency.condition:
                    try:
                        if not dependency.condition(workflow, completed_tasks, self._task_results):
                            return False
                    except Exception as e:
                        logger.error(f"Error evaluating condition for task {task.name}: {e}")
                        return False
        
        return True
    
    async def _execute_single_task(self, task: WorkflowTask, workflow: Workflow) -> AgentResponse:
        """Execute a single task with retry logic."""
        last_error = None
        
        for attempt in range(task.max_retries + 1):
            try:
                # Create agent request
                request = AgentRequest(
                    id=str(uuid4()),
                    user_id=workflow.created_by or "system",
                    agent_id=task.agent_id or "",  # Provide empty string if None
                    prompt=task.prompt,
                    context={
                        **workflow.context,
                        **task.context,
                        "workflow_id": workflow.id,
                        "task_id": task.id,
                        "task_name": task.name,
                        "attempt": attempt + 1,
                        "previous_results": self._get_previous_results(task, workflow),
                    },
                    priority=task.priority,
                    created_at=datetime.utcnow(),
                )
                
                # Add agent type to context if specified
                if task.agent_type:
                    request.context["agent_type"] = task.agent_type
                
                # Execute with timeout
                response = await asyncio.wait_for(
                    self.registry.route_request(request),
                    timeout=task.timeout
                )
                
                if response.status == ResponseStatus.SUCCESS:
                    return response
                else:
                    last_error = response.error_message or "Task execution failed"
                    
            except asyncio.TimeoutError:
                last_error = f"Task timed out after {task.timeout} seconds"
            except Exception as e:
                last_error = str(e)
            
            # Update retry count
            task.retry_count = attempt + 1
            
            # Wait before retry (exponential backoff)
            if attempt < task.max_retries:
                wait_time = min(2 ** attempt, 30)  # Max 30 seconds
                await asyncio.sleep(wait_time)
                logger.warning(f"Retrying task {task.name} (attempt {attempt + 2})")
        
        # All retries exhausted
        raise AgentError(f"Task {task.name} failed after {task.max_retries + 1} attempts: {last_error}")
    
    def _get_previous_results(self, task: WorkflowTask, workflow: Workflow) -> List[Dict[str, Any]]:
        """Get results from previous tasks that this task depends on."""
        previous_results = []
        
        for dependency in task.dependencies:
            if dependency.dependency_type in ["completion", "data"]:
                result_key = f"{workflow.id}_{dependency.task_id}"
                if result_key in self._task_results:
                    result = self._task_results[result_key]
                    previous_results.append({
                        "task_id": dependency.task_id,
                        "result": result.dict() if hasattr(result, 'dict') else result,
                    })
        
        return previous_results
    
    async def pause_workflow(self, workflow_id: str) -> None:
        """Pause a running workflow."""
        async with self._lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                raise AgentError(f"Workflow {workflow_id} not found")
            
            if workflow.status != WorkflowStatus.RUNNING:
                raise AgentError(f"Workflow {workflow_id} is not running")
            
            workflow.status = WorkflowStatus.PAUSED
            
            # Cancel running tasks for this workflow
            tasks_to_cancel = [
                task for task_id, task in self._running_tasks.items()
                if task_id.startswith(workflow_id)
            ]
            
            for task in tasks_to_cancel:
                task.cancel()
        
        logger.info(f"Paused workflow {workflow_id}")
    
    async def resume_workflow(self, workflow_id: str) -> None:
        """Resume a paused workflow."""
        async with self._lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                raise AgentError(f"Workflow {workflow_id} not found")
            
            if workflow.status != WorkflowStatus.PAUSED:
                raise AgentError(f"Workflow {workflow_id} is not paused")
            
            workflow.status = WorkflowStatus.RUNNING
        
        # Continue execution
        await self._execute_workflow_tasks(workflow)
        logger.info(f"Resumed workflow {workflow_id}")
    
    async def cancel_workflow(self, workflow_id: str) -> None:
        """Cancel a workflow."""
        async with self._lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                raise AgentError(f"Workflow {workflow_id} not found")
            
            if workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.CANCELLED]:
                return
            
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.utcnow()
            
            # Cancel running tasks for this workflow
            tasks_to_cancel = [
                task for task_id, task in self._running_tasks.items()
                if task_id.startswith(workflow_id)
            ]
            
            for task in tasks_to_cancel:
                task.cancel()
        
        logger.info(f"Cancelled workflow {workflow_id}")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status,
            "progress": workflow.progress,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "created_by": workflow.created_by,
            "error": workflow.error,
            "tasks": [
                {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status,
                    "retry_count": task.retry_count,
                    "execution_time": task.execution_time,
                    "error": task.error,
                }
                for task in workflow.tasks
            ],
        }
    
    def get_all_workflows(self) -> List[Dict[str, Any]]:
        """Get all workflows."""
        return [self.get_workflow_status(workflow_id) for workflow_id in self._workflows.keys()]
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get all active workflows."""
        return [
            self.get_workflow_status(workflow_id)
            for workflow_id, workflow in self._workflows.items()
            if workflow.status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]
        ]
    
    def get_workflow_templates(self) -> List[Dict[str, Any]]:
        """Get all available workflow templates."""
        return [
            {
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "estimated_duration": template.estimated_duration,
                "tags": template.tags,
                "task_count": len(template.tasks),
            }
            for template in self._workflow_templates.values()
        ]
    
    def add_workflow_template(self, template: WorkflowTemplate) -> None:
        """Add a new workflow template."""
        self._workflow_templates[template.id] = template
        logger.info(f"Added workflow template {template.id}")
    
    def register_progress_callback(self, workflow_id: str, callback: Callable) -> None:
        """Register a callback for workflow progress updates."""
        if workflow_id not in self._progress_callbacks:
            self._progress_callbacks[workflow_id] = []
        self._progress_callbacks[workflow_id].append(callback)
    
    async def _notify_progress_callbacks(self, workflow_id: str, progress: float) -> None:
        """Notify all progress callbacks for a workflow."""
        callbacks = self._progress_callbacks.get(workflow_id, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(workflow_id, progress)
                else:
                    callback(workflow_id, progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    async def cleanup_completed_workflows(self, older_than_hours: int = 24) -> int:
        """Clean up completed workflows older than specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        workflows_to_remove = []
        
        async with self._lock:
            for workflow_id, workflow in self._workflows.items():
                if (workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                    workflow.completed_at and workflow.completed_at < cutoff_time):
                    workflows_to_remove.append(workflow_id)
            
            for workflow_id in workflows_to_remove:
                del self._workflows[workflow_id]
                # Clean up task results
                result_keys_to_remove = [
                    key for key in self._task_results.keys()
                    if key.startswith(f"{workflow_id}_")
                ]
                for key in result_keys_to_remove:
                    del self._task_results[key]
                
                # Clean up progress callbacks
                if workflow_id in self._progress_callbacks:
                    del self._progress_callbacks[workflow_id]
        
        logger.info(f"Cleaned up {len(workflows_to_remove)} completed workflows")
        return len(workflows_to_remove)