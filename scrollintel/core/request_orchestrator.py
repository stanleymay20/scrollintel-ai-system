"""
Request Orchestrator for Visual Generation System

This module provides the RequestOrchestrator class for managing complex visual generation workflows,
including task decomposition, parallel processing, priority queuing, and resource allocation.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class ResourceType(Enum):
    """Available resource types."""
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ResourceRequirement:
    """Resource requirement specification."""
    resource_type: ResourceType
    amount: float
    unit: str = "units"
    
    def __post_init__(self):
        if self.amount <= 0:
            raise ValueError("Resource amount must be positive")


@dataclass
class TaskDefinition:
    """Definition of a task to be executed."""
    id: str
    name: str
    handler: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class TaskExecution:
    """Runtime information about task execution."""
    task: TaskDefinition
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    progress: float = 0.0
    allocated_resources: Dict[ResourceType, float] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate task execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]


@dataclass
class WorkflowDefinition:
    """Definition of a complete workflow."""
    id: str
    name: str
    tasks: List[TaskDefinition]
    description: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Validate task dependencies
        task_ids = {task.id for task in self.tasks}
        for task in self.tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    raise ValueError(f"Task {task.id} depends on non-existent task {dep_id}")


class ResourceManager:
    """Manages resource allocation and availability."""
    
    def __init__(self):
        self.available_resources: Dict[ResourceType, float] = {
            ResourceType.GPU: 8.0,
            ResourceType.CPU: 32.0,
            ResourceType.MEMORY: 128.0,  # GB
            ResourceType.STORAGE: 1000.0,  # GB
            ResourceType.NETWORK: 10.0  # Gbps
        }
        self.allocated_resources: Dict[ResourceType, float] = {
            resource_type: 0.0 for resource_type in ResourceType
        }
        self._lock = asyncio.Lock()
    
    async def can_allocate(self, requirements: List[ResourceRequirement]) -> bool:
        """Check if resources can be allocated."""
        async with self._lock:
            for req in requirements:
                available = self.available_resources[req.resource_type] - self.allocated_resources[req.resource_type]
                if available < req.amount:
                    return False
            return True
    
    def _can_allocate_sync(self, requirements: List[ResourceRequirement]) -> bool:
        """Synchronous version of can_allocate for internal use."""
        for req in requirements:
            available = self.available_resources[req.resource_type] - self.allocated_resources[req.resource_type]
            if available < req.amount:
                return False
        return True
    
    async def allocate_resources(self, requirements: List[ResourceRequirement]) -> Dict[ResourceType, float]:
        """Allocate resources for a task."""
        async with self._lock:
            # Check availability using sync version to avoid deadlock
            if not self._can_allocate_sync(requirements):
                raise RuntimeError("Insufficient resources available")
            
            allocated = {}
            for req in requirements:
                self.allocated_resources[req.resource_type] += req.amount
                allocated[req.resource_type] = req.amount
            
            logger.info(f"Allocated resources: {allocated}")
            return allocated
    
    async def release_resources(self, allocated: Dict[ResourceType, float]):
        """Release allocated resources."""
        async with self._lock:
            for resource_type, amount in allocated.items():
                self.allocated_resources[resource_type] -= amount
                if self.allocated_resources[resource_type] < 0:
                    self.allocated_resources[resource_type] = 0
            
            logger.info(f"Released resources: {allocated}")
    
    def get_resource_utilization(self) -> Dict[ResourceType, float]:
        """Get current resource utilization percentages."""
        return {
            resource_type: (self.allocated_resources[resource_type] / self.available_resources[resource_type]) * 100
            for resource_type in ResourceType
        }


class TaskQueue:
    """Priority queue for task management."""
    
    def __init__(self):
        self._queues: Dict[TaskPriority, List[TaskExecution]] = {
            priority: [] for priority in TaskPriority
        }
        self._lock = asyncio.Lock()
    
    async def enqueue(self, task_execution: TaskExecution):
        """Add task to appropriate priority queue."""
        async with self._lock:
            priority = task_execution.task.priority
            self._queues[priority].append(task_execution)
            logger.info(f"Enqueued task {task_execution.task.id} with priority {priority.name}")
    
    async def dequeue(self) -> Optional[TaskExecution]:
        """Get next task from highest priority queue."""
        async with self._lock:
            # Check queues in priority order (highest first)
            for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
                if self._queues[priority]:
                    task_execution = self._queues[priority].pop(0)
                    logger.info(f"Dequeued task {task_execution.task.id} from {priority.name} queue")
                    return task_execution
            return None
    
    async def get_queue_status(self) -> Dict[TaskPriority, int]:
        """Get current queue lengths by priority."""
        async with self._lock:
            return {priority: len(queue) for priority, queue in self._queues.items()}
    
    async def remove_task(self, task_id: str) -> bool:
        """Remove a specific task from queues."""
        async with self._lock:
            for priority_queue in self._queues.values():
                for i, task_execution in enumerate(priority_queue):
                    if task_execution.task.id == task_id:
                        priority_queue.pop(i)
                        logger.info(f"Removed task {task_id} from queue")
                        return True
            return False


class ProgressTracker:
    """Tracks progress and status of tasks and workflows."""
    
    def __init__(self):
        self.task_executions: Dict[str, TaskExecution] = {}
        self.workflow_progress: Dict[str, Dict[str, Any]] = {}
        self.workflow_tasks: Dict[str, List[str]] = {}  # workflow_id -> list of task_ids
        self._lock = asyncio.Lock()
    
    async def register_task(self, task_execution: TaskExecution):
        """Register a task for tracking."""
        async with self._lock:
            self.task_executions[task_execution.task.id] = task_execution
    
    async def update_task_status(self, task_id: str, status: TaskStatus, progress: float = None, error: Exception = None):
        """Update task status and progress."""
        async with self._lock:
            if task_id in self.task_executions:
                execution = self.task_executions[task_id]
                execution.status = status
                
                if progress is not None:
                    execution.progress = max(0.0, min(100.0, progress))
                
                if error:
                    execution.error = error
                
                if status == TaskStatus.RUNNING and not execution.started_at:
                    execution.started_at = datetime.now()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    execution.completed_at = datetime.now()
                
                logger.info(f"Updated task {task_id} status to {status.value}")
    
    async def get_task_status(self, task_id: str) -> Optional[TaskExecution]:
        """Get current status of a task."""
        async with self._lock:
            return self.task_executions.get(task_id)
    
    async def get_workflow_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Get overall workflow progress."""
        async with self._lock:
            if workflow_id not in self.workflow_tasks:
                return {"progress": 0.0, "status": "not_started", "tasks": {}}
            
            # Get current task executions for this workflow
            task_ids = self.workflow_tasks[workflow_id]
            task_executions = [self.task_executions[tid] for tid in task_ids if tid in self.task_executions]
            
            # Update progress based on current task states
            await self._update_workflow_progress_internal(workflow_id, task_executions)
            
            return self.workflow_progress.get(workflow_id, {"progress": 0.0, "status": "not_started", "tasks": {}}).copy()
    
    async def update_workflow_progress(self, workflow_id: str, tasks: List[TaskExecution]):
        """Update workflow progress based on task statuses."""
        async with self._lock:
            await self._update_workflow_progress_internal(workflow_id, tasks)
    
    async def _update_workflow_progress_internal(self, workflow_id: str, tasks: List[TaskExecution]):
        """Internal method to update workflow progress (assumes lock is held)."""
        total_tasks = len(tasks)
        if total_tasks == 0:
            return
        
        completed_tasks = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in tasks if task.status == TaskStatus.FAILED)
        running_tasks = sum(1 for task in tasks if task.status == TaskStatus.RUNNING)
        
        overall_progress = (completed_tasks / total_tasks) * 100
        
        if failed_tasks > 0:
            status = "failed"
        elif completed_tasks == total_tasks:
            status = "completed"
        elif running_tasks > 0:
            status = "running"
        else:
            status = "pending"
        
        self.workflow_progress[workflow_id] = {
            "progress": overall_progress,
            "status": status,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "running_tasks": running_tasks,
            "tasks": {task.task.id: {
                "status": task.status.value,
                "progress": task.progress,
                "error": str(task.error) if task.error else None
            } for task in tasks}
        }


class RequestOrchestrator:
    """
    Main orchestrator for managing complex visual generation workflows.
    
    Provides task decomposition, parallel processing, priority queuing,
    and resource allocation capabilities.
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.resource_manager = ResourceManager()
        self.task_queue = TaskQueue()
        self.progress_tracker = ProgressTracker()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self._shutdown = False
        self._worker_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the orchestrator worker."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Request orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator and cleanup resources."""
        self._shutdown = True
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        # Cancel running tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        self.executor.shutdown(wait=True)
        logger.info("Request orchestrator stopped")
    
    async def submit_task(self, task: TaskDefinition) -> str:
        """Submit a single task for execution."""
        task_execution = TaskExecution(task=task)
        await self.progress_tracker.register_task(task_execution)
        await self.task_queue.enqueue(task_execution)
        
        logger.info(f"Submitted task {task.id}: {task.name}")
        return task.id
    
    async def submit_workflow(self, workflow: WorkflowDefinition) -> str:
        """Submit a complete workflow for execution."""
        # Create task executions for all tasks
        task_executions = []
        for task in workflow.tasks:
            task_execution = TaskExecution(task=task)
            await self.progress_tracker.register_task(task_execution)
            task_executions.append(task_execution)
        
        # Initialize workflow progress tracking
        await self.progress_tracker.update_workflow_progress(workflow.id, task_executions)
        
        # Enqueue tasks respecting dependencies
        await self._enqueue_workflow_tasks(task_executions, workflow.id)
        
        logger.info(f"Submitted workflow {workflow.id}: {workflow.name} with {len(workflow.tasks)} tasks")
        return workflow.id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task."""
        # Try to remove from queue first
        if await self.task_queue.remove_task(task_id):
            await self.progress_tracker.update_task_status(task_id, TaskStatus.CANCELLED)
            return True
        
        # Cancel if currently running
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            await self.progress_tracker.update_task_status(task_id, TaskStatus.CANCELLED)
            return True
        
        return False
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a task."""
        execution = await self.progress_tracker.get_task_status(task_id)
        if not execution:
            return None
        
        return {
            "id": execution.task.id,
            "name": execution.task.name,
            "status": execution.status.value,
            "progress": execution.progress,
            "created_at": execution.created_at.isoformat(),
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "duration": str(execution.duration) if execution.duration else None,
            "error": str(execution.error) if execution.error else None,
            "allocated_resources": execution.allocated_resources
        }
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow."""
        return await self.progress_tracker.get_workflow_progress(workflow_id)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        queue_status = await self.task_queue.get_queue_status()
        resource_utilization = self.resource_manager.get_resource_utilization()
        
        return {
            "running_tasks": len(self.running_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "queue_status": {priority.name: count for priority, count in queue_status.items()},
            "resource_utilization": {resource.name: f"{util:.1f}%" for resource, util in resource_utilization.items()},
            "shutdown": self._shutdown
        }
    
    async def _worker_loop(self):
        """Main worker loop for processing tasks."""
        while not self._shutdown:
            try:
                # Check if we can run more tasks
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get next task from queue
                task_execution = await self.task_queue.dequeue()
                if not task_execution:
                    await asyncio.sleep(0.1)
                    continue
                
                # Check dependencies
                if not await self._check_dependencies(task_execution):
                    # Re-queue task if dependencies not met
                    await self.task_queue.enqueue(task_execution)
                    await asyncio.sleep(0.1)
                    continue
                
                # Check resource availability
                can_allocate = await self.resource_manager.can_allocate(task_execution.task.resource_requirements)
                if not can_allocate:
                    # Re-queue task if resources not available
                    await self.task_queue.enqueue(task_execution)
                    await asyncio.sleep(0.1)
                    continue
                
                # Start task execution
                task_coroutine = self._execute_task(task_execution)
                task_future = asyncio.create_task(task_coroutine)
                self.running_tasks[task_execution.task.id] = task_future
                
                # Set up completion callback
                task_future.add_done_callback(
                    lambda fut, task_id=task_execution.task.id: asyncio.create_task(self._task_completed(task_id))
                )
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task_execution: TaskExecution):
        """Execute a single task."""
        task = task_execution.task
        
        try:
            # Allocate resources
            allocated_resources = await self.resource_manager.allocate_resources(task.resource_requirements)
            task_execution.allocated_resources = allocated_resources
            
            # Update status to running
            await self.progress_tracker.update_task_status(task.id, TaskStatus.RUNNING)
            
            # Execute task with timeout
            if asyncio.iscoroutinefunction(task.handler):
                if task.timeout:
                    result = await asyncio.wait_for(
                        task.handler(*task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                else:
                    result = await task.handler(*task.args, **task.kwargs)
            else:
                # Run synchronous function in executor
                if task.timeout:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self.executor, task.handler, *task.args
                        ),
                        timeout=task.timeout
                    )
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor, task.handler, *task.args
                    )
            
            # Update status to completed
            task_execution.result = result
            await self.progress_tracker.update_task_status(task.id, TaskStatus.COMPLETED, progress=100.0)
            
        except asyncio.TimeoutError:
            error = TimeoutError(f"Task {task.id} timed out after {task.timeout} seconds")
            await self._handle_task_error(task_execution, error)
        except Exception as e:
            await self._handle_task_error(task_execution, e)
        finally:
            # Release resources
            if task_execution.allocated_resources:
                await self.resource_manager.release_resources(task_execution.allocated_resources)
    
    async def _handle_task_error(self, task_execution: TaskExecution, error: Exception):
        """Handle task execution errors with retry logic."""
        task = task_execution.task
        
        logger.error(f"Task {task.id} failed: {error}")
        
        # Check if we should retry
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            logger.info(f"Retrying task {task.id} (attempt {task.retry_count}/{task.max_retries})")
            
            # Reset task execution state
            task_execution.status = TaskStatus.PENDING
            task_execution.started_at = None
            task_execution.completed_at = None
            task_execution.error = None
            
            # Re-queue task
            await self.task_queue.enqueue(task_execution)
        else:
            # Mark as failed
            await self.progress_tracker.update_task_status(task.id, TaskStatus.FAILED, error=error)
    
    async def _task_completed(self, task_id: str):
        """Callback for when a task completes."""
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]
    
    async def _check_dependencies(self, task_execution: TaskExecution) -> bool:
        """Check if all task dependencies are completed."""
        for dep_id in task_execution.task.dependencies:
            dep_execution = await self.progress_tracker.get_task_status(dep_id)
            if not dep_execution or dep_execution.status != TaskStatus.COMPLETED:
                return False
        return True
    
    async def _enqueue_workflow_tasks(self, task_executions: List[TaskExecution], workflow_id: str):
        """Enqueue workflow tasks in dependency order."""
        # Store workflow mapping for progress tracking
        self.progress_tracker.workflow_tasks[workflow_id] = [te.task.id for te in task_executions]
        
        # Simple approach: enqueue all tasks, dependencies will be checked during execution
        for task_execution in task_executions:
            await self.task_queue.enqueue(task_execution)
    
    @asynccontextmanager
    async def managed_execution(self):
        """Context manager for orchestrator lifecycle."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()


# Utility functions for common workflow patterns

async def create_image_generation_workflow(
    prompts: List[str],
    model_preferences: Optional[List[str]] = None,
    priority: TaskPriority = TaskPriority.NORMAL
) -> WorkflowDefinition:
    """Create a workflow for batch image generation."""
    tasks = []
    
    for i, prompt in enumerate(prompts):
        model_pref = model_preferences[i] if model_preferences and i < len(model_preferences) else None
        
        task = TaskDefinition(
            id=f"image_gen_{i}",
            name=f"Generate image for prompt {i+1}",
            handler=lambda p=prompt, m=model_pref: {"prompt": p, "model": m},  # Placeholder handler
            priority=priority,
            resource_requirements=[
                ResourceRequirement(ResourceType.GPU, 1.0),
                ResourceRequirement(ResourceType.MEMORY, 4.0)
            ]
        )
        tasks.append(task)
    
    return WorkflowDefinition(
        id=f"batch_image_gen_{uuid.uuid4()}",
        name="Batch Image Generation",
        tasks=tasks,
        description=f"Generate {len(prompts)} images from text prompts"
    )


async def create_video_generation_workflow(
    prompt: str,
    duration: float = 5.0,
    resolution: tuple = (1280, 720),
    priority: TaskPriority = TaskPriority.HIGH
) -> WorkflowDefinition:
    """Create a workflow for video generation with post-processing."""
    
    # Task 1: Generate base video
    base_generation_task = TaskDefinition(
        id="video_base_gen",
        name="Generate base video",
        handler=lambda: {"prompt": prompt, "duration": duration, "resolution": resolution},
        priority=priority,
        resource_requirements=[
            ResourceRequirement(ResourceType.GPU, 2.0),
            ResourceRequirement(ResourceType.MEMORY, 8.0)
        ],
        timeout=300.0  # 5 minutes
    )
    
    # Task 2: Enhance video quality (depends on base generation)
    enhancement_task = TaskDefinition(
        id="video_enhancement",
        name="Enhance video quality",
        handler=lambda: {"enhancement": "quality_boost"},
        priority=priority,
        dependencies=["video_base_gen"],
        resource_requirements=[
            ResourceRequirement(ResourceType.GPU, 1.0),
            ResourceRequirement(ResourceType.MEMORY, 4.0)
        ],
        timeout=180.0  # 3 minutes
    )
    
    # Task 3: Apply post-processing effects (depends on enhancement)
    postprocess_task = TaskDefinition(
        id="video_postprocess",
        name="Apply post-processing",
        handler=lambda: {"postprocess": "final_effects"},
        priority=priority,
        dependencies=["video_enhancement"],
        resource_requirements=[
            ResourceRequirement(ResourceType.CPU, 2.0),
            ResourceRequirement(ResourceType.MEMORY, 2.0)
        ],
        timeout=120.0  # 2 minutes
    )
    
    return WorkflowDefinition(
        id=f"video_gen_{uuid.uuid4()}",
        name="Video Generation Pipeline",
        tasks=[base_generation_task, enhancement_task, postprocess_task],
        description=f"Generate and process video: {prompt}"
    )