"""
Tests for Request Orchestrator functionality.

Tests task management, workflow execution, resource allocation,
and progress tracking capabilities.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
import uuid

from scrollintel.core.request_orchestrator import (
    RequestOrchestrator,
    TaskDefinition,
    WorkflowDefinition,
    TaskExecution,
    TaskStatus,
    TaskPriority,
    ResourceType,
    ResourceRequirement,
    ResourceManager,
    TaskQueue,
    ProgressTracker,
    create_image_generation_workflow,
    create_video_generation_workflow
)


class TestResourceRequirement:
    """Test ResourceRequirement class."""
    
    def test_valid_resource_requirement(self):
        """Test creating valid resource requirement."""
        req = ResourceRequirement(ResourceType.GPU, 2.0, "cores")
        assert req.resource_type == ResourceType.GPU
        assert req.amount == 2.0
        assert req.unit == "cores"
    
    def test_invalid_resource_amount(self):
        """Test invalid resource amount raises error."""
        with pytest.raises(ValueError, match="Resource amount must be positive"):
            ResourceRequirement(ResourceType.GPU, -1.0)
        
        with pytest.raises(ValueError, match="Resource amount must be positive"):
            ResourceRequirement(ResourceType.CPU, 0.0)


class TestTaskDefinition:
    """Test TaskDefinition class."""
    
    def test_task_definition_creation(self):
        """Test creating task definition."""
        def dummy_handler():
            return "result"
        
        task = TaskDefinition(
            id="test_task",
            name="Test Task",
            handler=dummy_handler,
            priority=TaskPriority.HIGH
        )
        
        assert task.id == "test_task"
        assert task.name == "Test Task"
        assert task.handler == dummy_handler
        assert task.priority == TaskPriority.HIGH
        assert task.retry_count == 0
        assert task.max_retries == 3
    
    def test_auto_generated_id(self):
        """Test auto-generated task ID."""
        task = TaskDefinition(
            id="",
            name="Test Task",
            handler=lambda: None
        )
        
        assert task.id != ""
        assert len(task.id) == 36  # UUID length


class TestTaskExecution:
    """Test TaskExecution class."""
    
    def test_task_execution_creation(self):
        """Test creating task execution."""
        task = TaskDefinition(
            id="test_task",
            name="Test Task",
            handler=lambda: None
        )
        
        execution = TaskExecution(task=task)
        
        assert execution.task == task
        assert execution.status == TaskStatus.PENDING
        assert execution.progress == 0.0
        assert execution.result is None
        assert execution.error is None
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        task = TaskDefinition(id="test", name="Test", handler=lambda: None)
        execution = TaskExecution(task=task)
        
        # No duration when not started
        assert execution.duration is None
        
        # Set start time
        execution.started_at = datetime.now()
        assert execution.duration is None  # Still no end time
        
        # Set end time
        execution.completed_at = execution.started_at + timedelta(seconds=5)
        assert execution.duration == timedelta(seconds=5)
    
    def test_is_complete(self):
        """Test completion status check."""
        task = TaskDefinition(id="test", name="Test", handler=lambda: None)
        execution = TaskExecution(task=task)
        
        # Initially not complete
        assert not execution.is_complete
        
        # Running is not complete
        execution.status = TaskStatus.RUNNING
        assert not execution.is_complete
        
        # Terminal states are complete
        execution.status = TaskStatus.COMPLETED
        assert execution.is_complete
        
        execution.status = TaskStatus.FAILED
        assert execution.is_complete
        
        execution.status = TaskStatus.CANCELLED
        assert execution.is_complete


class TestWorkflowDefinition:
    """Test WorkflowDefinition class."""
    
    def test_workflow_creation(self):
        """Test creating workflow definition."""
        task1 = TaskDefinition(id="task1", name="Task 1", handler=lambda: None)
        task2 = TaskDefinition(id="task2", name="Task 2", handler=lambda: None, dependencies=["task1"])
        
        workflow = WorkflowDefinition(
            id="test_workflow",
            name="Test Workflow",
            tasks=[task1, task2]
        )
        
        assert workflow.id == "test_workflow"
        assert workflow.name == "Test Workflow"
        assert len(workflow.tasks) == 2
    
    def test_invalid_dependencies(self):
        """Test workflow with invalid dependencies."""
        task1 = TaskDefinition(id="task1", name="Task 1", handler=lambda: None)
        task2 = TaskDefinition(id="task2", name="Task 2", handler=lambda: None, dependencies=["nonexistent"])
        
        with pytest.raises(ValueError, match="depends on non-existent task"):
            WorkflowDefinition(
                id="test_workflow",
                name="Test Workflow",
                tasks=[task1, task2]
            )


class TestResourceManager:
    """Test ResourceManager class."""
    
    @pytest.fixture
    def resource_manager(self):
        """Create resource manager for testing."""
        return ResourceManager()
    
    @pytest.mark.asyncio
    async def test_resource_allocation(self, resource_manager):
        """Test resource allocation and release."""
        requirements = [
            ResourceRequirement(ResourceType.GPU, 2.0),
            ResourceRequirement(ResourceType.MEMORY, 4.0)
        ]
        
        # Check availability - fix the async call
        can_allocate = await resource_manager.can_allocate(requirements)
        assert can_allocate
        
        # Allocate resources
        allocated = await resource_manager.allocate_resources(requirements)
        assert allocated[ResourceType.GPU] == 2.0
        assert allocated[ResourceType.MEMORY] == 4.0
        
        # Check utilization
        utilization = resource_manager.get_resource_utilization()
        assert utilization[ResourceType.GPU] == 25.0  # 2/8 * 100
        assert utilization[ResourceType.MEMORY] > 0
        
        # Release resources
        await resource_manager.release_resources(allocated)
        
        # Check utilization after release
        utilization = resource_manager.get_resource_utilization()
        assert utilization[ResourceType.GPU] == 0.0
        assert utilization[ResourceType.MEMORY] == 0.0
    
    @pytest.mark.asyncio
    async def test_insufficient_resources(self, resource_manager):
        """Test handling of insufficient resources."""
        # Try to allocate more than available
        requirements = [ResourceRequirement(ResourceType.GPU, 10.0)]  # More than 8 available
        
        can_allocate = await resource_manager.can_allocate(requirements)
        assert not can_allocate
        
        with pytest.raises(RuntimeError, match="Insufficient resources"):
            await resource_manager.allocate_resources(requirements)


class TestTaskQueue:
    """Test TaskQueue class."""
    
    @pytest.fixture
    def task_queue(self):
        """Create task queue for testing."""
        return TaskQueue()
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, task_queue):
        """Test priority-based task ordering."""
        # Create tasks with different priorities
        low_task = TaskExecution(TaskDefinition(id="low", name="Low", handler=lambda: None, priority=TaskPriority.LOW))
        high_task = TaskExecution(TaskDefinition(id="high", name="High", handler=lambda: None, priority=TaskPriority.HIGH))
        urgent_task = TaskExecution(TaskDefinition(id="urgent", name="Urgent", handler=lambda: None, priority=TaskPriority.URGENT))
        
        # Enqueue in random order
        await task_queue.enqueue(low_task)
        await task_queue.enqueue(urgent_task)
        await task_queue.enqueue(high_task)
        
        # Dequeue should return highest priority first
        first = await task_queue.dequeue()
        assert first.task.id == "urgent"
        
        second = await task_queue.dequeue()
        assert second.task.id == "high"
        
        third = await task_queue.dequeue()
        assert third.task.id == "low"
        
        # Queue should be empty
        empty = await task_queue.dequeue()
        assert empty is None
    
    @pytest.mark.asyncio
    async def test_queue_status(self, task_queue):
        """Test queue status reporting."""
        # Initially empty
        status = await task_queue.get_queue_status()
        assert all(count == 0 for count in status.values())
        
        # Add tasks
        low_task = TaskExecution(TaskDefinition(id="low", name="Low", handler=lambda: None, priority=TaskPriority.LOW))
        high_task = TaskExecution(TaskDefinition(id="high", name="High", handler=lambda: None, priority=TaskPriority.HIGH))
        
        await task_queue.enqueue(low_task)
        await task_queue.enqueue(high_task)
        
        status = await task_queue.get_queue_status()
        assert status[TaskPriority.LOW] == 1
        assert status[TaskPriority.HIGH] == 1
        assert status[TaskPriority.NORMAL] == 0
    
    @pytest.mark.asyncio
    async def test_remove_task(self, task_queue):
        """Test removing specific task from queue."""
        task = TaskExecution(TaskDefinition(id="test", name="Test", handler=lambda: None))
        await task_queue.enqueue(task)
        
        # Remove existing task
        removed = await task_queue.remove_task("test")
        assert removed
        
        # Try to remove non-existent task
        not_removed = await task_queue.remove_task("nonexistent")
        assert not not_removed


class TestProgressTracker:
    """Test ProgressTracker class."""
    
    @pytest.fixture
    def progress_tracker(self):
        """Create progress tracker for testing."""
        return ProgressTracker()
    
    @pytest.mark.asyncio
    async def test_task_tracking(self, progress_tracker):
        """Test task registration and status updates."""
        task = TaskDefinition(id="test", name="Test", handler=lambda: None)
        execution = TaskExecution(task=task)
        
        # Register task
        await progress_tracker.register_task(execution)
        
        # Get initial status
        status = await progress_tracker.get_task_status("test")
        assert status.status == TaskStatus.PENDING
        assert status.progress == 0.0
        
        # Update status
        await progress_tracker.update_task_status("test", TaskStatus.RUNNING, progress=50.0)
        
        status = await progress_tracker.get_task_status("test")
        assert status.status == TaskStatus.RUNNING
        assert status.progress == 50.0
        assert status.started_at is not None
    
    @pytest.mark.asyncio
    async def test_workflow_progress(self, progress_tracker):
        """Test workflow progress calculation."""
        # Create tasks
        task1 = TaskDefinition(id="task1", name="Task 1", handler=lambda: None)
        task2 = TaskDefinition(id="task2", name="Task 2", handler=lambda: None)
        
        execution1 = TaskExecution(task=task1)
        execution2 = TaskExecution(task=task2)
        
        await progress_tracker.register_task(execution1)
        await progress_tracker.register_task(execution2)
        
        # Update workflow progress
        await progress_tracker.update_workflow_progress("workflow1", [execution1, execution2])
        
        progress = await progress_tracker.get_workflow_progress("workflow1")
        assert progress["status"] == "pending"
        assert progress["total_tasks"] == 2
        assert progress["completed_tasks"] == 0
        
        # Complete one task
        execution1.status = TaskStatus.COMPLETED
        await progress_tracker.update_workflow_progress("workflow1", [execution1, execution2])
        
        progress = await progress_tracker.get_workflow_progress("workflow1")
        assert progress["progress"] == 50.0
        assert progress["completed_tasks"] == 1


class TestRequestOrchestrator:
    """Test RequestOrchestrator class."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        return RequestOrchestrator(max_concurrent_tasks=2)
    
    @pytest.mark.asyncio
    async def test_task_submission(self, orchestrator):
        """Test task submission and execution."""
        async def test_handler():
            await asyncio.sleep(0.1)
            return "test_result"
        
        task = TaskDefinition(
            id="test_task",
            name="Test Task",
            handler=test_handler
        )
        
        async with orchestrator.managed_execution():
            task_id = await orchestrator.submit_task(task)
            assert task_id == "test_task"
            
            # Wait for task to complete
            await asyncio.sleep(0.5)
            
            status = await orchestrator.get_task_status(task_id)
            assert status["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_workflow_submission(self, orchestrator):
        """Test workflow submission and execution."""
        async def task1_handler():
            await asyncio.sleep(0.1)
            return "task1_result"
        
        async def task2_handler():
            await asyncio.sleep(0.1)
            return "task2_result"
        
        task1 = TaskDefinition(id="task1", name="Task 1", handler=task1_handler)
        task2 = TaskDefinition(id="task2", name="Task 2", handler=task2_handler, dependencies=["task1"])
        
        workflow = WorkflowDefinition(
            id="test_workflow",
            name="Test Workflow",
            tasks=[task1, task2]
        )
        
        async with orchestrator.managed_execution():
            workflow_id = await orchestrator.submit_workflow(workflow)
            
            # Wait for workflow to complete
            await asyncio.sleep(1.0)
            
            # Check individual task statuses
            task1_status = await orchestrator.get_task_status("task1")
            task2_status = await orchestrator.get_task_status("task2")
            
            assert task1_status["status"] == "completed"
            assert task2_status["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self, orchestrator):
        """Test task cancellation."""
        async def long_running_task():
            await asyncio.sleep(10)  # Long running task
            return "result"
        
        task = TaskDefinition(
            id="long_task",
            name="Long Task",
            handler=long_running_task
        )
        
        async with orchestrator.managed_execution():
            task_id = await orchestrator.submit_task(task)
            
            # Wait a bit then cancel
            await asyncio.sleep(0.1)
            success = await orchestrator.cancel_task(task_id)
            assert success
            
            # Check status
            await asyncio.sleep(0.1)
            status = await orchestrator.get_task_status(task_id)
            assert status["status"] == "cancelled"
    
    @pytest.mark.asyncio
    async def test_resource_constraints(self, orchestrator):
        """Test resource constraint handling."""
        async def resource_intensive_task():
            await asyncio.sleep(0.2)
            return "result"
        
        # Create tasks that require more resources than available
        tasks = []
        for i in range(5):
            task = TaskDefinition(
                id=f"task_{i}",
                name=f"Task {i}",
                handler=resource_intensive_task,
                resource_requirements=[ResourceRequirement(ResourceType.GPU, 3.0)]  # Each needs 3 GPUs
            )
            tasks.append(task)
        
        async with orchestrator.managed_execution():
            # Submit all tasks
            task_ids = []
            for task in tasks:
                task_id = await orchestrator.submit_task(task)
                task_ids.append(task_id)
            
            # Wait for execution
            await asyncio.sleep(1.0)
            
            # Check that not all tasks ran simultaneously due to resource constraints
            system_status = await orchestrator.get_system_status()
            assert system_status["running_tasks"] <= 2  # Max 2 can run with available GPUs
    
    @pytest.mark.asyncio
    async def test_system_status(self, orchestrator):
        """Test system status reporting."""
        async with orchestrator.managed_execution():
            status = await orchestrator.get_system_status()
            
            assert "running_tasks" in status
            assert "max_concurrent_tasks" in status
            assert "queue_status" in status
            assert "resource_utilization" in status
            assert status["max_concurrent_tasks"] == 2


class TestWorkflowUtilities:
    """Test workflow utility functions."""
    
    @pytest.mark.asyncio
    async def test_image_generation_workflow(self):
        """Test image generation workflow creation."""
        prompts = ["A cat", "A dog", "A bird"]
        models = ["dalle3", "stable_diffusion", "midjourney"]
        
        workflow = await create_image_generation_workflow(
            prompts=prompts,
            model_preferences=models,
            priority=TaskPriority.HIGH
        )
        
        assert len(workflow.tasks) == 3
        assert workflow.name == "Batch Image Generation"
        assert all(task.priority == TaskPriority.HIGH for task in workflow.tasks)
        
        # Check resource requirements
        for task in workflow.tasks:
            assert len(task.resource_requirements) == 2  # GPU and Memory
            gpu_req = next(req for req in task.resource_requirements if req.resource_type == ResourceType.GPU)
            assert gpu_req.amount == 1.0
    
    @pytest.mark.asyncio
    async def test_video_generation_workflow(self):
        """Test video generation workflow creation."""
        workflow = await create_video_generation_workflow(
            prompt="A flying car",
            duration=10.0,
            resolution=(1920, 1080),
            priority=TaskPriority.URGENT
        )
        
        assert len(workflow.tasks) == 3  # Base generation, enhancement, post-processing
        assert workflow.name == "Video Generation Pipeline"
        
        # Check task dependencies
        base_task = next(task for task in workflow.tasks if task.id == "video_base_gen")
        enhancement_task = next(task for task in workflow.tasks if task.id == "video_enhancement")
        postprocess_task = next(task for task in workflow.tasks if task.id == "video_postprocess")
        
        assert len(base_task.dependencies) == 0
        assert "video_base_gen" in enhancement_task.dependencies
        assert "video_enhancement" in postprocess_task.dependencies
        
        # Check timeouts
        assert base_task.timeout == 300.0
        assert enhancement_task.timeout == 180.0
        assert postprocess_task.timeout == 120.0


@pytest.mark.asyncio
async def test_orchestrator_lifecycle():
    """Test orchestrator startup and shutdown."""
    orchestrator = RequestOrchestrator()
    
    # Start orchestrator
    await orchestrator.start()
    assert orchestrator._worker_task is not None
    
    # Stop orchestrator
    await orchestrator.stop()
    assert orchestrator._shutdown


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in task execution."""
    def failing_task():
        raise ValueError("Task failed")
    
    task = TaskDefinition(
        id="failing_task",
        name="Failing Task",
        handler=failing_task,
        max_retries=2
    )
    
    orchestrator = RequestOrchestrator()
    
    async with orchestrator.managed_execution():
        task_id = await orchestrator.submit_task(task)
        
        # Wait for retries to complete
        await asyncio.sleep(1.0)
        
        status = await orchestrator.get_task_status(task_id)
        assert status["status"] == "failed"
        assert "Task failed" in status["error"]


@pytest.mark.asyncio
async def test_timeout_handling():
    """Test task timeout handling."""
    async def slow_task():
        await asyncio.sleep(2.0)  # Longer than timeout
        return "result"
    
    task = TaskDefinition(
        id="slow_task",
        name="Slow Task",
        handler=slow_task,
        timeout=0.5  # Short timeout
    )
    
    orchestrator = RequestOrchestrator()
    
    async with orchestrator.managed_execution():
        task_id = await orchestrator.submit_task(task)
        
        # Wait for timeout
        await asyncio.sleep(1.0)
        
        status = await orchestrator.get_task_status(task_id)
        assert status["status"] == "failed"
        assert "timed out" in status["error"]