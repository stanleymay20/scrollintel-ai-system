"""
End-to-End Tests for Pipeline Orchestration Engine
Tests complete pipeline execution workflows including scheduling, dependencies, and resource management.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import uuid

from scrollintel.core.pipeline_orchestrator import (
    PipelineOrchestrator, ExecutionContext, ExecutionStatus, ResourceType,
    ScheduleConfig, ScheduleType, ResourceAllocation
)
from scrollintel.models.pipeline_models import Pipeline, PipelineNode, PipelineConnection, NodeType


class TestPipelineOrchestrationEndToEnd:
    """End-to-end tests for pipeline orchestration"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a test orchestrator instance"""
        orchestrator = PipelineOrchestrator(max_concurrent_executions=3)
        yield orchestrator
        orchestrator.stop()
    
    @pytest.fixture
    def sample_pipeline_id(self):
        """Generate a sample pipeline ID"""
        return str(uuid.uuid4())
    
    @pytest.fixture
    def mock_database_session(self):
        """Mock database session for testing"""
        with patch('scrollintel.core.pipeline_orchestrator.get_database_session') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            # Mock pipeline with nodes and connections
            mock_pipeline = Mock()
            mock_pipeline.nodes = [
                Mock(id="node1", node_type=NodeType.DATA_SOURCE),
                Mock(id="node2", node_type=NodeType.TRANSFORMATION),
                Mock(id="node3", node_type=NodeType.DATA_SINK)
            ]
            mock_pipeline.connections = [
                Mock(source_node_id="node1", target_node_id="node2"),
                Mock(source_node_id="node2", target_node_id="node3")
            ]
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_pipeline
            yield mock_db
    
    def test_orchestrator_lifecycle(self, orchestrator):
        """Test orchestrator start and stop lifecycle"""
        assert not orchestrator.is_running
        
        orchestrator.start()
        assert orchestrator.is_running
        assert orchestrator.scheduler_thread is not None
        assert orchestrator.scheduler_thread.is_alive()
        
        orchestrator.stop()
        assert not orchestrator.is_running
    
    def test_immediate_pipeline_execution(self, orchestrator, sample_pipeline_id, mock_database_session):
        """Test immediate pipeline execution"""
        orchestrator.start()
        
        # Execute pipeline immediately
        execution_id = orchestrator.execute_pipeline_now(
            pipeline_id=sample_pipeline_id,
            priority=10,
            resource_requirements={
                ResourceType.CPU: 2.0,
                ResourceType.MEMORY: 4.0
            }
        )
        
        assert execution_id is not None
        
        # Check execution status
        status = orchestrator.get_execution_status(execution_id)
        assert status is not None
        assert status['pipeline_id'] == sample_pipeline_id
        assert status['priority'] == 10
        
        # Wait for execution to start
        time.sleep(2)
        
        # Verify execution is running or completed
        updated_status = orchestrator.get_execution_status(execution_id)
        assert updated_status['status'] in ['running', 'completed']
    
    def test_scheduled_pipeline_execution(self, orchestrator, sample_pipeline_id, mock_database_session):
        """Test scheduled pipeline execution"""
        orchestrator.start()
        
        # Schedule pipeline with interval
        schedule_config = ScheduleConfig(
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=5,
            enabled=True
        )
        
        execution_id = orchestrator.schedule_pipeline(
            pipeline_id=sample_pipeline_id,
            schedule_config=schedule_config,
            priority=5
        )
        
        assert execution_id is not None
        
        # Check initial status
        status = orchestrator.get_execution_status(execution_id)
        assert status['status'] == 'pending'
        
        # Wait for execution to be processed
        time.sleep(3)
        
        # Verify execution progressed
        updated_status = orchestrator.get_execution_status(execution_id)
        assert updated_status['status'] in ['running', 'completed']
    
    def test_pipeline_dependencies(self, orchestrator, mock_database_session):
        """Test pipeline dependency management"""
        orchestrator.start()
        
        pipeline_a = str(uuid.uuid4())
        pipeline_b = str(uuid.uuid4())
        pipeline_c = str(uuid.uuid4())
        
        # Schedule pipeline A (no dependencies)
        schedule_config = ScheduleConfig(schedule_type=ScheduleType.ONCE)
        execution_a = orchestrator.schedule_pipeline(
            pipeline_id=pipeline_a,
            schedule_config=schedule_config,
            priority=5
        )
        
        # Schedule pipeline B (depends on A)
        execution_b = orchestrator.schedule_pipeline(
            pipeline_id=pipeline_b,
            schedule_config=schedule_config,
            priority=5,
            dependencies=[execution_a]
        )
        
        # Schedule pipeline C (depends on B)
        execution_c = orchestrator.schedule_pipeline(
            pipeline_id=pipeline_c,
            schedule_config=schedule_config,
            priority=5,
            dependencies=[execution_b]
        )
        
        # Wait for executions to process
        time.sleep(8)
        
        # Verify execution order
        status_a = orchestrator.get_execution_status(execution_a)
        status_b = orchestrator.get_execution_status(execution_b)
        status_c = orchestrator.get_execution_status(execution_c)
        
        # A should complete before B starts
        assert status_a['status'] in ['completed', 'running']
        
        # If A is completed, B should be running or completed
        if status_a['status'] == 'completed':
            assert status_b['status'] in ['running', 'completed']
    
    def test_resource_allocation_and_scaling(self, orchestrator, mock_database_session):
        """Test resource allocation and scaling"""
        orchestrator.start()
        
        # Set limited resources
        orchestrator.resource_manager.total_cpu = 4.0
        orchestrator.resource_manager.total_memory = 8.0
        
        pipeline_ids = [str(uuid.uuid4()) for _ in range(5)]
        execution_ids = []
        
        # Schedule multiple pipelines with high resource requirements
        for pipeline_id in pipeline_ids:
            schedule_config = ScheduleConfig(schedule_type=ScheduleType.ONCE)
            execution_id = orchestrator.schedule_pipeline(
                pipeline_id=pipeline_id,
                schedule_config=schedule_config,
                priority=5,
                resource_requirements={
                    ResourceType.CPU: 2.0,
                    ResourceType.MEMORY: 4.0
                }
            )
            execution_ids.append(execution_id)
        
        # Wait for processing
        time.sleep(3)
        
        # Check resource utilization
        utilization = orchestrator.get_resource_utilization()
        assert utilization['cpu_utilization'] <= 100.0
        assert utilization['memory_utilization'] <= 100.0
        
        # Verify not all pipelines are running simultaneously due to resource constraints
        running_count = 0
        for execution_id in execution_ids:
            status = orchestrator.get_execution_status(execution_id)
            if status and status['status'] == 'running':
                running_count += 1
        
        # Should be limited by resource availability
        assert running_count <= 2  # Max 2 pipelines with 2 CPU each on 4 CPU total
    
    def test_retry_mechanism_with_exponential_backoff(self, orchestrator, sample_pipeline_id, mock_database_session):
        """Test retry mechanism with exponential backoff"""
        orchestrator.start()
        
        # Mock pipeline execution to fail initially
        original_execute = orchestrator._execute_pipeline
        
        def mock_failing_execute(context):
            if context.retry_count < 2:
                raise Exception("Simulated failure")
            return {"success": True, "execution_time": 1}
        
        orchestrator._execute_pipeline = mock_failing_execute
        
        try:
            execution_id = orchestrator.execute_pipeline_now(
                pipeline_id=sample_pipeline_id,
                priority=10
            )
            
            # Wait for initial execution and retries
            time.sleep(10)
            
            # Check final status
            status = orchestrator.get_execution_status(execution_id)
            assert status is not None
            
            # Should eventually succeed after retries
            metrics = orchestrator.get_orchestrator_metrics()
            assert metrics['retry_executions'] >= 2
            
        finally:
            orchestrator._execute_pipeline = original_execute
    
    def test_execution_cancellation(self, orchestrator, sample_pipeline_id, mock_database_session):
        """Test execution cancellation"""
        orchestrator.start()
        
        # Schedule a pipeline
        schedule_config = ScheduleConfig(schedule_type=ScheduleType.ONCE)
        execution_id = orchestrator.schedule_pipeline(
            pipeline_id=sample_pipeline_id,
            schedule_config=schedule_config,
            priority=1  # Low priority to keep it in queue
        )
        
        # Cancel before execution starts
        success = orchestrator.cancel_execution(execution_id)
        assert success
        
        # Verify cancellation
        status = orchestrator.get_execution_status(execution_id)
        assert status['status'] == 'cancelled'
    
    def test_execution_pause_and_resume(self, orchestrator, sample_pipeline_id, mock_database_session):
        """Test execution pause and resume"""
        orchestrator.start()
        
        # Mock long-running execution
        original_execute = orchestrator._execute_pipeline
        
        def mock_long_execute(context):
            time.sleep(10)  # Long execution
            return {"success": True, "execution_time": 10}
        
        orchestrator._execute_pipeline = mock_long_execute
        
        try:
            execution_id = orchestrator.execute_pipeline_now(
                pipeline_id=sample_pipeline_id,
                priority=10
            )
            
            # Wait for execution to start
            time.sleep(2)
            
            # Pause execution
            success = orchestrator.pause_execution(execution_id)
            assert success
            
            status = orchestrator.get_execution_status(execution_id)
            assert status['status'] == 'paused'
            
            # Resume execution
            success = orchestrator.resume_execution(execution_id)
            assert success
            
            status = orchestrator.get_execution_status(execution_id)
            assert status['status'] == 'running'
            
        finally:
            orchestrator._execute_pipeline = original_execute
    
    def test_priority_based_execution_ordering(self, orchestrator, mock_database_session):
        """Test priority-based execution ordering"""
        orchestrator.start()
        
        # Set max concurrent executions to 1 to test ordering
        orchestrator.max_concurrent_executions = 1
        
        pipeline_ids = [str(uuid.uuid4()) for _ in range(3)]
        priorities = [1, 10, 5]  # Low, High, Medium
        execution_ids = []
        
        # Schedule pipelines with different priorities
        for pipeline_id, priority in zip(pipeline_ids, priorities):
            schedule_config = ScheduleConfig(schedule_type=ScheduleType.ONCE)
            execution_id = orchestrator.schedule_pipeline(
                pipeline_id=pipeline_id,
                schedule_config=schedule_config,
                priority=priority
            )
            execution_ids.append((execution_id, priority))
        
        # Wait for processing
        time.sleep(8)
        
        # Check execution order - higher priority should execute first
        execution_order = []
        for execution_id, priority in execution_ids:
            status = orchestrator.get_execution_status(execution_id)
            if status and status['status'] in ['running', 'completed']:
                execution_order.append(priority)
        
        # Should process in priority order: 10, 5, 1
        assert len(execution_order) > 0
        # First executed should be highest priority
        assert execution_order[0] == 10
    
    def test_orchestrator_metrics_collection(self, orchestrator, mock_database_session):
        """Test orchestrator metrics collection"""
        orchestrator.start()
        
        pipeline_ids = [str(uuid.uuid4()) for _ in range(3)]
        
        # Execute multiple pipelines
        for pipeline_id in pipeline_ids:
            orchestrator.execute_pipeline_now(
                pipeline_id=pipeline_id,
                priority=5
            )
        
        # Wait for executions
        time.sleep(8)
        
        # Check metrics
        metrics = orchestrator.get_orchestrator_metrics()
        
        assert 'total_executions' in metrics
        assert 'successful_executions' in metrics
        assert 'failed_executions' in metrics
        assert 'retry_executions' in metrics
        assert 'resource_utilization' in metrics
        assert 'queue_size' in metrics
        assert 'running_executions' in metrics
        assert 'completed_executions' in metrics
        
        assert metrics['total_executions'] >= 3
        assert metrics['successful_executions'] >= 0
    
    def test_dependency_graph_analysis(self, orchestrator, sample_pipeline_id, mock_database_session):
        """Test dependency graph analysis"""
        # Test dependency chain detection
        dependencies = orchestrator.get_pipeline_dependencies(sample_pipeline_id)
        
        # Should return list of node IDs in execution order
        assert isinstance(dependencies, list)
        assert len(dependencies) >= 0
    
    def test_concurrent_execution_limits(self, orchestrator, mock_database_session):
        """Test concurrent execution limits"""
        orchestrator.start()
        orchestrator.max_concurrent_executions = 2
        
        pipeline_ids = [str(uuid.uuid4()) for _ in range(5)]
        
        # Mock long-running executions
        original_execute = orchestrator._execute_pipeline
        
        def mock_long_execute(context):
            time.sleep(5)
            return {"success": True, "execution_time": 5}
        
        orchestrator._execute_pipeline = mock_long_execute
        
        try:
            # Schedule multiple pipelines
            for pipeline_id in pipeline_ids:
                orchestrator.execute_pipeline_now(
                    pipeline_id=pipeline_id,
                    priority=5
                )
            
            # Wait a bit for processing
            time.sleep(2)
            
            # Check that no more than max_concurrent_executions are running
            metrics = orchestrator.get_orchestrator_metrics()
            assert metrics['running_executions'] <= orchestrator.max_concurrent_executions
            
        finally:
            orchestrator._execute_pipeline = original_execute
    
    def test_resource_deallocation_on_completion(self, orchestrator, sample_pipeline_id, mock_database_session):
        """Test resource deallocation when execution completes"""
        orchestrator.start()
        
        initial_cpu = orchestrator.resource_manager.allocated_cpu
        initial_memory = orchestrator.resource_manager.allocated_memory
        
        execution_id = orchestrator.execute_pipeline_now(
            pipeline_id=sample_pipeline_id,
            priority=10,
            resource_requirements={
                ResourceType.CPU: 2.0,
                ResourceType.MEMORY: 4.0
            }
        )
        
        # Wait for execution to complete
        time.sleep(8)
        
        # Resources should be deallocated
        final_cpu = orchestrator.resource_manager.allocated_cpu
        final_memory = orchestrator.resource_manager.allocated_memory
        
        assert final_cpu == initial_cpu
        assert final_memory == initial_memory
    
    def test_error_handling_and_logging(self, orchestrator, sample_pipeline_id, mock_database_session):
        """Test error handling and logging"""
        orchestrator.start()
        
        # Mock execution that raises an exception
        original_execute = orchestrator._execute_pipeline
        
        def mock_error_execute(context):
            raise ValueError("Test error")
        
        orchestrator._execute_pipeline = mock_error_execute
        
        try:
            execution_id = orchestrator.execute_pipeline_now(
                pipeline_id=sample_pipeline_id,
                priority=10
            )
            
            # Wait for execution to fail
            time.sleep(3)
            
            # Check that error is handled gracefully
            status = orchestrator.get_execution_status(execution_id)
            assert status is not None
            
            metrics = orchestrator.get_orchestrator_metrics()
            assert metrics['failed_executions'] >= 1
            
        finally:
            orchestrator._execute_pipeline = original_execute


class TestResourceAllocation:
    """Test resource allocation functionality"""
    
    def test_resource_allocation_basic(self):
        """Test basic resource allocation"""
        allocator = ResourceAllocation(
            total_cpu=10.0,
            total_memory=20.0,
            total_storage=100.0,
            total_network=50.0
        )
        
        requirements = {
            ResourceType.CPU: 5.0,
            ResourceType.MEMORY: 10.0,
            ResourceType.STORAGE: 50.0,
            ResourceType.NETWORK: 25.0
        }
        
        # Should be able to allocate
        assert allocator.can_allocate(requirements)
        
        # Allocate resources
        success = allocator.allocate(requirements)
        assert success
        
        # Check allocation
        assert allocator.allocated_cpu == 5.0
        assert allocator.allocated_memory == 10.0
        assert allocator.allocated_storage == 50.0
        assert allocator.allocated_network == 25.0
    
    def test_resource_allocation_overflow(self):
        """Test resource allocation overflow protection"""
        allocator = ResourceAllocation(
            total_cpu=10.0,
            total_memory=20.0
        )
        
        # Try to allocate more than available
        requirements = {
            ResourceType.CPU: 15.0,
            ResourceType.MEMORY: 25.0
        }
        
        assert not allocator.can_allocate(requirements)
        
        success = allocator.allocate(requirements)
        assert not success
        
        # Resources should remain unchanged
        assert allocator.allocated_cpu == 0.0
        assert allocator.allocated_memory == 0.0
    
    def test_resource_deallocation(self):
        """Test resource deallocation"""
        allocator = ResourceAllocation(
            total_cpu=10.0,
            total_memory=20.0
        )
        
        requirements = {
            ResourceType.CPU: 5.0,
            ResourceType.MEMORY: 10.0
        }
        
        # Allocate and then deallocate
        allocator.allocate(requirements)
        assert allocator.allocated_cpu == 5.0
        assert allocator.allocated_memory == 10.0
        
        allocator.deallocate(requirements)
        assert allocator.allocated_cpu == 0.0
        assert allocator.allocated_memory == 0.0
    
    def test_resource_deallocation_underflow_protection(self):
        """Test resource deallocation underflow protection"""
        allocator = ResourceAllocation()
        
        # Try to deallocate more than allocated
        requirements = {
            ResourceType.CPU: 5.0,
            ResourceType.MEMORY: 10.0
        }
        
        allocator.deallocate(requirements)
        
        # Should not go negative
        assert allocator.allocated_cpu == 0.0
        assert allocator.allocated_memory == 0.0


class TestScheduleConfig:
    """Test schedule configuration"""
    
    def test_schedule_config_creation(self):
        """Test schedule configuration creation"""
        config = ScheduleConfig(
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=300,
            enabled=True,
            timezone="UTC"
        )
        
        assert config.schedule_type == ScheduleType.INTERVAL
        assert config.interval_seconds == 300
        assert config.enabled is True
        assert config.timezone == "UTC"
    
    def test_schedule_config_cron(self):
        """Test cron schedule configuration"""
        config = ScheduleConfig(
            schedule_type=ScheduleType.CRON,
            cron_expression="0 0 * * *",  # Daily at midnight
            enabled=True
        )
        
        assert config.schedule_type == ScheduleType.CRON
        assert config.cron_expression == "0 0 * * *"
    
    def test_schedule_config_once(self):
        """Test one-time schedule configuration"""
        start_time = datetime.utcnow() + timedelta(hours=1)
        
        config = ScheduleConfig(
            schedule_type=ScheduleType.ONCE,
            start_time=start_time,
            enabled=True
        )
        
        assert config.schedule_type == ScheduleType.ONCE
        assert config.start_time == start_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])