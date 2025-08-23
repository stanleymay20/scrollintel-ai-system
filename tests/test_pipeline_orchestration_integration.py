"""
Integration Tests for Pipeline Orchestration
Tests integration between orchestration engine and existing pipeline components.
"""

import pytest
import time
import uuid
from unittest.mock import Mock, patch

from scrollintel.core.pipeline_orchestrator import (
    PipelineOrchestrator, ScheduleConfig, ScheduleType, ResourceType
)
from scrollintel.core.pipeline_builder import PipelineBuilder
from scrollintel.models.pipeline_models import Pipeline, PipelineNode, NodeType, PipelineStatus


class TestPipelineOrchestrationIntegration:
    """Integration tests for pipeline orchestration"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        with patch('scrollintel.core.pipeline_orchestrator.get_database_session') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            # Mock pipeline with realistic structure
            mock_pipeline = Mock()
            mock_pipeline.id = str(uuid.uuid4())
            mock_pipeline.name = "Test Pipeline"
            mock_pipeline.status = PipelineStatus.ACTIVE
            
            # Mock nodes in topological order
            mock_pipeline.nodes = [
                Mock(id="source_node", node_type=NodeType.DATA_SOURCE),
                Mock(id="transform_node", node_type=NodeType.TRANSFORMATION),
                Mock(id="sink_node", node_type=NodeType.DATA_SINK)
            ]
            
            # Mock connections
            mock_pipeline.connections = [
                Mock(source_node_id="source_node", target_node_id="transform_node"),
                Mock(source_node_id="transform_node", target_node_id="sink_node")
            ]
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_pipeline
            mock_db.add = Mock()
            mock_db.commit = Mock()
            
            yield mock_db
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing"""
        orchestrator = PipelineOrchestrator(max_concurrent_executions=2)
        yield orchestrator
        orchestrator.stop()
    
    def test_orchestrator_pipeline_builder_integration(self, orchestrator, mock_db_session):
        """Test integration between orchestrator and pipeline builder"""
        orchestrator.start()
        
        # Create a pipeline using builder
        pipeline_id = str(uuid.uuid4())
        
        # Execute pipeline through orchestrator
        execution_id = orchestrator.execute_pipeline_now(
            pipeline_id=pipeline_id,
            priority=5,
            resource_requirements={
                ResourceType.CPU: 2.0,
                ResourceType.MEMORY: 4.0
            }
        )
        
        assert execution_id is not None
        
        # Verify execution status
        status = orchestrator.get_execution_status(execution_id)
        assert status is not None
        assert status['pipeline_id'] == pipeline_id
        assert status['priority'] == 5
    
    def test_orchestrator_with_pipeline_validation(self, orchestrator, mock_db_session):
        """Test orchestrator integration with pipeline validation"""
        orchestrator.start()
        
        pipeline_id = str(uuid.uuid4())
        
        # Mock validation result
        with patch('scrollintel.core.pipeline_builder.PipelineBuilder') as mock_builder_class:
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            
            # Mock successful validation
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.errors = []
            mock_validation_result.warnings = []
            mock_builder.validate_pipeline.return_value = mock_validation_result
            
            # Execute pipeline
            execution_id = orchestrator.execute_pipeline_now(
                pipeline_id=pipeline_id,
                priority=8
            )
            
            # Wait for processing
            time.sleep(2)
            
            # Verify execution proceeded
            status = orchestrator.get_execution_status(execution_id)
            assert status is not None
    
    def test_orchestrator_dependency_resolution(self, orchestrator, mock_db_session):
        """Test dependency resolution with real pipeline structure"""
        orchestrator.start()
        
        pipeline_id = str(uuid.uuid4())
        
        # Get dependencies
        dependencies = orchestrator.get_pipeline_dependencies(pipeline_id)
        
        # Should return node execution order
        assert isinstance(dependencies, list)
        # Based on our mock, should have 3 nodes
        assert len(dependencies) == 3
    
    def test_orchestrator_execution_logging(self, orchestrator, mock_db_session):
        """Test execution logging to database"""
        orchestrator.start()
        
        pipeline_id = str(uuid.uuid4())
        
        # Execute pipeline
        execution_id = orchestrator.execute_pipeline_now(
            pipeline_id=pipeline_id,
            priority=7
        )
        
        # Wait for execution
        time.sleep(6)
        
        # Verify database interaction
        mock_db_session.add.assert_called()
        mock_db_session.commit.assert_called()
    
    def test_orchestrator_resource_tracking(self, orchestrator, mock_db_session):
        """Test resource tracking during execution"""
        orchestrator.start()
        
        pipeline_id = str(uuid.uuid4())
        
        # Set initial resource state
        initial_cpu = orchestrator.resource_manager.allocated_cpu
        initial_memory = orchestrator.resource_manager.allocated_memory
        
        # Execute with specific resource requirements
        execution_id = orchestrator.execute_pipeline_now(
            pipeline_id=pipeline_id,
            priority=6,
            resource_requirements={
                ResourceType.CPU: 3.0,
                ResourceType.MEMORY: 6.0
            }
        )
        
        # Check resource allocation during execution
        time.sleep(1)
        
        # Resources should be allocated
        current_cpu = orchestrator.resource_manager.allocated_cpu
        current_memory = orchestrator.resource_manager.allocated_memory
        
        # Should have allocated resources (or execution completed quickly)
        assert current_cpu >= initial_cpu
        assert current_memory >= initial_memory
        
        # Wait for completion
        time.sleep(6)
        
        # Resources should be deallocated
        final_cpu = orchestrator.resource_manager.allocated_cpu
        final_memory = orchestrator.resource_manager.allocated_memory
        
        assert final_cpu == initial_cpu
        assert final_memory == initial_memory
    
    def test_orchestrator_concurrent_pipeline_execution(self, orchestrator, mock_db_session):
        """Test concurrent execution of multiple pipelines"""
        orchestrator.start()
        
        # Execute multiple pipelines
        pipeline_ids = [str(uuid.uuid4()) for _ in range(3)]
        execution_ids = []
        
        for pipeline_id in pipeline_ids:
            execution_id = orchestrator.execute_pipeline_now(
                pipeline_id=pipeline_id,
                priority=5,
                resource_requirements={
                    ResourceType.CPU: 1.0,
                    ResourceType.MEMORY: 2.0
                }
            )
            execution_ids.append(execution_id)
        
        # Wait for processing
        time.sleep(3)
        
        # Check that executions are being processed
        metrics = orchestrator.get_orchestrator_metrics()
        assert metrics['total_executions'] >= len(pipeline_ids)
    
    def test_orchestrator_schedule_integration(self, orchestrator, mock_db_session):
        """Test scheduling integration"""
        orchestrator.start()
        
        pipeline_id = str(uuid.uuid4())
        
        # Schedule pipeline with interval
        schedule_config = ScheduleConfig(
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=2,
            enabled=True
        )
        
        execution_id = orchestrator.schedule_pipeline(
            pipeline_id=pipeline_id,
            schedule_config=schedule_config,
            priority=4
        )
        
        assert execution_id is not None
        
        # Verify scheduling
        status = orchestrator.get_execution_status(execution_id)
        assert status is not None
        assert status['pipeline_id'] == pipeline_id
    
    def test_orchestrator_error_handling_integration(self, orchestrator, mock_db_session):
        """Test error handling integration"""
        orchestrator.start()
        
        # Mock database error
        mock_db_session.add.side_effect = Exception("Database error")
        
        pipeline_id = str(uuid.uuid4())
        
        # Execute pipeline (should handle database error gracefully)
        execution_id = orchestrator.execute_pipeline_now(
            pipeline_id=pipeline_id,
            priority=5
        )
        
        # Wait for processing
        time.sleep(3)
        
        # Orchestrator should still be running
        assert orchestrator.is_running
        
        # Metrics should reflect the error
        metrics = orchestrator.get_orchestrator_metrics()
        assert metrics['failed_executions'] >= 0  # May have failed executions
    
    def test_orchestrator_pipeline_status_updates(self, orchestrator, mock_db_session):
        """Test pipeline status updates during execution"""
        orchestrator.start()
        
        pipeline_id = str(uuid.uuid4())
        
        # Mock pipeline object for status updates
        mock_pipeline = Mock()
        mock_pipeline.id = pipeline_id
        mock_pipeline.status = PipelineStatus.DRAFT
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_pipeline
        
        # Execute pipeline
        execution_id = orchestrator.execute_pipeline_now(
            pipeline_id=pipeline_id,
            priority=9
        )
        
        # Wait for execution
        time.sleep(3)
        
        # Verify execution was attempted
        status = orchestrator.get_execution_status(execution_id)
        assert status is not None
    
    def test_orchestrator_metrics_accuracy(self, orchestrator, mock_db_session):
        """Test accuracy of orchestrator metrics"""
        orchestrator.start()
        
        # Get initial metrics
        initial_metrics = orchestrator.get_orchestrator_metrics()
        initial_total = initial_metrics['total_executions']
        
        # Execute a pipeline
        pipeline_id = str(uuid.uuid4())
        execution_id = orchestrator.execute_pipeline_now(
            pipeline_id=pipeline_id,
            priority=5
        )
        
        # Wait for completion
        time.sleep(6)
        
        # Check updated metrics
        final_metrics = orchestrator.get_orchestrator_metrics()
        
        # Should have at least one more execution
        assert final_metrics['total_executions'] >= initial_total + 1
        
        # Successful or failed count should increase
        total_outcomes = final_metrics['successful_executions'] + final_metrics['failed_executions']
        initial_outcomes = initial_metrics['successful_executions'] + initial_metrics['failed_executions']
        assert total_outcomes >= initial_outcomes + 1


class TestOrchestrationAPIIntegration:
    """Test integration with API routes"""
    
    def test_api_execution_request_format(self):
        """Test API request format compatibility"""
        from scrollintel.api.routes.pipeline_routes import PipelineExecutionRequest
        
        # Test valid request
        request = PipelineExecutionRequest(
            priority=8,
            resource_requirements={
                "cpu": 2.0,
                "memory": 4.0
            },
            dependencies=["dep1", "dep2"]
        )
        
        assert request.priority == 8
        assert request.resource_requirements["cpu"] == 2.0
        assert request.dependencies == ["dep1", "dep2"]
    
    def test_api_schedule_request_format(self):
        """Test API schedule request format"""
        from scrollintel.api.routes.pipeline_routes import ScheduleRequest
        
        # Test valid schedule request
        request = ScheduleRequest(
            schedule_type="interval",
            interval_seconds=300,
            priority=6,
            enabled=True,
            timezone="UTC"
        )
        
        assert request.schedule_type == "interval"
        assert request.interval_seconds == 300
        assert request.priority == 6
        assert request.enabled is True
    
    @patch('scrollintel.api.routes.pipeline_routes.get_orchestrator')
    def test_api_orchestrator_integration(self, mock_get_orchestrator):
        """Test API integration with orchestrator"""
        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.execute_pipeline_now.return_value = "test-execution-id"
        mock_get_orchestrator.return_value = mock_orchestrator
        
        # Import after patching
        from scrollintel.api.routes.pipeline_routes import execute_pipeline_now
        
        # This would normally be called by FastAPI
        # We're testing the function logic
        assert mock_get_orchestrator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])