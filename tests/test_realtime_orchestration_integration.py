"""
Integration Tests for Real-Time Orchestration Engine

Tests integration with existing ScrollIntel components including agent registry,
message bus, and API routes.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch

from scrollintel.api.routes.realtime_orchestration_routes import router, get_orchestration_engine
from scrollintel.core.realtime_orchestration_engine import (
    RealTimeOrchestrationEngine,
    TaskPriority,
    CollaborationMode
)
from scrollintel.core.agent_registry import AgentRegistry
from scrollintel.core.message_bus import MessageBus


@pytest.fixture
def mock_orchestration_engine():
    """Create a mock orchestration engine for API testing"""
    engine = Mock(spec=RealTimeOrchestrationEngine)
    
    # Mock methods
    engine.submit_task = AsyncMock(return_value="task-123")
    engine.get_task_status = AsyncMock(return_value={
        "id": "task-123",
        "name": "Test Task",
        "description": "Test task description",
        "status": "completed",
        "priority": "NORMAL",
        "assigned_agent_id": "agent-1",
        "collaboration_mode": "sequential",
        "collaboration_agents": [],
        "created_at": "2024-01-01T00:00:00",
        "queued_at": "2024-01-01T00:00:01",
        "assigned_at": "2024-01-01T00:00:02",
        "started_at": "2024-01-01T00:00:03",
        "completed_at": "2024-01-01T00:00:10",
        "retry_count": 0,
        "error_message": None,
        "execution_metrics": {"execution_time_seconds": 7.0}
    })
    engine.cancel_task = AsyncMock(return_value=True)
    engine.get_active_tasks = Mock(return_value=[
        {
            "id": "task-456",
            "name": "Active Task",
            "status": "running",
            "priority": "HIGH",
            "assigned_agent_id": "agent-2",
            "created_at": "2024-01-01T00:01:00",
            "collaboration_mode": "parallel"
        }
    ])
    engine.get_engine_stats = Mock(return_value={
        "engine_status": {"running": True, "uptime_seconds": 3600},
        "task_statistics": {
            "tasks_processed": 10,
            "tasks_completed": 8,
            "tasks_failed": 1,
            "average_processing_time": 5.2,
            "concurrent_tasks_peak": 3,
            "active_tasks": 1,
            "queued_tasks": 0,
            "completed_tasks_cached": 8
        },
        "distribution_stats": {
            "tasks_distributed": 10,
            "successful_assignments": 9,
            "failed_assignments": 1
        },
        "load_balancing_stats": {
            "agent_count": 3,
            "average_load": 45.0,
            "max_load": 60.0,
            "min_load": 30.0,
            "rebalancing_stats": {"rebalancing_events": 2}
        },
        "coordination_stats": {
            "collaborations_started": 3,
            "collaborations_completed": 2,
            "collaborations_failed": 0,
            "active_collaborations": 1
        },
        "component_status": {
            "task_distributor": "active",
            "load_balancer": "active",
            "coordinator": "active",
            "message_bus": "active"
        }
    })
    
    # Mock load balancer and coordinator
    engine.load_balancer = Mock()
    engine.load_balancer.get_load_balance_stats = Mock(return_value={
        "agent_count": 3,
        "average_load": 45.0,
        "max_load": 60.0,
        "min_load": 30.0,
        "rebalancing_stats": {"rebalancing_events": 2}
    })
    
    engine.coordinator = Mock()
    engine.coordinator.get_coordination_stats = Mock(return_value={
        "collaborations_started": 3,
        "collaborations_completed": 2,
        "collaborations_failed": 0,
        "active_collaborations": 1
    })
    
    return engine


@pytest.fixture
def test_client(mock_orchestration_engine):
    """Create test client with mocked dependencies"""
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(router)
    
    # Override dependency
    app.dependency_overrides[get_orchestration_engine] = lambda: mock_orchestration_engine
    
    return TestClient(app)


class TestOrchestrationAPI:
    """Test the orchestration API endpoints"""
    
    def test_submit_task(self, test_client, mock_orchestration_engine):
        """Test task submission endpoint"""
        task_data = {
            "name": "Test API Task",
            "description": "Task submitted via API",
            "required_capabilities": ["data_analysis"],
            "payload": {"test": "data"},
            "priority": "HIGH",
            "collaboration_mode": "SEQUENTIAL",
            "timeout_seconds": 300.0
        }
        
        response = test_client.post("/api/v1/orchestration/tasks", json=task_data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["task_id"] == "task-123"
        assert result["status"] == "submitted"
        
        # Verify engine was called correctly
        mock_orchestration_engine.submit_task.assert_called_once()
        call_args = mock_orchestration_engine.submit_task.call_args
        assert call_args.kwargs["name"] == "Test API Task"
        assert call_args.kwargs["priority"] == TaskPriority.HIGH
        assert call_args.kwargs["collaboration_mode"] == CollaborationMode.SEQUENTIAL
    
    def test_submit_task_invalid_priority(self, test_client):
        """Test task submission with invalid priority"""
        task_data = {
            "name": "Invalid Priority Task",
            "description": "Task with invalid priority",
            "required_capabilities": ["data_analysis"],
            "priority": "INVALID_PRIORITY"
        }
        
        response = test_client.post("/api/v1/orchestration/tasks", json=task_data)
        
        assert response.status_code == 400
        assert "Invalid priority" in response.json()["detail"]
    
    def test_get_task_status(self, test_client, mock_orchestration_engine):
        """Test getting task status"""
        response = test_client.get("/api/v1/orchestration/tasks/task-123")
        
        assert response.status_code == 200
        result = response.json()
        assert result["id"] == "task-123"
        assert result["name"] == "Test Task"
        assert result["status"] == "completed"
        
        mock_orchestration_engine.get_task_status.assert_called_once_with("task-123")
    
    def test_get_task_status_not_found(self, test_client, mock_orchestration_engine):
        """Test getting status of non-existent task"""
        mock_orchestration_engine.get_task_status.return_value = None
        
        response = test_client.get("/api/v1/orchestration/tasks/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_cancel_task(self, test_client, mock_orchestration_engine):
        """Test task cancellation"""
        response = test_client.delete("/api/v1/orchestration/tasks/task-123")
        
        assert response.status_code == 200
        result = response.json()
        assert result["task_id"] == "task-123"
        assert result["status"] == "cancelled"
        
        mock_orchestration_engine.cancel_task.assert_called_once_with("task-123")
    
    def test_cancel_task_not_found(self, test_client, mock_orchestration_engine):
        """Test cancelling non-existent task"""
        mock_orchestration_engine.cancel_task.return_value = False
        
        response = test_client.delete("/api/v1/orchestration/tasks/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_get_active_tasks(self, test_client, mock_orchestration_engine):
        """Test getting active tasks"""
        response = test_client.get("/api/v1/orchestration/tasks")
        
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 1
        assert result[0]["id"] == "task-456"
        assert result[0]["name"] == "Active Task"
    
    def test_get_engine_stats(self, test_client, mock_orchestration_engine):
        """Test getting engine statistics"""
        response = test_client.get("/api/v1/orchestration/stats")
        
        assert response.status_code == 200
        result = response.json()
        assert "engine_status" in result
        assert "task_statistics" in result
        assert result["task_statistics"]["tasks_processed"] == 10
        assert result["engine_status"]["running"] is True
    
    def test_get_load_balancer_stats(self, test_client, mock_orchestration_engine):
        """Test getting load balancer statistics"""
        response = test_client.get("/api/v1/orchestration/load-balancer/stats")
        
        assert response.status_code == 200
        result = response.json()
        assert result["agent_count"] == 3
        assert result["average_load"] == 45.0
    
    def test_get_coordinator_stats(self, test_client, mock_orchestration_engine):
        """Test getting coordinator statistics"""
        response = test_client.get("/api/v1/orchestration/coordinator/stats")
        
        assert response.status_code == 200
        result = response.json()
        assert result["collaborations_started"] == 3
        assert result["collaborations_completed"] == 2
    
    def test_submit_sample_task(self, test_client, mock_orchestration_engine):
        """Test submitting sample task"""
        response = test_client.post("/api/v1/orchestration/test/submit-sample-task")
        
        assert response.status_code == 200
        result = response.json()
        assert result["task_id"] == "task-123"
        assert result["type"] == "sample_task"
        
        # Verify sample task parameters
        mock_orchestration_engine.submit_task.assert_called_once()
        call_args = mock_orchestration_engine.submit_task.call_args
        assert "Sample Analysis Task" in call_args.kwargs["name"]
        assert "data_analysis" in call_args.kwargs["required_capabilities"]
    
    def test_submit_collaborative_task(self, test_client, mock_orchestration_engine):
        """Test submitting collaborative task"""
        response = test_client.post("/api/v1/orchestration/test/submit-collaborative-task")
        
        assert response.status_code == 200
        result = response.json()
        assert result["task_id"] == "task-123"
        assert result["type"] == "collaborative_task"
        
        # Verify collaborative task parameters
        mock_orchestration_engine.submit_task.assert_called_once()
        call_args = mock_orchestration_engine.submit_task.call_args
        assert call_args.kwargs["collaboration_mode"] == CollaborationMode.PARALLEL
        assert len(call_args.kwargs["collaboration_agents"]) == 3


class TestOrchestrationEngineIntegration:
    """Test integration between orchestration engine and other components"""
    
    @pytest.mark.asyncio
    async def test_engine_with_real_message_bus(self):
        """Test orchestration engine with real message bus"""
        from scrollintel.core.message_bus import MessageBus
        
        # Create real message bus
        message_bus = MessageBus()
        await message_bus.start()
        
        try:
            # Create mock agent registry
            agent_registry = Mock(spec=AgentRegistry)
            agent_registry.get_available_agents = AsyncMock(return_value=[])
            agent_registry.get_registry_stats = AsyncMock(return_value={
                "agent_counts": {"total": 0, "active": 0}
            })
            
            # Create orchestration engine
            engine = RealTimeOrchestrationEngine(agent_registry, message_bus)
            
            # Test engine lifecycle
            await engine.start()
            assert engine._running
            
            # Test basic functionality
            stats = engine.get_engine_stats()
            assert "engine_status" in stats
            assert stats["engine_status"]["running"] is True
            
            await engine.stop()
            assert not engine._running
            
        finally:
            await message_bus.stop()
    
    @pytest.mark.asyncio
    async def test_task_distribution_integration(self):
        """Test task distribution with mock agents"""
        from scrollintel.core.message_bus import MessageBus
        from scrollintel.core.interfaces import AgentResponse, ResponseStatus
        
        message_bus = MessageBus()
        await message_bus.start()
        
        try:
            # Create agent registry with mock agents
            agent_registry = Mock(spec=AgentRegistry)
            agent_registry.get_available_agents = AsyncMock(return_value=[
                {
                    "id": "test-agent-1",
                    "name": "Test Agent",
                    "capabilities": [{"name": "data_analysis", "performance_score": 90.0}],
                    "current_load": 30.0,
                    "success_rate": 95.0,
                    "average_response_time": 2.0
                }
            ])
            agent_registry.get_registry_stats = AsyncMock(return_value={
                "agent_counts": {"total": 1, "active": 1}
            })
            
            # Mock successful response
            mock_response = AgentResponse(
                id="response-1",
                request_id="request-1", 
                agent_id="test-agent-1",
                status=ResponseStatus.SUCCESS,
                content={"result": "success"}
            )
            agent_registry.route_request = AsyncMock(return_value=mock_response)
            
            # Create and start engine
            engine = RealTimeOrchestrationEngine(agent_registry, message_bus)
            await engine.start()
            
            try:
                # Submit a task
                task_id = await engine.submit_task(
                    name="Integration Test Task",
                    description="Test task for integration",
                    required_capabilities=["data_analysis"],
                    payload={"test": "data"}
                )
                
                assert task_id is not None
                assert task_id in engine.active_tasks
                
                # Allow processing time
                await asyncio.sleep(0.1)
                
                # Check task was processed
                task_status = await engine.get_task_status(task_id)
                assert task_status is not None
                
            finally:
                await engine.stop()
                
        finally:
            await message_bus.stop()
    
    @pytest.mark.asyncio
    async def test_load_balancer_integration(self):
        """Test load balancer integration with task distributor"""
        from scrollintel.core.message_bus import MessageBus
        
        message_bus = MessageBus()
        agent_registry = Mock(spec=AgentRegistry)
        agent_registry.get_available_agents = AsyncMock(return_value=[])
        agent_registry.get_registry_stats = AsyncMock(return_value={
            "agent_counts": {"total": 0, "active": 0}
        })
        
        engine = RealTimeOrchestrationEngine(agent_registry, message_bus)
        
        # Test load balancer can be started/stopped
        await engine.load_balancer.start_monitoring()
        assert engine.load_balancer._running
        
        await engine.load_balancer.stop_monitoring()
        assert not engine.load_balancer._running
        
        # Test load balancer stats
        stats = engine.load_balancer.get_load_balance_stats()
        assert "error" in stats  # No workload metrics available
    
    @pytest.mark.asyncio
    async def test_coordinator_integration(self):
        """Test multi-agent coordinator integration"""
        from scrollintel.core.message_bus import MessageBus
        
        message_bus = MessageBus()
        agent_registry = Mock(spec=AgentRegistry)
        
        engine = RealTimeOrchestrationEngine(agent_registry, message_bus)
        
        # Test coordinator stats
        stats = engine.coordinator.get_coordination_stats()
        assert "collaborations_started" in stats
        assert "collaborations_completed" in stats
        assert "active_collaborations" in stats
        
        # Test coordinator has access to message bus and registry
        assert engine.coordinator.message_bus is message_bus
        assert engine.coordinator.agent_registry is agent_registry


@pytest.mark.asyncio
async def test_full_system_integration():
    """Test full system integration with all components"""
    from scrollintel.core.message_bus import MessageBus, get_message_bus
    from scrollintel.core.agent_registry import AgentRegistry
    
    # Use global message bus
    message_bus = get_message_bus()
    await message_bus.start()
    
    try:
        # Create agent registry
        agent_registry = AgentRegistry(message_bus)
        
        # Create orchestration engine
        engine = RealTimeOrchestrationEngine(agent_registry, message_bus)
        await engine.start()
        
        try:
            # Test engine is properly initialized
            assert engine._running
            assert engine.agent_registry is agent_registry
            assert engine.message_bus is message_bus
            
            # Test all components are connected
            assert engine.task_distributor.agent_registry is agent_registry
            assert engine.load_balancer.agent_registry is agent_registry
            assert engine.coordinator.message_bus is message_bus
            assert engine.coordinator.agent_registry is agent_registry
            
            # Test engine statistics
            stats = engine.get_engine_stats()
            assert stats["engine_status"]["running"] is True
            assert "component_status" in stats
            
            # Test component integration
            assert engine.load_balancer.task_distributor is engine.task_distributor
            
        finally:
            await engine.stop()
            
    finally:
        await message_bus.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])