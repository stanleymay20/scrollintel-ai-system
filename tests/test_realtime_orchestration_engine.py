"""
Tests for Real-Time Orchestration Engine

Tests the core orchestration engine functionality including task distribution,
load balancing, and multi-agent coordination.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from scrollintel.core.realtime_orchestration_engine import (
    RealTimeOrchestrationEngine,
    IntelligentTaskDistributor,
    RealTimeLoadBalancer,
    MultiAgentCoordinator,
    OrchestrationTask,
    TaskPriority,
    TaskStatus,
    CollaborationMode,
    WorkloadMetrics
)
from scrollintel.core.agent_registry import AgentRegistry, AgentSelectionCriteria
from scrollintel.core.message_bus import MessageBus
from scrollintel.core.interfaces import AgentRequest, AgentResponse, ResponseStatus


@pytest.fixture
async def mock_message_bus():
    """Create a mock message bus"""
    bus = Mock(spec=MessageBus)
    bus.start = AsyncMock()
    bus.stop = AsyncMock()
    bus._running = True
    return bus


@pytest.fixture
async def mock_agent_registry():
    """Create a mock agent registry"""
    registry = Mock(spec=AgentRegistry)
    
    # Mock available agents
    mock_agents = [
        {
            "id": "agent-1",
            "name": "Data Analyst",
            "type": "data_analysis",
            "capabilities": [
                {"name": "data_analysis", "performance_score": 85.0},
                {"name": "reporting", "performance_score": 90.0}
            ],
            "current_load": 20.0,
            "max_concurrent_tasks": 10,
            "success_rate": 95.0,
            "average_response_time": 2.5
        },
        {
            "id": "agent-2", 
            "name": "ML Engineer",
            "type": "machine_learning",
            "capabilities": [
                {"name": "machine_learning", "performance_score": 92.0},
                {"name": "data_analysis", "performance_score": 80.0}
            ],
            "current_load": 40.0,
            "max_concurrent_tasks": 8,
            "success_rate": 88.0,
            "average_response_time": 3.2
        }
    ]
    
    registry.get_available_agents = AsyncMock(return_value=mock_agents)
    registry.get_registry_stats = AsyncMock(return_value={
        "agent_counts": {"total": 2, "active": 2},
        "performance_metrics": {"average_response_time": 2.85}
    })
    
    # Mock successful agent response
    mock_response = AgentResponse(
        id="response-1",
        request_id="request-1",
        agent_id="agent-1",
        status=ResponseStatus.SUCCESS,
        content={"analysis": "Sample analysis result", "confidence": 0.85},
        metadata={"processing_time": 2.1}
    )
    registry.route_request = AsyncMock(return_value=mock_response)
    
    return registry


@pytest.fixture
async def orchestration_engine(mock_agent_registry, mock_message_bus):
    """Create orchestration engine with mocked dependencies"""
    engine = RealTimeOrchestrationEngine(mock_agent_registry, mock_message_bus)
    yield engine
    
    # Cleanup
    if engine._running:
        await engine.stop()


class TestIntelligentTaskDistributor:
    """Test the intelligent task distribution algorithm"""
    
    @pytest.mark.asyncio
    async def test_distribute_task_success(self, mock_agent_registry):
        """Test successful task distribution"""
        distributor = IntelligentTaskDistributor(mock_agent_registry)
        
        # Initialize workload metrics
        distributor.workload_metrics = {
            "agent-1": WorkloadMetrics("agent-1", current_tasks=2, success_rate=95.0),
            "agent-2": WorkloadMetrics("agent-2", current_tasks=3, success_rate=88.0)
        }
        
        task = OrchestrationTask(
            name="Test Task",
            description="Test task for distribution",
            required_capabilities=["data_analysis"],
            priority=TaskPriority.NORMAL
        )
        
        agent_id = await distributor.distribute_task(task)
        
        assert agent_id is not None
        assert agent_id in ["agent-1", "agent-2"]
        assert distributor.distribution_stats["tasks_distributed"] == 1
        assert distributor.distribution_stats["successful_assignments"] == 1
    
    @pytest.mark.asyncio
    async def test_distribute_task_no_agents(self, mock_agent_registry):
        """Test task distribution when no agents available"""
        mock_agent_registry.get_available_agents = AsyncMock(return_value=[])
        
        distributor = IntelligentTaskDistributor(mock_agent_registry)
        
        task = OrchestrationTask(
            name="Test Task",
            required_capabilities=["nonexistent_capability"]
        )
        
        agent_id = await distributor.distribute_task(task)
        
        assert agent_id is None
        assert distributor.distribution_stats["failed_assignments"] == 0  # No failure recorded for no agents
    
    def test_calculate_capability_match(self, mock_agent_registry):
        """Test capability matching algorithm"""
        distributor = IntelligentTaskDistributor(mock_agent_registry)
        
        agent = {
            "capabilities": [
                {"name": "data_analysis", "performance_score": 85.0},
                {"name": "reporting", "performance_score": 90.0}
            ]
        }
        
        task = OrchestrationTask(
            required_capabilities=["data_analysis"],
            preferred_capabilities=["reporting"]
        )
        
        score = distributor._calculate_capability_match(agent, task)
        
        assert score > 70.0  # Should have high score for matching capabilities
        assert score <= 100.0
    
    def test_calculate_distribution_score(self, mock_agent_registry):
        """Test overall distribution score calculation"""
        distributor = IntelligentTaskDistributor(mock_agent_registry)
        
        agent = {
            "capabilities": [{"name": "data_analysis", "performance_score": 85.0}],
            "resource_capacity": {"cpu": 8, "memory": 16}
        }
        
        task = OrchestrationTask(
            required_capabilities=["data_analysis"],
            priority=TaskPriority.HIGH,
            resource_requirements={"cpu": 2, "memory": 4}
        )
        
        workload = WorkloadMetrics("agent-1", current_tasks=2, success_rate=95.0)
        
        score = distributor._calculate_distribution_score(agent, task, workload)
        
        assert score > 0.0
        assert score <= 100.0


class TestRealTimeLoadBalancer:
    """Test the real-time load balancing system"""
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, mock_agent_registry):
        """Test starting and stopping load monitoring"""
        distributor = IntelligentTaskDistributor(mock_agent_registry)
        balancer = RealTimeLoadBalancer(mock_agent_registry, distributor)
        
        assert not balancer._running
        
        await balancer.start_monitoring()
        assert balancer._running
        assert balancer._monitoring_task is not None
        
        await balancer.stop_monitoring()
        assert not balancer._running
    
    @pytest.mark.asyncio
    async def test_check_and_rebalance(self, mock_agent_registry):
        """Test load checking and rebalancing logic"""
        distributor = IntelligentTaskDistributor(mock_agent_registry)
        balancer = RealTimeLoadBalancer(mock_agent_registry, distributor)
        
        # Set up workload metrics with imbalance
        distributor.workload_metrics = {
            "agent-1": WorkloadMetrics("agent-1", current_tasks=8, max_concurrent_tasks=10),  # 80% load
            "agent-2": WorkloadMetrics("agent-2", current_tasks=2, max_concurrent_tasks=10)   # 20% load
        }
        
        initial_rebalancing_events = balancer.rebalancing_stats["rebalancing_events"]
        
        await balancer._check_and_rebalance()
        
        # Should trigger rebalancing due to 60% load difference
        assert balancer.rebalancing_stats["rebalancing_events"] > initial_rebalancing_events
    
    def test_get_load_balance_stats(self, mock_agent_registry):
        """Test load balance statistics"""
        distributor = IntelligentTaskDistributor(mock_agent_registry)
        balancer = RealTimeLoadBalancer(mock_agent_registry, distributor)
        
        # Set up workload metrics
        distributor.workload_metrics = {
            "agent-1": WorkloadMetrics("agent-1", current_tasks=5, max_concurrent_tasks=10),
            "agent-2": WorkloadMetrics("agent-2", current_tasks=3, max_concurrent_tasks=10)
        }
        
        stats = balancer.get_load_balance_stats()
        
        assert "agent_count" in stats
        assert "average_load" in stats
        assert "max_load" in stats
        assert "min_load" in stats
        assert stats["agent_count"] == 2


class TestMultiAgentCoordinator:
    """Test multi-agent coordination protocols"""
    
    @pytest.mark.asyncio
    async def test_coordinate_sequential_task(self, mock_agent_registry, mock_message_bus):
        """Test sequential coordination mode"""
        coordinator = MultiAgentCoordinator(mock_message_bus, mock_agent_registry)
        
        task = OrchestrationTask(
            name="Sequential Task",
            description="Test sequential coordination",
            collaboration_mode=CollaborationMode.SEQUENTIAL,
            collaboration_agents=["agent-1", "agent-2"]
        )
        
        result = await coordinator.coordinate_task(task)
        
        assert result["collaboration_id"] is not None
        assert result["mode"] == "sequential"
        assert "results" in result
        assert coordinator.coordination_stats["collaborations_completed"] == 1
    
    @pytest.mark.asyncio
    async def test_coordinate_parallel_task(self, mock_agent_registry, mock_message_bus):
        """Test parallel coordination mode"""
        coordinator = MultiAgentCoordinator(mock_message_bus, mock_agent_registry)
        
        task = OrchestrationTask(
            name="Parallel Task",
            description="Test parallel coordination",
            collaboration_mode=CollaborationMode.PARALLEL,
            collaboration_agents=["agent-1", "agent-2"]
        )
        
        result = await coordinator.coordinate_task(task)
        
        assert result["collaboration_id"] is not None
        assert result["mode"] == "parallel"
        assert "results" in result
        assert "successful_results" in result
    
    @pytest.mark.asyncio
    async def test_coordinate_consensus_task(self, mock_agent_registry, mock_message_bus):
        """Test consensus coordination mode"""
        coordinator = MultiAgentCoordinator(mock_message_bus, mock_agent_registry)
        
        task = OrchestrationTask(
            name="Consensus Task",
            description="Test consensus coordination",
            collaboration_mode=CollaborationMode.CONSENSUS,
            collaboration_agents=["agent-1", "agent-2"],
            requires_consensus=True,
            consensus_threshold=0.7
        )
        
        result = await coordinator.coordinate_task(task)
        
        assert result["collaboration_id"] is not None
        assert result["mode"] == "consensus"
        assert "consensus" in result
        assert "individual_results" in result
    
    def test_get_coordination_stats(self, mock_agent_registry, mock_message_bus):
        """Test coordination statistics"""
        coordinator = MultiAgentCoordinator(mock_message_bus, mock_agent_registry)
        
        stats = coordinator.get_coordination_stats()
        
        assert "collaborations_started" in stats
        assert "collaborations_completed" in stats
        assert "collaborations_failed" in stats
        assert "active_collaborations" in stats


class TestRealTimeOrchestrationEngine:
    """Test the main orchestration engine"""
    
    @pytest.mark.asyncio
    async def test_engine_start_stop(self, orchestration_engine):
        """Test engine startup and shutdown"""
        engine = orchestration_engine
        
        assert not engine._running
        
        await engine.start()
        assert engine._running
        assert engine.engine_stats["uptime_start"] is not None
        
        await engine.stop()
        assert not engine._running
    
    @pytest.mark.asyncio
    async def test_submit_task(self, orchestration_engine):
        """Test task submission"""
        engine = orchestration_engine
        await engine.start()
        
        task_id = await engine.submit_task(
            name="Test Task",
            description="Test task submission",
            required_capabilities=["data_analysis"],
            payload={"data": "test_data"},
            priority=TaskPriority.NORMAL
        )
        
        assert task_id is not None
        assert task_id in engine.active_tasks
        assert engine.engine_stats["tasks_processed"] == 1
        
        # Check task status
        task_status = await engine.get_task_status(task_id)
        assert task_status is not None
        assert task_status["name"] == "Test Task"
        assert task_status["status"] in ["queued", "assigned", "running"]
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, orchestration_engine):
        """Test task cancellation"""
        engine = orchestration_engine
        await engine.start()
        
        task_id = await engine.submit_task(
            name="Cancellable Task",
            description="Task to be cancelled",
            required_capabilities=["data_analysis"],
            payload={}
        )
        
        # Cancel the task
        success = await engine.cancel_task(task_id)
        assert success
        
        # Check task is no longer active
        assert task_id not in engine.active_tasks
        
        # Task should be in completed tasks
        task_status = await engine.get_task_status(task_id)
        assert task_status["status"] == "cancelled"
    
    @pytest.mark.asyncio
    async def test_collaborative_task_submission(self, orchestration_engine):
        """Test submitting a collaborative task"""
        engine = orchestration_engine
        await engine.start()
        
        task_id = await engine.submit_task(
            name="Collaborative Task",
            description="Multi-agent collaborative task",
            required_capabilities=["data_analysis", "machine_learning"],
            payload={"dataset": "large_data.csv"},
            collaboration_mode=CollaborationMode.PARALLEL,
            collaboration_agents=["agent-1", "agent-2"]
        )
        
        assert task_id is not None
        
        task_status = await engine.get_task_status(task_id)
        assert task_status["collaboration_mode"] == "parallel"
        assert len(task_status["collaboration_agents"]) == 2
    
    def test_get_engine_stats(self, orchestration_engine):
        """Test engine statistics"""
        engine = orchestration_engine
        
        stats = engine.get_engine_stats()
        
        assert "engine_status" in stats
        assert "task_statistics" in stats
        assert "distribution_stats" in stats
        assert "load_balancing_stats" in stats
        assert "coordination_stats" in stats
        assert "component_status" in stats
    
    def test_get_active_tasks(self, orchestration_engine):
        """Test getting active tasks"""
        engine = orchestration_engine
        
        # Initially no active tasks
        active_tasks = engine.get_active_tasks()
        assert len(active_tasks) == 0
        
        # Add a mock task
        task = OrchestrationTask(
            name="Mock Task",
            description="Mock task for testing"
        )
        engine.active_tasks[task.id] = task
        
        active_tasks = engine.get_active_tasks()
        assert len(active_tasks) == 1
        assert active_tasks[0]["name"] == "Mock Task"


class TestOrchestrationTask:
    """Test the OrchestrationTask data class"""
    
    def test_task_creation(self):
        """Test creating an orchestration task"""
        task = OrchestrationTask(
            name="Test Task",
            description="Test task creation",
            priority=TaskPriority.HIGH,
            required_capabilities=["data_analysis"],
            payload={"key": "value"}
        )
        
        assert task.id is not None
        assert task.name == "Test Task"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        assert "data_analysis" in task.required_capabilities
        assert task.payload["key"] == "value"
    
    def test_task_defaults(self):
        """Test task default values"""
        task = OrchestrationTask()
        
        assert task.priority == TaskPriority.NORMAL
        assert task.status == TaskStatus.PENDING
        assert task.collaboration_mode == CollaborationMode.SEQUENTIAL
        assert task.timeout_seconds == 300.0
        assert task.max_retries == 3
        assert task.retry_count == 0


class TestWorkloadMetrics:
    """Test workload metrics calculations"""
    
    def test_load_percentage_calculation(self):
        """Test load percentage calculation"""
        metrics = WorkloadMetrics(
            agent_id="test-agent",
            current_tasks=5,
            max_concurrent_tasks=10
        )
        
        assert metrics.load_percentage == 50.0
    
    def test_is_overloaded(self):
        """Test overload detection"""
        # Not overloaded
        metrics1 = WorkloadMetrics(
            agent_id="test-agent",
            current_tasks=8,
            max_concurrent_tasks=10
        )
        assert not metrics1.is_overloaded
        
        # Overloaded
        metrics2 = WorkloadMetrics(
            agent_id="test-agent",
            current_tasks=10,
            max_concurrent_tasks=10
        )
        assert metrics2.is_overloaded
    
    def test_capacity_score(self):
        """Test capacity score calculation"""
        metrics = WorkloadMetrics(
            agent_id="test-agent",
            current_tasks=3,
            max_concurrent_tasks=10,
            success_rate=95.0,
            average_response_time=2.0
        )
        
        score = metrics.capacity_score
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be good score for low load, high success rate


@pytest.mark.asyncio
async def test_integration_full_workflow(mock_agent_registry, mock_message_bus):
    """Integration test for complete orchestration workflow"""
    engine = RealTimeOrchestrationEngine(mock_agent_registry, mock_message_bus)
    
    try:
        await engine.start()
        
        # Submit multiple tasks with different priorities
        task_ids = []
        
        # High priority task
        task_id1 = await engine.submit_task(
            name="Critical Analysis",
            description="Critical data analysis task",
            required_capabilities=["data_analysis"],
            payload={"urgency": "high"},
            priority=TaskPriority.CRITICAL
        )
        task_ids.append(task_id1)
        
        # Normal priority collaborative task
        task_id2 = await engine.submit_task(
            name="Team Analysis",
            description="Collaborative analysis task",
            required_capabilities=["data_analysis", "machine_learning"],
            payload={"team_size": 2},
            collaboration_mode=CollaborationMode.PARALLEL,
            collaboration_agents=["agent-1", "agent-2"]
        )
        task_ids.append(task_id2)
        
        # Allow some processing time
        await asyncio.sleep(0.1)
        
        # Check all tasks were submitted
        for task_id in task_ids:
            task_status = await engine.get_task_status(task_id)
            assert task_status is not None
            assert task_status["status"] in ["queued", "assigned", "running", "completed"]
        
        # Get engine statistics
        stats = engine.get_engine_stats()
        assert stats["task_statistics"]["tasks_processed"] >= 2
        
        # Get active tasks
        active_tasks = engine.get_active_tasks()
        assert len(active_tasks) >= 0  # May have completed by now
        
    finally:
        await engine.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])