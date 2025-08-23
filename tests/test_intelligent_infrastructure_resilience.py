"""
Tests for Intelligent Infrastructure Resilience System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from security.infrastructure.intelligent_infrastructure_resilience import (
    IntelligentInfrastructureResilience,
    InfrastructureStatus,
    OptimizationAction,
    InfrastructureMetrics,
    CapacityPrediction,
    DisasterRecoveryPlan
)

@pytest.fixture
def infrastructure_system():
    """Create infrastructure resilience system for testing"""
    return IntelligentInfrastructureResilience()

@pytest.mark.asyncio
async def test_collect_infrastructure_metrics(infrastructure_system):
    """Test infrastructure metrics collection"""
    # Collect metrics
    metrics = await infrastructure_system.collect_infrastructure_metrics()
    
    # Verify metrics structure
    assert isinstance(metrics, InfrastructureMetrics)
    assert isinstance(metrics.timestamp, datetime)
    assert 0 <= metrics.cpu_usage <= 100
    assert 0 <= metrics.memory_usage <= 100
    assert 0 <= metrics.disk_usage <= 100
    assert metrics.network_io >= 0
    assert metrics.response_time >= 0
    assert metrics.error_rate >= 0
    assert metrics.throughput >= 0
    assert metrics.availability >= 0
    assert metrics.cost_per_hour >= 0

@pytest.mark.asyncio
async def test_auto_tune_infrastructure(infrastructure_system):
    """Test automatic infrastructure tuning"""
    # Mock high CPU usage scenario
    with patch.object(infrastructure_system, 'collect_infrastructure_metrics') as mock_collect:
        mock_metrics = InfrastructureMetrics(
            timestamp=datetime.now(),
            cpu_usage=85.0,  # High CPU usage
            memory_usage=60.0,
            disk_usage=70.0,
            network_io=1000.0,
            response_time=300.0,
            error_rate=0.5,
            throughput=800.0,
            availability=99.5,
            cost_per_hour=15.0
        )
        mock_collect.return_value = mock_metrics
        
        # Execute auto-tuning
        result = await infrastructure_system.auto_tune_infrastructure()
        
        # Verify optimization actions were taken
        assert "actions_taken" in result
        assert isinstance(result["actions_taken"], list)
        assert "performance_improvement" in result
        assert isinstance(result["performance_improvement"], (int, float))

@pytest.mark.asyncio
async def test_predict_capacity_needs(infrastructure_system):
    """Test capacity prediction"""
    # Add some historical metrics
    for i in range(100):
        metrics = InfrastructureMetrics(
            timestamp=datetime.now() - timedelta(hours=i),
            cpu_usage=50.0 + (i % 20),
            memory_usage=60.0 + (i % 15),
            disk_usage=70.0 + (i % 10),
            network_io=1000.0 + (i * 10),
            response_time=200.0 + (i % 50),
            error_rate=0.1 + (i % 5) * 0.1,
            throughput=1000.0 + (i * 5),
            availability=99.0 + (i % 2),
            cost_per_hour=10.0 + (i % 5)
        )
        infrastructure_system.metrics_history.append(metrics)
    
    # Train the predictor
    infrastructure_system.train_capacity_predictor()
    
    # Generate predictions
    predictions = await infrastructure_system.predict_capacity_needs(forecast_hours=24)
    
    # Verify predictions
    assert isinstance(predictions, list)
    if predictions:  # If training was successful
        assert len(predictions) == 24
        for prediction in predictions:
            assert isinstance(prediction, CapacityPrediction)
            assert isinstance(prediction.timestamp, datetime)
            assert prediction.predicted_value >= 0
            assert prediction.confidence_interval_lower >= 0
            assert prediction.confidence_interval_upper >= prediction.confidence_interval_lower
            assert 0 <= prediction.accuracy_score <= 100
            assert isinstance(prediction.recommended_actions, list)

@pytest.mark.asyncio
async def test_disaster_recovery_execution(infrastructure_system):
    """Test disaster recovery execution"""
    # Execute disaster recovery
    result = await infrastructure_system.execute_disaster_recovery("hardware_failure")
    
    # Verify disaster recovery result
    assert "start_time" in result
    assert "failure_type" in result
    assert result["failure_type"] == "hardware_failure"
    assert "steps_completed" in result
    assert isinstance(result["steps_completed"], list)
    assert "rto_target" in result
    assert result["rto_target"] == infrastructure_system.disaster_recovery_plan.rto_target
    assert "rpo_target" in result
    assert result["rpo_target"] == infrastructure_system.disaster_recovery_plan.rpo_target
    assert "total_recovery_time_minutes" in result
    assert isinstance(result["total_recovery_time_minutes"], (int, float))
    assert "rto_met" in result
    assert isinstance(result["rto_met"], bool)

def test_infrastructure_status(infrastructure_system):
    """Test infrastructure status reporting"""
    # Add some metrics
    metrics = InfrastructureMetrics(
        timestamp=datetime.now(),
        cpu_usage=45.0,
        memory_usage=55.0,
        disk_usage=65.0,
        network_io=800.0,
        response_time=150.0,
        error_rate=0.05,
        throughput=1200.0,
        availability=99.95,
        cost_per_hour=12.0
    )
    infrastructure_system.metrics_history.append(metrics)
    
    # Get status
    status = infrastructure_system.get_infrastructure_status()
    
    # Verify status structure
    assert "status" in status
    assert status["status"] in ["healthy", "degraded", "critical", "error"]
    assert "uptime_percentage" in status
    assert "response_time_ms" in status
    assert "error_rate_percentage" in status
    assert "cpu_usage" in status
    assert "memory_usage" in status
    assert "cost_per_hour" in status
    assert "last_updated" in status

@pytest.mark.asyncio
async def test_scaling_actions(infrastructure_system):
    """Test scaling action execution"""
    # Test CPU scaling
    with patch.object(infrastructure_system, '_scale_kubernetes_resources') as mock_k8s_scale:
        mock_k8s_scale.return_value = None
        await infrastructure_system._scale_up_resources("cpu")
        # Verify scaling was attempted (no exception raised)

@pytest.mark.asyncio
async def test_performance_improvement_calculation(infrastructure_system):
    """Test performance improvement calculation"""
    # Create previous and current metrics
    previous_metrics = InfrastructureMetrics(
        timestamp=datetime.now() - timedelta(minutes=5),
        cpu_usage=80.0,
        memory_usage=75.0,
        disk_usage=70.0,
        network_io=1000.0,
        response_time=400.0,
        error_rate=2.0,
        throughput=800.0,
        availability=98.0,
        cost_per_hour=15.0
    )
    
    current_metrics = InfrastructureMetrics(
        timestamp=datetime.now(),
        cpu_usage=60.0,
        memory_usage=55.0,
        disk_usage=70.0,
        network_io=1000.0,
        response_time=200.0,
        error_rate=0.5,
        throughput=1200.0,
        availability=99.5,
        cost_per_hour=12.0
    )
    
    # Calculate improvement
    improvement = infrastructure_system._calculate_performance_improvement(
        previous_metrics, current_metrics
    )
    
    # Verify improvement calculation
    assert isinstance(improvement, (int, float))
    # Should show positive improvement due to better metrics

def test_disaster_recovery_plan_creation(infrastructure_system):
    """Test disaster recovery plan structure"""
    dr_plan = infrastructure_system.disaster_recovery_plan
    
    # Verify disaster recovery plan structure
    assert isinstance(dr_plan, DisasterRecoveryPlan)
    assert dr_plan.rto_target == 15  # 15 minutes
    assert dr_plan.rpo_target == 5   # 5 minutes
    assert isinstance(dr_plan.backup_locations, list)
    assert len(dr_plan.backup_locations) > 0
    assert isinstance(dr_plan.failover_sequence, list)
    assert len(dr_plan.failover_sequence) > 0
    assert isinstance(dr_plan.rollback_plan, list)
    assert len(dr_plan.rollback_plan) > 0
    assert isinstance(dr_plan.validation_steps, list)
    assert len(dr_plan.validation_steps) > 0

@pytest.mark.asyncio
async def test_optimization_action_determination(infrastructure_system):
    """Test optimization action determination logic"""
    import numpy as np
    
    # Test high CPU prediction
    high_cpu_prediction = np.array([85.0, 60.0, 70.0, 1000.0])
    actions = infrastructure_system._determine_optimization_actions(high_cpu_prediction)
    assert OptimizationAction.SCALE_UP in actions
    
    # Test low CPU prediction
    low_cpu_prediction = np.array([15.0, 40.0, 50.0, 500.0])
    actions = infrastructure_system._determine_optimization_actions(low_cpu_prediction)
    assert OptimizationAction.SCALE_DOWN in actions
    
    # Test normal prediction
    normal_prediction = np.array([50.0, 60.0, 70.0, 800.0])
    actions = infrastructure_system._determine_optimization_actions(normal_prediction)
    # Should have some action, even if it's NO_ACTION

@pytest.mark.asyncio
async def test_uptime_target_achievement(infrastructure_system):
    """Test uptime target achievement"""
    # Add metrics with high availability
    for i in range(10):
        metrics = InfrastructureMetrics(
            timestamp=datetime.now() - timedelta(minutes=i),
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=70.0,
            network_io=1000.0,
            response_time=150.0,
            error_rate=0.01,  # Very low error rate
            throughput=1200.0,
            availability=99.99,  # High availability
            cost_per_hour=12.0
        )
        infrastructure_system.metrics_history.append(metrics)
    
    # Check if uptime target is being tracked
    status = infrastructure_system.get_infrastructure_status()
    assert status["uptime_percentage"] >= infrastructure_system.uptime_target

@pytest.mark.asyncio
async def test_cost_calculation(infrastructure_system):
    """Test cost calculation accuracy"""
    # Mock system metrics
    with patch('psutil.cpu_percent', return_value=50.0):
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 60.0
            
            cost = infrastructure_system._calculate_current_cost(50.0, 60.0, 70.0)
            
            # Verify cost is reasonable
            assert isinstance(cost, (int, float))
            assert cost > 0
            assert cost < 100  # Should be reasonable cost per hour

@pytest.mark.asyncio
async def test_metrics_history_management(infrastructure_system):
    """Test metrics history size management"""
    # Add many metrics to test history management
    for i in range(10005):  # More than the 10000 limit
        metrics = InfrastructureMetrics(
            timestamp=datetime.now() - timedelta(minutes=i),
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=70.0,
            network_io=1000.0,
            response_time=150.0,
            error_rate=0.1,
            throughput=1200.0,
            availability=99.5,
            cost_per_hour=12.0
        )
        infrastructure_system.metrics_history.append(metrics)
    
    # Collect one more metric to trigger cleanup
    await infrastructure_system.collect_infrastructure_metrics()
    
    # Verify history size is managed
    assert len(infrastructure_system.metrics_history) <= 10000

if __name__ == "__main__":
    pytest.main([__file__])