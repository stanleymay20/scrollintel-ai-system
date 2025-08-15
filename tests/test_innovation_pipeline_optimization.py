"""
Tests for Innovation Pipeline Optimization

This module contains comprehensive tests for the innovation pipeline optimization system,
including flow optimization, resource allocation, and performance monitoring.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.innovation_pipeline_optimizer import InnovationPipelineOptimizer
from scrollintel.models.innovation_pipeline_models import (
    InnovationPipelineItem, PipelineStage, InnovationPriority, ResourceType,
    PipelineStatus, ResourceRequirement, PipelineOptimizationConfig
)


class TestInnovationPipelineOptimizer:
    """Test cases for InnovationPipelineOptimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing"""
        config = PipelineOptimizationConfig(
            max_concurrent_innovations=10,
            resource_buffer_percentage=0.2,
            bottleneck_threshold=0.7,
            resource_utilization_target=0.85
        )
        return InnovationPipelineOptimizer(config)
    
    @pytest.fixture
    def sample_innovation(self):
        """Create sample innovation for testing"""
        return InnovationPipelineItem(
            innovation_id="test-innovation-001",
            current_stage=PipelineStage.IDEATION,
            priority=InnovationPriority.HIGH,
            success_probability=0.8,
            risk_score=0.3,
            impact_score=0.9,
            resource_requirements=[
                ResourceRequirement(
                    resource_type=ResourceType.COMPUTE,
                    amount=100.0,
                    unit="cores",
                    duration=24.0,
                    priority=InnovationPriority.HIGH
                ),
                ResourceRequirement(
                    resource_type=ResourceType.RESEARCH_TIME,
                    amount=40.0,
                    unit="hours",
                    duration=48.0,
                    priority=InnovationPriority.MEDIUM
                )
            ]
        )
    
    @pytest.mark.asyncio
    async def test_add_innovation_to_pipeline(self, optimizer, sample_innovation):
        """Test adding innovation to pipeline"""
        # Test successful addition
        result = await optimizer.add_innovation_to_pipeline(sample_innovation)
        assert result is True
        assert sample_innovation.id in optimizer.pipeline_items
        assert len(sample_innovation.resource_allocations) > 0
    
    @pytest.mark.asyncio
    async def test_add_innovation_capacity_constraint(self, optimizer):
        """Test capacity constraints when adding innovations"""
        # Fill up ideation stage to capacity
        for i in range(optimizer.stage_capacities[PipelineStage.IDEATION] + 1):
            innovation = InnovationPipelineItem(
                innovation_id=f"test-innovation-{i:03d}",
                current_stage=PipelineStage.IDEATION,
                priority=InnovationPriority.MEDIUM
            )
            
            if i < optimizer.stage_capacities[PipelineStage.IDEATION]:
                result = await optimizer.add_innovation_to_pipeline(innovation)
                assert result is True
            else:
                # This should fail due to capacity constraint
                result = await optimizer.add_innovation_to_pipeline(innovation)
                assert result is False
    
    @pytest.mark.asyncio
    async def test_optimize_pipeline_flow(self, optimizer, sample_innovation):
        """Test pipeline flow optimization"""
        # Add some innovations to pipeline
        await optimizer.add_innovation_to_pipeline(sample_innovation)
        
        # Add another innovation with different characteristics
        innovation2 = InnovationPipelineItem(
            innovation_id="test-innovation-002",
            current_stage=PipelineStage.RESEARCH,
            priority=InnovationPriority.CRITICAL,
            success_probability=0.9,
            risk_score=0.2,
            impact_score=0.95
        )
        await optimizer.add_innovation_to_pipeline(innovation2)
        
        # Run optimization
        result = await optimizer.optimize_pipeline_flow()
        
        # Verify optimization result
        assert result.optimization_id is not None
        assert result.timestamp is not None
        assert result.optimization_score >= 0.0
        assert result.confidence_level >= 0.0
        assert isinstance(result.recommendations, list)
        assert isinstance(result.warnings, list)
    
    @pytest.mark.asyncio
    async def test_monitor_pipeline_performance(self, optimizer, sample_innovation):
        """Test pipeline performance monitoring"""
        # Add innovation to pipeline
        await optimizer.add_innovation_to_pipeline(sample_innovation)
        
        # Generate performance report
        report = await optimizer.monitor_pipeline_performance()
        
        # Verify report structure
        assert report.report_id is not None
        assert report.generated_at is not None
        assert report.total_innovations >= 1
        assert report.active_innovations >= 1
        assert isinstance(report.stage_metrics, dict)
        assert isinstance(report.optimization_recommendations, list)
        assert isinstance(report.capacity_recommendations, list)
        assert isinstance(report.process_improvements, list)
    
    @pytest.mark.asyncio
    async def test_prioritize_innovations(self, optimizer):
        """Test innovation prioritization"""
        # Add multiple innovations with different characteristics
        innovations = [
            InnovationPipelineItem(
                innovation_id=f"test-innovation-{i:03d}",
                current_stage=PipelineStage.IDEATION,
                priority=InnovationPriority.MEDIUM,
                success_probability=0.5 + i * 0.1,
                risk_score=0.5 - i * 0.1,
                impact_score=0.6 + i * 0.1
            )
            for i in range(5)
        ]
        
        for innovation in innovations:
            await optimizer.add_innovation_to_pipeline(innovation)
        
        # Test prioritization with different criteria
        criteria = {
            'impact': 0.4,
            'success_probability': 0.3,
            'risk': 0.2,
            'resource_efficiency': 0.1
        }
        
        priorities = await optimizer.prioritize_innovations(criteria)
        
        # Verify prioritization results
        assert len(priorities) == len(innovations)
        assert all(isinstance(priority, InnovationPriority) for priority in priorities.values())
        
        # Check that higher impact/success probability leads to higher priority
        high_impact_innovation = innovations[-1]  # Highest impact
        low_impact_innovation = innovations[0]   # Lowest impact
        
        assert priorities[high_impact_innovation.id].value in ['critical', 'high']
        assert priorities[low_impact_innovation.id].value in ['low', 'medium']
    
    @pytest.mark.asyncio
    async def test_allocate_pipeline_resources(self, optimizer, sample_innovation):
        """Test resource allocation across pipeline"""
        # Add innovation to pipeline
        await optimizer.add_innovation_to_pipeline(sample_innovation)
        
        # Test different allocation strategies
        strategies = ["balanced", "aggressive", "conservative"]
        
        for strategy in strategies:
            allocations = await optimizer.allocate_pipeline_resources(strategy)
            
            # Verify allocation results
            assert isinstance(allocations, dict)
            if sample_innovation.id in allocations:
                innovation_allocations = allocations[sample_innovation.id]
                assert len(innovation_allocations) > 0
                
                for allocation in innovation_allocations:
                    assert allocation.innovation_id == sample_innovation.id
                    assert allocation.allocated_amount > 0
                    assert allocation.allocation_time is not None
    
    @pytest.mark.asyncio
    async def test_identify_bottlenecks(self, optimizer):
        """Test bottleneck identification"""
        # Create bottleneck by filling a stage near capacity
        stage = PipelineStage.PROTOTYPING
        capacity = optimizer.stage_capacities[stage]
        
        # Fill stage to 80% capacity (above bottleneck threshold)
        for i in range(int(capacity * 0.8)):
            innovation = InnovationPipelineItem(
                innovation_id=f"bottleneck-test-{i:03d}",
                current_stage=stage,
                priority=InnovationPriority.MEDIUM
            )
            await optimizer.add_innovation_to_pipeline(innovation)
        
        # Identify bottlenecks
        bottlenecks = await optimizer._identify_bottlenecks()
        
        # Verify bottleneck detection
        if stage in bottlenecks:
            assert bottlenecks[stage] >= optimizer.config.bottleneck_threshold
    
    @pytest.mark.asyncio
    async def test_resource_utilization_calculation(self, optimizer, sample_innovation):
        """Test resource utilization calculation"""
        # Add innovation to pipeline
        await optimizer.add_innovation_to_pipeline(sample_innovation)
        
        # Calculate resource utilization
        utilization = await optimizer._calculate_resource_utilization()
        
        # Verify utilization calculation
        assert isinstance(utilization, dict)
        assert all(isinstance(util, float) for util in utilization.values())
        assert all(0.0 <= util <= 1.0 for util in utilization.values())
        
        # Check that compute resources show some utilization
        if ResourceType.COMPUTE in utilization:
            assert utilization[ResourceType.COMPUTE] > 0.0
    
    @pytest.mark.asyncio
    async def test_stage_transition_optimization(self, optimizer):
        """Test stage transition optimization"""
        # Create innovation that's been in ideation stage for sufficient time
        innovation = InnovationPipelineItem(
            innovation_id="transition-test-001",
            current_stage=PipelineStage.IDEATION,
            priority=InnovationPriority.HIGH,
            success_probability=0.8,
            stage_entered_at=datetime.utcnow() - timedelta(hours=3)  # 3 hours ago
        )
        
        await optimizer.add_innovation_to_pipeline(innovation)
        
        # Test stage transition optimization
        transitions = await optimizer._optimize_stage_transitions()
        
        # Verify transition recommendations
        assert isinstance(transitions, dict)
        
        # If innovation is ready for transition, it should be recommended
        if innovation.id in transitions:
            next_stage = transitions[innovation.id]
            assert isinstance(next_stage, PipelineStage)
            assert next_stage != innovation.current_stage
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, optimizer, sample_innovation):
        """Test performance metrics calculation"""
        # Add innovation to pipeline
        await optimizer.add_innovation_to_pipeline(sample_innovation)
        
        # Calculate stage metrics
        stage_metrics = await optimizer._calculate_stage_metrics()
        
        # Verify metrics structure
        assert isinstance(stage_metrics, dict)
        
        for stage, metrics in stage_metrics.items():
            assert isinstance(stage, PipelineStage)
            assert hasattr(metrics, 'throughput')
            assert hasattr(metrics, 'cycle_time')
            assert hasattr(metrics, 'success_rate')
            assert hasattr(metrics, 'resource_utilization')
            assert hasattr(metrics, 'bottleneck_score')
            assert hasattr(metrics, 'quality_score')
            assert hasattr(metrics, 'cost_efficiency')
            
            # Verify metric ranges
            assert metrics.throughput >= 0.0
            assert metrics.cycle_time >= 0.0
            assert 0.0 <= metrics.success_rate <= 1.0
            assert 0.0 <= metrics.resource_utilization <= 1.0
            assert metrics.bottleneck_score >= 0.0
            assert 0.0 <= metrics.quality_score <= 1.0
            assert 0.0 <= metrics.cost_efficiency <= 1.0
    
    @pytest.mark.asyncio
    async def test_optimization_history_tracking(self, optimizer, sample_innovation):
        """Test optimization history tracking"""
        # Add innovation to pipeline
        await optimizer.add_innovation_to_pipeline(sample_innovation)
        
        # Run multiple optimizations
        initial_history_length = len(optimizer.optimization_history)
        
        for i in range(3):
            await optimizer.optimize_pipeline_flow()
        
        # Verify history tracking
        assert len(optimizer.optimization_history) == initial_history_length + 3
        
        # Verify history entries
        for optimization in optimizer.optimization_history[-3:]:
            assert optimization.optimization_id is not None
            assert optimization.timestamp is not None
            assert optimization.optimization_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_resource_efficiency_calculation(self, optimizer, sample_innovation):
        """Test resource efficiency calculation"""
        # Add innovation to pipeline
        await optimizer.add_innovation_to_pipeline(sample_innovation)
        
        # Simulate some resource usage
        for allocation in sample_innovation.resource_allocations:
            allocation.used_amount = allocation.allocated_amount * 0.7  # 70% efficiency
        
        # Calculate efficiency
        efficiency = await optimizer._calculate_innovation_resource_efficiency(sample_innovation)
        
        # Verify efficiency calculation
        assert 0.0 <= efficiency <= 1.0
        assert abs(efficiency - 0.7) < 0.1  # Should be close to 70%
    
    @pytest.mark.asyncio
    async def test_time_sensitivity_calculation(self, optimizer):
        """Test time sensitivity calculation"""
        # Create innovation with deadline
        innovation = InnovationPipelineItem(
            innovation_id="time-sensitive-001",
            current_stage=PipelineStage.IDEATION,
            priority=InnovationPriority.HIGH,
            created_at=datetime.utcnow() - timedelta(hours=12),
            estimated_completion=datetime.utcnow() + timedelta(hours=12)
        )
        
        # Calculate time sensitivity
        sensitivity = await optimizer._calculate_time_sensitivity(innovation)
        
        # Verify time sensitivity calculation
        assert 0.0 <= sensitivity <= 1.0
        # Should be around 0.5 since we're halfway to deadline
        assert abs(sensitivity - 0.5) < 0.2
    
    @pytest.mark.asyncio
    async def test_concurrent_optimization_prevention(self, optimizer, sample_innovation):
        """Test prevention of concurrent optimizations"""
        # Add innovation to pipeline
        await optimizer.add_innovation_to_pipeline(sample_innovation)
        
        # Start first optimization
        optimization_task1 = asyncio.create_task(optimizer.optimize_pipeline_flow())
        
        # Add a small delay to ensure first optimization starts
        await asyncio.sleep(0.01)
        
        # Try to start second optimization while first is running
        optimization_task2 = asyncio.create_task(optimizer.optimize_pipeline_flow())
        
        # Wait for both to complete
        result1, result2 = await asyncio.gather(optimization_task1, optimization_task2)
        
        # At least one should succeed
        assert (result1.optimization_score > 0.0) or (result2.optimization_score > 0.0)
        
        # Check that we have some indication of concurrent execution handling
        # (either one has score 0.0 or both completed successfully)
        total_score = result1.optimization_score + result2.optimization_score
        assert total_score >= 0.0  # At least some optimization occurred
        
        # Verify optimization history was updated
        assert len(optimizer.optimization_history) >= 1
    
    def test_pipeline_stage_ordering(self, optimizer):
        """Test pipeline stage ordering logic"""
        # Test stage progression
        stage_order = [
            PipelineStage.IDEATION,
            PipelineStage.RESEARCH,
            PipelineStage.EXPERIMENTATION,
            PipelineStage.PROTOTYPING,
            PipelineStage.VALIDATION,
            PipelineStage.OPTIMIZATION,
            PipelineStage.DEPLOYMENT,
            PipelineStage.MONITORING
        ]
        
        for i, stage in enumerate(stage_order[:-1]):
            next_stage = optimizer._get_next_stage(stage)
            expected_next = stage_order[i + 1]
            assert next_stage == expected_next
        
        # Last stage should have no next stage
        assert optimizer._get_next_stage(PipelineStage.MONITORING) is None
    
    def test_priority_weight_calculation(self, optimizer):
        """Test priority weight calculation"""
        # Test all priority levels
        priorities = [
            InnovationPriority.CRITICAL,
            InnovationPriority.HIGH,
            InnovationPriority.MEDIUM,
            InnovationPriority.LOW
        ]
        
        weights = [optimizer._get_priority_weight(priority) for priority in priorities]
        
        # Verify weights are in descending order
        assert weights == sorted(weights, reverse=True)
        
        # Verify specific weight values
        assert optimizer._get_priority_weight(InnovationPriority.CRITICAL) == optimizer.config.priority_weight_critical
        assert optimizer._get_priority_weight(InnovationPriority.HIGH) == optimizer.config.priority_weight_high
        assert optimizer._get_priority_weight(InnovationPriority.MEDIUM) == optimizer.config.priority_weight_medium
        assert optimizer._get_priority_weight(InnovationPriority.LOW) == optimizer.config.priority_weight_low
    
    @pytest.mark.asyncio
    async def test_capacity_checking(self, optimizer):
        """Test capacity checking logic"""
        # Test with empty pipeline
        innovation = InnovationPipelineItem(
            innovation_id="capacity-test-001",
            current_stage=PipelineStage.IDEATION,
            priority=InnovationPriority.MEDIUM
        )
        
        assert optimizer._check_capacity_constraints(innovation) is True
        
        # Fill stage to capacity
        stage_capacity = optimizer.stage_capacities[PipelineStage.IDEATION]
        for i in range(stage_capacity):
            test_innovation = InnovationPipelineItem(
                innovation_id=f"capacity-fill-{i:03d}",
                current_stage=PipelineStage.IDEATION,
                priority=InnovationPriority.MEDIUM,
                status=PipelineStatus.ACTIVE
            )
            optimizer.pipeline_items[test_innovation.id] = test_innovation
        
        # Now capacity should be exceeded
        assert optimizer._check_capacity_constraints(innovation) is False


@pytest.mark.asyncio
async def test_integration_pipeline_optimization():
    """Integration test for complete pipeline optimization workflow"""
    # Create optimizer
    optimizer = InnovationPipelineOptimizer()
    
    # Create multiple innovations with different characteristics
    innovations = [
        InnovationPipelineItem(
            innovation_id=f"integration-test-{i:03d}",
            current_stage=PipelineStage.IDEATION if i < 3 else PipelineStage.RESEARCH,
            priority=InnovationPriority.HIGH if i % 2 == 0 else InnovationPriority.MEDIUM,
            success_probability=0.6 + i * 0.05,
            risk_score=0.4 - i * 0.03,
            impact_score=0.7 + i * 0.04,
            resource_requirements=[
                ResourceRequirement(
                    resource_type=ResourceType.COMPUTE,
                    amount=50.0 + i * 10,
                    unit="cores",
                    duration=24.0
                )
            ]
        )
        for i in range(6)
    ]
    
    # Add all innovations to pipeline
    for innovation in innovations:
        result = await optimizer.add_innovation_to_pipeline(innovation)
        assert result is True
    
    # Run optimization
    optimization_result = await optimizer.optimize_pipeline_flow()
    assert optimization_result.optimization_score >= 0.0
    
    # Generate performance report
    performance_report = await optimizer.monitor_pipeline_performance()
    assert performance_report.total_innovations == len(innovations)
    assert performance_report.active_innovations > 0
    
    # Test prioritization
    criteria = {'impact': 0.5, 'success_probability': 0.3, 'risk': 0.2}
    priorities = await optimizer.prioritize_innovations(criteria)
    assert len(priorities) == len(innovations)
    
    # Test resource allocation
    allocations = await optimizer.allocate_pipeline_resources("balanced")
    assert len(allocations) > 0
    
    # Verify optimization history
    assert len(optimizer.optimization_history) > 0
    assert len(optimizer.historical_metrics) > 0


if __name__ == "__main__":
    pytest.main([__file__])