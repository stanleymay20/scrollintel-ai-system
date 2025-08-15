"""
Tests for Innovation Acceleration System

This module contains comprehensive tests for the innovation acceleration system,
including bottleneck identification, timeline optimization, and acceleration strategies.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.innovation_acceleration_system import (
    InnovationAccelerationSystem, AccelerationType, BottleneckType,
    AccelerationStrategy, BottleneckAnalysis, TimelineOptimization
)
from scrollintel.models.innovation_pipeline_models import (
    InnovationPipelineItem, PipelineStage, InnovationPriority, ResourceType,
    PipelineStatus, ResourceRequirement, ResourceAllocation
)


class TestInnovationAccelerationSystem:
    """Test cases for InnovationAccelerationSystem"""
    
    @pytest.fixture
    def acceleration_system(self):
        """Create acceleration system instance for testing"""
        return InnovationAccelerationSystem()
    
    @pytest.fixture
    def sample_pipeline_items(self):
        """Create sample pipeline items for testing"""
        items = {}
        
        # High priority innovation with resource constraints
        items["innovation-001"] = InnovationPipelineItem(
            innovation_id="innovation-001",
            current_stage=PipelineStage.RESEARCH,
            priority=InnovationPriority.CRITICAL,
            success_probability=0.9,
            risk_score=0.2,
            impact_score=0.95,
            resource_requirements=[
                ResourceRequirement(
                    resource_type=ResourceType.RESEARCH_TIME,
                    amount=200.0,
                    unit="hours",
                    duration=168.0
                )
            ],
            resource_allocations=[
                ResourceAllocation(
                    innovation_id="innovation-001",
                    resource_type=ResourceType.RESEARCH_TIME,
                    allocated_amount=200.0,
                    used_amount=180.0
                )
            ]
        )
        
        # Medium priority innovation with quality issues
        items["innovation-002"] = InnovationPipelineItem(
            innovation_id="innovation-002",
            current_stage=PipelineStage.VALIDATION,
            priority=InnovationPriority.MEDIUM,
            success_probability=0.5,  # Low success probability
            risk_score=0.6,
            impact_score=0.7,
            resource_requirements=[
                ResourceRequirement(
                    resource_type=ResourceType.TESTING_TIME,
                    amount=100.0,
                    unit="hours",
                    duration=72.0
                )
            ]
        )
        
        # Innovation with dependencies
        items["innovation-003"] = InnovationPipelineItem(
            innovation_id="innovation-003",
            current_stage=PipelineStage.PROTOTYPING,
            priority=InnovationPriority.HIGH,
            success_probability=0.8,
            risk_score=0.3,
            impact_score=0.85,
            dependencies=["innovation-001", "innovation-004"]  # One exists, one doesn't
        )
        
        return items
    
    @pytest.mark.asyncio
    async def test_accelerate_innovation_development(self, acceleration_system, sample_pipeline_items):
        """Test complete innovation development acceleration"""
        result = await acceleration_system.accelerate_innovation_development(sample_pipeline_items)
        
        # Verify result structure
        assert result.acceleration_id is not None
        assert result.timestamp is not None
        assert result.innovations_accelerated >= 0
        assert result.total_time_saved >= 0.0
        assert result.bottlenecks_resolved >= 0
        assert isinstance(result.acceleration_strategies_applied, list)
        assert isinstance(result.timeline_optimizations, list)
        assert result.performance_improvement >= 0.0
        assert result.cost_efficiency >= 0.0
    
    @pytest.mark.asyncio
    async def test_identify_bottlenecks(self, acceleration_system, sample_pipeline_items):
        """Test bottleneck identification"""
        bottlenecks = await acceleration_system.identify_bottlenecks(sample_pipeline_items)
        
        # Verify bottlenecks were identified
        assert isinstance(bottlenecks, list)
        
        # Check for expected bottleneck types
        bottleneck_types = [b.bottleneck_type for b in bottlenecks]
        
        # Should identify quality bottleneck for innovation-002
        assert BottleneckType.QUALITY_GATE in bottleneck_types
        
        # Should identify dependency bottleneck for innovation-003
        assert BottleneckType.DEPENDENCY_BLOCK in bottleneck_types
        
        # Verify bottleneck structure
        for bottleneck in bottlenecks:
            assert bottleneck.bottleneck_id is not None
            assert bottleneck.innovation_id in sample_pipeline_items
            assert isinstance(bottleneck.bottleneck_type, BottleneckType)
            assert isinstance(bottleneck.affected_stage, PipelineStage)
            assert 0.0 <= bottleneck.severity <= 1.0
            assert bottleneck.estimated_delay >= 0.0
            assert isinstance(bottleneck.root_causes, list)
            assert isinstance(bottleneck.resolution_strategies, list)
    
    @pytest.mark.asyncio
    async def test_optimize_innovation_timeline(self, acceleration_system, sample_pipeline_items):
        """Test timeline optimization for specific innovation"""
        innovation = sample_pipeline_items["innovation-001"]
        optimization = await acceleration_system.optimize_innovation_timeline(innovation)
        
        # Verify optimization structure
        assert optimization.optimization_id is not None
        assert optimization.innovation_id == innovation.innovation_id
        assert optimization.original_timeline > 0.0
        assert optimization.optimized_timeline > 0.0
        assert optimization.time_savings >= 0.0
        assert isinstance(optimization.acceleration_strategies, list)
        assert 0.0 <= optimization.confidence_level <= 1.0
        assert 0.0 <= optimization.risk_assessment <= 1.0
        
        # Timeline should be optimized (reduced or same)
        assert optimization.optimized_timeline <= optimization.original_timeline
        
        # Time savings should match the difference
        expected_savings = optimization.original_timeline - optimization.optimized_timeline
        assert abs(optimization.time_savings - expected_savings) < 0.1
    
    @pytest.mark.asyncio
    async def test_resource_bottleneck_identification(self, acceleration_system):
        """Test identification of resource constraint bottlenecks"""
        # Create pipeline with high resource utilization
        pipeline_items = {
            "high-resource-innovation": InnovationPipelineItem(
                innovation_id="high-resource-innovation",
                current_stage=PipelineStage.EXPERIMENTATION,
                priority=InnovationPriority.HIGH,
                resource_requirements=[
                    ResourceRequirement(
                        resource_type=ResourceType.COMPUTE,
                        amount=1000.0,  # High demand
                        unit="cores",
                        duration=168.0
                    )
                ],
                resource_allocations=[
                    ResourceAllocation(
                        innovation_id="high-resource-innovation",
                        resource_type=ResourceType.COMPUTE,
                        allocated_amount=1000.0,
                        used_amount=950.0  # 95% utilization
                    )
                ]
            )
        }
        
        bottlenecks = await acceleration_system._analyze_resource_bottlenecks(pipeline_items)
        
        # Should identify resource bottleneck
        resource_bottlenecks = [
            b for b in bottlenecks 
            if b.bottleneck_type == BottleneckType.RESOURCE_CONSTRAINT
        ]
        assert len(resource_bottlenecks) > 0
        
        # Verify bottleneck details
        bottleneck = resource_bottlenecks[0]
        assert bottleneck.severity > 0.9  # High severity due to high utilization
        assert "utilization" in bottleneck.root_causes[0].lower()
    
    @pytest.mark.asyncio
    async def test_capacity_bottleneck_identification(self, acceleration_system):
        """Test identification of capacity limit bottlenecks"""
        # Create many innovations in same stage to exceed capacity
        pipeline_items = {}
        for i in range(25):  # Exceed typical research stage capacity
            pipeline_items[f"research-innovation-{i}"] = InnovationPipelineItem(
                innovation_id=f"research-innovation-{i}",
                current_stage=PipelineStage.RESEARCH,
                priority=InnovationPriority.MEDIUM,
                status=PipelineStatus.ACTIVE
            )
        
        bottlenecks = await acceleration_system._analyze_capacity_bottlenecks(pipeline_items)
        
        # Should identify capacity bottlenecks
        capacity_bottlenecks = [
            b for b in bottlenecks 
            if b.bottleneck_type == BottleneckType.CAPACITY_LIMIT
        ]
        assert len(capacity_bottlenecks) > 0
        
        # All should be for research stage
        for bottleneck in capacity_bottlenecks:
            assert bottleneck.affected_stage == PipelineStage.RESEARCH
    
    @pytest.mark.asyncio
    async def test_dependency_bottleneck_identification(self, acceleration_system):
        """Test identification of dependency blocking bottlenecks"""
        pipeline_items = {
            "dependent-innovation": InnovationPipelineItem(
                innovation_id="dependent-innovation",
                current_stage=PipelineStage.PROTOTYPING,
                priority=InnovationPriority.HIGH,
                dependencies=["blocking-innovation"],
                status=PipelineStatus.ACTIVE
            ),
            "blocking-innovation": InnovationPipelineItem(
                innovation_id="blocking-innovation",
                current_stage=PipelineStage.RESEARCH,
                priority=InnovationPriority.MEDIUM,
                status=PipelineStatus.ACTIVE  # Not completed, so blocking
            )
        }
        
        bottlenecks = await acceleration_system._analyze_dependency_bottlenecks(pipeline_items)
        
        # Should identify dependency bottleneck
        dependency_bottlenecks = [
            b for b in bottlenecks 
            if b.bottleneck_type == BottleneckType.DEPENDENCY_BLOCK
        ]
        assert len(dependency_bottlenecks) > 0
        
        # Verify bottleneck details
        bottleneck = dependency_bottlenecks[0]
        assert bottleneck.innovation_id == "dependent-innovation"
        assert "blocking-innovation" in bottleneck.root_causes[0]
    
    @pytest.mark.asyncio
    async def test_quality_bottleneck_identification(self, acceleration_system):
        """Test identification of quality gate bottlenecks"""
        pipeline_items = {
            "low-quality-innovation": InnovationPipelineItem(
                innovation_id="low-quality-innovation",
                current_stage=PipelineStage.VALIDATION,
                priority=InnovationPriority.HIGH,
                success_probability=0.4,  # Low success probability
                status=PipelineStatus.ACTIVE
            )
        }
        
        bottlenecks = await acceleration_system._analyze_quality_bottlenecks(pipeline_items)
        
        # Should identify quality bottleneck
        quality_bottlenecks = [
            b for b in bottlenecks 
            if b.bottleneck_type == BottleneckType.QUALITY_GATE
        ]
        assert len(quality_bottlenecks) > 0
        
        # Verify bottleneck details
        bottleneck = quality_bottlenecks[0]
        assert bottleneck.innovation_id == "low-quality-innovation"
        assert bottleneck.severity > 0.0  # Should have some severity
        assert "success probability" in bottleneck.root_causes[0].lower()
    
    @pytest.mark.asyncio
    async def test_acceleration_strategy_generation(self, acceleration_system, sample_pipeline_items):
        """Test generation of acceleration strategies"""
        # First identify bottlenecks
        bottlenecks = await acceleration_system.identify_bottlenecks(sample_pipeline_items)
        
        # Generate strategies
        strategies = await acceleration_system._generate_acceleration_strategies(
            sample_pipeline_items, bottlenecks
        )
        
        # Verify strategies were generated
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        
        # Verify strategy structure
        for strategy in strategies:
            assert strategy.id is not None
            assert strategy.innovation_id in sample_pipeline_items
            assert isinstance(strategy.acceleration_type, AccelerationType)
            assert isinstance(strategy.target_stage, PipelineStage)
            assert strategy.expected_time_reduction > 0.0
            assert strategy.resource_cost > 0.0
            assert 0.0 <= strategy.success_probability <= 1.0
            assert 0.0 <= strategy.risk_factor <= 1.0
            assert isinstance(strategy.prerequisites, list)
            assert isinstance(strategy.side_effects, list)
        
        # Strategies should meet quality thresholds
        for strategy in strategies:
            assert strategy.success_probability >= acceleration_system.acceleration_threshold
            assert strategy.risk_factor <= acceleration_system.risk_tolerance
    
    @pytest.mark.asyncio
    async def test_research_stage_strategies(self, acceleration_system):
        """Test strategy generation for research stage"""
        innovation = InnovationPipelineItem(
            innovation_id="research-innovation",
            current_stage=PipelineStage.RESEARCH,
            priority=InnovationPriority.CRITICAL
        )
        
        strategies = await acceleration_system._generate_research_strategies(innovation)
        
        # Should generate research-specific strategies
        assert len(strategies) > 0
        
        # Check for expected strategy types
        strategy_types = [s.acceleration_type for s in strategies]
        assert AccelerationType.PARALLEL_PROCESSING in strategy_types or AccelerationType.RESOURCE_BOOST in strategy_types
        
        # All strategies should target research stage
        for strategy in strategies:
            assert strategy.target_stage == PipelineStage.RESEARCH
    
    @pytest.mark.asyncio
    async def test_experimentation_stage_strategies(self, acceleration_system):
        """Test strategy generation for experimentation stage"""
        innovation = InnovationPipelineItem(
            innovation_id="experiment-innovation",
            current_stage=PipelineStage.EXPERIMENTATION,
            priority=InnovationPriority.HIGH
        )
        
        strategies = await acceleration_system._generate_experimentation_strategies(innovation)
        
        # Should generate experimentation-specific strategies
        assert len(strategies) > 0
        
        # Check for expected strategy types
        strategy_types = [s.acceleration_type for s in strategies]
        expected_types = [AccelerationType.AUTOMATED_TRANSITION, AccelerationType.PARALLEL_PROCESSING]
        assert any(t in strategy_types for t in expected_types)
        
        # All strategies should target experimentation stage
        for strategy in strategies:
            assert strategy.target_stage == PipelineStage.EXPERIMENTATION
    
    @pytest.mark.asyncio
    async def test_timeline_estimation(self, acceleration_system):
        """Test timeline estimation for innovations"""
        # Test different stages and complexities
        innovations = [
            InnovationPipelineItem(
                innovation_id="simple-research",
                current_stage=PipelineStage.RESEARCH,
                success_probability=0.9  # Simple (high success probability)
            ),
            InnovationPipelineItem(
                innovation_id="complex-research",
                current_stage=PipelineStage.RESEARCH,
                success_probability=0.4  # Complex (low success probability)
            ),
            InnovationPipelineItem(
                innovation_id="prototyping",
                current_stage=PipelineStage.PROTOTYPING,
                success_probability=0.7
            )
        ]
        
        for innovation in innovations:
            timeline = await acceleration_system._estimate_current_timeline(innovation)
            
            # Timeline should be positive
            assert timeline > 0.0
            
            # Complex innovations should have longer timelines
            if innovation.success_probability < 0.5:
                assert timeline > 200.0  # Should be longer due to complexity
    
    @pytest.mark.asyncio
    async def test_timeline_optimization_calculation(self, acceleration_system):
        """Test timeline optimization calculation"""
        innovation = InnovationPipelineItem(
            innovation_id="test-innovation",
            current_stage=PipelineStage.RESEARCH,
            success_probability=0.8
        )
        
        # Create mock strategies
        strategies = [
            AccelerationStrategy(
                id="strategy-1",
                innovation_id=innovation.id,
                acceleration_type=AccelerationType.PARALLEL_PROCESSING,
                target_stage=PipelineStage.RESEARCH,
                expected_time_reduction=48.0,
                resource_cost=1.5,
                success_probability=0.8,
                risk_factor=0.2
            ),
            AccelerationStrategy(
                id="strategy-2",
                innovation_id=innovation.id,
                acceleration_type=AccelerationType.RESOURCE_BOOST,
                target_stage=PipelineStage.RESEARCH,
                expected_time_reduction=24.0,
                resource_cost=1.3,
                success_probability=0.9,
                risk_factor=0.1
            )
        ]
        
        original_timeline = await acceleration_system._estimate_current_timeline(innovation)
        optimized_timeline = await acceleration_system._calculate_optimized_timeline(
            innovation, strategies
        )
        
        # Optimized timeline should be shorter
        assert optimized_timeline < original_timeline
        
        # Should not be reduced below minimum threshold
        min_timeline = original_timeline * 0.3
        assert optimized_timeline >= min_timeline
    
    @pytest.mark.asyncio
    async def test_strategy_application(self, acceleration_system, sample_pipeline_items):
        """Test application of acceleration strategies"""
        innovation = sample_pipeline_items["innovation-001"]
        
        # Create test strategy
        strategy = AccelerationStrategy(
            id="test-strategy",
            innovation_id=innovation.id,
            acceleration_type=AccelerationType.RESOURCE_BOOST,
            target_stage=innovation.current_stage,
            expected_time_reduction=48.0,
            resource_cost=1.5,
            success_probability=0.8,
            risk_factor=0.2
        )
        
        # Apply strategy
        await acceleration_system._apply_strategy(strategy, innovation)
        
        # Verify strategy was applied
        assert 'accelerations' in innovation.metadata
        accelerations = innovation.metadata['accelerations']
        assert len(accelerations) > 0
        
        applied_acceleration = accelerations[-1]
        assert applied_acceleration['strategy_id'] == strategy.id
        assert applied_acceleration['type'] == strategy.acceleration_type.value
        assert applied_acceleration['expected_reduction'] == strategy.expected_time_reduction
    
    @pytest.mark.asyncio
    async def test_performance_improvement_calculation(self, acceleration_system, sample_pipeline_items):
        """Test performance improvement calculation"""
        # Create test strategies
        strategies = [
            AccelerationStrategy(
                id="strategy-1",
                innovation_id="innovation-001",
                acceleration_type=AccelerationType.PARALLEL_PROCESSING,
                target_stage=PipelineStage.RESEARCH,
                expected_time_reduction=48.0,
                resource_cost=1.5,
                success_probability=0.8,
                risk_factor=0.2
            ),
            AccelerationStrategy(
                id="strategy-2",
                innovation_id="innovation-002",
                acceleration_type=AccelerationType.RESOURCE_BOOST,
                target_stage=PipelineStage.VALIDATION,
                expected_time_reduction=24.0,
                resource_cost=1.3,
                success_probability=0.9,
                risk_factor=0.1
            )
        ]
        
        improvement = await acceleration_system._calculate_performance_improvement(
            sample_pipeline_items, strategies
        )
        
        # Should calculate positive improvement
        assert improvement >= 0.0
        
        # Should be weighted by priority and impact
        assert improvement > 0.0  # Should have some improvement with valid strategies
    
    @pytest.mark.asyncio
    async def test_cost_efficiency_calculation(self, acceleration_system):
        """Test cost efficiency calculation"""
        strategies = [
            AccelerationStrategy(
                id="efficient-strategy",
                innovation_id="test-innovation",
                acceleration_type=AccelerationType.AUTOMATED_TRANSITION,
                target_stage=PipelineStage.VALIDATION,
                expected_time_reduction=48.0,
                resource_cost=1.2,  # Low cost
                success_probability=0.9,  # High success
                risk_factor=0.1
            ),
            AccelerationStrategy(
                id="expensive-strategy",
                innovation_id="test-innovation-2",
                acceleration_type=AccelerationType.PARALLEL_PROCESSING,
                target_stage=PipelineStage.EXPERIMENTATION,
                expected_time_reduction=24.0,
                resource_cost=2.0,  # High cost
                success_probability=0.7,
                risk_factor=0.3
            )
        ]
        
        efficiency = await acceleration_system._calculate_cost_efficiency(strategies)
        
        # Should calculate positive efficiency
        assert efficiency > 0.0
        
        # Should be reasonable ratio of benefit to cost
        assert efficiency < 100.0  # Sanity check
    
    def test_acceleration_metrics(self, acceleration_system):
        """Test acceleration metrics collection"""
        # Add some mock history
        from scrollintel.engines.innovation_acceleration_system import AccelerationResult
        
        acceleration_system.acceleration_history = [
            AccelerationResult(
                acceleration_id="test-1",
                innovations_accelerated=5,
                total_time_saved=120.0,
                bottlenecks_resolved=3,
                performance_improvement=0.25,
                cost_efficiency=1.5
            ),
            AccelerationResult(
                acceleration_id="test-2",
                innovations_accelerated=3,
                total_time_saved=80.0,
                bottlenecks_resolved=2,
                performance_improvement=0.15,
                cost_efficiency=1.2
            )
        ]
        
        metrics = acceleration_system.get_acceleration_metrics()
        
        # Verify metrics structure
        assert 'average_time_saved' in metrics
        assert 'average_innovations_accelerated' in metrics
        assert 'average_performance_improvement' in metrics
        assert 'average_cost_efficiency' in metrics
        assert 'bottleneck_resolution_rate' in metrics
        
        # Verify calculated values
        assert metrics['average_time_saved'] == 100.0  # (120 + 80) / 2
        assert metrics['average_innovations_accelerated'] == 4.0  # (5 + 3) / 2
    
    def test_bottleneck_patterns(self, acceleration_system):
        """Test bottleneck pattern tracking"""
        # Simulate some bottleneck patterns
        acceleration_system.bottleneck_patterns[BottleneckType.RESOURCE_CONSTRAINT] = 5
        acceleration_system.bottleneck_patterns[BottleneckType.CAPACITY_LIMIT] = 3
        acceleration_system.bottleneck_patterns[BottleneckType.QUALITY_GATE] = 2
        
        patterns = acceleration_system.get_bottleneck_patterns()
        
        # Verify patterns
        assert patterns[BottleneckType.RESOURCE_CONSTRAINT] == 5
        assert patterns[BottleneckType.CAPACITY_LIMIT] == 3
        assert patterns[BottleneckType.QUALITY_GATE] == 2
    
    @pytest.mark.asyncio
    async def test_confidence_and_risk_assessment(self, acceleration_system):
        """Test confidence and risk assessment for timeline optimization"""
        # Test with high-confidence strategies
        high_confidence_strategies = [
            AccelerationStrategy(
                id="safe-strategy",
                innovation_id="test-innovation",
                acceleration_type=AccelerationType.AUTOMATED_TRANSITION,
                target_stage=PipelineStage.VALIDATION,
                expected_time_reduction=24.0,
                resource_cost=1.1,
                success_probability=0.95,  # High success
                risk_factor=0.05  # Low risk
            )
        ]
        
        confidence = await acceleration_system._calculate_timeline_confidence(high_confidence_strategies)
        risk = await acceleration_system._assess_timeline_risk(high_confidence_strategies)
        
        assert confidence > 0.8  # Should be high confidence
        assert risk < 0.2  # Should be low risk
        
        # Test with low-confidence strategies
        low_confidence_strategies = [
            AccelerationStrategy(
                id="risky-strategy",
                innovation_id="test-innovation",
                acceleration_type=AccelerationType.BOTTLENECK_BYPASS,
                target_stage=PipelineStage.PROTOTYPING,
                expected_time_reduction=72.0,
                resource_cost=2.0,
                success_probability=0.6,  # Lower success
                risk_factor=0.4  # Higher risk
            )
        ]
        
        confidence_low = await acceleration_system._calculate_timeline_confidence(low_confidence_strategies)
        risk_high = await acceleration_system._assess_timeline_risk(low_confidence_strategies)
        
        assert confidence_low < confidence  # Should be lower confidence
        assert risk_high > risk  # Should be higher risk


@pytest.mark.asyncio
async def test_integration_acceleration_system():
    """Integration test for complete acceleration system workflow"""
    # Create acceleration system
    acceleration_system = InnovationAccelerationSystem()
    
    # Create test pipeline with various scenarios
    pipeline_items = {
        "critical-innovation": InnovationPipelineItem(
            innovation_id="critical-innovation",
            current_stage=PipelineStage.RESEARCH,
            priority=InnovationPriority.CRITICAL,
            success_probability=0.85,
            risk_score=0.2,
            impact_score=0.95,
            resource_requirements=[
                ResourceRequirement(
                    resource_type=ResourceType.RESEARCH_TIME,
                    amount=300.0,
                    unit="hours",
                    duration=168.0
                )
            ]
        ),
        "bottlenecked-innovation": InnovationPipelineItem(
            innovation_id="bottlenecked-innovation",
            current_stage=PipelineStage.VALIDATION,
            priority=InnovationPriority.HIGH,
            success_probability=0.4,  # Quality bottleneck
            risk_score=0.6,
            impact_score=0.8,
            dependencies=["critical-innovation"]  # Dependency bottleneck
        ),
        "capacity-constrained": InnovationPipelineItem(
            innovation_id="capacity-constrained",
            current_stage=PipelineStage.PROTOTYPING,
            priority=InnovationPriority.MEDIUM,
            success_probability=0.75,
            risk_score=0.3,
            impact_score=0.7
        )
    }
    
    # Run complete acceleration workflow
    result = await acceleration_system.accelerate_innovation_development(pipeline_items)
    
    # Verify comprehensive results
    assert result.acceleration_id is not None
    assert result.innovations_accelerated > 0
    assert result.total_time_saved > 0.0
    assert len(result.acceleration_strategies_applied) > 0
    assert len(result.timeline_optimizations) > 0
    
    # Verify bottlenecks were identified and addressed
    assert result.bottlenecks_resolved > 0
    
    # Verify performance metrics
    assert result.performance_improvement > 0.0
    assert result.cost_efficiency > 0.0
    
    # Verify history tracking
    assert len(acceleration_system.acceleration_history) > 0
    assert len(acceleration_system.bottleneck_history) > 0
    
    # Test individual timeline optimization
    innovation = pipeline_items["critical-innovation"]
    timeline_opt = await acceleration_system.optimize_innovation_timeline(innovation)
    
    assert timeline_opt.time_savings > 0.0
    assert timeline_opt.optimized_timeline < timeline_opt.original_timeline
    assert len(timeline_opt.acceleration_strategies) > 0
    
    # Test metrics collection
    metrics = acceleration_system.get_acceleration_metrics()
    assert 'average_time_saved' in metrics
    assert 'average_innovations_accelerated' in metrics
    
    patterns = acceleration_system.get_bottleneck_patterns()
    assert len(patterns) > 0


if __name__ == "__main__":
    pytest.main([__file__])