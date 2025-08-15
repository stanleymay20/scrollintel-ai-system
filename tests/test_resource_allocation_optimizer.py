"""
Tests for Resource Allocation Optimizer

Comprehensive tests for optimal resource distribution, allocation tracking,
and utilization monitoring capabilities.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.resource_allocation_optimizer import (
    ResourceAllocationOptimizer, AllocationEngine, AllocationTrackingSystem,
    UtilizationMonitor, OptimizationStrategy, AllocationScore
)
from scrollintel.models.resource_mobilization_models import (
    Resource, ResourceRequirement, ResourceAllocation, AllocationPlan,
    ResourceUtilization, ResourceType, ResourcePriority, AllocationStatus,
    ResourceCapability
)
from scrollintel.models.crisis_models_simple import Crisis, CrisisType, SeverityLevel


class TestResourceAllocationOptimizer:
    """Test cases for ResourceAllocationOptimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create ResourceAllocationOptimizer instance for testing"""
        return ResourceAllocationOptimizer()
    
    @pytest.fixture
    def sample_crisis(self):
        """Create sample crisis for testing"""
        return Crisis(
            id="test_crisis_001",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.HIGH,
            description="Test system outage requiring resource allocation"
        )
    
    @pytest.fixture
    def sample_requirements(self):
        """Create sample resource requirements"""
        return [
            ResourceRequirement(
                id="req_001",
                crisis_id="test_crisis_001",
                resource_type=ResourceType.HUMAN_RESOURCES,
                required_capabilities=["incident_response", "system_debugging"],
                quantity_needed=20.0,
                priority=ResourcePriority.CRITICAL,
                duration_needed=timedelta(hours=8),
                justification="Critical incident response team"
            ),
            ResourceRequirement(
                id="req_002",
                crisis_id="test_crisis_001",
                resource_type=ResourceType.TECHNICAL_INFRASTRUCTURE,
                required_capabilities=["high_availability", "load_balancing"],
                quantity_needed=500.0,
                priority=ResourcePriority.HIGH,
                duration_needed=timedelta(hours=12),
                justification="Infrastructure scaling for incident response"
            ),
            ResourceRequirement(
                id="req_003",
                crisis_id="test_crisis_001",
                resource_type=ResourceType.CLOUD_COMPUTE,
                required_capabilities=["elastic_scaling"],
                quantity_needed=1000.0,
                priority=ResourcePriority.MEDIUM,
                duration_needed=timedelta(hours=24),
                justification="Additional compute capacity"
            )
        ]
    
    @pytest.fixture
    def sample_resources(self):
        """Create sample available resources"""
        return [
            Resource(
                id="resource_001",
                name="Incident Response Team",
                resource_type=ResourceType.HUMAN_RESOURCES,
                capabilities=[
                    ResourceCapability(name="incident_response", proficiency_level=0.9),
                    ResourceCapability(name="system_debugging", proficiency_level=0.8)
                ],
                capacity=40.0,
                current_utilization=10.0,
                cost_per_hour=150.0
            ),
            Resource(
                id="resource_002",
                name="Primary Infrastructure",
                resource_type=ResourceType.TECHNICAL_INFRASTRUCTURE,
                capabilities=[
                    ResourceCapability(name="high_availability", proficiency_level=1.0),
                    ResourceCapability(name="load_balancing", proficiency_level=0.9)
                ],
                capacity=1000.0,
                current_utilization=400.0,
                cost_per_hour=75.0
            ),
            Resource(
                id="resource_003",
                name="Cloud Compute Pool",
                resource_type=ResourceType.CLOUD_COMPUTE,
                capabilities=[
                    ResourceCapability(name="elastic_scaling", proficiency_level=1.0)
                ],
                capacity=5000.0,
                current_utilization=2000.0,
                cost_per_hour=30.0
            )
        ]
    
    @pytest.mark.asyncio
    async def test_optimize_resource_allocation(self, optimizer, sample_crisis, sample_requirements, sample_resources):
        """Test comprehensive resource allocation optimization"""
        # Perform optimization
        result = await optimizer.optimize_resource_allocation(
            crisis=sample_crisis,
            requirements=sample_requirements,
            available_resources=sample_resources,
            strategy=OptimizationStrategy.BALANCED_OPTIMIZATION
        )
        
        # Verify result structure
        assert result is not None
        assert result.allocation_plan is not None
        assert isinstance(result.optimization_score, (int, float))
        assert isinstance(result.total_cost, (int, float))
        assert isinstance(result.total_time, timedelta)
        assert isinstance(result.resource_utilization, dict)
        assert isinstance(result.unmet_requirements, list)
        assert isinstance(result.optimization_metrics, dict)
        assert isinstance(result.recommendations, list)
        
        # Verify allocation plan
        plan = result.allocation_plan
        assert plan.crisis_id == sample_crisis.id
        assert len(plan.allocations) > 0
        assert plan.plan_status == "approved"
        
        # Verify allocations are valid
        for allocation in plan.allocations:
            assert allocation.crisis_id == sample_crisis.id
            assert allocation.allocated_quantity > 0
            assert allocation.status == AllocationStatus.APPROVED
    
    @pytest.mark.asyncio
    async def test_optimization_strategies(self, optimizer, sample_crisis, sample_requirements, sample_resources):
        """Test different optimization strategies"""
        strategies = [
            OptimizationStrategy.PRIORITY_BASED,
            OptimizationStrategy.COST_MINIMIZATION,
            OptimizationStrategy.TIME_MINIMIZATION,
            OptimizationStrategy.BALANCED_OPTIMIZATION,
            OptimizationStrategy.CAPACITY_MAXIMIZATION
        ]
        
        results = {}
        
        for strategy in strategies:
            result = await optimizer.optimize_resource_allocation(
                crisis=sample_crisis,
                requirements=sample_requirements,
                available_resources=sample_resources,
                strategy=strategy
            )
            results[strategy] = result
            
            # Verify each strategy produces valid results
            assert result.optimization_score >= 0
            assert result.total_cost >= 0
            assert len(result.allocation_plan.allocations) >= 0
        
        # Verify strategies produce different results
        scores = [result.optimization_score for result in results.values()]
        costs = [result.total_cost for result in results.values()]
        
        # At least some variation in scores or costs
        assert len(set(scores)) > 1 or len(set(costs)) > 1
    
    @pytest.mark.asyncio
    async def test_allocation_with_constraints(self, optimizer, sample_crisis, sample_requirements, sample_resources):
        """Test allocation optimization with constraints"""
        constraints = {
            'budget_limit': 50000.0,
            'time_limit': timedelta(hours=6),
            'location_constraint': 'Data Center',
            'required_skills': ['incident_response']
        }
        
        result = await optimizer.optimize_resource_allocation(
            crisis=sample_crisis,
            requirements=sample_requirements,
            available_resources=sample_resources,
            strategy=OptimizationStrategy.BALANCED_OPTIMIZATION,
            constraints=constraints
        )
        
        # Verify constraints are considered
        assert result.total_cost <= constraints['budget_limit']
        assert len(result.allocation_plan.allocations) >= 0
        
        # Check that allocations respect constraints where possible
        for allocation in result.allocation_plan.allocations:
            assert allocation.cost_estimate >= 0
    
    @pytest.mark.asyncio
    async def test_track_allocation_progress(self, optimizer):
        """Test allocation progress tracking"""
        allocation_id = "test_allocation_001"
        
        progress = await optimizer.track_allocation_progress(allocation_id)
        
        # Verify progress structure
        assert isinstance(progress, dict)
        assert 'allocation_id' in progress
        assert 'status' in progress
        assert 'progress_percentage' in progress
        assert 'performance_metrics' in progress
        
        # Verify data types
        assert isinstance(progress['progress_percentage'], (int, float))
        assert 0 <= progress['progress_percentage'] <= 100
    
    @pytest.mark.asyncio
    async def test_adjust_allocation(self, optimizer):
        """Test allocation adjustment"""
        allocation_id = "test_allocation_001"
        adjustments = {
            'allocated_quantity': 25.0,
            'priority': 'high',
            'notes': 'Increased allocation due to escalation'
        }
        
        success = await optimizer.adjust_allocation(allocation_id, adjustments)
        
        # Verify adjustment success
        assert isinstance(success, bool)
        assert success is True  # Mock implementation always succeeds
    
    @pytest.mark.asyncio
    async def test_monitor_resource_utilization(self, optimizer):
        """Test resource utilization monitoring"""
        resource_ids = ["resource_001", "resource_002", "resource_003"]
        time_window = timedelta(hours=24)
        
        utilization_data = await optimizer.monitor_resource_utilization(resource_ids, time_window)
        
        # Verify utilization data structure
        assert isinstance(utilization_data, dict)
        assert len(utilization_data) == len(resource_ids)
        
        for resource_id in resource_ids:
            assert resource_id in utilization_data
            util = utilization_data[resource_id]
            assert isinstance(util, ResourceUtilization)
            assert util.resource_id == resource_id
            assert 0 <= util.actual_utilization <= 1
            assert 0 <= util.efficiency_score <= 1
    
    @pytest.mark.asyncio
    async def test_optimize_utilization(self, optimizer):
        """Test utilization optimization"""
        # Create mock allocation plan
        allocation_plan = AllocationPlan(
            id="test_plan_001",
            crisis_id="test_crisis_001",
            plan_name="Test Allocation Plan"
        )
        
        optimization_results = await optimizer.optimize_utilization(allocation_plan)
        
        # Verify optimization results structure
        assert isinstance(optimization_results, dict)
        assert 'current_efficiency' in optimization_results
        assert 'optimized_efficiency' in optimization_results
        assert 'potential_savings' in optimization_results
        assert 'optimization_actions' in optimization_results
        
        # Verify data types and ranges
        assert 0 <= optimization_results['current_efficiency'] <= 1
        assert 0 <= optimization_results['optimized_efficiency'] <= 1
        assert optimization_results['potential_savings'] >= 0
        assert isinstance(optimization_results['optimization_actions'], list)


class TestAllocationEngine:
    """Test cases for AllocationEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create AllocationEngine instance for testing"""
        return AllocationEngine()
    
    @pytest.fixture
    def sample_requirement(self):
        """Create sample resource requirement"""
        return ResourceRequirement(
            id="req_001",
            crisis_id="crisis_001",
            resource_type=ResourceType.HUMAN_RESOURCES,
            quantity_needed=10.0,
            priority=ResourcePriority.HIGH,
            duration_needed=timedelta(hours=8)
        )
    
    @pytest.fixture
    def sample_resource(self):
        """Create sample resource"""
        return Resource(
            id="resource_001",
            name="Test Resource",
            resource_type=ResourceType.HUMAN_RESOURCES,
            capacity=20.0,
            current_utilization=5.0,
            cost_per_hour=100.0
        )
    
    @pytest.mark.asyncio
    async def test_create_allocation(self, engine, sample_requirement, sample_resource):
        """Test allocation creation"""
        quantity = 8.0
        
        allocation = await engine.create_allocation(sample_requirement, sample_resource, quantity)
        
        # Verify allocation structure
        assert isinstance(allocation, ResourceAllocation)
        assert allocation.crisis_id == sample_requirement.crisis_id
        assert allocation.requirement_id == sample_requirement.id
        assert allocation.resource_id == sample_resource.id
        assert allocation.allocated_quantity == quantity
        assert allocation.allocation_priority == sample_requirement.priority
        assert allocation.estimated_duration == sample_requirement.duration_needed
        
        # Verify cost calculation
        expected_cost = sample_resource.cost_per_hour * quantity * (sample_requirement.duration_needed.total_seconds() / 3600)
        assert allocation.cost_estimate == expected_cost


class TestAllocationTrackingSystem:
    """Test cases for AllocationTrackingSystem"""
    
    @pytest.fixture
    def tracking_system(self):
        """Create AllocationTrackingSystem instance for testing"""
        return AllocationTrackingSystem()
    
    @pytest.mark.asyncio
    async def test_track_allocation(self, tracking_system):
        """Test allocation tracking"""
        allocation_id = "test_allocation_001"
        
        tracking_data = await tracking_system.track_allocation(allocation_id)
        
        # Verify tracking data structure
        assert isinstance(tracking_data, dict)
        assert tracking_data['allocation_id'] == allocation_id
        assert 'status' in tracking_data
        assert 'progress_percentage' in tracking_data
        assert 'performance_metrics' in tracking_data
        
        # Verify data types
        assert isinstance(tracking_data['progress_percentage'], (int, float))
        assert isinstance(tracking_data['performance_metrics'], dict)
    
    @pytest.mark.asyncio
    async def test_adjust_allocation(self, tracking_system):
        """Test allocation adjustment"""
        allocation_id = "test_allocation_001"
        adjustments = {
            'quantity': 15.0,
            'priority': 'critical'
        }
        
        success = await tracking_system.adjust_allocation(allocation_id, adjustments)
        
        # Verify adjustment
        assert isinstance(success, bool)
        assert success is True
        
        # Verify adjustment is recorded in history
        assert len(tracking_system.allocation_history) > 0
        last_adjustment = tracking_system.allocation_history[-1]
        assert last_adjustment['allocation_id'] == allocation_id
        assert last_adjustment['adjustments'] == adjustments


class TestUtilizationMonitor:
    """Test cases for UtilizationMonitor"""
    
    @pytest.fixture
    def monitor(self):
        """Create UtilizationMonitor instance for testing"""
        return UtilizationMonitor()
    
    @pytest.mark.asyncio
    async def test_monitor_utilization(self, monitor):
        """Test utilization monitoring"""
        resource_ids = ["resource_001", "resource_002"]
        time_window = timedelta(hours=12)
        
        utilization_data = await monitor.monitor_utilization(resource_ids, time_window)
        
        # Verify utilization data
        assert isinstance(utilization_data, dict)
        assert len(utilization_data) == len(resource_ids)
        
        for resource_id in resource_ids:
            assert resource_id in utilization_data
            util = utilization_data[resource_id]
            assert isinstance(util, ResourceUtilization)
            assert util.resource_id == resource_id
            assert isinstance(util.utilization_period, dict)
            assert 'start' in util.utilization_period
            assert 'end' in util.utilization_period
    
    @pytest.mark.asyncio
    async def test_optimize_utilization(self, monitor):
        """Test utilization optimization"""
        allocation_plan = AllocationPlan(
            id="test_plan",
            crisis_id="test_crisis",
            plan_name="Test Plan"
        )
        
        optimization_results = await monitor.optimize_utilization(allocation_plan)
        
        # Verify optimization results
        assert isinstance(optimization_results, dict)
        assert 'current_efficiency' in optimization_results
        assert 'optimized_efficiency' in optimization_results
        assert 'potential_savings' in optimization_results
        assert 'optimization_actions' in optimization_results
        assert 'implementation_timeline' in optimization_results
        assert 'risk_assessment' in optimization_results
        
        # Verify efficiency values are reasonable
        assert 0 <= optimization_results['current_efficiency'] <= 1
        assert 0 <= optimization_results['optimized_efficiency'] <= 1
        assert optimization_results['optimized_efficiency'] >= optimization_results['current_efficiency']


class TestAllocationScore:
    """Test cases for AllocationScore calculations"""
    
    def test_allocation_score_creation(self):
        """Test AllocationScore creation and structure"""
        score = AllocationScore(
            resource_id="resource_001",
            requirement_id="req_001",
            priority_score=0.9,
            cost_score=0.7,
            capability_score=0.8,
            availability_score=1.0,
            efficiency_score=0.85,
            total_score=0.85,
            allocation_feasible=True
        )
        
        # Verify score structure
        assert score.resource_id == "resource_001"
        assert score.requirement_id == "req_001"
        assert score.priority_score == 0.9
        assert score.cost_score == 0.7
        assert score.capability_score == 0.8
        assert score.availability_score == 1.0
        assert score.efficiency_score == 0.85
        assert score.total_score == 0.85
        assert score.allocation_feasible is True
        assert len(score.constraints_violated) == 0
    
    def test_allocation_score_with_constraints(self):
        """Test AllocationScore with constraint violations"""
        score = AllocationScore(
            resource_id="resource_001",
            requirement_id="req_001",
            allocation_feasible=False,
            constraints_violated=["budget_limit", "location_constraint"]
        )
        
        # Verify constraint handling
        assert score.allocation_feasible is False
        assert len(score.constraints_violated) == 2
        assert "budget_limit" in score.constraints_violated
        assert "location_constraint" in score.constraints_violated


class TestIntegration:
    """Integration tests for resource allocation system"""
    
    @pytest.fixture
    def optimizer(self):
        """Create ResourceAllocationOptimizer for integration testing"""
        return ResourceAllocationOptimizer()
    
    @pytest.mark.asyncio
    async def test_full_allocation_workflow(self, optimizer):
        """Test complete allocation workflow"""
        # Create test data
        crisis = Crisis(
            id="integration_test_crisis",
            crisis_type=CrisisType.SECURITY_BREACH,
            severity_level=SeverityLevel.CRITICAL
        )
        
        requirements = [
            ResourceRequirement(
                crisis_id=crisis.id,
                resource_type=ResourceType.HUMAN_RESOURCES,
                required_capabilities=["security_analysis"],
                quantity_needed=30.0,
                priority=ResourcePriority.CRITICAL,
                duration_needed=timedelta(hours=12)
            )
        ]
        
        resources = [
            Resource(
                id="security_team",
                name="Security Response Team",
                resource_type=ResourceType.HUMAN_RESOURCES,
                capabilities=[
                    ResourceCapability(name="security_analysis", proficiency_level=0.95)
                ],
                capacity=40.0,
                current_utilization=10.0,
                cost_per_hour=175.0
            )
        ]
        
        # Step 1: Optimize allocation
        result = await optimizer.optimize_resource_allocation(
            crisis=crisis,
            requirements=requirements,
            available_resources=resources,
            strategy=OptimizationStrategy.PRIORITY_BASED
        )
        
        # Verify optimization result
        assert result.allocation_plan is not None
        assert len(result.allocation_plan.allocations) > 0
        
        # Step 2: Track allocation progress
        allocation_id = result.allocation_plan.allocations[0].id
        progress = await optimizer.track_allocation_progress(allocation_id)
        assert progress is not None
        
        # Step 3: Monitor utilization
        resource_ids = [alloc.resource_id for alloc in result.allocation_plan.allocations]
        utilization_data = await optimizer.monitor_resource_utilization(resource_ids)
        assert len(utilization_data) == len(resource_ids)
        
        # Step 4: Optimize utilization
        optimization_results = await optimizer.optimize_utilization(result.allocation_plan)
        assert optimization_results is not None
        assert 'optimization_actions' in optimization_results
    
    @pytest.mark.asyncio
    async def test_constraint_handling_integration(self, optimizer):
        """Test constraint handling in full workflow"""
        crisis = Crisis(id="constraint_test_crisis")
        
        # Create requirements that will challenge constraints
        requirements = [
            ResourceRequirement(
                crisis_id=crisis.id,
                resource_type=ResourceType.HUMAN_RESOURCES,
                quantity_needed=100.0,  # High quantity
                priority=ResourcePriority.EMERGENCY,
                duration_needed=timedelta(hours=48),  # Long duration
                budget_limit=10000.0  # Limited budget
            )
        ]
        
        resources = [
            Resource(
                id="limited_resource",
                resource_type=ResourceType.HUMAN_RESOURCES,
                capacity=50.0,  # Limited capacity
                current_utilization=20.0,
                cost_per_hour=200.0  # High cost
            )
        ]
        
        constraints = {
            'budget_limit': 15000.0,
            'time_limit': timedelta(hours=24)
        }
        
        # Perform optimization with constraints
        result = await optimizer.optimize_resource_allocation(
            crisis=crisis,
            requirements=requirements,
            available_resources=resources,
            strategy=OptimizationStrategy.COST_MINIMIZATION,
            constraints=constraints
        )
        
        # Verify constraints are respected
        assert result.total_cost <= constraints['budget_limit']
        assert len(result.unmet_requirements) >= 0  # May have unmet requirements due to constraints
        assert len(result.recommendations) > 0  # Should have recommendations for handling constraints


if __name__ == "__main__":
    pytest.main([__file__])