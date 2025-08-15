"""
Hyperscale Infrastructure Tests

Comprehensive tests for billion-user scale infrastructure management.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
import random

from scrollintel.models.hyperscale_models import (
    HyperscaleMetrics, RegionalMetrics, GlobalInfrastructure,
    CloudProvider, ResourceType, ScalingEvent, ScalingDirection,
    CostOptimization, CapacityPlan
)
from scrollintel.engines.global_infra_manager import GlobalInfraManager
from scrollintel.engines.hyperscale_autoscaler import HyperscaleAutoScaler
from scrollintel.engines.hyperscale_cost_optimizer import HyperscaleCostOptimizer


@pytest.fixture
def infra_manager():
    return GlobalInfraManager()


@pytest.fixture
def autoscaler():
    return HyperscaleAutoScaler()


@pytest.fixture
def cost_optimizer():
    return HyperscaleCostOptimizer()


@pytest.fixture
def billion_user_metrics():
    """Create metrics for billion-user scenario"""
    regional_metrics = {}
    
    # Create metrics for 20 regions (simplified for testing)
    regions = [f"region-{i}" for i in range(20)]
    
    for region in regions:
        regional_metrics[region] = RegionalMetrics(
            region=region,
            provider=random.choice(list(CloudProvider)),
            active_users=random.randint(40_000_000, 60_000_000),  # 40-60M users per region
            requests_per_second=random.randint(400_000, 600_000),  # 400-600k RPS per region
            cpu_utilization=random.uniform(60.0, 85.0),
            memory_utilization=random.uniform(65.0, 80.0),
            network_throughput=random.uniform(1000.0, 5000.0),
            storage_iops=random.randint(50000, 100000),
            latency_p95=random.uniform(50.0, 120.0),
            error_rate=random.uniform(0.001, 0.01),
            cost_per_hour=random.uniform(5000.0, 15000.0),
            timestamp=datetime.now()
        )
    
    return HyperscaleMetrics(
        id="billion-user-test",
        timestamp=datetime.now(),
        global_requests_per_second=sum(m.requests_per_second for m in regional_metrics.values()),
        active_users=sum(m.active_users for m in regional_metrics.values()),
        total_servers=200_000,  # 200k servers
        total_data_centers=20,
        infrastructure_utilization={
            "cpu": 72.5,
            "memory": 70.0,
            "network": 65.0,
            "storage": 80.0
        },
        performance_metrics={
            "avg_latency": 85.0,
            "p95_latency": 150.0,
            "error_rate": 0.005,
            "availability": 99.99
        },
        cost_metrics={
            "hourly_cost": 200_000.0,  # $200k/hour
            "monthly_cost": 144_000_000.0  # $144M/month
        },
        regional_distribution=regional_metrics
    )


@pytest.mark.asyncio
async def test_billion_user_capacity_planning(infra_manager):
    """Test capacity planning for billion users"""
    
    target_users = 1_000_000_000
    growth_timeline = {
        "phase_1": datetime.now() + timedelta(days=90),
        "phase_2": datetime.now() + timedelta(days=180),
        "phase_3": datetime.now() + timedelta(days=365)
    }
    performance_requirements = {
        "max_latency_ms": 100,
        "min_availability": 99.99,
        "max_error_rate": 0.01
    }
    
    capacity_plan = await infra_manager.plan_billion_user_capacity(
        target_users, growth_timeline, performance_requirements
    )
    
    # Validate capacity plan
    assert capacity_plan.target_users == target_users
    assert capacity_plan.target_rps >= 100_000_000  # At least 100M RPS
    assert len(capacity_plan.regions) >= 20  # Minimum 20 regions
    assert capacity_plan.estimated_cost > 0
    assert len(capacity_plan.risk_factors) > 0
    assert len(capacity_plan.contingency_plans) > 0
    
    # Validate resource requirements
    assert ResourceType.COMPUTE in capacity_plan.resource_requirements
    assert capacity_plan.resource_requirements[ResourceType.COMPUTE] >= 100_000
    
    print(f"Capacity plan for {target_users:,} users:")
    print(f"- Regions: {len(capacity_plan.regions)}")
    print(f"- Estimated cost: ${capacity_plan.estimated_cost:,.2f}/month")
    print(f"- Compute resources: {capacity_plan.resource_requirements[ResourceType.COMPUTE]:,}")


@pytest.mark.asyncio
async def test_hyperscale_performance_monitoring(infra_manager, billion_user_metrics):
    """Test performance monitoring at hyperscale"""
    
    # Set up infrastructure state
    infrastructure = GlobalInfrastructure(
        id="test-infra",
        name="Billion User Infrastructure",
        total_capacity={
            ResourceType.COMPUTE: 200_000,
            ResourceType.STORAGE: 50_000_000,
            ResourceType.NETWORK: 5_000_000
        },
        current_utilization={
            ResourceType.COMPUTE: 72.5,
            ResourceType.STORAGE: 80.0,
            ResourceType.NETWORK: 65.0
        },
        regions=list(billion_user_metrics.regional_distribution.keys()),
        providers=[CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP],
        auto_scaling_enabled=True,
        cost_optimization_enabled=True,
        performance_targets={"latency": 100.0, "availability": 99.99},
        cost_targets={"monthly_budget": 200_000_000.0},
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    infra_manager.infrastructure_state["test-infra"] = infrastructure
    
    # Mock regional metrics collection
    async def mock_collect_regional_metrics(region):
        return billion_user_metrics.regional_distribution[region]
    
    infra_manager._collect_regional_metrics = mock_collect_regional_metrics
    
    # Monitor performance
    monitored_metrics = await infra_manager.monitor_hyperscale_performance("test-infra")
    
    # Validate monitoring results
    assert monitored_metrics.active_users >= 800_000_000  # At least 800M users
    assert monitored_metrics.global_requests_per_second >= 8_000_000  # At least 8M RPS
    assert monitored_metrics.total_servers == 200_000
    assert monitored_metrics.total_data_centers == 20
    assert len(monitored_metrics.regional_distribution) == 20
    
    print(f"Monitoring results:")
    print(f"- Active users: {monitored_metrics.active_users:,}")
    print(f"- Global RPS: {monitored_metrics.global_requests_per_second:,}")
    print(f"- Regions monitored: {len(monitored_metrics.regional_distribution)}")


@pytest.mark.asyncio
async def test_auto_scaling_billion_users(autoscaler, billion_user_metrics):
    """Test auto-scaling for billion-user load"""
    
    # Create demand forecast
    demand_forecast = {}
    for region in billion_user_metrics.regional_distribution.keys():
        demand_forecast[f"{region}_demand"] = random.uniform(1.2, 2.0)  # 20-100% increase
    
    # Mock scaling execution
    autoscaler._execute_scaling_event = AsyncMock(return_value=True)
    
    # Perform auto-scaling
    scaling_events = await autoscaler.auto_scale_resources(
        billion_user_metrics, demand_forecast
    )
    
    # Validate scaling events
    assert len(scaling_events) > 0
    
    # Check for different types of scaling
    scale_up_events = [e for e in scaling_events if e.direction in [ScalingDirection.UP, ScalingDirection.OUT]]
    
    assert len(scale_up_events) > 0  # Should have scale-up events
    
    # Validate scaling event properties
    for event in scaling_events:
        assert event.region in billion_user_metrics.regional_distribution
        assert event.scale_factor > 0
        assert event.instances_after >= event.instances_before or event.direction in [ScalingDirection.DOWN, ScalingDirection.IN]
        assert event.cost_impact != 0
    
    print(f"Auto-scaling results:")
    print(f"- Total scaling events: {len(scaling_events)}")
    print(f"- Scale-up events: {len(scale_up_events)}")


@pytest.mark.asyncio
async def test_cost_optimization_billion_scale(cost_optimizer, billion_user_metrics):
    """Test cost optimization for billion-user infrastructure"""
    
    # Set cost targets
    cost_targets = {
        'cost_reduction_target': 30.0,  # 30% cost reduction target
        'compute_utilization_target': 80.0,
        'reserved_instance_ratio': 75.0
    }
    
    # Optimize costs
    optimizations = await cost_optimizer.optimize_infrastructure_costs(
        billion_user_metrics, cost_targets
    )
    
    # Validate optimizations
    assert len(optimizations) > 0
    
    # Check optimization categories
    categories = {opt.optimization_category for opt in optimizations}
    expected_categories = {
        'compute_rightsizing', 'storage_tiering', 'network_cdn',
        'database_optimization', 'multicloud_arbitrage', 'reserved_capacity'
    }
    assert len(categories.intersection(expected_categories)) > 0
    
    # Validate savings potential
    total_savings = sum(opt.savings_potential for opt in optimizations)
    assert total_savings > 0
    
    # Check for high-impact optimizations
    high_impact_opts = [opt for opt in optimizations if opt.savings_potential > 1_000_000]
    assert len(high_impact_opts) > 0  # Should have million-dollar savings opportunities
    
    print(f"Cost optimization results:")
    print(f"- Total optimizations: {len(optimizations)}")
    print(f"- Total potential savings: ${total_savings:,.2f}/month")
    print(f"- High-impact optimizations: {len(high_impact_opts)}")


@pytest.mark.asyncio
async def test_traffic_surge_handling(autoscaler):
    """Test emergency scaling for traffic surges"""
    
    region = "us-east-1"
    surge_magnitude = 10.0  # 10x traffic increase
    duration_estimate = 3600  # 1 hour
    
    # Mock capacity methods
    async def mock_get_current_capacity(region):
        return 100_000.0  # 100k RPS current capacity
    
    async def mock_get_current_instances(region, resource_type):
        return 1000  # 1000 current instances
    
    autoscaler._get_current_capacity = mock_get_current_capacity
    autoscaler._get_current_instances = mock_get_current_instances
    autoscaler._execute_scaling_event = AsyncMock(return_value=True)
    
    # Handle traffic surge
    scaling_events = await autoscaler.handle_traffic_surge(
        region, surge_magnitude, duration_estimate
    )
    
    # Validate emergency scaling
    assert len(scaling_events) > 0
    
    # Check for compute scaling
    compute_events = [e for e in scaling_events if e.resource_type == ResourceType.COMPUTE]
    assert len(compute_events) > 0
    
    # Validate scaling magnitude
    for event in compute_events:
        assert event.scale_factor >= surge_magnitude * 0.8  # At least 80% of required scaling
        assert event.trigger_metric == "emergency_scaling"
        assert event.trigger_value == surge_magnitude * 100_000.0  # Required capacity
    
    print(f"Emergency scaling results:")
    print(f"- Scaling events: {len(scaling_events)}")
    print(f"- Compute scaling factor: {compute_events[0].scale_factor:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])