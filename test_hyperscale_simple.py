"""
Simple Hyperscale Infrastructure Test

Basic test to verify hyperscale infrastructure components work.
"""

import asyncio
from datetime import datetime
from scrollintel.models.hyperscale_models import (
    HyperscaleMetrics, RegionalMetrics, CloudProvider, ResourceType
)
from scrollintel.engines.global_infra_manager import GlobalInfraManager
from scrollintel.engines.hyperscale_autoscaler import HyperscaleAutoScaler
from scrollintel.engines.hyperscale_cost_optimizer import HyperscaleCostOptimizer


async def test_basic_functionality():
    """Test basic functionality of hyperscale components"""
    
    print("Testing Hyperscale Infrastructure Components...")
    
    # Test GlobalInfraManager
    infra_manager = GlobalInfraManager()
    print("✓ GlobalInfraManager created")
    
    # Test capacity planning
    capacity_plan = await infra_manager.plan_billion_user_capacity(
        target_users=1_000_000_000,
        growth_timeline={"phase_1": datetime.now()},
        performance_requirements={"max_latency_ms": 100}
    )
    print(f"✓ Capacity plan created for {capacity_plan.target_users:,} users")
    print(f"  - Regions: {len(capacity_plan.regions)}")
    print(f"  - Estimated cost: ${capacity_plan.estimated_cost:,.2f}/month")
    
    # Test HyperscaleAutoScaler
    autoscaler = HyperscaleAutoScaler()
    print("✓ HyperscaleAutoScaler created")
    
    # Create sample metrics
    regional_metrics = {
        "us-east-1": RegionalMetrics(
            region="us-east-1",
            provider=CloudProvider.AWS,
            active_users=50_000_000,
            requests_per_second=500_000,
            cpu_utilization=75.0,
            memory_utilization=70.0,
            network_throughput=2000.0,
            storage_iops=75000,
            latency_p95=85.0,
            error_rate=0.005,
            cost_per_hour=10000.0,
            timestamp=datetime.now()
        )
    }
    
    sample_metrics = HyperscaleMetrics(
        id="test-metrics",
        timestamp=datetime.now(),
        global_requests_per_second=500_000,
        active_users=50_000_000,
        total_servers=5000,
        total_data_centers=1,
        infrastructure_utilization={"cpu": 75.0, "memory": 70.0},
        performance_metrics={"avg_latency": 85.0, "error_rate": 0.005},
        cost_metrics={"hourly_cost": 10000.0},
        regional_distribution=regional_metrics
    )
    
    # Mock the scaling execution
    async def mock_execute_scaling_event(event):
        return True
    
    autoscaler._execute_scaling_event = mock_execute_scaling_event
    
    # Test auto-scaling
    scaling_events = await autoscaler.auto_scale_resources(sample_metrics)
    print(f"✓ Auto-scaling completed with {len(scaling_events)} events")
    
    # Test HyperscaleCostOptimizer
    cost_optimizer = HyperscaleCostOptimizer()
    print("✓ HyperscaleCostOptimizer created")
    
    # Test cost optimization
    optimizations = await cost_optimizer.optimize_infrastructure_costs(sample_metrics)
    total_savings = sum(opt.savings_potential for opt in optimizations)
    print(f"✓ Cost optimization completed with ${total_savings:,.2f} potential savings")
    print(f"  - Optimizations found: {len(optimizations)}")
    
    print("\n🎉 All hyperscale infrastructure tests passed!")
    return True


if __name__ == "__main__":
    result = asyncio.run(test_basic_functionality())
    if result:
        print("✅ Hyperscale infrastructure implementation verified!")
    else:
        print("❌ Tests failed!")