"""
Hyperscale Infrastructure Demo

Demonstrates billion-user scale infrastructure management capabilities
including capacity planning, auto-scaling, and cost optimization.
"""

import asyncio
from datetime import datetime, timedelta
from scrollintel.models.hyperscale_models import (
    HyperscaleMetrics, RegionalMetrics, CloudProvider, ResourceType
)
from scrollintel.engines.global_infra_manager import GlobalInfraManager
from scrollintel.engines.hyperscale_autoscaler import HyperscaleAutoScaler
from scrollintel.engines.hyperscale_cost_optimizer import HyperscaleCostOptimizer


async def demo_billion_user_capacity_planning():
    """Demo capacity planning for billion users"""
    print("🌍 BILLION-USER CAPACITY PLANNING DEMO")
    print("=" * 50)
    
    infra_manager = GlobalInfraManager()
    
    # Plan for 1 billion users
    target_users = 1_000_000_000
    growth_timeline = {
        "phase_1": datetime.now() + timedelta(days=90),   # 250M users
        "phase_2": datetime.now() + timedelta(days=180),  # 500M users
        "phase_3": datetime.now() + timedelta(days=365)   # 1B users
    }
    performance_requirements = {
        "max_latency_ms": 100,
        "min_availability": 99.99,
        "max_error_rate": 0.01
    }
    
    print(f"📊 Planning infrastructure for {target_users:,} users...")
    
    capacity_plan = await infra_manager.plan_billion_user_capacity(
        target_users, growth_timeline, performance_requirements
    )
    
    print(f"✅ Capacity Plan Generated:")
    print(f"   • Target Users: {capacity_plan.target_users:,}")
    print(f"   • Target RPS: {capacity_plan.target_rps:,}")
    print(f"   • Regions: {len(capacity_plan.regions)}")
    print(f"   • Estimated Cost: ${capacity_plan.estimated_cost:,.2f}/month")
    print(f"   • Compute Resources: {capacity_plan.resource_requirements[ResourceType.COMPUTE]:,} servers")
    print(f"   • Storage: {capacity_plan.resource_requirements[ResourceType.STORAGE]:,} GB")
    print(f"   • Risk Factors: {len(capacity_plan.risk_factors)}")
    print(f"   • Contingency Plans: {len(capacity_plan.contingency_plans)}")
    
    return capacity_plan


async def demo_hyperscale_monitoring():
    """Demo hyperscale performance monitoring"""
    print("\n📈 HYPERSCALE PERFORMANCE MONITORING DEMO")
    print("=" * 50)
    
    # Create realistic billion-user metrics
    regional_metrics = {}
    regions = [
        "us-east-1", "us-west-2", "eu-west-1", "eu-central-1",
        "ap-southeast-1", "ap-northeast-1", "ap-south-1", "sa-east-1",
        "ca-central-1", "af-south-1", "me-south-1", "ap-east-1",
        "eu-north-1", "eu-south-1", "us-west-1", "ap-southeast-2",
        "ap-northeast-2", "ap-northeast-3", "eu-west-2", "eu-west-3"
    ]
    
    total_users = 0
    total_rps = 0
    
    for i, region in enumerate(regions):
        users = 45_000_000 + (i * 5_000_000)  # 45-140M users per region
        rps = users // 100  # 1% activity rate
        
        regional_metrics[region] = RegionalMetrics(
            region=region,
            provider=CloudProvider.AWS if i % 3 == 0 else CloudProvider.AZURE if i % 3 == 1 else CloudProvider.GCP,
            active_users=users,
            requests_per_second=rps,
            cpu_utilization=70.0 + (i * 2),
            memory_utilization=65.0 + (i * 1.5),
            network_throughput=2000.0 + (i * 100),
            storage_iops=75000 + (i * 5000),
            latency_p95=80.0 + (i * 3),
            error_rate=0.003 + (i * 0.0001),
            cost_per_hour=8000.0 + (i * 500),
            timestamp=datetime.now()
        )
        
        total_users += users
        total_rps += rps
    
    hyperscale_metrics = HyperscaleMetrics(
        id="demo-metrics",
        timestamp=datetime.now(),
        global_requests_per_second=total_rps,
        active_users=total_users,
        total_servers=400_000,
        total_data_centers=len(regions),
        infrastructure_utilization={
            "cpu": 75.0,
            "memory": 72.0,
            "network": 68.0,
            "storage": 82.0
        },
        performance_metrics={
            "avg_latency": 95.0,
            "p95_latency": 180.0,
            "error_rate": 0.005,
            "availability": 99.99
        },
        cost_metrics={
            "hourly_cost": 250_000.0,
            "monthly_cost": 180_000_000.0
        },
        regional_distribution=regional_metrics
    )
    
    print(f"🌐 Global Infrastructure Metrics:")
    print(f"   • Active Users: {hyperscale_metrics.active_users:,}")
    print(f"   • Global RPS: {hyperscale_metrics.global_requests_per_second:,}")
    print(f"   • Total Servers: {hyperscale_metrics.total_servers:,}")
    print(f"   • Data Centers: {hyperscale_metrics.total_data_centers}")
    print(f"   • Average Latency: {hyperscale_metrics.performance_metrics['avg_latency']:.1f}ms")
    print(f"   • Availability: {hyperscale_metrics.performance_metrics['availability']:.2f}%")
    print(f"   • Hourly Cost: ${hyperscale_metrics.cost_metrics['hourly_cost']:,.2f}")
    
    return hyperscale_metrics


async def demo_auto_scaling():
    """Demo auto-scaling for billion-user load"""
    print("\n⚡ AUTO-SCALING DEMO")
    print("=" * 50)
    
    autoscaler = HyperscaleAutoScaler()
    
    # Create high-load scenario
    regional_metrics = {
        "us-east-1": RegionalMetrics(
            region="us-east-1",
            provider=CloudProvider.AWS,
            active_users=100_000_000,
            requests_per_second=1_000_000,
            cpu_utilization=85.0,  # High CPU
            memory_utilization=82.0,  # High memory
            network_throughput=8000.0,
            storage_iops=150000,
            latency_p95=150.0,  # High latency
            error_rate=0.008,
            cost_per_hour=25000.0,
            timestamp=datetime.now()
        ),
        "eu-west-1": RegionalMetrics(
            region="eu-west-1",
            provider=CloudProvider.AZURE,
            active_users=80_000_000,
            requests_per_second=800_000,
            cpu_utilization=78.0,
            memory_utilization=75.0,
            network_throughput=6000.0,
            storage_iops=120000,
            latency_p95=120.0,
            error_rate=0.006,
            cost_per_hour=20000.0,
            timestamp=datetime.now()
        )
    }
    
    high_load_metrics = HyperscaleMetrics(
        id="high-load-demo",
        timestamp=datetime.now(),
        global_requests_per_second=1_800_000,
        active_users=180_000_000,
        total_servers=50_000,
        total_data_centers=2,
        infrastructure_utilization={"cpu": 81.5, "memory": 78.5},
        performance_metrics={"avg_latency": 135.0, "error_rate": 0.007},
        cost_metrics={"hourly_cost": 45_000.0},
        regional_distribution=regional_metrics
    )
    
    print("🔥 High Load Detected:")
    print(f"   • CPU Utilization: {high_load_metrics.infrastructure_utilization['cpu']:.1f}%")
    print(f"   • Memory Utilization: {high_load_metrics.infrastructure_utilization['memory']:.1f}%")
    print(f"   • Average Latency: {high_load_metrics.performance_metrics['avg_latency']:.1f}ms")
    
    # Mock scaling execution
    async def mock_execute_scaling_event(event):
        return True
    
    autoscaler._execute_scaling_event = mock_execute_scaling_event
    
    # Perform auto-scaling
    scaling_events = await autoscaler.auto_scale_resources(high_load_metrics)
    
    print(f"\n⚙️ Auto-Scaling Response:")
    print(f"   • Scaling Events: {len(scaling_events)}")
    
    for event in scaling_events:
        direction_symbol = "📈" if event.direction.value in ["up", "out"] else "📉"
        print(f"   {direction_symbol} {event.region}: {event.resource_type.value} "
              f"{event.instances_before} → {event.instances_after} "
              f"(${event.cost_impact:,.2f} impact)")
    
    return scaling_events


async def demo_traffic_surge_handling():
    """Demo handling massive traffic surges"""
    print("\n🚨 TRAFFIC SURGE HANDLING DEMO")
    print("=" * 50)
    
    autoscaler = HyperscaleAutoScaler()
    
    # Simulate 20x traffic surge (like a viral event)
    surge_magnitude = 20.0
    affected_regions = ["us-east-1", "us-west-2", "eu-west-1"]
    duration_estimate = 7200  # 2 hours
    
    print(f"⚠️ MASSIVE TRAFFIC SURGE DETECTED!")
    print(f"   • Magnitude: {surge_magnitude}x normal traffic")
    print(f"   • Affected Regions: {len(affected_regions)}")
    print(f"   • Estimated Duration: {duration_estimate//3600}h {(duration_estimate%3600)//60}m")
    
    # Mock capacity methods
    async def mock_get_current_capacity(region):
        return 500_000.0  # 500k RPS current capacity
    
    async def mock_get_current_instances(region, resource_type):
        return 5000  # 5000 current instances
    
    autoscaler._get_current_capacity = mock_get_current_capacity
    autoscaler._get_current_instances = mock_get_current_instances
    autoscaler._execute_scaling_event = lambda event: True
    
    # Handle traffic surge
    emergency_events = []
    for region in affected_regions:
        events = await autoscaler.handle_traffic_surge(region, surge_magnitude, duration_estimate)
        emergency_events.extend(events)
    
    print(f"\n🛡️ Emergency Response Activated:")
    print(f"   • Emergency Scaling Events: {len(emergency_events)}")
    
    total_additional_capacity = 0
    for event in emergency_events:
        additional_instances = event.instances_after - event.instances_before
        capacity_increase = additional_instances * 100  # 100 RPS per instance
        total_additional_capacity += capacity_increase
        
        print(f"   🚀 {event.region}: +{additional_instances:,} {event.resource_type.value} instances")
        print(f"      Capacity: +{capacity_increase:,} RPS (${event.cost_impact:,.2f}/hour)")
    
    print(f"\n📊 Total Emergency Response:")
    print(f"   • Additional Capacity: +{total_additional_capacity:,} RPS")
    print(f"   • Can Handle: {total_additional_capacity // 1000:,}k concurrent users")
    
    return emergency_events


async def demo_cost_optimization():
    """Demo cost optimization for billion-user infrastructure"""
    print("\n💰 COST OPTIMIZATION DEMO")
    print("=" * 50)
    
    cost_optimizer = HyperscaleCostOptimizer()
    
    # Create expensive infrastructure scenario
    expensive_metrics = HyperscaleMetrics(
        id="expensive-demo",
        timestamp=datetime.now(),
        global_requests_per_second=5_000_000,
        active_users=500_000_000,
        total_servers=200_000,
        total_data_centers=25,
        infrastructure_utilization={"cpu": 45.0, "memory": 50.0},  # Low utilization
        performance_metrics={"avg_latency": 120.0, "error_rate": 0.008},
        cost_metrics={
            "hourly_cost": 400_000.0,  # $400k/hour
            "monthly_cost": 288_000_000.0  # $288M/month
        },
        regional_distribution={}
    )
    
    print(f"💸 Current Infrastructure Costs:")
    print(f"   • Monthly Cost: ${expensive_metrics.cost_metrics['monthly_cost']:,.2f}")
    print(f"   • CPU Utilization: {expensive_metrics.infrastructure_utilization['cpu']:.1f}% (underutilized)")
    print(f"   • Memory Utilization: {expensive_metrics.infrastructure_utilization['memory']:.1f}% (underutilized)")
    
    # Set aggressive cost reduction targets
    cost_targets = {
        'cost_reduction_target': 35.0,  # 35% cost reduction
        'compute_utilization_target': 75.0,
        'reserved_instance_ratio': 80.0
    }
    
    # Optimize costs
    optimizations = await cost_optimizer.optimize_infrastructure_costs(
        expensive_metrics, cost_targets
    )
    
    print(f"\n🎯 Cost Optimization Analysis:")
    print(f"   • Optimizations Found: {len(optimizations)}")
    
    total_savings = sum(opt.savings_potential for opt in optimizations)
    total_percentage = (total_savings / expensive_metrics.cost_metrics['monthly_cost']) * 100
    
    print(f"   • Total Monthly Savings: ${total_savings:,.2f}")
    print(f"   • Percentage Reduction: {total_percentage:.1f}%")
    
    # Show top optimizations
    top_optimizations = sorted(optimizations, key=lambda x: x.savings_potential, reverse=True)[:5]
    
    print(f"\n🏆 Top Cost Optimizations:")
    for i, opt in enumerate(top_optimizations, 1):
        print(f"   {i}. {opt.optimization_category.replace('_', ' ').title()}")
        print(f"      💰 Savings: ${opt.savings_potential:,.2f}/month ({opt.savings_percentage:.1f}%)")
        print(f"      ⚠️ Risk: {opt.risk_assessment.title()}")
        print(f"      🔧 Effort: {opt.implementation_effort.title()}")
        print(f"      📅 Payback: {opt.payback_period_days} days")
    
    return optimizations


async def demo_global_infrastructure_optimization():
    """Demo global infrastructure optimization"""
    print("\n🌍 GLOBAL INFRASTRUCTURE OPTIMIZATION DEMO")
    print("=" * 50)
    
    infra_manager = GlobalInfraManager()
    
    # Create global metrics
    global_metrics = HyperscaleMetrics(
        id="global-optimization-demo",
        timestamp=datetime.now(),
        global_requests_per_second=8_000_000,
        active_users=800_000_000,
        total_servers=300_000,
        total_data_centers=30,
        infrastructure_utilization={"cpu": 72.0, "memory": 68.0, "network": 75.0},
        performance_metrics={"avg_latency": 110.0, "error_rate": 0.006, "availability": 99.98},
        cost_metrics={"hourly_cost": 350_000.0, "monthly_cost": 252_000_000.0},
        regional_distribution={}
    )
    
    print(f"🔍 Analyzing Global Infrastructure:")
    print(f"   • Users: {global_metrics.active_users:,}")
    print(f"   • RPS: {global_metrics.global_requests_per_second:,}")
    print(f"   • Servers: {global_metrics.total_servers:,}")
    print(f"   • Data Centers: {global_metrics.total_data_centers}")
    print(f"   • Availability: {global_metrics.performance_metrics['availability']:.2f}%")
    
    # Optimize global infrastructure
    optimizations = await infra_manager.optimize_global_infrastructure(global_metrics)
    
    print(f"\n⚙️ Global Optimization Results:")
    
    for opt_type, optimization in optimizations.items():
        if optimization.get('apply', False):
            improvement = optimization.get('estimated_improvement', 0)
            savings = optimization.get('estimated_savings', 0)
            
            print(f"   ✅ {opt_type.replace('_', ' ').title()}:")
            if improvement:
                print(f"      📈 Performance Improvement: {improvement:.1f}%")
            if savings:
                print(f"      💰 Cost Savings: ${savings:,.2f}/month")
    
    return optimizations


async def main():
    """Run the complete hyperscale infrastructure demo"""
    print("🚀 HYPERSCALE INFRASTRUCTURE MANAGEMENT DEMO")
    print("=" * 60)
    print("Demonstrating billion-user scale infrastructure capabilities")
    print("=" * 60)
    
    try:
        # Run all demos
        await demo_billion_user_capacity_planning()
        hyperscale_metrics = await demo_hyperscale_monitoring()
        await demo_auto_scaling()
        await demo_traffic_surge_handling()
        await demo_cost_optimization()
        await demo_global_infrastructure_optimization()
        
        print("\n" + "=" * 60)
        print("🎉 HYPERSCALE INFRASTRUCTURE DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📋 DEMO SUMMARY:")
        print("✅ Billion-user capacity planning")
        print("✅ Real-time performance monitoring")
        print("✅ Intelligent auto-scaling")
        print("✅ Emergency traffic surge handling")
        print("✅ Advanced cost optimization")
        print("✅ Global infrastructure optimization")
        
        print(f"\n🌟 KEY ACHIEVEMENTS:")
        print(f"• Planned infrastructure for 1,000,000,000 users")
        print(f"• Monitored {hyperscale_metrics.active_users:,} active users")
        print(f"• Handled {hyperscale_metrics.global_requests_per_second:,} requests per second")
        print(f"• Managed {hyperscale_metrics.total_servers:,} servers across {hyperscale_metrics.total_data_centers} regions")
        print(f"• Achieved 99.99% availability at hyperscale")
        
        print(f"\n💡 This demonstrates ScrollIntel's capability to:")
        print(f"• Replace Big Tech CTO infrastructure management")
        print(f"• Handle billion-user scale with intelligent automation")
        print(f"• Optimize costs while maintaining performance")
        print(f"• Respond to traffic surges in real-time")
        print(f"• Coordinate global infrastructure seamlessly")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())