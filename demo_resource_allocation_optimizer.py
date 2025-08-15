#!/usr/bin/env python3
"""
Demo script for Resource Allocation Optimizer.
Demonstrates optimal resource allocation for experimental activities.
"""
import sys
import os
sys.path.append('.')

from datetime import datetime, timedelta
from scrollintel.engines.resource_allocation_optimizer import (
    ResourceAllocationOptimizer, ResourceType, AllocationStatus,
    ResourceAvailability, ResourceAllocation
)
from scrollintel.models.experimental_design_models import ResourceRequirement

def main():
    """Demonstrate resource allocation optimizer capabilities."""
    print("🔬 ScrollIntel™ Resource Allocation Optimizer Demo")
    print("=" * 60)
    
    # Initialize the optimizer
    optimizer = ResourceAllocationOptimizer()
    
    # Demo 1: Basic Resource Allocation Optimization
    print("\n📊 Demo 1: Basic Resource Allocation Optimization")
    print("-" * 50)
    
    # Create experimental resource requirements
    requirements = [
        ResourceRequirement(
            resource_type="personnel",
            resource_name="Senior Research Scientist",
            quantity_needed=2,
            duration_needed=timedelta(weeks=8),
            cost_estimate=16000.0
        ),
        ResourceRequirement(
            resource_type="equipment",
            resource_name="High-Performance Computing Cluster",
            quantity_needed=4,
            duration_needed=timedelta(weeks=6),
            cost_estimate=12000.0
        ),
        ResourceRequirement(
            resource_type="materials",
            resource_name="Laboratory Reagents",
            quantity_needed=10,
            duration_needed=timedelta(weeks=4),
            cost_estimate=3000.0
        ),
        ResourceRequirement(
            resource_type="facilities",
            resource_name="Clean Room Access",
            quantity_needed=1,
            duration_needed=timedelta(weeks=12),
            cost_estimate=8000.0
        )
    ]
    
    # Create available resources
    available_resources = [
        ResourceAvailability(
            resource_id="personnel_pool_001",
            resource_type=ResourceType.PERSONNEL,
            resource_name="Research Scientists Pool",
            total_capacity=8.0,
            available_capacity=5.0,
            availability_schedule={},
            cost_per_unit=1000.0,
            location="Research Building A"
        ),
        ResourceAvailability(
            resource_id="hpc_cluster_001",
            resource_type=ResourceType.COMPUTATIONAL,
            resource_name="HPC Cluster Alpha",
            total_capacity=16.0,
            available_capacity=12.0,
            availability_schedule={},
            cost_per_unit=500.0,
            location="Data Center"
        ),
        ResourceAvailability(
            resource_id="materials_inventory_001",
            resource_type=ResourceType.MATERIALS,
            resource_name="Chemical Reagents Inventory",
            total_capacity=50.0,
            available_capacity=35.0,
            availability_schedule={},
            cost_per_unit=150.0,
            location="Chemical Storage"
        ),
        ResourceAvailability(
            resource_id="cleanroom_001",
            resource_type=ResourceType.FACILITIES,
            resource_name="Class 100 Clean Room",
            total_capacity=2.0,
            available_capacity=1.0,
            availability_schedule={},
            cost_per_unit=2000.0,
            location="Fabrication Facility"
        )
    ]
    
    print(f"📋 Resource Requirements:")
    for i, req in enumerate(requirements, 1):
        print(f"   {i}. {req.resource_name} ({req.resource_type})")
        print(f"      Quantity: {req.quantity_needed}, Duration: {req.duration_needed.days} days")
        print(f"      Estimated Cost: ${req.cost_estimate:,.2f}")
    
    print(f"\n🏭 Available Resources:")
    for i, res in enumerate(available_resources, 1):
        print(f"   {i}. {res.resource_name} ({res.resource_type.value})")
        print(f"      Capacity: {res.available_capacity}/{res.total_capacity}")
        print(f"      Cost per unit: ${res.cost_per_unit:,.2f}")
        print(f"      Location: {res.location}")
    
    # Perform optimization
    print(f"\n⚡ Performing Resource Allocation Optimization...")
    optimization = optimizer.optimize_resource_allocation(
        resource_requirements=requirements,
        available_resources=available_resources,
        constraints={'max_budget': 35000.0},
        optimization_goals=['efficiency', 'cost', 'utilization']
    )
    
    print(f"\n✅ Optimization Results:")
    print(f"   📊 Efficiency Gain: {optimization.efficiency_gain:.2f}%")
    print(f"   💰 Cost Savings: ${optimization.cost_savings:,.2f}")
    print(f"   🎯 Confidence Score: {optimization.confidence_score:.2f}")
    print(f"   📈 Optimization Strategies Applied:")
    for strategy in optimization.optimization_strategies:
        print(f"      • {strategy}")
    
    print(f"\n📋 Optimized Allocations:")
    for i, alloc in enumerate(optimization.optimized_allocations, 1):
        print(f"   {i}. Resource: {alloc.resource_id}")
        print(f"      Amount: {alloc.allocated_amount}, Priority: {alloc.priority}")
        print(f"      Duration: {alloc.start_time.strftime('%Y-%m-%d')} to {alloc.end_time.strftime('%Y-%m-%d')}")
        print(f"      Cost: ${alloc.cost:,.2f}" if alloc.cost else "      Cost: Not specified")
    
    # Demo 2: Multi-Experiment Resource Coordination
    print("\n\n🔄 Demo 2: Multi-Experiment Resource Coordination")
    print("-" * 50)
    
    # Simulate multiple experiments competing for resources
    experiments = [
        {
            'id': 'quantum_computing_exp',
            'priority': 9,
            'requirements': requirements[:2]  # High priority, needs personnel and computing
        },
        {
            'id': 'materials_research_exp',
            'priority': 6,
            'requirements': requirements[2:]  # Medium priority, needs materials and facilities
        },
        {
            'id': 'ai_optimization_exp',
            'priority': 7,
            'requirements': [requirements[1]]  # Medium-high priority, needs computing
        }
    ]
    
    print(f"🧪 Coordinating Resources for {len(experiments)} Experiments:")
    
    coordination_results = []
    for exp in experiments:
        print(f"\n   Experiment: {exp['id']} (Priority: {exp['priority']})")
        
        result = optimizer.coordinate_resource_allocation(
            experiment_id=exp['id'],
            resource_requirements=exp['requirements'],
            priority=exp['priority'],
            coordination_strategy="priority_coordination"
        )
        
        coordination_results.append(result)
        
        print(f"   ✅ Status: {result['coordination_status']}")
        print(f"   📦 Allocated Resources: {len(result['allocated_resources'])}")
        if result.get('conflicts_resolved'):
            print(f"   ⚡ Conflicts Resolved: {len(result['conflicts_resolved'])}")
    
    # Show coordination status
    print(f"\n📊 Overall Coordination Status:")
    status = optimizer.get_resource_coordination_status()
    print(f"   🔄 Coordination Enabled: {status['coordination_enabled']}")
    print(f"   🧪 Active Experiments: {status['active_experiments']}")
    print(f"   📦 Active Allocations: {status['active_allocations']}")
    print(f"   🔒 Locked Resources: {status['locked_resources']}")
    print(f"   📈 Coordination Efficiency: {status['coordination_metrics']['coordination_efficiency']:.2%}")
    
    # Demo 3: Resource Scheduling Algorithms
    print("\n\n⏰ Demo 3: Resource Scheduling Algorithms")
    print("-" * 50)
    
    # Create sample allocations for scheduling
    sample_allocations = [
        ResourceAllocation(
            allocation_id="alloc_urgent",
            resource_id="hpc_cluster_001",
            activity_id="urgent_computation",
            allocated_amount=4.0,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=6),
            priority=9  # Urgent
        ),
        ResourceAllocation(
            allocation_id="alloc_routine",
            resource_id="hpc_cluster_001",
            activity_id="routine_analysis",
            allocated_amount=2.0,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=12),
            priority=5  # Normal
        ),
        ResourceAllocation(
            allocation_id="alloc_background",
            resource_id="hpc_cluster_001",
            activity_id="background_processing",
            allocated_amount=1.0,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=24),
            priority=2  # Low
        )
    ]
    
    algorithms = ["priority_first", "earliest_deadline", "shortest_job"]
    
    for algorithm in algorithms:
        print(f"\n   📅 Scheduling Algorithm: {algorithm}")
        scheduled = optimizer.schedule_resources(
            allocations=sample_allocations.copy(),
            scheduling_algorithm=algorithm
        )
        
        print(f"   📋 Scheduled Order:")
        for i, alloc in enumerate(scheduled, 1):
            duration = alloc.end_time - alloc.start_time
            print(f"      {i}. {alloc.activity_id} (Priority: {alloc.priority}, Duration: {duration})")
    
    # Demo 4: Resource Utilization Tracking
    print("\n\n📊 Demo 4: Resource Utilization Tracking")
    print("-" * 50)
    
    # Track utilization for different resources
    resources_to_track = ["hpc_cluster_001", "personnel_pool_001", "cleanroom_001"]
    
    for resource_id in resources_to_track:
        print(f"\n   📈 Tracking Resource: {resource_id}")
        
        # Create time period for tracking
        start_time = datetime.now() - timedelta(days=7)
        end_time = datetime.now()
        
        utilization = optimizer.track_resource_utilization(
            resource_id=resource_id,
            time_period=(start_time, end_time),
            allocations=sample_allocations
        )
        
        print(f"   📊 Utilization Rate: {utilization.utilization_rate:.2%}")
        print(f"   ⚡ Peak Utilization: {utilization.peak_utilization:.2f}")
        print(f"   ⏱️  Idle Time: {utilization.idle_time.total_seconds()/3600:.1f} hours")
        print(f"   🎯 Efficiency Score: {utilization.efficiency_score:.2f}")
    
    # Demo 5: Dynamic Resource Reallocation
    print("\n\n🔄 Demo 5: Dynamic Resource Reallocation")
    print("-" * 50)
    
    print("   🚨 Simulating High-Priority Emergency Experiment...")
    
    emergency_requirements = [
        ResourceRequirement(
            resource_type="computational",
            resource_name="Emergency Computing",
            quantity_needed=8,
            duration_needed=timedelta(hours=4),
            cost_estimate=5000.0
        )
    ]
    
    emergency_result = optimizer.coordinate_resource_allocation(
        experiment_id="emergency_response_exp",
        resource_requirements=emergency_requirements,
        priority=10,  # Maximum priority
        coordination_strategy="dynamic_reallocation"
    )
    
    print(f"   ✅ Emergency Allocation Status: {emergency_result['coordination_status']}")
    print(f"   ⚡ Conflicts Resolved: {len(emergency_result.get('conflicts_resolved', []))}")
    print(f"   📦 Resources Allocated: {len(emergency_result['allocated_resources'])}")
    
    if emergency_result.get('conflicts_resolved'):
        print(f"   🔄 Preempted Lower Priority Experiments:")
        for conflict in emergency_result['conflicts_resolved']:
            print(f"      • {conflict.get('preempted_experiment', 'Unknown')}")
    
    # Final status
    print(f"\n\n📊 Final System Status:")
    final_status = optimizer.get_resource_coordination_status()
    print(f"   🧪 Total Active Experiments: {final_status['active_experiments']}")
    print(f"   📦 Total Active Allocations: {final_status['active_allocations']}")
    print(f"   🔒 Total Locked Resources: {final_status['locked_resources']}")
    print(f"   📈 System Efficiency: {final_status['coordination_metrics']['coordination_efficiency']:.2%}")
    print(f"   ⚡ Conflict Resolution Rate: {final_status['coordination_metrics']['conflict_resolution_rate']:.2%}")
    
    print(f"\n🎉 Resource Allocation Optimizer Demo Complete!")
    print("=" * 60)
    print("✅ Successfully demonstrated:")
    print("   • Optimal resource allocation for experimental activities")
    print("   • Multi-experiment resource coordination")
    print("   • Advanced scheduling algorithms")
    print("   • Real-time utilization tracking")
    print("   • Dynamic resource reallocation")
    print("   • Conflict detection and resolution")

if __name__ == "__main__":
    main()