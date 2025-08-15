"""
Innovation Pipeline Optimization Demo

This script demonstrates the capabilities of the innovation pipeline optimization system,
including flow optimization, resource allocation, and performance monitoring.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List

from scrollintel.engines.innovation_pipeline_optimizer import InnovationPipelineOptimizer
from scrollintel.models.innovation_pipeline_models import (
    InnovationPipelineItem, PipelineStage, InnovationPriority, ResourceType,
    ResourceRequirement, PipelineOptimizationConfig
)


async def create_sample_innovations() -> List[InnovationPipelineItem]:
    """Create sample innovations for demonstration"""
    innovations = []
    
    # Innovation 1: AI-powered drug discovery
    innovations.append(InnovationPipelineItem(
        innovation_id="ai-drug-discovery-001",
        current_stage=PipelineStage.RESEARCH,
        priority=InnovationPriority.CRITICAL,
        success_probability=0.85,
        risk_score=0.3,
        impact_score=0.95,
        resource_requirements=[
            ResourceRequirement(
                resource_type=ResourceType.COMPUTE,
                amount=500.0,
                unit="GPU-hours",
                duration=168.0,  # 1 week
                priority=InnovationPriority.CRITICAL
            ),
            ResourceRequirement(
                resource_type=ResourceType.RESEARCH_TIME,
                amount=200.0,
                unit="hours",
                duration=336.0,  # 2 weeks
                priority=InnovationPriority.HIGH
            ),
            ResourceRequirement(
                resource_type=ResourceType.BUDGET,
                amount=50000.0,
                unit="USD",
                duration=720.0,  # 1 month
                priority=InnovationPriority.CRITICAL
            )
        ],
        dependencies=[],
        metadata={
            "domain": "healthcare",
            "technology": "machine_learning",
            "market_size": "10B",
            "patent_potential": "high"
        }
    ))
    
    # Innovation 2: Quantum computing optimization
    innovations.append(InnovationPipelineItem(
        innovation_id="quantum-optimization-002",
        current_stage=PipelineStage.EXPERIMENTATION,
        priority=InnovationPriority.HIGH,
        success_probability=0.7,
        risk_score=0.5,
        impact_score=0.9,
        resource_requirements=[
            ResourceRequirement(
                resource_type=ResourceType.COMPUTE,
                amount=200.0,
                unit="quantum-hours",
                duration=72.0,  # 3 days
                priority=InnovationPriority.HIGH
            ),
            ResourceRequirement(
                resource_type=ResourceType.RESEARCH_TIME,
                amount=120.0,
                unit="hours",
                duration=168.0,  # 1 week
                priority=InnovationPriority.HIGH
            ),
            ResourceRequirement(
                resource_type=ResourceType.BUDGET,
                amount=30000.0,
                unit="USD",
                duration=480.0,  # 20 days
                priority=InnovationPriority.HIGH
            )
        ],
        dependencies=[],
        metadata={
            "domain": "computing",
            "technology": "quantum",
            "market_size": "5B",
            "patent_potential": "very_high"
        }
    ))
    
    # Innovation 3: Sustainable energy storage
    innovations.append(InnovationPipelineItem(
        innovation_id="energy-storage-003",
        current_stage=PipelineStage.PROTOTYPING,
        priority=InnovationPriority.HIGH,
        success_probability=0.8,
        risk_score=0.4,
        impact_score=0.88,
        resource_requirements=[
            ResourceRequirement(
                resource_type=ResourceType.DEVELOPMENT_TIME,
                amount=300.0,
                unit="hours",
                duration=240.0,  # 10 days
                priority=InnovationPriority.HIGH
            ),
            ResourceRequirement(
                resource_type=ResourceType.TESTING_TIME,
                amount=100.0,
                unit="hours",
                duration=120.0,  # 5 days
                priority=InnovationPriority.MEDIUM
            ),
            ResourceRequirement(
                resource_type=ResourceType.BUDGET,
                amount=75000.0,
                unit="USD",
                duration=600.0,  # 25 days
                priority=InnovationPriority.HIGH
            )
        ],
        dependencies=[],
        metadata={
            "domain": "energy",
            "technology": "materials_science",
            "market_size": "15B",
            "patent_potential": "high"
        }
    ))
    
    # Innovation 4: Autonomous vehicle navigation
    innovations.append(InnovationPipelineItem(
        innovation_id="autonomous-nav-004",
        current_stage=PipelineStage.VALIDATION,
        priority=InnovationPriority.MEDIUM,
        success_probability=0.75,
        risk_score=0.6,
        impact_score=0.82,
        resource_requirements=[
            ResourceRequirement(
                resource_type=ResourceType.COMPUTE,
                amount=300.0,
                unit="GPU-hours",
                duration=96.0,  # 4 days
                priority=InnovationPriority.MEDIUM
            ),
            ResourceRequirement(
                resource_type=ResourceType.TESTING_TIME,
                amount=200.0,
                unit="hours",
                duration=192.0,  # 8 days
                priority=InnovationPriority.HIGH
            ),
            ResourceRequirement(
                resource_type=ResourceType.BUDGET,
                amount=40000.0,
                unit="USD",
                duration=384.0,  # 16 days
                priority=InnovationPriority.MEDIUM
            )
        ],
        dependencies=[],
        metadata={
            "domain": "transportation",
            "technology": "ai_robotics",
            "market_size": "8B",
            "patent_potential": "medium"
        }
    ))
    
    # Innovation 5: Blockchain supply chain
    innovations.append(InnovationPipelineItem(
        innovation_id="blockchain-supply-005",
        current_stage=PipelineStage.IDEATION,
        priority=InnovationPriority.MEDIUM,
        success_probability=0.65,
        risk_score=0.45,
        impact_score=0.7,
        resource_requirements=[
            ResourceRequirement(
                resource_type=ResourceType.RESEARCH_TIME,
                amount=80.0,
                unit="hours",
                duration=120.0,  # 5 days
                priority=InnovationPriority.MEDIUM
            ),
            ResourceRequirement(
                resource_type=ResourceType.DEVELOPMENT_TIME,
                amount=150.0,
                unit="hours",
                duration=240.0,  # 10 days
                priority=InnovationPriority.MEDIUM
            ),
            ResourceRequirement(
                resource_type=ResourceType.BUDGET,
                amount=25000.0,
                unit="USD",
                duration=360.0,  # 15 days
                priority=InnovationPriority.MEDIUM
            )
        ],
        dependencies=[],
        metadata={
            "domain": "logistics",
            "technology": "blockchain",
            "market_size": "3B",
            "patent_potential": "medium"
        }
    ))
    
    # Innovation 6: Neural interface technology
    innovations.append(InnovationPipelineItem(
        innovation_id="neural-interface-006",
        current_stage=PipelineStage.RESEARCH,
        priority=InnovationPriority.LOW,
        success_probability=0.6,
        risk_score=0.7,
        impact_score=0.92,
        resource_requirements=[
            ResourceRequirement(
                resource_type=ResourceType.RESEARCH_TIME,
                amount=400.0,
                unit="hours",
                duration=480.0,  # 20 days
                priority=InnovationPriority.LOW
            ),
            ResourceRequirement(
                resource_type=ResourceType.BUDGET,
                amount=100000.0,
                unit="USD",
                duration=1440.0,  # 60 days
                priority=InnovationPriority.LOW
            )
        ],
        dependencies=[],
        metadata={
            "domain": "neurotechnology",
            "technology": "bioengineering",
            "market_size": "20B",
            "patent_potential": "very_high"
        }
    ))
    
    return innovations


async def demonstrate_pipeline_optimization():
    """Demonstrate innovation pipeline optimization capabilities"""
    print("🚀 Innovation Pipeline Optimization Demo")
    print("=" * 50)
    
    # Create optimizer with custom configuration
    config = PipelineOptimizationConfig(
        max_concurrent_innovations=20,
        resource_buffer_percentage=0.15,
        bottleneck_threshold=0.75,
        resource_utilization_target=0.9,
        rebalance_frequency_minutes=15
    )
    
    optimizer = InnovationPipelineOptimizer(config)
    
    # Create and add sample innovations
    print("\n📋 Creating sample innovations...")
    innovations = await create_sample_innovations()
    
    for innovation in innovations:
        success = await optimizer.add_innovation_to_pipeline(innovation)
        if success:
            print(f"✅ Added {innovation.innovation_id} to {innovation.current_stage.value} stage")
        else:
            print(f"❌ Failed to add {innovation.innovation_id}")
    
    print(f"\n📊 Pipeline Status: {len(optimizer.pipeline_items)} innovations active")
    
    # Demonstrate bottleneck identification
    print("\n🔍 Identifying pipeline bottlenecks...")
    bottlenecks = await optimizer._identify_bottlenecks()
    
    if bottlenecks:
        print("⚠️  Bottlenecks detected:")
        for stage, severity in bottlenecks.items():
            print(f"   - {stage.value}: {severity:.2%} utilization")
    else:
        print("✅ No significant bottlenecks detected")
    
    # Demonstrate resource utilization analysis
    print("\n💾 Analyzing resource utilization...")
    utilization = await optimizer._calculate_resource_utilization()
    
    print("Resource Utilization:")
    for resource_type, util in utilization.items():
        status = "🔴" if util > 0.9 else "🟡" if util > 0.7 else "🟢"
        print(f"   {status} {resource_type.value}: {util:.2%}")
    
    # Demonstrate innovation prioritization
    print("\n🎯 Prioritizing innovations...")
    criteria = {
        'impact': 0.35,
        'success_probability': 0.25,
        'risk': 0.2,
        'resource_efficiency': 0.15,
        'time_sensitivity': 0.05
    }
    
    priorities = await optimizer.prioritize_innovations(criteria)
    
    print("Innovation Priorities:")
    for innovation_id, priority in sorted(priorities.items(), 
                                        key=lambda x: optimizer._get_priority_weight(x[1]), 
                                        reverse=True):
        innovation = optimizer.pipeline_items[innovation_id]
        print(f"   {priority.value.upper()}: {innovation.innovation_id} "
              f"(Impact: {innovation.impact_score:.2f}, Success: {innovation.success_probability:.2f})")
    
    # Demonstrate resource allocation
    print("\n💰 Optimizing resource allocation...")
    allocations = await optimizer.allocate_pipeline_resources("balanced")
    
    total_allocated = {}
    for innovation_id, innovation_allocations in allocations.items():
        innovation = optimizer.pipeline_items[innovation_id]
        print(f"\n   📦 {innovation.innovation_id}:")
        
        for allocation in innovation_allocations:
            resource_type = allocation.resource_type
            amount = allocation.allocated_amount
            
            if resource_type not in total_allocated:
                total_allocated[resource_type] = 0
            total_allocated[resource_type] += amount
            
            print(f"      - {resource_type.value}: {amount:.1f} units")
    
    print(f"\n📈 Total Resource Allocation:")
    for resource_type, total in total_allocated.items():
        print(f"   - {resource_type.value}: {total:.1f} units")
    
    # Demonstrate pipeline optimization
    print("\n⚡ Running pipeline optimization...")
    optimization_result = await optimizer.optimize_pipeline_flow()
    
    print(f"Optimization Results:")
    print(f"   🎯 Optimization Score: {optimization_result.optimization_score:.3f}")
    print(f"   📊 Confidence Level: {optimization_result.confidence_level:.3f}")
    print(f"   📈 Expected Throughput Improvement: {optimization_result.expected_throughput_improvement:.2%}")
    print(f"   ⏱️  Expected Cycle Time Reduction: {optimization_result.expected_cycle_time_reduction:.2%}")
    print(f"   💰 Expected Resource Savings: {optimization_result.expected_resource_savings:.2%}")
    
    if optimization_result.recommendations:
        print(f"\n💡 Recommendations:")
        for i, recommendation in enumerate(optimization_result.recommendations, 1):
            print(f"   {i}. {recommendation}")
    
    if optimization_result.warnings:
        print(f"\n⚠️  Warnings:")
        for i, warning in enumerate(optimization_result.warnings, 1):
            print(f"   {i}. {warning}")
    
    # Demonstrate performance monitoring
    print("\n📊 Generating performance report...")
    performance_report = await optimizer.monitor_pipeline_performance()
    
    print(f"Performance Report:")
    print(f"   📋 Total Innovations: {performance_report.total_innovations}")
    print(f"   🟢 Active: {performance_report.active_innovations}")
    print(f"   ✅ Completed: {performance_report.completed_innovations}")
    print(f"   ❌ Failed: {performance_report.failed_innovations}")
    print(f"   📈 Overall Throughput: {performance_report.overall_throughput:.2f} innovations/day")
    print(f"   ⏱️  Average Cycle Time: {performance_report.average_cycle_time:.1f} hours")
    print(f"   🎯 Success Rate: {performance_report.overall_success_rate:.2%}")
    print(f"   💰 Cost per Innovation: ${performance_report.cost_per_innovation:,.2f}")
    
    # Show stage-wise metrics
    if performance_report.stage_metrics:
        print(f"\n📊 Stage-wise Metrics:")
        for stage, metrics in performance_report.stage_metrics.items():
            print(f"   {stage.value.title()}:")
            print(f"      - Throughput: {metrics.throughput:.2f}/hour")
            print(f"      - Cycle Time: {metrics.cycle_time:.1f} hours")
            print(f"      - Success Rate: {metrics.success_rate:.2%}")
            print(f"      - Quality Score: {metrics.quality_score:.2f}")
    
    # Show identified bottlenecks
    if performance_report.identified_bottlenecks:
        print(f"\n🚧 Identified Bottlenecks:")
        for stage in performance_report.identified_bottlenecks:
            severity = performance_report.bottleneck_severity.get(stage, 0.0)
            print(f"   - {stage.value}: {severity:.2%} severity")
    
    # Show recommendations
    if performance_report.optimization_recommendations:
        print(f"\n🎯 Optimization Recommendations:")
        for i, rec in enumerate(performance_report.optimization_recommendations, 1):
            print(f"   {i}. {rec}")
    
    if performance_report.capacity_recommendations:
        print(f"\n📈 Capacity Recommendations:")
        for i, rec in enumerate(performance_report.capacity_recommendations, 1):
            print(f"   {i}. {rec}")
    
    if performance_report.process_improvements:
        print(f"\n⚙️  Process Improvements:")
        for i, improvement in enumerate(performance_report.process_improvements, 1):
            print(f"   {i}. {improvement}")
    
    # Demonstrate continuous optimization
    print(f"\n🔄 Demonstrating continuous optimization...")
    
    # Simulate some time passing and innovations progressing
    for innovation in optimizer.pipeline_items.values():
        if innovation.current_stage == PipelineStage.IDEATION:
            innovation.stage_entered_at = datetime.utcnow() - timedelta(hours=3)
    
    # Run another optimization cycle
    print("   Running second optimization cycle...")
    optimization_result_2 = await optimizer.optimize_pipeline_flow()
    
    print(f"   Second Optimization Score: {optimization_result_2.optimization_score:.3f}")
    
    # Show optimization history
    print(f"\n📈 Optimization History:")
    for i, opt in enumerate(optimizer.optimization_history, 1):
        print(f"   {i}. Score: {opt.optimization_score:.3f}, "
              f"Confidence: {opt.confidence_level:.3f}, "
              f"Time: {opt.timestamp.strftime('%H:%M:%S')}")
    
    print(f"\n✨ Demo completed successfully!")
    print(f"   - {len(innovations)} innovations processed")
    print(f"   - {len(optimizer.optimization_history)} optimization cycles completed")
    print(f"   - {len(optimizer.historical_metrics)} performance reports generated")


async def demonstrate_advanced_scenarios():
    """Demonstrate advanced pipeline optimization scenarios"""
    print("\n🎯 Advanced Pipeline Optimization Scenarios")
    print("=" * 50)
    
    optimizer = InnovationPipelineOptimizer()
    
    # Scenario 1: High-load stress test
    print("\n🔥 Scenario 1: High-load stress test")
    print("Creating 50 innovations to test capacity limits...")
    
    stress_innovations = []
    for i in range(50):
        innovation = InnovationPipelineItem(
            innovation_id=f"stress-test-{i:03d}",
            current_stage=PipelineStage.IDEATION,
            priority=InnovationPriority.MEDIUM,
            success_probability=0.6 + (i % 10) * 0.04,
            risk_score=0.5 - (i % 8) * 0.05,
            impact_score=0.7 + (i % 6) * 0.05,
            resource_requirements=[
                ResourceRequirement(
                    resource_type=ResourceType.COMPUTE,
                    amount=10.0 + i,
                    unit="cores",
                    duration=24.0
                )
            ]
        )
        stress_innovations.append(innovation)
    
    added_count = 0
    for innovation in stress_innovations:
        if await optimizer.add_innovation_to_pipeline(innovation):
            added_count += 1
    
    print(f"   ✅ Successfully added {added_count}/{len(stress_innovations)} innovations")
    
    # Run optimization under high load
    stress_optimization = await optimizer.optimize_pipeline_flow()
    print(f"   📊 High-load optimization score: {stress_optimization.optimization_score:.3f}")
    
    # Scenario 2: Resource scarcity simulation
    print("\n💰 Scenario 2: Resource scarcity simulation")
    
    # Reduce available resources
    original_resources = optimizer.resource_pool.copy()
    for resource_type in optimizer.resource_pool:
        optimizer.resource_pool[resource_type] *= 0.3  # Reduce to 30%
    
    print("   Reduced available resources to 30% of original capacity")
    
    scarcity_optimization = await optimizer.optimize_pipeline_flow()
    print(f"   📊 Resource-constrained optimization score: {scarcity_optimization.optimization_score:.3f}")
    
    if scarcity_optimization.warnings:
        print("   ⚠️  Resource scarcity warnings:")
        for warning in scarcity_optimization.warnings:
            print(f"      - {warning}")
    
    # Restore resources
    optimizer.resource_pool = original_resources
    
    # Scenario 3: Priority rebalancing
    print("\n⚖️  Scenario 3: Dynamic priority rebalancing")
    
    # Change priorities based on market conditions
    market_shift_criteria = {
        'impact': 0.5,  # Increased focus on impact
        'success_probability': 0.2,  # Reduced focus on certainty
        'risk': 0.1,  # Reduced risk aversion
        'resource_efficiency': 0.2
    }
    
    new_priorities = await optimizer.prioritize_innovations(market_shift_criteria)
    
    priority_changes = 0
    for innovation_id, new_priority in new_priorities.items():
        if innovation_id in optimizer.pipeline_items:
            old_priority = optimizer.pipeline_items[innovation_id].priority
            if old_priority != new_priority:
                optimizer.pipeline_items[innovation_id].priority = new_priority
                priority_changes += 1
    
    print(f"   📈 Rebalanced priorities for {priority_changes} innovations")
    
    rebalance_optimization = await optimizer.optimize_pipeline_flow()
    print(f"   📊 Post-rebalancing optimization score: {rebalance_optimization.optimization_score:.3f}")
    
    print("\n✨ Advanced scenarios completed!")


async def main():
    """Main demo function"""
    try:
        await demonstrate_pipeline_optimization()
        await demonstrate_advanced_scenarios()
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())