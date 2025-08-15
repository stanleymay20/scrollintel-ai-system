"""
Innovation Acceleration System Demo

This script demonstrates the capabilities of the innovation acceleration system,
including bottleneck identification, timeline optimization, and acceleration strategies.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List

from scrollintel.engines.innovation_acceleration_system import (
    InnovationAccelerationSystem, AccelerationType, BottleneckType
)
from scrollintel.models.innovation_pipeline_models import (
    InnovationPipelineItem, PipelineStage, InnovationPriority, ResourceType,
    ResourceRequirement, ResourceAllocation, PipelineStatus
)


async def create_sample_pipeline() -> dict:
    """Create sample pipeline items for demonstration"""
    pipeline_items = {}
    
    # Critical innovation with resource constraints
    pipeline_items["ai-breakthrough-001"] = InnovationPipelineItem(
        innovation_id="ai-breakthrough-001",
        current_stage=PipelineStage.RESEARCH,
        priority=InnovationPriority.CRITICAL,
        success_probability=0.85,
        risk_score=0.25,
        impact_score=0.95,
        resource_requirements=[
            ResourceRequirement(
                resource_type=ResourceType.COMPUTE,
                amount=800.0,
                unit="GPU-hours",
                duration=168.0,  # 1 week
                priority=InnovationPriority.CRITICAL
            ),
            ResourceRequirement(
                resource_type=ResourceType.RESEARCH_TIME,
                amount=300.0,
                unit="hours",
                duration=336.0,  # 2 weeks
                priority=InnovationPriority.HIGH
            )
        ],
        resource_allocations=[
            ResourceAllocation(
                innovation_id="ai-breakthrough-001",
                resource_type=ResourceType.COMPUTE,
                allocated_amount=800.0,
                used_amount=750.0  # High utilization
            ),
            ResourceAllocation(
                innovation_id="ai-breakthrough-001",
                resource_type=ResourceType.RESEARCH_TIME,
                allocated_amount=300.0,
                used_amount=280.0
            )
        ],
        metadata={
            "domain": "artificial_intelligence",
            "complexity": "very_high",
            "market_potential": "10B"
        }
    )
    
    # High priority innovation with quality issues
    pipeline_items["quantum-computing-002"] = InnovationPipelineItem(
        innovation_id="quantum-computing-002",
        current_stage=PipelineStage.VALIDATION,
        priority=InnovationPriority.HIGH,
        success_probability=0.45,  # Quality bottleneck
        risk_score=0.65,
        impact_score=0.92,
        resource_requirements=[
            ResourceRequirement(
                resource_type=ResourceType.TESTING_TIME,
                amount=200.0,
                unit="hours",
                duration=120.0,
                priority=InnovationPriority.HIGH
            )
        ],
        metadata={
            "domain": "quantum_computing",
            "complexity": "extreme",
            "market_potential": "15B"
        }
    )
    
    # Innovation with dependencies
    pipeline_items["biotech-innovation-003"] = InnovationPipelineItem(
        innovation_id="biotech-innovation-003",
        current_stage=PipelineStage.PROTOTYPING,
        priority=InnovationPriority.HIGH,
        success_probability=0.75,
        risk_score=0.35,
        impact_score=0.88,
        dependencies=["ai-breakthrough-001", "missing-dependency"],  # One exists, one doesn't
        resource_requirements=[
            ResourceRequirement(
                resource_type=ResourceType.DEVELOPMENT_TIME,
                amount=400.0,
                unit="hours",
                duration=240.0,
                priority=InnovationPriority.HIGH
            )
        ],
        metadata={
            "domain": "biotechnology",
            "complexity": "high",
            "market_potential": "8B"
        }
    )
    
    # Medium priority innovation in experimentation
    pipeline_items["clean-energy-004"] = InnovationPipelineItem(
        innovation_id="clean-energy-004",
        current_stage=PipelineStage.EXPERIMENTATION,
        priority=InnovationPriority.MEDIUM,
        success_probability=0.70,
        risk_score=0.40,
        impact_score=0.80,
        resource_requirements=[
            ResourceRequirement(
                resource_type=ResourceType.TESTING_TIME,
                amount=150.0,
                unit="hours",
                duration=96.0,
                priority=InnovationPriority.MEDIUM
            )
        ],
        metadata={
            "domain": "clean_energy",
            "complexity": "medium",
            "market_potential": "5B"
        }
    )
    
    # Low priority innovation with high impact
    pipeline_items["space-tech-005"] = InnovationPipelineItem(
        innovation_id="space-tech-005",
        current_stage=PipelineStage.IDEATION,
        priority=InnovationPriority.LOW,
        success_probability=0.60,
        risk_score=0.70,
        impact_score=0.85,
        resource_requirements=[
            ResourceRequirement(
                resource_type=ResourceType.RESEARCH_TIME,
                amount=100.0,
                unit="hours",
                duration=168.0,
                priority=InnovationPriority.LOW
            )
        ],
        metadata={
            "domain": "space_technology",
            "complexity": "very_high",
            "market_potential": "12B"
        }
    )
    
    return pipeline_items


async def demonstrate_acceleration_system():
    """Demonstrate innovation acceleration system capabilities"""
    print("🚀 Innovation Acceleration System Demo")
    print("=" * 50)
    
    # Create acceleration system
    acceleration_system = InnovationAccelerationSystem()
    
    # Create sample pipeline
    print("\n📋 Creating sample innovation pipeline...")
    pipeline_items = await create_sample_pipeline()
    
    for innovation_id, item in pipeline_items.items():
        print(f"✅ {innovation_id}: {item.current_stage.value} stage, "
              f"{item.priority.value} priority, {item.success_probability:.2%} success rate")
    
    print(f"\n📊 Pipeline Status: {len(pipeline_items)} innovations active")
    
    # Demonstrate bottleneck identification
    print("\n🔍 Identifying pipeline bottlenecks...")
    bottlenecks = await acceleration_system.identify_bottlenecks(pipeline_items)
    
    if bottlenecks:
        print(f"⚠️  Found {len(bottlenecks)} bottlenecks:")
        for bottleneck in bottlenecks:
            severity_emoji = "🔴" if bottleneck.severity > 0.7 else "🟡" if bottleneck.severity > 0.4 else "🟢"
            print(f"   {severity_emoji} {bottleneck.bottleneck_type.value.replace('_', ' ').title()}")
            print(f"      Innovation: {bottleneck.innovation_id}")
            print(f"      Stage: {bottleneck.affected_stage.value}")
            print(f"      Severity: {bottleneck.severity:.2%}")
            print(f"      Estimated Delay: {bottleneck.estimated_delay:.1f} hours")
            print(f"      Root Causes: {', '.join(bottleneck.root_causes)}")
            print(f"      Resolution Strategies: {len(bottleneck.resolution_strategies)} available")
            print()
    else:
        print("✅ No significant bottlenecks detected")
    
    # Demonstrate timeline optimization for specific innovation
    print("⏱️  Optimizing timeline for critical innovation...")
    critical_innovation = pipeline_items["ai-breakthrough-001"]
    timeline_optimization = await acceleration_system.optimize_innovation_timeline(critical_innovation)
    
    print(f"Timeline Optimization Results:")
    print(f"   📊 Original Timeline: {timeline_optimization.original_timeline:.1f} hours")
    print(f"   ⚡ Optimized Timeline: {timeline_optimization.optimized_timeline:.1f} hours")
    print(f"   💰 Time Savings: {timeline_optimization.time_savings:.1f} hours "
          f"({timeline_optimization.time_savings/timeline_optimization.original_timeline*100:.1f}%)")
    print(f"   🎯 Confidence Level: {timeline_optimization.confidence_level:.2%}")
    print(f"   ⚠️  Risk Assessment: {timeline_optimization.risk_assessment:.2%}")
    print(f"   🔧 Acceleration Strategies: {len(timeline_optimization.acceleration_strategies)} available")
    
    if timeline_optimization.acceleration_strategies:
        print(f"\n   Recommended Strategies:")
        for i, strategy in enumerate(timeline_optimization.acceleration_strategies[:3], 1):
            print(f"      {i}. {strategy.acceleration_type.value.replace('_', ' ').title()}")
            print(f"         - Time Reduction: {strategy.expected_time_reduction:.1f} hours")
            print(f"         - Success Probability: {strategy.success_probability:.2%}")
            print(f"         - Risk Factor: {strategy.risk_factor:.2%}")
            print(f"         - Resource Cost: {strategy.resource_cost:.1f}x")
    
    # Demonstrate complete acceleration workflow
    print(f"\n⚡ Running complete acceleration workflow...")
    acceleration_result = await acceleration_system.accelerate_innovation_development(pipeline_items)
    
    print(f"Acceleration Results:")
    print(f"   🎯 Acceleration ID: {acceleration_result.acceleration_id}")
    print(f"   📈 Innovations Accelerated: {acceleration_result.innovations_accelerated}")
    print(f"   ⏰ Total Time Saved: {acceleration_result.total_time_saved:.1f} hours")
    print(f"   🚧 Bottlenecks Resolved: {acceleration_result.bottlenecks_resolved}")
    print(f"   📊 Performance Improvement: {acceleration_result.performance_improvement:.2%}")
    print(f"   💰 Cost Efficiency: {acceleration_result.cost_efficiency:.2f}")
    print(f"   🔧 Strategies Applied: {len(acceleration_result.acceleration_strategies_applied)}")
    print(f"   📋 Timeline Optimizations: {len(acceleration_result.timeline_optimizations)}")
    
    # Show applied strategies
    if acceleration_result.acceleration_strategies_applied:
        print(f"\n🔧 Applied Acceleration Strategies:")
        for strategy in acceleration_result.acceleration_strategies_applied:
            print(f"   • {strategy.acceleration_type.value.replace('_', ' ').title()}")
            print(f"     Innovation: {strategy.innovation_id}")
            print(f"     Target Stage: {strategy.target_stage.value}")
            print(f"     Expected Reduction: {strategy.expected_time_reduction:.1f} hours")
            print(f"     Success Probability: {strategy.success_probability:.2%}")
    
    # Show timeline optimizations
    if acceleration_result.timeline_optimizations:
        print(f"\n📋 Timeline Optimizations:")
        for optimization in acceleration_result.timeline_optimizations:
            savings_pct = (optimization.time_savings / optimization.original_timeline * 100) if optimization.original_timeline > 0 else 0
            print(f"   • {optimization.innovation_id}")
            print(f"     Time Savings: {optimization.time_savings:.1f} hours ({savings_pct:.1f}%)")
            print(f"     Confidence: {optimization.confidence_level:.2%}")
            print(f"     Risk: {optimization.risk_assessment:.2%}")
    
    # Demonstrate acceleration metrics
    print(f"\n📊 Acceleration System Metrics:")
    metrics = acceleration_system.get_acceleration_metrics()
    
    if metrics:
        print(f"   📈 Average Time Saved: {metrics.get('average_time_saved', 0):.1f} hours")
        print(f"   🎯 Average Innovations Accelerated: {metrics.get('average_innovations_accelerated', 0):.1f}")
        print(f"   📊 Average Performance Improvement: {metrics.get('average_performance_improvement', 0):.2%}")
        print(f"   💰 Average Cost Efficiency: {metrics.get('average_cost_efficiency', 0):.2f}")
        print(f"   🚧 Bottleneck Resolution Rate: {metrics.get('bottleneck_resolution_rate', 0):.1f}")
    else:
        print("   📊 Building metrics history...")
    
    # Demonstrate bottleneck patterns
    print(f"\n🔍 Bottleneck Pattern Analysis:")
    patterns = acceleration_system.get_bottleneck_patterns()
    
    if patterns:
        total_bottlenecks = sum(patterns.values())
        print(f"   Total Bottlenecks Analyzed: {total_bottlenecks}")
        
        # Sort by frequency
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        
        for bottleneck_type, count in sorted_patterns:
            percentage = (count / total_bottlenecks * 100) if total_bottlenecks > 0 else 0
            print(f"   • {bottleneck_type.value.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    else:
        print("   📊 No bottleneck patterns yet - building history...")
    
    print(f"\n✨ Demo completed successfully!")
    print(f"   - {len(pipeline_items)} innovations processed")
    print(f"   - {len(bottlenecks)} bottlenecks identified")
    print(f"   - {acceleration_result.innovations_accelerated} innovations accelerated")
    print(f"   - {acceleration_result.total_time_saved:.1f} hours saved")


async def demonstrate_advanced_scenarios():
    """Demonstrate advanced acceleration scenarios"""
    print("\n🎯 Advanced Acceleration Scenarios")
    print("=" * 50)
    
    acceleration_system = InnovationAccelerationSystem()
    
    # Scenario 1: High-bottleneck environment
    print("\n🚧 Scenario 1: High-bottleneck environment")
    print("Creating pipeline with multiple bottlenecks...")
    
    bottleneck_pipeline = {}
    
    # Create multiple innovations in same stage (capacity bottleneck)
    for i in range(15):
        bottleneck_pipeline[f"research-overload-{i:03d}"] = InnovationPipelineItem(
            innovation_id=f"research-overload-{i:03d}",
            current_stage=PipelineStage.RESEARCH,
            priority=InnovationPriority.HIGH,
            success_probability=0.8,
            status=PipelineStatus.ACTIVE
        )
    
    # Add quality bottleneck innovation
    bottleneck_pipeline["quality-issue"] = InnovationPipelineItem(
        innovation_id="quality-issue",
        current_stage=PipelineStage.VALIDATION,
        priority=InnovationPriority.CRITICAL,
        success_probability=0.3,  # Very low success probability
        risk_score=0.8,
        impact_score=0.9,
        status=PipelineStatus.ACTIVE
    )
    
    # Add dependency bottleneck
    bottleneck_pipeline["dependent-innovation"] = InnovationPipelineItem(
        innovation_id="dependent-innovation",
        current_stage=PipelineStage.PROTOTYPING,
        priority=InnovationPriority.HIGH,
        success_probability=0.8,
        dependencies=["quality-issue", "missing-dependency"],
        status=PipelineStatus.ACTIVE
    )
    
    bottlenecks = await acceleration_system.identify_bottlenecks(bottleneck_pipeline)
    print(f"   🔍 Identified {len(bottlenecks)} bottlenecks")
    
    # Count bottleneck types
    bottleneck_counts = {}
    for bottleneck in bottlenecks:
        bt = bottleneck.bottleneck_type
        bottleneck_counts[bt] = bottleneck_counts.get(bt, 0) + 1
    
    for bottleneck_type, count in bottleneck_counts.items():
        print(f"      - {bottleneck_type.value.replace('_', ' ').title()}: {count}")
    
    # Run acceleration on bottlenecked pipeline
    bottleneck_result = await acceleration_system.accelerate_innovation_development(bottleneck_pipeline)
    print(f"   ⚡ Acceleration Results:")
    print(f"      - Bottlenecks Resolved: {bottleneck_result.bottlenecks_resolved}")
    print(f"      - Time Saved: {bottleneck_result.total_time_saved:.1f} hours")
    print(f"      - Performance Improvement: {bottleneck_result.performance_improvement:.2%}")
    
    # Scenario 2: Priority-based acceleration
    print("\n🎯 Scenario 2: Priority-based acceleration")
    print("Testing acceleration with different priority levels...")
    
    priority_pipeline = {
        "critical-mission": InnovationPipelineItem(
            innovation_id="critical-mission",
            current_stage=PipelineStage.EXPERIMENTATION,
            priority=InnovationPriority.CRITICAL,
            success_probability=0.9,
            impact_score=0.95,
            status=PipelineStatus.ACTIVE
        ),
        "high-value": InnovationPipelineItem(
            innovation_id="high-value",
            current_stage=PipelineStage.PROTOTYPING,
            priority=InnovationPriority.HIGH,
            success_probability=0.8,
            impact_score=0.85,
            status=PipelineStatus.ACTIVE
        ),
        "medium-impact": InnovationPipelineItem(
            innovation_id="medium-impact",
            current_stage=PipelineStage.VALIDATION,
            priority=InnovationPriority.MEDIUM,
            success_probability=0.7,
            impact_score=0.7,
            status=PipelineStatus.ACTIVE
        ),
        "low-priority": InnovationPipelineItem(
            innovation_id="low-priority",
            current_stage=PipelineStage.IDEATION,
            priority=InnovationPriority.LOW,
            success_probability=0.6,
            impact_score=0.5,
            status=PipelineStatus.ACTIVE
        )
    }
    
    priority_result = await acceleration_system.accelerate_innovation_development(priority_pipeline)
    
    print(f"   📊 Priority-based Results:")
    print(f"      - Innovations Accelerated: {priority_result.innovations_accelerated}")
    print(f"      - Strategies Applied: {len(priority_result.acceleration_strategies_applied)}")
    
    # Show which innovations got accelerated
    accelerated_innovations = set(
        strategy.innovation_id for strategy in priority_result.acceleration_strategies_applied
    )
    
    print(f"   🎯 Accelerated Innovations:")
    for innovation_id in accelerated_innovations:
        innovation = priority_pipeline[innovation_id]
        print(f"      - {innovation_id}: {innovation.priority.value} priority")
    
    # Scenario 3: Stage-specific acceleration
    print("\n🔄 Scenario 3: Stage-specific acceleration patterns")
    print("Analyzing acceleration effectiveness by stage...")
    
    stage_pipeline = {}
    stages = [PipelineStage.RESEARCH, PipelineStage.EXPERIMENTATION, 
              PipelineStage.PROTOTYPING, PipelineStage.VALIDATION]
    
    for i, stage in enumerate(stages):
        for j in range(3):  # 3 innovations per stage
            innovation_id = f"{stage.value}-innovation-{j:02d}"
            stage_pipeline[innovation_id] = InnovationPipelineItem(
                innovation_id=innovation_id,
                current_stage=stage,
                priority=InnovationPriority.HIGH,
                success_probability=0.7 + j * 0.1,
                impact_score=0.8,
                status=PipelineStatus.ACTIVE
            )
    
    stage_result = await acceleration_system.accelerate_innovation_development(stage_pipeline)
    
    # Analyze strategies by stage
    stage_strategies = {}
    for strategy in stage_result.acceleration_strategies_applied:
        stage = strategy.target_stage
        if stage not in stage_strategies:
            stage_strategies[stage] = []
        stage_strategies[stage].append(strategy)
    
    print(f"   📋 Stage-specific Acceleration:")
    for stage, strategies in stage_strategies.items():
        avg_time_reduction = sum(s.expected_time_reduction for s in strategies) / len(strategies)
        avg_success_prob = sum(s.success_probability for s in strategies) / len(strategies)
        
        print(f"      - {stage.value.title()}: {len(strategies)} strategies")
        print(f"        Avg Time Reduction: {avg_time_reduction:.1f} hours")
        print(f"        Avg Success Probability: {avg_success_prob:.2%}")
    
    print(f"\n✨ Advanced scenarios completed!")
    print(f"   - Multiple bottleneck types identified and resolved")
    print(f"   - Priority-based acceleration demonstrated")
    print(f"   - Stage-specific patterns analyzed")


async def main():
    """Main demo function"""
    try:
        await demonstrate_acceleration_system()
        await demonstrate_advanced_scenarios()
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())