#!/usr/bin/env python3
"""
Demo script for Recovery Planning Engine

This script demonstrates the capabilities of the recovery planning engine
for post-crisis recovery strategy development, milestone tracking, and optimization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from scrollintel.engines.recovery_planning_engine import RecoveryPlanningEngine
from scrollintel.models.crisis_detection_models import CrisisModel


def main():
    """Main demo function"""
    print("🔄 Recovery Planning Engine Demo")
    print("=" * 50)
    
    # Initialize the recovery planning engine
    recovery_engine = RecoveryPlanningEngine()
    
    # Create a sample crisis for recovery planning
    sample_crisis = CrisisModel(
        id="crisis_20240104_001",
        crisis_type="system_failure",
        severity_level="high",
        status="resolved",
        start_time=datetime.now() - timedelta(days=2),
        affected_areas=["production_systems", "customer_portal", "payment_processing"],
        stakeholders_impacted=["customers", "employees", "partners", "investors"],
        resolution_time=datetime.now() - timedelta(hours=6)
    )
    
    # Define recovery objectives
    recovery_objectives = [
        "Restore full operational capacity",
        "Rebuild stakeholder confidence",
        "Implement preventive measures",
        "Enhance organizational resilience",
        "Document lessons learned"
    ]
    
    print("\n1. Developing Recovery Strategy")
    print("-" * 30)
    
    try:
        # Develop recovery strategy
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        print(f"✅ Recovery strategy developed: {strategy.id}")
        print(f"   Strategy name: {strategy.strategy_name}")
        print(f"   Recovery objectives: {len(strategy.recovery_objectives)}")
        print(f"   Milestones: {len(strategy.milestones)}")
        print(f"   Timeline phases: {len(strategy.timeline)}")
        
        # Display milestones by phase
        print("\n   Recovery Milestones by Phase:")
        for milestone in strategy.milestones:
            print(f"   • {milestone.phase.value.upper()}: {milestone.name}")
            print(f"     Priority: {milestone.priority.value}, Target: {milestone.target_date.strftime('%Y-%m-%d')}")
        
        # Display resource allocation
        print(f"\n   Resource Allocation:")
        for resource_type, allocation in strategy.resource_allocation.items():
            print(f"   • {resource_type.title()}: {allocation}")
        
        print(f"\n   Success Metrics:")
        for metric, target in strategy.success_metrics.items():
            print(f"   • {metric.replace('_', ' ').title()}: {target}%")
        
    except Exception as e:
        print(f"❌ Error developing recovery strategy: {str(e)}")
        return
    
    print("\n2. Tracking Recovery Progress")
    print("-" * 30)
    
    try:
        # Simulate some progress on milestones
        for i, milestone in enumerate(strategy.milestones[:3]):
            milestone.progress_percentage = min(100.0, (i + 1) * 30)
            if milestone.progress_percentage == 100.0:
                milestone.status = milestone.status.__class__.COMPLETED
                milestone.completion_date = datetime.now()
        
        # Track recovery progress
        progress = recovery_engine.track_recovery_progress(strategy.id)
        
        print(f"✅ Recovery progress tracked for strategy: {strategy.id}")
        print(f"   Overall progress: {progress.overall_progress:.1f}%")
        print(f"   Milestone completion rate: {progress.milestone_completion_rate:.1f}%")
        print(f"   Timeline adherence: {progress.timeline_adherence:.1f}%")
        
        print(f"\n   Phase Progress:")
        for phase, prog in progress.phase_progress.items():
            print(f"   • {phase.value.replace('_', ' ').title()}: {prog:.1f}%")
        
        print(f"\n   Resource Utilization:")
        for resource, utilization in progress.resource_utilization.items():
            print(f"   • {resource.replace('_', ' ').title()}: {utilization:.1f}%")
        
        print(f"\n   Success Metric Achievement:")
        for metric, achievement in progress.success_metric_achievement.items():
            print(f"   • {metric.replace('_', ' ').title()}: {achievement:.1f}%")
        
        if progress.identified_issues:
            print(f"\n   Identified Issues:")
            for issue in progress.identified_issues:
                print(f"   ⚠️  {issue}")
        
        if progress.recommended_adjustments:
            print(f"\n   Recommended Adjustments:")
            for adjustment in progress.recommended_adjustments:
                print(f"   💡 {adjustment}")
        
    except Exception as e:
        print(f"❌ Error tracking recovery progress: {str(e)}")
        return
    
    print("\n3. Optimizing Recovery Strategy")
    print("-" * 30)
    
    try:
        # Generate optimization recommendations
        optimizations = recovery_engine.optimize_recovery_strategy(strategy.id)
        
        print(f"✅ Recovery optimization completed for strategy: {strategy.id}")
        print(f"   Optimization recommendations: {len(optimizations)}")
        
        for i, optimization in enumerate(optimizations, 1):
            print(f"\n   Optimization {i}: {optimization.optimization_type.title()}")
            print(f"   Priority Score: {optimization.priority_score}/10")
            print(f"   Implementation Effort: {optimization.implementation_effort.title()}")
            
            print(f"   Current Performance:")
            for metric, value in optimization.current_performance.items():
                print(f"   • {metric.replace('_', ' ').title()}: {value:.1f}")
            
            print(f"   Target Performance:")
            for metric, value in optimization.target_performance.items():
                print(f"   • {metric.replace('_', ' ').title()}: {value:.1f}")
            
            print(f"   Recommended Actions:")
            for action in optimization.recommended_actions:
                print(f"   • {action}")
            
            print(f"   Expected Impact:")
            for impact, value in optimization.expected_impact.items():
                print(f"   • {impact.replace('_', ' ').title()}: +{value:.1f}")
        
    except Exception as e:
        print(f"❌ Error optimizing recovery strategy: {str(e)}")
        return
    
    print("\n4. Recovery Strategy Summary")
    print("-" * 30)
    
    try:
        # Display comprehensive recovery strategy summary
        print(f"Recovery Strategy: {strategy.strategy_name}")
        print(f"Created: {strategy.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Crisis ID: {strategy.crisis_id}")
        
        print(f"\nRecovery Timeline:")
        for phase, duration in strategy.timeline.items():
            print(f"• {phase.value.replace('_', ' ').title()}: {duration}")
        
        print(f"\nStakeholder Communication Plan:")
        comm_plan = strategy.stakeholder_communication_plan
        for group, details in comm_plan.get("stakeholder_groups", {}).items():
            print(f"• {group.title()}: {details['frequency']} via {', '.join(details['channels'])}")
        
        print(f"\nRisk Mitigation Measures:")
        for measure in strategy.risk_mitigation_measures:
            print(f"• {measure}")
        
        print(f"\nContingency Plans:")
        for scenario, plan in strategy.contingency_plans.items():
            print(f"• {scenario.replace('_', ' ').title()}: {len(plan['actions'])} actions defined")
        
        print(f"\n✅ Recovery Planning Engine Demo completed successfully!")
        print(f"   Strategy developed with {len(strategy.milestones)} milestones")
        print(f"   Progress tracking implemented with {len(progress.phase_progress)} phases")
        print(f"   {len(optimizations)} optimization recommendations generated")
        
    except Exception as e:
        print(f"❌ Error in recovery strategy summary: {str(e)}")


if __name__ == "__main__":
    main()