"""
Demo script for Strategy Optimization Engine

This script demonstrates the capabilities of the Strategy Optimization Engine
for cultural transformation leadership.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

from scrollintel.engines.strategy_optimization_engine import (
    StrategyOptimizationEngine,
    OptimizationContext,
    OptimizationMetric,
    OptimizationType,
    OptimizationPriority
)


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'-'*40}")
    print(f"{title}")
    print(f"{'-'*40}")


def create_sample_context(scenario: str) -> OptimizationContext:
    """Create sample optimization contexts for different scenarios"""
    scenarios = {
        "struggling_transformation": OptimizationContext(
            transformation_id="trans_struggling_001",
            current_progress=0.3,
            timeline_status="behind_schedule",
            budget_utilization=0.8,
            resistance_level=0.7,
            engagement_score=0.4,
            performance_metrics={
                "culture_alignment": 0.3,
                "behavior_adoption": 0.2,
                "leadership_buy_in": 0.5
            },
            external_factors=["economic_uncertainty", "competitor_pressure", "regulatory_changes"]
        ),
        "moderate_transformation": OptimizationContext(
            transformation_id="trans_moderate_001",
            current_progress=0.6,
            timeline_status="slightly_behind",
            budget_utilization=0.7,
            resistance_level=0.4,
            engagement_score=0.6,
            performance_metrics={
                "culture_alignment": 0.6,
                "behavior_adoption": 0.5,
                "leadership_buy_in": 0.7
            },
            external_factors=["market_volatility"]
        ),
        "successful_transformation": OptimizationContext(
            transformation_id="trans_successful_001",
            current_progress=0.8,
            timeline_status="on_track",
            budget_utilization=0.6,
            resistance_level=0.2,
            engagement_score=0.8,
            performance_metrics={
                "culture_alignment": 0.8,
                "behavior_adoption": 0.7,
                "leadership_buy_in": 0.9
            },
            external_factors=["favorable_market_conditions"]
        )
    }
    
    return scenarios.get(scenario, scenarios["moderate_transformation"])


def create_sample_metrics(scenario: str) -> List[OptimizationMetric]:
    """Create sample metrics for different scenarios"""
    base_time = datetime.now()
    
    if scenario == "struggling_transformation":
        return [
            OptimizationMetric(
                name="culture_alignment",
                current_value=0.3,
                target_value=0.8,
                weight=0.9,
                trend="declining",
                last_updated=base_time
            ),
            OptimizationMetric(
                name="behavior_adoption",
                current_value=0.2,
                target_value=0.7,
                weight=0.8,
                trend="stable",
                last_updated=base_time
            ),
            OptimizationMetric(
                name="employee_engagement",
                current_value=0.4,
                target_value=0.8,
                weight=0.7,
                trend="declining",
                last_updated=base_time
            ),
            OptimizationMetric(
                name="leadership_effectiveness",
                current_value=0.5,
                target_value=0.8,
                weight=0.8,
                trend="stable",
                last_updated=base_time
            )
        ]
    elif scenario == "moderate_transformation":
        return [
            OptimizationMetric(
                name="culture_alignment",
                current_value=0.6,
                target_value=0.8,
                weight=0.9,
                trend="improving",
                last_updated=base_time
            ),
            OptimizationMetric(
                name="behavior_adoption",
                current_value=0.5,
                target_value=0.7,
                weight=0.8,
                trend="stable",
                last_updated=base_time
            ),
            OptimizationMetric(
                name="employee_engagement",
                current_value=0.6,
                target_value=0.8,
                weight=0.7,
                trend="improving",
                last_updated=base_time
            )
        ]
    else:  # successful_transformation
        return [
            OptimizationMetric(
                name="culture_alignment",
                current_value=0.8,
                target_value=0.8,
                weight=0.9,
                trend="stable",
                last_updated=base_time
            ),
            OptimizationMetric(
                name="behavior_adoption",
                current_value=0.7,
                target_value=0.7,
                weight=0.8,
                trend="improving",
                last_updated=base_time
            ),
            OptimizationMetric(
                name="employee_engagement",
                current_value=0.8,
                target_value=0.8,
                weight=0.7,
                trend="stable",
                last_updated=base_time
            )
        ]


def display_context_summary(context: OptimizationContext):
    """Display optimization context summary"""
    print(f"Transformation ID: {context.transformation_id}")
    print(f"Current Progress: {context.current_progress:.1%}")
    print(f"Timeline Status: {context.timeline_status}")
    print(f"Budget Utilization: {context.budget_utilization:.1%}")
    print(f"Resistance Level: {context.resistance_level:.1%}")
    print(f"Engagement Score: {context.engagement_score:.1%}")
    print(f"Performance Metrics: {json.dumps(context.performance_metrics, indent=2)}")
    print(f"External Factors: {', '.join(context.external_factors)}")


def display_metrics_summary(metrics: List[OptimizationMetric]):
    """Display metrics summary"""
    print(f"{'Metric':<25} {'Current':<10} {'Target':<10} {'Gap':<10} {'Trend':<12} {'Weight':<8}")
    print("-" * 80)
    
    for metric in metrics:
        gap = metric.target_value - metric.current_value
        print(f"{metric.name:<25} {metric.current_value:<10.2f} {metric.target_value:<10.2f} "
              f"{gap:<10.2f} {metric.trend:<12} {metric.weight:<8.1f}")


def display_recommendations(recommendations: List):
    """Display optimization recommendations"""
    if not recommendations:
        print("No optimization recommendations generated.")
        return
    
    print(f"Generated {len(recommendations)} optimization recommendations:\n")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.title}")
        print(f"   Type: {rec.optimization_type.value}")
        print(f"   Priority: {rec.priority.value.upper()}")
        print(f"   Expected Impact: {rec.expected_impact:.1%}")
        print(f"   Implementation Effort: {rec.implementation_effort}")
        print(f"   Timeline: {rec.timeline}")
        print(f"   Success Probability: {rec.success_probability:.1%}")
        print(f"   Description: {rec.description}")
        
        if rec.dependencies:
            print(f"   Dependencies: {', '.join(rec.dependencies)}")
        
        if rec.risks:
            print(f"   Risks: {', '.join(rec.risks)}")
        
        print()


def demonstrate_strategy_adjustment(engine: StrategyOptimizationEngine, 
                                  transformation_id: str, 
                                  recommendation):
    """Demonstrate strategy adjustment creation and implementation"""
    print_section("Strategy Adjustment Process")
    
    # Original strategy
    original_strategy = {
        "approach": "gradual_change",
        "timeline": "12_months",
        "communication_frequency": "monthly",
        "training_intensity": "standard",
        "leadership_involvement": "moderate",
        "budget_allocation": {
            "training": 0.4,
            "communication": 0.2,
            "technology": 0.3,
            "consulting": 0.1
        }
    }
    
    print("Original Strategy:")
    print(json.dumps(original_strategy, indent=2))
    
    # Create strategy adjustment
    adjustment = engine.create_strategy_adjustment(
        transformation_id, recommendation, original_strategy
    )
    
    print(f"\nStrategy Adjustment Created:")
    print(f"Adjustment ID: {adjustment.id}")
    print(f"Type: {adjustment.adjustment_type}")
    print(f"Rationale: {adjustment.rationale}")
    print(f"Implementation Date: {adjustment.implementation_date}")
    
    print(f"\nAdjusted Strategy:")
    print(json.dumps(adjustment.adjusted_strategy, indent=2))
    
    print(f"\nExpected Outcomes:")
    for outcome in adjustment.expected_outcomes:
        print(f"  • {outcome}")
    
    # Implement adjustment
    print(f"\nImplementing adjustment...")
    success = engine.implement_adjustment(adjustment.id)
    
    if success:
        print("✅ Strategy adjustment implemented successfully!")
        
        # Monitor effectiveness
        print(f"\nMonitoring adjustment effectiveness...")
        post_metrics = [
            OptimizationMetric(
                name="implementation_progress",
                current_value=0.7,
                target_value=0.8,
                weight=1.0,
                trend="improving",
                last_updated=datetime.now()
            )
        ]
        
        effectiveness = engine.monitor_adjustment_effectiveness(adjustment.id, post_metrics)
        
        print(f"Effectiveness Score: {effectiveness['effectiveness_score']:.1%}")
        print(f"Impact Analysis: {json.dumps(effectiveness['impact_analysis'], indent=2)}")
        print(f"Next Steps: {', '.join(effectiveness['recommendations'])}")
    else:
        print("❌ Strategy adjustment implementation failed.")
    
    return adjustment


def demonstrate_continuous_improvement(engine: StrategyOptimizationEngine, 
                                     transformation_id: str):
    """Demonstrate continuous improvement plan generation"""
    print_section("Continuous Improvement Planning")
    
    # Sample optimization history
    optimization_history = [
        {
            "optimization_id": "opt_001",
            "type": "performance_based",
            "applied_date": "2024-01-15",
            "success": True,
            "effectiveness_score": 0.8,
            "lessons_learned": "Performance metrics improved significantly with focused interventions"
        },
        {
            "optimization_id": "opt_002",
            "type": "engagement_based",
            "applied_date": "2024-01-22",
            "success": True,
            "effectiveness_score": 0.9,
            "lessons_learned": "Employee engagement initiatives had high impact and acceptance"
        },
        {
            "optimization_id": "opt_003",
            "type": "resistance_based",
            "applied_date": "2024-02-01",
            "success": False,
            "effectiveness_score": 0.3,
            "lessons_learned": "Resistance mitigation strategies need more stakeholder involvement"
        },
        {
            "optimization_id": "opt_004",
            "type": "timeline_based",
            "applied_date": "2024-02-10",
            "success": True,
            "effectiveness_score": 0.7,
            "lessons_learned": "Timeline acceleration worked but required additional resources"
        }
    ]
    
    print("Optimization History:")
    for opt in optimization_history:
        status = "✅ Success" if opt["success"] else "❌ Failed"
        print(f"  {opt['optimization_id']}: {opt['type']} - {status} "
              f"(Effectiveness: {opt['effectiveness_score']:.1%})")
    
    # Generate improvement plan
    improvement_plan = engine.generate_continuous_improvement_plan(
        transformation_id, optimization_history
    )
    
    print(f"\nContinuous Improvement Plan:")
    print(f"Transformation ID: {improvement_plan['transformation_id']}")
    
    print(f"\nPatterns Identified:")
    patterns = improvement_plan['patterns_identified']
    print(f"  Most Common Optimizations: {', '.join(patterns['most_common_optimizations'])}")
    print(f"  Success Rates by Type:")
    for opt_type, rate in patterns['success_rate_by_type'].items():
        print(f"    {opt_type}: {rate:.1%}")
    print(f"  Seasonal Patterns: {patterns['seasonal_patterns']}")
    print(f"  Correlation Insights: {patterns['correlation_insights']}")
    
    print(f"\nImprovement Recommendations:")
    for rec in improvement_plan['improvement_recommendations']:
        print(f"  • {rec}")
    
    print(f"\nLearning Integration Plan:")
    learning_plan = improvement_plan['learning_integration_plan']
    print(f"  Knowledge Capture:")
    for key, value in learning_plan['knowledge_capture'].items():
        print(f"    {key}: {value}")
    
    print(f"  System Improvements:")
    for key, value in learning_plan['system_improvements'].items():
        print(f"    {key}: {value}")


async def run_optimization_demo():
    """Run the complete optimization demo"""
    print_header("ScrollIntel Strategy Optimization Engine Demo")
    
    # Initialize the optimization engine
    engine = StrategyOptimizationEngine()
    
    # Demo scenarios
    scenarios = [
        ("struggling_transformation", "Struggling Transformation"),
        ("moderate_transformation", "Moderate Transformation"),
        ("successful_transformation", "Successful Transformation")
    ]
    
    for scenario_key, scenario_name in scenarios:
        print_header(f"Scenario: {scenario_name}")
        
        # Create context and metrics
        context = create_sample_context(scenario_key)
        metrics = create_sample_metrics(scenario_key)
        
        print_section("Transformation Context")
        display_context_summary(context)
        
        print_section("Performance Metrics")
        display_metrics_summary(metrics)
        
        print_section("Optimization Analysis")
        print("Analyzing transformation context and generating optimization recommendations...")
        
        # Generate optimization recommendations
        recommendations = engine.optimize_strategy(context, metrics)
        
        display_recommendations(recommendations)
        
        # Demonstrate strategy adjustment for the first scenario
        if scenario_key == "struggling_transformation" and recommendations:
            adjustment = demonstrate_strategy_adjustment(
                engine, context.transformation_id, recommendations[0]
            )
        
        # Demonstrate continuous improvement for the moderate scenario
        if scenario_key == "moderate_transformation":
            demonstrate_continuous_improvement(engine, context.transformation_id)
        
        print("\n" + "="*60)
        input("Press Enter to continue to the next scenario...")
    
    print_header("Demo Complete")
    print("The Strategy Optimization Engine has demonstrated:")
    print("✅ Context-aware optimization analysis")
    print("✅ Multi-dimensional recommendation generation")
    print("✅ Prioritized optimization strategies")
    print("✅ Strategy adjustment creation and implementation")
    print("✅ Effectiveness monitoring and feedback")
    print("✅ Continuous improvement planning")
    print("✅ Learning integration and pattern recognition")
    
    print(f"\nThe engine successfully optimized strategies for {len(scenarios)} different scenarios,")
    print("demonstrating its ability to adapt to various transformation contexts and challenges.")


def run_quick_demo():
    """Run a quick demo for testing"""
    print_header("Quick Strategy Optimization Demo")
    
    engine = StrategyOptimizationEngine()
    context = create_sample_context("moderate_transformation")
    metrics = create_sample_metrics("moderate_transformation")
    
    print("Context: Moderate transformation with some challenges")
    print(f"Progress: {context.current_progress:.1%}, Resistance: {context.resistance_level:.1%}, "
          f"Engagement: {context.engagement_score:.1%}")
    
    recommendations = engine.optimize_strategy(context, metrics)
    
    print(f"\nGenerated {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
        print(f"{i}. {rec.title} (Priority: {rec.priority.value}, Impact: {rec.expected_impact:.1%})")
    
    if recommendations:
        print(f"\nCreating strategy adjustment for top recommendation...")
        original_strategy = {"approach": "standard"}
        adjustment = engine.create_strategy_adjustment(
            context.transformation_id, recommendations[0], original_strategy
        )
        print(f"✅ Created adjustment: {adjustment.id}")
        
        success = engine.implement_adjustment(adjustment.id)
        print(f"✅ Implementation {'successful' if success else 'failed'}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_demo()
    else:
        asyncio.run(run_optimization_demo())