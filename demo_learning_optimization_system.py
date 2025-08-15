"""
Demo script for Learning Optimization System

This script demonstrates the capabilities of the learning optimization system
including continuous learning optimization, effectiveness measurement, adaptive learning,
and innovation process enhancement.
"""

import asyncio
from datetime import datetime, timedelta
from scrollintel.engines.learning_optimization_system import LearningOptimizationSystem
from scrollintel.models.knowledge_integration_models import LearningMetric


async def demo_learning_optimization():
    """Demonstrate learning optimization system capabilities"""
    
    print("🧠 Learning Optimization System Demo")
    print("=" * 50)
    
    # Initialize system
    system = LearningOptimizationSystem()
    
    print("\n📊 Step 1: Setting Up Learning Context and Metrics")
    print("-" * 40)
    
    # Define learning context
    learning_context = {
        "domain": "autonomous_innovation_lab",
        "learner_profile": {
            "experience_level": "advanced",
            "learning_style": "hands_on_experimental",
            "preferred_pace": "accelerated",
            "focus_areas": ["pattern_recognition", "knowledge_synthesis", "adaptive_systems"]
        },
        "environment": {
            "available_time_hours_per_week": 40,
            "resource_budget": 50000,
            "team_size": 8,
            "infrastructure": "high_performance_computing"
        },
        "constraints": {
            "deadline_pressure": "high",
            "quality_requirements": "very_high",
            "innovation_targets": ["breakthrough_discoveries", "process_optimization"]
        }
    }
    
    print("🎯 Learning Context:")
    print(f"   Domain: {learning_context['domain']}")
    print(f"   Experience Level: {learning_context['learner_profile']['experience_level']}")
    print(f"   Learning Style: {learning_context['learner_profile']['learning_style']}")
    print(f"   Weekly Time Budget: {learning_context['environment']['available_time_hours_per_week']} hours")
    print(f"   Team Size: {learning_context['environment']['team_size']}")
    print()
    
    # Create sample learning metrics representing current performance
    current_metrics = [
        LearningMetric(
            metric_name="knowledge_acquisition_rate",
            value=0.72,
            timestamp=datetime.now() - timedelta(days=7),
            context={"activity": "pattern_recognition_training", "difficulty": "high"},
            improvement_rate=0.08
        ),
        LearningMetric(
            metric_name="knowledge_retention_rate",
            value=0.85,
            timestamp=datetime.now() - timedelta(days=5),
            context={"activity": "knowledge_synthesis_practice", "difficulty": "medium"},
            improvement_rate=0.05
        ),
        LearningMetric(
            metric_name="application_success_rate",
            value=0.68,
            timestamp=datetime.now() - timedelta(days=3),
            context={"activity": "real_world_problem_solving", "difficulty": "very_high"},
            improvement_rate=-0.03
        ),
        LearningMetric(
            metric_name="innovation_generation_rate",
            value=0.58,
            timestamp=datetime.now() - timedelta(days=2),
            context={"activity": "breakthrough_ideation", "difficulty": "extreme"},
            improvement_rate=0.12
        ),
        LearningMetric(
            metric_name="adaptive_learning_efficiency",
            value=0.75,
            timestamp=datetime.now() - timedelta(days=1),
            context={"activity": "process_optimization", "difficulty": "high"},
            improvement_rate=0.06
        ),
        LearningMetric(
            metric_name="knowledge_acquisition_rate",
            value=0.78,
            timestamp=datetime.now(),
            context={"activity": "advanced_pattern_analysis", "difficulty": "very_high"},
            improvement_rate=0.06
        )
    ]
    
    print("📈 Current Learning Metrics:")
    for metric in current_metrics:
        trend_icon = "📈" if metric.improvement_rate > 0 else "📉" if metric.improvement_rate < 0 else "➡️"
        print(f"   {trend_icon} {metric.metric_name}: {metric.value:.3f} (trend: {metric.improvement_rate:+.3f})")
    print()
    
    print("\n🚀 Step 2: Optimizing Continuous Learning")
    print("-" * 40)
    
    # Define optimization targets
    optimization_targets = [
        "knowledge_acquisition_rate",
        "application_success_rate", 
        "innovation_generation_rate"
    ]
    
    print(f"🎯 Optimization Targets: {', '.join(optimization_targets)}")
    print()
    
    # Optimize continuous learning
    optimization = await system.optimize_continuous_learning(
        learning_context, current_metrics, optimization_targets
    )
    
    print(f"✅ Learning Optimization Complete")
    print(f"   Optimization ID: {optimization.id}")
    print(f"   Strategy: {optimization.optimization_strategy}")
    print(f"   Effectiveness Score: {optimization.effectiveness_score:.3f}")
    print(f"   Target Metrics: {optimization.optimization_target}")
    print()
    
    print("⚙️ Optimization Parameters:")
    for param, value in optimization.parameters.items():
        print(f"   {param}: {value}")
    print()
    
    print("\n📊 Step 3: Measuring Learning Effectiveness")
    print("-" * 40)
    
    # Define learning activities
    learning_activities = [
        {
            "name": "advanced_pattern_recognition_workshop",
            "type": "hands_on",
            "duration_hours": 32,
            "feedback_available": True,
            "interactive": True,
            "complexity": "very_high",
            "reinforcement": True,
            "practical_application": True,
            "practical_focus": True,
            "mentorship": True,
            "innovation_focus": True
        },
        {
            "name": "knowledge_synthesis_laboratory",
            "type": "experimental",
            "duration_hours": 28,
            "feedback_available": True,
            "interactive": True,
            "complexity": "high",
            "reinforcement": True,
            "practical_application": True,
            "collaborative": True,
            "research_oriented": True
        },
        {
            "name": "adaptive_systems_theory_seminar",
            "type": "theoretical",
            "duration_hours": 16,
            "feedback_available": False,
            "interactive": False,
            "complexity": "extreme",
            "reinforcement": False,
            "practical_application": False,
            "depth_focused": True,
            "expert_led": True
        },
        {
            "name": "innovation_breakthrough_hackathon",
            "type": "competitive",
            "duration_hours": 48,
            "feedback_available": True,
            "interactive": True,
            "complexity": "extreme",
            "reinforcement": True,
            "practical_application": True,
            "practical_focus": True,
            "mentorship": True,
            "time_pressured": True,
            "innovation_focus": True
        },
        {
            "name": "autonomous_lab_simulation",
            "type": "simulation",
            "duration_hours": 24,
            "feedback_available": True,
            "interactive": True,
            "complexity": "very_high",
            "reinforcement": True,
            "practical_application": True,
            "realistic_scenarios": True,
            "performance_tracking": True
        }
    ]
    
    print(f"🎓 Learning Activities ({len(learning_activities)} total):")
    for activity in learning_activities:
        complexity_icon = "🔥" if activity["complexity"] == "extreme" else "⚡" if activity["complexity"] == "very_high" else "💡"
        print(f"   {complexity_icon} {activity['name']} ({activity['type']}, {activity['duration_hours']}h)")
    print()
    
    # Measure learning effectiveness
    time_window = timedelta(days=30)
    effectiveness = await system.measure_learning_effectiveness(learning_activities, time_window)
    
    print(f"📊 Learning Effectiveness Analysis (30-day window)")
    print(f"   Overall Effectiveness: {effectiveness['overall_effectiveness']:.3f}")
    print(f"   Learning Velocity: {effectiveness['learning_velocity']:.2f} activities/day")
    print(f"   Knowledge Retention: {effectiveness['knowledge_retention']:.3f}")
    print(f"   Application Success: {effectiveness['application_success']:.3f}")
    print()
    
    print("🎯 Activity-Specific Effectiveness:")
    for activity_name, activity_eff in effectiveness["activity_effectiveness"].items():
        score = activity_eff["effectiveness_score"]
        engagement = activity_eff["engagement_level"]
        retention = activity_eff["retention_rate"]
        
        score_icon = "🌟" if score > 0.8 else "⭐" if score > 0.6 else "💫"
        print(f"   {score_icon} {activity_name[:30]}...")
        print(f"      Effectiveness: {score:.3f}, Engagement: {engagement:.3f}, Retention: {retention:.3f}")
    print()
    
    print("📈 Improvement Trends:")
    for activity_name, trend in effectiveness["improvement_trends"].items():
        direction_icon = "📈" if trend["direction"] == "improving" else "📉" if trend["direction"] == "declining" else "➡️"
        print(f"   {direction_icon} {activity_name}: {trend['direction']} (strength: {trend['strength']:.3f})")
    print()
    
    print("\n🔄 Step 4: Implementing Adaptive Learning")
    print("-" * 40)
    
    # Define performance feedback
    performance_feedback = [
        {
            "feedback_type": "peer_review",
            "sentiment": "positive",
            "strength": "exceptional_pattern_recognition_skills",
            "timestamp": datetime.now() - timedelta(days=3),
            "source": "research_team_lead"
        },
        {
            "feedback_type": "performance_assessment",
            "sentiment": "negative",
            "issue": "slow_knowledge_synthesis_under_pressure",
            "timestamp": datetime.now() - timedelta(days=2),
            "source": "innovation_lab_supervisor"
        },
        {
            "feedback_type": "self_reflection",
            "sentiment": "neutral",
            "issue": "difficulty_with_extreme_complexity_problems",
            "strength": "strong_adaptive_learning_capability",
            "timestamp": datetime.now() - timedelta(days=1),
            "source": "self_assessment"
        },
        {
            "feedback_type": "mentor_evaluation",
            "sentiment": "positive",
            "strength": "innovative_problem_solving_approach",
            "timestamp": datetime.now(),
            "source": "senior_researcher"
        }
    ]
    
    adaptation_goals = [
        "improve_performance_under_pressure",
        "enhance_complex_problem_solving",
        "accelerate_knowledge_synthesis",
        "optimize_innovation_generation"
    ]
    
    print(f"📝 Performance Feedback ({len(performance_feedback)} items):")
    for feedback in performance_feedback:
        sentiment_icon = "✅" if feedback["sentiment"] == "positive" else "❌" if feedback["sentiment"] == "negative" else "⚠️"
        print(f"   {sentiment_icon} {feedback['feedback_type']}: {feedback.get('strength', feedback.get('issue', 'general_feedback'))}")
    print()
    
    print(f"🎯 Adaptation Goals: {', '.join(adaptation_goals)}")
    print()
    
    # Implement adaptive learning
    adaptive_result = await system.implement_adaptive_learning(
        learning_context, performance_feedback, adaptation_goals
    )
    
    print(f"🔄 Adaptive Learning Implementation Complete")
    print(f"   Adaptation ID: {adaptive_result['adaptation_id']}")
    print(f"   Effectiveness Improvement: {adaptive_result['effectiveness_improvement']:.3f}")
    print()
    
    print("🔍 Identified Adaptation Needs:")
    for i, need in enumerate(adaptive_result["adaptation_needs"], 1):
        priority_icon = "🔥" if need["priority"] == "high" else "⚡" if need["priority"] == "medium" else "💡"
        print(f"   {i}. {priority_icon} {need['description']} (Priority: {need['priority']})")
    print()
    
    print("🛠️ Adaptive Strategies:")
    for i, strategy in enumerate(adaptive_result["adaptive_strategies"], 1):
        print(f"   {i}. {strategy['strategy_type']}: {strategy['description']}")
        print(f"      Approach: {strategy['approach']}")
    print()
    
    print("📊 Implementation Results:")
    impl_results = adaptive_result["implementation_results"]
    successful_strategies = sum(1 for success in impl_results["implementation_success"].values() if success)
    total_strategies = len(impl_results["implemented_strategies"])
    
    print(f"   Successful Implementations: {successful_strategies}/{total_strategies}")
    print(f"   Challenges Encountered: {len(impl_results['challenges_encountered'])}")
    
    if impl_results["challenges_encountered"]:
        print("   ⚠️ Implementation Challenges:")
        for challenge in impl_results["challenges_encountered"]:
            print(f"     • {challenge['challenge']} (Impact: {challenge['impact']})")
    print()
    
    print("\n⚙️ Step 5: Enhancing Innovation Processes")
    print("-" * 40)
    
    # Define current innovation processes
    process_data = [
        {
            "name": "research_ideation",
            "efficiency": 0.78,
            "bottlenecks": ["limited_cross_domain_knowledge"],
            "duration_days": 3,
            "resource_utilization": 0.85,
            "innovation_output_rate": 0.72
        },
        {
            "name": "hypothesis_generation",
            "efficiency": 0.65,
            "bottlenecks": ["insufficient_pattern_recognition", "time_constraints"],
            "duration_days": 2,
            "resource_utilization": 0.70,
            "innovation_output_rate": 0.58
        },
        {
            "name": "experimental_design",
            "efficiency": 0.82,
            "bottlenecks": [],
            "duration_days": 4,
            "resource_utilization": 0.90,
            "innovation_output_rate": 0.85
        },
        {
            "name": "prototype_development",
            "efficiency": 0.45,  # Major bottleneck
            "bottlenecks": ["resource_constraints", "skill_gaps", "coordination_issues"],
            "duration_days": 12,
            "resource_utilization": 0.55,
            "innovation_output_rate": 0.38
        },
        {
            "name": "validation_testing",
            "efficiency": 0.88,
            "bottlenecks": [],
            "duration_days": 5,
            "resource_utilization": 0.92,
            "innovation_output_rate": 0.90
        },
        {
            "name": "knowledge_integration",
            "efficiency": 0.58,
            "bottlenecks": ["complexity_management", "synthesis_speed"],
            "duration_days": 6,
            "resource_utilization": 0.68,
            "innovation_output_rate": 0.52
        }
    ]
    
    enhancement_objectives = [
        "reduce_overall_innovation_cycle_time",
        "improve_prototype_development_efficiency",
        "enhance_knowledge_synthesis_speed",
        "increase_breakthrough_discovery_rate",
        "optimize_resource_utilization"
    ]
    
    print(f"🏭 Current Innovation Processes ({len(process_data)} processes):")
    for process in process_data:
        efficiency = process["efficiency"]
        efficiency_icon = "🌟" if efficiency > 0.8 else "⚡" if efficiency > 0.6 else "⚠️" if efficiency > 0.4 else "🚨"
        print(f"   {efficiency_icon} {process['name']}: {efficiency:.3f} efficiency ({process['duration_days']} days)")
        if process["bottlenecks"]:
            print(f"      Bottlenecks: {', '.join(process['bottlenecks'])}")
    print()
    
    print(f"🎯 Enhancement Objectives:")
    for i, objective in enumerate(enhancement_objectives, 1):
        print(f"   {i}. {objective.replace('_', ' ').title()}")
    print()
    
    # Enhance innovation processes
    enhancement_result = await system.enhance_innovation_processes(process_data, enhancement_objectives)
    
    print(f"⚙️ Process Enhancement Complete")
    print(f"   Enhancement ID: {enhancement_result['enhancement_id']}")
    print()
    
    print("📊 Process Analysis Results:")
    analysis = enhancement_result["process_analysis"]
    print(f"   Overall Process Efficiency: {analysis['process_efficiency']:.3f}")
    print(f"   Bottlenecks Identified: {len(analysis['bottlenecks'])}")
    print(f"   Strengths Identified: {len(analysis['strengths'])}")
    print()
    
    if analysis["bottlenecks"]:
        print("🚨 Critical Bottlenecks:")
        for bottleneck in analysis["bottlenecks"]:
            print(f"   • {bottleneck['process']}: {bottleneck['efficiency']:.3f} efficiency")
            print(f"     Issue: {bottleneck['issue']}")
    print()
    
    if analysis["strengths"]:
        print("🌟 Process Strengths:")
        for strength in analysis["strengths"]:
            print(f"   • {strength['process']}: {strength['efficiency']:.3f} efficiency")
            print(f"     Strength: {strength['strength']}")
    print()
    
    print("🎯 Prioritized Enhancements:")
    for i, enhancement in enumerate(enhancement_result["prioritized_enhancements"][:5], 1):
        priority_icon = "🔥" if enhancement["priority"] == "high" else "⚡" if enhancement["priority"] == "medium" else "💡"
        print(f"   {i}. {priority_icon} {enhancement['strategy_type']}")
        print(f"      Target: {enhancement['target']}")
        print(f"      Expected Improvement: {enhancement['expected_improvement']:.3f}")
        print(f"      Priority: {enhancement['priority']}")
    print()
    
    print("📋 Implementation Plan:")
    plan = enhancement_result["implementation_plan"]
    print(f"   Timeline: {plan['timeline']}")
    
    for phase in plan["phases"]:
        print(f"   📅 {phase['phase']} ({phase['duration']})")
        print(f"      Focus: {phase['focus']}")
        print(f"      Enhancements: {len(phase['enhancements'])}")
    print()
    
    print("📈 Expected Impact:")
    impact = enhancement_result["impact_estimation"]
    print(f"   Current Efficiency: {impact['current_efficiency']:.3f}")
    print(f"   Expected Efficiency: {impact['expected_efficiency']:.3f}")
    print(f"   Total Improvement: {impact['total_improvement']:.3f}")
    print(f"   ROI Estimate: {impact['roi_estimate']:.1f}x")
    print(f"   Implementation Risk: {impact['implementation_risk']}")
    print()
    
    print("\n🔧 Step 6: Parameter Optimization")
    print("-" * 40)
    
    # Simulate performance data over time
    performance_data = [
        {"performance_score": 0.68, "timestamp": datetime.now() - timedelta(days=5)},
        {"performance_score": 0.71, "timestamp": datetime.now() - timedelta(days=4)},
        {"performance_score": 0.69, "timestamp": datetime.now() - timedelta(days=3)},
        {"performance_score": 0.74, "timestamp": datetime.now() - timedelta(days=2)},
        {"performance_score": 0.76, "timestamp": datetime.now() - timedelta(days=1)},
        {"performance_score": 0.78, "timestamp": datetime.now()}
    ]
    
    print("📊 Recent Performance Data:")
    for data in performance_data:
        print(f"   {data['timestamp'].strftime('%Y-%m-%d')}: {data['performance_score']:.3f}")
    print()
    
    # Optimize learning parameters
    optimization_result = await system.optimize_learning_parameters(optimization.id, performance_data)
    
    print(f"🔧 Parameter Optimization Complete")
    print(f"   Optimization ID: {optimization_result['optimization_id']}")
    print(f"   New Effectiveness Score: {optimization_result['new_effectiveness_score']:.3f}")
    print()
    
    if optimization_result["parameter_adjustments"]:
        print("⚙️ Parameter Adjustments:")
        for param, adjustment in optimization_result["parameter_adjustments"].items():
            print(f"   {param}:")
            print(f"     Old Value: {adjustment['old_value']:.6f}")
            print(f"     New Value: {adjustment['new_value']:.6f}")
            print(f"     Reason: {adjustment['reason']}")
    else:
        print("✅ No parameter adjustments needed - current parameters are optimal")
    print()
    
    print("📈 Improvement Metrics:")
    improvement = optimization_result["improvement_metrics"]
    print(f"   Overall Improvement: {improvement['overall_improvement']:.3f}")
    print(f"   Expected Performance Gain: {improvement['expected_performance_gain']:.3f}")
    
    if improvement["parameter_improvements"]:
        print("   Parameter-Specific Improvements:")
        for param, imp in improvement["parameter_improvements"].items():
            print(f"     {param}: {imp:.3f}")
    print()
    
    print("\n📊 Step 7: System Statistics and Summary")
    print("-" * 40)
    
    # Display system statistics
    total_optimizations = len(system.learning_optimizations)
    total_metrics_tracked = len(system.learning_metrics_history)
    total_adaptive_params = len(system.adaptive_parameters)
    
    print(f"📊 Learning Optimization System Statistics:")
    print(f"   Total Optimizations Created: {total_optimizations}")
    print(f"   Metrics Types Tracked: {total_metrics_tracked}")
    print(f"   Adaptive Parameter Sets: {total_adaptive_params}")
    print()
    
    if system.learning_optimizations:
        effectiveness_scores = [opt.effectiveness_score for opt in system.learning_optimizations.values()]
        avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
        
        print(f"🎯 Optimization Effectiveness:")
        print(f"   Average Effectiveness Score: {avg_effectiveness:.3f}")
        print(f"   Best Performing Optimization: {max(effectiveness_scores):.3f}")
        print(f"   Optimization Strategies Used: {set(opt.optimization_strategy for opt in system.learning_optimizations.values())}")
    print()
    
    if system.learning_metrics_history:
        print("📈 Metrics Tracking Summary:")
        for metric_name, metrics_deque in system.learning_metrics_history.items():
            metrics_list = list(metrics_deque)
            if metrics_list:
                latest_value = metrics_list[-1].value
                trend = "improving" if len(metrics_list) > 1 and metrics_list[-1].value > metrics_list[0].value else "stable"
                print(f"   {metric_name}: {latest_value:.3f} ({len(metrics_list)} measurements, {trend})")
    print()
    
    print("\n🎉 Demo Complete!")
    print("=" * 50)
    print("The Learning Optimization System successfully demonstrated:")
    print("✅ Continuous learning process optimization")
    print("✅ Comprehensive learning effectiveness measurement")
    print("✅ Adaptive learning implementation with feedback integration")
    print("✅ Innovation process enhancement with bottleneck identification")
    print("✅ Dynamic parameter optimization based on performance data")
    print("✅ Real-time metrics tracking and trend analysis")
    print()
    print("🚀 Key Capabilities Showcased:")
    print("• Multi-dimensional learning optimization across various metrics")
    print("• Intelligent adaptation based on performance feedback")
    print("• Process enhancement with prioritized improvement strategies")
    print("• Dynamic parameter tuning for continuous improvement")
    print("• Comprehensive effectiveness measurement and trend analysis")
    print("• Scalable architecture for complex learning environments")


if __name__ == "__main__":
    asyncio.run(demo_learning_optimization())