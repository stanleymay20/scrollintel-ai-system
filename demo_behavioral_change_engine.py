"""
Demo script for Behavioral Change Engine

This script demonstrates the comprehensive behavioral change capabilities
including behavior analysis, modification strategies, and habit formation.
"""

import asyncio
from datetime import datetime
from uuid import uuid4

from scrollintel.engines.behavioral_analysis_engine import BehavioralAnalysisEngine
from scrollintel.engines.behavior_modification_engine import BehaviorModificationEngine
from scrollintel.engines.habit_formation_engine import HabitFormationEngine
from scrollintel.models.behavioral_analysis_models import (
    BehaviorObservation, BehaviorType
)
from scrollintel.models.habit_formation_models import HabitType


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


async def demo_behavioral_analysis():
    """Demonstrate behavioral analysis capabilities"""
    print_section("BEHAVIORAL ANALYSIS ENGINE DEMO")
    
    # Initialize the engine
    analysis_engine = BehavioralAnalysisEngine()
    
    # Create sample behavior observations
    observations = [
        BehaviorObservation(
            id=str(uuid4()),
            observer_id="manager_alice",
            observed_behavior="Team members actively collaborate on problem-solving",
            behavior_type=BehaviorType.COLLABORATION,
            context={"triggers": ["complex_problem"], "outcomes": ["solution_found", "team_bonding"]},
            participants=["john", "sarah", "mike", "lisa"],
            timestamp=datetime.now(),
            impact_assessment="Significantly improved problem resolution speed",
            cultural_relevance=0.9
        ),
        BehaviorObservation(
            id=str(uuid4()),
            observer_id="manager_alice",
            observed_behavior="Team members actively collaborate on problem-solving",
            behavior_type=BehaviorType.COLLABORATION,
            context={"triggers": ["complex_problem"], "outcomes": ["solution_found", "team_bonding"]},
            participants=["john", "sarah", "emma", "david"],
            timestamp=datetime.now(),
            impact_assessment="Significantly improved problem resolution speed",
            cultural_relevance=0.9
        ),
        BehaviorObservation(
            id=str(uuid4()),
            observer_id="manager_alice",
            observed_behavior="Team members actively collaborate on problem-solving",
            behavior_type=BehaviorType.COLLABORATION,
            context={"triggers": ["complex_problem"], "outcomes": ["solution_found", "team_bonding"]},
            participants=["sarah", "mike", "emma", "alex"],
            timestamp=datetime.now(),
            impact_assessment="Significantly improved problem resolution speed",
            cultural_relevance=0.9
        ),
        BehaviorObservation(
            id=str(uuid4()),
            observer_id="team_lead_bob",
            observed_behavior="Clear and timely communication of project updates",
            behavior_type=BehaviorType.COMMUNICATION,
            context={"triggers": ["milestone_reached"], "outcomes": ["stakeholder_alignment"]},
            participants=["project_manager", "stakeholders"],
            timestamp=datetime.now(),
            impact_assessment="Improved stakeholder confidence and alignment",
            cultural_relevance=0.8
        ),
        BehaviorObservation(
            id=str(uuid4()),
            observer_id="team_lead_bob",
            observed_behavior="Clear and timely communication of project updates",
            behavior_type=BehaviorType.COMMUNICATION,
            context={"triggers": ["milestone_reached"], "outcomes": ["stakeholder_alignment"]},
            participants=["project_manager", "stakeholders"],
            timestamp=datetime.now(),
            impact_assessment="Improved stakeholder confidence and alignment",
            cultural_relevance=0.8
        ),
        BehaviorObservation(
            id=str(uuid4()),
            observer_id="team_lead_bob",
            observed_behavior="Clear and timely communication of project updates",
            behavior_type=BehaviorType.COMMUNICATION,
            context={"triggers": ["milestone_reached"], "outcomes": ["stakeholder_alignment"]},
            participants=["project_manager", "stakeholders"],
            timestamp=datetime.now(),
            impact_assessment="Improved stakeholder confidence and alignment",
            cultural_relevance=0.8
        ),
        BehaviorObservation(
            id=str(uuid4()),
            observer_id="hr_manager",
            observed_behavior="Proactive learning and skill development",
            behavior_type=BehaviorType.LEARNING,
            context={"triggers": ["skill_gap_identified"], "outcomes": ["competency_improved"]},
            participants=["individual_contributors"],
            timestamp=datetime.now(),
            impact_assessment="Enhanced team capabilities and adaptability",
            cultural_relevance=0.85
        ),
        BehaviorObservation(
            id=str(uuid4()),
            observer_id="hr_manager",
            observed_behavior="Proactive learning and skill development",
            behavior_type=BehaviorType.LEARNING,
            context={"triggers": ["skill_gap_identified"], "outcomes": ["competency_improved"]},
            participants=["individual_contributors"],
            timestamp=datetime.now(),
            impact_assessment="Enhanced team capabilities and adaptability",
            cultural_relevance=0.85
        ),
        BehaviorObservation(
            id=str(uuid4()),
            observer_id="hr_manager",
            observed_behavior="Proactive learning and skill development",
            behavior_type=BehaviorType.LEARNING,
            context={"triggers": ["skill_gap_identified"], "outcomes": ["competency_improved"]},
            participants=["individual_contributors"],
            timestamp=datetime.now(),
            impact_assessment="Enhanced team capabilities and adaptability",
            cultural_relevance=0.85
        )
    ]
    
    # Define cultural values
    cultural_values = ["collaboration", "transparency", "continuous_learning", "innovation", "excellence"]
    
    print_subsection("Analyzing Organizational Behaviors")
    print(f"📊 Processing {len(observations)} behavior observations...")
    print(f"🎯 Cultural values: {', '.join(cultural_values)}")
    
    # Perform behavioral analysis
    analysis_result = analysis_engine.analyze_organizational_behaviors(
        organization_id="demo_org_001",
        observations=observations,
        cultural_values=cultural_values,
        analyst="ScrollIntel Demo"
    )
    
    print(f"\n✅ Analysis completed!")
    print(f"   • Analysis ID: {analysis_result.analysis_id}")
    print(f"   • Behavior patterns identified: {len(analysis_result.behavior_patterns)}")
    print(f"   • Behavioral norms assessed: {len(analysis_result.behavioral_norms)}")
    print(f"   • Culture alignments analyzed: {len(analysis_result.culture_alignments)}")
    print(f"   • Overall health score: {analysis_result.overall_health_score:.2f}")
    
    print_subsection("Behavior Patterns Identified")
    for i, pattern in enumerate(analysis_result.behavior_patterns, 1):
        print(f"{i}. {pattern.name}")
        print(f"   Type: {pattern.behavior_type.value}")
        print(f"   Frequency: {pattern.frequency.value}")
        print(f"   Strength: {pattern.strength:.2f}")
        print(f"   Participants: {len(pattern.participants)} people")
        print(f"   Triggers: {', '.join(pattern.triggers[:2])}...")
    
    print_subsection("Key Insights")
    for i, insight in enumerate(analysis_result.key_insights, 1):
        print(f"{i}. {insight}")
    
    print_subsection("Recommendations")
    for i, recommendation in enumerate(analysis_result.recommendations, 1):
        print(f"{i}. {recommendation}")
    
    # Calculate behavioral metrics
    metrics = analysis_engine.calculate_behavior_metrics(analysis_result)
    
    print_subsection("Behavioral Metrics")
    print(f"📈 Behavior diversity index: {metrics.behavior_diversity_index:.2f}")
    print(f"📊 Norm compliance average: {metrics.norm_compliance_average:.2f}")
    print(f"🎯 Culture alignment score: {metrics.culture_alignment_score:.2f}")
    print(f"🔄 Behavior consistency index: {metrics.behavior_consistency_index:.2f}")
    print(f"✅ Positive behavior ratio: {metrics.positive_behavior_ratio:.2f}")
    
    return analysis_result


async def demo_behavior_modification():
    """Demonstrate behavior modification capabilities"""
    print_section("BEHAVIOR MODIFICATION ENGINE DEMO")
    
    # Initialize the engine
    modification_engine = BehaviorModificationEngine()
    
    print_subsection("Developing Behavior Modification Strategy")
    
    # Define modification parameters
    target_behavior = "improve cross-team communication and collaboration"
    current_state = {
        'complexity': 'medium',
        'participant_count': 25,
        'strength': 0.4,
        'frequency': 0.3,
        'quality': 0.5
    }
    desired_outcome = "achieve seamless cross-team collaboration with 90% satisfaction"
    constraints = {
        'urgency': 'high',
        'budget': 'medium',
        'timeline_weeks': 16
    }
    stakeholders = ["team_leads", "project_managers", "hr_department", "executive_team"]
    
    print(f"🎯 Target behavior: {target_behavior}")
    print(f"📊 Current state: {current_state}")
    print(f"🏆 Desired outcome: {desired_outcome}")
    print(f"⚡ Constraints: {constraints}")
    print(f"👥 Stakeholders: {', '.join(stakeholders)}")
    
    # Develop modification strategy
    strategy = modification_engine.develop_modification_strategy(
        target_behavior=target_behavior,
        current_state=current_state,
        desired_outcome=desired_outcome,
        constraints=constraints,
        stakeholders=stakeholders
    )
    
    print(f"\n✅ Strategy developed!")
    print(f"   • Strategy ID: {strategy.id}")
    print(f"   • Timeline: {strategy.timeline_weeks} weeks")
    print(f"   • Techniques: {len(strategy.techniques)}")
    print(f"   • Success criteria: {len(strategy.success_criteria)}")
    print(f"   • Resources required: {len(strategy.resources_required)}")
    
    print_subsection("Modification Techniques Selected")
    for i, technique in enumerate(strategy.techniques, 1):
        print(f"{i}. {technique.value.replace('_', ' ').title()}")
    
    print_subsection("Success Criteria")
    for i, criterion in enumerate(strategy.success_criteria, 1):
        print(f"{i}. {criterion}")
    
    print_subsection("Risk Factors & Mitigation")
    print("🚨 Risk Factors:")
    for i, risk in enumerate(strategy.risk_factors, 1):
        print(f"   {i}. {risk}")
    
    print("\n🛡️ Mitigation Strategies:")
    for i, mitigation in enumerate(strategy.mitigation_strategies, 1):
        print(f"   {i}. {mitigation}")
    
    # Create interventions
    print_subsection("Creating Modification Interventions")
    participants = ["alice", "bob", "charlie", "diana", "eve", "frank"]
    facilitators = ["senior_coach", "change_manager"]
    
    interventions = modification_engine.create_modification_interventions(
        strategy=strategy,
        participants=participants,
        facilitators=facilitators
    )
    
    print(f"✅ Created {len(interventions)} interventions")
    
    for i, intervention in enumerate(interventions, 1):
        print(f"\n{i}. {intervention.intervention_name}")
        print(f"   Technique: {intervention.technique.value}")
        print(f"   Duration: {intervention.duration_days} days")
        print(f"   Frequency: {intervention.frequency}")
        print(f"   Participants: {len(intervention.target_participants)}")
        print(f"   Steps: {len(intervention.implementation_steps)}")
    
    # Track progress for some participants
    print_subsection("Tracking Behavior Change Progress")
    
    progress_results = []
    for i, participant in enumerate(participants[:3]):
        # Simulate different progress levels
        current_measurement = 0.3 + (i * 0.2)  # 0.3, 0.5, 0.7
        
        progress = modification_engine.track_behavior_change_progress(
            strategy_id=strategy.id,
            participant_id=participant,
            current_measurement=current_measurement,
            baseline_measurement=0.3,
            target_measurement=0.8
        )
        progress_results.append(progress)
        
        print(f"\n👤 {participant.title()}")
        print(f"   Progress level: {progress.progress_level.value}")
        print(f"   Current measurement: {progress.current_measurement:.2f}")
        print(f"   Improvement rate: {progress.improvement_rate:.2f}")
        print(f"   Milestones achieved: {len(progress.milestones_achieved)}")
        print(f"   Next review: {progress.next_review_date.strftime('%Y-%m-%d')}")
    
    # Optimize strategy
    print_subsection("Strategy Optimization")
    
    optimization = modification_engine.optimize_modification_strategy(
        strategy_id=strategy.id,
        progress_data=progress_results,
        effectiveness_data={"participant_feedback": 0.75, "facilitator_assessment": 0.8}
    )
    
    print(f"📊 Current effectiveness: {optimization.current_effectiveness:.2f}")
    print(f"📈 Expected improvement: {optimization.expected_improvement:.2f}")
    print(f"⚡ Implementation effort: {optimization.implementation_effort}")
    print(f"⚠️ Risk level: {optimization.risk_level}")
    print(f"🎯 Priority score: {optimization.priority_score:.2f}")
    
    print("\n🔧 Optimization Opportunities:")
    for i, opportunity in enumerate(optimization.optimization_opportunities, 1):
        print(f"   {i}. {opportunity}")
    
    print("\n💡 Recommended Adjustments:")
    for i, adjustment in enumerate(optimization.recommended_adjustments[:3], 1):
        print(f"   {i}. {adjustment}")
    
    return strategy


async def demo_habit_formation():
    """Demonstrate habit formation capabilities"""
    print_section("HABIT FORMATION ENGINE DEMO")
    
    # Initialize the engine
    habit_engine = HabitFormationEngine()
    
    print_subsection("Designing Organizational Habits")
    
    # Design multiple habits
    habits = []
    
    # Habit 1: Daily Standup
    habit1 = habit_engine.design_organizational_habit(
        name="Daily Team Standup",
        description="15-minute daily synchronization meeting for team alignment",
        habit_type=HabitType.COMMUNICATION,
        target_behavior="improve daily team communication and coordination",
        participants=["team_alpha", "alice", "bob", "charlie", "diana"],
        cultural_values=["transparency", "collaboration", "accountability"],
        business_objectives=["improve team productivity", "reduce miscommunication"]
    )
    habits.append(habit1)
    
    # Habit 2: Weekly Learning Session
    habit2 = habit_engine.design_organizational_habit(
        name="Weekly Learning Hour",
        description="Dedicated time for team learning and skill development",
        habit_type=HabitType.LEARNING,
        target_behavior="foster continuous learning and knowledge sharing",
        participants=["team_alpha", "eve", "frank", "grace", "henry"],
        cultural_values=["growth", "curiosity", "excellence"],
        business_objectives=["build team capabilities", "stay competitive"]
    )
    habits.append(habit2)
    
    # Habit 3: Innovation Friday
    habit3 = habit_engine.design_organizational_habit(
        name="Innovation Friday",
        description="Monthly innovation session for creative problem-solving",
        habit_type=HabitType.INNOVATION,
        target_behavior="encourage creative thinking and innovation",
        participants=["team_alpha", "alice", "eve", "grace", "ivan"],
        cultural_values=["creativity", "experimentation", "innovation"],
        business_objectives=["drive innovation", "improve processes"]
    )
    habits.append(habit3)
    
    print(f"✅ Designed {len(habits)} organizational habits")
    
    for i, habit in enumerate(habits, 1):
        print(f"\n{i}. {habit.name}")
        print(f"   Type: {habit.habit_type.value}")
        print(f"   Frequency: {habit.frequency.value}")
        print(f"   Duration: {habit.duration_minutes} minutes")
        print(f"   Participants: {len(habit.participants)}")
        print(f"   Cultural alignment: {habit.cultural_alignment:.2f}")
        print(f"   Triggers: {', '.join(habit.trigger_conditions[:2])}...")
    
    # Create formation strategies
    print_subsection("Creating Habit Formation Strategies")
    
    strategies = []
    for habit in habits:
        organizational_context = {
            'change_readiness': 'high',
            'resource_availability': 'medium',
            'leadership_support': 'strong'
        }
        
        strategy = habit_engine.create_habit_formation_strategy(
            habit=habit,
            organizational_context=organizational_context
        )
        strategies.append(strategy)
        
        print(f"\n📋 Strategy for {habit.name}")
        print(f"   Timeline: {strategy.timeline_weeks} weeks")
        print(f"   Phases: {len(strategy.formation_phases)}")
        print(f"   Milestones: {len(strategy.key_milestones)}")
        print(f"   Success metrics: {len(strategy.success_metrics)}")
    
    # Implement sustainability mechanisms
    print_subsection("Implementing Sustainability Mechanisms")
    
    sustainability_mechanisms = []
    for habit in habits:
        formation_progress = {
            'participation_rate': 0.85,
            'consistency_rate': 0.78,
            'weeks_since_formation': 8
        }
        
        sustainability = habit_engine.implement_habit_sustainability_mechanisms(
            habit=habit,
            formation_progress=formation_progress
        )
        sustainability_mechanisms.append(sustainability)
        
        print(f"\n🔄 Sustainability for {habit.name}")
        print(f"   Level: {sustainability.sustainability_level.value}")
        print(f"   Score: {sustainability.sustainability_score:.2f}")
        print(f"   Cultural integration: {sustainability.cultural_integration:.2f}")
        print(f"   Reinforcement systems: {len(sustainability.reinforcement_systems)}")
        print(f"   Monitoring mechanisms: {len(sustainability.monitoring_mechanisms)}")
    
    # Track habit progress
    print_subsection("Tracking Habit Progress")
    
    participants = ["alice", "bob", "charlie"]
    for habit in habits[:2]:  # Track first 2 habits
        print(f"\n📊 Progress for {habit.name}")
        
        for participant in participants:
            # Simulate different progress levels
            execution_count = 4 if participant == "alice" else 3 if participant == "bob" else 5
            target_count = 5
            quality_score = 0.8 if participant == "alice" else 0.7 if participant == "bob" else 0.9
            engagement_level = 0.9 if participant == "alice" else 0.75 if participant == "bob" else 0.85
            
            progress = habit_engine.track_habit_progress(
                habit_id=habit.id,
                participant_id=participant,
                tracking_period="2024-W01",
                execution_count=execution_count,
                target_count=target_count,
                quality_score=quality_score,
                engagement_level=engagement_level
            )
            
            print(f"   👤 {participant.title()}: {progress.consistency_rate:.1%} consistency, "
                  f"{progress.quality_score:.1f} quality, {progress.engagement_level:.1%} engagement")
    
    # Calculate comprehensive metrics
    print_subsection("Habit Formation Metrics")
    
    metrics = habit_engine.calculate_habit_formation_metrics("team_alpha")
    
    print(f"📊 Comprehensive Metrics for Organization")
    print(f"   • Total habits designed: {metrics.total_habits_designed}")
    print(f"   • Habits in formation: {metrics.habits_in_formation}")
    print(f"   • Habits established: {metrics.habits_established}")
    print(f"   • Average formation time: {metrics.average_formation_time_weeks:.1f} weeks")
    print(f"   • Overall success rate: {metrics.overall_success_rate:.1%}")
    print(f"   • Participant engagement: {metrics.participant_engagement_average:.1%}")
    print(f"   • Sustainability index: {metrics.sustainability_index:.2f}")
    print(f"   • Cultural integration: {metrics.cultural_integration_score:.2f}")
    print(f"   • ROI achieved: {metrics.roi_achieved:.1f}x")
    
    return habits


async def demo_integrated_behavioral_change():
    """Demonstrate integrated behavioral change workflow"""
    print_section("INTEGRATED BEHAVIORAL CHANGE WORKFLOW")
    
    print_subsection("Workflow Overview")
    print("🔄 This demo shows how the three engines work together:")
    print("   1. Behavioral Analysis → Identify current patterns")
    print("   2. Behavior Modification → Create change strategies")
    print("   3. Habit Formation → Establish sustainable practices")
    
    # Step 1: Analyze current behaviors
    print_subsection("Step 1: Behavioral Analysis")
    analysis_result = await demo_behavioral_analysis()
    
    # Step 2: Develop modification strategies based on analysis
    print_subsection("Step 2: Behavior Modification Strategy")
    
    # Use insights from analysis to inform modification
    key_behavior_to_modify = "enhance collaboration patterns identified in analysis"
    print(f"🎯 Targeting behavior based on analysis: {key_behavior_to_modify}")
    
    modification_strategy = await demo_behavior_modification()
    
    # Step 3: Create sustainable habits
    print_subsection("Step 3: Habit Formation for Sustainability")
    
    print("🌱 Creating habits to sustain the behavioral changes:")
    habits = await demo_habit_formation()
    
    # Integration summary
    print_subsection("Integration Summary")
    print("✅ Behavioral Change Engine Integration Complete!")
    print(f"   • Analyzed {len(analysis_result.behavior_patterns)} behavior patterns")
    print(f"   • Developed modification strategy with {len(modification_strategy.techniques)} techniques")
    print(f"   • Created {len(habits)} sustainable organizational habits")
    print(f"   • Overall health score improved from baseline")
    print(f"   • Cultural alignment strengthened across all interventions")
    
    print("\n🎯 Key Success Factors:")
    print("   1. Data-driven analysis of current behavioral patterns")
    print("   2. Systematic modification strategies with multiple techniques")
    print("   3. Sustainable habit formation with reinforcement mechanisms")
    print("   4. Continuous monitoring and optimization")
    print("   5. Cultural alignment throughout the transformation process")
    
    print("\n🚀 Next Steps:")
    print("   • Implement the modification interventions")
    print("   • Begin habit formation strategies")
    print("   • Monitor progress and adjust approaches")
    print("   • Scale successful patterns across the organization")
    print("   • Measure long-term cultural transformation impact")


async def main():
    """Main demo function"""
    print("🎭 ScrollIntel Behavioral Change Engine Demo")
    print("=" * 60)
    print("This demo showcases comprehensive behavioral transformation capabilities")
    print("including analysis, modification, and sustainable habit formation.")
    
    try:
        # Run individual engine demos
        await demo_behavioral_analysis()
        await demo_behavior_modification()
        await demo_habit_formation()
        
        # Run integrated workflow demo
        await demo_integrated_behavioral_change()
        
        print_section("DEMO COMPLETE")
        print("🎉 Behavioral Change Engine demo completed successfully!")
        print("📈 The system demonstrated comprehensive capabilities for:")
        print("   • Analyzing organizational behavior patterns")
        print("   • Developing systematic modification strategies")
        print("   • Creating sustainable organizational habits")
        print("   • Integrating all components for cultural transformation")
        
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())