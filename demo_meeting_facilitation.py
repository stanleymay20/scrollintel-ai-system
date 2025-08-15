"""
Demo script for Meeting Facilitation Engine
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from scrollintel.engines.meeting_facilitation_engine import (
    MeetingFacilitationEngine, FacilitationPhase, InteractionType,
    FacilitationGuidance, MeetingFlow, EngagementMetrics,
    FacilitationIntervention, MeetingOutcome
)
from scrollintel.models.meeting_preparation_models import (
    MeetingPreparation, BoardMember, MeetingObjective, AgendaItem,
    MeetingType, PreparationStatus, ContentType
)


def create_sample_meeting_preparation() -> MeetingPreparation:
    """Create sample meeting preparation for demonstration"""
    board_members = [
        BoardMember(
            id="member_001",
            name="Elizabeth Warren",
            role="Board Chair",
            expertise_areas=["Finance", "Strategy", "Governance"],
            communication_preferences={
                "style": "direct",
                "detail_level": "high",
                "preferred_format": "data_driven"
            },
            influence_level=0.95,
            typical_concerns=[
                "Financial performance",
                "Risk management",
                "Shareholder value"
            ],
            decision_patterns={
                "speed": "deliberate",
                "style": "consensus_building",
                "risk_tolerance": "moderate"
            }
        ),
        BoardMember(
            id="member_002",
            name="Dr. Satya Nadella",
            role="Technology Committee Chair",
            expertise_areas=["Technology", "Innovation", "Digital Transformation"],
            communication_preferences={
                "style": "visionary",
                "detail_level": "medium",
                "preferred_format": "strategic_narrative"
            },
            influence_level=0.90,
            typical_concerns=[
                "Technology strategy",
                "Innovation pipeline",
                "Digital transformation"
            ],
            decision_patterns={
                "speed": "quick",
                "style": "data_driven",
                "risk_tolerance": "high"
            }
        ),
        BoardMember(
            id="member_003",
            name="Mary Barra",
            role="Audit Committee Chair",
            expertise_areas=["Operations", "Manufacturing", "Quality"],
            communication_preferences={
                "style": "operational",
                "detail_level": "high",
                "preferred_format": "process_focused"
            },
            influence_level=0.85,
            typical_concerns=[
                "Operational efficiency",
                "Quality control",
                "Process improvement"
            ],
            decision_patterns={
                "speed": "thorough",
                "style": "risk_averse",
                "risk_tolerance": "low"
            }
        )
    ]
    
    objectives = [
        MeetingObjective(
            id="obj_001",
            title="AI Technology Investment Strategy",
            description="Present and approve comprehensive AI technology investment strategy",
            priority=1,
            success_criteria=[
                "Technology strategy clearly articulated",
                "Investment approval obtained",
                "Implementation timeline agreed"
            ],
            required_decisions=[
                "Approve $50M AI technology investment",
                "Approve hiring plan for AI talent",
                "Set implementation timeline"
            ],
            stakeholders=["CTO", "Board", "Technology Committee"]
        ),
        MeetingObjective(
            id="obj_002",
            title="Q4 Financial Performance Review",
            description="Review Q4 financial results and approve annual budget",
            priority=2,
            success_criteria=[
                "Financial results clearly communicated",
                "Budget approved by board",
                "Key performance metrics discussed"
            ],
            required_decisions=[
                "Approve 2025 annual budget",
                "Approve dividend policy"
            ],
            stakeholders=["CFO", "Board", "Shareholders"]
        )
    ]
    
    agenda_items = [
        AgendaItem(
            id="item_001",
            title="Meeting Opening & Welcome",
            description="Welcome board members and review agenda",
            presenter="Board Chair",
            duration_minutes=10,
            content_type=ContentType.PRESENTATION,
            objectives=[],
            materials_required=["Agenda", "Previous minutes"],
            key_messages=["Meeting officially commenced", "Agenda confirmed"],
            anticipated_questions=[],
            decision_required=True,
            priority=1
        ),
        AgendaItem(
            id="item_002",
            title="AI Technology Investment Strategy",
            description="Comprehensive presentation of AI investment strategy",
            presenter="CTO/ScrollIntel",
            duration_minutes=45,
            content_type=ContentType.STRATEGIC_UPDATE,
            objectives=["obj_001"],
            materials_required=[
                "AI strategy presentation",
                "Investment analysis",
                "ROI projections",
                "Implementation timeline"
            ],
            key_messages=[
                "AI technology is critical for competitive advantage",
                "Investment will drive significant business value",
                "Implementation plan is comprehensive and achievable"
            ],
            anticipated_questions=[
                "What are the key risks of this investment?",
                "How does this compare to competitor investments?",
                "What happens if AI technology doesn't deliver expected ROI?"
            ],
            decision_required=True,
            priority=1
        ),
        AgendaItem(
            id="item_003",
            title="Q4 Financial Performance Review",
            description="Detailed review of Q4 financial results",
            presenter="CFO",
            duration_minutes=30,
            content_type=ContentType.FINANCIAL_REPORT,
            objectives=["obj_002"],
            materials_required=[
                "Q4 financial statements",
                "Performance analysis",
                "Budget proposal"
            ],
            key_messages=[
                "Strong Q4 financial performance",
                "Annual targets exceeded",
                "2025 budget is conservative and achievable"
            ],
            anticipated_questions=[
                "What drove the strong Q4 performance?",
                "Are these results sustainable?",
                "How does 2025 budget compare to industry benchmarks?"
            ],
            decision_required=True,
            priority=2
        ),
        AgendaItem(
            id="item_004",
            title="Next Steps & Adjournment",
            description="Summary of decisions and next steps",
            presenter="Board Chair",
            duration_minutes=15,
            content_type=ContentType.PRESENTATION,
            objectives=[],
            materials_required=[],
            key_messages=[
                "Clear decisions made",
                "Action items assigned",
                "Next meeting scheduled"
            ],
            anticipated_questions=[],
            decision_required=False,
            priority=999
        )
    ]
    
    return MeetingPreparation(
        id="prep_demo_facilitation",
        meeting_id="board_meeting_demo_facilitation",
        meeting_type=MeetingType.BOARD_MEETING,
        meeting_date=datetime.now() + timedelta(hours=2),
        board_members=board_members,
        objectives=objectives,
        agenda_items=agenda_items,
        content_materials=[],
        preparation_tasks=[],
        success_metrics=[],
        status=PreparationStatus.COMPLETED,
        created_at=datetime.now() - timedelta(hours=1),
        updated_at=datetime.now()
    )


async def demonstrate_meeting_facilitation():
    """Demonstrate comprehensive meeting facilitation capabilities"""
    print("🎯 ScrollIntel Board Executive Mastery - Meeting Facilitation Engine Demo")
    print("=" * 80)
    
    # Initialize the meeting facilitation engine
    engine = MeetingFacilitationEngine()
    
    # Create sample meeting preparation
    preparation = create_sample_meeting_preparation()
    
    print(f"\n📋 Meeting Setup:")
    print(f"   Meeting ID: {preparation.meeting_id}")
    print(f"   Type: {preparation.meeting_type.value}")
    print(f"   Board Members: {len(preparation.board_members)}")
    print(f"   Agenda Items: {len(preparation.agenda_items)}")
    print(f"   Total Duration: {sum(item.duration_minutes for item in preparation.agenda_items)} minutes")
    
    # Step 1: Generate comprehensive facilitation guidance
    print(f"\n🎯 Step 1: Generating Facilitation Guidance")
    print("-" * 50)
    
    guidance_list = engine.generate_facilitation_guidance(
        preparation, FacilitationPhase.PRE_MEETING
    )
    
    print(f"✅ Generated {len(guidance_list)} facilitation guidance items!")
    
    # Display key guidance items
    for i, guidance in enumerate(guidance_list[:3], 1):  # Show first 3
        print(f"\n📖 Guidance {i}: {guidance.title}")
        print(f"   Phase: {guidance.phase.value}")
        print(f"   Type: {guidance.guidance_type}")
        print(f"   Agenda Item: {guidance.agenda_item_id}")
        
        print(f"   Key Actions:")
        for action in guidance.key_actions[:3]:  # Show first 3 actions
            print(f"   • {action}")
        
        print(f"   Engagement Strategies:")
        for strategy in guidance.engagement_strategies[:2]:  # Show first 2 strategies
            print(f"   • {strategy}")
        
        print(f"   Success Indicators:")
        for indicator in guidance.success_indicators[:2]:  # Show first 2 indicators
            print(f"   • {indicator}")
    
    if len(guidance_list) > 3:
        print(f"\n   ... and {len(guidance_list) - 3} more guidance items")
    
    # Step 2: Simulate meeting flow monitoring
    print(f"\n⏱️  Step 2: Monitoring Meeting Flow (Real-time Simulation)")
    print("-" * 50)
    
    # Simulate different meeting phases
    meeting_phases = [
        ("Opening", "item_001", 5),
        ("AI Strategy Discussion", "item_002", 25),
        ("Financial Review", "item_003", 55),
        ("Closing", "item_004", 85)
    ]
    
    for phase_name, current_item, elapsed_minutes in meeting_phases:
        current_time = preparation.created_at + timedelta(minutes=elapsed_minutes)
        
        meeting_flow = engine.monitor_meeting_flow(
            preparation, current_time, current_item
        )
        
        print(f"\n🔄 {phase_name} Phase (Elapsed: {elapsed_minutes} min)")
        print(f"   Current Phase: {meeting_flow.current_phase.value}")
        print(f"   Flow Status: {meeting_flow.flow_status}")
        print(f"   Engagement Level: {meeting_flow.engagement_level:.1%}")
        print(f"   Remaining Time: {meeting_flow.remaining_time} minutes")
        
        if meeting_flow.next_actions:
            print(f"   Next Actions:")
            for action in meeting_flow.next_actions:
                print(f"   • {action}")
        
        if meeting_flow.flow_adjustments:
            print(f"   Flow Adjustments:")
            for adjustment in meeting_flow.flow_adjustments:
                print(f"   • {adjustment}")
    
    # Step 3: Track engagement metrics
    print(f"\n📊 Step 3: Tracking Engagement Metrics")
    print("-" * 50)
    
    current_time = preparation.created_at + timedelta(minutes=45)
    engagement_metrics = engine.track_engagement_metrics(
        preparation.meeting_id,
        preparation.board_members,
        current_time
    )
    
    print(f"✅ Engagement metrics tracked successfully!")
    print(f"   Overall Engagement: {engagement_metrics.overall_engagement:.1%}")
    print(f"   Participation Balance: {engagement_metrics.participation_balance:.1%}")
    print(f"   Discussion Quality: {engagement_metrics.discussion_quality:.1%}")
    print(f"   Decision Momentum: {engagement_metrics.decision_momentum:.1%}")
    print(f"   Energy Level: {engagement_metrics.energy_level:.1%}")
    
    print(f"\n👥 Individual Engagement Levels:")
    for member_id, engagement in engagement_metrics.individual_engagement.items():
        member = next(m for m in preparation.board_members if m.id == member_id)
        print(f"   • {member.name}: {engagement:.1%}")
    
    # Step 4: Suggest facilitation interventions
    print(f"\n🚨 Step 4: Suggesting Facilitation Interventions")
    print("-" * 50)
    
    # Create a scenario with some challenges
    challenging_flow = MeetingFlow(
        id="flow_challenging",
        meeting_preparation_id=preparation.id,
        current_phase=FacilitationPhase.DISCUSSION,
        current_agenda_item="item_002",
        elapsed_time=60,
        remaining_time=30,
        flow_status="behind_schedule",
        engagement_level=0.55,  # Low engagement
        decision_progress={"item_002": 0.3},
        next_actions=["Focus on key decisions"],
        flow_adjustments=["Accelerate discussion"]
    )
    
    challenging_metrics = EngagementMetrics(
        id="metrics_challenging",
        meeting_id=preparation.meeting_id,
        timestamp=current_time,
        overall_engagement=0.55,  # Low engagement
        individual_engagement={
            "member_001": 0.8,  # Chair engaged
            "member_002": 0.6,  # Moderate engagement
            "member_003": 0.3   # Low engagement - trigger intervention
        },
        participation_balance=0.4,  # Imbalanced - trigger intervention
        discussion_quality=0.6,
        decision_momentum=0.4,  # Low momentum - trigger intervention
        energy_level=0.5
    )
    
    interventions = engine.suggest_facilitation_interventions(
        challenging_flow, challenging_metrics
    )
    
    print(f"✅ Generated {len(interventions)} facilitation interventions!")
    
    for i, intervention in enumerate(interventions, 1):
        print(f"\n🔧 Intervention {i}: {intervention.intervention_type.replace('_', ' ').title()}")
        print(f"   Trigger: {intervention.trigger}")
        print(f"   Description: {intervention.description}")
        print(f"   Expected Outcome: {intervention.expected_outcome}")
        
        print(f"   Actions to Take:")
        for action in intervention.actions_taken:
            print(f"   • {action}")
    
    # Step 5: Track meeting outcomes
    print(f"\n📈 Step 5: Tracking Meeting Outcomes")
    print("-" * 50)
    
    completed_items = ["item_001", "item_002", "item_003"]
    outcomes = engine.track_meeting_outcomes(preparation, completed_items)
    
    print(f"✅ Tracked {len(outcomes)} meeting outcomes!")
    
    for outcome in outcomes:
        agenda_item = next(item for item in preparation.agenda_items if item.id == outcome.agenda_item_id)
        print(f"\n📋 Outcome: {agenda_item.title}")
        print(f"   Type: {outcome.outcome_type}")
        print(f"   Success Score: {outcome.success_score:.1%}")
        print(f"   Follow-up Required: {'Yes' if outcome.follow_up_required else 'No'}")
        
        if outcome.decisions_made:
            print(f"   Decisions Made:")
            for decision in outcome.decisions_made:
                print(f"   • {decision}")
        
        if outcome.action_items:
            print(f"   Action Items:")
            for action_item in outcome.action_items[:2]:  # Show first 2
                print(f"   • {action_item['description']} (Due: {action_item['due_date'][:10]})")
        
        print(f"   Stakeholder Satisfaction:")
        for member_id, satisfaction in outcome.stakeholder_satisfaction.items():
            member = next(m for m in preparation.board_members if m.id == member_id)
            print(f"   • {member.name}: {satisfaction:.1%}")
    
    # Step 6: Generate post-meeting insights
    print(f"\n💡 Step 6: Generating Post-Meeting Insights")
    print("-" * 50)
    
    # Create engagement metrics timeline
    engagement_timeline = [
        EngagementMetrics(
            id="metrics_start",
            meeting_id=preparation.meeting_id,
            timestamp=preparation.created_at + timedelta(minutes=10),
            overall_engagement=0.75,
            individual_engagement={"member_001": 0.8, "member_002": 0.7, "member_003": 0.75},
            participation_balance=0.8,
            discussion_quality=0.7,
            decision_momentum=0.6,
            energy_level=0.8
        ),
        EngagementMetrics(
            id="metrics_mid",
            meeting_id=preparation.meeting_id,
            timestamp=preparation.created_at + timedelta(minutes=45),
            overall_engagement=0.65,
            individual_engagement={"member_001": 0.8, "member_002": 0.6, "member_003": 0.55},
            participation_balance=0.6,
            discussion_quality=0.7,
            decision_momentum=0.7,
            energy_level=0.6
        ),
        EngagementMetrics(
            id="metrics_end",
            meeting_id=preparation.meeting_id,
            timestamp=preparation.created_at + timedelta(minutes=90),
            overall_engagement=0.8,
            individual_engagement={"member_001": 0.9, "member_002": 0.8, "member_003": 0.7},
            participation_balance=0.85,
            discussion_quality=0.85,
            decision_momentum=0.9,
            energy_level=0.8
        )
    ]
    
    final_flow = MeetingFlow(
        id="flow_final",
        meeting_preparation_id=preparation.id,
        current_phase=FacilitationPhase.CLOSING,
        current_agenda_item="item_004",
        elapsed_time=90,
        remaining_time=10,
        flow_status="on_schedule",
        engagement_level=0.8,
        decision_progress={"item_002": 1.0, "item_003": 1.0},
        next_actions=[],
        flow_adjustments=[]
    )
    
    insights = engine.generate_post_meeting_insights(
        preparation, final_flow, engagement_timeline, outcomes
    )
    
    print(f"✅ Generated comprehensive post-meeting insights!")
    print(f"   Overall Effectiveness: {insights['overall_effectiveness']:.1%}")
    
    print(f"\n📊 Objective Achievement:")
    for obj_id, achievement in insights['objective_achievement'].items():
        objective = next(obj for obj in preparation.objectives if obj.id == obj_id)
        print(f"   • {objective.title}: {achievement:.1%}")
    
    print(f"\n📈 Engagement Analysis:")
    engagement_analysis = insights['engagement_analysis']
    print(f"   • Average Engagement: {engagement_analysis['average_engagement']:.1%}")
    print(f"   • Peak Engagement: {engagement_analysis['peak_engagement']:.1%}")
    print(f"   • Engagement Trend: {engagement_analysis['trend'].title()}")
    
    print(f"\n⏰ Time Management:")
    time_analysis = insights['time_management']
    print(f"   • Planned Duration: {time_analysis['planned_duration']} minutes")
    print(f"   • Actual Duration: {time_analysis['actual_duration']} minutes")
    print(f"   • Variance: {time_analysis['variance']:+d} minutes")
    print(f"   • Efficiency: {time_analysis['efficiency']:.1%}")
    
    print(f"\n⚖️  Decision Quality:")
    decision_analysis = insights['decision_quality']
    print(f"   • Decisions Made: {decision_analysis['decisions_made']}")
    print(f"   • Average Quality: {decision_analysis['average_quality']:.1%}")
    print(f"   • Stakeholder Satisfaction: {decision_analysis['stakeholder_satisfaction']:.1%}")
    
    print(f"\n🌟 Success Factors:")
    for factor in insights['success_factors']:
        print(f"   • {factor}")
    
    print(f"\n🔧 Improvement Areas:")
    for area in insights['improvement_areas']:
        print(f"   • {area}")
    
    print(f"\n💡 Recommendations:")
    for recommendation in insights['recommendations']:
        print(f"   • {recommendation}")
    
    # Summary
    print(f"\n🎉 Meeting Facilitation Engine Demo Complete!")
    print("=" * 80)
    print(f"✅ Successfully demonstrated real-time meeting facilitation")
    print(f"✅ Generated {len(guidance_list)} facilitation guidance items")
    print(f"✅ Monitored meeting flow across multiple phases")
    print(f"✅ Tracked engagement metrics with {engagement_metrics.overall_engagement:.1%} overall engagement")
    print(f"✅ Suggested {len(interventions)} targeted facilitation interventions")
    print(f"✅ Tracked {len(outcomes)} meeting outcomes with {sum(o.success_score for o in outcomes)/len(outcomes):.1%} average success")
    print(f"✅ Generated comprehensive insights with {insights['overall_effectiveness']:.1%} overall effectiveness")
    print(f"\n🚀 ScrollIntel is ready to facilitate board meetings with executive-level sophistication!")


if __name__ == "__main__":
    asyncio.run(demonstrate_meeting_facilitation())