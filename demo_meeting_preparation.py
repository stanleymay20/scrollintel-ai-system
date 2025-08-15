"""
Demo script for Meeting Preparation Framework
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from scrollintel.engines.meeting_preparation_engine import MeetingPreparationEngine
from scrollintel.models.meeting_preparation_models import (
    MeetingPreparation, BoardMember, MeetingObjective, AgendaItem,
    MeetingType, PreparationStatus, ContentType
)


def create_sample_board_members() -> List[BoardMember]:
    """Create sample board members for demonstration"""
    return [
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
                "Shareholder value",
                "Regulatory compliance"
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
                "Digital transformation",
                "Competitive positioning"
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
                "Process improvement",
                "Cost management"
            ],
            decision_patterns={
                "speed": "thorough",
                "style": "risk_averse",
                "risk_tolerance": "low"
            }
        ),
        BoardMember(
            id="member_004",
            name="Reid Hoffman",
            role="Independent Director",
            expertise_areas=["Venture Capital", "Scaling", "Networks"],
            communication_preferences={
                "style": "entrepreneurial",
                "detail_level": "medium",
                "preferred_format": "growth_focused"
            },
            influence_level=0.80,
            typical_concerns=[
                "Growth strategy",
                "Market expansion",
                "Competitive advantage",
                "Talent acquisition"
            ],
            decision_patterns={
                "speed": "quick",
                "style": "opportunity_focused",
                "risk_tolerance": "high"
            }
        ),
        BoardMember(
            id="member_005",
            name="Ursula Burns",
            role="Compensation Committee Chair",
            expertise_areas=["Leadership", "Transformation", "Diversity"],
            communication_preferences={
                "style": "transformational",
                "detail_level": "medium",
                "preferred_format": "leadership_focused"
            },
            influence_level=0.75,
            typical_concerns=[
                "Leadership development",
                "Organizational transformation",
                "Diversity and inclusion",
                "Executive compensation"
            ],
            decision_patterns={
                "speed": "balanced",
                "style": "people_focused",
                "risk_tolerance": "moderate"
            }
        )
    ]


def create_sample_objectives() -> List[MeetingObjective]:
    """Create sample meeting objectives for demonstration"""
    return [
        MeetingObjective(
            id="obj_001",
            title="Q4 2024 Financial Performance Review",
            description="Comprehensive review of Q4 financial results, annual performance, and budget approval for 2025",
            priority=1,
            success_criteria=[
                "Q4 financial results clearly communicated and understood",
                "Annual performance benchmarks reviewed and assessed",
                "2025 budget approved by board with clear rationale",
                "Key financial risks identified and mitigation strategies approved",
                "Dividend policy for 2025 confirmed"
            ],
            required_decisions=[
                "Approve 2025 annual budget",
                "Approve dividend policy for 2025",
                "Approve capital allocation strategy",
                "Approve financial risk management framework"
            ],
            stakeholders=["CFO", "Board", "Shareholders", "Audit Committee"]
        ),
        MeetingObjective(
            id="obj_002",
            title="AI Technology Investment Strategy",
            description="Present comprehensive AI technology investment strategy and secure board approval for major technology initiatives",
            priority=2,
            success_criteria=[
                "AI technology strategy clearly articulated with business impact",
                "Investment requirements and ROI projections presented",
                "Implementation timeline and milestones agreed upon",
                "Risk assessment and mitigation strategies approved",
                "Resource allocation and team structure confirmed"
            ],
            required_decisions=[
                "Approve $50M AI technology investment",
                "Approve hiring plan for AI talent acquisition",
                "Approve technology partnership agreements",
                "Set implementation timeline and milestones"
            ],
            stakeholders=["CTO", "Board", "Technology Committee", "Executive Team"]
        ),
        MeetingObjective(
            id="obj_003",
            title="Market Expansion and Competitive Positioning",
            description="Review market expansion opportunities and approve strategic initiatives to strengthen competitive position",
            priority=3,
            success_criteria=[
                "Market analysis and expansion opportunities presented",
                "Competitive landscape assessment completed",
                "Strategic initiatives clearly defined with success metrics",
                "Resource requirements and timeline established",
                "Risk factors and mitigation strategies identified"
            ],
            required_decisions=[
                "Approve market expansion strategy",
                "Approve strategic partnership initiatives",
                "Approve competitive response framework",
                "Set market expansion budget and timeline"
            ],
            stakeholders=["CEO", "Board", "Strategy Committee", "Sales Leadership"]
        ),
        MeetingObjective(
            id="obj_004",
            title="ESG and Sustainability Framework",
            description="Present comprehensive ESG framework and sustainability initiatives for board approval and oversight",
            priority=4,
            success_criteria=[
                "ESG framework clearly articulated with measurable goals",
                "Sustainability initiatives presented with impact projections",
                "Governance structure for ESG oversight established",
                "Reporting and accountability mechanisms defined",
                "Stakeholder engagement strategy approved"
            ],
            required_decisions=[
                "Approve ESG framework and sustainability goals",
                "Approve ESG governance structure",
                "Approve sustainability investment budget",
                "Set ESG reporting and disclosure requirements"
            ],
            stakeholders=["Board", "ESG Committee", "Sustainability Team", "Stakeholders"]
        )
    ]


async def demonstrate_meeting_preparation():
    """Demonstrate comprehensive meeting preparation capabilities"""
    print("🎯 ScrollIntel Board Executive Mastery - Meeting Preparation Framework Demo")
    print("=" * 80)
    
    # Initialize the meeting preparation engine
    engine = MeetingPreparationEngine()
    
    # Create sample data
    board_members = create_sample_board_members()
    objectives = create_sample_objectives()
    meeting_date = datetime.now() + timedelta(days=14)  # Meeting in 2 weeks
    
    print(f"\n📅 Meeting Details:")
    print(f"   Date: {meeting_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"   Type: Board Meeting")
    print(f"   Board Members: {len(board_members)}")
    print(f"   Objectives: {len(objectives)}")
    
    # Step 1: Create comprehensive meeting preparation
    print(f"\n🔧 Step 1: Creating Comprehensive Meeting Preparation")
    print("-" * 50)
    
    preparation = engine.create_meeting_preparation(
        meeting_id="board_meeting_2024_q4",
        meeting_type=MeetingType.BOARD_MEETING,
        meeting_date=meeting_date,
        board_members=board_members,
        objectives=objectives
    )
    
    print(f"✅ Meeting preparation created successfully!")
    print(f"   Preparation ID: {preparation.id}")
    print(f"   Status: {preparation.status.value}")
    print(f"   Preparation Score: {preparation.preparation_score}/10")
    print(f"   Agenda Items: {len(preparation.agenda_items)}")
    print(f"   Preparation Tasks: {len(preparation.preparation_tasks)}")
    print(f"   Success Metrics: {len(preparation.success_metrics)}")
    
    # Display agenda overview
    print(f"\n📋 Generated Agenda Overview:")
    for i, item in enumerate(preparation.agenda_items, 1):
        print(f"   {i}. {item.title} ({item.duration_minutes} min)")
        print(f"      Presenter: {item.presenter}")
        print(f"      Type: {item.content_type.value}")
        if item.decision_required:
            print(f"      ⚠️  Decision Required")
    
    # Display preparation tasks
    print(f"\n📝 Preparation Tasks Overview:")
    for task in preparation.preparation_tasks[:5]:  # Show first 5 tasks
        print(f"   • {task.title}")
        print(f"     Assignee: {task.assignee}")
        print(f"     Due: {task.due_date.strftime('%Y-%m-%d')}")
        print(f"     Status: {task.status.value}")
    
    if len(preparation.preparation_tasks) > 5:
        print(f"   ... and {len(preparation.preparation_tasks) - 5} more tasks")
    
    # Step 2: Optimize meeting agenda
    print(f"\n⚡ Step 2: Optimizing Meeting Agenda")
    print("-" * 50)
    
    optimization = engine.optimize_agenda(preparation)
    
    print(f"✅ Agenda optimization completed!")
    print(f"   Optimization ID: {optimization.id}")
    print(f"   Flow Improvements: {len(optimization.flow_improvements)}")
    print(f"   Engagement Enhancements: {len(optimization.engagement_enhancements)}")
    print(f"   Decision Optimizations: {len(optimization.decision_optimization)}")
    
    print(f"\n🔄 Optimization Insights:")
    print(f"   Rationale: {optimization.optimization_rationale}")
    
    print(f"\n   Flow Improvements:")
    for improvement in optimization.flow_improvements:
        print(f"   • {improvement}")
    
    print(f"\n   Engagement Enhancements:")
    for enhancement in optimization.engagement_enhancements:
        print(f"   • {enhancement}")
    
    # Step 3: Prepare content for key agenda items
    print(f"\n📄 Step 3: Preparing Content for Key Agenda Items")
    print("-" * 50)
    
    # Find the AI technology objective agenda item
    ai_agenda_item = next(
        (item for item in preparation.agenda_items 
         if "AI Technology" in item.title or "obj_002" in item.objectives),
        preparation.agenda_items[1]  # Fallback to second item
    )
    
    content_prep = engine.prepare_content(preparation, ai_agenda_item)
    
    print(f"✅ Content preparation completed for: {ai_agenda_item.title}")
    print(f"   Content ID: {content_prep.content_id}")
    print(f"   Target Audience: {len(content_prep.target_audience)} board members")
    print(f"   Key Messages: {len(content_prep.key_messages)}")
    print(f"   Supporting Evidence: {len(content_prep.supporting_evidence)}")
    print(f"   Visual Aids: {len(content_prep.visual_aids)}")
    
    print(f"\n💡 Key Messages:")
    for i, message in enumerate(content_prep.key_messages, 1):
        print(f"   {i}. {message}")
    
    print(f"\n📊 Supporting Evidence:")
    for evidence in content_prep.supporting_evidence:
        print(f"   • {evidence}")
    
    print(f"\n🎨 Visual Aids:")
    for aid in content_prep.visual_aids:
        print(f"   • {aid}")
    
    print(f"\n📖 Narrative Structure:")
    print(f"   {content_prep.narrative_structure}")
    
    # Step 4: Predict meeting success
    print(f"\n🔮 Step 4: Predicting Meeting Success")
    print("-" * 50)
    
    prediction = engine.predict_meeting_success(preparation)
    
    print(f"✅ Success prediction completed!")
    print(f"   Prediction ID: {prediction.id}")
    print(f"   Overall Success Probability: {prediction.overall_success_probability:.1%}")
    print(f"   Engagement Prediction: {prediction.engagement_prediction:.1%}")
    print(f"   Decision Quality Prediction: {prediction.decision_quality_prediction:.1%}")
    
    print(f"\n📊 Objective Achievement Probabilities:")
    for obj_id, probability in prediction.objective_achievement_probabilities.items():
        objective = next(obj for obj in objectives if obj.id == obj_id)
        print(f"   • {objective.title}: {probability:.1%}")
    
    print(f"\n👥 Stakeholder Satisfaction Predictions:")
    for member_id, satisfaction in prediction.stakeholder_satisfaction_prediction.items():
        member = next(member for member in board_members if member.id == member_id)
        print(f"   • {member.name}: {satisfaction:.1%}")
    
    print(f"\n⚠️  Risk Factors:")
    for risk in prediction.risk_factors:
        print(f"   • {risk['description']} (Probability: {risk['probability']:.1%}, Impact: {risk['impact']})")
    
    print(f"\n🚀 Enhancement Recommendations:")
    for i, recommendation in enumerate(prediction.enhancement_recommendations, 1):
        print(f"   {i}. {recommendation}")
    
    print(f"\n📈 Confidence Interval: {prediction.confidence_interval['lower']:.1%} - {prediction.confidence_interval['upper']:.1%}")
    
    # Step 5: Generate preparation insights
    print(f"\n💡 Step 5: Generating Preparation Insights")
    print("-" * 50)
    
    insights = engine.generate_preparation_insights(preparation)
    
    print(f"✅ Generated {len(insights)} preparation insights!")
    
    for insight in insights:
        print(f"\n🔍 {insight.title}")
        print(f"   Type: {insight.insight_type}")
        print(f"   Impact Level: {insight.impact_level.upper()}")
        print(f"   Confidence: {insight.confidence_score:.1%}")
        print(f"   Description: {insight.description}")
        
        print(f"   Recommendations:")
        for rec in insight.actionable_recommendations:
            print(f"   • {rec}")
    
    # Step 6: Display final preparation status
    print(f"\n📋 Final Preparation Status")
    print("-" * 50)
    
    print(f"Meeting: {preparation.meeting_id}")
    print(f"Date: {preparation.meeting_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Status: {preparation.status.value}")
    print(f"Preparation Score: {preparation.preparation_score}/10")
    print(f"Success Prediction: {preparation.success_prediction:.1%}")
    
    # Task completion status
    task_status_counts = {}
    for task in preparation.preparation_tasks:
        status = task.status.value
        task_status_counts[status] = task_status_counts.get(status, 0) + 1
    
    print(f"\nTask Status:")
    for status, count in task_status_counts.items():
        print(f"   {status.replace('_', ' ').title()}: {count}")
    
    print(f"\nRisk Factors: {len(preparation.risk_factors)}")
    for risk in preparation.risk_factors:
        print(f"   • {risk}")
    
    print(f"\nMitigation Strategies: {len(preparation.mitigation_strategies)}")
    for strategy in preparation.mitigation_strategies[:3]:  # Show first 3
        print(f"   • {strategy}")
    
    # Summary
    print(f"\n🎉 Meeting Preparation Framework Demo Complete!")
    print("=" * 80)
    print(f"✅ Successfully demonstrated comprehensive board meeting preparation")
    print(f"✅ Generated optimized agenda with {len(preparation.agenda_items)} items")
    print(f"✅ Created {len(preparation.preparation_tasks)} preparation tasks")
    print(f"✅ Prepared detailed content for key agenda items")
    print(f"✅ Predicted meeting success with {prediction.overall_success_probability:.1%} probability")
    print(f"✅ Generated {len(insights)} actionable preparation insights")
    print(f"\n🚀 ScrollIntel is ready to excel in board-level executive engagement!")


if __name__ == "__main__":
    asyncio.run(demonstrate_meeting_preparation())