"""
Cultural Leadership Assessment Demo

Demonstrates the cultural leadership assessment, development planning, 
and effectiveness measurement capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from scrollintel.engines.cultural_leadership_assessment_engine import CulturalLeadershipAssessmentEngine
from scrollintel.models.cultural_leadership_models import CulturalCompetency, LeadershipLevel


async def demo_cultural_leadership_assessment():
    """Demonstrate cultural leadership assessment system"""
    print("🎯 Cultural Leadership Assessment Demo")
    print("=" * 50)
    
    # Initialize assessment engine
    assessment_engine = CulturalLeadershipAssessmentEngine()
    
    # Demo 1: Comprehensive Leadership Assessment
    print("\n1. Comprehensive Cultural Leadership Assessment")
    print("-" * 45)
    
    # Sample assessment data for a senior manager
    assessment_data = {
        "assessment_method": "comprehensive",
        "assessor_id": "assessor_001",
        "self_assessment": False,
        "role": "Senior Manager",
        
        # Vision Creation competency
        "vision_creation": {
            "self_rating": 72,
            "peer_ratings": [68, 75, 70, 78],
            "manager_rating": 76,
            "behaviors": [
                "Creates compelling team vision statements",
                "Communicates vision effectively in team meetings",
                "Aligns project goals with organizational vision",
                "Inspires team members with future possibilities"
            ],
            "evidence": [
                "Led successful vision workshop for product team",
                "Improved team alignment scores by 25%",
                "Received positive feedback on vision communication"
            ]
        },
        
        # Communication competency
        "communication": {
            "self_rating": 85,
            "peer_ratings": [82, 88, 85, 90],
            "manager_rating": 87,
            "behaviors": [
                "Communicates clearly and persuasively",
                "Listens actively to team concerns",
                "Adapts communication style to different audiences",
                "Facilitates difficult conversations effectively"
            ],
            "evidence": [
                "Successfully mediated team conflict",
                "Received high scores on presentation skills",
                "Improved team communication satisfaction by 30%"
            ]
        },
        
        # Change Leadership competency
        "change_leadership": {
            "self_rating": 58,
            "peer_ratings": [55, 62, 60, 58],
            "manager_rating": 61,
            "behaviors": [
                "Supports organizational change initiatives",
                "Helps team members navigate change",
                "Shows resilience during transitions"
            ],
            "evidence": [
                "Led team through recent reorganization",
                "Maintained team morale during change period"
            ]
        },
        
        # Additional assessment dimensions
        "cultural_impact": {
            "team_culture_improvement": 78,
            "cultural_initiative_success": 82,
            "employee_engagement_change": 15,  # 15% improvement
            "cultural_alignment_score": 85
        },
        
        "vision_clarity": {
            "clarity_rating": 80,
            "alignment_rating": 75,
            "inspiration_rating": 85
        },
        
        "team_engagement": {
            "overall_engagement": 82
        },
        
        "change_readiness": {
            "adaptability": 65,
            "resilience": 70,
            "change_advocacy": 58
        },
        
        "peer_feedback": [
            {
                "feedback": "Strong communicator who inspires the team",
                "rating": 85,
                "competency": "communication"
            },
            {
                "feedback": "Could improve change leadership skills",
                "rating": 60,
                "competency": "change_leadership"
            }
        ],
        
        "direct_report_feedback": [
            {
                "feedback": "Creates clear vision but could be more inspiring",
                "rating": 75,
                "competency": "vision_creation"
            },
            {
                "feedback": "Excellent listener and communicator",
                "rating": 90,
                "competency": "communication"
            }
        ]
    }
    
    # Conduct assessment
    assessment = assessment_engine.assess_cultural_leadership(
        leader_id="leader_sarah_chen",
        organization_id="techcorp_inc",
        assessment_data=assessment_data
    )
    
    print(f"Assessment ID: {assessment.id}")
    print(f"Leader: {assessment.leader_id}")
    print(f"Overall Score: {assessment.overall_score:.1f}/100")
    print(f"Leadership Level: {assessment.leadership_level.value.title()}")
    print(f"Cultural Impact Score: {assessment.cultural_impact_score:.1f}/100")
    print(f"Vision Clarity Score: {assessment.vision_clarity_score:.1f}/100")
    print(f"Communication Effectiveness: {assessment.communication_effectiveness:.1f}/100")
    print(f"Change Readiness: {assessment.change_readiness:.1f}/100")
    print(f"Team Engagement Score: {assessment.team_engagement_score:.1f}/100")
    
    print("\nCompetency Breakdown:")
    for score in assessment.competency_scores:
        print(f"  {score.competency.value.replace('_', ' ').title()}: "
              f"{score.score:.1f} ({score.current_level.value} → {score.target_level.value})")
    
    print("\nKey Recommendations:")
    for i, rec in enumerate(assessment.recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Generate insights
    insights = assessment_engine.get_assessment_insights(assessment)
    
    print(f"\nLeadership Style: {insights['leadership_style']}")
    print(f"Cultural Impact Potential: {insights['cultural_impact_potential']}")
    
    print("\nTop Strengths:")
    for strength in insights['strengths'][:3]:
        print(f"  • {strength}")
    
    print("\nDevelopment Opportunities:")
    for opportunity in insights['development_opportunities'][:3]:
        print(f"  • {opportunity}")
    
    # Demo 2: Development Plan Creation
    print("\n\n2. Leadership Development Plan Creation")
    print("-" * 42)
    
    development_preferences = {
        "duration_days": 180,  # 6-month plan
        "coaching_sessions": 8,
        "learning_style": "blended",
        "coach_id": "coach_michael_torres",
        "peer_learning": True,
        "budget_range": "medium"
    }
    
    development_plan = assessment_engine.create_development_plan(
        assessment, development_preferences
    )
    
    print(f"Development Plan ID: {development_plan.id}")
    print(f"Duration: {(development_plan.target_completion - development_plan.created_date).days} days")
    print(f"Priority Competencies: {[comp.value.replace('_', ' ').title() for comp in development_plan.priority_competencies]}")
    
    print("\nDevelopment Goals:")
    for i, goal in enumerate(development_plan.development_goals, 1):
        print(f"  {i}. {goal}")
    
    print(f"\nLearning Activities ({len(development_plan.learning_activities)} total):")
    for activity in development_plan.learning_activities[:3]:  # Show first 3
        print(f"  • {activity.title} ({activity.activity_type}, {activity.estimated_duration}h)")
        print(f"    Target: {[comp.value.replace('_', ' ').title() for comp in activity.target_competencies]}")
    
    print(f"\nCoaching Schedule ({len(development_plan.coaching_sessions)} sessions):")
    for i, session in enumerate(development_plan.coaching_sessions[:3], 1):  # Show first 3
        print(f"  Session {i}: {session.session_date.strftime('%Y-%m-%d')} "
              f"({session.duration} min)")
        print(f"    Focus: {[area.value.replace('_', ' ').title() for area in session.focus_areas]}")
    
    print(f"\nProgress Milestones ({len(development_plan.progress_milestones)} total):")
    for milestone in development_plan.progress_milestones:
        print(f"  • {milestone.title} (Due: {milestone.target_date.strftime('%Y-%m-%d')})")
        print(f"    Criteria: {milestone.completion_criteria[0]}")
    
    print("\nSuccess Metrics:")
    for metric in development_plan.success_metrics[:3]:
        print(f"  • {metric}")
    
    # Demo 3: Leadership Effectiveness Measurement
    print("\n\n3. Leadership Effectiveness Measurement")
    print("-" * 40)
    
    # Simulate effectiveness metrics after 6 months
    effectiveness_data = {
        "team_engagement_score": 88,  # Improved from 82
        "cultural_alignment_score": 85,
        "change_success_rate": 78,  # Improved change leadership
        "vision_clarity_rating": 85,  # Improved from 80
        "communication_effectiveness": 92,  # Strong area maintained
        "influence_reach": 150,  # Number of people influenced
        "retention_rate": 94,
        "promotion_rate": 18,  # % of team members promoted
        "peer_leadership_rating": 84,
        "direct_report_satisfaction": 89,
        "cultural_initiative_success": 82,
        "innovation_fostered": 7,  # Number of innovations
        "conflict_resolution_success": 91
    }
    
    effectiveness_metrics = assessment_engine.measure_leadership_effectiveness(
        leader_id="leader_sarah_chen",
        measurement_period="Q3_2024",
        metrics_data=effectiveness_data
    )
    
    print(f"Measurement Period: {effectiveness_metrics.measurement_period}")
    print(f"Team Engagement: {effectiveness_metrics.team_engagement_score}/100")
    print(f"Cultural Alignment: {effectiveness_metrics.cultural_alignment_score}/100")
    print(f"Change Success Rate: {effectiveness_metrics.change_success_rate}%")
    print(f"Vision Clarity: {effectiveness_metrics.vision_clarity_rating}/100")
    print(f"Communication Effectiveness: {effectiveness_metrics.communication_effectiveness}/100")
    print(f"Influence Reach: {effectiveness_metrics.influence_reach} people")
    print(f"Team Retention Rate: {effectiveness_metrics.retention_rate}%")
    print(f"Team Promotion Rate: {effectiveness_metrics.promotion_rate}%")
    print(f"Peer Leadership Rating: {effectiveness_metrics.peer_leadership_rating}/100")
    print(f"Direct Report Satisfaction: {effectiveness_metrics.direct_report_satisfaction}/100")
    print(f"Cultural Initiative Success: {effectiveness_metrics.cultural_initiative_success}/100")
    print(f"Innovations Fostered: {effectiveness_metrics.innovation_fostered}")
    print(f"Conflict Resolution Success: {effectiveness_metrics.conflict_resolution_success}%")
    
    # Demo 4: Competency Deep Dive
    print("\n\n4. Cultural Leadership Competencies Overview")
    print("-" * 45)
    
    print("Core Cultural Leadership Competencies:")
    competency_descriptions = {
        CulturalCompetency.VISION_CREATION: "Creating compelling organizational visions that inspire action",
        CulturalCompetency.VALUES_ALIGNMENT: "Ensuring organizational values are lived consistently",
        CulturalCompetency.CHANGE_LEADERSHIP: "Leading organizational and cultural transformation",
        CulturalCompetency.COMMUNICATION: "Communicating with clarity, influence, and cultural sensitivity",
        CulturalCompetency.INFLUENCE: "Influencing others through trust, credibility, and inspiration",
        CulturalCompetency.EMPATHY: "Understanding and responding to others' emotions and perspectives",
        CulturalCompetency.AUTHENTICITY: "Demonstrating genuine, consistent, and transparent leadership",
        CulturalCompetency.RESILIENCE: "Maintaining effectiveness under pressure and bouncing back",
        CulturalCompetency.ADAPTABILITY: "Adjusting approach based on changing circumstances",
        CulturalCompetency.SYSTEMS_THINKING: "Understanding organizational complexity and interconnections"
    }
    
    for competency, description in competency_descriptions.items():
        weight = assessment_engine.competency_weights[competency]
        print(f"\n{competency.value.replace('_', ' ').title()} (Weight: {weight:.1%})")
        print(f"  {description}")
    
    # Demo 5: Assessment Framework Options
    print("\n\n5. Available Assessment Frameworks")
    print("-" * 37)
    
    frameworks = {
        "comprehensive": {
            "name": "Comprehensive 360° Assessment",
            "duration": "2-3 hours",
            "participants": ["Self", "Manager", "Peers", "Direct Reports"],
            "methods": ["Self-rating", "360 feedback", "Behavioral observation"]
        },
        "360_feedback": {
            "name": "360-Degree Feedback",
            "duration": "1-2 hours", 
            "participants": ["Self", "Manager", "Peers", "Direct Reports"],
            "methods": ["Self-rating", "Multi-source feedback"]
        },
        "self_assessment": {
            "name": "Self-Assessment Tool",
            "duration": "45-60 minutes",
            "participants": ["Self"],
            "methods": ["Self-reflection", "Competency rating"]
        }
    }
    
    for framework_id, framework in frameworks.items():
        print(f"\n{framework['name']}:")
        print(f"  Duration: {framework['duration']}")
        print(f"  Participants: {', '.join(framework['participants'])}")
        print(f"  Methods: {', '.join(framework['methods'])}")
    
    print("\n" + "=" * 50)
    print("✅ Cultural Leadership Assessment Demo Complete!")
    print("\nKey Capabilities Demonstrated:")
    print("• Comprehensive cultural leadership assessment")
    print("• Multi-source feedback integration (360°)")
    print("• Personalized development plan creation")
    print("• Learning activity and coaching recommendations")
    print("• Progress milestone and success metric definition")
    print("• Leadership effectiveness measurement")
    print("• Assessment insights and recommendations")
    print("• Multiple assessment framework options")


if __name__ == "__main__":
    asyncio.run(demo_cultural_leadership_assessment())