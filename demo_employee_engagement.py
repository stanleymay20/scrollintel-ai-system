"""
Demo script for Employee Engagement System
"""

import asyncio
import json
from datetime import datetime, timedelta
from scrollintel.engines.employee_engagement_engine import EmployeeEngagementEngine
from scrollintel.models.employee_engagement_models import (
    Employee, EngagementFeedback, EngagementLevel
)


async def main():
    """
    Demonstrate the Employee Engagement System capabilities
    """
    print("🚀 ScrollIntel Employee Engagement System Demo")
    print("=" * 60)
    
    # Initialize the engine
    engine = EmployeeEngagementEngine()
    
    # Demo organization data
    organization_id = "demo_company_2024"
    target_groups = ["engineering", "sales", "marketing", "hr"]
    cultural_objectives = [
        "Foster innovation and creativity",
        "Improve cross-team collaboration",
        "Enhance communication transparency",
        "Build inclusive culture",
        "Promote continuous learning"
    ]
    
    # Current engagement data (simulated from surveys/assessments)
    current_engagement_data = {
        "engagement_levels": {
            "engineering": 0.78,
            "sales": 0.52,
            "marketing": 0.65,
            "hr": 0.71
        },
        "communication_scores": {
            "internal": 0.62,
            "management": 0.55,
            "cross_team": 0.58
        },
        "satisfaction_scores": {
            "job_satisfaction": 0.68,
            "culture_fit": 0.64,
            "growth_opportunities": 0.59
        }
    }
    
    print("\n📊 Current Engagement Data:")
    print(json.dumps(current_engagement_data, indent=2))
    
    # Step 1: Develop Engagement Strategy
    print("\n🎯 Step 1: Developing Employee Engagement Strategy...")
    strategy = engine.develop_engagement_strategy(
        organization_id=organization_id,
        target_groups=target_groups,
        cultural_objectives=cultural_objectives,
        current_engagement_data=current_engagement_data
    )
    
    print(f"✅ Strategy Created: {strategy.name}")
    print(f"   - Target Groups: {', '.join(strategy.target_groups)}")
    print(f"   - Objectives: {len(strategy.objectives)} defined")
    print(f"   - Budget Allocated: ${strategy.budget_allocated:,.2f}")
    print(f"   - Status: {strategy.status}")
    
    # Create sample employee profiles
    employee_profiles = [
        Employee(
            id="emp_001",
            name="Alice Johnson",
            department="engineering",
            role="senior_developer",
            manager_id="mgr_eng",
            hire_date=datetime.now() - timedelta(days=730),
            engagement_level=EngagementLevel.HIGHLY_ENGAGED,
            cultural_alignment_score=0.85
        ),
        Employee(
            id="emp_002",
            name="Bob Smith",
            department="sales",
            role="account_manager",
            manager_id="mgr_sales",
            hire_date=datetime.now() - timedelta(days=365),
            engagement_level=EngagementLevel.SOMEWHAT_ENGAGED,
            cultural_alignment_score=0.55
        ),
        Employee(
            id="emp_003",
            name="Carol Davis",
            department="marketing",
            role="marketing_specialist",
            manager_id="mgr_marketing",
            hire_date=datetime.now() - timedelta(days=180),
            engagement_level=EngagementLevel.ENGAGED,
            cultural_alignment_score=0.72
        ),
        Employee(
            id="emp_004",
            name="David Wilson",
            department="sales",
            role="sales_representative",
            manager_id="mgr_sales",
            hire_date=datetime.now() - timedelta(days=90),
            engagement_level=EngagementLevel.DISENGAGED,
            cultural_alignment_score=0.38
        ),
        Employee(
            id="emp_005",
            name="Eva Martinez",
            department="hr",
            role="hr_specialist",
            manager_id="mgr_hr",
            hire_date=datetime.now() - timedelta(days=540),
            engagement_level=EngagementLevel.ENGAGED,
            cultural_alignment_score=0.78
        )
    ]
    
    print(f"\n👥 Employee Profiles: {len(employee_profiles)} employees")
    for emp in employee_profiles:
        print(f"   - {emp.name} ({emp.department}): {emp.engagement_level.value}")
    
    # Step 2: Design Engagement Activities
    print("\n🎨 Step 2: Designing Engagement Activities...")
    activities = engine.design_engagement_activities(
        strategy=strategy,
        employee_profiles=employee_profiles
    )
    
    print(f"✅ Activities Designed: {len(activities)} activities")
    for i, activity in enumerate(activities[:5], 1):  # Show first 5
        print(f"   {i}. {activity.name}")
        print(f"      Type: {activity.activity_type.value}")
        print(f"      Duration: {activity.duration_minutes} minutes")
        print(f"      Facilitator: {activity.facilitator}")
        print(f"      Objectives: {len(activity.objectives)} defined")
    
    if len(activities) > 5:
        print(f"   ... and {len(activities) - 5} more activities")
    
    # Step 3: Execute Sample Activity
    print("\n⚡ Step 3: Executing Sample Engagement Activity...")
    if activities:
        sample_activity = activities[0]
        participants = ["emp_001", "emp_002", "emp_003"]
        execution_context = {
            "send_pre_survey": True,
            "venue": "conference_room_a",
            "virtual_option": False
        }
        
        execution_report = engine.execute_engagement_activity(
            activity_id=sample_activity.id,
            participants=participants,
            execution_context=execution_context
        )
        
        print(f"✅ Activity Executed: {sample_activity.name}")
        print(f"   - Participants: {len(participants)}")
        print(f"   - Attendance Rate: {execution_report['execution_results']['attendance_rate']:.1%}")
        print(f"   - Engagement Score: {execution_report['execution_results']['engagement_score']:.1f}/5.0")
        print(f"   - Completion Rate: {execution_report['execution_results']['completion_rate']:.1%}")
        print(f"   - Success Level: {execution_report['success_indicators']['success_level']}")
    
    # Step 4: Process Employee Feedback
    print("\n💬 Step 4: Processing Employee Feedback...")
    
    # Simulate various types of feedback
    feedback_samples = [
        {
            "employee_id": "emp_001",
            "feedback_type": "activity",
            "rating": 4.5,
            "comments": "Excellent workshop! I learned great communication techniques and enjoyed the interactive exercises.",
            "suggestions": ["More hands-on practice", "Follow-up sessions"]
        },
        {
            "employee_id": "emp_002",
            "feedback_type": "general",
            "rating": 3.0,
            "comments": "The session was okay but felt rushed. Need better time management and clearer instructions.",
            "suggestions": ["Longer duration", "Better preparation"]
        },
        {
            "employee_id": "emp_004",
            "feedback_type": "suggestion",
            "rating": 2.5,
            "comments": "I'm frustrated with the lack of career development opportunities and poor communication from management.",
            "suggestions": ["Career planning sessions", "Regular manager check-ins"]
        }
    ]
    
    processed_feedback = []
    for feedback_data in feedback_samples:
        feedback = EngagementFeedback(
            id=f"feedback_{feedback_data['employee_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            employee_id=feedback_data["employee_id"],
            feedback_type=feedback_data["feedback_type"],
            rating=feedback_data["rating"],
            comments=feedback_data["comments"],
            suggestions=feedback_data["suggestions"],
            sentiment="neutral",
            themes=[],
            submitted_date=datetime.now(),
            processed=False
        )
        
        processing_results = engine.process_engagement_feedback(feedback)
        processed_feedback.append(processing_results)
        
        print(f"✅ Processed feedback from {feedback_data['employee_id']}:")
        print(f"   - Sentiment: {processing_results['sentiment_analysis']['sentiment']}")
        print(f"   - Themes: {', '.join(processing_results['themes'])}")
        print(f"   - Key Insights: {len(processing_results['insights'])} identified")
    
    # Step 5: Measure Engagement Effectiveness
    print("\n📈 Step 5: Measuring Engagement Effectiveness...")
    measurement_period = {
        "start": datetime.now() - timedelta(days=30),
        "end": datetime.now()
    }
    
    effectiveness_report = engine.measure_engagement_effectiveness(
        organization_id=organization_id,
        measurement_period=measurement_period
    )
    
    print(f"✅ Effectiveness Report Generated:")
    print(f"   - Overall Engagement Score: {effectiveness_report.overall_engagement_score:.2f}/5.0")
    print(f"   - Department Scores:")
    for dept, score in effectiveness_report.engagement_by_department.items():
        print(f"     • {dept.title()}: {score:.2f}")
    
    print(f"   - Activity Effectiveness:")
    for activity_type, effectiveness in effectiveness_report.activity_effectiveness.items():
        print(f"     • {activity_type.replace('_', ' ').title()}: {effectiveness:.1%}")
    
    print(f"   - Key Insights ({len(effectiveness_report.key_insights)}):")
    for insight in effectiveness_report.key_insights[:3]:
        print(f"     • {insight}")
    
    print(f"   - Recommendations ({len(effectiveness_report.recommendations)}):")
    for rec in effectiveness_report.recommendations[:3]:
        print(f"     • {rec}")
    
    # Step 6: Create Improvement Plan
    print("\n🎯 Step 6: Creating Engagement Improvement Plan...")
    improvement_goals = [
        "Increase overall engagement score to 4.0+",
        "Improve sales team engagement by 30%",
        "Enhance communication effectiveness",
        "Reduce employee turnover by 15%",
        "Increase participation in development programs"
    ]
    
    improvement_plan = engine.create_improvement_plan(
        engagement_report=effectiveness_report,
        improvement_goals=improvement_goals
    )
    
    print(f"✅ Improvement Plan Created:")
    print(f"   - Current Overall Engagement: {improvement_plan.current_state['overall_engagement']:.2f}")
    print(f"   - Target Overall Engagement: {improvement_plan.target_state['overall_engagement']:.2f}")
    print(f"   - Improvement Strategies: {len(improvement_plan.improvement_strategies)}")
    
    for i, strategy in enumerate(improvement_plan.improvement_strategies, 1):
        print(f"     {i}. {strategy.name}")
        print(f"        Budget: ${strategy.budget_allocated:,.2f}")
        print(f"        Objectives: {len(strategy.objectives)}")
    
    print(f"   - Success Criteria: {len(improvement_plan.success_criteria)}")
    for criteria in improvement_plan.success_criteria[:3]:
        print(f"     • {criteria}")
    
    print(f"   - Resource Requirements:")
    resources = improvement_plan.resource_requirements
    print(f"     • Total Budget: ${resources['budget']:,.2f}")
    print(f"     • Personnel: {resources['personnel']}")
    print(f"     • Time Commitment: {resources['time_commitment']}")
    
    # Step 7: Analytics and Insights
    print("\n📊 Step 7: Advanced Analytics and Insights...")
    
    # Engagement trends analysis
    trends = effectiveness_report.engagement_trends
    print("✅ Engagement Trends Analysis:")
    for metric, trend_data in trends.items():
        if len(trend_data) >= 2:
            change = trend_data[-1] - trend_data[-2]
            direction = "↗️" if change > 0 else "↘️" if change < 0 else "➡️"
            print(f"   {direction} {metric.replace('_', ' ').title()}: {trend_data[-1]:.3f} ({change:+.3f})")
    
    # Feedback sentiment analysis
    print("\n💭 Feedback Sentiment Analysis:")
    sentiment_summary = {}
    for feedback_result in processed_feedback:
        sentiment = feedback_result['sentiment_analysis']['sentiment']
        sentiment_summary[sentiment] = sentiment_summary.get(sentiment, 0) + 1
    
    total_feedback = len(processed_feedback)
    for sentiment, count in sentiment_summary.items():
        percentage = (count / total_feedback) * 100
        print(f"   • {sentiment.title()}: {count}/{total_feedback} ({percentage:.1f}%)")
    
    # Risk assessment
    print("\n⚠️ Risk Assessment:")
    risks = improvement_plan.risk_mitigation
    for i, risk in enumerate(risks[:3], 1):
        print(f"   {i}. {risk}")
    
    # Success predictions
    print("\n🎯 Success Predictions:")
    success_probability = 0.85  # Simulated based on plan quality
    print(f"   • Plan Success Probability: {success_probability:.1%}")
    print(f"   • Expected Timeline: {len(improvement_plan.implementation_timeline)} phases")
    print(f"   • Monitoring Frequency: {improvement_plan.monitoring_plan['monitoring_frequency']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 EMPLOYEE ENGAGEMENT SYSTEM SUMMARY")
    print("=" * 60)
    print(f"✅ Strategy Developed: {strategy.name}")
    print(f"✅ Activities Designed: {len(activities)} activities")
    print(f"✅ Activities Executed: 1 sample activity")
    print(f"✅ Feedback Processed: {len(processed_feedback)} feedback items")
    print(f"✅ Effectiveness Measured: Comprehensive report generated")
    print(f"✅ Improvement Plan Created: {len(improvement_plan.improvement_strategies)} strategies")
    
    print(f"\n🎯 Key Outcomes:")
    print(f"   • Current Engagement Score: {effectiveness_report.overall_engagement_score:.2f}/5.0")
    print(f"   • Target Engagement Score: {improvement_plan.target_state['overall_engagement']:.2f}/5.0")
    print(f"   • Improvement Potential: {improvement_plan.target_state['overall_engagement'] - effectiveness_report.overall_engagement_score:.2f} points")
    print(f"   • Investment Required: ${improvement_plan.resource_requirements['budget']:,.2f}")
    print(f"   • Expected ROI: High (improved retention, productivity, satisfaction)")
    
    print(f"\n🚀 Next Steps:")
    print("   1. Review and approve improvement plan")
    print("   2. Allocate resources and assign ownership")
    print("   3. Begin implementation of priority strategies")
    print("   4. Establish monitoring and feedback loops")
    print("   5. Track progress against success criteria")
    
    print("\n🎉 Employee Engagement System Demo Complete!")
    print("The system successfully demonstrates comprehensive employee engagement")
    print("capabilities including strategy development, activity design, execution,")
    print("feedback processing, effectiveness measurement, and improvement planning.")


if __name__ == "__main__":
    asyncio.run(main())