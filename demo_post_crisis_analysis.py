#!/usr/bin/env python3
"""
Post-Crisis Analysis System Demo

Demonstrates comprehensive crisis response analysis and evaluation capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from scrollintel.engines.post_crisis_analysis_engine import PostCrisisAnalysisEngine
from scrollintel.models.crisis_detection_models import Crisis, CrisisType, SeverityLevel, CrisisStatus


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


async def demo_post_crisis_analysis():
    """Demonstrate post-crisis analysis capabilities"""
    
    print_section("POST-CRISIS ANALYSIS SYSTEM DEMO")
    
    # Initialize the analysis engine
    engine = PostCrisisAnalysisEngine()
    print("✅ Post-Crisis Analysis Engine initialized")
    
    # Create sample crisis
    crisis = Crisis(
        id="crisis_20240101_001",
        crisis_type=CrisisType.SYSTEM_OUTAGE,
        severity_level=SeverityLevel.HIGH,
        start_time=datetime.now() - timedelta(hours=3),
        affected_areas=["production_api", "user_dashboard", "payment_system"],
        stakeholders_impacted=["customers", "partners", "internal_teams"],
        current_status=CrisisStatus.RESOLVED,
        response_actions=[
            "emergency_team_activated",
            "stakeholder_notification_sent",
            "backup_systems_activated",
            "root_cause_investigation_started",
            "fix_deployed",
            "systems_restored"
        ],
        resolution_time=datetime.now()
    )
    
    print(f"📋 Sample Crisis Created: {crisis.id}")
    print(f"   Type: {crisis.crisis_type.value}")
    print(f"   Severity: {crisis.severity_level.value}")
    print(f"   Duration: {(crisis.resolution_time - crisis.start_time).total_seconds() / 3600:.1f} hours")
    
    # Sample response data
    response_data = {
        "response_time": 22.0,  # minutes
        "communication_score": 88.0,  # percentage
        "resource_efficiency": 82.0,  # percentage
        "stakeholder_satisfaction": 79,  # percentage
        "team_coordination_score": 91,  # percentage
        "communication_delays": 1,  # count
        "resource_shortages": 0,  # count
        "customer_impact": "Medium",
        "employee_impact": "Low",
        "investor_impact": "Medium",
        "partner_impact": "Low",
        "revenue_impact": 75000,  # dollars
        "operational_disruption": "High",
        "recovery_time": 180,  # minutes
        "response_cost": 35000,  # dollars
        "media_sentiment": "Neutral",
        "social_impact": "Medium",
        "brand_impact": -8,  # percentage change
        "reputation_recovery_time": 45  # days
    }
    
    print_subsection("1. COMPREHENSIVE CRISIS ANALYSIS")
    
    # Conduct comprehensive analysis
    analysis = engine.conduct_comprehensive_analysis(
        crisis=crisis,
        response_data=response_data,
        analyst_id="senior_analyst_001"
    )
    
    print(f"✅ Analysis completed: {analysis.id}")
    print(f"   Overall Performance Score: {analysis.overall_performance_score:.1f}%")
    print(f"   Crisis Duration: {analysis.crisis_duration:.1f} hours")
    print(f"   Confidence Level: {analysis.confidence_level:.1f}%")
    
    print(f"\n📊 Response Metrics ({len(analysis.response_metrics)} metrics):")
    for metric in analysis.response_metrics:
        status = "✅" if metric.performance_score >= 80 else "⚠️" if metric.performance_score >= 60 else "❌"
        print(f"   {status} {metric.metric_name}: {metric.actual_value} {metric.measurement_unit} "
              f"(Score: {metric.performance_score:.1f}%)")
    
    print(f"\n💪 Strengths Identified ({len(analysis.strengths_identified)}):")
    for strength in analysis.strengths_identified:
        print(f"   ✅ {strength}")
    
    print(f"\n⚠️ Weaknesses Identified ({len(analysis.weaknesses_identified)}):")
    for weakness in analysis.weaknesses_identified:
        print(f"   ❌ {weakness}")
    
    print_subsection("2. LESSONS LEARNED IDENTIFICATION")
    
    # Identify lessons learned
    lessons = engine.identify_lessons_learned(
        crisis=crisis,
        response_data=response_data
    )
    
    print(f"✅ Identified {len(lessons)} lessons learned")
    
    for i, lesson in enumerate(lessons, 1):
        print(f"\n📚 Lesson {i}: {lesson.title}")
        print(f"   Category: {lesson.category.value}")
        print(f"   Description: {lesson.description}")
        print(f"   Impact: {lesson.impact_assessment}")
        print(f"   Status: {lesson.validation_status}")
    
    print_subsection("3. IMPROVEMENT RECOMMENDATIONS")
    
    # Generate improvement recommendations
    recommendations = engine.generate_improvement_recommendations(lessons)
    
    print(f"✅ Generated {len(recommendations)} improvement recommendations")
    
    for i, rec in enumerate(recommendations, 1):
        priority_icon = "🔴" if rec.priority.value == "critical" else "🟡" if rec.priority.value == "high" else "🟢"
        print(f"\n{priority_icon} Recommendation {i}: {rec.title}")
        print(f"   Priority: {rec.priority.value.upper()}")
        print(f"   Description: {rec.description}")
        print(f"   Expected Impact: {rec.expected_impact}")
        print(f"   Responsible Team: {rec.responsible_team}")
        print(f"   Target Completion: {rec.target_completion.strftime('%Y-%m-%d')}")
        print(f"   Status: {rec.implementation_status}")
    
    print_subsection("4. IMPACT ASSESSMENT")
    
    print("🎯 Stakeholder Impact:")
    for stakeholder, impact in analysis.stakeholder_impact.items():
        print(f"   • {stakeholder.title()}: {impact}")
    
    print("\n💼 Business Impact:")
    for aspect, impact in analysis.business_impact.items():
        if isinstance(impact, (int, float)) and aspect in ["revenue_impact", "response_cost"]:
            print(f"   • {aspect.replace('_', ' ').title()}: ${impact:,}")
        else:
            print(f"   • {aspect.replace('_', ' ').title()}: {impact}")
    
    print("\n🏢 Reputation Impact:")
    for aspect, impact in analysis.reputation_impact.items():
        print(f"   • {aspect.replace('_', ' ').title()}: {impact}")
    
    print_subsection("5. ANALYSIS REPORT GENERATION")
    
    # Generate analysis report
    report = engine.generate_analysis_report(analysis, "comprehensive")
    
    print(f"✅ Analysis report generated: {report.report_title}")
    print(f"   Format: {report.report_format}")
    print(f"   Generated: {report.generated_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📄 Executive Summary:")
    print(f"   {report.executive_summary}")
    
    print(f"\n📊 Detailed Findings:")
    print(f"   {report.detailed_findings}")
    
    print(f"\n💡 Recommendations Summary:")
    print(f"   {report.recommendations_summary}")
    
    print(f"\n📎 Appendices ({len(report.appendices)}):")
    for appendix in report.appendices:
        print(f"   • {appendix}")
    
    print_subsection("6. ANALYSIS INSIGHTS")
    
    # Calculate additional insights
    high_priority_recs = [r for r in recommendations if r.priority.value in ["critical", "high"]]
    avg_performance = analysis.overall_performance_score
    
    print(f"📈 Key Insights:")
    print(f"   • Overall crisis response performance: {avg_performance:.1f}%")
    print(f"   • High-priority improvement areas: {len(high_priority_recs)}")
    print(f"   • Crisis resolution efficiency: {'Excellent' if analysis.crisis_duration < 2 else 'Good' if analysis.crisis_duration < 4 else 'Needs Improvement'}")
    print(f"   • Stakeholder satisfaction: {analysis.stakeholder_impact.get('overall_satisfaction', 'N/A')}%")
    
    # Performance categorization
    if avg_performance >= 90:
        performance_category = "🏆 Exceptional Performance"
    elif avg_performance >= 80:
        performance_category = "✅ Strong Performance"
    elif avg_performance >= 70:
        performance_category = "⚠️ Adequate Performance"
    else:
        performance_category = "❌ Performance Needs Improvement"
    
    print(f"   • Performance Category: {performance_category}")
    
    print_subsection("7. CONTINUOUS IMPROVEMENT TRACKING")
    
    print("🔄 Improvement Tracking:")
    print(f"   • Total lessons learned: {len(lessons)}")
    print(f"   • Actionable recommendations: {len(recommendations)}")
    print(f"   • Critical/High priority items: {len(high_priority_recs)}")
    print(f"   • Estimated implementation timeline: 90 days")
    
    # Mock implementation progress
    implementation_progress = {
        "not_started": len([r for r in recommendations if r.implementation_status == "not_started"]),
        "in_progress": 0,
        "completed": 0
    }
    
    print(f"   • Implementation Status:")
    print(f"     - Not Started: {implementation_progress['not_started']}")
    print(f"     - In Progress: {implementation_progress['in_progress']}")
    print(f"     - Completed: {implementation_progress['completed']}")
    
    print_section("DEMO COMPLETED SUCCESSFULLY")
    print("🎉 Post-Crisis Analysis System demonstration completed!")
    print("\n📋 Summary:")
    print(f"   • Crisis analyzed: {crisis.id}")
    print(f"   • Performance score: {analysis.overall_performance_score:.1f}%")
    print(f"   • Lessons identified: {len(lessons)}")
    print(f"   • Recommendations generated: {len(recommendations)}")
    print(f"   • Analysis confidence: {analysis.confidence_level:.1f}%")
    
    return {
        "analysis": analysis,
        "lessons": lessons,
        "recommendations": recommendations,
        "report": report,
        "performance_score": avg_performance
    }


if __name__ == "__main__":
    asyncio.run(demo_post_crisis_analysis())