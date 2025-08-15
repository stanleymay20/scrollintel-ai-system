#!/usr/bin/env python3
"""
Performance Monitoring System De
This demo showcases the real-time team performance tracking, issue identification,
and optimization capabilities during crisis situations.
"""

import asyncio
import json
from datetime import datetime
from scrollintel.engines.performance_monitoring_engine import PerformanceMonitoringEngine
from scrollintel.models.performance_monitoring_models import InterventionType, SupportType


async def main():
    """Main demo function"""
    print("🚨 Crisis Leadership Excellence - Performance Monitoring System Demo")
    print("=" * 70)
    
    # Initialize performance monitoring engine
    engine = PerformanceMonitoringEngine()
    
    # Demo scenario: Major system outage crisis
    crisis_id = "system_outage_001"
    team_members = ["alice_tech", "bob_ops", "carol_comm", "david_mgmt", "eve_support"]
    
    print(f"\n📊 CRISIS SCENARIO: Major System Outage")
    print(f"Crisis ID: {crisis_id}")
    print(f"Team Members: {', '.join(team_members)}")
    
    # Step 1: Track initial team performance
    print(f"\n🔍 Step 1: Tracking Initial Team Performance")
    print("-" * 50)
    
    team_overview = await engine.track_team_performance(crisis_id, team_members)
    
    print(f"Overall Performance Score: {team_overview.overall_performance_score:.1f}/100")
    print(f"Team Efficiency: {team_overview.team_efficiency:.1f}%")
    print(f"Collaboration Index: {team_overview.collaboration_index:.1f}%")
    print(f"Average Stress Level: {team_overview.stress_level_avg:.1f}/10")
    print(f"Task Completion Rate: {team_overview.task_completion_rate:.1f}%")
    print(f"Average Response Time: {team_overview.response_time_avg:.1f} minutes")
    
    print(f"\n👥 Individual Member Performance:")
    for member in team_overview.member_performances:
        status_emoji = {
            "excellent": "🟢",
            "good": "🟡", 
            "average": "🟠",
            "below_average": "🔴",
            "critical": "🚨"
        }.get(member.performance_status.value, "❓")
        
        print(f"  {status_emoji} {member.member_name}: {member.overall_score:.1f}/100 "
              f"(Stress: {member.stress_level:.1f}/10)")
    
    # Step 2: Identify performance issues
    print(f"\n⚠️  Step 2: Identifying Performance Issues")
    print("-" * 50)
    
    issues = await engine.identify_performance_issues(crisis_id)
    
    if issues:
        print(f"Found {len(issues)} performance issues:")
        for issue in issues:
            severity_emoji = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️"}.get(issue.severity, "❓")
            print(f"  {severity_emoji} {issue.issue_type}: {issue.description}")
            print(f"     Member: {issue.member_id} | Severity: {issue.severity}")
            print(f"     Impact: {issue.impact_assessment}")
    else:
        print("✅ No critical performance issues detected")
    
    # Step 3: Implement interventions
    print(f"\n🛠️  Step 3: Implementing Performance Interventions")
    print("-" * 50)
    
    # Implement coaching intervention for a team member
    intervention = await engine.implement_intervention(
        crisis_id, "alice_tech", InterventionType.COACHING
    )
    
    print(f"✅ Implemented {intervention.intervention_type.value} intervention:")
    print(f"   Target: {intervention.member_id}")
    print(f"   Description: {intervention.description}")
    print(f"   Expected Outcome: {intervention.expected_outcome}")
    print(f"   Status: {intervention.completion_status}")
    
    # Implement additional support
    additional_support = await engine.implement_intervention(
        crisis_id, "bob_ops", InterventionType.ADDITIONAL_SUPPORT
    )
    
    print(f"\n✅ Implemented {additional_support.intervention_type.value} intervention:")
    print(f"   Target: {additional_support.member_id}")
    print(f"   Description: {additional_support.description}")
    
    # Step 4: Provide support
    print(f"\n🤝 Step 4: Providing Team Member Support")
    print("-" * 50)
    
    # Provide technical support
    tech_support = await engine.provide_support(
        crisis_id, "alice_tech", SupportType.TECHNICAL_SUPPORT, "senior_engineer"
    )
    
    print(f"✅ Provided {tech_support.support_type.value}:")
    print(f"   Recipient: {tech_support.member_id}")
    print(f"   Provider: {tech_support.provider}")
    print(f"   Description: {tech_support.description}")
    
    # Provide emotional support
    emotional_support = await engine.provide_support(
        crisis_id, "carol_comm", SupportType.EMOTIONAL_SUPPORT, "crisis_counselor"
    )
    
    print(f"\n✅ Provided {emotional_support.support_type.value}:")
    print(f"   Recipient: {emotional_support.member_id}")
    print(f"   Provider: {emotional_support.provider}")
    print(f"   Description: {emotional_support.description}")
    
    # Step 5: Generate performance alerts
    print(f"\n🚨 Step 5: Generating Performance Alerts")
    print("-" * 50)
    
    alerts = await engine.generate_performance_alerts(crisis_id)
    
    if alerts:
        print(f"Generated {len(alerts)} performance alerts:")
        for alert in alerts:
            severity_emoji = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️"}.get(alert.severity, "❓")
            print(f"  {severity_emoji} {alert.alert_type}: {alert.message}")
            print(f"     Severity: {alert.severity} | Time: {alert.triggered_at.strftime('%H:%M:%S')}")
    else:
        print("✅ No critical alerts generated")
    
    # Step 6: Optimize team performance
    print(f"\n🎯 Step 6: Generating Performance Optimizations")
    print("-" * 50)
    
    optimizations = await engine.optimize_team_performance(crisis_id)
    
    if optimizations:
        print(f"Generated {len(optimizations)} optimization recommendations:")
        for opt in optimizations:
            priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(opt.priority_level, "❓")
            print(f"\n  {priority_emoji} {opt.target_area} Optimization:")
            print(f"     Current: {opt.current_performance:.1f} → Target: {opt.target_performance:.1f}")
            print(f"     Strategy: {opt.optimization_strategy}")
            print(f"     Priority: {opt.priority_level} | ETA: {opt.estimated_completion_time} minutes")
            print(f"     Expected Impact: {opt.expected_impact}")
    else:
        print("✅ Team performance is optimal - no optimizations needed")
    
    # Step 7: Track performance after interventions
    print(f"\n📈 Step 7: Tracking Performance After Interventions")
    print("-" * 50)
    
    # Simulate some time passing and re-track performance
    await asyncio.sleep(1)  # Simulate time passage
    updated_overview = await engine.track_team_performance(crisis_id, team_members)
    
    print(f"Updated Performance Metrics:")
    print(f"  Overall Score: {team_overview.overall_performance_score:.1f} → {updated_overview.overall_performance_score:.1f}")
    print(f"  Team Efficiency: {team_overview.team_efficiency:.1f}% → {updated_overview.team_efficiency:.1f}%")
    print(f"  Stress Level: {team_overview.stress_level_avg:.1f} → {updated_overview.stress_level_avg:.1f}")
    
    # Step 8: Generate comprehensive performance report
    print(f"\n📋 Step 8: Generating Comprehensive Performance Report")
    print("-" * 50)
    
    report = await engine.generate_performance_report(crisis_id, 1)  # 1 hour report
    
    print(f"Performance Report Generated:")
    print(f"  Report ID: {report.report_id}")
    print(f"  Time Period: {report.time_period_start.strftime('%H:%M')} - {report.time_period_end.strftime('%H:%M')}")
    print(f"  Report Type: {report.report_type}")
    
    print(f"\n📊 Key Performance Insights:")
    for insight in report.key_insights:
        print(f"  • {insight}")
    
    print(f"\n📈 Success Metrics:")
    for metric, value in report.success_metrics.items():
        print(f"  • {metric.replace('_', ' ').title()}: {value:.1f}")
    
    print(f"\n🎯 Optimization Recommendations:")
    for rec in report.recommendations:
        print(f"  • {rec.target_area}: {rec.optimization_strategy}")
    
    # Step 9: Show intervention and support history
    print(f"\n📚 Step 9: Intervention and Support History")
    print("-" * 50)
    
    if crisis_id in engine.intervention_history:
        interventions = engine.intervention_history[crisis_id]
        print(f"Total Interventions Implemented: {len(interventions)}")
        for intervention in interventions:
            print(f"  • {intervention.intervention_type.value} for {intervention.member_id}")
            print(f"    Status: {intervention.completion_status}")
    
    support_provisions = [s for s in engine.support_provisions.values() if s.crisis_id == crisis_id]
    print(f"\nTotal Support Provisions: {len(support_provisions)}")
    for support in support_provisions:
        print(f"  • {support.support_type.value} for {support.member_id} by {support.provider}")
    
    # Step 10: Performance monitoring summary
    print(f"\n✅ Step 10: Performance Monitoring Summary")
    print("-" * 50)
    
    print(f"Crisis Performance Monitoring Complete!")
    print(f"  • Team Members Monitored: {len(team_members)}")
    print(f"  • Performance Issues Identified: {len(issues)}")
    print(f"  • Interventions Implemented: {len(engine.intervention_history.get(crisis_id, []))}")
    print(f"  • Support Provisions Made: {len(support_provisions)}")
    print(f"  • Alerts Generated: {len(alerts)}")
    print(f"  • Optimizations Recommended: {len(optimizations)}")
    
    final_score = updated_overview.overall_performance_score
    if final_score >= 85:
        print(f"  🎉 Final Team Performance: EXCELLENT ({final_score:.1f}/100)")
    elif final_score >= 75:
        print(f"  ✅ Final Team Performance: GOOD ({final_score:.1f}/100)")
    else:
        print(f"  ⚠️  Final Team Performance: NEEDS IMPROVEMENT ({final_score:.1f}/100)")
    
    print(f"\n🚀 Performance monitoring system successfully demonstrated!")
    print(f"   Real-time tracking, issue identification, and optimization complete.")


if __name__ == "__main__":
    asyncio.run(main())