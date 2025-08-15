"""
Culture Maintenance Demo

Demonstrates the cultural sustainability assessment and maintenance framework.
"""

import asyncio
from datetime import datetime, timedelta

from scrollintel.engines.culture_maintenance_engine import CultureMaintenanceEngine
from scrollintel.models.cultural_assessment_models import CultureMap, CulturalTransformation


async def main():
    """Main demo function"""
    print("🏛️ ScrollIntel Culture Maintenance System Demo")
    print("=" * 60)
    
    # Initialize engine
    maintenance_engine = CultureMaintenanceEngine()
    
    # Demo organization data
    organization_id = "demo_company"
    
    # Create current culture
    current_culture = CultureMap(
        organization_id=organization_id,
        assessment_date=datetime.now(),
        cultural_dimensions={},
        values=[],
        behaviors=[],
        norms=[],
        subcultures=[],
        health_metrics=[],
        overall_health_score=0.72,
        assessment_confidence=0.8,
        data_sources=["survey", "interviews"]
    )
    
    # Create target culture
    target_culture = CultureMap(
        organization_id=organization_id,
        assessment_date=datetime.now(),
        cultural_dimensions={},
        values=[],
        behaviors=[],
        norms=[],
        subcultures=[],
        health_metrics=[],
        overall_health_score=0.85,
        assessment_confidence=0.8,
        data_sources=["survey", "interviews"]
    )
    
    # Create transformation
    transformation = CulturalTransformation(
        id="culture_transformation_2024",
        organization_id=organization_id,
        current_culture={},
        target_culture={},
        vision={},
        roadmap={},
        interventions=[],
        progress=0.75,
        start_date=datetime.now() - timedelta(days=120),
        target_completion=datetime.now() + timedelta(days=60)
    )
    
    print("\n1. 📊 Assessing Cultural Sustainability")
    print("-" * 40)
    
    # Assess cultural sustainability
    sustainability_assessment = maintenance_engine.assess_cultural_sustainability(
        organization_id, transformation, current_culture
    )
    
    print(f"Organization: {sustainability_assessment.organization_id}")
    print(f"Sustainability Level: {sustainability_assessment.sustainability_level.value.upper()}")
    print(f"Overall Score: {sustainability_assessment.overall_score:.2f}")
    print(f"Risk Factors: {len(sustainability_assessment.risk_factors)}")
    print(f"Protective Factors: {len(sustainability_assessment.protective_factors)}")
    
    print("\n📈 Health Indicators:")
    for indicator in sustainability_assessment.health_indicators:
        print(f"  • {indicator.name}: {indicator.current_value:.2f} → {indicator.target_value:.2f} ({indicator.trend})")
    
    print("\n⚠️ Risk Factors:")
    for risk in sustainability_assessment.risk_factors:
        print(f"  • {risk}")
    
    print("\n🛡️ Protective Factors:")
    for factor in sustainability_assessment.protective_factors:
        print(f"  • {factor}")
    
    print("\n2. 🎯 Developing Maintenance Strategies")
    print("-" * 40)
    
    # Develop maintenance strategies
    maintenance_strategies = maintenance_engine.develop_maintenance_strategy(
        organization_id, sustainability_assessment, target_culture
    )
    
    print(f"Generated {len(maintenance_strategies)} maintenance strategies:")
    
    for i, strategy in enumerate(maintenance_strategies, 1):
        print(f"\n  Strategy {i}: {strategy.target_culture_elements[0] if strategy.target_culture_elements else 'General'}")
        print(f"    Activities: {len(strategy.maintenance_activities)}")
        print(f"    Success Metrics: {len(strategy.success_metrics)}")
        print(f"    Review Frequency: {strategy.review_frequency}")
        
        print("    Key Activities:")
        for activity in strategy.maintenance_activities[:2]:  # Show first 2 activities
            print(f"      • {activity.get('description', 'N/A')} ({activity.get('frequency', 'N/A')})")
    
    print("\n3. 📋 Creating Maintenance Plan")
    print("-" * 40)
    
    # Create maintenance plan
    maintenance_plan = maintenance_engine.create_maintenance_plan(
        organization_id, sustainability_assessment, maintenance_strategies
    )
    
    print(f"Plan ID: {maintenance_plan.plan_id}")
    print(f"Status: {maintenance_plan.status.value.upper()}")
    print(f"Strategies: {len(maintenance_plan.maintenance_strategies)}")
    print(f"Intervention Triggers: {len(maintenance_plan.intervention_triggers)}")
    
    print("\n📊 Monitoring Framework:")
    framework = maintenance_plan.monitoring_framework
    print(f"  • Indicators: {len(framework.get('indicators', []))}")
    print(f"  • Measurement Frequency: {framework.get('measurement_frequency', 'N/A')}")
    print(f"  • Reporting Schedule: {framework.get('reporting_schedule', 'N/A')}")
    
    print("\n💰 Resource Allocation:")
    resources = maintenance_plan.resource_allocation
    print(f"  • Time Investment: {resources.get('total_time_hours_monthly', 0)} hours/month")
    print(f"  • Budget Requirement: {resources.get('budget_requirement', 'N/A')}")
    print(f"  • Personnel Needed: {resources.get('personnel_needed', 0)}")
    
    print("\n🚨 Intervention Triggers:")
    for trigger in maintenance_plan.intervention_triggers[:3]:  # Show first 3 triggers
        print(f"  • {trigger.get('trigger_type', 'N/A')}: {trigger.get('condition', 'N/A')}")
    
    print("\n4. 📈 Long-term Health Monitoring")
    print("-" * 40)
    
    # Monitor long-term health
    monitoring_result = maintenance_engine.monitor_long_term_health(
        organization_id, maintenance_plan, monitoring_period_days=90
    )
    
    print(f"Monitoring Period: {monitoring_result.monitoring_period['start'].strftime('%Y-%m-%d')} to {monitoring_result.monitoring_period['end'].strftime('%Y-%m-%d')}")
    
    print("\n📊 Health Trends:")
    for metric, values in monitoring_result.health_trends.items():
        if values:
            trend = "↗️" if values[-1] > values[0] else "↘️" if values[-1] < values[0] else "➡️"
            print(f"  • {metric.title()}: {values[0]:.2f} → {values[-1]:.2f} {trend}")
    
    print("\n📈 Sustainability Metrics:")
    for metric, value in list(monitoring_result.sustainability_metrics.items())[:5]:  # Show first 5 metrics
        print(f"  • {metric.replace('_', ' ').title()}: {value:.3f}")
    
    print("\n⚠️ Current Risk Indicators:")
    if monitoring_result.risk_indicators:
        for risk in monitoring_result.risk_indicators:
            print(f"  • {risk}")
    else:
        print("  • No significant risks detected")
    
    print("\n💡 Recommendations:")
    for recommendation in monitoring_result.recommendations:
        print(f"  • {recommendation}")
    
    print("\n🎯 Next Actions:")
    for action in monitoring_result.next_actions:
        priority_emoji = "🔴" if action.get('priority') == 'high' else "🟡" if action.get('priority') == 'medium' else "🟢"
        print(f"  {priority_emoji} {action.get('action', 'N/A').replace('_', ' ').title()}")
        print(f"    Timeline: {action.get('timeline', 'N/A')}")
        print(f"    Responsible: {action.get('responsible', 'N/A')}")
    
    print("\n5. 🎯 Culture Maintenance Success Metrics")
    print("-" * 40)
    
    # Calculate success metrics
    success_metrics = {
        "sustainability_improvement": (sustainability_assessment.overall_score - 0.6) * 100,
        "risk_reduction": max(0, 5 - len(sustainability_assessment.risk_factors)) * 20,
        "protective_factor_strength": len(sustainability_assessment.protective_factors) * 25,
        "monitoring_coverage": len(maintenance_plan.monitoring_framework.get('indicators', [])) * 10,
        "strategy_comprehensiveness": len(maintenance_strategies) * 15
    }
    
    print("📊 Success Metrics:")
    for metric, value in success_metrics.items():
        print(f"  • {metric.replace('_', ' ').title()}: {value:.1f}%")
    
    overall_success = sum(success_metrics.values()) / len(success_metrics)
    print(f"\n🏆 Overall Maintenance Framework Success: {overall_success:.1f}%")
    
    if overall_success >= 80:
        print("✅ Excellent culture maintenance framework established!")
    elif overall_success >= 60:
        print("✅ Good culture maintenance framework with room for improvement")
    else:
        print("⚠️ Culture maintenance framework needs strengthening")
    
    print("\n" + "=" * 60)
    print("🎉 Culture Maintenance Demo Complete!")
    print("The system successfully demonstrated:")
    print("  ✅ Cultural sustainability assessment")
    print("  ✅ Maintenance strategy development")
    print("  ✅ Comprehensive maintenance planning")
    print("  ✅ Long-term health monitoring")
    print("  ✅ Continuous optimization recommendations")


if __name__ == "__main__":
    asyncio.run(main())