#!/usr/bin/env python3
"""
Organizational Resilience System Demo

Demonstrates the comprehensive organizational resilience assessment, enhancement,
and monitoring capabilities of ScrollIntel.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from scrollintel.engines.organizational_resilience_engine import OrganizationalResilienceEngine
from scrollintel.models.organizational_resilience_models import (
    ResilienceCategory, ResilienceMetricType
)


async def demonstrate_resilience_assessment():
    """Demonstrate organizational resilience assessment"""
    print("=" * 80)
    print("ORGANIZATIONAL RESILIENCE ASSESSMENT DEMO")
    print("=" * 80)
    
    engine = OrganizationalResilienceEngine()
    organization_id = "demo_org_001"
    
    try:
        # Conduct comprehensive assessment
        print(f"\n🔍 Conducting resilience assessment for organization: {organization_id}")
        assessment = await engine.assess_organizational_resilience(organization_id)
        
        print(f"\n📊 ASSESSMENT RESULTS:")
        print(f"   Overall Resilience Level: {assessment.overall_resilience_level.value.upper()}")
        print(f"   Confidence Score: {assessment.confidence_score:.2f}")
        
        print(f"\n📈 CATEGORY SCORES:")
        for category, score in assessment.category_scores.items():
            print(f"   {category.value.title()}: {score:.2f}")
        
        print(f"\n💪 STRENGTHS:")
        for strength in assessment.strengths:
            print(f"   • {strength}")
        
        print(f"\n⚠️  VULNERABILITIES:")
        for vulnerability in assessment.vulnerabilities:
            print(f"   • {vulnerability}")
        
        print(f"\n🎯 IMPROVEMENT AREAS:")
        for area in assessment.improvement_areas:
            print(f"   • {area}")
        
        return assessment
        
    except Exception as e:
        print(f"❌ Error in resilience assessment: {str(e)}")
        return None


async def demonstrate_strategy_development(assessment):
    """Demonstrate resilience strategy development"""
    print("\n" + "=" * 80)
    print("RESILIENCE STRATEGY DEVELOPMENT DEMO")
    print("=" * 80)
    
    if not assessment:
        print("❌ No assessment available for strategy development")
        return None
    
    engine = OrganizationalResilienceEngine()
    
    try:
        # Develop resilience strategy
        print(f"\n🎯 Developing resilience strategy based on assessment...")
        strategic_priorities = ["operational_excellence", "technology_modernization"]
        
        strategy = await engine.develop_resilience_strategy(
            assessment=assessment,
            strategic_priorities=strategic_priorities
        )
        
        print(f"\n📋 STRATEGY OVERVIEW:")
        print(f"   Strategy Name: {strategy.strategy_name}")
        print(f"   Target Categories: {[cat.value for cat in strategy.target_categories]}")
        
        print(f"\n🎯 OBJECTIVES:")
        for objective in strategy.objectives:
            print(f"   • {objective}")
        
        print(f"\n🚀 KEY INITIATIVES:")
        for initiative in strategy.initiatives[:5]:  # Show first 5
            print(f"   • {initiative}")
        
        print(f"\n📊 RESOURCE REQUIREMENTS:")
        for resource, requirement in strategy.resource_requirements.items():
            print(f"   {resource.title()}: {requirement}")
        
        print(f"\n📈 SUCCESS METRICS:")
        for metric in strategy.success_metrics[:3]:  # Show first 3
            print(f"   • {metric}")
        
        print(f"\n⚠️  RISK FACTORS:")
        for risk in strategy.risk_factors:
            print(f"   • {risk}")
        
        print(f"\n🎯 EXPECTED IMPACT:")
        for category, impact in strategy.expected_impact.items():
            print(f"   {category.value.title()}: +{impact:.1%}")
        
        return strategy
        
    except Exception as e:
        print(f"❌ Error in strategy development: {str(e)}")
        return None


async def demonstrate_resilience_monitoring():
    """Demonstrate continuous resilience monitoring"""
    print("\n" + "=" * 80)
    print("CONTINUOUS RESILIENCE MONITORING DEMO")
    print("=" * 80)
    
    engine = OrganizationalResilienceEngine()
    organization_id = "demo_org_001"
    
    try:
        # Monitor resilience
        print(f"\n📊 Monitoring resilience for organization: {organization_id}")
        monitoring_data = await engine.monitor_resilience_continuously(
            organization_id=organization_id,
            monitoring_frequency="daily"
        )
        
        print(f"\n📈 CURRENT METRICS:")
        for metric_type, value in monitoring_data.metric_values.items():
            print(f"   {metric_type.value.replace('_', ' ').title()}: {value:.2f}")
        
        print(f"\n📊 TREND ANALYSIS:")
        for category, analysis in monitoring_data.trend_analysis.items():
            if isinstance(analysis, dict):
                trend = analysis.get('trend', 'unknown')
                print(f"   {category.title()}: {trend}")
        
        if monitoring_data.alert_triggers:
            print(f"\n🚨 ALERTS:")
            for alert in monitoring_data.alert_triggers:
                print(f"   • {alert}")
        else:
            print(f"\n✅ No critical alerts detected")
        
        if monitoring_data.anomaly_detection:
            print(f"\n🔍 ANOMALIES DETECTED:")
            for anomaly in monitoring_data.anomaly_detection:
                print(f"   • {anomaly}")
        else:
            print(f"\n✅ No anomalies detected")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for recommendation in monitoring_data.recommendations:
            print(f"   • {recommendation}")
        
        return monitoring_data
        
    except Exception as e:
        print(f"❌ Error in resilience monitoring: {str(e)}")
        return None


async def demonstrate_continuous_improvement(monitoring_data_list):
    """Demonstrate continuous improvement implementation"""
    print("\n" + "=" * 80)
    print("CONTINUOUS IMPROVEMENT DEMO")
    print("=" * 80)
    
    if not monitoring_data_list:
        print("❌ No monitoring data available for improvement analysis")
        return None
    
    engine = OrganizationalResilienceEngine()
    organization_id = "demo_org_001"
    
    try:
        # Implement continuous improvement
        print(f"\n🔄 Implementing continuous improvement for: {organization_id}")
        improvements = await engine.implement_continuous_improvement(
            organization_id=organization_id,
            monitoring_data=monitoring_data_list,
            improvement_cycle="quarterly"
        )
        
        print(f"\n📋 IMPROVEMENT RECOMMENDATIONS ({len(improvements)} total):")
        
        for i, improvement in enumerate(improvements, 1):
            print(f"\n   {i}. {improvement.improvement_type}")
            print(f"      Category: {improvement.category.value.title()}")
            print(f"      Priority: {improvement.priority.upper()}")
            print(f"      Description: {improvement.description}")
            print(f"      Timeline: {improvement.estimated_timeline}")
            
            if improvement.implementation_steps:
                print(f"      Steps:")
                for step in improvement.implementation_steps[:3]:  # Show first 3
                    print(f"        • {step}")
            
            if improvement.expected_benefits:
                print(f"      Benefits:")
                for benefit in improvement.expected_benefits[:2]:  # Show first 2
                    print(f"        • {benefit}")
        
        return improvements
        
    except Exception as e:
        print(f"❌ Error in continuous improvement: {str(e)}")
        return None


async def demonstrate_report_generation(assessment, monitoring_data_list, improvements):
    """Demonstrate comprehensive report generation"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESILIENCE REPORT DEMO")
    print("=" * 80)
    
    if not all([assessment, monitoring_data_list, improvements]):
        print("❌ Insufficient data for report generation")
        return None
    
    engine = OrganizationalResilienceEngine()
    organization_id = "demo_org_001"
    
    try:
        # Generate comprehensive report
        print(f"\n📄 Generating comprehensive resilience report...")
        report = await engine.generate_resilience_report(
            organization_id=organization_id,
            assessment=assessment,
            monitoring_data=monitoring_data_list,
            improvements=improvements
        )
        
        print(f"\n📊 EXECUTIVE SUMMARY:")
        print(f"{report.executive_summary}")
        
        print(f"\n📈 OVERALL RESILIENCE SCORE: {report.overall_resilience_score:.2f}")
        
        print(f"\n📊 CATEGORY BREAKDOWN:")
        for category, breakdown in report.category_breakdown.items():
            if isinstance(breakdown, dict):
                score = breakdown.get('current_score', 0)
                trend = breakdown.get('trend', 'unknown')
                potential = breakdown.get('improvement_potential', 0)
                print(f"   {category.value.title()}:")
                print(f"     Current Score: {score:.2f}")
                print(f"     Trend: {trend}")
                print(f"     Improvement Potential: {potential:.2f}")
        
        print(f"\n📈 TREND ANALYSIS:")
        for key, value in report.trend_analysis.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n📊 BENCHMARK COMPARISON:")
        for benchmark, value in report.benchmark_comparison.items():
            print(f"   {benchmark.replace('_', ' ').title()}: {value:.2f}")
        
        print(f"\n🔍 KEY FINDINGS:")
        for finding in report.key_findings:
            print(f"   • {finding}")
        
        print(f"\n🎯 ACTION PLAN:")
        for action in report.action_plan:
            print(f"   • {action}")
        
        print(f"\n📅 Next Assessment: {report.next_assessment_date.strftime('%Y-%m-%d')}")
        
        return report
        
    except Exception as e:
        print(f"❌ Error in report generation: {str(e)}")
        return None


async def demonstrate_resilience_scenarios():
    """Demonstrate resilience assessment under different scenarios"""
    print("\n" + "=" * 80)
    print("RESILIENCE SCENARIO ANALYSIS DEMO")
    print("=" * 80)
    
    engine = OrganizationalResilienceEngine()
    
    scenarios = [
        {
            "name": "High-Growth Startup",
            "org_id": "startup_001",
            "scope": [ResilienceCategory.OPERATIONAL, ResilienceCategory.FINANCIAL]
        },
        {
            "name": "Enterprise Corporation",
            "org_id": "enterprise_001",
            "scope": [ResilienceCategory.TECHNOLOGICAL, ResilienceCategory.STRATEGIC]
        },
        {
            "name": "Non-Profit Organization",
            "org_id": "nonprofit_001",
            "scope": [ResilienceCategory.HUMAN_CAPITAL, ResilienceCategory.CULTURAL]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🏢 SCENARIO: {scenario['name']}")
        print(f"   Organization ID: {scenario['org_id']}")
        print(f"   Assessment Scope: {[cat.value for cat in scenario['scope']]}")
        
        try:
            assessment = await engine.assess_organizational_resilience(
                organization_id=scenario['org_id'],
                assessment_scope=scenario['scope']
            )
            
            print(f"   Overall Level: {assessment.overall_resilience_level.value.upper()}")
            print(f"   Confidence: {assessment.confidence_score:.2f}")
            print(f"   Key Strengths: {len(assessment.strengths)}")
            print(f"   Vulnerabilities: {len(assessment.vulnerabilities)}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")


async def main():
    """Main demo function"""
    print("🚀 Starting Organizational Resilience System Demo")
    print(f"⏰ Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Resilience Assessment
    assessment = await demonstrate_resilience_assessment()
    
    # Step 2: Strategy Development
    strategy = await demonstrate_strategy_development(assessment)
    
    # Step 3: Continuous Monitoring
    monitoring_data = await demonstrate_resilience_monitoring()
    monitoring_data_list = [monitoring_data] if monitoring_data else []
    
    # Step 4: Continuous Improvement
    improvements = await demonstrate_continuous_improvement(monitoring_data_list)
    
    # Step 5: Report Generation
    report = await demonstrate_report_generation(
        assessment, monitoring_data_list, improvements
    )
    
    # Step 6: Scenario Analysis
    await demonstrate_resilience_scenarios()
    
    print("\n" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    print(f"✅ Resilience Assessment: {'Completed' if assessment else 'Failed'}")
    print(f"✅ Strategy Development: {'Completed' if strategy else 'Failed'}")
    print(f"✅ Continuous Monitoring: {'Completed' if monitoring_data else 'Failed'}")
    print(f"✅ Continuous Improvement: {'Completed' if improvements else 'Failed'}")
    print(f"✅ Report Generation: {'Completed' if report else 'Failed'}")
    
    print(f"\n🎉 Organizational Resilience System Demo completed!")
    print(f"⏰ Demo finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main())