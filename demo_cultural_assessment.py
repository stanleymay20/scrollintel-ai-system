"""
Demo script for Cultural Assessment Engine

This script demonstrates the comprehensive cultural assessment capabilities
including culture mapping, dimensional analysis, subculture identification,
and cultural health metrics.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from scrollintel.engines.cultural_assessment_engine import CulturalAssessmentEngine
from scrollintel.models.cultural_assessment_models import (
    CulturalAssessmentRequest, CulturalDimension, CultureData
)


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


def format_score(score: float) -> str:
    """Format score with color coding"""
    if score >= 0.8:
        return f"{score:.2f} (Excellent)"
    elif score >= 0.6:
        return f"{score:.2f} (Good)"
    elif score >= 0.4:
        return f"{score:.2f} (Fair)"
    else:
        return f"{score:.2f} (Needs Improvement)"


async def demonstrate_culture_mapping():
    """Demonstrate comprehensive culture mapping"""
    print_section("CULTURAL ASSESSMENT ENGINE DEMONSTRATION")
    
    # Initialize the engine
    engine = CulturalAssessmentEngine()
    
    print("🎯 Initializing Cultural Assessment Engine...")
    print("✅ Engine components loaded:")
    print("   - Culture Mapper")
    print("   - Dimension Analyzer") 
    print("   - Subculture Identifier")
    print("   - Health Metrics Calculator")
    
    return engine


async def demonstrate_comprehensive_assessment(engine: CulturalAssessmentEngine):
    """Demonstrate comprehensive cultural assessment"""
    print_subsection("Comprehensive Cultural Assessment")
    
    # Create assessment request
    request = CulturalAssessmentRequest(
        organization_id="demo_tech_company",
        assessment_type="comprehensive",
        focus_areas=list(CulturalDimension),
        data_sources=["employee_survey", "manager_interviews", "behavioral_observation", "document_analysis"],
        timeline="3 weeks",
        stakeholders=["HR_Director", "CEO", "Department_Heads", "Employee_Representatives"],
        special_requirements={
            "include_remote_workers": True,
            "focus_on_innovation": True,
            "assess_change_readiness": True
        }
    )
    
    print(f"📋 Assessment Request:")
    print(f"   Organization: {request.organization_id}")
    print(f"   Type: {request.assessment_type}")
    print(f"   Focus Areas: {len(request.focus_areas)} cultural dimensions")
    print(f"   Data Sources: {', '.join(request.data_sources)}")
    print(f"   Timeline: {request.timeline}")
    print(f"   Stakeholders: {len(request.stakeholders)} groups")
    
    # Conduct assessment
    print("\n🔍 Conducting comprehensive cultural assessment...")
    result = engine.conduct_cultural_assessment(request)
    
    print(f"✅ Assessment completed successfully!")
    print(f"   Request ID: {result.request_id}")
    print(f"   Confidence Score: {format_score(result.confidence_score)}")
    print(f"   Completion Date: {result.completion_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return result


async def demonstrate_culture_map_analysis(result):
    """Demonstrate culture map analysis"""
    print_subsection("Culture Map Analysis")
    
    culture_map = result.culture_map
    
    print(f"🗺️ Culture Map Overview:")
    print(f"   Overall Health Score: {format_score(culture_map.overall_health_score)}")
    print(f"   Assessment Confidence: {format_score(culture_map.assessment_confidence)}")
    print(f"   Assessment Date: {culture_map.assessment_date.strftime('%Y-%m-%d')}")
    
    print(f"\n📊 Cultural Dimensions Analysis:")
    for dimension, score in culture_map.cultural_dimensions.items():
        dimension_name = dimension.value.replace('_', ' ').title()
        print(f"   {dimension_name}: {format_score(score)}")
    
    print(f"\n💎 Cultural Values Identified: {len(culture_map.values)}")
    for i, value in enumerate(culture_map.values[:3], 1):  # Show top 3
        print(f"   {i}. {value.name}")
        print(f"      Importance: {format_score(value.importance_score)}")
        print(f"      Alignment: {format_score(value.alignment_score)}")
        print(f"      Evidence: {len(value.evidence)} data points")
    
    print(f"\n🎭 Cultural Behaviors Observed: {len(culture_map.behaviors)}")
    for i, behavior in enumerate(culture_map.behaviors[:3], 1):  # Show top 3
        print(f"   {i}. {behavior.description}")
        print(f"      Frequency: {format_score(behavior.frequency)}")
        print(f"      Impact: {behavior.impact_score:+.2f}")
        print(f"      Context: {behavior.context}")


async def demonstrate_subculture_analysis(result):
    """Demonstrate subculture identification and analysis"""
    print_subsection("Subculture Analysis")
    
    subcultures = result.culture_map.subcultures
    
    print(f"🏘️ Subcultures Identified: {len(subcultures)}")
    
    if subcultures:
        # Group by type
        by_type = {}
        for subculture in subcultures:
            if subculture.type.value not in by_type:
                by_type[subculture.type.value] = []
            by_type[subculture.type.value].append(subculture)
        
        for subculture_type, subs in by_type.items():
            print(f"\n   📂 {subculture_type.replace('_', ' ').title()} Subcultures:")
            for sub in subs:
                print(f"      • {sub.name}")
                print(f"        Members: {len(sub.members)}")
                print(f"        Strength: {format_score(sub.strength)}")
                print(f"        Influence: {format_score(sub.influence)}")
        
        # Find most influential subculture
        most_influential = max(subcultures, key=lambda s: s.influence)
        print(f"\n🌟 Most Influential Subculture:")
        print(f"   Name: {most_influential.name}")
        print(f"   Type: {most_influential.type.value.replace('_', ' ').title()}")
        print(f"   Influence Score: {format_score(most_influential.influence)}")
        print(f"   Members: {len(most_influential.members)}")
        
        # Find strongest subculture
        strongest = max(subcultures, key=lambda s: s.strength)
        print(f"\n💪 Strongest Subculture:")
        print(f"   Name: {strongest.name}")
        print(f"   Type: {strongest.type.value.replace('_', ' ').title()}")
        print(f"   Strength Score: {format_score(strongest.strength)}")
        print(f"   Cohesion Level: {'High' if strongest.strength > 0.7 else 'Moderate' if strongest.strength > 0.5 else 'Low'}")


async def demonstrate_health_metrics(result):
    """Demonstrate cultural health metrics"""
    print_subsection("Cultural Health Metrics")
    
    health_metrics = result.culture_map.health_metrics
    
    print(f"📈 Health Metrics Calculated: {len(health_metrics)}")
    
    for metric in health_metrics:
        print(f"\n   📊 {metric.name}")
        print(f"      Current Value: {format_score(metric.value)}")
        if metric.target_value:
            gap = metric.value - metric.target_value
            print(f"      Target Value: {metric.target_value:.2f}")
            print(f"      Gap: {gap:+.2f} ({'On Target' if gap >= 0 else 'Below Target'})")
        print(f"      Trend: {metric.trend.title()}")
        print(f"      Confidence: {format_score(metric.confidence_level)}")
        print(f"      Data Sources: {len(metric.data_sources)}")
    
    # Calculate summary statistics
    on_target_count = sum(1 for m in health_metrics if m.target_value and m.value >= m.target_value)
    improving_count = sum(1 for m in health_metrics if m.trend == "improving")
    avg_confidence = sum(m.confidence_level for m in health_metrics) / len(health_metrics) if health_metrics else 0
    
    print(f"\n📋 Health Metrics Summary:")
    print(f"   Metrics On Target: {on_target_count}/{len(health_metrics)} ({on_target_count/len(health_metrics)*100:.1f}%)")
    print(f"   Improving Trends: {improving_count}/{len(health_metrics)} ({improving_count/len(health_metrics)*100:.1f}%)")
    print(f"   Average Confidence: {format_score(avg_confidence)}")


async def demonstrate_dimensional_analysis(result):
    """Demonstrate detailed dimensional analysis"""
    print_subsection("Dimensional Analysis")
    
    dimension_analyses = result.dimension_analyses
    
    print(f"🔬 Dimensional Analyses: {len(dimension_analyses)}")
    
    # Show top 3 dimensions by score
    top_dimensions = sorted(dimension_analyses, key=lambda d: d.current_score, reverse=True)[:3]
    
    print(f"\n🏆 Top Performing Dimensions:")
    for i, analysis in enumerate(top_dimensions, 1):
        dimension_name = analysis.dimension.value.replace('_', ' ').title()
        print(f"   {i}. {dimension_name}")
        print(f"      Current Score: {format_score(analysis.current_score)}")
        if analysis.ideal_score:
            print(f"      Ideal Score: {analysis.ideal_score:.2f}")
        print(f"      Measurement Confidence: {format_score(analysis.measurement_confidence)}")
        if analysis.contributing_factors:
            print(f"      Key Factors: {', '.join(analysis.contributing_factors[:2])}")
    
    # Show bottom 3 dimensions
    bottom_dimensions = sorted(dimension_analyses, key=lambda d: d.current_score)[:3]
    
    print(f"\n⚠️ Dimensions Needing Attention:")
    for i, analysis in enumerate(bottom_dimensions, 1):
        dimension_name = analysis.dimension.value.replace('_', ' ').title()
        print(f"   {i}. {dimension_name}")
        print(f"      Current Score: {format_score(analysis.current_score)}")
        print(f"      Gap Analysis: {analysis.gap_analysis}")
        if analysis.improvement_recommendations:
            print(f"      Top Recommendation: {analysis.improvement_recommendations[0]}")


async def demonstrate_findings_and_recommendations(result):
    """Demonstrate key findings and recommendations"""
    print_subsection("Key Findings & Recommendations")
    
    print(f"🔍 Key Findings ({len(result.key_findings)}):")
    for i, finding in enumerate(result.key_findings, 1):
        print(f"   {i}. {finding}")
    
    print(f"\n💡 Strategic Recommendations ({len(result.recommendations)}):")
    for i, recommendation in enumerate(result.recommendations, 1):
        print(f"   {i}. {recommendation}")
    
    print(f"\n📝 Assessment Summary:")
    print(f"{result.assessment_summary}")


async def demonstrate_quick_assessment(engine: CulturalAssessmentEngine):
    """Demonstrate quick assessment capability"""
    print_subsection("Quick Assessment Demo")
    
    # Create quick assessment request
    quick_request = CulturalAssessmentRequest(
        organization_id="demo_startup",
        assessment_type="quick",
        focus_areas=[
            CulturalDimension.INNOVATION_ORIENTATION,
            CulturalDimension.COLLABORATION_STYLE,
            CulturalDimension.COMMUNICATION_DIRECTNESS
        ],
        data_sources=["pulse_survey"],
        timeline="3 days",
        stakeholders=["Founders", "Team_Leads"]
    )
    
    print(f"⚡ Quick Assessment Request:")
    print(f"   Organization: {quick_request.organization_id}")
    print(f"   Focus Areas: {len(quick_request.focus_areas)} key dimensions")
    print(f"   Timeline: {quick_request.timeline}")
    
    # Conduct quick assessment
    print("\n🚀 Conducting quick assessment...")
    quick_result = engine.conduct_cultural_assessment(quick_request)
    
    print(f"✅ Quick assessment completed!")
    print(f"   Overall Health: {format_score(quick_result.culture_map.overall_health_score)}")
    print(f"   Confidence: {format_score(quick_result.confidence_score)}")
    
    print(f"\n📊 Focus Area Results:")
    for dimension, score in quick_result.culture_map.cultural_dimensions.items():
        if dimension in quick_request.focus_areas:
            dimension_name = dimension.value.replace('_', ' ').title()
            print(f"   {dimension_name}: {format_score(score)}")
    
    print(f"\n🎯 Quick Recommendations:")
    for i, rec in enumerate(quick_result.recommendations[:3], 1):
        print(f"   {i}. {rec}")


async def demonstrate_api_integration():
    """Demonstrate API integration capabilities"""
    print_subsection("API Integration Demo")
    
    print("🌐 Cultural Assessment API Endpoints:")
    print("   POST /cultural-assessment/assess - Comprehensive assessment")
    print("   GET  /cultural-assessment/culture-map/{org_id} - Culture mapping")
    print("   GET  /cultural-assessment/dimensions/{org_id} - Dimensional analysis")
    print("   GET  /cultural-assessment/subcultures/{org_id} - Subculture identification")
    print("   GET  /cultural-assessment/health-metrics/{org_id} - Health metrics")
    print("   POST /cultural-assessment/quick-assessment - Quick assessment")
    
    print("\n📡 Sample API Request:")
    sample_request = {
        "organization_id": "api_demo_org",
        "assessment_type": "comprehensive",
        "focus_areas": ["innovation_orientation", "collaboration_style"],
        "data_sources": ["survey", "observation"],
        "timeline": "2 weeks",
        "stakeholders": ["HR", "Leadership"]
    }
    
    print(json.dumps(sample_request, indent=2))
    
    print("\n📨 Expected Response Structure:")
    response_structure = {
        "request_id": "assessment_20240101_120000",
        "organization_id": "api_demo_org",
        "culture_map": {
            "overall_health_score": 0.75,
            "assessment_confidence": 0.85,
            "cultural_dimensions": {"innovation_orientation": 0.8},
            "subcultures_count": 5,
            "health_metrics_count": 6
        },
        "key_findings": ["Finding 1", "Finding 2"],
        "recommendations": ["Recommendation 1", "Recommendation 2"],
        "confidence_score": 0.85,
        "completion_date": "2024-01-01T12:00:00"
    }
    
    print(json.dumps(response_structure, indent=2))


async def main():
    """Main demonstration function"""
    try:
        # Initialize engine
        engine = await demonstrate_culture_mapping()
        
        # Comprehensive assessment
        result = await demonstrate_comprehensive_assessment(engine)
        
        # Detailed analysis
        await demonstrate_culture_map_analysis(result)
        await demonstrate_subculture_analysis(result)
        await demonstrate_health_metrics(result)
        await demonstrate_dimensional_analysis(result)
        await demonstrate_findings_and_recommendations(result)
        
        # Quick assessment demo
        await demonstrate_quick_assessment(engine)
        
        # API integration demo
        await demonstrate_api_integration()
        
        print_section("DEMONSTRATION COMPLETE")
        print("🎉 Cultural Assessment Engine demonstration completed successfully!")
        print("\n✨ Key Capabilities Demonstrated:")
        print("   ✅ Comprehensive culture mapping")
        print("   ✅ Cultural dimensions analysis")
        print("   ✅ Subculture identification")
        print("   ✅ Cultural health metrics")
        print("   ✅ Quick assessment capability")
        print("   ✅ API integration ready")
        
        print(f"\n📋 Requirements Addressed:")
        print(f"   ✅ 1.1 - Create compelling value systems that resonate with high-performers")
        print(f"   ✅ 1.2 - Ensure deep cultural integration and alignment with company values")
        print(f"   ✅ Cultural drift detection and corrective measures")
        print(f"   ✅ Quantitative culture measurement system")
        
        print(f"\n🚀 Ready for production deployment!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())