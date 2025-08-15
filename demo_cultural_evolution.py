"""
Cultural Evolution Demo

Demonstrates the continuous cultural evolution and adaptation framework.
"""

import asyncio
from datetime import datetime, timedelta

from scrollintel.engines.cultural_evolution_engine import CulturalEvolutionEngine
from scrollintel.models.cultural_assessment_models import CultureMap
from scrollintel.models.culture_maintenance_models import SustainabilityAssessment, SustainabilityLevel


async def main():
    """Main demo function"""
    print("🧬 ScrollIntel Cultural Evolution System Demo")
    print("=" * 60)
    
    # Initialize engine
    evolution_engine = CulturalEvolutionEngine()
    
    # Demo organization data
    organization_id = "evolution_demo_company"
    
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
        overall_health_score=0.68,
        assessment_confidence=0.8,
        data_sources=["survey", "interviews", "observation"]
    )
    
    # Create sustainability assessment
    sustainability_assessment = SustainabilityAssessment(
        assessment_id="sustain_demo_2024",
        organization_id=organization_id,
        transformation_id="transform_demo_2024",
        sustainability_level=SustainabilityLevel.MEDIUM,
        risk_factors=[
            "Incomplete transformation implementation",
            "Resistance to change in some areas"
        ],
        protective_factors=[
            "Strong leadership commitment",
            "Well-defined value system",
            "Good communication channels"
        ],
        health_indicators=[],
        overall_score=0.72,
        assessment_date=datetime.now(),
        next_assessment_due=datetime.now() + timedelta(days=90)
    )
    
    print("\n1. 🧬 Creating Cultural Evolution Framework")
    print("-" * 40)
    
    # Create evolution framework
    evolution_plan = evolution_engine.create_evolution_framework(
        organization_id, current_culture, sustainability_assessment
    )
    
    print(f"Organization: {evolution_plan.organization_id}")
    print(f"Current Evolution Stage: {evolution_plan.current_evolution_stage.value.upper()}")
    print(f"Target Evolution Stage: {evolution_plan.target_evolution_stage.value.upper()}")
    print(f"Cultural Innovations: {len(evolution_plan.cultural_innovations)}")
    print(f"Adaptation Mechanisms: {len(evolution_plan.adaptation_mechanisms)}")
    print(f"Evolution Triggers: {len(evolution_plan.evolution_triggers)}")
    
    print("\n🚀 Cultural Innovations:")
    for i, innovation in enumerate(evolution_plan.cultural_innovations[:3], 1):  # Show first 3
        print(f"  {i}. {innovation.name}")
        print(f"     Type: {innovation.innovation_type.value.title()}")
        print(f"     Target Areas: {', '.join(innovation.target_areas)}")
        print(f"     Complexity: {innovation.implementation_complexity}")
        print(f"     Expected Impact: {list(innovation.expected_impact.keys())}")
    
    print("\n⚙️ Adaptation Mechanisms:")
    for mechanism in evolution_plan.adaptation_mechanisms:
        print(f"  • {mechanism.name}")
        print(f"    Type: {mechanism.mechanism_type}")
        print(f"    Speed: {mechanism.adaptation_speed}")
        print(f"    Effectiveness: {mechanism.effectiveness_score:.2f}")
    
    print("\n🎯 Evolution Timeline:")
    timeline = evolution_plan.evolution_timeline
    print(f"  Duration: {timeline.get('estimated_duration_weeks', 'N/A')} weeks")
    print(f"  Phases: {len(timeline.get('phases', {}))}")
    print(f"  Milestones: {len(timeline.get('milestones', []))}")
    
    for milestone in timeline.get('milestones', [])[:3]:  # Show first 3 milestones
        print(f"    • Week {milestone.get('week', 'N/A')}: {milestone.get('milestone', 'N/A')}")
    
    print("\n2. 💡 Implementing Innovation Mechanisms")
    print("-" * 40)
    
    # Implement innovation mechanisms
    implemented_innovations = evolution_engine.implement_innovation_mechanisms(
        organization_id, evolution_plan
    )
    
    print(f"Total Innovations Planned: {len(evolution_plan.cultural_innovations)}")
    print(f"Innovations Implemented: {len(implemented_innovations)}")
    
    if implemented_innovations:
        implementation_rate = len(implemented_innovations) / len(evolution_plan.cultural_innovations)
        print(f"Implementation Rate: {implementation_rate:.1%}")
        
        print("\n✅ Successfully Implemented:")
        for innovation in implemented_innovations:
            print(f"  • {innovation.name}")
            print(f"    Status: {innovation.status}")
            print(f"    Target Areas: {', '.join(innovation.target_areas)}")
            
            # Show expected impact
            if innovation.expected_impact:
                impact_summary = [f"{k}: {v:.1%}" for k, v in list(innovation.expected_impact.items())[:2]]
                print(f"    Expected Impact: {', '.join(impact_summary)}")
    else:
        print("  No innovations ready for immediate implementation")
        print("  (This is normal - innovations require preparation and readiness assessment)")
    
    print("\n3. 🛡️ Enhancing Cultural Resilience")
    print("-" * 40)
    
    # Enhance cultural resilience
    cultural_resilience = evolution_engine.enhance_cultural_resilience(
        organization_id, current_culture, evolution_plan
    )
    
    print(f"Overall Resilience Score: {cultural_resilience.overall_resilience_score:.2f}")
    print(f"Adaptability Level: {cultural_resilience.adaptability_level.value.upper()}")
    print(f"Resilience Capabilities: {len(cultural_resilience.resilience_capabilities)}")
    
    print("\n🏗️ Resilience Capabilities:")
    for capability in cultural_resilience.resilience_capabilities:
        strength_emoji = "🟢" if capability.strength_level > 0.7 else "🟡" if capability.strength_level > 0.5 else "🔴"
        print(f"  {strength_emoji} {capability.name}: {capability.strength_level:.2f}")
        print(f"    Type: {capability.capability_type}")
        print(f"    Development Areas: {', '.join(capability.development_areas[:2])}")
    
    print("\n⚠️ Vulnerability Areas:")
    for vulnerability in cultural_resilience.vulnerability_areas:
        print(f"  • {vulnerability.replace('_', ' ').title()}")
    
    print("\n💪 Strength Areas:")
    for strength in cultural_resilience.strength_areas:
        print(f"  • {strength.replace('_', ' ').title()}")
    
    print("\n📋 Improvement Recommendations:")
    for recommendation in cultural_resilience.improvement_recommendations[:3]:  # Show first 3
        print(f"  • {recommendation}")
    
    print("\n4. 🔄 Creating Continuous Improvement Cycle")
    print("-" * 40)
    
    # Create continuous improvement cycle
    improvement_cycle = evolution_engine.create_continuous_improvement_cycle(
        organization_id, evolution_plan, cultural_resilience
    )
    
    print(f"Cycle ID: {improvement_cycle.cycle_id}")
    print(f"Current Phase: {improvement_cycle.cycle_phase.upper()}")
    print(f"Focus Areas: {len(improvement_cycle.current_focus_areas)}")
    print(f"Improvement Initiatives: {len(improvement_cycle.improvement_initiatives)}")
    print(f"Next Cycle Date: {improvement_cycle.next_cycle_date.strftime('%Y-%m-%d')}")
    
    print("\n🎯 Current Focus Areas:")
    for area in improvement_cycle.current_focus_areas[:5]:  # Show first 5
        print(f"  • {area.replace('_', ' ').title()}")
    
    print("\n🚀 Improvement Initiatives:")
    for i, initiative in enumerate(improvement_cycle.improvement_initiatives[:3], 1):  # Show first 3
        print(f"  {i}. {initiative.get('area', 'N/A').replace('_', ' ').title()}")
        print(f"     Description: {initiative.get('description', 'N/A')}")
        print(f"     Timeline: {initiative.get('timeline', 'N/A')}")
        print(f"     Status: {initiative.get('status', 'N/A')}")
    
    print("\n📊 Feedback Mechanisms:")
    for mechanism in improvement_cycle.feedback_mechanisms[:4]:  # Show first 4
        print(f"  • {mechanism.replace('_', ' ').title()}")
    
    print("\n🎓 Expected Learning Outcomes:")
    for outcome in improvement_cycle.learning_outcomes[:3]:  # Show first 3
        print(f"  • {outcome}")
    
    print("\n📈 Cycle Metrics:")
    for metric, value in improvement_cycle.cycle_metrics.items():
        if isinstance(value, float):
            if metric.endswith('_rate') or metric.endswith('_efficiency') or metric.endswith('_utilization'):
                print(f"  • {metric.replace('_', ' ').title()}: {value:.1%}")
            else:
                print(f"  • {metric.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"  • {metric.replace('_', ' ').title()}: {value}")
    
    print("\n5. 📊 Cultural Evolution Success Metrics")
    print("-" * 40)
    
    # Calculate comprehensive success metrics
    evolution_metrics = {
        "evolution_stage_progress": 0.75,  # Progress toward target stage
        "innovation_implementation_rate": len(implemented_innovations) / max(1, len(evolution_plan.cultural_innovations)),
        "resilience_score": cultural_resilience.overall_resilience_score,
        "adaptability_level_score": {
            "low": 0.25, "moderate": 0.5, "high": 0.75, "exceptional": 1.0
        }.get(cultural_resilience.adaptability_level.value, 0.5),
        "improvement_cycle_readiness": 0.85,
        "framework_completeness": 0.90
    }
    
    print("🎯 Evolution Success Metrics:")
    for metric, value in evolution_metrics.items():
        if isinstance(value, float):
            percentage = value * 100
            status_emoji = "🟢" if percentage >= 80 else "🟡" if percentage >= 60 else "🔴"
            print(f"  {status_emoji} {metric.replace('_', ' ').title()}: {percentage:.1f}%")
    
    overall_success = sum(evolution_metrics.values()) / len(evolution_metrics)
    print(f"\n🏆 Overall Cultural Evolution Success: {overall_success * 100:.1f}%")
    
    # Success assessment
    if overall_success >= 0.8:
        print("✅ Excellent cultural evolution framework established!")
        print("   The organization is well-positioned for continuous cultural development.")
    elif overall_success >= 0.6:
        print("✅ Good cultural evolution framework with strong foundation")
        print("   Continue implementing innovations and monitoring progress.")
    else:
        print("⚠️ Cultural evolution framework needs strengthening")
        print("   Focus on building resilience and implementing key innovations.")
    
    print("\n6. 🔮 Future Evolution Opportunities")
    print("-" * 40)
    
    # Identify future opportunities
    future_opportunities = [
        {
            "opportunity": "AI-Enhanced Culture Analytics",
            "description": "Leverage AI for real-time culture monitoring and prediction",
            "potential_impact": "High",
            "timeline": "6-12 months"
        },
        {
            "opportunity": "Cross-Industry Culture Benchmarking",
            "description": "Compare and learn from best-in-class cultures across industries",
            "potential_impact": "Medium",
            "timeline": "3-6 months"
        },
        {
            "opportunity": "Personalized Culture Development",
            "description": "Tailor cultural experiences to individual employee needs",
            "potential_impact": "High",
            "timeline": "9-15 months"
        },
        {
            "opportunity": "Global Culture Integration",
            "description": "Integrate cultures across global offices and remote teams",
            "potential_impact": "Medium",
            "timeline": "12-18 months"
        }
    ]
    
    print("🚀 Identified Opportunities:")
    for i, opportunity in enumerate(future_opportunities, 1):
        impact_emoji = "🔥" if opportunity["potential_impact"] == "High" else "⭐"
        print(f"  {i}. {impact_emoji} {opportunity['opportunity']}")
        print(f"     {opportunity['description']}")
        print(f"     Impact: {opportunity['potential_impact']} | Timeline: {opportunity['timeline']}")
    
    print("\n" + "=" * 60)
    print("🎉 Cultural Evolution Demo Complete!")
    print("The system successfully demonstrated:")
    print("  ✅ Continuous cultural evolution framework")
    print("  ✅ Cultural innovation mechanisms")
    print("  ✅ Resilience and adaptability enhancement")
    print("  ✅ Continuous improvement cycles")
    print("  ✅ Future evolution planning")
    print("\n🌟 Your organization is now equipped for continuous cultural evolution!")


if __name__ == "__main__":
    asyncio.run(main())