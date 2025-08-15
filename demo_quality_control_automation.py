"""
Demo script for Quality Control Automation system

This script demonstrates the autonomous quality control and validation capabilities
for all innovation lab processes.
"""

import asyncio
import json
from datetime import datetime
from scrollintel.engines.quality_control_automation import QualityControlAutomation

async def demo_quality_control_automation():
    """Demonstrate quality control automation capabilities"""
    print("🔍 Quality Control Automation Demo")
    print("=" * 50)
    
    # Initialize quality control system
    quality_control = QualityControlAutomation()
    print("✅ Quality Control Automation system initialized")
    
    # Demo 1: Research Quality Assessment
    print("\n📊 Demo 1: Research Quality Assessment")
    print("-" * 40)
    
    research_data = {
        "clarity_score": 0.85,
        "methodology_score": 0.9,
        "literature_score": 0.8,
        "power_analysis": 0.85,
        "reproducibility": 0.8,
        "testable": True,
        "peer_reviewed": True,
        "comprehensive_literature": True
    }
    
    research_assessment = await quality_control.assess_quality(
        process_id="research_001",
        process_type="research",
        process_data=research_data
    )
    
    print(f"Process ID: {research_assessment.process_id}")
    print(f"Overall Score: {research_assessment.overall_score:.3f}")
    print(f"Quality Level: {research_assessment.quality_level.value}")
    print(f"Metrics Count: {len(research_assessment.metrics)}")
    print(f"Issues: {len(research_assessment.issues)}")
    print(f"Recommendations: {len(research_assessment.recommendations)}")
    
    if research_assessment.issues:
        print("Issues found:")
        for issue in research_assessment.issues:
            print(f"  - {issue}")
    
    if research_assessment.recommendations:
        print("Recommendations:")
        for rec in research_assessment.recommendations:
            print(f"  - {rec}")
    
    # Demo 2: Experiment Quality Assessment
    print("\n🧪 Demo 2: Experiment Quality Assessment")
    print("-" * 40)
    
    experiment_data = {
        "design_score": 0.9,
        "control_score": 0.85,
        "sample_adequacy": 0.8,
        "precision_score": 0.95,
        "bias_score": 0.9,
        "has_control": True,
        "adequate_sample": True,
        "calibrated": True
    }
    
    experiment_assessment = await quality_control.assess_quality(
        process_id="experiment_001",
        process_type="experiment",
        process_data=experiment_data
    )
    
    print(f"Process ID: {experiment_assessment.process_id}")
    print(f"Overall Score: {experiment_assessment.overall_score:.3f}")
    print(f"Quality Level: {experiment_assessment.quality_level.value}")
    
    # Demo 3: Quality Standards Enforcement
    print("\n⚖️ Demo 3: Quality Standards Enforcement")
    print("-" * 40)
    
    # Test with good quality
    good_allowed = await quality_control.enforce_quality_standards(
        "research_001", research_assessment
    )
    print(f"Research process allowed: {good_allowed}")
    
    # Test with poor quality data
    poor_data = {
        "clarity_score": 0.4,
        "methodology_score": 0.3,
        "literature_score": 0.2,
        "power_analysis": 0.3,
        "reproducibility": 0.2,
        "testable": False,
        "peer_reviewed": False,
        "comprehensive_literature": False
    }
    
    poor_assessment = await quality_control.assess_quality(
        process_id="research_002",
        process_type="research",
        process_data=poor_data
    )
    
    poor_allowed = await quality_control.enforce_quality_standards(
        "research_002", poor_assessment
    )
    print(f"Poor quality process allowed: {poor_allowed}")
    print(f"Poor quality level: {poor_assessment.quality_level.value}")
    
    # Demo 4: Prototype Quality Assessment
    print("\n🔧 Demo 4: Prototype Quality Assessment")
    print("-" * 40)
    
    prototype_data = {
        "functionality_score": 0.85,
        "performance_score": 0.8,
        "reliability_score": 0.9,
        "usability_score": 0.75,
        "maintainability_score": 0.8,
        "core_functions_work": True,
        "meets_performance": True,
        "acceptable_failure_rate": True
    }
    
    prototype_assessment = await quality_control.assess_quality(
        process_id="prototype_001",
        process_type="prototype",
        process_data=prototype_data
    )
    
    print(f"Prototype Overall Score: {prototype_assessment.overall_score:.3f}")
    print(f"Prototype Quality Level: {prototype_assessment.quality_level.value}")
    
    # Demo 5: Quality Standards Overview
    print("\n📋 Demo 5: Quality Standards Overview")
    print("-" * 40)
    
    print("Available Quality Standards:")
    for standard_type, standard_def in quality_control.quality_standards.items():
        print(f"\n{standard_type.value.upper()}:")
        print(f"  Metrics: {', '.join(standard_def.metrics[:3])}...")
        print(f"  Rules: {len(standard_def.validation_rules)} validation rules")
    
    # Demo 6: Quality History
    print("\n📈 Demo 6: Quality History")
    print("-" * 40)
    
    print("Quality Assessment History:")
    for process_id, history in quality_control.quality_history.items():
        latest = history[-1]
        print(f"  {process_id}: {latest.quality_level.value} (score: {latest.overall_score:.3f})")
    
    # Demo 7: Quality Optimization
    print("\n⚡ Demo 7: Quality Process Optimization")
    print("-" * 40)
    
    optimization_results = await quality_control.optimize_quality_processes()
    print("Optimization Results:")
    print(f"  Threshold adjustments: {len(optimization_results.get('threshold_adjustments', {}))}")
    print(f"  Weight optimizations: {len(optimization_results.get('weight_optimizations', {}))}")
    print(f"  New metrics identified: {len(optimization_results.get('new_metrics', []))}")
    print(f"  Process improvements: {len(optimization_results.get('process_improvements', []))}")
    
    # Demo 8: Continuous Monitoring (brief demo)
    print("\n🔄 Demo 8: Continuous Monitoring")
    print("-" * 40)
    
    print("Starting continuous monitoring...")
    monitoring_task = asyncio.create_task(quality_control.start_continuous_monitoring())
    
    # Let it run briefly
    await asyncio.sleep(2)
    print(f"Monitoring active: {quality_control.monitoring_active}")
    
    # Stop monitoring
    quality_control.stop_continuous_monitoring()
    print("Monitoring stopped")
    
    # Wait for task to complete
    try:
        await asyncio.wait_for(monitoring_task, timeout=1.0)
    except asyncio.TimeoutError:
        monitoring_task.cancel()
    
    # Demo 9: Quality Metrics Analysis
    print("\n📊 Demo 9: Quality Metrics Analysis")
    print("-" * 40)
    
    # Analyze metrics from assessments
    all_scores = []
    quality_levels = {}
    
    for history in quality_control.quality_history.values():
        for assessment in history:
            all_scores.append(assessment.overall_score)
            level = assessment.quality_level.value
            quality_levels[level] = quality_levels.get(level, 0) + 1
    
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        print(f"Average Quality Score: {avg_score:.3f}")
        print(f"Total Assessments: {len(all_scores)}")
        print("Quality Level Distribution:")
        for level, count in quality_levels.items():
            print(f"  {level}: {count}")
    
    # Demo 10: Improvement Suggestions
    print("\n💡 Demo 10: Improvement Suggestions")
    print("-" * 40)
    
    print("Improvement Suggestions by Process:")
    for process_id, suggestions in quality_control.improvement_suggestions.items():
        if suggestions:
            print(f"\n{process_id}:")
            for suggestion in suggestions[:3]:  # Show first 3
                print(f"  - {suggestion}")
    
    print("\n🎉 Quality Control Automation Demo Complete!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(demo_quality_control_automation())