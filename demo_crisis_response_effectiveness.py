"""
Crisis Response Effectiveness Testing Demo

Demonstrates the comprehensive crisis response effectiveness testing and validation system.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

from scrollintel.engines.crisis_response_effectiveness_testing import (
    CrisisResponseEffectivenessTesting,
    EffectivenessMetric
)

async def demonstrate_crisis_response_effectiveness_testing():
    """Demonstrate comprehensive crisis response effectiveness testing"""
    print("🚨 Crisis Response Effectiveness Testing Demo")
    print("=" * 60)
    
    # Initialize the testing engine
    effectiveness_testing = CrisisResponseEffectivenessTesting()
    
    # Demo 1: Start effectiveness test
    print("\n1. Starting Crisis Response Effectiveness Test")
    print("-" * 50)
    
    crisis_scenario = "Major data breach affecting 100,000 customer records with potential regulatory implications"
    test_id = await effectiveness_testing.start_effectiveness_test(
        crisis_scenario=crisis_scenario,
        test_type="comprehensive"
    )
    
    print(f"✅ Started effectiveness test: {test_id}")
    print(f"📋 Crisis scenario: {crisis_scenario}")
    
    # Demo 2: Measure response speed
    print("\n2. Measuring Response Speed")
    print("-" * 50)
    
    detection_time = datetime.now()
    first_response_time = detection_time + timedelta(minutes=4)  # 4 minutes to first response
    full_response_time = detection_time + timedelta(minutes=25)  # 25 minutes to full response
    
    speed_score = await effectiveness_testing.measure_response_speed(
        test_id=test_id,
        detection_time=detection_time,
        first_response_time=first_response_time,
        full_response_time=full_response_time
    )
    
    print(f"⚡ Response Speed Score: {speed_score.score:.3f}")
    print(f"📊 First Response Time: {speed_score.details['detection_to_first_response_seconds']:.0f} seconds")
    print(f"📊 Full Response Time: {speed_score.details['detection_to_full_response_seconds']:.0f} seconds")
    print(f"🎯 Confidence Level: {speed_score.confidence_level:.1%}")
    
    # Demo 3: Measure decision quality
    print("\n3. Measuring Decision Quality")
    print("-" * 50)
    
    decisions_made = [
        {
            "id": "decision_1",
            "type": "immediate_containment",
            "description": "Immediately isolate affected systems",
            "timestamp": detection_time.isoformat()
        },
        {
            "id": "decision_2",
            "type": "stakeholder_notification",
            "description": "Notify legal team and prepare regulatory filings",
            "timestamp": (detection_time + timedelta(minutes=10)).isoformat()
        },
        {
            "id": "decision_3",
            "type": "customer_communication",
            "description": "Prepare customer notification and support response",
            "timestamp": (detection_time + timedelta(minutes=15)).isoformat()
        }
    ]
    
    decision_outcomes = [
        {
            "information_completeness": 0.9,
            "stakeholder_consideration": 0.85,
            "risk_assessment_accuracy": 0.8,
            "implementation_feasibility": 0.95,
            "outcome_effectiveness": 0.9
        },
        {
            "information_completeness": 0.8,
            "stakeholder_consideration": 0.9,
            "risk_assessment_accuracy": 0.85,
            "implementation_feasibility": 0.8,
            "outcome_effectiveness": 0.85
        },
        {
            "information_completeness": 0.75,
            "stakeholder_consideration": 0.95,
            "risk_assessment_accuracy": 0.7,
            "implementation_feasibility": 0.9,
            "outcome_effectiveness": 0.8
        }
    ]
    
    quality_score = await effectiveness_testing.measure_decision_quality(
        test_id=test_id,
        decisions_made=decisions_made,
        decision_outcomes=decision_outcomes
    )
    
    print(f"🧠 Decision Quality Score: {quality_score.score:.3f}")
    print(f"📊 Decisions Evaluated: {quality_score.details['decisions_evaluated']}")
    print(f"📈 Average Score: {quality_score.details['average_score']:.3f}")
    
    for i, decision_detail in enumerate(quality_score.details['decision_details']):
        print(f"   Decision {i+1}: {decision_detail['overall_score']:.3f} - {decisions_made[i]['description']}")
    
    # Demo 4: Measure communication effectiveness
    print("\n4. Measuring Communication Effectiveness")
    print("-" * 50)
    
    communications_sent = [
        {
            "id": "comm_1",
            "channel": "email",
            "audience": "affected_customers",
            "message": "Security incident notification and next steps",
            "timestamp": (detection_time + timedelta(hours=2)).isoformat()
        },
        {
            "id": "comm_2",
            "channel": "press_release",
            "audience": "media_public",
            "message": "Official statement on security incident",
            "timestamp": (detection_time + timedelta(hours=4)).isoformat()
        },
        {
            "id": "comm_3",
            "channel": "slack",
            "audience": "internal_team",
            "message": "Crisis response status and coordination updates",
            "timestamp": (detection_time + timedelta(minutes=30)).isoformat()
        }
    ]
    
    stakeholder_feedback = [
        {
            "communication_id": "comm_1",
            "clarity_rating": 0.85,
            "timeliness_rating": 0.8,
            "completeness_rating": 0.9,
            "appropriateness_rating": 0.85
        },
        {
            "communication_id": "comm_2",
            "clarity_rating": 0.9,
            "timeliness_rating": 0.75,
            "completeness_rating": 0.8,
            "appropriateness_rating": 0.9
        },
        {
            "communication_id": "comm_3",
            "clarity_rating": 0.95,
            "timeliness_rating": 0.9,
            "completeness_rating": 0.85,
            "appropriateness_rating": 0.9
        }
    ]
    
    comm_score = await effectiveness_testing.measure_communication_effectiveness(
        test_id=test_id,
        communications_sent=communications_sent,
        stakeholder_feedback=stakeholder_feedback
    )
    
    print(f"📢 Communication Effectiveness Score: {comm_score.score:.3f}")
    print(f"📊 Communications Evaluated: {comm_score.details['communications_evaluated']}")
    
    for comm_detail in comm_score.details['communication_details']:
        print(f"   {comm_detail['channel'].title()}: {comm_detail['overall_score']:.3f} "
              f"(Clarity: {comm_detail['clarity_score']:.2f}, "
              f"Timeliness: {comm_detail['timeliness_score']:.2f})")
    
    # Demo 5: Measure outcome success
    print("\n5. Measuring Outcome Success")
    print("-" * 50)
    
    crisis_objectives = [
        "Contain data breach within 2 hours",
        "Notify all affected customers within 24 hours",
        "File regulatory notifications within required timeframes",
        "Maintain customer trust and minimize churn",
        "Implement additional security measures to prevent recurrence"
    ]
    
    achieved_outcomes = [
        {
            "completion_rate": 0.95,
            "quality_rating": 0.9,
            "stakeholder_satisfaction": 0.85,
            "long_term_impact_score": 0.9
        },
        {
            "completion_rate": 1.0,
            "quality_rating": 0.85,
            "stakeholder_satisfaction": 0.8,
            "long_term_impact_score": 0.85
        },
        {
            "completion_rate": 1.0,
            "quality_rating": 0.95,
            "stakeholder_satisfaction": 0.9,
            "long_term_impact_score": 0.95
        },
        {
            "completion_rate": 0.8,
            "quality_rating": 0.75,
            "stakeholder_satisfaction": 0.7,
            "long_term_impact_score": 0.8
        },
        {
            "completion_rate": 0.9,
            "quality_rating": 0.85,
            "stakeholder_satisfaction": 0.85,
            "long_term_impact_score": 0.9
        }
    ]
    
    outcome_score = await effectiveness_testing.measure_outcome_success(
        test_id=test_id,
        crisis_objectives=crisis_objectives,
        achieved_outcomes=achieved_outcomes
    )
    
    print(f"🎯 Outcome Success Score: {outcome_score.score:.3f}")
    print(f"📊 Objectives Evaluated: {outcome_score.details['objectives_evaluated']}")
    
    for i, outcome_detail in enumerate(outcome_score.details['outcome_details']):
        print(f"   Objective {i+1}: {outcome_detail['objective_score']:.3f}")
        print(f"      {crisis_objectives[i]}")
        print(f"      Completion: {outcome_detail['completion_rate']:.1%}, "
              f"Quality: {outcome_detail['quality_rating']:.2f}")
    
    # Demo 6: Measure leadership effectiveness
    print("\n6. Measuring Leadership Effectiveness")
    print("-" * 50)
    
    leadership_actions = [
        {
            "type": "decision_making",
            "description": "Made rapid decision to activate crisis response team",
            "effectiveness_rating": 0.9
        },
        {
            "type": "communication",
            "description": "Provided clear direction to response team",
            "effectiveness_rating": 0.85
        },
        {
            "type": "stakeholder_management",
            "description": "Managed board and investor communications",
            "effectiveness_rating": 0.8
        },
        {
            "type": "team_coordination",
            "description": "Coordinated cross-functional response efforts",
            "effectiveness_rating": 0.9
        },
        {
            "type": "crisis_composure",
            "description": "Maintained calm and focused leadership throughout",
            "effectiveness_rating": 0.95
        }
    ]
    
    team_feedback = [
        {
            "leadership_clarity": 0.9,
            "decision_confidence": 0.85,
            "communication_effectiveness": 0.9
        },
        {
            "leadership_clarity": 0.85,
            "decision_confidence": 0.9,
            "communication_effectiveness": 0.85
        },
        {
            "leadership_clarity": 0.9,
            "decision_confidence": 0.8,
            "communication_effectiveness": 0.9
        }
    ]
    
    stakeholder_confidence = {
        "board_members": 0.85,
        "customers": 0.75,
        "employees": 0.9,
        "regulators": 0.8,
        "media": 0.7
    }
    
    leadership_score = await effectiveness_testing.measure_leadership_effectiveness(
        test_id=test_id,
        leadership_actions=leadership_actions,
        team_feedback=team_feedback,
        stakeholder_confidence=stakeholder_confidence
    )
    
    print(f"👑 Leadership Effectiveness Score: {leadership_score.score:.3f}")
    print(f"📊 Leadership Dimensions:")
    for dimension, score in leadership_score.details['leadership_dimensions'].items():
        print(f"   {dimension.replace('_', ' ').title()}: {score:.3f}")
    
    print(f"📊 Team Ratings:")
    for rating, score in leadership_score.details['team_ratings'].items():
        print(f"   {rating.replace('_', ' ').title()}: {score:.3f}")
    
    print(f"📊 Stakeholder Confidence:")
    for stakeholder, confidence in stakeholder_confidence.items():
        print(f"   {stakeholder.replace('_', ' ').title()}: {confidence:.3f}")
    
    # Demo 7: Complete effectiveness test
    print("\n7. Completing Effectiveness Test")
    print("-" * 50)
    
    completed_test = await effectiveness_testing.complete_effectiveness_test(test_id)
    
    print(f"✅ Test Completed: {completed_test.test_id}")
    print(f"⏱️  Duration: {(completed_test.end_time - completed_test.start_time).total_seconds():.0f} seconds")
    print(f"🏆 Overall Effectiveness Score: {completed_test.overall_score:.3f}")
    
    print(f"\n📊 Individual Metric Scores:")
    for score in completed_test.effectiveness_scores:
        print(f"   {score.metric.value.replace('_', ' ').title()}: {score.score:.3f}")
    
    print(f"\n💡 Recommendations ({len(completed_test.recommendations)}):")
    for i, recommendation in enumerate(completed_test.recommendations, 1):
        print(f"   {i}. {recommendation}")
    
    # Demo 8: Set baselines and benchmark
    print("\n8. Benchmarking Against Baselines")
    print("-" * 50)
    
    # Set baseline metrics
    baselines = {
        EffectivenessMetric.RESPONSE_SPEED: 0.7,
        EffectivenessMetric.DECISION_QUALITY: 0.75,
        EffectivenessMetric.COMMUNICATION_CLARITY: 0.8,
        EffectivenessMetric.OUTCOME_SUCCESS: 0.8,
        EffectivenessMetric.LEADERSHIP_EFFECTIVENESS: 0.75
    }
    
    effectiveness_testing.update_baseline_metrics(baselines)
    print("📊 Baseline metrics updated")
    
    # Benchmark the completed test
    benchmark_results = await effectiveness_testing.benchmark_against_baseline(test_id)
    
    print(f"\n🎯 Benchmark Results for Test: {benchmark_results['test_id']}")
    print(f"🏆 Overall Score: {benchmark_results['overall_score']:.3f}")
    
    for metric, comparison in benchmark_results['metric_comparisons'].items():
        improvement = comparison['improvement']
        improvement_pct = comparison['improvement_percentage']
        performance_level = comparison['performance_level']
        
        print(f"\n   {metric.replace('_', ' ').title()}:")
        print(f"      Current: {comparison['current_score']:.3f}")
        print(f"      Baseline: {comparison['baseline_score']:.3f}")
        print(f"      Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")
        print(f"      Performance Level: {performance_level.upper()}")
    
    # Demo 9: Export test results
    print("\n9. Exporting Test Results")
    print("-" * 50)
    
    exported_results = await effectiveness_testing.export_test_results(test_id)
    
    print(f"📄 Exported results for test: {exported_results['test_id']}")
    print(f"📊 Test duration: {exported_results['duration_seconds']:.0f} seconds")
    print(f"📈 Effectiveness scores: {len(exported_results['effectiveness_scores'])}")
    print(f"💡 Recommendations: {len(exported_results['recommendations'])}")
    
    # Save to file
    with open(f"crisis_effectiveness_test_{test_id}.json", "w") as f:
        json.dump(exported_results, f, indent=2, default=str)
    
    print(f"💾 Results saved to: crisis_effectiveness_test_{test_id}.json")
    
    # Demo 10: Analyze trends (simulate with multiple tests)
    print("\n10. Effectiveness Trends Analysis")
    print("-" * 50)
    
    # Create additional test history for trend analysis
    print("📈 Simulating historical test data...")
    
    for i in range(5):
        historical_test_id = await effectiveness_testing.start_effectiveness_test(
            crisis_scenario=f"Historical crisis scenario {i+1}",
            test_type="historical_simulation"
        )
        
        # Add some scores with slight variations
        base_score = 0.7 + (i * 0.05)  # Improving trend
        
        await effectiveness_testing.measure_response_speed(
            test_id=historical_test_id,
            detection_time=datetime.now() - timedelta(days=30-i*5),
            first_response_time=datetime.now() - timedelta(days=30-i*5) + timedelta(minutes=5),
            full_response_time=datetime.now() - timedelta(days=30-i*5) + timedelta(minutes=20)
        )
        
        await effectiveness_testing.complete_effectiveness_test(historical_test_id)
    
    # Analyze trends
    overall_trends = await effectiveness_testing.get_effectiveness_trends()
    speed_trends = await effectiveness_testing.get_effectiveness_trends(
        metric=EffectivenessMetric.RESPONSE_SPEED,
        time_period=timedelta(days=60)
    )
    
    print(f"\n📊 Overall Effectiveness Trends:")
    print(f"   Tests Analyzed: {overall_trends['tests_analyzed']}")
    print(f"   Average Score: {overall_trends['average_score']:.3f}")
    print(f"   Trend Direction: {overall_trends['trend'].upper()}")
    
    print(f"\n⚡ Response Speed Trends:")
    print(f"   Tests Analyzed: {speed_trends['tests_analyzed']}")
    print(f"   Average Score: {speed_trends['average_score']:.3f}")
    print(f"   Trend Direction: {speed_trends['trend'].upper()}")
    
    print("\n" + "=" * 60)
    print("🎉 Crisis Response Effectiveness Testing Demo Complete!")
    print("=" * 60)
    
    return {
        "test_completed": completed_test,
        "benchmark_results": benchmark_results,
        "exported_results": exported_results,
        "trends": {
            "overall": overall_trends,
            "response_speed": speed_trends
        }
    }

if __name__ == "__main__":
    # Run the demo
    results = asyncio.run(demonstrate_crisis_response_effectiveness_testing())
    
    print(f"\n📋 Demo Summary:")
    print(f"   Overall Effectiveness Score: {results['test_completed'].overall_score:.3f}")
    print(f"   Recommendations Generated: {len(results['test_completed'].recommendations)}")
    print(f"   Metrics Benchmarked: {len(results['benchmark_results']['metric_comparisons'])}")
    print(f"   Trend Analysis Period: {results['trends']['overall']['tests_analyzed']} tests")