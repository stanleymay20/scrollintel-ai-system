"""
Demo: Cultural Transformation Testing and Validation Framework

This demo showcases the comprehensive testing and validation framework
for cultural transformation initiatives.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from scrollintel.engines.cultural_transformation_validator import CulturalTransformationValidator
from scrollintel.engines.cultural_transformation_outcome_validator import CulturalTransformationOutcomeValidator
from scrollintel.engines.cultural_sustainability_tester import CulturalSustainabilityTester
from scrollintel.engines.transformation_roi_assessor import TransformationROIAssessor
from scrollintel.models.transformation_outcome_models import (
    TransformationOutcome, TransformationStatus, create_transformation_outcome
)


def demo_cultural_assessment_validation():
    """Demo cultural assessment accuracy validation"""
    print("=" * 60)
    print("CULTURAL ASSESSMENT VALIDATION DEMO")
    print("=" * 60)
    
    validator = CulturalTransformationValidator()
    
    # Sample assessment results
    assessment_results = {
        "cultural_dimensions": {
            "innovation": 0.82,
            "collaboration": 0.78,
            "accountability": 0.75,
            "transparency": 0.80,
            "adaptability": 0.77,
            "performance_orientation": 0.83
        },
        "confidence_score": 0.87,
        "data_points": list(range(1, 101)),  # 100 data points
        "assessment_method": "comprehensive_survey_and_observation",
        "sample_size": 450
    }
    
    # Validate assessment accuracy
    result = validator.validate_cultural_assessment_accuracy(assessment_results)
    
    print(f"Assessment Validation Results:")
    print(f"  Test Name: {result.test_name}")
    print(f"  Passed: {result.passed}")
    print(f"  Accuracy Score: {result.score:.3f}")
    print(f"  Confidence Level: {result.confidence:.3f}")
    print(f"  Details:")
    for key, value in result.details.items():
        if isinstance(value, (int, float)):
            print(f"    {key}: {value:.3f}")
        else:
            print(f"    {key}: {value}")
    
    return result


def demo_transformation_strategy_validation():
    """Demo transformation strategy effectiveness validation"""
    print("\n" + "=" * 60)
    print("TRANSFORMATION STRATEGY VALIDATION DEMO")
    print("=" * 60)
    
    validator = CulturalTransformationValidator()
    
    # Sample strategy and outcome data
    strategy_data = {
        "goals": [
            {"id": "innovation_boost", "target": 0.85},
            {"id": "collaboration_improvement", "target": 0.80},
            {"id": "engagement_increase", "target": 0.88}
        ],
        "planned_resources": 500000,
        "investment": 650000,
        "timeline_months": 18
    }
    
    outcome_data = {
        "achievements": [
            {"goal_id": "innovation_boost", "score": 0.87},
            {"goal_id": "collaboration_improvement", "score": 0.82},
            {"goal_id": "engagement_increase", "score": 0.90}
        ],
        "actual_resources": 580000,
        "benefits": 850000,
        "impact_score": 0.85,
        "timeline_adherence": 0.92
    }
    
    # Validate strategy effectiveness
    result = validator.validate_transformation_strategy_effectiveness(strategy_data, outcome_data)
    
    print(f"Strategy Validation Results:")
    print(f"  Test Name: {result.test_name}")
    print(f"  Passed: {result.passed}")
    print(f"  Effectiveness Score: {result.score:.3f}")
    print(f"  Confidence Level: {result.confidence:.3f}")
    print(f"  Key Metrics:")
    for key, value in result.details.items():
        if isinstance(value, (int, float)):
            print(f"    {key}: {value:.3f}")
        elif isinstance(value, str):
            print(f"    {key}: {value}")
    
    return result


def demo_transformation_success_measurement():
    """Demo transformation success measurement"""
    print("\n" + "=" * 60)
    print("TRANSFORMATION SUCCESS MEASUREMENT DEMO")
    print("=" * 60)
    
    outcome_validator = CulturalTransformationOutcomeValidator()
    
    # Create sample transformation outcome
    transformation_data = {
        "transformation_id": "trans_demo_001",
        "organization_id": "org_demo_001",
        "start_date": datetime.now() - timedelta(days=365),
        "end_date": datetime.now() - timedelta(days=30),
        "status": "completed",
        "target_metrics": {
            "cultural_health": 0.80,
            "employee_engagement": 0.85,
            "performance_improvement": 0.15,
            "innovation_index": 0.75
        },
        "achieved_metrics": {
            "cultural_health": 0.84,
            "employee_engagement": 0.89,
            "performance_improvement": 0.18,
            "innovation_index": 0.78
        },
        "success_criteria": [
            {"metric": "cultural_health", "threshold": 0.75, "weight": 0.3},
            {"metric": "employee_engagement", "threshold": 0.80, "weight": 0.3},
            {"metric": "performance_improvement", "threshold": 0.10, "weight": 0.2},
            {"metric": "innovation_index", "threshold": 0.70, "weight": 0.2}
        ]
    }
    
    transformation_outcome = create_transformation_outcome(transformation_data)
    
    # Measure transformation success
    success_assessment = outcome_validator.measure_transformation_success(transformation_outcome)
    
    print(f"Transformation Success Assessment:")
    print(f"  Overall Success Score: {success_assessment.overall_success_score:.3f}")
    print(f"  Criteria Met: {success_assessment.criteria_met_percentage:.1%}")
    print(f"  Success Level: {success_assessment.success_level}")
    print(f"  Exceeded Targets: {', '.join(success_assessment.exceeded_targets)}")
    if success_assessment.underperformed_areas:
        print(f"  Underperformed Areas: {', '.join(success_assessment.underperformed_areas)}")
    print(f"  Key Success Factors:")
    for factor in success_assessment.success_factors:
        print(f"    - {factor}")
    
    return success_assessment


def demo_cultural_sustainability_testing():
    """Demo cultural sustainability testing"""
    print("\n" + "=" * 60)
    print("CULTURAL SUSTAINABILITY TESTING DEMO")
    print("=" * 60)
    
    sustainability_tester = CulturalSustainabilityTester()
    
    # Sample sustainability data
    sustainability_data = {
        "transformation_completion": datetime.now() - timedelta(days=180),
        "measurement_points": [
            {"date": datetime.now() - timedelta(days=150), "cultural_health": 0.84, "engagement": 0.89},
            {"date": datetime.now() - timedelta(days=120), "cultural_health": 0.83, "engagement": 0.88},
            {"date": datetime.now() - timedelta(days=90), "cultural_health": 0.85, "engagement": 0.90},
            {"date": datetime.now() - timedelta(days=60), "cultural_health": 0.82, "engagement": 0.87},
            {"date": datetime.now() - timedelta(days=30), "cultural_health": 0.84, "engagement": 0.89},
            {"date": datetime.now(), "cultural_health": 0.83, "engagement": 0.88}
        ],
        "reinforcement_mechanisms": [
            {"type": "leadership_modeling", "strength": 0.85, "consistency": 0.92},
            {"type": "recognition_systems", "strength": 0.78, "consistency": 0.85},
            {"type": "performance_integration", "strength": 0.88, "consistency": 0.87},
            {"type": "communication_reinforcement", "strength": 0.80, "consistency": 0.90}
        ]
    }
    
    # Measure sustainability
    sustainability_metrics = sustainability_tester.measure_sustainability(sustainability_data)
    
    print(f"Cultural Sustainability Assessment:")
    print(f"  Stability Score: {sustainability_metrics.stability_score:.3f}")
    print(f"  Trend Consistency: {sustainability_metrics.trend_consistency:.3f}")
    print(f"  Reinforcement Strength: {sustainability_metrics.reinforcement_strength:.3f}")
    print(f"  Decay Resistance: {sustainability_metrics.decay_resistance:.3f}")
    print(f"  Sustainability Level: {sustainability_metrics.sustainability_level}")
    if sustainability_metrics.risk_factors:
        print(f"  Risk Factors:")
        for risk in sustainability_metrics.risk_factors:
            print(f"    - {risk}")
    
    # Test drift detection
    drift_data = {
        "baseline_culture": {"innovation": 0.84, "collaboration": 0.89, "accountability": 0.78},
        "current_culture": {"innovation": 0.81, "collaboration": 0.87, "accountability": 0.75},
        "measurement_history": [
            {"date": datetime.now() - timedelta(days=90), "innovation": 0.83, "collaboration": 0.88, "accountability": 0.77},
            {"date": datetime.now() - timedelta(days=60), "innovation": 0.82, "collaboration": 0.87, "accountability": 0.76},
            {"date": datetime.now() - timedelta(days=30), "innovation": 0.81, "collaboration": 0.87, "accountability": 0.75}
        ]
    }
    
    drift_assessment = sustainability_tester.detect_cultural_drift(drift_data)
    
    print(f"\nCultural Drift Assessment:")
    print(f"  Drift Detected: {drift_assessment.drift_detected}")
    print(f"  Drift Severity: {drift_assessment.drift_severity}")
    print(f"  Drift Rate: {drift_assessment.drift_rate:.3f}")
    if drift_assessment.affected_dimensions:
        print(f"  Affected Dimensions: {', '.join(drift_assessment.affected_dimensions)}")
    if drift_assessment.recommended_interventions:
        print(f"  Recommended Interventions:")
        for intervention in drift_assessment.recommended_interventions:
            print(f"    - {intervention}")
    
    return sustainability_metrics, drift_assessment


def demo_transformation_roi_assessment():
    """Demo transformation ROI assessment"""
    print("\n" + "=" * 60)
    print("TRANSFORMATION ROI ASSESSMENT DEMO")
    print("=" * 60)
    
    roi_assessor = TransformationROIAssessor()
    
    # Sample ROI data
    roi_data = {
        "investment_data": {
            "direct_costs": {
                "consulting_fees": 300000,
                "training_programs": 180000,
                "technology_tools": 90000,
                "internal_resources": 250000
            },
            "indirect_costs": {
                "opportunity_cost": 120000,
                "productivity_loss": 60000,
                "change_management": 100000
            },
            "timeline": 15  # months
        },
        "benefit_data": {
            "quantifiable_benefits": {
                "productivity_improvement": 220000,  # annual
                "retention_savings": 150000,  # annual
                "reduced_turnover_costs": 110000,  # annual
                "innovation_revenue": 280000  # annual
            },
            "qualitative_benefits": [
                {"benefit": "improved_reputation", "value_estimate": 75000},
                {"benefit": "enhanced_agility", "value_estimate": 100000},
                {"benefit": "better_decision_making", "value_estimate": 85000}
            ],
            "measurement_period": 30  # months
        }
    }
    
    # Calculate financial ROI
    roi_assessment = roi_assessor.calculate_financial_roi(roi_data)
    
    print(f"Financial ROI Assessment:")
    print(f"  ROI Percentage: {roi_assessment.roi_percentage:.1f}%")
    print(f"  Payback Period: {roi_assessment.payback_period:.1f} months")
    print(f"  Net Present Value: ${roi_assessment.net_present_value:,.0f}")
    print(f"  Benefit-Cost Ratio: {roi_assessment.benefit_cost_ratio:.2f}")
    print(f"  Total Investment: ${roi_assessment.total_investment:,.0f}")
    print(f"  Total Benefits: ${roi_assessment.total_benefits:,.0f}")
    print(f"  Assessment Confidence: {roi_assessment.assessment_confidence:.3f}")
    
    # Assess comprehensive value
    value_assessment = roi_assessor.assess_comprehensive_value(roi_data)
    
    print(f"\nComprehensive Value Assessment:")
    print(f"  Total Value Score: {value_assessment.total_value_score:.3f}")
    print(f"  Tangible Value: ${value_assessment.tangible_value:,.0f}")
    print(f"  Intangible Value: ${value_assessment.intangible_value:,.0f}")
    print(f"  Value Sustainability: {value_assessment.value_sustainability:.3f}")
    print(f"  Top Value Drivers:")
    sorted_drivers = sorted(value_assessment.value_drivers, 
                          key=lambda x: x.get("annual_value", 0), reverse=True)
    for driver in sorted_drivers[:3]:
        print(f"    - {driver['name']}: ${driver['annual_value']:,.0f} ({driver['contribution_percentage']:.1f}%)")
    
    return roi_assessment, value_assessment


def demo_comprehensive_validation_suite():
    """Demo comprehensive validation suite"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE VALIDATION SUITE DEMO")
    print("=" * 60)
    
    validator = CulturalTransformationValidator()
    
    # Sample comprehensive transformation data
    transformation_data = {
        "assessment_results": {
            "cultural_dimensions": {
                "innovation": 0.84,
                "collaboration": 0.81,
                "accountability": 0.78,
                "transparency": 0.82
            },
            "confidence_score": 0.88,
            "data_points": list(range(1, 201))
        },
        "strategy_data": {
            "goals": [
                {"id": "culture_health", "target": 0.85},
                {"id": "engagement", "target": 0.90}
            ],
            "planned_resources": 600000,
            "investment": 750000
        },
        "outcome_data": {
            "achievements": [
                {"goal_id": "culture_health", "score": 0.87},
                {"goal_id": "engagement", "score": 0.92}
            ],
            "actual_resources": 680000,
            "benefits": 950000,
            "impact_score": 0.88
        },
        "baseline_behaviors": [
            {"type": "collaboration_frequency", "value": 2.5},
            {"type": "innovation_attempts", "value": 1.2},
            {"type": "feedback_giving", "value": 0.9}
        ],
        "current_behaviors": [
            {"type": "collaboration_frequency", "value": 4.1},
            {"type": "innovation_attempts", "value": 2.8},
            {"type": "feedback_giving", "value": 2.3}
        ],
        "communication_data": {
            "reach_metrics": {"coverage": 0.95, "target": 1.0, "confidence": 0.9},
            "engagement_metrics": {"average_engagement": 0.84, "confidence": 0.85},
            "behavior_influence": {"influence_score": 0.78, "confidence": 0.8}
        },
        "tracking_data": {
            "predictions": [0.82, 0.85, 0.87, 0.89],
            "reliability_score": 0.88,
            "confidence": 0.85
        },
        "actual_outcomes": {
            "actual_values": [0.84, 0.86, 0.88, 0.90],
            "confidence": 0.87
        }
    }
    
    # Run comprehensive validation suite
    results = validator.run_comprehensive_validation_suite(transformation_data)
    
    print(f"Comprehensive Validation Results:")
    print(f"  Total Tests Run: {len(results)}")
    
    passed_tests = [r for r in results if r.passed]
    failed_tests = [r for r in results if not r.passed]
    
    print(f"  Tests Passed: {len(passed_tests)}")
    print(f"  Tests Failed: {len(failed_tests)}")
    print(f"  Overall Success Rate: {len(passed_tests)/len(results):.1%}")
    
    print(f"\nDetailed Results:")
    for result in results:
        status = "✓ PASSED" if result.passed else "✗ FAILED"
        print(f"    {result.test_name}: {status} (Score: {result.score:.3f}, Confidence: {result.confidence:.3f})")
    
    return results


def main():
    """Run all demos"""
    print("Cultural Transformation Testing and Validation Framework Demo")
    print("=" * 80)
    
    try:
        # Run individual demos
        demo_cultural_assessment_validation()
        demo_transformation_strategy_validation()
        demo_transformation_success_measurement()
        demo_cultural_sustainability_testing()
        demo_transformation_roi_assessment()
        demo_comprehensive_validation_suite()
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nKey Capabilities Demonstrated:")
        print("✓ Cultural Assessment Accuracy Validation")
        print("✓ Transformation Strategy Effectiveness Testing")
        print("✓ Transformation Success Measurement")
        print("✓ Cultural Sustainability Testing")
        print("✓ Cultural Drift Detection")
        print("✓ Transformation ROI Assessment")
        print("✓ Comprehensive Value Analysis")
        print("✓ Integrated Validation Suite")
        
        print("\nThe Cultural Transformation Testing Framework provides:")
        print("• Comprehensive validation of cultural assessments")
        print("• Rigorous testing of transformation strategies")
        print("• Accurate measurement of transformation outcomes")
        print("• Long-term sustainability monitoring")
        print("• Financial ROI validation and assessment")
        print("• Integrated testing across all transformation phases")
        
    except Exception as e:
        print(f"\nDemo encountered an error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()