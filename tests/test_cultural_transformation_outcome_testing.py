"""
Cultural Transformation Outcome Testing

This module provides comprehensive testing for transformation success measurement,
cultural sustainability testing, and transformation ROI assessment.
Requirements: 1.2, 2.2, 3.2, 4.2, 5.2
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

from scrollintel.engines.cultural_transformation_outcome_validator import CulturalTransformationOutcomeValidator
from scrollintel.engines.cultural_sustainability_tester import CulturalSustainabilityTester
from scrollintel.engines.transformation_roi_assessor import TransformationROIAssessor
from scrollintel.models.transformation_outcome_models import (
    TransformationOutcome, SustainabilityMetrics
)
from scrollintel.engines.transformation_roi_assessor import ROIAssessment


class TestTransformationSuccessMeasurement:
    """Test transformation success measurement and validation"""
    
    @pytest.fixture
    def outcome_validator(self):
        return CulturalTransformationOutcomeValidator()
    
    @pytest.fixture
    def sample_transformation_outcome(self):
        return TransformationOutcome(
            transformation_id="trans_001",
            organization_id="org_001",
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now() - timedelta(days=30),
            target_metrics={
                "cultural_health": 0.8,
                "employee_engagement": 0.85,
                "performance_improvement": 0.15,
                "innovation_index": 0.7
            },
            achieved_metrics={
                "cultural_health": 0.82,
                "employee_engagement": 0.87,
                "performance_improvement": 0.18,
                "innovation_index": 0.75
            },
            success_criteria=[
                {"metric": "cultural_health", "threshold": 0.75, "weight": 0.3},
                {"metric": "employee_engagement", "threshold": 0.8, "weight": 0.3},
                {"metric": "performance_improvement", "threshold": 0.1, "weight": 0.2},
                {"metric": "innovation_index", "threshold": 0.65, "weight": 0.2}
            ]
        )
    
    def test_overall_transformation_success_measurement(self, outcome_validator, sample_transformation_outcome):
        """Test measurement of overall transformation success"""
        success_assessment = outcome_validator.measure_transformation_success(sample_transformation_outcome)
        
        assert success_assessment.overall_success_score >= 0.8
        assert success_assessment.criteria_met_percentage >= 0.9
        assert success_assessment.success_level in ["high", "exceptional"]
        assert len(success_assessment.exceeded_targets) >= 2
    
    def test_goal_achievement_validation(self, outcome_validator, sample_transformation_outcome):
        """Test validation of specific goal achievement"""
        goal_assessment = outcome_validator.validate_goal_achievement(
            sample_transformation_outcome.target_metrics,
            sample_transformation_outcome.achieved_metrics,
            sample_transformation_outcome.success_criteria
        )
        
        assert goal_assessment.achievement_rate >= 0.9
        assert len(goal_assessment.fully_achieved_goals) >= 3
        assert goal_assessment.weighted_success_score >= 0.85
        
        # Test individual goal validation
        for goal in goal_assessment.goal_results:
            if goal.metric_name == "cultural_health":
                assert goal.achieved_value >= goal.target_value
                assert goal.success_margin > 0
    
    def test_transformation_impact_measurement(self, outcome_validator, sample_transformation_outcome):
        """Test measurement of transformation impact on organization"""
        impact_data = {
            "baseline_performance": {
                "productivity": 0.7,
                "retention": 0.8,
                "satisfaction": 0.65,
                "innovation_rate": 0.3
            },
            "post_transformation_performance": {
                "productivity": 0.85,
                "retention": 0.92,
                "satisfaction": 0.82,
                "innovation_rate": 0.55
            },
            "external_factors": [
                {"factor": "market_conditions", "impact": 0.05},
                {"factor": "industry_trends", "impact": 0.03}
            ]
        }
        
        impact_assessment = outcome_validator.measure_transformation_impact(
            sample_transformation_outcome, impact_data
        )
        
        assert impact_assessment.net_impact_score >= 0.7
        assert impact_assessment.attribution_confidence >= 0.8
        assert len(impact_assessment.significant_improvements) >= 3
        
        # Test impact attribution
        productivity_impact = next(
            (imp for imp in impact_assessment.impact_breakdown 
             if imp.metric == "productivity"), None
        )
        assert productivity_impact is not None
        assert productivity_impact.transformation_attribution >= 0.8
    
    def test_success_validation_against_benchmarks(self, outcome_validator):
        """Test validation of success against industry benchmarks"""
        transformation_results = {
            "cultural_health_improvement": 0.25,
            "engagement_increase": 0.15,
            "performance_boost": 0.18,
            "retention_improvement": 0.12
        }
        
        benchmark_data = {
            "industry_averages": {
                "cultural_health_improvement": 0.15,
                "engagement_increase": 0.10,
                "performance_boost": 0.12,
                "retention_improvement": 0.08
            },
            "top_quartile": {
                "cultural_health_improvement": 0.22,
                "engagement_increase": 0.18,
                "performance_boost": 0.20,
                "retention_improvement": 0.15
            }
        }
        
        benchmark_assessment = outcome_validator.validate_against_benchmarks(
            transformation_results, benchmark_data
        )
        
        assert benchmark_assessment.overall_ranking >= "above_average"
        assert benchmark_assessment.percentile_score >= 70
        assert len(benchmark_assessment.top_quartile_metrics) >= 2
    
    def test_transformation_quality_assessment(self, outcome_validator, sample_transformation_outcome):
        """Test assessment of transformation quality and depth"""
        quality_indicators = {
            "behavior_change_depth": 0.8,
            "cultural_integration": 0.85,
            "leadership_alignment": 0.9,
            "employee_adoption": 0.82,
            "process_embedding": 0.75,
            "value_internalization": 0.78
        }
        
        quality_assessment = outcome_validator.assess_transformation_quality(
            sample_transformation_outcome, quality_indicators
        )
        
        assert quality_assessment.overall_quality_score >= 0.8
        assert quality_assessment.depth_score >= 0.75
        assert quality_assessment.integration_score >= 0.8
        assert quality_assessment.quality_level in ["high", "exceptional"]


class TestCulturalSustainabilityTesting:
    """Test cultural sustainability testing and verification"""
    
    @pytest.fixture
    def sustainability_tester(self):
        return CulturalSustainabilityTester()
    
    @pytest.fixture
    def sample_sustainability_data(self):
        return {
            "transformation_completion": datetime.now() - timedelta(days=180),
            "measurement_points": [
                {"date": datetime.now() - timedelta(days=150), "cultural_health": 0.82, "engagement": 0.87},
                {"date": datetime.now() - timedelta(days=120), "cultural_health": 0.81, "engagement": 0.86},
                {"date": datetime.now() - timedelta(days=90), "cultural_health": 0.83, "engagement": 0.88},
                {"date": datetime.now() - timedelta(days=60), "cultural_health": 0.80, "engagement": 0.85},
                {"date": datetime.now() - timedelta(days=30), "cultural_health": 0.82, "engagement": 0.87},
                {"date": datetime.now(), "cultural_health": 0.81, "engagement": 0.86}
            ],
            "reinforcement_mechanisms": [
                {"type": "leadership_modeling", "strength": 0.8, "consistency": 0.9},
                {"type": "recognition_systems", "strength": 0.7, "consistency": 0.8},
                {"type": "performance_integration", "strength": 0.85, "consistency": 0.85},
                {"type": "communication_reinforcement", "strength": 0.75, "consistency": 0.9}
            ]
        }
    
    def test_cultural_change_sustainability_measurement(self, sustainability_tester, sample_sustainability_data):
        """Test measurement of cultural change sustainability"""
        sustainability_metrics = sustainability_tester.measure_sustainability(sample_sustainability_data)
        
        assert isinstance(sustainability_metrics, SustainabilityMetrics)
        assert sustainability_metrics.stability_score >= 0.7
        assert sustainability_metrics.trend_consistency >= 0.8
        assert sustainability_metrics.reinforcement_strength >= 0.75
        assert sustainability_metrics.decay_resistance >= 0.7
    
    def test_long_term_stability_validation(self, sustainability_tester, sample_sustainability_data):
        """Test validation of long-term cultural stability"""
        stability_assessment = sustainability_tester.validate_long_term_stability(
            sample_sustainability_data["measurement_points"]
        )
        
        assert stability_assessment.stability_coefficient >= 0.8
        assert stability_assessment.variance_within_acceptable_range
        assert stability_assessment.trend_direction in ["stable", "improving"]
        assert stability_assessment.confidence_level >= 0.85
        
        # Test stability across different time windows
        short_term_stability = stability_assessment.stability_by_period["30_days"]
        medium_term_stability = stability_assessment.stability_by_period["90_days"]
        long_term_stability = stability_assessment.stability_by_period["180_days"]
        
        assert all(s >= 0.7 for s in [short_term_stability, medium_term_stability, long_term_stability])
    
    def test_reinforcement_mechanism_effectiveness(self, sustainability_tester, sample_sustainability_data):
        """Test effectiveness of cultural reinforcement mechanisms"""
        reinforcement_analysis = sustainability_tester.analyze_reinforcement_effectiveness(
            sample_sustainability_data["reinforcement_mechanisms"]
        )
        
        assert reinforcement_analysis.overall_effectiveness >= 0.75
        assert len(reinforcement_analysis.strong_mechanisms) >= 2
        assert reinforcement_analysis.consistency_score >= 0.8
        
        # Test individual mechanism effectiveness
        leadership_mechanism = next(
            (mech for mech in reinforcement_analysis.mechanism_scores 
             if mech.type == "leadership_modeling"), None
        )
        assert leadership_mechanism is not None
        assert leadership_mechanism.effectiveness_score >= 0.8
    
    def test_cultural_drift_detection(self, sustainability_tester):
        """Test detection of cultural drift and degradation"""
        drift_data = {
            "baseline_culture": {"innovation": 0.8, "collaboration": 0.85, "accountability": 0.75},
            "current_culture": {"innovation": 0.72, "collaboration": 0.83, "accountability": 0.70},
            "measurement_history": [
                {"date": datetime.now() - timedelta(days=90), "innovation": 0.79, "collaboration": 0.84, "accountability": 0.74},
                {"date": datetime.now() - timedelta(days=60), "innovation": 0.76, "collaboration": 0.83, "accountability": 0.72},
                {"date": datetime.now() - timedelta(days=30), "innovation": 0.74, "collaboration": 0.83, "accountability": 0.71}
            ]
        }
        
        drift_assessment = sustainability_tester.detect_cultural_drift(drift_data)
        
        assert drift_assessment.drift_detected
        assert drift_assessment.drift_severity in ["moderate", "significant"]
        assert "innovation" in drift_assessment.affected_dimensions
        assert len(drift_assessment.recommended_interventions) >= 2
    
    def test_sustainability_risk_assessment(self, sustainability_tester, sample_sustainability_data):
        """Test assessment of sustainability risks"""
        risk_factors = {
            "leadership_changes": {"probability": 0.3, "impact": 0.8},
            "organizational_restructuring": {"probability": 0.2, "impact": 0.9},
            "market_pressures": {"probability": 0.6, "impact": 0.5},
            "resource_constraints": {"probability": 0.4, "impact": 0.6}
        }
        
        risk_assessment = sustainability_tester.assess_sustainability_risks(
            sample_sustainability_data, risk_factors
        )
        
        assert risk_assessment.overall_risk_level in ["low", "medium", "high"]
        assert risk_assessment.risk_score >= 0
        assert len(risk_assessment.mitigation_strategies) >= 2
        assert risk_assessment.monitoring_recommendations is not None
    
    def test_cultural_resilience_measurement(self, sustainability_tester):
        """Test measurement of cultural resilience to challenges"""
        resilience_data = {
            "challenge_events": [
                {"type": "leadership_change", "date": datetime.now() - timedelta(days=120), "severity": 0.7},
                {"type": "market_downturn", "date": datetime.now() - timedelta(days=80), "severity": 0.6},
                {"type": "restructuring", "date": datetime.now() - timedelta(days=40), "severity": 0.8}
            ],
            "recovery_metrics": [
                {"event_id": 1, "recovery_time": 30, "recovery_completeness": 0.9},
                {"event_id": 2, "recovery_time": 20, "recovery_completeness": 0.95},
                {"event_id": 3, "recovery_time": 45, "recovery_completeness": 0.85}
            ]
        }
        
        resilience_assessment = sustainability_tester.measure_cultural_resilience(resilience_data)
        
        assert resilience_assessment.resilience_score >= 0.8
        assert resilience_assessment.average_recovery_time <= 35
        assert resilience_assessment.recovery_completeness >= 0.85
        assert resilience_assessment.resilience_level in ["high", "exceptional"]


class TestTransformationROIAssessment:
    """Test transformation ROI assessment and validation"""
    
    @pytest.fixture
    def roi_assessor(self):
        return TransformationROIAssessor()
    
    @pytest.fixture
    def sample_roi_data(self):
        return {
            "investment_data": {
                "direct_costs": {
                    "consulting_fees": 250000,
                    "training_programs": 150000,
                    "technology_tools": 75000,
                    "internal_resources": 200000
                },
                "indirect_costs": {
                    "opportunity_cost": 100000,
                    "productivity_loss": 50000,
                    "change_management": 80000
                },
                "timeline": 12  # months
            },
            "benefit_data": {
                "quantifiable_benefits": {
                    "productivity_improvement": 180000,  # annual
                    "retention_savings": 120000,  # annual
                    "reduced_turnover_costs": 90000,  # annual
                    "innovation_revenue": 200000  # annual
                },
                "qualitative_benefits": [
                    {"benefit": "improved_reputation", "value_estimate": 50000},
                    {"benefit": "enhanced_agility", "value_estimate": 75000},
                    {"benefit": "better_decision_making", "value_estimate": 60000}
                ],
                "measurement_period": 24  # months
            }
        }
    
    def test_financial_roi_calculation(self, roi_assessor, sample_roi_data):
        """Test calculation of financial ROI"""
        roi_assessment = roi_assessor.calculate_financial_roi(sample_roi_data)
        
        assert isinstance(roi_assessment, ROIAssessment)
        assert roi_assessment.roi_percentage >= 50  # Expect positive ROI
        assert roi_assessment.payback_period <= 18  # months
        assert roi_assessment.net_present_value > 0
        assert roi_assessment.benefit_cost_ratio >= 1.5
    
    def test_comprehensive_value_assessment(self, roi_assessor, sample_roi_data):
        """Test comprehensive value assessment including intangible benefits"""
        value_assessment = roi_assessor.assess_comprehensive_value(sample_roi_data)
        
        assert value_assessment.total_value_score >= 0.8
        assert value_assessment.tangible_value >= 500000  # Annual benefits
        assert value_assessment.intangible_value >= 150000  # Estimated intangible value
        assert len(value_assessment.value_drivers) >= 5
        
        # Test value driver analysis
        productivity_driver = next(
            (driver for driver in value_assessment.value_drivers 
             if driver.name == "productivity_improvement"), None
        )
        assert productivity_driver is not None
        assert productivity_driver.annual_value >= 150000
    
    def test_roi_validation_and_verification(self, roi_assessor, sample_roi_data):
        """Test validation and verification of ROI calculations"""
        validation_result = roi_assessor.validate_roi_calculation(sample_roi_data)
        
        assert validation_result.calculation_accuracy >= 0.9
        assert validation_result.data_completeness >= 0.85
        assert validation_result.assumption_validity >= 0.8
        assert len(validation_result.validation_warnings) <= 2
        
        # Test sensitivity analysis
        sensitivity_analysis = validation_result.sensitivity_analysis
        assert sensitivity_analysis.best_case_roi >= validation_result.base_case_roi
        assert sensitivity_analysis.worst_case_roi <= validation_result.base_case_roi
        assert sensitivity_analysis.confidence_interval[1] - sensitivity_analysis.confidence_interval[0] <= 0.3
    
    def test_long_term_value_projection(self, roi_assessor, sample_roi_data):
        """Test projection of long-term value creation"""
        projection_data = {
            **sample_roi_data,
            "projection_period": 60,  # months
            "growth_assumptions": {
                "productivity_growth": 0.05,  # annual
                "benefit_sustainability": 0.9,
                "cost_inflation": 0.03
            }
        }
        
        long_term_projection = roi_assessor.project_long_term_value(projection_data)
        
        assert long_term_projection.five_year_roi >= 200  # %
        assert long_term_projection.cumulative_value >= 2000000
        assert long_term_projection.value_sustainability_score >= 0.8
        assert len(long_term_projection.value_milestones) >= 5
    
    def test_roi_benchmarking(self, roi_assessor, sample_roi_data):
        """Test ROI benchmarking against industry standards"""
        benchmark_data = {
            "industry_averages": {
                "cultural_transformation_roi": 85,  # %
                "payback_period": 24,  # months
                "success_rate": 0.65
            },
            "best_practices": {
                "top_quartile_roi": 150,  # %
                "optimal_payback": 18,  # months
                "high_success_rate": 0.85
            }
        }
        
        roi_calculation = roi_assessor.calculate_financial_roi(sample_roi_data)
        benchmark_assessment = roi_assessor.benchmark_roi_performance(roi_calculation, benchmark_data)
        
        assert benchmark_assessment.performance_ranking in ["above_average", "top_quartile"]
        assert benchmark_assessment.roi_percentile >= 70
        assert benchmark_assessment.competitive_advantage_score >= 0.7
    
    def test_risk_adjusted_roi_calculation(self, roi_assessor, sample_roi_data):
        """Test risk-adjusted ROI calculation"""
        risk_factors = {
            "implementation_risk": {"probability": 0.2, "impact": 0.3},
            "adoption_risk": {"probability": 0.3, "impact": 0.4},
            "sustainability_risk": {"probability": 0.25, "impact": 0.35},
            "external_risk": {"probability": 0.15, "impact": 0.2}
        }
        
        risk_adjusted_roi = roi_assessor.calculate_risk_adjusted_roi(sample_roi_data, risk_factors)
        
        assert risk_adjusted_roi.adjusted_roi_percentage >= 0
        assert risk_adjusted_roi.risk_discount_factor <= 1.0
        assert risk_adjusted_roi.confidence_level >= 0.7
        assert len(risk_adjusted_roi.risk_mitigation_recommendations) >= 2


@pytest.mark.integration
class TestTransformationOutcomeIntegration:
    """Integration tests for complete transformation outcome testing"""
    
    def test_end_to_end_outcome_validation(self):
        """Test complete outcome validation process"""
        # This would test the entire outcome validation pipeline
        # from success measurement through sustainability to ROI
        pass
    
    def test_cross_metric_correlation_analysis(self):
        """Test correlation analysis across different outcome metrics"""
        # This would validate correlations between success, sustainability, and ROI metrics
        pass
    
    def test_longitudinal_outcome_tracking(self):
        """Test long-term outcome tracking and validation"""
        # This would test outcome tracking over extended periods
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])