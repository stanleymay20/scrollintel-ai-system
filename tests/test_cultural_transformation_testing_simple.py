"""
Simple Cultural Transformation Testing Suite

Basic tests for the cultural transformation testing framework.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from scrollintel.engines.cultural_transformation_validator import CulturalTransformationValidator
from scrollintel.engines.cultural_transformation_outcome_validator import CulturalTransformationOutcomeValidator
from scrollintel.engines.cultural_sustainability_tester import CulturalSustainabilityTester
from scrollintel.engines.transformation_roi_assessor import TransformationROIAssessor


class TestCulturalTransformationValidator:
    """Test the cultural transformation validator"""
    
    @pytest.fixture
    def validator(self):
        return CulturalTransformationValidator()
    
    def test_validator_initialization(self, validator):
        """Test validator initializes correctly"""
        assert validator is not None
        assert validator.accuracy_threshold == 0.8
        assert validator.effectiveness_threshold == 0.7
        assert validator.confidence_threshold == 0.85
    
    def test_validate_cultural_assessment_accuracy(self, validator):
        """Test cultural assessment accuracy validation"""
        assessment_results = {
            "cultural_dimensions": {
                "innovation": 0.8,
                "collaboration": 0.75,
                "accountability": 0.7
            },
            "confidence_score": 0.85,
            "data_points": [1, 2, 3, 4, 5]
        }
        
        result = validator.validate_cultural_assessment_accuracy(assessment_results)
        
        assert result.test_name == "cultural_assessment_accuracy"
        assert isinstance(result.passed, bool)
        assert 0 <= result.score <= 1
        assert 0 <= result.confidence <= 1
        assert isinstance(result.details, dict)
    
    def test_validate_transformation_strategy_effectiveness(self, validator):
        """Test transformation strategy effectiveness validation"""
        strategy_data = {
            "goals": [
                {"id": "goal1", "target": 0.8},
                {"id": "goal2", "target": 0.75}
            ],
            "planned_resources": 100000,
            "investment": 150000
        }
        
        outcome_data = {
            "achievements": [
                {"goal_id": "goal1", "score": 0.85},
                {"goal_id": "goal2", "score": 0.8}
            ],
            "actual_resources": 120000,
            "benefits": 200000,
            "impact_score": 0.8
        }
        
        result = validator.validate_transformation_strategy_effectiveness(strategy_data, outcome_data)
        
        assert result.test_name == "transformation_strategy_effectiveness"
        assert isinstance(result.passed, bool)
        assert 0 <= result.score <= 1


class TestCulturalTransformationOutcomeValidator:
    """Test the outcome validator"""
    
    @pytest.fixture
    def outcome_validator(self):
        return CulturalTransformationOutcomeValidator()
    
    @pytest.fixture
    def sample_transformation_outcome(self):
        from scrollintel.models.transformation_outcome_models import TransformationOutcome, TransformationStatus
        return TransformationOutcome(
            transformation_id="trans_001",
            organization_id="org_001",
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now() - timedelta(days=30),
            status=TransformationStatus.COMPLETED,
            target_metrics={
                "cultural_health": 0.8,
                "employee_engagement": 0.85
            },
            achieved_metrics={
                "cultural_health": 0.82,
                "employee_engagement": 0.87
            },
            success_criteria=[
                {"metric": "cultural_health", "threshold": 0.75, "weight": 0.5},
                {"metric": "employee_engagement", "threshold": 0.8, "weight": 0.5}
            ],
            investment_data={},
            benefit_data={},
            sustainability_data={},
            validation_results=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def test_outcome_validator_initialization(self, outcome_validator):
        """Test outcome validator initializes correctly"""
        assert outcome_validator is not None
        assert "exceptional" in outcome_validator.success_thresholds
        assert "high" in outcome_validator.quality_thresholds
    
    def test_measure_transformation_success(self, outcome_validator, sample_transformation_outcome):
        """Test transformation success measurement"""
        success_assessment = outcome_validator.measure_transformation_success(sample_transformation_outcome)
        
        assert success_assessment is not None
        assert 0 <= success_assessment.overall_success_score <= 1
        assert 0 <= success_assessment.criteria_met_percentage <= 1
        assert success_assessment.success_level in ["exceptional", "high", "moderate", "low", "failed"]
        assert isinstance(success_assessment.exceeded_targets, list)
        assert isinstance(success_assessment.underperformed_areas, list)


class TestCulturalSustainabilityTester:
    """Test the sustainability tester"""
    
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
                {"date": datetime.now() - timedelta(days=90), "cultural_health": 0.83, "engagement": 0.88}
            ],
            "reinforcement_mechanisms": [
                {"type": "leadership_modeling", "strength": 0.8, "consistency": 0.9},
                {"type": "recognition_systems", "strength": 0.7, "consistency": 0.8}
            ]
        }
    
    def test_sustainability_tester_initialization(self, sustainability_tester):
        """Test sustainability tester initializes correctly"""
        assert sustainability_tester is not None
        assert "excellent" in sustainability_tester.stability_thresholds
        assert "minimal" in sustainability_tester.drift_thresholds
    
    def test_measure_sustainability(self, sustainability_tester, sample_sustainability_data):
        """Test sustainability measurement"""
        sustainability_metrics = sustainability_tester.measure_sustainability(sample_sustainability_data)
        
        assert sustainability_metrics is not None
        assert 0 <= sustainability_metrics.stability_score <= 1
        assert 0 <= sustainability_metrics.trend_consistency <= 1
        assert 0 <= sustainability_metrics.reinforcement_strength <= 1
        assert 0 <= sustainability_metrics.decay_resistance <= 1
        assert sustainability_metrics.sustainability_level in ["excellent", "good", "acceptable", "concerning", "poor"]


class TestTransformationROIAssessor:
    """Test the ROI assessor"""
    
    @pytest.fixture
    def roi_assessor(self):
        return TransformationROIAssessor()
    
    @pytest.fixture
    def sample_roi_data(self):
        return {
            "investment_data": {
                "direct_costs": {
                    "consulting_fees": 250000,
                    "training_programs": 150000
                },
                "indirect_costs": {
                    "opportunity_cost": 100000
                }
            },
            "benefit_data": {
                "quantifiable_benefits": {
                    "productivity_improvement": 180000,
                    "retention_savings": 120000
                },
                "measurement_period": 24
            }
        }
    
    def test_roi_assessor_initialization(self, roi_assessor):
        """Test ROI assessor initializes correctly"""
        assert roi_assessor is not None
        assert roi_assessor.discount_rate == 0.08
        assert roi_assessor.risk_free_rate == 0.03
        assert "high" in roi_assessor.confidence_thresholds
    
    def test_calculate_financial_roi(self, roi_assessor, sample_roi_data):
        """Test financial ROI calculation"""
        roi_assessment = roi_assessor.calculate_financial_roi(sample_roi_data)
        
        assert roi_assessment is not None
        assert isinstance(roi_assessment.roi_percentage, (int, float))
        assert isinstance(roi_assessment.payback_period, (int, float))
        assert isinstance(roi_assessment.net_present_value, (int, float))
        assert isinstance(roi_assessment.benefit_cost_ratio, (int, float))
        assert roi_assessment.total_investment > 0
        assert roi_assessment.total_benefits > 0
        assert 0 <= roi_assessment.assessment_confidence <= 1


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    def test_comprehensive_validation_workflow(self):
        """Test a complete validation workflow"""
        # Initialize all validators
        validator = CulturalTransformationValidator()
        outcome_validator = CulturalTransformationOutcomeValidator()
        sustainability_tester = CulturalSustainabilityTester()
        roi_assessor = TransformationROIAssessor()
        
        # Verify all components are initialized
        assert validator is not None
        assert outcome_validator is not None
        assert sustainability_tester is not None
        assert roi_assessor is not None
    
    def test_validation_result_consistency(self):
        """Test that validation results are consistent"""
        validator = CulturalTransformationValidator()
        
        # Test with same data multiple times
        assessment_results = {
            "cultural_dimensions": {"innovation": 0.8},
            "confidence_score": 0.85,
            "data_points": [1, 2, 3]
        }
        
        result1 = validator.validate_cultural_assessment_accuracy(assessment_results)
        result2 = validator.validate_cultural_assessment_accuracy(assessment_results)
        
        # Results should be consistent
        assert result1.test_name == result2.test_name
        assert result1.score == result2.score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])