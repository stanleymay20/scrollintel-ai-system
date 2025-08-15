"""
Tests for Risk-Benefit Analysis Engine

This module tests the risk-benefit analysis capabilities including
response option evaluation, mitigation strategy generation, and benefit optimization.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.risk_benefit_analyzer import RiskBenefitAnalyzer
from scrollintel.models.risk_benefit_models import (
    RiskFactor, BenefitFactor, ResponseOption, MitigationStrategy,
    TradeOffAnalysis, RiskBenefitEvaluation, OptimizationResult,
    RiskLevel, BenefitType, UncertaintyLevel
)


class TestRiskBenefitAnalyzer:
    """Test suite for RiskBenefitAnalyzer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = RiskBenefitAnalyzer()
        
        # Create sample risk factors
        self.sample_risks = [
            RiskFactor(
                name="System Downtime",
                description="Risk of extended system outage",
                category="operational",
                probability=0.3,
                impact_severity=RiskLevel.HIGH,
                potential_impact="Service disruption affecting customers",
                time_horizon="immediate",
                uncertainty_level=UncertaintyLevel.MODERATE
            ),
            RiskFactor(
                name="Reputation Damage",
                description="Risk of negative public perception",
                category="reputational",
                probability=0.4,
                impact_severity=RiskLevel.MEDIUM,
                potential_impact="Loss of customer trust",
                time_horizon="short_term",
                uncertainty_level=UncertaintyLevel.HIGH
            )
        ]
        
        # Create sample benefit factors
        self.sample_benefits = [
            BenefitFactor(
                name="Cost Savings",
                description="Reduced operational costs",
                benefit_type=BenefitType.FINANCIAL,
                expected_value=0.8,
                probability_of_realization=0.7,
                time_to_realization="short_term",
                sustainability="permanent",
                uncertainty_level=UncertaintyLevel.LOW
            ),
            BenefitFactor(
                name="Process Improvement",
                description="Enhanced operational efficiency",
                benefit_type=BenefitType.OPERATIONAL,
                expected_value=0.6,
                probability_of_realization=0.8,
                time_to_realization="medium_term",
                sustainability="permanent",
                uncertainty_level=UncertaintyLevel.MODERATE
            )
        ]
        
        # Create sample response options
        self.sample_options = [
            ResponseOption(
                name="Emergency Shutdown",
                description="Immediate system shutdown and repair",
                category="emergency_response",
                implementation_complexity="low",
                resource_requirements={"personnel": 5, "budget": 10000},
                time_to_implement="immediate",
                risks=self.sample_risks[:1],  # Only system downtime risk
                benefits=self.sample_benefits[:1],  # Only cost savings
                dependencies=["technical_team_availability"],
                success_criteria=["system_restored_within_4_hours"]
            ),
            ResponseOption(
                name="Gradual Migration",
                description="Planned migration to backup systems",
                category="planned_response",
                implementation_complexity="medium",
                resource_requirements={"personnel": 10, "budget": 25000},
                time_to_implement="short_term",
                risks=self.sample_risks,  # Both risks
                benefits=self.sample_benefits,  # Both benefits
                dependencies=["backup_system_ready", "stakeholder_approval"],
                success_criteria=["zero_data_loss", "minimal_service_disruption"]
            )
        ]
    
    def test_initialization(self):
        """Test analyzer initialization"""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'risk_weights')
        assert hasattr(self.analyzer, 'uncertainty_adjustments')
        assert hasattr(self.analyzer, 'benefit_weights')
        
        # Check weight configurations
        assert RiskLevel.CRITICAL in self.analyzer.risk_weights
        assert UncertaintyLevel.VERY_HIGH in self.analyzer.uncertainty_adjustments
        assert BenefitType.FINANCIAL in self.analyzer.benefit_weights
    
    def test_calculate_risk_score(self):
        """Test risk score calculation"""
        # Test with sample risks
        risk_score = self.analyzer._calculate_risk_score(
            self.sample_risks, 
            RiskLevel.MEDIUM
        )
        
        assert 0.0 <= risk_score <= 1.0
        assert risk_score > 0  # Should have some risk
        
        # Test with empty risks
        empty_risk_score = self.analyzer._calculate_risk_score([], RiskLevel.MEDIUM)
        assert empty_risk_score == 0.0
        
        # Test with different risk tolerances
        low_tolerance_score = self.analyzer._calculate_risk_score(
            self.sample_risks, 
            RiskLevel.LOW
        )
        high_tolerance_score = self.analyzer._calculate_risk_score(
            self.sample_risks, 
            RiskLevel.HIGH
        )
        
        # Low tolerance should result in higher risk perception
        assert low_tolerance_score > high_tolerance_score
    
    def test_calculate_benefit_score(self):
        """Test benefit score calculation"""
        # Test with sample benefits
        benefit_score = self.analyzer._calculate_benefit_score(self.sample_benefits)
        
        assert 0.0 <= benefit_score <= 1.0
        assert benefit_score > 0  # Should have some benefit
        
        # Test with empty benefits
        empty_benefit_score = self.analyzer._calculate_benefit_score([])
        assert empty_benefit_score == 0.0
        
        # Test with high-value benefits
        high_value_benefits = [
            BenefitFactor(
                name="High Value Benefit",
                benefit_type=BenefitType.FINANCIAL,
                expected_value=1.0,
                probability_of_realization=1.0,
                uncertainty_level=UncertaintyLevel.VERY_LOW
            )
        ]
        
        high_benefit_score = self.analyzer._calculate_benefit_score(high_value_benefits)
        assert high_benefit_score > benefit_score
    
    def test_calculate_overall_score(self):
        """Test overall score calculation"""
        risk_score = 0.3
        benefit_score = 0.7
        criteria = {"risk_aversion": 0.4, "benefit_focus": 0.6}
        
        overall_score = self.analyzer._calculate_overall_score(
            risk_score, benefit_score, criteria, "moderate"
        )
        
        assert 0.0 <= overall_score <= 1.0
        
        # Test with different time pressures
        immediate_score = self.analyzer._calculate_overall_score(
            risk_score, benefit_score, criteria, "immediate"
        )
        low_pressure_score = self.analyzer._calculate_overall_score(
            risk_score, benefit_score, criteria, "low"
        )
        
        # Low pressure should allow for slightly better scores
        assert low_pressure_score >= immediate_score
    
    def test_evaluate_response_options(self):
        """Test complete response option evaluation"""
        evaluation_criteria = {
            "risk_aversion": 0.4,
            "benefit_focus": 0.6
        }
        
        evaluation = self.analyzer.evaluate_response_options(
            crisis_id="test_crisis_001",
            response_options=self.sample_options,
            evaluation_criteria=evaluation_criteria,
            risk_tolerance=RiskLevel.MEDIUM,
            time_pressure="moderate"
        )
        
        # Verify evaluation structure
        assert isinstance(evaluation, RiskBenefitEvaluation)
        assert evaluation.crisis_id == "test_crisis_001"
        assert evaluation.recommended_option is not None
        assert 0.0 <= evaluation.confidence_score <= 1.0
        assert len(evaluation.response_options) == len(self.sample_options)
        
        # Verify trade-off analyses
        assert len(evaluation.trade_off_analyses) > 0
        for analysis in evaluation.trade_off_analyses:
            assert isinstance(analysis, TradeOffAnalysis)
            assert analysis.recommendation is not None
            assert 0.0 <= analysis.confidence_level <= 1.0
        
        # Verify mitigation plan
        assert len(evaluation.mitigation_plan) > 0
        for strategy in evaluation.mitigation_plan:
            assert isinstance(strategy, MitigationStrategy)
            assert strategy.name is not None
            assert 0.0 <= strategy.effectiveness_score <= 1.0
        
        # Verify monitoring requirements
        assert len(evaluation.monitoring_requirements) > 0
        assert isinstance(evaluation.monitoring_requirements, list)
    
    def test_generate_mitigation_strategies(self):
        """Test mitigation strategy generation"""
        strategies = self.analyzer._generate_mitigation_strategies(self.sample_risks)
        
        assert len(strategies) > 0
        
        for strategy in strategies:
            assert isinstance(strategy, MitigationStrategy)
            assert strategy.name is not None
            assert strategy.description is not None
            assert len(strategy.target_risks) > 0
            assert 0.0 <= strategy.effectiveness_score <= 1.0
            assert 0.0 <= strategy.success_probability <= 1.0
        
        # Test with high-severity risks
        critical_risk = RiskFactor(
            name="Critical System Failure",
            category="operational",
            impact_severity=RiskLevel.CRITICAL,
            probability=0.8
        )
        
        critical_strategies = self.analyzer._generate_mitigation_strategies([critical_risk])
        assert len(critical_strategies) > 0
    
    def test_generate_trade_off_analyses(self):
        """Test trade-off analysis generation"""
        # Create scored options for testing
        scored_options = [
            {
                'option': self.sample_options[0],
                'risk_score': 0.3,
                'benefit_score': 0.7,
                'overall_score': 0.4
            },
            {
                'option': self.sample_options[1],
                'risk_score': 0.5,
                'benefit_score': 0.6,
                'overall_score': 0.1
            }
        ]
        
        analyses = self.analyzer._generate_trade_off_analyses(scored_options)
        
        assert len(analyses) > 0
        
        for analysis in analyses:
            assert isinstance(analysis, TradeOffAnalysis)
            assert analysis.option_a_id is not None
            assert analysis.option_b_id is not None
            assert analysis.recommendation is not None
            assert 0.0 <= analysis.confidence_level <= 1.0
            assert analysis.decision_rationale is not None
    
    def test_optimize_benefits(self):
        """Test benefit optimization"""
        # Create a sample evaluation
        evaluation = RiskBenefitEvaluation(
            crisis_id="test_crisis",
            response_options=self.sample_options,
            recommended_option=self.sample_options[0].id
        )
        
        optimization_result = self.analyzer.optimize_benefits(
            evaluation=evaluation,
            optimization_objective="maximize_total_value"
        )
        
        assert isinstance(optimization_result, OptimizationResult)
        assert optimization_result.evaluation_id == evaluation.id
        assert optimization_result.optimization_objective == "maximize_total_value"
        assert len(optimization_result.optimized_benefits) > 0
        assert len(optimization_result.optimization_strategies) > 0
        assert optimization_result.expected_improvement >= 0.0
        assert 0.0 <= optimization_result.success_probability <= 1.0
        
        # Verify optimized benefits
        for benefit in optimization_result.optimized_benefits:
            assert isinstance(benefit, BenefitFactor)
            assert benefit.expected_value > 0.0
            assert benefit.probability_of_realization > 0.0
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation"""
        confidence = self.analyzer._calculate_confidence_score(self.sample_options[0])
        
        assert 0.0 <= confidence <= 1.0
        
        # Test with high uncertainty option
        high_uncertainty_risks = [
            RiskFactor(
                name="Uncertain Risk",
                uncertainty_level=UncertaintyLevel.VERY_HIGH,
                probability=0.5,
                impact_severity=RiskLevel.MEDIUM
            )
        ]
        
        high_uncertainty_benefits = [
            BenefitFactor(
                name="Uncertain Benefit",
                uncertainty_level=UncertaintyLevel.VERY_HIGH,
                expected_value=0.5,
                probability_of_realization=0.5
            )
        ]
        
        uncertain_option = ResponseOption(
            name="Uncertain Option",
            risks=high_uncertainty_risks,
            benefits=high_uncertainty_benefits
        )
        
        uncertain_confidence = self.analyzer._calculate_confidence_score(uncertain_option)
        assert uncertain_confidence < confidence  # Should be lower confidence
    
    def test_identify_uncertainty_factors(self):
        """Test uncertainty factor identification"""
        uncertainty_factors = self.analyzer._identify_uncertainty_factors(self.sample_options)
        
        assert isinstance(uncertainty_factors, list)
        # Should identify high uncertainty factors from sample data
        assert len(uncertainty_factors) > 0
    
    def test_generate_monitoring_requirements(self):
        """Test monitoring requirements generation"""
        requirements = self.analyzer._generate_monitoring_requirements(self.sample_options[0])
        
        assert isinstance(requirements, list)
        assert len(requirements) > 0
        
        # Should include general monitoring requirements
        general_requirements = [
            "Monitor implementation progress against timeline",
            "Track resource utilization and availability",
            "Assess stakeholder reactions and feedback",
            "Monitor external environment changes"
        ]
        
        for req in general_requirements:
            assert req in requirements
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with no options
        with pytest.raises(Exception):
            self.analyzer.evaluate_response_options(
                crisis_id="test",
                response_options=[],
                evaluation_criteria={}
            )
        
        # Test with invalid risk tolerance
        evaluation = self.analyzer.evaluate_response_options(
            crisis_id="test",
            response_options=self.sample_options,
            evaluation_criteria={"risk_aversion": 0.5},
            risk_tolerance=RiskLevel.MEDIUM
        )
        assert evaluation is not None
        
        # Test optimization with invalid evaluation
        invalid_evaluation = RiskBenefitEvaluation(
            crisis_id="test",
            response_options=[],
            recommended_option="nonexistent"
        )
        
        with pytest.raises(ValueError):
            self.analyzer.optimize_benefits(invalid_evaluation)
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger datasets"""
        # Create larger dataset
        large_risks = []
        large_benefits = []
        
        for i in range(20):
            risk = RiskFactor(
                name=f"Risk {i}",
                probability=0.1 + (i * 0.04),
                impact_severity=RiskLevel.MEDIUM,
                uncertainty_level=UncertaintyLevel.MODERATE
            )
            large_risks.append(risk)
            
            benefit = BenefitFactor(
                name=f"Benefit {i}",
                expected_value=0.1 + (i * 0.04),
                probability_of_realization=0.5 + (i * 0.02),
                benefit_type=BenefitType.OPERATIONAL,
                uncertainty_level=UncertaintyLevel.MODERATE
            )
            large_benefits.append(benefit)
        
        large_options = []
        for i in range(10):
            option = ResponseOption(
                name=f"Option {i}",
                risks=large_risks[i*2:(i*2)+2],
                benefits=large_benefits[i*2:(i*2)+2]
            )
            large_options.append(option)
        
        # Test evaluation with large dataset
        start_time = datetime.now()
        evaluation = self.analyzer.evaluate_response_options(
            crisis_id="large_test",
            response_options=large_options,
            evaluation_criteria={"risk_aversion": 0.4, "benefit_focus": 0.6}
        )
        end_time = datetime.now()
        
        # Should complete within reasonable time (less than 5 seconds)
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 5.0
        
        # Should still produce valid results
        assert evaluation is not None
        assert evaluation.recommended_option is not None
        assert len(evaluation.trade_off_analyses) > 0


@pytest.fixture
def sample_analyzer():
    """Fixture providing a configured analyzer"""
    return RiskBenefitAnalyzer()


@pytest.fixture
def sample_crisis_data():
    """Fixture providing sample crisis data"""
    return {
        "crisis_id": "test_crisis_001",
        "response_options": [
            {
                "name": "Immediate Response",
                "description": "Quick action to address crisis",
                "risks": [
                    {
                        "name": "Implementation Risk",
                        "probability": 0.3,
                        "impact_severity": "medium",
                        "uncertainty_level": "moderate"
                    }
                ],
                "benefits": [
                    {
                        "name": "Quick Resolution",
                        "expected_value": 0.8,
                        "probability_of_realization": 0.7,
                        "benefit_type": "operational",
                        "uncertainty_level": "low"
                    }
                ]
            }
        ],
        "evaluation_criteria": {
            "risk_aversion": 0.4,
            "benefit_focus": 0.6
        }
    }


def test_integration_with_sample_data(sample_analyzer, sample_crisis_data):
    """Integration test with sample data"""
    # This would be used for integration testing
    assert sample_analyzer is not None
    assert sample_crisis_data["crisis_id"] == "test_crisis_001"