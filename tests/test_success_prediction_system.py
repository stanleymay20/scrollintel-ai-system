"""
Tests for Success Prediction System.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.success_prediction_system import SuccessPredictionSystem, SuccessFactorCategory, RiskCategory
from scrollintel.models.validation_models import (
    Innovation, SuccessPrediction, SuccessProbability
)


@pytest.fixture
async def prediction_system():
    """Create and initialize success prediction system for testing."""
    system = SuccessPredictionSystem()
    await system.start()
    return system


@pytest.fixture
def sample_innovation():
    """Create sample innovation for testing."""
    return Innovation(
        title="AI-Powered Healthcare Assistant",
        description="An AI system that assists healthcare professionals with diagnosis and treatment recommendations",
        category="Healthcare Technology",
        domain="Healthcare",
        technology_stack=["Python", "TensorFlow", "React", "PostgreSQL", "AWS"],
        target_market="Healthcare providers",
        problem_statement="Healthcare professionals need better tools for accurate diagnosis",
        proposed_solution="AI-powered assistant that analyzes patient data and provides recommendations",
        unique_value_proposition="Reduces diagnostic errors by 40% and improves treatment outcomes",
        competitive_advantages=["Advanced AI algorithms", "Integration with existing systems"],
        estimated_timeline="18 months",
        estimated_cost=2000000.0,
        potential_revenue=10000000.0,
        risk_factors=["Regulatory approval", "Data privacy concerns"],
        success_metrics=["Diagnostic accuracy", "User adoption rate", "Patient outcomes"]
    )


@pytest.fixture
def high_success_innovation():
    """Create high-success probability innovation for testing."""
    return Innovation(
        title="Revolutionary AI Platform",
        description="Breakthrough AI platform with proven technology and strong market demand",
        category="AI Technology",
        domain="Technology",
        technology_stack=["Python", "JavaScript", "React", "AWS", "PostgreSQL"],  # All mature technologies
        target_market="Global enterprise customers",
        problem_statement="Clear market need for advanced AI solutions",
        proposed_solution="Revolutionary AI platform with proven capabilities",
        unique_value_proposition="Revolutionary breakthrough in AI technology with 10x performance improvement",
        competitive_advantages=["Patent protection", "Proprietary algorithms", "Strong team", "Market validation"],
        estimated_timeline="12 months",
        estimated_cost=1000000.0,
        potential_revenue=100000000.0,  # High revenue potential
        risk_factors=[],  # Minimal risks
        success_metrics=["Performance metrics", "Customer adoption", "Revenue growth", "Market share"]
    )


class TestSuccessPredictionSystem:
    """Test cases for SuccessPredictionSystem class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, prediction_system):
        """Test success prediction system initialization."""
        assert prediction_system.engine_id == "success_prediction_system"
        assert prediction_system.name == "Innovation Success Prediction System"
        assert len(prediction_system.prediction_models) > 0
        assert len(prediction_system.success_factors) > 0
        assert len(prediction_system.risk_factors) > 0
        assert len(prediction_system.success_patterns) > 0
        assert len(prediction_system.failure_patterns) > 0
        assert prediction_system.prediction_context is not None
    
    @pytest.mark.asyncio
    async def test_predict_innovation_success(self, prediction_system, sample_innovation):
        """Test innovation success prediction."""
        # Predict success
        prediction = await prediction_system.predict_innovation_success(sample_innovation)
        
        # Verify prediction structure
        assert isinstance(prediction, SuccessPrediction)
        assert prediction.innovation_id == sample_innovation.id
        assert 0.0 <= prediction.overall_probability <= 1.0
        assert prediction.probability_category in SuccessProbability
        assert 0.0 <= prediction.technical_success_probability <= 1.0
        assert 0.0 <= prediction.market_success_probability <= 1.0
        assert 0.0 <= prediction.financial_success_probability <= 1.0
        assert 0.0 <= prediction.timeline_success_probability <= 1.0
        
        # Verify additional data
        assert len(prediction.key_success_factors) >= 0
        assert len(prediction.critical_risks) >= 0
        assert len(prediction.success_scenarios) > 0
        assert len(prediction.failure_scenarios) > 0
        assert len(prediction.mitigation_strategies) >= 0
        assert len(prediction.optimization_opportunities) >= 0
        assert len(prediction.confidence_intervals) > 0
        assert 0.0 <= prediction.model_accuracy <= 1.0
        assert 0.0 <= prediction.data_quality_score <= 1.0
        assert prediction.created_at is not None
        assert prediction.expires_at is not None
    
    @pytest.mark.asyncio
    async def test_analyze_success_factors(self, prediction_system, sample_innovation):
        """Test success factors analysis."""
        analysis = await prediction_system.analyze_success_factors(sample_innovation)
        
        assert isinstance(analysis, dict)
        
        # Check all success factor categories are analyzed
        for category in SuccessFactorCategory:
            assert category.value in analysis
            category_analysis = analysis[category.value]
            assert "score" in category_analysis
            assert "factors" in category_analysis
            assert 0.0 <= category_analysis["score"] <= 1.0
        
        # Check critical factors and scores
        assert "critical_factors" in analysis
        assert "factor_scores" in analysis
        assert isinstance(analysis["critical_factors"], list)
        assert isinstance(analysis["factor_scores"], dict)
    
    @pytest.mark.asyncio
    async def test_identify_critical_risks(self, prediction_system, sample_innovation):
        """Test critical risks identification."""
        risk_analysis = await prediction_system.identify_critical_risks(sample_innovation)
        
        assert isinstance(risk_analysis, dict)
        
        # Check all risk categories are analyzed
        for category in RiskCategory:
            assert category.value in risk_analysis
            category_risks = risk_analysis[category.value]
            assert "score" in category_risks
            assert "risks" in category_risks
            assert 0.0 <= category_risks["score"] <= 1.0
        
        # Check critical risks and scores
        assert "critical_risks" in risk_analysis
        assert "risk_scores" in risk_analysis
        assert isinstance(risk_analysis["critical_risks"], list)
        assert isinstance(risk_analysis["risk_scores"], dict)
    
    @pytest.mark.asyncio
    async def test_generate_success_scenarios(self, prediction_system, sample_innovation):
        """Test success scenarios generation."""
        scenarios = await prediction_system.generate_success_scenarios(sample_innovation)
        
        assert isinstance(scenarios, list)
        assert len(scenarios) >= 3  # Best, likely, conservative
        
        for scenario in scenarios:
            assert isinstance(scenario, dict)
            assert "scenario" in scenario
            assert "probability" in scenario
            assert "description" in scenario
            assert 0.0 <= scenario["probability"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_generate_failure_scenarios(self, prediction_system, sample_innovation):
        """Test failure scenarios generation."""
        scenarios = await prediction_system.generate_failure_scenarios(sample_innovation)
        
        assert isinstance(scenarios, list)
        assert len(scenarios) >= 4  # Technical, market, financial, competitive
        
        for scenario in scenarios:
            assert isinstance(scenario, dict)
            assert "scenario" in scenario
            assert "probability" in scenario
            assert "description" in scenario
            assert 0.0 <= scenario["probability"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_generate_optimization_strategies(self, prediction_system, sample_innovation):
        """Test optimization strategies generation."""
        strategies = await prediction_system.generate_optimization_strategies(sample_innovation)
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        
        for strategy in strategies:
            assert isinstance(strategy, str)
            assert len(strategy) > 0
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_intervals(self, prediction_system, sample_innovation):
        """Test confidence intervals calculation."""
        intervals = await prediction_system.calculate_confidence_intervals(sample_innovation)
        
        assert isinstance(intervals, dict)
        
        expected_intervals = [
            "overall_success", "technical_success", "market_success", 
            "financial_success", "timeline_success"
        ]
        
        for interval_name in expected_intervals:
            assert interval_name in intervals
            interval = intervals[interval_name]
            assert isinstance(interval, tuple)
            assert len(interval) == 2
            assert 0.0 <= interval[0] <= interval[1] <= 1.0
    
    @pytest.mark.asyncio
    async def test_high_success_innovation_prediction(self, prediction_system, high_success_innovation):
        """Test prediction for high-success probability innovation."""
        prediction = await prediction_system.predict_innovation_success(high_success_innovation)
        
        # High-success innovation should have higher probabilities
        assert prediction.overall_probability > 0.5
        assert prediction.probability_category in [
            SuccessProbability.MEDIUM, SuccessProbability.HIGH, SuccessProbability.VERY_HIGH
        ]
        
        # Should have good technical probability due to mature tech stack
        assert prediction.technical_success_probability > 0.5
        
        # Should have good financial probability due to high revenue potential
        assert prediction.financial_success_probability > 0.5
    
    @pytest.mark.asyncio
    async def test_probability_category_determination(self, prediction_system):
        """Test probability category determination logic."""
        test_cases = [
            (0.9, SuccessProbability.VERY_HIGH),
            (0.7, SuccessProbability.HIGH),
            (0.5, SuccessProbability.MEDIUM),
            (0.3, SuccessProbability.LOW),
            (0.1, SuccessProbability.VERY_LOW)
        ]
        
        for probability, expected_category in test_cases:
            category = prediction_system._determine_probability_category(probability)
            assert category == expected_category
    
    @pytest.mark.asyncio
    async def test_individual_probability_calculations(self, prediction_system, sample_innovation):
        """Test individual probability calculations."""
        # Test technical probability
        tech_prob = await prediction_system._calculate_technical_success_probability(sample_innovation)
        assert 0.0 <= tech_prob <= 1.0
        
        # Test market probability
        market_prob = await prediction_system._calculate_market_success_probability(sample_innovation)
        assert 0.0 <= market_prob <= 1.0
        
        # Test financial probability
        financial_prob = await prediction_system._calculate_financial_success_probability(sample_innovation)
        assert 0.0 <= financial_prob <= 1.0
        
        # Test timeline probability
        timeline_prob = await prediction_system._calculate_timeline_success_probability(sample_innovation)
        assert 0.0 <= timeline_prob <= 1.0
    
    @pytest.mark.asyncio
    async def test_overall_probability_calculation(self, prediction_system):
        """Test overall probability calculation from individual probabilities."""
        # Test with various probability combinations
        test_cases = [
            (0.8, 0.8, 0.8, 0.8),  # All high
            (0.5, 0.5, 0.5, 0.5),  # All medium
            (0.2, 0.2, 0.2, 0.2),  # All low
            (0.9, 0.1, 0.9, 0.1),  # Mixed
        ]
        
        for tech, market, financial, timeline in test_cases:
            overall = await prediction_system._calculate_overall_success_probability(
                tech, market, financial, timeline
            )
            assert 0.0 <= overall <= 1.0
    
    @pytest.mark.asyncio
    async def test_process_method(self, prediction_system, sample_innovation):
        """Test the process method."""
        result = await prediction_system.process(sample_innovation)
        
        assert isinstance(result, SuccessPrediction)
        assert result.innovation_id == sample_innovation.id
    
    @pytest.mark.asyncio
    async def test_process_invalid_input(self, prediction_system):
        """Test process method with invalid input."""
        with pytest.raises(ValueError, match="Input must be Innovation object"):
            await prediction_system.process("invalid_input")
    
    @pytest.mark.asyncio
    async def test_historical_predictions_tracking(self, prediction_system, sample_innovation):
        """Test that predictions are tracked in history."""
        initial_count = len(prediction_system.historical_predictions)
        
        # Perform prediction
        await prediction_system.predict_innovation_success(sample_innovation)
        
        # Check that prediction was added to history
        assert len(prediction_system.historical_predictions) == initial_count + 1
    
    @pytest.mark.asyncio
    async def test_get_status(self, prediction_system):
        """Test getting system status."""
        status = prediction_system.get_status()
        
        assert isinstance(status, dict)
        assert "healthy" in status
        assert "prediction_models_loaded" in status
        assert "success_factors_loaded" in status
        assert "risk_factors_loaded" in status
        assert "historical_predictions" in status
        assert "success_patterns" in status
        assert "failure_patterns" in status
        assert "context_initialized" in status
        assert status["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_health_check(self, prediction_system):
        """Test system health check."""
        is_healthy = await prediction_system.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_cleanup(self, prediction_system, sample_innovation):
        """Test system cleanup."""
        # Add some data
        await prediction_system.predict_innovation_success(sample_innovation)
        
        # Verify data exists
        assert len(prediction_system.historical_predictions) > 0
        
        # Cleanup
        await prediction_system.cleanup()
        
        # Verify cleanup
        assert len(prediction_system.prediction_models) == 0
        assert len(prediction_system.success_factors) == 0
        assert len(prediction_system.risk_factors) == 0
        assert len(prediction_system.historical_predictions) == 0
        assert len(prediction_system.success_patterns) == 0
        assert len(prediction_system.failure_patterns) == 0


class TestSuccessFactorAnalysis:
    """Test cases for success factor analysis."""
    
    @pytest.mark.asyncio
    async def test_technology_maturity_assessment(self, prediction_system, sample_innovation):
        """Test technology maturity assessment."""
        maturity = await prediction_system._assess_technology_maturity(sample_innovation)
        
        assert 0.0 <= maturity <= 1.0
        # Should be high due to mature tech stack (Python, React, AWS, PostgreSQL)
        assert maturity > 0.5
    
    @pytest.mark.asyncio
    async def test_technical_complexity_assessment(self, prediction_system, sample_innovation):
        """Test technical complexity assessment."""
        complexity = await prediction_system._assess_technical_complexity(sample_innovation)
        
        assert 0.0 <= complexity <= 1.0
        # Complexity should be based on technology stack size
        expected_complexity = len(sample_innovation.technology_stack) / 10.0
        assert abs(complexity - expected_complexity) < 0.1
    
    @pytest.mark.asyncio
    async def test_innovation_novelty_assessment(self, prediction_system, sample_innovation):
        """Test innovation novelty assessment."""
        novelty = await prediction_system._assess_innovation_novelty(sample_innovation)
        
        assert 0.0 <= novelty <= 1.0
        # Should have some novelty due to AI content
        assert novelty > 0.3
    
    @pytest.mark.asyncio
    async def test_market_size_score_assessment(self, prediction_system, sample_innovation):
        """Test market size score assessment."""
        market_score = await prediction_system._assess_market_size_score(sample_innovation)
        
        assert 0.0 <= market_score <= 1.0
        # Should be medium due to $10M revenue potential
        assert 0.4 <= market_score <= 0.8
    
    @pytest.mark.asyncio
    async def test_roi_potential_assessment(self, prediction_system, sample_innovation):
        """Test ROI potential assessment."""
        roi_score = await prediction_system._assess_roi_potential(sample_innovation)
        
        assert 0.0 <= roi_score <= 1.0
        # Should be positive due to good revenue vs cost ratio
        assert roi_score > 0.0


class TestRiskAnalysis:
    """Test cases for risk analysis."""
    
    @pytest.mark.asyncio
    async def test_financial_risk_assessment(self, prediction_system, sample_innovation):
        """Test financial risk assessment."""
        risk = await prediction_system._assess_financial_risk(sample_innovation)
        
        assert 0.0 <= risk <= 1.0
        # Should have some risk but not maximum
        assert 0.2 <= risk <= 0.8
    
    @pytest.mark.asyncio
    async def test_regulatory_complexity_assessment(self, prediction_system, sample_innovation):
        """Test regulatory complexity assessment."""
        complexity = await prediction_system._assess_regulatory_complexity(sample_innovation)
        
        assert 0.0 <= complexity <= 1.0
        # Healthcare domain should have high regulatory complexity
        assert complexity > 0.5


class TestScenarioGeneration:
    """Test cases for scenario generation."""
    
    @pytest.mark.asyncio
    async def test_scenario_probability_consistency(self, prediction_system, sample_innovation):
        """Test that scenario probabilities are consistent."""
        success_scenarios = await prediction_system.generate_success_scenarios(sample_innovation)
        failure_scenarios = await prediction_system.generate_failure_scenarios(sample_innovation)
        
        # All probabilities should be valid
        for scenario in success_scenarios + failure_scenarios:
            assert 0.0 <= scenario["probability"] <= 1.0
        
        # Success scenarios should generally have lower probabilities than 1.0
        for scenario in success_scenarios:
            assert scenario["probability"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_scenario_content_quality(self, prediction_system, sample_innovation):
        """Test quality of scenario content."""
        scenarios = await prediction_system.generate_success_scenarios(sample_innovation)
        
        for scenario in scenarios:
            assert len(scenario["description"]) > 10  # Meaningful description
            assert "scenario" in scenario
            assert scenario["scenario"] != ""


class TestSuccessPredictionIntegration:
    """Integration tests for success prediction system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_prediction_flow(self, prediction_system, sample_innovation):
        """Test complete prediction flow from innovation to prediction."""
        # Step 1: Predict success
        prediction = await prediction_system.predict_innovation_success(sample_innovation)
        
        # Step 2: Verify comprehensive prediction
        assert isinstance(prediction, SuccessPrediction)
        assert prediction.innovation_id == sample_innovation.id
        
        # Step 3: Verify all probability dimensions
        assert 0.0 <= prediction.overall_probability <= 1.0
        assert 0.0 <= prediction.technical_success_probability <= 1.0
        assert 0.0 <= prediction.market_success_probability <= 1.0
        assert 0.0 <= prediction.financial_success_probability <= 1.0
        assert 0.0 <= prediction.timeline_success_probability <= 1.0
        
        # Step 4: Verify analysis components
        assert len(prediction.success_scenarios) > 0
        assert len(prediction.failure_scenarios) > 0
        assert len(prediction.confidence_intervals) > 0
        
        # Step 5: Verify prediction is in history
        assert prediction in prediction_system.historical_predictions
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_predictions(self, prediction_system):
        """Test handling multiple concurrent predictions."""
        innovations = [
            Innovation(title=f"Innovation {i}", domain="Technology", estimated_cost=1000000.0, potential_revenue=5000000.0)
            for i in range(3)
        ]
        
        # Start concurrent predictions
        tasks = [
            prediction_system.predict_innovation_success(innovation)
            for innovation in innovations
        ]
        
        # Wait for all to complete
        predictions = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        assert len(predictions) == 3
        for prediction in predictions:
            assert isinstance(prediction, SuccessPrediction)
            assert prediction.created_at is not None
    
    @pytest.mark.asyncio
    async def test_prediction_with_minimal_data(self, prediction_system):
        """Test prediction with minimal innovation data."""
        minimal_innovation = Innovation(
            title="Minimal Innovation",
            domain="Technology"
            # Missing many fields
        )
        
        # Should still complete prediction
        prediction = await prediction_system.predict_innovation_success(minimal_innovation)
        
        assert isinstance(prediction, SuccessPrediction)
        assert prediction.innovation_id == minimal_innovation.id
        # Should have reasonable default values
        assert 0.0 <= prediction.overall_probability <= 1.0
        assert prediction.data_quality_score < 1.0  # Should reflect missing data


if __name__ == "__main__":
    pytest.main([__file__])