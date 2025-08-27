"""
Comprehensive integration tests for A/B Testing Engine.
Tests complete statistical analysis, experiment scheduling, automation,
winner selection, and results visualization.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.experiment_engine import (
    ExperimentEngine, ExperimentConfig, StatisticalAnalysis, 
    ExperimentResults, TestType
)
from scrollintel.models.experiment_models import (
    ExperimentStatus, StatisticalSignificance
)


class TestABTestingIntegration:
    """Integration tests for A/B Testing Engine."""
    
    @pytest.fixture
    async def experiment_engine(self):
        """Create and initialize experiment engine."""
        engine = ExperimentEngine()
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    @pytest.fixture
    def sample_experiment_config(self):
        """Sample experiment configuration."""
        return ExperimentConfig(
            name="Prompt Optimization Test",
            prompt_id="prompt_123",
            hypothesis="New prompt will improve accuracy by 10%",
            variants=[
                {
                    "name": "Control",
                    "prompt_content": "Original prompt content",
                    "type": "control",
                    "traffic_weight": 0.5
                },
                {
                    "name": "Treatment",
                    "prompt_content": "Optimized prompt content",
                    "type": "treatment",
                    "traffic_weight": 0.5
                }
            ],
            success_metrics=["accuracy", "response_time", "user_satisfaction"],
            target_sample_size=1000,
            confidence_level=0.95,
            minimum_effect_size=0.1,
            duration_hours=24
        )
    
    @pytest.mark.asyncio
    async def test_create_experiment(self, experiment_engine, sample_experiment_config):
        """Test experiment creation."""
        result = await experiment_engine.create_experiment(sample_experiment_config)
        
        assert result["status"] == "created"
        assert "experiment_id" in result
        assert result["variants_count"] == 2
        assert result["target_sample_size"] == 1000
        
        # Verify experiment is stored
        experiment_id = result["experiment_id"]
        assert experiment_id in experiment_engine.running_experiments
    
    @pytest.mark.asyncio
    async def test_start_stop_experiment(self, experiment_engine, sample_experiment_config):
        """Test starting and stopping experiments."""
        # Create experiment
        create_result = await experiment_engine.create_experiment(sample_experiment_config)
        experiment_id = create_result["experiment_id"]
        
        # Start experiment
        start_result = await experiment_engine.start_experiment(experiment_id)
        assert start_result["status"] == "running"
        assert "start_date" in start_result
        
        # Verify experiment status
        experiment = experiment_engine.running_experiments[experiment_id]
        assert experiment.status == ExperimentStatus.RUNNING.value
        assert experiment.start_date is not None
        
        # Stop experiment
        stop_result = await experiment_engine.stop_experiment(experiment_id)
        assert stop_result["status"] == "completed"
        assert "end_date" in stop_result
        assert "results" in stop_result
    
    @pytest.mark.asyncio
    async def test_statistical_analysis(self, experiment_engine):
        """Test comprehensive statistical analysis."""
        # Mock variant data with different performance
        control_data = [0.80, 0.82, 0.78, 0.81, 0.79] * 20  # 100 samples
        treatment_data = [0.85, 0.87, 0.83, 0.86, 0.84] * 20  # 100 samples, higher mean
        
        variant_data = {
            "control_variant": [{"accuracy": val} for val in control_data],
            "treatment_variant": [{"accuracy": val} for val in treatment_data]
        }
        
        # Perform analysis
        analysis = await experiment_engine._perform_statistical_analysis(
            variant_data, "accuracy", 0.95
        )
        
        assert analysis is not None
        assert analysis.metric_name == "accuracy"
        assert analysis.control_mean < analysis.treatment_mean
        assert analysis.effect_size > 0
        assert analysis.p_value < 0.05  # Should be significant
        assert analysis.statistical_significance != StatisticalSignificance.NOT_SIGNIFICANT
        assert len(analysis.confidence_interval) == 2
        assert analysis.statistical_power > 0
        assert "recommend" in analysis.recommendation.lower()
    
    @pytest.mark.asyncio
    async def test_experiment_results_analysis(self, experiment_engine, sample_experiment_config):
        """Test complete experiment results analysis."""
        # Create and start experiment
        create_result = await experiment_engine.create_experiment(sample_experiment_config)
        experiment_id = create_result["experiment_id"]
        await experiment_engine.start_experiment(experiment_id)
        
        # Mock the _get_variant_metrics method to return test data
        def mock_get_metrics(variant_id):
            if "control" in variant_id.lower():
                return [
                    {"accuracy": 0.80, "response_time": 1.2, "user_satisfaction": 4.0},
                    {"accuracy": 0.82, "response_time": 1.1, "user_satisfaction": 4.1},
                    {"accuracy": 0.78, "response_time": 1.3, "user_satisfaction": 3.9}
                ] * 50  # 150 samples
            else:
                return [
                    {"accuracy": 0.85, "response_time": 1.0, "user_satisfaction": 4.3},
                    {"accuracy": 0.87, "response_time": 0.9, "user_satisfaction": 4.4},
                    {"accuracy": 0.83, "response_time": 1.1, "user_satisfaction": 4.2}
                ] * 50  # 150 samples
        
        experiment_engine._get_variant_metrics = mock_get_metrics
        
        # Analyze results
        results = await experiment_engine.analyze_experiment_results(experiment_id)
        
        assert isinstance(results, ExperimentResults)
        assert results.experiment_id == experiment_id
        assert len(results.analyses) == 3  # Three success metrics
        assert results.confidence_level == 0.95
        assert results.total_sample_size > 0
        assert "visualizations" in results.__dict__
        
        # Check individual analyses
        for analysis in results.analyses:
            assert analysis.metric_name in ["accuracy", "response_time", "user_satisfaction"]
            assert analysis.control_sample_size > 0
            assert analysis.treatment_sample_size > 0
            assert analysis.p_value >= 0
            assert analysis.statistical_power >= 0
    
    @pytest.mark.asyncio
    async def test_winner_selection(self, experiment_engine, sample_experiment_config):
        """Test winner selection and promotion."""
        # Create experiment with clear winner
        create_result = await experiment_engine.create_experiment(sample_experiment_config)
        experiment_id = create_result["experiment_id"]
        
        # Mock strong treatment effect
        def mock_get_metrics(variant_id):
            if "control" in variant_id.lower():
                return [{"accuracy": 0.70}] * 100
            else:
                return [{"accuracy": 0.90}] * 100  # Much better performance
        
        experiment_engine._get_variant_metrics = mock_get_metrics
        
        # Promote winner
        promotion_result = await experiment_engine.promote_winner(experiment_id)
        
        assert promotion_result["status"] == "promoted"
        assert "winner_variant_id" in promotion_result
        assert promotion_result["experiment_id"] == experiment_id
    
    @pytest.mark.asyncio
    async def test_no_winner_scenario(self, experiment_engine, sample_experiment_config):
        """Test scenario where no clear winner exists."""
        create_result = await experiment_engine.create_experiment(sample_experiment_config)
        experiment_id = create_result["experiment_id"]
        
        # Mock similar performance (no significant difference)
        def mock_get_metrics(variant_id):
            return [{"accuracy": 0.80 + np.random.normal(0, 0.01)}] * 50
        
        experiment_engine._get_variant_metrics = mock_get_metrics
        
        promotion_result = await experiment_engine.promote_winner(experiment_id)
        
        assert promotion_result["status"] == "no_winner"
        assert "no statistically significant winner" in promotion_result["message"].lower()
    
    @pytest.mark.asyncio
    async def test_experiment_scheduling(self, experiment_engine):
        """Test experiment scheduling functionality."""
        config = ExperimentConfig(
            name="Scheduled Test",
            prompt_id="prompt_456",
            hypothesis="Scheduled experiment test",
            variants=[
                {"name": "Control", "prompt_content": "Control", "type": "control"},
                {"name": "Treatment", "prompt_content": "Treatment", "type": "treatment"}
            ],
            success_metrics=["accuracy"],
            auto_start=True,
            auto_stop=True,
            schedule_config={
                "type": "daily",
                "cron_expression": "0 9 * * *"  # Daily at 9 AM
            }
        )
        
        result = await experiment_engine.create_experiment(config)
        assert result["status"] == "created"
        
        # Verify scheduler is running
        status = experiment_engine.get_status()
        assert status["scheduler_active"] is True
    
    @pytest.mark.asyncio
    async def test_multiple_metrics_analysis(self, experiment_engine):
        """Test analysis with multiple success metrics."""
        # Create experiment with multiple metrics
        config = ExperimentConfig(
            name="Multi-Metric Test",
            prompt_id="prompt_789",
            hypothesis="Test multiple metrics",
            variants=[
                {"name": "Control", "prompt_content": "Control", "type": "control"},
                {"name": "Treatment", "prompt_content": "Treatment", "type": "treatment"}
            ],
            success_metrics=["accuracy", "response_time", "user_satisfaction", "conversion_rate"]
        )
        
        create_result = await experiment_engine.create_experiment(config)
        experiment_id = create_result["experiment_id"]
        
        # Mock diverse metric data
        def mock_get_metrics(variant_id):
            if "control" in variant_id.lower():
                return [
                    {
                        "accuracy": 0.80,
                        "response_time": 1.5,
                        "user_satisfaction": 4.0,
                        "conversion_rate": 0.15
                    }
                ] * 100
            else:
                return [
                    {
                        "accuracy": 0.85,  # Better
                        "response_time": 1.2,  # Better (lower)
                        "user_satisfaction": 4.3,  # Better
                        "conversion_rate": 0.18  # Better
                    }
                ] * 100
        
        experiment_engine._get_variant_metrics = mock_get_metrics
        
        results = await experiment_engine.analyze_experiment_results(experiment_id)
        
        assert len(results.analyses) == 4
        metric_names = [a.metric_name for a in results.analyses]
        assert "accuracy" in metric_names
        assert "response_time" in metric_names
        assert "user_satisfaction" in metric_names
        assert "conversion_rate" in metric_names
    
    @pytest.mark.asyncio
    async def test_statistical_test_selection(self, experiment_engine):
        """Test automatic selection of appropriate statistical tests."""
        # Test with small sample (should use Mann-Whitney)
        small_sample_data = {
            "control": [{"metric": val} for val in [1, 2, 3, 4, 5]],
            "treatment": [{"metric": val} for val in [2, 3, 4, 5, 6]]
        }
        
        analysis = await experiment_engine._perform_statistical_analysis(
            small_sample_data, "metric", 0.95
        )
        
        assert analysis.test_type == TestType.MANN_WHITNEY.value
        
        # Test with binary data (should use proportion test)
        binary_data = {
            "control": [{"converted": val} for val in [0, 1, 0, 1, 0] * 20],
            "treatment": [{"converted": val} for val in [1, 1, 0, 1, 1] * 20]
        }
        
        analysis = await experiment_engine._perform_statistical_analysis(
            binary_data, "converted", 0.95
        )
        
        assert analysis.test_type == TestType.PROPORTION_TEST.value
    
    @pytest.mark.asyncio
    async def test_visualization_generation(self, experiment_engine, sample_experiment_config):
        """Test generation of visualization data."""
        create_result = await experiment_engine.create_experiment(sample_experiment_config)
        experiment_id = create_result["experiment_id"]
        
        # Mock data for visualization
        def mock_get_metrics(variant_id):
            return [
                {"accuracy": 0.80, "response_time": 1.2},
                {"accuracy": 0.82, "response_time": 1.1}
            ] * 25
        
        experiment_engine._get_variant_metrics = mock_get_metrics
        
        results = await experiment_engine.analyze_experiment_results(experiment_id)
        
        assert "visualizations" in results.__dict__
        viz = results.visualizations
        
        assert "metric_comparisons" in viz
        assert "confidence_intervals" in viz
        assert "sample_sizes" in viz
        
        # Check metric comparisons structure
        for comparison in viz["metric_comparisons"]:
            assert "metric" in comparison
            assert "control_mean" in comparison
            assert "treatment_mean" in comparison
            assert "p_value" in comparison
            assert "effect_size" in comparison
    
    @pytest.mark.asyncio
    async def test_engine_status_reporting(self, experiment_engine, sample_experiment_config):
        """Test engine status reporting."""
        initial_status = experiment_engine.get_status()
        
        assert initial_status["engine_id"] == "experiment_engine"
        assert initial_status["name"] == "ExperimentEngine"
        assert initial_status["healthy"] is True
        assert initial_status["running_experiments"] == 0
        assert initial_status["scheduler_active"] is True
        
        # Create experiment and check status update
        await experiment_engine.create_experiment(sample_experiment_config)
        
        updated_status = experiment_engine.get_status()
        assert updated_status["running_experiments"] == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, experiment_engine):
        """Test error handling in various scenarios."""
        # Test with invalid experiment ID
        result = await experiment_engine.start_experiment("invalid_id")
        assert result["status"] == "failed"
        assert "not found" in result["error"].lower()
        
        # Test promotion with invalid experiment
        result = await experiment_engine.promote_winner("invalid_id")
        assert result["status"] == "failed"
        
        # Test analysis with missing data
        with patch.object(experiment_engine, '_get_variant_metrics', return_value=[]):
            try:
                await experiment_engine.analyze_experiment_results("invalid_id")
                assert False, "Should have raised an exception"
            except ValueError:
                pass  # Expected
    
    @pytest.mark.asyncio
    async def test_confidence_interval_calculation(self, experiment_engine):
        """Test confidence interval calculations."""
        # Create controlled data with known properties
        control_data = [0.8] * 100  # Mean = 0.8, std = 0
        treatment_data = [0.9] * 100  # Mean = 0.9, std = 0
        
        variant_data = {
            "control": [{"accuracy": val} for val in control_data],
            "treatment": [{"accuracy": val} for val in treatment_data]
        }
        
        analysis = await experiment_engine._perform_statistical_analysis(
            variant_data, "accuracy", 0.95
        )
        
        assert analysis is not None
        assert len(analysis.confidence_interval) == 2
        assert analysis.confidence_interval[0] < analysis.confidence_interval[1]
        
        # The difference should be within the confidence interval
        mean_diff = analysis.treatment_mean - analysis.control_mean
        assert (analysis.confidence_interval[0] <= mean_diff <= 
                analysis.confidence_interval[1])
    
    @pytest.mark.asyncio
    async def test_power_analysis(self, experiment_engine):
        """Test statistical power calculations."""
        # Test with large effect size (should have high power)
        large_effect_power = experiment_engine._calculate_statistical_power(
            effect_size=0.8, n1=100, n2=100, alpha=0.05
        )
        assert large_effect_power > 0.8
        
        # Test with small effect size (should have lower power)
        small_effect_power = experiment_engine._calculate_statistical_power(
            effect_size=0.1, n1=20, n2=20, alpha=0.05
        )
        assert small_effect_power < large_effect_power
    
    @pytest.mark.asyncio
    async def test_recommendation_generation(self, experiment_engine):
        """Test recommendation generation logic."""
        # Strong positive effect
        rec1 = experiment_engine._generate_recommendation(
            effect_size=0.5, p_value=0.001, 
            significance=StatisticalSignificance.HIGHLY_SIGNIFICANT, power=0.9
        )
        assert "strong evidence" in rec1.lower()
        assert "implement" in rec1.lower()
        
        # Moderate effect
        rec2 = experiment_engine._generate_recommendation(
            effect_size=0.15, p_value=0.03,
            significance=StatisticalSignificance.SIGNIFICANT, power=0.8
        )
        assert "moderate evidence" in rec2.lower()
        
        # Low power
        rec3 = experiment_engine._generate_recommendation(
            effect_size=0.1, p_value=0.1,
            significance=StatisticalSignificance.NOT_SIGNIFICANT, power=0.5
        )
        assert "insufficient" in rec3.lower() or "sample size" in rec3.lower()
    
    def test_experiment_config_validation(self):
        """Test experiment configuration validation."""
        # Valid config
        config = ExperimentConfig(
            name="Test",
            prompt_id="123",
            hypothesis="Test hypothesis",
            variants=[{"name": "A", "prompt_content": "Content A"}],
            success_metrics=["accuracy"]
        )
        
        assert config.name == "Test"
        assert config.confidence_level == 0.95  # Default
        assert config.minimum_effect_size == 0.05  # Default
        assert config.target_sample_size == 1000  # Default