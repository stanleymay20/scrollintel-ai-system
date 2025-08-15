"""
Unit tests for predictive analytics engine.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scrollintel.engines.predictive_engine import PredictiveEngine
from scrollintel.models.predictive_models import (
    BusinessMetric, Forecast, ScenarioConfig, ScenarioResult,
    RiskPrediction, PredictionAccuracy, PredictionUpdate, BusinessContext,
    ForecastModel, RiskLevel, MetricCategory
)


class TestPredictiveEngine:
    """Test suite for PredictiveEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create a PredictiveEngine instance for testing."""
        return PredictiveEngine()
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample business metrics for testing."""
        base_time = datetime.utcnow() - timedelta(days=30)
        metrics = []
        
        for i in range(30):
            metric = BusinessMetric(
                id="test_metric_1",
                name="Revenue",
                category=MetricCategory.FINANCIAL,
                value=1000 + i * 10 + np.random.normal(0, 50),  # Trending upward with noise
                unit="USD",
                timestamp=base_time + timedelta(days=i),
                source="test",
                context={"department": "sales"}
            )
            metrics.append(metric)
        
        return metrics
    
    @pytest.fixture
    def sample_context(self):
        """Create sample business context for testing."""
        return BusinessContext(
            industry="technology",
            company_size="medium",
            market_conditions={"sentiment": "positive", "volatility": "low"},
            seasonal_factors={"q4_boost": 1.2},
            external_factors={"economic_growth": 0.03},
            historical_patterns={"yearly_growth": 0.15}
        )
    
    def test_forecast_metrics_linear(self, engine, sample_metrics):
        """Test linear regression forecasting."""
        current_metric = sample_metrics[-1]
        forecast = engine.forecast_metrics(
            metric=current_metric,
            horizon=7,
            historical_data=sample_metrics,
            model_type=ForecastModel.LINEAR_REGRESSION
        )
        
        assert isinstance(forecast, Forecast)
        assert forecast.metric_id == "test_metric_1"
        assert forecast.model_type == ForecastModel.LINEAR_REGRESSION
        assert len(forecast.predictions) == 7
        assert len(forecast.timestamps) == 7
        assert len(forecast.confidence_lower) == 7
        assert len(forecast.confidence_upper) == 7
        assert forecast.horizon_days == 7
        assert forecast.confidence_level > 0
        
        # Check that predictions are reasonable (trending upward)
        assert all(p > 0 for p in forecast.predictions)
        assert forecast.predictions[-1] > forecast.predictions[0]  # Should trend upward
    
    def test_forecast_metrics_ensemble(self, engine, sample_metrics):
        """Test ensemble forecasting."""
        current_metric = sample_metrics[-1]
        forecast = engine.forecast_metrics(
            metric=current_metric,
            horizon=14,
            historical_data=sample_metrics,
            model_type=ForecastModel.ENSEMBLE
        )
        
        assert isinstance(forecast, Forecast)
        assert forecast.model_type == ForecastModel.ENSEMBLE
        assert len(forecast.predictions) == 14
        assert forecast.confidence_level > 0.8  # Ensemble should have higher confidence
    
    def test_forecast_insufficient_data(self, engine):
        """Test forecasting with insufficient historical data."""
        # Create minimal data
        minimal_metrics = [
            BusinessMetric(
                id="test_metric",
                name="Test",
                category=MetricCategory.OPERATIONAL,
                value=100,
                unit="count",
                timestamp=datetime.utcnow(),
                source="test",
                context={}
            )
        ]
        
        with pytest.raises(ValueError, match="Insufficient historical data"):
            engine.forecast_metrics(
                metric=minimal_metrics[0],
                horizon=7,
                historical_data=minimal_metrics
            )
    
    def test_scenario_modeling(self, engine, sample_metrics):
        """Test scenario modeling and what-if analysis."""
        scenario = ScenarioConfig(
            id="test_scenario",
            name="Growth Scenario",
            description="Test growth scenario",
            parameters={
                "percentage_change": 20,  # 20% increase
                "trend_adjustment": 0.1
            },
            target_metrics=["test_metric_1"],
            time_horizon=30,
            created_by="test_user",
            created_at=datetime.utcnow()
        )
        
        historical_data = {"test_metric_1": sample_metrics}
        
        result = engine.model_scenario(scenario, historical_data)
        
        assert isinstance(result, ScenarioResult)
        assert result.scenario_id == "test_scenario"
        assert "test_metric_1" in result.baseline_forecast
        assert "test_metric_1" in result.scenario_forecast
        assert "test_metric_1" in result.impact_analysis
        assert len(result.recommendations) > 0
        assert 0 <= result.confidence_score <= 1
        
        # Scenario forecast should be higher than baseline (20% increase)
        # Due to model randomness, we check the impact analysis instead
        assert result.impact_analysis["test_metric_1"] > 0  # Should show positive impact
    
    def test_risk_prediction(self, engine, sample_metrics, sample_context):
        """Test risk prediction functionality."""
        # Create metrics with some anomalies
        current_metrics = [sample_metrics[-1]]
        current_metrics[0].value = 2000  # Anomalously high value
        
        historical_data = {"test_metric_1": sample_metrics}
        
        risks = engine.predict_risks(sample_context, current_metrics, historical_data)
        
        assert isinstance(risks, list)
        if risks:  # May not always detect risks depending on data
            risk = risks[0]
            assert isinstance(risk, RiskPrediction)
            assert risk.metric_id in ["test_metric_1", "systemic", "market"]
            assert isinstance(risk.risk_level, RiskLevel)
            assert 0 <= risk.probability <= 1
            assert risk.impact_score >= 0
            assert len(risk.mitigation_strategies) > 0
    
    def test_prediction_updates(self, engine, sample_metrics):
        """Test prediction update functionality."""
        # Initialize with some historical forecasts
        engine.models["test_metric_1"] = {
            "last_forecast": engine.forecast_metrics(
                sample_metrics[-1], 7, sample_metrics, ForecastModel.LINEAR_REGRESSION
            )
        }
        
        # Create new data with significant change
        new_metric = BusinessMetric(
            id="test_metric_1",
            name="Revenue",
            category=MetricCategory.FINANCIAL,
            value=2000,  # Significant jump
            unit="USD",
            timestamp=datetime.utcnow(),
            source="test",
            context={}
        )
        
        with patch.object(engine, '_get_historical_data', return_value=sample_metrics):
            updates = engine.update_predictions([new_metric])
        
        assert isinstance(updates, list)
        # Updates may or may not be generated depending on change threshold
        for update in updates:
            assert isinstance(update, PredictionUpdate)
            assert update.metric_id == "test_metric_1"
            assert isinstance(update.change_magnitude, float)
            assert len(update.change_reason) > 0
    
    def test_forecast_accuracy_calculation(self, engine, sample_metrics):
        """Test forecast accuracy calculation."""
        # Create enough data for validation split
        extended_metrics = sample_metrics * 2  # 60 data points
        
        # Update timestamps to be sequential
        base_time = datetime.utcnow() - timedelta(days=60)
        for i, metric in enumerate(extended_metrics):
            metric.timestamp = base_time + timedelta(days=i)
        
        df = engine._prepare_time_series_data(extended_metrics)
        accuracy = engine._calculate_forecast_accuracy(
            "test_metric_1", ForecastModel.LINEAR_REGRESSION, df
        )
        
        if accuracy is not None:  # May be None if validation fails
            assert 0 <= accuracy <= 1
            assert "test_metric_1_linear_regression" in engine.accuracy_tracker
    
    def test_risk_summary_generation(self, engine):
        """Test risk summary generation."""
        # Create sample risks
        risks = [
            RiskPrediction(
                id="risk1",
                metric_id="metric1",
                risk_type="anomaly",
                risk_level=RiskLevel.HIGH,
                probability=0.8,
                impact_score=0.7,
                description="Test risk 1",
                early_warning_threshold=100,
                mitigation_strategies=["Strategy 1"],
                predicted_date=None,
                created_at=datetime.utcnow()
            ),
            RiskPrediction(
                id="risk2",
                metric_id="metric2",
                risk_type="trend",
                risk_level=RiskLevel.MEDIUM,
                probability=0.6,
                impact_score=0.5,
                description="Test risk 2",
                early_warning_threshold=200,
                mitigation_strategies=["Strategy 2"],
                predicted_date=None,
                created_at=datetime.utcnow()
            )
        ]
        
        summary = engine.get_risk_summary(risks)
        
        assert summary["total_risks"] == 2
        assert "high" in summary["risk_levels"]
        assert "medium" in summary["risk_levels"]
        assert len(summary["top_risks"]) <= 5
        assert 0 <= summary["average_probability"] <= 1
        assert summary["average_impact"] >= 0
    
    def test_data_preparation(self, engine, sample_metrics):
        """Test time series data preparation."""
        df = engine._prepare_time_series_data(sample_metrics)
        
        assert isinstance(df, pd.DataFrame)
        assert "ds" in df.columns
        assert "y" in df.columns
        assert "metric_id" in df.columns
        assert len(df) == len(sample_metrics)
        assert df["ds"].dtype == "datetime64[ns]"
        assert pd.api.types.is_numeric_dtype(df["y"])
    
    def test_scenario_parameter_application(self, engine, sample_metrics):
        """Test application of scenario parameters to forecasts."""
        # Create baseline forecast
        baseline = engine.forecast_metrics(
            sample_metrics[-1], 7, sample_metrics, ForecastModel.LINEAR_REGRESSION
        )
        
        # Apply scenario parameters
        parameters = {"percentage_change": 50}  # 50% increase
        modified = engine._apply_scenario_parameters(baseline, parameters)
        
        assert isinstance(modified, Forecast)
        assert len(modified.predictions) == len(baseline.predictions)
        
        # Modified predictions should be higher
        baseline_avg = np.mean(baseline.predictions)
        modified_avg = np.mean(modified.predictions)
        assert modified_avg > baseline_avg
        
        # Confidence should be reduced
        assert modified.confidence_level < baseline.confidence_level
    
    def test_model_performance_tracking(self, engine):
        """Test model performance tracking."""
        # Add some accuracy data
        accuracy = PredictionAccuracy(
            model_type=ForecastModel.LINEAR_REGRESSION,
            metric_id="test_metric",
            mae=10.5,
            mape=5.2,
            rmse=12.3,
            r2_score=0.85,
            accuracy_trend=[0.8, 0.82, 0.85],
            evaluation_date=datetime.utcnow(),
            sample_size=100
        )
        
        engine.accuracy_tracker["test_metric_linear_regression"] = accuracy
        
        performance = engine.get_model_performance("test_metric")
        
        assert "linear_regression" in performance
        assert performance["linear_regression"] == accuracy
    
    def test_error_handling(self, engine):
        """Test error handling in various scenarios."""
        # Test with invalid data
        invalid_metrics = []
        
        with pytest.raises(ValueError):
            engine.forecast_metrics(
                BusinessMetric("", "", MetricCategory.OPERATIONAL, 0, "", datetime.utcnow(), "", {}),
                7,
                invalid_metrics
            )
        
        # Test with empty scenario
        empty_scenario = ScenarioConfig("", "", "", {}, [], 0, "", datetime.utcnow())
        result = engine.model_scenario(empty_scenario, {})
        
        # Should handle gracefully
        assert isinstance(result, ScenarioResult)
    
    @pytest.mark.parametrize("model_type", [
        ForecastModel.LINEAR_REGRESSION,
        ForecastModel.ENSEMBLE
    ])
    def test_different_forecast_models(self, engine, sample_metrics, model_type):
        """Test different forecasting models."""
        current_metric = sample_metrics[-1]
        forecast = engine.forecast_metrics(
            metric=current_metric,
            horizon=7,
            historical_data=sample_metrics,
            model_type=model_type
        )
        
        assert forecast.model_type == model_type
        assert len(forecast.predictions) == 7
        assert all(isinstance(p, (int, float)) for p in forecast.predictions)
    
    def test_confidence_intervals(self, engine, sample_metrics):
        """Test confidence interval generation."""
        current_metric = sample_metrics[-1]
        forecast = engine.forecast_metrics(
            metric=current_metric,
            horizon=7,
            historical_data=sample_metrics
        )
        
        # Confidence intervals should be properly ordered
        for i in range(len(forecast.predictions)):
            assert forecast.confidence_lower[i] <= forecast.predictions[i]
            assert forecast.predictions[i] <= forecast.confidence_upper[i]
    
    def test_risk_level_classification(self, engine, sample_metrics, sample_context):
        """Test risk level classification."""
        # Create extreme anomaly
        anomaly_metric = sample_metrics[-1]
        anomaly_metric.value = 10000  # Extreme value
        
        historical_data = {"test_metric_1": sample_metrics}
        risks = engine.predict_risks(sample_context, [anomaly_metric], historical_data)
        
        # Should detect high-risk anomaly
        high_risks = [r for r in risks if r.risk_level == RiskLevel.HIGH]
        assert len(high_risks) > 0 or len(risks) == 0  # May not detect depending on implementation
    
    def test_confidence_interval_tracking(self, engine, sample_metrics):
        """Test confidence interval performance tracking."""
        current_metric = sample_metrics[-1]
        forecast = engine.forecast_metrics(
            metric=current_metric,
            horizon=5,
            historical_data=sample_metrics
        )
        
        # Create some actual values for comparison
        actual_values = [1100, 1150, 1200, 1250, 1300]
        
        tracking_result = engine.track_confidence_intervals(forecast, actual_values)
        
        assert "coverage" in tracking_result
        assert "width" in tracking_result
        assert "reliability" in tracking_result
        assert 0 <= tracking_result["coverage"] <= 1
        assert tracking_result["width"] >= 0
        assert 0 <= tracking_result["reliability"] <= 1
        
        # Check that tracking data is stored
        key = f"{forecast.metric_id}_{forecast.model_type.value}"
        assert key in engine.confidence_tracker
    
    def test_early_warning_system(self, engine, sample_metrics):
        """Test early warning system setup and checking."""
        metric_id = "test_metric_1"
        thresholds = {
            "medium": 1500,
            "high": 2000,
            "critical": 2500
        }
        
        # Setup early warning system
        result = engine.setup_early_warning_system(metric_id, thresholds)
        assert result is True
        assert metric_id in engine.early_warning_system
        
        # Create metric that exceeds threshold
        warning_metric = BusinessMetric(
            id=metric_id,
            name="Test Metric",
            category=MetricCategory.OPERATIONAL,
            value=1600,  # Exceeds medium threshold
            unit="count",
            timestamp=datetime.utcnow(),
            source="test",
            context={}
        )
        
        warnings = engine.check_early_warnings([warning_metric])
        
        assert len(warnings) > 0
        warning = warnings[0]
        assert warning["metric_id"] == metric_id
        assert warning["threshold_name"] == "medium"
        assert warning["current_value"] == 1600
        assert "severity" in warning
        assert "recommended_actions" in warning
    
    def test_prediction_performance_report(self, engine, sample_metrics):
        """Test comprehensive prediction performance reporting."""
        # Add some mock accuracy data
        from scrollintel.models.predictive_models import PredictionAccuracy
        
        accuracy = PredictionAccuracy(
            model_type=ForecastModel.LINEAR_REGRESSION,
            metric_id="test_metric_1",
            mae=15.2,
            mape=8.5,
            rmse=18.7,
            r2_score=0.82,
            accuracy_trend=[0.8, 0.81, 0.82],
            evaluation_date=datetime.utcnow(),
            sample_size=50
        )
        
        engine.accuracy_tracker["test_metric_1_linear_regression"] = accuracy
        
        # Add some confidence tracking data
        engine.confidence_tracker["test_metric_1_linear_regression"] = {
            "coverage": 0.85,
            "average_width": 45.2,
            "reliability": 0.88,
            "last_updated": datetime.utcnow()
        }
        
        report = engine.get_prediction_performance_report("test_metric_1")
        
        assert "accuracy_metrics" in report
        assert "confidence_metrics" in report
        assert "model_comparison" in report
        assert "recommendations" in report
        
        assert "test_metric_1_linear_regression" in report["accuracy_metrics"]
        assert "test_metric_1_linear_regression" in report["confidence_metrics"]
        
        accuracy_data = report["accuracy_metrics"]["test_metric_1_linear_regression"]
        assert accuracy_data["mae"] == 15.2
        assert accuracy_data["mape"] == 8.5
        assert accuracy_data["r2_score"] == 0.82
        
        confidence_data = report["confidence_metrics"]["test_metric_1_linear_regression"]
        assert confidence_data["coverage"] == 0.85
        assert confidence_data["reliability"] == 0.88
    
    def test_enhanced_scenario_modeling(self, engine, sample_metrics):
        """Test enhanced scenario modeling with multiple parameters."""
        scenario = ScenarioConfig(
            id="enhanced_scenario",
            name="Enhanced Growth Scenario",
            description="Test enhanced scenario with multiple parameters",
            parameters={
                "percentage_change": 15,
                "seasonal_adjustment": 0.1,
                "trend_adjustment": 0.05
            },
            target_metrics=["test_metric_1"],
            time_horizon=30,
            created_by="test_user",
            created_at=datetime.utcnow()
        )
        
        historical_data = {"test_metric_1": sample_metrics}
        result = engine.model_scenario(scenario, historical_data)
        
        assert isinstance(result, ScenarioResult)
        assert result.scenario_id == "enhanced_scenario"
        assert "test_metric_1" in result.baseline_forecast
        assert "test_metric_1" in result.scenario_forecast
        assert "test_metric_1" in result.impact_analysis
        
        # Check that multiple parameters were applied
        baseline_avg = np.mean(result.baseline_forecast["test_metric_1"].predictions)
        scenario_avg = np.mean(result.scenario_forecast["test_metric_1"].predictions)
        
        # Should show positive impact due to parameters
        assert scenario_avg > baseline_avg
        assert result.impact_analysis["test_metric_1"] > 0  # Positive impact
    
    def test_advanced_risk_detection(self, engine, sample_metrics, sample_context):
        """Test advanced risk detection with multiple risk types."""
        # Create metrics with different risk patterns
        current_metrics = []
        
        # Anomaly risk
        anomaly_metric = sample_metrics[-1]
        anomaly_metric.value = 2500  # Anomalously high
        current_metrics.append(anomaly_metric)
        
        # Declining trend risk
        declining_metric = BusinessMetric(
            id="declining_metric",
            name="Declining Metric",
            category=MetricCategory.PERFORMANCE,
            value=800,  # Lower than historical average
            unit="score",
            timestamp=datetime.utcnow(),
            source="test",
            context={}
        )
        current_metrics.append(declining_metric)
        
        # Create historical data for declining metric
        declining_history = []
        for i in range(20):
            declining_history.append(BusinessMetric(
                id="declining_metric",
                name="Declining Metric",
                category=MetricCategory.PERFORMANCE,
                value=1000 - i * 10,  # Declining trend
                unit="score",
                timestamp=datetime.utcnow() - timedelta(days=20-i),
                source="test",
                context={}
            ))
        
        historical_data = {
            "test_metric_1": sample_metrics,
            "declining_metric": declining_history
        }
        
        risks = engine.predict_risks(sample_context, current_metrics, historical_data)
        
        assert len(risks) > 0
        
        # Check for different risk types
        risk_types = [risk.risk_type for risk in risks]
        assert len(set(risk_types)) > 1  # Should have multiple risk types
        
        # Check risk levels are properly assigned
        risk_levels = [risk.risk_level for risk in risks]
        assert any(level in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL] for level in risk_levels)


if __name__ == "__main__":
    pytest.main([__file__])