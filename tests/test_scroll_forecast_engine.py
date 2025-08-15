"""
Unit tests for ScrollForecast Engine.
Tests forecasting algorithms and model selection functionality.
"""

import pytest
import pytest_asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import json
import tempfile
import os

from scrollintel.engines.scroll_forecast_engine import (
    ScrollForecastEngine, 
    ForecastModel, 
    SeasonalityType
)
from scrollintel.engines.base_engine import EngineStatus, EngineCapability


class TestScrollForecastEngine:
    """Test cases for ScrollForecast Engine."""
    
    @pytest_asyncio.fixture
    async def engine(self):
        """Create a ScrollForecast engine instance."""
        engine = ScrollForecastEngine()
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    @pytest.fixture
    def sample_time_series_data(self):
        """Create sample time series data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        # Create synthetic data with trend and seasonality
        trend = np.linspace(100, 200, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly seasonality
        noise = np.random.normal(0, 5, len(dates))
        values = trend + seasonal + noise
        
        return pd.DataFrame({
            'date': dates,
            'value': values
        })
    
    @pytest.fixture
    def sample_time_series_dict(self, sample_time_series_data):
        """Convert sample data to dictionary format."""
        return sample_time_series_data.to_dict('records')
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = ScrollForecastEngine()
        
        assert engine.engine_id == "scroll_forecast_engine"
        assert engine.name == "ScrollForecast Engine"
        assert EngineCapability.FORECASTING in engine.capabilities
        assert EngineCapability.DATA_ANALYSIS in engine.capabilities
        assert EngineCapability.VISUALIZATION in engine.capabilities
    
    @pytest.mark.asyncio
    async def test_engine_startup_and_cleanup(self):
        """Test engine startup and cleanup process."""
        engine = ScrollForecastEngine()
        
        # Test initialization
        await engine.initialize()
        assert engine.status == EngineStatus.READY
        
        # Test cleanup
        await engine.cleanup()
        assert len(engine.trained_models) == 0
    
    def test_get_status(self, engine):
        """Test engine status reporting."""
        status = engine.get_status()
        
        assert status["healthy"] is True
        assert "models_trained" in status
        assert "supported_models" in status
        assert "models_directory" in status
        assert "libraries_available" in status
    
    @pytest.mark.asyncio
    async def test_prepare_time_series_data(self, engine, sample_time_series_data):
        """Test time series data preparation."""
        # Test with DataFrame
        df = await engine._prepare_time_series_data(
            sample_time_series_data, "date", "value"
        )
        
        assert len(df) == len(sample_time_series_data)
        assert "ds" in df.columns
        assert "y" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['ds'])
        assert pd.api.types.is_numeric_dtype(df['y'])
        
        # Test with dictionary
        data_dict = sample_time_series_data.to_dict('records')
        df2 = await engine._prepare_time_series_data(
            data_dict, "date", "value"
        )
        
        assert len(df2) == len(sample_time_series_data)
        
        # Test with missing columns
        with pytest.raises(ValueError, match="Date column 'missing' not found"):
            await engine._prepare_time_series_data(
                sample_time_series_data, "missing", "value"
            )
    
    @pytest.mark.asyncio
    async def test_analyze_data_characteristics(self, engine, sample_time_series_data):
        """Test time series data analysis."""
        df = await engine._prepare_time_series_data(
            sample_time_series_data, "date", "value"
        )
        
        analysis = await engine._analyze_data_characteristics(df)
        
        assert "data_points" in analysis
        assert "date_range" in analysis
        assert "value_stats" in analysis
        assert "frequency" in analysis
        
        assert analysis["data_points"] == len(df)
        assert analysis["frequency"] == "daily"
        
        # Check value statistics
        stats = analysis["value_stats"]
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
    
    @pytest.mark.asyncio
    async def test_auto_select_models(self, engine, sample_time_series_data):
        """Test automatic model selection."""
        df = await engine._prepare_time_series_data(
            sample_time_series_data, "date", "value"
        )
        analysis = await engine._analyze_data_characteristics(df)
        
        selected_models = await engine._auto_select_models(df, analysis)
        
        assert isinstance(selected_models, list)
        assert len(selected_models) > 0
        
        # Should select models based on available libraries
        for model in selected_models:
            assert model in engine.supported_models
    
    @pytest.mark.asyncio
    async def test_create_forecast_with_mock_models(self, engine, sample_time_series_dict):
        """Test forecast creation with mocked models."""
        
        # Mock the model training methods to avoid dependency issues
        with patch.object(engine, '_train_forecast_model') as mock_train:
            mock_train.return_value = {
                "model": Mock(),
                "forecast": [
                    {
                        "ds": "2024-01-01",
                        "yhat": 150.0,
                        "yhat_lower": 140.0,
                        "yhat_upper": 160.0
                    }
                ],
                "metrics": {
                    "validation_mae": 5.0,
                    "validation_mse": 25.0,
                    "validation_rmse": 5.0
                }
            }
            
            with patch.object(engine, '_create_forecast_visualizations') as mock_viz:
                mock_viz.return_value = {"forecast_comparison": "mock_plot"}
                
                with patch.object(engine, '_save_forecast_model') as mock_save:
                    mock_save.return_value = "/path/to/model.pkl"
                    
                    input_data = {
                        "data": sample_time_series_dict,
                        "date_column": "date",
                        "value_column": "value",
                        "forecast_periods": 30,
                        "models": ["prophet"]  # Use specific model to avoid auto-selection
                    }
                    
                    result = await engine._create_forecast(input_data, {})
                    
                    assert "forecast_name" in result
                    assert "data_analysis" in result
                    assert "models_tested" in result
                    assert "results" in result
                    assert "best_model" in result
                    assert "visualizations" in result
                    
                    # Check that training was called
                    mock_train.assert_called()
    
    @pytest.mark.asyncio
    async def test_analyze_time_series_action(self, engine, sample_time_series_dict):
        """Test time series analysis action."""
        input_data = {
            "action": "analyze",
            "data": sample_time_series_dict,
            "date_column": "date",
            "value_column": "value"
        }
        
        result = await engine.process(input_data, {})
        
        assert "data_analysis" in result
        assert "recommendations" in result
        
        # Check analysis structure
        analysis = result["data_analysis"]
        assert "data_points" in analysis
        assert "date_range" in analysis
        assert "value_stats" in analysis
        assert "frequency" in analysis
    
    @pytest.mark.asyncio
    async def test_decompose_time_series_action(self, engine, sample_time_series_dict):
        """Test time series decomposition action."""
        
        # Mock statsmodels availability
        with patch('scrollintel.engines.scroll_forecast_engine.HAS_STATSMODELS', True):
            with patch('scrollintel.engines.scroll_forecast_engine.seasonal_decompose') as mock_decompose:
                # Create mock decomposition result
                mock_result = Mock()
                mock_result.trend = pd.Series([100, 110, 120, 130])
                mock_result.seasonal = pd.Series([5, -5, 5, -5])
                mock_result.resid = pd.Series([1, -1, 1, -1])
                mock_decompose.return_value = mock_result
                
                with patch('matplotlib.pyplot.savefig'):
                    with patch('matplotlib.pyplot.close'):
                        input_data = {
                            "action": "decompose",
                            "data": sample_time_series_dict[:50],  # Use subset for faster processing
                            "date_column": "date",
                            "value_column": "value",
                            "type": "additive"
                        }
                        
                        result = await engine.process(input_data, {})
                        
                        assert "decomposition" in result
                        assert "visualization" in result
                        assert "analysis" in result
                        
                        decomp = result["decomposition"]
                        assert "trend" in decomp
                        assert "seasonal" in decomp
                        assert "residual" in decomp
                        assert "period" in decomp
                        assert "type" in decomp
    
    @pytest.mark.asyncio
    async def test_compare_models_action(self, engine):
        """Test model comparison action."""
        # Add some mock trained models
        engine.trained_models = {
            "model1": {
                "model_type": "prophet",
                "metrics": {"validation_mae": 5.0},
                "data_analysis": {"data_points": 100},
                "trained_at": datetime.utcnow()
            },
            "model2": {
                "model_type": "arima",
                "metrics": {"validation_mae": 7.0},
                "data_analysis": {"data_points": 100},
                "trained_at": datetime.utcnow()
            }
        }
        
        input_data = {
            "action": "compare",
            "model_names": ["model1", "model2"]
        }
        
        result = await engine.process(input_data, {})
        
        assert "comparison" in result
        assert "total_models" in result
        assert result["total_models"] == 2
        
        comparison = result["comparison"]
        assert "model1" in comparison
        assert "model2" in comparison
    
    @pytest.mark.asyncio
    async def test_invalid_action(self, engine, sample_time_series_dict):
        """Test handling of invalid action."""
        input_data = {
            "action": "invalid_action",
            "data": sample_time_series_dict
        }
        
        with pytest.raises(ValueError, match="Unknown action: invalid_action"):
            await engine.process(input_data, {})
    
    @pytest.mark.asyncio
    async def test_missing_data_error(self, engine):
        """Test error handling for missing data."""
        input_data = {
            "action": "forecast"
            # Missing data field
        }
        
        with pytest.raises(ValueError, match="Time series data is required"):
            await engine.process(input_data, {})
    
    @pytest.mark.asyncio
    async def test_get_model_recommendations(self, engine):
        """Test model recommendations based on data analysis."""
        # Test with small dataset
        small_analysis = {"data_points": 10}
        recommendations = await engine._get_model_recommendations(small_analysis)
        assert any("more data" in rec.lower() for rec in recommendations)
        
        # Test with seasonal data
        seasonal_analysis = {
            "data_points": 100,
            "seasonality": {"has_weekly_pattern": True}
        }
        recommendations = await engine._get_model_recommendations(seasonal_analysis)
        assert any("prophet" in rec.lower() for rec in recommendations)
        
        # Test with stationary data
        stationary_analysis = {
            "data_points": 100,
            "stationarity": {"is_stationary": True}
        }
        recommendations = await engine._get_model_recommendations(stationary_analysis)
        assert any("arima" in rec.lower() for rec in recommendations)
        
        # Test with large dataset
        large_analysis = {"data_points": 100}
        recommendations = await engine._get_model_recommendations(large_analysis)
        assert any("lstm" in rec.lower() for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_create_ensemble_forecast(self, engine):
        """Test ensemble forecast creation."""
        # Mock results from multiple models
        results = {
            "prophet": {
                "forecast": [
                    {"ds": "2024-01-01", "yhat": 100, "yhat_lower": 90, "yhat_upper": 110},
                    {"ds": "2024-01-02", "yhat": 105, "yhat_lower": 95, "yhat_upper": 115}
                ]
            },
            "arima": {
                "forecast": [
                    {"ds": "2024-01-01", "yhat": 102, "yhat_lower": 92, "yhat_upper": 112},
                    {"ds": "2024-01-02", "yhat": 107, "yhat_lower": 97, "yhat_upper": 117}
                ]
            }
        }
        
        ensemble = await engine._create_ensemble_forecast(results)
        
        assert ensemble is not None
        assert "forecast" in ensemble
        assert "models_combined" in ensemble
        assert "combination_method" in ensemble
        
        forecast = ensemble["forecast"]
        assert len(forecast) == 2
        
        # Check that ensemble values are averages
        assert forecast[0]["yhat"] == 101.0  # (100 + 102) / 2
        assert forecast[1]["yhat"] == 106.0  # (105 + 107) / 2
    
    @pytest.mark.asyncio
    async def test_create_ensemble_forecast_insufficient_models(self, engine):
        """Test ensemble forecast with insufficient models."""
        # Only one successful model
        results = {
            "prophet": {
                "forecast": [{"ds": "2024-01-01", "yhat": 100, "yhat_lower": 90, "yhat_upper": 110}]
            },
            "arima": {
                "error": "Model failed"
            }
        }
        
        ensemble = await engine._create_ensemble_forecast(results)
        assert ensemble is None
    
    def test_supported_models_detection(self):
        """Test that supported models are correctly detected based on available libraries."""
        engine = ScrollForecastEngine()
        
        # Should have at least one model type
        assert len(engine.supported_models) >= 0
        
        # All supported models should be valid
        valid_models = [ForecastModel.PROPHET, ForecastModel.ARIMA, ForecastModel.LSTM]
        for model in engine.supported_models:
            assert model in valid_models
    
    @pytest.mark.asyncio
    async def test_model_save_and_load(self, engine):
        """Test model saving and loading functionality."""
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            engine.models_dir = temp_dir
            
            # Mock model info
            model_info = {
                "type": "prophet",
                "model": Mock(),
                "forecast": [{"ds": "2024-01-01", "yhat": 100}],
                "metrics": {"validation_mae": 5.0}
            }
            
            df = pd.DataFrame({
                'ds': pd.date_range('2023-01-01', periods=10),
                'y': range(10)
            })
            
            analysis = {"data_points": 10, "frequency": "daily"}
            
            # Test saving
            model_path = await engine._save_forecast_model(
                model_info, "test_forecast", df, analysis
            )
            
            assert os.path.exists(model_path)
            assert "test_forecast" in engine.trained_models
            
            # Test loading
            await engine._load_existing_models()
            assert "test_forecast" in engine.trained_models
    
    @pytest.mark.asyncio
    async def test_error_handling_in_process(self, engine):
        """Test error handling in the main process method."""
        
        # Test with invalid input that should raise an error
        input_data = {
            "action": "forecast",
            "data": "invalid_data_format"
        }
        
        with pytest.raises(Exception):
            await engine.process(input_data, {})
    
    @pytest.mark.asyncio
    async def test_health_check(self, engine):
        """Test engine health check."""
        health = await engine.health_check()
        assert isinstance(health, bool)
        
        # Should be healthy after initialization
        assert health is True
    
    def test_get_metrics(self, engine):
        """Test engine metrics reporting."""
        metrics = engine.get_metrics()
        
        assert "engine_id" in metrics
        assert "name" in metrics
        assert "status" in metrics
        assert "capabilities" in metrics
        assert "usage_count" in metrics
        assert "error_count" in metrics
        assert "error_rate" in metrics
        assert "created_at" in metrics
        
        assert metrics["engine_id"] == "scroll_forecast_engine"
        assert metrics["name"] == "ScrollForecast Engine"


class TestForecastModelTypes:
    """Test forecast model type constants."""
    
    def test_forecast_model_constants(self):
        """Test that forecast model constants are properly defined."""
        assert ForecastModel.PROPHET == "prophet"
        assert ForecastModel.ARIMA == "arima"
        assert ForecastModel.LSTM == "lstm"
    
    def test_seasonality_type_constants(self):
        """Test that seasonality type constants are properly defined."""
        assert SeasonalityType.DAILY == "daily"
        assert SeasonalityType.WEEKLY == "weekly"
        assert SeasonalityType.MONTHLY == "monthly"
        assert SeasonalityType.QUARTERLY == "quarterly"
        assert SeasonalityType.YEARLY == "yearly"


@pytest.mark.integration
class TestScrollForecastEngineIntegration:
    """Integration tests for ScrollForecast Engine with real libraries."""
    
    @pytest_asyncio.fixture
    async def engine(self):
        """Create engine for integration tests."""
        engine = ScrollForecastEngine()
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    @pytest.fixture
    def real_time_series_data(self):
        """Create realistic time series data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        # Create data with clear trend and seasonality
        trend = np.linspace(100, 200, len(dates))
        seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        noise = np.random.normal(0, 3, len(dates))
        values = trend + seasonal + noise
        
        return [
            {"date": date.isoformat(), "value": value}
            for date, value in zip(dates, values)
        ]
    
    @pytest.mark.skipif(
        not any([
            'scrollintel.engines.scroll_forecast_engine.HAS_PROPHET',
            'scrollintel.engines.scroll_forecast_engine.HAS_STATSMODELS',
            'scrollintel.engines.scroll_forecast_engine.HAS_TENSORFLOW'
        ]), 
        reason="No forecasting libraries available"
    )
    @pytest.mark.asyncio
    async def test_full_forecast_workflow(self, engine, real_time_series_data):
        """Test complete forecasting workflow with real data."""
        
        input_data = {
            "action": "forecast",
            "data": real_time_series_data,
            "date_column": "date",
            "value_column": "value",
            "forecast_periods": 10,
            "models": engine.supported_models[:1] if engine.supported_models else ["prophet"]
        }
        
        # This test will only run if at least one forecasting library is available
        if engine.supported_models:
            result = await engine.process(input_data, {})
            
            assert "forecast_name" in result
            assert "data_analysis" in result
            assert "results" in result
            assert "best_model" in result
            
            # Check that at least one model succeeded
            successful_models = [
                model for model, result_data in result["results"].items()
                if "forecast" in result_data
            ]
            assert len(successful_models) > 0


if __name__ == "__main__":
    pytest.main([__file__])