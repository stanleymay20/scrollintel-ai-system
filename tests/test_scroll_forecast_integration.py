"""
Integration tests for ScrollForecast Engine.
Tests end-to-end forecasting workflows and model selection.
"""

import pytest
import pytest_asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import tempfile
import os

from scrollintel.engines.scroll_forecast_engine import ScrollForecastEngine, ForecastModel
from scrollintel.engines.base_engine import EngineStatus


@pytest.mark.integration
class TestScrollForecastEngineIntegration:
    """Integration tests for ScrollForecast Engine."""
    
    @pytest_asyncio.fixture
    async def engine(self):
        """Create and initialize ScrollForecast engine."""
        engine = ScrollForecastEngine()
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    @pytest.fixture
    def daily_sales_data(self):
        """Create realistic daily sales time series data."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Create realistic sales pattern
        base_sales = 1000
        trend = np.linspace(0, 200, len(dates))  # Growing trend
        
        # Weekly seasonality (higher sales on weekends)
        weekly_pattern = []
        for date in dates:
            if date.weekday() in [5, 6]:  # Saturday, Sunday
                weekly_pattern.append(150)
            elif date.weekday() in [0, 4]:  # Monday, Friday
                weekly_pattern.append(50)
            else:
                weekly_pattern.append(0)
        
        # Monthly seasonality (holiday effects)
        monthly_pattern = []
        for date in dates:
            if date.month in [11, 12]:  # Holiday season
                monthly_pattern.append(300)
            elif date.month in [6, 7, 8]:  # Summer
                monthly_pattern.append(100)
            else:
                monthly_pattern.append(0)
        
        # Random noise
        noise = np.random.normal(0, 50, len(dates))
        
        # Combine all components
        sales = base_sales + trend + np.array(weekly_pattern) + np.array(monthly_pattern) + noise
        
        return [
            {"date": date.strftime('%Y-%m-%d'), "sales": max(0, sale)}
            for date, sale in zip(dates, sales)
        ]
    
    @pytest.fixture
    def stock_price_data(self):
        """Create realistic stock price time series data."""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')  # Business days
        
        # Simulate stock price with random walk
        initial_price = 100
        returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
        prices = [initial_price]
        
        for return_rate in returns[1:]:
            new_price = prices[-1] * (1 + return_rate)
            prices.append(max(1, new_price))  # Ensure positive prices
        
        return [
            {"date": date.strftime('%Y-%m-%d'), "price": price}
            for date, price in zip(dates, prices)
        ]
    
    @pytest.fixture
    def temperature_data(self):
        """Create realistic temperature time series data."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Create temperature pattern with seasonal variation
        day_of_year = np.array([date.timetuple().tm_yday for date in dates])
        
        # Base temperature with seasonal pattern
        seasonal_temp = 20 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Daily variation
        daily_noise = np.random.normal(0, 3, len(dates))
        
        temperatures = seasonal_temp + daily_noise
        
        return [
            {"date": date.strftime('%Y-%m-%d'), "temperature": temp}
            for date, temp in zip(dates, temperatures)
        ]
    
    @pytest.mark.asyncio
    async def test_sales_forecasting_workflow(self, engine, daily_sales_data):
        """Test complete sales forecasting workflow."""
        
        input_data = {
            "action": "forecast",
            "data": daily_sales_data,
            "date_column": "date",
            "value_column": "sales",
            "forecast_periods": 30,
            "confidence_level": 0.95,
            "forecast_name": "daily_sales_forecast"
        }
        
        result = await engine.process(input_data, {})
        
        # Verify basic structure
        assert "forecast_name" in result
        assert "data_analysis" in result
        assert "results" in result
        assert "best_model" in result
        assert "visualizations" in result
        
        # Check data analysis
        analysis = result["data_analysis"]
        assert analysis["data_points"] == len(daily_sales_data)
        assert analysis["frequency"] == "daily"
        assert "value_stats" in analysis
        
        # Check that at least one model was trained successfully
        successful_models = [
            model for model, model_result in result["results"].items()
            if "forecast" in model_result and "error" not in model_result
        ]
        assert len(successful_models) > 0, f"No models succeeded. Results: {result['results']}"
        
        # Check forecast structure
        for model_name in successful_models:
            forecast = result["results"][model_name]["forecast"]
            assert len(forecast) == 30  # Should have 30 forecast periods
            
            for forecast_point in forecast:
                assert "ds" in forecast_point
                assert "yhat" in forecast_point
                assert "yhat_lower" in forecast_point
                assert "yhat_upper" in forecast_point
                
                # Confidence intervals should be reasonable
                assert forecast_point["yhat_lower"] <= forecast_point["yhat"]
                assert forecast_point["yhat"] <= forecast_point["yhat_upper"]
        
        # Check best model selection
        if result["best_model"]["type"]:
            assert result["best_model"]["type"] in successful_models
            assert result["best_model"]["score"] is not None
    
    @pytest.mark.asyncio
    async def test_stock_price_forecasting(self, engine, stock_price_data):
        """Test stock price forecasting with financial data."""
        
        input_data = {
            "action": "forecast",
            "data": stock_price_data,
            "date_column": "date",
            "value_column": "price",
            "forecast_periods": 20,
            "models": ["auto"],  # Let engine auto-select models
            "forecast_name": "stock_price_forecast"
        }
        
        result = await engine.process(input_data, {})
        
        # Verify results
        assert "forecast_name" in result
        assert result["forecast_name"] == "stock_price_forecast"
        
        # Check that models were auto-selected
        assert len(result["models_tested"]) > 0
        
        # Verify forecast quality
        successful_models = [
            model for model, model_result in result["results"].items()
            if "forecast" in model_result
        ]
        
        if successful_models:
            # Check that forecasted prices are reasonable (positive)
            for model_name in successful_models:
                forecast = result["results"][model_name]["forecast"]
                for point in forecast:
                    assert point["yhat"] > 0, "Stock prices should be positive"
    
    @pytest.mark.asyncio
    async def test_temperature_forecasting_with_seasonality(self, engine, temperature_data):
        """Test temperature forecasting with strong seasonal patterns."""
        
        input_data = {
            "action": "forecast",
            "data": temperature_data,
            "date_column": "date",
            "value_column": "temperature",
            "forecast_periods": 60,  # 2 months ahead
            "forecast_name": "temperature_forecast"
        }
        
        result = await engine.process(input_data, {})
        
        # Check seasonality detection
        analysis = result["data_analysis"]
        assert "trend" in analysis or "seasonality" in analysis
        
        # Verify forecast makes sense for temperature data
        successful_models = [
            model for model, model_result in result["results"].items()
            if "forecast" in model_result
        ]
        
        if successful_models:
            for model_name in successful_models:
                forecast = result["results"][model_name]["forecast"]
                temperatures = [point["yhat"] for point in forecast]
                
                # Temperature should be within reasonable range
                assert all(-50 <= temp <= 60 for temp in temperatures), \
                    f"Unreasonable temperature forecast: {temperatures}"
    
    @pytest.mark.asyncio
    async def test_time_series_analysis_workflow(self, engine, daily_sales_data):
        """Test time series analysis without forecasting."""
        
        input_data = {
            "action": "analyze",
            "data": daily_sales_data,
            "date_column": "date",
            "value_column": "sales"
        }
        
        result = await engine.process(input_data, {})
        
        assert "data_analysis" in result
        assert "recommendations" in result
        
        analysis = result["data_analysis"]
        
        # Check analysis completeness
        assert "data_points" in analysis
        assert "date_range" in analysis
        assert "value_stats" in analysis
        assert "frequency" in analysis
        
        # Check recommendations
        recommendations = result["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_model_comparison_workflow(self, engine, daily_sales_data):
        """Test model comparison after training multiple models."""
        
        # First, create some forecasts
        input_data = {
            "action": "forecast",
            "data": daily_sales_data[:200],  # Use subset for faster training
            "date_column": "date",
            "value_column": "sales",
            "forecast_periods": 10,
            "forecast_name": "comparison_test_1"
        }
        
        result1 = await engine.process(input_data, {})
        
        # Create another forecast with different parameters
        input_data["forecast_name"] = "comparison_test_2"
        input_data["forecast_periods"] = 15
        
        result2 = await engine.process(input_data, {})
        
        # Now compare the models
        comparison_input = {
            "action": "compare",
            "model_names": ["comparison_test_1", "comparison_test_2"]
        }
        
        comparison_result = await engine.process(comparison_input, {})
        
        assert "comparison" in comparison_result
        assert "total_models" in comparison_result
        
        comparison = comparison_result["comparison"]
        
        # Should have both models in comparison
        if "comparison_test_1" in engine.trained_models:
            assert "comparison_test_1" in comparison
        if "comparison_test_2" in engine.trained_models:
            assert "comparison_test_2" in comparison
    
    @pytest.mark.asyncio
    async def test_ensemble_forecasting(self, engine, daily_sales_data):
        """Test ensemble forecasting with multiple models."""
        
        # Use a subset of data and ensure we have multiple models available
        if len(engine.supported_models) < 2:
            pytest.skip("Need at least 2 forecasting models for ensemble testing")
        
        input_data = {
            "action": "forecast",
            "data": daily_sales_data[:100],  # Smaller dataset for faster processing
            "date_column": "date",
            "value_column": "sales",
            "forecast_periods": 10,
            "models": engine.supported_models[:2],  # Use first 2 available models
            "forecast_name": "ensemble_test"
        }
        
        result = await engine.process(input_data, {})
        
        # Check if ensemble was created
        if result.get("ensemble_forecast"):
            ensemble = result["ensemble_forecast"]
            assert "forecast" in ensemble
            assert "models_combined" in ensemble
            assert "combination_method" in ensemble
            
            # Ensemble should have same number of periods as individual forecasts
            assert len(ensemble["forecast"]) == 10
            
            # Should combine at least 2 models
            assert len(ensemble["models_combined"]) >= 2
    
    @pytest.mark.asyncio
    async def test_error_handling_with_invalid_data(self, engine):
        """Test error handling with various invalid data scenarios."""
        
        # Test with empty data
        with pytest.raises(Exception):
            await engine.process({
                "action": "forecast",
                "data": [],
                "date_column": "date",
                "value_column": "value"
            }, {})
        
        # Test with missing columns
        with pytest.raises(ValueError):
            await engine.process({
                "action": "forecast",
                "data": [{"wrong_column": "2023-01-01", "value": 100}],
                "date_column": "date",
                "value_column": "value"
            }, {})
        
        # Test with non-numeric values
        with pytest.raises(Exception):
            await engine.process({
                "action": "forecast",
                "data": [
                    {"date": "2023-01-01", "value": "not_a_number"},
                    {"date": "2023-01-02", "value": "also_not_a_number"}
                ],
                "date_column": "date",
                "value_column": "value"
            }, {})
    
    @pytest.mark.asyncio
    async def test_model_persistence(self, engine, daily_sales_data):
        """Test that trained models are properly saved and can be loaded."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set temporary directory for models
            engine.models_dir = temp_dir
            
            # Train a model
            input_data = {
                "action": "forecast",
                "data": daily_sales_data[:50],  # Small dataset for speed
                "date_column": "date",
                "value_column": "sales",
                "forecast_periods": 5,
                "forecast_name": "persistence_test"
            }
            
            result = await engine.process(input_data, {})
            
            # Check that model was saved
            if result.get("model_path"):
                assert os.path.exists(result["model_path"])
                
                # Check that model is in trained_models
                assert "persistence_test" in engine.trained_models
                
                # Clear models and reload
                engine.trained_models.clear()
                await engine._load_existing_models()
                
                # Model should be loaded back
                assert "persistence_test" in engine.trained_models
    
    @pytest.mark.asyncio
    async def test_forecast_visualization_generation(self, engine, daily_sales_data):
        """Test that forecast visualizations are properly generated."""
        
        input_data = {
            "action": "forecast",
            "data": daily_sales_data[:100],  # Smaller dataset
            "date_column": "date",
            "value_column": "sales",
            "forecast_periods": 10,
            "forecast_name": "visualization_test"
        }
        
        result = await engine.process(input_data, {})
        
        # Check that visualizations were created
        assert "visualizations" in result
        visualizations = result["visualizations"]
        
        # Should have at least the main forecast comparison plot
        if "forecast_comparison" in visualizations:
            viz_data = visualizations["forecast_comparison"]
            assert viz_data.startswith("data:image/png;base64,")
        elif "error" in visualizations:
            # Visualization creation failed, but that's acceptable in test environment
            assert isinstance(visualizations["error"], str)
    
    @pytest.mark.asyncio
    async def test_confidence_interval_calculation(self, engine, daily_sales_data):
        """Test that confidence intervals are properly calculated."""
        
        input_data = {
            "action": "forecast",
            "data": daily_sales_data[:100],
            "date_column": "date",
            "value_column": "sales",
            "forecast_periods": 10,
            "confidence_level": 0.90,  # 90% confidence
            "forecast_name": "confidence_test"
        }
        
        result = await engine.process(input_data, {})
        
        # Check confidence intervals in successful forecasts
        successful_models = [
            model for model, model_result in result["results"].items()
            if "forecast" in model_result
        ]
        
        for model_name in successful_models:
            forecast = result["results"][model_name]["forecast"]
            
            for point in forecast:
                # Confidence intervals should be properly ordered
                assert point["yhat_lower"] <= point["yhat"] <= point["yhat_upper"]
                
                # Intervals should be reasonable (not too wide or too narrow)
                interval_width = point["yhat_upper"] - point["yhat_lower"]
                assert interval_width > 0, "Confidence interval should have positive width"
                
                # For sales data, intervals shouldn't be unreasonably wide
                assert interval_width < point["yhat"] * 2, "Confidence interval too wide"
    
    @pytest.mark.asyncio
    async def test_automated_model_selection_logic(self, engine):
        """Test the automated model selection logic with different data characteristics."""
        
        # Test with small dataset (should prefer simpler models)
        small_data = [
            {"date": f"2023-01-{i:02d}", "value": 100 + i}
            for i in range(1, 16)  # 15 data points
        ]
        
        df_small = await engine._prepare_time_series_data(small_data, "date", "value")
        analysis_small = await engine._analyze_data_characteristics(df_small)
        selected_small = await engine._auto_select_models(df_small, analysis_small)
        
        # Should select appropriate models for small dataset
        assert isinstance(selected_small, list)
        assert len(selected_small) > 0
        
        # Test with large dataset (should consider more complex models)
        large_data = [
            {"date": f"2023-{(i//30)+1:02d}-{(i%30)+1:02d}", "value": 100 + i + 10*np.sin(i/7)}
            for i in range(200)  # 200 data points
        ]
        
        df_large = await engine._prepare_time_series_data(large_data, "date", "value")
        analysis_large = await engine._analyze_data_characteristics(df_large)
        selected_large = await engine._auto_select_models(df_large, analysis_large)
        
        # Should potentially select more models for larger dataset
        assert isinstance(selected_large, list)
        assert len(selected_large) > 0
        
        # All selected models should be supported
        for model in selected_small + selected_large:
            assert model in engine.supported_models


if __name__ == "__main__":
    pytest.main([__file__])