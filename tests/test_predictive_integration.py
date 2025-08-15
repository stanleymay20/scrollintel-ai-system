"""
Integration tests for predictive analytics system.
"""
import pytest
import asyncio
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scrollintel.api.routes.predictive_routes import router
from scrollintel.models.predictive_models import ForecastModel, MetricCategory
from fastapi import FastAPI

# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestPredictiveIntegration:
    """Integration tests for predictive analytics API."""
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical data for testing."""
        base_time = datetime.utcnow() - timedelta(days=30)
        data = []
        
        for i in range(30):
            data.append({
                "id": "revenue_metric",
                "name": "Monthly Revenue",
                "category": "financial",
                "value": 10000 + i * 100 + (i % 7) * 50,  # Trending with weekly pattern
                "unit": "USD",
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "source": "sales_system",
                "context": {"department": "sales", "region": "north_america"}
            })
        
        return data
    
    @pytest.fixture
    def sample_scenario_data(self, sample_historical_data):
        """Create sample scenario data for testing."""
        return {
            "name": "Q4 Growth Scenario",
            "description": "Modeling 25% growth in Q4",
            "parameters": {
                "percentage_change": 25,
                "seasonal_adjustment": 0.15,
                "trend_adjustment": 0.1
            },
            "target_metrics": ["revenue_metric"],
            "time_horizon": 90,
            "created_by": "test_analyst",
            "historical_data": {
                "revenue_metric": sample_historical_data
            }
        }
    
    @pytest.fixture
    def sample_risk_data(self, sample_historical_data):
        """Create sample risk analysis data."""
        return {
            "context": {
                "industry": "technology",
                "company_size": "medium",
                "market_conditions": {
                    "sentiment": "neutral",
                    "volatility": "medium",
                    "economic_indicators": {"gdp_growth": 0.02, "inflation": 0.03}
                },
                "seasonal_factors": {
                    "q4_boost": 1.2,
                    "summer_slowdown": 0.9
                },
                "external_factors": {
                    "competition_level": "high",
                    "regulatory_changes": "minimal"
                }
            },
            "current_metrics": [
                {
                    "id": "revenue_metric",
                    "name": "Monthly Revenue",
                    "category": "financial",
                    "value": 15000,  # Higher than recent trend
                    "unit": "USD",
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "sales_system",
                    "context": {"department": "sales"}
                }
            ],
            "historical_data": {
                "revenue_metric": sample_historical_data
            }
        }
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/api/predictive/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "predictive_analytics"
        assert "timestamp" in data
        assert "available_models" in data
        assert len(data["available_models"]) > 0
    
    def test_generate_forecast_linear(self, sample_historical_data):
        """Test forecast generation with linear regression."""
        response = client.post(
            "/api/predictive/forecast",
            params={
                "metric_id": "revenue_metric",
                "horizon_days": 14,
                "model_type": "linear_regression"
            },
            json=sample_historical_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["metric_id"] == "revenue_metric"
        assert data["model_type"] == "linear_regression"
        assert data["horizon_days"] == 14
        assert len(data["predictions"]) == 14
        assert len(data["timestamps"]) == 14
        assert "confidence_intervals" in data
        assert len(data["confidence_intervals"]["lower"]) == 14
        assert len(data["confidence_intervals"]["upper"]) == 14
        assert 0 <= data["confidence_intervals"]["level"] <= 1
        
        # Predictions should be positive (revenue)
        assert all(p > 0 for p in data["predictions"])
    
    def test_generate_forecast_ensemble(self, sample_historical_data):
        """Test forecast generation with ensemble model."""
        response = client.post(
            "/api/predictive/forecast",
            params={
                "metric_id": "revenue_metric",
                "horizon_days": 30,
                "model_type": "ensemble"
            },
            json=sample_historical_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_type"] == "ensemble"
        assert data["horizon_days"] == 30
        assert len(data["predictions"]) == 30
        
        # Ensemble should have higher confidence
        assert data["confidence_intervals"]["level"] >= 0.8
    
    def test_forecast_insufficient_data(self):
        """Test forecast with insufficient historical data."""
        minimal_data = [
            {
                "id": "test_metric",
                "name": "Test Metric",
                "category": "operational",
                "value": 100,
                "unit": "count",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "test",
                "context": {}
            }
        ]
        
        response = client.post(
            "/api/predictive/forecast",
            params={"metric_id": "test_metric", "horizon_days": 7},
            json=minimal_data
        )
        
        assert response.status_code == 500
        assert "Insufficient historical data" in response.json()["detail"]
    
    def test_scenario_modeling(self, sample_scenario_data):
        """Test scenario modeling endpoint."""
        response = client.post("/api/predictive/scenario", json=sample_scenario_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "scenario_id" in data
        assert "baseline_forecasts" in data
        assert "scenario_forecasts" in data
        assert "impact_analysis" in data
        assert "recommendations" in data
        assert "confidence_score" in data
        
        # Should have forecasts for target metric
        assert "revenue_metric" in data["baseline_forecasts"]
        assert "revenue_metric" in data["scenario_forecasts"]
        assert "revenue_metric" in data["impact_analysis"]
        
        # Impact should be positive (25% growth scenario)
        assert data["impact_analysis"]["revenue_metric"] > 0
        
        # Should have recommendations
        assert len(data["recommendations"]) > 0
        assert isinstance(data["recommendations"][0], str)
    
    def test_risk_prediction(self, sample_risk_data):
        """Test risk prediction endpoint."""
        response = client.post("/api/predictive/risks", json=sample_risk_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "risks" in data
        assert "summary" in data
        assert "analysis_timestamp" in data
        
        # Summary should have required fields
        summary = data["summary"]
        assert "total_risks" in summary
        assert "risk_levels" in summary
        assert "top_risks" in summary
        assert isinstance(summary["total_risks"], int)
        
        # If risks are detected, validate structure
        for risk in data["risks"]:
            assert "id" in risk
            assert "metric_id" in risk
            assert "risk_type" in risk
            assert "risk_level" in risk
            assert "probability" in risk
            assert "impact_score" in risk
            assert "description" in risk
            assert "mitigation_strategies" in risk
            assert 0 <= risk["probability"] <= 1
            assert risk["impact_score"] >= 0
            assert len(risk["mitigation_strategies"]) > 0
    
    def test_prediction_updates(self, sample_historical_data):
        """Test prediction update endpoint."""
        # First, generate a forecast to have something to update
        client.post(
            "/api/predictive/forecast",
            params={"metric_id": "revenue_metric", "horizon_days": 7},
            json=sample_historical_data
        )
        
        # Now send new data for update
        new_data = [
            {
                "id": "revenue_metric",
                "name": "Monthly Revenue",
                "category": "financial",
                "value": 20000,  # Significant jump
                "unit": "USD",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "sales_system",
                "context": {"department": "sales"}
            }
        ]
        
        response = client.post(
            "/api/predictive/update",
            json={"new_data": new_data}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "updates" in data
        assert "total_updates" in data
        assert "processing_timestamp" in data
        assert isinstance(data["total_updates"], int)
        
        # If updates are generated, validate structure
        for update in data["updates"]:
            assert "metric_id" in update
            assert "change_magnitude" in update
            assert "change_reason" in update
            assert "previous_forecast" in update
            assert "updated_forecast" in update
            assert isinstance(update["change_magnitude"], float)
            assert isinstance(update["change_reason"], str)
    
    def test_model_performance(self):
        """Test model performance endpoint."""
        response = client.get("/api/predictive/models/revenue_metric/performance")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "metric_id" in data
        assert "models" in data
        assert "retrieved_at" in data
        assert data["metric_id"] == "revenue_metric"
        
        # Models may be empty if no performance data exists yet
        for model_type, performance in data["models"].items():
            assert "mae" in performance
            assert "mape" in performance
            assert "rmse" in performance
            assert "r2_score" in performance
            assert "sample_size" in performance
            assert "evaluation_date" in performance
            assert performance["mae"] >= 0
            assert performance["rmse"] >= 0
            assert performance["sample_size"] >= 0
    
    def test_forecast_parameter_validation(self, sample_historical_data):
        """Test forecast parameter validation."""
        # Test invalid horizon
        response = client.post(
            "/api/predictive/forecast",
            params={"metric_id": "test", "horizon_days": 0},
            json=sample_historical_data
        )
        assert response.status_code == 422  # Validation error
        
        # Test invalid model type
        response = client.post(
            "/api/predictive/forecast",
            params={"metric_id": "test", "model_type": "invalid_model"},
            json=sample_historical_data
        )
        assert response.status_code == 422  # Validation error
    
    def test_concurrent_requests(self, sample_historical_data):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        def make_forecast_request():
            return client.post(
                "/api/predictive/forecast",
                params={"metric_id": "concurrent_test", "horizon_days": 7},
                json=sample_historical_data
            )
        
        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_forecast_request) for _ in range(5)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 7
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create large historical dataset
        base_time = datetime.utcnow() - timedelta(days=365)
        large_dataset = []
        
        for i in range(365):  # One year of daily data
            large_dataset.append({
                "id": "large_metric",
                "name": "Daily Metric",
                "category": "operational",
                "value": 1000 + i + (i % 30) * 10,  # Monthly pattern
                "unit": "count",
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "source": "system",
                "context": {}
            })
        
        response = client.post(
            "/api/predictive/forecast",
            params={"metric_id": "large_metric", "horizon_days": 30},
            json=large_dataset
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 30
    
    def test_error_handling(self):
        """Test error handling for various edge cases."""
        # Test with empty data
        response = client.post(
            "/api/predictive/forecast",
            params={"metric_id": "empty", "horizon_days": 7},
            json=[]
        )
        assert response.status_code == 400
        
        # Test with malformed data
        malformed_data = [{"invalid": "data"}]
        response = client.post(
            "/api/predictive/forecast",
            params={"metric_id": "malformed", "horizon_days": 7},
            json=malformed_data
        )
        assert response.status_code == 500  # Should handle gracefully
    
    def test_forecast_consistency(self, sample_historical_data):
        """Test that forecasts are consistent across multiple calls."""
        # Make the same forecast request multiple times
        responses = []
        for _ in range(3):
            response = client.post(
                "/api/predictive/forecast",
                params={"metric_id": "consistency_test", "horizon_days": 7},
                json=sample_historical_data
            )
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Predictions should be similar (allowing for some randomness in ensemble)
        predictions = [response.json()["predictions"] for response in responses]
        
        # Check that predictions are reasonably consistent
        for i in range(len(predictions[0])):
            values = [pred[i] for pred in predictions]
            std_dev = (max(values) - min(values)) / max(values)
            assert std_dev < 0.1  # Less than 10% variation


if __name__ == "__main__":
    pytest.main([__file__])