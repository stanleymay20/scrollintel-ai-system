"""
Tests for usage tracking API routes.
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import FastAPI

from scrollintel.api.routes.usage_tracking_routes import router
from scrollintel.models.usage_tracking_models import GenerationType, ResourceType


# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestUsageTrackingRoutes:
    """Test suite for usage tracking API routes."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/api/v1/usage/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "usage_tracking"
        assert "timestamp" in data
    
    def test_start_tracking(self):
        """Test starting generation tracking."""
        request_data = {
            "user_id": "test_user_123",
            "generation_type": "image",
            "model_used": "stable_diffusion_xl",
            "prompt": "A beautiful landscape",
            "parameters": {
                "resolution": [1024, 1024],
                "steps": 50
            }
        }
        
        response = client.post("/api/v1/usage/tracking/start", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "tracking_started"
        assert "test_user_123" in data["message"]
        
        return data["session_id"]
    
    def test_track_resource(self):
        """Test tracking resource usage."""
        # First start tracking
        session_id = self.test_start_tracking()
        
        request_data = {
            "session_id": session_id,
            "resource_type": "gpu_seconds",
            "amount": 10.5,
            "metadata": {
                "gpu_type": "A100"
            }
        }
        
        response = client.post("/api/v1/usage/tracking/resource", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "resource_tracked"
        assert "10.5" in data["message"]
        assert "gpu_seconds" in data["message"]
    
    def test_end_tracking(self):
        """Test ending generation tracking."""
        # Start tracking and add resource usage
        session_id = self.test_start_tracking()
        
        # Add resource usage
        resource_request = {
            "session_id": session_id,
            "resource_type": "gpu_seconds",
            "amount": 5.0
        }
        client.post("/api/v1/usage/tracking/resource", json=resource_request)
        
        # End tracking
        end_request = {
            "session_id": session_id,
            "success": True,
            "quality_score": 0.85
        }
        
        response = client.post("/api/v1/usage/tracking/end", json=end_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["session_id"] == session_id
        assert data["status"] == "tracking_completed"
        assert data["success"] is True
        assert data["quality_score"] == 0.85
        assert data["duration_seconds"] > 0
        assert data["total_cost"] > 0
    
    def test_end_tracking_nonexistent_session(self):
        """Test ending tracking for non-existent session."""
        request_data = {
            "session_id": "nonexistent_session",
            "success": True
        }
        
        response = client.post("/api/v1/usage/tracking/end", json=request_data)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_get_usage_summary_empty(self):
        """Test getting usage summary for user with no data."""
        response = client.get("/api/v1/usage/summary/empty_user?days=30")
        assert response.status_code == 200
        
        data = response.json()
        assert data["user_id"] == "empty_user"
        assert data["total_generations"] == 0
        assert data["total_cost"] == 0.0
        assert data["average_cost_per_generation"] == 0.0
    
    def test_get_usage_summary_with_data(self):
        """Test getting usage summary with actual data."""
        user_id = "summary_test_user"
        
        # Create a complete generation session
        start_request = {
            "user_id": user_id,
            "generation_type": "image",
            "model_used": "dalle3",
            "prompt": "Test image"
        }
        
        start_response = client.post("/api/v1/usage/tracking/start", json=start_request)
        session_id = start_response.json()["session_id"]
        
        # Add resource usage
        resource_request = {
            "session_id": session_id,
            "resource_type": "gpu_seconds",
            "amount": 8.0
        }
        client.post("/api/v1/usage/tracking/resource", json=resource_request)
        
        # End tracking
        end_request = {
            "session_id": session_id,
            "success": True,
            "quality_score": 0.9
        }
        client.post("/api/v1/usage/tracking/end", json=end_request)
        
        # Get summary
        response = client.get(f"/api/v1/usage/summary/{user_id}?days=30")
        assert response.status_code == 200
        
        data = response.json()
        assert data["user_id"] == user_id
        assert data["total_generations"] == 1
        assert data["successful_generations"] == 1
        assert data["image_generations"] == 1
        assert data["total_gpu_seconds"] == 8.0
        assert data["total_cost"] == 8.0 * 0.05  # GPU cost
        assert data["average_quality_score"] == 0.9
    
    def test_get_budget_alerts(self):
        """Test getting budget alerts."""
        user_id = "budget_alert_user"
        
        # Create expensive session
        start_request = {
            "user_id": user_id,
            "generation_type": "video",
            "model_used": "runway_ml",
            "prompt": "Test video"
        }
        
        start_response = client.post("/api/v1/usage/tracking/start", json=start_request)
        session_id = start_response.json()["session_id"]
        
        # Add expensive resource usage
        resource_request = {
            "session_id": session_id,
            "resource_type": "gpu_seconds",
            "amount": 40.0  # $2.00 cost
        }
        client.post("/api/v1/usage/tracking/resource", json=resource_request)
        
        # End tracking
        end_request = {"session_id": session_id, "success": True}
        client.post("/api/v1/usage/tracking/end", json=end_request)
        
        # Check budget alerts with $3.00 budget
        response = client.get(f"/api/v1/usage/budget-alerts/{user_id}?budget_limit=3.0&period_days=30")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) >= 2  # Should have 50% and 75% alerts
        
        for alert in data:
            assert alert["current_usage"] == 2.0
            assert alert["budget_limit"] == 3.0
            assert alert["usage_percentage"] > 50.0
    
    def test_get_usage_forecast(self):
        """Test getting usage forecast."""
        user_id = "forecast_test_user"
        
        response = client.get(f"/api/v1/usage/forecast/{user_id}?forecast_days=30&historical_days=90")
        assert response.status_code == 200
        
        data = response.json()
        assert data["user_id"] == user_id
        assert data["forecast_period_days"] == 30
        assert "predicted_cost" in data
        assert "confidence_interval" in data
        assert "usage_trend" in data
        assert "generated_at" in data
    
    def test_get_cost_optimization_recommendations(self):
        """Test getting cost optimization recommendations."""
        user_id = "optimization_test_user"
        
        response = client.get(f"/api/v1/usage/recommendations/{user_id}?analysis_days=30")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        # May be empty if no optimization opportunities found
    
    def test_get_cost_estimate_image(self):
        """Test getting cost estimate for image generation."""
        request_data = {
            "generation_type": "image",
            "model_name": "stable_diffusion_xl",
            "parameters": {
                "resolution": [2048, 2048],
                "steps": 100
            }
        }
        
        response = client.post("/api/v1/usage/cost-estimate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["generation_type"] == "image"
        assert data["model_name"] == "stable_diffusion_xl"
        assert data["base_cost"] > 0
        assert data["multiplier"] > 1.0  # Should be higher due to increased resolution and steps
        assert data["estimated_cost"] > data["base_cost"]
        assert data["currency"] == "USD"
        assert "resolution" in data["parameters_considered"]
        assert "steps" in data["parameters_considered"]
    
    def test_get_cost_estimate_video(self):
        """Test getting cost estimate for video generation."""
        request_data = {
            "generation_type": "video",
            "model_name": "runway_ml",
            "parameters": {
                "duration": 10.0,
                "fps": 30,
                "resolution": [1920, 1080]
            }
        }
        
        response = client.post("/api/v1/usage/cost-estimate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["generation_type"] == "video"
        assert data["model_name"] == "runway_ml"
        assert data["estimated_cost"] > 0
        assert data["currency"] == "USD"
    
    def test_get_usage_analytics(self):
        """Test getting comprehensive usage analytics."""
        user_id = "analytics_test_user"
        
        # Create some test data first
        start_request = {
            "user_id": user_id,
            "generation_type": "image",
            "model_used": "stable_diffusion_xl",
            "prompt": "Analytics test"
        }
        
        start_response = client.post("/api/v1/usage/tracking/start", json=start_request)
        session_id = start_response.json()["session_id"]
        
        resource_request = {
            "session_id": session_id,
            "resource_type": "gpu_seconds",
            "amount": 5.0
        }
        client.post("/api/v1/usage/tracking/resource", json=resource_request)
        
        end_request = {
            "session_id": session_id,
            "success": True,
            "quality_score": 0.8
        }
        client.post("/api/v1/usage/tracking/end", json=end_request)
        
        # Get analytics
        response = client.get(f"/api/v1/usage/analytics/{user_id}?days=30")
        assert response.status_code == 200
        
        data = response.json()
        assert data["user_id"] == user_id
        assert "analysis_period" in data
        assert "usage_summary" in data
        assert "forecast" in data
        assert "recommendations_count" in data
        assert "top_recommendations" in data
        
        # Verify usage summary
        usage_summary = data["usage_summary"]
        assert usage_summary["total_generations"] == 1
        assert usage_summary["success_rate"] == 100.0
        assert usage_summary["total_cost"] == 0.25  # 5.0 * 0.05
        assert usage_summary["average_quality_score"] == 0.8
    
    def test_invalid_parameters(self):
        """Test API endpoints with invalid parameters."""
        # Invalid days parameter
        response = client.get("/api/v1/usage/summary/test_user?days=0")
        assert response.status_code == 422  # Validation error
        
        response = client.get("/api/v1/usage/summary/test_user?days=400")
        assert response.status_code == 422  # Validation error
        
        # Invalid budget limit
        response = client.get("/api/v1/usage/budget-alerts/test_user?budget_limit=-1.0")
        assert response.status_code == 422  # Validation error
    
    def test_track_resource_invalid_session(self):
        """Test tracking resource for invalid session."""
        request_data = {
            "session_id": "invalid_session",
            "resource_type": "gpu_seconds",
            "amount": 1.0
        }
        
        response = client.post("/api/v1/usage/tracking/resource", json=request_data)
        # Should succeed but log warning (doesn't raise exception)
        assert response.status_code == 200
    
    def test_complete_workflow_integration(self):
        """Test complete workflow integration through API."""
        user_id = "integration_workflow_user"
        
        # 1. Start tracking
        start_request = {
            "user_id": user_id,
            "generation_type": "image",
            "model_used": "dalle3",
            "prompt": "Integration test image",
            "parameters": {"resolution": [1024, 1024]}
        }
        
        start_response = client.post("/api/v1/usage/tracking/start", json=start_request)
        assert start_response.status_code == 200
        session_id = start_response.json()["session_id"]
        
        # 2. Track multiple resources
        resources = [
            {"resource_type": "api_calls", "amount": 1},
            {"resource_type": "gpu_seconds", "amount": 12.0},
            {"resource_type": "storage_gb", "amount": 0.2}
        ]
        
        for resource in resources:
            resource_request = {
                "session_id": session_id,
                **resource
            }
            response = client.post("/api/v1/usage/tracking/resource", json=resource_request)
            assert response.status_code == 200
        
        # 3. End tracking
        end_request = {
            "session_id": session_id,
            "success": True,
            "quality_score": 0.88
        }
        
        end_response = client.post("/api/v1/usage/tracking/end", json=end_request)
        assert end_response.status_code == 200
        
        end_data = end_response.json()
        assert end_data["success"] is True
        assert end_data["quality_score"] == 0.88
        assert end_data["total_cost"] > 0
        
        # 4. Get usage summary
        summary_response = client.get(f"/api/v1/usage/summary/{user_id}?days=1")
        assert summary_response.status_code == 200
        
        summary_data = summary_response.json()
        assert summary_data["total_generations"] == 1
        assert summary_data["successful_generations"] == 1
        assert summary_data["total_gpu_seconds"] == 12.0
        assert summary_data["total_storage_gb"] == 0.2
        assert summary_data["total_api_calls"] == 1
        
        # 5. Get cost estimate for similar request
        estimate_request = {
            "generation_type": "image",
            "model_name": "dalle3",
            "parameters": {"resolution": [1024, 1024]}
        }
        
        estimate_response = client.post("/api/v1/usage/cost-estimate", json=estimate_request)
        assert estimate_response.status_code == 200
        
        estimate_data = estimate_response.json()
        assert estimate_data["estimated_cost"] > 0
        
        # 6. Get analytics
        analytics_response = client.get(f"/api/v1/usage/analytics/{user_id}?days=1")
        assert analytics_response.status_code == 200
        
        analytics_data = analytics_response.json()
        assert analytics_data["usage_summary"]["total_generations"] == 1
        assert analytics_data["usage_summary"]["success_rate"] == 100.0


class TestUsageTrackingValidation:
    """Test input validation for usage tracking routes."""
    
    def test_start_tracking_validation(self):
        """Test validation for start tracking endpoint."""
        # Missing required fields
        response = client.post("/api/v1/usage/tracking/start", json={})
        assert response.status_code == 422
        
        # Invalid generation type
        invalid_request = {
            "user_id": "test_user",
            "generation_type": "invalid_type",
            "model_used": "test_model",
            "prompt": "test prompt"
        }
        response = client.post("/api/v1/usage/tracking/start", json=invalid_request)
        assert response.status_code == 422
    
    def test_track_resource_validation(self):
        """Test validation for track resource endpoint."""
        # Invalid resource type
        invalid_request = {
            "session_id": "test_session",
            "resource_type": "invalid_resource",
            "amount": 1.0
        }
        response = client.post("/api/v1/usage/tracking/resource", json=invalid_request)
        assert response.status_code == 422
        
        # Negative amount
        invalid_request = {
            "session_id": "test_session",
            "resource_type": "gpu_seconds",
            "amount": -1.0
        }
        response = client.post("/api/v1/usage/tracking/resource", json=invalid_request)
        # Should be handled by the application logic, not validation
        assert response.status_code == 200
    
    def test_cost_estimate_validation(self):
        """Test validation for cost estimate endpoint."""
        # Invalid generation type
        invalid_request = {
            "generation_type": "invalid_type",
            "model_name": "test_model",
            "parameters": {}
        }
        response = client.post("/api/v1/usage/cost-estimate", json=invalid_request)
        assert response.status_code == 422