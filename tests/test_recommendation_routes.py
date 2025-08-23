"""
Integration tests for the Recommendation Engine API routes
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json
from unittest.mock import Mock, patch

from scrollintel.api.routes.recommendation_routes import router


# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestRecommendationRoutes:
    """Test cases for recommendation API routes"""
    
    @pytest.fixture
    def sample_schema_request(self):
        """Sample schema request data"""
        return {
            "name": "test_schema",
            "columns": [
                {"name": "id", "type": "integer"},
                {"name": "name", "type": "string"},
                {"name": "age", "type": "string"}
            ],
            "data_types": {
                "id": "int64",
                "name": "object",
                "age": "object"
            }
        }
    
    @pytest.fixture
    def sample_dataset_request(self, sample_schema_request):
        """Sample dataset request data"""
        return {
            "name": "test_dataset",
            "schema": sample_schema_request,
            "row_count": 10000,
            "size_mb": 50.0,
            "quality_score": 0.85
        }
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/recommendations/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "recommendation_engine"
        assert "timestamp" in data
    
    def test_get_transformation_recommendations(self, sample_schema_request):
        """Test transformation recommendations endpoint"""
        target_schema = sample_schema_request.copy()
        target_schema["data_types"]["age"] = "int64"  # Different type to trigger recommendation
        
        request_data = {
            "source_schema": sample_schema_request,
            "target_schema": target_schema
        }
        
        response = client.post("/api/v1/recommendations/transformations", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        if len(data) > 0:
            recommendation = data[0]
            assert "name" in recommendation
            assert "type" in recommendation
            assert "description" in recommendation
            assert "confidence" in recommendation
            assert "parameters" in recommendation
            assert "estimated_performance_impact" in recommendation
            
            # Confidence should be between 0 and 1
            assert 0 <= recommendation["confidence"] <= 1
    
    def test_get_transformation_recommendations_invalid_data(self):
        """Test transformation recommendations with invalid data"""
        invalid_request = {
            "source_schema": {
                "name": "invalid",
                "columns": [],  # Invalid: empty columns
                "data_types": {}
            },
            "target_schema": {
                "name": "target",
                "columns": [{"name": "id", "type": "integer"}],
                "data_types": {"id": "int64"}
            }
        }
        
        response = client.post("/api/v1/recommendations/transformations", json=invalid_request)
        
        # Should handle gracefully
        assert response.status_code in [200, 422]  # 422 for validation error
    
    def test_get_optimization_recommendations(self):
        """Test optimization recommendations endpoint"""
        request_data = {
            "pipeline": {
                "name": "test_pipeline",
                "steps": ["extract", "transform", "load"]
            },
            "metrics": {
                "execution_time_seconds": 600,  # Long execution time
                "memory_usage_mb": 2000,  # High memory usage
                "rows_processed": 1000000,
                "error_rate": 0.05  # High error rate
            }
        }
        
        response = client.post("/api/v1/recommendations/optimizations", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        if len(data) > 0:
            optimization = data[0]
            assert "category" in optimization
            assert "description" in optimization
            assert "impact" in optimization
            assert "implementation_effort" in optimization
            assert "estimated_improvement" in optimization
            assert "priority" in optimization
            
            # Estimated improvement should be between 0 and 1
            assert 0 <= optimization["estimated_improvement"] <= 1
            assert optimization["priority"] > 0
    
    def test_get_join_recommendation(self, sample_dataset_request):
        """Test join recommendation endpoint"""
        # Create second dataset with common column
        right_dataset = sample_dataset_request.copy()
        right_dataset["name"] = "orders"
        right_dataset["schema"]["name"] = "orders_schema"
        right_dataset["schema"]["columns"].append({"name": "customer_id", "type": "integer"})
        right_dataset["schema"]["data_types"]["customer_id"] = "int64"
        
        # Add common column to left dataset too
        left_dataset = sample_dataset_request.copy()
        left_dataset["schema"]["columns"].append({"name": "customer_id", "type": "integer"})
        left_dataset["schema"]["data_types"]["customer_id"] = "int64"
        
        request_data = {
            "left_dataset": left_dataset,
            "right_dataset": right_dataset
        }
        
        response = client.post("/api/v1/recommendations/join-strategy", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "join_type" in data
        assert "left_key" in data
        assert "right_key" in data
        assert "confidence" in data
        assert "estimated_rows" in data
        assert "performance_score" in data
        
        # Validate data types and ranges
        assert data["join_type"] in ["inner", "left", "right", "outer", "cross"]
        assert 0 <= data["confidence"] <= 1
        assert data["estimated_rows"] >= 0
        assert 0 <= data["performance_score"] <= 1
    
    def test_analyze_data_patterns(self):
        """Test data pattern analysis endpoint"""
        sample_data = {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", None, "David", "Eve"],
            "age": ["25", "30", "35", "40", "45"],  # String but numeric
            "salary": [50000, 60000, 70000, 1000000, 55000],  # Has outlier
            "created_at": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        }
        
        request_data = {"data_sample": sample_data}
        
        response = client.post("/api/v1/recommendations/data-patterns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "patterns" in data
        assert "anomalies" in data
        assert "recommendations" in data
        assert "quality_issues" in data
        
        # All should be lists
        assert isinstance(data["patterns"], list)
        assert isinstance(data["anomalies"], list)
        assert isinstance(data["recommendations"], list)
        assert isinstance(data["quality_issues"], list)
    
    def test_analyze_data_patterns_empty_data(self):
        """Test data pattern analysis with empty data"""
        request_data = {"data_sample": {}}
        
        response = client.post("/api/v1/recommendations/data-patterns", json=request_data)
        
        # Should handle gracefully
        assert response.status_code in [200, 422]
    
    def test_submit_feedback(self):
        """Test feedback submission endpoint"""
        feedback_data = {
            "recommendation_id": "test_rec_001",
            "feedback_type": "rating",
            "rating": 4,
            "comment": "Very helpful recommendation",
            "was_helpful": True,
            "implementation_difficulty": "easy",
            "actual_benefit": "Improved performance by 30%",
            "user_id": "test_user"
        }
        
        response = client.post("/api/v1/recommendations/feedback", json=feedback_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "recommendation_id" in data
        assert data["recommendation_id"] == "test_rec_001"
    
    def test_submit_feedback_minimal(self):
        """Test feedback submission with minimal data"""
        feedback_data = {
            "recommendation_id": "test_rec_002",
            "feedback_type": "usage"
        }
        
        response = client.post("/api/v1/recommendations/feedback", json=feedback_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["recommendation_id"] == "test_rec_002"
    
    def test_submit_feedback_invalid_rating(self):
        """Test feedback submission with invalid rating"""
        feedback_data = {
            "recommendation_id": "test_rec_003",
            "feedback_type": "rating",
            "rating": 10  # Invalid: should be 1-5
        }
        
        response = client.post("/api/v1/recommendations/feedback", json=feedback_data)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_get_recommendation_history(self):
        """Test recommendation history endpoint"""
        response = client.get("/api/v1/recommendations/history")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "recommendations" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        
        assert isinstance(data["recommendations"], list)
        assert isinstance(data["total"], int)
        assert data["limit"] == 50  # Default limit
        assert data["offset"] == 0  # Default offset
    
    def test_get_recommendation_history_with_filters(self):
        """Test recommendation history with filters"""
        params = {
            "recommendation_type": "transformation",
            "limit": 10,
            "offset": 5
        }
        
        response = client.get("/api/v1/recommendations/history", params=params)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["limit"] == 10
        assert data["offset"] == 5
    
    def test_get_recommendation_history_invalid_params(self):
        """Test recommendation history with invalid parameters"""
        params = {
            "limit": -1,  # Invalid: negative limit
            "offset": -5  # Invalid: negative offset
        }
        
        response = client.get("/api/v1/recommendations/history", params=params)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_get_recommendation_metrics(self):
        """Test recommendation metrics endpoint"""
        response = client.get("/api/v1/recommendations/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required metrics
        assert "total_recommendations" in data
        assert "acceptance_rate" in data
        assert "implementation_rate" in data
        assert "average_confidence_score" in data
        assert "average_user_rating" in data
        assert "performance_improvement" in data
        assert "recommendation_types" in data
        assert "recent_trends" in data
        
        # Validate data types and ranges
        assert isinstance(data["total_recommendations"], int)
        assert 0 <= data["acceptance_rate"] <= 1
        assert 0 <= data["implementation_rate"] <= 1
        assert 0 <= data["average_confidence_score"] <= 1
        assert 1 <= data["average_user_rating"] <= 5
        assert isinstance(data["recommendation_types"], dict)
        assert isinstance(data["recent_trends"], dict)
    
    def test_store_performance_baseline(self):
        """Test performance baseline storage endpoint"""
        baseline_data = {
            "execution_time_seconds": 120.5,
            "memory_usage_mb": 512.0,
            "cpu_usage_percent": 75.0,
            "rows_processed": 100000,
            "data_size_mb": 250.0,
            "error_rate": 0.01,
            "data_quality_score": 0.95,
            "environment": {
                "platform": "linux",
                "python_version": "3.9.0"
            },
            "resources": {
                "cpu_cores": 4,
                "memory_gb": 8
            }
        }
        
        response = client.post(
            "/api/v1/recommendations/performance-baseline",
            params={"pipeline_id": "test_pipeline_001"},
            json=baseline_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "pipeline_id" in data
        assert "baseline_id" in data
        assert data["pipeline_id"] == "test_pipeline_001"
    
    def test_store_performance_baseline_minimal(self):
        """Test performance baseline storage with minimal data"""
        baseline_data = {
            "execution_time_seconds": 60.0,
            "memory_usage_mb": 256.0,
            "cpu_usage_percent": 50.0,
            "rows_processed": 50000,
            "data_size_mb": 100.0
        }
        
        response = client.post(
            "/api/v1/recommendations/performance-baseline",
            params={"pipeline_id": "test_pipeline_002"},
            json=baseline_data
        )
        
        assert response.status_code == 200
    
    def test_missing_required_fields(self):
        """Test API endpoints with missing required fields"""
        # Test transformation recommendations without required fields
        response = client.post("/api/v1/recommendations/transformations", json={})
        assert response.status_code == 422
        
        # Test optimization recommendations without required fields
        response = client.post("/api/v1/recommendations/optimizations", json={})
        assert response.status_code == 422
        
        # Test join recommendations without required fields
        response = client.post("/api/v1/recommendations/join-strategy", json={})
        assert response.status_code == 422
        
        # Test feedback without required fields
        response = client.post("/api/v1/recommendations/feedback", json={})
        assert response.status_code == 422
    
    def test_malformed_json(self):
        """Test API endpoints with malformed JSON"""
        malformed_json = '{"invalid": json}'
        
        response = client.post(
            "/api/v1/recommendations/transformations",
            data=malformed_json,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_large_data_handling(self):
        """Test API endpoints with large data payloads"""
        # Create large sample data
        large_data = {f"col_{i}": list(range(1000)) for i in range(10)}
        
        request_data = {"data_sample": large_data}
        
        response = client.post("/api/v1/recommendations/data-patterns", json=request_data)
        
        # Should handle large data gracefully
        assert response.status_code in [200, 413, 422]  # 413 for payload too large
    
    @patch('scrollintel.engines.recommendation_engine.RecommendationEngine.recommend_transformations')
    def test_engine_error_handling(self, mock_recommend):
        """Test error handling when recommendation engine fails"""
        # Mock engine to raise an exception
        mock_recommend.side_effect = Exception("Engine error")
        
        request_data = {
            "source_schema": {
                "name": "source",
                "columns": [{"name": "id", "type": "integer"}],
                "data_types": {"id": "int64"}
            },
            "target_schema": {
                "name": "target",
                "columns": [{"name": "id", "type": "integer"}],
                "data_types": {"id": "int64"}
            }
        }
        
        response = client.post("/api/v1/recommendations/transformations", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Engine error" in data["detail"]
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get("/api/v1/recommendations/health")
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__])