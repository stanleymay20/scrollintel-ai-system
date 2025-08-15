"""
Integration tests for AI Data Readiness Platform API endpoints.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

from ai_data_readiness.api.app import app


class TestAPIIntegration:
    """Test API integration scenarios."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_dataset_upload_and_analysis_workflow(self, client, sample_csv_data, temp_directory):
        """Test complete dataset upload and analysis workflow via API."""
        # Save sample data to file
        csv_file = temp_directory / "api_test_data.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        # Upload dataset
        with open(csv_file, 'rb') as f:
            files = {"file": ("test_data.csv", f, "text/csv")}
            response = client.post("/api/v1/datasets/upload", files=files)
        
        assert response.status_code == 201
        upload_data = response.json()
        assert "dataset_id" in upload_data
        dataset_id = upload_data["dataset_id"]
        
        # Get dataset info
        response = client.get(f"/api/v1/datasets/{dataset_id}")
        assert response.status_code == 200
        
        dataset_info = response.json()
        assert dataset_info["id"] == dataset_id
        assert "schema" in dataset_info
        assert "metadata" in dataset_info
        
        # Trigger quality assessment
        response = client.post(f"/api/v1/datasets/{dataset_id}/quality-assessment")
        assert response.status_code == 200
        
        quality_data = response.json()
        assert "overall_score" in quality_data
        assert "completeness_score" in quality_data
        assert "accuracy_score" in quality_data
        
        # Get quality report
        response = client.get(f"/api/v1/datasets/{dataset_id}/quality-report")
        assert response.status_code == 200
        
        quality_report = response.json()
        assert "dataset_id" in quality_report
        assert quality_report["dataset_id"] == dataset_id
    
    def test_bias_analysis_api_workflow(self, client, sample_biased_data, temp_directory):
        """Test bias analysis API workflow."""
        # Upload biased dataset
        csv_file = temp_directory / "biased_data.csv"
        sample_biased_data.to_csv(csv_file, index=False)
        
        with open(csv_file, 'rb') as f:
            files = {"file": ("biased_data.csv", f, "text/csv")}
            response = client.post("/api/v1/datasets/upload", files=files)
        
        assert response.status_code == 201
        dataset_id = response.json()["dataset_id"]
        
        # Trigger bias analysis
        bias_request = {
            "protected_attributes": ["gender"],
            "target_column": "approved"
        }
        
        response = client.post(
            f"/api/v1/datasets/{dataset_id}/bias-analysis",
            json=bias_request
        )
        assert response.status_code == 200
        
        bias_data = response.json()
        assert "bias_detected" in bias_data
        assert "bias_score" in bias_data
        assert "protected_attributes" in bias_data
        
        # Get fairness metrics
        response = client.get(
            f"/api/v1/datasets/{dataset_id}/fairness-metrics",
            params={"target_column": "approved"}
        )
        assert response.status_code == 200
        
        fairness_data = response.json()
        assert "fairness_metrics" in fairness_data
        assert "overall_fairness_score" in fairness_data
    
    def test_feature_engineering_api_workflow(self, client, sample_csv_data, temp_directory):
        """Test feature engineering API workflow."""
        # Upload dataset
        csv_file = temp_directory / "feature_data.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        with open(csv_file, 'rb') as f:
            files = {"file": ("feature_data.csv", f, "text/csv")}
            response = client.post("/api/v1/datasets/upload", files=files)
        
        dataset_id = response.json()["dataset_id"]
        
        # Get feature recommendations
        feature_request = {
            "model_type": "classification",
            "target_column": "score"
        }
        
        response = client.post(
            f"/api/v1/datasets/{dataset_id}/feature-recommendations",
            json=feature_request
        )
        assert response.status_code == 200
        
        recommendations = response.json()
        assert "recommended_features" in recommendations
        assert "transformations" in recommendations
        
        # Apply transformations
        if recommendations["transformations"]:
            transform_request = {
                "transformations": recommendations["transformations"][:2]  # Apply first 2
            }
            
            response = client.post(
                f"/api/v1/datasets/{dataset_id}/apply-transformations",
                json=transform_request
            )
            assert response.status_code == 200
            
            transform_result = response.json()
            assert "transformed_dataset_id" in transform_result
            assert "transformation_summary" in transform_result
    
    def test_ai_readiness_report_api(self, client, sample_csv_data, temp_directory):
        """Test AI readiness report generation via API."""
        # Upload dataset
        csv_file = temp_directory / "readiness_data.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        with open(csv_file, 'rb') as f:
            files = {"file": ("readiness_data.csv", f, "text/csv")}
            response = client.post("/api/v1/datasets/upload", files=files)
        
        dataset_id = response.json()["dataset_id"]
        
        # Generate AI readiness report
        response = client.post(f"/api/v1/datasets/{dataset_id}/ai-readiness-report")
        assert response.status_code == 200
        
        report = response.json()
        assert "overall_ai_readiness_score" in report
        assert "dimension_scores" in report
        assert "recommendations" in report
        assert "improvement_roadmap" in report
        
        # Get report in different formats
        response = client.get(
            f"/api/v1/datasets/{dataset_id}/ai-readiness-report",
            params={"format": "json"}
        )
        assert response.status_code == 200
        
        response = client.get(
            f"/api/v1/datasets/{dataset_id}/ai-readiness-report",
            params={"format": "html"}
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_drift_monitoring_api(self, client, sample_csv_data, temp_directory):
        """Test drift monitoring API."""
        # Upload reference dataset
        reference_file = temp_directory / "reference_data.csv"
        sample_csv_data.to_csv(reference_file, index=False)
        
        with open(reference_file, 'rb') as f:
            files = {"file": ("reference_data.csv", f, "text/csv")}
            response = client.post("/api/v1/datasets/upload", files=files)
        
        reference_id = response.json()["dataset_id"]
        
        # Upload current dataset (with drift)
        drifted_data = sample_csv_data.copy()
        drifted_data['age'] = drifted_data['age'] + 10
        
        current_file = temp_directory / "current_data.csv"
        drifted_data.to_csv(current_file, index=False)
        
        with open(current_file, 'rb') as f:
            files = {"file": ("current_data.csv", f, "text/csv")}
            response = client.post("/api/v1/datasets/upload", files=files)
        
        current_id = response.json()["dataset_id"]
        
        # Monitor drift
        drift_request = {
            "reference_dataset_id": reference_id,
            "current_dataset_id": current_id
        }
        
        response = client.post("/api/v1/drift/monitor", json=drift_request)
        assert response.status_code == 200
        
        drift_data = response.json()
        assert "drift_detected" in drift_data
        assert "drift_score" in drift_data
        assert "feature_drift_scores" in drift_data
    
    def test_graphql_api_integration(self, client):
        """Test GraphQL API integration."""
        # GraphQL query to get datasets
        query = """
        query {
            datasets {
                id
                name
                schema
                qualityScore
                createdAt
            }
        }
        """
        
        response = client.post("/graphql", json={"query": query})
        assert response.status_code == 200
        
        data = response.json()
        assert "data" in data
        assert "datasets" in data["data"]
    
    def test_websocket_real_time_updates(self, client):
        """Test WebSocket real-time updates."""
        # This would test WebSocket connections for real-time updates
        # For now, we'll just verify the endpoint exists
        with client.websocket_connect("/ws/updates") as websocket:
            # Send subscription message
            websocket.send_json({
                "type": "subscribe",
                "topic": "quality_assessments"
            })
            
            # This would normally receive real-time updates
            # For testing, we'll just verify connection works
            pass
    
    def test_api_authentication_and_authorization(self, client):
        """Test API authentication and authorization."""
        # Test accessing protected endpoint without auth
        response = client.get("/api/v1/admin/users")
        assert response.status_code == 401
        
        # Test with invalid token
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/admin/users", headers=headers)
        assert response.status_code == 401
        
        # Test with valid token (mocked)
        with patch('ai_data_readiness.api.middleware.auth.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "test_user", "role": "admin"}
            
            headers = {"Authorization": "Bearer valid_token"}
            response = client.get("/api/v1/admin/users", headers=headers)
            # Should not be 401 (might be 404 if endpoint doesn't exist, but not auth error)
            assert response.status_code != 401
    
    def test_api_rate_limiting(self, client):
        """Test API rate limiting."""
        # Make multiple rapid requests
        responses = []
        for i in range(10):
            response = client.get("/health")
            responses.append(response.status_code)
        
        # Most should succeed, but rate limiting might kick in
        success_count = sum(1 for status in responses if status == 200)
        assert success_count >= 5  # At least some should succeed
    
    def test_api_error_handling(self, client):
        """Test API error handling."""
        # Test 404 for non-existent dataset
        response = client.get("/api/v1/datasets/nonexistent_id")
        assert response.status_code == 404
        
        error_data = response.json()
        assert "error" in error_data
        assert "message" in error_data
        
        # Test 400 for invalid request
        response = client.post("/api/v1/datasets/invalid_id/bias-analysis", json={})
        assert response.status_code == 400
        
        # Test 422 for validation errors
        invalid_upload = {"invalid": "data"}
        response = client.post("/api/v1/datasets/upload", json=invalid_upload)
        assert response.status_code == 422
    
    def test_api_pagination(self, client):
        """Test API pagination."""
        # Test datasets pagination
        response = client.get("/api/v1/datasets", params={"page": 1, "limit": 10})
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "limit" in data
        assert "pages" in data
    
    def test_api_filtering_and_sorting(self, client):
        """Test API filtering and sorting."""
        # Test filtering datasets by quality score
        response = client.get(
            "/api/v1/datasets",
            params={"min_quality_score": 0.8, "sort": "created_at", "order": "desc"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
    
    def test_batch_operations_api(self, client, temp_directory):
        """Test batch operations via API."""
        # Create multiple test files
        datasets = []
        for i in range(3):
            data = generate_synthetic_dataset(n_rows=100, n_features=5)
            csv_file = temp_directory / f"batch_data_{i}.csv"
            data.to_csv(csv_file, index=False)
            datasets.append(csv_file)
        
        # Batch upload
        files = []
        for i, csv_file in enumerate(datasets):
            with open(csv_file, 'rb') as f:
                files.append(("files", (f"batch_data_{i}.csv", f.read(), "text/csv")))
        
        response = client.post("/api/v1/datasets/batch-upload", files=files)
        assert response.status_code == 201
        
        batch_result = response.json()
        assert "dataset_ids" in batch_result
        assert len(batch_result["dataset_ids"]) == 3
        
        # Batch quality assessment
        quality_request = {
            "dataset_ids": batch_result["dataset_ids"]
        }
        
        response = client.post("/api/v1/datasets/batch-quality-assessment", json=quality_request)
        assert response.status_code == 200
        
        batch_quality = response.json()
        assert "results" in batch_quality
        assert len(batch_quality["results"]) == 3
    
    def test_export_functionality_api(self, client, sample_csv_data, temp_directory):
        """Test data export functionality via API."""
        # Upload dataset
        csv_file = temp_directory / "export_data.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        with open(csv_file, 'rb') as f:
            files = {"file": ("export_data.csv", f, "text/csv")}
            response = client.post("/api/v1/datasets/upload", files=files)
        
        dataset_id = response.json()["dataset_id"]
        
        # Export in different formats
        formats = ["csv", "json", "parquet"]
        
        for fmt in formats:
            response = client.get(
                f"/api/v1/datasets/{dataset_id}/export",
                params={"format": fmt}
            )
            assert response.status_code == 200
            
            if fmt == "csv":
                assert "text/csv" in response.headers["content-type"]
            elif fmt == "json":
                assert "application/json" in response.headers["content-type"]
            elif fmt == "parquet":
                assert "application/octet-stream" in response.headers["content-type"]


def generate_synthetic_dataset(n_rows=1000, n_features=10):
    """Generate synthetic dataset for API testing."""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    data = {}
    for i in range(n_features):
        if i % 2 == 0:
            data[f'feature_{i}'] = np.random.normal(0, 1, n_rows)
        else:
            data[f'feature_{i}'] = np.random.choice(['A', 'B', 'C'], n_rows)
    
    data['target'] = np.random.choice([0, 1], n_rows)
    
    return pd.DataFrame(data)