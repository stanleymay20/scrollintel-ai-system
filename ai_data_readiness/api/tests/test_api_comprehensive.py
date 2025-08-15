"""Comprehensive API tests for AI Data Readiness Platform."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

from ..app import app
from ..middleware.auth import create_access_token


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_token():
    """Create authentication token for testing."""
    token_data = {
        "user_id": "test_user",
        "username": "testuser",
        "email": "test@example.com",
        "permissions": ["read", "write", "delete"]
    }
    return create_access_token(token_data)


@pytest.fixture
def auth_headers(auth_token):
    """Create authorization headers."""
    return {"Authorization": f"Bearer {auth_token}"}


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_basic_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_api_health_check(self, client):
        """Test API health check."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "components" in data
    
    def test_detailed_health_check(self, client):
        """Test detailed health check."""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "process" in data
        assert "config" in data


class TestAuthenticationEndpoints:
    """Test authentication endpoints."""
    
    def test_login_success(self, client):
        """Test successful login."""
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "access_token" in data
        assert "user" in data
    
    def test_login_failure(self, client):
        """Test failed login."""
        login_data = {
            "username": "invalid",
            "password": "invalid"
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 401
    
    def test_logout(self, client):
        """Test logout."""
        response = client.post("/api/v1/auth/logout")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestDatasetEndpoints:
    """Test dataset management endpoints."""
    
    def test_create_dataset(self, client, auth_headers):
        """Test dataset creation."""
        dataset_data = {
            "name": "Test Dataset",
            "description": "Test dataset description",
            "source": "test_source",
            "format": "csv",
            "tags": ["test", "sample"]
        }
        response = client.post("/api/v1/datasets", json=dataset_data, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == dataset_data["name"]
        assert data["description"] == dataset_data["description"]
        assert "id" in data
    
    def test_list_datasets(self, client, auth_headers):
        """Test dataset listing."""
        response = client.get("/api/v1/datasets", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "datasets" in data
        assert "total" in data
        assert "page" in data
    
    def test_get_dataset(self, client, auth_headers):
        """Test getting a specific dataset."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"  # Valid UUID format
        response = client.get(f"/api/v1/datasets/{dataset_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset_id
    
    def test_update_dataset(self, client, auth_headers):
        """Test dataset update."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        update_data = {
            "name": "Updated Dataset",
            "description": "Updated description"
        }
        response = client.put(f"/api/v1/datasets/{dataset_id}", json=update_data, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
    
    def test_delete_dataset(self, client, auth_headers):
        """Test dataset deletion."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        response = client.delete(f"/api/v1/datasets/{dataset_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_get_metrics(self, client, auth_headers):
        """Test system metrics."""
        response = client.get("/api/v1/datasets/metrics", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_datasets" in data
        assert "active_jobs" in data
        assert "average_quality_score" in data


class TestQualityEndpoints:
    """Test quality assessment endpoints."""
    
    def test_assess_quality(self, client, auth_headers):
        """Test quality assessment."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        assessment_data = {
            "dataset_id": dataset_id,
            "dimensions": ["completeness", "accuracy"],
            "generate_recommendations": True
        }
        response = client.post(f"/api/v1/datasets/{dataset_id}/quality", json=assessment_data, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset_id
        assert "overall_score" in data
        assert "completeness_score" in data
    
    def test_get_quality_report(self, client, auth_headers):
        """Test getting quality report."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        response = client.get(f"/api/v1/datasets/{dataset_id}/quality", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset_id
    
    def test_get_ai_readiness(self, client, auth_headers):
        """Test AI readiness assessment."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        response = client.get(f"/api/v1/datasets/{dataset_id}/ai-readiness", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "overall_score" in data
        assert "data_quality_score" in data


class TestBiasEndpoints:
    """Test bias analysis endpoints."""
    
    def test_analyze_bias(self, client, auth_headers):
        """Test bias analysis."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        bias_data = {
            "dataset_id": dataset_id,
            "protected_attributes": ["gender", "age"],
            "target_column": "outcome"
        }
        response = client.post(f"/api/v1/datasets/{dataset_id}/bias", json=bias_data, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset_id
        assert "bias_metrics" in data
    
    def test_get_bias_report(self, client, auth_headers):
        """Test getting bias report."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        response = client.get(f"/api/v1/datasets/{dataset_id}/bias", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset_id


class TestFeatureEndpoints:
    """Test feature engineering endpoints."""
    
    def test_recommend_features(self, client, auth_headers):
        """Test feature recommendations."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        feature_data = {
            "dataset_id": dataset_id,
            "model_type": "classification",
            "target_column": "target"
        }
        response = client.post(f"/api/v1/datasets/{dataset_id}/features", json=feature_data, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset_id
        assert "recommendations" in data


class TestComplianceEndpoints:
    """Test compliance checking endpoints."""
    
    def test_check_compliance(self, client, auth_headers):
        """Test compliance checking."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        compliance_data = {
            "dataset_id": dataset_id,
            "regulations": ["GDPR", "CCPA"]
        }
        response = client.post(f"/api/v1/datasets/{dataset_id}/compliance", json=compliance_data, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset_id
        assert "compliance_score" in data


class TestLineageEndpoints:
    """Test data lineage endpoints."""
    
    def test_get_lineage(self, client, auth_headers):
        """Test getting data lineage."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        response = client.get(f"/api/v1/datasets/{dataset_id}/lineage", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset_id
        assert "source_datasets" in data


class TestDriftEndpoints:
    """Test drift monitoring endpoints."""
    
    def test_monitor_drift(self, client, auth_headers):
        """Test drift monitoring setup."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        drift_data = {
            "dataset_id": dataset_id,
            "reference_dataset_id": "550e8400-e29b-41d4-a716-446655440001",
            "monitoring_frequency": "daily",
            "drift_threshold": 0.1
        }
        response = client.post(f"/api/v1/datasets/{dataset_id}/drift", json=drift_data, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset_id
        assert "drift_score" in data


class TestProcessingEndpoints:
    """Test processing job endpoints."""
    
    def test_create_processing_job(self, client, auth_headers):
        """Test creating a processing job."""
        job_data = {
            "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
            "job_type": "quality_assessment",
            "parameters": {},
            "priority": "normal"
        }
        response = client.post("/api/v1/processing/jobs", json=job_data, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["job_type"] == job_data["job_type"]
    
    def test_get_processing_job(self, client, auth_headers):
        """Test getting a processing job."""
        job_id = "550e8400-e29b-41d4-a716-446655440000"
        response = client.get(f"/api/v1/processing/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id


class TestErrorHandling:
    """Test error handling."""
    
    def test_unauthorized_access(self, client):
        """Test unauthorized access."""
        response = client.get("/api/v1/datasets")
        assert response.status_code == 401
    
    def test_invalid_dataset_id(self, client, auth_headers):
        """Test invalid dataset ID."""
        response = client.get("/api/v1/datasets/invalid-id", headers=auth_headers)
        assert response.status_code == 400
    
    def test_not_found(self, client, auth_headers):
        """Test 404 handling."""
        response = client.get("/api/v1/nonexistent", headers=auth_headers)
        assert response.status_code == 404


class TestRateLimiting:
    """Test rate limiting."""
    
    def test_rate_limit_not_exceeded(self, client, auth_headers):
        """Test normal request within rate limit."""
        response = client.get("/api/v1/datasets", headers=auth_headers)
        assert response.status_code == 200
    
    # Note: Testing actual rate limiting would require many requests
    # This is a placeholder for rate limiting tests


class TestValidation:
    """Test request validation."""
    
    def test_invalid_json(self, client, auth_headers):
        """Test invalid JSON handling."""
        response = client.post(
            "/api/v1/datasets",
            data="invalid json",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self, client, auth_headers):
        """Test missing required fields."""
        dataset_data = {
            "description": "Missing name field"
        }
        response = client.post("/api/v1/datasets", json=dataset_data, headers=auth_headers)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__])