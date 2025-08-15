"""
Tests for deployment automation API routes.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from scrollintel.api.routes.deployment_routes import router
from scrollintel.models.deployment_models import (
    CloudProvider, DeploymentEnvironment
)


@pytest.fixture
def client():
    """Create test client."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_user():
    """Mock authenticated user."""
    return {"id": "user-123", "username": "testuser"}


class TestDeploymentRoutes:
    """Test deployment API routes."""
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_create_deployment_config(self, mock_auth, client, mock_user):
        """Test creating deployment configuration."""
        mock_auth.return_value = mock_user
        
        request_data = {
            "application_id": "app-123",
            "environment": "production",
            "cloud_provider": "aws",
            "config": {
                "region": "us-west-2",
                "instance_type": "t3.micro"
            }
        }
        
        response = client.post("/api/v1/deployment/config", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["application_id"] == "app-123"
        assert data["environment"] == "production"
        assert data["cloud_provider"] == "aws"
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_get_deployment_config_not_found(self, mock_auth, client, mock_user):
        """Test getting non-existent deployment configuration."""
        mock_auth.return_value = mock_user
        
        response = client.get("/api/v1/deployment/config/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_validate_deployment_config(self, mock_auth, client, mock_user):
        """Test validating deployment configuration."""
        mock_auth.return_value = mock_user
        
        response = client.post("/api/v1/deployment/config/test-config/validate")
        
        assert response.status_code == 200
        data = response.json()
        assert "is_valid" in data
        assert "errors" in data
        assert "warnings" in data
        assert "recommendations" in data
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_deploy_application_dry_run(self, mock_auth, client, mock_user):
        """Test deploying application with dry run."""
        mock_auth.return_value = mock_user
        
        request_data = {
            "deployment_config_id": "config-123",
            "dry_run": True
        }
        
        response = client.post("/api/v1/deployment/deploy", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "dry_run_success"
        assert "dry_run_success" in data["status"]
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_deploy_application_actual(self, mock_auth, client, mock_user):
        """Test deploying application actual deployment."""
        mock_auth.return_value = mock_user
        
        request_data = {
            "deployment_config_id": "config-123",
            "dry_run": False
        }
        
        response = client.post("/api/v1/deployment/deploy", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_list_cloud_providers(self, mock_auth, client):
        """Test listing supported cloud providers."""
        response = client.get("/api/v1/deployment/providers")
        
        assert response.status_code == 200
        data = response.json()
        assert "aws" in data
        assert "azure" in data
        assert "gcp" in data
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_list_deployment_environments(self, mock_auth, client):
        """Test listing supported deployment environments."""
        response = client.get("/api/v1/deployment/environments")
        
        assert response.status_code == 200
        data = response.json()
        assert "development" in data
        assert "staging" in data
        assert "production" in data
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_generate_dockerfile(self, mock_auth, client, mock_user):
        """Test generating Dockerfile."""
        mock_auth.return_value = mock_user
        
        response = client.post("/api/v1/deployment/config/test-config/dockerfile")
        
        assert response.status_code == 200
        data = response.json()
        assert "dockerfile" in data
        assert "FROM python:3.11-slim" in data["dockerfile"]
        assert "message" in data
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_generate_infrastructure_code_terraform(self, mock_auth, client, mock_user):
        """Test generating Terraform infrastructure code."""
        mock_auth.return_value = mock_user
        
        response = client.post(
            "/api/v1/deployment/config/test-config/infrastructure?template_type=terraform"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "infrastructure_code" in data
        assert "template_type" in data
        assert data["template_type"] == "terraform"
        assert "aws_instance" in data["infrastructure_code"]
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_generate_cicd_pipeline_github(self, mock_auth, client, mock_user):
        """Test generating GitHub Actions CI/CD pipeline."""
        mock_auth.return_value = mock_user
        
        response = client.post(
            "/api/v1/deployment/config/test-config/pipeline?platform=github"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "pipeline_content" in data
        assert "platform" in data
        assert data["platform"] == "github"
        assert "CI/CD Pipeline" in data["pipeline_content"]
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_create_deployment_template(self, mock_auth, client, mock_user):
        """Test creating deployment template."""
        mock_auth.return_value = mock_user
        
        template_data = {
            "name": "Python Web App Template",
            "description": "Template for Python web applications",
            "cloud_provider": "aws",
            "dockerfile_template": "FROM python:3.11-slim\n...",
            "infrastructure_template": "resource \"aws_instance\" ...",
            "cicd_template": "name: CI/CD Pipeline\n..."
        }
        
        response = client.post("/api/v1/deployment/template", json=template_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "template_id" in data
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_list_deployment_templates(self, mock_auth, client, mock_user):
        """Test listing deployment templates."""
        mock_auth.return_value = mock_user
        
        response = client.get("/api/v1/deployment/template")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_list_deployments(self, mock_auth, client, mock_user):
        """Test listing deployments."""
        mock_auth.return_value = mock_user
        
        response = client.get("/api/v1/deployment/deploy")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_list_deployments_with_filters(self, mock_auth, client, mock_user):
        """Test listing deployments with filters."""
        mock_auth.return_value = mock_user
        
        response = client.get(
            "/api/v1/deployment/deploy?application_id=app-123&environment=production&status=success"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_get_deployment_status_not_found(self, mock_auth, client, mock_user):
        """Test getting deployment status for non-existent deployment."""
        mock_auth.return_value = mock_user
        
        response = client.get("/api/v1/deployment/deploy/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_update_deployment_config_not_found(self, mock_auth, client, mock_user):
        """Test updating non-existent deployment configuration."""
        mock_auth.return_value = mock_user
        
        update_data = {
            "auto_scaling": {
                "min_instances": 2,
                "max_instances": 20
            }
        }
        
        response = client.put("/api/v1/deployment/config/nonexistent", json=update_data)
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_delete_deployment_config(self, mock_auth, client, mock_user):
        """Test deleting deployment configuration."""
        mock_auth.return_value = mock_user
        
        response = client.delete("/api/v1/deployment/config/test-config")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "deleted successfully" in data["message"]
    
    def test_unauthorized_access(self, client):
        """Test unauthorized access to protected endpoints."""
        # Test without authentication
        response = client.post("/api/v1/deployment/config", json={
            "application_id": "app-123",
            "environment": "production",
            "cloud_provider": "aws"
        })
        
        # Should return 422 due to missing dependency (auth)
        assert response.status_code == 422
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_invalid_deployment_request(self, mock_auth, client, mock_user):
        """Test creating deployment config with invalid data."""
        mock_auth.return_value = mock_user
        
        # Missing required fields
        request_data = {
            "application_id": "app-123"
            # Missing environment and cloud_provider
        }
        
        response = client.post("/api/v1/deployment/config", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_invalid_cloud_provider(self, mock_auth, client, mock_user):
        """Test creating deployment config with invalid cloud provider."""
        mock_auth.return_value = mock_user
        
        request_data = {
            "application_id": "app-123",
            "environment": "production",
            "cloud_provider": "invalid_provider"
        }
        
        response = client.post("/api/v1/deployment/config", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    @patch('scrollintel.api.routes.deployment_routes.get_current_user')
    def test_invalid_environment(self, mock_auth, client, mock_user):
        """Test creating deployment config with invalid environment."""
        mock_auth.return_value = mock_user
        
        request_data = {
            "application_id": "app-123",
            "environment": "invalid_environment",
            "cloud_provider": "aws"
        }
        
        response = client.post("/api/v1/deployment/config", json=request_data)
        
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__])