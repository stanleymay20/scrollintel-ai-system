"""
Simple tests for deployment automation API routes without full auth system.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Create a simple test app without the full gateway
app = FastAPI()

# Mock the auth dependency
def mock_get_current_user():
    return {"id": "test-user", "username": "testuser"}

# Import and setup the router with mocked auth
with patch('scrollintel.api.routes.deployment_routes.get_current_user', mock_get_current_user):
    from scrollintel.api.routes.deployment_routes import router
    app.include_router(router)

client = TestClient(app)


class TestDeploymentRoutesSimple:
    """Simple tests for deployment routes."""
    
    def test_list_cloud_providers(self):
        """Test listing supported cloud providers."""
        response = client.get("/api/v1/deployment/providers")
        
        assert response.status_code == 200
        data = response.json()
        assert "aws" in data
        assert "azure" in data
        assert "gcp" in data
        assert "docker" in data
        assert "kubernetes" in data
    
    def test_list_deployment_environments(self):
        """Test listing supported deployment environments."""
        response = client.get("/api/v1/deployment/environments")
        
        assert response.status_code == 200
        data = response.json()
        assert "development" in data
        assert "staging" in data
        assert "production" in data
        assert "test" in data
    
    def test_create_deployment_config(self):
        """Test creating deployment configuration."""
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
        assert "container_config" in data
        assert "infrastructure_code" in data
        assert "cicd_pipeline" in data
    
    def test_deploy_application_dry_run(self):
        """Test deploying application with dry run."""
        request_data = {
            "deployment_config_id": "config-123",
            "dry_run": True
        }
        
        response = client.post("/api/v1/deployment/deploy", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "dry_run_success"
        assert "deployment_id" in data
    
    def test_deploy_application_actual(self):
        """Test deploying application actual deployment."""
        request_data = {
            "deployment_config_id": "config-123",
            "dry_run": False
        }
        
        response = client.post("/api/v1/deployment/deploy", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "deployment_id" in data
        assert "container_image" in data
        assert "endpoints" in data
    
    def test_generate_dockerfile(self):
        """Test generating Dockerfile."""
        response = client.post("/api/v1/deployment/config/test-config/dockerfile")
        
        assert response.status_code == 200
        data = response.json()
        assert "dockerfile" in data
        assert "FROM python:3.11-slim" in data["dockerfile"]
        assert "message" in data
    
    def test_generate_infrastructure_code_terraform(self):
        """Test generating Terraform infrastructure code."""
        response = client.post(
            "/api/v1/deployment/config/test-config/infrastructure?template_type=terraform"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "infrastructure_code" in data
        assert "template_type" in data
        assert data["template_type"] == "terraform"
        assert "aws_instance" in data["infrastructure_code"]
    
    def test_generate_cicd_pipeline_github(self):
        """Test generating GitHub Actions CI/CD pipeline."""
        response = client.post(
            "/api/v1/deployment/config/test-config/pipeline?platform=github"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "pipeline_content" in data
        assert "platform" in data
        assert data["platform"] == "github"
        assert "CI/CD Pipeline" in data["pipeline_content"]
    
    def test_create_deployment_template(self):
        """Test creating deployment template."""
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
    
    def test_list_deployment_templates(self):
        """Test listing deployment templates."""
        response = client.get("/api/v1/deployment/template")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_list_deployments(self):
        """Test listing deployments."""
        response = client.get("/api/v1/deployment/deploy")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_invalid_deployment_request(self):
        """Test creating deployment config with invalid data."""
        # Missing required fields
        request_data = {
            "application_id": "app-123"
            # Missing environment and cloud_provider
        }
        
        response = client.post("/api/v1/deployment/config", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_cloud_provider(self):
        """Test creating deployment config with invalid cloud provider."""
        request_data = {
            "application_id": "app-123",
            "environment": "production",
            "cloud_provider": "invalid_provider"
        }
        
        response = client.post("/api/v1/deployment/config", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_environment(self):
        """Test creating deployment config with invalid environment."""
        request_data = {
            "application_id": "app-123",
            "environment": "invalid_environment",
            "cloud_provider": "aws"
        }
        
        response = client.post("/api/v1/deployment/config", json=request_data)
        
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__])