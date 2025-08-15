"""
Tests for deployment models.
"""

import pytest
from datetime import datetime

from scrollintel.models.deployment_models import (
    DeploymentConfig, ContainerConfig, InfrastructureCode, CICDPipeline,
    DeploymentResult, DeploymentValidation, DeploymentTemplate,
    CloudProvider, DeploymentEnvironment
)


class TestDeploymentModels:
    """Test deployment model validation and functionality."""
    
    def test_container_config_creation(self):
        """Test ContainerConfig model creation."""
        config = ContainerConfig(
            base_image="python:3.11-slim",
            dockerfile_content="FROM python:3.11-slim\nWORKDIR /app",
            build_args={"APP_NAME": "test-app"},
            environment_vars={"PORT": "8000"},
            exposed_ports=[8000],
            volumes=["/app/data"],
            health_check={"test": ["CMD", "curl", "-f", "http://localhost:8000/health"]},
            resource_limits={"memory": "512m", "cpu": "0.5"}
        )
        
        assert config.base_image == "python:3.11-slim"
        assert config.exposed_ports == [8000]
        assert config.build_args["APP_NAME"] == "test-app"
        assert config.environment_vars["PORT"] == "8000"
        assert config.health_check["test"] == ["CMD", "curl", "-f", "http://localhost:8000/health"]
    
    def test_infrastructure_code_creation(self):
        """Test InfrastructureCode model creation."""
        infrastructure = InfrastructureCode(
            provider=CloudProvider.AWS,
            template_type="terraform",
            template_content="resource \"aws_instance\" \"app\" { ... }",
            variables={"region": "us-west-2", "instance_type": "t3.micro"},
            outputs={"instance_ip": "Instance IP address"},
            dependencies=["vpc", "security_group"]
        )
        
        assert infrastructure.provider == CloudProvider.AWS
        assert infrastructure.template_type == "terraform"
        assert "aws_instance" in infrastructure.template_content
        assert infrastructure.variables["region"] == "us-west-2"
        assert infrastructure.outputs["instance_ip"] == "Instance IP address"
        assert "vpc" in infrastructure.dependencies
    
    def test_cicd_pipeline_creation(self):
        """Test CICDPipeline model creation."""
        pipeline = CICDPipeline(
            platform="github",
            pipeline_content="name: CI/CD Pipeline\non: [push]",
            stages=["test", "build", "deploy"],
            triggers=["push", "pull_request"],
            environment_configs={
                "staging": {"auto_deploy": True},
                "production": {"manual_approval": True}
            },
            secrets=["DOCKER_TOKEN", "AWS_CREDENTIALS"]
        )
        
        assert pipeline.platform == "github"
        assert "CI/CD Pipeline" in pipeline.pipeline_content
        assert "test" in pipeline.stages
        assert "push" in pipeline.triggers
        assert pipeline.environment_configs["staging"]["auto_deploy"] is True
        assert "DOCKER_TOKEN" in pipeline.secrets
    
    def test_deployment_config_creation(self):
        """Test DeploymentConfig model creation."""
        container_config = ContainerConfig(
            base_image="python:3.11-slim",
            dockerfile_content="FROM python:3.11-slim",
            build_args={},
            environment_vars={},
            exposed_ports=[8000],
            volumes=[],
            health_check=None,
            resource_limits={}
        )
        
        infrastructure_code = InfrastructureCode(
            provider=CloudProvider.AWS,
            template_type="terraform",
            template_content="resource \"aws_instance\" \"app\" {}",
            variables={},
            outputs={},
            dependencies=[]
        )
        
        cicd_pipeline = CICDPipeline(
            platform="github",
            pipeline_content="name: CI/CD",
            stages=["test"],
            triggers=["push"],
            environment_configs={},
            secrets=[]
        )
        
        config = DeploymentConfig(
            id="test-config-123",
            name="test-deployment",
            application_id="app-123",
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            container_config=container_config,
            infrastructure_code=infrastructure_code,
            cicd_pipeline=cicd_pipeline,
            auto_scaling={"min_instances": 1, "max_instances": 5},
            load_balancing={"type": "application"},
            monitoring={"metrics_enabled": True},
            security={"https_only": True},
            created_by="test-user"
        )
        
        assert config.id == "test-config-123"
        assert config.name == "test-deployment"
        assert config.application_id == "app-123"
        assert config.environment == DeploymentEnvironment.PRODUCTION
        assert config.cloud_provider == CloudProvider.AWS
        assert config.auto_scaling["min_instances"] == 1
        assert config.load_balancing["type"] == "application"
        assert config.monitoring["metrics_enabled"] is True
        assert config.security["https_only"] is True
        assert config.created_by == "test-user"
        assert isinstance(config.created_at, datetime)
    
    def test_deployment_result_creation(self):
        """Test DeploymentResult model creation."""
        result = DeploymentResult(
            deployment_id="deploy-123",
            status="success",
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            container_image="myapp:latest",
            infrastructure_resources=[
                {"type": "aws_instance", "id": "i-1234567890abcdef0"},
                {"type": "aws_vpc", "id": "vpc-12345678"}
            ],
            endpoints=["https://myapp.example.com"],
            logs=["Deployment started", "Container built", "Deployment completed"],
            errors=[]
        )
        
        assert result.deployment_id == "deploy-123"
        assert result.status == "success"
        assert result.environment == DeploymentEnvironment.PRODUCTION
        assert result.cloud_provider == CloudProvider.AWS
        assert result.container_image == "myapp:latest"
        assert len(result.infrastructure_resources) == 2
        assert "https://myapp.example.com" in result.endpoints
        assert "Deployment completed" in result.logs
        assert len(result.errors) == 0
        assert isinstance(result.started_at, datetime)
    
    def test_deployment_validation_creation(self):
        """Test DeploymentValidation model creation."""
        validation = DeploymentValidation(
            is_valid=True,
            errors=[],
            warnings=["Consider enabling WAF for better security"],
            recommendations=[
                "Enable HTTPS-only for better security",
                "Consider increasing max instances for better availability"
            ],
            estimated_cost=150.50,
            security_score=85
        )
        
        assert validation.is_valid is True
        assert len(validation.errors) == 0
        assert len(validation.warnings) == 1
        assert "WAF" in validation.warnings[0]
        assert len(validation.recommendations) == 2
        assert "HTTPS-only" in validation.recommendations[0]
        assert validation.estimated_cost == 150.50
        assert validation.security_score == 85
    
    def test_deployment_template_creation(self):
        """Test DeploymentTemplate model creation."""
        template = DeploymentTemplate(
            id="template-123",
            name="Python Web App Template",
            description="Template for Python web applications",
            cloud_provider=CloudProvider.AWS,
            dockerfile_template="FROM python:3.11-slim\n...",
            infrastructure_template="resource \"aws_instance\" ...",
            cicd_template="name: CI/CD Pipeline\n...",
            variables={
                "app_name": {"type": "string", "description": "Application name"},
                "instance_type": {"type": "string", "default": "t3.micro"}
            },
            version="1.0.0"
        )
        
        assert template.id == "template-123"
        assert template.name == "Python Web App Template"
        assert template.cloud_provider == CloudProvider.AWS
        assert "FROM python:3.11-slim" in template.dockerfile_template
        assert "aws_instance" in template.infrastructure_template
        assert "CI/CD Pipeline" in template.cicd_template
        assert template.variables["app_name"]["type"] == "string"
        assert template.version == "1.0.0"
        assert isinstance(template.created_at, datetime)
    
    def test_cloud_provider_enum(self):
        """Test CloudProvider enum values."""
        assert CloudProvider.AWS == "aws"
        assert CloudProvider.AZURE == "azure"
        assert CloudProvider.GCP == "gcp"
        assert CloudProvider.DOCKER == "docker"
        assert CloudProvider.KUBERNETES == "kubernetes"
        
        # Test enum membership
        assert "aws" in [provider.value for provider in CloudProvider]
        assert "azure" in [provider.value for provider in CloudProvider]
    
    def test_deployment_environment_enum(self):
        """Test DeploymentEnvironment enum values."""
        assert DeploymentEnvironment.DEVELOPMENT == "development"
        assert DeploymentEnvironment.STAGING == "staging"
        assert DeploymentEnvironment.PRODUCTION == "production"
        assert DeploymentEnvironment.TEST == "test"
        
        # Test enum membership
        assert "production" in [env.value for env in DeploymentEnvironment]
        assert "staging" in [env.value for env in DeploymentEnvironment]
    
    def test_deployment_config_validation_errors(self):
        """Test DeploymentConfig validation with invalid data."""
        with pytest.raises(ValueError):
            # Invalid cloud provider
            DeploymentConfig(
                id="test-config",
                name="test",
                application_id="app-123",
                environment="invalid_environment",  # This should fail validation
                cloud_provider=CloudProvider.AWS,
                container_config=None,
                infrastructure_code=None,
                cicd_pipeline=None,
                created_by="test-user"
            )
    
    def test_deployment_result_with_errors(self):
        """Test DeploymentResult with errors."""
        result = DeploymentResult(
            deployment_id="deploy-failed-123",
            status="failed",
            environment=DeploymentEnvironment.STAGING,
            cloud_provider=CloudProvider.AZURE,
            container_image=None,
            infrastructure_resources=[],
            endpoints=[],
            logs=["Deployment started", "Error occurred"],
            errors=["Failed to build container", "Infrastructure creation failed"]
        )
        
        assert result.status == "failed"
        assert result.container_image is None
        assert len(result.infrastructure_resources) == 0
        assert len(result.endpoints) == 0
        assert len(result.errors) == 2
        assert "Failed to build container" in result.errors
        assert "Infrastructure creation failed" in result.errors
    
    def test_deployment_validation_with_errors(self):
        """Test DeploymentValidation with validation errors."""
        validation = DeploymentValidation(
            is_valid=False,
            errors=[
                "Dockerfile content is required",
                "Infrastructure template content is required",
                "CI/CD pipeline content is required"
            ],
            warnings=["No health check defined"],
            recommendations=["Add health check for better reliability"],
            estimated_cost=None,
            security_score=None
        )
        
        assert validation.is_valid is False
        assert len(validation.errors) == 3
        assert "Dockerfile content is required" in validation.errors
        assert len(validation.warnings) == 1
        assert "health check" in validation.warnings[0]
        assert len(validation.recommendations) == 1
        assert validation.estimated_cost is None
        assert validation.security_score is None


if __name__ == "__main__":
    pytest.main([__file__])