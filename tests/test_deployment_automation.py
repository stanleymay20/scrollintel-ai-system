"""
Tests for deployment automation functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.deployment_automation import (
    DeploymentAutomation, ContainerBuilder, InfrastructureGenerator, CICDGenerator
)
from scrollintel.models.deployment_models import (
    DeploymentConfig, ContainerConfig, InfrastructureCode, CICDPipeline,
    DeploymentResult, DeploymentValidation, CloudProvider, DeploymentEnvironment
)
from scrollintel.models.code_generation_models import GeneratedApplication, CodeComponent


@pytest.fixture
def sample_application():
    """Create a sample application for testing."""
    return GeneratedApplication(
        id="test-app-123",
        name="test-application",
        description="Test application for deployment",
        requirements=None,
        architecture=None,
        code_components=[
            CodeComponent(
                id="comp-1",
                name="main.py",
                type="backend",
                language="python",
                code="print('Hello World')",
                dependencies=[],
                tests=[]
            )
        ],
        tests=None,
        deployment_config=None
    )


@pytest.fixture
def container_builder():
    """Create ContainerBuilder instance."""
    return ContainerBuilder()


@pytest.fixture
def infrastructure_generator():
    """Create InfrastructureGenerator instance."""
    return InfrastructureGenerator()


@pytest.fixture
def cicd_generator():
    """Create CICDGenerator instance."""
    return CICDGenerator()


@pytest.fixture
def deployment_automation():
    """Create DeploymentAutomation instance."""
    return DeploymentAutomation()


class TestContainerBuilder:
    """Test ContainerBuilder functionality."""
    
    def test_generate_dockerfile_python(self, container_builder, sample_application):
        """Test Dockerfile generation for Python application."""
        dockerfile = container_builder.generate_dockerfile(sample_application)
        
        assert "FROM python:3.11-slim" in dockerfile
        assert "WORKDIR /app" in dockerfile
        assert "COPY requirements.txt" in dockerfile
        assert "RUN pip install" in dockerfile
        assert "HEALTHCHECK" in dockerfile
        assert "CMD" in dockerfile
    
    def test_build_container_config(self, container_builder, sample_application):
        """Test container configuration building."""
        config = container_builder.build_container_config(sample_application)
        
        assert isinstance(config, ContainerConfig)
        assert config.base_image == "python:3.11-slim"
        assert config.dockerfile_content is not None
        assert 8000 in config.exposed_ports
        assert config.health_check is not None
        assert config.resource_limits is not None
    
    def test_detect_language(self, container_builder, sample_application):
        """Test language detection."""
        language = container_builder._detect_language(sample_application)
        assert language == "python"
    
    def test_generate_node_dockerfile_content(self, container_builder, sample_application):
        """Test Node.js specific Dockerfile content."""
        content = container_builder._generate_node_dockerfile_content(sample_application)
        
        assert "package*.json" in content
        assert "npm ci" in content
        assert "nodejs" in content
        assert "EXPOSE 3000" in content


class TestInfrastructureGenerator:
    """Test InfrastructureGenerator functionality."""
    
    def test_generate_terraform_aws(self, infrastructure_generator, sample_application):
        """Test Terraform AWS template generation."""
        config = {"environment": "production", "region": "us-west-2"}
        terraform_code = infrastructure_generator.generate_terraform_aws(sample_application, config)
        
        assert "terraform {" in terraform_code
        assert "provider \"aws\"" in terraform_code
        assert "resource \"aws_vpc\"" in terraform_code
        assert "resource \"aws_ecs_cluster\"" in terraform_code
        assert "resource \"aws_lb\"" in terraform_code
        assert sample_application.name in terraform_code
    
    def test_generate_cloudformation_aws(self, infrastructure_generator, sample_application):
        """Test CloudFormation template generation."""
        config = {"environment": "production"}
        cf_template = infrastructure_generator.generate_cloudformation_aws(sample_application, config)
        
        assert "AWSTemplateFormatVersion" in cf_template
        assert "AWS::EC2::VPC" in cf_template
        assert "AWS::EC2::InternetGateway" in cf_template
        assert sample_application.name in cf_template
    
    def test_generate_azure_arm(self, infrastructure_generator, sample_application):
        """Test Azure ARM template generation."""
        config = {"environment": "production"}
        arm_template = infrastructure_generator.generate_azure_arm(sample_application, config)
        
        assert "schema.management.azure.com" in arm_template
        assert "Microsoft.Web/serverfarms" in arm_template
        assert "Microsoft.Web/sites" in arm_template
        assert sample_application.name in arm_template
    
    def test_generate_gcp_deployment_manager(self, infrastructure_generator, sample_application):
        """Test GCP Deployment Manager template generation."""
        config = {"environment": "production"}
        gcp_template = infrastructure_generator.generate_gcp_deployment_manager(sample_application, config)
        
        assert "resources:" in gcp_template
        assert "compute.v1.network" in gcp_template
        assert "compute.v1.instanceTemplate" in gcp_template
        assert sample_application.name in gcp_template


class TestCICDGenerator:
    """Test CICDGenerator functionality."""
    
    def test_generate_github_actions(self, cicd_generator, sample_application):
        """Test GitHub Actions workflow generation."""
        config = {"environment": "production"}
        workflow = cicd_generator.generate_github_actions(sample_application, config)
        
        assert "name: CI/CD Pipeline" in workflow
        assert "on:" in workflow
        assert "jobs:" in workflow
        assert "test:" in workflow
        assert "build-and-push:" in workflow
        assert "deploy-production:" in workflow
        assert sample_application.name in workflow
    
    def test_generate_gitlab_ci(self, cicd_generator, sample_application):
        """Test GitLab CI configuration generation."""
        config = {"environment": "production"}
        gitlab_ci = cicd_generator.generate_gitlab_ci(sample_application, config)
        
        assert "stages:" in gitlab_ci
        assert "test:" in gitlab_ci
        assert "build:" in gitlab_ci
        assert "deploy-production:" in gitlab_ci
        assert sample_application.name in gitlab_ci
    
    def test_generate_jenkins_pipeline(self, cicd_generator, sample_application):
        """Test Jenkins pipeline generation."""
        config = {"environment": "production"}
        jenkins_pipeline = cicd_generator.generate_jenkins_pipeline(sample_application, config)
        
        assert "pipeline {" in jenkins_pipeline
        assert "stages {" in jenkins_pipeline
        assert "stage('Test')" in jenkins_pipeline
        assert "stage('Build')" in jenkins_pipeline
        assert "stage('Deploy to Production')" in jenkins_pipeline
        assert sample_application.name in jenkins_pipeline


class TestDeploymentAutomation:
    """Test DeploymentAutomation functionality."""
    
    def test_generate_deployment_config(self, deployment_automation, sample_application):
        """Test deployment configuration generation."""
        config = deployment_automation.generate_deployment_config(
            application=sample_application,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            config={"region": "us-west-2"}
        )
        
        assert isinstance(config, DeploymentConfig)
        assert config.application_id == sample_application.id
        assert config.environment == DeploymentEnvironment.PRODUCTION
        assert config.cloud_provider == CloudProvider.AWS
        assert config.container_config is not None
        assert config.infrastructure_code is not None
        assert config.cicd_pipeline is not None
    
    def test_validate_deployment_config_valid(self, deployment_automation, sample_application):
        """Test deployment configuration validation - valid config."""
        config = deployment_automation.generate_deployment_config(
            application=sample_application,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            config={"region": "us-west-2"}
        )
        
        validation = deployment_automation.validate_deployment_config(config)
        
        assert isinstance(validation, DeploymentValidation)
        assert validation.is_valid is True
        assert len(validation.errors) == 0
        assert validation.estimated_cost is not None
        assert validation.security_score is not None
    
    def test_validate_deployment_config_invalid(self, deployment_automation):
        """Test deployment configuration validation - invalid config."""
        # Create invalid config with missing required fields
        config = DeploymentConfig(
            id="test-config",
            name="test",
            application_id="test-app",
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            container_config=ContainerConfig(
                base_image="python:3.11",
                dockerfile_content="",  # Empty content should cause error
                build_args={},
                environment_vars={},
                exposed_ports=[],
                volumes=[],
                health_check=None,
                resource_limits={}
            ),
            infrastructure_code=InfrastructureCode(
                provider=CloudProvider.AWS,
                template_type="terraform",
                template_content="",  # Empty content should cause error
                variables={},
                outputs={},
                dependencies=[]
            ),
            cicd_pipeline=CICDPipeline(
                platform="github",
                pipeline_content="",  # Empty content should cause error
                stages=[],
                triggers=[],
                environment_configs={},
                secrets=[]
            ),
            created_by="test"
        )
        
        validation = deployment_automation.validate_deployment_config(config)
        
        assert validation.is_valid is False
        assert len(validation.errors) > 0
        assert "Dockerfile content is required" in validation.errors
        assert "Infrastructure template content is required" in validation.errors
        assert "CI/CD pipeline content is required" in validation.errors
    
    def test_deploy_application_dry_run(self, deployment_automation, sample_application):
        """Test application deployment - dry run."""
        config = deployment_automation.generate_deployment_config(
            application=sample_application,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            config={"region": "us-west-2"}
        )
        
        result = deployment_automation.deploy_application(config, dry_run=True)
        
        assert isinstance(result, DeploymentResult)
        assert result.status == "dry_run_success"
        assert result.environment == DeploymentEnvironment.PRODUCTION
        assert result.cloud_provider == CloudProvider.AWS
        assert "Dry run completed successfully" in result.logs
    
    def test_deploy_application_actual(self, deployment_automation, sample_application):
        """Test application deployment - actual deployment."""
        config = deployment_automation.generate_deployment_config(
            application=sample_application,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            config={"region": "us-west-2"}
        )
        
        result = deployment_automation.deploy_application(config, dry_run=False)
        
        assert isinstance(result, DeploymentResult)
        assert result.status == "success"
        assert result.container_image is not None
        assert len(result.endpoints) > 0
        assert "Deployment completed successfully" in result.logs
    
    def test_estimate_cost(self, deployment_automation, sample_application):
        """Test cost estimation."""
        config = deployment_automation.generate_deployment_config(
            application=sample_application,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            config={"region": "us-west-2"}
        )
        
        cost = deployment_automation._estimate_cost(config)
        
        assert isinstance(cost, float)
        assert cost > 0
    
    def test_calculate_security_score(self, deployment_automation, sample_application):
        """Test security score calculation."""
        config = deployment_automation.generate_deployment_config(
            application=sample_application,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            config={"region": "us-west-2"}
        )
        
        score = deployment_automation._calculate_security_score(config)
        
        assert isinstance(score, int)
        assert 0 <= score <= 100
    
    def test_generate_infrastructure_code_aws(self, deployment_automation, sample_application):
        """Test infrastructure code generation for AWS."""
        infrastructure_code = deployment_automation._generate_infrastructure_code(
            application=sample_application,
            cloud_provider=CloudProvider.AWS,
            config={"region": "us-west-2"}
        )
        
        assert isinstance(infrastructure_code, InfrastructureCode)
        assert infrastructure_code.provider == CloudProvider.AWS
        assert infrastructure_code.template_type == "terraform"
        assert infrastructure_code.template_content is not None
        assert len(infrastructure_code.template_content) > 0
    
    def test_generate_infrastructure_code_azure(self, deployment_automation, sample_application):
        """Test infrastructure code generation for Azure."""
        infrastructure_code = deployment_automation._generate_infrastructure_code(
            application=sample_application,
            cloud_provider=CloudProvider.AZURE,
            config={"region": "eastus"}
        )
        
        assert isinstance(infrastructure_code, InfrastructureCode)
        assert infrastructure_code.provider == CloudProvider.AZURE
        assert infrastructure_code.template_type == "arm"
        assert infrastructure_code.template_content is not None
    
    def test_generate_cicd_pipeline_github(self, deployment_automation, sample_application):
        """Test CI/CD pipeline generation for GitHub."""
        cicd_pipeline = deployment_automation._generate_cicd_pipeline(
            application=sample_application,
            config={"cicd_platform": "github"}
        )
        
        assert isinstance(cicd_pipeline, CICDPipeline)
        assert cicd_pipeline.platform == "github"
        assert cicd_pipeline.pipeline_content is not None
        assert "test" in cicd_pipeline.stages
        assert "build" in cicd_pipeline.stages
        assert "deploy" in cicd_pipeline.stages
    
    def test_generate_cicd_pipeline_gitlab(self, deployment_automation, sample_application):
        """Test CI/CD pipeline generation for GitLab."""
        cicd_pipeline = deployment_automation._generate_cicd_pipeline(
            application=sample_application,
            config={"cicd_platform": "gitlab"}
        )
        
        assert isinstance(cicd_pipeline, CICDPipeline)
        assert cicd_pipeline.platform == "gitlab"
        assert cicd_pipeline.pipeline_content is not None


@pytest.mark.asyncio
class TestDeploymentIntegration:
    """Integration tests for deployment automation."""
    
    async def test_end_to_end_deployment_workflow(self, deployment_automation, sample_application):
        """Test complete end-to-end deployment workflow."""
        # Generate deployment configuration
        config = deployment_automation.generate_deployment_config(
            application=sample_application,
            environment=DeploymentEnvironment.STAGING,
            cloud_provider=CloudProvider.AWS,
            config={
                "region": "us-west-2",
                "cicd_platform": "github"
            }
        )
        
        # Validate configuration
        validation = deployment_automation.validate_deployment_config(config)
        assert validation.is_valid is True
        
        # Perform dry run deployment
        dry_run_result = deployment_automation.deploy_application(config, dry_run=True)
        assert dry_run_result.status == "dry_run_success"
        
        # Perform actual deployment
        deployment_result = deployment_automation.deploy_application(config, dry_run=False)
        assert deployment_result.status == "success"
        assert deployment_result.container_image is not None
        assert len(deployment_result.endpoints) > 0
    
    async def test_multi_cloud_deployment(self, deployment_automation, sample_application):
        """Test deployment configuration for multiple cloud providers."""
        cloud_providers = [CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP]
        
        for provider in cloud_providers:
            config = deployment_automation.generate_deployment_config(
                application=sample_application,
                environment=DeploymentEnvironment.PRODUCTION,
                cloud_provider=provider,
                config={"region": "us-west-2"}
            )
            
            assert config.cloud_provider == provider
            assert config.infrastructure_code.provider == provider
            
            validation = deployment_automation.validate_deployment_config(config)
            assert validation.is_valid is True
    
    async def test_multi_environment_deployment(self, deployment_automation, sample_application):
        """Test deployment configuration for multiple environments."""
        environments = [
            DeploymentEnvironment.DEVELOPMENT,
            DeploymentEnvironment.STAGING,
            DeploymentEnvironment.PRODUCTION
        ]
        
        for env in environments:
            config = deployment_automation.generate_deployment_config(
                application=sample_application,
                environment=env,
                cloud_provider=CloudProvider.AWS,
                config={"region": "us-west-2"}
            )
            
            assert config.environment == env
            
            validation = deployment_automation.validate_deployment_config(config)
            assert validation.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__])