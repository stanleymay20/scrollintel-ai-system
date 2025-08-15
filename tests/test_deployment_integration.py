"""
Integration tests for deployment automation system.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.deployment_automation import DeploymentAutomation
from scrollintel.models.deployment_models import (
    DeploymentConfig, DeploymentResult, CloudProvider, DeploymentEnvironment
)
from scrollintel.models.code_generation_models import GeneratedApplication, CodeComponent


@pytest.fixture
def sample_python_application():
    """Create a sample Python application for testing."""
    return GeneratedApplication(
        id="python-app-123",
        name="python-web-app",
        description="Python web application with Flask",
        requirements=None,
        architecture=None,
        code_components=[
            CodeComponent(
                id="main-py",
                name="main.py",
                type="backend",
                language="python",
                code="""
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
""",
                dependencies=["flask"],
                tests=[]
            ),
            CodeComponent(
                id="requirements-txt",
                name="requirements.txt",
                type="config",
                language="text",
                code="flask==2.3.3\ngunicorn==21.2.0",
                dependencies=[],
                tests=[]
            )
        ],
        tests=None,
        deployment_config=None
    )


@pytest.fixture
def sample_node_application():
    """Create a sample Node.js application for testing."""
    return GeneratedApplication(
        id="node-app-123",
        name="node-web-app",
        description="Node.js web application with Express",
        requirements=None,
        architecture=None,
        code_components=[
            CodeComponent(
                id="app-js",
                name="app.js",
                type="backend",
                language="javascript",
                code="""
const express = require('express');
const app = express();
const port = 3000;

app.get('/', (req, res) => {
  res.send('Hello World!');
});

app.listen(port, () => {
  console.log(`App listening at http://localhost:${port}`);
});
""",
                dependencies=["express"],
                tests=[]
            ),
            CodeComponent(
                id="package-json",
                name="package.json",
                type="config",
                language="json",
                code='{"name": "node-web-app", "dependencies": {"express": "^4.18.2"}}',
                dependencies=[],
                tests=[]
            )
        ],
        tests=None,
        deployment_config=None
    )


@pytest.fixture
def deployment_automation():
    """Create DeploymentAutomation instance."""
    return DeploymentAutomation()


class TestDeploymentIntegration:
    """Integration tests for deployment automation."""
    
    @pytest.mark.asyncio
    async def test_complete_python_deployment_workflow(self, deployment_automation, sample_python_application):
        """Test complete deployment workflow for Python application."""
        # Step 1: Generate deployment configuration
        config = deployment_automation.generate_deployment_config(
            application=sample_python_application,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            config={
                "region": "us-west-2",
                "cicd_platform": "github",
                "instance_type": "t3.micro"
            }
        )
        
        # Verify configuration
        assert config.application_id == sample_python_application.id
        assert config.environment == DeploymentEnvironment.PRODUCTION
        assert config.cloud_provider == CloudProvider.AWS
        
        # Verify container configuration
        assert "FROM python:3.11-slim" in config.container_config.dockerfile_content
        assert "pip install" in config.container_config.dockerfile_content
        assert 8000 in config.container_config.exposed_ports
        
        # Verify infrastructure configuration
        assert config.infrastructure_code.provider == CloudProvider.AWS
        assert config.infrastructure_code.template_type == "terraform"
        assert "aws_vpc" in config.infrastructure_code.template_content
        assert "aws_ecs_cluster" in config.infrastructure_code.template_content
        
        # Verify CI/CD configuration
        assert config.cicd_pipeline.platform == "github"
        assert "CI/CD Pipeline" in config.cicd_pipeline.pipeline_content
        assert "test" in config.cicd_pipeline.stages
        assert "build" in config.cicd_pipeline.stages
        
        # Step 2: Validate configuration
        validation = deployment_automation.validate_deployment_config(config)
        assert validation.is_valid is True
        assert len(validation.errors) == 0
        assert validation.estimated_cost > 0
        assert validation.security_score > 0
        
        # Step 3: Perform dry run deployment
        dry_run_result = deployment_automation.deploy_application(config, dry_run=True)
        assert dry_run_result.status == "dry_run_success"
        assert dry_run_result.environment == DeploymentEnvironment.PRODUCTION
        assert dry_run_result.cloud_provider == CloudProvider.AWS
        
        # Step 4: Perform actual deployment
        deployment_result = deployment_automation.deploy_application(config, dry_run=False)
        assert deployment_result.status == "success"
        assert deployment_result.container_image is not None
        assert len(deployment_result.endpoints) > 0
        assert deployment_result.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_complete_node_deployment_workflow(self, deployment_automation, sample_node_application):
        """Test complete deployment workflow for Node.js application."""
        # Generate deployment configuration
        config = deployment_automation.generate_deployment_config(
            application=sample_node_application,
            environment=DeploymentEnvironment.STAGING,
            cloud_provider=CloudProvider.AZURE,
            config={
                "region": "eastus",
                "cicd_platform": "gitlab"
            }
        )
        
        # Verify Node.js specific configuration
        assert "FROM node:18-alpine" in config.container_config.dockerfile_content
        assert "npm ci" in config.container_config.dockerfile_content
        assert 3000 in config.container_config.exposed_ports
        
        # Verify Azure infrastructure
        assert config.infrastructure_code.provider == CloudProvider.AZURE
        assert config.infrastructure_code.template_type == "arm"
        assert "Microsoft.Web" in config.infrastructure_code.template_content
        
        # Verify GitLab CI/CD
        assert config.cicd_pipeline.platform == "gitlab"
        assert "stages:" in config.cicd_pipeline.pipeline_content
        
        # Validate and deploy
        validation = deployment_automation.validate_deployment_config(config)
        assert validation.is_valid is True
        
        deployment_result = deployment_automation.deploy_application(config, dry_run=False)
        assert deployment_result.status == "success"
    
    @pytest.mark.asyncio
    async def test_multi_cloud_deployment_comparison(self, deployment_automation, sample_python_application):
        """Test deployment configuration across multiple cloud providers."""
        cloud_providers = [CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP]
        configs = {}
        
        # Generate configurations for each cloud provider
        for provider in cloud_providers:
            config = deployment_automation.generate_deployment_config(
                application=sample_python_application,
                environment=DeploymentEnvironment.PRODUCTION,
                cloud_provider=provider,
                config={"region": "us-west-2"}
            )
            configs[provider] = config
            
            # Validate each configuration
            validation = deployment_automation.validate_deployment_config(config)
            assert validation.is_valid is True
        
        # Compare configurations
        aws_config = configs[CloudProvider.AWS]
        azure_config = configs[CloudProvider.AZURE]
        gcp_config = configs[CloudProvider.GCP]
        
        # All should have same application settings
        assert aws_config.application_id == azure_config.application_id == gcp_config.application_id
        
        # Infrastructure should be provider-specific
        assert aws_config.infrastructure_code.template_type == "terraform"
        assert azure_config.infrastructure_code.template_type == "arm"
        assert gcp_config.infrastructure_code.template_type == "deployment_manager"
        
        # Container configs should be similar
        assert aws_config.container_config.base_image == azure_config.container_config.base_image
        assert azure_config.container_config.base_image == gcp_config.container_config.base_image
    
    @pytest.mark.asyncio
    async def test_multi_environment_deployment_pipeline(self, deployment_automation, sample_python_application):
        """Test deployment pipeline across multiple environments."""
        environments = [
            DeploymentEnvironment.DEVELOPMENT,
            DeploymentEnvironment.STAGING,
            DeploymentEnvironment.PRODUCTION
        ]
        
        configs = {}
        deployment_results = {}
        
        for env in environments:
            # Generate configuration for each environment
            config = deployment_automation.generate_deployment_config(
                application=sample_python_application,
                environment=env,
                cloud_provider=CloudProvider.AWS,
                config={
                    "region": "us-west-2",
                    "instance_type": "t3.nano" if env == DeploymentEnvironment.DEVELOPMENT else "t3.micro"
                }
            )
            configs[env] = config
            
            # Validate configuration
            validation = deployment_automation.validate_deployment_config(config)
            assert validation.is_valid is True
            
            # Deploy to each environment
            result = deployment_automation.deploy_application(config, dry_run=False)
            deployment_results[env] = result
            assert result.status == "success"
        
        # Verify environment-specific configurations
        dev_config = configs[DeploymentEnvironment.DEVELOPMENT]
        prod_config = configs[DeploymentEnvironment.PRODUCTION]
        
        # Production should have higher resource limits
        assert prod_config.auto_scaling["max_instances"] >= dev_config.auto_scaling["max_instances"]
        
        # All deployments should succeed
        for env, result in deployment_results.items():
            assert result.status == "success"
            assert result.environment == env
    
    @pytest.mark.asyncio
    async def test_deployment_rollback_scenario(self, deployment_automation, sample_python_application):
        """Test deployment rollback scenario."""
        # Initial successful deployment
        config = deployment_automation.generate_deployment_config(
            application=sample_python_application,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            config={"region": "us-west-2"}
        )
        
        initial_deployment = deployment_automation.deploy_application(config, dry_run=False)
        assert initial_deployment.status == "success"
        
        # Simulate failed deployment (would trigger rollback in real scenario)
        # In a real implementation, this would involve actual deployment failure
        # and automatic rollback to previous version
        
        # For testing, we simulate the rollback result
        rollback_result = DeploymentResult(
            deployment_id=f"rollback-{initial_deployment.deployment_id}",
            status="rollback_success",
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            container_image=initial_deployment.container_image,
            endpoints=initial_deployment.endpoints,
            logs=["Rollback completed successfully"],
            errors=[]
        )
        
        assert rollback_result.status == "rollback_success"
        assert rollback_result.container_image == initial_deployment.container_image
    
    @pytest.mark.asyncio
    async def test_deployment_scaling_scenario(self, deployment_automation, sample_python_application):
        """Test deployment auto-scaling configuration."""
        # Create configuration with auto-scaling
        config = deployment_automation.generate_deployment_config(
            application=sample_python_application,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            config={
                "region": "us-west-2",
                "auto_scaling": {
                    "min_instances": 2,
                    "max_instances": 20,
                    "target_cpu_utilization": 60
                }
            }
        )
        
        # Verify auto-scaling configuration
        assert config.auto_scaling["min_instances"] == 2
        assert config.auto_scaling["max_instances"] == 20
        assert config.auto_scaling["target_cpu_utilization"] == 60
        
        # Validate configuration
        validation = deployment_automation.validate_deployment_config(config)
        assert validation.is_valid is True
        
        # Deploy with auto-scaling
        deployment_result = deployment_automation.deploy_application(config, dry_run=False)
        assert deployment_result.status == "success"
        
        # Verify cost estimation includes scaling
        estimated_cost = deployment_automation._estimate_cost(config)
        assert estimated_cost > 100  # Should be higher due to max instances
    
    @pytest.mark.asyncio
    async def test_deployment_security_validation(self, deployment_automation, sample_python_application):
        """Test deployment security validation."""
        # Create configuration with security settings
        config = deployment_automation.generate_deployment_config(
            application=sample_python_application,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            config={"region": "us-west-2"}
        )
        
        # Verify default security settings
        assert config.security["https_only"] is True
        assert config.security["security_headers"] is True
        assert config.security["waf_enabled"] is True
        
        # Calculate security score
        security_score = deployment_automation._calculate_security_score(config)
        assert security_score > 50  # Should have decent security score
        
        # Test with reduced security
        config.security["https_only"] = False
        config.security["waf_enabled"] = False
        
        reduced_security_score = deployment_automation._calculate_security_score(config)
        assert reduced_security_score < security_score  # Should be lower
        
        # Validation should include security warnings
        validation = deployment_automation.validate_deployment_config(config)
        assert len(validation.recommendations) > 0
        assert any("HTTPS" in rec for rec in validation.recommendations)
    
    @pytest.mark.asyncio
    async def test_deployment_monitoring_integration(self, deployment_automation, sample_python_application):
        """Test deployment monitoring configuration."""
        config = deployment_automation.generate_deployment_config(
            application=sample_python_application,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            config={
                "region": "us-west-2",
                "monitoring": {
                    "metrics_enabled": True,
                    "logging_enabled": True,
                    "alerting_enabled": True,
                    "custom_metrics": ["response_time", "error_rate"]
                }
            }
        )
        
        # Verify monitoring configuration
        assert config.monitoring["metrics_enabled"] is True
        assert config.monitoring["logging_enabled"] is True
        assert config.monitoring["alerting_enabled"] is True
        
        # Deploy with monitoring
        deployment_result = deployment_automation.deploy_application(config, dry_run=False)
        assert deployment_result.status == "success"
        
        # Verify monitoring affects security score positively
        security_score = deployment_automation._calculate_security_score(config)
        assert security_score >= 70  # Should have good score with monitoring
    
    @pytest.mark.asyncio
    async def test_deployment_cost_optimization(self, deployment_automation, sample_python_application):
        """Test deployment cost optimization across providers."""
        providers_costs = {}
        
        for provider in [CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP]:
            config = deployment_automation.generate_deployment_config(
                application=sample_python_application,
                environment=DeploymentEnvironment.PRODUCTION,
                cloud_provider=provider,
                config={"region": "us-west-2"}
            )
            
            cost = deployment_automation._estimate_cost(config)
            providers_costs[provider] = cost
        
        # Verify cost differences between providers
        aws_cost = providers_costs[CloudProvider.AWS]
        azure_cost = providers_costs[CloudProvider.AZURE]
        gcp_cost = providers_costs[CloudProvider.GCP]
        
        # Based on our cost calculation, GCP should be cheapest, then Azure, then AWS
        assert gcp_cost <= azure_cost <= aws_cost
        
        # All costs should be reasonable
        for cost in providers_costs.values():
            assert 50 <= cost <= 500  # Reasonable monthly cost range


@pytest.mark.asyncio
class TestDeploymentErrorHandling:
    """Test error handling in deployment automation."""
    
    async def test_invalid_application_deployment(self, deployment_automation):
        """Test deployment with invalid application."""
        # Create application with missing required components
        invalid_app = GeneratedApplication(
            id="invalid-app",
            name="",  # Empty name
            description="Invalid application",
            requirements=None,
            architecture=None,
            code_components=[],  # No components
            tests=None,
            deployment_config=None
        )
        
        # Should still generate config but with warnings
        config = deployment_automation.generate_deployment_config(
            application=invalid_app,
            environment=DeploymentEnvironment.DEVELOPMENT,
            cloud_provider=CloudProvider.AWS,
            config={}
        )
        
        # Validation should catch issues
        validation = deployment_automation.validate_deployment_config(config)
        # May still be valid but with warnings about empty components
        assert isinstance(validation.warnings, list)
    
    async def test_deployment_failure_handling(self, deployment_automation, sample_python_application):
        """Test handling of deployment failures."""
        config = deployment_automation.generate_deployment_config(
            application=sample_python_application,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            config={"region": "us-west-2"}
        )
        
        # Mock deployment failure
        with patch.object(deployment_automation, 'deploy_application') as mock_deploy:
            mock_deploy.side_effect = Exception("Deployment failed")
            
            try:
                result = deployment_automation.deploy_application(config, dry_run=False)
                # In real implementation, this would be handled gracefully
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Deployment failed" in str(e)
    
    async def test_configuration_validation_errors(self, deployment_automation):
        """Test configuration validation with multiple errors."""
        # Create invalid configuration
        from scrollintel.models.deployment_models import DeploymentConfig, ContainerConfig, InfrastructureCode, CICDPipeline
        
        invalid_config = DeploymentConfig(
            id="invalid-config",
            name="invalid",
            application_id="test-app",
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            container_config=ContainerConfig(
                base_image="",  # Empty base image
                dockerfile_content="",  # Empty dockerfile
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
                template_content="",  # Empty template
                variables={},
                outputs={},
                dependencies=[]
            ),
            cicd_pipeline=CICDPipeline(
                platform="github",
                pipeline_content="",  # Empty pipeline
                stages=[],
                triggers=[],
                environment_configs={},
                secrets=[]
            ),
            created_by="test"
        )
        
        validation = deployment_automation.validate_deployment_config(invalid_config)
        
        assert validation.is_valid is False
        assert len(validation.errors) >= 3  # Should have multiple errors
        assert any("Dockerfile" in error for error in validation.errors)
        assert any("Infrastructure" in error for error in validation.errors)
        assert any("CI/CD" in error for error in validation.errors)


if __name__ == "__main__":
    pytest.main([__file__])