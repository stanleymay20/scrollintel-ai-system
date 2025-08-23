"""
CI/CD Integration Tests

This module contains comprehensive tests for the CI/CD integration system,
covering provider implementations, deployment automation, and testing workflows.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from scrollintel.core.cicd_integration import (
    CICDIntegration, JenkinsProvider, GitLabCIProvider, GitHubActionsProvider
)
from scrollintel.models.cicd_models import (
    CICDProvider, DeploymentStatus, TestStatus,
    DeploymentRequest, CICDConfigurationRequest
)


class TestJenkinsProvider:
    """Test Jenkins CI/CD provider"""
    
    @pytest.fixture
    def jenkins_config(self):
        return {
            "base_url": "https://jenkins.example.com",
            "job_name": "deploy-app"
        }
    
    @pytest.fixture
    def jenkins_credentials(self):
        return {
            "username": "admin",
            "api_token": "test-token"
        }
    
    @pytest.fixture
    def jenkins_provider(self, jenkins_config, jenkins_credentials):
        return JenkinsProvider(jenkins_config, jenkins_credentials)
    
    @pytest.mark.asyncio
    async def test_trigger_pipeline_success(self, jenkins_provider):
        """Test successful pipeline trigger"""
        deployment_request = DeploymentRequest(
            pipeline_id="pipeline-1",
            version="v1.0.0",
            environment="production"
        )
        
        with patch.object(jenkins_provider.session, 'post') as mock_post:
            mock_post.return_value.status_code = 201
            mock_post.return_value.headers = {"Location": "https://jenkins.example.com/queue/item/123/"}
            
            with patch.object(jenkins_provider, '_get_build_number_from_queue', 
                            return_value="456") as mock_get_build:
                execution_id = await jenkins_provider.trigger_pipeline(
                    {"job_name": "deploy-app"},
                    deployment_request
                )
                
                assert execution_id == "deploy-app#456"
                mock_post.assert_called_once()
                mock_get_build.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_pipeline_status_running(self, jenkins_provider):
        """Test getting running pipeline status"""
        with patch.object(jenkins_provider.session, 'get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "building": True,
                "result": None
            }
            
            status = await jenkins_provider.get_pipeline_status("pipeline-1", "deploy-app#123")
            assert status == DeploymentStatus.RUNNING
    
    @pytest.mark.asyncio
    async def test_get_pipeline_status_success(self, jenkins_provider):
        """Test getting successful pipeline status"""
        with patch.object(jenkins_provider.session, 'get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "building": False,
                "result": "SUCCESS"
            }
            
            status = await jenkins_provider.get_pipeline_status("pipeline-1", "deploy-app#123")
            assert status == DeploymentStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_get_pipeline_logs(self, jenkins_provider):
        """Test getting pipeline logs"""
        with patch.object(jenkins_provider.session, 'get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.text = "Build logs here..."
            
            logs = await jenkins_provider.get_pipeline_logs("pipeline-1", "deploy-app#123")
            assert logs == "Build logs here..."
    
    @pytest.mark.asyncio
    async def test_cancel_pipeline(self, jenkins_provider):
        """Test cancelling pipeline"""
        with patch.object(jenkins_provider.session, 'post') as mock_post:
            mock_post.return_value.status_code = 200
            
            result = await jenkins_provider.cancel_pipeline("pipeline-1", "deploy-app#123")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_configuration_success(self, jenkins_provider):
        """Test successful configuration validation"""
        with patch.object(jenkins_provider.session, 'get') as mock_get:
            mock_get.return_value.status_code = 200
            
            result = await jenkins_provider.validate_configuration({})
            assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_configuration_failure(self, jenkins_provider):
        """Test failed configuration validation"""
        with patch.object(jenkins_provider.session, 'get') as mock_get:
            mock_get.return_value.status_code = 401
            
            result = await jenkins_provider.validate_configuration({})
            assert result is False


class TestGitLabCIProvider:
    """Test GitLab CI provider"""
    
    @pytest.fixture
    def gitlab_config(self):
        return {
            "base_url": "https://gitlab.com",
            "project_id": "123"
        }
    
    @pytest.fixture
    def gitlab_credentials(self):
        return {
            "access_token": "test-token"
        }
    
    @pytest.fixture
    def gitlab_provider(self, gitlab_config, gitlab_credentials):
        return GitLabCIProvider(gitlab_config, gitlab_credentials)
    
    @pytest.mark.asyncio
    async def test_trigger_pipeline_success(self, gitlab_provider):
        """Test successful GitLab pipeline trigger"""
        deployment_request = DeploymentRequest(
            pipeline_id="pipeline-1",
            version="v1.0.0",
            environment="production"
        )
        
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={"id": 456})
        
        with patch('aiohttp.ClientSession.post', return_value=mock_response):
            execution_id = await gitlab_provider.trigger_pipeline(
                {"ref": "main"},
                deployment_request
            )
            
            assert execution_id == "456"
    
    @pytest.mark.asyncio
    async def test_get_pipeline_status_success(self, gitlab_provider):
        """Test getting GitLab pipeline status"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        
        with patch('aiohttp.ClientSession.get', return_value=mock_response):
            status = await gitlab_provider.get_pipeline_status("pipeline-1", "456")
            assert status == DeploymentStatus.SUCCESS


class TestGitHubActionsProvider:
    """Test GitHub Actions provider"""
    
    @pytest.fixture
    def github_config(self):
        return {
            "owner": "example",
            "repo": "test-repo"
        }
    
    @pytest.fixture
    def github_credentials(self):
        return {
            "access_token": "test-token"
        }
    
    @pytest.fixture
    def github_provider(self, github_config, github_credentials):
        return GitHubActionsProvider(github_config, github_credentials)
    
    @pytest.mark.asyncio
    async def test_trigger_pipeline_success(self, github_provider):
        """Test successful GitHub Actions workflow trigger"""
        deployment_request = DeploymentRequest(
            pipeline_id="pipeline-1",
            version="v1.0.0",
            environment="production"
        )
        
        mock_response = AsyncMock()
        mock_response.status = 204
        
        with patch('aiohttp.ClientSession.post', return_value=mock_response):
            with patch.object(github_provider, '_get_latest_run_id', return_value="123456"):
                execution_id = await github_provider.trigger_pipeline(
                    {"workflow_id": "deploy.yml", "ref": "main"},
                    deployment_request
                )
                
                assert execution_id == "123456"
    
    @pytest.mark.asyncio
    async def test_get_pipeline_status_completed_success(self, github_provider):
        """Test getting completed successful GitHub Actions run status"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "status": "completed",
            "conclusion": "success"
        })
        
        with patch('aiohttp.ClientSession.get', return_value=mock_response):
            status = await github_provider.get_pipeline_status("pipeline-1", "123456")
            assert status == DeploymentStatus.SUCCESS


class TestCICDIntegration:
    """Test main CI/CD integration system"""
    
    @pytest.fixture
    def cicd_integration(self):
        return CICDIntegration(encryption_key="test-key-32-chars-long-12345678")
    
    @pytest.fixture
    def mock_jenkins_provider(self):
        provider = Mock(spec=JenkinsProvider)
        provider.validate_configuration = AsyncMock(return_value=True)
        provider.trigger_pipeline = AsyncMock(return_value="deploy-app#123")
        provider.get_pipeline_status = AsyncMock(return_value=DeploymentStatus.SUCCESS)
        provider.get_pipeline_logs = AsyncMock(return_value="Build successful")
        return provider
    
    @pytest.mark.asyncio
    async def test_create_cicd_configuration_success(self, cicd_integration):
        """Test successful CI/CD configuration creation"""
        config_request = {
            "name": "Test Jenkins",
            "provider": CICDProvider.JENKINS,
            "config": {"base_url": "https://jenkins.example.com"},
            "credentials": {"username": "admin", "api_token": "token"}
        }
        
        with patch.object(cicd_integration, '_create_provider_instance') as mock_create:
            mock_provider = Mock()
            mock_provider.validate_configuration = AsyncMock(return_value=True)
            mock_create.return_value = mock_provider
            
            config_id = await cicd_integration.create_cicd_configuration(config_request)
            
            assert config_id is not None
            assert len(config_id) > 0
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_cicd_configuration_invalid(self, cicd_integration):
        """Test CI/CD configuration creation with invalid config"""
        config_request = {
            "name": "Invalid Jenkins",
            "provider": CICDProvider.JENKINS,
            "config": {"base_url": "invalid-url"},
            "credentials": {"username": "admin", "api_token": "token"}
        }
        
        with patch.object(cicd_integration, '_create_provider_instance') as mock_create:
            mock_provider = Mock()
            mock_provider.validate_configuration = AsyncMock(return_value=False)
            mock_create.return_value = mock_provider
            
            with pytest.raises(Exception, match="Invalid CI/CD configuration"):
                await cicd_integration.create_cicd_configuration(config_request)
    
    @pytest.mark.asyncio
    async def test_trigger_deployment_success(self, cicd_integration, mock_jenkins_provider):
        """Test successful deployment trigger"""
        # Register mock provider
        cicd_integration.register_provider("config-1", mock_jenkins_provider)
        
        # Mock pipeline config
        with patch.object(cicd_integration, '_get_pipeline_config') as mock_get_config:
            mock_get_config.return_value = {
                "cicd_config_id": "config-1",
                "pipeline_config": {"job_name": "deploy-app"}
            }
            
            deployment_request = DeploymentRequest(
                pipeline_id="pipeline-1",
                version="v1.0.0",
                environment="production"
            )
            
            deployment_id = await cicd_integration.trigger_deployment(deployment_request)
            
            assert deployment_id is not None
            assert len(deployment_id) > 0
            mock_jenkins_provider.trigger_pipeline.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trigger_deployment_provider_not_found(self, cicd_integration):
        """Test deployment trigger with missing provider"""
        with patch.object(cicd_integration, '_get_pipeline_config') as mock_get_config:
            mock_get_config.return_value = {
                "cicd_config_id": "nonexistent-config",
                "pipeline_config": {"job_name": "deploy-app"}
            }
            
            deployment_request = DeploymentRequest(
                pipeline_id="pipeline-1",
                version="v1.0.0",
                environment="production"
            )
            
            with pytest.raises(Exception, match="CI/CD provider not found"):
                await cicd_integration.trigger_deployment(deployment_request)
    
    @pytest.mark.asyncio
    async def test_get_deployment_status_success(self, cicd_integration):
        """Test getting deployment status"""
        # Create a mock deployment
        from scrollintel.models.cicd_models import Deployment
        deployment = Deployment(
            id="deployment-1",
            pipeline_id="pipeline-1",
            version="v1.0.0",
            environment="production",
            status=DeploymentStatus.SUCCESS.value,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )
        
        cicd_integration.deployments["deployment-1"] = deployment
        
        result = await cicd_integration.get_deployment_status("deployment-1")
        
        assert result is not None
        assert result.id == "deployment-1"
        assert result.status == DeploymentStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_get_deployment_status_not_found(self, cicd_integration):
        """Test getting status for nonexistent deployment"""
        result = await cicd_integration.get_deployment_status("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_rollback_deployment_success(self, cicd_integration, mock_jenkins_provider):
        """Test successful deployment rollback"""
        # Register mock provider
        cicd_integration.register_provider("config-1", mock_jenkins_provider)
        
        # Create current deployment
        from scrollintel.models.cicd_models import Deployment
        current_deployment = Deployment(
            id="deployment-1",
            pipeline_id="pipeline-1",
            version="v2.0.0",
            environment="production",
            status=DeploymentStatus.FAILED.value,
            started_at=datetime.utcnow()
        )
        
        cicd_integration.deployments["deployment-1"] = current_deployment
        
        # Mock finding previous deployment
        previous_deployment = Deployment(
            id="deployment-0",
            pipeline_id="pipeline-1",
            version="v1.0.0",
            environment="production",
            status=DeploymentStatus.SUCCESS.value,
            started_at=datetime.utcnow()
        )
        
        with patch.object(cicd_integration, '_find_previous_deployment', 
                         return_value=previous_deployment):
            with patch.object(cicd_integration, '_get_pipeline_config') as mock_get_config:
                mock_get_config.return_value = {
                    "cicd_config_id": "config-1",
                    "pipeline_config": {"job_name": "deploy-app"}
                }
                
                rollback_id = await cicd_integration.rollback_deployment(
                    "deployment-1", 
                    "Failed deployment"
                )
                
                assert rollback_id is not None
                assert current_deployment.rollback_deployment_id == rollback_id
    
    @pytest.mark.asyncio
    async def test_rollback_deployment_not_found(self, cicd_integration):
        """Test rollback for nonexistent deployment"""
        with pytest.raises(Exception, match="Deployment not found"):
            await cicd_integration.rollback_deployment("nonexistent", "Test rollback")
    
    @pytest.mark.asyncio
    async def test_rollback_no_previous_deployment(self, cicd_integration):
        """Test rollback when no previous deployment exists"""
        # Create current deployment
        from scrollintel.models.cicd_models import Deployment
        current_deployment = Deployment(
            id="deployment-1",
            pipeline_id="pipeline-1",
            version="v1.0.0",
            environment="production",
            status=DeploymentStatus.FAILED.value,
            started_at=datetime.utcnow()
        )
        
        cicd_integration.deployments["deployment-1"] = current_deployment
        
        with patch.object(cicd_integration, '_find_previous_deployment', return_value=None):
            with pytest.raises(Exception, match="No previous deployment found for rollback"):
                await cicd_integration.rollback_deployment("deployment-1", "Test rollback")
    
    @pytest.mark.asyncio
    async def test_get_test_executions(self, cicd_integration):
        """Test getting test executions for deployment"""
        # Create test executions
        from scrollintel.models.cicd_models import TestExecution
        test1 = TestExecution(
            id="test-1",
            deployment_id="deployment-1",
            test_suite="unit-tests",
            test_type="unit",
            status=TestStatus.PASSED.value,
            started_at=datetime.utcnow()
        )
        
        test2 = TestExecution(
            id="test-2",
            deployment_id="deployment-1",
            test_suite="integration-tests",
            test_type="integration",
            status=TestStatus.PASSED.value,
            started_at=datetime.utcnow()
        )
        
        cicd_integration.test_executions["test-1"] = test1
        cicd_integration.test_executions["test-2"] = test2
        
        executions = await cicd_integration.get_test_executions("deployment-1")
        
        assert len(executions) == 2
        assert executions[0]["id"] == "test-1"
        assert executions[1]["id"] == "test-2"
    
    def test_encrypt_decrypt_credentials(self, cicd_integration):
        """Test credential encryption and decryption"""
        credentials = {
            "username": "admin",
            "password": "secret",
            "api_token": "token123"
        }
        
        encrypted = cicd_integration._encrypt_credentials(credentials)
        decrypted = cicd_integration._decrypt_credentials(encrypted)
        
        assert decrypted == credentials
    
    @pytest.mark.asyncio
    async def test_run_health_check_success(self, cicd_integration):
        """Test successful health check execution"""
        from scrollintel.models.cicd_models import Deployment
        deployment = Deployment(
            id="deployment-1",
            pipeline_id="pipeline-1",
            version="v1.0.0",
            environment="production",
            status=DeploymentStatus.SUCCESS.value,
            started_at=datetime.utcnow()
        )
        
        config = {"health_url": "https://api.example.com/health"}
        
        mock_response = AsyncMock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession.get', return_value=mock_response):
            result = await cicd_integration._run_health_check(deployment, config)
            
            assert result["success"] is True
            assert result["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_run_health_check_failure(self, cicd_integration):
        """Test failed health check execution"""
        from scrollintel.models.cicd_models import Deployment
        deployment = Deployment(
            id="deployment-1",
            pipeline_id="pipeline-1",
            version="v1.0.0",
            environment="production",
            status=DeploymentStatus.SUCCESS.value,
            started_at=datetime.utcnow()
        )
        
        config = {"health_url": "https://api.example.com/health"}
        
        mock_response = AsyncMock()
        mock_response.status = 500
        
        with patch('aiohttp.ClientSession.get', return_value=mock_response):
            result = await cicd_integration._run_health_check(deployment, config)
            
            assert result["success"] is False
            assert result["status_code"] == 500


class TestCICDModels:
    """Test CI/CD data models"""
    
    def test_cicd_configuration_request_validation(self):
        """Test CI/CD configuration request validation"""
        from scrollintel.models.cicd_models import CICDConfigurationRequest
        
        request = CICDConfigurationRequest(
            name="Test Config",
            provider=CICDProvider.JENKINS,
            config={"base_url": "https://jenkins.example.com"},
            credentials={"username": "admin", "api_token": "token"}
        )
        
        assert request.name == "Test Config"
        assert request.provider == CICDProvider.JENKINS
        assert request.config["base_url"] == "https://jenkins.example.com"
    
    def test_deployment_request_validation(self):
        """Test deployment request validation"""
        from scrollintel.models.cicd_models import DeploymentRequest
        
        request = DeploymentRequest(
            pipeline_id="pipeline-1",
            version="v1.0.0",
            environment="production",
            metadata={"build_number": 123}
        )
        
        assert request.pipeline_id == "pipeline-1"
        assert request.version == "v1.0.0"
        assert request.environment == "production"
        assert request.metadata["build_number"] == 123
    
    def test_deployment_status_enum(self):
        """Test deployment status enum values"""
        assert DeploymentStatus.PENDING.value == "pending"
        assert DeploymentStatus.RUNNING.value == "running"
        assert DeploymentStatus.SUCCESS.value == "success"
        assert DeploymentStatus.FAILED.value == "failed"
        assert DeploymentStatus.CANCELLED.value == "cancelled"
        assert DeploymentStatus.ROLLBACK.value == "rollback"
    
    def test_test_status_enum(self):
        """Test test status enum values"""
        assert TestStatus.PENDING.value == "pending"
        assert TestStatus.RUNNING.value == "running"
        assert TestStatus.PASSED.value == "passed"
        assert TestStatus.FAILED.value == "failed"
        assert TestStatus.SKIPPED.value == "skipped"


if __name__ == "__main__":
    pytest.main([__file__])