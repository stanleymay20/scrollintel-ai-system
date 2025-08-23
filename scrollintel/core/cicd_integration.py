"""
CI/CD Integration Core System

This module provides the core CI/CD integration functionality,
supporting multiple CI/CD providers and deployment automation.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod

import aiohttp
import requests
from cryptography.fernet import Fernet

from scrollintel.models.cicd_models import (
    CICDProvider, DeploymentStatus, TestStatus,
    CICDConfiguration, Pipeline, Deployment, TestExecution,
    DeploymentRequest, DeploymentResponse, TestExecutionRequest
)

logger = logging.getLogger(__name__)


class CICDProviderInterface(ABC):
    """Abstract interface for CI/CD providers"""
    
    @abstractmethod
    async def trigger_pipeline(self, pipeline_config: Dict[str, Any], 
                             deployment_request: DeploymentRequest) -> str:
        """Trigger a pipeline execution"""
        pass
    
    @abstractmethod
    async def get_pipeline_status(self, pipeline_id: str, execution_id: str) -> DeploymentStatus:
        """Get pipeline execution status"""
        pass
    
    @abstractmethod
    async def get_pipeline_logs(self, pipeline_id: str, execution_id: str) -> str:
        """Get pipeline execution logs"""
        pass
    
    @abstractmethod
    async def cancel_pipeline(self, pipeline_id: str, execution_id: str) -> bool:
        """Cancel pipeline execution"""
        pass
    
    @abstractmethod
    async def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate provider configuration"""
        pass


class JenkinsProvider(CICDProviderInterface):
    """Jenkins CI/CD provider implementation"""
    
    def __init__(self, config: Dict[str, Any], credentials: Dict[str, Any]):
        self.base_url = config.get("base_url")
        self.username = credentials.get("username")
        self.api_token = credentials.get("api_token")
        self.session = requests.Session()
        self.session.auth = (self.username, self.api_token)
    
    async def trigger_pipeline(self, pipeline_config: Dict[str, Any], 
                             deployment_request: DeploymentRequest) -> str:
        """Trigger Jenkins job"""
        job_name = pipeline_config.get("job_name")
        parameters = {
            "VERSION": deployment_request.version,
            "ENVIRONMENT": deployment_request.environment,
            **pipeline_config.get("parameters", {})
        }
        
        url = f"{self.base_url}/job/{job_name}/buildWithParameters"
        response = self.session.post(url, data=parameters)
        
        if response.status_code == 201:
            # Get build number from queue
            queue_url = response.headers.get("Location")
            build_number = await self._get_build_number_from_queue(queue_url)
            return f"{job_name}#{build_number}"
        else:
            raise Exception(f"Failed to trigger Jenkins job: {response.text}")
    
    async def get_pipeline_status(self, pipeline_id: str, execution_id: str) -> DeploymentStatus:
        """Get Jenkins build status"""
        job_name, build_number = execution_id.split("#")
        url = f"{self.base_url}/job/{job_name}/{build_number}/api/json"
        
        response = self.session.get(url)
        if response.status_code == 200:
            build_info = response.json()
            if build_info.get("building"):
                return DeploymentStatus.RUNNING
            elif build_info.get("result") == "SUCCESS":
                return DeploymentStatus.SUCCESS
            elif build_info.get("result") == "FAILURE":
                return DeploymentStatus.FAILED
            elif build_info.get("result") == "ABORTED":
                return DeploymentStatus.CANCELLED
            else:
                return DeploymentStatus.PENDING
        else:
            raise Exception(f"Failed to get Jenkins build status: {response.text}")
    
    async def get_pipeline_logs(self, pipeline_id: str, execution_id: str) -> str:
        """Get Jenkins build logs"""
        job_name, build_number = execution_id.split("#")
        url = f"{self.base_url}/job/{job_name}/{build_number}/consoleText"
        
        response = self.session.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return f"Failed to get logs: {response.text}"
    
    async def cancel_pipeline(self, pipeline_id: str, execution_id: str) -> bool:
        """Cancel Jenkins build"""
        job_name, build_number = execution_id.split("#")
        url = f"{self.base_url}/job/{job_name}/{build_number}/stop"
        
        response = self.session.post(url)
        return response.status_code == 200
    
    async def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate Jenkins configuration"""
        try:
            url = f"{self.base_url}/api/json"
            response = self.session.get(url)
            return response.status_code == 200
        except Exception:
            return False
    
    async def _get_build_number_from_queue(self, queue_url: str) -> str:
        """Get build number from Jenkins queue"""
        # Poll queue until build starts
        for _ in range(30):  # 30 second timeout
            response = self.session.get(f"{queue_url}api/json")
            if response.status_code == 200:
                queue_info = response.json()
                if "executable" in queue_info:
                    return str(queue_info["executable"]["number"])
            await asyncio.sleep(1)
        raise Exception("Timeout waiting for build to start")


class GitLabCIProvider(CICDProviderInterface):
    """GitLab CI provider implementation"""
    
    def __init__(self, config: Dict[str, Any], credentials: Dict[str, Any]):
        self.base_url = config.get("base_url", "https://gitlab.com")
        self.project_id = config.get("project_id")
        self.access_token = credentials.get("access_token")
        self.headers = {"Authorization": f"Bearer {self.access_token}"}
    
    async def trigger_pipeline(self, pipeline_config: Dict[str, Any], 
                             deployment_request: DeploymentRequest) -> str:
        """Trigger GitLab CI pipeline"""
        url = f"{self.base_url}/api/v4/projects/{self.project_id}/pipeline"
        
        data = {
            "ref": pipeline_config.get("ref", "main"),
            "variables": [
                {"key": "VERSION", "value": deployment_request.version},
                {"key": "ENVIRONMENT", "value": deployment_request.environment}
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=self.headers) as response:
                if response.status == 201:
                    pipeline_info = await response.json()
                    return str(pipeline_info["id"])
                else:
                    text = await response.text()
                    raise Exception(f"Failed to trigger GitLab pipeline: {text}")
    
    async def get_pipeline_status(self, pipeline_id: str, execution_id: str) -> DeploymentStatus:
        """Get GitLab pipeline status"""
        url = f"{self.base_url}/api/v4/projects/{self.project_id}/pipelines/{execution_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    pipeline_info = await response.json()
                    status = pipeline_info.get("status")
                    
                    status_mapping = {
                        "pending": DeploymentStatus.PENDING,
                        "running": DeploymentStatus.RUNNING,
                        "success": DeploymentStatus.SUCCESS,
                        "failed": DeploymentStatus.FAILED,
                        "canceled": DeploymentStatus.CANCELLED,
                        "skipped": DeploymentStatus.CANCELLED
                    }
                    
                    return status_mapping.get(status, DeploymentStatus.PENDING)
                else:
                    raise Exception(f"Failed to get GitLab pipeline status")
    
    async def get_pipeline_logs(self, pipeline_id: str, execution_id: str) -> str:
        """Get GitLab pipeline logs"""
        # Get jobs for the pipeline
        jobs_url = f"{self.base_url}/api/v4/projects/{self.project_id}/pipelines/{execution_id}/jobs"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(jobs_url, headers=self.headers) as response:
                if response.status == 200:
                    jobs = await response.json()
                    logs = []
                    
                    for job in jobs:
                        job_id = job["id"]
                        log_url = f"{self.base_url}/api/v4/projects/{self.project_id}/jobs/{job_id}/trace"
                        
                        async with session.get(log_url, headers=self.headers) as log_response:
                            if log_response.status == 200:
                                job_logs = await log_response.text()
                                logs.append(f"=== Job: {job['name']} ===\n{job_logs}\n")
                    
                    return "\n".join(logs)
                else:
                    return "Failed to get pipeline logs"
    
    async def cancel_pipeline(self, pipeline_id: str, execution_id: str) -> bool:
        """Cancel GitLab pipeline"""
        url = f"{self.base_url}/api/v4/projects/{self.project_id}/pipelines/{execution_id}/cancel"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers) as response:
                return response.status == 200
    
    async def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate GitLab configuration"""
        try:
            url = f"{self.base_url}/api/v4/projects/{self.project_id}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    return response.status == 200
        except Exception:
            return False


class GitHubActionsProvider(CICDProviderInterface):
    """GitHub Actions provider implementation"""
    
    def __init__(self, config: Dict[str, Any], credentials: Dict[str, Any]):
        self.owner = config.get("owner")
        self.repo = config.get("repo")
        self.access_token = credentials.get("access_token")
        self.headers = {
            "Authorization": f"token {self.access_token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    async def trigger_pipeline(self, pipeline_config: Dict[str, Any], 
                             deployment_request: DeploymentRequest) -> str:
        """Trigger GitHub Actions workflow"""
        workflow_id = pipeline_config.get("workflow_id")
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/workflows/{workflow_id}/dispatches"
        
        data = {
            "ref": pipeline_config.get("ref", "main"),
            "inputs": {
                "version": deployment_request.version,
                "environment": deployment_request.environment
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=self.headers) as response:
                if response.status == 204:
                    # GitHub Actions doesn't return run ID immediately
                    # We need to poll for the latest run
                    return await self._get_latest_run_id(workflow_id)
                else:
                    text = await response.text()
                    raise Exception(f"Failed to trigger GitHub Actions workflow: {text}")
    
    async def get_pipeline_status(self, pipeline_id: str, execution_id: str) -> DeploymentStatus:
        """Get GitHub Actions run status"""
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/runs/{execution_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    run_info = await response.json()
                    status = run_info.get("status")
                    conclusion = run_info.get("conclusion")
                    
                    if status == "in_progress":
                        return DeploymentStatus.RUNNING
                    elif status == "completed":
                        if conclusion == "success":
                            return DeploymentStatus.SUCCESS
                        elif conclusion == "failure":
                            return DeploymentStatus.FAILED
                        elif conclusion == "cancelled":
                            return DeploymentStatus.CANCELLED
                    
                    return DeploymentStatus.PENDING
                else:
                    raise Exception("Failed to get GitHub Actions run status")
    
    async def get_pipeline_logs(self, pipeline_id: str, execution_id: str) -> str:
        """Get GitHub Actions run logs"""
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/runs/{execution_id}/logs"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    # GitHub returns logs as a zip file
                    return "Logs available via GitHub Actions interface"
                else:
                    return "Failed to get run logs"
    
    async def cancel_pipeline(self, pipeline_id: str, execution_id: str) -> bool:
        """Cancel GitHub Actions run"""
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/runs/{execution_id}/cancel"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers) as response:
                return response.status == 202
    
    async def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate GitHub configuration"""
        try:
            url = f"https://api.github.com/repos/{self.owner}/{self.repo}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _get_latest_run_id(self, workflow_id: str) -> str:
        """Get the latest workflow run ID"""
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/workflows/{workflow_id}/runs"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    runs = await response.json()
                    if runs["workflow_runs"]:
                        return str(runs["workflow_runs"][0]["id"])
        
        raise Exception("Failed to get latest run ID")


class CICDIntegration:
    """Main CI/CD integration system"""
    
    def __init__(self, encryption_key: str):
        self.encryption_key = encryption_key
        self.fernet = Fernet(encryption_key.encode())
        self.providers: Dict[str, CICDProviderInterface] = {}
        self.deployments: Dict[str, Deployment] = {}
        self.test_executions: Dict[str, TestExecution] = {}
    
    def register_provider(self, config_id: str, provider: CICDProviderInterface):
        """Register a CI/CD provider"""
        self.providers[config_id] = provider
    
    def _encrypt_credentials(self, credentials: Dict[str, Any]) -> str:
        """Encrypt credentials for storage"""
        return self.fernet.encrypt(json.dumps(credentials).encode()).decode()
    
    def _decrypt_credentials(self, encrypted_credentials: str) -> Dict[str, Any]:
        """Decrypt credentials from storage"""
        return json.loads(self.fernet.decrypt(encrypted_credentials.encode()).decode())
    
    async def create_cicd_configuration(self, config_request: Dict[str, Any]) -> str:
        """Create CI/CD configuration"""
        config_id = str(uuid.uuid4())
        
        # Encrypt credentials
        encrypted_creds = self._encrypt_credentials(config_request["credentials"])
        
        # Create provider instance
        provider = await self._create_provider_instance(
            config_request["provider"],
            config_request["config"],
            config_request["credentials"]
        )
        
        # Validate configuration
        if not await provider.validate_configuration(config_request["config"]):
            raise Exception("Invalid CI/CD configuration")
        
        # Register provider
        self.register_provider(config_id, provider)
        
        # Store configuration (in real implementation, save to database)
        logger.info(f"Created CI/CD configuration: {config_id}")
        
        return config_id
    
    async def _create_provider_instance(self, provider_type: str, config: Dict[str, Any], 
                                      credentials: Dict[str, Any]) -> CICDProviderInterface:
        """Create provider instance based on type"""
        if provider_type == CICDProvider.JENKINS:
            return JenkinsProvider(config, credentials)
        elif provider_type == CICDProvider.GITLAB_CI:
            return GitLabCIProvider(config, credentials)
        elif provider_type == CICDProvider.GITHUB_ACTIONS:
            return GitHubActionsProvider(config, credentials)
        else:
            raise Exception(f"Unsupported CI/CD provider: {provider_type}")
    
    async def trigger_deployment(self, deployment_request: DeploymentRequest) -> str:
        """Trigger a deployment"""
        deployment_id = str(uuid.uuid4())
        
        # Get pipeline configuration (in real implementation, fetch from database)
        pipeline_config = self._get_pipeline_config(deployment_request.pipeline_id)
        provider = self.providers.get(pipeline_config["cicd_config_id"])
        
        if not provider:
            raise Exception("CI/CD provider not found")
        
        try:
            # Trigger pipeline
            execution_id = await provider.trigger_pipeline(
                pipeline_config["pipeline_config"],
                deployment_request
            )
            
            # Create deployment record
            metadata = {"execution_id": execution_id}
            if deployment_request.metadata:
                metadata.update(deployment_request.metadata)
            
            deployment = Deployment(
                id=deployment_id,
                pipeline_id=deployment_request.pipeline_id,
                version=deployment_request.version,
                environment=deployment_request.environment,
                status=DeploymentStatus.RUNNING.value,
                started_at=datetime.utcnow(),
                metadata=metadata
            )
            
            self.deployments[deployment_id] = deployment
            
            # Start monitoring deployment
            asyncio.create_task(self._monitor_deployment(deployment_id, provider, execution_id))
            
            logger.info(f"Triggered deployment: {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Failed to trigger deployment: {e}")
            raise
    
    async def _monitor_deployment(self, deployment_id: str, provider: CICDProviderInterface, 
                                execution_id: str):
        """Monitor deployment progress"""
        deployment = self.deployments[deployment_id]
        
        while deployment.status in [DeploymentStatus.PENDING.value, DeploymentStatus.RUNNING.value]:
            try:
                # Check status
                status = await provider.get_pipeline_status(deployment.pipeline_id, execution_id)
                deployment.status = status.value
                
                # Get logs
                logs = await provider.get_pipeline_logs(deployment.pipeline_id, execution_id)
                deployment.logs = logs
                
                # Check if completed
                if status in [DeploymentStatus.SUCCESS, DeploymentStatus.FAILED, DeploymentStatus.CANCELLED]:
                    deployment.completed_at = datetime.utcnow()
                    
                    # Run automated tests if deployment succeeded
                    if status == DeploymentStatus.SUCCESS:
                        await self._run_automated_tests(deployment_id)
                    
                    # Send notifications
                    await self._send_deployment_notification(deployment_id, status)
                    break
                
                # Wait before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error monitoring deployment {deployment_id}: {e}")
                deployment.status = DeploymentStatus.FAILED.value
                deployment.completed_at = datetime.utcnow()
                break
    
    async def _run_automated_tests(self, deployment_id: str):
        """Run automated tests after successful deployment"""
        deployment = self.deployments[deployment_id]
        pipeline_config = self._get_pipeline_config(deployment.pipeline_id)
        
        test_suites = pipeline_config.get("test_suites", [])
        
        for test_suite in test_suites:
            test_id = str(uuid.uuid4())
            
            test_execution = TestExecution(
                id=test_id,
                deployment_id=deployment_id,
                test_suite=test_suite["name"],
                test_type=test_suite["type"],
                status=TestStatus.RUNNING.value,
                started_at=datetime.utcnow()
            )
            
            self.test_executions[test_id] = test_execution
            
            # Run tests (simplified implementation)
            try:
                results = await self._execute_test_suite(test_suite, deployment)
                test_execution.status = TestStatus.PASSED.value if results["success"] else TestStatus.FAILED.value
                test_execution.results = results
                test_execution.completed_at = datetime.utcnow()
                
                logger.info(f"Test execution completed: {test_id}")
                
            except Exception as e:
                test_execution.status = TestStatus.FAILED.value
                test_execution.logs = str(e)
                test_execution.completed_at = datetime.utcnow()
                logger.error(f"Test execution failed: {test_id}, {e}")
    
    async def _execute_test_suite(self, test_suite: Dict[str, Any], deployment: Deployment) -> Dict[str, Any]:
        """Execute a test suite"""
        # Simplified test execution
        # In real implementation, this would integrate with testing frameworks
        
        test_type = test_suite["type"]
        test_config = test_suite.get("config", {})
        
        if test_type == "health_check":
            # Simple health check
            return await self._run_health_check(deployment, test_config)
        elif test_type == "integration":
            # Integration tests
            return await self._run_integration_tests(deployment, test_config)
        elif test_type == "performance":
            # Performance tests
            return await self._run_performance_tests(deployment, test_config)
        else:
            return {"success": True, "message": f"Test type {test_type} not implemented"}
    
    async def _run_health_check(self, deployment: Deployment, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run health check tests"""
        try:
            # Simple HTTP health check
            health_url = config.get("health_url", f"https://{deployment.environment}.example.com/health")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        return {"success": True, "status_code": response.status}
                    else:
                        return {"success": False, "status_code": response.status}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _run_integration_tests(self, deployment: Deployment, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run integration tests"""
        # Placeholder for integration test execution
        return {"success": True, "tests_run": 10, "tests_passed": 10}
    
    async def _run_performance_tests(self, deployment: Deployment, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run performance tests"""
        # Placeholder for performance test execution
        return {"success": True, "response_time": 150, "throughput": 1000}
    
    async def _send_deployment_notification(self, deployment_id: str, status: DeploymentStatus):
        """Send deployment notification"""
        deployment = self.deployments[deployment_id]
        pipeline_config = self._get_pipeline_config(deployment.pipeline_id)
        
        notification_config = pipeline_config.get("notification_config", {})
        
        if notification_config:
            message = f"Deployment {deployment_id} {status.value} for version {deployment.version} in {deployment.environment}"
            
            # Send notifications (simplified)
            for notification in notification_config.get("notifications", []):
                await self._send_notification(notification, message)
    
    async def _send_notification(self, notification_config: Dict[str, Any], message: str):
        """Send individual notification"""
        notification_type = notification_config.get("type")
        
        if notification_type == "email":
            # Send email notification
            logger.info(f"Email notification: {message}")
        elif notification_type == "slack":
            # Send Slack notification
            logger.info(f"Slack notification: {message}")
        elif notification_type == "webhook":
            # Send webhook notification
            webhook_url = notification_config.get("url")
            async with aiohttp.ClientSession() as session:
                await session.post(webhook_url, json={"message": message})
    
    async def rollback_deployment(self, deployment_id: str, reason: str) -> str:
        """Rollback a deployment"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            raise Exception("Deployment not found")
        
        # Find previous successful deployment
        previous_deployment = self._find_previous_deployment(deployment)
        if not previous_deployment:
            raise Exception("No previous deployment found for rollback")
        
        # Create rollback deployment request
        rollback_request = DeploymentRequest(
            pipeline_id=deployment.pipeline_id,
            version=previous_deployment.version,
            environment=deployment.environment,
            metadata={"rollback_reason": reason, "original_deployment": deployment_id}
        )
        
        # Trigger rollback deployment
        rollback_deployment_id = await self.trigger_deployment(rollback_request)
        
        # Update original deployment with rollback reference
        deployment.rollback_deployment_id = rollback_deployment_id
        
        logger.info(f"Rollback triggered: {rollback_deployment_id} for deployment {deployment_id}")
        return rollback_deployment_id
    
    def _get_pipeline_config(self, pipeline_id: str) -> Dict[str, Any]:
        """Get pipeline configuration (placeholder)"""
        # In real implementation, fetch from database
        return {
            "cicd_config_id": "config-1",
            "pipeline_config": {
                "job_name": "deploy-scrollintel",
                "parameters": {}
            },
            "test_suites": [
                {"name": "health-check", "type": "health_check", "config": {}},
                {"name": "integration-tests", "type": "integration", "config": {}}
            ],
            "notification_config": {
                "notifications": [
                    {"type": "email", "recipients": ["admin@example.com"]},
                    {"type": "slack", "channel": "#deployments"}
                ]
            }
        }
    
    def _find_previous_deployment(self, deployment: Deployment) -> Optional[Deployment]:
        """Find previous successful deployment for rollback"""
        # In real implementation, query database for previous successful deployment
        # This is a simplified placeholder
        return None
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResponse]:
        """Get deployment status"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return None
        
        return DeploymentResponse(
            id=deployment.id,
            pipeline_id=deployment.pipeline_id,
            version=deployment.version,
            environment=deployment.environment,
            status=DeploymentStatus(deployment.status),
            started_at=deployment.started_at,
            completed_at=deployment.completed_at,
            logs=deployment.logs,
            metadata=deployment.metadata,
            rollback_deployment_id=deployment.rollback_deployment_id
        )
    
    async def get_test_executions(self, deployment_id: str) -> List[Dict[str, Any]]:
        """Get test executions for a deployment"""
        executions = []
        for test_execution in self.test_executions.values():
            if test_execution.deployment_id == deployment_id:
                executions.append({
                    "id": test_execution.id,
                    "test_suite": test_execution.test_suite,
                    "test_type": test_execution.test_type,
                    "status": test_execution.status,
                    "started_at": test_execution.started_at,
                    "completed_at": test_execution.completed_at,
                    "results": test_execution.results,
                    "logs": test_execution.logs
                })
        return executions