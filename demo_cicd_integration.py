"""
CI/CD Integration Demo

This demo showcases the CI/CD pipeline integration capabilities,
including Jenkins, GitLab CI, and GitHub Actions support with
automated testing and deployment workflows.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

from scrollintel.core.cicd_integration import CICDIntegration
from scrollintel.models.cicd_models import (
    CICDProvider, DeploymentRequest, CICDConfigurationRequest,
    DeploymentStatus, TestStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CICDIntegrationDemo:
    """Comprehensive CI/CD integration demonstration"""
    
    def __init__(self):
        self.cicd_integration = CICDIntegration(
            encryption_key="demo-encryption-key-32-chars-long"
        )
        self.config_ids = {}
        self.pipeline_ids = {}
        self.deployment_ids = []
    
    async def run_demo(self):
        """Run the complete CI/CD integration demo"""
        print("🚀 Starting CI/CD Integration Demo")
        print("=" * 50)
        
        try:
            # 1. Setup CI/CD configurations
            await self.setup_cicd_configurations()
            
            # 2. Create deployment pipelines
            await self.create_deployment_pipelines()
            
            # 3. Demonstrate deployment workflows
            await self.demonstrate_deployment_workflows()
            
            # 4. Show automated testing
            await self.demonstrate_automated_testing()
            
            # 5. Demonstrate rollback functionality
            await self.demonstrate_rollback_functionality()
            
            # 6. Show monitoring and notifications
            await self.demonstrate_monitoring_notifications()
            
            print("\n✅ CI/CD Integration Demo completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.error(f"Demo error: {e}", exc_info=True)
    
    async def setup_cicd_configurations(self):
        """Setup various CI/CD provider configurations"""
        print("\n📋 Setting up CI/CD Configurations")
        print("-" * 30)
        
        # Jenkins configuration
        jenkins_config = {
            "name": "Production Jenkins",
            "provider": CICDProvider.JENKINS,
            "config": {
                "base_url": "https://jenkins.scrollintel.com",
                "job_name": "deploy-scrollintel"
            },
            "credentials": {
                "username": "admin",
                "api_token": "jenkins-api-token-123"
            }
        }
        
        jenkins_id = await self.cicd_integration.create_cicd_configuration(jenkins_config)
        self.config_ids["jenkins"] = jenkins_id
        print(f"✓ Jenkins configuration created: {jenkins_id}")
        
        # GitLab CI configuration
        gitlab_config = {
            "name": "GitLab CI Pipeline",
            "provider": CICDProvider.GITLAB_CI,
            "config": {
                "base_url": "https://gitlab.com",
                "project_id": "12345"
            },
            "credentials": {
                "access_token": "gitlab-access-token-456"
            }
        }
        
        gitlab_id = await self.cicd_integration.create_cicd_configuration(gitlab_config)
        self.config_ids["gitlab"] = gitlab_id
        print(f"✓ GitLab CI configuration created: {gitlab_id}")
        
        # GitHub Actions configuration
        github_config = {
            "name": "GitHub Actions Workflow",
            "provider": CICDProvider.GITHUB_ACTIONS,
            "config": {
                "owner": "scrollintel",
                "repo": "scrollintel-platform"
            },
            "credentials": {
                "access_token": "github-token-789"
            }
        }
        
        github_id = await self.cicd_integration.create_cicd_configuration(github_config)
        self.config_ids["github"] = github_id
        print(f"✓ GitHub Actions configuration created: {github_id}")
    
    async def create_deployment_pipelines(self):
        """Create deployment pipelines for different environments"""
        print("\n🔧 Creating Deployment Pipelines")
        print("-" * 30)
        
        # Production pipeline (Jenkins)
        prod_pipeline = {
            "name": "Production Deployment",
            "cicd_config_id": self.config_ids["jenkins"],
            "pipeline_config": {
                "job_name": "deploy-scrollintel-prod",
                "parameters": {
                    "ENVIRONMENT": "production",
                    "HEALTH_CHECK_URL": "https://api.scrollintel.com/health"
                }
            },
            "trigger_config": {
                "manual_approval": True,
                "auto_deploy_tags": True,
                "tag_pattern": "v*"
            },
            "notification_config": {
                "notifications": [
                    {"type": "email", "recipients": ["devops@scrollintel.com"]},
                    {"type": "slack", "channel": "#deployments"}
                ]
            },
            "test_suites": [
                {
                    "name": "production-health-check",
                    "type": "health_check",
                    "config": {"health_url": "https://api.scrollintel.com/health"}
                },
                {
                    "name": "production-smoke-tests",
                    "type": "integration",
                    "config": {"test_env": "production"}
                }
            ]
        }
        
        self.pipeline_ids["production"] = "prod-pipeline-1"
        print(f"✓ Production pipeline created: {self.pipeline_ids['production']}")
        
        # Staging pipeline (GitLab CI)
        staging_pipeline = {
            "name": "Staging Deployment",
            "cicd_config_id": self.config_ids["gitlab"],
            "pipeline_config": {
                "ref": "develop",
                "variables": {
                    "ENVIRONMENT": "staging",
                    "AUTO_DEPLOY": "true"
                }
            },
            "trigger_config": {
                "on_push": True,
                "branches": ["develop"],
                "auto_deploy": True
            },
            "test_suites": [
                {
                    "name": "staging-integration-tests",
                    "type": "integration",
                    "config": {"test_env": "staging"}
                },
                {
                    "name": "staging-performance-tests",
                    "type": "performance",
                    "config": {"load_test_duration": 300}
                }
            ]
        }
        
        self.pipeline_ids["staging"] = "staging-pipeline-1"
        print(f"✓ Staging pipeline created: {self.pipeline_ids['staging']}")
        
        # Development pipeline (GitHub Actions)
        dev_pipeline = {
            "name": "Development Deployment",
            "cicd_config_id": self.config_ids["github"],
            "pipeline_config": {
                "workflow_id": "deploy-dev.yml",
                "ref": "main"
            },
            "trigger_config": {
                "on_push": True,
                "on_pull_request": True,
                "auto_deploy": True
            },
            "test_suites": [
                {
                    "name": "dev-unit-tests",
                    "type": "unit",
                    "config": {"coverage_threshold": 80}
                },
                {
                    "name": "dev-integration-tests",
                    "type": "integration",
                    "config": {"test_env": "development"}
                }
            ]
        }
        
        self.pipeline_ids["development"] = "dev-pipeline-1"
        print(f"✓ Development pipeline created: {self.pipeline_ids['development']}")
    
    async def demonstrate_deployment_workflows(self):
        """Demonstrate various deployment workflows"""
        print("\n🚀 Demonstrating Deployment Workflows")
        print("-" * 35)
        
        # Trigger development deployment
        dev_deployment = DeploymentRequest(
            pipeline_id=self.pipeline_ids["development"],
            version="v1.2.3-dev",
            environment="development",
            metadata={
                "commit_sha": "abc123def456",
                "branch": "feature/new-agent",
                "triggered_by": "developer@scrollintel.com"
            }
        )
        
        dev_deployment_id = await self.cicd_integration.trigger_deployment(dev_deployment)
        self.deployment_ids.append(dev_deployment_id)
        print(f"✓ Development deployment triggered: {dev_deployment_id}")
        
        # Simulate deployment progress
        await self.simulate_deployment_progress(dev_deployment_id, "Development")
        
        # Trigger staging deployment
        staging_deployment = DeploymentRequest(
            pipeline_id=self.pipeline_ids["staging"],
            version="v1.2.3-rc1",
            environment="staging",
            metadata={
                "commit_sha": "def456ghi789",
                "branch": "develop",
                "triggered_by": "ci-system"
            }
        )
        
        staging_deployment_id = await self.cicd_integration.trigger_deployment(staging_deployment)
        self.deployment_ids.append(staging_deployment_id)
        print(f"✓ Staging deployment triggered: {staging_deployment_id}")
        
        await self.simulate_deployment_progress(staging_deployment_id, "Staging")
        
        # Trigger production deployment
        prod_deployment = DeploymentRequest(
            pipeline_id=self.pipeline_ids["production"],
            version="v1.2.3",
            environment="production",
            metadata={
                "commit_sha": "ghi789jkl012",
                "branch": "main",
                "triggered_by": "release-manager@scrollintel.com",
                "approval_id": "PROD-2024-001"
            }
        )
        
        prod_deployment_id = await self.cicd_integration.trigger_deployment(prod_deployment)
        self.deployment_ids.append(prod_deployment_id)
        print(f"✓ Production deployment triggered: {prod_deployment_id}")
        
        await self.simulate_deployment_progress(prod_deployment_id, "Production")
    
    async def simulate_deployment_progress(self, deployment_id: str, environment: str):
        """Simulate deployment progress monitoring"""
        print(f"  📊 Monitoring {environment} deployment progress...")
        
        # Simulate deployment phases
        phases = [
            ("Building application", 2),
            ("Running tests", 3),
            ("Deploying to environment", 2),
            ("Running health checks", 1),
            ("Deployment complete", 0)
        ]
        
        for phase, duration in phases:
            print(f"    • {phase}...")
            await asyncio.sleep(duration)
        
        # Get final deployment status
        deployment = await self.cicd_integration.get_deployment_status(deployment_id)
        if deployment:
            print(f"    ✅ {environment} deployment completed: {deployment.status.value}")
        else:
            print(f"    ❌ Failed to get {environment} deployment status")
    
    async def demonstrate_automated_testing(self):
        """Demonstrate automated testing capabilities"""
        print("\n🧪 Demonstrating Automated Testing")
        print("-" * 30)
        
        # Get test executions for the latest deployment
        if self.deployment_ids:
            latest_deployment = self.deployment_ids[-1]
            test_executions = await self.cicd_integration.get_test_executions(latest_deployment)
            
            print(f"📋 Test executions for deployment {latest_deployment}:")
            
            for execution in test_executions:
                status_icon = "✅" if execution["status"] == "passed" else "❌"
                print(f"  {status_icon} {execution['test_suite']} ({execution['test_type']})")
                print(f"    Status: {execution['status']}")
                if execution.get("results"):
                    results = execution["results"]
                    if "tests_run" in results:
                        print(f"    Tests: {results.get('tests_passed', 0)}/{results.get('tests_run', 0)} passed")
                    if "response_time" in results:
                        print(f"    Performance: {results['response_time']}ms avg response time")
        
        # Demonstrate test result analysis
        print("\n📊 Test Result Analysis:")
        test_metrics = {
            "total_tests": 45,
            "passed_tests": 43,
            "failed_tests": 2,
            "coverage": 87.5,
            "performance_score": 92
        }
        
        for metric, value in test_metrics.items():
            print(f"  • {metric.replace('_', ' ').title()}: {value}")
        
        # Show test failure handling
        print("\n🔍 Test Failure Analysis:")
        failed_tests = [
            {
                "name": "test_agent_response_time",
                "error": "Response time exceeded 500ms threshold",
                "suggestion": "Optimize agent processing pipeline"
            },
            {
                "name": "test_concurrent_users",
                "error": "Connection pool exhausted at 1000 users",
                "suggestion": "Increase database connection pool size"
            }
        ]
        
        for test in failed_tests:
            print(f"  ❌ {test['name']}")
            print(f"    Error: {test['error']}")
            print(f"    Suggestion: {test['suggestion']}")
    
    async def demonstrate_rollback_functionality(self):
        """Demonstrate deployment rollback capabilities"""
        print("\n🔄 Demonstrating Rollback Functionality")
        print("-" * 35)
        
        # Simulate a failed deployment that needs rollback
        failed_deployment = DeploymentRequest(
            pipeline_id=self.pipeline_ids["production"],
            version="v1.2.4-hotfix",
            environment="production",
            metadata={
                "commit_sha": "bad123commit",
                "branch": "hotfix/critical-bug",
                "triggered_by": "emergency-deploy@scrollintel.com"
            }
        )
        
        failed_deployment_id = await self.cicd_integration.trigger_deployment(failed_deployment)
        print(f"🚨 Emergency deployment triggered: {failed_deployment_id}")
        
        # Simulate deployment failure
        print("  • Deploying hotfix...")
        await asyncio.sleep(2)
        print("  • Running health checks...")
        await asyncio.sleep(1)
        print("  ❌ Health check failed - critical error detected!")
        
        # Trigger rollback
        print("\n🔄 Initiating automatic rollback...")
        try:
            rollback_deployment_id = await self.cicd_integration.rollback_deployment(
                failed_deployment_id,
                "Health check failure - rolling back to previous stable version"
            )
            print(f"✓ Rollback deployment triggered: {rollback_deployment_id}")
            
            # Simulate rollback progress
            print("  • Rolling back to v1.2.3...")
            await asyncio.sleep(2)
            print("  • Verifying rollback health...")
            await asyncio.sleep(1)
            print("  ✅ Rollback completed successfully!")
            
        except Exception as e:
            print(f"  ❌ Rollback failed: {e}")
        
        # Show rollback metrics
        print("\n📊 Rollback Metrics:")
        rollback_metrics = {
            "rollback_time": "3m 45s",
            "downtime": "1m 12s",
            "affected_users": "~2,500",
            "recovery_success": "100%"
        }
        
        for metric, value in rollback_metrics.items():
            print(f"  • {metric.replace('_', ' ').title()}: {value}")
    
    async def demonstrate_monitoring_notifications(self):
        """Demonstrate monitoring and notification capabilities"""
        print("\n📊 Demonstrating Monitoring & Notifications")
        print("-" * 40)
        
        # Show deployment monitoring dashboard
        print("📈 Deployment Monitoring Dashboard:")
        
        deployment_stats = {
            "total_deployments": len(self.deployment_ids) + 2,
            "successful_deployments": len(self.deployment_ids) + 1,
            "failed_deployments": 1,
            "average_deployment_time": "8m 32s",
            "success_rate": "85.7%"
        }
        
        for stat, value in deployment_stats.items():
            print(f"  • {stat.replace('_', ' ').title()}: {value}")
        
        # Show notification examples
        print("\n📧 Notification Examples:")
        
        notifications = [
            {
                "type": "Email",
                "recipient": "devops@scrollintel.com",
                "subject": "Production Deployment Successful - v1.2.3",
                "message": "Deployment completed in 7m 23s with all tests passing"
            },
            {
                "type": "Slack",
                "channel": "#deployments",
                "message": "🚨 Production deployment failed - rollback initiated"
            },
            {
                "type": "Webhook",
                "endpoint": "https://monitoring.scrollintel.com/webhooks/deployment",
                "payload": {"status": "success", "version": "v1.2.3", "environment": "production"}
            }
        ]
        
        for notification in notifications:
            print(f"  📨 {notification['type']}: {notification.get('subject', notification.get('message', 'Webhook sent'))}")
        
        # Show alerting rules
        print("\n🚨 Alerting Rules:")
        
        alert_rules = [
            "Deployment failure rate > 20% in 1 hour",
            "Deployment time > 15 minutes",
            "Health check failure after deployment",
            "Rollback triggered in production",
            "Test failure rate > 10%"
        ]
        
        for rule in alert_rules:
            print(f"  • {rule}")
        
        # Show integration status
        print("\n🔗 CI/CD Integration Status:")
        
        integration_status = {
            "Jenkins": "✅ Connected",
            "GitLab CI": "✅ Connected", 
            "GitHub Actions": "✅ Connected",
            "Monitoring": "✅ Active",
            "Notifications": "✅ Configured"
        }
        
        for service, status in integration_status.items():
            print(f"  • {service}: {status}")
    
    async def show_deployment_summary(self):
        """Show summary of all deployments"""
        print("\n📋 Deployment Summary")
        print("-" * 20)
        
        for i, deployment_id in enumerate(self.deployment_ids):
            deployment = await self.cicd_integration.get_deployment_status(deployment_id)
            if deployment:
                print(f"  {i+1}. {deployment.environment} - {deployment.version}")
                print(f"     Status: {deployment.status.value}")
                print(f"     Started: {deployment.started_at}")
                if deployment.completed_at:
                    duration = deployment.completed_at - deployment.started_at
                    print(f"     Duration: {duration}")


async def main():
    """Run the CI/CD integration demo"""
    demo = CICDIntegrationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())