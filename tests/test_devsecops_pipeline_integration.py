"""
Comprehensive tests for DevSecOps Pipeline Integration
Tests all components of the DevSecOps pipeline integration system
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from security.devsecops.devsecops_orchestrator import DevSecOpsOrchestrator
from security.devsecops.pipeline_security_gates import (
    PipelineSecurityGates, SecurityGateType, SecurityGateStatus
)
from security.devsecops.container_vulnerability_scanner import (
    ContainerVulnerabilityScanner, VulnerabilitySeverity
)
from security.devsecops.deployment_strategies import (
    SecureDeploymentStrategies, DeploymentStrategy, DeploymentConfig
)
from security.devsecops.infrastructure_change_workflows import (
    InfrastructureChangeWorkflows, ChangeType, ChangeRisk
)
from security.devsecops.security_policy_enforcement import (
    SecurityPolicyEnforcement, PolicyType, PolicySeverity
)
from security.devsecops.automated_rollback_system import (
    AutomatedRollbackSystem, RollbackTrigger, IncidentSeverity
)

class TestPipelineSecurityGates:
    """Test pipeline security gates functionality"""
    
    @pytest.fixture
    def security_gates(self):
        config = {}
        return PipelineSecurityGates(config)
    
    @pytest.mark.asyncio
    async def test_execute_sast_scan(self, security_gates):
        """Test SAST scan execution"""
        pipeline_context = {
            "pipeline_id": "test-pipeline-001",
            "application_name": "test-app",
            "git_commit": "abc123"
        }
        
        result = await security_gates.execute_security_gate(
            SecurityGateType.SAST_SCAN, pipeline_context
        )
        
        assert result.gate_type == SecurityGateType.SAST_SCAN
        assert result.status in [SecurityGateStatus.APPROVED, SecurityGateStatus.PENDING]
        assert isinstance(result.score, float)
        assert isinstance(result.findings, list)
        assert isinstance(result.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_execute_dast_scan(self, security_gates):
        """Test DAST scan execution"""
        pipeline_context = {
            "pipeline_id": "test-pipeline-002",
            "application_name": "test-app",
            "app_url": "https://test-app.example.com"
        }
        
        result = await security_gates.execute_security_gate(
            SecurityGateType.DAST_SCAN, pipeline_context
        )
        
        assert result.gate_type == SecurityGateType.DAST_SCAN
        assert result.status in [SecurityGateStatus.APPROVED, SecurityGateStatus.PENDING]
        assert result.execution_time >= 0
    
    @pytest.mark.asyncio
    async def test_container_scan(self, security_gates):
        """Test container vulnerability scan"""
        pipeline_context = {
            "pipeline_id": "test-pipeline-003",
            "container_image": "test-app:latest"
        }
        
        result = await security_gates.execute_security_gate(
            SecurityGateType.CONTAINER_SCAN, pipeline_context
        )
        
        assert result.gate_type == SecurityGateType.CONTAINER_SCAN
        assert isinstance(result.findings, list)
    
    @pytest.mark.asyncio
    async def test_approval_workflow_creation(self, security_gates):
        """Test approval workflow creation"""
        pipeline_context = {
            "pipeline_id": "test-pipeline-004",
            "author": "test-user"
        }
        
        # Execute a gate that requires approval
        gate_result = await security_gates.execute_security_gate(
            SecurityGateType.SECURITY_REVIEW, pipeline_context
        )
        
        # Create approval workflow
        workflow = await security_gates.create_approval_workflow(
            gate_result, pipeline_context
        )
        
        assert workflow.workflow_id is not None
        assert len(workflow.required_approvers) > 0
        assert workflow.approval_threshold > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_security_status(self, security_gates):
        """Test getting pipeline security status"""
        pipeline_id = "test-pipeline-005"
        
        # Execute multiple gates
        pipeline_context = {"pipeline_id": pipeline_id}
        
        await security_gates.execute_security_gate(
            SecurityGateType.SAST_SCAN, pipeline_context
        )
        await security_gates.execute_security_gate(
            SecurityGateType.DEPENDENCY_SCAN, pipeline_context
        )
        
        # Get overall status
        status = await security_gates.get_pipeline_security_status(pipeline_id)
        
        assert "status" in status
        assert "overall_score" in status
        assert "gate_results" in status

class TestContainerVulnerabilityScanner:
    """Test container vulnerability scanner functionality"""
    
    @pytest.fixture
    def vulnerability_scanner(self):
        config = {}
        return ContainerVulnerabilityScanner(config)
    
    @pytest.mark.asyncio
    async def test_scan_image_basic(self, vulnerability_scanner):
        """Test basic image scanning"""
        with patch('docker.from_env') as mock_docker:
            mock_client = Mock()
            mock_docker.return_value = mock_client
            
            # Mock image exists
            mock_client.images.get.return_value = Mock()
            
            # Mock subprocess for trivy
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = Mock()
                mock_process.returncode = 0
                mock_process.communicate.return_value = (
                    b'{"Results": [{"Vulnerabilities": []}]}', b''
                )
                mock_subprocess.return_value = mock_process
                
                result = await vulnerability_scanner.scan_image("test-app", "latest")
                
                assert result.image_name == "test-app"
                assert result.image_tag == "latest"
                assert isinstance(result.vulnerabilities, list)
                assert isinstance(result.misconfigurations, list)
                assert isinstance(result.security_score, float)
    
    @pytest.mark.asyncio
    async def test_scan_with_vulnerabilities(self, vulnerability_scanner):
        """Test scanning image with vulnerabilities"""
        with patch('docker.from_env') as mock_docker:
            mock_client = Mock()
            mock_docker.return_value = mock_client
            mock_client.images.get.return_value = Mock()
            
            # Mock trivy output with vulnerabilities
            trivy_output = {
                "Results": [{
                    "Vulnerabilities": [{
                        "VulnerabilityID": "CVE-2021-1234",
                        "Severity": "HIGH",
                        "PkgName": "openssl",
                        "InstalledVersion": "1.1.1f",
                        "FixedVersion": "1.1.1g",
                        "Description": "Test vulnerability",
                        "CVSS": {"nvd": {"V3Score": 7.5}},
                        "References": ["https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-1234"]
                    }]
                }]
            }
            
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = Mock()
                mock_process.returncode = 0
                mock_process.communicate.return_value = (
                    json.dumps(trivy_output).encode(), b''
                )
                mock_subprocess.return_value = mock_process
                
                result = await vulnerability_scanner.scan_image("vulnerable-app", "latest")
                
                assert len(result.vulnerabilities) > 0
                assert result.vulnerabilities[0].cve_id == "CVE-2021-1234"
                assert result.vulnerabilities[0].severity == VulnerabilitySeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_misconfiguration_detection(self, vulnerability_scanner):
        """Test container misconfiguration detection"""
        with patch('docker.from_env') as mock_docker:
            mock_client = Mock()
            mock_docker.return_value = mock_client
            
            # Mock image with misconfigurations
            mock_image = Mock()
            mock_image.attrs = {
                "Config": {
                    "User": "",  # Running as root
                    "Env": ["PASSWORD=secret123"],  # Exposed secret
                    "ExposedPorts": {"22/tcp": {}}  # SSH port exposed
                }
            }
            mock_image.history.return_value = [
                {"CreatedBy": "/bin/sh -c USER root"}
            ]
            mock_client.images.get.return_value = mock_image
            
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = Mock()
                mock_process.returncode = 0
                mock_process.communicate.return_value = (
                    b'{"Results": []}', b''
                )
                mock_subprocess.return_value = mock_process
                
                result = await vulnerability_scanner.scan_image("misconfigured-app", "latest")
                
                assert len(result.misconfigurations) > 0
                # Should detect root user and exposed secrets
                misconfig_types = [m.check_id for m in result.misconfigurations]
                assert "USER_ROOT" in misconfig_types or "EXPOSED_SECRETS" in misconfig_types

class TestSecureDeploymentStrategies:
    """Test secure deployment strategies"""
    
    @pytest.fixture
    def deployment_strategies(self):
        config = {}
        with patch('kubernetes.config.load_incluster_config'):
            with patch('kubernetes.config.load_kube_config'):
                return SecureDeploymentStrategies(config)
    
    @pytest.mark.asyncio
    async def test_blue_green_deployment(self, deployment_strategies):
        """Test blue-green deployment strategy"""
        deployment_config = DeploymentConfig(
            strategy=DeploymentStrategy.BLUE_GREEN,
            namespace="test",
            service_name="test-service",
            image="test-app:v2.0.0",
            replicas=3,
            security_validations=["runtime_security_scan"],
            rollback_threshold=80.0,
            validation_timeout=300
        )
        
        with patch.object(deployment_strategies, '_create_green_deployment', return_value="test-service-green"):
            with patch.object(deployment_strategies, '_wait_for_deployment_ready'):
                with patch.object(deployment_strategies, '_run_security_validations', return_value=[]):
                    with patch.object(deployment_strategies, '_switch_traffic_to_green'):
                        with patch.object(deployment_strategies, '_monitor_post_deployment', return_value={"error_rate": 0.01}):
                            with patch.object(deployment_strategies, '_cleanup_blue_deployment'):
                                
                                result = await deployment_strategies.deploy_blue_green(deployment_config)
                                
                                assert result.deployment_id is not None
                                assert result.strategy == DeploymentStrategy.BLUE_GREEN
                                assert result.status.value in ["completed", "rolled_back"]
    
    @pytest.mark.asyncio
    async def test_canary_deployment(self, deployment_strategies):
        """Test canary deployment strategy"""
        deployment_config = DeploymentConfig(
            strategy=DeploymentStrategy.CANARY,
            namespace="test",
            service_name="test-service",
            image="test-app:v2.0.0",
            replicas=3,
            security_validations=["runtime_security_scan"],
            rollback_threshold=80.0,
            validation_timeout=300,
            canary_config={
                "initial_traffic": 10,
                "increment_traffic": 20,
                "max_traffic": 100,
                "validation_interval": 60
            }
        )
        
        with patch.object(deployment_strategies, '_create_canary_deployment', return_value="test-service-canary"):
            with patch.object(deployment_strategies, '_wait_for_deployment_ready'):
                with patch.object(deployment_strategies, '_run_security_validations', return_value=[]):
                    with patch.object(deployment_strategies, '_update_traffic_split'):
                        with patch.object(deployment_strategies, '_collect_canary_metrics', return_value={"error_rate": 0.01}):
                            with patch.object(deployment_strategies, '_complete_canary_deployment'):
                                
                                result = await deployment_strategies.deploy_canary(deployment_config)
                                
                                assert result.deployment_id is not None
                                assert result.strategy == DeploymentStrategy.CANARY

class TestInfrastructureChangeWorkflows:
    """Test infrastructure change workflows"""
    
    @pytest.fixture
    def change_workflows(self):
        config = {}
        return InfrastructureChangeWorkflows(config)
    
    @pytest.mark.asyncio
    async def test_submit_change_request(self, change_workflows):
        """Test submitting infrastructure change request"""
        change_data = {
            "title": "Update security group rules",
            "description": "Add new ingress rules for API access",
            "change_type": "network",
            "files_changed": ["terraform/security-groups.tf"],
            "implementation_plan": {"steps": ["terraform plan", "terraform apply"]},
            "rollback_plan": {"steps": ["terraform destroy"]},
            "testing_plan": {"steps": ["connectivity test"]}
        }
        
        change = await change_workflows.submit_change_request(change_data, "test-user")
        
        assert change.change_id is not None
        assert change.title == change_data["title"]
        assert change.change_type == ChangeType.NETWORK
        assert change.submitter == "test-user"
        assert len(change.required_approvers) > 0
    
    @pytest.mark.asyncio
    async def test_security_assessment(self, change_workflows):
        """Test security assessment of infrastructure change"""
        change_data = {
            "title": "Deploy new application",
            "description": "Deploy new microservice",
            "change_type": "infrastructure",
            "files_changed": ["k8s/deployment.yaml", "terraform/rds.tf"],
            "implementation_plan": {"steps": ["deploy"]},
            "rollback_plan": {"steps": ["rollback"]},
            "testing_plan": {"steps": ["test"]}
        }
        
        change = await change_workflows.submit_change_request(change_data, "test-user")
        
        # Security assessment should be triggered automatically
        assert change.security_assessment is not None
        assert change.security_assessment.risk_score >= 0
        assert isinstance(change.security_assessment.security_findings, list)
    
    @pytest.mark.asyncio
    async def test_approval_workflow(self, change_workflows):
        """Test change approval workflow"""
        from security.devsecops.infrastructure_change_workflows import ApprovalStatus
        
        change_data = {
            "title": "Minor configuration update",
            "description": "Update application config",
            "change_type": "configuration",
            "files_changed": ["config/app.yaml"],
            "implementation_plan": {"steps": ["update config"]},
            "rollback_plan": {"steps": ["revert config"]},
            "testing_plan": {"steps": ["test config"]}
        }
        
        change = await change_workflows.submit_change_request(change_data, "test-user")
        
        # Submit approval
        success = await change_workflows.submit_approval(
            change.change_id,
            change.required_approvers[0],
            ApprovalStatus.APPROVED,
            "Looks good to me"
        )
        
        assert success is True
        assert len(change.approvals) > 0
        assert change.approvals[0].status == ApprovalStatus.APPROVED

class TestSecurityPolicyEnforcement:
    """Test security policy enforcement"""
    
    @pytest.fixture
    def policy_enforcement(self):
        config = {}
        return SecurityPolicyEnforcement(config)
    
    @pytest.mark.asyncio
    async def test_create_policy(self, policy_enforcement):
        """Test creating security policy"""
        policy_data = {
            "name": "Test Container Security Policy",
            "description": "Test policy for container security",
            "policy_type": "security",
            "severity": "error",
            "action": "block",
            "scope": "global",
            "rules": [
                {
                    "type": "container_security_context",
                    "condition": "runAsUser != 0"
                }
            ],
            "environments": ["test", "staging", "production"]
        }
        
        policy = await policy_enforcement.create_policy(policy_data, "test-user")
        
        assert policy.policy_id is not None
        assert policy.name == policy_data["name"]
        assert policy.policy_type == PolicyType.SECURITY
        assert policy.severity == PolicySeverity.ERROR
    
    @pytest.mark.asyncio
    async def test_policy_enforcement(self, policy_enforcement):
        """Test enforcing policies"""
        resource_context = {
            "kubernetes_resources": [
                {
                    "kind": "Deployment",
                    "metadata": {"name": "test-app"},
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": "app",
                                    "securityContext": {
                                        "runAsUser": 0  # Running as root - violation
                                    }
                                }]
                            }
                        }
                    }
                }
            ]
        }
        
        result = await policy_enforcement.enforce_policies("test", resource_context)
        
        assert result.enforcement_id is not None
        assert result.policies_evaluated > 0
        # Should find violations for root user
        assert result.violations_found >= 0
    
    @pytest.mark.asyncio
    async def test_compliance_report(self, policy_enforcement):
        """Test generating compliance report"""
        # First enforce some policies to generate violations
        resource_context = {
            "kubernetes_resources": [
                {
                    "kind": "Deployment",
                    "metadata": {"name": "test-app"},
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": "app",
                                    "securityContext": {"runAsUser": 0}
                                }]
                            }
                        }
                    }
                }
            ]
        }
        
        await policy_enforcement.enforce_policies("test", resource_context)
        
        # Generate compliance report
        report = policy_enforcement.get_policy_compliance_report("test")
        
        assert report["environment"] == "test"
        assert "overall_compliance_score" in report
        assert "policy_compliance" in report
        assert "violation_summary" in report

class TestAutomatedRollbackSystem:
    """Test automated rollback system"""
    
    @pytest.fixture
    def rollback_system(self):
        config = {}
        with patch('kubernetes.config.load_incluster_config'):
            with patch('kubernetes.config.load_kube_config'):
                return AutomatedRollbackSystem(config)
    
    @pytest.mark.asyncio
    async def test_service_registration(self, rollback_system):
        """Test registering service for rollback monitoring"""
        rollback_config = {
            "strategy": "immediate",
            "triggers": ["security_incident", "performance_degradation"],
            "thresholds": {
                "max_error_rate": 5.0,
                "max_response_time": 200
            },
            "auto_rollback_enabled": True,
            "approval_required": False
        }
        
        config = await rollback_system.register_service(
            "test-service", "test", rollback_config
        )
        
        assert config.service_name == "test-service"
        assert config.environment == "test"
        assert config.auto_rollback_enabled is True
        assert RollbackTrigger.SECURITY_INCIDENT in config.rollback_triggers
    
    @pytest.mark.asyncio
    async def test_manual_rollback(self, rollback_system):
        """Test manual rollback trigger"""
        # First register service
        rollback_config = {
            "strategy": "immediate",
            "triggers": ["manual_trigger"],
            "auto_rollback_enabled": True,
            "approval_required": False
        }
        
        await rollback_system.register_service("test-service", "test", rollback_config)
        
        # Mock Kubernetes operations
        with patch.object(rollback_system, '_get_current_version', return_value="v1.2.3"):
            with patch.object(rollback_system, '_update_deployment_image'):
                
                rollback_execution = await rollback_system.manual_rollback(
                    "test-service", "test", "v1.2.2", "Testing rollback", "test-user"
                )
                
                assert rollback_execution.rollback_id is not None
                assert rollback_execution.service_name == "test-service"
                assert rollback_execution.target_version == "v1.2.2"
                assert rollback_execution.trigger == RollbackTrigger.MANUAL_TRIGGER

class TestDevSecOpsOrchestrator:
    """Test DevSecOps orchestrator integration"""
    
    @pytest.fixture
    def orchestrator(self):
        config = {
            "security_gates": {},
            "vulnerability_scanner": {},
            "deployment_strategies": {},
            "change_workflows": {},
            "policy_enforcement": {},
            "rollback_system": {}
        }
        
        with patch('kubernetes.config.load_incluster_config'):
            with patch('kubernetes.config.load_kube_config'):
                with patch('docker.from_env'):
                    return DevSecOpsOrchestrator(config)
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_execution(self, orchestrator):
        """Test complete pipeline execution"""
        pipeline_config = {
            "application_name": "test-app",
            "environment": "test",
            "git_commit": "abc123",
            "image_tag": "test-app:v1.0.0",
            "deployment_strategy": "blue_green",
            "replicas": 3,
            "author": "test-user",
            "kubernetes_resources": [],
            "aws_resources": []
        }
        
        # Mock all the underlying operations
        with patch.object(orchestrator, '_execute_security_gates_phase'):
            with patch.object(orchestrator, '_execute_vulnerability_scanning_phase'):
                with patch.object(orchestrator, '_execute_policy_enforcement_phase'):
                    with patch.object(orchestrator, '_execute_deployment_phase'):
                        with patch.object(orchestrator, '_execute_monitoring_phase'):
                            
                            pipeline_execution = await orchestrator.execute_pipeline(pipeline_config)
                            
                            assert pipeline_execution.pipeline_id is not None
                            assert pipeline_execution.application_name == "test-app"
                            assert pipeline_execution.status in ["completed", "failed"]
    
    @pytest.mark.asyncio
    async def test_security_dashboard(self, orchestrator):
        """Test security dashboard data generation"""
        dashboard_data = await orchestrator.get_security_dashboard()
        
        assert "timestamp" in dashboard_data
        assert "pipeline_statistics" in dashboard_data
        assert "policy_violations" in dashboard_data
        assert "active_rollbacks" in dashboard_data
        assert "security_gates" in dashboard_data
        assert "infrastructure_changes" in dashboard_data
    
    @pytest.mark.asyncio
    async def test_pipeline_failure_rollback(self, orchestrator):
        """Test pipeline failure triggers rollback"""
        pipeline_config = {
            "application_name": "test-app",
            "environment": "test",
            "git_commit": "abc123",
            "image_tag": "test-app:v1.0.0",
            "deployment_strategy": "blue_green",
            "replicas": 3,
            "author": "test-user"
        }
        
        # Mock security gates to pass
        with patch.object(orchestrator, '_execute_security_gates_phase'):
            with patch.object(orchestrator, '_execute_vulnerability_scanning_phase'):
                with patch.object(orchestrator, '_execute_policy_enforcement_phase'):
                    # Mock deployment to fail
                    with patch.object(orchestrator, '_execute_deployment_phase', side_effect=Exception("Deployment failed")):
                        with patch.object(orchestrator, '_trigger_emergency_rollback') as mock_rollback:
                            
                            pipeline_execution = await orchestrator.execute_pipeline(pipeline_config)
                            
                            assert pipeline_execution.status == "failed"
                            # Should not trigger rollback if deployment never succeeded
                            mock_rollback.assert_not_called()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])