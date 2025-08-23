"""
Demo script for DevSecOps Pipeline Integration
Demonstrates comprehensive DevSecOps pipeline capabilities
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

from security.devsecops.devsecops_orchestrator import DevSecOpsOrchestrator
from security.devsecops.pipeline_security_gates import SecurityGateType
from security.devsecops.deployment_strategies import DeploymentStrategy
from security.devsecops.infrastructure_change_workflows import ChangeType
from security.devsecops.automated_rollback_system import RollbackTrigger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DevSecOpsPipelineDemo:
    """
    Comprehensive demo of DevSecOps pipeline integration capabilities
    """
    
    def __init__(self):
        self.config = {
            "security_gates": {
                "sast_enabled": True,
                "dast_enabled": True,
                "container_scan_enabled": True,
                "dependency_scan_enabled": True
            },
            "vulnerability_scanner": {
                "scanners": ["trivy", "grype"],
                "severity_threshold": "high"
            },
            "deployment_strategies": {
                "default_strategy": "blue_green",
                "security_validations_required": True
            },
            "change_workflows": {
                "approval_required": True,
                "security_assessment_enabled": True
            },
            "policy_enforcement": {
                "enforcement_enabled": True,
                "blocking_violations": ["critical", "error"]
            },
            "rollback_system": {
                "auto_rollback_enabled": True,
                "monitoring_interval": 30
            }
        }
        
        self.orchestrator = DevSecOpsOrchestrator(self.config)
    
    async def run_complete_demo(self):
        """Run complete DevSecOps pipeline demo"""
        print("🚀 Starting DevSecOps Pipeline Integration Demo")
        print("=" * 60)
        
        try:
            # Demo 1: Complete Pipeline Execution
            await self.demo_complete_pipeline_execution()
            
            # Demo 2: Security Gates
            await self.demo_security_gates()
            
            # Demo 3: Container Vulnerability Scanning
            await self.demo_container_vulnerability_scanning()
            
            # Demo 4: Deployment Strategies
            await self.demo_deployment_strategies()
            
            # Demo 5: Infrastructure Change Workflows
            await self.demo_infrastructure_change_workflows()
            
            # Demo 6: Security Policy Enforcement
            await self.demo_security_policy_enforcement()
            
            # Demo 7: Automated Rollback System
            await self.demo_automated_rollback_system()
            
            # Demo 8: Security Dashboard
            await self.demo_security_dashboard()
            
            print("\n✅ DevSecOps Pipeline Integration Demo Completed Successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            print(f"\n❌ Demo failed: {str(e)}")
    
    async def demo_complete_pipeline_execution(self):
        """Demo complete pipeline execution"""
        print("\n📋 Demo 1: Complete Pipeline Execution")
        print("-" * 40)
        
        pipeline_config = {
            "application_name": "payment-service",
            "environment": "staging",
            "git_commit": "a1b2c3d4e5f6",
            "image_tag": "payment-service:v2.1.0",
            "deployment_strategy": "blue_green",
            "replicas": 5,
            "rollback_threshold": 85.0,
            "validation_timeout": 300,
            "author": "dev-team-lead",
            "kubernetes_resources": [
                {
                    "kind": "Deployment",
                    "metadata": {"name": "payment-service"},
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": "payment-service",
                                    "image": "payment-service:v2.1.0",
                                    "securityContext": {
                                        "runAsNonRoot": True,
                                        "runAsUser": 1000,
                                        "readOnlyRootFilesystem": True
                                    },
                                    "resources": {
                                        "limits": {"memory": "512Mi", "cpu": "500m"},
                                        "requests": {"memory": "256Mi", "cpu": "250m"}
                                    }
                                }]
                            }
                        }
                    }
                }
            ],
            "aws_resources": [
                {
                    "resource_type": "aws_rds_instance",
                    "resource_id": "payment-db",
                    "configuration": {
                        "storage_encrypted": True,
                        "backup_retention_period": 7
                    }
                }
            ]
        }
        
        print(f"Executing pipeline for {pipeline_config['application_name']}...")
        
        try:
            pipeline_execution = await self.orchestrator.execute_pipeline(pipeline_config)
            
            print(f"✅ Pipeline ID: {pipeline_execution.pipeline_id}")
            print(f"   Status: {pipeline_execution.status}")
            print(f"   Security Gates Passed: {pipeline_execution.security_gates_passed}")
            print(f"   Deployment Successful: {pipeline_execution.deployment_successful}")
            print(f"   Started: {pipeline_execution.started_at}")
            
            # Show execution log
            print("\n📝 Execution Log:")
            for i, log_entry in enumerate(pipeline_execution.execution_log[-5:], 1):
                print(f"   {i}. {log_entry['message']}")
            
        except Exception as e:
            print(f"❌ Pipeline execution failed: {str(e)}")
    
    async def demo_security_gates(self):
        """Demo security gates functionality"""
        print("\n🔒 Demo 2: Security Gates")
        print("-" * 40)
        
        pipeline_context = {
            "pipeline_id": "demo-pipeline-001",
            "application_name": "user-service",
            "git_commit": "def456",
            "author": "security-team"
        }
        
        security_gates = [
            SecurityGateType.SAST_SCAN,
            SecurityGateType.DAST_SCAN,
            SecurityGateType.DEPENDENCY_SCAN,
            SecurityGateType.COMPLIANCE_CHECK
        ]
        
        print("Executing security gates...")
        
        for gate_type in security_gates:
            try:
                result = await self.orchestrator.security_gates.execute_security_gate(
                    gate_type, pipeline_context
                )
                
                print(f"✅ {gate_type.value}:")
                print(f"   Status: {result.status.value}")
                print(f"   Score: {result.score:.1f}")
                print(f"   Findings: {len(result.findings)}")
                print(f"   Execution Time: {result.execution_time:.2f}s")
                
                if result.findings:
                    print(f"   Sample Finding: {result.findings[0].get('type', 'N/A')}")
                
            except Exception as e:
                print(f"❌ {gate_type.value} failed: {str(e)}")
        
        # Get overall pipeline security status
        try:
            security_status = await self.orchestrator.security_gates.get_pipeline_security_status(
                pipeline_context["pipeline_id"]
            )
            
            print(f"\n📊 Overall Security Status:")
            print(f"   Status: {security_status['status']}")
            print(f"   Overall Score: {security_status['overall_score']:.1f}")
            print(f"   Gates Executed: {len(security_status['gate_results'])}")
            
        except Exception as e:
            print(f"❌ Failed to get security status: {str(e)}")
    
    async def demo_container_vulnerability_scanning(self):
        """Demo container vulnerability scanning"""
        print("\n🐳 Demo 3: Container Vulnerability Scanning")
        print("-" * 40)
        
        test_images = [
            ("nginx", "1.20"),
            ("node", "14-alpine"),
            ("python", "3.9-slim")
        ]
        
        for image_name, image_tag in test_images:
            print(f"\nScanning {image_name}:{image_tag}...")
            
            try:
                scan_result = await self.orchestrator.vulnerability_scanner.scan_image(
                    image_name, image_tag
                )
                
                print(f"✅ Scan completed:")
                print(f"   Security Score: {scan_result.security_score:.1f}")
                print(f"   Compliance Score: {scan_result.compliance_score:.1f}")
                print(f"   Vulnerabilities: {len(scan_result.vulnerabilities)}")
                print(f"   Misconfigurations: {len(scan_result.misconfigurations)}")
                print(f"   Scan Duration: {scan_result.scan_duration:.2f}s")
                
                # Show vulnerability breakdown
                vuln_by_severity = {}
                for vuln in scan_result.vulnerabilities:
                    severity = vuln.severity.value
                    vuln_by_severity[severity] = vuln_by_severity.get(severity, 0) + 1
                
                if vuln_by_severity:
                    print("   Vulnerability Breakdown:")
                    for severity, count in vuln_by_severity.items():
                        print(f"     {severity.title()}: {count}")
                
                # Generate scan report
                scan_report = await self.orchestrator.vulnerability_scanner.generate_scan_report(scan_result)
                print(f"   Report Generated: {len(scan_report['recommendations'])} recommendations")
                
            except Exception as e:
                print(f"❌ Scan failed for {image_name}:{image_tag}: {str(e)}")
    
    async def demo_deployment_strategies(self):
        """Demo deployment strategies"""
        print("\n🚀 Demo 4: Deployment Strategies")
        print("-" * 40)
        
        from security.devsecops.deployment_strategies import DeploymentConfig
        
        # Demo Blue-Green Deployment
        print("\n🔵🟢 Blue-Green Deployment:")
        
        bg_config = DeploymentConfig(
            strategy=DeploymentStrategy.BLUE_GREEN,
            namespace="demo",
            service_name="api-service",
            image="api-service:v1.5.0",
            replicas=3,
            security_validations=[
                "runtime_security_scan",
                "network_policy_check",
                "compliance_check"
            ],
            rollback_threshold=80.0,
            validation_timeout=300
        )
        
        try:
            # Note: This would normally deploy to Kubernetes
            # For demo, we'll simulate the process
            print(f"   Deploying {bg_config.service_name} using blue-green strategy...")
            print(f"   Image: {bg_config.image}")
            print(f"   Replicas: {bg_config.replicas}")
            print(f"   Security Validations: {len(bg_config.security_validations)}")
            print("   ✅ Blue-green deployment simulation completed")
            
        except Exception as e:
            print(f"   ❌ Blue-green deployment failed: {str(e)}")
        
        # Demo Canary Deployment
        print("\n🐤 Canary Deployment:")
        
        canary_config = DeploymentConfig(
            strategy=DeploymentStrategy.CANARY,
            namespace="demo",
            service_name="frontend-service",
            image="frontend-service:v2.0.0",
            replicas=5,
            security_validations=["runtime_security_scan"],
            rollback_threshold=85.0,
            validation_timeout=300,
            canary_config={
                "initial_traffic": 5,
                "increment_traffic": 15,
                "max_traffic": 100,
                "validation_interval": 120
            }
        )
        
        try:
            print(f"   Deploying {canary_config.service_name} using canary strategy...")
            print(f"   Initial Traffic: {canary_config.canary_config['initial_traffic']}%")
            print(f"   Traffic Increment: {canary_config.canary_config['increment_traffic']}%")
            print(f"   Validation Interval: {canary_config.canary_config['validation_interval']}s")
            print("   ✅ Canary deployment simulation completed")
            
        except Exception as e:
            print(f"   ❌ Canary deployment failed: {str(e)}")
    
    async def demo_infrastructure_change_workflows(self):
        """Demo infrastructure change workflows"""
        print("\n🏗️ Demo 5: Infrastructure Change Workflows")
        print("-" * 40)
        
        # Submit infrastructure change
        change_data = {
            "title": "Update Kubernetes Security Policies",
            "description": "Implement new pod security standards and network policies",
            "change_type": "security_policy",
            "files_changed": [
                "k8s/security-policies/pod-security-policy.yaml",
                "k8s/network-policies/default-deny.yaml",
                "k8s/rbac/security-rbac.yaml"
            ],
            "implementation_plan": {
                "steps": [
                    "Review current security policies",
                    "Apply new pod security policies",
                    "Update network policies",
                    "Test security controls",
                    "Validate compliance"
                ]
            },
            "rollback_plan": {
                "steps": [
                    "Revert to previous security policies",
                    "Restore original network policies",
                    "Validate rollback"
                ]
            },
            "testing_plan": {
                "steps": [
                    "Unit tests for policy validation",
                    "Integration tests for network isolation",
                    "Security compliance tests"
                ]
            },
            "metadata": {
                "priority": "high",
                "compliance_frameworks": ["SOC2", "ISO27001"]
            }
        }
        
        try:
            print("Submitting infrastructure change request...")
            
            change = await self.orchestrator.submit_infrastructure_change(
                change_data, "infrastructure-team"
            )
            
            print(f"✅ Change Request Submitted:")
            print(f"   Change ID: {change.change_id}")
            print(f"   Title: {change.title}")
            print(f"   Type: {change.change_type.value}")
            print(f"   Risk Level: {change.risk_level.value}")
            print(f"   Status: {change.status.value}")
            print(f"   Required Approvers: {', '.join(change.required_approvers)}")
            
            # Show security assessment
            if change.security_assessment:
                print(f"\n🔍 Security Assessment:")
                print(f"   Risk Score: {change.security_assessment.risk_score:.1f}")
                print(f"   Findings: {len(change.security_assessment.security_findings)}")
                print(f"   Compliance Impact: {', '.join(change.security_assessment.compliance_impact)}")
                print(f"   Recommendations: {len(change.security_assessment.recommendations)}")
            
            # Simulate approval process
            print(f"\n✅ Simulating approval process...")
            
            for approver in change.required_approvers[:2]:  # Approve with first 2 approvers
                success = await self.orchestrator.approve_infrastructure_change(
                    change.change_id, approver, "approved", f"Approved by {approver}"
                )
                
                if success:
                    print(f"   ✅ Approved by {approver}")
                else:
                    print(f"   ❌ Approval failed for {approver}")
            
        except Exception as e:
            print(f"❌ Infrastructure change workflow failed: {str(e)}")
    
    async def demo_security_policy_enforcement(self):
        """Demo security policy enforcement"""
        print("\n🛡️ Demo 6: Security Policy Enforcement")
        print("-" * 40)
        
        # Create test resource context
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
                                    "image": "test-app:latest",
                                    "securityContext": {
                                        "runAsUser": 0,  # Violation: running as root
                                        "privileged": False
                                    },
                                    "resources": {
                                        "limits": {"memory": "512Mi"}
                                        # Missing CPU limits - violation
                                    }
                                }]
                            }
                        }
                    }
                },
                {
                    "kind": "Ingress",
                    "metadata": {"name": "test-ingress"},
                    "spec": {
                        "rules": [{"host": "test.example.com"}]
                        # Missing TLS configuration - violation
                    }
                }
            ],
            "aws_resources": [
                {
                    "resource_type": "aws_s3_bucket",
                    "resource_id": "test-bucket",
                    "configuration": {
                        "versioning": True
                        # Missing encryption configuration - violation
                    }
                }
            ]
        }
        
        environments = ["development", "staging", "production"]
        
        for environment in environments:
            print(f"\n🔍 Enforcing policies in {environment}:")
            
            try:
                enforcement_result = await self.orchestrator.policy_enforcement.enforce_policies(
                    environment, resource_context
                )
                
                print(f"   ✅ Enforcement completed:")
                print(f"      Policies Evaluated: {enforcement_result.policies_evaluated}")
                print(f"      Violations Found: {enforcement_result.violations_found}")
                print(f"      Actions Taken: {len(enforcement_result.actions_taken)}")
                print(f"      Execution Time: {enforcement_result.execution_time:.2f}s")
                
                # Show violation breakdown
                if enforcement_result.violations:
                    violation_by_severity = {}
                    for violation in enforcement_result.violations:
                        severity = violation.severity.value
                        violation_by_severity[severity] = violation_by_severity.get(severity, 0) + 1
                    
                    print("      Violation Breakdown:")
                    for severity, count in violation_by_severity.items():
                        print(f"        {severity.title()}: {count}")
                
                # Generate compliance report
                compliance_report = self.orchestrator.policy_enforcement.get_policy_compliance_report(environment)
                print(f"      Compliance Score: {compliance_report['overall_compliance_score']:.1f}")
                
            except Exception as e:
                print(f"   ❌ Policy enforcement failed: {str(e)}")
    
    async def demo_automated_rollback_system(self):
        """Demo automated rollback system"""
        print("\n🔄 Demo 7: Automated Rollback System")
        print("-" * 40)
        
        # Register services for rollback monitoring
        services = [
            {
                "name": "payment-service",
                "environment": "production",
                "config": {
                    "strategy": "immediate",
                    "triggers": ["security_incident", "error_rate_spike"],
                    "thresholds": {
                        "max_error_rate": 3.0,
                        "max_response_time": 150,
                        "max_cpu_usage": 85
                    },
                    "auto_rollback_enabled": True,
                    "approval_required": False
                }
            },
            {
                "name": "user-service",
                "environment": "production",
                "config": {
                    "strategy": "gradual",
                    "triggers": ["performance_degradation", "health_check_failure"],
                    "thresholds": {
                        "max_error_rate": 5.0,
                        "max_response_time": 200
                    },
                    "auto_rollback_enabled": True,
                    "approval_required": True
                }
            }
        ]
        
        print("Registering services for rollback monitoring...")
        
        for service in services:
            try:
                rollback_config = await self.orchestrator.rollback_system.register_service(
                    service["name"], service["environment"], service["config"]
                )
                
                print(f"✅ {service['name']} registered:")
                print(f"   Strategy: {rollback_config.strategy.value}")
                print(f"   Auto-rollback: {rollback_config.auto_rollback_enabled}")
                print(f"   Triggers: {[t.value for t in rollback_config.rollback_triggers]}")
                
            except Exception as e:
                print(f"❌ Failed to register {service['name']}: {str(e)}")
        
        # Demo manual rollback
        print(f"\n🔄 Triggering manual rollback...")
        
        try:
            rollback_execution = await self.orchestrator.rollback_system.manual_rollback(
                "payment-service",
                "production",
                "v2.0.5",
                "Critical security vulnerability discovered",
                "security-team"
            )
            
            print(f"✅ Manual rollback initiated:")
            print(f"   Rollback ID: {rollback_execution.rollback_id}")
            print(f"   Service: {rollback_execution.service_name}")
            print(f"   Target Version: {rollback_execution.target_version}")
            print(f"   Status: {rollback_execution.status.value}")
            print(f"   Trigger: {rollback_execution.trigger.value}")
            
            # Simulate waiting for rollback completion
            await asyncio.sleep(2)
            
            # Check rollback status
            updated_status = await self.orchestrator.rollback_system.get_rollback_status(
                rollback_execution.rollback_id
            )
            
            if updated_status:
                print(f"   Updated Status: {updated_status.status.value}")
                print(f"   Steps Completed: {len(updated_status.steps_completed)}")
            
        except Exception as e:
            print(f"❌ Manual rollback failed: {str(e)}")
    
    async def demo_security_dashboard(self):
        """Demo security dashboard"""
        print("\n📊 Demo 8: Security Dashboard")
        print("-" * 40)
        
        try:
            dashboard_data = await self.orchestrator.get_security_dashboard()
            
            print("✅ Security Dashboard Data:")
            print(f"   Timestamp: {dashboard_data['timestamp']}")
            
            # Pipeline Statistics
            pipeline_stats = dashboard_data['pipeline_statistics']
            print(f"\n📈 Pipeline Statistics:")
            print(f"   Total Executions: {pipeline_stats['total_executions']}")
            print(f"   Success Rate: {pipeline_stats['success_rate']:.1f}%")
            print(f"   Security Gates Passed: {pipeline_stats['security_gates_passed']}")
            print(f"   Successful Deployments: {pipeline_stats['deployments_successful']}")
            print(f"   Rollbacks Triggered: {pipeline_stats['rollbacks_triggered']}")
            
            # Policy Violations
            policy_violations = dashboard_data['policy_violations']
            print(f"\n🚨 Policy Violations:")
            print(f"   Total: {policy_violations['total']}")
            print(f"   Critical: {policy_violations['critical']}")
            print(f"   Active: {policy_violations['active']}")
            
            # Active Rollbacks
            print(f"\n🔄 Active Rollbacks: {dashboard_data['active_rollbacks']}")
            
            # Security Gates
            security_gates = dashboard_data['security_gates']
            print(f"\n🔒 Security Gates:")
            print(f"   Total Executions: {security_gates['total_executions']}")
            print(f"   Approval Workflows: {security_gates['approval_workflows']}")
            
            # Infrastructure Changes
            infra_changes = dashboard_data['infrastructure_changes']
            print(f"\n🏗️ Infrastructure Changes:")
            print(f"   Pending Approval: {infra_changes['pending_approval']}")
            
        except Exception as e:
            print(f"❌ Failed to get security dashboard: {str(e)}")
    
    def print_summary(self):
        """Print demo summary"""
        print("\n" + "=" * 60)
        print("📋 DevSecOps Pipeline Integration Summary")
        print("=" * 60)
        
        features = [
            "✅ Pipeline Security Gates (SAST, DAST, Container Scanning)",
            "✅ Container Vulnerability Scanning with Misconfiguration Detection",
            "✅ Blue-Green and Canary Deployment Strategies with Security Validation",
            "✅ Infrastructure Change Review and Approval Workflows",
            "✅ Automated Security Policy Enforcement Across Environments",
            "✅ Automated Rollback System Triggered by Security Incidents",
            "✅ Comprehensive Security Dashboard and Reporting",
            "✅ Integration with Kubernetes, Docker, and Cloud Platforms"
        ]
        
        for feature in features:
            print(feature)
        
        print("\n🎯 Key Benefits:")
        benefits = [
            "• Shift-left security with automated gates",
            "• Continuous vulnerability management",
            "• Zero-downtime secure deployments",
            "• Automated compliance and governance",
            "• Intelligent incident response and rollback",
            "• Enterprise-grade security at scale"
        ]
        
        for benefit in benefits:
            print(benefit)

async def main():
    """Main demo function"""
    demo = DevSecOpsPipelineDemo()
    
    try:
        await demo.run_complete_demo()
        demo.print_summary()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Demo failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())