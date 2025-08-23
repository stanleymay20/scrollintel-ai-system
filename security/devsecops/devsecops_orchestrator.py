"""
DevSecOps Pipeline Integration Orchestrator
Coordinates all DevSecOps components for comprehensive pipeline security
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from .pipeline_security_gates import PipelineSecurityGates, SecurityGateType, SecurityGateStatus
from .container_vulnerability_scanner import ContainerVulnerabilityScanner
from .deployment_strategies import SecureDeploymentStrategies, DeploymentStrategy, DeploymentConfig
from .infrastructure_change_workflows import InfrastructureChangeWorkflows, ChangeType
from .security_policy_enforcement import SecurityPolicyEnforcement
from .automated_rollback_system import AutomatedRollbackSystem, RollbackTrigger

logger = logging.getLogger(__name__)

@dataclass
class PipelineExecution:
    pipeline_id: str
    application_name: str
    environment: str
    git_commit: str
    image_tag: str
    started_at: datetime
    status: str
    security_gates_passed: bool
    deployment_successful: bool
    rollback_triggered: bool
    execution_log: List[Dict[str, Any]]

class DevSecOpsOrchestrator:
    """
    Main orchestrator for DevSecOps pipeline integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.security_gates = PipelineSecurityGates(config.get("security_gates", {}))
        self.vulnerability_scanner = ContainerVulnerabilityScanner(config.get("vulnerability_scanner", {}))
        self.deployment_strategies = SecureDeploymentStrategies(config.get("deployment_strategies", {}))
        self.change_workflows = InfrastructureChangeWorkflows(config.get("change_workflows", {}))
        self.policy_enforcement = SecurityPolicyEnforcement(config.get("policy_enforcement", {}))
        self.rollback_system = AutomatedRollbackSystem(config.get("rollback_system", {}))
        
        self.active_pipelines = {}
        
    async def execute_pipeline(
        self, 
        pipeline_config: Dict[str, Any]
    ) -> PipelineExecution:
        """Execute complete DevSecOps pipeline"""
        pipeline_id = f"pipeline-{int(datetime.now().timestamp())}"
        
        pipeline_execution = PipelineExecution(
            pipeline_id=pipeline_id,
            application_name=pipeline_config["application_name"],
            environment=pipeline_config["environment"],
            git_commit=pipeline_config["git_commit"],
            image_tag=pipeline_config["image_tag"],
            started_at=datetime.now(),
            status="running",
            security_gates_passed=False,
            deployment_successful=False,
            rollback_triggered=False,
            execution_log=[]
        )
        
        self.active_pipelines[pipeline_id] = pipeline_execution
        
        try:
            # Phase 1: Security Gates
            await self._execute_security_gates_phase(pipeline_execution, pipeline_config)
            
            # Phase 2: Container Vulnerability Scanning
            await self._execute_vulnerability_scanning_phase(pipeline_execution, pipeline_config)
            
            # Phase 3: Policy Enforcement
            await self._execute_policy_enforcement_phase(pipeline_execution, pipeline_config)
            
            # Phase 4: Secure Deployment
            await self._execute_deployment_phase(pipeline_execution, pipeline_config)
            
            # Phase 5: Post-deployment Monitoring
            await self._execute_monitoring_phase(pipeline_execution, pipeline_config)
            
            pipeline_execution.status = "completed"
            
        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} failed: {str(e)}")
            pipeline_execution.status = "failed"
            
            # Trigger rollback if deployment was attempted
            if pipeline_execution.deployment_successful:
                await self._trigger_emergency_rollback(pipeline_execution, str(e))
        
        return pipeline_execution
    
    async def _execute_security_gates_phase(
        self, 
        pipeline_execution: PipelineExecution,
        pipeline_config: Dict[str, Any]
    ):
        """Execute security gates phase"""
        self._log_pipeline_event(pipeline_execution, "Starting security gates phase")
        
        # Define security gates to execute
        security_gates = [
            SecurityGateType.SAST_SCAN,
            SecurityGateType.DAST_SCAN,
            SecurityGateType.DEPENDENCY_SCAN,
            SecurityGateType.COMPLIANCE_CHECK
        ]
        
        pipeline_context = {
            "pipeline_id": pipeline_execution.pipeline_id,
            "application_name": pipeline_execution.application_name,
            "git_commit": pipeline_execution.git_commit,
            "author": pipeline_config.get("author", "unknown")
        }
        
        all_gates_passed = True
        
        for gate_type in security_gates:
            self._log_pipeline_event(pipeline_execution, f"Executing security gate: {gate_type.value}")
            
            gate_result = await self.security_gates.execute_security_gate(gate_type, pipeline_context)
            
            if gate_result.status == SecurityGateStatus.FAILED:
                all_gates_passed = False
                self._log_pipeline_event(
                    pipeline_execution, 
                    f"Security gate {gate_type.value} failed",
                    {"findings": gate_result.findings}
                )
                break
            elif gate_result.status == SecurityGateStatus.PENDING:
                # Create approval workflow
                workflow = await self.security_gates.create_approval_workflow(gate_result, pipeline_context)
                self._log_pipeline_event(
                    pipeline_execution,
                    f"Security gate {gate_type.value} requires approval",
                    {"workflow_id": workflow.workflow_id}
                )
                
                # For demo purposes, auto-approve after delay
                await asyncio.sleep(2)
                await self.security_gates.process_approval(
                    workflow.workflow_id, 
                    "auto-approver", 
                    SecurityGateStatus.APPROVED,
                    "Auto-approved for demo"
                )
        
        if not all_gates_passed:
            raise Exception("Security gates failed")
        
        pipeline_execution.security_gates_passed = True
        self._log_pipeline_event(pipeline_execution, "Security gates phase completed successfully")
    
    async def _execute_vulnerability_scanning_phase(
        self, 
        pipeline_execution: PipelineExecution,
        pipeline_config: Dict[str, Any]
    ):
        """Execute container vulnerability scanning phase"""
        self._log_pipeline_event(pipeline_execution, "Starting vulnerability scanning phase")
        
        # Extract image name and tag
        image_parts = pipeline_execution.image_tag.split(":")
        image_name = image_parts[0]
        image_tag = image_parts[1] if len(image_parts) > 1 else "latest"
        
        # Scan container image
        scan_result = await self.vulnerability_scanner.scan_image(image_name, image_tag)
        
        # Check if scan results are acceptable
        critical_vulns = [v for v in scan_result.vulnerabilities if v.severity.value == "critical"]
        high_vulns = [v for v in scan_result.vulnerabilities if v.severity.value == "high"]
        
        if critical_vulns:
            self._log_pipeline_event(
                pipeline_execution,
                f"Critical vulnerabilities found: {len(critical_vulns)}",
                {"vulnerabilities": [v.cve_id for v in critical_vulns]}
            )
            raise Exception(f"Critical vulnerabilities found: {len(critical_vulns)}")
        
        if len(high_vulns) > 5:  # Configurable threshold
            self._log_pipeline_event(
                pipeline_execution,
                f"Too many high severity vulnerabilities: {len(high_vulns)}",
                {"vulnerabilities": [v.cve_id for v in high_vulns]}
            )
            raise Exception(f"Too many high severity vulnerabilities: {len(high_vulns)}")
        
        self._log_pipeline_event(
            pipeline_execution,
            "Vulnerability scanning completed",
            {
                "security_score": scan_result.security_score,
                "total_vulnerabilities": len(scan_result.vulnerabilities),
                "critical": len(critical_vulns),
                "high": len(high_vulns)
            }
        )
    
    async def _execute_policy_enforcement_phase(
        self, 
        pipeline_execution: PipelineExecution,
        pipeline_config: Dict[str, Any]
    ):
        """Execute security policy enforcement phase"""
        self._log_pipeline_event(pipeline_execution, "Starting policy enforcement phase")
        
        # Create resource context for policy evaluation
        resource_context = {
            "kubernetes_resources": pipeline_config.get("kubernetes_resources", []),
            "aws_resources": pipeline_config.get("aws_resources", []),
            "application_name": pipeline_execution.application_name,
            "environment": pipeline_execution.environment
        }
        
        # Enforce policies
        enforcement_result = await self.policy_enforcement.enforce_policies(
            pipeline_execution.environment,
            resource_context
        )
        
        # Check for blocking violations
        blocking_violations = [
            v for v in enforcement_result.violations 
            if v.severity.value in ["critical", "error"]
        ]
        
        if blocking_violations:
            self._log_pipeline_event(
                pipeline_execution,
                f"Policy violations found: {len(blocking_violations)}",
                {"violations": [v.violation_id for v in blocking_violations]}
            )
            raise Exception(f"Policy violations found: {len(blocking_violations)}")
        
        self._log_pipeline_event(
            pipeline_execution,
            "Policy enforcement completed",
            {
                "policies_evaluated": enforcement_result.policies_evaluated,
                "violations_found": enforcement_result.violations_found
            }
        )
    
    async def _execute_deployment_phase(
        self, 
        pipeline_execution: PipelineExecution,
        pipeline_config: Dict[str, Any]
    ):
        """Execute secure deployment phase"""
        self._log_pipeline_event(pipeline_execution, "Starting deployment phase")
        
        # Create deployment configuration
        deployment_config = DeploymentConfig(
            strategy=DeploymentStrategy(pipeline_config.get("deployment_strategy", "blue_green")),
            namespace=pipeline_execution.environment,
            service_name=pipeline_execution.application_name,
            image=pipeline_execution.image_tag,
            replicas=pipeline_config.get("replicas", 3),
            security_validations=[
                "runtime_security_scan",
                "network_policy_check",
                "rbac_validation",
                "compliance_check"
            ],
            rollback_threshold=pipeline_config.get("rollback_threshold", 80.0),
            validation_timeout=pipeline_config.get("validation_timeout", 300)
        )
        
        # Execute deployment based on strategy
        if deployment_config.strategy == DeploymentStrategy.BLUE_GREEN:
            deployment_result = await self.deployment_strategies.deploy_blue_green(deployment_config)
        elif deployment_config.strategy == DeploymentStrategy.CANARY:
            deployment_result = await self.deployment_strategies.deploy_canary(deployment_config)
        else:
            raise Exception(f"Unsupported deployment strategy: {deployment_config.strategy}")
        
        # Check deployment result
        if deployment_result.status.value in ["completed"]:
            pipeline_execution.deployment_successful = True
            self._log_pipeline_event(
                pipeline_execution,
                "Deployment completed successfully",
                {
                    "deployment_id": deployment_result.deployment_id,
                    "strategy": deployment_result.strategy.value,
                    "security_validations_passed": len(deployment_result.security_validations)
                }
            )
        else:
            self._log_pipeline_event(
                pipeline_execution,
                f"Deployment failed: {deployment_result.status.value}",
                {"rollback_reason": deployment_result.rollback_reason}
            )
            raise Exception(f"Deployment failed: {deployment_result.status.value}")
    
    async def _execute_monitoring_phase(
        self, 
        pipeline_execution: PipelineExecution,
        pipeline_config: Dict[str, Any]
    ):
        """Execute post-deployment monitoring phase"""
        self._log_pipeline_event(pipeline_execution, "Starting monitoring phase")
        
        # Register service with rollback system
        rollback_config = {
            "strategy": "immediate",
            "triggers": ["security_incident", "performance_degradation", "error_rate_spike"],
            "thresholds": {
                "max_error_rate": 5.0,
                "max_response_time": 200,
                "max_cpu_usage": 80
            },
            "auto_rollback_enabled": True,
            "approval_required": False
        }
        
        await self.rollback_system.register_service(
            pipeline_execution.application_name,
            pipeline_execution.environment,
            rollback_config
        )
        
        self._log_pipeline_event(
            pipeline_execution,
            "Service registered for automated rollback monitoring"
        )
    
    async def _trigger_emergency_rollback(
        self, 
        pipeline_execution: PipelineExecution,
        reason: str
    ):
        """Trigger emergency rollback"""
        self._log_pipeline_event(
            pipeline_execution,
            f"Triggering emergency rollback: {reason}"
        )
        
        try:
            rollback_result = await self.rollback_system.manual_rollback(
                pipeline_execution.application_name,
                pipeline_execution.environment,
                "previous-stable",
                reason,
                "pipeline-orchestrator"
            )
            
            pipeline_execution.rollback_triggered = True
            self._log_pipeline_event(
                pipeline_execution,
                "Emergency rollback initiated",
                {"rollback_id": rollback_result.rollback_id}
            )
            
        except Exception as e:
            self._log_pipeline_event(
                pipeline_execution,
                f"Emergency rollback failed: {str(e)}"
            )
    
    def _log_pipeline_event(
        self, 
        pipeline_execution: PipelineExecution,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log pipeline event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "details": details or {}
        }
        
        pipeline_execution.execution_log.append(event)
        logger.info(f"Pipeline {pipeline_execution.pipeline_id}: {message}")
    
    async def submit_infrastructure_change(
        self, 
        change_data: Dict[str, Any],
        submitter: str
    ):
        """Submit infrastructure change request"""
        return await self.change_workflows.submit_change_request(change_data, submitter)
    
    async def approve_infrastructure_change(
        self, 
        change_id: str,
        approver: str,
        decision: str,
        comments: str = ""
    ):
        """Approve infrastructure change"""
        from .infrastructure_change_workflows import ApprovalStatus
        
        approval_status = ApprovalStatus.APPROVED if decision == "approved" else ApprovalStatus.REJECTED
        
        return await self.change_workflows.submit_approval(
            change_id, approver, approval_status, comments
        )
    
    async def get_pipeline_status(self, pipeline_id: str) -> Optional[PipelineExecution]:
        """Get pipeline execution status"""
        return self.active_pipelines.get(pipeline_id)
    
    async def list_active_pipelines(self) -> List[PipelineExecution]:
        """List all active pipelines"""
        return list(self.active_pipelines.values())
    
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data"""
        # Get policy violations
        policy_violations = self.policy_enforcement.get_policy_violations()
        
        # Get active rollbacks
        active_rollbacks = self.rollback_system.list_active_rollbacks()
        
        # Get pipeline statistics
        pipeline_stats = self._calculate_pipeline_statistics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "pipeline_statistics": pipeline_stats,
            "policy_violations": {
                "total": len(policy_violations),
                "critical": len([v for v in policy_violations if v.severity.value == "critical"]),
                "active": len([v for v in policy_violations if v.status.value == "active"])
            },
            "active_rollbacks": len(active_rollbacks),
            "security_gates": {
                "total_executions": len(self.security_gates.gate_results),
                "approval_workflows": len(self.security_gates.approval_workflows)
            },
            "infrastructure_changes": {
                "pending_approval": len(self.change_workflows.list_changes_by_status(
                    self.change_workflows.ChangeStatus.UNDER_REVIEW
                ))
            }
        }
    
    def _calculate_pipeline_statistics(self) -> Dict[str, Any]:
        """Calculate pipeline execution statistics"""
        pipelines = list(self.active_pipelines.values())
        
        if not pipelines:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_duration": 0.0
            }
        
        completed_pipelines = [p for p in pipelines if p.status in ["completed", "failed"]]
        successful_pipelines = [p for p in completed_pipelines if p.status == "completed"]
        
        success_rate = len(successful_pipelines) / len(completed_pipelines) * 100 if completed_pipelines else 0
        
        return {
            "total_executions": len(pipelines),
            "completed": len(completed_pipelines),
            "successful": len(successful_pipelines),
            "success_rate": success_rate,
            "security_gates_passed": len([p for p in pipelines if p.security_gates_passed]),
            "deployments_successful": len([p for p in pipelines if p.deployment_successful]),
            "rollbacks_triggered": len([p for p in pipelines if p.rollback_triggered])
        }