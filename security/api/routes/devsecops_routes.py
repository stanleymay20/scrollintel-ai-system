"""
DevSecOps Pipeline Integration API Routes
Provides REST API endpoints for DevSecOps pipeline operations
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ...devsecops.devsecops_orchestrator import DevSecOpsOrchestrator
from ...devsecops.pipeline_security_gates import SecurityGateType, SecurityGateStatus
from ...devsecops.deployment_strategies import DeploymentStrategy
from ...devsecops.infrastructure_change_workflows import ChangeType, ChangeRisk

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/devsecops", tags=["DevSecOps"])
security = HTTPBearer()

# Global orchestrator instance (in production, this would be dependency injected)
orchestrator = None

def get_orchestrator():
    """Get DevSecOps orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        config = {
            "security_gates": {},
            "vulnerability_scanner": {},
            "deployment_strategies": {},
            "change_workflows": {},
            "policy_enforcement": {},
            "rollback_system": {}
        }
        orchestrator = DevSecOpsOrchestrator(config)
    return orchestrator

# Pydantic models for request/response
class PipelineExecutionRequest(BaseModel):
    application_name: str = Field(..., description="Name of the application")
    environment: str = Field(..., description="Target environment")
    git_commit: str = Field(..., description="Git commit hash")
    image_tag: str = Field(..., description="Container image tag")
    deployment_strategy: str = Field(default="blue_green", description="Deployment strategy")
    replicas: int = Field(default=3, description="Number of replicas")
    rollback_threshold: float = Field(default=80.0, description="Rollback threshold score")
    validation_timeout: int = Field(default=300, description="Validation timeout in seconds")
    author: Optional[str] = Field(None, description="Pipeline author")
    kubernetes_resources: Optional[List[Dict[str, Any]]] = Field(default=[], description="Kubernetes resources")
    aws_resources: Optional[List[Dict[str, Any]]] = Field(default=[], description="AWS resources")

class SecurityGateExecutionRequest(BaseModel):
    gate_type: str = Field(..., description="Security gate type")
    pipeline_context: Dict[str, Any] = Field(..., description="Pipeline context")

class ContainerScanRequest(BaseModel):
    image_name: str = Field(..., description="Container image name")
    image_tag: str = Field(default="latest", description="Container image tag")
    scan_options: Optional[Dict[str, Any]] = Field(default={}, description="Scan options")

class InfrastructureChangeRequest(BaseModel):
    title: str = Field(..., description="Change title")
    description: str = Field(..., description="Change description")
    change_type: str = Field(..., description="Type of change")
    files_changed: List[str] = Field(..., description="List of files changed")
    implementation_plan: Dict[str, Any] = Field(..., description="Implementation plan")
    rollback_plan: Dict[str, Any] = Field(..., description="Rollback plan")
    testing_plan: Optional[Dict[str, Any]] = Field(default={}, description="Testing plan")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")

class ChangeApprovalRequest(BaseModel):
    decision: str = Field(..., description="Approval decision (approved/rejected)")
    comments: str = Field(default="", description="Approval comments")

class ManualRollbackRequest(BaseModel):
    service_name: str = Field(..., description="Service name")
    environment: str = Field(..., description="Environment")
    target_version: str = Field(..., description="Target version to rollback to")
    reason: str = Field(..., description="Reason for rollback")

class ServiceRegistrationRequest(BaseModel):
    service_name: str = Field(..., description="Service name")
    environment: str = Field(..., description="Environment")
    rollback_config: Dict[str, Any] = Field(..., description="Rollback configuration")

# Pipeline execution endpoints
@router.post("/pipeline/execute")
async def execute_pipeline(
    request: PipelineExecutionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Execute DevSecOps pipeline"""
    try:
        orchestrator_instance = get_orchestrator()
        
        pipeline_config = request.dict()
        
        # Execute pipeline in background
        pipeline_execution = await orchestrator_instance.execute_pipeline(pipeline_config)
        
        return {
            "pipeline_id": pipeline_execution.pipeline_id,
            "status": pipeline_execution.status,
            "started_at": pipeline_execution.started_at.isoformat(),
            "application_name": pipeline_execution.application_name,
            "environment": pipeline_execution.environment,
            "message": "Pipeline execution started"
        }
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/{pipeline_id}/status")
async def get_pipeline_status(
    pipeline_id: str,
    token: str = Depends(security)
):
    """Get pipeline execution status"""
    try:
        orchestrator_instance = get_orchestrator()
        
        pipeline_execution = await orchestrator_instance.get_pipeline_status(pipeline_id)
        
        if not pipeline_execution:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return {
            "pipeline_id": pipeline_execution.pipeline_id,
            "application_name": pipeline_execution.application_name,
            "environment": pipeline_execution.environment,
            "status": pipeline_execution.status,
            "started_at": pipeline_execution.started_at.isoformat(),
            "security_gates_passed": pipeline_execution.security_gates_passed,
            "deployment_successful": pipeline_execution.deployment_successful,
            "rollback_triggered": pipeline_execution.rollback_triggered,
            "execution_log": pipeline_execution.execution_log
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/active")
async def list_active_pipelines(
    token: str = Depends(security)
):
    """List all active pipelines"""
    try:
        orchestrator_instance = get_orchestrator()
        
        active_pipelines = await orchestrator_instance.list_active_pipelines()
        
        return {
            "active_pipelines": [
                {
                    "pipeline_id": p.pipeline_id,
                    "application_name": p.application_name,
                    "environment": p.environment,
                    "status": p.status,
                    "started_at": p.started_at.isoformat()
                } for p in active_pipelines
            ],
            "total_count": len(active_pipelines)
        }
        
    except Exception as e:
        logger.error(f"Error listing active pipelines: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Security gates endpoints
@router.post("/security-gates/execute")
async def execute_security_gate(
    request: SecurityGateExecutionRequest,
    token: str = Depends(security)
):
    """Execute individual security gate"""
    try:
        orchestrator_instance = get_orchestrator()
        
        gate_type = SecurityGateType(request.gate_type)
        
        gate_result = await orchestrator_instance.security_gates.execute_security_gate(
            gate_type, request.pipeline_context
        )
        
        return {
            "gate_type": gate_result.gate_type.value,
            "status": gate_result.status.value,
            "score": gate_result.score,
            "findings": gate_result.findings,
            "recommendations": gate_result.recommendations,
            "execution_time": gate_result.execution_time,
            "timestamp": gate_result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Security gate execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/security-gates/pipeline/{pipeline_id}/status")
async def get_pipeline_security_status(
    pipeline_id: str,
    token: str = Depends(security)
):
    """Get security status for pipeline"""
    try:
        orchestrator_instance = get_orchestrator()
        
        security_status = await orchestrator_instance.security_gates.get_pipeline_security_status(pipeline_id)
        
        return security_status
        
    except Exception as e:
        logger.error(f"Error getting pipeline security status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Container vulnerability scanning endpoints
@router.post("/vulnerability-scan/container")
async def scan_container(
    request: ContainerScanRequest,
    token: str = Depends(security)
):
    """Scan container for vulnerabilities"""
    try:
        orchestrator_instance = get_orchestrator()
        
        scan_result = await orchestrator_instance.vulnerability_scanner.scan_image(
            request.image_name, request.image_tag, request.scan_options
        )
        
        # Generate scan report
        scan_report = await orchestrator_instance.vulnerability_scanner.generate_scan_report(scan_result)
        
        return scan_report
        
    except Exception as e:
        logger.error(f"Container vulnerability scan failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Infrastructure change management endpoints
@router.post("/infrastructure/changes")
async def submit_infrastructure_change(
    request: InfrastructureChangeRequest,
    submitter: str = "api-user",
    token: str = Depends(security)
):
    """Submit infrastructure change request"""
    try:
        orchestrator_instance = get_orchestrator()
        
        change_data = request.dict()
        
        change = await orchestrator_instance.submit_infrastructure_change(change_data, submitter)
        
        return {
            "change_id": change.change_id,
            "title": change.title,
            "change_type": change.change_type.value,
            "risk_level": change.risk_level.value,
            "status": change.status.value,
            "required_approvers": change.required_approvers,
            "submission_time": change.submission_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Infrastructure change submission failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/infrastructure/changes/{change_id}/approve")
async def approve_infrastructure_change(
    change_id: str,
    request: ChangeApprovalRequest,
    approver: str = "api-user",
    token: str = Depends(security)
):
    """Approve infrastructure change"""
    try:
        orchestrator_instance = get_orchestrator()
        
        success = await orchestrator_instance.approve_infrastructure_change(
            change_id, approver, request.decision, request.comments
        )
        
        return {
            "change_id": change_id,
            "approver": approver,
            "decision": request.decision,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Infrastructure change approval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/infrastructure/changes/{change_id}")
async def get_infrastructure_change(
    change_id: str,
    token: str = Depends(security)
):
    """Get infrastructure change details"""
    try:
        orchestrator_instance = get_orchestrator()
        
        change = orchestrator_instance.change_workflows.get_change_status(change_id)
        
        if not change:
            raise HTTPException(status_code=404, detail="Change not found")
        
        return {
            "change_id": change.change_id,
            "title": change.title,
            "description": change.description,
            "change_type": change.change_type.value,
            "risk_level": change.risk_level.value,
            "status": change.status.value,
            "submitter": change.submitter,
            "submission_time": change.submission_time.isoformat(),
            "required_approvers": change.required_approvers,
            "approvals": [
                {
                    "approver": a.approver,
                    "status": a.status.value,
                    "comments": a.comments,
                    "timestamp": a.timestamp.isoformat()
                } for a in change.approvals
            ],
            "security_assessment": {
                "risk_score": change.security_assessment.risk_score if change.security_assessment else 0,
                "findings_count": len(change.security_assessment.security_findings) if change.security_assessment else 0
            } if change.security_assessment else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting infrastructure change: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Policy enforcement endpoints
@router.post("/policy/enforce/{environment}")
async def enforce_policies(
    environment: str,
    resource_context: Dict[str, Any],
    token: str = Depends(security)
):
    """Enforce security policies for environment"""
    try:
        orchestrator_instance = get_orchestrator()
        
        enforcement_result = await orchestrator_instance.policy_enforcement.enforce_policies(
            environment, resource_context
        )
        
        return {
            "enforcement_id": enforcement_result.enforcement_id,
            "environment": enforcement_result.environment,
            "timestamp": enforcement_result.timestamp.isoformat(),
            "policies_evaluated": enforcement_result.policies_evaluated,
            "violations_found": enforcement_result.violations_found,
            "violations": [
                {
                    "violation_id": v.violation_id,
                    "policy_id": v.policy_id,
                    "resource_id": v.resource_id,
                    "severity": v.severity.value,
                    "message": v.message,
                    "status": v.status.value
                } for v in enforcement_result.violations
            ],
            "actions_taken": enforcement_result.actions_taken,
            "execution_time": enforcement_result.execution_time
        }
        
    except Exception as e:
        logger.error(f"Policy enforcement failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/policy/violations")
async def get_policy_violations(
    environment: Optional[str] = None,
    status: Optional[str] = None,
    token: str = Depends(security)
):
    """Get policy violations with optional filters"""
    try:
        orchestrator_instance = get_orchestrator()
        
        from ...devsecops.security_policy_enforcement import ViolationStatus
        
        violation_status = ViolationStatus(status) if status else None
        
        violations = orchestrator_instance.policy_enforcement.get_policy_violations(
            environment, violation_status
        )
        
        return {
            "violations": [
                {
                    "violation_id": v.violation_id,
                    "policy_id": v.policy_id,
                    "resource_id": v.resource_id,
                    "resource_type": v.resource_type,
                    "environment": v.environment,
                    "severity": v.severity.value,
                    "message": v.message,
                    "status": v.status.value,
                    "detected_at": v.detected_at.isoformat()
                } for v in violations
            ],
            "total_count": len(violations)
        }
        
    except Exception as e:
        logger.error(f"Error getting policy violations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/policy/compliance/{environment}")
async def get_compliance_report(
    environment: str,
    token: str = Depends(security)
):
    """Get policy compliance report for environment"""
    try:
        orchestrator_instance = get_orchestrator()
        
        compliance_report = orchestrator_instance.policy_enforcement.get_policy_compliance_report(environment)
        
        return compliance_report
        
    except Exception as e:
        logger.error(f"Error getting compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Automated rollback endpoints
@router.post("/rollback/register")
async def register_service_for_rollback(
    request: ServiceRegistrationRequest,
    token: str = Depends(security)
):
    """Register service for automated rollback monitoring"""
    try:
        orchestrator_instance = get_orchestrator()
        
        rollback_config = await orchestrator_instance.rollback_system.register_service(
            request.service_name, request.environment, request.rollback_config
        )
        
        return {
            "service_name": rollback_config.service_name,
            "environment": rollback_config.environment,
            "strategy": rollback_config.strategy.value,
            "auto_rollback_enabled": rollback_config.auto_rollback_enabled,
            "rollback_triggers": [t.value for t in rollback_config.rollback_triggers],
            "message": "Service registered for rollback monitoring"
        }
        
    except Exception as e:
        logger.error(f"Service registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rollback/manual")
async def trigger_manual_rollback(
    request: ManualRollbackRequest,
    initiated_by: str = "api-user",
    token: str = Depends(security)
):
    """Trigger manual rollback"""
    try:
        orchestrator_instance = get_orchestrator()
        
        rollback_execution = await orchestrator_instance.rollback_system.manual_rollback(
            request.service_name,
            request.environment,
            request.target_version,
            request.reason,
            initiated_by
        )
        
        return {
            "rollback_id": rollback_execution.rollback_id,
            "service_name": rollback_execution.service_name,
            "environment": rollback_execution.environment,
            "target_version": rollback_execution.target_version,
            "status": rollback_execution.status.value,
            "start_time": rollback_execution.start_time.isoformat(),
            "message": "Manual rollback initiated"
        }
        
    except Exception as e:
        logger.error(f"Manual rollback failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rollback/{rollback_id}/status")
async def get_rollback_status(
    rollback_id: str,
    token: str = Depends(security)
):
    """Get rollback execution status"""
    try:
        orchestrator_instance = get_orchestrator()
        
        rollback_execution = await orchestrator_instance.rollback_system.get_rollback_status(rollback_id)
        
        if not rollback_execution:
            raise HTTPException(status_code=404, detail="Rollback not found")
        
        return {
            "rollback_id": rollback_execution.rollback_id,
            "service_name": rollback_execution.service_name,
            "environment": rollback_execution.environment,
            "trigger": rollback_execution.trigger.value,
            "strategy": rollback_execution.strategy.value,
            "status": rollback_execution.status.value,
            "start_time": rollback_execution.start_time.isoformat(),
            "end_time": rollback_execution.end_time.isoformat() if rollback_execution.end_time else None,
            "current_version": rollback_execution.current_version,
            "target_version": rollback_execution.target_version,
            "steps_completed": rollback_execution.steps_completed,
            "error_message": rollback_execution.error_message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting rollback status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rollback/active")
async def list_active_rollbacks(
    token: str = Depends(security)
):
    """List all active rollbacks"""
    try:
        orchestrator_instance = get_orchestrator()
        
        active_rollbacks = orchestrator_instance.rollback_system.list_active_rollbacks()
        
        return {
            "active_rollbacks": [
                {
                    "rollback_id": r.rollback_id,
                    "service_name": r.service_name,
                    "environment": r.environment,
                    "status": r.status.value,
                    "start_time": r.start_time.isoformat()
                } for r in active_rollbacks
            ],
            "total_count": len(active_rollbacks)
        }
        
    except Exception as e:
        logger.error(f"Error listing active rollbacks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard and reporting endpoints
@router.get("/dashboard/security")
async def get_security_dashboard(
    token: str = Depends(security)
):
    """Get comprehensive security dashboard data"""
    try:
        orchestrator_instance = get_orchestrator()
        
        dashboard_data = await orchestrator_instance.get_security_dashboard()
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting security dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "devsecops-pipeline-integration"
    }