"""
Infrastructure Change Review and Approval Workflows with Security Assessment
Implements automated workflows for infrastructure changes with security validation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import json
import yaml
import hashlib
import git
from pathlib import Path

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    INFRASTRUCTURE = "infrastructure"
    CONFIGURATION = "configuration"
    SECURITY_POLICY = "security_policy"
    NETWORK = "network"
    ACCESS_CONTROL = "access_control"
    COMPLIANCE = "compliance"

class ChangeRisk(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ChangeStatus(Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    SECURITY_ASSESSMENT = "security_assessment"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    ROLLED_BACK = "rolled_back"

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"

@dataclass
class SecurityAssessment:
    assessment_id: str
    change_id: str
    risk_score: float
    security_findings: List[Dict[str, Any]]
    compliance_impact: List[str]
    recommendations: List[str]
    assessor: str
    timestamp: datetime
    automated_checks: Dict[str, Any]

@dataclass
class ChangeApproval:
    approver: str
    status: ApprovalStatus
    comments: str
    timestamp: datetime
    conditions: List[str] = field(default_factory=list)

@dataclass
class InfrastructureChange:
    change_id: str
    title: str
    description: str
    change_type: ChangeType
    risk_level: ChangeRisk
    submitter: str
    submission_time: datetime
    status: ChangeStatus
    files_changed: List[str]
    security_assessment: Optional[SecurityAssessment]
    approvals: List[ChangeApproval]
    required_approvers: List[str]
    implementation_plan: Dict[str, Any]
    rollback_plan: Dict[str, Any]
    testing_plan: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

class InfrastructureChangeWorkflows:
    """
    Manages infrastructure change review and approval workflows with security assessment
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_changes = {}
        self.approval_rules = self._load_approval_rules()
        self.security_assessors = self._load_security_assessors()
        self.change_templates = self._load_change_templates()
        
    def _load_approval_rules(self) -> Dict[str, Any]:
        """Load approval rules based on change type and risk"""
        return {
            "infrastructure": {
                "low": ["team-lead"],
                "medium": ["team-lead", "infrastructure-lead"],
                "high": ["team-lead", "infrastructure-lead", "security-team"],
                "critical": ["team-lead", "infrastructure-lead", "security-team", "cto"]
            },
            "security_policy": {
                "low": ["security-team"],
                "medium": ["security-team", "security-lead"],
                "high": ["security-team", "security-lead", "ciso"],
                "critical": ["security-team", "security-lead", "ciso", "cto"]
            },
            "network": {
                "low": ["network-admin"],
                "medium": ["network-admin", "infrastructure-lead"],
                "high": ["network-admin", "infrastructure-lead", "security-team"],
                "critical": ["network-admin", "infrastructure-lead", "security-team", "ciso"]
            },
            "access_control": {
                "low": ["security-team"],
                "medium": ["security-team", "security-lead"],
                "high": ["security-team", "security-lead", "ciso"],
                "critical": ["security-team", "security-lead", "ciso", "compliance-officer"]
            },
            "compliance": {
                "low": ["compliance-officer"],
                "medium": ["compliance-officer", "security-team"],
                "high": ["compliance-officer", "security-team", "ciso"],
                "critical": ["compliance-officer", "security-team", "ciso", "legal-team"]
            }
        }
    
    def _load_security_assessors(self) -> Dict[str, Callable]:
        """Load security assessment functions"""
        return {
            "terraform_security_scan": self._assess_terraform_security,
            "kubernetes_security_scan": self._assess_kubernetes_security,
            "network_security_scan": self._assess_network_security,
            "iam_security_scan": self._assess_iam_security,
            "compliance_impact_assessment": self._assess_compliance_impact,
            "risk_analysis": self._assess_risk_analysis
        }
    
    def _load_change_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load change request templates"""
        return {
            "infrastructure": {
                "required_fields": [
                    "title", "description", "change_type", "files_changed",
                    "implementation_plan", "rollback_plan", "testing_plan"
                ],
                "security_checks": [
                    "terraform_security_scan", "compliance_impact_assessment"
                ]
            },
            "security_policy": {
                "required_fields": [
                    "title", "description", "policy_changes", "impact_analysis",
                    "implementation_plan", "rollback_plan"
                ],
                "security_checks": [
                    "iam_security_scan", "compliance_impact_assessment", "risk_analysis"
                ]
            },
            "network": {
                "required_fields": [
                    "title", "description", "network_changes", "security_impact",
                    "implementation_plan", "rollback_plan", "testing_plan"
                ],
                "security_checks": [
                    "network_security_scan", "compliance_impact_assessment"
                ]
            }
        }
    
    async def submit_change_request(
        self, 
        change_data: Dict[str, Any],
        submitter: str
    ) -> InfrastructureChange:
        """Submit a new infrastructure change request"""
        change_id = self._generate_change_id(change_data)
        
        # Validate change request
        self._validate_change_request(change_data)
        
        # Determine risk level
        risk_level = await self._assess_change_risk(change_data)
        
        # Create change request
        change = InfrastructureChange(
            change_id=change_id,
            title=change_data["title"],
            description=change_data["description"],
            change_type=ChangeType(change_data["change_type"]),
            risk_level=risk_level,
            submitter=submitter,
            submission_time=datetime.now(),
            status=ChangeStatus.SUBMITTED,
            files_changed=change_data.get("files_changed", []),
            security_assessment=None,
            approvals=[],
            required_approvers=self._get_required_approvers(
                ChangeType(change_data["change_type"]), risk_level
            ),
            implementation_plan=change_data.get("implementation_plan", {}),
            rollback_plan=change_data.get("rollback_plan", {}),
            testing_plan=change_data.get("testing_plan", {}),
            metadata=change_data.get("metadata", {})
        )
        
        self.active_changes[change_id] = change
        
        # Trigger security assessment
        await self._trigger_security_assessment(change)
        
        logger.info(f"Change request {change_id} submitted by {submitter}")
        return change
    
    def _generate_change_id(self, change_data: Dict[str, Any]) -> str:
        """Generate unique change ID"""
        content = json.dumps(change_data, sort_keys=True)
        hash_object = hashlib.md5(content.encode())
        timestamp = int(datetime.now().timestamp())
        return f"CHG-{timestamp}-{hash_object.hexdigest()[:8]}"
    
    def _validate_change_request(self, change_data: Dict[str, Any]):
        """Validate change request data"""
        change_type = change_data.get("change_type")
        if not change_type or change_type not in [ct.value for ct in ChangeType]:
            raise ValueError(f"Invalid change type: {change_type}")
        
        template = self.change_templates.get(change_type, {})
        required_fields = template.get("required_fields", [])
        
        for field in required_fields:
            if field not in change_data:
                raise ValueError(f"Required field missing: {field}")
    
    async def _assess_change_risk(self, change_data: Dict[str, Any]) -> ChangeRisk:
        """Assess risk level of change"""
        risk_factors = []
        
        # File-based risk assessment
        files_changed = change_data.get("files_changed", [])
        
        # High-risk files
        high_risk_patterns = [
            "security", "iam", "rbac", "network", "firewall",
            "production", "prod", "critical"
        ]
        
        for file_path in files_changed:
            file_lower = file_path.lower()
            if any(pattern in file_lower for pattern in high_risk_patterns):
                risk_factors.append("high_risk_file")
        
        # Change type risk
        change_type = ChangeType(change_data["change_type"])
        if change_type in [ChangeType.SECURITY_POLICY, ChangeType.ACCESS_CONTROL]:
            risk_factors.append("security_change")
        
        # Scope assessment
        if len(files_changed) > 10:
            risk_factors.append("large_scope")
        
        # Determine overall risk
        if len(risk_factors) >= 3:
            return ChangeRisk.CRITICAL
        elif len(risk_factors) >= 2:
            return ChangeRisk.HIGH
        elif len(risk_factors) >= 1:
            return ChangeRisk.MEDIUM
        else:
            return ChangeRisk.LOW
    
    def _get_required_approvers(
        self, 
        change_type: ChangeType,
        risk_level: ChangeRisk
    ) -> List[str]:
        """Get required approvers based on change type and risk"""
        type_key = change_type.value
        risk_key = risk_level.value
        
        return self.approval_rules.get(type_key, {}).get(risk_key, ["team-lead"])
    
    async def _trigger_security_assessment(self, change: InfrastructureChange):
        """Trigger automated security assessment"""
        change.status = ChangeStatus.SECURITY_ASSESSMENT
        
        assessment_id = f"SA-{change.change_id}"
        
        # Run security checks based on change type
        template = self.change_templates.get(change.change_type.value, {})
        security_checks = template.get("security_checks", [])
        
        all_findings = []
        all_recommendations = []
        compliance_impacts = []
        automated_checks = {}
        
        for check_name in security_checks:
            if check_name in self.security_assessors:
                logger.info(f"Running security check: {check_name}")
                
                try:
                    assessor = self.security_assessors[check_name]
                    result = await assessor(change)
                    
                    all_findings.extend(result.get("findings", []))
                    all_recommendations.extend(result.get("recommendations", []))
                    compliance_impacts.extend(result.get("compliance_impact", []))
                    automated_checks[check_name] = result
                    
                except Exception as e:
                    logger.error(f"Security check {check_name} failed: {str(e)}")
                    automated_checks[check_name] = {"error": str(e)}
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(all_findings, change.risk_level)
        
        # Create security assessment
        assessment = SecurityAssessment(
            assessment_id=assessment_id,
            change_id=change.change_id,
            risk_score=risk_score,
            security_findings=all_findings,
            compliance_impact=list(set(compliance_impacts)),
            recommendations=list(set(all_recommendations)),
            assessor="automated-security-assessment",
            timestamp=datetime.now(),
            automated_checks=automated_checks
        )
        
        change.security_assessment = assessment
        change.status = ChangeStatus.UNDER_REVIEW
        
        # Notify approvers
        await self._notify_approvers(change)
        
        logger.info(f"Security assessment completed for change {change.change_id}")
    
    async def _assess_terraform_security(self, change: InfrastructureChange) -> Dict[str, Any]:
        """Assess Terraform configuration security"""
        findings = []
        recommendations = []
        compliance_impact = []
        
        # Simulate Terraform security scan
        await asyncio.sleep(2)
        
        terraform_files = [f for f in change.files_changed if f.endswith('.tf')]
        
        for tf_file in terraform_files:
            # Simulate security checks
            if "aws_s3_bucket" in tf_file:
                findings.append({
                    "severity": "medium",
                    "type": "s3_security",
                    "file": tf_file,
                    "message": "S3 bucket may not have proper encryption",
                    "line": 15
                })
                recommendations.append("Enable S3 bucket encryption")
                compliance_impact.append("SOC2")
            
            if "aws_security_group" in tf_file:
                findings.append({
                    "severity": "high",
                    "type": "network_security",
                    "file": tf_file,
                    "message": "Security group allows broad access",
                    "line": 8
                })
                recommendations.append("Restrict security group rules to minimum required access")
                compliance_impact.append("ISO27001")
        
        return {
            "findings": findings,
            "recommendations": recommendations,
            "compliance_impact": compliance_impact,
            "scan_tool": "tfsec",
            "files_scanned": len(terraform_files)
        }
    
    async def _assess_kubernetes_security(self, change: InfrastructureChange) -> Dict[str, Any]:
        """Assess Kubernetes configuration security"""
        findings = []
        recommendations = []
        compliance_impact = []
        
        await asyncio.sleep(1)
        
        k8s_files = [f for f in change.files_changed if f.endswith(('.yaml', '.yml'))]
        
        for k8s_file in k8s_files:
            # Simulate Kubernetes security checks
            if "deployment" in k8s_file.lower():
                findings.append({
                    "severity": "medium",
                    "type": "pod_security",
                    "file": k8s_file,
                    "message": "Pod may be running as root",
                    "line": 25
                })
                recommendations.append("Configure pod to run as non-root user")
                compliance_impact.append("CIS_Kubernetes")
        
        return {
            "findings": findings,
            "recommendations": recommendations,
            "compliance_impact": compliance_impact,
            "scan_tool": "kube-score",
            "files_scanned": len(k8s_files)
        }
    
    async def _assess_network_security(self, change: InfrastructureChange) -> Dict[str, Any]:
        """Assess network security impact"""
        findings = []
        recommendations = []
        compliance_impact = []
        
        await asyncio.sleep(1)
        
        # Simulate network security assessment
        network_changes = change.metadata.get("network_changes", [])
        
        for network_change in network_changes:
            if "firewall" in network_change.get("type", "").lower():
                findings.append({
                    "severity": "high",
                    "type": "firewall_rule",
                    "message": "New firewall rule may expose services",
                    "details": network_change
                })
                recommendations.append("Review firewall rule necessity and scope")
                compliance_impact.append("PCI_DSS")
        
        return {
            "findings": findings,
            "recommendations": recommendations,
            "compliance_impact": compliance_impact,
            "assessment_type": "network_security"
        }
    
    async def _assess_iam_security(self, change: InfrastructureChange) -> Dict[str, Any]:
        """Assess IAM and access control security"""
        findings = []
        recommendations = []
        compliance_impact = []
        
        await asyncio.sleep(1)
        
        # Simulate IAM security assessment
        iam_changes = change.metadata.get("policy_changes", [])
        
        for iam_change in iam_changes:
            if "admin" in iam_change.get("permissions", []):
                findings.append({
                    "severity": "critical",
                    "type": "excessive_permissions",
                    "message": "Policy grants administrative permissions",
                    "details": iam_change
                })
                recommendations.append("Apply principle of least privilege")
                compliance_impact.extend(["SOC2", "ISO27001"])
        
        return {
            "findings": findings,
            "recommendations": recommendations,
            "compliance_impact": compliance_impact,
            "assessment_type": "iam_security"
        }
    
    async def _assess_compliance_impact(self, change: InfrastructureChange) -> Dict[str, Any]:
        """Assess compliance framework impact"""
        findings = []
        recommendations = []
        compliance_impact = []
        
        await asyncio.sleep(1)
        
        # Assess impact on various compliance frameworks
        frameworks = ["SOC2", "GDPR", "HIPAA", "PCI_DSS", "ISO27001"]
        
        for framework in frameworks:
            # Simulate compliance impact assessment
            if change.change_type == ChangeType.SECURITY_POLICY:
                compliance_impact.append(framework)
                recommendations.append(f"Review {framework} compliance requirements")
        
        return {
            "findings": findings,
            "recommendations": recommendations,
            "compliance_impact": compliance_impact,
            "frameworks_assessed": frameworks
        }
    
    async def _assess_risk_analysis(self, change: InfrastructureChange) -> Dict[str, Any]:
        """Perform comprehensive risk analysis"""
        findings = []
        recommendations = []
        compliance_impact = []
        
        await asyncio.sleep(2)
        
        # Simulate risk analysis
        risk_factors = {
            "data_exposure": 0.3,
            "service_disruption": 0.2,
            "compliance_violation": 0.4,
            "security_breach": 0.5
        }
        
        for risk_type, probability in risk_factors.items():
            if probability > 0.3:
                findings.append({
                    "severity": "medium" if probability < 0.5 else "high",
                    "type": "risk_factor",
                    "message": f"Elevated risk of {risk_type}",
                    "probability": probability
                })
                recommendations.append(f"Implement mitigation for {risk_type} risk")
        
        return {
            "findings": findings,
            "recommendations": recommendations,
            "compliance_impact": compliance_impact,
            "risk_factors": risk_factors
        }
    
    def _calculate_risk_score(
        self, 
        findings: List[Dict[str, Any]],
        base_risk: ChangeRisk
    ) -> float:
        """Calculate overall risk score"""
        base_scores = {
            ChangeRisk.LOW: 20.0,
            ChangeRisk.MEDIUM: 40.0,
            ChangeRisk.HIGH: 60.0,
            ChangeRisk.CRITICAL: 80.0
        }
        
        base_score = base_scores[base_risk]
        
        # Add points for findings
        for finding in findings:
            severity = finding.get("severity", "low")
            if severity == "critical":
                base_score += 15
            elif severity == "high":
                base_score += 10
            elif severity == "medium":
                base_score += 5
            elif severity == "low":
                base_score += 2
        
        return min(100.0, base_score)
    
    async def _notify_approvers(self, change: InfrastructureChange):
        """Notify required approvers"""
        logger.info(f"Notifying approvers for change {change.change_id}: {change.required_approvers}")
        
        # In a real implementation, this would send notifications via email, Slack, etc.
        notification_data = {
            "change_id": change.change_id,
            "title": change.title,
            "submitter": change.submitter,
            "risk_level": change.risk_level.value,
            "security_score": change.security_assessment.risk_score if change.security_assessment else 0,
            "required_action": "review_and_approve"
        }
        
        # Simulate notification sending
        await asyncio.sleep(0.1)
    
    async def submit_approval(
        self, 
        change_id: str,
        approver: str,
        status: ApprovalStatus,
        comments: str = "",
        conditions: List[str] = None
    ) -> bool:
        """Submit approval decision for change request"""
        if change_id not in self.active_changes:
            raise ValueError(f"Change {change_id} not found")
        
        change = self.active_changes[change_id]
        
        if approver not in change.required_approvers:
            raise ValueError(f"Approver {approver} not authorized for change {change_id}")
        
        # Check if already approved by this approver
        existing_approval = next(
            (a for a in change.approvals if a.approver == approver), None
        )
        
        if existing_approval:
            # Update existing approval
            existing_approval.status = status
            existing_approval.comments = comments
            existing_approval.timestamp = datetime.now()
            existing_approval.conditions = conditions or []
        else:
            # Add new approval
            approval = ChangeApproval(
                approver=approver,
                status=status,
                comments=comments,
                timestamp=datetime.now(),
                conditions=conditions or []
            )
            change.approvals.append(approval)
        
        # Check if change is fully approved or rejected
        await self._evaluate_change_status(change)
        
        logger.info(f"Approval submitted by {approver} for change {change_id}: {status.value}")
        return True
    
    async def _evaluate_change_status(self, change: InfrastructureChange):
        """Evaluate overall change status based on approvals"""
        approvals_by_approver = {a.approver: a for a in change.approvals}
        
        # Check if any approver rejected
        rejected_approvals = [a for a in change.approvals if a.status == ApprovalStatus.REJECTED]
        if rejected_approvals:
            change.status = ChangeStatus.REJECTED
            await self._notify_change_status_update(change, "rejected")
            return
        
        # Check if all required approvers have approved
        approved_count = 0
        conditional_count = 0
        
        for required_approver in change.required_approvers:
            approval = approvals_by_approver.get(required_approver)
            if approval:
                if approval.status == ApprovalStatus.APPROVED:
                    approved_count += 1
                elif approval.status == ApprovalStatus.CONDITIONAL:
                    conditional_count += 1
        
        total_required = len(change.required_approvers)
        
        if approved_count == total_required:
            change.status = ChangeStatus.APPROVED
            await self._notify_change_status_update(change, "approved")
        elif approved_count + conditional_count == total_required:
            # All approvers responded, some with conditions
            change.status = ChangeStatus.APPROVED
            await self._notify_change_status_update(change, "conditionally_approved")
    
    async def _notify_change_status_update(self, change: InfrastructureChange, status: str):
        """Notify stakeholders of change status update"""
        logger.info(f"Change {change.change_id} status updated to: {status}")
        
        # In a real implementation, this would send notifications
        notification_data = {
            "change_id": change.change_id,
            "title": change.title,
            "status": status,
            "submitter": change.submitter,
            "approvals": [
                {
                    "approver": a.approver,
                    "status": a.status.value,
                    "comments": a.comments
                } for a in change.approvals
            ]
        }
        
        await asyncio.sleep(0.1)
    
    async def implement_change(self, change_id: str, implementer: str) -> bool:
        """Implement approved change"""
        if change_id not in self.active_changes:
            raise ValueError(f"Change {change_id} not found")
        
        change = self.active_changes[change_id]
        
        if change.status != ChangeStatus.APPROVED:
            raise ValueError(f"Change {change_id} is not approved for implementation")
        
        logger.info(f"Implementing change {change_id} by {implementer}")
        
        try:
            # Execute implementation plan
            implementation_result = await self._execute_implementation_plan(change)
            
            if implementation_result["success"]:
                change.status = ChangeStatus.IMPLEMENTED
                change.metadata["implementation_result"] = implementation_result
                change.metadata["implementer"] = implementer
                change.metadata["implementation_time"] = datetime.now().isoformat()
                
                logger.info(f"Change {change_id} implemented successfully")
                return True
            else:
                logger.error(f"Change {change_id} implementation failed: {implementation_result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Change {change_id} implementation error: {str(e)}")
            return False
    
    async def _execute_implementation_plan(self, change: InfrastructureChange) -> Dict[str, Any]:
        """Execute the implementation plan"""
        implementation_plan = change.implementation_plan
        
        # Simulate implementation execution
        await asyncio.sleep(5)
        
        # In a real implementation, this would execute the actual changes
        # such as applying Terraform, updating Kubernetes resources, etc.
        
        return {
            "success": True,
            "steps_executed": implementation_plan.get("steps", []),
            "execution_time": 5.0,
            "changes_applied": change.files_changed
        }
    
    async def rollback_change(self, change_id: str, rollback_reason: str) -> bool:
        """Rollback implemented change"""
        if change_id not in self.active_changes:
            raise ValueError(f"Change {change_id} not found")
        
        change = self.active_changes[change_id]
        
        if change.status != ChangeStatus.IMPLEMENTED:
            raise ValueError(f"Change {change_id} is not in implemented status")
        
        logger.info(f"Rolling back change {change_id}: {rollback_reason}")
        
        try:
            # Execute rollback plan
            rollback_result = await self._execute_rollback_plan(change)
            
            if rollback_result["success"]:
                change.status = ChangeStatus.ROLLED_BACK
                change.metadata["rollback_result"] = rollback_result
                change.metadata["rollback_reason"] = rollback_reason
                change.metadata["rollback_time"] = datetime.now().isoformat()
                
                logger.info(f"Change {change_id} rolled back successfully")
                return True
            else:
                logger.error(f"Change {change_id} rollback failed: {rollback_result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Change {change_id} rollback error: {str(e)}")
            return False
    
    async def _execute_rollback_plan(self, change: InfrastructureChange) -> Dict[str, Any]:
        """Execute the rollback plan"""
        rollback_plan = change.rollback_plan
        
        # Simulate rollback execution
        await asyncio.sleep(3)
        
        return {
            "success": True,
            "steps_executed": rollback_plan.get("steps", []),
            "execution_time": 3.0,
            "changes_reverted": change.files_changed
        }
    
    def get_change_status(self, change_id: str) -> Optional[InfrastructureChange]:
        """Get status of change request"""
        return self.active_changes.get(change_id)
    
    def list_changes_by_status(self, status: ChangeStatus) -> List[InfrastructureChange]:
        """List changes by status"""
        return [
            change for change in self.active_changes.values()
            if change.status == status
        ]
    
    def list_pending_approvals(self, approver: str) -> List[InfrastructureChange]:
        """List changes pending approval from specific approver"""
        pending_changes = []
        
        for change in self.active_changes.values():
            if (change.status == ChangeStatus.UNDER_REVIEW and
                approver in change.required_approvers):
                
                # Check if this approver hasn't approved yet
                existing_approval = next(
                    (a for a in change.approvals if a.approver == approver), None
                )
                
                if not existing_approval:
                    pending_changes.append(change)
        
        return pending_changes
    
    async def generate_change_report(self, change_id: str) -> Dict[str, Any]:
        """Generate comprehensive change report"""
        if change_id not in self.active_changes:
            raise ValueError(f"Change {change_id} not found")
        
        change = self.active_changes[change_id]
        
        return {
            "change_summary": {
                "id": change.change_id,
                "title": change.title,
                "type": change.change_type.value,
                "risk_level": change.risk_level.value,
                "status": change.status.value,
                "submitter": change.submitter,
                "submission_time": change.submission_time.isoformat()
            },
            "security_assessment": {
                "risk_score": change.security_assessment.risk_score if change.security_assessment else 0,
                "findings_count": len(change.security_assessment.security_findings) if change.security_assessment else 0,
                "compliance_impact": change.security_assessment.compliance_impact if change.security_assessment else [],
                "recommendations": change.security_assessment.recommendations if change.security_assessment else []
            },
            "approval_status": {
                "required_approvers": change.required_approvers,
                "approvals_received": len(change.approvals),
                "approval_details": [
                    {
                        "approver": a.approver,
                        "status": a.status.value,
                        "timestamp": a.timestamp.isoformat(),
                        "comments": a.comments
                    } for a in change.approvals
                ]
            },
            "implementation": {
                "files_changed": change.files_changed,
                "implementation_plan": change.implementation_plan,
                "rollback_plan": change.rollback_plan,
                "testing_plan": change.testing_plan
            },
            "metadata": change.metadata
        }