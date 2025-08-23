"""
DevSecOps Pipeline Security Gates Implementation
Integrates security gates into CI/CD pipeline with automated approval workflows
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import json
import yaml

logger = logging.getLogger(__name__)

class SecurityGateStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    BYPASSED = "bypassed"
    FAILED = "failed"

class SecurityGateType(Enum):
    SAST_SCAN = "sast_scan"
    DAST_SCAN = "dast_scan"
    CONTAINER_SCAN = "container_scan"
    DEPENDENCY_SCAN = "dependency_scan"
    COMPLIANCE_CHECK = "compliance_check"
    SECURITY_REVIEW = "security_review"
    PENETRATION_TEST = "penetration_test"

@dataclass
class SecurityGateResult:
    gate_type: SecurityGateType
    status: SecurityGateStatus
    score: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    execution_time: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class ApprovalWorkflow:
    workflow_id: str
    required_approvers: List[str]
    current_approvers: List[str]
    approval_threshold: int
    auto_approve_conditions: Dict[str, Any]
    escalation_rules: Dict[str, Any]
    timeout_minutes: int

class PipelineSecurityGates:
    """
    Implements security gates for CI/CD pipeline with automated approval workflows
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gate_results = {}
        self.approval_workflows = {}
        self.security_policies = self._load_security_policies()
        
    def _load_security_policies(self) -> Dict[str, Any]:
        """Load security policies from configuration"""
        return {
            "sast_thresholds": {
                "critical": 0,
                "high": 5,
                "medium": 20,
                "low": 50
            },
            "dast_thresholds": {
                "critical": 0,
                "high": 3,
                "medium": 15,
                "low": 30
            },
            "container_thresholds": {
                "critical": 0,
                "high": 2,
                "medium": 10,
                "low": 25
            },
            "dependency_thresholds": {
                "critical": 0,
                "high": 5,
                "medium": 15,
                "low": 40
            },
            "auto_approve_conditions": {
                "score_threshold": 85.0,
                "max_critical_findings": 0,
                "max_high_findings": 2,
                "trusted_authors": ["security-team", "lead-dev"]
            }
        }
    
    async def execute_security_gate(
        self, 
        gate_type: SecurityGateType,
        pipeline_context: Dict[str, Any]
    ) -> SecurityGateResult:
        """Execute a specific security gate"""
        start_time = datetime.now()
        
        try:
            if gate_type == SecurityGateType.SAST_SCAN:
                result = await self._execute_sast_scan(pipeline_context)
            elif gate_type == SecurityGateType.DAST_SCAN:
                result = await self._execute_dast_scan(pipeline_context)
            elif gate_type == SecurityGateType.CONTAINER_SCAN:
                result = await self._execute_container_scan(pipeline_context)
            elif gate_type == SecurityGateType.DEPENDENCY_SCAN:
                result = await self._execute_dependency_scan(pipeline_context)
            elif gate_type == SecurityGateType.COMPLIANCE_CHECK:
                result = await self._execute_compliance_check(pipeline_context)
            elif gate_type == SecurityGateType.SECURITY_REVIEW:
                result = await self._execute_security_review(pipeline_context)
            else:
                raise ValueError(f"Unsupported gate type: {gate_type}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # Store result for approval workflow
            gate_id = f"{pipeline_context.get('pipeline_id')}_{gate_type.value}"
            self.gate_results[gate_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Security gate {gate_type} failed: {str(e)}")
            return SecurityGateResult(
                gate_type=gate_type,
                status=SecurityGateStatus.FAILED,
                score=0.0,
                findings=[{"error": str(e)}],
                recommendations=["Fix security gate execution error"],
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )
    
    async def _execute_sast_scan(self, context: Dict[str, Any]) -> SecurityGateResult:
        """Execute Static Application Security Testing"""
        # Simulate SAST scan
        await asyncio.sleep(2)  # Simulate scan time
        
        findings = [
            {
                "severity": "medium",
                "type": "sql_injection",
                "file": "src/api/routes.py",
                "line": 45,
                "description": "Potential SQL injection vulnerability"
            },
            {
                "severity": "low",
                "type": "hardcoded_secret",
                "file": "src/config.py",
                "line": 12,
                "description": "Hardcoded API key detected"
            }
        ]
        
        score = self._calculate_security_score(findings, "sast")
        status = SecurityGateStatus.APPROVED if score >= 80 else SecurityGateStatus.PENDING
        
        return SecurityGateResult(
            gate_type=SecurityGateType.SAST_SCAN,
            status=status,
            score=score,
            findings=findings,
            recommendations=self._generate_recommendations(findings),
            execution_time=0.0,
            timestamp=datetime.now(),
            metadata={"scan_tool": "semgrep", "rules_version": "1.2.3"}
        )
    
    async def _execute_dast_scan(self, context: Dict[str, Any]) -> SecurityGateResult:
        """Execute Dynamic Application Security Testing"""
        await asyncio.sleep(5)  # Simulate scan time
        
        findings = [
            {
                "severity": "high",
                "type": "xss",
                "url": "/api/search",
                "parameter": "query",
                "description": "Cross-site scripting vulnerability"
            }
        ]
        
        score = self._calculate_security_score(findings, "dast")
        status = SecurityGateStatus.PENDING if findings else SecurityGateStatus.APPROVED
        
        return SecurityGateResult(
            gate_type=SecurityGateType.DAST_SCAN,
            status=status,
            score=score,
            findings=findings,
            recommendations=self._generate_recommendations(findings),
            execution_time=0.0,
            timestamp=datetime.now(),
            metadata={"scan_tool": "zap", "target_url": context.get("app_url")}
        )
    
    async def _execute_container_scan(self, context: Dict[str, Any]) -> SecurityGateResult:
        """Execute container vulnerability scanning"""
        await asyncio.sleep(3)  # Simulate scan time
        
        findings = [
            {
                "severity": "critical",
                "type": "vulnerability",
                "package": "openssl",
                "version": "1.1.1f",
                "cve": "CVE-2021-3711",
                "description": "Buffer overflow in OpenSSL"
            },
            {
                "severity": "medium",
                "type": "misconfiguration",
                "check": "USER_ROOT",
                "description": "Container running as root user"
            }
        ]
        
        score = self._calculate_security_score(findings, "container")
        status = SecurityGateStatus.REJECTED if any(f["severity"] == "critical" for f in findings) else SecurityGateStatus.APPROVED
        
        return SecurityGateResult(
            gate_type=SecurityGateType.CONTAINER_SCAN,
            status=status,
            score=score,
            findings=findings,
            recommendations=self._generate_recommendations(findings),
            execution_time=0.0,
            timestamp=datetime.now(),
            metadata={"scan_tool": "trivy", "image": context.get("container_image")}
        )
    
    async def _execute_dependency_scan(self, context: Dict[str, Any]) -> SecurityGateResult:
        """Execute dependency vulnerability scanning"""
        await asyncio.sleep(1)  # Simulate scan time
        
        findings = [
            {
                "severity": "high",
                "type": "vulnerability",
                "package": "requests",
                "version": "2.25.1",
                "cve": "CVE-2021-33503",
                "description": "Inefficient regular expression complexity"
            }
        ]
        
        score = self._calculate_security_score(findings, "dependency")
        status = SecurityGateStatus.APPROVED if score >= 75 else SecurityGateStatus.PENDING
        
        return SecurityGateResult(
            gate_type=SecurityGateType.DEPENDENCY_SCAN,
            status=status,
            score=score,
            findings=findings,
            recommendations=self._generate_recommendations(findings),
            execution_time=0.0,
            timestamp=datetime.now(),
            metadata={"scan_tool": "safety", "package_manager": "pip"}
        )
    
    async def _execute_compliance_check(self, context: Dict[str, Any]) -> SecurityGateResult:
        """Execute compliance checks"""
        await asyncio.sleep(1)  # Simulate check time
        
        findings = []
        score = 95.0
        
        return SecurityGateResult(
            gate_type=SecurityGateType.COMPLIANCE_CHECK,
            status=SecurityGateStatus.APPROVED,
            score=score,
            findings=findings,
            recommendations=[],
            execution_time=0.0,
            timestamp=datetime.now(),
            metadata={"frameworks": ["SOC2", "GDPR"], "compliance_score": score}
        )
    
    async def _execute_security_review(self, context: Dict[str, Any]) -> SecurityGateResult:
        """Execute security review process"""
        # This would typically require human review
        findings = []
        score = 90.0
        
        return SecurityGateResult(
            gate_type=SecurityGateType.SECURITY_REVIEW,
            status=SecurityGateStatus.PENDING,  # Always requires approval
            score=score,
            findings=findings,
            recommendations=["Security team review required"],
            execution_time=0.0,
            timestamp=datetime.now(),
            metadata={"reviewer_required": True}
        )
    
    def _calculate_security_score(self, findings: List[Dict[str, Any]], scan_type: str) -> float:
        """Calculate security score based on findings"""
        if not findings:
            return 100.0
        
        severity_weights = {
            "critical": 25,
            "high": 10,
            "medium": 5,
            "low": 1
        }
        
        total_penalty = sum(severity_weights.get(f.get("severity", "low"), 1) for f in findings)
        score = max(0, 100 - total_penalty)
        
        return score
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []
        
        for finding in findings:
            severity = finding.get("severity", "low")
            finding_type = finding.get("type", "unknown")
            
            if finding_type == "sql_injection":
                recommendations.append("Use parameterized queries to prevent SQL injection")
            elif finding_type == "xss":
                recommendations.append("Implement proper input validation and output encoding")
            elif finding_type == "vulnerability":
                recommendations.append(f"Update {finding.get('package')} to latest secure version")
            elif finding_type == "misconfiguration":
                recommendations.append("Review and fix container security configuration")
            elif finding_type == "hardcoded_secret":
                recommendations.append("Move secrets to secure configuration management")
        
        return recommendations
    
    async def create_approval_workflow(
        self, 
        gate_result: SecurityGateResult,
        pipeline_context: Dict[str, Any]
    ) -> ApprovalWorkflow:
        """Create approval workflow for security gate"""
        workflow_id = f"approval_{pipeline_context.get('pipeline_id')}_{gate_result.gate_type.value}"
        
        # Determine required approvers based on findings severity
        required_approvers = self._determine_required_approvers(gate_result)
        
        workflow = ApprovalWorkflow(
            workflow_id=workflow_id,
            required_approvers=required_approvers,
            current_approvers=[],
            approval_threshold=len(required_approvers) // 2 + 1,  # Majority approval
            auto_approve_conditions=self.security_policies["auto_approve_conditions"],
            escalation_rules={
                "timeout_escalation": ["security-lead", "cto"],
                "critical_findings_escalation": ["security-team", "ciso"]
            },
            timeout_minutes=240  # 4 hours
        )
        
        self.approval_workflows[workflow_id] = workflow
        
        # Check for auto-approval conditions
        if self._check_auto_approval_conditions(gate_result, pipeline_context):
            workflow.status = SecurityGateStatus.APPROVED
            workflow.current_approvers = ["auto-approved"]
        
        return workflow
    
    def _determine_required_approvers(self, gate_result: SecurityGateResult) -> List[str]:
        """Determine required approvers based on gate result"""
        approvers = []
        
        # Check for critical findings
        critical_findings = [f for f in gate_result.findings if f.get("severity") == "critical"]
        high_findings = [f for f in gate_result.findings if f.get("severity") == "high"]
        
        if critical_findings:
            approvers.extend(["security-team", "ciso"])
        elif high_findings:
            approvers.extend(["security-team", "security-lead"])
        elif gate_result.score < 80:
            approvers.append("security-lead")
        else:
            approvers.append("team-lead")
        
        return list(set(approvers))  # Remove duplicates
    
    def _check_auto_approval_conditions(
        self, 
        gate_result: SecurityGateResult,
        pipeline_context: Dict[str, Any]
    ) -> bool:
        """Check if gate result meets auto-approval conditions"""
        conditions = self.security_policies["auto_approve_conditions"]
        
        # Check score threshold
        if gate_result.score < conditions["score_threshold"]:
            return False
        
        # Check critical findings
        critical_count = len([f for f in gate_result.findings if f.get("severity") == "critical"])
        if critical_count > conditions["max_critical_findings"]:
            return False
        
        # Check high findings
        high_count = len([f for f in gate_result.findings if f.get("severity") == "high"])
        if high_count > conditions["max_high_findings"]:
            return False
        
        # Check trusted authors
        author = pipeline_context.get("author")
        if author in conditions["trusted_authors"]:
            return True
        
        return gate_result.score >= 95.0 and not gate_result.findings
    
    async def process_approval(
        self, 
        workflow_id: str,
        approver: str,
        decision: SecurityGateStatus,
        comments: str = ""
    ) -> bool:
        """Process approval decision"""
        if workflow_id not in self.approval_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.approval_workflows[workflow_id]
        
        if approver not in workflow.required_approvers:
            raise ValueError(f"Approver {approver} not authorized for this workflow")
        
        if decision == SecurityGateStatus.APPROVED:
            workflow.current_approvers.append(approver)
            
            # Check if approval threshold is met
            if len(workflow.current_approvers) >= workflow.approval_threshold:
                workflow.status = SecurityGateStatus.APPROVED
                return True
        elif decision == SecurityGateStatus.REJECTED:
            workflow.status = SecurityGateStatus.REJECTED
            return True
        
        return False
    
    async def get_pipeline_security_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get overall security status for pipeline"""
        gate_results = {
            k: v for k, v in self.gate_results.items() 
            if k.startswith(pipeline_id)
        }
        
        if not gate_results:
            return {"status": "no_gates_executed", "overall_score": 0.0}
        
        # Calculate overall score
        scores = [result.score for result in gate_results.values()]
        overall_score = sum(scores) / len(scores)
        
        # Determine overall status
        statuses = [result.status for result in gate_results.values()]
        
        if SecurityGateStatus.FAILED in statuses:
            overall_status = "failed"
        elif SecurityGateStatus.REJECTED in statuses:
            overall_status = "rejected"
        elif SecurityGateStatus.PENDING in statuses:
            overall_status = "pending_approval"
        elif all(status == SecurityGateStatus.APPROVED for status in statuses):
            overall_status = "approved"
        else:
            overall_status = "mixed"
        
        return {
            "status": overall_status,
            "overall_score": overall_score,
            "gate_results": {k: {
                "type": v.gate_type.value,
                "status": v.status.value,
                "score": v.score,
                "findings_count": len(v.findings)
            } for k, v in gate_results.items()},
            "recommendations": self._get_consolidated_recommendations(gate_results.values())
        }
    
    def _get_consolidated_recommendations(self, results) -> List[str]:
        """Get consolidated recommendations from all gate results"""
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations