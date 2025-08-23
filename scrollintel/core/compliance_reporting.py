"""
Compliance Reporting and Governance Integration
Handles compliance reporting for various frameworks (SOX, GDPR, HIPAA, etc.)
"""
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from jinja2 import Template
import pandas as pd

from ..models.security_audit_models import (
    ComplianceReport, SecurityAuditLog, ComplianceReportCreate,
    SeverityLevel, SecurityEventType
)
from ..core.database_connection_manager import get_sync_session
from ..core.logging_config import get_logger

logger = get_logger(__name__)

class ComplianceFramework(str, Enum):
    """Supported compliance frameworks"""
    SOX = "sox"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    CCPA = "ccpa"
    SOC2 = "soc2"

class ComplianceStatus(str, Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"

@dataclass
class ComplianceRequirement:
    """Individual compliance requirement"""
    requirement_id: str
    framework: ComplianceFramework
    title: str
    description: str
    control_objectives: List[str]
    evidence_requirements: List[str]
    testing_procedures: List[str]
    severity: str

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    requirement_id: str
    description: str
    severity: SeverityLevel
    detected_at: datetime
    evidence: List[Dict[str, Any]]
    remediation_steps: List[str]
    status: str

@dataclass
class ComplianceMetrics:
    """Compliance metrics and KPIs"""
    framework: str
    overall_score: float
    compliant_controls: int
    total_controls: int
    violations_count: int
    critical_violations: int
    last_assessment: datetime

class ComplianceReportingEngine:
    """Comprehensive compliance reporting and governance engine"""
    
    def __init__(self):
        self.compliance_requirements = self._initialize_compliance_requirements()
        self.report_templates = self._initialize_report_templates()
    
    def _initialize_compliance_requirements(self) -> Dict[ComplianceFramework, List[ComplianceRequirement]]:
        """Initialize compliance requirements for different frameworks"""
        requirements = {
            ComplianceFramework.SOX: [
                ComplianceRequirement(
                    requirement_id="SOX-302",
                    framework=ComplianceFramework.SOX,
                    title="Management Assessment of Internal Controls",
                    description="Management must assess and report on internal controls over financial reporting",
                    control_objectives=[
                        "Ensure accuracy of financial data",
                        "Prevent unauthorized access to financial systems",
                        "Maintain audit trails for financial transactions"
                    ],
                    evidence_requirements=[
                        "Access logs for financial systems",
                        "Change management records",
                        "User access reviews"
                    ],
                    testing_procedures=[
                        "Review access controls quarterly",
                        "Test segregation of duties",
                        "Validate audit trail completeness"
                    ],
                    severity="high"
                ),
                ComplianceRequirement(
                    requirement_id="SOX-404",
                    framework=ComplianceFramework.SOX,
                    title="Internal Control Assessment",
                    description="Annual assessment of internal control effectiveness",
                    control_objectives=[
                        "Document internal control procedures",
                        "Test control effectiveness",
                        "Report control deficiencies"
                    ],
                    evidence_requirements=[
                        "Control documentation",
                        "Testing results",
                        "Deficiency reports"
                    ],
                    testing_procedures=[
                        "Annual control testing",
                        "Management review",
                        "External audit validation"
                    ],
                    severity="critical"
                )
            ],
            ComplianceFramework.GDPR: [
                ComplianceRequirement(
                    requirement_id="GDPR-Art32",
                    framework=ComplianceFramework.GDPR,
                    title="Security of Processing",
                    description="Implement appropriate technical and organizational measures",
                    control_objectives=[
                        "Ensure confidentiality of personal data",
                        "Maintain integrity of personal data",
                        "Ensure availability of personal data",
                        "Implement access controls"
                    ],
                    evidence_requirements=[
                        "Security policies and procedures",
                        "Access control logs",
                        "Encryption implementation",
                        "Incident response records"
                    ],
                    testing_procedures=[
                        "Security assessment",
                        "Penetration testing",
                        "Access review",
                        "Incident response testing"
                    ],
                    severity="high"
                ),
                ComplianceRequirement(
                    requirement_id="GDPR-Art33",
                    framework=ComplianceFramework.GDPR,
                    title="Notification of Personal Data Breach",
                    description="Notify supervisory authority of data breaches within 72 hours",
                    control_objectives=[
                        "Detect data breaches promptly",
                        "Assess breach impact",
                        "Notify authorities timely",
                        "Document breach response"
                    ],
                    evidence_requirements=[
                        "Breach detection logs",
                        "Impact assessments",
                        "Notification records",
                        "Response documentation"
                    ],
                    testing_procedures=[
                        "Breach simulation exercises",
                        "Response time testing",
                        "Notification process validation"
                    ],
                    severity="critical"
                )
            ],
            ComplianceFramework.HIPAA: [
                ComplianceRequirement(
                    requirement_id="HIPAA-164.312",
                    framework=ComplianceFramework.HIPAA,
                    title="Technical Safeguards",
                    description="Implement technical safeguards for PHI",
                    control_objectives=[
                        "Control access to PHI",
                        "Audit access to PHI",
                        "Ensure data integrity",
                        "Implement encryption"
                    ],
                    evidence_requirements=[
                        "Access control policies",
                        "Audit logs",
                        "Encryption documentation",
                        "User access reviews"
                    ],
                    testing_procedures=[
                        "Access control testing",
                        "Audit log review",
                        "Encryption validation"
                    ],
                    severity="high"
                )
            ]
        }
        return requirements
    
    def _initialize_report_templates(self) -> Dict[str, str]:
        """Initialize report templates for different frameworks"""
        return {
            "executive_summary": """
# Compliance Report - {{ framework }}
## Executive Summary

**Report Period:** {{ period_start }} to {{ period_end }}
**Generated:** {{ generated_date }}
**Overall Compliance Score:** {{ overall_score }}%

### Key Findings
{% for finding in key_findings %}
- {{ finding }}
{% endfor %}

### Critical Issues
{% for issue in critical_issues %}
- **{{ issue.title }}:** {{ issue.description }}
{% endfor %}

### Recommendations
{% for recommendation in recommendations %}
- {{ recommendation }}
{% endfor %}
            """,
            "detailed_findings": """
## Detailed Compliance Assessment

{% for requirement in requirements %}
### {{ requirement.requirement_id }}: {{ requirement.title }}

**Status:** {{ requirement.status }}
**Score:** {{ requirement.score }}%

**Control Objectives:**
{% for objective in requirement.control_objectives %}
- {{ objective }}
{% endfor %}

**Findings:**
{% for finding in requirement.findings %}
- {{ finding }}
{% endfor %}

**Evidence Reviewed:**
{% for evidence in requirement.evidence %}
- {{ evidence }}
{% endfor %}

{% if requirement.violations %}
**Violations:**
{% for violation in requirement.violations %}
- **{{ violation.severity }}:** {{ violation.description }}
{% endfor %}
{% endif %}

---
{% endfor %}
            """,
            "remediation_plan": """
## Remediation Plan

{% for violation in violations %}
### {{ violation.violation_id }}: {{ violation.description }}

**Severity:** {{ violation.severity }}
**Detected:** {{ violation.detected_at }}

**Remediation Steps:**
{% for step in violation.remediation_steps %}
{{ loop.index }}. {{ step }}
{% endfor %}

**Timeline:** {{ violation.timeline }}
**Owner:** {{ violation.owner }}

---
{% endfor %}
            """
        }
    
    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        period_start: datetime,
        period_end: datetime,
        report_type: str = "full",
        generated_by: Optional[str] = None
    ) -> str:
        """Generate comprehensive compliance report"""
        try:
            report_id = str(uuid.uuid4())
            
            # Collect compliance data
            compliance_data = await self._collect_compliance_data(
                framework, period_start, period_end
            )
            
            # Assess compliance status
            assessment_results = await self._assess_compliance_requirements(
                framework, compliance_data
            )
            
            # Identify violations
            violations = await self._identify_violations(
                framework, compliance_data, assessment_results
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                framework, assessment_results, violations
            )
            
            # Calculate overall score
            overall_score = self._calculate_compliance_score(assessment_results)
            
            # Create report content
            report_content = await self._create_report_content(
                framework=framework,
                period_start=period_start,
                period_end=period_end,
                assessment_results=assessment_results,
                violations=violations,
                recommendations=recommendations,
                overall_score=overall_score,
                report_type=report_type
            )
            
            # Save report to database
            report = ComplianceReport(
                id=report_id,
                report_type=report_type,
                framework=framework.value,
                period_start=period_start,
                period_end=period_end,
                status="completed",
                findings=assessment_results,
                violations=[v.__dict__ for v in violations],
                recommendations=recommendations,
                generated_by=generated_by
            )
            
            with get_sync_session() as db:
                db.add(report)
                db.commit()
            
            logger.info(f"Compliance report generated: {report_id}")
            return report_id
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {str(e)}")
            raise
    
    async def _collect_compliance_data(
        self,
        framework: ComplianceFramework,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Collect relevant data for compliance assessment"""
        try:
            with get_sync_session() as db:
                # Get all security events in the period
                security_events = db.query(SecurityAuditLog).filter(
                    SecurityAuditLog.timestamp >= period_start,
                    SecurityAuditLog.timestamp <= period_end
                ).all()
                
                # Categorize events by type
                events_by_type = {}
                for event in security_events:
                    event_type = event.event_type
                    if event_type not in events_by_type:
                        events_by_type[event_type] = []
                    events_by_type[event_type].append(event)
                
                # Get access control data
                access_events = events_by_type.get(SecurityEventType.AUTHENTICATION.value, [])
                authorization_events = events_by_type.get(SecurityEventType.AUTHORIZATION.value, [])
                data_access_events = events_by_type.get(SecurityEventType.DATA_ACCESS.value, [])
                config_changes = events_by_type.get(SecurityEventType.CONFIGURATION_CHANGE.value, [])
                
                # Calculate metrics
                failed_logins = len([e for e in access_events if e.outcome == "failure"])
                successful_logins = len([e for e in access_events if e.outcome == "success"])
                unauthorized_access_attempts = len([e for e in authorization_events if e.outcome == "failure"])
                
                return {
                    "total_events": len(security_events),
                    "events_by_type": events_by_type,
                    "access_metrics": {
                        "failed_logins": failed_logins,
                        "successful_logins": successful_logins,
                        "unauthorized_attempts": unauthorized_access_attempts
                    },
                    "data_access_events": len(data_access_events),
                    "configuration_changes": len(config_changes),
                    "period_start": period_start,
                    "period_end": period_end
                }
                
        except Exception as e:
            logger.error(f"Failed to collect compliance data: {str(e)}")
            return {}
    
    async def _assess_compliance_requirements(
        self,
        framework: ComplianceFramework,
        compliance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess compliance against framework requirements"""
        try:
            requirements = self.compliance_requirements.get(framework, [])
            assessment_results = {}
            
            for requirement in requirements:
                result = await self._assess_individual_requirement(
                    requirement, compliance_data
                )
                assessment_results[requirement.requirement_id] = result
            
            return assessment_results
            
        except Exception as e:
            logger.error(f"Failed to assess compliance requirements: {str(e)}")
            return {}
    
    async def _assess_individual_requirement(
        self,
        requirement: ComplianceRequirement,
        compliance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess individual compliance requirement"""
        try:
            # Framework-specific assessment logic
            if requirement.framework == ComplianceFramework.SOX:
                return await self._assess_sox_requirement(requirement, compliance_data)
            elif requirement.framework == ComplianceFramework.GDPR:
                return await self._assess_gdpr_requirement(requirement, compliance_data)
            elif requirement.framework == ComplianceFramework.HIPAA:
                return await self._assess_hipaa_requirement(requirement, compliance_data)
            else:
                return await self._assess_generic_requirement(requirement, compliance_data)
                
        except Exception as e:
            logger.error(f"Failed to assess requirement {requirement.requirement_id}: {str(e)}")
            return {
                "status": ComplianceStatus.UNDER_REVIEW.value,
                "score": 0,
                "findings": [f"Assessment failed: {str(e)}"],
                "evidence": [],
                "violations": []
            }
    
    async def _assess_sox_requirement(
        self,
        requirement: ComplianceRequirement,
        compliance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess SOX-specific requirements"""
        findings = []
        violations = []
        score = 100
        
        if requirement.requirement_id == "SOX-302":
            # Check access controls for financial systems
            access_metrics = compliance_data.get("access_metrics", {})
            failed_logins = access_metrics.get("failed_logins", 0)
            unauthorized_attempts = access_metrics.get("unauthorized_attempts", 0)
            
            if failed_logins > 100:  # Threshold for concern
                findings.append(f"High number of failed login attempts: {failed_logins}")
                score -= 20
            
            if unauthorized_attempts > 10:
                findings.append(f"Unauthorized access attempts detected: {unauthorized_attempts}")
                violations.append({
                    "description": "Excessive unauthorized access attempts",
                    "severity": "medium",
                    "count": unauthorized_attempts
                })
                score -= 30
            
            # Check for proper audit trails
            config_changes = compliance_data.get("configuration_changes", 0)
            if config_changes == 0:
                findings.append("No configuration changes logged - verify audit trail completeness")
                score -= 10
            
        elif requirement.requirement_id == "SOX-404":
            # Annual control assessment
            findings.append("Manual review required for annual control assessment")
            score = 85  # Assume partial compliance pending manual review
        
        status = ComplianceStatus.COMPLIANT.value if score >= 90 else \
                ComplianceStatus.PARTIALLY_COMPLIANT.value if score >= 70 else \
                ComplianceStatus.NON_COMPLIANT.value
        
        return {
            "status": status,
            "score": score,
            "findings": findings,
            "evidence": [
                f"Access events analyzed: {compliance_data.get('access_metrics', {}).get('successful_logins', 0)}",
                f"Configuration changes: {compliance_data.get('configuration_changes', 0)}"
            ],
            "violations": violations
        }
    
    async def _assess_gdpr_requirement(
        self,
        requirement: ComplianceRequirement,
        compliance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess GDPR-specific requirements"""
        findings = []
        violations = []
        score = 100
        
        if requirement.requirement_id == "GDPR-Art32":
            # Security of processing
            data_access_events = compliance_data.get("data_access_events", 0)
            
            # Check for proper access controls
            if data_access_events > 0:
                findings.append(f"Data access events recorded: {data_access_events}")
            else:
                findings.append("No data access events - verify monitoring completeness")
                score -= 15
            
            # Check for security incidents
            events_by_type = compliance_data.get("events_by_type", {})
            security_incidents = events_by_type.get(SecurityEventType.SYSTEM_BREACH.value, [])
            
            if security_incidents:
                violations.append({
                    "description": f"Security incidents detected: {len(security_incidents)}",
                    "severity": "high",
                    "count": len(security_incidents)
                })
                score -= 40
        
        elif requirement.requirement_id == "GDPR-Art33":
            # Breach notification
            events_by_type = compliance_data.get("events_by_type", {})
            breaches = events_by_type.get(SecurityEventType.SYSTEM_BREACH.value, [])
            
            if breaches:
                # Check if breaches were handled within 72 hours
                for breach in breaches:
                    # This would need more sophisticated tracking
                    findings.append(f"Breach detected at {breach.timestamp} - verify notification timeline")
            else:
                findings.append("No data breaches detected in reporting period")
        
        status = ComplianceStatus.COMPLIANT.value if score >= 90 else \
                ComplianceStatus.PARTIALLY_COMPLIANT.value if score >= 70 else \
                ComplianceStatus.NON_COMPLIANT.value
        
        return {
            "status": status,
            "score": score,
            "findings": findings,
            "evidence": [
                f"Data access events: {compliance_data.get('data_access_events', 0)}",
                f"Security incidents: {len(compliance_data.get('events_by_type', {}).get(SecurityEventType.SYSTEM_BREACH.value, []))}"
            ],
            "violations": violations
        }
    
    async def _assess_hipaa_requirement(
        self,
        requirement: ComplianceRequirement,
        compliance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess HIPAA-specific requirements"""
        findings = []
        violations = []
        score = 100
        
        if requirement.requirement_id == "HIPAA-164.312":
            # Technical safeguards
            access_metrics = compliance_data.get("access_metrics", {})
            unauthorized_attempts = access_metrics.get("unauthorized_attempts", 0)
            
            if unauthorized_attempts > 5:  # Lower threshold for PHI
                violations.append({
                    "description": f"Unauthorized PHI access attempts: {unauthorized_attempts}",
                    "severity": "high",
                    "count": unauthorized_attempts
                })
                score -= 50
            
            # Check audit trail completeness
            total_events = compliance_data.get("total_events", 0)
            if total_events == 0:
                findings.append("No audit events recorded - verify audit system functionality")
                score -= 30
        
        status = ComplianceStatus.COMPLIANT.value if score >= 95 else \
                ComplianceStatus.PARTIALLY_COMPLIANT.value if score >= 80 else \
                ComplianceStatus.NON_COMPLIANT.value
        
        return {
            "status": status,
            "score": score,
            "findings": findings,
            "evidence": [
                f"Total audit events: {compliance_data.get('total_events', 0)}",
                f"Unauthorized attempts: {unauthorized_attempts}"
            ],
            "violations": violations
        }
    
    async def _assess_generic_requirement(
        self,
        requirement: ComplianceRequirement,
        compliance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess generic compliance requirement"""
        return {
            "status": ComplianceStatus.UNDER_REVIEW.value,
            "score": 75,
            "findings": ["Manual review required for this requirement"],
            "evidence": ["Automated assessment not available"],
            "violations": []
        }
    
    async def _identify_violations(
        self,
        framework: ComplianceFramework,
        compliance_data: Dict[str, Any],
        assessment_results: Dict[str, Any]
    ) -> List[ComplianceViolation]:
        """Identify compliance violations"""
        violations = []
        
        try:
            for requirement_id, result in assessment_results.items():
                for violation_data in result.get("violations", []):
                    violation = ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        requirement_id=requirement_id,
                        description=violation_data["description"],
                        severity=SeverityLevel(violation_data["severity"]),
                        detected_at=datetime.utcnow(),
                        evidence=[{
                            "type": "audit_data",
                            "description": f"Count: {violation_data.get('count', 0)}"
                        }],
                        remediation_steps=self._get_remediation_steps(
                            framework, requirement_id, violation_data
                        ),
                        status="open"
                    )
                    violations.append(violation)
            
            return violations
            
        except Exception as e:
            logger.error(f"Failed to identify violations: {str(e)}")
            return []
    
    def _get_remediation_steps(
        self,
        framework: ComplianceFramework,
        requirement_id: str,
        violation_data: Dict[str, Any]
    ) -> List[str]:
        """Get remediation steps for violation"""
        remediation_map = {
            "SOX-302": [
                "Review and strengthen access controls",
                "Implement additional monitoring",
                "Conduct user access review",
                "Update security policies"
            ],
            "GDPR-Art32": [
                "Implement additional security measures",
                "Conduct security assessment",
                "Review data processing activities",
                "Update privacy impact assessments"
            ],
            "HIPAA-164.312": [
                "Strengthen PHI access controls",
                "Implement additional audit logging",
                "Conduct security risk assessment",
                "Update technical safeguards"
            ]
        }
        
        return remediation_map.get(requirement_id, [
            "Review compliance requirement",
            "Implement corrective measures",
            "Monitor for improvement",
            "Document remediation actions"
        ])
    
    def _generate_recommendations(
        self,
        framework: ComplianceFramework,
        assessment_results: Dict[str, Any],
        violations: List[ComplianceViolation]
    ) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # General recommendations based on violations
        if violations:
            critical_violations = [v for v in violations if v.severity == SeverityLevel.CRITICAL]
            high_violations = [v for v in violations if v.severity == SeverityLevel.HIGH]
            
            if critical_violations:
                recommendations.append(f"Address {len(critical_violations)} critical compliance violations immediately")
            
            if high_violations:
                recommendations.append(f"Prioritize remediation of {len(high_violations)} high-severity violations")
        
        # Framework-specific recommendations
        if framework == ComplianceFramework.SOX:
            recommendations.extend([
                "Implement quarterly access reviews",
                "Enhance segregation of duties controls",
                "Strengthen change management processes"
            ])
        elif framework == ComplianceFramework.GDPR:
            recommendations.extend([
                "Conduct privacy impact assessments",
                "Implement data minimization practices",
                "Enhance breach detection capabilities"
            ])
        elif framework == ComplianceFramework.HIPAA:
            recommendations.extend([
                "Implement role-based access controls for PHI",
                "Enhance audit logging for all PHI access",
                "Conduct regular security risk assessments"
            ])
        
        # Assessment-based recommendations
        low_scores = [req_id for req_id, result in assessment_results.items() 
                     if result.get("score", 0) < 80]
        
        if low_scores:
            recommendations.append(f"Focus improvement efforts on requirements: {', '.join(low_scores)}")
        
        return recommendations
    
    def _calculate_compliance_score(self, assessment_results: Dict[str, Any]) -> float:
        """Calculate overall compliance score"""
        if not assessment_results:
            return 0.0
        
        total_score = sum(result.get("score", 0) for result in assessment_results.values())
        return round(total_score / len(assessment_results), 2)
    
    async def _create_report_content(
        self,
        framework: ComplianceFramework,
        period_start: datetime,
        period_end: datetime,
        assessment_results: Dict[str, Any],
        violations: List[ComplianceViolation],
        recommendations: List[str],
        overall_score: float,
        report_type: str
    ) -> str:
        """Create formatted report content"""
        try:
            # Prepare template data
            template_data = {
                "framework": framework.value.upper(),
                "period_start": period_start.strftime("%Y-%m-%d"),
                "period_end": period_end.strftime("%Y-%m-%d"),
                "generated_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "overall_score": overall_score,
                "key_findings": [
                    f"Overall compliance score: {overall_score}%",
                    f"Total requirements assessed: {len(assessment_results)}",
                    f"Violations identified: {len(violations)}",
                    f"Critical violations: {len([v for v in violations if v.severity == SeverityLevel.CRITICAL])}"
                ],
                "critical_issues": [
                    {
                        "title": v.requirement_id,
                        "description": v.description
                    }
                    for v in violations if v.severity == SeverityLevel.CRITICAL
                ],
                "recommendations": recommendations,
                "requirements": [
                    {
                        "requirement_id": req_id,
                        "title": f"Requirement {req_id}",
                        "status": result.get("status", "unknown"),
                        "score": result.get("score", 0),
                        "control_objectives": [],  # Would be populated from requirement definition
                        "findings": result.get("findings", []),
                        "evidence": result.get("evidence", []),
                        "violations": result.get("violations", [])
                    }
                    for req_id, result in assessment_results.items()
                ],
                "violations": [
                    {
                        "violation_id": v.violation_id,
                        "description": v.description,
                        "severity": v.severity.value,
                        "detected_at": v.detected_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "remediation_steps": v.remediation_steps,
                        "timeline": "30 days",  # Default timeline
                        "owner": "Security Team"  # Default owner
                    }
                    for v in violations
                ]
            }
            
            # Generate report sections
            executive_template = Template(self.report_templates["executive_summary"])
            executive_content = executive_template.render(**template_data)
            
            if report_type == "full":
                detailed_template = Template(self.report_templates["detailed_findings"])
                detailed_content = detailed_template.render(**template_data)
                
                remediation_template = Template(self.report_templates["remediation_plan"])
                remediation_content = remediation_template.render(**template_data)
                
                return f"{executive_content}\n\n{detailed_content}\n\n{remediation_content}"
            else:
                return executive_content
                
        except Exception as e:
            logger.error(f"Failed to create report content: {str(e)}")
            return f"Report generation failed: {str(e)}"
    
    def get_compliance_metrics(
        self,
        framework: ComplianceFramework,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> ComplianceMetrics:
        """Get compliance metrics for framework"""
        try:
            if not period_start:
                period_start = datetime.utcnow() - timedelta(days=90)
            if not period_end:
                period_end = datetime.utcnow()
            
            with get_sync_session() as db:
                # Get recent reports for this framework
                reports = db.query(ComplianceReport).filter(
                    ComplianceReport.framework == framework.value,
                    ComplianceReport.period_start >= period_start,
                    ComplianceReport.period_end <= period_end
                ).order_by(ComplianceReport.created_at.desc()).limit(5).all()
                
                if not reports:
                    return ComplianceMetrics(
                        framework=framework.value,
                        overall_score=0.0,
                        compliant_controls=0,
                        total_controls=0,
                        violations_count=0,
                        critical_violations=0,
                        last_assessment=datetime.utcnow()
                    )
                
                # Calculate metrics from most recent report
                latest_report = reports[0]
                findings = latest_report.findings or {}
                violations = latest_report.violations or []
                
                total_controls = len(findings)
                compliant_controls = len([f for f in findings.values() 
                                        if f.get("status") == ComplianceStatus.COMPLIANT.value])
                
                overall_score = sum(f.get("score", 0) for f in findings.values()) / total_controls if total_controls > 0 else 0
                
                critical_violations = len([v for v in violations 
                                         if v.get("severity") == SeverityLevel.CRITICAL.value])
                
                return ComplianceMetrics(
                    framework=framework.value,
                    overall_score=round(overall_score, 2),
                    compliant_controls=compliant_controls,
                    total_controls=total_controls,
                    violations_count=len(violations),
                    critical_violations=critical_violations,
                    last_assessment=latest_report.created_at
                )
                
        except Exception as e:
            logger.error(f"Failed to get compliance metrics: {str(e)}")
            return ComplianceMetrics(
                framework=framework.value,
                overall_score=0.0,
                compliant_controls=0,
                total_controls=0,
                violations_count=0,
                critical_violations=0,
                last_assessment=datetime.utcnow()
            )

# Global compliance reporting engine
compliance_engine = ComplianceReportingEngine()