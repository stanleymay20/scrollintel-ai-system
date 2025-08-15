"""
ScrollComplianceAgent - Regulatory Compliance and Auditing
GDPR, SOC2, ISO, and comprehensive regulatory compliance automation.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum
import logging

from scrollintel.core.interfaces import BaseAgent, AgentType, AgentRequest, AgentResponse, AgentCapability, ResponseStatus

logger = logging.getLogger(__name__)


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    NIST = "nist"
    SOX = "sox"
    EU_AI_ACT = "eu_ai_act"
    PIPEDA = "pipeda"
    LGPD = "lgpd"


class ComplianceStatus(str, Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
    IN_PROGRESS = "in_progress"
    NOT_APPLICABLE = "not_applicable"


class RiskLevel(str, Enum):
    """Risk levels for compliance issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class AuditType(str, Enum):
    """Types of compliance audits."""
    INTERNAL = "internal"
    EXTERNAL = "external"
    SELF_ASSESSMENT = "self_assessment"
    THIRD_PARTY = "third_party"
    REGULATORY = "regulatory"
    CONTINUOUS = "continuous"


@dataclass
class ComplianceRequirement:
    """Individual compliance requirement."""
    id: str
    framework: ComplianceFramework
    requirement_id: str
    title: str
    description: str
    category: str
    mandatory: bool
    evidence_required: List[str]
    testing_procedures: List[str]
    remediation_guidance: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ComplianceAssessment:
    """Compliance assessment result."""
    id: str
    framework: ComplianceFramework
    assessment_date: datetime
    assessor: str
    scope: str
    requirements_assessed: List[str]
    overall_status: ComplianceStatus
    compliance_score: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    remediation_plan: Dict[str, Any]
    next_assessment_date: datetime
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ComplianceFinding:
    """Individual compliance finding."""
    id: str
    assessment_id: str
    requirement_id: str
    finding_type: str  # "gap", "weakness", "non_compliance", "observation"
    severity: RiskLevel
    description: str
    evidence: List[str]
    impact: str
    recommendation: str
    remediation_effort: str
    due_date: Optional[datetime] = None
    status: str = "open"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ComplianceReport:
    """Comprehensive compliance report."""
    id: str
    title: str
    frameworks: List[ComplianceFramework]
    report_type: str
    executive_summary: str
    detailed_findings: List[ComplianceFinding]
    compliance_matrix: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    remediation_roadmap: Dict[str, Any]
    appendices: List[str]
    generated_at: datetime = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.utcnow()


class ScrollComplianceAgent(BaseAgent):
    """Advanced compliance and regulatory auditing agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="scroll-compliance-agent",
            name="ScrollCompliance Agent",
            agent_type=AgentType.AI_ENGINEER
        )
        
        self.capabilities = [
            AgentCapability(
                name="compliance_assessment",
                description="Conduct comprehensive compliance assessments against regulatory frameworks",
                input_types=["system_documentation", "policies", "procedures"],
                output_types=["compliance_report", "gap_analysis", "remediation_plan"]
            ),
            AgentCapability(
                name="regulatory_monitoring",
                description="Monitor regulatory changes and assess impact on compliance",
                input_types=["regulatory_updates", "system_changes"],
                output_types=["impact_analysis", "compliance_alerts", "action_items"]
            ),
            AgentCapability(
                name="audit_preparation",
                description="Prepare for regulatory audits and assessments",
                input_types=["audit_scope", "framework_requirements"],
                output_types=["audit_readiness_report", "evidence_package", "response_templates"]
            ),
            AgentCapability(
                name="policy_generation",
                description="Generate compliance policies and procedures",
                input_types=["framework_requirements", "organizational_context"],
                output_types=["policy_documents", "procedures", "training_materials"]
            )
        ]
        
        # Compliance state
        self.active_assessments = {}
        self.compliance_requirements = {}
        self.assessment_history = []
        self.compliance_findings = {}
        
        # Initialize compliance frameworks
        self._initialize_compliance_frameworks()
    
    def _initialize_compliance_frameworks(self):
        """Initialize compliance framework requirements."""
        self.framework_requirements = {
            ComplianceFramework.GDPR: self._get_gdpr_requirements(),
            ComplianceFramework.SOC2: self._get_soc2_requirements(),
            ComplianceFramework.HIPAA: self._get_hipaa_requirements(),
            ComplianceFramework.ISO27001: self._get_iso27001_requirements(),
            ComplianceFramework.EU_AI_ACT: self._get_eu_ai_act_requirements()
        }
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process compliance requests."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            prompt = request.prompt.lower()
            context = request.context or {}
            
            if "assess" in prompt or "audit" in prompt:
                content = await self._conduct_compliance_assessment(request.prompt, context)
            elif "gdpr" in prompt:
                content = await self._assess_gdpr_compliance(request.prompt, context)
            elif "soc2" in prompt:
                content = await self._assess_soc2_compliance(request.prompt, context)
            elif "policy" in prompt or "procedure" in prompt:
                content = await self._generate_compliance_policies(request.prompt, context)
            elif "monitor" in prompt or "track" in prompt:
                content = await self._monitor_compliance_status(request.prompt, context)
            elif "report" in prompt:
                content = await self._generate_compliance_report(request.prompt, context)
            else:
                content = await self._general_compliance_analysis(request.prompt, context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return AgentResponse(
                id=f"compliance-{uuid4()}",
                request_id=request.id,
                content=content,
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.SUCCESS
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"compliance-{uuid4()}",
                request_id=request.id,
                content=f"Error in compliance analysis: {str(e)}",
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )
    
    async def _conduct_compliance_assessment(self, prompt: str, context: Dict[str, Any]) -> str:
        """Conduct comprehensive compliance assessment."""
        frameworks = context.get("frameworks", [ComplianceFramework.GDPR])
        scope = context.get("scope", "Full system assessment")
        assessment_type = context.get("assessment_type", AuditType.SELF_ASSESSMENT)
        
        # Ensure frameworks are enum instances
        frameworks = [ComplianceFramework(f) if isinstance(f, str) else f for f in frameworks]
        
        # Create assessment
        assessment = ComplianceAssessment(
            id=f"assessment-{uuid4()}",
            framework=frameworks[0],  # Primary framework
            assessment_date=datetime.utcnow(),
            assessor=context.get("assessor", "ScrollCompliance Agent"),
            scope=scope,
            requirements_assessed=[],
            overall_status=ComplianceStatus.IN_PROGRESS,
            compliance_score=0.0,
            findings=[],
            recommendations=[],
            remediation_plan={},
            next_assessment_date=datetime.utcnow() + timedelta(days=365)
        )
        
        # Assess each framework
        all_findings = []
        total_score = 0.0
        
        for framework in frameworks:
            framework_findings = await self._assess_framework_compliance(framework, context)
            all_findings.extend(framework_findings)
            
            # Calculate framework score
            framework_score = await self._calculate_framework_score(framework_findings)
            total_score += framework_score
        
        # Update assessment
        assessment.findings = all_findings
        assessment.compliance_score = total_score / len(frameworks)
        assessment.overall_status = await self._determine_overall_status(assessment.compliance_score)
        assessment.recommendations = await self._generate_assessment_recommendations(all_findings)
        assessment.remediation_plan = await self._create_remediation_plan(all_findings)
        
        # Store assessment
        self.active_assessments[assessment.id] = assessment
        self.assessment_history.append(assessment)
        
        return f"""
# Compliance Assessment Report

## Assessment Overview
- **Assessment ID**: {assessment.id}
- **Frameworks Assessed**: {[f.value for f in frameworks]}
- **Scope**: {assessment.scope}
- **Assessment Date**: {assessment.assessment_date.strftime('%Y-%m-%d')}
- **Assessor**: {assessment.assessor}

## Overall Compliance Status
- **Status**: {assessment.overall_status.value.upper()}
- **Compliance Score**: {assessment.compliance_score:.1f}%
- **Next Assessment**: {assessment.next_assessment_date.strftime('%Y-%m-%d')}

## Framework-Specific Results
{await self._format_framework_results(frameworks, all_findings)}

## Key Findings Summary
{await self._summarize_key_findings(all_findings)}

## Risk Assessment
{await self._assess_compliance_risks(all_findings)}

## Recommendations
{chr(10).join(f"- {rec}" for rec in assessment.recommendations)}

## Remediation Roadmap
{await self._format_remediation_roadmap(assessment.remediation_plan)}

## Next Steps
{await self._suggest_compliance_next_steps(assessment)}
"""
    
    async def _assess_gdpr_compliance(self, prompt: str, context: Dict[str, Any]) -> str:
        """Assess GDPR compliance specifically."""
        system_data = context.get("system_data", {})
        data_processing_activities = context.get("data_processing_activities", [])
        
        # GDPR-specific assessment
        gdpr_findings = await self._assess_framework_compliance(ComplianceFramework.GDPR, context)
        
        # GDPR-specific analysis
        gdpr_analysis = {
            "lawful_basis": await self._assess_lawful_basis(data_processing_activities),
            "data_subject_rights": await self._assess_data_subject_rights(system_data),
            "privacy_by_design": await self._assess_privacy_by_design(system_data),
            "data_protection_impact": await self._assess_dpia_requirements(data_processing_activities),
            "international_transfers": await self._assess_international_transfers(system_data),
            "breach_notification": await self._assess_breach_procedures(system_data)
        }
        
        return f"""
# GDPR Compliance Assessment

## GDPR Compliance Overview
- **Assessment Date**: {datetime.utcnow().strftime('%Y-%m-%d')}
- **Scope**: {context.get('scope', 'Full GDPR assessment')}
- **Data Processing Activities**: {len(data_processing_activities)}

## Article-by-Article Assessment
{await self._format_gdpr_articles_assessment(gdpr_findings)}

## Key GDPR Areas Analysis

### Lawful Basis for Processing
{gdpr_analysis['lawful_basis']}

### Data Subject Rights
{gdpr_analysis['data_subject_rights']}

### Privacy by Design and Default
{gdpr_analysis['privacy_by_design']}

### Data Protection Impact Assessment (DPIA)
{gdpr_analysis['data_protection_impact']}

### International Data Transfers
{gdpr_analysis['international_transfers']}

### Personal Data Breach Procedures
{gdpr_analysis['breach_notification']}

## GDPR Compliance Score
{await self._calculate_gdpr_score(gdpr_findings)}

## Priority Actions
{await self._prioritize_gdpr_actions(gdpr_findings)}

## Documentation Requirements
{await self._list_gdpr_documentation_requirements()}
"""
    
    async def _assess_soc2_compliance(self, prompt: str, context: Dict[str, Any]) -> str:
        """Assess SOC2 compliance specifically."""
        system_controls = context.get("system_controls", {})
        trust_service_criteria = context.get("trust_service_criteria", ["security", "availability"])
        
        # SOC2-specific assessment
        soc2_findings = await self._assess_framework_compliance(ComplianceFramework.SOC2, context)
        
        # SOC2 Trust Service Criteria analysis
        criteria_analysis = {}
        for criterion in trust_service_criteria:
            criteria_analysis[criterion] = await self._assess_trust_service_criterion(criterion, system_controls)
        
        return f"""
# SOC2 Compliance Assessment

## SOC2 Overview
- **Assessment Date**: {datetime.utcnow().strftime('%Y-%m-%d')}
- **Trust Service Criteria**: {trust_service_criteria}
- **Assessment Type**: {context.get('soc2_type', 'Type II')}

## Trust Service Criteria Assessment
{await self._format_trust_service_criteria(criteria_analysis)}

## Control Environment Analysis
{await self._analyze_control_environment(system_controls)}

## SOC2 Findings
{await self._format_soc2_findings(soc2_findings)}

## Control Deficiencies
{await self._identify_control_deficiencies(soc2_findings)}

## Remediation Timeline
{await self._create_soc2_remediation_timeline(soc2_findings)}

## Audit Readiness
{await self._assess_soc2_audit_readiness(criteria_analysis)}
"""
    
    async def _assess_framework_compliance(self, framework: ComplianceFramework, context: Dict[str, Any]) -> List[ComplianceFinding]:
        """Assess compliance against a specific framework."""
        findings = []
        
        # Get framework requirements
        requirements = self.framework_requirements.get(framework, [])
        
        for requirement in requirements:
            # Mock assessment - in production would analyze actual system
            finding = ComplianceFinding(
                id=f"finding-{uuid4()}",
                assessment_id=f"assessment-{uuid4()}",
                requirement_id=requirement["id"],
                finding_type="gap" if requirement["mandatory"] else "observation",
                severity=RiskLevel.MEDIUM,
                description=f"Assessment of {requirement['title']}",
                evidence=["System documentation review", "Policy analysis"],
                impact="Medium impact on compliance posture",
                recommendation=f"Implement controls for {requirement['title']}",
                remediation_effort="2-4 weeks"
            )
            findings.append(finding)
        
        return findings
    
    def _get_gdpr_requirements(self) -> List[Dict[str, Any]]:
        """Get GDPR requirements."""
        return [
            {
                "id": "gdpr_art_6",
                "title": "Lawfulness of processing",
                "description": "Processing shall be lawful only if and to the extent that at least one legal basis applies",
                "mandatory": True,
                "category": "Legal Basis"
            },
            {
                "id": "gdpr_art_7",
                "title": "Conditions for consent",
                "description": "Where processing is based on consent, demonstrate that consent has been given",
                "mandatory": True,
                "category": "Consent"
            },
            {
                "id": "gdpr_art_25",
                "title": "Data protection by design and by default",
                "description": "Implement appropriate technical and organisational measures",
                "mandatory": True,
                "category": "Privacy by Design"
            },
            {
                "id": "gdpr_art_32",
                "title": "Security of processing",
                "description": "Implement appropriate technical and organisational measures to ensure security",
                "mandatory": True,
                "category": "Security"
            },
            {
                "id": "gdpr_art_33",
                "title": "Notification of a personal data breach",
                "description": "Notify supervisory authority of breach within 72 hours",
                "mandatory": True,
                "category": "Breach Notification"
            }
        ]
    
    def _get_soc2_requirements(self) -> List[Dict[str, Any]]:
        """Get SOC2 requirements."""
        return [
            {
                "id": "cc1_1",
                "title": "Control Environment - Integrity and Ethical Values",
                "description": "Demonstrates commitment to integrity and ethical values",
                "mandatory": True,
                "category": "Control Environment"
            },
            {
                "id": "cc2_1",
                "title": "Communication and Information - Internal Communication",
                "description": "Communicates information internally to support functioning of internal control",
                "mandatory": True,
                "category": "Communication"
            },
            {
                "id": "cc6_1",
                "title": "Logical and Physical Access Controls - Logical Access",
                "description": "Implements logical access security software and infrastructure",
                "mandatory": True,
                "category": "Access Controls"
            },
            {
                "id": "cc7_1",
                "title": "System Operations - System Monitoring",
                "description": "Monitors system components and operation of controls",
                "mandatory": True,
                "category": "Monitoring"
            }
        ]
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities."""
        return self.capabilities
    
    async def health_check(self) -> bool:
        """Check agent health."""
        return True
    
    # Helper methods (simplified implementations)
    async def _calculate_framework_score(self, findings: List[ComplianceFinding]) -> float:
        """Calculate compliance score for a framework."""
        if not findings:
            return 100.0
        
        total_findings = len(findings)
        critical_findings = sum(1 for f in findings if f.severity == RiskLevel.CRITICAL)
        high_findings = sum(1 for f in findings if f.severity == RiskLevel.HIGH)
        
        # Simple scoring algorithm
        score = 100.0 - (critical_findings * 20) - (high_findings * 10) - ((total_findings - critical_findings - high_findings) * 5)
        return max(0.0, score)
    
    async def _determine_overall_status(self, score: float) -> ComplianceStatus:
        """Determine overall compliance status based on score."""
        if score >= 90:
            return ComplianceStatus.COMPLIANT
        elif score >= 70:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT
    
    # Placeholder implementations for complex methods
    async def _assess_lawful_basis(self, activities: List[Any]) -> str:
        """Assess lawful basis for data processing."""
        return "Lawful basis assessment: Consent and legitimate interest identified for most processing activities"
    
    async def _assess_data_subject_rights(self, system_data: Dict[str, Any]) -> str:
        """Assess data subject rights implementation."""
        return "Data subject rights: Procedures in place for access, rectification, and erasure requests"
    
    async def _assess_privacy_by_design(self, system_data: Dict[str, Any]) -> str:
        """Assess privacy by design implementation."""
        return "Privacy by design: Technical measures implemented, organizational measures need improvement"