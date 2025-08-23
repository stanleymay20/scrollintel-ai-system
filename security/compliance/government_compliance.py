"""
Government Compliance Module
Implements FedRAMP, FISMA, and classified data handling
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging

from .industry_compliance_framework import (
    BaseComplianceModule, ComplianceRule, ComplianceAssessment,
    ComplianceStatus, IndustryType
)

logger = logging.getLogger(__name__)

class GovernmentComplianceModule(BaseComplianceModule):
    """Government industry compliance module with FedRAMP, FISMA, and classified data handling controls"""
    
    def __init__(self):
        super().__init__(IndustryType.GOVERNMENT)
    
    def _load_rules(self):
        """Load government-specific compliance rules"""
        
        # FedRAMP Rules
        self.rules["FEDRAMP_AC"] = ComplianceRule(
            rule_id="FEDRAMP_AC",
            name="Access Control",
            description="Implement comprehensive access control measures",
            regulation="FedRAMP Moderate Baseline",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement role-based access control",
                "Deploy multi-factor authentication",
                "Regular access reviews and certifications"
            ],
            evidence_requirements=["access_control_policies", "mfa_configuration", "access_reviews"]
        )
        
        self.rules["FEDRAMP_AU"] = ComplianceRule(
            rule_id="FEDRAMP_AU",
            name="Audit and Accountability",
            description="Maintain comprehensive audit trails and accountability measures",
            regulation="FedRAMP Moderate Baseline",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement comprehensive audit logging",
                "Deploy log analysis and monitoring",
                "Maintain audit trail integrity"
            ],
            evidence_requirements=["audit_logs", "monitoring_configuration", "integrity_verification"]
        )
        
        self.rules["FEDRAMP_SC"] = ComplianceRule(
            rule_id="FEDRAMP_SC",
            name="System and Communications Protection",
            description="Protect system and communications integrity",
            regulation="FedRAMP Moderate Baseline",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement encryption for data in transit and at rest",
                "Deploy network security controls",
                "Maintain communication integrity"
            ],
            evidence_requirements=["encryption_configuration", "network_controls", "integrity_checks"]
        )
        
        self.rules["FEDRAMP_SI"] = ComplianceRule(
            rule_id="FEDRAMP_SI",
            name="System and Information Integrity",
            description="Maintain system and information integrity",
            regulation="FedRAMP Moderate Baseline",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Deploy malware protection",
                "Implement vulnerability scanning",
                "Maintain system integrity monitoring"
            ],
            evidence_requirements=["malware_protection", "vulnerability_scans", "integrity_monitoring"]
        )
        
        # FISMA Rules
        self.rules["FISMA_CATEGORIZATION"] = ComplianceRule(
            rule_id="FISMA_CATEGORIZATION",
            name="Information System Categorization",
            description="Categorize information systems according to FIPS 199",
            regulation="FISMA",
            severity="high",
            automated_check=False,
            remediation_steps=[
                "Conduct system categorization analysis",
                "Document security categorization",
                "Regular categorization reviews"
            ],
            evidence_requirements=["categorization_analysis", "categorization_documentation", "review_records"]
        )
        
        self.rules["FISMA_SECURITY_CONTROLS"] = ComplianceRule(
            rule_id="FISMA_SECURITY_CONTROLS",
            name="Security Control Implementation",
            description="Implement appropriate security controls based on system categorization",
            regulation="FISMA",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Select appropriate security controls",
                "Implement security controls",
                "Document control implementation"
            ],
            evidence_requirements=["control_selection", "implementation_documentation", "control_testing"]
        )
        
        self.rules["FISMA_CONTINUOUS_MONITORING"] = ComplianceRule(
            rule_id="FISMA_CONTINUOUS_MONITORING",
            name="Continuous Monitoring",
            description="Implement continuous monitoring of security controls",
            regulation="FISMA",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Deploy continuous monitoring tools",
                "Establish monitoring procedures",
                "Regular monitoring reports"
            ],
            evidence_requirements=["monitoring_tools", "monitoring_procedures", "monitoring_reports"]
        )
        
        # Classified Data Handling Rules
        self.rules["CLASSIFIED_STORAGE"] = ComplianceRule(
            rule_id="CLASSIFIED_STORAGE",
            name="Classified Data Storage",
            description="Secure storage of classified information",
            regulation="Executive Order 13526",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement classified storage systems",
                "Deploy access controls for classified data",
                "Maintain storage audit trails"
            ],
            evidence_requirements=["storage_systems", "access_controls", "audit_trails"]
        )
        
        self.rules["CLASSIFIED_TRANSMISSION"] = ComplianceRule(
            rule_id="CLASSIFIED_TRANSMISSION",
            name="Classified Data Transmission",
            description="Secure transmission of classified information",
            regulation="Executive Order 13526",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Use approved transmission methods",
                "Implement end-to-end encryption",
                "Maintain transmission logs"
            ],
            evidence_requirements=["transmission_methods", "encryption_verification", "transmission_logs"]
        )
        
        self.rules["CLASSIFIED_PERSONNEL"] = ComplianceRule(
            rule_id="CLASSIFIED_PERSONNEL",
            name="Personnel Security for Classified Systems",
            description="Ensure personnel handling classified information have appropriate clearances",
            regulation="Executive Order 13526",
            severity="critical",
            automated_check=False,
            remediation_steps=[
                "Verify security clearances",
                "Implement need-to-know principles",
                "Regular personnel security reviews"
            ],
            evidence_requirements=["clearance_verification", "access_justification", "security_reviews"]
        )
        
        # Additional Government Security Rules
        self.rules["NIST_800_53"] = ComplianceRule(
            rule_id="NIST_800_53",
            name="NIST 800-53 Security Controls",
            description="Implement NIST 800-53 security control families",
            regulation="NIST 800-53",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Implement all applicable control families",
                "Document control implementation",
                "Regular control assessments"
            ],
            evidence_requirements=["control_implementation", "control_documentation", "assessment_reports"]
        )
    
    def assess_compliance(self, system_data: Dict[str, Any]) -> ComplianceAssessment:
        """Assess government compliance"""
        assessment_id = f"government_assessment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        rules_evaluated = list(self.rules.keys())
        compliant_rules = []
        non_compliant_rules = []
        recommendations = []
        
        # FedRAMP Assessments
        if self._check_fedramp_access_control(system_data):
            compliant_rules.append("FEDRAMP_AC")
        else:
            non_compliant_rules.append("FEDRAMP_AC")
            recommendations.append("Implement comprehensive access control measures")
        
        if self._check_fedramp_audit(system_data):
            compliant_rules.append("FEDRAMP_AU")
        else:
            non_compliant_rules.append("FEDRAMP_AU")
            recommendations.append("Deploy comprehensive audit and accountability controls")
        
        if self._check_fedramp_system_protection(system_data):
            compliant_rules.append("FEDRAMP_SC")
        else:
            non_compliant_rules.append("FEDRAMP_SC")
            recommendations.append("Implement system and communications protection")
        
        if self._check_fedramp_system_integrity(system_data):
            compliant_rules.append("FEDRAMP_SI")
        else:
            non_compliant_rules.append("FEDRAMP_SI")
            recommendations.append("Deploy system and information integrity controls")
        
        # FISMA Assessments
        if self._check_fisma_categorization(system_data):
            compliant_rules.append("FISMA_CATEGORIZATION")
        else:
            non_compliant_rules.append("FISMA_CATEGORIZATION")
            recommendations.append("Complete system categorization according to FIPS 199")
        
        if self._check_fisma_security_controls(system_data):
            compliant_rules.append("FISMA_SECURITY_CONTROLS")
        else:
            non_compliant_rules.append("FISMA_SECURITY_CONTROLS")
            recommendations.append("Implement appropriate security controls")
        
        if self._check_fisma_continuous_monitoring(system_data):
            compliant_rules.append("FISMA_CONTINUOUS_MONITORING")
        else:
            non_compliant_rules.append("FISMA_CONTINUOUS_MONITORING")
            recommendations.append("Deploy continuous monitoring capabilities")
        
        # Classified Data Handling Assessments
        if self._check_classified_storage(system_data):
            compliant_rules.append("CLASSIFIED_STORAGE")
        else:
            non_compliant_rules.append("CLASSIFIED_STORAGE")
            recommendations.append("Implement secure classified data storage")
        
        if self._check_classified_transmission(system_data):
            compliant_rules.append("CLASSIFIED_TRANSMISSION")
        else:
            non_compliant_rules.append("CLASSIFIED_TRANSMISSION")
            recommendations.append("Secure classified data transmission methods")
        
        if self._check_classified_personnel(system_data):
            compliant_rules.append("CLASSIFIED_PERSONNEL")
        else:
            non_compliant_rules.append("CLASSIFIED_PERSONNEL")
            recommendations.append("Verify personnel security clearances")
        
        # NIST 800-53 Assessment
        if self._check_nist_800_53(system_data):
            compliant_rules.append("NIST_800_53")
        else:
            non_compliant_rules.append("NIST_800_53")
            recommendations.append("Implement NIST 800-53 security controls")
        
        # Determine overall status
        compliance_rate = len(compliant_rules) / len(rules_evaluated)
        if compliance_rate >= 0.95:
            overall_status = ComplianceStatus.COMPLIANT
        elif compliance_rate >= 0.80:
            overall_status = ComplianceStatus.PENDING_REVIEW
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT
        
        risk_score = self.calculate_risk_score(non_compliant_rules)
        
        return ComplianceAssessment(
            assessment_id=assessment_id,
            industry=self.industry,
            timestamp=datetime.utcnow(),
            rules_evaluated=rules_evaluated,
            compliant_rules=compliant_rules,
            non_compliant_rules=non_compliant_rules,
            overall_status=overall_status,
            risk_score=risk_score,
            recommendations=recommendations
        )
    
    def generate_report(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate government compliance report"""
        return {
            "assessment_id": assessment.assessment_id,
            "industry": "Government",
            "timestamp": assessment.timestamp.isoformat(),
            "overall_status": assessment.overall_status.value,
            "compliance_rate": (len(assessment.compliant_rules) / len(assessment.rules_evaluated)) * 100,
            "risk_score": assessment.risk_score,
            "regulatory_frameworks": {
                "FedRAMP": self._get_framework_status(assessment, "FEDRAMP"),
                "FISMA": self._get_framework_status(assessment, "FISMA"),
                "Classified_Data": self._get_framework_status(assessment, "CLASSIFIED"),
                "NIST_800_53": self._get_framework_status(assessment, "NIST")
            },
            "security_clearance_status": self._assess_security_clearance(assessment),
            "ato_readiness": self._assess_ato_readiness(assessment),
            "compliant_rules": assessment.compliant_rules,
            "non_compliant_rules": assessment.non_compliant_rules,
            "recommendations": assessment.recommendations,
            "next_assessment_due": (assessment.timestamp + timedelta(days=365)).isoformat()
        }
    
    def _check_fedramp_access_control(self, system_data: Dict[str, Any]) -> bool:
        """Check FedRAMP access control requirements"""
        access_config = system_data.get("access_control_config", {})
        return (
            access_config.get("rbac_implemented", False) and
            access_config.get("mfa_enabled", False) and
            access_config.get("regular_access_reviews", False)
        )
    
    def _check_fedramp_audit(self, system_data: Dict[str, Any]) -> bool:
        """Check FedRAMP audit and accountability requirements"""
        audit_config = system_data.get("audit_config", {})
        return (
            audit_config.get("comprehensive_logging", False) and
            audit_config.get("log_analysis", False) and
            audit_config.get("audit_trail_integrity", False)
        )
    
    def _check_fedramp_system_protection(self, system_data: Dict[str, Any]) -> bool:
        """Check FedRAMP system and communications protection"""
        protection_config = system_data.get("system_protection_config", {})
        return (
            protection_config.get("encryption_in_transit", False) and
            protection_config.get("encryption_at_rest", False) and
            protection_config.get("network_security_controls", False)
        )
    
    def _check_fedramp_system_integrity(self, system_data: Dict[str, Any]) -> bool:
        """Check FedRAMP system and information integrity"""
        integrity_config = system_data.get("system_integrity_config", {})
        return (
            integrity_config.get("malware_protection", False) and
            integrity_config.get("vulnerability_scanning", False) and
            integrity_config.get("integrity_monitoring", False)
        )
    
    def _check_fisma_categorization(self, system_data: Dict[str, Any]) -> bool:
        """Check FISMA system categorization"""
        categorization_config = system_data.get("categorization_config", {})
        return (
            categorization_config.get("categorization_completed", False) and
            categorization_config.get("categorization_documented", False) and
            categorization_config.get("regular_reviews", False)
        )
    
    def _check_fisma_security_controls(self, system_data: Dict[str, Any]) -> bool:
        """Check FISMA security control implementation"""
        controls_config = system_data.get("security_controls_config", {})
        return (
            controls_config.get("controls_selected", False) and
            controls_config.get("controls_implemented", False) and
            controls_config.get("controls_documented", False)
        )
    
    def _check_fisma_continuous_monitoring(self, system_data: Dict[str, Any]) -> bool:
        """Check FISMA continuous monitoring"""
        monitoring_config = system_data.get("continuous_monitoring_config", {})
        return (
            monitoring_config.get("monitoring_tools_deployed", False) and
            monitoring_config.get("monitoring_procedures", False) and
            monitoring_config.get("regular_reports", False)
        )
    
    def _check_classified_storage(self, system_data: Dict[str, Any]) -> bool:
        """Check classified data storage requirements"""
        storage_config = system_data.get("classified_storage_config", {})
        return (
            storage_config.get("secure_storage_systems", False) and
            storage_config.get("access_controls", False) and
            storage_config.get("audit_trails", False)
        )
    
    def _check_classified_transmission(self, system_data: Dict[str, Any]) -> bool:
        """Check classified data transmission requirements"""
        transmission_config = system_data.get("classified_transmission_config", {})
        return (
            transmission_config.get("approved_methods", False) and
            transmission_config.get("end_to_end_encryption", False) and
            transmission_config.get("transmission_logs", False)
        )
    
    def _check_classified_personnel(self, system_data: Dict[str, Any]) -> bool:
        """Check personnel security for classified systems"""
        personnel_config = system_data.get("personnel_security_config", {})
        return (
            personnel_config.get("clearance_verification", False) and
            personnel_config.get("need_to_know", False) and
            personnel_config.get("regular_reviews", False)
        )
    
    def _check_nist_800_53(self, system_data: Dict[str, Any]) -> bool:
        """Check NIST 800-53 security controls"""
        nist_config = system_data.get("nist_800_53_config", {})
        return (
            nist_config.get("control_families_implemented", False) and
            nist_config.get("controls_documented", False) and
            nist_config.get("regular_assessments", False)
        )
    
    def _assess_security_clearance(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Assess security clearance requirements compliance"""
        clearance_rules = ["CLASSIFIED_PERSONNEL"]
        compliant_clearance_rules = [rule for rule in assessment.compliant_rules if rule in clearance_rules]
        
        if compliant_clearance_rules:
            status = "clearance_verified"
        else:
            status = "clearance_verification_required"
        
        return {
            "status": status,
            "compliant_controls": len(compliant_clearance_rules),
            "total_controls": len(clearance_rules)
        }
    
    def _assess_ato_readiness(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Assess Authority to Operate (ATO) readiness"""
        ato_critical_rules = ["FEDRAMP_AC", "FEDRAMP_AU", "FEDRAMP_SC", "FISMA_SECURITY_CONTROLS"]
        compliant_ato_rules = [rule for rule in assessment.compliant_rules if rule in ato_critical_rules]
        
        ato_readiness = len(compliant_ato_rules) / len(ato_critical_rules)
        
        if ato_readiness >= 1.0:
            status = "ato_ready"
        elif ato_readiness >= 0.75:
            status = "ato_pending"
        else:
            status = "ato_not_ready"
        
        return {
            "status": status,
            "readiness_level": ato_readiness * 100,
            "compliant_critical_controls": len(compliant_ato_rules),
            "total_critical_controls": len(ato_critical_rules)
        }
    
    def _get_framework_status(self, assessment: ComplianceAssessment, framework_prefix: str) -> Dict[str, Any]:
        """Get status for specific regulatory framework"""
        framework_rules = [rule for rule in assessment.rules_evaluated if rule.startswith(framework_prefix)]
        compliant_framework_rules = [rule for rule in assessment.compliant_rules if rule.startswith(framework_prefix)]
        
        if not framework_rules:
            return {"status": "not_applicable", "compliance_rate": 0}
        
        compliance_rate = (len(compliant_framework_rules) / len(framework_rules)) * 100
        status = "compliant" if compliance_rate >= 95 else "non_compliant" if compliance_rate < 80 else "partial"
        
        return {
            "status": status,
            "compliance_rate": compliance_rate,
            "total_rules": len(framework_rules),
            "compliant_rules": len(compliant_framework_rules)
        }