"""
Healthcare Compliance Module
Implements HIPAA, HITECH, and FDA 21 CFR Part 11 support
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging

from .industry_compliance_framework import (
    BaseComplianceModule, ComplianceRule, ComplianceAssessment,
    ComplianceStatus, IndustryType
)

logger = logging.getLogger(__name__)

class HealthcareComplianceModule(BaseComplianceModule):
    """Healthcare industry compliance module with HIPAA, HITECH, and FDA 21 CFR Part 11 controls"""
    
    def __init__(self):
        super().__init__(IndustryType.HEALTHCARE)
    
    def _load_rules(self):
        """Load healthcare-specific compliance rules"""
        
        # HIPAA Rules
        self.rules["HIPAA_PRIVACY"] = ComplianceRule(
            rule_id="HIPAA_PRIVACY",
            name="Privacy Rule Compliance",
            description="Protect the privacy of individually identifiable health information",
            regulation="HIPAA Privacy Rule",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement privacy policies and procedures",
                "Conduct privacy impact assessments",
                "Train workforce on privacy requirements"
            ],
            evidence_requirements=["privacy_policies", "training_records", "access_logs"]
        )
        
        self.rules["HIPAA_SECURITY"] = ComplianceRule(
            rule_id="HIPAA_SECURITY",
            name="Security Rule Compliance",
            description="Ensure the confidentiality, integrity, and availability of ePHI",
            regulation="HIPAA Security Rule",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement administrative safeguards",
                "Deploy physical safeguards",
                "Configure technical safeguards"
            ],
            evidence_requirements=["security_policies", "access_controls", "audit_logs"]
        )
        
        self.rules["HIPAA_BREACH"] = ComplianceRule(
            rule_id="HIPAA_BREACH",
            name="Breach Notification Rule",
            description="Notify individuals and authorities of PHI breaches",
            regulation="HIPAA Breach Notification Rule",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Implement breach detection procedures",
                "Establish notification processes",
                "Maintain breach documentation"
            ],
            evidence_requirements=["breach_procedures", "notification_records", "incident_reports"]
        )
        
        # HITECH Rules
        self.rules["HITECH_AUDIT"] = ComplianceRule(
            rule_id="HITECH_AUDIT",
            name="Audit Controls",
            description="Implement hardware, software, and procedural mechanisms for audit controls",
            regulation="HITECH Act",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Deploy comprehensive audit logging",
                "Implement log monitoring and analysis",
                "Regular audit log reviews"
            ],
            evidence_requirements=["audit_logs", "monitoring_reports", "review_documentation"]
        )
        
        self.rules["HITECH_INTEGRITY"] = ComplianceRule(
            rule_id="HITECH_INTEGRITY",
            name="Information Integrity",
            description="Protect ePHI from improper alteration or destruction",
            regulation="HITECH Act",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement data integrity controls",
                "Deploy checksums and digital signatures",
                "Regular integrity verification"
            ],
            evidence_requirements=["integrity_controls", "verification_reports", "change_logs"]
        )
        
        # FDA 21 CFR Part 11 Rules
        self.rules["FDA_ELECTRONIC_RECORDS"] = ComplianceRule(
            rule_id="FDA_ELECTRONIC_RECORDS",
            name="Electronic Records Requirements",
            description="Ensure electronic records are trustworthy, reliable, and equivalent to paper records",
            regulation="FDA 21 CFR Part 11",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement electronic record controls",
                "Ensure data integrity and authenticity",
                "Maintain audit trails"
            ],
            evidence_requirements=["record_controls", "audit_trails", "validation_documentation"]
        )
        
        self.rules["FDA_ELECTRONIC_SIGNATURES"] = ComplianceRule(
            rule_id="FDA_ELECTRONIC_SIGNATURES",
            name="Electronic Signatures Requirements",
            description="Ensure electronic signatures are legally binding and secure",
            regulation="FDA 21 CFR Part 11",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Implement secure signature systems",
                "Establish signature controls",
                "Maintain signature records"
            ],
            evidence_requirements=["signature_system", "signature_controls", "signature_records"]
        )
        
        self.rules["FDA_VALIDATION"] = ComplianceRule(
            rule_id="FDA_VALIDATION",
            name="System Validation",
            description="Validate computer systems used in FDA-regulated activities",
            regulation="FDA 21 CFR Part 11",
            severity="critical",
            automated_check=False,
            remediation_steps=[
                "Develop validation protocols",
                "Execute validation testing",
                "Maintain validation documentation"
            ],
            evidence_requirements=["validation_protocols", "test_results", "validation_reports"]
        )
    
    def assess_compliance(self, system_data: Dict[str, Any]) -> ComplianceAssessment:
        """Assess healthcare compliance"""
        assessment_id = f"healthcare_assessment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        rules_evaluated = list(self.rules.keys())
        compliant_rules = []
        non_compliant_rules = []
        recommendations = []
        
        # HIPAA Assessments
        if self._check_privacy_controls(system_data):
            compliant_rules.append("HIPAA_PRIVACY")
        else:
            non_compliant_rules.append("HIPAA_PRIVACY")
            recommendations.append("Implement comprehensive privacy controls for PHI")
        
        if self._check_security_safeguards(system_data):
            compliant_rules.append("HIPAA_SECURITY")
        else:
            non_compliant_rules.append("HIPAA_SECURITY")
            recommendations.append("Deploy administrative, physical, and technical safeguards")
        
        if self._check_breach_notification(system_data):
            compliant_rules.append("HIPAA_BREACH")
        else:
            non_compliant_rules.append("HIPAA_BREACH")
            recommendations.append("Establish breach detection and notification procedures")
        
        # HITECH Assessments
        if self._check_audit_controls(system_data):
            compliant_rules.append("HITECH_AUDIT")
        else:
            non_compliant_rules.append("HITECH_AUDIT")
            recommendations.append("Implement comprehensive audit controls and monitoring")
        
        if self._check_information_integrity(system_data):
            compliant_rules.append("HITECH_INTEGRITY")
        else:
            non_compliant_rules.append("HITECH_INTEGRITY")
            recommendations.append("Deploy data integrity protection mechanisms")
        
        # FDA 21 CFR Part 11 Assessments
        if self._check_electronic_records(system_data):
            compliant_rules.append("FDA_ELECTRONIC_RECORDS")
        else:
            non_compliant_rules.append("FDA_ELECTRONIC_RECORDS")
            recommendations.append("Implement electronic records controls and validation")
        
        if self._check_electronic_signatures(system_data):
            compliant_rules.append("FDA_ELECTRONIC_SIGNATURES")
        else:
            non_compliant_rules.append("FDA_ELECTRONIC_SIGNATURES")
            recommendations.append("Deploy secure electronic signature systems")
        
        if self._check_system_validation(system_data):
            compliant_rules.append("FDA_VALIDATION")
        else:
            non_compliant_rules.append("FDA_VALIDATION")
            recommendations.append("Complete system validation for FDA compliance")
        
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
        """Generate healthcare compliance report"""
        return {
            "assessment_id": assessment.assessment_id,
            "industry": "Healthcare",
            "timestamp": assessment.timestamp.isoformat(),
            "overall_status": assessment.overall_status.value,
            "compliance_rate": (len(assessment.compliant_rules) / len(assessment.rules_evaluated)) * 100,
            "risk_score": assessment.risk_score,
            "regulatory_frameworks": {
                "HIPAA": self._get_framework_status(assessment, "HIPAA"),
                "HITECH": self._get_framework_status(assessment, "HITECH"),
                "FDA_21_CFR_Part_11": self._get_framework_status(assessment, "FDA")
            },
            "phi_protection_status": self._assess_phi_protection(assessment),
            "compliant_rules": assessment.compliant_rules,
            "non_compliant_rules": assessment.non_compliant_rules,
            "recommendations": assessment.recommendations,
            "next_assessment_due": (assessment.timestamp + timedelta(days=180)).isoformat()
        }
    
    def _check_privacy_controls(self, system_data: Dict[str, Any]) -> bool:
        """Check HIPAA privacy controls"""
        privacy_config = system_data.get("privacy_config", {})
        return (
            privacy_config.get("privacy_policies_implemented", False) and
            privacy_config.get("minimum_necessary_standard", False) and
            privacy_config.get("individual_rights_procedures", False)
        )
    
    def _check_security_safeguards(self, system_data: Dict[str, Any]) -> bool:
        """Check HIPAA security safeguards"""
        security_config = system_data.get("security_config", {})
        return (
            security_config.get("administrative_safeguards", False) and
            security_config.get("physical_safeguards", False) and
            security_config.get("technical_safeguards", False)
        )
    
    def _check_breach_notification(self, system_data: Dict[str, Any]) -> bool:
        """Check breach notification procedures"""
        breach_config = system_data.get("breach_config", {})
        return (
            breach_config.get("breach_detection_procedures", False) and
            breach_config.get("notification_processes", False) and
            breach_config.get("documentation_procedures", False)
        )
    
    def _check_audit_controls(self, system_data: Dict[str, Any]) -> bool:
        """Check HITECH audit controls"""
        audit_config = system_data.get("audit_config", {})
        return (
            audit_config.get("comprehensive_logging", False) and
            audit_config.get("log_monitoring", False) and
            audit_config.get("regular_reviews", False)
        )
    
    def _check_information_integrity(self, system_data: Dict[str, Any]) -> bool:
        """Check information integrity controls"""
        integrity_config = system_data.get("integrity_config", {})
        return (
            integrity_config.get("data_integrity_controls", False) and
            integrity_config.get("checksums_implemented", False) and
            integrity_config.get("regular_verification", False)
        )
    
    def _check_electronic_records(self, system_data: Dict[str, Any]) -> bool:
        """Check FDA electronic records compliance"""
        records_config = system_data.get("electronic_records_config", {})
        return (
            records_config.get("record_controls_implemented", False) and
            records_config.get("audit_trails_maintained", False) and
            records_config.get("data_integrity_ensured", False)
        )
    
    def _check_electronic_signatures(self, system_data: Dict[str, Any]) -> bool:
        """Check FDA electronic signatures compliance"""
        signature_config = system_data.get("electronic_signatures_config", {})
        return (
            signature_config.get("secure_signature_system", False) and
            signature_config.get("signature_controls", False) and
            signature_config.get("signature_records_maintained", False)
        )
    
    def _check_system_validation(self, system_data: Dict[str, Any]) -> bool:
        """Check FDA system validation"""
        validation_config = system_data.get("validation_config", {})
        return (
            validation_config.get("validation_protocols_developed", False) and
            validation_config.get("validation_testing_completed", False) and
            validation_config.get("validation_documentation_maintained", False)
        )
    
    def _assess_phi_protection(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Assess overall PHI protection status"""
        phi_rules = ["HIPAA_PRIVACY", "HIPAA_SECURITY", "HITECH_INTEGRITY"]
        compliant_phi_rules = [rule for rule in assessment.compliant_rules if rule in phi_rules]
        
        protection_level = len(compliant_phi_rules) / len(phi_rules)
        
        if protection_level >= 1.0:
            status = "fully_protected"
        elif protection_level >= 0.67:
            status = "partially_protected"
        else:
            status = "inadequately_protected"
        
        return {
            "status": status,
            "protection_level": protection_level * 100,
            "compliant_controls": len(compliant_phi_rules),
            "total_controls": len(phi_rules)
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