"""
Banking Compliance Module
Implements PCI DSS, SOX, Basel III, and AML controls
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta
import re
import logging

from .industry_compliance_framework import (
    BaseComplianceModule, ComplianceRule, ComplianceAssessment,
    ComplianceStatus, IndustryType
)

logger = logging.getLogger(__name__)

class BankingComplianceModule(BaseComplianceModule):
    """Banking industry compliance module with PCI DSS, SOX, Basel III, and AML controls"""
    
    def __init__(self):
        super().__init__(IndustryType.BANKING)
    
    def _load_rules(self):
        """Load banking-specific compliance rules"""
        
        # PCI DSS Rules
        self.rules["PCI_DSS_1"] = ComplianceRule(
            rule_id="PCI_DSS_1",
            name="Install and maintain firewall configuration",
            description="Firewalls must be installed and maintained to protect cardholder data",
            regulation="PCI DSS Requirement 1",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Configure firewall rules to restrict access",
                "Document firewall configuration",
                "Regular firewall rule reviews"
            ],
            evidence_requirements=["firewall_config", "access_logs", "review_documentation"]
        )
        
        self.rules["PCI_DSS_2"] = ComplianceRule(
            rule_id="PCI_DSS_2",
            name="Do not use vendor-supplied defaults",
            description="Change vendor-supplied defaults and remove unnecessary default accounts",
            regulation="PCI DSS Requirement 2",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Change default passwords",
                "Remove default accounts",
                "Disable unnecessary services"
            ],
            evidence_requirements=["system_hardening_report", "account_audit", "service_inventory"]
        )
        
        self.rules["PCI_DSS_3"] = ComplianceRule(
            rule_id="PCI_DSS_3",
            name="Protect stored cardholder data",
            description="Cardholder data must be encrypted when stored",
            regulation="PCI DSS Requirement 3",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement strong encryption for stored data",
                "Secure key management",
                "Data retention policies"
            ],
            evidence_requirements=["encryption_verification", "key_management_audit", "data_inventory"]
        )
        
        # SOX Rules
        self.rules["SOX_404"] = ComplianceRule(
            rule_id="SOX_404",
            name="Internal controls over financial reporting",
            description="Establish and maintain internal controls over financial reporting",
            regulation="SOX Section 404",
            severity="critical",
            automated_check=False,
            remediation_steps=[
                "Document financial processes",
                "Implement control testing",
                "Management assessment of controls"
            ],
            evidence_requirements=["control_documentation", "testing_results", "management_assessment"]
        )
        
        # Basel III Rules
        self.rules["BASEL_III_CAR"] = ComplianceRule(
            rule_id="BASEL_III_CAR",
            name="Capital Adequacy Ratio",
            description="Maintain minimum capital adequacy ratio as per Basel III",
            regulation="Basel III Capital Requirements",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Monitor capital ratios",
                "Stress testing",
                "Capital planning"
            ],
            evidence_requirements=["capital_calculations", "stress_test_results", "regulatory_reports"]
        )
        
        # AML Rules
        self.rules["AML_KYC"] = ComplianceRule(
            rule_id="AML_KYC",
            name="Know Your Customer procedures",
            description="Implement comprehensive KYC procedures for customer identification",
            regulation="AML/KYC Requirements",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Customer identification program",
                "Enhanced due diligence",
                "Ongoing monitoring"
            ],
            evidence_requirements=["kyc_documentation", "monitoring_reports", "training_records"]
        )
        
        self.rules["AML_SAR"] = ComplianceRule(
            rule_id="AML_SAR",
            name="Suspicious Activity Reporting",
            description="Monitor and report suspicious activities",
            regulation="AML SAR Requirements",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Transaction monitoring system",
                "SAR filing procedures",
                "Staff training"
            ],
            evidence_requirements=["monitoring_alerts", "sar_filings", "training_documentation"]
        )
    
    def assess_compliance(self, system_data: Dict[str, Any]) -> ComplianceAssessment:
        """Assess banking compliance"""
        assessment_id = f"banking_assessment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        rules_evaluated = list(self.rules.keys())
        compliant_rules = []
        non_compliant_rules = []
        recommendations = []
        
        # PCI DSS Assessments
        if self._check_firewall_configuration(system_data):
            compliant_rules.append("PCI_DSS_1")
        else:
            non_compliant_rules.append("PCI_DSS_1")
            recommendations.append("Configure and maintain proper firewall rules")
        
        if self._check_default_credentials(system_data):
            compliant_rules.append("PCI_DSS_2")
        else:
            non_compliant_rules.append("PCI_DSS_2")
            recommendations.append("Remove default credentials and harden systems")
        
        if self._check_data_encryption(system_data):
            compliant_rules.append("PCI_DSS_3")
        else:
            non_compliant_rules.append("PCI_DSS_3")
            recommendations.append("Implement strong encryption for cardholder data")
        
        # SOX Assessment
        if self._check_internal_controls(system_data):
            compliant_rules.append("SOX_404")
        else:
            non_compliant_rules.append("SOX_404")
            recommendations.append("Establish comprehensive internal controls")
        
        # Basel III Assessment
        if self._check_capital_adequacy(system_data):
            compliant_rules.append("BASEL_III_CAR")
        else:
            non_compliant_rules.append("BASEL_III_CAR")
            recommendations.append("Maintain adequate capital ratios")
        
        # AML Assessments
        if self._check_kyc_procedures(system_data):
            compliant_rules.append("AML_KYC")
        else:
            non_compliant_rules.append("AML_KYC")
            recommendations.append("Implement comprehensive KYC procedures")
        
        if self._check_suspicious_activity_monitoring(system_data):
            compliant_rules.append("AML_SAR")
        else:
            non_compliant_rules.append("AML_SAR")
            recommendations.append("Enhance suspicious activity monitoring")
        
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
        """Generate banking compliance report"""
        return {
            "assessment_id": assessment.assessment_id,
            "industry": "Banking",
            "timestamp": assessment.timestamp.isoformat(),
            "overall_status": assessment.overall_status.value,
            "compliance_rate": (len(assessment.compliant_rules) / len(assessment.rules_evaluated)) * 100,
            "risk_score": assessment.risk_score,
            "regulatory_frameworks": {
                "PCI_DSS": self._get_framework_status(assessment, "PCI_DSS"),
                "SOX": self._get_framework_status(assessment, "SOX"),
                "Basel_III": self._get_framework_status(assessment, "BASEL_III"),
                "AML": self._get_framework_status(assessment, "AML")
            },
            "compliant_rules": assessment.compliant_rules,
            "non_compliant_rules": assessment.non_compliant_rules,
            "recommendations": assessment.recommendations,
            "next_assessment_due": (assessment.timestamp + timedelta(days=90)).isoformat()
        }
    
    def _check_firewall_configuration(self, system_data: Dict[str, Any]) -> bool:
        """Check firewall configuration compliance"""
        firewall_config = system_data.get("firewall_config", {})
        return (
            firewall_config.get("enabled", False) and
            firewall_config.get("rules_configured", False) and
            firewall_config.get("logging_enabled", False)
        )
    
    def _check_default_credentials(self, system_data: Dict[str, Any]) -> bool:
        """Check for default credentials"""
        security_config = system_data.get("security_config", {})
        return (
            not security_config.get("default_passwords_present", True) and
            not security_config.get("default_accounts_present", True)
        )
    
    def _check_data_encryption(self, system_data: Dict[str, Any]) -> bool:
        """Check data encryption compliance"""
        encryption_config = system_data.get("encryption_config", {})
        return (
            encryption_config.get("data_at_rest_encrypted", False) and
            encryption_config.get("data_in_transit_encrypted", False) and
            encryption_config.get("key_management_implemented", False)
        )
    
    def _check_internal_controls(self, system_data: Dict[str, Any]) -> bool:
        """Check internal controls implementation"""
        controls = system_data.get("internal_controls", {})
        return (
            controls.get("documented", False) and
            controls.get("tested", False) and
            controls.get("management_assessed", False)
        )
    
    def _check_capital_adequacy(self, system_data: Dict[str, Any]) -> bool:
        """Check capital adequacy ratio"""
        capital_data = system_data.get("capital_data", {})
        car_ratio = capital_data.get("capital_adequacy_ratio", 0)
        return car_ratio >= 8.0  # Basel III minimum
    
    def _check_kyc_procedures(self, system_data: Dict[str, Any]) -> bool:
        """Check KYC procedures implementation"""
        kyc_config = system_data.get("kyc_config", {})
        return (
            kyc_config.get("customer_identification_program", False) and
            kyc_config.get("enhanced_due_diligence", False) and
            kyc_config.get("ongoing_monitoring", False)
        )
    
    def _check_suspicious_activity_monitoring(self, system_data: Dict[str, Any]) -> bool:
        """Check suspicious activity monitoring"""
        aml_config = system_data.get("aml_config", {})
        return (
            aml_config.get("transaction_monitoring", False) and
            aml_config.get("alert_system", False) and
            aml_config.get("sar_filing_process", False)
        )
    
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