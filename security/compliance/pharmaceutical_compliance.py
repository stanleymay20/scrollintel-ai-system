"""
Pharmaceutical Compliance Module
Implements GxP compliance and clinical trial data integrity
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging

from .industry_compliance_framework import (
    BaseComplianceModule, ComplianceRule, ComplianceAssessment,
    ComplianceStatus, IndustryType
)

logger = logging.getLogger(__name__)

class PharmaceuticalComplianceModule(BaseComplianceModule):
    """Pharmaceutical industry compliance module with GxP compliance and clinical trial data integrity"""
    
    def __init__(self):
        super().__init__(IndustryType.PHARMACEUTICAL)
    
    def _load_rules(self):
        """Load pharmaceutical-specific compliance rules"""
        
        # GxP Compliance Rules
        self.rules["GMP_QUALITY_SYSTEM"] = ComplianceRule(
            rule_id="GMP_QUALITY_SYSTEM",
            name="Good Manufacturing Practice Quality System",
            description="Implement comprehensive GMP quality management system",
            regulation="FDA 21 CFR Part 211",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement quality management system",
                "Establish manufacturing procedures",
                "Deploy quality control processes"
            ],
            evidence_requirements=["quality_system_documentation", "manufacturing_procedures", "qc_processes"]
        )
        
        self.rules["GLP_LABORATORY"] = ComplianceRule(
            rule_id="GLP_LABORATORY",
            name="Good Laboratory Practice",
            description="Ensure laboratory operations comply with GLP standards",
            regulation="FDA 21 CFR Part 58",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement GLP laboratory procedures",
                "Establish data integrity controls",
                "Deploy laboratory information management system"
            ],
            evidence_requirements=["glp_procedures", "data_integrity_controls", "lims_system"]
        )
        
        self.rules["GCP_CLINICAL_TRIALS"] = ComplianceRule(
            rule_id="GCP_CLINICAL_TRIALS",
            name="Good Clinical Practice",
            description="Ensure clinical trials comply with GCP standards",
            regulation="ICH E6 GCP Guidelines",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement GCP procedures",
                "Deploy clinical trial management system",
                "Establish data monitoring procedures"
            ],
            evidence_requirements=["gcp_procedures", "ctms_system", "data_monitoring"]
        )
        
        self.rules["GDP_DISTRIBUTION"] = ComplianceRule(
            rule_id="GDP_DISTRIBUTION",
            name="Good Distribution Practice",
            description="Ensure pharmaceutical distribution complies with GDP",
            regulation="EU GDP Guidelines",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Implement GDP procedures",
                "Deploy supply chain monitoring",
                "Establish temperature control systems"
            ],
            evidence_requirements=["gdp_procedures", "supply_chain_monitoring", "temperature_controls"]
        )
        
        # Clinical Trial Data Integrity Rules
        self.rules["CLINICAL_DATA_INTEGRITY"] = ComplianceRule(
            rule_id="CLINICAL_DATA_INTEGRITY",
            name="Clinical Trial Data Integrity",
            description="Ensure integrity of clinical trial data throughout lifecycle",
            regulation="FDA Data Integrity Guidance",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement ALCOA+ principles",
                "Deploy electronic data capture systems",
                "Establish audit trail procedures"
            ],
            evidence_requirements=["alcoa_implementation", "edc_systems", "audit_trails"]
        )
        
        self.rules["ELECTRONIC_RECORDS_PHARMA"] = ComplianceRule(
            rule_id="ELECTRONIC_RECORDS_PHARMA",
            name="Electronic Records in Clinical Trials",
            description="Ensure electronic records meet regulatory requirements",
            regulation="FDA 21 CFR Part 11",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement electronic record controls",
                "Deploy digital signature systems",
                "Establish record retention procedures"
            ],
            evidence_requirements=["record_controls", "signature_systems", "retention_procedures"]
        )
        
        self.rules["CLINICAL_DATA_MONITORING"] = ComplianceRule(
            rule_id="CLINICAL_DATA_MONITORING",
            name="Clinical Data Monitoring",
            description="Implement comprehensive clinical data monitoring",
            regulation="ICH E6 GCP Guidelines",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Deploy clinical data monitoring systems",
                "Implement risk-based monitoring",
                "Establish data review procedures"
            ],
            evidence_requirements=["monitoring_systems", "rbm_procedures", "data_review_procedures"]
        )
        
        # Pharmacovigilance Rules
        self.rules["PHARMACOVIGILANCE"] = ComplianceRule(
            rule_id="PHARMACOVIGILANCE",
            name="Pharmacovigilance System",
            description="Implement comprehensive pharmacovigilance system",
            regulation="FDA FAERS/EU EudraVigilance",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Deploy adverse event reporting system",
                "Implement signal detection procedures",
                "Establish safety database"
            ],
            evidence_requirements=["aer_system", "signal_detection", "safety_database"]
        )
        
        self.rules["ADVERSE_EVENT_REPORTING"] = ComplianceRule(
            rule_id="ADVERSE_EVENT_REPORTING",
            name="Adverse Event Reporting",
            description="Ensure timely and accurate adverse event reporting",
            regulation="FDA 21 CFR Part 312/314",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Implement automated AE reporting",
                "Deploy case processing workflows",
                "Establish regulatory submission procedures"
            ],
            evidence_requirements=["automated_reporting", "case_workflows", "submission_procedures"]
        )
        
        # Regulatory Submission Rules
        self.rules["REGULATORY_SUBMISSIONS"] = ComplianceRule(
            rule_id="REGULATORY_SUBMISSIONS",
            name="Regulatory Submission Management",
            description="Manage regulatory submissions and communications",
            regulation="FDA eCTD/EU CTD",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Implement submission management system",
                "Deploy document lifecycle management",
                "Establish regulatory tracking procedures"
            ],
            evidence_requirements=["submission_system", "document_management", "tracking_procedures"]
        )
        
        self.rules["PRODUCT_LABELING"] = ComplianceRule(
            rule_id="PRODUCT_LABELING",
            name="Product Labeling Compliance",
            description="Ensure product labeling meets regulatory requirements",
            regulation="FDA 21 CFR Part 201",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Implement labeling management system",
                "Deploy labeling review workflows",
                "Establish change control procedures"
            ],
            evidence_requirements=["labeling_system", "review_workflows", "change_control"]
        )
    
    def assess_compliance(self, system_data: Dict[str, Any]) -> ComplianceAssessment:
        """Assess pharmaceutical compliance"""
        assessment_id = f"pharmaceutical_assessment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        rules_evaluated = list(self.rules.keys())
        compliant_rules = []
        non_compliant_rules = []
        recommendations = []
        
        # GxP Compliance Assessments
        if self._check_gmp_quality_system(system_data):
            compliant_rules.append("GMP_QUALITY_SYSTEM")
        else:
            non_compliant_rules.append("GMP_QUALITY_SYSTEM")
            recommendations.append("Implement comprehensive GMP quality management system")
        
        if self._check_glp_laboratory(system_data):
            compliant_rules.append("GLP_LABORATORY")
        else:
            non_compliant_rules.append("GLP_LABORATORY")
            recommendations.append("Deploy GLP-compliant laboratory procedures")
        
        if self._check_gcp_clinical_trials(system_data):
            compliant_rules.append("GCP_CLINICAL_TRIALS")
        else:
            non_compliant_rules.append("GCP_CLINICAL_TRIALS")
            recommendations.append("Implement GCP procedures for clinical trials")
        
        if self._check_gdp_distribution(system_data):
            compliant_rules.append("GDP_DISTRIBUTION")
        else:
            non_compliant_rules.append("GDP_DISTRIBUTION")
            recommendations.append("Deploy GDP-compliant distribution procedures")
        
        # Clinical Trial Data Integrity Assessments
        if self._check_clinical_data_integrity(system_data):
            compliant_rules.append("CLINICAL_DATA_INTEGRITY")
        else:
            non_compliant_rules.append("CLINICAL_DATA_INTEGRITY")
            recommendations.append("Implement ALCOA+ data integrity principles")
        
        if self._check_electronic_records_pharma(system_data):
            compliant_rules.append("ELECTRONIC_RECORDS_PHARMA")
        else:
            non_compliant_rules.append("ELECTRONIC_RECORDS_PHARMA")
            recommendations.append("Deploy compliant electronic records systems")
        
        if self._check_clinical_data_monitoring(system_data):
            compliant_rules.append("CLINICAL_DATA_MONITORING")
        else:
            non_compliant_rules.append("CLINICAL_DATA_MONITORING")
            recommendations.append("Implement comprehensive clinical data monitoring")
        
        # Pharmacovigilance Assessments
        if self._check_pharmacovigilance(system_data):
            compliant_rules.append("PHARMACOVIGILANCE")
        else:
            non_compliant_rules.append("PHARMACOVIGILANCE")
            recommendations.append("Deploy comprehensive pharmacovigilance system")
        
        if self._check_adverse_event_reporting(system_data):
            compliant_rules.append("ADVERSE_EVENT_REPORTING")
        else:
            non_compliant_rules.append("ADVERSE_EVENT_REPORTING")
            recommendations.append("Implement automated adverse event reporting")
        
        # Regulatory Submission Assessments
        if self._check_regulatory_submissions(system_data):
            compliant_rules.append("REGULATORY_SUBMISSIONS")
        else:
            non_compliant_rules.append("REGULATORY_SUBMISSIONS")
            recommendations.append("Deploy regulatory submission management system")
        
        if self._check_product_labeling(system_data):
            compliant_rules.append("PRODUCT_LABELING")
        else:
            non_compliant_rules.append("PRODUCT_LABELING")
            recommendations.append("Implement product labeling compliance system")
        
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
        """Generate pharmaceutical compliance report"""
        return {
            "assessment_id": assessment.assessment_id,
            "industry": "Pharmaceutical",
            "timestamp": assessment.timestamp.isoformat(),
            "overall_status": assessment.overall_status.value,
            "compliance_rate": (len(assessment.compliant_rules) / len(assessment.rules_evaluated)) * 100,
            "risk_score": assessment.risk_score,
            "gxp_domains": {
                "GMP": self._get_domain_status(assessment, "GMP"),
                "GLP": self._get_domain_status(assessment, "GLP"),
                "GCP": self._get_domain_status(assessment, "GCP"),
                "GDP": self._get_domain_status(assessment, "GDP")
            },
            "data_integrity_status": self._assess_data_integrity(assessment),
            "pharmacovigilance_status": self._assess_pharmacovigilance(assessment),
            "regulatory_readiness": self._assess_regulatory_readiness(assessment),
            "compliant_rules": assessment.compliant_rules,
            "non_compliant_rules": assessment.non_compliant_rules,
            "recommendations": assessment.recommendations,
            "next_assessment_due": (assessment.timestamp + timedelta(days=180)).isoformat()
        }
    
    def _check_gmp_quality_system(self, system_data: Dict[str, Any]) -> bool:
        """Check GMP quality system implementation"""
        gmp_config = system_data.get("gmp_config", {})
        return (
            gmp_config.get("quality_system_documented", False) and
            gmp_config.get("manufacturing_procedures", False) and
            gmp_config.get("quality_control_processes", False)
        )
    
    def _check_glp_laboratory(self, system_data: Dict[str, Any]) -> bool:
        """Check GLP laboratory compliance"""
        glp_config = system_data.get("glp_config", {})
        return (
            glp_config.get("glp_procedures", False) and
            glp_config.get("data_integrity_controls", False) and
            glp_config.get("lims_system", False)
        )
    
    def _check_gcp_clinical_trials(self, system_data: Dict[str, Any]) -> bool:
        """Check GCP clinical trial compliance"""
        gcp_config = system_data.get("gcp_config", {})
        return (
            gcp_config.get("gcp_procedures", False) and
            gcp_config.get("ctms_system", False) and
            gcp_config.get("data_monitoring", False)
        )
    
    def _check_gdp_distribution(self, system_data: Dict[str, Any]) -> bool:
        """Check GDP distribution compliance"""
        gdp_config = system_data.get("gdp_config", {})
        return (
            gdp_config.get("gdp_procedures", False) and
            gdp_config.get("supply_chain_monitoring", False) and
            gdp_config.get("temperature_controls", False)
        )
    
    def _check_clinical_data_integrity(self, system_data: Dict[str, Any]) -> bool:
        """Check clinical data integrity implementation"""
        data_integrity_config = system_data.get("clinical_data_integrity_config", {})
        return (
            data_integrity_config.get("alcoa_principles", False) and
            data_integrity_config.get("edc_systems", False) and
            data_integrity_config.get("audit_trails", False)
        )
    
    def _check_electronic_records_pharma(self, system_data: Dict[str, Any]) -> bool:
        """Check pharmaceutical electronic records compliance"""
        records_config = system_data.get("pharma_electronic_records_config", {})
        return (
            records_config.get("record_controls", False) and
            records_config.get("signature_systems", False) and
            records_config.get("retention_procedures", False)
        )
    
    def _check_clinical_data_monitoring(self, system_data: Dict[str, Any]) -> bool:
        """Check clinical data monitoring implementation"""
        monitoring_config = system_data.get("clinical_monitoring_config", {})
        return (
            monitoring_config.get("monitoring_systems", False) and
            monitoring_config.get("rbm_procedures", False) and
            monitoring_config.get("data_review_procedures", False)
        )
    
    def _check_pharmacovigilance(self, system_data: Dict[str, Any]) -> bool:
        """Check pharmacovigilance system implementation"""
        pv_config = system_data.get("pharmacovigilance_config", {})
        return (
            pv_config.get("aer_system", False) and
            pv_config.get("signal_detection", False) and
            pv_config.get("safety_database", False)
        )
    
    def _check_adverse_event_reporting(self, system_data: Dict[str, Any]) -> bool:
        """Check adverse event reporting implementation"""
        aer_config = system_data.get("adverse_event_config", {})
        return (
            aer_config.get("automated_reporting", False) and
            aer_config.get("case_workflows", False) and
            aer_config.get("submission_procedures", False)
        )
    
    def _check_regulatory_submissions(self, system_data: Dict[str, Any]) -> bool:
        """Check regulatory submission management"""
        submission_config = system_data.get("regulatory_submission_config", {})
        return (
            submission_config.get("submission_system", False) and
            submission_config.get("document_management", False) and
            submission_config.get("tracking_procedures", False)
        )
    
    def _check_product_labeling(self, system_data: Dict[str, Any]) -> bool:
        """Check product labeling compliance"""
        labeling_config = system_data.get("product_labeling_config", {})
        return (
            labeling_config.get("labeling_system", False) and
            labeling_config.get("review_workflows", False) and
            labeling_config.get("change_control", False)
        )
    
    def _assess_data_integrity(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Assess clinical trial data integrity status"""
        integrity_rules = ["CLINICAL_DATA_INTEGRITY", "ELECTRONIC_RECORDS_PHARMA", "CLINICAL_DATA_MONITORING"]
        compliant_integrity_rules = [rule for rule in assessment.compliant_rules if rule in integrity_rules]
        
        integrity_level = len(compliant_integrity_rules) / len(integrity_rules)
        
        if integrity_level >= 1.0:
            status = "fully_compliant"
        elif integrity_level >= 0.67:
            status = "partially_compliant"
        else:
            status = "non_compliant"
        
        return {
            "status": status,
            "integrity_level": integrity_level * 100,
            "compliant_controls": len(compliant_integrity_rules),
            "total_controls": len(integrity_rules)
        }
    
    def _assess_pharmacovigilance(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Assess pharmacovigilance system status"""
        pv_rules = ["PHARMACOVIGILANCE", "ADVERSE_EVENT_REPORTING"]
        compliant_pv_rules = [rule for rule in assessment.compliant_rules if rule in pv_rules]
        
        pv_level = len(compliant_pv_rules) / len(pv_rules)
        
        if pv_level >= 1.0:
            status = "fully_operational"
        elif pv_level >= 0.5:
            status = "partially_operational"
        else:
            status = "needs_implementation"
        
        return {
            "status": status,
            "operational_level": pv_level * 100,
            "compliant_controls": len(compliant_pv_rules),
            "total_controls": len(pv_rules)
        }
    
    def _assess_regulatory_readiness(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Assess regulatory submission readiness"""
        regulatory_rules = ["REGULATORY_SUBMISSIONS", "PRODUCT_LABELING"]
        compliant_regulatory_rules = [rule for rule in assessment.compliant_rules if rule in regulatory_rules]
        
        readiness_level = len(compliant_regulatory_rules) / len(regulatory_rules)
        
        if readiness_level >= 1.0:
            status = "submission_ready"
        elif readiness_level >= 0.5:
            status = "preparation_needed"
        else:
            status = "not_ready"
        
        return {
            "status": status,
            "readiness_level": readiness_level * 100,
            "compliant_controls": len(compliant_regulatory_rules),
            "total_controls": len(regulatory_rules)
        }
    
    def _get_domain_status(self, assessment: ComplianceAssessment, domain_prefix: str) -> Dict[str, Any]:
        """Get status for specific GxP domain"""
        domain_rules = [rule for rule in assessment.rules_evaluated if rule.startswith(domain_prefix)]
        compliant_domain_rules = [rule for rule in assessment.compliant_rules if rule.startswith(domain_prefix)]
        
        if not domain_rules:
            return {"status": "not_applicable", "compliance_rate": 0}
        
        compliance_rate = (len(compliant_domain_rules) / len(domain_rules)) * 100
        status = "compliant" if compliance_rate >= 95 else "non_compliant" if compliance_rate < 80 else "partial"
        
        return {
            "status": status,
            "compliance_rate": compliance_rate,
            "total_rules": len(domain_rules),
            "compliant_rules": len(compliant_domain_rules)
        }