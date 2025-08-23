"""
Tests for Industry-Tailored Compliance Modules
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from security.compliance.industry_compliance_framework import (
    IndustryComplianceOrchestrator,
    IndustryType,
    ComplianceStatus
)
from security.compliance.banking_compliance import BankingComplianceModule
from security.compliance.healthcare_compliance import HealthcareComplianceModule
from security.compliance.manufacturing_compliance import ManufacturingComplianceModule
from security.compliance.government_compliance import GovernmentComplianceModule
from security.compliance.financial_services_compliance import FinancialServicesComplianceModule
from security.compliance.pharmaceutical_compliance import PharmaceuticalComplianceModule

class TestIndustryComplianceOrchestrator:
    """Test the main compliance orchestrator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.orchestrator = IndustryComplianceOrchestrator()
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initializes all industry modules"""
        assert len(self.orchestrator.modules) == 6
        assert IndustryType.BANKING in self.orchestrator.modules
        assert IndustryType.HEALTHCARE in self.orchestrator.modules
        assert IndustryType.MANUFACTURING in self.orchestrator.modules
        assert IndustryType.GOVERNMENT in self.orchestrator.modules
        assert IndustryType.FINANCIAL_SERVICES in self.orchestrator.modules
        assert IndustryType.PHARMACEUTICAL in self.orchestrator.modules
    
    def test_get_industry_rules(self):
        """Test getting rules for specific industry"""
        banking_rules = self.orchestrator.get_industry_rules(IndustryType.BANKING)
        assert len(banking_rules) > 0
        assert all(rule.rule_id for rule in banking_rules)
        assert all(rule.name for rule in banking_rules)
    
    def test_assess_industry_compliance(self):
        """Test assessing compliance for specific industry"""
        system_data = {
            "firewall_config": {"enabled": True, "rules_configured": True, "logging_enabled": True},
            "security_config": {"default_passwords_present": False, "default_accounts_present": False},
            "encryption_config": {"data_at_rest_encrypted": True, "data_in_transit_encrypted": True, "key_management_implemented": True}
        }
        
        assessment = self.orchestrator.assess_industry_compliance(IndustryType.BANKING, system_data)
        
        assert assessment.industry == IndustryType.BANKING
        assert assessment.assessment_id
        assert assessment.timestamp
        assert isinstance(assessment.rules_evaluated, list)
        assert isinstance(assessment.compliant_rules, list)
        assert isinstance(assessment.non_compliant_rules, list)
        assert assessment.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.PENDING_REVIEW, ComplianceStatus.NON_COMPLIANT]
        assert 0 <= assessment.risk_score <= 100
    
    def test_generate_cross_industry_report(self):
        """Test generating cross-industry compliance report"""
        # Create mock assessments
        assessment1 = Mock()
        assessment1.industry = IndustryType.BANKING
        assessment1.rules_evaluated = ["rule1", "rule2", "rule3"]
        assessment1.compliant_rules = ["rule1", "rule2"]
        assessment1.non_compliant_rules = ["rule3"]
        assessment1.overall_status = ComplianceStatus.PENDING_REVIEW
        assessment1.risk_score = 25.0
        
        assessment2 = Mock()
        assessment2.industry = IndustryType.HEALTHCARE
        assessment2.rules_evaluated = ["rule4", "rule5"]
        assessment2.compliant_rules = ["rule4", "rule5"]
        assessment2.non_compliant_rules = []
        assessment2.overall_status = ComplianceStatus.COMPLIANT
        assessment2.risk_score = 0.0
        
        report = self.orchestrator.generate_cross_industry_report([assessment1, assessment2])
        
        assert "timestamp" in report
        assert report["industries_assessed"] == 2
        assert "overall_compliance_rate" in report
        assert "industry_breakdown" in report
        assert "banking" in report["industry_breakdown"]
        assert "healthcare" in report["industry_breakdown"]

class TestBankingComplianceModule:
    """Test banking compliance module"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.module = BankingComplianceModule()
    
    def test_module_initialization(self):
        """Test module initializes with correct industry and rules"""
        assert self.module.industry == IndustryType.BANKING
        assert len(self.module.rules) > 0
        assert "PCI_DSS_1" in self.module.rules
        assert "SOX_404" in self.module.rules
        assert "BASEL_III_CAR" in self.module.rules
        assert "AML_KYC" in self.module.rules
    
    def test_assess_compliance_compliant_system(self):
        """Test assessment of compliant banking system"""
        system_data = {
            "firewall_config": {"enabled": True, "rules_configured": True, "logging_enabled": True},
            "security_config": {"default_passwords_present": False, "default_accounts_present": False},
            "encryption_config": {"data_at_rest_encrypted": True, "data_in_transit_encrypted": True, "key_management_implemented": True},
            "internal_controls": {"documented": True, "tested": True, "management_assessed": True},
            "capital_data": {"capital_adequacy_ratio": 12.0},
            "kyc_config": {"customer_identification_program": True, "enhanced_due_diligence": True, "ongoing_monitoring": True},
            "aml_config": {"transaction_monitoring": True, "alert_system": True, "sar_filing_process": True}
        }
        
        assessment = self.module.assess_compliance(system_data)
        
        assert assessment.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.PENDING_REVIEW]
        assert len(assessment.compliant_rules) > len(assessment.non_compliant_rules)
        assert assessment.risk_score < 50.0
    
    def test_assess_compliance_non_compliant_system(self):
        """Test assessment of non-compliant banking system"""
        system_data = {
            "firewall_config": {"enabled": False, "rules_configured": False, "logging_enabled": False},
            "security_config": {"default_passwords_present": True, "default_accounts_present": True},
            "encryption_config": {"data_at_rest_encrypted": False, "data_in_transit_encrypted": False, "key_management_implemented": False},
            "internal_controls": {"documented": False, "tested": False, "management_assessed": False},
            "capital_data": {"capital_adequacy_ratio": 5.0},
            "kyc_config": {"customer_identification_program": False, "enhanced_due_diligence": False, "ongoing_monitoring": False},
            "aml_config": {"transaction_monitoring": False, "alert_system": False, "sar_filing_process": False}
        }
        
        assessment = self.module.assess_compliance(system_data)
        
        assert assessment.overall_status == ComplianceStatus.NON_COMPLIANT
        assert len(assessment.non_compliant_rules) > len(assessment.compliant_rules)
        assert assessment.risk_score > 50.0
    
    def test_generate_report(self):
        """Test generating banking compliance report"""
        system_data = {
            "firewall_config": {"enabled": True, "rules_configured": True, "logging_enabled": True},
            "security_config": {"default_passwords_present": False, "default_accounts_present": False}
        }
        
        assessment = self.module.assess_compliance(system_data)
        report = self.module.generate_report(assessment)
        
        assert report["industry"] == "Banking"
        assert "regulatory_frameworks" in report
        assert "PCI_DSS" in report["regulatory_frameworks"]
        assert "SOX" in report["regulatory_frameworks"]
        assert "Basel_III" in report["regulatory_frameworks"]
        assert "AML" in report["regulatory_frameworks"]

class TestHealthcareComplianceModule:
    """Test healthcare compliance module"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.module = HealthcareComplianceModule()
    
    def test_module_initialization(self):
        """Test module initializes with correct industry and rules"""
        assert self.module.industry == IndustryType.HEALTHCARE
        assert len(self.module.rules) > 0
        assert "HIPAA_PRIVACY" in self.module.rules
        assert "HITECH_AUDIT" in self.module.rules
        assert "FDA_ELECTRONIC_RECORDS" in self.module.rules
    
    def test_assess_compliance_compliant_system(self):
        """Test assessment of compliant healthcare system"""
        system_data = {
            "privacy_config": {"privacy_policies_implemented": True, "minimum_necessary_standard": True, "individual_rights_procedures": True},
            "security_config": {"administrative_safeguards": True, "physical_safeguards": True, "technical_safeguards": True},
            "breach_config": {"breach_detection_procedures": True, "notification_processes": True, "documentation_procedures": True},
            "audit_config": {"comprehensive_logging": True, "log_monitoring": True, "regular_reviews": True},
            "integrity_config": {"data_integrity_controls": True, "checksums_implemented": True, "regular_verification": True},
            "electronic_records_config": {"record_controls_implemented": True, "audit_trails_maintained": True, "data_integrity_ensured": True},
            "electronic_signatures_config": {"secure_signature_system": True, "signature_controls": True, "signature_records_maintained": True},
            "validation_config": {"validation_protocols_developed": True, "validation_testing_completed": True, "validation_documentation_maintained": True}
        }
        
        assessment = self.module.assess_compliance(system_data)
        
        assert assessment.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.PENDING_REVIEW]
        assert len(assessment.compliant_rules) > len(assessment.non_compliant_rules)
    
    def test_generate_report(self):
        """Test generating healthcare compliance report"""
        system_data = {
            "privacy_config": {"privacy_policies_implemented": True, "minimum_necessary_standard": True, "individual_rights_procedures": True}
        }
        
        assessment = self.module.assess_compliance(system_data)
        report = self.module.generate_report(assessment)
        
        assert report["industry"] == "Healthcare"
        assert "regulatory_frameworks" in report
        assert "HIPAA" in report["regulatory_frameworks"]
        assert "HITECH" in report["regulatory_frameworks"]
        assert "FDA_21_CFR_Part_11" in report["regulatory_frameworks"]
        assert "phi_protection_status" in report

class TestManufacturingComplianceModule:
    """Test manufacturing compliance module"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.module = ManufacturingComplianceModule()
    
    def test_module_initialization(self):
        """Test module initializes with correct industry and rules"""
        assert self.module.industry == IndustryType.MANUFACTURING
        assert len(self.module.rules) > 0
        assert "IOT_DEVICE_AUTH" in self.module.rules
        assert "OT_IT_SEGMENTATION" in self.module.rules
        assert "SUPPLY_CHAIN_INTEGRITY" in self.module.rules
    
    def test_assess_compliance_compliant_system(self):
        """Test assessment of compliant manufacturing system"""
        system_data = {
            "iot_config": {"certificate_based_auth": True, "device_identity_management": True, "credential_rotation": True, "end_to_end_encryption": True, "secure_protocols": True, "key_management": True},
            "firmware_config": {"secure_boot": True, "signed_updates": True, "vulnerability_scanning": True},
            "network_config": {"ot_it_segmentation": True, "dmz_implemented": True, "traffic_monitoring": True},
            "ot_monitoring_config": {"ot_specific_tools": True, "anomaly_detection": True, "real_time_alerting": True},
            "protocol_config": {"protocol_security_features": True, "protocol_gateways": True, "communication_monitoring": True},
            "supply_chain_config": {"supplier_assessments": True, "supply_chain_monitoring": True, "incident_response_procedures": True},
            "traceability_config": {"tracking_system": True, "traceability_records": True, "regular_audits": True},
            "safety_config": {"safety_systems_deployed": True, "regular_testing": True, "safety_documentation": True}
        }
        
        assessment = self.module.assess_compliance(system_data)
        
        assert assessment.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.PENDING_REVIEW]
        assert len(assessment.compliant_rules) > len(assessment.non_compliant_rules)
    
    def test_generate_report(self):
        """Test generating manufacturing compliance report"""
        system_data = {
            "iot_config": {"certificate_based_auth": True, "device_identity_management": True, "credential_rotation": True}
        }
        
        assessment = self.module.assess_compliance(system_data)
        report = self.module.generate_report(assessment)
        
        assert report["industry"] == "Manufacturing"
        assert "security_domains" in report
        assert "IoT_Security" in report["security_domains"]
        assert "OT_IT_Convergence" in report["security_domains"]
        assert "ot_it_convergence_status" in report

class TestGovernmentComplianceModule:
    """Test government compliance module"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.module = GovernmentComplianceModule()
    
    def test_module_initialization(self):
        """Test module initializes with correct industry and rules"""
        assert self.module.industry == IndustryType.GOVERNMENT
        assert len(self.module.rules) > 0
        assert "FEDRAMP_AC" in self.module.rules
        assert "FISMA_CATEGORIZATION" in self.module.rules
        assert "CLASSIFIED_STORAGE" in self.module.rules
    
    def test_assess_compliance_compliant_system(self):
        """Test assessment of compliant government system"""
        system_data = {
            "access_control_config": {"rbac_implemented": True, "mfa_enabled": True, "regular_access_reviews": True},
            "audit_config": {"comprehensive_logging": True, "log_analysis": True, "audit_trail_integrity": True},
            "system_protection_config": {"encryption_in_transit": True, "encryption_at_rest": True, "network_security_controls": True},
            "system_integrity_config": {"malware_protection": True, "vulnerability_scanning": True, "integrity_monitoring": True},
            "categorization_config": {"categorization_completed": True, "categorization_documented": True, "regular_reviews": True},
            "security_controls_config": {"controls_selected": True, "controls_implemented": True, "controls_documented": True},
            "continuous_monitoring_config": {"monitoring_tools_deployed": True, "monitoring_procedures": True, "regular_reports": True},
            "classified_storage_config": {"secure_storage_systems": True, "access_controls": True, "audit_trails": True},
            "classified_transmission_config": {"approved_methods": True, "end_to_end_encryption": True, "transmission_logs": True},
            "personnel_security_config": {"clearance_verification": True, "need_to_know": True, "regular_reviews": True},
            "nist_800_53_config": {"control_families_implemented": True, "controls_documented": True, "regular_assessments": True}
        }
        
        assessment = self.module.assess_compliance(system_data)
        
        assert assessment.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.PENDING_REVIEW]
        assert len(assessment.compliant_rules) > len(assessment.non_compliant_rules)
    
    def test_generate_report(self):
        """Test generating government compliance report"""
        system_data = {
            "access_control_config": {"rbac_implemented": True, "mfa_enabled": True, "regular_access_reviews": True}
        }
        
        assessment = self.module.assess_compliance(system_data)
        report = self.module.generate_report(assessment)
        
        assert report["industry"] == "Government"
        assert "regulatory_frameworks" in report
        assert "FedRAMP" in report["regulatory_frameworks"]
        assert "FISMA" in report["regulatory_frameworks"]
        assert "ato_readiness" in report

class TestFinancialServicesComplianceModule:
    """Test financial services compliance module"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.module = FinancialServicesComplianceModule()
    
    def test_module_initialization(self):
        """Test module initializes with correct industry and rules"""
        assert self.module.industry == IndustryType.FINANCIAL_SERVICES
        assert len(self.module.rules) > 0
        assert "FRAUD_DETECTION_RT" in self.module.rules
        assert "REGULATORY_REPORTING" in self.module.rules
        assert "MARKET_RISK_MGMT" in self.module.rules
    
    def test_assess_compliance_compliant_system(self):
        """Test assessment of compliant financial services system"""
        system_data = {
            "fraud_detection_config": {"ml_models_deployed": True, "real_time_monitoring": True, "alert_procedures": True},
            "transaction_monitoring_config": {"comprehensive_monitoring": True, "behavioral_analytics": True, "audit_trails": True},
            "risk_scoring_config": {"scoring_algorithms": True, "dynamic_assessment": True, "model_validation": True},
            "regulatory_reporting_config": {"automated_systems": True, "data_quality_controls": True, "validation_procedures": True},
            "stress_testing_config": {"testing_framework": True, "scenario_analysis": True, "documented_results": True},
            "liquidity_monitoring_config": {"monitoring_systems": True, "ratio_calculations": True, "regular_reporting": True},
            "market_risk_config": {"measurement_systems": True, "var_stress_testing": True, "risk_limits": True},
            "trading_surveillance_config": {"surveillance_systems": True, "abuse_detection": True, "investigation_procedures": True},
            "consumer_protection_config": {"fair_lending_practices": True, "complaint_management": True, "disclosure_procedures": True},
            "financial_privacy_config": {"privacy_controls": True, "encryption_masking": True, "breach_procedures": True}
        }
        
        assessment = self.module.assess_compliance(system_data)
        
        assert assessment.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.PENDING_REVIEW]
        assert len(assessment.compliant_rules) > len(assessment.non_compliant_rules)
    
    def test_generate_report(self):
        """Test generating financial services compliance report"""
        system_data = {
            "fraud_detection_config": {"ml_models_deployed": True, "real_time_monitoring": True, "alert_procedures": True}
        }
        
        assessment = self.module.assess_compliance(system_data)
        report = self.module.generate_report(assessment)
        
        assert report["industry"] == "Financial Services"
        assert "compliance_domains" in report
        assert "fraud_detection_effectiveness" in report
        assert "regulatory_reporting_status" in report

class TestPharmaceuticalComplianceModule:
    """Test pharmaceutical compliance module"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.module = PharmaceuticalComplianceModule()
    
    def test_module_initialization(self):
        """Test module initializes with correct industry and rules"""
        assert self.module.industry == IndustryType.PHARMACEUTICAL
        assert len(self.module.rules) > 0
        assert "GMP_QUALITY_SYSTEM" in self.module.rules
        assert "CLINICAL_DATA_INTEGRITY" in self.module.rules
        assert "PHARMACOVIGILANCE" in self.module.rules
    
    def test_assess_compliance_compliant_system(self):
        """Test assessment of compliant pharmaceutical system"""
        system_data = {
            "gmp_config": {"quality_system_documented": True, "manufacturing_procedures": True, "quality_control_processes": True},
            "glp_config": {"glp_procedures": True, "data_integrity_controls": True, "lims_system": True},
            "gcp_config": {"gcp_procedures": True, "ctms_system": True, "data_monitoring": True},
            "gdp_config": {"gdp_procedures": True, "supply_chain_monitoring": True, "temperature_controls": True},
            "clinical_data_integrity_config": {"alcoa_principles": True, "edc_systems": True, "audit_trails": True},
            "pharma_electronic_records_config": {"record_controls": True, "signature_systems": True, "retention_procedures": True},
            "clinical_monitoring_config": {"monitoring_systems": True, "rbm_procedures": True, "data_review_procedures": True},
            "pharmacovigilance_config": {"aer_system": True, "signal_detection": True, "safety_database": True},
            "adverse_event_config": {"automated_reporting": True, "case_workflows": True, "submission_procedures": True},
            "regulatory_submission_config": {"submission_system": True, "document_management": True, "tracking_procedures": True},
            "product_labeling_config": {"labeling_system": True, "review_workflows": True, "change_control": True}
        }
        
        assessment = self.module.assess_compliance(system_data)
        
        assert assessment.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.PENDING_REVIEW]
        assert len(assessment.compliant_rules) > len(assessment.non_compliant_rules)
    
    def test_generate_report(self):
        """Test generating pharmaceutical compliance report"""
        system_data = {
            "gmp_config": {"quality_system_documented": True, "manufacturing_procedures": True, "quality_control_processes": True}
        }
        
        assessment = self.module.assess_compliance(system_data)
        report = self.module.generate_report(assessment)
        
        assert report["industry"] == "Pharmaceutical"
        assert "gxp_domains" in report
        assert "data_integrity_status" in report
        assert "pharmacovigilance_status" in report
        assert "regulatory_readiness" in report

if __name__ == "__main__":
    pytest.main([__file__])