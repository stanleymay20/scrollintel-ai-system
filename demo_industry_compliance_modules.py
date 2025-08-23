"""
Demo script for Industry-Tailored Compliance Modules
Demonstrates the comprehensive compliance capabilities across different industries
"""

import asyncio
import json
from datetime import datetime
from security.compliance.industry_compliance_framework import (
    IndustryComplianceOrchestrator, IndustryType
)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

def demo_banking_compliance():
    """Demonstrate banking compliance assessment"""
    print_section("BANKING COMPLIANCE DEMONSTRATION")
    
    orchestrator = IndustryComplianceOrchestrator()
    
    # Get banking rules
    print_subsection("Banking Compliance Rules")
    banking_rules = orchestrator.get_industry_rules(IndustryType.BANKING)
    print(f"Total Banking Rules: {len(banking_rules)}")
    
    for rule in banking_rules[:3]:  # Show first 3 rules
        print(f"\n• {rule.name} ({rule.rule_id})")
        print(f"  Regulation: {rule.regulation}")
        print(f"  Severity: {rule.severity}")
        print(f"  Automated: {rule.automated_check}")
    
    # Simulate compliant banking system
    print_subsection("Compliant Banking System Assessment")
    compliant_system = {
        "firewall_config": {
            "enabled": True,
            "rules_configured": True,
            "logging_enabled": True
        },
        "security_config": {
            "default_passwords_present": False,
            "default_accounts_present": False
        },
        "encryption_config": {
            "data_at_rest_encrypted": True,
            "data_in_transit_encrypted": True,
            "key_management_implemented": True
        },
        "internal_controls": {
            "documented": True,
            "tested": True,
            "management_assessed": True
        },
        "capital_data": {
            "capital_adequacy_ratio": 12.5
        },
        "kyc_config": {
            "customer_identification_program": True,
            "enhanced_due_diligence": True,
            "ongoing_monitoring": True
        },
        "aml_config": {
            "transaction_monitoring": True,
            "alert_system": True,
            "sar_filing_process": True
        }
    }
    
    assessment = orchestrator.assess_industry_compliance(IndustryType.BANKING, compliant_system)
    
    print(f"Assessment ID: {assessment.assessment_id}")
    print(f"Overall Status: {assessment.overall_status.value}")
    print(f"Compliance Rate: {(len(assessment.compliant_rules) / len(assessment.rules_evaluated)) * 100:.1f}%")
    print(f"Risk Score: {assessment.risk_score:.1f}")
    print(f"Compliant Rules: {len(assessment.compliant_rules)}")
    print(f"Non-Compliant Rules: {len(assessment.non_compliant_rules)}")
    
    # Generate detailed report
    banking_module = orchestrator.modules[IndustryType.BANKING]
    report = banking_module.generate_report(assessment)
    
    print_subsection("Regulatory Framework Status")
    for framework, status in report["regulatory_frameworks"].items():
        print(f"• {framework}: {status['status']} ({status['compliance_rate']:.1f}%)")

def demo_healthcare_compliance():
    """Demonstrate healthcare compliance assessment"""
    print_section("HEALTHCARE COMPLIANCE DEMONSTRATION")
    
    orchestrator = IndustryComplianceOrchestrator()
    
    # Simulate healthcare system with mixed compliance
    healthcare_system = {
        "privacy_config": {
            "privacy_policies_implemented": True,
            "minimum_necessary_standard": True,
            "individual_rights_procedures": False  # Non-compliant
        },
        "security_config": {
            "administrative_safeguards": True,
            "physical_safeguards": True,
            "technical_safeguards": True
        },
        "breach_config": {
            "breach_detection_procedures": True,
            "notification_processes": False,  # Non-compliant
            "documentation_procedures": True
        },
        "audit_config": {
            "comprehensive_logging": True,
            "log_monitoring": True,
            "regular_reviews": True
        },
        "integrity_config": {
            "data_integrity_controls": True,
            "checksums_implemented": False,  # Non-compliant
            "regular_verification": True
        },
        "electronic_records_config": {
            "record_controls_implemented": True,
            "audit_trails_maintained": True,
            "data_integrity_ensured": True
        },
        "electronic_signatures_config": {
            "secure_signature_system": True,
            "signature_controls": True,
            "signature_records_maintained": True
        },
        "validation_config": {
            "validation_protocols_developed": False,  # Non-compliant
            "validation_testing_completed": False,   # Non-compliant
            "validation_documentation_maintained": False  # Non-compliant
        }
    }
    
    assessment = orchestrator.assess_industry_compliance(IndustryType.HEALTHCARE, healthcare_system)
    
    print(f"Assessment ID: {assessment.assessment_id}")
    print(f"Overall Status: {assessment.overall_status.value}")
    print(f"Compliance Rate: {(len(assessment.compliant_rules) / len(assessment.rules_evaluated)) * 100:.1f}%")
    print(f"Risk Score: {assessment.risk_score:.1f}")
    
    # Generate detailed report
    healthcare_module = orchestrator.modules[IndustryType.HEALTHCARE]
    report = healthcare_module.generate_report(assessment)
    
    print_subsection("PHI Protection Status")
    phi_status = report["phi_protection_status"]
    print(f"Status: {phi_status['status']}")
    print(f"Protection Level: {phi_status['protection_level']:.1f}%")
    
    print_subsection("Top Recommendations")
    for i, recommendation in enumerate(assessment.recommendations[:3], 1):
        print(f"{i}. {recommendation}")

def demo_manufacturing_compliance():
    """Demonstrate manufacturing compliance assessment"""
    print_section("MANUFACTURING COMPLIANCE DEMONSTRATION")
    
    orchestrator = IndustryComplianceOrchestrator()
    
    # Simulate manufacturing system focused on IoT and OT/IT convergence
    manufacturing_system = {
        "iot_config": {
            "certificate_based_auth": True,
            "device_identity_management": True,
            "credential_rotation": True,
            "end_to_end_encryption": True,
            "secure_protocols": True,
            "key_management": True
        },
        "firmware_config": {
            "secure_boot": True,
            "signed_updates": True,
            "vulnerability_scanning": False  # Non-compliant
        },
        "network_config": {
            "ot_it_segmentation": True,
            "dmz_implemented": True,
            "traffic_monitoring": True
        },
        "ot_monitoring_config": {
            "ot_specific_tools": True,
            "anomaly_detection": False,  # Non-compliant
            "real_time_alerting": True
        },
        "protocol_config": {
            "protocol_security_features": True,
            "protocol_gateways": True,
            "communication_monitoring": True
        },
        "supply_chain_config": {
            "supplier_assessments": False,  # Non-compliant
            "supply_chain_monitoring": True,
            "incident_response_procedures": True
        },
        "traceability_config": {
            "tracking_system": True,
            "traceability_records": True,
            "regular_audits": True
        },
        "safety_config": {
            "safety_systems_deployed": True,
            "regular_testing": True,
            "safety_documentation": True
        }
    }
    
    assessment = orchestrator.assess_industry_compliance(IndustryType.MANUFACTURING, manufacturing_system)
    
    print(f"Assessment ID: {assessment.assessment_id}")
    print(f"Overall Status: {assessment.overall_status.value}")
    print(f"Compliance Rate: {(len(assessment.compliant_rules) / len(assessment.rules_evaluated)) * 100:.1f}%")
    print(f"Risk Score: {assessment.risk_score:.1f}")
    
    # Generate detailed report
    manufacturing_module = orchestrator.modules[IndustryType.MANUFACTURING]
    report = manufacturing_module.generate_report(assessment)
    
    print_subsection("OT/IT Convergence Status")
    convergence_status = report["ot_it_convergence_status"]
    print(f"Status: {convergence_status['status']}")
    print(f"Security Level: {convergence_status['security_level']:.1f}%")
    
    print_subsection("Security Domains")
    for domain, status in report["security_domains"].items():
        print(f"• {domain}: {status['status']} ({status['compliance_rate']:.1f}%)")

def demo_government_compliance():
    """Demonstrate government compliance assessment"""
    print_section("GOVERNMENT COMPLIANCE DEMONSTRATION")
    
    orchestrator = IndustryComplianceOrchestrator()
    
    # Simulate government system with high security requirements
    government_system = {
        "access_control_config": {
            "rbac_implemented": True,
            "mfa_enabled": True,
            "regular_access_reviews": True
        },
        "audit_config": {
            "comprehensive_logging": True,
            "log_analysis": True,
            "audit_trail_integrity": True
        },
        "system_protection_config": {
            "encryption_in_transit": True,
            "encryption_at_rest": True,
            "network_security_controls": True
        },
        "system_integrity_config": {
            "malware_protection": True,
            "vulnerability_scanning": True,
            "integrity_monitoring": True
        },
        "categorization_config": {
            "categorization_completed": True,
            "categorization_documented": True,
            "regular_reviews": True
        },
        "security_controls_config": {
            "controls_selected": True,
            "controls_implemented": True,
            "controls_documented": True
        },
        "continuous_monitoring_config": {
            "monitoring_tools_deployed": True,
            "monitoring_procedures": True,
            "regular_reports": True
        },
        "classified_storage_config": {
            "secure_storage_systems": True,
            "access_controls": True,
            "audit_trails": True
        },
        "classified_transmission_config": {
            "approved_methods": True,
            "end_to_end_encryption": True,
            "transmission_logs": True
        },
        "personnel_security_config": {
            "clearance_verification": True,
            "need_to_know": True,
            "regular_reviews": True
        },
        "nist_800_53_config": {
            "control_families_implemented": True,
            "controls_documented": True,
            "regular_assessments": True
        }
    }
    
    assessment = orchestrator.assess_industry_compliance(IndustryType.GOVERNMENT, government_system)
    
    print(f"Assessment ID: {assessment.assessment_id}")
    print(f"Overall Status: {assessment.overall_status.value}")
    print(f"Compliance Rate: {(len(assessment.compliant_rules) / len(assessment.rules_evaluated)) * 100:.1f}%")
    print(f"Risk Score: {assessment.risk_score:.1f}")
    
    # Generate detailed report
    government_module = orchestrator.modules[IndustryType.GOVERNMENT]
    report = government_module.generate_report(assessment)
    
    print_subsection("ATO Readiness")
    ato_status = report["ato_readiness"]
    print(f"Status: {ato_status['status']}")
    print(f"Readiness Level: {ato_status['readiness_level']:.1f}%")

def demo_financial_services_compliance():
    """Demonstrate financial services compliance assessment"""
    print_section("FINANCIAL SERVICES COMPLIANCE DEMONSTRATION")
    
    orchestrator = IndustryComplianceOrchestrator()
    
    # Simulate financial services system with fraud detection focus
    financial_system = {
        "fraud_detection_config": {
            "ml_models_deployed": True,
            "real_time_monitoring": True,
            "alert_procedures": True
        },
        "transaction_monitoring_config": {
            "comprehensive_monitoring": True,
            "behavioral_analytics": True,
            "audit_trails": True
        },
        "risk_scoring_config": {
            "scoring_algorithms": True,
            "dynamic_assessment": True,
            "model_validation": False  # Non-compliant
        },
        "regulatory_reporting_config": {
            "automated_systems": True,
            "data_quality_controls": True,
            "validation_procedures": True
        },
        "stress_testing_config": {
            "testing_framework": False,  # Non-compliant
            "scenario_analysis": False,  # Non-compliant
            "documented_results": False  # Non-compliant
        },
        "liquidity_monitoring_config": {
            "monitoring_systems": True,
            "ratio_calculations": True,
            "regular_reporting": True
        },
        "market_risk_config": {
            "measurement_systems": True,
            "var_stress_testing": True,
            "risk_limits": True
        },
        "trading_surveillance_config": {
            "surveillance_systems": True,
            "abuse_detection": True,
            "investigation_procedures": True
        },
        "consumer_protection_config": {
            "fair_lending_practices": True,
            "complaint_management": True,
            "disclosure_procedures": True
        },
        "financial_privacy_config": {
            "privacy_controls": True,
            "encryption_masking": True,
            "breach_procedures": True
        }
    }
    
    assessment = orchestrator.assess_industry_compliance(IndustryType.FINANCIAL_SERVICES, financial_system)
    
    print(f"Assessment ID: {assessment.assessment_id}")
    print(f"Overall Status: {assessment.overall_status.value}")
    print(f"Compliance Rate: {(len(assessment.compliant_rules) / len(assessment.rules_evaluated)) * 100:.1f}%")
    print(f"Risk Score: {assessment.risk_score:.1f}")
    
    # Generate detailed report
    financial_module = orchestrator.modules[IndustryType.FINANCIAL_SERVICES]
    report = financial_module.generate_report(assessment)
    
    print_subsection("Fraud Detection Effectiveness")
    fraud_status = report["fraud_detection_effectiveness"]
    print(f"Status: {fraud_status['status']}")
    print(f"Effectiveness Level: {fraud_status['effectiveness_level']:.1f}%")

def demo_pharmaceutical_compliance():
    """Demonstrate pharmaceutical compliance assessment"""
    print_section("PHARMACEUTICAL COMPLIANCE DEMONSTRATION")
    
    orchestrator = IndustryComplianceOrchestrator()
    
    # Simulate pharmaceutical system with GxP focus
    pharmaceutical_system = {
        "gmp_config": {
            "quality_system_documented": True,
            "manufacturing_procedures": True,
            "quality_control_processes": True
        },
        "glp_config": {
            "glp_procedures": True,
            "data_integrity_controls": True,
            "lims_system": True
        },
        "gcp_config": {
            "gcp_procedures": True,
            "ctms_system": True,
            "data_monitoring": True
        },
        "gdp_config": {
            "gdp_procedures": True,
            "supply_chain_monitoring": False,  # Non-compliant
            "temperature_controls": True
        },
        "clinical_data_integrity_config": {
            "alcoa_principles": True,
            "edc_systems": True,
            "audit_trails": True
        },
        "pharma_electronic_records_config": {
            "record_controls": True,
            "signature_systems": True,
            "retention_procedures": True
        },
        "clinical_monitoring_config": {
            "monitoring_systems": True,
            "rbm_procedures": False,  # Non-compliant
            "data_review_procedures": True
        },
        "pharmacovigilance_config": {
            "aer_system": True,
            "signal_detection": True,
            "safety_database": True
        },
        "adverse_event_config": {
            "automated_reporting": True,
            "case_workflows": True,
            "submission_procedures": True
        },
        "regulatory_submission_config": {
            "submission_system": True,
            "document_management": True,
            "tracking_procedures": True
        },
        "product_labeling_config": {
            "labeling_system": True,
            "review_workflows": True,
            "change_control": True
        }
    }
    
    assessment = orchestrator.assess_industry_compliance(IndustryType.PHARMACEUTICAL, pharmaceutical_system)
    
    print(f"Assessment ID: {assessment.assessment_id}")
    print(f"Overall Status: {assessment.overall_status.value}")
    print(f"Compliance Rate: {(len(assessment.compliant_rules) / len(assessment.rules_evaluated)) * 100:.1f}%")
    print(f"Risk Score: {assessment.risk_score:.1f}")
    
    # Generate detailed report
    pharmaceutical_module = orchestrator.modules[IndustryType.PHARMACEUTICAL]
    report = pharmaceutical_module.generate_report(assessment)
    
    print_subsection("GxP Domains Status")
    for domain, status in report["gxp_domains"].items():
        print(f"• {domain}: {status['status']} ({status['compliance_rate']:.1f}%)")
    
    print_subsection("Data Integrity Status")
    integrity_status = report["data_integrity_status"]
    print(f"Status: {integrity_status['status']}")
    print(f"Integrity Level: {integrity_status['integrity_level']:.1f}%")

def demo_cross_industry_assessment():
    """Demonstrate cross-industry compliance assessment"""
    print_section("CROSS-INDUSTRY COMPLIANCE ASSESSMENT")
    
    orchestrator = IndustryComplianceOrchestrator()
    
    # Simulate assessments for multiple industries
    industries_data = {
        IndustryType.BANKING: {
            "firewall_config": {"enabled": True, "rules_configured": True, "logging_enabled": True},
            "security_config": {"default_passwords_present": False, "default_accounts_present": False},
            "encryption_config": {"data_at_rest_encrypted": True, "data_in_transit_encrypted": True, "key_management_implemented": True}
        },
        IndustryType.HEALTHCARE: {
            "privacy_config": {"privacy_policies_implemented": True, "minimum_necessary_standard": True, "individual_rights_procedures": True},
            "security_config": {"administrative_safeguards": True, "physical_safeguards": True, "technical_safeguards": True}
        },
        IndustryType.MANUFACTURING: {
            "iot_config": {"certificate_based_auth": True, "device_identity_management": True, "credential_rotation": True},
            "network_config": {"ot_it_segmentation": True, "dmz_implemented": True, "traffic_monitoring": True}
        }
    }
    
    assessments = []
    for industry, system_data in industries_data.items():
        assessment = orchestrator.assess_industry_compliance(industry, system_data)
        assessments.append(assessment)
        print(f"• {industry.value}: {assessment.overall_status.value} ({(len(assessment.compliant_rules) / len(assessment.rules_evaluated)) * 100:.1f}%)")
    
    # Generate cross-industry report
    cross_report = orchestrator.generate_cross_industry_report(assessments)
    
    print_subsection("Cross-Industry Summary")
    print(f"Industries Assessed: {cross_report['industries_assessed']}")
    print(f"Overall Compliance Rate: {cross_report['overall_compliance_rate']:.1f}%")
    print(f"Critical Findings: {len(cross_report['critical_findings'])}")
    
    print_subsection("Industry Breakdown")
    for industry, breakdown in cross_report["industry_breakdown"].items():
        print(f"• {industry.title()}: {breakdown['status']} ({breakdown['compliance_rate']:.1f}%)")

def main():
    """Main demo function"""
    print_section("INDUSTRY-TAILORED COMPLIANCE MODULES DEMONSTRATION")
    print("Demonstrating enterprise-grade compliance capabilities across industries")
    print("Exceeding Palantir and Databricks compliance standards")
    
    try:
        # Demo each industry compliance module
        demo_banking_compliance()
        demo_healthcare_compliance()
        demo_manufacturing_compliance()
        demo_government_compliance()
        demo_financial_services_compliance()
        demo_pharmaceutical_compliance()
        
        # Demo cross-industry assessment
        demo_cross_industry_assessment()
        
        print_section("DEMONSTRATION COMPLETE")
        print("✅ All industry compliance modules demonstrated successfully")
        print("✅ Banking: PCI DSS, SOX, Basel III, AML controls")
        print("✅ Healthcare: HIPAA, HITECH, FDA 21 CFR Part 11")
        print("✅ Manufacturing: IoT security, OT/IT convergence")
        print("✅ Government: FedRAMP, FISMA, classified data handling")
        print("✅ Financial Services: Real-time fraud detection, regulatory reporting")
        print("✅ Pharmaceutical: GxP compliance, clinical trial data integrity")
        print("✅ Cross-industry assessment and reporting capabilities")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()