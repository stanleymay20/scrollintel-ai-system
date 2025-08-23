"""
Security Requirement Templates for Vendor Contracts and SLAs
Implements comprehensive security requirement templates and contract management
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class ContractType(Enum):
    SOFTWARE_LICENSE = "software_license"
    CLOUD_SERVICE = "cloud_service"
    DATA_PROCESSING = "data_processing"
    CONSULTING_SERVICE = "consulting_service"
    MANAGED_SERVICE = "managed_service"
    SAAS_SUBSCRIPTION = "saas_subscription"

class ComplianceFramework(Enum):
    SOC2_TYPE_II = "soc2_type_ii"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    CCPA = "ccpa"
    FISMA = "fisma"
    FEDRAMP = "fedramp"

class SecurityRequirementCategory(Enum):
    DATA_PROTECTION = "data_protection"
    ACCESS_CONTROL = "access_control"
    INCIDENT_RESPONSE = "incident_response"
    VULNERABILITY_MANAGEMENT = "vulnerability_management"
    BUSINESS_CONTINUITY = "business_continuity"
    COMPLIANCE_REPORTING = "compliance_reporting"
    AUDIT_LOGGING = "audit_logging"
    ENCRYPTION = "encryption"

@dataclass
class SecurityRequirement:
    requirement_id: str
    category: SecurityRequirementCategory
    title: str
    description: str
    mandatory: bool
    compliance_frameworks: List[ComplianceFramework]
    verification_method: str
    acceptance_criteria: List[str]
    sla_metrics: Dict[str, Any]
    penalty_clause: Optional[str]
    review_frequency: str  # monthly, quarterly, annually

@dataclass
class ContractTemplate:
    template_id: str
    name: str
    contract_type: ContractType
    version: str
    created_at: datetime
    updated_at: datetime
    security_requirements: List[SecurityRequirement]
    standard_clauses: Dict[str, str]
    compliance_requirements: List[ComplianceFramework]
    risk_assessment_required: bool
    security_assessment_frequency: str

class SecurityContractTemplateManager:
    def __init__(self, config_path: str = "security/config/contract_config.yaml"):
        self.config = self._load_config(config_path)
        self.templates = {}
        self.requirement_library = self._initialize_requirement_library()
        self.standard_clauses = self._initialize_standard_clauses()
        self._create_default_templates()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load contract template configuration"""
        default_config = {
            "template_versioning": {
                "major_version_triggers": [
                    "new_compliance_framework",
                    "significant_security_change",
                    "legal_requirement_change"
                ],
                "minor_version_triggers": [
                    "requirement_clarification",
                    "sla_adjustment",
                    "process_improvement"
                ]
            },
            "default_sla_metrics": {
                "incident_response_time": "4 hours",
                "vulnerability_patching": "30 days for high, 90 days for medium",
                "security_assessment_frequency": "annually",
                "compliance_reporting": "quarterly"
            },
            "penalty_structures": {
                "data_breach": "Up to $1M or 2% of annual contract value",
                "sla_breach": "5% reduction in monthly fees per breach",
                "compliance_failure": "Immediate contract review and potential termination"
            }
        }
        
        try:
            # In production, load from actual config file
            return default_config
        except Exception:
            return default_config
    
    def _initialize_requirement_library(self) -> Dict[str, SecurityRequirement]:
        """Initialize comprehensive security requirement library"""
        requirements = {}
        
        # Data Protection Requirements
        requirements["DP001"] = SecurityRequirement(
            requirement_id="DP001",
            category=SecurityRequirementCategory.DATA_PROTECTION,
            title="Data Encryption at Rest",
            description="All data must be encrypted at rest using AES-256 or equivalent encryption standard approved by NIST.",
            mandatory=True,
            compliance_frameworks=[ComplianceFramework.SOC2_TYPE_II, ComplianceFramework.ISO27001, ComplianceFramework.PCI_DSS],
            verification_method="Technical audit and certification review",
            acceptance_criteria=[
                "AES-256 encryption or NIST-approved equivalent implemented",
                "Encryption keys managed through HSM or equivalent secure key management",
                "Regular key rotation procedures documented and implemented",
                "Encryption implementation validated by third-party security assessment"
            ],
            sla_metrics={
                "encryption_coverage": "100% of sensitive data",
                "key_rotation_frequency": "annually",
                "compliance_verification": "annually"
            },
            penalty_clause="Failure to maintain encryption standards may result in immediate contract suspension",
            review_frequency="annually"
        )
        
        requirements["DP002"] = SecurityRequirement(
            requirement_id="DP002",
            category=SecurityRequirementCategory.DATA_PROTECTION,
            title="Data Encryption in Transit",
            description="All data transmissions must use TLS 1.3 or higher with perfect forward secrecy.",
            mandatory=True,
            compliance_frameworks=[ComplianceFramework.SOC2_TYPE_II, ComplianceFramework.ISO27001, ComplianceFramework.PCI_DSS, ComplianceFramework.HIPAA],
            verification_method="Network security assessment and SSL/TLS configuration review",
            acceptance_criteria=[
                "TLS 1.3 or higher implemented for all data transmissions",
                "Perfect forward secrecy enabled",
                "Strong cipher suites only (no deprecated algorithms)",
                "Certificate management procedures documented"
            ],
            sla_metrics={
                "tls_version_compliance": "100%",
                "certificate_validity": "continuous",
                "security_scan_frequency": "monthly"
            },
            penalty_clause="Use of deprecated encryption protocols results in immediate remediation requirement",
            review_frequency="quarterly"
        )
        
        requirements["DP003"] = SecurityRequirement(
            requirement_id="DP003",
            category=SecurityRequirementCategory.DATA_PROTECTION,
            title="Data Loss Prevention",
            description="Implement comprehensive data loss prevention controls to prevent unauthorized data exfiltration.",
            mandatory=True,
            compliance_frameworks=[ComplianceFramework.SOC2_TYPE_II, ComplianceFramework.ISO27001, ComplianceFramework.GDPR],
            verification_method="DLP system configuration review and testing",
            acceptance_criteria=[
                "DLP solution deployed and configured",
                "Data classification system implemented",
                "Automated blocking of unauthorized data transfers",
                "Regular DLP policy updates and testing"
            ],
            sla_metrics={
                "dlp_coverage": "100% of sensitive data flows",
                "false_positive_rate": "<5%",
                "incident_detection_time": "real-time"
            },
            penalty_clause="Data exfiltration incidents due to DLP failures subject to breach penalties",
            review_frequency="quarterly"
        )
        
        # Access Control Requirements
        requirements["AC001"] = SecurityRequirement(
            requirement_id="AC001",
            category=SecurityRequirementCategory.ACCESS_CONTROL,
            title="Multi-Factor Authentication",
            description="Multi-factor authentication must be enforced for all user accounts accessing sensitive systems or data.",
            mandatory=True,
            compliance_frameworks=[ComplianceFramework.SOC2_TYPE_II, ComplianceFramework.ISO27001, ComplianceFramework.PCI_DSS, ComplianceFramework.HIPAA],
            verification_method="Access control system audit and user access testing",
            acceptance_criteria=[
                "MFA enabled for all user accounts",
                "Support for multiple authentication factors (TOTP, SMS, biometric)",
                "MFA bypass procedures documented and approved",
                "Regular MFA compliance monitoring"
            ],
            sla_metrics={
                "mfa_coverage": "100% of user accounts",
                "mfa_bypass_approval_time": "24 hours",
                "compliance_monitoring": "continuous"
            },
            penalty_clause="Unauthorized access due to MFA failures subject to security incident penalties",
            review_frequency="quarterly"
        )
        
        requirements["AC002"] = SecurityRequirement(
            requirement_id="AC002",
            category=SecurityRequirementCategory.ACCESS_CONTROL,
            title="Privileged Access Management",
            description="Implement just-in-time privileged access with comprehensive monitoring and approval workflows.",
            mandatory=True,
            compliance_frameworks=[ComplianceFramework.SOC2_TYPE_II, ComplianceFramework.ISO27001, ComplianceFramework.PCI_DSS],
            verification_method="PAM system configuration review and access log analysis",
            acceptance_criteria=[
                "Just-in-time access provisioning implemented",
                "Privileged access approval workflows configured",
                "Session recording for privileged access",
                "Regular privileged access reviews and certifications"
            ],
            sla_metrics={
                "access_approval_time": "2 hours during business hours",
                "session_recording_coverage": "100% of privileged sessions",
                "access_review_frequency": "monthly"
            },
            penalty_clause="Unauthorized privileged access incidents subject to immediate investigation",
            review_frequency="monthly"
        )
        
        # Incident Response Requirements
        requirements["IR001"] = SecurityRequirement(
            requirement_id="IR001",
            category=SecurityRequirementCategory.INCIDENT_RESPONSE,
            title="Security Incident Response Plan",
            description="Maintain and regularly test a comprehensive security incident response plan with defined escalation procedures.",
            mandatory=True,
            compliance_frameworks=[ComplianceFramework.SOC2_TYPE_II, ComplianceFramework.ISO27001, ComplianceFramework.HIPAA],
            verification_method="Incident response plan review and tabletop exercise",
            acceptance_criteria=[
                "Documented incident response plan with clear procedures",
                "24/7 incident response capability",
                "Regular incident response training and testing",
                "Integration with customer incident response procedures"
            ],
            sla_metrics={
                "incident_acknowledgment": "15 minutes",
                "initial_response": "1 hour",
                "customer_notification": "2 hours for high/critical incidents",
                "plan_testing_frequency": "annually"
            },
            penalty_clause="Failure to meet incident response SLAs results in service credits",
            review_frequency="annually"
        )
        
        requirements["IR002"] = SecurityRequirement(
            requirement_id="IR002",
            category=SecurityRequirementCategory.INCIDENT_RESPONSE,
            title="Security Incident Notification",
            description="Provide timely notification of security incidents that may affect customer data or services.",
            mandatory=True,
            compliance_frameworks=[ComplianceFramework.SOC2_TYPE_II, ComplianceFramework.GDPR, ComplianceFramework.HIPAA, ComplianceFramework.CCPA],
            verification_method="Incident notification procedures review and communication testing",
            acceptance_criteria=[
                "Clear incident classification and notification criteria",
                "Multiple communication channels for notifications",
                "Detailed incident reports provided within specified timeframes",
                "Regular communication during incident resolution"
            ],
            sla_metrics={
                "critical_incident_notification": "2 hours",
                "high_incident_notification": "8 hours",
                "incident_report_delivery": "72 hours",
                "resolution_updates": "daily for ongoing incidents"
            },
            penalty_clause="Late notification of incidents may result in contract penalties",
            review_frequency="quarterly"
        )
        
        # Vulnerability Management Requirements
        requirements["VM001"] = SecurityRequirement(
            requirement_id="VM001",
            category=SecurityRequirementCategory.VULNERABILITY_MANAGEMENT,
            title="Vulnerability Assessment and Patching",
            description="Conduct regular vulnerability assessments and maintain timely patching procedures for identified vulnerabilities.",
            mandatory=True,
            compliance_frameworks=[ComplianceFramework.SOC2_TYPE_II, ComplianceFramework.ISO27001, ComplianceFramework.PCI_DSS],
            verification_method="Vulnerability scan reports and patch management documentation review",
            acceptance_criteria=[
                "Monthly vulnerability scans of all systems",
                "Critical vulnerabilities patched within 72 hours",
                "High vulnerabilities patched within 30 days",
                "Vulnerability management reporting provided quarterly"
            ],
            sla_metrics={
                "critical_patch_time": "72 hours",
                "high_patch_time": "30 days",
                "medium_patch_time": "90 days",
                "scan_frequency": "monthly"
            },
            penalty_clause="Unpatched critical vulnerabilities may result in service suspension",
            review_frequency="quarterly"
        )
        
        # Business Continuity Requirements
        requirements["BC001"] = SecurityRequirement(
            requirement_id="BC001",
            category=SecurityRequirementCategory.BUSINESS_CONTINUITY,
            title="Business Continuity and Disaster Recovery",
            description="Maintain comprehensive business continuity and disaster recovery capabilities with regular testing.",
            mandatory=True,
            compliance_frameworks=[ComplianceFramework.SOC2_TYPE_II, ComplianceFramework.ISO27001],
            verification_method="BCP/DR plan review and recovery testing validation",
            acceptance_criteria=[
                "Documented business continuity plan",
                "Disaster recovery procedures with defined RTO/RPO",
                "Regular backup testing and validation",
                "Annual disaster recovery testing"
            ],
            sla_metrics={
                "rto": "4 hours",
                "rpo": "1 hour",
                "backup_frequency": "daily",
                "dr_test_frequency": "annually"
            },
            penalty_clause="Failure to meet RTO/RPO commitments results in service level credits",
            review_frequency="annually"
        )
        
        # Compliance Reporting Requirements
        requirements["CR001"] = SecurityRequirement(
            requirement_id="CR001",
            category=SecurityRequirementCategory.COMPLIANCE_REPORTING,
            title="Compliance Reporting and Attestation",
            description="Provide regular compliance reports and maintain current security certifications.",
            mandatory=True,
            compliance_frameworks=[ComplianceFramework.SOC2_TYPE_II, ComplianceFramework.ISO27001, ComplianceFramework.PCI_DSS],
            verification_method="Compliance report review and certification validation",
            acceptance_criteria=[
                "Current SOC 2 Type II report provided annually",
                "Quarterly compliance status reports",
                "Immediate notification of compliance status changes",
                "Third-party audit results shared upon request"
            ],
            sla_metrics={
                "annual_report_delivery": "within 90 days of audit completion",
                "quarterly_report_delivery": "within 30 days of quarter end",
                "compliance_change_notification": "within 48 hours"
            },
            penalty_clause="Loss of required certifications may result in contract termination",
            review_frequency="quarterly"
        )
        
        # Audit Logging Requirements
        requirements["AL001"] = SecurityRequirement(
            requirement_id="AL001",
            category=SecurityRequirementCategory.AUDIT_LOGGING,
            title="Comprehensive Audit Logging",
            description="Implement comprehensive audit logging with secure log storage and retention.",
            mandatory=True,
            compliance_frameworks=[ComplianceFramework.SOC2_TYPE_II, ComplianceFramework.ISO27001, ComplianceFramework.PCI_DSS, ComplianceFramework.HIPAA],
            verification_method="Audit log configuration review and log analysis",
            acceptance_criteria=[
                "All security-relevant events logged",
                "Tamper-evident log storage",
                "Log retention for minimum 7 years",
                "Real-time log monitoring and alerting"
            ],
            sla_metrics={
                "log_completeness": "100% of security events",
                "log_integrity": "cryptographically protected",
                "log_availability": "99.9% uptime",
                "retention_period": "7 years minimum"
            },
            penalty_clause="Log tampering or unavailability incidents subject to security breach penalties",
            review_frequency="quarterly"
        )
        
        # Encryption Requirements
        requirements["EN001"] = SecurityRequirement(
            requirement_id="EN001",
            category=SecurityRequirementCategory.ENCRYPTION,
            title="Cryptographic Key Management",
            description="Implement secure cryptographic key management using hardware security modules or equivalent.",
            mandatory=True,
            compliance_frameworks=[ComplianceFramework.SOC2_TYPE_II, ComplianceFramework.ISO27001, ComplianceFramework.PCI_DSS, ComplianceFramework.FEDRAMP],
            verification_method="Key management system audit and cryptographic implementation review",
            acceptance_criteria=[
                "HSM or equivalent secure key storage",
                "Key lifecycle management procedures",
                "Regular key rotation and escrow procedures",
                "Cryptographic algorithm compliance with current standards"
            ],
            sla_metrics={
                "key_availability": "99.99%",
                "key_rotation_frequency": "annually for encryption keys",
                "algorithm_compliance": "NIST-approved algorithms only"
            },
            penalty_clause="Cryptographic failures may result in immediate security review",
            review_frequency="annually"
        )
        
        return requirements
    
    def _initialize_standard_clauses(self) -> Dict[str, str]:
        """Initialize standard contract clauses"""
        return {
            "data_protection_clause": """
DATA PROTECTION AND PRIVACY

Vendor shall implement and maintain appropriate technical and organizational measures to protect Customer Data against unauthorized access, disclosure, alteration, or destruction. Such measures shall include but not be limited to:

a) Encryption of data at rest and in transit using industry-standard encryption algorithms
b) Access controls limiting data access to authorized personnel only
c) Regular security assessments and vulnerability management
d) Incident response procedures with timely notification requirements
e) Data backup and recovery procedures
f) Compliance with applicable data protection regulations including GDPR, CCPA, and HIPAA where applicable

Vendor warrants that all personnel with access to Customer Data have undergone appropriate background checks and security training.
            """,
            
            "security_incident_clause": """
SECURITY INCIDENT RESPONSE

In the event of any actual or suspected security incident affecting Customer Data or Services:

a) Vendor shall notify Customer within two (2) hours of becoming aware of any critical security incident
b) Vendor shall provide detailed incident reports within seventy-two (72) hours
c) Vendor shall cooperate fully with Customer's incident response procedures
d) Vendor shall implement immediate containment and remediation measures
e) Vendor shall provide regular status updates until incident resolution
f) Vendor shall conduct post-incident analysis and implement preventive measures

Security incidents include but are not limited to unauthorized access, data breaches, malware infections, and service disruptions with security implications.
            """,
            
            "compliance_clause": """
REGULATORY COMPLIANCE

Vendor shall maintain compliance with all applicable laws, regulations, and industry standards relevant to the Services provided, including but not limited to:

a) SOC 2 Type II certification (to be maintained throughout contract term)
b) ISO 27001 certification where applicable
c) Industry-specific regulations (PCI DSS, HIPAA, etc.) as relevant
d) Data protection regulations (GDPR, CCPA, etc.)
e) Export control and trade sanctions regulations

Vendor shall provide evidence of compliance upon Customer request and notify Customer immediately of any compliance status changes or regulatory violations.
            """,
            
            "audit_rights_clause": """
AUDIT RIGHTS

Customer reserves the right to audit Vendor's security controls and compliance with this Agreement:

a) Customer may conduct security audits annually or upon reasonable suspicion of non-compliance
b) Vendor shall provide reasonable cooperation and access for audit activities
c) Audits may be conducted by Customer or qualified third-party auditors
d) Vendor shall remediate any identified deficiencies within agreed timeframes
e) Audit costs shall be borne by Customer unless material non-compliance is identified
f) Vendor shall maintain audit logs and documentation for review

Vendor shall provide SOC 2 Type II reports and other compliance documentation annually.
            """,
            
            "data_return_clause": """
DATA RETURN AND DESTRUCTION

Upon termination or expiration of this Agreement:

a) Vendor shall return all Customer Data in a commonly used electronic format within thirty (30) days
b) Vendor shall securely destroy all copies of Customer Data within ninety (90) days
c) Destruction shall be performed using NIST-approved methods
d) Vendor shall provide written certification of data destruction
e) Backup copies may be retained only as required by law with continued protection obligations
f) Any data retention shall be subject to the same security and confidentiality requirements

Customer may request data return or destruction at any time during the contract term.
            """,
            
            "liability_clause": """
SECURITY LIABILITY

Vendor's liability for security breaches and data protection failures:

a) Vendor shall be liable for direct damages resulting from security breaches caused by Vendor's negligence
b) Liability cap shall not apply to security breaches involving Customer Data
c) Vendor shall maintain cyber liability insurance of not less than $10 million
d) Vendor shall indemnify Customer for regulatory fines and penalties resulting from Vendor's non-compliance
e) Vendor shall bear costs of breach notification and credit monitoring services
f) Liability shall include business interruption and reputational damages

This liability clause supplements and does not limit other liability provisions in this Agreement.
            """,
            
            "termination_clause": """
SECURITY-RELATED TERMINATION

Customer may terminate this Agreement immediately upon:

a) Material security breach by Vendor affecting Customer Data
b) Vendor's failure to maintain required security certifications
c) Vendor's non-compliance with applicable data protection regulations
d) Vendor's failure to remediate critical security vulnerabilities within required timeframes
e) Loss or compromise of Vendor's security certifications
f) Vendor's refusal to cooperate with security audits or investigations

Upon security-related termination, Vendor shall immediately secure all Customer Data and systems.
            """
        }
    
    def _create_default_templates(self):
        """Create default contract templates for common scenarios"""
        
        # Cloud Service Template
        cloud_template = ContractTemplate(
            template_id="TMPL-CLOUD-001",
            name="Cloud Service Provider Security Requirements",
            contract_type=ContractType.CLOUD_SERVICE,
            version="1.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            security_requirements=[
                self.requirement_library["DP001"],  # Data Encryption at Rest
                self.requirement_library["DP002"],  # Data Encryption in Transit
                self.requirement_library["DP003"],  # Data Loss Prevention
                self.requirement_library["AC001"],  # Multi-Factor Authentication
                self.requirement_library["AC002"],  # Privileged Access Management
                self.requirement_library["IR001"],  # Incident Response Plan
                self.requirement_library["IR002"],  # Incident Notification
                self.requirement_library["VM001"],  # Vulnerability Management
                self.requirement_library["BC001"],  # Business Continuity
                self.requirement_library["CR001"],  # Compliance Reporting
                self.requirement_library["AL001"],  # Audit Logging
                self.requirement_library["EN001"]   # Key Management
            ],
            standard_clauses={
                "data_protection": self.standard_clauses["data_protection_clause"],
                "security_incident": self.standard_clauses["security_incident_clause"],
                "compliance": self.standard_clauses["compliance_clause"],
                "audit_rights": self.standard_clauses["audit_rights_clause"],
                "data_return": self.standard_clauses["data_return_clause"],
                "liability": self.standard_clauses["liability_clause"],
                "termination": self.standard_clauses["termination_clause"]
            },
            compliance_requirements=[
                ComplianceFramework.SOC2_TYPE_II,
                ComplianceFramework.ISO27001,
                ComplianceFramework.GDPR
            ],
            risk_assessment_required=True,
            security_assessment_frequency="annually"
        )
        
        # SaaS Subscription Template
        saas_template = ContractTemplate(
            template_id="TMPL-SAAS-001",
            name="SaaS Application Security Requirements",
            contract_type=ContractType.SAAS_SUBSCRIPTION,
            version="1.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            security_requirements=[
                self.requirement_library["DP001"],  # Data Encryption at Rest
                self.requirement_library["DP002"],  # Data Encryption in Transit
                self.requirement_library["AC001"],  # Multi-Factor Authentication
                self.requirement_library["IR001"],  # Incident Response Plan
                self.requirement_library["IR002"],  # Incident Notification
                self.requirement_library["VM001"],  # Vulnerability Management
                self.requirement_library["CR001"],  # Compliance Reporting
                self.requirement_library["AL001"]   # Audit Logging
            ],
            standard_clauses={
                "data_protection": self.standard_clauses["data_protection_clause"],
                "security_incident": self.standard_clauses["security_incident_clause"],
                "compliance": self.standard_clauses["compliance_clause"],
                "audit_rights": self.standard_clauses["audit_rights_clause"],
                "data_return": self.standard_clauses["data_return_clause"]
            },
            compliance_requirements=[
                ComplianceFramework.SOC2_TYPE_II,
                ComplianceFramework.GDPR
            ],
            risk_assessment_required=True,
            security_assessment_frequency="annually"
        )
        
        # Data Processing Template
        data_processing_template = ContractTemplate(
            template_id="TMPL-DPA-001",
            name="Data Processing Agreement Security Requirements",
            contract_type=ContractType.DATA_PROCESSING,
            version="1.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            security_requirements=[
                self.requirement_library["DP001"],  # Data Encryption at Rest
                self.requirement_library["DP002"],  # Data Encryption in Transit
                self.requirement_library["DP003"],  # Data Loss Prevention
                self.requirement_library["AC001"],  # Multi-Factor Authentication
                self.requirement_library["AC002"],  # Privileged Access Management
                self.requirement_library["IR001"],  # Incident Response Plan
                self.requirement_library["IR002"],  # Incident Notification
                self.requirement_library["AL001"],  # Audit Logging
                self.requirement_library["EN001"]   # Key Management
            ],
            standard_clauses={
                "data_protection": self.standard_clauses["data_protection_clause"],
                "security_incident": self.standard_clauses["security_incident_clause"],
                "compliance": self.standard_clauses["compliance_clause"],
                "audit_rights": self.standard_clauses["audit_rights_clause"],
                "data_return": self.standard_clauses["data_return_clause"],
                "liability": self.standard_clauses["liability_clause"]
            },
            compliance_requirements=[
                ComplianceFramework.GDPR,
                ComplianceFramework.CCPA,
                ComplianceFramework.SOC2_TYPE_II
            ],
            risk_assessment_required=True,
            security_assessment_frequency="annually"
        )
        
        # Store templates
        self.templates[cloud_template.template_id] = cloud_template
        self.templates[saas_template.template_id] = saas_template
        self.templates[data_processing_template.template_id] = data_processing_template
    
    def get_template(self, template_id: str) -> Optional[ContractTemplate]:
        """Get contract template by ID"""
        return self.templates.get(template_id)
    
    def get_templates_by_type(self, contract_type: ContractType) -> List[ContractTemplate]:
        """Get templates by contract type"""
        return [template for template in self.templates.values() 
                if template.contract_type == contract_type]
    
    def create_custom_template(self, name: str, contract_type: ContractType,
                             requirement_ids: List[str], 
                             compliance_frameworks: List[ComplianceFramework],
                             additional_clauses: Optional[Dict[str, str]] = None) -> ContractTemplate:
        """Create custom contract template"""
        template_id = f"TMPL-CUSTOM-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Get requirements
        requirements = []
        for req_id in requirement_ids:
            if req_id in self.requirement_library:
                requirements.append(self.requirement_library[req_id])
        
        # Combine standard and additional clauses
        clauses = {
            "data_protection": self.standard_clauses["data_protection_clause"],
            "security_incident": self.standard_clauses["security_incident_clause"],
            "compliance": self.standard_clauses["compliance_clause"],
            "audit_rights": self.standard_clauses["audit_rights_clause"]
        }
        
        if additional_clauses:
            clauses.update(additional_clauses)
        
        template = ContractTemplate(
            template_id=template_id,
            name=name,
            contract_type=contract_type,
            version="1.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            security_requirements=requirements,
            standard_clauses=clauses,
            compliance_requirements=compliance_frameworks,
            risk_assessment_required=True,
            security_assessment_frequency="annually"
        )
        
        self.templates[template_id] = template
        return template
    
    def generate_contract_language(self, template_id: str, 
                                 vendor_name: str,
                                 service_description: str,
                                 custom_requirements: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate complete contract language from template"""
        template = self.get_template(template_id)
        if not template:
            return {"error": "Template not found"}
        
        contract_sections = {
            "header": self._generate_contract_header(vendor_name, service_description, template),
            "security_requirements": self._generate_security_requirements_section(template, custom_requirements),
            "compliance_requirements": self._generate_compliance_section(template),
            "sla_metrics": self._generate_sla_section(template),
            "standard_clauses": template.standard_clauses,
            "appendices": self._generate_appendices(template)
        }
        
        return {
            "template_id": template_id,
            "vendor_name": vendor_name,
            "service_description": service_description,
            "generated_at": datetime.now().isoformat(),
            "contract_sections": contract_sections,
            "compliance_frameworks": [cf.value for cf in template.compliance_requirements],
            "total_requirements": len(template.security_requirements)
        }
    
    def _generate_contract_header(self, vendor_name: str, service_description: str, 
                                template: ContractTemplate) -> str:
        """Generate contract header section"""
        return f"""
SECURITY REQUIREMENTS ADDENDUM

This Security Requirements Addendum ("Addendum") is incorporated into and forms part of the agreement between Customer and {vendor_name} ("Vendor") for {service_description} ("Services").

Template: {template.name} (Version {template.version})
Contract Type: {template.contract_type.value}
Generated: {datetime.now().strftime('%B %d, %Y')}

The security requirements outlined in this Addendum are mandatory and shall be maintained throughout the term of the Agreement. Failure to comply with these requirements may result in contract termination and associated penalties.
        """
    
    def _generate_security_requirements_section(self, template: ContractTemplate,
                                              custom_requirements: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate security requirements section"""
        sections = {}
        
        # Group requirements by category
        by_category = {}
        for req in template.security_requirements:
            category = req.category.value
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(req)
        
        # Generate section for each category
        for category, requirements in by_category.items():
            section_content = f"\n{category.replace('_', ' ').title()}\n\n"
            
            for req in requirements:
                section_content += f"{req.requirement_id}: {req.title}\n"
                section_content += f"Description: {req.description}\n"
                section_content += f"Mandatory: {'Yes' if req.mandatory else 'No'}\n"
                section_content += f"Verification: {req.verification_method}\n"
                
                section_content += "Acceptance Criteria:\n"
                for criteria in req.acceptance_criteria:
                    section_content += f"  • {criteria}\n"
                
                section_content += f"Review Frequency: {req.review_frequency}\n"
                
                if req.penalty_clause:
                    section_content += f"Penalty Clause: {req.penalty_clause}\n"
                
                section_content += "\n"
            
            sections[category] = section_content
        
        # Add custom requirements if provided
        if custom_requirements:
            sections["custom_requirements"] = "\n".join(custom_requirements)
        
        return sections
    
    def _generate_compliance_section(self, template: ContractTemplate) -> str:
        """Generate compliance requirements section"""
        compliance_text = "COMPLIANCE REQUIREMENTS\n\n"
        compliance_text += "Vendor shall maintain compliance with the following frameworks:\n\n"
        
        for framework in template.compliance_requirements:
            compliance_text += f"• {framework.value.upper().replace('_', ' ')}\n"
        
        compliance_text += f"\nSecurity assessments shall be conducted {template.security_assessment_frequency}.\n"
        
        if template.risk_assessment_required:
            compliance_text += "Risk assessments are required prior to service commencement and annually thereafter.\n"
        
        return compliance_text
    
    def _generate_sla_section(self, template: ContractTemplate) -> Dict[str, Any]:
        """Generate SLA metrics section"""
        sla_metrics = {}
        
        for req in template.security_requirements:
            if req.sla_metrics:
                sla_metrics[req.requirement_id] = {
                    "title": req.title,
                    "metrics": req.sla_metrics
                }
        
        return sla_metrics
    
    def _generate_appendices(self, template: ContractTemplate) -> Dict[str, Any]:
        """Generate contract appendices"""
        return {
            "appendix_a": "Security Requirements Detail Matrix",
            "appendix_b": "Compliance Framework Mapping",
            "appendix_c": "SLA Metrics and Penalties",
            "appendix_d": "Incident Response Procedures",
            "appendix_e": "Audit and Assessment Schedule"
        }
    
    def validate_vendor_compliance(self, template_id: str, 
                                 vendor_certifications: List[str],
                                 vendor_capabilities: Dict[str, bool]) -> Dict[str, Any]:
        """Validate vendor compliance against template requirements"""
        template = self.get_template(template_id)
        if not template:
            return {"error": "Template not found"}
        
        compliance_results = {
            "template_id": template_id,
            "vendor_certifications": vendor_certifications,
            "assessment_date": datetime.now().isoformat(),
            "overall_compliance": True,
            "requirement_compliance": [],
            "missing_certifications": [],
            "recommendations": []
        }
        
        # Check compliance framework requirements
        required_frameworks = [cf.value for cf in template.compliance_requirements]
        missing_certs = [cf for cf in required_frameworks if cf not in vendor_certifications]
        compliance_results["missing_certifications"] = missing_certs
        
        if missing_certs:
            compliance_results["overall_compliance"] = False
        
        # Check individual requirements
        for req in template.security_requirements:
            req_compliance = {
                "requirement_id": req.requirement_id,
                "title": req.title,
                "mandatory": req.mandatory,
                "compliant": True,
                "gaps": []
            }
            
            # Check if vendor has required capabilities
            for criteria in req.acceptance_criteria:
                # Simplified capability check (in production, this would be more sophisticated)
                capability_key = req.requirement_id.lower()
                if capability_key in vendor_capabilities:
                    if not vendor_capabilities[capability_key]:
                        req_compliance["compliant"] = False
                        req_compliance["gaps"].append(criteria)
                        if req.mandatory:
                            compliance_results["overall_compliance"] = False
            
            compliance_results["requirement_compliance"].append(req_compliance)
        
        # Generate recommendations
        if missing_certs:
            compliance_results["recommendations"].append(
                f"Obtain missing certifications: {', '.join(missing_certs)}"
            )
        
        non_compliant_mandatory = [
            rc for rc in compliance_results["requirement_compliance"]
            if not rc["compliant"] and rc["mandatory"]
        ]
        
        if non_compliant_mandatory:
            compliance_results["recommendations"].append(
                "Address mandatory requirement gaps before contract execution"
            )
        
        return compliance_results
    
    def export_template(self, template_id: str, format: str = "json") -> Optional[str]:
        """Export template in specified format"""
        template = self.get_template(template_id)
        if not template:
            return None
        
        if format.lower() == "json":
            return json.dumps(asdict(template), indent=2, default=str)
        elif format.lower() == "yaml":
            # In production, implement YAML export
            return "YAML export not implemented"
        else:
            return None
    
    def get_requirement_by_id(self, requirement_id: str) -> Optional[SecurityRequirement]:
        """Get security requirement by ID"""
        return self.requirement_library.get(requirement_id)
    
    def search_requirements(self, category: Optional[SecurityRequirementCategory] = None,
                          compliance_framework: Optional[ComplianceFramework] = None,
                          mandatory_only: bool = False) -> List[SecurityRequirement]:
        """Search requirements by criteria"""
        results = list(self.requirement_library.values())
        
        if category:
            results = [req for req in results if req.category == category]
        
        if compliance_framework:
            results = [req for req in results if compliance_framework in req.compliance_frameworks]
        
        if mandatory_only:
            results = [req for req in results if req.mandatory]
        
        return results
    
    def generate_requirement_matrix(self, template_id: str) -> Dict[str, Any]:
        """Generate requirements traceability matrix"""
        template = self.get_template(template_id)
        if not template:
            return {"error": "Template not found"}
        
        matrix = {
            "template_id": template_id,
            "template_name": template.name,
            "generated_at": datetime.now().isoformat(),
            "requirements_by_category": {},
            "compliance_mapping": {},
            "sla_summary": {}
        }
        
        # Group by category
        for req in template.security_requirements:
            category = req.category.value
            if category not in matrix["requirements_by_category"]:
                matrix["requirements_by_category"][category] = []
            
            matrix["requirements_by_category"][category].append({
                "id": req.requirement_id,
                "title": req.title,
                "mandatory": req.mandatory,
                "review_frequency": req.review_frequency
            })
        
        # Compliance mapping
        for framework in template.compliance_requirements:
            framework_reqs = [
                req.requirement_id for req in template.security_requirements
                if framework in req.compliance_frameworks
            ]
            matrix["compliance_mapping"][framework.value] = framework_reqs
        
        # SLA summary
        for req in template.security_requirements:
            if req.sla_metrics:
                matrix["sla_summary"][req.requirement_id] = req.sla_metrics
        
        return matrix