"""
Manufacturing Compliance Module
Implements IoT security and OT/IT convergence capabilities
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging

from .industry_compliance_framework import (
    BaseComplianceModule, ComplianceRule, ComplianceAssessment,
    ComplianceStatus, IndustryType
)

logger = logging.getLogger(__name__)

class ManufacturingComplianceModule(BaseComplianceModule):
    """Manufacturing industry compliance module with IoT security and OT/IT convergence controls"""
    
    def __init__(self):
        super().__init__(IndustryType.MANUFACTURING)
    
    def _load_rules(self):
        """Load manufacturing-specific compliance rules"""
        
        # IoT Security Rules
        self.rules["IOT_DEVICE_AUTH"] = ComplianceRule(
            rule_id="IOT_DEVICE_AUTH",
            name="IoT Device Authentication",
            description="Implement strong authentication for all IoT devices",
            regulation="IEC 62443 Industrial Security",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Deploy certificate-based authentication",
                "Implement device identity management",
                "Regular credential rotation"
            ],
            evidence_requirements=["device_certificates", "authentication_logs", "credential_policies"]
        )
        
        self.rules["IOT_ENCRYPTION"] = ComplianceRule(
            rule_id="IOT_ENCRYPTION",
            name="IoT Communication Encryption",
            description="Encrypt all IoT device communications",
            regulation="IEC 62443 Industrial Security",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Implement end-to-end encryption",
                "Use secure communication protocols",
                "Regular encryption key updates"
            ],
            evidence_requirements=["encryption_configuration", "protocol_analysis", "key_management"]
        )
        
        self.rules["IOT_FIRMWARE"] = ComplianceRule(
            rule_id="IOT_FIRMWARE",
            name="IoT Firmware Security",
            description="Ensure secure firmware management and updates",
            regulation="IEC 62443 Industrial Security",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Implement secure boot processes",
                "Deploy signed firmware updates",
                "Firmware vulnerability scanning"
            ],
            evidence_requirements=["firmware_signatures", "update_logs", "vulnerability_scans"]
        )
        
        # OT/IT Convergence Rules
        self.rules["OT_IT_SEGMENTATION"] = ComplianceRule(
            rule_id="OT_IT_SEGMENTATION",
            name="OT/IT Network Segmentation",
            description="Implement proper network segmentation between OT and IT systems",
            regulation="NIST Cybersecurity Framework",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Deploy network segmentation controls",
                "Implement DMZ for OT/IT communication",
                "Monitor cross-network traffic"
            ],
            evidence_requirements=["network_topology", "segmentation_rules", "traffic_analysis"]
        )
        
        self.rules["OT_MONITORING"] = ComplianceRule(
            rule_id="OT_MONITORING",
            name="OT System Monitoring",
            description="Implement comprehensive monitoring of operational technology systems",
            regulation="IEC 62443 Industrial Security",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Deploy OT-specific monitoring tools",
                "Implement anomaly detection",
                "Real-time alerting systems"
            ],
            evidence_requirements=["monitoring_configuration", "alert_logs", "anomaly_reports"]
        )
        
        self.rules["INDUSTRIAL_PROTOCOLS"] = ComplianceRule(
            rule_id="INDUSTRIAL_PROTOCOLS",
            name="Industrial Protocol Security",
            description="Secure industrial communication protocols (Modbus, DNP3, etc.)",
            regulation="IEC 62443 Industrial Security",
            severity="high",
            automated_check=True,
            remediation_steps=[
                "Implement protocol security features",
                "Deploy protocol gateways",
                "Monitor protocol communications"
            ],
            evidence_requirements=["protocol_configuration", "gateway_logs", "communication_analysis"]
        )
        
        # Supply Chain Security Rules
        self.rules["SUPPLY_CHAIN_INTEGRITY"] = ComplianceRule(
            rule_id="SUPPLY_CHAIN_INTEGRITY",
            name="Supply Chain Integrity",
            description="Ensure integrity of manufacturing supply chain",
            regulation="NIST Supply Chain Risk Management",
            severity="critical",
            automated_check=False,
            remediation_steps=[
                "Implement supplier security assessments",
                "Deploy supply chain monitoring",
                "Establish incident response procedures"
            ],
            evidence_requirements=["supplier_assessments", "monitoring_reports", "incident_procedures"]
        )
        
        self.rules["COMPONENT_TRACEABILITY"] = ComplianceRule(
            rule_id="COMPONENT_TRACEABILITY",
            name="Component Traceability",
            description="Maintain traceability of all manufacturing components",
            regulation="ISO 9001 Quality Management",
            severity="medium",
            automated_check=True,
            remediation_steps=[
                "Implement component tracking systems",
                "Maintain traceability records",
                "Regular traceability audits"
            ],
            evidence_requirements=["tracking_system", "traceability_records", "audit_reports"]
        )
        
        # Safety and Environmental Rules
        self.rules["SAFETY_SYSTEMS"] = ComplianceRule(
            rule_id="SAFETY_SYSTEMS",
            name="Safety Instrumented Systems",
            description="Implement and maintain safety instrumented systems",
            regulation="IEC 61511 Functional Safety",
            severity="critical",
            automated_check=True,
            remediation_steps=[
                "Deploy safety instrumented systems",
                "Regular safety system testing",
                "Maintain safety documentation"
            ],
            evidence_requirements=["safety_system_config", "test_results", "safety_documentation"]
        )
    
    def assess_compliance(self, system_data: Dict[str, Any]) -> ComplianceAssessment:
        """Assess manufacturing compliance"""
        assessment_id = f"manufacturing_assessment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        rules_evaluated = list(self.rules.keys())
        compliant_rules = []
        non_compliant_rules = []
        recommendations = []
        
        # IoT Security Assessments
        if self._check_iot_device_authentication(system_data):
            compliant_rules.append("IOT_DEVICE_AUTH")
        else:
            non_compliant_rules.append("IOT_DEVICE_AUTH")
            recommendations.append("Implement strong authentication for all IoT devices")
        
        if self._check_iot_encryption(system_data):
            compliant_rules.append("IOT_ENCRYPTION")
        else:
            non_compliant_rules.append("IOT_ENCRYPTION")
            recommendations.append("Deploy end-to-end encryption for IoT communications")
        
        if self._check_iot_firmware_security(system_data):
            compliant_rules.append("IOT_FIRMWARE")
        else:
            non_compliant_rules.append("IOT_FIRMWARE")
            recommendations.append("Implement secure firmware management processes")
        
        # OT/IT Convergence Assessments
        if self._check_ot_it_segmentation(system_data):
            compliant_rules.append("OT_IT_SEGMENTATION")
        else:
            non_compliant_rules.append("OT_IT_SEGMENTATION")
            recommendations.append("Deploy proper network segmentation between OT and IT")
        
        if self._check_ot_monitoring(system_data):
            compliant_rules.append("OT_MONITORING")
        else:
            non_compliant_rules.append("OT_MONITORING")
            recommendations.append("Implement comprehensive OT system monitoring")
        
        if self._check_industrial_protocols(system_data):
            compliant_rules.append("INDUSTRIAL_PROTOCOLS")
        else:
            non_compliant_rules.append("INDUSTRIAL_PROTOCOLS")
            recommendations.append("Secure industrial communication protocols")
        
        # Supply Chain Assessments
        if self._check_supply_chain_integrity(system_data):
            compliant_rules.append("SUPPLY_CHAIN_INTEGRITY")
        else:
            non_compliant_rules.append("SUPPLY_CHAIN_INTEGRITY")
            recommendations.append("Enhance supply chain security controls")
        
        if self._check_component_traceability(system_data):
            compliant_rules.append("COMPONENT_TRACEABILITY")
        else:
            non_compliant_rules.append("COMPONENT_TRACEABILITY")
            recommendations.append("Implement comprehensive component traceability")
        
        # Safety System Assessment
        if self._check_safety_systems(system_data):
            compliant_rules.append("SAFETY_SYSTEMS")
        else:
            non_compliant_rules.append("SAFETY_SYSTEMS")
            recommendations.append("Deploy and maintain safety instrumented systems")
        
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
        """Generate manufacturing compliance report"""
        return {
            "assessment_id": assessment.assessment_id,
            "industry": "Manufacturing",
            "timestamp": assessment.timestamp.isoformat(),
            "overall_status": assessment.overall_status.value,
            "compliance_rate": (len(assessment.compliant_rules) / len(assessment.rules_evaluated)) * 100,
            "risk_score": assessment.risk_score,
            "security_domains": {
                "IoT_Security": self._get_domain_status(assessment, "IOT"),
                "OT_IT_Convergence": self._get_domain_status(assessment, ["OT_IT", "OT_MONITORING", "INDUSTRIAL"]),
                "Supply_Chain": self._get_domain_status(assessment, ["SUPPLY_CHAIN", "COMPONENT"]),
                "Safety_Systems": self._get_domain_status(assessment, "SAFETY")
            },
            "ot_it_convergence_status": self._assess_ot_it_convergence(assessment),
            "compliant_rules": assessment.compliant_rules,
            "non_compliant_rules": assessment.non_compliant_rules,
            "recommendations": assessment.recommendations,
            "next_assessment_due": (assessment.timestamp + timedelta(days=90)).isoformat()
        }
    
    def _check_iot_device_authentication(self, system_data: Dict[str, Any]) -> bool:
        """Check IoT device authentication"""
        iot_config = system_data.get("iot_config", {})
        return (
            iot_config.get("certificate_based_auth", False) and
            iot_config.get("device_identity_management", False) and
            iot_config.get("credential_rotation", False)
        )
    
    def _check_iot_encryption(self, system_data: Dict[str, Any]) -> bool:
        """Check IoT communication encryption"""
        iot_config = system_data.get("iot_config", {})
        return (
            iot_config.get("end_to_end_encryption", False) and
            iot_config.get("secure_protocols", False) and
            iot_config.get("key_management", False)
        )
    
    def _check_iot_firmware_security(self, system_data: Dict[str, Any]) -> bool:
        """Check IoT firmware security"""
        firmware_config = system_data.get("firmware_config", {})
        return (
            firmware_config.get("secure_boot", False) and
            firmware_config.get("signed_updates", False) and
            firmware_config.get("vulnerability_scanning", False)
        )
    
    def _check_ot_it_segmentation(self, system_data: Dict[str, Any]) -> bool:
        """Check OT/IT network segmentation"""
        network_config = system_data.get("network_config", {})
        return (
            network_config.get("ot_it_segmentation", False) and
            network_config.get("dmz_implemented", False) and
            network_config.get("traffic_monitoring", False)
        )
    
    def _check_ot_monitoring(self, system_data: Dict[str, Any]) -> bool:
        """Check OT system monitoring"""
        monitoring_config = system_data.get("ot_monitoring_config", {})
        return (
            monitoring_config.get("ot_specific_tools", False) and
            monitoring_config.get("anomaly_detection", False) and
            monitoring_config.get("real_time_alerting", False)
        )
    
    def _check_industrial_protocols(self, system_data: Dict[str, Any]) -> bool:
        """Check industrial protocol security"""
        protocol_config = system_data.get("protocol_config", {})
        return (
            protocol_config.get("protocol_security_features", False) and
            protocol_config.get("protocol_gateways", False) and
            protocol_config.get("communication_monitoring", False)
        )
    
    def _check_supply_chain_integrity(self, system_data: Dict[str, Any]) -> bool:
        """Check supply chain integrity"""
        supply_chain_config = system_data.get("supply_chain_config", {})
        return (
            supply_chain_config.get("supplier_assessments", False) and
            supply_chain_config.get("supply_chain_monitoring", False) and
            supply_chain_config.get("incident_response_procedures", False)
        )
    
    def _check_component_traceability(self, system_data: Dict[str, Any]) -> bool:
        """Check component traceability"""
        traceability_config = system_data.get("traceability_config", {})
        return (
            traceability_config.get("tracking_system", False) and
            traceability_config.get("traceability_records", False) and
            traceability_config.get("regular_audits", False)
        )
    
    def _check_safety_systems(self, system_data: Dict[str, Any]) -> bool:
        """Check safety instrumented systems"""
        safety_config = system_data.get("safety_config", {})
        return (
            safety_config.get("safety_systems_deployed", False) and
            safety_config.get("regular_testing", False) and
            safety_config.get("safety_documentation", False)
        )
    
    def _assess_ot_it_convergence(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Assess OT/IT convergence security status"""
        convergence_rules = ["OT_IT_SEGMENTATION", "OT_MONITORING", "INDUSTRIAL_PROTOCOLS"]
        compliant_convergence_rules = [rule for rule in assessment.compliant_rules if rule in convergence_rules]
        
        convergence_level = len(compliant_convergence_rules) / len(convergence_rules)
        
        if convergence_level >= 1.0:
            status = "fully_secured"
        elif convergence_level >= 0.67:
            status = "partially_secured"
        else:
            status = "inadequately_secured"
        
        return {
            "status": status,
            "security_level": convergence_level * 100,
            "compliant_controls": len(compliant_convergence_rules),
            "total_controls": len(convergence_rules)
        }
    
    def _get_domain_status(self, assessment: ComplianceAssessment, domain_prefix) -> Dict[str, Any]:
        """Get status for specific security domain"""
        if isinstance(domain_prefix, list):
            domain_rules = [rule for rule in assessment.rules_evaluated 
                          if any(rule.startswith(prefix) for prefix in domain_prefix)]
            compliant_domain_rules = [rule for rule in assessment.compliant_rules 
                                    if any(rule.startswith(prefix) for prefix in domain_prefix)]
        else:
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