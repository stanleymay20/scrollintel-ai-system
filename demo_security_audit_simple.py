"""
Simple Demo: Security Audit and SIEM Integration System
Demonstrates the security audit system without database dependencies
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from scrollintel.models.security_audit_models import (
    SecurityEventType, SeverityLevel, SIEMPlatform, ComplianceFramework,
    SecurityAuditLogCreate, SIEMIntegrationCreate, ThreatDetectionRuleCreate
)

class SecurityAuditSimpleDemo:
    """Simple security audit system demonstration"""
    
    def __init__(self):
        self.demo_events = []
        self.demo_alerts = []
        self.demo_metrics = {
            "total_events": 0,
            "critical_events": 0,
            "high_severity_events": 0,
            "threat_detections": 0
        }
    
    async def run_demo(self):
        """Run simple security audit system demonstration"""
        print("🔒 Security Audit and SIEM Integration System Demo")
        print("=" * 60)
        
        # 1. Security Event Models
        self.demo_security_event_models()
        
        # 2. SIEM Integration Models
        self.demo_siem_integration_models()
        
        # 3. Threat Detection Patterns
        self.demo_threat_detection_patterns()
        
        # 4. Compliance Framework Support
        self.demo_compliance_frameworks()
        
        # 5. Security Event Processing
        await self.demo_security_event_processing()
        
        # 6. SIEM Event Formatting
        self.demo_siem_event_formatting()
        
        # 7. Compliance Reporting Structure
        self.demo_compliance_reporting_structure()
        
        # 8. Security Metrics
        self.demo_security_metrics()
        
        print("\n✅ Security Audit System Demo Complete!")
        print("All security models and patterns demonstrated successfully.")
    
    def demo_security_event_models(self):
        """Demonstrate security event models"""
        print("\n📝 1. Security Event Models")
        print("-" * 40)
        
        # Authentication event
        auth_event = SecurityAuditLogCreate(
            event_type=SecurityEventType.AUTHENTICATION,
            action="login",
            outcome="success",
            severity=SeverityLevel.LOW,
            user_id="alice.admin",
            source_ip="192.168.1.100",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            details={
                "method": "password_mfa",
                "mfa_type": "totp",
                "session_duration": 3600
            }
        )
        
        print(f"✓ Authentication Event Model:")
        print(f"  Type: {auth_event.event_type.value}")
        print(f"  Action: {auth_event.action}")
        print(f"  Severity: {auth_event.severity.value}")
        print(f"  User: {auth_event.user_id}")
        
        # Data access event
        data_event = SecurityAuditLogCreate(
            event_type=SecurityEventType.DATA_ACCESS,
            action="read",
            outcome="success",
            severity=SeverityLevel.MEDIUM,
            user_id="bob.analyst",
            resource="customer_database",
            details={
                "query_type": "SELECT",
                "record_count": 1500,
                "classification": "confidential"
            }
        )
        
        print(f"\n✓ Data Access Event Model:")
        print(f"  Type: {data_event.event_type.value}")
        print(f"  Resource: {data_event.resource}")
        print(f"  Classification: {data_event.details['classification']}")
        
        # Threat detection event
        threat_event = SecurityAuditLogCreate(
            event_type=SecurityEventType.THREAT_DETECTED,
            action="malware_detection",
            outcome="blocked",
            severity=SeverityLevel.CRITICAL,
            details={
                "threat_type": "malware",
                "detection_method": "signature_based",
                "confidence_score": 0.95
            }
        )
        
        print(f"\n✓ Threat Detection Event Model:")
        print(f"  Type: {threat_event.event_type.value}")
        print(f"  Severity: {threat_event.severity.value}")
        print(f"  Confidence: {threat_event.details['confidence_score']}")
        
        self.demo_events.extend([auth_event, data_event, threat_event])
        self.demo_metrics["total_events"] += 3
        self.demo_metrics["critical_events"] += 1
        self.demo_metrics["high_severity_events"] += 1
    
    def demo_siem_integration_models(self):
        """Demonstrate SIEM integration models"""
        print("\n🔗 2. SIEM Integration Models")
        print("-" * 40)
        
        # Splunk integration
        splunk_integration = SIEMIntegrationCreate(
            name="Production Splunk",
            platform=SIEMPlatform.SPLUNK,
            endpoint_url="https://splunk.company.com:8088",
            api_key="demo-splunk-key-123"
        )
        
        print(f"✓ Splunk Integration Model:")
        print(f"  Name: {splunk_integration.name}")
        print(f"  Platform: {splunk_integration.platform.value}")
        print(f"  Endpoint: {splunk_integration.endpoint_url}")
        
        # ELK Stack integration
        elk_integration = SIEMIntegrationCreate(
            name="Security ELK Stack",
            platform=SIEMPlatform.ELK_STACK,
            endpoint_url="https://elasticsearch.security.com:9200",
            username="elastic_user",
            password="secure_password"
        )
        
        print(f"\n✓ ELK Stack Integration Model:")
        print(f"  Name: {elk_integration.name}")
        print(f"  Platform: {elk_integration.platform.value}")
        print(f"  Authentication: Username/Password")
        
        # QRadar integration
        qradar_integration = SIEMIntegrationCreate(
            name="IBM QRadar SIEM",
            platform=SIEMPlatform.QRADAR,
            endpoint_url="https://qradar.enterprise.com/api",
            api_key="qradar-api-token-456"
        )
        
        print(f"\n✓ QRadar Integration Model:")
        print(f"  Name: {qradar_integration.name}")
        print(f"  Platform: {qradar_integration.platform.value}")
        print(f"  Authentication: API Token")
    
    def demo_threat_detection_patterns(self):
        """Demonstrate threat detection patterns"""
        print("\n🚨 3. Threat Detection Patterns")
        print("-" * 40)
        
        # Brute force detection rule
        brute_force_rule = ThreatDetectionRuleCreate(
            name="Brute Force Login Detection",
            description="Detect multiple failed login attempts from same IP",
            rule_type="frequency",
            pattern="failed_login_attempts >= 5 within 300 seconds",
            severity=SeverityLevel.HIGH,
            threshold=5,
            time_window=300,
            actions=[
                {"action": "block_ip", "duration": 3600},
                {"action": "alert_security_team"},
                {"action": "log_incident"}
            ]
        )
        
        print(f"✓ Brute Force Detection Rule:")
        print(f"  Name: {brute_force_rule.name}")
        print(f"  Type: {brute_force_rule.rule_type}")
        print(f"  Threshold: {brute_force_rule.threshold} attempts in {brute_force_rule.time_window}s")
        print(f"  Actions: {len(brute_force_rule.actions)} automated responses")
        
        # Privilege escalation rule
        privilege_escalation_rule = ThreatDetectionRuleCreate(
            name="Privilege Escalation Detection",
            description="Detect unauthorized privilege escalation attempts",
            rule_type="behavioral",
            pattern="access_level_increase without authorization",
            severity=SeverityLevel.CRITICAL,
            threshold=1,
            actions=[
                {"action": "disable_user_account"},
                {"action": "escalate_to_soc"},
                {"action": "forensic_capture"}
            ]
        )
        
        print(f"\n✓ Privilege Escalation Rule:")
        print(f"  Name: {privilege_escalation_rule.name}")
        print(f"  Severity: {privilege_escalation_rule.severity.value}")
        print(f"  Response: Immediate account lockdown")
        
        # Data exfiltration rule
        data_exfiltration_rule = ThreatDetectionRuleCreate(
            name="Data Exfiltration Detection",
            description="Detect unusual data access patterns indicating exfiltration",
            rule_type="anomaly",
            pattern="data_volume > baseline * 3 AND after_hours = true",
            severity=SeverityLevel.HIGH,
            threshold=1,
            actions=[
                {"action": "quarantine_session"},
                {"action": "alert_data_protection_officer"},
                {"action": "audit_data_access"}
            ]
        )
        
        print(f"\n✓ Data Exfiltration Rule:")
        print(f"  Name: {data_exfiltration_rule.name}")
        print(f"  Detection: Anomaly-based analysis")
        print(f"  Focus: Volume and timing patterns")
    
    def demo_compliance_frameworks(self):
        """Demonstrate compliance framework support"""
        print("\n📋 4. Compliance Framework Support")
        print("-" * 40)
        
        frameworks = [
            (ComplianceFramework.SOX, "Sarbanes-Oxley Act", "Financial reporting controls"),
            (ComplianceFramework.GDPR, "General Data Protection Regulation", "EU privacy protection"),
            (ComplianceFramework.HIPAA, "Health Insurance Portability Act", "Healthcare data protection"),
            (ComplianceFramework.PCI_DSS, "Payment Card Industry DSS", "Payment data security"),
            (ComplianceFramework.ISO_27001, "ISO 27001", "Information security management"),
            (ComplianceFramework.NIST, "NIST Cybersecurity Framework", "US cybersecurity standards")
        ]
        
        for framework, full_name, description in frameworks:
            print(f"✓ {framework.value.upper()}: {full_name}")
            print(f"  Focus: {description}")
            
            # Sample compliance requirements
            if framework == ComplianceFramework.SOX:
                print(f"  Key Requirements: Access controls, audit trails, segregation of duties")
            elif framework == ComplianceFramework.GDPR:
                print(f"  Key Requirements: Data protection, breach notification, consent management")
            elif framework == ComplianceFramework.HIPAA:
                print(f"  Key Requirements: PHI protection, access logging, risk assessments")
            
            print()
    
    async def demo_security_event_processing(self):
        """Demonstrate security event processing workflow"""
        print("\n⚡ 5. Security Event Processing Workflow")
        print("-" * 40)
        
        # Simulate processing security events
        print("🔍 Processing security events...")
        
        for i, event in enumerate(self.demo_events, 1):
            print(f"\n📝 Event {i}: {event.event_type.value}")
            print(f"   Action: {event.action} -> {event.outcome}")
            print(f"   Severity: {event.severity.value}")
            
            # Simulate risk scoring
            risk_score = self._calculate_risk_score(event)
            print(f"   Risk Score: {risk_score}/10")
            
            # Simulate threat analysis
            if risk_score >= 7:
                threat_alert = {
                    "alert_id": f"ALERT-{i:04d}",
                    "event_id": f"EVENT-{i:04d}",
                    "threat_type": "high_risk_activity",
                    "confidence": 0.85,
                    "recommended_actions": [
                        "Investigate user activity",
                        "Review access patterns",
                        "Consider account restrictions"
                    ]
                }
                self.demo_alerts.append(threat_alert)
                self.demo_metrics["threat_detections"] += 1
                print(f"   🚨 THREAT ALERT: {threat_alert['alert_id']}")
                print(f"   Confidence: {threat_alert['confidence']:.2f}")
            
            # Simulate correlation
            correlation_id = f"CORR-{datetime.now().strftime('%Y%m%d')}-{i}"
            print(f"   Correlation ID: {correlation_id}")
            
            await asyncio.sleep(0.1)  # Simulate processing time
        
        print(f"\n📊 Processing Summary:")
        print(f"   Events Processed: {len(self.demo_events)}")
        print(f"   Alerts Generated: {len(self.demo_alerts)}")
        print(f"   Average Risk Score: {sum(self._calculate_risk_score(e) for e in self.demo_events) / len(self.demo_events):.1f}")
    
    def demo_siem_event_formatting(self):
        """Demonstrate SIEM event formatting"""
        print("\n📤 6. SIEM Event Formatting")
        print("-" * 40)
        
        sample_event = self.demo_events[0]  # Use first event
        
        # Splunk HEC format
        splunk_format = {
            "time": int(datetime.now().timestamp()),
            "event": {
                "event_id": "EVENT-0001",
                "event_type": sample_event.event_type.value,
                "severity": sample_event.severity.value,
                "source_ip": sample_event.source_ip,
                "user": sample_event.user_id,
                "action": sample_event.action,
                "outcome": sample_event.outcome,
                "details": sample_event.details
            },
            "source": "scrollintel",
            "sourcetype": "security_audit",
            "index": "security"
        }
        
        print("✓ Splunk HEC Format:")
        print(json.dumps(splunk_format, indent=2))
        
        # ELK format
        elk_format = {
            "@timestamp": datetime.now().isoformat(),
            "event_id": "EVENT-0001",
            "event_type": sample_event.event_type.value,
            "severity": sample_event.severity.value,
            "source": {
                "ip": sample_event.source_ip,
                "user": sample_event.user_id
            },
            "action": sample_event.action,
            "outcome": sample_event.outcome,
            "scrollintel": {
                "version": "1.0",
                "component": "security_audit"
            },
            "details": sample_event.details
        }
        
        print(f"\n✓ ELK Stack Format:")
        print(json.dumps(elk_format, indent=2))
    
    def demo_compliance_reporting_structure(self):
        """Demonstrate compliance reporting structure"""
        print("\n📋 7. Compliance Reporting Structure")
        print("-" * 40)
        
        # Sample SOX compliance report structure
        sox_report_structure = {
            "report_id": "SOX-2024-Q4",
            "framework": "sox",
            "period": {
                "start": "2024-10-01",
                "end": "2024-12-31"
            },
            "overall_score": 92.5,
            "requirements_assessed": [
                {
                    "requirement_id": "SOX-302",
                    "title": "Management Assessment of Internal Controls",
                    "status": "compliant",
                    "score": 95,
                    "findings": [
                        "Access controls properly implemented",
                        "Audit trails complete and accurate",
                        "Segregation of duties maintained"
                    ],
                    "evidence": [
                        "Access control logs reviewed",
                        "Change management records validated",
                        "User access reviews completed"
                    ]
                },
                {
                    "requirement_id": "SOX-404",
                    "title": "Internal Control Assessment",
                    "status": "partially_compliant",
                    "score": 85,
                    "findings": [
                        "Control documentation complete",
                        "Minor gaps in testing procedures"
                    ],
                    "recommendations": [
                        "Enhance testing documentation",
                        "Implement automated control testing"
                    ]
                }
            ],
            "violations": [
                {
                    "violation_id": "SOX-V001",
                    "requirement": "SOX-404",
                    "severity": "medium",
                    "description": "Incomplete testing documentation for Q3",
                    "remediation_status": "in_progress"
                }
            ]
        }
        
        print("✓ SOX Compliance Report Structure:")
        print(f"  Report ID: {sox_report_structure['report_id']}")
        print(f"  Overall Score: {sox_report_structure['overall_score']}%")
        print(f"  Requirements: {len(sox_report_structure['requirements_assessed'])}")
        print(f"  Violations: {len(sox_report_structure['violations'])}")
        
        # Sample GDPR compliance metrics
        gdpr_metrics = {
            "framework": "gdpr",
            "data_protection_score": 88.0,
            "breach_response_time": "45 minutes (target: <72 hours)",
            "consent_management": "compliant",
            "data_subject_requests": {
                "total": 156,
                "completed_on_time": 152,
                "compliance_rate": "97.4%"
            },
            "privacy_impact_assessments": {
                "required": 12,
                "completed": 12,
                "compliance_rate": "100%"
            }
        }
        
        print(f"\n✓ GDPR Compliance Metrics:")
        print(f"  Data Protection Score: {gdpr_metrics['data_protection_score']}%")
        print(f"  Breach Response: {gdpr_metrics['breach_response_time']}")
        print(f"  Subject Requests: {gdpr_metrics['data_subject_requests']['compliance_rate']}")
    
    def demo_security_metrics(self):
        """Demonstrate security metrics and KPIs"""
        print("\n📊 8. Security Metrics and KPIs")
        print("-" * 40)
        
        # Current metrics
        print("🔍 Current Security Metrics:")
        print(f"   Total Events: {self.demo_metrics['total_events']}")
        print(f"   Critical Events: {self.demo_metrics['critical_events']}")
        print(f"   High Severity Events: {self.demo_metrics['high_severity_events']}")
        print(f"   Threat Detections: {self.demo_metrics['threat_detections']}")
        
        # Calculate additional metrics
        if self.demo_metrics['total_events'] > 0:
            critical_rate = (self.demo_metrics['critical_events'] / self.demo_metrics['total_events']) * 100
            detection_rate = (self.demo_metrics['threat_detections'] / self.demo_metrics['total_events']) * 100
            
            print(f"   Critical Event Rate: {critical_rate:.1f}%")
            print(f"   Threat Detection Rate: {detection_rate:.1f}%")
        
        # Security trends (simulated)
        print(f"\n📈 Security Trends (Last 30 Days):")
        print(f"   Authentication Events: 15,420 (+5.2%)")
        print(f"   Data Access Events: 8,750 (+2.1%)")
        print(f"   Configuration Changes: 342 (-1.8%)")
        print(f"   Threat Detections: 23 (-12.5%)")
        
        # Risk assessment
        print(f"\n⚠️ Risk Assessment:")
        print(f"   Overall Risk Level: MEDIUM")
        print(f"   Top Risk Factors:")
        print(f"     • Failed login attempts: 156 (last 24h)")
        print(f"     • After-hours access: 23 events")
        print(f"     • Privileged account usage: 45 events")
        
        # Compliance status
        print(f"\n✅ Compliance Status:")
        print(f"   SOX: 92.5% (Compliant)")
        print(f"   GDPR: 88.0% (Compliant)")
        print(f"   HIPAA: 94.2% (Compliant)")
        print(f"   PCI DSS: 89.7% (Compliant)")
        
        # Incident response metrics
        print(f"\n🚨 Incident Response Metrics:")
        print(f"   Mean Time to Detection: 4.2 minutes")
        print(f"   Mean Time to Response: 12.8 minutes")
        print(f"   Mean Time to Resolution: 2.3 hours")
        print(f"   False Positive Rate: 3.2%")
    
    def _calculate_risk_score(self, event: SecurityAuditLogCreate) -> int:
        """Calculate risk score for an event"""
        base_scores = {
            SeverityLevel.LOW: 2,
            SeverityLevel.MEDIUM: 5,
            SeverityLevel.HIGH: 8,
            SeverityLevel.CRITICAL: 10
        }
        
        score = base_scores.get(event.severity, 1)
        
        # Adjust based on event type
        if event.event_type == SecurityEventType.THREAT_DETECTED:
            score = min(score + 2, 10)
        elif event.event_type == SecurityEventType.SYSTEM_BREACH:
            score = 10
        elif event.event_type == SecurityEventType.DATA_ACCESS and event.details:
            if event.details.get("classification") in ["confidential", "restricted"]:
                score = min(score + 1, 10)
        
        return score

async def main():
    """Run the simple security audit demo"""
    demo = SecurityAuditSimpleDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())