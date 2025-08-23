"""
Simple Tests for Security Audit System
Tests the security audit models and basic functionality
"""
import pytest
from datetime import datetime

from scrollintel.models.security_audit_models import (
    SecurityEventType, SeverityLevel, SIEMPlatform, ComplianceFramework,
    SecurityAuditLogCreate, SIEMIntegrationCreate, ThreatDetectionRuleCreate
)

class TestSecurityAuditModels:
    """Test security audit models"""
    
    def test_security_event_creation(self):
        """Test creating security audit log entries"""
        event = SecurityAuditLogCreate(
            event_type=SecurityEventType.AUTHENTICATION,
            action="login",
            outcome="success",
            severity=SeverityLevel.LOW,
            user_id="test_user"
        )
        
        assert event.event_type == SecurityEventType.AUTHENTICATION
        assert event.action == "login"
        assert event.outcome == "success"
        assert event.user_id == "test_user"
        assert event.severity == SeverityLevel.LOW
    
    def test_siem_integration_creation(self):
        """Test creating SIEM integration configurations"""
        integration = SIEMIntegrationCreate(
            name="Test Splunk",
            platform=SIEMPlatform.SPLUNK,
            endpoint_url="https://test.splunk.com"
        )
        
        assert integration.name == "Test Splunk"
        assert integration.platform == SIEMPlatform.SPLUNK
        assert integration.endpoint_url == "https://test.splunk.com"
    
    def test_threat_detection_rule_creation(self):
        """Test creating threat detection rules"""
        rule = ThreatDetectionRuleCreate(
            name="Test Rule",
            rule_type="frequency",
            pattern="test_pattern",
            severity=SeverityLevel.HIGH
        )
        
        assert rule.name == "Test Rule"
        assert rule.rule_type == "frequency"
        assert rule.severity == SeverityLevel.HIGH
        assert rule.threshold == 1  # Default
    
    def test_security_event_types(self):
        """Test all security event types are available"""
        expected_types = [
            "authentication", "authorization", "data_access",
            "configuration_change", "system_breach", "suspicious_activity",
            "compliance_violation", "threat_detected"
        ]
        
        for event_type in expected_types:
            assert hasattr(SecurityEventType, event_type.upper())
    
    def test_severity_levels(self):
        """Test all severity levels are available"""
        expected_levels = ["low", "medium", "high", "critical"]
        
        for level in expected_levels:
            assert hasattr(SeverityLevel, level.upper())
    
    def test_siem_platforms(self):
        """Test all SIEM platforms are supported"""
        expected_platforms = [
            "splunk", "elk_stack", "qradar", 
            "azure_sentinel", "arcsight", "sumo_logic"
        ]
        
        platform_mapping = {
            "splunk": "SPLUNK",
            "elk_stack": "ELK_STACK", 
            "qradar": "QRADAR",
            "azure_sentinel": "SENTINEL",
            "arcsight": "ARCSIGHT",
            "sumo_logic": "SUMO_LOGIC"
        }
        
        for platform in expected_platforms:
            platform_name = platform_mapping[platform]
            assert hasattr(SIEMPlatform, platform_name)
    
    def test_compliance_frameworks(self):
        """Test all compliance frameworks are supported"""
        expected_frameworks = [
            "sox", "gdpr", "hipaa", "pci_dss", 
            "iso_27001", "nist", "ccpa", "soc2"
        ]
        
        for framework in expected_frameworks:
            framework_name = framework.upper().replace("_", "_")
            assert hasattr(ComplianceFramework, framework_name)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])