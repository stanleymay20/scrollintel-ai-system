"""
Security and Compliance Framework Demonstration
Showcases enterprise-grade security controls and compliance reporting
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_demo_security_config():
    """Create demonstration security configuration"""
    from scrollintel.models.security_compliance_models import SecurityConfig, EncryptionConfig
    
    return SecurityConfig(
        password_policy={
            "min_length": 12,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_symbols": True,
            "max_age_days": 90
        },
        session_timeout_minutes=480,  # 8 hours
        max_failed_attempts=5,
        account_lockout_minutes=30,
        mfa_required=True,
        sso_enabled=True,
        audit_retention_days=2555,  # 7 years
        encryption=EncryptionConfig(
            algorithm="AES-256-GCM",
            key_rotation_days=90,
            backup_encryption=True,
            transit_encryption=True,
            at_rest_encryption=True
        )
    )

def demonstrate_password_security():
    """Demonstrate password hashing and verification"""
    print("\n" + "="*60)
    print("PASSWORD SECURITY DEMONSTRATION")
    print("="*60)
    
    from scrollintel.security.enterprise_security_framework import EnterpriseSecurityFramework
    
    config = create_demo_security_config()
    security = EnterpriseSecurityFramework(config)
    
    # Test passwords
    passwords = [
        "WeakPassword123",
        "StrongP@ssw0rd!2024",
        "UltraSecure#P@ssw0rd$2024!"
    ]
    
    for password in passwords:
        print(f"\nTesting password: {password}")
        
        # Hash password
        hashed = security.hash_password(password)
        print(f"Hashed: {hashed[:50]}...")
        
        # Verify correct password
        is_valid = security.verify_password(password, hashed)
        print(f"Verification (correct): {is_valid}")
        
        # Verify wrong password
        is_invalid = security.verify_password("wrong_password", hashed)
        print(f"Verification (wrong): {is_invalid}")

def demonstrate_mfa_setup():
    """Demonstrate multi-factor authentication setup"""
    print("\n" + "="*60)
    print("MULTI-FACTOR AUTHENTICATION DEMONSTRATION")
    print("="*60)
    
    from scrollintel.security.enterprise_security_framework import EnterpriseSecurityFramework
    
    config = create_demo_security_config()
    security = EnterpriseSecurityFramework(config)
    
    # Generate MFA secret
    mfa_secret = security.generate_mfa_secret()
    print(f"MFA Secret: {mfa_secret}")
    
    # Generate QR code URL
    qr_url = security.generate_mfa_qr_url("demo@scrollintel.com", mfa_secret, "ScrollIntel Demo")
    print(f"QR Code URL: {qr_url}")
    
    # Generate backup codes
    backup_codes = security.generate_backup_codes(10)
    print(f"Backup Codes: {backup_codes}")
    
    # Simulate MFA verification (would normally use actual TOTP)
    print("\nMFA Verification Simulation:")
    print("In production, this would verify against actual TOTP codes")

def demonstrate_data_encryption():
    """Demonstrate data encryption and decryption"""
    print("\n" + "="*60)
    print("DATA ENCRYPTION DEMONSTRATION")
    print("="*60)
    
    from scrollintel.security.enterprise_security_framework import EnterpriseSecurityFramework
    
    config = create_demo_security_config()
    security = EnterpriseSecurityFramework(config)
    
    # Test data
    sensitive_data = [
        "Personal information: John Doe, SSN: 123-45-6789",
        "Financial data: Account 9876543210, Balance: $50,000",
        "Health record: Patient ID 12345, Diagnosis: Confidential"
    ]
    
    for data in sensitive_data:
        print(f"\nOriginal: {data}")
        
        # Encrypt data
        encrypted = security.encrypt_data(data)
        print(f"Encrypted: {encrypted[:50]}...")
        
        # Decrypt data
        decrypted = security.decrypt_data(encrypted)
        print(f"Decrypted: {decrypted.decode('utf-8')}")
        
        # Verify integrity
        integrity_check = decrypted.decode('utf-8') == data
        print(f"Integrity Check: {integrity_check}")

def demonstrate_audit_logging():
    """Demonstrate comprehensive audit logging"""
    print("\n" + "="*60)
    print("AUDIT LOGGING DEMONSTRATION")
    print("="*60)
    
    from scrollintel.security.enterprise_security_framework import EnterpriseSecurityFramework
    from scrollintel.models.security_compliance_models import AuditEventType, ComplianceFramework
    
    config = create_demo_security_config()
    security = EnterpriseSecurityFramework(config)
    
    # Mock database session
    class MockDB:
        def __init__(self):
            self.audit_logs = []
        
        def add(self, log):
            self.audit_logs.append(log)
        
        def commit(self):
            pass
    
    mock_db = MockDB()
    
    # Simulate various audit events
    audit_events = [
        {
            "event_type": AuditEventType.AUTHENTICATION,
            "category": "login_success",
            "description": "User successfully authenticated",
            "user_id": "user123",
            "success": True
        },
        {
            "event_type": AuditEventType.AUTHORIZATION,
            "category": "permission_denied",
            "description": "User attempted to access restricted resource",
            "user_id": "user456",
            "success": False,
            "error_code": "INSUFFICIENT_PERMISSIONS"
        },
        {
            "event_type": AuditEventType.DATA_ACCESS,
            "category": "sensitive_data_access",
            "description": "User accessed personal data",
            "user_id": "user789",
            "success": True,
            "sensitive_data_accessed": True,
            "compliance_frameworks": [ComplianceFramework.GDPR]
        },
        {
            "event_type": AuditEventType.SECURITY_INCIDENT,
            "category": "brute_force_attempt",
            "description": "Multiple failed login attempts detected",
            "success": False,
            "ip_address": "192.168.1.100"
        }
    ]
    
    print("Logging audit events:")
    for i, event in enumerate(audit_events, 1):
        print(f"\n{i}. {event['description']}")
        
        security._log_audit_event(
            db=mock_db,
            user_id=event.get("user_id"),
            event_type=event["event_type"],
            event_category=event["category"],
            event_description=event["description"],
            action="demo_action",
            success=event.get("success", True),
            error_code=event.get("error_code"),
            ip_address=event.get("ip_address", "127.0.0.1"),
            sensitive_data_accessed=event.get("sensitive_data_accessed", False),
            compliance_frameworks=event.get("compliance_frameworks")
        )
    
    print(f"\nTotal audit events logged: {len(mock_db.audit_logs)}")

def demonstrate_compliance_checking():
    """Demonstrate automated compliance checking"""
    print("\n" + "="*60)
    print("COMPLIANCE CHECKING DEMONSTRATION")
    print("="*60)
    
    from scrollintel.security.enterprise_security_framework import EnterpriseSecurityFramework
    from scrollintel.core.audit_compliance_system import ComprehensiveAuditSystem
    from scrollintel.models.security_compliance_models import ComplianceFramework
    
    config = create_demo_security_config()
    security = EnterpriseSecurityFramework(config)
    audit_system = ComprehensiveAuditSystem(security)
    
    # Display compliance rules
    print("Available Compliance Rules:")
    
    frameworks = [ComplianceFramework.GDPR, ComplianceFramework.SOX, ComplianceFramework.HIPAA, ComplianceFramework.ISO_27001]
    
    for framework in frameworks:
        print(f"\n{framework.value.upper()} Rules:")
        framework_rules = [rule for rule in audit_system.compliance_rules.values() 
                          if rule.framework == framework]
        
        for rule in framework_rules:
            print(f"  - {rule.id}: {rule.title}")
            print(f"    Risk Level: {rule.risk_level.value}")
            print(f"    Automated: {rule.automated_check}")

async def demonstrate_compliance_reporting():
    """Demonstrate compliance report generation"""
    print("\n" + "="*60)
    print("COMPLIANCE REPORTING DEMONSTRATION")
    print("="*60)
    
    from scrollintel.security.enterprise_security_framework import EnterpriseSecurityFramework
    from scrollintel.core.audit_compliance_system import ComprehensiveAuditSystem
    from scrollintel.models.security_compliance_models import ComplianceFramework, AuditLog, AuditEventType
    
    config = create_demo_security_config()
    security = EnterpriseSecurityFramework(config)
    audit_system = ComprehensiveAuditSystem(security)
    
    # Mock database with sample data
    class MockDB:
        def __init__(self):
            self.audit_logs = self._create_sample_audit_logs()
        
        def query(self, model):
            return MockQuery(self.audit_logs)
        
        def _create_sample_audit_logs(self):
            logs = []
            base_time = datetime.utcnow() - timedelta(days=15)
            
            # Create sample audit events
            for i in range(100):
                log = type('MockAuditLog', (), {
                    'id': f'log_{i}',
                    'timestamp': base_time + timedelta(hours=i),
                    'event_type': AuditEventType.DATA_ACCESS.value if i % 3 == 0 else AuditEventType.AUTHENTICATION.value,
                    'success': i % 10 != 0,  # 10% failure rate
                    'sensitive_data_accessed': i % 5 == 0,  # 20% sensitive data access
                    'user_id': f'user_{i % 10}',
                    'resource_type': 'personal_data' if i % 5 == 0 else 'system_data',
                    'metadata': {'severity': 'high' if i % 20 == 0 else 'medium'}
                })()
                logs.append(log)
            
            return logs
    
    class MockQuery:
        def __init__(self, data):
            self.data = data
        
        def filter(self, *args):
            return self
        
        def all(self):
            return self.data
        
        def count(self):
            return len(self.data)
    
    mock_db = MockDB()
    
    # Generate compliance report for GDPR
    print("Generating GDPR Compliance Report...")
    
    report = await audit_system.generate_comprehensive_compliance_report(
        db=mock_db,
        framework=ComplianceFramework.GDPR,
        start_date=datetime.utcnow() - timedelta(days=30),
        end_date=datetime.utcnow()
    )
    
    # Display report summary
    print(f"\nCompliance Report Summary:")
    print(f"Framework: {report['framework']}")
    print(f"Overall Score: {report['compliance_metrics']['overall_score']:.1f}%")
    print(f"Total Events: {report['compliance_metrics']['total_events']:,}")
    print(f"Violations: {report['compliance_metrics']['violation_events']}")
    
    print(f"\nRisk Assessment:")
    for severity, count in report['risk_assessment']['severity_breakdown'].items():
        print(f"  {severity.title()}: {count}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"  {i}. {rec['recommendation']} (Priority: {rec['priority']})")
    
    print(f"\nExecutive Summary:")
    print(report['executive_summary'][:300] + "...")

def demonstrate_security_metrics():
    """Demonstrate security metrics collection"""
    print("\n" + "="*60)
    print("SECURITY METRICS DEMONSTRATION")
    print("="*60)
    
    from scrollintel.models.security_compliance_models import SecurityMetrics
    
    # Simulate security metrics
    metrics = SecurityMetrics(
        active_users=1250,
        active_sessions=890,
        failed_login_attempts=23,
        security_incidents=2,
        compliance_score=94.5,
        encryption_coverage=100.0,
        audit_events_today=15420
    )
    
    print("Current Security Metrics:")
    print(f"  Active Users: {metrics.active_users:,}")
    print(f"  Active Sessions: {metrics.active_sessions:,}")
    print(f"  Failed Login Attempts (Today): {metrics.failed_login_attempts}")
    print(f"  Security Incidents (Today): {metrics.security_incidents}")
    print(f"  Compliance Score: {metrics.compliance_score}%")
    print(f"  Encryption Coverage: {metrics.encryption_coverage}%")
    print(f"  Audit Events (Today): {metrics.audit_events_today:,}")
    print(f"  Timestamp: {metrics.timestamp}")
    
    # Security status assessment
    if metrics.compliance_score >= 95:
        status = "EXCELLENT"
    elif metrics.compliance_score >= 85:
        status = "GOOD"
    elif metrics.compliance_score >= 75:
        status = "ACCEPTABLE"
    else:
        status = "NEEDS IMPROVEMENT"
    
    print(f"\nOverall Security Status: {status}")

def demonstrate_key_management():
    """Demonstrate encryption key management"""
    print("\n" + "="*60)
    print("ENCRYPTION KEY MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    from scrollintel.security.enterprise_security_framework import EnterpriseSecurityFramework
    
    config = create_demo_security_config()
    security = EnterpriseSecurityFramework(config)
    
    print("Encryption Key Purposes:")
    for purpose in security.encryption_keys.keys():
        print(f"  - {purpose}")
    
    print(f"\nKey Rotation Schedule: Every {config.encryption.key_rotation_days} days")
    print(f"Encryption Algorithm: {config.encryption.algorithm}")
    print(f"Backup Encryption: {config.encryption.backup_encryption}")
    print(f"Transit Encryption: {config.encryption.transit_encryption}")
    print(f"At-Rest Encryption: {config.encryption.at_rest_encryption}")
    
    # Simulate key rotation
    print("\nSimulating Key Rotation...")
    
    class MockDB:
        def add(self, item):
            print(f"  Added encryption key record: {item}")
        
        def commit(self):
            print("  Committed key rotation to database")
    
    mock_db = MockDB()
    
    # Store original key count
    original_key_count = len(security.encryption_keys)
    
    # Rotate keys
    security.rotate_encryption_keys(mock_db)
    
    print(f"  Rotated {original_key_count} encryption keys")
    print("  Key rotation completed successfully")

def demonstrate_security_validation():
    """Demonstrate security configuration validation"""
    print("\n" + "="*60)
    print("SECURITY CONFIGURATION VALIDATION")
    print("="*60)
    
    from scrollintel.security.enterprise_security_framework import EnterpriseSecurityFramework
    
    # Test with secure configuration
    secure_config = create_demo_security_config()
    security = EnterpriseSecurityFramework(secure_config)
    
    print("Validating Secure Configuration:")
    issues = security.validate_security_config()
    
    if not issues:
        print("  ✓ Configuration is secure - no issues found")
    else:
        print("  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    
    # Test with insecure configuration
    print("\nValidating Insecure Configuration:")
    insecure_config = create_demo_security_config()
    insecure_config.password_policy['min_length'] = 6  # Too short
    insecure_config.session_timeout_minutes = 1440  # Too long (24 hours)
    insecure_config.mfa_required = False  # Insecure
    insecure_config.audit_retention_days = 30  # Too short
    
    insecure_security = EnterpriseSecurityFramework(insecure_config)
    issues = insecure_security.validate_security_config()
    
    if issues:
        print("  Issues found:")
        for issue in issues:
            print(f"    ✗ {issue}")
    else:
        print("  ✓ No issues found")

async def main():
    """Run all security and compliance demonstrations"""
    print("SCROLLINTEL SECURITY & COMPLIANCE FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    print("Showcasing enterprise-grade security controls and compliance reporting")
    print("=" * 80)
    
    try:
        # Run demonstrations
        demonstrate_password_security()
        demonstrate_mfa_setup()
        demonstrate_data_encryption()
        demonstrate_audit_logging()
        demonstrate_compliance_checking()
        await demonstrate_compliance_reporting()
        demonstrate_security_metrics()
        demonstrate_key_management()
        demonstrate_security_validation()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print("The Security & Compliance Framework provides:")
        print("✓ Military-grade encryption and data protection")
        print("✓ Multi-factor authentication and SSO integration")
        print("✓ Comprehensive audit logging and compliance reporting")
        print("✓ Real-time security monitoring and incident detection")
        print("✓ Automated compliance checking for GDPR, SOX, HIPAA, ISO 27001")
        print("✓ Role-based access control with fine-grained permissions")
        print("✓ Enterprise-grade key management and rotation")
        print("✓ Zero-tolerance security validation and monitoring")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\nError during demonstration: {e}")

if __name__ == "__main__":
    asyncio.run(main())