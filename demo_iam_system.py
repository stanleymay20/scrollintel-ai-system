"""
Demo script for Identity and Access Management System
Demonstrates all IAM capabilities including MFA, JIT access, RBAC, UEBA, and session management
"""

import asyncio
import time
from datetime import datetime, timedelta
from security.iam.iam_integration import IAMSystem
from security.iam.mfa_system import MFAMethod
from security.iam.session_manager import SessionType
from security.iam.ueba_system import UserActivity

async def demo_iam_system():
    """Comprehensive demo of IAM system capabilities"""
    
    print("🔐 ScrollIntel Enterprise Security - IAM System Demo")
    print("=" * 60)
    
    # Initialize IAM system
    config = {
        "mfa": {},
        "jit": {},
        "rbac": {},
        "ueba": {"learning_window_days": 30},
        "session": {
            "max_concurrent_sessions": 5,
            "session_timeout_minutes": 30,
            "absolute_timeout_hours": 8
        }
    }
    
    iam_system = IAMSystem(config)
    
    # Demo 1: User Setup and MFA Configuration
    print("\n1. 👤 User Setup and MFA Configuration")
    print("-" * 40)
    
    user_id = "demo_user"
    phone_number = "+1234567890"
    
    # Setup test user with MFA
    setup_result = iam_system.setup_test_user(user_id, phone_number)
    print(f"✅ User setup completed: {user_id}")
    print(f"📱 TOTP Secret: {setup_result['totp_secret'][:10]}...")
    print(f"🔑 Backup codes generated: {len(setup_result['backup_codes'])} codes")
    
    # Demo 2: Basic Authentication
    print("\n2. 🔑 Authentication Flow")
    print("-" * 40)
    
    # First authentication (will require MFA)
    auth_result = await iam_system.authenticate_user(
        username=user_id,
        password="test_password",
        ip_address="192.168.1.100",
        user_agent="Demo Browser 1.0",
        device_fingerprint="demo_device_001",
        session_type=SessionType.WEB
    )
    
    print(f"🔐 Authentication result: {auth_result.success}")
    print(f"🛡️ Requires MFA: {auth_result.requires_mfa}")
    print(f"📊 Risk score: {auth_result.risk_score:.2f}")
    
    if auth_result.requires_mfa:
        print(f"🔢 Available MFA methods: {[m.value for m in auth_result.available_mfa_methods]}")
        
        # Complete MFA with TOTP
        import pyotp
        totp = pyotp.TOTP(setup_result['totp_secret'])
        token = totp.now()
        
        mfa_result = await iam_system.complete_mfa_authentication(
            user_id=user_id,
            mfa_method=MFAMethod.TOTP,
            mfa_token=token,
            ip_address="192.168.1.100",
            session_type=SessionType.WEB
        )
        
        print(f"✅ MFA completed: {mfa_result.success}")
        session_id = mfa_result.session_id
        session_token = mfa_result.session_token
    else:
        session_id = auth_result.session_id
        session_token = auth_result.session_token
    
    print(f"🎫 Session created: {session_id[:16]}...")
    
    # Demo 3: Authorization and RBAC
    print("\n3. 🛡️ Authorization and RBAC")
    print("-" * 40)
    
    # Test basic read access (should be allowed)
    auth_result = await iam_system.authorize_access(
        session_id=session_id,
        session_token=session_token,
        resource_id="documents/report.pdf",
        resource_type="document",
        action="read",
        ip_address="192.168.1.100"
    )
    
    print(f"📖 Read access to document: {'✅ Allowed' if auth_result.allowed else '❌ Denied'}")
    print(f"📝 Reason: {auth_result.reason}")
    
    # Test admin access (should be denied)
    admin_auth_result = await iam_system.authorize_access(
        session_id=session_id,
        session_token=session_token,
        resource_id="admin/user_management",
        resource_type="admin",
        action="admin",
        ip_address="192.168.1.100"
    )
    
    print(f"⚙️ Admin access: {'✅ Allowed' if admin_auth_result.allowed else '❌ Denied'}")
    print(f"📝 Reason: {admin_auth_result.reason}")
    
    # Demo 4: Just-in-Time (JIT) Access
    print("\n4. ⏰ Just-in-Time Access Request")
    print("-" * 40)
    
    # Request temporary admin access
    jit_request_id = await iam_system.request_jit_access(
        session_id=session_id,
        session_token=session_token,
        resource_id="admin/sensitive_data",
        permissions=["read", "write"],
        justification="Need access for security audit",
        duration_hours=4
    )
    
    print(f"📋 JIT access request submitted: {jit_request_id[:16]}...")
    
    # Check request status
    request = iam_system.jit_system.access_requests[jit_request_id]
    print(f"📊 Request status: {request.status.value}")
    print(f"⏱️ Requested duration: {request.requested_duration}")
    
    # Demo auto-approval for low-risk request
    low_risk_request_id = await iam_system.request_jit_access(
        session_id=session_id,
        session_token=session_token,
        resource_id="public/reports",
        permissions=["read"],
        justification="Need to review quarterly reports",
        duration_hours=2
    )
    
    low_risk_request = iam_system.jit_system.access_requests[low_risk_request_id]
    print(f"🚀 Low-risk request status: {low_risk_request.status.value}")
    
    # Demo 5: User and Entity Behavior Analytics (UEBA)
    print("\n5. 🧠 Behavior Analytics and Anomaly Detection")
    print("-" * 40)
    
    # Simulate normal user behavior
    print("📊 Simulating normal user behavior...")
    for hour in range(9, 17):  # Business hours
        for i in range(5):
            activity = UserActivity(
                activity_id=f"normal_{hour}_{i}",
                user_id=user_id,
                timestamp=datetime.utcnow().replace(hour=hour),
                action="read",
                resource_id=f"document_{i}",
                resource_type="document",
                ip_address="192.168.1.100"
            )
            iam_system.ueba_system.record_activity(activity)
    
    # Update user profile
    iam_system.ueba_system._update_user_profile(user_id)
    profile = iam_system.ueba_system.user_profiles[user_id]
    print(f"👤 User profile updated - Typical hours: {profile.typical_hours}")
    print(f"🌐 Typical locations: {profile.typical_locations}")
    
    # Simulate anomalous behavior
    print("🚨 Simulating anomalous behavior...")
    
    # Late night access from different location
    anomalous_activity = UserActivity(
        activity_id="anomaly_1",
        user_id=user_id,
        timestamp=datetime.utcnow().replace(hour=2),  # 2 AM
        action="admin_access",
        resource_id="admin/user_data",
        resource_type="admin",
        ip_address="10.0.0.50"  # Different IP
    )
    iam_system.ueba_system.record_activity(anomalous_activity)
    
    # Check for alerts
    alerts = iam_system.ueba_system.get_active_alerts(user_id)
    print(f"🚨 Security alerts generated: {len(alerts)}")
    
    for alert in alerts:
        print(f"  ⚠️ {alert.anomaly_type.value}: {alert.description}")
        print(f"     Risk: {alert.risk_level.value}, Confidence: {alert.confidence_score:.2f}")
    
    # Demo 6: Session Management
    print("\n6. 🎫 Session Management")
    print("-" * 40)
    
    # Create multiple sessions
    sessions = []
    for i in range(3):
        sid, stoken = iam_system.session_manager.create_session(
            user_id=user_id,
            session_type=SessionType.WEB,
            ip_address=f"192.168.1.{100+i}",
            device_fingerprint=f"device_{i}"
        )
        sessions.append((sid, stoken))
    
    user_sessions = iam_system.session_manager.get_user_sessions(user_id)
    print(f"👥 Active sessions for user: {len(user_sessions)}")
    
    for session in user_sessions[:3]:  # Show first 3
        print(f"  🎫 {session.session_id[:16]}... - {session.session_type.value}")
        print(f"     Created: {session.created_at.strftime('%H:%M:%S')}")
        print(f"     IP: {session.ip_address}")
    
    # Test concurrent session limits
    print(f"🔒 Max concurrent sessions: {iam_system.session_manager.config.max_concurrent_sessions}")
    
    # Demo 7: MFA Methods
    print("\n7. 🔐 Multi-Factor Authentication Methods")
    print("-" * 40)
    
    # SMS Challenge
    sms_challenge_id = iam_system.mfa_system.initiate_sms_challenge(user_id, phone_number)
    print(f"📱 SMS challenge initiated: {sms_challenge_id[:16]}...")
    
    # Get the SMS code (in production, this would be sent to phone)
    challenge = iam_system.mfa_system.active_challenges[sms_challenge_id]
    sms_code = challenge.challenge_data["code"]
    print(f"💬 SMS code (for demo): {sms_code}")
    
    # Verify SMS code
    sms_valid = iam_system.mfa_system.verify_sms_challenge(sms_challenge_id, sms_code)
    print(f"✅ SMS verification: {'Success' if sms_valid else 'Failed'}")
    
    # Test backup codes
    backup_codes = setup_result['backup_codes']
    test_backup_code = backup_codes[0]
    backup_valid = iam_system.mfa_system.verify_backup_code(user_id, test_backup_code)
    print(f"🔑 Backup code verification: {'Success' if backup_valid else 'Failed'}")
    
    # Demo 8: System Status and Monitoring
    print("\n8. 📊 System Status and Monitoring")
    print("-" * 40)
    
    system_status = iam_system.get_system_status()
    print(f"🏥 System health: {system_status['system_health']}")
    print(f"🎫 Active sessions: {system_status['session_statistics']['active_sessions']}")
    print(f"👥 Unique users: {system_status['session_statistics']['unique_users']}")
    print(f"🚨 Active alerts: {system_status['active_security_alerts']}")
    print(f"📋 Pending JIT requests: {system_status['pending_jit_requests']}")
    
    # Demo 9: Cleanup and Maintenance
    print("\n9. 🧹 System Cleanup")
    print("-" * 40)
    
    print("🔄 Running system cleanup...")
    await iam_system.cleanup_expired_data()
    
    # Show final statistics
    final_stats = iam_system.session_manager.get_session_statistics()
    print(f"📈 Final session count: {final_stats['active_sessions']}")
    
    # Demo 10: Security Audit
    print("\n10. 🔍 Security Audit")
    print("-" * 40)
    
    audit_report = iam_system.rbac_system.audit_user_permissions(user_id)
    print(f"👤 User: {audit_report['user_id']}")
    print(f"🎭 Active roles: {audit_report['active_roles']}")
    print(f"🔑 Effective permissions: {audit_report['effective_permissions']}")
    
    print("\n" + "=" * 60)
    print("🎉 IAM System Demo Completed Successfully!")
    print("✅ All enterprise security features demonstrated:")
    print("   • Multi-Factor Authentication (TOTP, SMS, Backup Codes)")
    print("   • Just-in-Time Access Provisioning")
    print("   • Role-Based Access Control (RBAC)")
    print("   • User & Entity Behavior Analytics (UEBA)")
    print("   • Advanced Session Management")
    print("   • Real-time Security Monitoring")
    print("   • Automated Anomaly Detection")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_iam_system())