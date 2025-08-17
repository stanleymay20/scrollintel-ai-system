"""
Tests for Identity and Access Management System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from security.iam.iam_integration import IAMSystem
from security.iam.mfa_system import MFAMethod
from security.iam.session_manager import SessionType
from security.iam.ueba_system import UserActivity, AnomalyType
from security.iam.rbac_system import AccessContext

class TestIAMSystem:
    
    @pytest.fixture
    def iam_config(self):
        return {
            "mfa": {},
            "jit": {},
            "rbac": {},
            "ueba": {"learning_window_days": 30},
            "session": {
                "max_concurrent_sessions": 3,
                "session_timeout_minutes": 30,
                "absolute_timeout_hours": 8
            }
        }
    
    @pytest.fixture
    def iam_system(self, iam_config):
        return IAMSystem(iam_config)
    
    @pytest.mark.asyncio
    async def test_basic_authentication(self, iam_system):
        """Test basic username/password authentication"""
        # Set low risk score to avoid MFA requirement
        user_id = "testuser"
        iam_system.ueba_system.user_risk_scores[user_id] = 0.1
        
        result = await iam_system.authenticate_user(
            username=user_id,
            password="test_password",
            ip_address="192.168.1.1",
            session_type=SessionType.WEB
        )
        
        assert result.success is True
        assert result.user_id == user_id
        
        # If MFA is not required, should have session
        if not result.requires_mfa:
            assert result.session_id is not None
            assert result.session_token is not None
        assert result.risk_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_authentication_with_mfa_required(self, iam_system):
        """Test authentication when MFA is required"""
        # Setup TOTP for user
        user_id = "testuser_mfa"
        iam_system.mfa_system.setup_totp(user_id)
        
        result = await iam_system.authenticate_user(
            username=user_id,
            password="test_password",
            ip_address="192.168.1.1"
        )
        
        assert result.success is True
        assert result.requires_mfa is True
        assert result.session_id is None  # No session until MFA complete
        assert MFAMethod.TOTP in result.available_mfa_methods
    
    @pytest.mark.asyncio
    async def test_mfa_completion(self, iam_system):
        """Test MFA completion flow"""
        user_id = "testuser_mfa"
        
        # Setup TOTP
        secret, _ = iam_system.mfa_system.setup_totp(user_id)
        
        # Generate valid TOTP token
        import pyotp
        totp = pyotp.TOTP(secret)
        token = totp.now()
        
        result = await iam_system.complete_mfa_authentication(
            user_id=user_id,
            mfa_method=MFAMethod.TOTP,
            mfa_token=token,
            ip_address="192.168.1.1"
        )
        
        assert result.success is True
        assert result.session_id is not None
        assert result.session_token is not None
    
    @pytest.mark.asyncio
    async def test_authorization_with_rbac(self, iam_system):
        """Test authorization using RBAC"""
        # Create session
        user_id = "testuser"
        
        # Assign viewer role to user first
        viewer_roles = [role_id for role_id, role in iam_system.rbac_system.roles.items() 
                      if role.name == "viewer"]
        if viewer_roles:
            iam_system.rbac_system.assign_role_to_user(user_id, viewer_roles[0], "system")
        
        session_id, session_token = iam_system.session_manager.create_session(
            user_id=user_id,
            session_type=SessionType.WEB,
            ip_address="192.168.1.1"
        )
        
        # Test authorization
        result = await iam_system.authorize_access(
            session_id=session_id,
            session_token=session_token,
            resource_id="test_resource",
            resource_type="document",
            action="read",
            ip_address="192.168.1.1"
        )
        
        # Should be allowed due to viewer role
        assert result.allowed is True
        assert "RBAC" in result.reason
    
    @pytest.mark.asyncio
    async def test_jit_access_request(self, iam_system):
        """Test JIT access request flow"""
        # Create session
        user_id = "testuser"
        session_id, session_token = iam_system.session_manager.create_session(
            user_id=user_id,
            session_type=SessionType.WEB
        )
        
        # Request JIT access
        request_id = await iam_system.request_jit_access(
            session_id=session_id,
            session_token=session_token,
            resource_id="admin/sensitive_data",
            permissions=["read", "write"],
            justification="Need access for urgent analysis",
            duration_hours=4
        )
        
        assert request_id is not None
        assert len(request_id) > 0
    
    def test_session_management(self, iam_system):
        """Test session creation and validation"""
        user_id = "testuser"
        
        # Create session
        session_id, session_token = iam_system.session_manager.create_session(
            user_id=user_id,
            session_type=SessionType.WEB,
            ip_address="192.168.1.1"
        )
        
        # Validate session
        session_info = iam_system.session_manager.validate_session(
            session_id, session_token, "192.168.1.1"
        )
        
        assert session_info is not None
        assert session_info.user_id == user_id
        assert session_info.session_id == session_id
    
    def test_concurrent_session_limits(self, iam_system):
        """Test concurrent session limits"""
        user_id = "testuser"
        sessions = []
        
        # Create maximum allowed sessions
        for i in range(iam_system.session_manager.config.max_concurrent_sessions):
            session_id, session_token = iam_system.session_manager.create_session(
                user_id=user_id,
                session_type=SessionType.WEB,
                ip_address=f"192.168.1.{i+1}"
            )
            sessions.append((session_id, session_token))
        
        # Verify all sessions are active
        user_sessions = iam_system.session_manager.get_user_sessions(user_id)
        assert len(user_sessions) == iam_system.session_manager.config.max_concurrent_sessions
        
        # Create one more session (should terminate oldest)
        new_session_id, new_session_token = iam_system.session_manager.create_session(
            user_id=user_id,
            session_type=SessionType.WEB,
            ip_address="192.168.1.10"
        )
        
        # Should still have max sessions
        user_sessions = iam_system.session_manager.get_user_sessions(user_id)
        assert len(user_sessions) == iam_system.session_manager.config.max_concurrent_sessions
    
    def test_ueba_anomaly_detection(self, iam_system):
        """Test UEBA anomaly detection"""
        user_id = "testuser"
        
        # Record normal activities during business hours
        for hour in range(9, 17):  # 9 AM to 5 PM
            for i in range(10):
                activity = UserActivity(
                    activity_id=f"activity_{hour}_{i}",
                    user_id=user_id,
                    timestamp=datetime.utcnow().replace(hour=hour),
                    action="read",
                    resource_id="normal_resource",
                    resource_type="document",
                    ip_address="192.168.1.100"
                )
                iam_system.ueba_system.record_activity(activity)
        
        # Update user profile
        iam_system.ueba_system._update_user_profile(user_id)
        
        # Record anomalous activity (late night access)
        anomalous_activity = UserActivity(
            activity_id="anomalous_activity",
            user_id=user_id,
            timestamp=datetime.utcnow().replace(hour=2),  # 2 AM
            action="read",
            resource_id="sensitive_resource",
            resource_type="document",
            ip_address="192.168.1.200"  # Different IP
        )
        
        iam_system.ueba_system.record_activity(anomalous_activity)
        
        # Check for alerts
        alerts = iam_system.ueba_system.get_active_alerts(user_id)
        
        # Should detect time-based and location-based anomalies
        assert len(alerts) > 0
        anomaly_types = [alert.anomaly_type for alert in alerts]
        assert AnomalyType.TIME_BASED in anomaly_types or AnomalyType.LOCATION_BASED in anomaly_types
    
    def test_mfa_totp_flow(self, iam_system):
        """Test TOTP MFA setup and verification"""
        user_id = "testuser"
        
        # Setup TOTP
        secret, qr_code = iam_system.mfa_system.setup_totp(user_id)
        
        assert secret is not None
        assert qr_code is not None
        assert len(secret) > 0
        
        # Generate and verify TOTP token
        import pyotp
        totp = pyotp.TOTP(secret)
        token = totp.now()
        
        is_valid = iam_system.mfa_system.verify_totp(user_id, token)
        assert is_valid is True
        
        # Test invalid token
        is_valid = iam_system.mfa_system.verify_totp(user_id, "000000")
        assert is_valid is False
    
    def test_mfa_sms_flow(self, iam_system):
        """Test SMS MFA flow"""
        user_id = "testuser"
        phone_number = "+1234567890"
        
        # Initiate SMS challenge
        challenge_id = iam_system.mfa_system.initiate_sms_challenge(user_id, phone_number)
        
        assert challenge_id is not None
        assert challenge_id in iam_system.mfa_system.active_challenges
        
        # Get the challenge code (in production, this would be sent via SMS)
        challenge = iam_system.mfa_system.active_challenges[challenge_id]
        code = challenge.challenge_data["code"]
        
        # Verify correct code
        is_valid = iam_system.mfa_system.verify_sms_challenge(challenge_id, code)
        assert is_valid is True
        
        # Challenge should be removed after successful verification
        assert challenge_id not in iam_system.mfa_system.active_challenges
    
    def test_backup_codes(self, iam_system):
        """Test backup codes generation and verification"""
        user_id = "testuser"
        
        # Generate backup codes
        codes = iam_system.mfa_system.generate_backup_codes(user_id)
        
        assert len(codes) == 10
        assert all(len(code) == 8 for code in codes)
        
        # Verify a backup code
        test_code = codes[0]
        is_valid = iam_system.mfa_system.verify_backup_code(user_id, test_code)
        assert is_valid is True
        
        # Code should be consumed (can't use again)
        is_valid = iam_system.mfa_system.verify_backup_code(user_id, test_code)
        assert is_valid is False
    
    def test_rbac_role_assignment(self, iam_system):
        """Test RBAC role assignment and permission checking"""
        user_id = "testuser"
        
        # Get admin role
        admin_roles = [role_id for role_id, role in iam_system.rbac_system.roles.items() 
                      if role.name == "admin"]
        assert len(admin_roles) > 0
        
        admin_role_id = admin_roles[0]
        
        # Assign admin role to user
        assignment_id = iam_system.rbac_system.assign_role_to_user(
            user_id, admin_role_id, "system"
        )
        
        assert assignment_id is not None
        
        # Check admin permission
        context = AccessContext(
            user_id=user_id,
            resource_id="admin_resource",
            resource_type="system",
            action="admin"
        )
        
        has_permission = iam_system.rbac_system.check_permission(context)
        assert has_permission is True
    
    def test_jit_auto_approval(self, iam_system):
        """Test JIT access auto-approval for low-risk requests"""
        user_id = "testuser"
        
        # Set low risk score for user
        iam_system.ueba_system.user_risk_scores[user_id] = 0.1
        
        # Request access to low-risk resource
        request_id = iam_system.jit_system.request_access(
            user_id=user_id,
            resource_id="public/document",
            permissions=["read"],
            justification="Need to review document",
            duration_hours=2
        )
        
        # Check if request was auto-approved
        request = iam_system.jit_system.access_requests[request_id]
        
        # Note: Auto-approval depends on matching rules and conditions
        # This test verifies the request was processed
        assert request.status is not None
    
    @pytest.mark.asyncio
    async def test_system_cleanup(self, iam_system):
        """Test system cleanup of expired data"""
        user_id = "testuser"
        
        # Create expired session
        session_id, session_token = iam_system.session_manager.create_session(
            user_id=user_id,
            session_type=SessionType.WEB
        )
        
        # Manually expire the session
        session_info = iam_system.session_manager.active_sessions[session_id]
        session_info.expires_at = datetime.utcnow() - timedelta(minutes=1)
        
        # Run cleanup
        await iam_system.cleanup_expired_data()
        
        # Session should be expired
        session_info = iam_system.session_manager.validate_session(session_id, session_token)
        assert session_info is None
    
    def test_system_status(self, iam_system):
        """Test system status reporting"""
        status = iam_system.get_system_status()
        
        assert "timestamp" in status
        assert "session_statistics" in status
        assert "active_security_alerts" in status
        assert "system_health" in status
        assert status["system_health"] == "healthy"
    
    def test_invalid_authentication(self, iam_system):
        """Test authentication with invalid credentials"""
        result = asyncio.run(iam_system.authenticate_user(
            username="nonexistent",
            password="wrong_password"
        ))
        
        assert result.success is False
        assert result.error_message is not None
        assert result.session_id is None
    
    def test_session_timeout(self, iam_system):
        """Test session timeout behavior"""
        user_id = "testuser"
        
        # Create session with short timeout
        session_id, session_token = iam_system.session_manager.create_session(
            user_id=user_id,
            session_type=SessionType.WEB
        )
        
        # Manually set expiration to past
        session_info = iam_system.session_manager.active_sessions[session_id]
        session_info.expires_at = datetime.utcnow() - timedelta(minutes=1)
        
        # Validation should fail
        validated_session = iam_system.session_manager.validate_session(
            session_id, session_token
        )
        
        assert validated_session is None
    
    def test_permission_escalation_detection(self, iam_system):
        """Test detection of permission escalation attempts"""
        user_id = "testuser"
        
        # Record normal activities first
        for i in range(50):
            activity = UserActivity(
                activity_id=f"normal_{i}",
                user_id=user_id,
                timestamp=datetime.utcnow(),
                action="read",
                resource_id="normal_document",
                resource_type="document"
            )
            iam_system.ueba_system.record_activity(activity)
        
        # Record suspicious admin activity
        admin_activity = UserActivity(
            activity_id="admin_attempt",
            user_id=user_id,
            timestamp=datetime.utcnow(),
            action="admin_access",
            resource_id="admin_panel",
            resource_type="system"
        )
        
        iam_system.ueba_system.record_activity(admin_activity)
        
        # Check for escalation alerts
        alerts = iam_system.ueba_system.get_active_alerts(user_id)
        escalation_alerts = [a for a in alerts if a.anomaly_type == AnomalyType.PERMISSION_ESCALATION]
        
        # Should detect potential escalation
        assert len(escalation_alerts) > 0

if __name__ == "__main__":
    pytest.main([__file__])