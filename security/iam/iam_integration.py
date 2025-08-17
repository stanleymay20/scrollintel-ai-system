"""
IAM System Integration
Combines all IAM components into a unified system
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

from .mfa_system import MFASystem, MFAMethod
from .jit_access import JITAccessSystem, AccessRequest
from .rbac_system import RBACSystem, AccessContext
from .ueba_system import UEBASystem, UserActivity, AnomalyAlert
from .session_manager import SessionManager, SessionConfig, SessionType, SessionInfo

logger = logging.getLogger(__name__)

@dataclass
class AuthenticationResult:
    success: bool
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    session_token: Optional[str] = None
    requires_mfa: bool = False
    mfa_challenge_id: Optional[str] = None
    available_mfa_methods: List[MFAMethod] = None
    error_message: Optional[str] = None
    risk_score: float = 0.0

@dataclass
class AuthorizationResult:
    allowed: bool
    reason: Optional[str] = None
    temporary_access: bool = False
    access_expires_at: Optional[datetime] = None
    risk_factors: List[str] = None

class IAMSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize subsystems
        self.mfa_system = MFASystem(config.get("mfa", {}))
        self.jit_system = JITAccessSystem(config.get("jit", {}))
        self.rbac_system = RBACSystem(config.get("rbac", {}))
        self.ueba_system = UEBASystem(config.get("ueba", {}))
        
        session_config = SessionConfig(**config.get("session", {}))
        self.session_manager = SessionManager(session_config)
        
        # User credentials storage (in production, use secure database)
        self.user_credentials: Dict[str, Dict[str, Any]] = {}
        
        logger.info("IAM System initialized")
    
    async def authenticate_user(self, username: str, password: str,
                              ip_address: Optional[str] = None,
                              user_agent: Optional[str] = None,
                              device_fingerprint: Optional[str] = None,
                              session_type: SessionType = SessionType.WEB) -> AuthenticationResult:
        """Authenticate user with primary credentials"""
        try:
            # Validate primary credentials
            if not self._validate_credentials(username, password):
                logger.warning(f"Authentication failed for user {username}")
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid credentials"
                )
            
            user_id = username  # In production, map username to user_id
            
            # Record authentication activity
            activity = UserActivity(
                activity_id=f"auth_{user_id}_{int(datetime.utcnow().timestamp())}",
                user_id=user_id,
                timestamp=datetime.utcnow(),
                action="authenticate",
                resource_id="auth_system",
                resource_type="authentication",
                ip_address=ip_address,
                user_agent=user_agent,
                success=True
            )
            
            self.ueba_system.record_activity(activity)
            
            # Get user risk score
            risk_score = self.ueba_system.get_user_risk_score(user_id)
            
            # Check if MFA is required
            available_mfa_methods = self.mfa_system.get_user_mfa_methods(user_id)
            requires_mfa = len(available_mfa_methods) > 0 or risk_score > 0.5
            
            if requires_mfa:
                return AuthenticationResult(
                    success=True,
                    user_id=user_id,
                    requires_mfa=True,
                    available_mfa_methods=available_mfa_methods,
                    risk_score=risk_score
                )
            
            # Create session
            session_id, session_token = self.session_manager.create_session(
                user_id=user_id,
                session_type=session_type,
                ip_address=ip_address,
                user_agent=user_agent,
                device_fingerprint=device_fingerprint
            )
            
            logger.info(f"User authenticated successfully: {user_id}")
            
            return AuthenticationResult(
                success=True,
                user_id=user_id,
                session_id=session_id,
                session_token=session_token,
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return AuthenticationResult(
                success=False,
                error_message="Authentication system error"
            )
    
    async def complete_mfa_authentication(self, user_id: str, mfa_method: MFAMethod,
                                        mfa_token: str, challenge_id: Optional[str] = None,
                                        ip_address: Optional[str] = None,
                                        user_agent: Optional[str] = None,
                                        device_fingerprint: Optional[str] = None,
                                        session_type: SessionType = SessionType.WEB) -> AuthenticationResult:
        """Complete MFA authentication"""
        try:
            mfa_success = False
            
            # Verify MFA based on method
            if mfa_method == MFAMethod.TOTP:
                mfa_success = self.mfa_system.verify_totp(user_id, mfa_token)
            elif mfa_method == MFAMethod.SMS and challenge_id:
                mfa_success = self.mfa_system.verify_sms_challenge(challenge_id, mfa_token)
            elif mfa_method == MFAMethod.BACKUP_CODES:
                mfa_success = self.mfa_system.verify_backup_code(user_id, mfa_token)
            elif mfa_method == MFAMethod.BIOMETRIC:
                # In production, biometric_data would be provided
                biometric_data = mfa_token.encode()  # Simplified
                mfa_success = self.mfa_system.verify_biometric(user_id, "fingerprint", biometric_data)
            
            if not mfa_success:
                logger.warning(f"MFA verification failed for user {user_id}")
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid MFA token"
                )
            
            # Record MFA activity
            activity = UserActivity(
                activity_id=f"mfa_{user_id}_{int(datetime.utcnow().timestamp())}",
                user_id=user_id,
                timestamp=datetime.utcnow(),
                action="mfa_verify",
                resource_id="mfa_system",
                resource_type="authentication",
                ip_address=ip_address,
                user_agent=user_agent,
                success=True,
                metadata={"mfa_method": mfa_method.value}
            )
            
            self.ueba_system.record_activity(activity)
            
            # Create session
            session_id, session_token = self.session_manager.create_session(
                user_id=user_id,
                session_type=session_type,
                ip_address=ip_address,
                user_agent=user_agent,
                device_fingerprint=device_fingerprint
            )
            
            logger.info(f"MFA authentication completed for user {user_id}")
            
            return AuthenticationResult(
                success=True,
                user_id=user_id,
                session_id=session_id,
                session_token=session_token,
                risk_score=self.ueba_system.get_user_risk_score(user_id)
            )
            
        except Exception as e:
            logger.error(f"MFA authentication error: {str(e)}")
            return AuthenticationResult(
                success=False,
                error_message="MFA authentication system error"
            )
    
    async def authorize_access(self, session_id: str, session_token: str,
                             resource_id: str, resource_type: str, action: str,
                             ip_address: Optional[str] = None,
                             additional_context: Optional[Dict[str, Any]] = None) -> AuthorizationResult:
        """Authorize access to resource"""
        try:
            # Validate session
            session_info = self.session_manager.validate_session(
                session_id, session_token, ip_address
            )
            
            if not session_info:
                return AuthorizationResult(
                    allowed=False,
                    reason="Invalid or expired session"
                )
            
            user_id = session_info.user_id
            
            # Create access context
            context = AccessContext(
                user_id=user_id,
                resource_id=resource_id,
                resource_type=resource_type,
                action=action,
                ip_address=ip_address,
                timestamp=datetime.utcnow(),
                additional_context=additional_context or {}
            )
            
            # Record access activity
            activity = UserActivity(
                activity_id=f"access_{user_id}_{int(datetime.utcnow().timestamp())}",
                user_id=user_id,
                timestamp=datetime.utcnow(),
                action=action,
                resource_id=resource_id,
                resource_type=resource_type,
                ip_address=ip_address,
                success=True  # Will be updated based on authorization result
            )
            
            # Check RBAC permissions
            rbac_allowed = self.rbac_system.check_permission(context)
            
            if rbac_allowed:
                activity.success = True
                self.ueba_system.record_activity(activity)
                
                return AuthorizationResult(
                    allowed=True,
                    reason="RBAC permission granted"
                )
            
            # Check JIT access
            jit_allowed = self.jit_system.check_access(user_id, resource_id, action)
            
            if jit_allowed:
                activity.success = True
                activity.metadata["access_type"] = "jit"
                self.ueba_system.record_activity(activity)
                
                # Get JIT access details
                user_accesses = self.jit_system.get_user_active_accesses(user_id)
                matching_access = next(
                    (access for access in user_accesses 
                     if access.resource_id == resource_id and action in access.permissions),
                    None
                )
                
                return AuthorizationResult(
                    allowed=True,
                    reason="JIT access granted",
                    temporary_access=True,
                    access_expires_at=matching_access.expires_at if matching_access else None
                )
            
            # Access denied
            activity.success = False
            activity.metadata["denial_reason"] = "insufficient_permissions"
            self.ueba_system.record_activity(activity)
            
            logger.warning(f"Access denied for user {user_id} to {resource_id}")
            
            return AuthorizationResult(
                allowed=False,
                reason="Insufficient permissions"
            )
            
        except Exception as e:
            logger.error(f"Authorization error: {str(e)}")
            return AuthorizationResult(
                allowed=False,
                reason="Authorization system error"
            )
    
    async def request_jit_access(self, session_id: str, session_token: str,
                               resource_id: str, permissions: List[str],
                               justification: str, duration_hours: int = 8) -> str:
        """Request just-in-time access"""
        try:
            # Validate session
            session_info = self.session_manager.validate_session(session_id, session_token)
            
            if not session_info:
                raise ValueError("Invalid or expired session")
            
            user_id = session_info.user_id
            
            # Submit access request
            request_id = self.jit_system.request_access(
                user_id=user_id,
                resource_id=resource_id,
                permissions=permissions,
                justification=justification,
                duration_hours=duration_hours
            )
            
            logger.info(f"JIT access request submitted: {request_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"JIT access request error: {str(e)}")
            raise
    
    def get_security_alerts(self, user_id: Optional[str] = None) -> List[AnomalyAlert]:
        """Get security alerts from UEBA system"""
        return self.ueba_system.get_active_alerts(user_id)
    
    def get_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """Get active sessions for user"""
        return self.session_manager.get_user_sessions(user_id)
    
    def terminate_session(self, session_id: str, reason: str = "user_logout") -> bool:
        """Terminate a session"""
        return self.session_manager.terminate_session(session_id, reason)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall IAM system status"""
        try:
            session_stats = self.session_manager.get_session_statistics()
            active_alerts = len(self.ueba_system.get_active_alerts())
            pending_requests = sum(
                len(self.jit_system.get_pending_approvals(approver))
                for approver in self.jit_system.pending_approvals.keys()
            )
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "session_statistics": session_stats,
                "active_security_alerts": active_alerts,
                "pending_jit_requests": pending_requests,
                "system_health": "healthy"
            }
            
        except Exception as e:
            logger.error(f"System status error: {str(e)}")
            return {"system_health": "error", "error": str(e)}
    
    async def cleanup_expired_data(self):
        """Cleanup expired sessions, challenges, and accesses"""
        try:
            # Cleanup expired sessions
            expired_sessions = self.session_manager.cleanup_expired_sessions()
            
            # Cleanup expired MFA challenges
            self.mfa_system.cleanup_expired_challenges()
            
            # Cleanup expired JIT accesses
            self.jit_system.cleanup_expired_accesses()
            
            # Cleanup expired RBAC assignments
            self.rbac_system.cleanup_expired_assignments()
            
            logger.info(f"Cleanup completed: {expired_sessions} sessions expired")
            
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
    
    def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials (simplified implementation)"""
        # In production, use secure password hashing and database storage
        if username not in self.user_credentials:
            # Only create user if password is correct
            if password == "test_password":
                self.user_credentials[username] = {
                    "password_hash": "test_hash",  # In production, use proper hashing
                    "created_at": datetime.utcnow()
                }
                return True
            return False
        
        # Simplified validation - in production, use proper password verification
        return password == "test_password"
    
    # Convenience methods for setting up test data
    def setup_test_user(self, user_id: str, phone_number: Optional[str] = None):
        """Setup test user with MFA"""
        try:
            # Setup TOTP
            secret, qr_code = self.mfa_system.setup_totp(user_id)
            
            # Generate backup codes
            backup_codes = self.mfa_system.generate_backup_codes(user_id)
            
            # Create basic RBAC role
            viewer_roles = [role_id for role_id, role in self.rbac_system.roles.items() 
                          if role.name == "viewer"]
            if viewer_roles:
                self.rbac_system.assign_role_to_user(user_id, viewer_roles[0], "system")
            
            logger.info(f"Test user setup completed: {user_id}")
            
            return {
                "totp_secret": secret,
                "qr_code": qr_code,
                "backup_codes": backup_codes
            }
            
        except Exception as e:
            logger.error(f"Test user setup failed: {str(e)}")
            raise