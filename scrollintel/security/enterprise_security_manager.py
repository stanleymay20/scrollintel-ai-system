"""
Enterprise Security Manager for Agent Steering System
Implements multi-factor authentication, SSO, encryption, RBAC, and audit logging
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for different operations"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

class EncryptionLevel(Enum):
    """Encryption levels"""
    BASIC = "aes_256"
    ADVANCED = "rsa_4096"
    QUANTUM_SAFE = "post_quantum"

class AuthenticationMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    MFA = "multi_factor"
    SSO = "single_sign_on"
    BIOMETRIC = "biometric"
    CERTIFICATE = "certificate"

@dataclass
class UserCredentials:
    """User credentials for authentication"""
    username: str
    password: Optional[str] = None
    mfa_token: Optional[str] = None
    sso_token: Optional[str] = None
    biometric_data: Optional[str] = None
    certificate: Optional[str] = None

@dataclass
class User:
    """User entity"""
    id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    security_clearance: SecurityLevel
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    is_locked: bool = False

@dataclass
class AuthenticationResult:
    """Result of authentication attempt"""
    success: bool
    user: Optional[User] = None
    token: Optional[str] = None
    permissions: List[str] = None
    reason: Optional[str] = None
    expires_at: Optional[datetime] = None

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_type: str
    user_id: Optional[str]
    resource: Optional[str]
    action: str
    result: str
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = None

@dataclass
class AuditCriteria:
    """Criteria for audit report generation"""
    start_date: datetime
    end_date: datetime
    event_types: List[str] = None
    user_ids: List[str] = None
    resources: List[str] = None

@dataclass
class AuditReport:
    """Audit report"""
    criteria: AuditCriteria
    events: List[SecurityEvent]
    summary: Dict[str, Any]
    generated_at: datetime

class EnterpriseIdentityProvider:
    """Enterprise identity provider for authentication"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.mfa_secrets: Dict[str, str] = {}
        self.sso_providers: Dict[str, Dict] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.lockout_duration = timedelta(minutes=30)
        
    async def verify_mfa(self, credentials: UserCredentials) -> Dict[str, Any]:
        """Verify multi-factor authentication"""
        try:
            if not credentials.mfa_token:
                return {"success": False, "reason": "MFA token required"}
                
            # Simulate TOTP verification
            user_secret = self.mfa_secrets.get(credentials.username)
            if not user_secret:
                return {"success": False, "reason": "MFA not configured"}
                
            # In real implementation, verify TOTP token
            # For demo, accept tokens starting with "123"
            if credentials.mfa_token.startswith("123"):
                return {"success": True}
            else:
                return {"success": False, "reason": "Invalid MFA token"}
                
        except Exception as e:
            logger.error(f"MFA verification error: {e}")
            return {"success": False, "reason": "MFA verification failed"}
    
    async def validate_sso(self, credentials: UserCredentials) -> Dict[str, Any]:
        """Validate single sign-on token"""
        try:
            if not credentials.sso_token:
                return {"success": False, "reason": "SSO token required"}
                
            # Decode and validate JWT token
            try:
                payload = jwt.decode(
                    credentials.sso_token, 
                    "secret_key",  # In production, use proper key management
                    algorithms=["HS256"]
                )
                
                user_data = payload.get("user")
                if not user_data:
                    return {"success": False, "reason": "Invalid token payload"}
                    
                # Create or update user
                user = User(
                    id=user_data.get("id", credentials.username),
                    username=credentials.username,
                    email=user_data.get("email", f"{credentials.username}@company.com"),
                    roles=user_data.get("roles", ["user"]),
                    permissions=user_data.get("permissions", []),
                    security_clearance=SecurityLevel(user_data.get("clearance", "internal"))
                )
                
                return {"success": True, "user": user}
                
            except jwt.InvalidTokenError as e:
                return {"success": False, "reason": f"Invalid SSO token: {e}"}
                
        except Exception as e:
            logger.error(f"SSO validation error: {e}")
            return {"success": False, "reason": "SSO validation failed"}

class QuantumSafeEncryption:
    """Quantum-safe encryption service"""
    
    def __init__(self):
        self.symmetric_key = Fernet.generate_key()
        self.fernet = Fernet(self.symmetric_key)
        self.rsa_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        
    async def encrypt_data(self, data: bytes, level: EncryptionLevel) -> bytes:
        """Encrypt data with specified encryption level"""
        try:
            if level == EncryptionLevel.BASIC:
                return self.fernet.encrypt(data)
            elif level == EncryptionLevel.ADVANCED:
                # RSA encryption for smaller data
                if len(data) > 446:  # RSA 4096 limit
                    # Hybrid encryption: RSA for key, AES for data
                    aes_key = Fernet.generate_key()
                    aes_cipher = Fernet(aes_key)
                    encrypted_data = aes_cipher.encrypt(data)
                    
                    # Encrypt AES key with RSA
                    encrypted_key = self.rsa_key.public_key().encrypt(
                        aes_key,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                    
                    return encrypted_key + b"||" + encrypted_data
                else:
                    return self.rsa_key.public_key().encrypt(
                        data,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
            else:
                # Quantum-safe encryption (placeholder for post-quantum algorithms)
                return self.fernet.encrypt(data)
                
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    async def decrypt_data(self, encrypted_data: bytes, level: EncryptionLevel) -> bytes:
        """Decrypt data with specified encryption level"""
        try:
            if level == EncryptionLevel.BASIC:
                return self.fernet.decrypt(encrypted_data)
            elif level == EncryptionLevel.ADVANCED:
                if b"||" in encrypted_data:
                    # Hybrid decryption
                    encrypted_key, encrypted_data_part = encrypted_data.split(b"||", 1)
                    
                    # Decrypt AES key with RSA
                    aes_key = self.rsa_key.decrypt(
                        encrypted_key,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                    
                    # Decrypt data with AES
                    aes_cipher = Fernet(aes_key)
                    return aes_cipher.decrypt(encrypted_data_part)
                else:
                    return self.rsa_key.decrypt(
                        encrypted_data,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
            else:
                # Quantum-safe decryption
                return self.fernet.decrypt(encrypted_data)
                
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise

class ComplianceAuditLogger:
    """Compliance audit logger for security events"""
    
    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.log_file = "security_audit.log"
        
    async def log_security_event(self, event: SecurityEvent) -> None:
        """Log security event"""
        try:
            # Add to memory store
            self.events.append(event)
            
            # Write to file
            log_entry = {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "user_id": event.user_id,
                "resource": event.resource,
                "action": event.action,
                "result": event.result,
                "ip_address": event.ip_address,
                "user_agent": event.user_agent,
                "details": event.details
            }
            
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
            logger.info(f"Security event logged: {event.event_type}")
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    async def log_failed_authentication(self, username: str, reason: str = None) -> None:
        """Log failed authentication attempt"""
        event = SecurityEvent(
            event_type="authentication_failed",
            user_id=username,
            resource="authentication",
            action="login",
            result="failed",
            timestamp=datetime.utcnow(),
            details={"reason": reason}
        )
        await self.log_security_event(event)
    
    async def log_successful_authentication(self, username: str) -> None:
        """Log successful authentication"""
        event = SecurityEvent(
            event_type="authentication_success",
            user_id=username,
            resource="authentication",
            action="login",
            result="success",
            timestamp=datetime.utcnow()
        )
        await self.log_security_event(event)
    
    async def generate_audit_report(self, criteria: AuditCriteria) -> AuditReport:
        """Generate audit report based on criteria"""
        try:
            # Filter events based on criteria
            filtered_events = []
            
            for event in self.events:
                if event.timestamp < criteria.start_date or event.timestamp > criteria.end_date:
                    continue
                    
                if criteria.event_types and event.event_type not in criteria.event_types:
                    continue
                    
                if criteria.user_ids and event.user_id not in criteria.user_ids:
                    continue
                    
                if criteria.resources and event.resource not in criteria.resources:
                    continue
                    
                filtered_events.append(event)
            
            # Generate summary
            summary = {
                "total_events": len(filtered_events),
                "event_types": {},
                "users": {},
                "resources": {},
                "success_rate": 0
            }
            
            successful_events = 0
            
            for event in filtered_events:
                # Count by event type
                summary["event_types"][event.event_type] = summary["event_types"].get(event.event_type, 0) + 1
                
                # Count by user
                if event.user_id:
                    summary["users"][event.user_id] = summary["users"].get(event.user_id, 0) + 1
                
                # Count by resource
                if event.resource:
                    summary["resources"][event.resource] = summary["resources"].get(event.resource, 0) + 1
                
                # Count successful events
                if event.result == "success":
                    successful_events += 1
            
            if len(filtered_events) > 0:
                summary["success_rate"] = successful_events / len(filtered_events)
            
            return AuditReport(
                criteria=criteria,
                events=filtered_events,
                summary=summary,
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to generate audit report: {e}")
            raise

class RoleBasedAccessController:
    """Role-based access control system"""
    
    def __init__(self):
        self.roles: Dict[str, List[str]] = {
            "admin": ["*"],  # All permissions
            "manager": [
                "agents.read", "agents.manage", "data.read", "data.write",
                "reports.read", "reports.generate", "users.read"
            ],
            "analyst": [
                "agents.read", "data.read", "reports.read", "reports.generate"
            ],
            "user": [
                "agents.read", "data.read", "reports.read"
            ],
            "guest": [
                "agents.read"
            ]
        }
        
        self.user_roles: Dict[str, List[str]] = {}
        
    async def get_user_permissions(self, user: User) -> List[str]:
        """Get all permissions for a user"""
        permissions = set()
        
        for role in user.roles:
            role_permissions = self.roles.get(role, [])
            permissions.update(role_permissions)
        
        # Add direct permissions
        permissions.update(user.permissions)
        
        return list(permissions)
    
    async def check_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission"""
        user_permissions = await self.get_user_permissions(user)
        
        # Check for wildcard permission
        if "*" in user_permissions:
            return True
            
        # Check for exact permission
        if permission in user_permissions:
            return True
            
        # Check for parent permissions (e.g., "agents.*" covers "agents.read")
        for perm in user_permissions:
            if perm.endswith("*"):
                prefix = perm[:-1]
                if permission.startswith(prefix):
                    return True
        
        return False

class EnterpriseSecurityManager:
    """Main enterprise security manager"""
    
    def __init__(self):
        self.identity_provider = EnterpriseIdentityProvider()
        self.encryption_service = QuantumSafeEncryption()
        self.audit_logger = ComplianceAuditLogger()
        self.access_controller = RoleBasedAccessController()
        self.session_tokens: Dict[str, Dict] = {}
        self.jwt_secret = secrets.token_urlsafe(32)
        
    async def authenticate_user(self, credentials: UserCredentials) -> AuthenticationResult:
        """Authenticate user with multi-factor authentication and SSO"""
        try:
            # Check if user is locked
            user = self.identity_provider.users.get(credentials.username)
            if user and user.is_locked:
                await self.audit_logger.log_failed_authentication(
                    credentials.username, "Account locked"
                )
                return AuthenticationResult(
                    success=False,
                    reason="Account is locked"
                )
            
            # Multi-factor authentication
            mfa_result = await self.identity_provider.verify_mfa(credentials)
            if not mfa_result["success"]:
                await self.audit_logger.log_failed_authentication(
                    credentials.username, mfa_result["reason"]
                )
                return AuthenticationResult(
                    success=False,
                    reason=mfa_result["reason"]
                )
            
            # Single sign-on integration
            sso_result = await self.identity_provider.validate_sso(credentials)
            if not sso_result["success"]:
                await self.audit_logger.log_failed_authentication(
                    credentials.username, sso_result["reason"]
                )
                return AuthenticationResult(
                    success=False,
                    reason=sso_result["reason"]
                )
            
            # Generate secure session token
            user = sso_result["user"]
            session_token = await self.generate_secure_token(user)
            
            # Get user permissions
            permissions = await self.access_controller.get_user_permissions(user)
            
            # Update user login info
            user.last_login = datetime.utcnow()
            user.failed_attempts = 0
            self.identity_provider.users[user.username] = user
            
            await self.audit_logger.log_successful_authentication(credentials.username)
            
            return AuthenticationResult(
                success=True,
                user=user,
                token=session_token,
                permissions=permissions,
                expires_at=datetime.utcnow() + timedelta(hours=8)
            )
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            await self.audit_logger.log_failed_authentication(
                credentials.username, f"System error: {e}"
            )
            return AuthenticationResult(
                success=False,
                reason="Authentication system error"
            )
    
    async def generate_secure_token(self, user: User) -> str:
        """Generate secure JWT session token"""
        payload = {
            "user_id": user.id,
            "username": user.username,
            "roles": user.roles,
            "security_clearance": user.security_clearance.value,
            "iat": int(time.time()),
            "exp": int(time.time()) + 28800,  # 8 hours
            "jti": secrets.token_urlsafe(16)  # JWT ID
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        
        # Store session info
        self.session_tokens[token] = {
            "user": user,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=8)
        }
        
        return token
    
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate session token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if token exists in session store
            session_info = self.session_tokens.get(token)
            if not session_info:
                return None
                
            # Check expiration
            if datetime.utcnow() > session_info["expires_at"]:
                del self.session_tokens[token]
                return None
                
            return session_info["user"]
            
        except jwt.InvalidTokenError:
            return None
    
    async def encrypt_sensitive_data(self, data: Union[str, bytes], level: EncryptionLevel = EncryptionLevel.BASIC) -> bytes:
        """Encrypt sensitive data"""
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        return await self.encryption_service.encrypt_data(data, level)
    
    async def decrypt_sensitive_data(self, encrypted_data: bytes, level: EncryptionLevel = EncryptionLevel.BASIC) -> bytes:
        """Decrypt sensitive data"""
        return await self.encryption_service.decrypt_data(encrypted_data, level)
    
    async def authorize_action(self, user: User, action: str, resource: str = None) -> bool:
        """Authorize user action with fine-grained permissions"""
        try:
            # Construct permission string
            permission = f"{resource}.{action}" if resource else action
            
            # Check permission
            has_permission = await self.access_controller.check_permission(user, permission)
            
            # Log authorization attempt
            event = SecurityEvent(
                event_type="authorization",
                user_id=user.id,
                resource=resource,
                action=action,
                result="success" if has_permission else "denied",
                timestamp=datetime.utcnow(),
                details={"permission": permission}
            )
            await self.audit_logger.log_security_event(event)
            
            return has_permission
            
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            return False
    
    async def generate_compliance_report(self, criteria: AuditCriteria) -> AuditReport:
        """Generate comprehensive compliance report"""
        return await self.audit_logger.generate_audit_report(criteria)
    
    async def setup_demo_data(self):
        """Setup demo users and data for testing"""
        # Create demo users
        admin_user = User(
            id="admin_001",
            username="admin",
            email="admin@company.com",
            roles=["admin"],
            permissions=[],
            security_clearance=SecurityLevel.SECRET
        )
        
        manager_user = User(
            id="mgr_001",
            username="manager",
            email="manager@company.com",
            roles=["manager"],
            permissions=[],
            security_clearance=SecurityLevel.CONFIDENTIAL
        )
        
        analyst_user = User(
            id="analyst_001",
            username="analyst",
            email="analyst@company.com",
            roles=["analyst"],
            permissions=[],
            security_clearance=SecurityLevel.INTERNAL
        )
        
        # Store users
        self.identity_provider.users["admin"] = admin_user
        self.identity_provider.users["manager"] = manager_user
        self.identity_provider.users["analyst"] = analyst_user
        
        # Setup MFA secrets
        self.identity_provider.mfa_secrets["admin"] = "ADMIN_SECRET"
        self.identity_provider.mfa_secrets["manager"] = "MANAGER_SECRET"
        self.identity_provider.mfa_secrets["analyst"] = "ANALYST_SECRET"
        
        logger.info("Demo security data setup complete")

# Global security manager instance
security_manager = EnterpriseSecurityManager()