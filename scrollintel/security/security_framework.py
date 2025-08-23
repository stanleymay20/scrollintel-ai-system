"""
Enterprise Security and Compliance Framework
Implements multi-factor authentication, encryption, RBAC, and audit logging
"""

import asyncio
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import logging

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class AuthenticationMethod(Enum):
    """Supported authentication methods"""
    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    MFA_SMS = "mfa_sms"
    SSO_SAML = "sso_saml"
    SSO_OAUTH = "sso_oauth"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"

class PermissionType(Enum):
    """Permission types for RBAC"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    AUDIT = "audit"

@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    session_id: str
    permissions: List[str]
    security_level: SecurityLevel
    authentication_methods: List[AuthenticationMethod]
    ip_address: str
    user_agent: str
    timestamp: datetime
    expires_at: datetime

@dataclass
class AuditEvent:
    """Audit event structure"""
    event_id: str
    user_id: str
    action: str
    resource: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    success: bool
    details: Dict[str, Any]
    security_level: SecurityLevel
    compliance_tags: List[str]

class EncryptionManager:
    """Handles all encryption operations"""
    
    def __init__(self):
        self.master_key = self._get_or_create_master_key()
        self.fernet = Fernet(self.master_key)
        self.rsa_private_key = self._get_or_create_rsa_key()
        self.rsa_public_key = self.rsa_private_key.public_key()
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key"""
        key_file = "security/keys/master.key"
        os.makedirs(os.path.dirname(key_file), exist_ok=True)
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def _get_or_create_rsa_key(self):
        """Get or create RSA key pair"""
        key_file = "security/keys/rsa_private.pem"
        os.makedirs(os.path.dirname(key_file), exist_ok=True)
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return serialization.load_pem_private_key(f.read(), password=None)
        else:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            with open(key_file, 'wb') as f:
                f.write(pem)
            
            return private_key
    
    def encrypt_data(self, data: Union[str, bytes], security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> str:
        """Encrypt data with appropriate security level"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if security_level in [SecurityLevel.RESTRICTED, SecurityLevel.TOP_SECRET]:
            # Use RSA for highest security
            encrypted = self.rsa_public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return base64.b64encode(encrypted).decode('utf-8')
        else:
            # Use Fernet for standard encryption
            encrypted = self.fernet.encrypt(data)
            return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str, security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> bytes:
        """Decrypt data based on security level"""
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        
        if security_level in [SecurityLevel.RESTRICTED, SecurityLevel.TOP_SECRET]:
            # Use RSA for highest security
            return self.rsa_private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        else:
            # Use Fernet for standard encryption
            return self.fernet.decrypt(encrypted_bytes)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = os.urandom(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode('utf-8'))
        
        return base64.b64encode(key).decode('utf-8'), base64.b64encode(salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """Verify password against hash"""
        try:
            salt_bytes = base64.b64decode(salt.encode('utf-8'))
            expected_hash, _ = self.hash_password(password, salt_bytes)
            return hmac.compare_digest(expected_hash, hashed_password)
        except Exception:
            return False

class MultiFactorAuth:
    """Multi-factor authentication manager"""
    
    def __init__(self):
        self.totp_secrets = {}
        self.sms_codes = {}
        self.backup_codes = {}
    
    def generate_totp_secret(self, user_id: str) -> str:
        """Generate TOTP secret for user"""
        secret = base64.b32encode(os.urandom(20)).decode('utf-8')
        self.totp_secrets[user_id] = secret
        return secret
    
    def verify_totp(self, user_id: str, token: str) -> bool:
        """Verify TOTP token"""
        if user_id not in self.totp_secrets:
            return False
        
        # Simplified TOTP verification (in production, use proper TOTP library)
        secret = self.totp_secrets[user_id]
        current_time = int(time.time() // 30)
        
        for time_window in [current_time - 1, current_time, current_time + 1]:
            expected_token = self._generate_totp_token(secret, time_window)
            if hmac.compare_digest(token, expected_token):
                return True
        
        return False
    
    def _generate_totp_token(self, secret: str, time_counter: int) -> str:
        """Generate TOTP token for given time counter"""
        # Simplified implementation - use proper TOTP library in production
        key = base64.b32decode(secret)
        time_bytes = time_counter.to_bytes(8, 'big')
        
        hmac_hash = hmac.new(key, time_bytes, hashlib.sha1).digest()
        offset = hmac_hash[-1] & 0x0f
        
        code = (
            (hmac_hash[offset] & 0x7f) << 24 |
            (hmac_hash[offset + 1] & 0xff) << 16 |
            (hmac_hash[offset + 2] & 0xff) << 8 |
            (hmac_hash[offset + 3] & 0xff)
        )
        
        return str(code % 1000000).zfill(6)
    
    def send_sms_code(self, user_id: str, phone_number: str) -> str:
        """Send SMS verification code"""
        code = str(secrets.randbelow(1000000)).zfill(6)
        self.sms_codes[user_id] = {
            'code': code,
            'expires_at': datetime.utcnow() + timedelta(minutes=5),
            'phone': phone_number
        }
        
        # In production, integrate with SMS service
        logger.info(f"SMS code {code} sent to {phone_number} for user {user_id}")
        return code
    
    def verify_sms_code(self, user_id: str, code: str) -> bool:
        """Verify SMS code"""
        if user_id not in self.sms_codes:
            return False
        
        stored_code = self.sms_codes[user_id]
        if datetime.utcnow() > stored_code['expires_at']:
            del self.sms_codes[user_id]
            return False
        
        if hmac.compare_digest(code, stored_code['code']):
            del self.sms_codes[user_id]
            return True
        
        return False
    
    def generate_backup_codes(self, user_id: str, count: int = 10) -> List[str]:
        """Generate backup codes for user"""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(8)
            codes.append(code)
        
        self.backup_codes[user_id] = [
            hashlib.sha256(code.encode()).hexdigest() for code in codes
        ]
        
        return codes
    
    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify and consume backup code"""
        if user_id not in self.backup_codes:
            return False
        
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        if code_hash in self.backup_codes[user_id]:
            self.backup_codes[user_id].remove(code_hash)
            return True
        
        return False

class RoleBasedAccessControl:
    """Role-based access control system"""
    
    def __init__(self):
        self.roles = {}
        self.user_roles = {}
        self.permissions = {}
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default roles and permissions"""
        # Define default roles
        self.create_role("admin", "System Administrator", [
            "system.admin", "user.manage", "audit.view", "security.manage"
        ])
        
        self.create_role("user", "Standard User", [
            "data.read", "agent.interact", "dashboard.view"
        ])
        
        self.create_role("analyst", "Data Analyst", [
            "data.read", "data.analyze", "report.create", "dashboard.view"
        ])
        
        self.create_role("auditor", "Security Auditor", [
            "audit.view", "compliance.check", "security.audit"
        ])
    
    def create_role(self, role_name: str, description: str, permissions: List[str]):
        """Create a new role with permissions"""
        self.roles[role_name] = {
            'name': role_name,
            'description': description,
            'permissions': set(permissions),
            'created_at': datetime.utcnow()
        }
    
    def assign_role(self, user_id: str, role_name: str):
        """Assign role to user"""
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} does not exist")
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role_name)
    
    def revoke_role(self, user_id: str, role_name: str):
        """Revoke role from user"""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission"""
        if user_id not in self.user_roles:
            return False
        
        user_permissions = self.get_user_permissions(user_id)
        return permission in user_permissions
    
    def get_user_permissions(self, user_id: str) -> set:
        """Get all permissions for user"""
        if user_id not in self.user_roles:
            return set()
        
        permissions = set()
        for role_name in self.user_roles[user_id]:
            if role_name in self.roles:
                permissions.update(self.roles[role_name]['permissions'])
        
        return permissions
    
    def get_user_roles(self, user_id: str) -> List[str]:
        """Get all roles for user"""
        return list(self.user_roles.get(user_id, set()))

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.audit_log = []
        self.compliance_rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules for different regulations"""
        return {
            'gdpr': {
                'data_retention_days': 2555,  # 7 years
                'required_fields': ['user_id', 'action', 'timestamp', 'legal_basis'],
                'encryption_required': True
            },
            'sox': {
                'data_retention_days': 2555,  # 7 years
                'required_fields': ['user_id', 'action', 'resource', 'timestamp'],
                'immutable_required': True
            },
            'hipaa': {
                'data_retention_days': 2190,  # 6 years
                'required_fields': ['user_id', 'action', 'phi_accessed', 'timestamp'],
                'encryption_required': True,
                'access_logging_required': True
            }
        }
    
    def log_event(
        self,
        user_id: str,
        action: str,
        resource: str,
        success: bool,
        details: Dict[str, Any],
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
        compliance_tags: List[str] = None,
        ip_address: str = None,
        user_agent: str = None
    ) -> str:
        """Log audit event"""
        event_id = secrets.token_hex(16)
        
        event = AuditEvent(
            event_id=event_id,
            user_id=user_id,
            action=action,
            resource=resource,
            timestamp=datetime.utcnow(),
            ip_address=ip_address or "unknown",
            user_agent=user_agent or "unknown",
            success=success,
            details=details,
            security_level=security_level,
            compliance_tags=compliance_tags or []
        )
        
        # Encrypt sensitive audit data
        encrypted_event = self._encrypt_audit_event(event)
        self.audit_log.append(encrypted_event)
        
        # Check compliance requirements
        self._check_compliance(event)
        
        logger.info(f"Audit event logged: {event_id} - {action} on {resource}")
        return event_id
    
    def _encrypt_audit_event(self, event: AuditEvent) -> Dict[str, Any]:
        """Encrypt audit event based on security level"""
        event_dict = asdict(event)
        
        # Always encrypt sensitive fields
        sensitive_fields = ['details', 'user_agent']
        for field in sensitive_fields:
            if field in event_dict and event_dict[field]:
                event_dict[field] = self.encryption_manager.encrypt_data(
                    json.dumps(event_dict[field]),
                    event.security_level
                )
        
        # Convert datetime to ISO string
        event_dict['timestamp'] = event.timestamp.isoformat()
        
        return event_dict
    
    def _check_compliance(self, event: AuditEvent):
        """Check event against compliance requirements"""
        for tag in event.compliance_tags:
            if tag in self.compliance_rules:
                rules = self.compliance_rules[tag]
                
                # Check required fields
                event_dict = asdict(event)
                for required_field in rules.get('required_fields', []):
                    if required_field not in event_dict or not event_dict[required_field]:
                        logger.warning(f"Compliance violation: Missing required field {required_field} for {tag}")
    
    def search_audit_log(
        self,
        user_id: str = None,
        action: str = None,
        resource: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search audit log with filters"""
        results = []
        
        for encrypted_event in self.audit_log:
            # Basic filtering (in production, use proper database queries)
            if user_id and encrypted_event.get('user_id') != user_id:
                continue
            if action and encrypted_event.get('action') != action:
                continue
            if resource and encrypted_event.get('resource') != resource:
                continue
            
            # Time filtering
            event_time = datetime.fromisoformat(encrypted_event['timestamp'])
            if start_time and event_time < start_time:
                continue
            if end_time and event_time > end_time:
                continue
            
            results.append(encrypted_event)
            
            if len(results) >= limit:
                break
        
        return results
    
    def generate_compliance_report(self, regulation: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specific regulation"""
        if regulation not in self.compliance_rules:
            raise ValueError(f"Unknown regulation: {regulation}")
        
        events = self.search_audit_log(start_time=start_date, end_time=end_date)
        
        # Filter events with compliance tag
        compliance_events = [
            event for event in events
            if regulation in event.get('compliance_tags', [])
        ]
        
        report = {
            'regulation': regulation,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_events': len(compliance_events),
            'event_summary': self._summarize_events(compliance_events),
            'compliance_status': self._assess_compliance(compliance_events, regulation),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return report
    
    def _summarize_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize audit events"""
        summary = {
            'actions': {},
            'users': set(),
            'resources': set(),
            'success_rate': 0
        }
        
        successful_events = 0
        
        for event in events:
            # Count actions
            action = event.get('action', 'unknown')
            summary['actions'][action] = summary['actions'].get(action, 0) + 1
            
            # Track users and resources
            summary['users'].add(event.get('user_id', 'unknown'))
            summary['resources'].add(event.get('resource', 'unknown'))
            
            # Count successful events
            if event.get('success', False):
                successful_events += 1
        
        # Calculate success rate
        if events:
            summary['success_rate'] = successful_events / len(events)
        
        # Convert sets to lists for JSON serialization
        summary['users'] = list(summary['users'])
        summary['resources'] = list(summary['resources'])
        
        return summary
    
    def _assess_compliance(self, events: List[Dict[str, Any]], regulation: str) -> Dict[str, Any]:
        """Assess compliance status"""
        rules = self.compliance_rules[regulation]
        violations = []
        
        for event in events:
            # Check required fields
            for required_field in rules.get('required_fields', []):
                if required_field not in event or not event[required_field]:
                    violations.append(f"Missing {required_field} in event {event.get('event_id')}")
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'violation_count': len(violations)
        }

class SecurityFramework:
    """Main security framework orchestrator"""
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.mfa = MultiFactorAuth()
        self.rbac = RoleBasedAccessControl()
        self.audit_logger = AuditLogger(self.encryption_manager)
        self.active_sessions = {}
        self.failed_attempts = {}
        self.jwt_secret = self._get_jwt_secret()
    
    def _get_jwt_secret(self) -> str:
        """Get or create JWT secret"""
        secret_file = "security/keys/jwt_secret.key"
        os.makedirs(os.path.dirname(secret_file), exist_ok=True)
        
        if os.path.exists(secret_file):
            with open(secret_file, 'r') as f:
                return f.read().strip()
        else:
            secret = secrets.token_hex(32)
            with open(secret_file, 'w') as f:
                f.write(secret)
            return secret
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        mfa_token: str = None,
        ip_address: str = None,
        user_agent: str = None
    ) -> Optional[SecurityContext]:
        """Authenticate user with optional MFA"""
        
        # Check for too many failed attempts
        if self._is_rate_limited(username, ip_address):
            self.audit_logger.log_event(
                user_id=username,
                action="authentication_blocked",
                resource="auth_system",
                success=False,
                details={"reason": "rate_limited"},
                ip_address=ip_address,
                user_agent=user_agent
            )
            return None
        
        # Verify password (in production, check against database)
        if not self._verify_user_credentials(username, password):
            self._record_failed_attempt(username, ip_address)
            self.audit_logger.log_event(
                user_id=username,
                action="authentication_failed",
                resource="auth_system",
                success=False,
                details={"reason": "invalid_credentials"},
                ip_address=ip_address,
                user_agent=user_agent
            )
            return None
        
        # Check MFA if required
        if self._requires_mfa(username):
            if not mfa_token or not self.mfa.verify_totp(username, mfa_token):
                self.audit_logger.log_event(
                    user_id=username,
                    action="mfa_failed",
                    resource="auth_system",
                    success=False,
                    details={"reason": "invalid_mfa_token"},
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                return None
        
        # Create security context
        session_id = secrets.token_hex(32)
        permissions = list(self.rbac.get_user_permissions(username))
        
        security_context = SecurityContext(
            user_id=username,
            session_id=session_id,
            permissions=permissions,
            security_level=SecurityLevel.INTERNAL,
            authentication_methods=[AuthenticationMethod.PASSWORD, AuthenticationMethod.MFA_TOTP],
            ip_address=ip_address or "unknown",
            user_agent=user_agent or "unknown",
            timestamp=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=8)
        )
        
        # Store active session
        self.active_sessions[session_id] = security_context
        
        # Clear failed attempts
        self._clear_failed_attempts(username, ip_address)
        
        # Log successful authentication
        self.audit_logger.log_event(
            user_id=username,
            action="authentication_success",
            resource="auth_system",
            success=True,
            details={"session_id": session_id, "permissions": permissions},
            ip_address=ip_address,
            user_agent=user_agent,
            compliance_tags=["gdpr", "sox"]
        )
        
        return security_context
    
    def _verify_user_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials (mock implementation)"""
        # In production, check against encrypted database
        test_users = {
            "admin": ("admin123", "salt123"),
            "user": ("user123", "salt456"),
            "analyst": ("analyst123", "salt789")
        }
        
        if username in test_users:
            stored_password, salt = test_users[username]
            return self.encryption_manager.verify_password(password, stored_password, salt)
        
        return False
    
    def _requires_mfa(self, username: str) -> bool:
        """Check if user requires MFA"""
        # In production, check user settings
        admin_users = ["admin", "auditor"]
        return username in admin_users
    
    def _is_rate_limited(self, username: str, ip_address: str) -> bool:
        """Check if user/IP is rate limited"""
        now = datetime.utcnow()
        
        # Check username attempts
        if username in self.failed_attempts:
            attempts = self.failed_attempts[username]
            recent_attempts = [
                attempt for attempt in attempts
                if now - attempt['timestamp'] < timedelta(minutes=15)
            ]
            if len(recent_attempts) >= 5:
                return True
        
        return False
    
    def _record_failed_attempt(self, username: str, ip_address: str):
        """Record failed authentication attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append({
            'timestamp': datetime.utcnow(),
            'ip_address': ip_address
        })
    
    def _clear_failed_attempts(self, username: str, ip_address: str):
        """Clear failed attempts for user"""
        if username in self.failed_attempts:
            del self.failed_attempts[username]
    
    def generate_jwt_token(self, security_context: SecurityContext) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            'user_id': security_context.user_id,
            'session_id': security_context.session_id,
            'permissions': security_context.permissions,
            'iat': int(security_context.timestamp.timestamp()),
            'exp': int(security_context.expires_at.timestamp())
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[SecurityContext]:
        """Verify JWT token and return security context"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            session_id = payload['session_id']
            
            if session_id in self.active_sessions:
                context = self.active_sessions[session_id]
                if datetime.utcnow() < context.expires_at:
                    return context
                else:
                    # Session expired
                    del self.active_sessions[session_id]
            
            return None
        except jwt.InvalidTokenError:
            return None
    
    def check_permission(self, security_context: SecurityContext, permission: str) -> bool:
        """Check if user has permission"""
        return permission in security_context.permissions
    
    def logout_user(self, session_id: str, ip_address: str = None, user_agent: str = None):
        """Logout user and invalidate session"""
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            
            self.audit_logger.log_event(
                user_id=context.user_id,
                action="logout",
                resource="auth_system",
                success=True,
                details={"session_id": session_id},
                ip_address=ip_address,
                user_agent=user_agent
            )
    
    def encrypt_sensitive_data(self, data: Any, security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> str:
        """Encrypt sensitive data"""
        if isinstance(data, dict):
            data = json.dumps(data)
        elif not isinstance(data, str):
            data = str(data)
        
        return self.encryption_manager.encrypt_data(data, security_level)
    
    def decrypt_sensitive_data(self, encrypted_data: str, security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> str:
        """Decrypt sensitive data"""
        decrypted_bytes = self.encryption_manager.decrypt_data(encrypted_data, security_level)
        return decrypted_bytes.decode('utf-8')
    
    def get_compliance_report(self, regulation: str, days: int = 30) -> Dict[str, Any]:
        """Get compliance report for regulation"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        return self.audit_logger.generate_compliance_report(regulation, start_date, end_date)

# Global security framework instance
security_framework = SecurityFramework()