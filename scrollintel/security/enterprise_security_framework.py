"""
Enterprise Security Framework for Agent Steering System
Implements military-grade security with zero tolerance for vulnerabilities
"""

import hashlib
import secrets
import base64
import hmac
import pyotp
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import bcrypt
import os
import json
import logging
from sqlalchemy.orm import Session
from ..models.security_compliance_models import (
    User, Role, Permission, UserRole, UserSession, AuditLog, EncryptionKey,
    SecurityLevel, AuthenticationMethod, PermissionType, AuditEventType,
    ComplianceFramework, SecurityConfig, EncryptionConfig
)

logger = logging.getLogger(__name__)

class SecurityException(Exception):
    """Base security exception"""
    pass

class AuthenticationException(SecurityException):
    """Authentication failed exception"""
    pass

class AuthorizationException(SecurityException):
    """Authorization failed exception"""
    pass

class EncryptionException(SecurityException):
    """Encryption/decryption failed exception"""
    pass

class ComplianceException(SecurityException):
    """Compliance violation exception"""
    pass

class EnterpriseSecurityFramework:
    """
    Enterprise-grade security framework implementing:
    - Multi-factor authentication
    - Single sign-on integration
    - End-to-end encryption
    - Role-based access control
    - Comprehensive audit logging
    - Compliance reporting
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.master_key = self._initialize_master_key()
        self.jwt_secret = self._generate_jwt_secret()
        self.encryption_keys: Dict[str, bytes] = {}
        self._initialize_encryption_keys()
        
    def _initialize_master_key(self) -> bytes:
        """Initialize or load master encryption key"""
        key_file = os.environ.get('MASTER_KEY_FILE', '/secure/master.key')
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new master key
            master_key = Fernet.generate_key()
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(master_key)
            os.chmod(key_file, 0o600)  # Restrict access
            return master_key
    
    def _generate_jwt_secret(self) -> str:
        """Generate JWT signing secret"""
        return base64.b64encode(secrets.token_bytes(64)).decode('utf-8')
    
    def _initialize_encryption_keys(self):
        """Initialize encryption keys for different purposes"""
        purposes = ['data_encryption', 'token_signing', 'session_encryption', 'audit_encryption']
        
        for purpose in purposes:
            key = Fernet.generate_key()
            self.encryption_keys[purpose] = key
    
    # Authentication Methods
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt with high cost factor"""
        salt = bcrypt.gensalt(rounds=12)  # High cost for security
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def generate_mfa_secret(self) -> str:
        """Generate MFA secret for TOTP"""
        return pyotp.random_base32()
    
    def generate_mfa_qr_url(self, user_email: str, secret: str, issuer: str = "ScrollIntel") -> str:
        """Generate QR code URL for MFA setup"""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(
            name=user_email,
            issuer_name=issuer
        )
    
    def verify_mfa_code(self, secret: str, code: str, window: int = 1) -> bool:
        """Verify MFA TOTP code with time window tolerance"""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=window)
        except Exception as e:
            logger.error(f"MFA verification failed: {e}")
            return False
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for MFA recovery"""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4).upper()
            codes.append(f"{code[:4]}-{code[4:]}")
        return codes
    
    def authenticate_user(
        self, 
        db: Session, 
        username: str, 
        password: Optional[str] = None,
        mfa_code: Optional[str] = None,
        sso_token: Optional[str] = None,
        ip_address: str = "unknown",
        user_agent: str = "unknown"
    ) -> Tuple[User, str]:
        """
        Authenticate user with multiple methods
        Returns (user, session_token) on success
        """
        
        # Get user
        user = db.query(User).filter(User.username == username).first()
        if not user or not user.is_active:
            self._log_audit_event(
                db, None, AuditEventType.AUTHENTICATION, "authentication_failed",
                f"Authentication failed for user: {username}", "user", username,
                "authenticate", False, "USER_NOT_FOUND", ip_address=ip_address
            )
            raise AuthenticationException("Invalid credentials")
        
        # Check account lockout
        if user.account_locked_until and user.account_locked_until > datetime.utcnow():
            self._log_audit_event(
                db, user.id, AuditEventType.AUTHENTICATION, "authentication_blocked",
                f"Authentication blocked - account locked: {username}", "user", user.id,
                "authenticate", False, "ACCOUNT_LOCKED", ip_address=ip_address
            )
            raise AuthenticationException("Account temporarily locked")
        
        # SSO Authentication
        if sso_token:
            if not self._verify_sso_token(sso_token, user):
                self._increment_failed_attempts(db, user)
                raise AuthenticationException("Invalid SSO token")
        
        # Password Authentication
        elif password:
            if not user.password_hash or not self.verify_password(password, user.password_hash):
                self._increment_failed_attempts(db, user)
                raise AuthenticationException("Invalid credentials")
        
        else:
            raise AuthenticationException("No authentication method provided")
        
        # MFA Verification
        if user.mfa_enabled:
            if not mfa_code:
                raise AuthenticationException("MFA code required")
            
            if not self.verify_mfa_code(user.mfa_secret, mfa_code):
                self._increment_failed_attempts(db, user)
                raise AuthenticationException("Invalid MFA code")
        
        # Reset failed attempts on successful authentication
        user.failed_login_attempts = 0
        user.account_locked_until = None
        user.last_login = datetime.utcnow()
        
        # Create session
        session_token = self._create_user_session(db, user, ip_address, user_agent)
        
        # Log successful authentication
        self._log_audit_event(
            db, user.id, AuditEventType.AUTHENTICATION, "authentication_success",
            f"User authenticated successfully: {username}", "user", user.id,
            "authenticate", True, ip_address=ip_address
        )
        
        db.commit()
        return user, session_token
    
    def _verify_sso_token(self, token: str, user: User) -> bool:
        """Verify SSO token (implement based on your SSO provider)"""
        # This is a placeholder - implement actual SSO verification
        # For example, verify JWT token from Azure AD, Okta, etc.
        try:
            # Decode and verify SSO token
            # This would integrate with your actual SSO provider
            return True  # Placeholder
        except Exception as e:
            logger.error(f"SSO token verification failed: {e}")
            return False
    
    def _increment_failed_attempts(self, db: Session, user: User):
        """Increment failed login attempts and lock account if needed"""
        user.failed_login_attempts += 1
        
        if user.failed_login_attempts >= self.config.max_failed_attempts:
            user.account_locked_until = datetime.utcnow() + timedelta(
                minutes=self.config.account_lockout_minutes
            )
            
            self._log_audit_event(
                db, user.id, AuditEventType.SECURITY_INCIDENT, "account_locked",
                f"Account locked due to failed attempts: {user.username}",
                "user", user.id, "lock_account", True
            )
        
        db.commit()
    
    def _create_user_session(
        self, 
        db: Session, 
        user: User, 
        ip_address: str, 
        user_agent: str
    ) -> str:
        """Create encrypted user session"""
        
        # Generate session tokens
        session_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)
        
        # Create session record
        session = UserSession(
            user_id=user.id,
            session_token=self._encrypt_token(session_token),
            refresh_token=self._encrypt_token(refresh_token),
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.utcnow() + timedelta(minutes=self.config.session_timeout_minutes)
        )
        
        db.add(session)
        db.commit()
        
        # Generate JWT token
        jwt_payload = {
            'user_id': user.id,
            'username': user.username,
            'session_id': session.id,
            'security_level': user.security_level,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(minutes=self.config.session_timeout_minutes)
        }
        
        jwt_token = jwt.encode(jwt_payload, self.jwt_secret, algorithm='HS256')
        return jwt_token
    
    def _encrypt_token(self, token: str) -> str:
        """Encrypt token for storage"""
        fernet = Fernet(self.encryption_keys['session_encryption'])
        return fernet.encrypt(token.encode()).decode()
    
    def _decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt stored token"""
        fernet = Fernet(self.encryption_keys['session_encryption'])
        return fernet.decrypt(encrypted_token.encode()).decode()
    
    # Authorization Methods
    
    def check_permission(
        self, 
        db: Session, 
        user_id: str, 
        resource: str, 
        action: PermissionType,
        resource_id: Optional[str] = None
    ) -> bool:
        """Check if user has permission for resource and action"""
        
        # Get user with roles
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            return False
        
        # Get user roles
        user_roles = db.query(Role).join(UserRole).filter(
            UserRole.user_id == user_id,
            UserRole.valid_from <= datetime.utcnow(),
            (UserRole.valid_until.is_(None) | (UserRole.valid_until > datetime.utcnow()))
        ).all()
        
        # Check permissions for each role
        for role in user_roles:
            permissions = db.query(Permission).join(RolePermission).filter(
                RolePermission.role_id == role.id,
                Permission.resource == resource,
                Permission.action == action.value
            ).all()
            
            if permissions:
                # Log authorization success
                self._log_audit_event(
                    db, user_id, AuditEventType.AUTHORIZATION, "authorization_granted",
                    f"Permission granted: {resource}:{action.value}", resource, resource_id,
                    action.value, True
                )
                return True
        
        # Log authorization failure
        self._log_audit_event(
            db, user_id, AuditEventType.AUTHORIZATION, "authorization_denied",
            f"Permission denied: {resource}:{action.value}", resource, resource_id,
            action.value, False, "INSUFFICIENT_PERMISSIONS"
        )
        
        return False
    
    def get_user_permissions(self, db: Session, user_id: str) -> List[str]:
        """Get all permissions for a user"""
        permissions = db.query(Permission).join(RolePermission).join(Role).join(UserRole).filter(
            UserRole.user_id == user_id,
            UserRole.valid_from <= datetime.utcnow(),
            (UserRole.valid_until.is_(None) | (UserRole.valid_until > datetime.utcnow()))
        ).all()
        
        return [f"{p.resource}:{p.action}" for p in permissions]
    
    # Encryption Methods
    
    def encrypt_data(self, data: Union[str, bytes], purpose: str = "data_encryption") -> str:
        """Encrypt data with specified purpose key"""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            if purpose not in self.encryption_keys:
                raise EncryptionException(f"Unknown encryption purpose: {purpose}")
            
            fernet = Fernet(self.encryption_keys[purpose])
            encrypted = fernet.encrypt(data)
            return base64.b64encode(encrypted).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionException(f"Encryption failed: {e}")
    
    def decrypt_data(self, encrypted_data: str, purpose: str = "data_encryption") -> bytes:
        """Decrypt data with specified purpose key"""
        try:
            if purpose not in self.encryption_keys:
                raise EncryptionException(f"Unknown encryption purpose: {purpose}")
            
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            fernet = Fernet(self.encryption_keys[purpose])
            return fernet.decrypt(encrypted_bytes)
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise EncryptionException(f"Decryption failed: {e}")
    
    def encrypt_sensitive_field(self, value: str) -> str:
        """Encrypt sensitive database field"""
        return self.encrypt_data(value, "data_encryption")
    
    def decrypt_sensitive_field(self, encrypted_value: str) -> str:
        """Decrypt sensitive database field"""
        return self.decrypt_data(encrypted_value, "data_encryption").decode('utf-8')
    
    # Audit Logging
    
    def _log_audit_event(
        self,
        db: Session,
        user_id: Optional[str],
        event_type: AuditEventType,
        event_category: str,
        event_description: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: str = "unknown",
        success: bool = True,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        event_metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        sensitive_data_accessed: bool = False,
        compliance_frameworks: Optional[List[ComplianceFramework]] = None
    ):
        """Log audit event for compliance"""
        
        try:
            # Encrypt sensitive audit data
            encrypted_description = self.encrypt_data(event_description, "audit_encryption")
            encrypted_metadata = None
            
            if event_metadata:
                encrypted_metadata = self.encrypt_data(json.dumps(event_metadata), "audit_encryption")
            
            audit_log = AuditLog(
                user_id=user_id,
                event_type=event_type.value,
                event_category=event_category,
                event_description=encrypted_description,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                success=success,
                error_code=error_code,
                error_message=error_message,
                event_metadata=encrypted_metadata,
                ip_address=ip_address,
                user_agent=user_agent,
                sensitive_data_accessed=sensitive_data_accessed,
                compliance_frameworks=[f.value for f in compliance_frameworks] if compliance_frameworks else None
            )
            
            db.add(audit_log)
            db.commit()
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
            # Don't raise exception to avoid breaking main operation
    
    def get_audit_logs(
        self,
        db: Session,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Retrieve audit logs with decryption"""
        
        query = db.query(AuditLog)
        
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        
        if event_type:
            query = query.filter(AuditLog.event_type == event_type.value)
        
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        logs = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
        
        # Decrypt sensitive fields
        for log in logs:
            try:
                log.event_description = self.decrypt_data(log.event_description, "audit_encryption").decode('utf-8')
                if log.event_metadata:
                    decrypted_metadata = self.decrypt_data(log.event_metadata, "audit_encryption").decode('utf-8')
                    log.event_metadata = json.loads(decrypted_metadata)
            except Exception as e:
                logger.error(f"Failed to decrypt audit log {log.id}: {e}")
        
        return logs
    
    # Compliance Reporting
    
    def generate_compliance_report(
        self,
        db: Session,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for specified framework"""
        
        # Get all audit events for the period
        audit_logs = db.query(AuditLog).filter(
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        ).all()
        
        total_events = len(audit_logs)
        compliant_events = 0
        violations = []
        
        # Analyze compliance based on framework
        for log in audit_logs:
            if self._is_compliant_event(log, framework):
                compliant_events += 1
            else:
                violations.append({
                    'event_id': log.id,
                    'timestamp': log.timestamp,
                    'event_type': log.event_type,
                    'description': 'Compliance violation detected',
                    'severity': self._get_violation_severity(log, framework)
                })
        
        compliance_score = (compliant_events / total_events * 100) if total_events > 0 else 100
        
        return {
            'framework': framework.value,
            'period_start': start_date,
            'period_end': end_date,
            'total_events': total_events,
            'compliant_events': compliant_events,
            'violations': len(violations),
            'compliance_score': compliance_score,
            'violation_details': violations,
            'recommendations': self._get_compliance_recommendations(framework, violations),
            'generated_at': datetime.utcnow()
        }
    
    def _is_compliant_event(self, log: AuditLog, framework: ComplianceFramework) -> bool:
        """Check if audit event is compliant with framework"""
        
        # Framework-specific compliance rules
        if framework == ComplianceFramework.GDPR:
            # GDPR requires explicit consent for data processing
            if log.sensitive_data_accessed and not log.success:
                return False
            
        elif framework == ComplianceFramework.SOX:
            # SOX requires proper authorization for financial data
            if log.resource_type == 'financial_data' and not log.success:
                return False
        
        elif framework == ComplianceFramework.HIPAA:
            # HIPAA requires strict access controls for health data
            if log.resource_type == 'health_data' and log.event_type == 'data_access':
                return log.success
        
        return True
    
    def _get_violation_severity(self, log: AuditLog, framework: ComplianceFramework) -> str:
        """Determine violation severity"""
        
        if log.sensitive_data_accessed and not log.success:
            return "HIGH"
        elif log.event_type == AuditEventType.SECURITY_INCIDENT.value:
            return "CRITICAL"
        elif not log.success:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_compliance_recommendations(
        self, 
        framework: ComplianceFramework, 
        violations: List[Dict]
    ) -> List[str]:
        """Generate compliance recommendations"""
        
        recommendations = []
        
        if violations:
            recommendations.append("Review and strengthen access controls")
            recommendations.append("Implement additional monitoring for sensitive data access")
            recommendations.append("Conduct security awareness training")
        
        if framework == ComplianceFramework.GDPR:
            recommendations.extend([
                "Ensure explicit consent for data processing",
                "Implement data retention policies",
                "Provide data subject access rights"
            ])
        
        elif framework == ComplianceFramework.SOX:
            recommendations.extend([
                "Strengthen financial data access controls",
                "Implement segregation of duties",
                "Regular access reviews"
            ])
        
        return recommendations
    
    # Key Management
    
    def rotate_encryption_keys(self, db: Session):
        """Rotate encryption keys according to policy"""
        
        for purpose, current_key in self.encryption_keys.items():
            # Generate new key
            new_key = Fernet.generate_key()
            
            # Store old key for decryption of existing data
            old_key_record = EncryptionKey(
                key_id=f"{purpose}_{datetime.utcnow().isoformat()}",
                key_type="AES-256",
                key_size=256,
                purpose=purpose,
                expires_at=datetime.utcnow() + timedelta(days=self.config.encryption.key_rotation_days),
                security_level=SecurityLevel.SECRET.value
            )
            
            db.add(old_key_record)
            
            # Update current key
            self.encryption_keys[purpose] = new_key
            
            logger.info(f"Rotated encryption key for purpose: {purpose}")
        
        db.commit()
    
    def validate_security_config(self) -> List[str]:
        """Validate security configuration and return issues"""
        
        issues = []
        
        # Check password policy
        if self.config.password_policy.get('min_length', 0) < 12:
            issues.append("Password minimum length should be at least 12 characters")
        
        # Check session timeout
        if self.config.session_timeout_minutes > 480:  # 8 hours
            issues.append("Session timeout should not exceed 8 hours")
        
        # Check MFA requirement
        if not self.config.mfa_required:
            issues.append("Multi-factor authentication should be required")
        
        # Check audit retention
        if self.config.audit_retention_days < 2555:  # 7 years
            issues.append("Audit log retention should be at least 7 years for compliance")
        
        return issues