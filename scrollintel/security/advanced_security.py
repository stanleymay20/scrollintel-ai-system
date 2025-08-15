"""
ScrollIntel Advanced Security Framework
Implements enterprise-grade security features and threat detection.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis
import ipaddress
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityEventType(Enum):
    AUTHENTICATION_FAILURE = "auth_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    MALICIOUS_REQUEST = "malicious_request"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"

@dataclass
class SecurityEvent:
    event_type: SecurityEventType
    threat_level: ThreatLevel
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime
    details: Dict[str, Any]
    action_taken: str

@dataclass
class SecurityPolicy:
    name: str
    rules: List[Dict[str, Any]]
    enforcement_level: str
    exceptions: List[str]
    created_at: datetime
    updated_at: datetime

class AdvancedThreatDetection:
    """Advanced threat detection and prevention system"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.threat_patterns = self._load_threat_patterns()
        self.ip_whitelist = set()
        self.ip_blacklist = set()
        
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load threat detection patterns"""
        return {
            'sql_injection': [
                r"(\%27)|(\')|(\-\-)|(\%23)|(#)",
                r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
                r"\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))",
                r"((\%27)|(\'))union"
            ],
            'xss_injection': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>.*?</iframe>"
            ],
            'command_injection': [
                r"[;&|`]",
                r"\$\([^)]*\)",
                r"`[^`]*`",
                r"\|\s*\w+"
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c"
            ]
        }
    
    async def analyze_request(
        self, 
        request_data: Dict[str, Any], 
        user_id: Optional[str] = None
    ) -> Tuple[ThreatLevel, List[str]]:
        """Analyze request for potential threats"""
        threats_detected = []
        max_threat_level = ThreatLevel.LOW
        
        # Check IP address
        ip_address = request_data.get('ip_address', '')
        if await self._is_suspicious_ip(ip_address):
            threats_detected.append("Suspicious IP address")
            max_threat_level = ThreatLevel.HIGH
        
        # Check for malicious patterns
        request_content = json.dumps(request_data)
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, request_content, re.IGNORECASE):
                    threats_detected.append(f"{threat_type.replace('_', ' ').title()} detected")
                    max_threat_level = max(max_threat_level, ThreatLevel.HIGH, key=lambda x: x.value)
        
        # Check rate limiting
        if await self._check_rate_limiting(ip_address, user_id):
            threats_detected.append("Rate limit exceeded")
            max_threat_level = max(max_threat_level, ThreatLevel.MEDIUM, key=lambda x: x.value)
        
        # Check for privilege escalation attempts
        if await self._detect_privilege_escalation(request_data, user_id):
            threats_detected.append("Privilege escalation attempt")
            max_threat_level = ThreatLevel.CRITICAL
        
        return max_threat_level, threats_detected
    
    async def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is suspicious"""
        if not ip_address:
            return True
        
        # Check blacklist
        if ip_address in self.ip_blacklist:
            return True
        
        # Check whitelist
        if ip_address in self.ip_whitelist:
            return False
        
        # Check for recent suspicious activity
        suspicious_key = f"suspicious_ip:{ip_address}"
        suspicious_count = await self.redis_client.get(suspicious_key)
        
        if suspicious_count and int(suspicious_count) > 5:
            return True
        
        # Check geolocation and known threat feeds (placeholder)
        # In production, integrate with threat intelligence feeds
        
        return False
    
    async def _check_rate_limiting(self, ip_address: str, user_id: Optional[str]) -> bool:
        """Check if rate limits are exceeded"""
        current_time = int(time.time())
        window_size = 60  # 1 minute window
        
        # IP-based rate limiting
        ip_key = f"rate_limit:ip:{ip_address}:{current_time // window_size}"
        ip_count = await self.redis_client.incr(ip_key)
        await self.redis_client.expire(ip_key, window_size)
        
        if ip_count > 100:  # 100 requests per minute per IP
            return True
        
        # User-based rate limiting
        if user_id:
            user_key = f"rate_limit:user:{user_id}:{current_time // window_size}"
            user_count = await self.redis_client.incr(user_key)
            await self.redis_client.expire(user_key, window_size)
            
            if user_count > 1000:  # 1000 requests per minute per user
                return True
        
        return False
    
    async def _detect_privilege_escalation(
        self, 
        request_data: Dict[str, Any], 
        user_id: Optional[str]
    ) -> bool:
        """Detect privilege escalation attempts"""
        if not user_id:
            return False
        
        # Check for admin endpoint access by non-admin users
        endpoint = request_data.get('endpoint', '')
        if '/admin/' in endpoint or '/api/admin/' in endpoint:
            # Check if user has admin privileges (placeholder)
            # In production, check actual user permissions
            return True
        
        # Check for role manipulation attempts
        request_body = request_data.get('body', {})
        if isinstance(request_body, dict):
            suspicious_fields = ['role', 'permissions', 'admin', 'superuser']
            for field in suspicious_fields:
                if field in request_body:
                    return True
        
        return False

class DataEncryptionManager:
    """Advanced data encryption and key management"""
    
    def __init__(self, master_key: bytes):
        self.master_key = master_key
        self.encryption_keys = {}
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption system"""
        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'scrollintel_salt',
            iterations=100000,
        )
        key = kdf.derive(self.master_key)
        self.fernet = Fernet(Fernet.generate_key())
    
    def encrypt_sensitive_data(self, data: str, context: str = "default") -> str:
        """Encrypt sensitive data with context-specific keys"""
        if context not in self.encryption_keys:
            self.encryption_keys[context] = Fernet.generate_key()
        
        fernet = Fernet(self.encryption_keys[context])
        encrypted_data = fernet.encrypt(data.encode())
        return encrypted_data.decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str, context: str = "default") -> str:
        """Decrypt sensitive data"""
        if context not in self.encryption_keys:
            raise ValueError(f"No encryption key found for context: {context}")
        
        fernet = Fernet(self.encryption_keys[context])
        decrypted_data = fernet.decrypt(encrypted_data.encode())
        return decrypted_data.decode()
    
    def rotate_encryption_keys(self, context: str = "default"):
        """Rotate encryption keys for enhanced security"""
        old_key = self.encryption_keys.get(context)
        new_key = Fernet.generate_key()
        
        # Store both keys temporarily for data migration
        self.encryption_keys[f"{context}_old"] = old_key
        self.encryption_keys[context] = new_key
        
        logger.info(f"Encryption key rotated for context: {context}")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token"""
        import secrets
        return secrets.token_urlsafe(length)

class AccessControlManager:
    """Advanced access control and permission management"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.permission_cache = {}
        self.role_hierarchy = {
            'admin': ['user', 'analyst', 'developer'],
            'analyst': ['user'],
            'developer': ['user'],
            'user': []
        }
    
    async def check_permission(
        self, 
        user_id: str, 
        resource: str, 
        action: str
    ) -> bool:
        """Check if user has permission for specific resource and action"""
        # Check cache first
        cache_key = f"permission:{user_id}:{resource}:{action}"
        cached_result = self.permission_cache.get(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        # Get user roles
        user_roles = await self._get_user_roles(user_id)
        
        # Check permissions for each role
        has_permission = False
        for role in user_roles:
            if await self._role_has_permission(role, resource, action):
                has_permission = True
                break
        
        # Cache result
        self.permission_cache[cache_key] = has_permission
        
        return has_permission
    
    async def _get_user_roles(self, user_id: str) -> List[str]:
        """Get user roles from database/cache"""
        roles_key = f"user_roles:{user_id}"
        cached_roles = await self.redis_client.get(roles_key)
        
        if cached_roles:
            return json.loads(cached_roles)
        
        # In production, fetch from database
        # For now, return default role
        default_roles = ['user']
        
        # Cache roles
        await self.redis_client.setex(roles_key, 3600, json.dumps(default_roles))
        
        return default_roles
    
    async def _role_has_permission(self, role: str, resource: str, action: str) -> bool:
        """Check if role has specific permission"""
        permission_key = f"role_permission:{role}:{resource}:{action}"
        cached_permission = await self.redis_client.get(permission_key)
        
        if cached_permission:
            return cached_permission.decode() == 'true'
        
        # Default permissions (in production, load from database)
        default_permissions = {
            'admin': True,
            'analyst': resource in ['dashboard', 'reports', 'data'],
            'developer': resource in ['api', 'models', 'code'],
            'user': resource in ['dashboard', 'basic_reports'] and action == 'read'
        }
        
        has_permission = default_permissions.get(role, False)
        
        # Cache permission
        await self.redis_client.setex(
            permission_key, 
            3600, 
            'true' if has_permission else 'false'
        )
        
        return has_permission
    
    def create_security_policy(
        self, 
        name: str, 
        rules: List[Dict[str, Any]]
    ) -> SecurityPolicy:
        """Create new security policy"""
        policy = SecurityPolicy(
            name=name,
            rules=rules,
            enforcement_level="strict",
            exceptions=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        return policy

class SecurityAuditLogger:
    """Comprehensive security audit logging"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.audit_queue = asyncio.Queue()
        
    async def log_security_event(self, event: SecurityEvent):
        """Log security event with full context"""
        event_data = {
            'event_type': event.event_type.value,
            'threat_level': event.threat_level.value,
            'user_id': event.user_id,
            'ip_address': event.ip_address,
            'user_agent': event.user_agent,
            'timestamp': event.timestamp.isoformat(),
            'details': event.details,
            'action_taken': event.action_taken
        }
        
        # Store in Redis for real-time monitoring
        await self.redis_client.lpush(
            "security_events",
            json.dumps(event_data)
        )
        
        # Keep only last 10000 events
        await self.redis_client.ltrim("security_events", 0, 9999)
        
        # Log to file for permanent storage
        logger.warning(f"Security Event: {event.event_type.value} - {event.threat_level.value}")
        
        # Trigger alerts for high-severity events
        if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            await self._trigger_security_alert(event)
    
    async def _trigger_security_alert(self, event: SecurityEvent):
        """Trigger security alert for high-severity events"""
        alert_data = {
            'alert_type': 'security_threat',
            'severity': event.threat_level.value,
            'event_type': event.event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'details': event.details
        }
        
        # Store alert
        await self.redis_client.lpush(
            "security_alerts",
            json.dumps(alert_data)
        )
        
        # In production, send notifications (email, Slack, etc.)
        logger.critical(f"SECURITY ALERT: {event.event_type.value} - {event.threat_level.value}")
    
    async def get_security_events(
        self, 
        limit: int = 100, 
        event_type: Optional[SecurityEventType] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve security events for analysis"""
        events = await self.redis_client.lrange("security_events", 0, limit - 1)
        
        parsed_events = []
        for event_data in events:
            event = json.loads(event_data)
            
            # Filter by event type if specified
            if event_type and event['event_type'] != event_type.value:
                continue
            
            parsed_events.append(event)
        
        return parsed_events

class SecurityMiddleware:
    """Security middleware for request processing"""
    
    def __init__(
        self, 
        threat_detector: AdvancedThreatDetection,
        access_control: AccessControlManager,
        audit_logger: SecurityAuditLogger
    ):
        self.threat_detector = threat_detector
        self.access_control = access_control
        self.audit_logger = audit_logger
    
    async def process_request(
        self, 
        request_data: Dict[str, Any], 
        user_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Process request through security pipeline"""
        
        # Threat detection
        threat_level, threats = await self.threat_detector.analyze_request(
            request_data, user_id
        )
        
        # Block high-threat requests
        if threat_level == ThreatLevel.CRITICAL:
            await self._log_blocked_request(request_data, user_id, threats)
            return False, "Request blocked due to security threat"
        
        # Access control check
        resource = request_data.get('resource', 'unknown')
        action = request_data.get('action', 'unknown')
        
        if user_id:
            has_access = await self.access_control.check_permission(
                user_id, resource, action
            )
            
            if not has_access:
                await self._log_access_denied(request_data, user_id)
                return False, "Access denied"
        
        # Log successful request
        if threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH]:
            await self._log_suspicious_request(request_data, user_id, threats)
        
        return True, "Request approved"
    
    async def _log_blocked_request(
        self, 
        request_data: Dict[str, Any], 
        user_id: Optional[str], 
        threats: List[str]
    ):
        """Log blocked request"""
        event = SecurityEvent(
            event_type=SecurityEventType.MALICIOUS_REQUEST,
            threat_level=ThreatLevel.CRITICAL,
            user_id=user_id,
            ip_address=request_data.get('ip_address', ''),
            user_agent=request_data.get('user_agent', ''),
            timestamp=datetime.utcnow(),
            details={'threats': threats, 'request': request_data},
            action_taken="Request blocked"
        )
        
        await self.audit_logger.log_security_event(event)
    
    async def _log_access_denied(
        self, 
        request_data: Dict[str, Any], 
        user_id: str
    ):
        """Log access denied event"""
        event = SecurityEvent(
            event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
            threat_level=ThreatLevel.MEDIUM,
            user_id=user_id,
            ip_address=request_data.get('ip_address', ''),
            user_agent=request_data.get('user_agent', ''),
            timestamp=datetime.utcnow(),
            details={'resource': request_data.get('resource'), 'action': request_data.get('action')},
            action_taken="Access denied"
        )
        
        await self.audit_logger.log_security_event(event)
    
    async def _log_suspicious_request(
        self, 
        request_data: Dict[str, Any], 
        user_id: Optional[str], 
        threats: List[str]
    ):
        """Log suspicious request"""
        event = SecurityEvent(
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            threat_level=ThreatLevel.MEDIUM,
            user_id=user_id,
            ip_address=request_data.get('ip_address', ''),
            user_agent=request_data.get('user_agent', ''),
            timestamp=datetime.utcnow(),
            details={'threats': threats},
            action_taken="Request monitored"
        )
        
        await self.audit_logger.log_security_event(event)

class ScrollIntelAdvancedSecurity:
    """Main advanced security system coordinator"""
    
    def __init__(self, redis_client: redis.Redis, master_key: bytes):
        self.threat_detector = AdvancedThreatDetection(redis_client)
        self.encryption_manager = DataEncryptionManager(master_key)
        self.access_control = AccessControlManager(redis_client)
        self.audit_logger = SecurityAuditLogger(redis_client)
        self.security_middleware = SecurityMiddleware(
            self.threat_detector,
            self.access_control,
            self.audit_logger
        )
        
    async def initialize_security_system(self):
        """Initialize advanced security system"""
        logger.info("Initializing ScrollIntel advanced security system")
        
        # Start security monitoring
        asyncio.create_task(self._continuous_security_monitoring())
        
        logger.info("Advanced security system initialized successfully")
    
    async def _continuous_security_monitoring(self):
        """Continuous security monitoring loop"""
        while True:
            try:
                # Monitor for security events
                events = await self.audit_logger.get_security_events(limit=50)
                
                # Analyze patterns and trends
                await self._analyze_security_patterns(events)
                
                # Update threat intelligence
                await self._update_threat_intelligence()
                
                # Wait before next monitoring cycle
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_security_patterns(self, events: List[Dict[str, Any]]):
        """Analyze security event patterns"""
        if not events:
            return
        
        # Count events by type
        event_counts = {}
        for event in events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Check for unusual patterns
        for event_type, count in event_counts.items():
            if count > 10:  # Threshold for unusual activity
                logger.warning(f"Unusual security activity detected: {event_type} ({count} events)")
    
    async def _update_threat_intelligence(self):
        """Update threat intelligence data"""
        # In production, integrate with threat intelligence feeds
        # For now, just log the update
        logger.debug("Threat intelligence updated")

# Global security system instance
_security_system = None

def get_security_system(redis_client: redis.Redis = None, master_key: bytes = None):
    """Get global security system instance"""
    global _security_system
    
    if _security_system is None and redis_client and master_key:
        _security_system = ScrollIntelAdvancedSecurity(redis_client, master_key)
    
    return _security_system

async def initialize_advanced_security():
    """Initialize advanced security system"""
    logger.info("Starting ScrollIntel advanced security initialization")
    
    # This would be called during system initialization
    # security_system = get_security_system(redis_client, master_key)
    # await security_system.initialize_security_system()
    
    logger.info("ScrollIntel advanced security initialization completed")