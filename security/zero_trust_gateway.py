"""
Zero Trust Network Gateway Implementation
Implements zero-trust principles with identity verification, risk assessment, and policy enforcement
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import hashlib
import time
import jwt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AccessDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    CHALLENGE = "challenge"

@dataclass
class Identity:
    user_id: str
    roles: List[str]
    attributes: Dict[str, Any]
    trust_score: float
    last_verified: datetime

@dataclass
class SecurityContext:
    source_ip: str
    user_agent: str
    device_fingerprint: str
    location: Optional[str]
    time_of_access: datetime
    network_segment: str

@dataclass
class AccessRequest:
    identity: Identity
    resource: str
    action: str
    context: SecurityContext
    credentials: Dict[str, Any]

@dataclass
class PolicyDecision:
    decision: AccessDecision
    risk_score: float
    reasons: List[str]
    required_controls: List[str]

class IdentityVerifier:
    """Verifies and validates user identities"""
    
    def __init__(self, jwt_secret: str):
        self.jwt_secret = jwt_secret
        self.trusted_devices = {}
        
    def verify(self, credentials: Dict[str, Any]) -> Optional[Identity]:
        """Verify user credentials and return identity"""
        try:
            token = credentials.get('token')
            if not token:
                return None
                
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            identity = Identity(
                user_id=payload['user_id'],
                roles=payload.get('roles', []),
                attributes=payload.get('attributes', {}),
                trust_score=payload.get('trust_score', 0.5),
                last_verified=datetime.fromisoformat(payload['last_verified'])
            )
            
            return identity
            
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token provided")
            return None
        except Exception as e:
            logger.error(f"Identity verification failed: {e}")
            return None

class RiskAssessor:
    """Assesses risk based on context and behavior"""
    
    def __init__(self):
        self.anomaly_threshold = 0.7
        self.location_whitelist = set()
        self.device_trust_scores = {}
        
    def calculate_risk(self, identity: Identity, resource: str, context: SecurityContext) -> float:
        """Calculate risk score for access request"""
        risk_factors = []
        
        # Identity trust score
        identity_risk = 1.0 - identity.trust_score
        risk_factors.append(identity_risk * 0.3)
        
        # Time-based risk
        time_risk = self._assess_time_risk(context.time_of_access)
        risk_factors.append(time_risk * 0.2)
        
        # Location risk
        location_risk = self._assess_location_risk(context.location, context.source_ip)
        risk_factors.append(location_risk * 0.2)
        
        # Device risk
        device_risk = self._assess_device_risk(context.device_fingerprint)
        risk_factors.append(device_risk * 0.15)
        
        # Resource sensitivity
        resource_risk = self._assess_resource_sensitivity(resource)
        risk_factors.append(resource_risk * 0.15)
        
        total_risk = sum(risk_factors)
        return min(total_risk, 1.0)
    
    def _assess_time_risk(self, access_time: datetime) -> float:
        """Assess risk based on time of access"""
        hour = access_time.hour
        # Higher risk for off-hours access
        if hour < 6 or hour > 22:
            return 0.6
        elif hour < 8 or hour > 18:
            return 0.3
        return 0.1
    
    def _assess_location_risk(self, location: Optional[str], ip: str) -> float:
        """Assess risk based on location and IP"""
        if location and location in self.location_whitelist:
            return 0.1
        
        # Check for suspicious IP patterns
        if self._is_suspicious_ip(ip):
            return 0.8
            
        return 0.4
    
    def _assess_device_risk(self, device_fingerprint: str) -> float:
        """Assess risk based on device fingerprint"""
        trust_score = self.device_trust_scores.get(device_fingerprint, 0.5)
        return 1.0 - trust_score
    
    def _assess_resource_sensitivity(self, resource: str) -> float:
        """Assess risk based on resource sensitivity"""
        sensitive_patterns = ['/admin/', '/api/v1/users/', '/config/', '/secrets/']
        
        for pattern in sensitive_patterns:
            if pattern in resource:
                return 0.8
                
        return 0.3
    
    def _is_suspicious_ip(self, ip: str) -> bool:
        """Check if IP is suspicious"""
        # Implement IP reputation checking
        suspicious_ranges = ['10.0.0.0/8', '192.168.0.0/16']  # Example
        return False  # Simplified implementation

class PolicyEngine:
    """Evaluates access policies and makes decisions"""
    
    def __init__(self):
        self.policies = self._load_default_policies()
        
    def evaluate(self, identity: Identity, resource: str, risk_score: float) -> PolicyDecision:
        """Evaluate access request against policies"""
        reasons = []
        required_controls = []
        
        # High-risk automatic deny
        if risk_score > 0.8:
            return PolicyDecision(
                decision=AccessDecision.DENY,
                risk_score=risk_score,
                reasons=["Risk score exceeds threshold"],
                required_controls=[]
            )
        
        # Check role-based access
        if not self._check_role_access(identity.roles, resource):
            return PolicyDecision(
                decision=AccessDecision.DENY,
                risk_score=risk_score,
                reasons=["Insufficient role permissions"],
                required_controls=[]
            )
        
        # Medium risk requires additional controls
        if risk_score > 0.5:
            required_controls.extend(["mfa", "device_verification"])
            return PolicyDecision(
                decision=AccessDecision.CHALLENGE,
                risk_score=risk_score,
                reasons=["Medium risk requires additional verification"],
                required_controls=required_controls
            )
        
        # Low risk allow
        return PolicyDecision(
            decision=AccessDecision.ALLOW,
            risk_score=risk_score,
            reasons=["Low risk, access granted"],
            required_controls=[]
        )
    
    def _load_default_policies(self) -> Dict[str, Any]:
        """Load default security policies"""
        return {
            "admin_resources": {
                "required_roles": ["admin", "security_admin"],
                "max_risk_score": 0.3
            },
            "user_resources": {
                "required_roles": ["user", "admin"],
                "max_risk_score": 0.7
            },
            "public_resources": {
                "required_roles": [],
                "max_risk_score": 1.0
            }
        }
    
    def _check_role_access(self, user_roles: List[str], resource: str) -> bool:
        """Check if user roles allow access to resource"""
        if '/admin/' in resource:
            return any(role in ['admin', 'security_admin'] for role in user_roles)
        elif '/api/' in resource:
            return any(role in ['user', 'admin', 'api_user'] for role in user_roles)
        
        return True  # Public resources

class ZeroTrustGateway:
    """Main zero trust gateway implementation"""
    
    def __init__(self, jwt_secret: str):
        self.identity_verifier = IdentityVerifier(jwt_secret)
        self.risk_assessor = RiskAssessor()
        self.policy_engine = PolicyEngine()
        self.audit_log = []
        
    def authorize_request(self, request: AccessRequest) -> PolicyDecision:
        """Authorize access request using zero trust principles"""
        start_time = time.time()
        
        try:
            # Step 1: Verify identity
            if not request.identity:
                return PolicyDecision(
                    decision=AccessDecision.DENY,
                    risk_score=1.0,
                    reasons=["Identity verification failed"],
                    required_controls=[]
                )
            
            # Step 2: Assess risk
            risk_score = self.risk_assessor.calculate_risk(
                request.identity, request.resource, request.context
            )
            
            # Step 3: Apply policy
            decision = self.policy_engine.evaluate(
                request.identity, request.resource, risk_score
            )
            
            # Step 4: Audit log
            self._log_access_attempt(request, decision, time.time() - start_time)
            
            return decision
            
        except Exception as e:
            logger.error(f"Authorization failed: {e}")
            return PolicyDecision(
                decision=AccessDecision.DENY,
                risk_score=1.0,
                reasons=[f"System error: {str(e)}"],
                required_controls=[]
            )
    
    def _log_access_attempt(self, request: AccessRequest, decision: PolicyDecision, duration: float):
        """Log access attempt for audit purposes"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": request.identity.user_id,
            "resource": request.resource,
            "action": request.action,
            "decision": decision.decision.value,
            "risk_score": decision.risk_score,
            "source_ip": request.context.source_ip,
            "duration_ms": duration * 1000,
            "reasons": decision.reasons
        }
        
        self.audit_log.append(log_entry)
        logger.info(f"Access attempt: {log_entry}")
    
    def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit logs"""
        return self.audit_log[-limit:]
    
    def update_device_trust(self, device_fingerprint: str, trust_score: float):
        """Update device trust score"""
        self.risk_assessor.device_trust_scores[device_fingerprint] = trust_score
    
    def add_trusted_location(self, location: str):
        """Add location to whitelist"""
        self.risk_assessor.location_whitelist.add(location)