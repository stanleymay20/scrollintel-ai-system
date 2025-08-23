"""
Vendor Access Monitoring with Time-Limited Access Controls
Implements comprehensive monitoring and control of vendor access
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import jwt
import secrets

class AccessType(Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"

class AccessStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"
    PENDING_APPROVAL = "pending_approval"

class MonitoringLevel(Enum):
    BASIC = "basic"
    ENHANCED = "enhanced"
    COMPREHENSIVE = "comprehensive"

@dataclass
class AccessRequest:
    request_id: str
    vendor_id: str
    requester_email: str
    access_type: AccessType
    resources_requested: List[str]
    business_justification: str
    requested_duration: timedelta
    emergency_access: bool
    approver_required: bool
    created_at: datetime
    status: AccessStatus
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

@dataclass
class AccessGrant:
    grant_id: str
    vendor_id: str
    user_email: str
    access_type: AccessType
    resources: List[str]
    granted_at: datetime
    expires_at: datetime
    status: AccessStatus
    access_token: str
    refresh_token: Optional[str]
    monitoring_level: MonitoringLevel
    session_metadata: Dict[str, Any]
    last_activity: Optional[datetime] = None
    activity_count: int = 0

@dataclass
class AccessActivity:
    activity_id: str
    grant_id: str
    vendor_id: str
    user_email: str
    action: str
    resource: str
    timestamp: datetime
    source_ip: str
    user_agent: str
    success: bool
    risk_score: float
    anomaly_detected: bool
    session_id: str
    additional_metadata: Dict[str, Any]

class VendorAccessMonitor:
    def __init__(self, config_path: str = "security/config/vendor_access_config.yaml"):
        self.config = self._load_config(config_path)
        self.active_grants = {}  # grant_id -> AccessGrant
        self.access_requests = {}  # request_id -> AccessRequest
        self.activity_log = []  # List of AccessActivity
        self.jwt_secret = self._generate_jwt_secret()
        self.anomaly_detector = self._initialize_anomaly_detector()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load vendor access configuration"""
        default_config = {
            "default_access_duration": {
                "read_only": 8,  # hours
                "read_write": 4,
                "admin": 2,
                "emergency": 1,
                "maintenance": 12
            },
            "max_access_duration": {
                "read_only": 24,  # hours
                "read_write": 8,
                "admin": 4,
                "emergency": 2,
                "maintenance": 24
            },
            "approval_required": {
                "read_only": False,
                "read_write": True,
                "admin": True,
                "emergency": False,  # Auto-approved but heavily monitored
                "maintenance": True
            },
            "monitoring_settings": {
                "activity_retention_days": 90,
                "anomaly_threshold": 0.7,
                "alert_on_anomaly": True,
                "session_timeout_minutes": 30
            },
            "risk_thresholds": {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.8
            },
            "allowed_ip_ranges": [],  # Empty means all IPs allowed
            "blocked_countries": ["CN", "RU", "KP"],  # ISO country codes
            "rate_limits": {
                "requests_per_minute": 100,
                "requests_per_hour": 1000
            }
        }
        
        try:
            # In production, load from actual config file
            return default_config
        except Exception:
            return default_config
    
    def _generate_jwt_secret(self) -> str:
        """Generate JWT secret for token signing"""
        return secrets.token_urlsafe(32)
    
    def _initialize_anomaly_detector(self) -> Dict[str, Any]:
        """Initialize anomaly detection system"""
        return {
            "baseline_patterns": {},
            "user_profiles": {},
            "risk_indicators": [
                "unusual_time_access",
                "geographic_anomaly",
                "excessive_requests",
                "privilege_escalation_attempt",
                "suspicious_resource_access"
            ]
        }
    
    async def request_access(self, vendor_id: str, requester_email: str, 
                           access_type: AccessType, resources: List[str],
                           business_justification: str, duration_hours: Optional[int] = None,
                           emergency: bool = False) -> AccessRequest:
        """Request vendor access"""
        request_id = self._generate_request_id(vendor_id, requester_email)
        
        # Determine duration
        if duration_hours is None:
            duration_hours = self.config["default_access_duration"][access_type.value]
        
        # Cap duration at maximum allowed
        max_duration = self.config["max_access_duration"][access_type.value]
        duration_hours = min(duration_hours, max_duration)
        
        # Check if approval is required
        approval_required = self.config["approval_required"][access_type.value]
        if emergency:
            approval_required = False  # Emergency access auto-approved
        
        access_request = AccessRequest(
            request_id=request_id,
            vendor_id=vendor_id,
            requester_email=requester_email,
            access_type=access_type,
            resources_requested=resources,
            business_justification=business_justification,
            requested_duration=timedelta(hours=duration_hours),
            emergency_access=emergency,
            approver_required=approval_required,
            created_at=datetime.now(),
            status=AccessStatus.PENDING_APPROVAL if approval_required else AccessStatus.ACTIVE
        )
        
        self.access_requests[request_id] = access_request
        
        # Auto-approve if no approval required
        if not approval_required:
            await self._auto_approve_request(access_request)
        
        return access_request
    
    async def approve_access_request(self, request_id: str, approver: str) -> bool:
        """Approve access request"""
        if request_id not in self.access_requests:
            return False
        
        access_request = self.access_requests[request_id]
        
        if access_request.status != AccessStatus.PENDING_APPROVAL:
            return False
        
        access_request.status = AccessStatus.ACTIVE
        access_request.approved_by = approver
        access_request.approved_at = datetime.now()
        
        # Create access grant
        await self._create_access_grant(access_request)
        
        return True
    
    async def _auto_approve_request(self, access_request: AccessRequest):
        """Auto-approve access request"""
        access_request.status = AccessStatus.ACTIVE
        access_request.approved_by = "system_auto_approval"
        access_request.approved_at = datetime.now()
        
        # Create access grant
        await self._create_access_grant(access_request)
    
    async def _create_access_grant(self, access_request: AccessRequest) -> AccessGrant:
        """Create access grant from approved request"""
        grant_id = self._generate_grant_id(access_request.vendor_id, access_request.requester_email)
        
        # Generate access tokens
        access_token = self._generate_access_token(access_request)
        refresh_token = self._generate_refresh_token(access_request) if access_request.requested_duration.total_seconds() > 3600 else None
        
        # Determine monitoring level
        monitoring_level = self._determine_monitoring_level(access_request)
        
        access_grant = AccessGrant(
            grant_id=grant_id,
            vendor_id=access_request.vendor_id,
            user_email=access_request.requester_email,
            access_type=access_request.access_type,
            resources=access_request.resources_requested,
            granted_at=datetime.now(),
            expires_at=datetime.now() + access_request.requested_duration,
            status=AccessStatus.ACTIVE,
            access_token=access_token,
            refresh_token=refresh_token,
            monitoring_level=monitoring_level,
            session_metadata={
                "emergency_access": access_request.emergency_access,
                "business_justification": access_request.business_justification,
                "approved_by": access_request.approved_by
            }
        )
        
        self.active_grants[grant_id] = access_grant
        
        # Start monitoring session
        await self._start_session_monitoring(access_grant)
        
        return access_grant
    
    def _determine_monitoring_level(self, access_request: AccessRequest) -> MonitoringLevel:
        """Determine monitoring level based on access request"""
        if access_request.emergency_access:
            return MonitoringLevel.COMPREHENSIVE
        elif access_request.access_type in [AccessType.ADMIN, AccessType.READ_WRITE]:
            return MonitoringLevel.ENHANCED
        else:
            return MonitoringLevel.BASIC
    
    def _generate_access_token(self, access_request: AccessRequest) -> str:
        """Generate JWT access token"""
        payload = {
            "vendor_id": access_request.vendor_id,
            "user_email": access_request.requester_email,
            "access_type": access_request.access_type.value,
            "resources": access_request.resources_requested,
            "iat": datetime.now().timestamp(),
            "exp": (datetime.now() + access_request.requested_duration).timestamp(),
            "emergency": access_request.emergency_access
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def _generate_refresh_token(self, access_request: AccessRequest) -> str:
        """Generate refresh token for long-duration access"""
        payload = {
            "vendor_id": access_request.vendor_id,
            "user_email": access_request.requester_email,
            "token_type": "refresh",
            "iat": datetime.now().timestamp(),
            "exp": (datetime.now() + access_request.requested_duration + timedelta(hours=1)).timestamp()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    async def validate_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate access token and return payload"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if token is expired
            if datetime.fromtimestamp(payload["exp"]) < datetime.now():
                return None
            
            # Find corresponding grant
            grant = self._find_grant_by_token(token)
            if not grant or grant.status != AccessStatus.ACTIVE:
                return None
            
            return payload
        
        except jwt.InvalidTokenError:
            return None
    
    def _find_grant_by_token(self, token: str) -> Optional[AccessGrant]:
        """Find access grant by token"""
        for grant in self.active_grants.values():
            if grant.access_token == token:
                return grant
        return None
    
    async def log_access_activity(self, token: str, action: str, resource: str,
                                source_ip: str, user_agent: str, success: bool,
                                additional_metadata: Optional[Dict[str, Any]] = None) -> AccessActivity:
        """Log vendor access activity"""
        # Validate token and get grant
        payload = await self.validate_access_token(token)
        if not payload:
            raise ValueError("Invalid or expired access token")
        
        grant = self._find_grant_by_token(token)
        if not grant:
            raise ValueError("Access grant not found")
        
        # Generate activity ID
        activity_id = self._generate_activity_id(grant.grant_id, action, resource)
        
        # Calculate risk score
        risk_score = await self._calculate_activity_risk(grant, action, resource, source_ip, user_agent)
        
        # Detect anomalies
        anomaly_detected = await self._detect_anomaly(grant, action, resource, source_ip, risk_score)
        
        # Create activity record
        activity = AccessActivity(
            activity_id=activity_id,
            grant_id=grant.grant_id,
            vendor_id=grant.vendor_id,
            user_email=grant.user_email,
            action=action,
            resource=resource,
            timestamp=datetime.now(),
            source_ip=source_ip,
            user_agent=user_agent,
            success=success,
            risk_score=risk_score,
            anomaly_detected=anomaly_detected,
            session_id=grant.grant_id,  # Using grant_id as session_id for simplicity
            additional_metadata=additional_metadata or {}
        )
        
        # Store activity
        self.activity_log.append(activity)
        
        # Update grant activity tracking
        grant.last_activity = datetime.now()
        grant.activity_count += 1
        
        # Handle anomalies
        if anomaly_detected:
            await self._handle_anomaly(activity, grant)
        
        return activity
    
    async def _calculate_activity_risk(self, grant: AccessGrant, action: str, resource: str,
                                     source_ip: str, user_agent: str) -> float:
        """Calculate risk score for activity"""
        risk_score = 0.0
        
        # Base risk by access type
        access_type_risk = {
            AccessType.READ_ONLY: 0.1,
            AccessType.READ_WRITE: 0.3,
            AccessType.ADMIN: 0.6,
            AccessType.EMERGENCY: 0.8,
            AccessType.MAINTENANCE: 0.4
        }
        risk_score += access_type_risk.get(grant.access_type, 0.5)
        
        # Time-based risk (higher risk outside business hours)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Outside 6 AM - 10 PM
            risk_score += 0.2
        
        # Weekend risk
        if datetime.now().weekday() >= 5:  # Saturday or Sunday
            risk_score += 0.1
        
        # Action-based risk
        high_risk_actions = ["delete", "modify", "admin", "config_change"]
        if any(risk_action in action.lower() for risk_action in high_risk_actions):
            risk_score += 0.3
        
        # Resource sensitivity risk
        sensitive_resources = ["database", "config", "user_data", "financial"]
        if any(sensitive in resource.lower() for sensitive in sensitive_resources):
            risk_score += 0.2
        
        # Geographic risk (simplified - in production, use GeoIP)
        if self._is_suspicious_ip(source_ip):
            risk_score += 0.4
        
        # Emergency access additional risk
        if grant.session_metadata.get("emergency_access", False):
            risk_score += 0.2
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def _is_suspicious_ip(self, ip: str) -> bool:
        """Check if IP is suspicious (simplified implementation)"""
        # In production, integrate with threat intelligence feeds
        suspicious_patterns = ["10.0.0.", "192.168.", "127.0.0."]
        return not any(ip.startswith(pattern) for pattern in suspicious_patterns)
    
    async def _detect_anomaly(self, grant: AccessGrant, action: str, resource: str,
                            source_ip: str, risk_score: float) -> bool:
        """Detect anomalous activity"""
        anomaly_threshold = self.config["monitoring_settings"]["anomaly_threshold"]
        
        # High risk score is an anomaly
        if risk_score >= anomaly_threshold:
            return True
        
        # Check for unusual patterns
        user_profile = self.anomaly_detector["user_profiles"].get(grant.user_email, {})
        
        # Unusual time access
        current_hour = datetime.now().hour
        typical_hours = user_profile.get("typical_access_hours", [])
        if typical_hours and current_hour not in typical_hours:
            return True
        
        # Excessive requests
        recent_activities = [
            a for a in self.activity_log
            if a.user_email == grant.user_email and
            a.timestamp > datetime.now() - timedelta(minutes=10)
        ]
        if len(recent_activities) > 50:  # More than 50 requests in 10 minutes
            return True
        
        # Privilege escalation attempt
        if grant.access_type == AccessType.READ_ONLY and "write" in action.lower():
            return True
        
        return False
    
    async def _handle_anomaly(self, activity: AccessActivity, grant: AccessGrant):
        """Handle detected anomaly"""
        if self.config["monitoring_settings"]["alert_on_anomaly"]:
            await self._send_anomaly_alert(activity, grant)
        
        # For high-risk anomalies, consider suspending access
        if activity.risk_score > 0.8:
            await self._suspend_access(grant.grant_id, "High-risk anomaly detected")
    
    async def _send_anomaly_alert(self, activity: AccessActivity, grant: AccessGrant):
        """Send anomaly alert to security team"""
        alert_data = {
            "alert_type": "vendor_access_anomaly",
            "timestamp": datetime.now().isoformat(),
            "vendor_id": grant.vendor_id,
            "user_email": grant.user_email,
            "activity": {
                "action": activity.action,
                "resource": activity.resource,
                "risk_score": activity.risk_score,
                "source_ip": activity.source_ip
            },
            "grant_info": {
                "access_type": grant.access_type.value,
                "expires_at": grant.expires_at.isoformat(),
                "emergency_access": grant.session_metadata.get("emergency_access", False)
            }
        }
        
        # In production, send to SIEM/alerting system
        print(f"ANOMALY ALERT: {json.dumps(alert_data, indent=2)}")
    
    async def revoke_access(self, grant_id: str, reason: str) -> bool:
        """Revoke vendor access"""
        if grant_id not in self.active_grants:
            return False
        
        grant = self.active_grants[grant_id]
        grant.status = AccessStatus.REVOKED
        
        # Log revocation
        await self._log_access_revocation(grant, reason)
        
        return True
    
    async def _suspend_access(self, grant_id: str, reason: str) -> bool:
        """Suspend vendor access"""
        if grant_id not in self.active_grants:
            return False
        
        grant = self.active_grants[grant_id]
        grant.status = AccessStatus.SUSPENDED
        
        # Log suspension
        await self._log_access_suspension(grant, reason)
        
        return True
    
    async def _log_access_revocation(self, grant: AccessGrant, reason: str):
        """Log access revocation"""
        revocation_activity = AccessActivity(
            activity_id=self._generate_activity_id(grant.grant_id, "revoke_access", "system"),
            grant_id=grant.grant_id,
            vendor_id=grant.vendor_id,
            user_email=grant.user_email,
            action="revoke_access",
            resource="system",
            timestamp=datetime.now(),
            source_ip="system",
            user_agent="system",
            success=True,
            risk_score=0.0,
            anomaly_detected=False,
            session_id=grant.grant_id,
            additional_metadata={"reason": reason}
        )
        
        self.activity_log.append(revocation_activity)
    
    async def _log_access_suspension(self, grant: AccessGrant, reason: str):
        """Log access suspension"""
        suspension_activity = AccessActivity(
            activity_id=self._generate_activity_id(grant.grant_id, "suspend_access", "system"),
            grant_id=grant.grant_id,
            vendor_id=grant.vendor_id,
            user_email=grant.user_email,
            action="suspend_access",
            resource="system",
            timestamp=datetime.now(),
            source_ip="system",
            user_agent="system",
            success=True,
            risk_score=0.0,
            anomaly_detected=False,
            session_id=grant.grant_id,
            additional_metadata={"reason": reason}
        )
        
        self.activity_log.append(suspension_activity)
    
    async def _start_session_monitoring(self, grant: AccessGrant):
        """Start monitoring session for automatic expiration"""
        # In production, this would be handled by a background task scheduler
        asyncio.create_task(self._monitor_session_expiration(grant))
    
    async def _monitor_session_expiration(self, grant: AccessGrant):
        """Monitor session for expiration"""
        while grant.status == AccessStatus.ACTIVE:
            if datetime.now() >= grant.expires_at:
                grant.status = AccessStatus.EXPIRED
                await self._log_access_expiration(grant)
                break
            
            # Check for session timeout
            if grant.last_activity:
                timeout_minutes = self.config["monitoring_settings"]["session_timeout_minutes"]
                if datetime.now() - grant.last_activity > timedelta(minutes=timeout_minutes):
                    grant.status = AccessStatus.EXPIRED
                    await self._log_access_timeout(grant)
                    break
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _log_access_expiration(self, grant: AccessGrant):
        """Log access expiration"""
        expiration_activity = AccessActivity(
            activity_id=self._generate_activity_id(grant.grant_id, "access_expired", "system"),
            grant_id=grant.grant_id,
            vendor_id=grant.vendor_id,
            user_email=grant.user_email,
            action="access_expired",
            resource="system",
            timestamp=datetime.now(),
            source_ip="system",
            user_agent="system",
            success=True,
            risk_score=0.0,
            anomaly_detected=False,
            session_id=grant.grant_id,
            additional_metadata={"expiration_reason": "time_limit_reached"}
        )
        
        self.activity_log.append(expiration_activity)
    
    async def _log_access_timeout(self, grant: AccessGrant):
        """Log access timeout"""
        timeout_activity = AccessActivity(
            activity_id=self._generate_activity_id(grant.grant_id, "session_timeout", "system"),
            grant_id=grant.grant_id,
            vendor_id=grant.vendor_id,
            user_email=grant.user_email,
            action="session_timeout",
            resource="system",
            timestamp=datetime.now(),
            source_ip="system",
            user_agent="system",
            success=True,
            risk_score=0.0,
            anomaly_detected=False,
            session_id=grant.grant_id,
            additional_metadata={"timeout_minutes": self.config["monitoring_settings"]["session_timeout_minutes"]}
        )
        
        self.activity_log.append(timeout_activity)
    
    def get_active_grants(self, vendor_id: Optional[str] = None) -> List[AccessGrant]:
        """Get active access grants"""
        grants = [g for g in self.active_grants.values() if g.status == AccessStatus.ACTIVE]
        
        if vendor_id:
            grants = [g for g in grants if g.vendor_id == vendor_id]
        
        return grants
    
    def get_access_activities(self, vendor_id: Optional[str] = None, 
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[AccessActivity]:
        """Get access activities with optional filters"""
        activities = self.activity_log
        
        if vendor_id:
            activities = [a for a in activities if a.vendor_id == vendor_id]
        
        if start_date:
            activities = [a for a in activities if a.timestamp >= start_date]
        
        if end_date:
            activities = [a for a in activities if a.timestamp <= end_date]
        
        return sorted(activities, key=lambda x: x.timestamp, reverse=True)
    
    async def generate_access_report(self, vendor_id: str, 
                                   report_period_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive access report for vendor"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=report_period_days)
        
        # Get activities for the period
        activities = self.get_access_activities(vendor_id, start_date, end_date)
        
        # Get current active grants
        active_grants = self.get_active_grants(vendor_id)
        
        # Calculate statistics
        total_activities = len(activities)
        successful_activities = len([a for a in activities if a.success])
        failed_activities = total_activities - successful_activities
        anomalous_activities = len([a for a in activities if a.anomaly_detected])
        
        # Risk analysis
        avg_risk_score = sum(a.risk_score for a in activities) / total_activities if total_activities > 0 else 0
        high_risk_activities = len([a for a in activities if a.risk_score > 0.7])
        
        # Access patterns
        access_by_hour = {}
        for activity in activities:
            hour = activity.timestamp.hour
            access_by_hour[hour] = access_by_hour.get(hour, 0) + 1
        
        # Resource access patterns
        resource_access = {}
        for activity in activities:
            resource_access[activity.resource] = resource_access.get(activity.resource, 0) + 1
        
        return {
            "report_metadata": {
                "vendor_id": vendor_id,
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": report_period_days
                },
                "generated_at": datetime.now().isoformat()
            },
            "access_summary": {
                "active_grants": len(active_grants),
                "total_activities": total_activities,
                "successful_activities": successful_activities,
                "failed_activities": failed_activities,
                "anomalous_activities": anomalous_activities,
                "success_rate": (successful_activities / total_activities * 100) if total_activities > 0 else 0
            },
            "risk_analysis": {
                "average_risk_score": round(avg_risk_score, 3),
                "high_risk_activities": high_risk_activities,
                "anomaly_rate": (anomalous_activities / total_activities * 100) if total_activities > 0 else 0
            },
            "access_patterns": {
                "access_by_hour": access_by_hour,
                "resource_access": dict(sorted(resource_access.items(), key=lambda x: x[1], reverse=True)[:10])
            },
            "active_grants_details": [
                {
                    "grant_id": g.grant_id,
                    "user_email": g.user_email,
                    "access_type": g.access_type.value,
                    "expires_at": g.expires_at.isoformat(),
                    "activity_count": g.activity_count,
                    "last_activity": g.last_activity.isoformat() if g.last_activity else None
                }
                for g in active_grants
            ],
            "recommendations": self._generate_access_recommendations(activities, active_grants)
        }
    
    def _generate_access_recommendations(self, activities: List[AccessActivity], 
                                       active_grants: List[AccessGrant]) -> List[str]:
        """Generate access recommendations based on analysis"""
        recommendations = []
        
        if not activities:
            return ["No access activity recorded for this vendor"]
        
        # High anomaly rate
        anomaly_rate = len([a for a in activities if a.anomaly_detected]) / len(activities)
        if anomaly_rate > 0.1:  # More than 10% anomalous
            recommendations.append("High anomaly rate detected - review vendor access patterns")
        
        # High risk activities
        high_risk_count = len([a for a in activities if a.risk_score > 0.7])
        if high_risk_count > len(activities) * 0.05:  # More than 5% high risk
            recommendations.append("Frequent high-risk activities - consider additional monitoring")
        
        # Long-running grants
        for grant in active_grants:
            if grant.expires_at - datetime.now() > timedelta(hours=24):
                recommendations.append(f"Long-running access grant for {grant.user_email} - review necessity")
        
        # Inactive grants
        for grant in active_grants:
            if grant.last_activity and datetime.now() - grant.last_activity > timedelta(hours=4):
                recommendations.append(f"Inactive access grant for {grant.user_email} - consider revocation")
        
        if not recommendations:
            recommendations.append("Access patterns appear normal - continue regular monitoring")
        
        return recommendations
    
    def _generate_request_id(self, vendor_id: str, requester_email: str) -> str:
        """Generate unique request ID"""
        timestamp = datetime.now().isoformat()
        content = f"{vendor_id}_{requester_email}_{timestamp}"
        return f"REQ-{hashlib.sha256(content.encode()).hexdigest()[:12]}"
    
    def _generate_grant_id(self, vendor_id: str, user_email: str) -> str:
        """Generate unique grant ID"""
        timestamp = datetime.now().isoformat()
        content = f"{vendor_id}_{user_email}_{timestamp}"
        return f"GRT-{hashlib.sha256(content.encode()).hexdigest()[:12]}"
    
    def _generate_activity_id(self, grant_id: str, action: str, resource: str) -> str:
        """Generate unique activity ID"""
        timestamp = datetime.now().isoformat()
        content = f"{grant_id}_{action}_{resource}_{timestamp}"
        return f"ACT-{hashlib.sha256(content.encode()).hexdigest()[:12]}"