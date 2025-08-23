"""
Threat Detection and Response Engine
Implements security event monitoring, alerting, and automated response
"""
import re
import json
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

from ..models.security_audit_models import (
    ThreatDetectionRule, SecurityIncident, SecurityAuditLog,
    SeverityLevel, SecurityEventType, ThreatDetectionRuleCreate,
    SecurityIncidentCreate
)
from ..core.database_connection_manager import get_sync_session
from ..core.logging_config import get_logger
from ..core.security_audit_logger import audit_logger

logger = get_logger(__name__)

class ThreatType(str, Enum):
    """Types of security threats"""
    BRUTE_FORCE = "brute_force"
    ANOMALOUS_ACCESS = "anomalous_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    SUSPICIOUS_LOGIN = "suspicious_login"
    CONFIGURATION_TAMPERING = "configuration_tampering"
    MALICIOUS_ACTIVITY = "malicious_activity"
    INSIDER_THREAT = "insider_threat"

class ResponseAction(str, Enum):
    """Automated response actions"""
    ALERT_ONLY = "alert_only"
    BLOCK_IP = "block_ip"
    DISABLE_USER = "disable_user"
    QUARANTINE_SESSION = "quarantine_session"
    ESCALATE_INCIDENT = "escalate_incident"
    NOTIFY_ADMIN = "notify_admin"

@dataclass
class ThreatPattern:
    """Threat detection pattern"""
    pattern_id: str
    name: str
    description: str
    pattern_type: str
    conditions: List[Dict[str, Any]]
    threshold: int
    time_window: int  # seconds
    severity: SeverityLevel
    actions: List[ResponseAction]

@dataclass
class ThreatAlert:
    """Threat detection alert"""
    alert_id: str
    threat_type: ThreatType
    severity: SeverityLevel
    confidence: float
    description: str
    affected_resources: List[str]
    indicators: List[Dict[str, Any]]
    recommended_actions: List[str]
    timestamp: datetime

class ThreatDetectionEngine:
    """Advanced threat detection and response engine"""
    
    def __init__(self):
        self.active_rules = {}
        self.event_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.user_baselines = {}
        self.ip_reputation = {}
        self.active_incidents = {}
        
        # Initialize built-in threat patterns
        self._initialize_threat_patterns()
    
    def _initialize_threat_patterns(self):
        """Initialize built-in threat detection patterns"""
        self.built_in_patterns = [
            ThreatPattern(
                pattern_id="brute_force_login",
                name="Brute Force Login Attempts",
                description="Multiple failed login attempts from same IP",
                pattern_type="frequency",
                conditions=[
                    {"field": "event_type", "operator": "equals", "value": "authentication"},
                    {"field": "outcome", "operator": "equals", "value": "failure"},
                    {"field": "source_ip", "operator": "exists"}
                ],
                threshold=5,
                time_window=300,  # 5 minutes
                severity=SeverityLevel.HIGH,
                actions=[ResponseAction.BLOCK_IP, ResponseAction.ALERT_ONLY]
            ),
            ThreatPattern(
                pattern_id="anomalous_data_access",
                name="Anomalous Data Access Pattern",
                description="Unusual data access patterns for user",
                pattern_type="anomaly",
                conditions=[
                    {"field": "event_type", "operator": "equals", "value": "data_access"},
                    {"field": "outcome", "operator": "equals", "value": "success"}
                ],
                threshold=1,
                time_window=3600,  # 1 hour
                severity=SeverityLevel.MEDIUM,
                actions=[ResponseAction.ALERT_ONLY, ResponseAction.NOTIFY_ADMIN]
            ),
            ThreatPattern(
                pattern_id="privilege_escalation",
                name="Privilege Escalation Attempt",
                description="User attempting to access resources above their privilege level",
                pattern_type="rule_based",
                conditions=[
                    {"field": "event_type", "operator": "equals", "value": "authorization"},
                    {"field": "outcome", "operator": "equals", "value": "failure"},
                    {"field": "details.privilege_required", "operator": "greater_than", "value": "user"}
                ],
                threshold=3,
                time_window=600,  # 10 minutes
                severity=SeverityLevel.HIGH,
                actions=[ResponseAction.ESCALATE_INCIDENT, ResponseAction.NOTIFY_ADMIN]
            ),
            ThreatPattern(
                pattern_id="after_hours_access",
                name="After Hours Access",
                description="Access attempts outside normal business hours",
                pattern_type="temporal",
                conditions=[
                    {"field": "timestamp", "operator": "outside_hours", "value": "09:00-17:00"},
                    {"field": "event_type", "operator": "in", "value": ["authentication", "data_access"]}
                ],
                threshold=1,
                time_window=3600,
                severity=SeverityLevel.MEDIUM,
                actions=[ResponseAction.ALERT_ONLY]
            ),
            ThreatPattern(
                pattern_id="configuration_tampering",
                name="Configuration Tampering",
                description="Unauthorized configuration changes",
                pattern_type="rule_based",
                conditions=[
                    {"field": "event_type", "operator": "equals", "value": "configuration_change"},
                    {"field": "details.authorized", "operator": "equals", "value": False}
                ],
                threshold=1,
                time_window=0,
                severity=SeverityLevel.CRITICAL,
                actions=[ResponseAction.ESCALATE_INCIDENT, ResponseAction.DISABLE_USER]
            )
        ]
    
    async def analyze_security_event(self, audit_log: SecurityAuditLog) -> List[ThreatAlert]:
        """Analyze security event for threats"""
        alerts = []
        
        try:
            # Add event to buffer for pattern analysis
            self._add_event_to_buffer(audit_log)
            
            # Check against all active rules
            for pattern in self.built_in_patterns:
                if await self._check_pattern_match(audit_log, pattern):
                    alert = await self._generate_threat_alert(audit_log, pattern)
                    if alert:
                        alerts.append(alert)
                        await self._handle_threat_response(alert, pattern)
            
            # Check custom rules from database
            custom_alerts = await self._check_custom_rules(audit_log)
            alerts.extend(custom_alerts)
            
            # Perform behavioral analysis
            behavioral_alerts = await self._perform_behavioral_analysis(audit_log)
            alerts.extend(behavioral_alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to analyze security event: {str(e)}")
            return []
    
    async def _check_pattern_match(self, audit_log: SecurityAuditLog, pattern: ThreatPattern) -> bool:
        """Check if event matches threat pattern"""
        try:
            # Check basic conditions
            if not self._evaluate_conditions(audit_log, pattern.conditions):
                return False
            
            # Check frequency-based patterns
            if pattern.pattern_type == "frequency":
                return await self._check_frequency_pattern(audit_log, pattern)
            
            # Check anomaly-based patterns
            elif pattern.pattern_type == "anomaly":
                return await self._check_anomaly_pattern(audit_log, pattern)
            
            # Check rule-based patterns
            elif pattern.pattern_type == "rule_based":
                return True  # Already checked conditions
            
            # Check temporal patterns
            elif pattern.pattern_type == "temporal":
                return self._check_temporal_pattern(audit_log, pattern)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check pattern match: {str(e)}")
            return False
    
    def _evaluate_conditions(self, audit_log: SecurityAuditLog, conditions: List[Dict[str, Any]]) -> bool:
        """Evaluate pattern conditions against audit log"""
        for condition in conditions:
            field = condition["field"]
            operator = condition["operator"]
            value = condition["value"]
            
            # Get field value from audit log
            field_value = self._get_field_value(audit_log, field)
            
            # Evaluate condition
            if not self._evaluate_condition(field_value, operator, value):
                return False
        
        return True
    
    def _get_field_value(self, audit_log: SecurityAuditLog, field: str) -> Any:
        """Get field value from audit log"""
        if "." in field:
            # Handle nested fields like "details.privilege_required"
            parts = field.split(".")
            value = audit_log
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value
        else:
            return getattr(audit_log, field, None)
    
    def _evaluate_condition(self, field_value: Any, operator: str, expected_value: Any) -> bool:
        """Evaluate individual condition"""
        if field_value is None:
            return operator == "not_exists"
        
        if operator == "equals":
            return field_value == expected_value
        elif operator == "not_equals":
            return field_value != expected_value
        elif operator == "greater_than":
            return field_value > expected_value
        elif operator == "less_than":
            return field_value < expected_value
        elif operator == "contains":
            return expected_value in str(field_value)
        elif operator == "in":
            return field_value in expected_value
        elif operator == "exists":
            return True
        elif operator == "not_exists":
            return False
        elif operator == "outside_hours":
            return self._is_outside_hours(field_value, expected_value)
        
        return False
    
    def _is_outside_hours(self, timestamp: datetime, hours_range: str) -> bool:
        """Check if timestamp is outside business hours"""
        try:
            start_hour, end_hour = hours_range.split("-")
            start_time = int(start_hour.split(":")[0])
            end_time = int(end_hour.split(":")[0])
            
            current_hour = timestamp.hour
            return current_hour < start_time or current_hour >= end_time
        except:
            return False
    
    async def _check_frequency_pattern(self, audit_log: SecurityAuditLog, pattern: ThreatPattern) -> bool:
        """Check frequency-based threat pattern"""
        try:
            # Get events from buffer within time window
            cutoff_time = datetime.utcnow() - timedelta(seconds=pattern.time_window)
            
            # Create key for grouping (e.g., by IP, user, etc.)
            group_key = self._create_group_key(audit_log, pattern)
            events = self.event_buffer[group_key]
            
            # Count matching events within time window
            matching_count = 0
            for event in events:
                if event.timestamp >= cutoff_time and self._evaluate_conditions(event, pattern.conditions):
                    matching_count += 1
            
            return matching_count >= pattern.threshold
            
        except Exception as e:
            logger.error(f"Failed to check frequency pattern: {str(e)}")
            return False
    
    async def _check_anomaly_pattern(self, audit_log: SecurityAuditLog, pattern: ThreatPattern) -> bool:
        """Check anomaly-based threat pattern"""
        try:
            if not audit_log.user_id:
                return False
            
            # Get user baseline behavior
            baseline = self.user_baselines.get(audit_log.user_id, {})
            
            # Check for anomalies based on pattern
            if pattern.pattern_id == "anomalous_data_access":
                return self._check_data_access_anomaly(audit_log, baseline)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check anomaly pattern: {str(e)}")
            return False
    
    def _check_temporal_pattern(self, audit_log: SecurityAuditLog, pattern: ThreatPattern) -> bool:
        """Check temporal-based threat pattern"""
        return self._evaluate_conditions(audit_log, pattern.conditions)
    
    def _check_data_access_anomaly(self, audit_log: SecurityAuditLog, baseline: Dict[str, Any]) -> bool:
        """Check for data access anomalies"""
        try:
            # Simple anomaly detection based on access patterns
            current_hour = audit_log.timestamp.hour
            typical_hours = baseline.get("typical_access_hours", [])
            
            if typical_hours and current_hour not in typical_hours:
                return True
            
            # Check for unusual resource access
            resource = audit_log.resource
            if resource:
                typical_resources = baseline.get("typical_resources", [])
                if typical_resources and resource not in typical_resources:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check data access anomaly: {str(e)}")
            return False
    
    def _create_group_key(self, audit_log: SecurityAuditLog, pattern: ThreatPattern) -> str:
        """Create grouping key for pattern matching"""
        # Default grouping by source IP
        if pattern.pattern_id == "brute_force_login":
            return f"ip:{audit_log.source_ip}"
        elif pattern.pattern_id == "privilege_escalation":
            return f"user:{audit_log.user_id}"
        else:
            return f"general:{audit_log.source_ip or audit_log.user_id}"
    
    def _add_event_to_buffer(self, audit_log: SecurityAuditLog):
        """Add event to analysis buffer"""
        group_key = f"all_events"
        self.event_buffer[group_key].append(audit_log)
        
        # Also add to specific buffers
        if audit_log.source_ip:
            ip_key = f"ip:{audit_log.source_ip}"
            self.event_buffer[ip_key].append(audit_log)
        
        if audit_log.user_id:
            user_key = f"user:{audit_log.user_id}"
            self.event_buffer[user_key].append(audit_log)
    
    async def _generate_threat_alert(self, audit_log: SecurityAuditLog, pattern: ThreatPattern) -> Optional[ThreatAlert]:
        """Generate threat alert from pattern match"""
        try:
            alert_id = str(uuid.uuid4())
            
            # Determine threat type from pattern
            threat_type_mapping = {
                "brute_force_login": ThreatType.BRUTE_FORCE,
                "anomalous_data_access": ThreatType.ANOMALOUS_ACCESS,
                "privilege_escalation": ThreatType.PRIVILEGE_ESCALATION,
                "after_hours_access": ThreatType.SUSPICIOUS_LOGIN,
                "configuration_tampering": ThreatType.CONFIGURATION_TAMPERING
            }
            
            threat_type = threat_type_mapping.get(pattern.pattern_id, ThreatType.MALICIOUS_ACTIVITY)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(audit_log, pattern)
            
            # Generate indicators
            indicators = self._generate_indicators(audit_log, pattern)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(pattern, audit_log)
            
            alert = ThreatAlert(
                alert_id=alert_id,
                threat_type=threat_type,
                severity=pattern.severity,
                confidence=confidence,
                description=f"{pattern.name}: {pattern.description}",
                affected_resources=[audit_log.resource] if audit_log.resource else [],
                indicators=indicators,
                recommended_actions=recommendations,
                timestamp=datetime.utcnow()
            )
            
            # Log the threat detection
            await audit_logger.log_threat_detection(
                threat_type=threat_type.value,
                severity=pattern.severity,
                source_ip=audit_log.source_ip,
                user_id=audit_log.user_id,
                details={
                    "alert_id": alert_id,
                    "pattern_id": pattern.pattern_id,
                    "confidence": confidence,
                    "indicators": indicators
                }
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to generate threat alert: {str(e)}")
            return None
    
    def _calculate_confidence_score(self, audit_log: SecurityAuditLog, pattern: ThreatPattern) -> float:
        """Calculate confidence score for threat detection"""
        base_confidence = 0.7
        
        # Adjust based on pattern type
        if pattern.pattern_type == "frequency":
            base_confidence = 0.9
        elif pattern.pattern_type == "rule_based":
            base_confidence = 0.95
        elif pattern.pattern_type == "anomaly":
            base_confidence = 0.6
        
        # Adjust based on severity
        if pattern.severity == SeverityLevel.CRITICAL:
            base_confidence += 0.1
        elif pattern.severity == SeverityLevel.LOW:
            base_confidence -= 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _generate_indicators(self, audit_log: SecurityAuditLog, pattern: ThreatPattern) -> List[Dict[str, Any]]:
        """Generate threat indicators"""
        indicators = []
        
        if audit_log.source_ip:
            indicators.append({
                "type": "ip_address",
                "value": audit_log.source_ip,
                "description": "Source IP address"
            })
        
        if audit_log.user_id:
            indicators.append({
                "type": "user_account",
                "value": audit_log.user_id,
                "description": "User account involved"
            })
        
        if audit_log.resource:
            indicators.append({
                "type": "resource",
                "value": audit_log.resource,
                "description": "Affected resource"
            })
        
        indicators.append({
            "type": "event_pattern",
            "value": pattern.pattern_id,
            "description": f"Matched pattern: {pattern.name}"
        })
        
        return indicators
    
    def _generate_recommendations(self, pattern: ThreatPattern, audit_log: SecurityAuditLog) -> List[str]:
        """Generate recommended actions"""
        recommendations = []
        
        for action in pattern.actions:
            if action == ResponseAction.BLOCK_IP and audit_log.source_ip:
                recommendations.append(f"Block IP address: {audit_log.source_ip}")
            elif action == ResponseAction.DISABLE_USER and audit_log.user_id:
                recommendations.append(f"Disable user account: {audit_log.user_id}")
            elif action == ResponseAction.ESCALATE_INCIDENT:
                recommendations.append("Escalate to security incident response team")
            elif action == ResponseAction.NOTIFY_ADMIN:
                recommendations.append("Notify system administrators")
            elif action == ResponseAction.QUARANTINE_SESSION:
                recommendations.append("Quarantine user session")
            else:
                recommendations.append("Monitor and alert on similar activities")
        
        return recommendations
    
    async def _handle_threat_response(self, alert: ThreatAlert, pattern: ThreatPattern):
        """Handle automated threat response"""
        try:
            for action in pattern.actions:
                if action == ResponseAction.ESCALATE_INCIDENT:
                    await self._create_security_incident(alert, pattern)
                elif action == ResponseAction.NOTIFY_ADMIN:
                    await self._notify_administrators(alert)
                # Add more automated response actions as needed
                
        except Exception as e:
            logger.error(f"Failed to handle threat response: {str(e)}")
    
    async def _create_security_incident(self, alert: ThreatAlert, pattern: ThreatPattern):
        """Create security incident from threat alert"""
        try:
            incident_id = str(uuid.uuid4())
            
            incident = SecurityIncident(
                id=incident_id,
                title=f"Security Threat Detected: {alert.threat_type.value}",
                description=alert.description,
                severity=alert.severity.value,
                status="open",
                detection_time=alert.timestamp,
                impact_assessment={
                    "threat_type": alert.threat_type.value,
                    "confidence": alert.confidence,
                    "affected_resources": alert.affected_resources,
                    "indicators": alert.indicators
                },
                response_actions={
                    "recommended": alert.recommended_actions,
                    "automated": [action.value for action in pattern.actions]
                }
            )
            
            with get_sync_session() as db:
                db.add(incident)
                db.commit()
            
            self.active_incidents[incident_id] = incident
            logger.warning(f"Security incident created: {incident_id}")
            
        except Exception as e:
            logger.error(f"Failed to create security incident: {str(e)}")
    
    async def _notify_administrators(self, alert: ThreatAlert):
        """Notify administrators of security threat"""
        try:
            # This would integrate with notification system
            logger.critical(f"SECURITY ALERT: {alert.description}")
            logger.critical(f"Threat Type: {alert.threat_type.value}")
            logger.critical(f"Severity: {alert.severity.value}")
            logger.critical(f"Confidence: {alert.confidence}")
            
        except Exception as e:
            logger.error(f"Failed to notify administrators: {str(e)}")
    
    async def _check_custom_rules(self, audit_log: SecurityAuditLog) -> List[ThreatAlert]:
        """Check custom threat detection rules from database"""
        alerts = []
        
        try:
            with get_sync_session() as db:
                rules = db.query(ThreatDetectionRule).filter(
                    ThreatDetectionRule.is_active == True
                ).all()
            
            for rule in rules:
                if await self._evaluate_custom_rule(audit_log, rule):
                    alert = await self._generate_custom_alert(audit_log, rule)
                    if alert:
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to check custom rules: {str(e)}")
            return []
    
    async def _evaluate_custom_rule(self, audit_log: SecurityAuditLog, rule: ThreatDetectionRule) -> bool:
        """Evaluate custom threat detection rule"""
        try:
            # Parse rule pattern (could be regex, JSON conditions, etc.)
            if rule.rule_type == "regex":
                pattern = re.compile(rule.pattern)
                text_to_match = f"{audit_log.action} {audit_log.outcome} {audit_log.details}"
                return bool(pattern.search(text_to_match))
            
            elif rule.rule_type == "conditions":
                conditions = json.loads(rule.pattern)
                return self._evaluate_conditions(audit_log, conditions)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate custom rule: {str(e)}")
            return False
    
    async def _generate_custom_alert(self, audit_log: SecurityAuditLog, rule: ThreatDetectionRule) -> Optional[ThreatAlert]:
        """Generate alert from custom rule"""
        try:
            alert_id = str(uuid.uuid4())
            
            alert = ThreatAlert(
                alert_id=alert_id,
                threat_type=ThreatType.MALICIOUS_ACTIVITY,
                severity=SeverityLevel(rule.severity),
                confidence=0.8,
                description=f"Custom Rule Match: {rule.name}",
                affected_resources=[audit_log.resource] if audit_log.resource else [],
                indicators=[{
                    "type": "custom_rule",
                    "value": rule.id,
                    "description": rule.description or "Custom threat detection rule"
                }],
                recommended_actions=rule.actions or [],
                timestamp=datetime.utcnow()
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to generate custom alert: {str(e)}")
            return None
    
    async def _perform_behavioral_analysis(self, audit_log: SecurityAuditLog) -> List[ThreatAlert]:
        """Perform behavioral analysis for threat detection"""
        alerts = []
        
        try:
            if audit_log.user_id:
                # Update user baseline
                await self._update_user_baseline(audit_log)
                
                # Check for behavioral anomalies
                if await self._detect_behavioral_anomaly(audit_log):
                    alert = ThreatAlert(
                        alert_id=str(uuid.uuid4()),
                        threat_type=ThreatType.ANOMALOUS_ACCESS,
                        severity=SeverityLevel.MEDIUM,
                        confidence=0.7,
                        description="Behavioral anomaly detected",
                        affected_resources=[audit_log.resource] if audit_log.resource else [],
                        indicators=[{
                            "type": "behavioral_anomaly",
                            "value": audit_log.user_id,
                            "description": "User behavior deviates from baseline"
                        }],
                        recommended_actions=["Review user activity", "Verify user identity"],
                        timestamp=datetime.utcnow()
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to perform behavioral analysis: {str(e)}")
            return []
    
    async def _update_user_baseline(self, audit_log: SecurityAuditLog):
        """Update user behavioral baseline"""
        try:
            user_id = audit_log.user_id
            if not user_id:
                return
            
            if user_id not in self.user_baselines:
                self.user_baselines[user_id] = {
                    "typical_access_hours": [],
                    "typical_resources": [],
                    "typical_actions": [],
                    "last_updated": datetime.utcnow()
                }
            
            baseline = self.user_baselines[user_id]
            
            # Update typical access hours
            current_hour = audit_log.timestamp.hour
            if current_hour not in baseline["typical_access_hours"]:
                baseline["typical_access_hours"].append(current_hour)
            
            # Update typical resources
            if audit_log.resource and audit_log.resource not in baseline["typical_resources"]:
                baseline["typical_resources"].append(audit_log.resource)
                # Keep only recent resources (limit to 50)
                if len(baseline["typical_resources"]) > 50:
                    baseline["typical_resources"] = baseline["typical_resources"][-50:]
            
            # Update typical actions
            if audit_log.action not in baseline["typical_actions"]:
                baseline["typical_actions"].append(audit_log.action)
            
            baseline["last_updated"] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to update user baseline: {str(e)}")
    
    async def _detect_behavioral_anomaly(self, audit_log: SecurityAuditLog) -> bool:
        """Detect behavioral anomalies"""
        try:
            user_id = audit_log.user_id
            if not user_id or user_id not in self.user_baselines:
                return False
            
            baseline = self.user_baselines[user_id]
            
            # Check for time-based anomalies
            current_hour = audit_log.timestamp.hour
            typical_hours = baseline.get("typical_access_hours", [])
            
            if len(typical_hours) > 5 and current_hour not in typical_hours:
                return True
            
            # Check for resource access anomalies
            if audit_log.resource:
                typical_resources = baseline.get("typical_resources", [])
                if len(typical_resources) > 10 and audit_log.resource not in typical_resources:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to detect behavioral anomaly: {str(e)}")
            return False

# Global threat detection engine
threat_engine = ThreatDetectionEngine()