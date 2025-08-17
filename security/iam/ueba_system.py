"""
User and Entity Behavior Analytics (UEBA) System
Detects anomalous access patterns and behaviors
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import math

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    TIME_BASED = "time_based"
    LOCATION_BASED = "location_based"
    RESOURCE_ACCESS = "resource_access"
    PERMISSION_ESCALATION = "permission_escalation"
    VOLUME_BASED = "volume_based"
    PATTERN_DEVIATION = "pattern_deviation"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class UserActivity:
    activity_id: str
    user_id: str
    timestamp: datetime
    action: str
    resource_id: str
    resource_type: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BehaviorProfile:
    user_id: str
    created_at: datetime
    updated_at: datetime
    typical_hours: List[int] = field(default_factory=list)
    typical_locations: List[str] = field(default_factory=list)
    typical_resources: Dict[str, int] = field(default_factory=dict)
    typical_actions: Dict[str, int] = field(default_factory=dict)
    activity_volume_stats: Dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.0

@dataclass
class AnomalyAlert:
    alert_id: str
    user_id: str
    anomaly_type: AnomalyType
    risk_level: RiskLevel
    confidence_score: float
    description: str
    detected_at: datetime
    activities: List[UserActivity]
    baseline_data: Dict[str, Any] = field(default_factory=dict)
    is_acknowledged: bool = False

class UEBASystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_profiles: Dict[str, BehaviorProfile] = {}
        self.activity_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.anomaly_alerts: Dict[str, AnomalyAlert] = {}
        self.user_risk_scores: Dict[str, float] = {}  # Add this for direct access
        self.learning_window_days = config.get("learning_window_days", 30)
        self.anomaly_thresholds = config.get("anomaly_thresholds", {
            "time_deviation": 2.0,
            "location_deviation": 0.8,
            "volume_multiplier": 3.0,
            "new_resource_threshold": 0.9
        })
    
    def record_activity(self, activity: UserActivity):
        """Record user activity for analysis"""
        try:
            # Store activity
            self.activity_history[activity.user_id].append(activity)
            
            # Update or create user profile
            if activity.user_id not in self.user_profiles:
                self._create_user_profile(activity.user_id)
            
            # Analyze for anomalies
            anomalies = self._analyze_activity_for_anomalies(activity)
            
            # Generate alerts for significant anomalies
            for anomaly in anomalies:
                if anomaly.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    self._generate_alert(anomaly)
            
            # Update user profile periodically
            if len(self.activity_history[activity.user_id]) % 100 == 0:
                self._update_user_profile(activity.user_id)
            
            logger.debug(f"Activity recorded for user {activity.user_id}")
            
        except Exception as e:
            logger.error(f"Activity recording failed: {str(e)}")
    
    def _create_user_profile(self, user_id: str):
        """Create initial behavior profile for user"""
        profile = BehaviorProfile(
            user_id=user_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.user_profiles[user_id] = profile
        logger.info(f"Created behavior profile for user {user_id}")
    
    def _update_user_profile(self, user_id: str):
        """Update user behavior profile based on recent activities"""
        try:
            if user_id not in self.user_profiles:
                self._create_user_profile(user_id)
            
            profile = self.user_profiles[user_id]
            activities = list(self.activity_history[user_id])
            
            if not activities:
                return
            
            # Filter activities within learning window
            cutoff_date = datetime.utcnow() - timedelta(days=self.learning_window_days)
            recent_activities = [a for a in activities if a.timestamp > cutoff_date]
            
            if not recent_activities:
                return
            
            # Update typical hours
            hours = [a.timestamp.hour for a in recent_activities]
            profile.typical_hours = self._calculate_typical_hours(hours)
            
            # Update typical locations (IP addresses)
            locations = [a.ip_address for a in recent_activities if a.ip_address]
            profile.typical_locations = self._calculate_typical_locations(locations)
            
            # Update typical resources
            resources = [a.resource_id for a in recent_activities]
            profile.typical_resources = self._calculate_frequency_distribution(resources)
            
            # Update typical actions
            actions = [a.action for a in recent_activities]
            profile.typical_actions = self._calculate_frequency_distribution(actions)
            
            # Update activity volume statistics
            profile.activity_volume_stats = self._calculate_volume_stats(recent_activities)
            
            # Update risk score
            profile.risk_score = self._calculate_user_risk_score(user_id)
            
            profile.updated_at = datetime.utcnow()
            
            logger.debug(f"Updated behavior profile for user {user_id}")
            
        except Exception as e:
            logger.error(f"Profile update failed for user {user_id}: {str(e)}")
    
    def _analyze_activity_for_anomalies(self, activity: UserActivity) -> List[AnomalyAlert]:
        """Analyze activity for anomalous patterns"""
        anomalies = []
        
        try:
            if activity.user_id not in self.user_profiles:
                return anomalies
            
            profile = self.user_profiles[activity.user_id]
            
            # Time-based anomaly detection
            time_anomaly = self._detect_time_anomaly(activity, profile)
            if time_anomaly:
                anomalies.append(time_anomaly)
            
            # Location-based anomaly detection
            location_anomaly = self._detect_location_anomaly(activity, profile)
            if location_anomaly:
                anomalies.append(location_anomaly)
            
            # Resource access anomaly detection
            resource_anomaly = self._detect_resource_anomaly(activity, profile)
            if resource_anomaly:
                anomalies.append(resource_anomaly)
            
            # Volume-based anomaly detection
            volume_anomaly = self._detect_volume_anomaly(activity, profile)
            if volume_anomaly:
                anomalies.append(volume_anomaly)
            
            # Permission escalation detection
            escalation_anomaly = self._detect_permission_escalation(activity)
            if escalation_anomaly:
                anomalies.append(escalation_anomaly)
            
        except Exception as e:
            logger.error(f"Anomaly analysis failed: {str(e)}")
        
        return anomalies
    
    def _detect_time_anomaly(self, activity: UserActivity, profile: BehaviorProfile) -> Optional[AnomalyAlert]:
        """Detect time-based anomalies"""
        if not profile.typical_hours:
            return None
        
        current_hour = activity.timestamp.hour
        
        # Calculate deviation from typical hours
        typical_hours_set = set(profile.typical_hours)
        
        if current_hour not in typical_hours_set:
            # Calculate how far this is from typical hours
            min_distance = min(
                min(abs(current_hour - h), 24 - abs(current_hour - h))
                for h in typical_hours_set
            )
            
            if min_distance >= self.anomaly_thresholds["time_deviation"]:
                confidence = min(min_distance / 12.0, 1.0)  # Normalize to 0-1
                risk_level = self._calculate_risk_level(confidence)
                
                return AnomalyAlert(
                    alert_id=f"time_anomaly_{activity.user_id}_{int(activity.timestamp.timestamp())}",
                    user_id=activity.user_id,
                    anomaly_type=AnomalyType.TIME_BASED,
                    risk_level=risk_level,
                    confidence_score=confidence,
                    description=f"User accessed system at unusual time: {current_hour}:00",
                    detected_at=datetime.utcnow(),
                    activities=[activity],
                    baseline_data={"typical_hours": profile.typical_hours}
                )
        
        return None
    
    def _detect_location_anomaly(self, activity: UserActivity, profile: BehaviorProfile) -> Optional[AnomalyAlert]:
        """Detect location-based anomalies"""
        if not activity.ip_address or not profile.typical_locations:
            return None
        
        if activity.ip_address not in profile.typical_locations:
            # New location detected
            confidence = self.anomaly_thresholds["location_deviation"]
            risk_level = self._calculate_risk_level(confidence)
            
            return AnomalyAlert(
                alert_id=f"location_anomaly_{activity.user_id}_{int(activity.timestamp.timestamp())}",
                user_id=activity.user_id,
                anomaly_type=AnomalyType.LOCATION_BASED,
                risk_level=risk_level,
                confidence_score=confidence,
                description=f"User accessed system from new location: {activity.ip_address}",
                detected_at=datetime.utcnow(),
                activities=[activity],
                baseline_data={"typical_locations": profile.typical_locations}
            )
        
        return None
    
    def _detect_resource_anomaly(self, activity: UserActivity, profile: BehaviorProfile) -> Optional[AnomalyAlert]:
        """Detect resource access anomalies"""
        if not profile.typical_resources:
            return None
        
        resource_frequency = profile.typical_resources.get(activity.resource_id, 0)
        total_accesses = sum(profile.typical_resources.values())
        
        if total_accesses == 0:
            return None
        
        resource_probability = resource_frequency / total_accesses
        
        if resource_probability < (1 - self.anomaly_thresholds["new_resource_threshold"]):
            confidence = 1 - resource_probability
            risk_level = self._calculate_risk_level(confidence)
            
            return AnomalyAlert(
                alert_id=f"resource_anomaly_{activity.user_id}_{int(activity.timestamp.timestamp())}",
                user_id=activity.user_id,
                anomaly_type=AnomalyType.RESOURCE_ACCESS,
                risk_level=risk_level,
                confidence_score=confidence,
                description=f"User accessed unusual resource: {activity.resource_id}",
                detected_at=datetime.utcnow(),
                activities=[activity],
                baseline_data={"typical_resources": dict(list(profile.typical_resources.items())[:10])}
            )
        
        return None
    
    def _detect_volume_anomaly(self, activity: UserActivity, profile: BehaviorProfile) -> Optional[AnomalyAlert]:
        """Detect volume-based anomalies"""
        if not profile.activity_volume_stats:
            return None
        
        # Get recent activity count for the user
        recent_activities = [
            a for a in self.activity_history[activity.user_id]
            if a.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        current_hourly_volume = len(recent_activities)
        avg_hourly_volume = profile.activity_volume_stats.get("avg_hourly", 1)
        
        if current_hourly_volume > avg_hourly_volume * self.anomaly_thresholds["volume_multiplier"]:
            confidence = min(current_hourly_volume / (avg_hourly_volume * self.anomaly_thresholds["volume_multiplier"]), 1.0)
            risk_level = self._calculate_risk_level(confidence)
            
            return AnomalyAlert(
                alert_id=f"volume_anomaly_{activity.user_id}_{int(activity.timestamp.timestamp())}",
                user_id=activity.user_id,
                anomaly_type=AnomalyType.VOLUME_BASED,
                risk_level=risk_level,
                confidence_score=confidence,
                description=f"Unusual activity volume: {current_hourly_volume} actions in past hour",
                detected_at=datetime.utcnow(),
                activities=recent_activities[-10:],  # Last 10 activities
                baseline_data={"avg_hourly_volume": avg_hourly_volume}
            )
        
        return None
    
    def _detect_permission_escalation(self, activity: UserActivity) -> Optional[AnomalyAlert]:
        """Detect permission escalation attempts"""
        # Look for patterns indicating privilege escalation
        escalation_indicators = [
            "admin", "root", "sudo", "privilege", "escalate", "elevate"
        ]
        
        activity_text = f"{activity.action} {activity.resource_id}".lower()
        
        if any(indicator in activity_text for indicator in escalation_indicators):
            # Check if this is unusual for the user
            recent_activities = list(self.activity_history[activity.user_id])[-100:]
            
            similar_activities = [
                a for a in recent_activities
                if any(indicator in f"{a.action} {a.resource_id}".lower() 
                      for indicator in escalation_indicators)
            ]
            
            if len(similar_activities) <= 2:  # Unusual pattern
                confidence = 0.8
                risk_level = RiskLevel.HIGH
                
                return AnomalyAlert(
                    alert_id=f"escalation_anomaly_{activity.user_id}_{int(activity.timestamp.timestamp())}",
                    user_id=activity.user_id,
                    anomaly_type=AnomalyType.PERMISSION_ESCALATION,
                    risk_level=risk_level,
                    confidence_score=confidence,
                    description=f"Potential permission escalation attempt: {activity.action}",
                    detected_at=datetime.utcnow(),
                    activities=[activity],
                    baseline_data={"similar_activities_count": len(similar_activities)}
                )
        
        return None
    
    def _generate_alert(self, anomaly: AnomalyAlert):
        """Generate and store anomaly alert"""
        self.anomaly_alerts[anomaly.alert_id] = anomaly
        logger.warning(f"Anomaly alert generated: {anomaly.alert_id} - {anomaly.description}")
    
    def get_user_risk_score(self, user_id: str) -> float:
        """Get current risk score for user"""
        # Check direct risk scores first
        if user_id in self.user_risk_scores:
            return self.user_risk_scores[user_id]
            
        if user_id not in self.user_profiles:
            return 0.5  # Default medium risk for unknown users
        
        return self.user_profiles[user_id].risk_score
    
    def get_active_alerts(self, user_id: Optional[str] = None, 
                         risk_level: Optional[RiskLevel] = None) -> List[AnomalyAlert]:
        """Get active anomaly alerts"""
        alerts = list(self.anomaly_alerts.values())
        
        if user_id:
            alerts = [a for a in alerts if a.user_id == user_id]
        
        if risk_level:
            alerts = [a for a in alerts if a.risk_level == risk_level]
        
        # Filter out acknowledged alerts
        alerts = [a for a in alerts if not a.is_acknowledged]
        
        # Sort by risk level and confidence
        risk_order = {RiskLevel.CRITICAL: 4, RiskLevel.HIGH: 3, RiskLevel.MEDIUM: 2, RiskLevel.LOW: 1}
        alerts.sort(key=lambda x: (risk_order[x.risk_level], x.confidence_score), reverse=True)
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an anomaly alert"""
        if alert_id not in self.anomaly_alerts:
            return False
        
        alert = self.anomaly_alerts[alert_id]
        alert.is_acknowledged = True
        alert.baseline_data["acknowledged_by"] = acknowledged_by
        alert.baseline_data["acknowledged_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True
    
    def _calculate_typical_hours(self, hours: List[int]) -> List[int]:
        """Calculate typical hours of activity"""
        if not hours:
            return []
        
        # Count frequency of each hour
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1
        
        # Find hours with significant activity (above average)
        avg_count = len(hours) / 24
        typical_hours = [hour for hour, count in hour_counts.items() if count > avg_count]
        
        return sorted(typical_hours)
    
    def _calculate_typical_locations(self, locations: List[str]) -> List[str]:
        """Calculate typical locations (IP addresses)"""
        if not locations:
            return []
        
        # Count frequency of each location
        location_counts = defaultdict(int)
        for location in locations:
            location_counts[location] += 1
        
        # Return locations that appear in at least 5% of activities
        min_count = max(1, len(locations) * 0.05)
        typical_locations = [loc for loc, count in location_counts.items() if count >= min_count]
        
        return typical_locations
    
    def _calculate_frequency_distribution(self, items: List[str]) -> Dict[str, int]:
        """Calculate frequency distribution of items"""
        distribution = defaultdict(int)
        for item in items:
            distribution[item] += 1
        return dict(distribution)
    
    def _calculate_volume_stats(self, activities: List[UserActivity]) -> Dict[str, float]:
        """Calculate activity volume statistics"""
        if not activities:
            return {}
        
        # Group activities by hour
        hourly_counts = defaultdict(int)
        for activity in activities:
            hour_key = activity.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1
        
        if not hourly_counts:
            return {}
        
        counts = list(hourly_counts.values())
        
        return {
            "avg_hourly": np.mean(counts),
            "std_hourly": np.std(counts),
            "max_hourly": max(counts),
            "min_hourly": min(counts)
        }
    
    def _calculate_user_risk_score(self, user_id: str) -> float:
        """Calculate overall risk score for user"""
        try:
            # Get recent alerts for user
            recent_alerts = [
                alert for alert in self.anomaly_alerts.values()
                if (alert.user_id == user_id and 
                    alert.detected_at > datetime.utcnow() - timedelta(days=7))
            ]
            
            if not recent_alerts:
                return 0.1  # Low risk for users with no recent alerts
            
            # Calculate risk based on alert frequency and severity
            risk_weights = {
                RiskLevel.LOW: 0.1,
                RiskLevel.MEDIUM: 0.3,
                RiskLevel.HIGH: 0.7,
                RiskLevel.CRITICAL: 1.0
            }
            
            total_risk = sum(risk_weights[alert.risk_level] * alert.confidence_score 
                           for alert in recent_alerts)
            
            # Normalize risk score
            max_possible_risk = len(recent_alerts) * 1.0
            normalized_risk = min(total_risk / max_possible_risk if max_possible_risk > 0 else 0, 1.0)
            
            return normalized_risk
            
        except Exception as e:
            logger.error(f"Risk score calculation failed: {str(e)}")
            return 0.5
    
    def _calculate_risk_level(self, confidence: float) -> RiskLevel:
        """Calculate risk level based on confidence score"""
        if confidence >= 0.9:
            return RiskLevel.CRITICAL
        elif confidence >= 0.7:
            return RiskLevel.HIGH
        elif confidence >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW