"""
Behavioral Analytics Engine for real-time anomaly detection and threat hunting
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json

from .ml_siem_engine import SecurityEvent, ThreatLevel

logger = logging.getLogger(__name__)

@dataclass
class UserBehaviorProfile:
    user_id: str
    baseline_features: Dict[str, float]
    typical_hours: List[int]
    typical_locations: List[str]
    typical_resources: List[str]
    risk_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    anomaly_count: int = 0

@dataclass
class BehaviorAnomaly:
    anomaly_id: str
    user_id: str
    event: SecurityEvent
    anomaly_type: str
    severity: ThreatLevel
    confidence: float
    baseline_deviation: float
    detected_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThreatHuntingQuery:
    query_id: str
    name: str
    description: str
    query_logic: Dict[str, Any]
    indicators: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    hit_count: int = 0

class BehavioralAnalyticsEngine:
    """
    Real-time behavioral analytics engine for anomaly detection and threat hunting
    """
    
    def __init__(self, learning_window_days: int = 30):
        self.learning_window_days = learning_window_days
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}
        self.behavior_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # ML models for behavior analysis
        self.anomaly_detector = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=100
        )
        self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        
        # Threat hunting queries
        self.hunting_queries: Dict[str, ThreatHuntingQuery] = {}
        
        # Real-time monitoring
        self.anomaly_buffer: deque = deque(maxlen=1000)
        self.hunting_results: deque = deque(maxlen=500)
        
        # Performance metrics
        self.metrics = {
            'users_profiled': 0,
            'anomalies_detected': 0,
            'hunting_queries_executed': 0,
            'true_positives': 0,
            'false_positives': 0,
            'detection_accuracy': 0.0
        }
        
        # Initialize default hunting queries
        self._initialize_hunting_queries()
    
    async def initialize(self):
        """Initialize the behavioral analytics engine"""
        try:
            # Initialize any async components if needed
            logger.info("Behavioral Analytics Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Behavioral Analytics Engine: {e}")
            raise
    
    def _initialize_hunting_queries(self):
        """Initialize default threat hunting queries"""
        
        # Suspicious login patterns
        login_hunt = ThreatHuntingQuery(
            query_id="hunt_login_001",
            name="Suspicious Login Patterns",
            description="Hunt for unusual login patterns indicating compromise",
            query_logic={
                "conditions": [
                    {"field": "event_type", "operator": "equals", "value": "LOGIN_ATTEMPT"},
                    {"field": "raw_data.success", "operator": "equals", "value": True}
                ],
                "aggregations": [
                    {"field": "source_ip", "function": "count", "threshold": 10},
                    {"field": "user_id", "function": "distinct_count", "threshold": 5}
                ],
                "time_window": "1h"
            },
            indicators=["multiple_ips_per_user", "multiple_users_per_ip", "off_hours_login"]
        )
        self.hunting_queries[login_hunt.query_id] = login_hunt
        
        # Data access anomalies
        data_access_hunt = ThreatHuntingQuery(
            query_id="hunt_data_001",
            name="Abnormal Data Access Patterns",
            description="Hunt for unusual data access indicating insider threat",
            query_logic={
                "conditions": [
                    {"field": "event_type", "operator": "equals", "value": "FILE_ACCESS"}
                ],
                "aggregations": [
                    {"field": "resource", "function": "count", "threshold": 100},
                    {"field": "raw_data.bytes_accessed", "function": "sum", "threshold": 1000000}
                ],
                "time_window": "4h"
            },
            indicators=["bulk_data_access", "sensitive_file_access", "unusual_access_volume"]
        )
        self.hunting_queries[data_access_hunt.query_id] = data_access_hunt
        
        # Privilege escalation hunting
        privesc_hunt = ThreatHuntingQuery(
            query_id="hunt_privesc_001",
            name="Privilege Escalation Indicators",
            description="Hunt for privilege escalation attempts",
            query_logic={
                "conditions": [
                    {"field": "event_type", "operator": "in", "value": ["PRIVILEGE_ESCALATION", "FILE_ACCESS"]}
                ],
                "sequence": [
                    {"event_type": "PRIVILEGE_ESCALATION"},
                    {"event_type": "FILE_ACCESS", "within_minutes": 30}
                ]
            },
            indicators=["admin_access_after_escalation", "sensitive_file_after_escalation"]
        )
        self.hunting_queries[privesc_hunt.query_id] = privesc_hunt
    
    async def analyze_user_behavior(self, event: SecurityEvent) -> Optional[BehaviorAnomaly]:
        """
        Analyze user behavior and detect anomalies in real-time
        """
        if not event.user_id:
            return None
        
        try:
            # Get or create user profile
            profile = await self._get_or_create_profile(event.user_id)
            
            # Add event to behavior buffer
            self.behavior_buffer[event.user_id].append(event)
            
            # Extract behavior features
            features = self._extract_behavior_features(event, profile)
            
            # Detect anomalies
            anomaly = await self._detect_behavior_anomaly(event, profile, features)
            
            # Update profile
            await self._update_user_profile(profile, event, features)
            
            if anomaly:
                self.anomaly_buffer.append(anomaly)
                self.metrics['anomalies_detected'] += 1
                logger.info(f"Behavior anomaly detected for user {event.user_id}: {anomaly.anomaly_type}")
            
            return anomaly
            
        except Exception as e:
            logger.error(f"Error analyzing behavior for user {event.user_id}: {e}")
            return None
    
    async def _get_or_create_profile(self, user_id: str) -> UserBehaviorProfile:
        """Get existing user profile or create new one"""
        if user_id not in self.user_profiles:
            # Create new profile with default baseline
            profile = UserBehaviorProfile(
                user_id=user_id,
                baseline_features={
                    'avg_events_per_hour': 10.0,
                    'unique_resources_per_day': 5.0,
                    'avg_risk_score': 0.1,
                    'typical_session_duration': 480.0,  # 8 hours in minutes
                    'data_access_volume': 1000.0
                },
                typical_hours=list(range(8, 18)),  # 8 AM to 6 PM
                typical_locations=['office_network'],
                typical_resources=['email', 'documents', 'applications']
            )
            
            self.user_profiles[user_id] = profile
            self.metrics['users_profiled'] += 1
            
            # Learn baseline from historical data if available
            await self._learn_baseline(profile)
        
        return self.user_profiles[user_id]
    
    async def _learn_baseline(self, profile: UserBehaviorProfile):
        """Learn baseline behavior from historical events"""
        user_events = list(self.behavior_buffer[profile.user_id])
        
        if len(user_events) < 10:
            return  # Not enough data
        
        # Calculate baseline features
        hourly_activity = defaultdict(int)
        daily_resources = defaultdict(set)
        risk_scores = []
        locations = set()
        
        for event in user_events[-100:]:  # Use last 100 events
            hourly_activity[event.timestamp.hour] += 1
            daily_resources[event.timestamp.date()].add(event.resource)
            risk_scores.append(event.risk_score)
            
            # Extract location from IP (simplified)
            if event.source_ip.startswith('192.168'):
                locations.add('office_network')
            elif event.source_ip.startswith('10.0'):
                locations.add('vpn_network')
            else:
                locations.add('external_network')
        
        # Update profile
        profile.typical_hours = [hour for hour, count in hourly_activity.items() if count > 2]
        profile.typical_locations = list(locations)
        profile.typical_resources = list(set(event.resource for event in user_events[-50:]))
        
        profile.baseline_features.update({
            'avg_events_per_hour': len(user_events) / max(1, len(set(e.timestamp.hour for e in user_events))),
            'unique_resources_per_day': np.mean([len(resources) for resources in daily_resources.values()]) if daily_resources else 5.0,
            'avg_risk_score': np.mean(risk_scores) if risk_scores else 0.1
        })
    
    def _extract_behavior_features(self, event: SecurityEvent, profile: UserBehaviorProfile) -> Dict[str, float]:
        """Extract behavioral features from event"""
        current_hour = event.timestamp.hour
        current_day = event.timestamp.weekday()
        
        # Time-based features
        is_typical_hour = 1.0 if current_hour in profile.typical_hours else 0.0
        is_weekend = 1.0 if current_day >= 5 else 0.0
        is_night = 1.0 if current_hour < 6 or current_hour > 22 else 0.0
        
        # Location features
        location = self._get_location_from_ip(event.source_ip)
        is_typical_location = 1.0 if location in profile.typical_locations else 0.0
        
        # Resource features
        is_typical_resource = 1.0 if event.resource in profile.typical_resources else 0.0
        
        # Activity features
        recent_events = [e for e in self.behavior_buffer[event.user_id] 
                        if (event.timestamp - e.timestamp).total_seconds() < 3600]  # Last hour
        events_last_hour = len(recent_events)
        
        # Risk features
        risk_deviation = abs(event.risk_score - profile.baseline_features['avg_risk_score'])
        
        return {
            'is_typical_hour': is_typical_hour,
            'is_weekend': is_weekend,
            'is_night': is_night,
            'is_typical_location': is_typical_location,
            'is_typical_resource': is_typical_resource,
            'events_last_hour': events_last_hour,
            'risk_deviation': risk_deviation,
            'current_risk_score': event.risk_score
        }
    
    def _get_location_from_ip(self, ip: str) -> str:
        """Determine location from IP address (simplified)"""
        if ip.startswith('192.168') or ip.startswith('10.0'):
            return 'office_network'
        elif ip.startswith('172.16'):
            return 'vpn_network'
        else:
            return 'external_network'
    
    async def _detect_behavior_anomaly(self, event: SecurityEvent, profile: UserBehaviorProfile, 
                                     features: Dict[str, float]) -> Optional[BehaviorAnomaly]:
        """Detect behavioral anomalies using ML models"""
        
        # Calculate anomaly scores for different behavior aspects
        anomalies = []
        
        # Time-based anomaly
        if features['is_night'] and not features['is_typical_hour']:
            anomalies.append(('unusual_time', 0.7, 'Activity during unusual hours'))
        
        # Location-based anomaly
        if not features['is_typical_location']:
            anomalies.append(('unusual_location', 0.6, 'Access from unusual location'))
        
        # Activity volume anomaly
        baseline_activity = profile.baseline_features['avg_events_per_hour']
        if features['events_last_hour'] > baseline_activity * 3:
            anomalies.append(('high_activity', 0.8, 'Unusually high activity volume'))
        
        # Risk score anomaly
        if features['risk_deviation'] > 0.5:
            anomalies.append(('high_risk', 0.9, 'Significantly elevated risk score'))
        
        # Resource access anomaly
        if not features['is_typical_resource'] and event.risk_score > 0.5:
            anomalies.append(('unusual_resource', 0.7, 'Access to unusual high-risk resource'))
        
        # Weekend/holiday activity
        if features['is_weekend'] and features['events_last_hour'] > baseline_activity:
            anomalies.append(('weekend_activity', 0.6, 'High activity during weekend'))
        
        # Select highest severity anomaly
        if anomalies:
            anomaly_type, confidence, description = max(anomalies, key=lambda x: x[1])
            
            # Determine severity based on confidence
            if confidence >= 0.8:
                severity = ThreatLevel.HIGH
            elif confidence >= 0.6:
                severity = ThreatLevel.MEDIUM
            else:
                severity = ThreatLevel.LOW
            
            return BehaviorAnomaly(
                anomaly_id=f"ba_{event.event_id}_{int(datetime.now().timestamp())}",
                user_id=event.user_id,
                event=event,
                anomaly_type=anomaly_type,
                severity=severity,
                confidence=confidence,
                baseline_deviation=features['risk_deviation'],
                context={
                    'description': description,
                    'features': features,
                    'baseline': profile.baseline_features
                }
            )
        
        return None
    
    async def _update_user_profile(self, profile: UserBehaviorProfile, event: SecurityEvent, 
                                 features: Dict[str, float]):
        """Update user profile with new behavior data"""
        
        # Update typical hours (learning)
        current_hour = event.timestamp.hour
        if current_hour not in profile.typical_hours and features['events_last_hour'] > 3:
            profile.typical_hours.append(current_hour)
            profile.typical_hours = sorted(profile.typical_hours)
        
        # Update typical locations
        location = self._get_location_from_ip(event.source_ip)
        if location not in profile.typical_locations and event.risk_score < 0.3:
            profile.typical_locations.append(location)
        
        # Update typical resources
        if event.resource not in profile.typical_resources and event.risk_score < 0.3:
            profile.typical_resources.append(event.resource)
            # Keep only recent resources
            profile.typical_resources = profile.typical_resources[-20:]
        
        # Update baseline features (exponential moving average)
        alpha = 0.1  # Learning rate
        profile.baseline_features['avg_risk_score'] = (
            (1 - alpha) * profile.baseline_features['avg_risk_score'] + 
            alpha * event.risk_score
        )
        
        profile.last_updated = datetime.now()
    
    async def execute_threat_hunting(self) -> List[Dict[str, Any]]:
        """Execute threat hunting queries and return results"""
        hunting_results = []
        
        for query in self.hunting_queries.values():
            try:
                results = await self._execute_hunting_query(query)
                if results:
                    hunting_results.extend(results)
                    query.hit_count += len(results)
                
                query.last_executed = datetime.now()
                self.metrics['hunting_queries_executed'] += 1
                
            except Exception as e:
                logger.error(f"Error executing hunting query {query.query_id}: {e}")
        
        # Store results
        self.hunting_results.extend(hunting_results)
        
        return hunting_results
    
    async def _execute_hunting_query(self, query: ThreatHuntingQuery) -> List[Dict[str, Any]]:
        """Execute individual hunting query"""
        results = []
        
        # Get time window
        time_window = query.query_logic.get('time_window', '1h')
        cutoff_time = datetime.now() - self._parse_time_window(time_window)
        
        # Collect relevant events
        relevant_events = []
        for user_events in self.behavior_buffer.values():
            for event in user_events:
                if event.timestamp >= cutoff_time:
                    if self._event_matches_conditions(event, query.query_logic.get('conditions', [])):
                        relevant_events.append(event)
        
        # Apply aggregations
        aggregations = query.query_logic.get('aggregations', [])
        for agg in aggregations:
            agg_results = self._apply_aggregation(relevant_events, agg)
            if agg_results:
                results.extend(agg_results)
        
        # Check for sequences
        if 'sequence' in query.query_logic:
            sequence_results = self._check_event_sequences(relevant_events, query.query_logic['sequence'])
            results.extend(sequence_results)
        
        # Filter by indicators
        filtered_results = []
        for result in results:
            if any(indicator in result.get('indicators', []) for indicator in query.indicators):
                result['query_id'] = query.query_id
                result['query_name'] = query.name
                result['detected_at'] = datetime.now().isoformat()
                filtered_results.append(result)
        
        return filtered_results
    
    def _parse_time_window(self, time_window: str) -> timedelta:
        """Parse time window string to timedelta"""
        if time_window.endswith('h'):
            return timedelta(hours=int(time_window[:-1]))
        elif time_window.endswith('m'):
            return timedelta(minutes=int(time_window[:-1]))
        elif time_window.endswith('d'):
            return timedelta(days=int(time_window[:-1]))
        else:
            return timedelta(hours=1)  # Default
    
    def _event_matches_conditions(self, event: SecurityEvent, conditions: List[Dict[str, Any]]) -> bool:
        """Check if event matches hunting query conditions"""
        for condition in conditions:
            field = condition['field']
            operator = condition['operator']
            value = condition['value']
            
            # Get field value
            if field == 'event_type':
                field_value = event.event_type.value
            elif field == 'source_ip':
                field_value = event.source_ip
            elif field == 'user_id':
                field_value = event.user_id
            elif field.startswith('raw_data.'):
                key = field.split('.', 1)[1]
                field_value = event.raw_data.get(key)
            else:
                continue
            
            # Apply operator
            if operator == 'equals' and field_value != value:
                return False
            elif operator == 'in' and field_value not in value:
                return False
            elif operator == 'greater_than' and (field_value is None or field_value <= value):
                return False
        
        return True
    
    def _apply_aggregation(self, events: List[SecurityEvent], aggregation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply aggregation to events"""
        field = aggregation['field']
        function = aggregation['function']
        threshold = aggregation['threshold']
        
        # Group events by field
        groups = defaultdict(list)
        for event in events:
            if field == 'source_ip':
                key = event.source_ip
            elif field == 'user_id':
                key = event.user_id
            elif field == 'resource':
                key = event.resource
            else:
                continue
            
            groups[key].append(event)
        
        results = []
        for key, group_events in groups.items():
            if function == 'count':
                value = len(group_events)
            elif function == 'distinct_count':
                if field == 'source_ip':
                    value = len(set(e.user_id for e in group_events if e.user_id))
                elif field == 'user_id':
                    value = len(set(e.source_ip for e in group_events))
                else:
                    value = len(group_events)
            else:
                continue
            
            if value >= threshold:
                results.append({
                    'type': 'aggregation_hit',
                    'field': field,
                    'key': key,
                    'function': function,
                    'value': value,
                    'threshold': threshold,
                    'events': len(group_events),
                    'indicators': [f"{function}_{field}_threshold_exceeded"]
                })
        
        return results
    
    def _check_event_sequences(self, events: List[SecurityEvent], sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for event sequences in hunting query"""
        results = []
        
        if len(sequence) < 2:
            return results
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Look for sequences
        for i, first_event in enumerate(sorted_events):
            if first_event.event_type.value == sequence[0]['event_type']:
                # Look for subsequent events
                sequence_events = [first_event]
                
                for j in range(i + 1, len(sorted_events)):
                    next_event = sorted_events[j]
                    
                    # Check if this matches next sequence step
                    for seq_step in sequence[1:]:
                        if next_event.event_type.value == seq_step['event_type']:
                            # Check time constraint
                            time_diff = (next_event.timestamp - first_event.timestamp).total_seconds() / 60
                            if time_diff <= seq_step.get('within_minutes', 60):
                                sequence_events.append(next_event)
                                break
                
                # Check if we found complete sequence
                if len(sequence_events) >= len(sequence):
                    results.append({
                        'type': 'sequence_detected',
                        'sequence_length': len(sequence_events),
                        'time_span_minutes': (sequence_events[-1].timestamp - sequence_events[0].timestamp).total_seconds() / 60,
                        'events': [e.event_id for e in sequence_events],
                        'indicators': ['event_sequence_detected']
                    })
        
        return results
    
    async def get_user_risk_assessment(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive risk assessment for user"""
        if user_id not in self.user_profiles:
            return {'error': 'User profile not found'}
        
        profile = self.user_profiles[user_id]
        recent_events = list(self.behavior_buffer[user_id])[-50:]  # Last 50 events
        
        # Calculate risk factors
        recent_anomalies = [a for a in self.anomaly_buffer if a.user_id == user_id and 
                           (datetime.now() - a.detected_at).days <= 7]
        
        risk_factors = {
            'anomaly_count_7days': len(recent_anomalies),
            'avg_risk_score': np.mean([e.risk_score for e in recent_events]) if recent_events else 0.0,
            'unusual_hours_activity': sum(1 for e in recent_events if e.timestamp.hour not in profile.typical_hours),
            'external_access_count': sum(1 for e in recent_events if not self._get_location_from_ip(e.source_ip).startswith('office')),
            'high_risk_events': sum(1 for e in recent_events if e.risk_score > 0.7)
        }
        
        # Calculate overall risk score
        overall_risk = min(1.0, (
            risk_factors['anomaly_count_7days'] * 0.2 +
            risk_factors['avg_risk_score'] * 0.3 +
            risk_factors['unusual_hours_activity'] / max(1, len(recent_events)) * 0.2 +
            risk_factors['external_access_count'] / max(1, len(recent_events)) * 0.15 +
            risk_factors['high_risk_events'] / max(1, len(recent_events)) * 0.15
        ))
        
        return {
            'user_id': user_id,
            'overall_risk_score': overall_risk,
            'risk_level': self._get_risk_level(overall_risk),
            'risk_factors': risk_factors,
            'recent_anomalies': len(recent_anomalies),
            'profile_last_updated': profile.last_updated.isoformat(),
            'recommendations': self._get_risk_recommendations(overall_risk, risk_factors)
        }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 0.8:
            return 'CRITICAL'
        elif risk_score >= 0.6:
            return 'HIGH'
        elif risk_score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_risk_recommendations(self, risk_score: float, risk_factors: Dict[str, float]) -> List[str]:
        """Get risk mitigation recommendations"""
        recommendations = []
        
        if risk_score >= 0.7:
            recommendations.append("Consider immediate security review")
        
        if risk_factors['anomaly_count_7days'] > 5:
            recommendations.append("Investigate recent anomalous activities")
        
        if risk_factors['external_access_count'] > 10:
            recommendations.append("Review external access patterns")
        
        if risk_factors['unusual_hours_activity'] > 5:
            recommendations.append("Verify after-hours activity legitimacy")
        
        if risk_factors['high_risk_events'] > 3:
            recommendations.append("Review high-risk event details")
        
        return recommendations
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get behavioral analytics performance metrics"""
        return {
            'users_profiled': self.metrics['users_profiled'],
            'anomalies_detected': self.metrics['anomalies_detected'],
            'hunting_queries_executed': self.metrics['hunting_queries_executed'],
            'active_hunting_queries': len(self.hunting_queries),
            'detection_accuracy': self.metrics.get('detection_accuracy', 0.0),
            'avg_anomalies_per_user': (
                self.metrics['anomalies_detected'] / max(1, self.metrics['users_profiled'])
            ),
            'recent_hunting_hits': len([r for r in self.hunting_results 
                                      if (datetime.now() - datetime.fromisoformat(r['detected_at'])).hours <= 24])
        }