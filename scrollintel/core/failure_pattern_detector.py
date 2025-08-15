"""
Failure Pattern Detection System

This module provides advanced failure pattern detection with proactive prevention,
machine learning-based pattern recognition, and automated response systems.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import defaultdict, deque, Counter
import re
import hashlib

logger = logging.getLogger(__name__)

class FailureType(Enum):
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    USER_ERROR = "user_error"
    NETWORK_ISSUE = "network_issue"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    DATA_CORRUPTION = "data_corruption"

class PatternSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class PreventionAction(Enum):
    SCALE_RESOURCES = "scale_resources"
    RESTART_SERVICE = "restart_service"
    CLEAR_CACHE = "clear_cache"
    SWITCH_DEPENDENCY = "switch_dependency"
    ENABLE_CIRCUIT_BREAKER = "enable_circuit_breaker"
    INCREASE_TIMEOUT = "increase_timeout"
    REDUCE_LOAD = "reduce_load"
    ALERT_OPERATORS = "alert_operators"

@dataclass
class FailureEvent:
    """Individual failure event"""
    id: str
    timestamp: datetime
    failure_type: FailureType
    component: str
    error_message: str
    stack_trace: Optional[str]
    user_id: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
    context: Dict[str, Any]
    severity: PatternSeverity
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class FailurePattern:
    """Detected failure pattern"""
    pattern_id: str
    pattern_type: str
    failure_type: FailureType
    component: str
    frequency: int
    first_occurrence: datetime
    last_occurrence: datetime
    pattern_signature: str
    confidence: float
    severity: PatternSeverity
    description: str
    root_cause_hypothesis: str
    prevention_actions: List[PreventionAction]
    similar_events: List[str]  # Event IDs
    trend: str  # increasing, decreasing, stable
    
@dataclass
class PreventionRule:
    """Proactive prevention rule"""
    rule_id: str
    pattern_signature: str
    trigger_conditions: Dict[str, Any]
    prevention_actions: List[PreventionAction]
    success_rate: float
    last_triggered: Optional[datetime]
    enabled: bool = True

class FailurePatternDetector:
    """
    Advanced failure pattern detection system with proactive prevention
    """
    
    def __init__(self):
        self.failure_events = deque(maxlen=100000)
        self.detected_patterns = {}
        self.prevention_rules = {}
        self.pattern_signatures = {}
        self.component_baselines = {}
        self.detection_active = False
        
        # Pattern detection parameters
        self.min_pattern_frequency = 3
        self.pattern_time_window = timedelta(hours=24)
        self.similarity_threshold = 0.8
        
        # Initialize default prevention rules
        self._setup_default_prevention_rules()
    
    def _setup_default_prevention_rules(self):
        """Setup default prevention rules"""
        self.prevention_rules = {
            "memory_leak_pattern": PreventionRule(
                rule_id="memory_leak_prevention",
                pattern_signature="memory_usage_increasing",
                trigger_conditions={
                    "memory_trend": "increasing",
                    "duration_hours": 2,
                    "increase_rate": 0.1  # 10% per hour
                },
                prevention_actions=[PreventionAction.RESTART_SERVICE, PreventionAction.CLEAR_CACHE],
                success_rate=0.85,
                last_triggered=None
            ),
            "cascade_failure_pattern": PreventionRule(
                rule_id="cascade_failure_prevention",
                pattern_signature="dependency_failure_cascade",
                trigger_conditions={
                    "failure_rate": 0.1,
                    "affected_components": 3,
                    "time_window_minutes": 15
                },
                prevention_actions=[PreventionAction.ENABLE_CIRCUIT_BREAKER, PreventionAction.REDUCE_LOAD],
                success_rate=0.9,
                last_triggered=None
            ),
            "resource_exhaustion_pattern": PreventionRule(
                rule_id="resource_exhaustion_prevention",
                pattern_signature="resource_usage_spike",
                trigger_conditions={
                    "cpu_usage": 0.8,
                    "memory_usage": 0.8,
                    "sustained_minutes": 10
                },
                prevention_actions=[PreventionAction.SCALE_RESOURCES, PreventionAction.REDUCE_LOAD],
                success_rate=0.8,
                last_triggered=None
            )
        }
    
    async def start_detection(self):
        """Start failure pattern detection"""
        self.detection_active = True
        logger.info("Failure Pattern Detector started")
        
        tasks = [
            asyncio.create_task(self._detect_patterns()),
            asyncio.create_task(self._monitor_prevention_rules()),
            asyncio.create_task(self._update_baselines()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop_detection(self):
        """Stop failure pattern detection"""
        self.detection_active = False
        logger.info("Failure Pattern Detector stopped")
    
    def record_failure(self, failure_type: FailureType, component: str, error_message: str,
                      stack_trace: str = None, user_id: str = None, session_id: str = None,
                      request_id: str = None, context: Dict[str, Any] = None,
                      severity: PatternSeverity = PatternSeverity.MEDIUM) -> str:
        """Record a failure event"""
        event_id = f"failure_{int(datetime.now().timestamp())}_{hash(error_message) % 10000}"
        
        event = FailureEvent(
            id=event_id,
            timestamp=datetime.now(),
            failure_type=failure_type,
            component=component,
            error_message=error_message,
            stack_trace=stack_trace,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            context=context or {},
            severity=severity
        )
        
        self.failure_events.append(event)
        logger.info(f"Failure recorded: {event_id} - {component}: {error_message}")
        
        return event_id
    
    async def _detect_patterns(self):
        """Main pattern detection loop"""
        while self.detection_active:
            try:
                current_time = datetime.now()
                cutoff_time = current_time - self.pattern_time_window
                
                # Get recent failure events
                recent_events = [e for e in self.failure_events if e.timestamp >= cutoff_time]
                
                if len(recent_events) < self.min_pattern_frequency:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    continue
                
                # Detect patterns by different criteria
                await self._detect_error_message_patterns(recent_events)
                await self._detect_component_failure_patterns(recent_events)
                await self._detect_temporal_patterns(recent_events)
                await self._detect_user_behavior_patterns(recent_events)
                await self._detect_cascade_patterns(recent_events)
                
                # Update pattern trends
                self._update_pattern_trends()
                
                await asyncio.sleep(300)  # Run detection every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in pattern detection: {e}")
                await asyncio.sleep(600)
    
    async def _detect_error_message_patterns(self, events: List[FailureEvent]):
        """Detect patterns in error messages"""
        try:
            # Group events by similar error messages
            error_groups = defaultdict(list)
            
            for event in events:
                # Create a normalized error signature
                normalized_error = self._normalize_error_message(event.error_message)
                error_groups[normalized_error].append(event)
            
            # Look for patterns in groups with sufficient frequency
            for normalized_error, group_events in error_groups.items():
                if len(group_events) >= self.min_pattern_frequency:
                    pattern_id = f"error_pattern_{hashlib.md5(normalized_error.encode()).hexdigest()[:8]}"
                    
                    if pattern_id not in self.detected_patterns:
                        # Analyze the pattern
                        components = [e.component for e in group_events]
                        most_common_component = Counter(components).most_common(1)[0][0]
                        
                        # Determine severity based on frequency and component criticality
                        severity = self._calculate_pattern_severity(group_events)
                        
                        # Generate prevention actions
                        prevention_actions = self._generate_prevention_actions(group_events)
                        
                        pattern = FailurePattern(
                            pattern_id=pattern_id,
                            pattern_type="error_message_pattern",
                            failure_type=group_events[0].failure_type,
                            component=most_common_component,
                            frequency=len(group_events),
                            first_occurrence=min(e.timestamp for e in group_events),
                            last_occurrence=max(e.timestamp for e in group_events),
                            pattern_signature=normalized_error,
                            confidence=min(0.95, len(group_events) / 10.0),
                            severity=severity,
                            description=f"Recurring error pattern: {normalized_error[:100]}...",
                            root_cause_hypothesis=self._generate_root_cause_hypothesis(group_events),
                            prevention_actions=prevention_actions,
                            similar_events=[e.id for e in group_events],
                            trend="stable"
                        )
                        
                        self.detected_patterns[pattern_id] = pattern
                        await self._trigger_pattern_alert(pattern)
        
        except Exception as e:
            logger.error(f"Error detecting error message patterns: {e}")
    
    async def _detect_component_failure_patterns(self, events: List[FailureEvent]):
        """Detect patterns in component failures"""
        try:
            # Group events by component
            component_groups = defaultdict(list)
            
            for event in events:
                component_groups[event.component].append(event)
            
            # Analyze each component's failure pattern
            for component, component_events in component_groups.items():
                if len(component_events) >= self.min_pattern_frequency:
                    # Check for increasing failure rate
                    failure_rate_trend = self._analyze_failure_rate_trend(component_events)
                    
                    if failure_rate_trend == "increasing":
                        pattern_id = f"component_degradation_{component}_{int(datetime.now().timestamp())}"
                        
                        pattern = FailurePattern(
                            pattern_id=pattern_id,
                            pattern_type="component_degradation",
                            failure_type=FailureType.PERFORMANCE_DEGRADATION,
                            component=component,
                            frequency=len(component_events),
                            first_occurrence=min(e.timestamp for e in component_events),
                            last_occurrence=max(e.timestamp for e in component_events),
                            pattern_signature=f"component_failure_increase_{component}",
                            confidence=0.8,
                            severity=self._calculate_pattern_severity(component_events),
                            description=f"Increasing failure rate in component: {component}",
                            root_cause_hypothesis=f"Component {component} may be experiencing degradation or resource issues",
                            prevention_actions=[PreventionAction.RESTART_SERVICE, PreventionAction.SCALE_RESOURCES],
                            similar_events=[e.id for e in component_events],
                            trend="increasing"
                        )
                        
                        self.detected_patterns[pattern_id] = pattern
                        await self._trigger_pattern_alert(pattern)
        
        except Exception as e:
            logger.error(f"Error detecting component failure patterns: {e}")
    
    async def _detect_temporal_patterns(self, events: List[FailureEvent]):
        """Detect temporal patterns in failures"""
        try:
            # Group events by hour of day
            hourly_failures = defaultdict(list)
            
            for event in events:
                hour = event.timestamp.hour
                hourly_failures[hour].append(event)
            
            # Look for time-based patterns
            if len(hourly_failures) >= 3:
                failure_counts = [len(hourly_failures.get(hour, [])) for hour in range(24)]
                
                # Find peak failure hours
                max_failures = max(failure_counts)
                avg_failures = statistics.mean(failure_counts)
                
                if max_failures > avg_failures * 2:  # Significant spike
                    peak_hours = [hour for hour, count in enumerate(failure_counts) if count == max_failures]
                    
                    pattern_id = f"temporal_pattern_{int(datetime.now().timestamp())}"
                    
                    pattern = FailurePattern(
                        pattern_id=pattern_id,
                        pattern_type="temporal_pattern",
                        failure_type=FailureType.SYSTEM_ERROR,
                        component="system",
                        frequency=max_failures,
                        first_occurrence=min(e.timestamp for e in events),
                        last_occurrence=max(e.timestamp for e in events),
                        pattern_signature=f"peak_failures_hours_{peak_hours}",
                        confidence=0.7,
                        severity=PatternSeverity.MEDIUM,
                        description=f"Failure spike during hours: {peak_hours}",
                        root_cause_hypothesis="Time-based load or scheduled process causing failures",
                        prevention_actions=[PreventionAction.SCALE_RESOURCES, PreventionAction.REDUCE_LOAD],
                        similar_events=[e.id for events_list in hourly_failures.values() for e in events_list],
                        trend="stable"
                    )
                    
                    self.detected_patterns[pattern_id] = pattern
                    await self._trigger_pattern_alert(pattern)
        
        except Exception as e:
            logger.error(f"Error detecting temporal patterns: {e}")
    
    async def _detect_user_behavior_patterns(self, events: List[FailureEvent]):
        """Detect patterns related to user behavior"""
        try:
            # Group events by user
            user_events = defaultdict(list)
            
            for event in events:
                if event.user_id:
                    user_events[event.user_id].append(event)
            
            # Look for users with high failure rates
            problematic_users = []
            for user_id, user_failures in user_events.items():
                if len(user_failures) >= 5:  # User with many failures
                    problematic_users.append((user_id, user_failures))
            
            if problematic_users:
                # Check if it's a pattern across multiple users or specific users
                if len(problematic_users) >= 3:
                    # Multiple users having issues - likely system problem
                    all_user_events = [e for _, events in problematic_users for e in events]
                    
                    pattern_id = f"user_behavior_pattern_{int(datetime.now().timestamp())}"
                    
                    pattern = FailurePattern(
                        pattern_id=pattern_id,
                        pattern_type="user_behavior_pattern",
                        failure_type=FailureType.USER_ERROR,
                        component="user_interface",
                        frequency=len(all_user_events),
                        first_occurrence=min(e.timestamp for e in all_user_events),
                        last_occurrence=max(e.timestamp for e in all_user_events),
                        pattern_signature="multiple_users_high_failure_rate",
                        confidence=0.8,
                        severity=PatternSeverity.HIGH,
                        description=f"Multiple users experiencing high failure rates",
                        root_cause_hypothesis="UI/UX issue causing user errors or system instability",
                        prevention_actions=[PreventionAction.ALERT_OPERATORS],
                        similar_events=[e.id for e in all_user_events],
                        trend="stable"
                    )
                    
                    self.detected_patterns[pattern_id] = pattern
                    await self._trigger_pattern_alert(pattern)
        
        except Exception as e:
            logger.error(f"Error detecting user behavior patterns: {e}")
    
    async def _detect_cascade_patterns(self, events: List[FailureEvent]):
        """Detect cascade failure patterns"""
        try:
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda x: x.timestamp)
            
            # Look for rapid succession of failures across different components
            cascade_window = timedelta(minutes=15)
            cascade_groups = []
            current_group = []
            
            for event in sorted_events:
                if not current_group:
                    current_group = [event]
                else:
                    time_diff = event.timestamp - current_group[-1].timestamp
                    if time_diff <= cascade_window:
                        current_group.append(event)
                    else:
                        if len(current_group) >= 3:  # Potential cascade
                            cascade_groups.append(current_group)
                        current_group = [event]
            
            # Check final group
            if len(current_group) >= 3:
                cascade_groups.append(current_group)
            
            # Analyze cascade groups
            for cascade_events in cascade_groups:
                components = set(e.component for e in cascade_events)
                
                if len(components) >= 2:  # Multiple components affected
                    pattern_id = f"cascade_pattern_{int(cascade_events[0].timestamp.timestamp())}"
                    
                    pattern = FailurePattern(
                        pattern_id=pattern_id,
                        pattern_type="cascade_failure",
                        failure_type=FailureType.DEPENDENCY_FAILURE,
                        component="multiple",
                        frequency=len(cascade_events),
                        first_occurrence=cascade_events[0].timestamp,
                        last_occurrence=cascade_events[-1].timestamp,
                        pattern_signature=f"cascade_{len(components)}_components",
                        confidence=0.9,
                        severity=PatternSeverity.CRITICAL,
                        description=f"Cascade failure across {len(components)} components",
                        root_cause_hypothesis="Initial failure triggered cascade across dependent components",
                        prevention_actions=[PreventionAction.ENABLE_CIRCUIT_BREAKER, PreventionAction.RESTART_SERVICE],
                        similar_events=[e.id for e in cascade_events],
                        trend="stable"
                    )
                    
                    self.detected_patterns[pattern_id] = pattern
                    await self._trigger_pattern_alert(pattern)
        
        except Exception as e:
            logger.error(f"Error detecting cascade patterns: {e}")
    
    def _normalize_error_message(self, error_message: str) -> str:
        """Normalize error message for pattern matching"""
        # Remove specific IDs, timestamps, and variable data
        normalized = re.sub(r'\b\d+\b', 'NUM', error_message)  # Replace numbers
        normalized = re.sub(r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b', 'UUID', normalized)  # UUIDs
        normalized = re.sub(r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b', 'TIMESTAMP', normalized)  # ISO timestamps
        normalized = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'IP', normalized)  # IP addresses
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        
        return normalized.strip().lower()
    
    def _calculate_pattern_severity(self, events: List[FailureEvent]) -> PatternSeverity:
        """Calculate pattern severity based on events"""
        frequency = len(events)
        time_span = (max(e.timestamp for e in events) - min(e.timestamp for e in events)).total_seconds() / 3600  # hours
        
        # Consider frequency and rate
        if frequency >= 20 or (frequency >= 10 and time_span <= 1):
            return PatternSeverity.CRITICAL
        elif frequency >= 10 or (frequency >= 5 and time_span <= 1):
            return PatternSeverity.HIGH
        elif frequency >= 5:
            return PatternSeverity.MEDIUM
        else:
            return PatternSeverity.LOW
    
    def _generate_prevention_actions(self, events: List[FailureEvent]) -> List[PreventionAction]:
        """Generate prevention actions based on failure events"""
        actions = []
        
        # Analyze failure types
        failure_types = [e.failure_type for e in events]
        most_common_type = Counter(failure_types).most_common(1)[0][0]
        
        if most_common_type == FailureType.RESOURCE_EXHAUSTION:
            actions.extend([PreventionAction.SCALE_RESOURCES, PreventionAction.CLEAR_CACHE])
        elif most_common_type == FailureType.DEPENDENCY_FAILURE:
            actions.extend([PreventionAction.ENABLE_CIRCUIT_BREAKER, PreventionAction.SWITCH_DEPENDENCY])
        elif most_common_type == FailureType.PERFORMANCE_DEGRADATION:
            actions.extend([PreventionAction.RESTART_SERVICE, PreventionAction.REDUCE_LOAD])
        elif most_common_type == FailureType.NETWORK_ISSUE:
            actions.extend([PreventionAction.INCREASE_TIMEOUT, PreventionAction.ENABLE_CIRCUIT_BREAKER])
        else:
            actions.append(PreventionAction.ALERT_OPERATORS)
        
        return actions
    
    def _generate_root_cause_hypothesis(self, events: List[FailureEvent]) -> str:
        """Generate root cause hypothesis based on events"""
        components = [e.component for e in events]
        most_common_component = Counter(components).most_common(1)[0][0]
        
        failure_types = [e.failure_type for e in events]
        most_common_type = Counter(failure_types).most_common(1)[0][0]
        
        time_span = (max(e.timestamp for e in events) - min(e.timestamp for e in events)).total_seconds() / 3600
        
        if time_span < 1:
            return f"Rapid failure burst in {most_common_component} suggests sudden {most_common_type.value} issue"
        elif len(set(components)) > 1:
            return f"Multiple component failures suggest system-wide {most_common_type.value} or cascade effect"
        else:
            return f"Recurring {most_common_type.value} in {most_common_component} suggests component-specific issue"
    
    def _analyze_failure_rate_trend(self, events: List[FailureEvent]) -> str:
        """Analyze failure rate trend"""
        if len(events) < 6:
            return "stable"
        
        # Sort by timestamp
        sorted_events = sorted(events, key=lambda x: x.timestamp)
        
        # Split into two halves and compare rates
        mid_point = len(sorted_events) // 2
        first_half = sorted_events[:mid_point]
        second_half = sorted_events[mid_point:]
        
        first_half_duration = (first_half[-1].timestamp - first_half[0].timestamp).total_seconds() / 3600
        second_half_duration = (second_half[-1].timestamp - second_half[0].timestamp).total_seconds() / 3600
        
        if first_half_duration > 0 and second_half_duration > 0:
            first_rate = len(first_half) / first_half_duration
            second_rate = len(second_half) / second_half_duration
            
            if second_rate > first_rate * 1.5:
                return "increasing"
            elif second_rate < first_rate * 0.7:
                return "decreasing"
        
        return "stable"
    
    def _update_pattern_trends(self):
        """Update trends for existing patterns"""
        current_time = datetime.now()
        
        for pattern in self.detected_patterns.values():
            # Get recent events for this pattern
            recent_events = [
                e for e in self.failure_events
                if e.timestamp >= current_time - timedelta(hours=6) and
                   e.id in pattern.similar_events
            ]
            
            if len(recent_events) >= 3:
                pattern.trend = self._analyze_failure_rate_trend(recent_events)
                pattern.last_occurrence = max(e.timestamp for e in recent_events)
    
    async def _monitor_prevention_rules(self):
        """Monitor and trigger prevention rules"""
        while self.detection_active:
            try:
                current_time = datetime.now()
                
                for rule in self.prevention_rules.values():
                    if not rule.enabled:
                        continue
                    
                    # Check if rule conditions are met
                    if await self._evaluate_prevention_rule(rule):
                        await self._trigger_prevention_action(rule)
                        rule.last_triggered = current_time
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring prevention rules: {e}")
                await asyncio.sleep(300)
    
    async def _evaluate_prevention_rule(self, rule: PreventionRule) -> bool:
        """Evaluate if a prevention rule should be triggered"""
        # This is a simplified evaluation - in practice, this would be more sophisticated
        current_time = datetime.now()
        
        # Don't trigger too frequently
        if rule.last_triggered and (current_time - rule.last_triggered).total_seconds() < 3600:
            return False
        
        # Check if we have a matching pattern
        for pattern in self.detected_patterns.values():
            if pattern.pattern_signature == rule.pattern_signature:
                # Check if pattern is recent and active
                if (current_time - pattern.last_occurrence).total_seconds() < 1800:  # 30 minutes
                    return True
        
        return False
    
    async def _trigger_prevention_action(self, rule: PreventionRule):
        """Trigger prevention actions"""
        logger.warning(f"PREVENTION RULE TRIGGERED: {rule.rule_id}")
        
        for action in rule.prevention_actions:
            await self._execute_prevention_action(action, rule)
    
    async def _execute_prevention_action(self, action: PreventionAction, rule: PreventionRule):
        """Execute a specific prevention action"""
        logger.info(f"Executing prevention action: {action.value}")
        
        # In a real implementation, these would trigger actual system actions
        action_data = {
            "action": action.value,
            "rule_id": rule.rule_id,
            "timestamp": datetime.now().isoformat(),
            "expected_success_rate": rule.success_rate
        }
        
        logger.info(f"Prevention action data: {json.dumps(action_data, indent=2)}")
    
    async def _update_baselines(self):
        """Update component baselines"""
        while self.detection_active:
            try:
                current_time = datetime.now()
                baseline_window = current_time - timedelta(days=7)
                
                # Calculate baselines for each component
                component_events = defaultdict(list)
                
                for event in self.failure_events:
                    if event.timestamp >= baseline_window:
                        component_events[event.component].append(event)
                
                for component, events in component_events.items():
                    if len(events) >= 10:  # Minimum events for baseline
                        # Calculate failure rate baseline
                        total_hours = (current_time - baseline_window).total_seconds() / 3600
                        failure_rate = len(events) / total_hours
                        
                        self.component_baselines[component] = {
                            "failure_rate": failure_rate,
                            "last_updated": current_time,
                            "sample_size": len(events)
                        }
                
                await asyncio.sleep(3600)  # Update baselines every hour
                
            except Exception as e:
                logger.error(f"Error updating baselines: {e}")
                await asyncio.sleep(7200)
    
    async def _cleanup_old_data(self):
        """Clean up old patterns and data"""
        while self.detection_active:
            try:
                current_time = datetime.now()
                cleanup_threshold = current_time - timedelta(days=30)
                
                # Remove old patterns
                old_patterns = [
                    pattern_id for pattern_id, pattern in self.detected_patterns.items()
                    if pattern.last_occurrence < cleanup_threshold
                ]
                
                for pattern_id in old_patterns:
                    del self.detected_patterns[pattern_id]
                
                logger.info(f"Cleaned up {len(old_patterns)} old patterns")
                
                await asyncio.sleep(86400)  # Clean up daily
                
            except Exception as e:
                logger.error(f"Error cleaning up old data: {e}")
                await asyncio.sleep(86400)
    
    async def _trigger_pattern_alert(self, pattern: FailurePattern):
        """Trigger alert for detected pattern"""
        logger.warning(f"FAILURE PATTERN DETECTED: {pattern.description}")
        
        pattern_data = {
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type,
            "component": pattern.component,
            "frequency": pattern.frequency,
            "severity": pattern.severity.value,
            "confidence": pattern.confidence,
            "description": pattern.description,
            "root_cause_hypothesis": pattern.root_cause_hypothesis,
            "prevention_actions": [action.value for action in pattern.prevention_actions]
        }
        
        logger.info(f"Pattern data: {json.dumps(pattern_data, indent=2)}")
    
    # Public API methods
    
    def get_detected_patterns(self, severity: PatternSeverity = None) -> List[Dict[str, Any]]:
        """Get detected failure patterns"""
        patterns = list(self.detected_patterns.values())
        
        if severity:
            patterns = [p for p in patterns if p.severity == severity]
        
        return [asdict(pattern) for pattern in sorted(patterns, key=lambda x: x.last_occurrence, reverse=True)]
    
    def get_component_health(self, component: str = None) -> Dict[str, Any]:
        """Get component health information"""
        current_time = datetime.now()
        recent_cutoff = current_time - timedelta(hours=24)
        
        if component:
            components = [component]
        else:
            components = list(set(e.component for e in self.failure_events))
        
        health_data = {}
        
        for comp in components:
            recent_failures = [e for e in self.failure_events 
                             if e.component == comp and e.timestamp >= recent_cutoff]
            
            baseline = self.component_baselines.get(comp, {"failure_rate": 0})
            current_rate = len(recent_failures) / 24  # failures per hour
            
            health_data[comp] = {
                "recent_failures": len(recent_failures),
                "failure_rate": current_rate,
                "baseline_rate": baseline.get("failure_rate", 0),
                "health_status": "healthy" if current_rate <= baseline.get("failure_rate", 0) * 1.5 else "degraded",
                "patterns_detected": len([p for p in self.detected_patterns.values() if p.component == comp])
            }
        
        return health_data
    
    def get_prevention_status(self) -> Dict[str, Any]:
        """Get prevention system status"""
        current_time = datetime.now()
        
        active_rules = sum(1 for rule in self.prevention_rules.values() if rule.enabled)
        recently_triggered = sum(
            1 for rule in self.prevention_rules.values() 
            if rule.last_triggered and (current_time - rule.last_triggered).total_seconds() < 3600
        )
        
        return {
            "total_rules": len(self.prevention_rules),
            "active_rules": active_rules,
            "recently_triggered": recently_triggered,
            "average_success_rate": statistics.mean([rule.success_rate for rule in self.prevention_rules.values()]),
            "patterns_with_prevention": len([p for p in self.detected_patterns.values() if p.prevention_actions])
        }

# Global failure pattern detector instance
failure_pattern_detector = FailurePatternDetector()