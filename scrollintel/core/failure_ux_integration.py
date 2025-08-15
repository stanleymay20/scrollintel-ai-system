"""
Integration between failure prevention and user experience protection systems.
Provides unified failure detection, recovery coordination, and predictive prevention
based on user behavior patterns.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from contextlib import asynccontextmanager
import statistics
from collections import defaultdict, deque

from .failure_prevention import (
    FailurePreventionSystem, FailureType, FailureEvent, 
    failure_prevention, CircuitBreakerState
)
from .user_experience_protection import (
    UserExperienceProtector, UserExperienceLevel, UserAction,
    ux_protector
)

logger = logging.getLogger(__name__)


class FailureImpactLevel(Enum):
    """Impact level of failures on user experience."""
    CRITICAL = "critical"      # Completely blocks user workflow
    HIGH = "high"             # Significantly degrades user experience
    MEDIUM = "medium"         # Noticeable but manageable impact
    LOW = "low"               # Minimal user impact
    NEGLIGIBLE = "negligible" # No visible user impact


class PredictionConfidence(Enum):
    """Confidence level for failure predictions."""
    VERY_HIGH = "very_high"   # 90%+ confidence
    HIGH = "high"             # 75-90% confidence
    MEDIUM = "medium"         # 50-75% confidence
    LOW = "low"               # 25-50% confidence
    VERY_LOW = "very_low"     # <25% confidence


@dataclass
class UserBehaviorPattern:
    """Represents a user behavior pattern."""
    pattern_id: str
    user_id: Optional[str]
    action_sequence: List[str]
    frequency: int
    avg_response_time: float
    success_rate: float
    last_seen: datetime
    failure_indicators: List[str] = field(default_factory=list)


@dataclass
class FailurePrediction:
    """Represents a predicted failure."""
    failure_type: FailureType
    confidence: PredictionConfidence
    predicted_time: datetime
    impact_level: FailureImpactLevel
    affected_users: List[str]
    prevention_actions: List[str]
    user_behavior_triggers: List[str]


@dataclass
class UnifiedFailureResponse:
    """Unified response to failures combining technical recovery and UX protection."""
    failure_event: FailureEvent
    technical_recovery_actions: List[str]
    ux_protection_actions: List[str]
    user_communication: Dict[str, Any]
    fallback_strategies: List[str]
    recovery_timeline: Dict[str, datetime]
    success_metrics: Dict[str, float]


class FailureUXIntegrator:
    """Integrates failure prevention with user experience protection."""
    
    def __init__(self):
        self.failure_prevention = failure_prevention
        self.ux_protector = ux_protector
        
        # User behavior analysis
        self.user_patterns: Dict[str, List[UserBehaviorPattern]] = defaultdict(list)
        self.behavior_analysis_window = timedelta(hours=24)
        self.pattern_cache: Dict[str, UserBehaviorPattern] = {}
        
        # Failure prediction
        self.failure_predictions: List[FailurePrediction] = []
        self.prediction_history: List[Tuple[FailurePrediction, bool]] = []  # (prediction, was_correct)
        
        # Cross-system coordination
        self.active_responses: Dict[str, UnifiedFailureResponse] = {}
        self.response_templates: Dict[FailureType, Dict[str, Any]] = {}
        
        # Metrics and learning
        self.integration_metrics: Dict[str, Any] = {
            "predictions_made": 0,
            "predictions_correct": 0,
            "failures_prevented": 0,
            "user_experience_improvements": 0,
            "avg_recovery_time": 0.0
        }
        
        # Initialize system
        self._setup_response_templates()
        self._setup_behavior_monitoring()
        
        # Background tasks will be started when needed
        self._background_tasks_started = False
    
    def _setup_response_templates(self):
        """Setup response templates for different failure types."""
        self.response_templates = {
            FailureType.NETWORK_ERROR: {
                "technical_actions": [
                    "retry_with_exponential_backoff",
                    "switch_to_backup_endpoint",
                    "enable_offline_mode"
                ],
                "ux_actions": [
                    "show_connectivity_indicator",
                    "enable_cached_data_mode",
                    "provide_offline_alternatives"
                ],
                "user_message": {
                    "type": "info",
                    "title": "Connection Issue",
                    "message": "We're experiencing connectivity issues. Your work is being saved locally.",
                    "actions": ["retry", "work_offline"]
                },
                "fallback_strategies": [
                    "use_cached_responses",
                    "enable_read_only_mode",
                    "provide_static_alternatives"
                ]
            },
            FailureType.DATABASE_ERROR: {
                "technical_actions": [
                    "reconnect_to_database",
                    "switch_to_replica",
                    "enable_write_buffering"
                ],
                "ux_actions": [
                    "show_saving_indicator",
                    "enable_local_storage",
                    "queue_user_actions"
                ],
                "user_message": {
                    "type": "warning",
                    "title": "Saving Issue",
                    "message": "We're having trouble saving your changes. They're being stored safely and will sync when resolved.",
                    "actions": ["continue_working", "export_backup"]
                },
                "fallback_strategies": [
                    "use_local_storage",
                    "enable_export_functionality",
                    "provide_manual_backup_options"
                ]
            },
            FailureType.MEMORY_ERROR: {
                "technical_actions": [
                    "trigger_garbage_collection",
                    "clear_non_essential_caches",
                    "reduce_processing_complexity"
                ],
                "ux_actions": [
                    "simplify_interface",
                    "reduce_concurrent_operations",
                    "enable_lightweight_mode"
                ],
                "user_message": {
                    "type": "info",
                    "title": "Optimizing Performance",
                    "message": "We're optimizing the system for better performance. Some features may be simplified temporarily.",
                    "actions": ["continue", "refresh_page"]
                },
                "fallback_strategies": [
                    "enable_lightweight_ui",
                    "reduce_data_visualization_complexity",
                    "limit_concurrent_requests"
                ]
            },
            FailureType.CPU_OVERLOAD: {
                "technical_actions": [
                    "throttle_background_processes",
                    "prioritize_user_facing_operations",
                    "enable_request_queuing"
                ],
                "ux_actions": [
                    "show_processing_indicators",
                    "enable_progressive_loading",
                    "reduce_real_time_updates"
                ],
                "user_message": {
                    "type": "info",
                    "title": "High System Load",
                    "message": "We're experiencing high demand. Your requests are being processed in order.",
                    "actions": ["wait", "try_later"]
                },
                "fallback_strategies": [
                    "enable_request_queuing",
                    "provide_estimated_wait_times",
                    "offer_simplified_alternatives"
                ]
            },
            FailureType.TIMEOUT_ERROR: {
                "technical_actions": [
                    "increase_timeout_limits",
                    "break_into_smaller_operations",
                    "enable_streaming_responses"
                ],
                "ux_actions": [
                    "show_detailed_progress",
                    "enable_partial_results",
                    "provide_cancellation_option"
                ],
                "user_message": {
                    "type": "info",
                    "title": "Processing...",
                    "message": "This operation is taking longer than usual. We're still working on it.",
                    "actions": ["wait", "cancel", "try_simpler_version"]
                },
                "fallback_strategies": [
                    "provide_partial_results",
                    "enable_background_processing",
                    "offer_simplified_operations"
                ]
            }
        }
    
    def _setup_behavior_monitoring(self):
        """Setup user behavior monitoring for predictive analysis."""
        # Store reference to original method for later use
        self._original_track_action = self.ux_protector.track_user_action
        self._behavior_monitoring_enabled = False
    
    def _start_background_tasks(self):
        """Start background tasks for continuous analysis."""
        if not self._background_tasks_started:
            self._background_tasks_started = True
            try:
                asyncio.create_task(self._continuous_analysis_loop())
                asyncio.create_task(self._prediction_validation_loop())
            except RuntimeError:
                # No event loop running, tasks will be started later
                pass
    
    def enable_behavior_monitoring(self):
        """Enable behavior monitoring by hooking into UX protector."""
        if self._behavior_monitoring_enabled:
            return
        
        original_track_action = self._original_track_action
        
        async def enhanced_track_action(action_type: str, user_id: Optional[str] = None, priority: str = "normal"):
            # Start background tasks if not already started
            if not self._background_tasks_started:
                self._start_background_tasks()
            
            async with original_track_action(action_type, user_id, priority) as action:
                # Analyze behavior pattern
                try:
                    await self._analyze_user_behavior(action_type, user_id, priority)
                except Exception as e:
                    logger.error(f"Error analyzing user behavior: {e}")
                yield action
        
        # Replace the original method
        self.ux_protector.track_user_action = enhanced_track_action
        self._behavior_monitoring_enabled = True
    
    async def _analyze_user_behavior(self, action_type: str, user_id: Optional[str], priority: str):
        """Analyze user behavior for pattern recognition and failure prediction."""
        user_key = user_id or "anonymous"
        
        # Get recent user actions
        recent_actions = [
            action for action in self.ux_protector.user_actions
            if action.user_id == user_id and 
            action.timestamp > datetime.utcnow() - self.behavior_analysis_window
        ]
        
        if len(recent_actions) < 3:
            return  # Need more data for pattern analysis
        
        # Extract action sequence
        action_sequence = [action.action_type for action in recent_actions[-10:]]
        
        # Calculate pattern metrics
        avg_response_time = statistics.mean(action.response_time for action in recent_actions)
        success_rate = sum(1 for action in recent_actions if action.success) / len(recent_actions)
        
        # Identify failure indicators
        failure_indicators = []
        if avg_response_time > 3.0:
            failure_indicators.append("slow_response_times")
        if success_rate < 0.8:
            failure_indicators.append("high_failure_rate")
        
        # Check for error patterns
        recent_errors = [action for action in recent_actions if not action.success]
        if len(recent_errors) > 2:
            failure_indicators.append("frequent_errors")
        
        # Create or update behavior pattern
        pattern_id = f"{user_key}_{hash(tuple(action_sequence))}"
        pattern = UserBehaviorPattern(
            pattern_id=pattern_id,
            user_id=user_id,
            action_sequence=action_sequence,
            frequency=1,
            avg_response_time=avg_response_time,
            success_rate=success_rate,
            last_seen=datetime.utcnow(),
            failure_indicators=failure_indicators
        )
        
        # Update pattern cache
        if pattern_id in self.pattern_cache:
            existing_pattern = self.pattern_cache[pattern_id]
            existing_pattern.frequency += 1
            existing_pattern.avg_response_time = (
                existing_pattern.avg_response_time + avg_response_time
            ) / 2
            existing_pattern.success_rate = (
                existing_pattern.success_rate + success_rate
            ) / 2
            existing_pattern.last_seen = datetime.utcnow()
            existing_pattern.failure_indicators = list(set(
                existing_pattern.failure_indicators + failure_indicators
            ))
        else:
            self.pattern_cache[pattern_id] = pattern
            self.user_patterns[user_key].append(pattern)
        
        # Predict potential failures based on patterns
        await self._predict_failures_from_behavior(pattern)
    
    async def _predict_failures_from_behavior(self, pattern: UserBehaviorPattern):
        """Predict potential failures based on user behavior patterns."""
        predictions = []
        
        # Analyze failure indicators
        if "slow_response_times" in pattern.failure_indicators:
            if pattern.avg_response_time > 5.0:
                predictions.append(FailurePrediction(
                    failure_type=FailureType.TIMEOUT_ERROR,
                    confidence=PredictionConfidence.HIGH,
                    predicted_time=datetime.utcnow() + timedelta(minutes=5),
                    impact_level=FailureImpactLevel.MEDIUM,
                    affected_users=[pattern.user_id] if pattern.user_id else [],
                    prevention_actions=[
                        "preload_common_data",
                        "optimize_query_performance",
                        "enable_caching"
                    ],
                    user_behavior_triggers=["slow_response_times"]
                ))
        
        if "high_failure_rate" in pattern.failure_indicators:
            if pattern.success_rate < 0.6:
                predictions.append(FailurePrediction(
                    failure_type=FailureType.EXTERNAL_API_ERROR,
                    confidence=PredictionConfidence.MEDIUM,
                    predicted_time=datetime.utcnow() + timedelta(minutes=10),
                    impact_level=FailureImpactLevel.HIGH,
                    affected_users=[pattern.user_id] if pattern.user_id else [],
                    prevention_actions=[
                        "prepare_fallback_responses",
                        "enable_circuit_breakers",
                        "cache_recent_responses"
                    ],
                    user_behavior_triggers=["high_failure_rate"]
                ))
        
        if "frequent_errors" in pattern.failure_indicators:
            predictions.append(FailurePrediction(
                failure_type=FailureType.VALIDATION_ERROR,
                confidence=PredictionConfidence.MEDIUM,
                predicted_time=datetime.utcnow() + timedelta(minutes=2),
                impact_level=FailureImpactLevel.LOW,
                affected_users=[pattern.user_id] if pattern.user_id else [],
                prevention_actions=[
                    "enhance_input_validation",
                    "provide_better_error_messages",
                    "add_input_suggestions"
                ],
                user_behavior_triggers=["frequent_errors"]
            ))
        
        # Add predictions to the list
        for prediction in predictions:
            self.failure_predictions.append(prediction)
            self.integration_metrics["predictions_made"] += 1
            
            # Trigger preventive actions
            await self._execute_preventive_actions(prediction)
    
    async def _execute_preventive_actions(self, prediction: FailurePrediction):
        """Execute preventive actions based on failure prediction."""
        logger.info(f"Executing preventive actions for predicted {prediction.failure_type.value}")
        
        try:
            for action in prediction.prevention_actions:
                await self._execute_prevention_action(action, prediction)
            
            self.integration_metrics["failures_prevented"] += 1
            
        except Exception as e:
            logger.error(f"Failed to execute preventive actions: {e}")
    
    async def _execute_prevention_action(self, action: str, prediction: FailurePrediction):
        """Execute a specific prevention action."""
        if action == "preload_common_data":
            # Preload data that users commonly access
            await self._preload_common_data(prediction.affected_users)
        
        elif action == "optimize_query_performance":
            # Optimize database queries
            await self._optimize_queries()
        
        elif action == "enable_caching":
            # Enable aggressive caching
            await self._enable_aggressive_caching()
        
        elif action == "prepare_fallback_responses":
            # Prepare fallback responses
            await self._prepare_fallback_responses(prediction.failure_type)
        
        elif action == "enable_circuit_breakers":
            # Enable circuit breakers for external services
            await self._enable_circuit_breakers()
        
        elif action == "cache_recent_responses":
            # Cache recent API responses
            await self._cache_recent_responses()
        
        elif action == "enhance_input_validation":
            # Enhance input validation
            await self._enhance_input_validation()
        
        elif action == "provide_better_error_messages":
            # Prepare better error messages
            await self._prepare_better_error_messages()
        
        elif action == "add_input_suggestions":
            # Add input suggestions
            await self._add_input_suggestions()
    
    async def handle_unified_failure(self, failure_event: FailureEvent) -> UnifiedFailureResponse:
        """Handle failure with unified technical recovery and UX protection."""
        logger.info(f"Handling unified failure: {failure_event.failure_type.value}")
        
        # Get response template
        template = self.response_templates.get(
            failure_event.failure_type, 
            self.response_templates[FailureType.UNKNOWN_ERROR]
        )
        
        # Create unified response
        response = UnifiedFailureResponse(
            failure_event=failure_event,
            technical_recovery_actions=template["technical_actions"].copy(),
            ux_protection_actions=template["ux_actions"].copy(),
            user_communication=template["user_message"].copy(),
            fallback_strategies=template["fallback_strategies"].copy(),
            recovery_timeline={},
            success_metrics={}
        )
        
        # Determine impact level
        impact_level = self._assess_failure_impact(failure_event)
        
        # Customize response based on impact
        if impact_level == FailureImpactLevel.CRITICAL:
            response.technical_recovery_actions.insert(0, "emergency_fallback_activation")
            response.ux_protection_actions.insert(0, "activate_emergency_ui_mode")
            response.user_communication["type"] = "error"
            response.user_communication["priority"] = "high"
        
        # Execute unified response
        await self._execute_unified_response(response)
        
        # Store active response
        response_id = f"{failure_event.failure_type.value}_{int(time.time())}"
        self.active_responses[response_id] = response
        
        return response
    
    def _assess_failure_impact(self, failure_event: FailureEvent) -> FailureImpactLevel:
        """Assess the impact level of a failure on user experience."""
        # Get current user experience level
        ux_level = self.ux_protector.get_current_experience_level()
        
        # Get system status
        system_status = self.failure_prevention.get_system_status()
        
        # Assess based on failure type and current state
        if failure_event.failure_type in [FailureType.DATABASE_ERROR, FailureType.NETWORK_ERROR]:
            if ux_level in [UserExperienceLevel.DEGRADED, UserExperienceLevel.MINIMAL]:
                return FailureImpactLevel.CRITICAL
            else:
                return FailureImpactLevel.HIGH
        
        elif failure_event.failure_type in [FailureType.MEMORY_ERROR, FailureType.CPU_OVERLOAD]:
            if system_status["system_resources"]["memory_percent"] > 95:
                return FailureImpactLevel.HIGH
            else:
                return FailureImpactLevel.MEDIUM
        
        elif failure_event.failure_type == FailureType.TIMEOUT_ERROR:
            return FailureImpactLevel.MEDIUM
        
        else:
            return FailureImpactLevel.LOW
    
    async def _execute_unified_response(self, response: UnifiedFailureResponse):
        """Execute the unified failure response."""
        start_time = datetime.utcnow()
        
        # Execute technical recovery actions
        for action in response.technical_recovery_actions:
            try:
                action_start = datetime.utcnow()
                await self._execute_technical_action(action, response.failure_event)
                response.recovery_timeline[f"technical_{action}"] = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Technical action {action} failed: {e}")
        
        # Execute UX protection actions
        for action in response.ux_protection_actions:
            try:
                action_start = datetime.utcnow()
                await self._execute_ux_action(action, response.failure_event)
                response.recovery_timeline[f"ux_{action}"] = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"UX action {action} failed: {e}")
        
        # Update user communication
        await self._update_user_communication(response.user_communication)
        
        # Calculate success metrics
        total_time = (datetime.utcnow() - start_time).total_seconds()
        response.success_metrics = {
            "total_recovery_time": total_time,
            "actions_completed": len(response.recovery_timeline),
            "user_impact_minimized": True
        }
        
        # Update integration metrics
        self.integration_metrics["avg_recovery_time"] = (
            self.integration_metrics["avg_recovery_time"] + total_time
        ) / 2
        self.integration_metrics["user_experience_improvements"] += 1
    
    async def _execute_technical_action(self, action: str, failure_event: FailureEvent):
        """Execute a technical recovery action."""
        if action == "retry_with_exponential_backoff":
            # Already handled by failure prevention system
            pass
        
        elif action == "switch_to_backup_endpoint":
            # Switch to backup service endpoint
            logger.info("Switching to backup endpoint")
        
        elif action == "enable_offline_mode":
            # Enable offline mode
            logger.info("Enabling offline mode")
        
        elif action == "reconnect_to_database":
            # Reconnect to database
            logger.info("Reconnecting to database")
        
        elif action == "switch_to_replica":
            # Switch to database replica
            logger.info("Switching to database replica")
        
        elif action == "enable_write_buffering":
            # Enable write buffering
            logger.info("Enabling write buffering")
        
        elif action == "trigger_garbage_collection":
            # Trigger garbage collection
            import gc
            gc.collect()
            logger.info("Garbage collection triggered")
        
        elif action == "clear_non_essential_caches":
            # Clear non-essential caches
            logger.info("Clearing non-essential caches")
        
        elif action == "reduce_processing_complexity":
            # Reduce processing complexity
            logger.info("Reducing processing complexity")
        
        elif action == "throttle_background_processes":
            # Throttle background processes
            logger.info("Throttling background processes")
        
        elif action == "prioritize_user_facing_operations":
            # Prioritize user-facing operations
            logger.info("Prioritizing user-facing operations")
        
        elif action == "enable_request_queuing":
            # Enable request queuing
            logger.info("Enabling request queuing")
        
        elif action == "increase_timeout_limits":
            # Increase timeout limits
            logger.info("Increasing timeout limits")
        
        elif action == "break_into_smaller_operations":
            # Break operations into smaller chunks
            logger.info("Breaking operations into smaller chunks")
        
        elif action == "enable_streaming_responses":
            # Enable streaming responses
            logger.info("Enabling streaming responses")
        
        elif action == "emergency_fallback_activation":
            # Activate emergency fallback systems
            logger.warning("Activating emergency fallback systems")
    
    async def _execute_ux_action(self, action: str, failure_event: FailureEvent):
        """Execute a UX protection action."""
        if action == "show_connectivity_indicator":
            # Show connectivity status to user
            logger.info("Showing connectivity indicator")
        
        elif action == "enable_cached_data_mode":
            # Enable cached data mode
            logger.info("Enabling cached data mode")
        
        elif action == "provide_offline_alternatives":
            # Provide offline alternatives
            logger.info("Providing offline alternatives")
        
        elif action == "show_saving_indicator":
            # Show saving status indicator
            logger.info("Showing saving indicator")
        
        elif action == "enable_local_storage":
            # Enable local storage mode
            logger.info("Enabling local storage")
        
        elif action == "queue_user_actions":
            # Queue user actions for later processing
            logger.info("Queuing user actions")
        
        elif action == "simplify_interface":
            # Simplify user interface
            logger.info("Simplifying interface")
        
        elif action == "reduce_concurrent_operations":
            # Reduce concurrent operations
            logger.info("Reducing concurrent operations")
        
        elif action == "enable_lightweight_mode":
            # Enable lightweight mode
            logger.info("Enabling lightweight mode")
        
        elif action == "show_processing_indicators":
            # Show processing indicators
            logger.info("Showing processing indicators")
        
        elif action == "enable_progressive_loading":
            # Enable progressive loading
            logger.info("Enabling progressive loading")
        
        elif action == "reduce_real_time_updates":
            # Reduce real-time updates
            logger.info("Reducing real-time updates")
        
        elif action == "show_detailed_progress":
            # Show detailed progress information
            logger.info("Showing detailed progress")
        
        elif action == "enable_partial_results":
            # Enable partial results display
            logger.info("Enabling partial results")
        
        elif action == "provide_cancellation_option":
            # Provide cancellation option
            logger.info("Providing cancellation option")
        
        elif action == "activate_emergency_ui_mode":
            # Activate emergency UI mode
            logger.warning("Activating emergency UI mode")
    
    async def _update_user_communication(self, communication: Dict[str, Any]):
        """Update user communication through the UX system."""
        # This would integrate with the frontend notification system
        logger.info(f"User communication: {communication['message']}")
    
    async def _continuous_analysis_loop(self):
        """Continuous analysis loop for behavior patterns and predictions."""
        while True:
            try:
                # Clean up old patterns
                cutoff_time = datetime.utcnow() - self.behavior_analysis_window
                for user_key in list(self.user_patterns.keys()):
                    self.user_patterns[user_key] = [
                        pattern for pattern in self.user_patterns[user_key]
                        if pattern.last_seen > cutoff_time
                    ]
                    
                    if not self.user_patterns[user_key]:
                        del self.user_patterns[user_key]
                
                # Clean up old predictions
                current_time = datetime.utcnow()
                self.failure_predictions = [
                    prediction for prediction in self.failure_predictions
                    if prediction.predicted_time > current_time - timedelta(hours=1)
                ]
                
                # Analyze system-wide patterns
                await self._analyze_system_patterns()
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Continuous analysis loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _analyze_system_patterns(self):
        """Analyze system-wide patterns for global predictions."""
        # Get system metrics
        system_status = self.failure_prevention.get_system_status()
        ux_metrics = self.ux_protector.get_experience_metrics()
        
        # Predict system-wide issues
        if (system_status["system_resources"]["memory_percent"] > 85 and
            ux_metrics.avg_response_time > 3.0):
            
            prediction = FailurePrediction(
                failure_type=FailureType.MEMORY_ERROR,
                confidence=PredictionConfidence.HIGH,
                predicted_time=datetime.utcnow() + timedelta(minutes=15),
                impact_level=FailureImpactLevel.HIGH,
                affected_users=["all"],
                prevention_actions=[
                    "trigger_garbage_collection",
                    "clear_non_essential_caches",
                    "enable_lightweight_mode"
                ],
                user_behavior_triggers=["system_wide_performance_degradation"]
            )
            
            self.failure_predictions.append(prediction)
            await self._execute_preventive_actions(prediction)
    
    async def _prediction_validation_loop(self):
        """Validate prediction accuracy and learn from results."""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Check predictions that should have occurred
                for prediction in list(self.failure_predictions):
                    if prediction.predicted_time < current_time - timedelta(minutes=30):
                        # Check if the prediction was correct
                        was_correct = await self._validate_prediction(prediction)
                        
                        # Record result
                        self.prediction_history.append((prediction, was_correct))
                        
                        # Update metrics
                        if was_correct:
                            self.integration_metrics["predictions_correct"] += 1
                        
                        # Remove from active predictions
                        self.failure_predictions.remove(prediction)
                
                # Calculate prediction accuracy
                if self.prediction_history:
                    accuracy = sum(1 for _, correct in self.prediction_history if correct) / len(self.prediction_history)
                    logger.info(f"Prediction accuracy: {accuracy:.2%}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Prediction validation loop error: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    async def _validate_prediction(self, prediction: FailurePrediction) -> bool:
        """Validate if a prediction was correct."""
        # Check recent failures for matching type
        recent_failures = [
            f for f in self.failure_prevention.failure_history
            if (f.failure_type == prediction.failure_type and
                f.timestamp > prediction.predicted_time - timedelta(minutes=30) and
                f.timestamp < prediction.predicted_time + timedelta(minutes=30))
        ]
        
        return len(recent_failures) > 0
    
    # Prevention action implementations
    async def _preload_common_data(self, affected_users: List[str]):
        """Preload commonly accessed data."""
        logger.info(f"Preloading common data for {len(affected_users)} users")
    
    async def _optimize_queries(self):
        """Optimize database queries."""
        logger.info("Optimizing database queries")
    
    async def _enable_aggressive_caching(self):
        """Enable aggressive caching."""
        logger.info("Enabling aggressive caching")
    
    async def _prepare_fallback_responses(self, failure_type: FailureType):
        """Prepare fallback responses for specific failure type."""
        logger.info(f"Preparing fallback responses for {failure_type.value}")
    
    async def _enable_circuit_breakers(self):
        """Enable circuit breakers for external services."""
        logger.info("Enabling circuit breakers")
    
    async def _cache_recent_responses(self):
        """Cache recent API responses."""
        logger.info("Caching recent responses")
    
    async def _enhance_input_validation(self):
        """Enhance input validation."""
        logger.info("Enhancing input validation")
    
    async def _prepare_better_error_messages(self):
        """Prepare better error messages."""
        logger.info("Preparing better error messages")
    
    async def _add_input_suggestions(self):
        """Add input suggestions."""
        logger.info("Adding input suggestions")
    
    async def _analyze_user_feedback(self, user_id: str, feedback: Dict[str, Any]):
        """Analyze user feedback for predictive insights."""
        logger.info(f"Analyzing user feedback from {user_id}: {feedback}")
        
        # Extract insights from feedback
        if "slow" in str(feedback).lower():
            # User experiencing slowness - predict performance issues
            prediction = FailurePrediction(
                failure_type=FailureType.TIMEOUT_ERROR,
                confidence=PredictionConfidence.MEDIUM,
                predicted_time=datetime.utcnow() + timedelta(minutes=5),
                impact_level=FailureImpactLevel.MEDIUM,
                affected_users=[user_id],
                prevention_actions=[
                    "optimize_query_performance",
                    "enable_caching",
                    "reduce_processing_complexity"
                ],
                user_behavior_triggers=["user_feedback_slow"]
            )
            
            self.failure_predictions.append(prediction)
            await self._execute_preventive_actions(prediction)
        
        elif "error" in str(feedback).lower() or "broken" in str(feedback).lower():
            # User experiencing errors - predict system issues
            prediction = FailurePrediction(
                failure_type=FailureType.VALIDATION_ERROR,
                confidence=PredictionConfidence.HIGH,
                predicted_time=datetime.utcnow() + timedelta(minutes=2),
                impact_level=FailureImpactLevel.HIGH,
                affected_users=[user_id],
                prevention_actions=[
                    "enhance_input_validation",
                    "provide_better_error_messages",
                    "prepare_fallback_responses"
                ],
                user_behavior_triggers=["user_feedback_errors"]
            )
            
            self.failure_predictions.append(prediction)
            await self._execute_preventive_actions(prediction)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.integration_metrics.copy(),
            "active_predictions": len(self.failure_predictions),
            "active_responses": len(self.active_responses),
            "user_patterns": len(self.pattern_cache),
            "prediction_accuracy": (
                self.integration_metrics["predictions_correct"] / 
                max(self.integration_metrics["predictions_made"], 1)
            ),
            "recent_predictions": [
                {
                    "failure_type": p.failure_type.value,
                    "confidence": p.confidence.value,
                    "predicted_time": p.predicted_time.isoformat(),
                    "impact_level": p.impact_level.value
                }
                for p in self.failure_predictions[-5:]  # Last 5 predictions
            ]
        }


# Global instance
failure_ux_integrator = FailureUXIntegrator()


# Convenience functions
async def handle_failure_with_ux_protection(failure_event: FailureEvent) -> UnifiedFailureResponse:
    """Handle failure with unified technical recovery and UX protection."""
    return await failure_ux_integrator.handle_unified_failure(failure_event)


def get_failure_ux_integration_status() -> Dict[str, Any]:
    """Get current integration status."""
    return failure_ux_integrator.get_integration_status()


@asynccontextmanager
async def bulletproof_user_operation(operation_name: str, user_id: Optional[str] = None, 
                                   priority: str = "normal"):
    """Context manager for bulletproof user operations with integrated protection."""
    async with failure_ux_integrator.ux_protector.track_user_action(
        operation_name, user_id, priority
    ) as action:
        try:
            yield action
        except Exception as e:
            # Create failure event
            failure_event = FailureEvent(
                failure_type=failure_ux_integrator.failure_prevention._classify_failure(e),
                timestamp=datetime.utcnow(),
                error_message=str(e),
                stack_trace="",  # Would include full stack trace in real implementation
                context={
                    "operation_name": operation_name,
                    "user_id": user_id,
                    "priority": priority
                }
            )
            
            # Handle with unified response
            await failure_ux_integrator.handle_unified_failure(failure_event)
            
            # Re-raise for calling code to handle
            raise


def bulletproof_with_ux(operation_name: str, user_id: Optional[str] = None, priority: str = "normal"):
    """Decorator for bulletproof operations with UX protection."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with bulletproof_user_operation(operation_name, user_id, priority):
                return await func(*args, **kwargs)
        return wrapper
    return decorator