"""
Simplified integration between failure prevention and user experience protection systems.
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
        # Basic initialization only
        self.failure_prevention = failure_prevention
        self.ux_protector = ux_protector
        
        # Simple data structures
        self.user_patterns: Dict[str, List[UserBehaviorPattern]] = defaultdict(list)
        self.failure_predictions: List[FailurePrediction] = []
        self.active_responses: Dict[str, UnifiedFailureResponse] = {}
        
        # Metrics
        self.integration_metrics: Dict[str, Any] = {
            "predictions_made": 0,
            "predictions_correct": 0,
            "failures_prevented": 0,
            "user_experience_improvements": 0,
            "avg_recovery_time": 0.0
        }
        
        # Lazy initialization flags
        self._response_templates: Optional[Dict[FailureType, Dict[str, Any]]] = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Ensure the system is fully initialized."""
        if not self._initialized:
            self._setup_response_templates()
            self._initialized = True
    
    def _setup_response_templates(self):
        """Setup response templates for different failure types."""
        self._response_templates = {
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
            }
        }
    
    async def handle_unified_failure(self, failure_event: FailureEvent) -> UnifiedFailureResponse:
        """Handle failure with unified technical recovery and UX protection."""
        self._ensure_initialized()
        
        logger.info(f"Handling unified failure: {failure_event.failure_type.value}")
        
        # Get response template
        template = self._response_templates.get(
            failure_event.failure_type, 
            self._response_templates.get(FailureType.NETWORK_ERROR, {
                "technical_actions": ["basic_recovery"],
                "ux_actions": ["show_error_message"],
                "user_message": {
                    "type": "error",
                    "title": "System Issue",
                    "message": "We're experiencing a technical issue. Please try again.",
                    "actions": ["retry"]
                },
                "fallback_strategies": ["basic_fallback"]
            })
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
        
        # Execute response
        await self._execute_unified_response(response)
        
        # Store active response
        response_id = f"{failure_event.failure_type.value}_{int(time.time())}"
        self.active_responses[response_id] = response
        
        # Update metrics
        self.integration_metrics["user_experience_improvements"] += 1
        
        return response
    
    async def _execute_unified_response(self, response: UnifiedFailureResponse):
        """Execute the unified failure response."""
        start_time = datetime.utcnow()
        
        # Execute technical recovery actions
        for action in response.technical_recovery_actions:
            try:
                await self._execute_technical_action(action, response.failure_event)
                response.recovery_timeline[f"technical_{action}"] = datetime.utcnow()
            except Exception as e:
                logger.error(f"Technical action {action} failed: {e}")
        
        # Execute UX protection actions
        for action in response.ux_protection_actions:
            try:
                await self._execute_ux_action(action, response.failure_event)
                response.recovery_timeline[f"ux_{action}"] = datetime.utcnow()
            except Exception as e:
                logger.error(f"UX action {action} failed: {e}")
        
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
    
    async def _execute_technical_action(self, action: str, failure_event: FailureEvent):
        """Execute a technical recovery action."""
        logger.info(f"Executing technical action: {action}")
        # Simulate action execution
        await asyncio.sleep(0.01)
    
    async def _execute_ux_action(self, action: str, failure_event: FailureEvent):
        """Execute a UX protection action."""
        logger.info(f"Executing UX action: {action}")
        # Simulate action execution
        await asyncio.sleep(0.01)
    
    async def analyze_user_behavior(self, action_type: str, user_id: Optional[str], 
                                  response_time: float, success: bool):
        """Analyze user behavior for pattern recognition."""
        if not user_id:
            return
        
        # Simple behavior analysis
        user_key = user_id
        current_time = datetime.utcnow()
        
        # Create simple pattern
        pattern = UserBehaviorPattern(
            pattern_id=f"{user_key}_{action_type}_{int(time.time())}",
            user_id=user_id,
            action_sequence=[action_type],
            frequency=1,
            avg_response_time=response_time,
            success_rate=1.0 if success else 0.0,
            last_seen=current_time,
            failure_indicators=["slow_response"] if response_time > 3.0 else []
        )
        
        self.user_patterns[user_key].append(pattern)
        
        # Simple prediction logic
        if response_time > 5.0 or not success:
            prediction = FailurePrediction(
                failure_type=FailureType.TIMEOUT_ERROR if response_time > 5.0 else FailureType.VALIDATION_ERROR,
                confidence=PredictionConfidence.MEDIUM,
                predicted_time=current_time + timedelta(minutes=5),
                impact_level=FailureImpactLevel.MEDIUM,
                affected_users=[user_id],
                prevention_actions=["optimize_performance", "enhance_validation"],
                user_behavior_triggers=["slow_response" if response_time > 5.0 else "validation_error"]
            )
            
            self.failure_predictions.append(prediction)
            self.integration_metrics["predictions_made"] += 1
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.integration_metrics.copy(),
            "active_predictions": len(self.failure_predictions),
            "active_responses": len(self.active_responses),
            "user_patterns": sum(len(patterns) for patterns in self.user_patterns.values()),
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
    start_time = time.time()
    success = False
    error = None
    
    try:
        yield
        success = True
    except Exception as e:
        error = e
        success = False
        
        # Create failure event
        failure_event = FailureEvent(
            failure_type=failure_prevention._classify_failure(e),
            timestamp=datetime.utcnow(),
            error_message=str(e),
            stack_trace="",
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
    finally:
        # Analyze user behavior
        response_time = time.time() - start_time
        await failure_ux_integrator.analyze_user_behavior(
            operation_name, user_id, response_time, success
        )


def bulletproof_with_ux(operation_name: str, user_id: Optional[str] = None, priority: str = "normal"):
    """Decorator for bulletproof operations with UX protection."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with bulletproof_user_operation(operation_name, user_id, priority):
                return await func(*args, **kwargs)
        return wrapper
    return decorator