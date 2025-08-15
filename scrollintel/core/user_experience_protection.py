"""
User experience protection system for ScrollIntel.
Ensures users always have a smooth, responsive experience even during system issues.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class UserExperienceLevel(Enum):
    """User experience quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    MINIMAL = "minimal"


@dataclass
class UserAction:
    """Represents a user action."""
    action_type: str
    timestamp: datetime
    user_id: Optional[str]
    response_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ExperienceMetrics:
    """User experience metrics."""
    avg_response_time: float
    success_rate: float
    error_count: int
    user_satisfaction_score: float
    experience_level: UserExperienceLevel


class UserExperienceProtector:
    """Protects and optimizes user experience."""
    
    def __init__(self):
        self.user_actions: List[UserAction] = []
        self.response_time_targets = {
            "critical": 0.5,    # 500ms for critical actions
            "important": 1.0,   # 1s for important actions
            "normal": 2.0,      # 2s for normal actions
            "background": 5.0   # 5s for background actions
        }
        self.experience_thresholds = {
            UserExperienceLevel.EXCELLENT: {"response_time": 0.5, "success_rate": 0.99},
            UserExperienceLevel.GOOD: {"response_time": 1.0, "success_rate": 0.95},
            UserExperienceLevel.ACCEPTABLE: {"response_time": 2.0, "success_rate": 0.90},
            UserExperienceLevel.DEGRADED: {"response_time": 5.0, "success_rate": 0.80},
            UserExperienceLevel.MINIMAL: {"response_time": 10.0, "success_rate": 0.60}
        }
        self.user_feedback_cache: Dict[str, Any] = {}
        self.loading_states: Dict[str, Dict[str, Any]] = {}
        self.progressive_enhancement_enabled = True
        
        # Initialize user-friendly messages
        self._setup_user_messages()
    
    def _setup_user_messages(self):
        """Setup user-friendly messages for different scenarios."""
        self.user_messages = {
            "loading": {
                "fast": "Just a moment...",
                "normal": "Processing your request...",
                "slow": "This is taking longer than usual, please wait...",
                "very_slow": "We're working hard to get your results. Thank you for your patience."
            },
            "errors": {
                "network": "Connection issue detected. Retrying automatically...",
                "server": "Our servers are busy. We're working to resolve this quickly.",
                "timeout": "This is taking longer than expected. We're still working on it.",
                "general": "Something went wrong, but we're fixing it. Please try again."
            },
            "degradation": {
                "minor": "We're optimizing performance. Some features may be simplified.",
                "major": "We're experiencing high demand. Core features remain available.",
                "emergency": "We're in maintenance mode. Basic functionality is available."
            },
            "success": {
                "fast": "Done!",
                "normal": "Complete!",
                "recovered": "All systems restored. Thank you for your patience."
            }
        }
    
    @asynccontextmanager
    async def track_user_action(self, action_type: str, user_id: Optional[str] = None,
                               priority: str = "normal"):
        """Context manager to track user actions and ensure good experience."""
        start_time = time.time()
        action = UserAction(
            action_type=action_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            response_time=0.0,
            success=False
        )
        
        # Set up loading state
        loading_key = f"{user_id or 'anonymous'}_{action_type}_{start_time}"
        self.loading_states[loading_key] = {
            "start_time": start_time,
            "action_type": action_type,
            "priority": priority,
            "user_id": user_id,
            "message": self._get_loading_message(0, priority)
        }
        
        # Start background task to update loading messages
        update_task = asyncio.create_task(
            self._update_loading_messages(loading_key, priority)
        )
        
        try:
            yield action
            
            # Action completed successfully
            action.success = True
            action.response_time = time.time() - start_time
            
            # Update loading state with success
            if loading_key in self.loading_states:
                self.loading_states[loading_key]["completed"] = True
                self.loading_states[loading_key]["message"] = self._get_success_message(action.response_time)
            
        except Exception as e:
            # Action failed
            action.success = False
            action.response_time = time.time() - start_time
            action.error_message = str(e)
            
            # Update loading state with error
            if loading_key in self.loading_states:
                self.loading_states[loading_key]["error"] = True
                self.loading_states[loading_key]["message"] = self._get_error_message(e)
            
            # Don't re-raise - let the calling code handle it
            
        finally:
            # Clean up
            update_task.cancel()
            self.user_actions.append(action)
            
            # Remove old loading state after a delay
            asyncio.create_task(self._cleanup_loading_state(loading_key, delay=5))
            
            # Clean up old actions (keep last 1000)
            if len(self.user_actions) > 1000:
                self.user_actions = self.user_actions[-1000:]
    
    async def _update_loading_messages(self, loading_key: str, priority: str):
        """Update loading messages based on elapsed time."""
        try:
            while loading_key in self.loading_states:
                loading_state = self.loading_states[loading_key]
                
                if loading_state.get("completed") or loading_state.get("error"):
                    break
                
                elapsed = time.time() - loading_state["start_time"]
                new_message = self._get_loading_message(elapsed, priority)
                
                if new_message != loading_state["message"]:
                    loading_state["message"] = new_message
                    loading_state["elapsed"] = elapsed
                
                await asyncio.sleep(1)  # Update every second
                
        except asyncio.CancelledError:
            pass
    
    async def _cleanup_loading_state(self, loading_key: str, delay: int = 5):
        """Clean up loading state after delay."""
        await asyncio.sleep(delay)
        self.loading_states.pop(loading_key, None)
    
    def _get_loading_message(self, elapsed_time: float, priority: str) -> str:
        """Get appropriate loading message based on elapsed time."""
        target_time = self.response_time_targets.get(priority, 2.0)
        
        if elapsed_time < target_time:
            return self.user_messages["loading"]["fast"]
        elif elapsed_time < target_time * 2:
            return self.user_messages["loading"]["normal"]
        elif elapsed_time < target_time * 4:
            return self.user_messages["loading"]["slow"]
        else:
            return self.user_messages["loading"]["very_slow"]
    
    def _get_success_message(self, response_time: float) -> str:
        """Get success message based on response time."""
        if response_time < 1.0:
            return self.user_messages["success"]["fast"]
        else:
            return self.user_messages["success"]["normal"]
    
    def _get_error_message(self, error: Exception) -> str:
        """Get user-friendly error message."""
        error_msg = str(error).lower()
        
        if "network" in error_msg or "connection" in error_msg:
            return self.user_messages["errors"]["network"]
        elif "timeout" in error_msg:
            return self.user_messages["errors"]["timeout"]
        elif "server" in error_msg or "internal" in error_msg:
            return self.user_messages["errors"]["server"]
        else:
            return self.user_messages["errors"]["general"]
    
    def get_current_experience_level(self) -> UserExperienceLevel:
        """Calculate current user experience level."""
        if not self.user_actions:
            return UserExperienceLevel.EXCELLENT
        
        # Analyze recent actions (last 5 minutes)
        recent_cutoff = datetime.utcnow() - timedelta(minutes=5)
        recent_actions = [
            action for action in self.user_actions 
            if action.timestamp > recent_cutoff
        ]
        
        if not recent_actions:
            return UserExperienceLevel.EXCELLENT
        
        # Calculate metrics
        avg_response_time = sum(action.response_time for action in recent_actions) / len(recent_actions)
        success_rate = sum(1 for action in recent_actions if action.success) / len(recent_actions)
        
        # Determine experience level
        for level, thresholds in self.experience_thresholds.items():
            if (avg_response_time <= thresholds["response_time"] and 
                success_rate >= thresholds["success_rate"]):
                return level
        
        return UserExperienceLevel.MINIMAL
    
    def get_experience_metrics(self) -> ExperienceMetrics:
        """Get comprehensive experience metrics."""
        if not self.user_actions:
            return ExperienceMetrics(
                avg_response_time=0.0,
                success_rate=1.0,
                error_count=0,
                user_satisfaction_score=1.0,
                experience_level=UserExperienceLevel.EXCELLENT
            )
        
        # Analyze recent actions
        recent_cutoff = datetime.utcnow() - timedelta(minutes=10)
        recent_actions = [
            action for action in self.user_actions 
            if action.timestamp > recent_cutoff
        ]
        
        if not recent_actions:
            recent_actions = self.user_actions[-10:]  # Last 10 actions
        
        avg_response_time = sum(action.response_time for action in recent_actions) / len(recent_actions)
        success_rate = sum(1 for action in recent_actions if action.success) / len(recent_actions)
        error_count = sum(1 for action in recent_actions if not action.success)
        
        # Calculate satisfaction score (0-1)
        satisfaction_score = min(1.0, success_rate * (2.0 / max(avg_response_time, 0.1)))
        
        return ExperienceMetrics(
            avg_response_time=avg_response_time,
            success_rate=success_rate,
            error_count=error_count,
            user_satisfaction_score=satisfaction_score,
            experience_level=self.get_current_experience_level()
        )
    
    def get_loading_states(self) -> Dict[str, Any]:
        """Get current loading states for UI updates."""
        return {
            key: {
                "message": state["message"],
                "elapsed": time.time() - state["start_time"],
                "action_type": state["action_type"],
                "completed": state.get("completed", False),
                "error": state.get("error", False)
            }
            for key, state in self.loading_states.items()
        }
    
    def should_show_progress_indicator(self, action_type: str, elapsed_time: float) -> bool:
        """Determine if progress indicator should be shown."""
        # Always show for actions taking longer than 1 second
        if elapsed_time > 1.0:
            return True
        
        # Show for critical actions taking longer than 500ms
        if action_type in ["login", "save", "upload"] and elapsed_time > 0.5:
            return True
        
        return False
    
    def get_user_feedback_prompt(self) -> Optional[Dict[str, Any]]:
        """Get feedback prompt if user experience is degraded."""
        experience_level = self.get_current_experience_level()
        
        if experience_level in [UserExperienceLevel.DEGRADED, UserExperienceLevel.MINIMAL]:
            return {
                "show_feedback": True,
                "message": "We notice you might be experiencing some issues. How can we improve?",
                "options": [
                    "The app is too slow",
                    "Features aren't working",
                    "Error messages are confusing",
                    "Everything is fine"
                ]
            }
        
        return None
    
    def record_user_feedback(self, user_id: str, feedback: Dict[str, Any]):
        """Record user feedback for experience improvement."""
        self.user_feedback_cache[user_id] = {
            "timestamp": datetime.utcnow().isoformat(),
            "feedback": feedback,
            "experience_level": self.get_current_experience_level().value
        }
        
        logger.info(f"User feedback recorded: {feedback}")
        
        # Notify integration system about user feedback
        try:
            from .failure_ux_integration import failure_ux_integrator
            asyncio.create_task(failure_ux_integrator._analyze_user_feedback(user_id, feedback))
        except ImportError:
            # Integration system not available
            pass
        except Exception as e:
            logger.error(f"Failed to notify integration system about feedback: {e}")
    
    def get_user_behavior_indicators(self) -> Dict[str, Any]:
        """Get user behavior indicators for predictive analysis."""
        if not self.user_actions:
            return {}
        
        recent_actions = [
            action for action in self.user_actions
            if action.timestamp > datetime.utcnow() - timedelta(minutes=30)
        ]
        
        if not recent_actions:
            return {}
        
        # Calculate behavior indicators
        avg_response_time = sum(action.response_time for action in recent_actions) / len(recent_actions)
        success_rate = sum(1 for action in recent_actions if action.success) / len(recent_actions)
        error_frequency = len([action for action in recent_actions if not action.success])
        
        # Identify concerning patterns
        indicators = {
            "avg_response_time": avg_response_time,
            "success_rate": success_rate,
            "error_frequency": error_frequency,
            "total_actions": len(recent_actions),
            "concerning_patterns": []
        }
        
        if avg_response_time > 3.0:
            indicators["concerning_patterns"].append("slow_response_times")
        
        if success_rate < 0.8:
            indicators["concerning_patterns"].append("high_failure_rate")
        
        if error_frequency > 3:
            indicators["concerning_patterns"].append("frequent_errors")
        
        return indicators
    
    def register_experience_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for experience level changes."""
        if not hasattr(self, '_experience_callbacks'):
            self._experience_callbacks = []
        self._experience_callbacks.append(callback)
    
    async def _notify_experience_callbacks(self, experience_data: Dict[str, Any]):
        """Notify registered callbacks about experience changes."""
        if hasattr(self, '_experience_callbacks'):
            for callback in self._experience_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(experience_data)
                    else:
                        callback(experience_data)
                except Exception as e:
                    logger.error(f"Experience callback error: {e}")
    
    def get_progressive_enhancement_config(self) -> Dict[str, Any]:
        """Get configuration for progressive enhancement based on current performance."""
        experience_level = self.get_current_experience_level()
        
        config = {
            "enable_animations": True,
            "enable_real_time_updates": True,
            "enable_advanced_features": True,
            "max_concurrent_requests": 10,
            "cache_duration": 300,  # 5 minutes
            "image_quality": "high",
            "chart_complexity": "full"
        }
        
        if experience_level == UserExperienceLevel.DEGRADED:
            config.update({
                "enable_animations": False,
                "max_concurrent_requests": 5,
                "cache_duration": 600,  # 10 minutes
                "image_quality": "medium",
                "chart_complexity": "simplified"
            })
        elif experience_level == UserExperienceLevel.MINIMAL:
            config.update({
                "enable_animations": False,
                "enable_real_time_updates": False,
                "enable_advanced_features": False,
                "max_concurrent_requests": 2,
                "cache_duration": 1800,  # 30 minutes
                "image_quality": "low",
                "chart_complexity": "basic"
            })
        
        return config
    
    def get_user_notification(self) -> Optional[Dict[str, Any]]:
        """Get user notification about current system status."""
        experience_level = self.get_current_experience_level()
        
        if experience_level == UserExperienceLevel.DEGRADED:
            return {
                "type": "info",
                "message": self.user_messages["degradation"]["minor"],
                "dismissible": True,
                "auto_dismiss": 10000  # 10 seconds
            }
        elif experience_level == UserExperienceLevel.MINIMAL:
            return {
                "type": "warning",
                "message": self.user_messages["degradation"]["major"],
                "dismissible": True,
                "auto_dismiss": 15000  # 15 seconds
            }
        
        return None
    
    def optimize_for_user_experience(self) -> Dict[str, Any]:
        """Get optimization recommendations based on current experience."""
        metrics = self.get_experience_metrics()
        recommendations = []
        
        if metrics.avg_response_time > 2.0:
            recommendations.append("Enable caching for faster responses")
            recommendations.append("Reduce data processing complexity")
        
        if metrics.success_rate < 0.9:
            recommendations.append("Implement better error handling")
            recommendations.append("Add retry mechanisms")
        
        if metrics.error_count > 5:
            recommendations.append("Investigate frequent error sources")
            recommendations.append("Improve system stability")
        
        return {
            "current_level": metrics.experience_level.value,
            "satisfaction_score": metrics.user_satisfaction_score,
            "recommendations": recommendations,
            "progressive_config": self.get_progressive_enhancement_config()
        }


# Global instance
ux_protector = UserExperienceProtector()


# Convenience functions and decorators
def track_user_action(action_type: str, user_id: Optional[str] = None, priority: str = "normal"):
    """Decorator to track user actions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with ux_protector.track_user_action(action_type, user_id, priority) as action:
                result = await func(*args, **kwargs)
                return result
        return wrapper
    return decorator


async def with_user_experience_protection(func: Callable, action_type: str, 
                                        user_id: Optional[str] = None, 
                                        priority: str = "normal"):
    """Execute function with user experience protection."""
    async with ux_protector.track_user_action(action_type, user_id, priority):
        return await func()


def get_user_experience_status() -> Dict[str, Any]:
    """Get current user experience status."""
    metrics = ux_protector.get_experience_metrics()
    return {
        "experience_level": metrics.experience_level.value,
        "avg_response_time": metrics.avg_response_time,
        "success_rate": metrics.success_rate,
        "satisfaction_score": metrics.user_satisfaction_score,
        "loading_states": ux_protector.get_loading_states(),
        "notification": ux_protector.get_user_notification(),
        "feedback_prompt": ux_protector.get_user_feedback_prompt(),
        "progressive_config": ux_protector.get_progressive_enhancement_config()
    }