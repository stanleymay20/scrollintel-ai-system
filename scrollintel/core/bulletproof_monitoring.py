"""
Main bulletproof monitoring system that integrates with all bulletproof components.
Provides centralized monitoring, metrics collection, and health tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MonitoringLevel(Enum):
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

@dataclass
class MonitoringConfig:
    """Configuration for bulletproof monitoring."""
    level: MonitoringLevel = MonitoringLevel.DETAILED
    auto_record_performance: bool = True
    auto_record_failures: bool = True
    auto_record_user_actions: bool = True
    performance_threshold_ms: float = 1000.0
    error_rate_threshold: float = 0.05
    satisfaction_threshold: float = 3.0

class BulletproofMonitoring:
    """
    Main bulletproof monitoring system that integrates with all components.
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.active_sessions = {}
        self.component_monitors = {}
        self.performance_baselines = {}
        self.is_monitoring = False
        
    async def start_monitoring(self) -> None:
        """Start the monitoring system."""
        try:
            self.is_monitoring = True
            logger.info("Bulletproof monitoring system started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring system: {e}")
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        try:
            self.is_monitoring = False
            logger.info("Bulletproof monitoring system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring system: {e}")
    
    async def record_user_action(
        self,
        user_id: str,
        action: str,
        success: bool = True,
        duration_ms: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Record a user action for monitoring."""
        try:
            # Import analytics here to avoid circular imports
            from scrollintel.core.bulletproof_monitoring_analytics import (
                bulletproof_analytics,
                UserExperienceMetric,
                MetricType
            )
            
            if not self.config.auto_record_user_actions:
                return
            
            # Record performance if duration provided
            if duration_ms is not None:
                performance_metric = UserExperienceMetric(
                    timestamp=datetime.now(),
                    user_id=user_id,
                    metric_type=MetricType.PERFORMANCE,
                    value=duration_ms,
                    context={
                        "action": action,
                        "success": success,
                        **(context or {})
                    },
                    session_id=session_id,
                    component="user_action"
                )
                
                await bulletproof_analytics.record_metric(performance_metric)
            
            # Record failure if action failed
            if not success:
                failure_metric = UserExperienceMetric(
                    timestamp=datetime.now(),
                    user_id=user_id,
                    metric_type=MetricType.FAILURE_RATE,
                    value=1.0,
                    context={
                        "action": action,
                        "failure_type": "user_action_failed",
                        **(context or {})
                    },
                    session_id=session_id,
                    component="user_action"
                )
                
                await bulletproof_analytics.record_metric(failure_metric)
            
        except Exception as e:
            logger.error(f"Error recording user action: {e}")
    
    async def record_user_satisfaction(
        self,
        user_id: str,
        satisfaction_score: float,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Record user satisfaction score."""
        try:
            # Import analytics here to avoid circular imports
            from scrollintel.core.bulletproof_monitoring_analytics import (
                bulletproof_analytics,
                UserExperienceMetric,
                MetricType
            )
            
            satisfaction_metric = UserExperienceMetric(
                timestamp=datetime.now(),
                user_id=user_id,
                metric_type=MetricType.USER_SATISFACTION,
                value=satisfaction_score,
                context=context or {},
                session_id=session_id,
                component="user_feedback"
            )
            
            await bulletproof_analytics.record_metric(satisfaction_metric)
            
            # Check satisfaction threshold
            if satisfaction_score < self.config.satisfaction_threshold:
                logger.warning(
                    f"Low user satisfaction detected: {satisfaction_score} "
                    f"(threshold: {self.config.satisfaction_threshold}) for user {user_id}"
                )
            
        except Exception as e:
            logger.error(f"Error recording user satisfaction: {e}")
    
    async def record_system_health(
        self,
        component: str,
        health_score: float,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record system health metrics."""
        try:
            # Import analytics here to avoid circular imports
            from scrollintel.core.bulletproof_monitoring_analytics import (
                bulletproof_analytics,
                UserExperienceMetric,
                MetricType
            )
            
            health_metric = UserExperienceMetric(
                timestamp=datetime.now(),
                user_id="system",
                metric_type=MetricType.SYSTEM_HEALTH,
                value=health_score,
                context=metrics or {},
                component=component
            )
            
            await bulletproof_analytics.record_metric(health_metric)
            
        except Exception as e:
            logger.error(f"Error recording system health: {e}")
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status."""
        try:
            # Import analytics here to avoid circular imports
            from scrollintel.core.bulletproof_monitoring_analytics import bulletproof_analytics
            
            return {
                "is_monitoring": self.is_monitoring,
                "config": {
                    "level": self.config.level.value,
                    "auto_record_performance": self.config.auto_record_performance,
                    "auto_record_failures": self.config.auto_record_failures,
                    "auto_record_user_actions": self.config.auto_record_user_actions,
                    "performance_threshold_ms": self.config.performance_threshold_ms,
                    "error_rate_threshold": self.config.error_rate_threshold,
                    "satisfaction_threshold": self.config.satisfaction_threshold
                },
                "monitored_components": list(self.component_monitors.keys()),
                "active_sessions": len(self.active_sessions),
                "performance_baselines": {
                    component: {
                        "mean_ms": baseline["mean"],
                        "updated_at": baseline["updated_at"].isoformat()
                    }
                    for component, baseline in self.performance_baselines.items()
                },
                "metrics_buffer_size": len(bulletproof_analytics.metrics_buffer),
                "active_alerts": len(bulletproof_analytics.active_alerts)
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring status: {e}")
            return {"error": str(e)}
    
    async def update_config(self, new_config: MonitoringConfig) -> None:
        """Update monitoring configuration."""
        try:
            self.config = new_config
            logger.info(f"Monitoring configuration updated: level={new_config.level.value}")
            
        except Exception as e:
            logger.error(f"Error updating monitoring config: {e}")
    
    def monitor_component(self, component_name: str):
        """Decorator to monitor a component's methods."""
        def decorator(cls):
            # Store component info
            self.component_monitors[component_name] = {
                "class": cls,
                "methods": [name for name, method in cls.__dict__.items() 
                           if callable(method) and not name.startswith('_')],
                "registered_at": datetime.now()
            }
            
            return cls
        return decorator

# Global monitoring instance
bulletproof_monitoring = BulletproofMonitoring()

# Convenience functions
async def record_user_action(
    user_id: str,
    action: str,
    success: bool = True,
    duration_ms: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> None:
    """Convenience function to record user actions."""
    await bulletproof_monitoring.record_user_action(
        user_id=user_id,
        action=action,
        success=success,
        duration_ms=duration_ms,
        context=context,
        session_id=session_id
    )

async def record_user_satisfaction(
    user_id: str,
    satisfaction_score: float,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> None:
    """Convenience function to record user satisfaction."""
    await bulletproof_monitoring.record_user_satisfaction(
        user_id=user_id,
        satisfaction_score=satisfaction_score,
        context=context,
        session_id=session_id
    )