"""
Central Bulletproof Orchestrator for ScrollIntel.
Coordinates all protection systems to ensure users never experience failures.
Provides unified configuration, management interface, comprehensive system health dashboard,
and automated recovery coordination.
"""

import asyncio
import logging
import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import psutil
import os
from pathlib import Path

from .never_fail_decorators import (
    intelligent_retry, fallback_generator, timeout_handler,
    IntelligentRetryStrategy, ContextAwareFallbackGenerator, ProgressiveTimeoutHandler
)
from .failure_prevention import failure_prevention, FailurePreventionSystem, FailureEvent
from .graceful_degradation import degradation_manager, IntelligentDegradationManager, DegradationLevel
from .user_experience_protection import ux_protector, UserExperienceProtector, UserExperienceLevel
from .bulletproof_middleware import BulletproofMiddleware

logger = logging.getLogger(__name__)


class SystemHealthStatus(Enum):
    """Overall system health status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ProtectionMode(Enum):
    """Protection system operation modes."""
    FULL_PROTECTION = "full_protection"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    RESOURCE_CONSERVATIVE = "resource_conservative"
    EMERGENCY_MODE = "emergency_mode"


@dataclass
class SystemHealthMetrics:
    """Comprehensive system health metrics."""
    overall_status: SystemHealthStatus
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    error_rate: float
    response_time: float
    active_users: int
    protection_effectiveness: float
    recovery_success_rate: float
    user_satisfaction_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProtectionSystemStatus:
    """Status of individual protection systems."""
    system_name: str
    is_active: bool
    health_score: float
    last_action: Optional[str]
    last_action_time: Optional[datetime]
    metrics: Dict[str, Any]
    alerts: List[str] = field(default_factory=list)


@dataclass
class UnifiedConfiguration:
    """Unified configuration for all protection systems."""
    protection_mode: ProtectionMode
    auto_recovery_enabled: bool
    predictive_prevention_enabled: bool
    user_experience_optimization: bool
    intelligent_routing_enabled: bool
    fallback_generation_enabled: bool
    degradation_learning_enabled: bool
    real_time_monitoring: bool
    alert_thresholds: Dict[str, float]
    recovery_timeouts: Dict[str, float]
    user_notification_settings: Dict[str, Any]


class BulletproofOrchestrator:
    """Central orchestrator that coordinates all bulletproof protection systems."""
    
    def __init__(self, config: Optional[UnifiedConfiguration] = None):
        self.config = config or self._create_default_config()
        self.is_active = False
        self.start_time = datetime.utcnow()
        
        # Protection system components
        self.failure_prevention = failure_prevention
        self.degradation_manager = degradation_manager
        self.ux_protector = ux_protector
        self.intelligent_retry = intelligent_retry
        self.fallback_generator = fallback_generator
        self.timeout_handler = timeout_handler
        
        # Orchestrator-specific components
        self.system_health_monitor = SystemHealthMonitor()
        self.recovery_coordinator = RecoveryCoordinator()
        self.alert_manager = AlertManager()
        self.dashboard_manager = DashboardManager()
        self.configuration_manager = ConfigurationManager(self.config)
        
        # Cross-system coordination
        self.protection_systems: Dict[str, Any] = {}
        self.system_metrics_history: List[SystemHealthMetrics] = []
        self.active_alerts: List[Dict[str, Any]] = []
        self.recovery_actions: List[Dict[str, Any]] = []
        
        # Monitoring and coordination
        self._monitoring_active = False
        self._monitoring_thread = None
        self._coordination_lock = threading.Lock()
        
        # Initialize systems
        self._initialize_protection_systems()
        self._setup_cross_system_callbacks()
        
    def _create_default_config(self) -> UnifiedConfiguration:
        """Create default unified configuration."""
        return UnifiedConfiguration(
            protection_mode=ProtectionMode.FULL_PROTECTION,
            auto_recovery_enabled=True,
            predictive_prevention_enabled=True,
            user_experience_optimization=True,
            intelligent_routing_enabled=True,
            fallback_generation_enabled=True,
            degradation_learning_enabled=True,
            real_time_monitoring=True,
            alert_thresholds={
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "error_rate": 0.05,
                "response_time": 3.0,
                "user_satisfaction": 0.7
            },
            recovery_timeouts={
                "automatic_recovery": 30.0,
                "degradation_recovery": 60.0,
                "system_restart": 300.0
            },
            user_notification_settings={
                "show_system_status": True,
                "show_recovery_progress": True,
                "show_degradation_notices": True,
                "auto_dismiss_success": True
            }
        )
    
    def _initialize_protection_systems(self):
        """Initialize and register all protection systems."""
        self.protection_systems = {
            "failure_prevention": {
                "instance": self.failure_prevention,
                "health_check": self._check_failure_prevention_health,
                "metrics_collector": self._collect_failure_prevention_metrics
            },
            "degradation_manager": {
                "instance": self.degradation_manager,
                "health_check": self._check_degradation_manager_health,
                "metrics_collector": self._collect_degradation_manager_metrics
            },
            "ux_protector": {
                "instance": self.ux_protector,
                "health_check": self._check_ux_protector_health,
                "metrics_collector": self._collect_ux_protector_metrics
            },
            "intelligent_retry": {
                "instance": self.intelligent_retry,
                "health_check": self._check_intelligent_retry_health,
                "metrics_collector": self._collect_intelligent_retry_metrics
            },
            "fallback_generator": {
                "instance": self.fallback_generator,
                "health_check": self._check_fallback_generator_health,
                "metrics_collector": self._collect_fallback_generator_metrics
            }
        }
    
    def _setup_cross_system_callbacks(self):
        """Setup callbacks for cross-system coordination."""
        # Register failure callbacks for coordination
        self.failure_prevention.register_failure_callback(self._handle_system_failure)
        
        # Register experience callbacks
        self.ux_protector.register_experience_callback(self._handle_experience_change)
        
        # Setup degradation coordination
        if hasattr(self.degradation_manager, 'register_degradation_callback'):
            self.degradation_manager.register_degradation_callback(self._handle_degradation_event)
    
    async def start(self):
        """Start the bulletproof orchestrator and all protection systems."""
        if self.is_active:
            logger.warning("Bulletproof orchestrator is already active")
            return
        
        logger.info("Starting bulletproof orchestrator...")
        
        try:
            # Start individual protection systems
            await self._start_protection_systems()
            
            # Start system health monitoring
            self.system_health_monitor.start()
            
            # Start recovery coordinator
            self.recovery_coordinator.start()
            
            # Start alert manager
            self.alert_manager.start()
            
            # Start dashboard manager
            self.dashboard_manager.start()
            
            # Start orchestrator monitoring
            self._start_orchestrator_monitoring()
            
            self.is_active = True
            logger.info("Bulletproof orchestrator started successfully")
            
            # Send startup notification
            await self._send_system_notification("system_started", {
                "message": "All protection systems are now active",
                "timestamp": datetime.utcnow().isoformat(),
                "protection_mode": self.config.protection_mode.value
            })
            
        except Exception as e:
            logger.error(f"Failed to start bulletproof orchestrator: {e}")
            await self._emergency_startup_recovery()
            raise
    
    async def stop(self):
        """Stop the bulletproof orchestrator and all protection systems."""
        if not self.is_active:
            return
        
        logger.info("Stopping bulletproof orchestrator...")
        
        try:
            # Stop orchestrator monitoring
            self._stop_orchestrator_monitoring()
            
            # Stop managers
            self.dashboard_manager.stop()
            self.alert_manager.stop()
            self.recovery_coordinator.stop()
            self.system_health_monitor.stop()
            
            # Stop protection systems
            await self._stop_protection_systems()
            
            self.is_active = False
            logger.info("Bulletproof orchestrator stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping bulletproof orchestrator: {e}")
    
    async def _start_protection_systems(self):
        """Start all protection systems."""
        for system_name, system_info in self.protection_systems.items():
            try:
                system_instance = system_info["instance"]
                
                # Start system if it has a start method
                if hasattr(system_instance, 'start'):
                    if asyncio.iscoroutinefunction(system_instance.start):
                        await system_instance.start()
                    else:
                        system_instance.start()
                
                logger.info(f"Started protection system: {system_name}")
                
            except Exception as e:
                logger.error(f"Failed to start protection system {system_name}: {e}")
                # Continue with other systems
    
    async def _stop_protection_systems(self):
        """Stop all protection systems."""
        for system_name, system_info in self.protection_systems.items():
            try:
                system_instance = system_info["instance"]
                
                # Stop system if it has a stop method
                if hasattr(system_instance, 'stop'):
                    if asyncio.iscoroutinefunction(system_instance.stop):
                        await system_instance.stop()
                    else:
                        system_instance.stop()
                
                logger.info(f"Stopped protection system: {system_name}")
                
            except Exception as e:
                logger.error(f"Failed to stop protection system {system_name}: {e}")
    
    def _start_orchestrator_monitoring(self):
        """Start orchestrator monitoring thread."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True
            )
            self._monitoring_thread.start()
    
    def _stop_orchestrator_monitoring(self):
        """Stop orchestrator monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """Main monitoring loop for orchestrator."""
        while self._monitoring_active:
            try:
                # Collect system health metrics
                asyncio.run(self._collect_system_health_metrics())
                
                # Check for alerts
                asyncio.run(self._check_system_alerts())
                
                # Coordinate recovery actions
                asyncio.run(self._coordinate_recovery_actions())
                
                # Update dashboard
                asyncio.run(self._update_dashboard())
                
                # Sleep before next iteration
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Orchestrator monitoring loop error: {e}")
                time.sleep(30)  # Wait longer on error
    
    async def _collect_system_health_metrics(self):
        """Collect comprehensive system health metrics."""
        try:
            # Collect system resource metrics
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Collect protection system metrics
            protection_metrics = {}
            for system_name, system_info in self.protection_systems.items():
                try:
                    metrics_collector = system_info["metrics_collector"]
                    protection_metrics[system_name] = await metrics_collector()
                except Exception as e:
                    logger.error(f"Failed to collect metrics for {system_name}: {e}")
                    protection_metrics[system_name] = {"error": str(e)}
            
            # Calculate derived metrics
            error_rate = self._calculate_system_error_rate()
            response_time = self._calculate_avg_response_time()
            active_users = self._estimate_active_users()
            protection_effectiveness = self._calculate_protection_effectiveness()
            recovery_success_rate = self._calculate_recovery_success_rate()
            user_satisfaction = self._calculate_user_satisfaction()
            
            # Determine overall health status
            overall_status = self._determine_overall_health_status(
                cpu_usage, memory.percent, error_rate, response_time, user_satisfaction
            )
            
            # Create health metrics
            health_metrics = SystemHealthMetrics(
                overall_status=overall_status,
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_latency=self._estimate_network_latency(),
                error_rate=error_rate,
                response_time=response_time,
                active_users=active_users,
                protection_effectiveness=protection_effectiveness,
                recovery_success_rate=recovery_success_rate,
                user_satisfaction_score=user_satisfaction
            )
            
            # Store metrics
            self.system_metrics_history.append(health_metrics)
            
            # Keep only recent metrics (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.system_metrics_history = [
                m for m in self.system_metrics_history
                if m.timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Failed to collect system health metrics: {e}")
    
    def _determine_overall_health_status(self, cpu_usage: float, memory_usage: float,
                                       error_rate: float, response_time: float,
                                       user_satisfaction: float) -> SystemHealthStatus:
        """Determine overall system health status."""
        # Critical conditions
        if (cpu_usage > 95 or memory_usage > 95 or 
            error_rate > 0.2 or response_time > 10 or user_satisfaction < 0.3):
            return SystemHealthStatus.EMERGENCY
        
        # Degraded conditions
        if (cpu_usage > 85 or memory_usage > 85 or 
            error_rate > 0.1 or response_time > 5 or user_satisfaction < 0.5):
            return SystemHealthStatus.CRITICAL
        
        # Warning conditions
        if (cpu_usage > 75 or memory_usage > 75 or 
            error_rate > 0.05 or response_time > 3 or user_satisfaction < 0.7):
            return SystemHealthStatus.DEGRADED
        
        # Good conditions
        if (cpu_usage > 50 or memory_usage > 50 or 
            error_rate > 0.01 or response_time > 1.5 or user_satisfaction < 0.9):
            return SystemHealthStatus.GOOD
        
        return SystemHealthStatus.EXCELLENT
    
    async def _check_system_alerts(self):
        """Check for system alerts and trigger appropriate responses."""
        if not self.system_metrics_history:
            return
        
        latest_metrics = self.system_metrics_history[-1]
        new_alerts = []
        
        # Check alert thresholds
        thresholds = self.config.alert_thresholds
        
        if latest_metrics.cpu_usage > thresholds["cpu_usage"]:
            new_alerts.append({
                "type": "cpu_high",
                "severity": "warning",
                "message": f"High CPU usage: {latest_metrics.cpu_usage:.1f}%",
                "value": latest_metrics.cpu_usage,
                "threshold": thresholds["cpu_usage"],
                "timestamp": datetime.utcnow().isoformat()
            })
        
        if latest_metrics.memory_usage > thresholds["memory_usage"]:
            new_alerts.append({
                "type": "memory_high",
                "severity": "warning",
                "message": f"High memory usage: {latest_metrics.memory_usage:.1f}%",
                "value": latest_metrics.memory_usage,
                "threshold": thresholds["memory_usage"],
                "timestamp": datetime.utcnow().isoformat()
            })
        
        if latest_metrics.error_rate > thresholds["error_rate"]:
            new_alerts.append({
                "type": "error_rate_high",
                "severity": "critical",
                "message": f"High error rate: {latest_metrics.error_rate:.3f}",
                "value": latest_metrics.error_rate,
                "threshold": thresholds["error_rate"],
                "timestamp": datetime.utcnow().isoformat()
            })
        
        if latest_metrics.response_time > thresholds["response_time"]:
            new_alerts.append({
                "type": "response_time_high",
                "severity": "warning",
                "message": f"High response time: {latest_metrics.response_time:.2f}s",
                "value": latest_metrics.response_time,
                "threshold": thresholds["response_time"],
                "timestamp": datetime.utcnow().isoformat()
            })
        
        if latest_metrics.user_satisfaction_score < thresholds["user_satisfaction"]:
            new_alerts.append({
                "type": "user_satisfaction_low",
                "severity": "critical",
                "message": f"Low user satisfaction: {latest_metrics.user_satisfaction_score:.2f}",
                "value": latest_metrics.user_satisfaction_score,
                "threshold": thresholds["user_satisfaction"],
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Add new alerts
        for alert in new_alerts:
            await self.alert_manager.add_alert(alert)
            self.active_alerts.append(alert)
        
        # Clean up old alerts
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        self.active_alerts = [
            alert for alert in self.active_alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]
    
    async def _coordinate_recovery_actions(self):
        """Coordinate recovery actions across all protection systems."""
        if not self.config.auto_recovery_enabled:
            return
        
        if not self.system_metrics_history:
            return
        
        latest_metrics = self.system_metrics_history[-1]
        
        # Determine if recovery actions are needed
        recovery_needed = []
        
        if latest_metrics.overall_status in [SystemHealthStatus.CRITICAL, SystemHealthStatus.EMERGENCY]:
            recovery_needed.append("system_critical")
        
        if latest_metrics.error_rate > 0.1:
            recovery_needed.append("high_error_rate")
        
        if latest_metrics.response_time > 5.0:
            recovery_needed.append("slow_response")
        
        if latest_metrics.user_satisfaction_score < 0.5:
            recovery_needed.append("poor_user_experience")
        
        # Execute recovery actions
        for recovery_type in recovery_needed:
            await self._execute_recovery_action(recovery_type, latest_metrics)
    
    async def _execute_recovery_action(self, recovery_type: str, metrics: SystemHealthMetrics):
        """Execute specific recovery action."""
        recovery_action = {
            "type": recovery_type,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_snapshot": {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "error_rate": metrics.error_rate,
                "response_time": metrics.response_time
            },
            "actions_taken": [],
            "success": False
        }
        
        try:
            if recovery_type == "system_critical":
                # Trigger emergency mode across all systems
                await self._trigger_emergency_mode()
                recovery_action["actions_taken"].append("emergency_mode_activated")
            
            elif recovery_type == "high_error_rate":
                # Increase retry attempts and enable more aggressive fallbacks
                await self._enhance_error_recovery()
                recovery_action["actions_taken"].append("enhanced_error_recovery")
            
            elif recovery_type == "slow_response":
                # Enable performance optimizations
                await self._optimize_performance()
                recovery_action["actions_taken"].append("performance_optimization")
            
            elif recovery_type == "poor_user_experience":
                # Improve user experience through degradation and notifications
                await self._improve_user_experience()
                recovery_action["actions_taken"].append("user_experience_improvement")
            
            recovery_action["success"] = True
            logger.info(f"Recovery action completed: {recovery_type}")
            
        except Exception as e:
            logger.error(f"Recovery action failed: {recovery_type} - {e}")
            recovery_action["error"] = str(e)
        
        self.recovery_actions.append(recovery_action)
        
        # Keep only recent recovery actions
        if len(self.recovery_actions) > 100:
            self.recovery_actions = self.recovery_actions[-100:]
    
    async def _trigger_emergency_mode(self):
        """Trigger emergency mode across all protection systems."""
        logger.warning("Triggering emergency mode")
        
        # Set degradation to emergency mode
        if hasattr(self.degradation_manager, 'set_emergency_mode'):
            await self.degradation_manager.set_emergency_mode()
        
        # Enable maximum protection in failure prevention
        if hasattr(self.failure_prevention, 'enable_maximum_protection'):
            await self.failure_prevention.enable_maximum_protection()
        
        # Notify users about emergency mode
        await self._send_system_notification("emergency_mode", {
            "message": "System is in emergency mode. Core functionality remains available.",
            "estimated_recovery": "5-15 minutes"
        })
    
    async def _enhance_error_recovery(self):
        """Enhance error recovery mechanisms."""
        # Increase retry attempts
        self.intelligent_retry.config.max_attempts = 5
        
        # Enable more aggressive fallback generation
        if hasattr(self.fallback_generator, 'enable_aggressive_mode'):
            self.fallback_generator.enable_aggressive_mode()
        
        logger.info("Enhanced error recovery mechanisms")
    
    async def _optimize_performance(self):
        """Optimize system performance."""
        # Enable performance mode in degradation manager
        if hasattr(self.degradation_manager, 'enable_performance_mode'):
            await self.degradation_manager.enable_performance_mode()
        
        # Optimize UX protector settings
        if hasattr(self.ux_protector, 'enable_performance_mode'):
            self.ux_protector.enable_performance_mode()
        
        logger.info("Performance optimization enabled")
    
    async def _improve_user_experience(self):
        """Improve user experience through various mechanisms."""
        # Enable user experience optimization
        if hasattr(self.ux_protector, 'enable_optimization_mode'):
            self.ux_protector.enable_optimization_mode()
        
        # Send user notification about improvements
        await self._send_system_notification("experience_improvement", {
            "message": "We're optimizing your experience. Thank you for your patience."
        })
        
        logger.info("User experience improvement measures activated")
    
    async def _update_dashboard(self):
        """Update the system health dashboard."""
        if not self.system_metrics_history:
            return
        
        latest_metrics = self.system_metrics_history[-1]
        
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": latest_metrics.overall_status.value,
            "system_metrics": {
                "cpu_usage": latest_metrics.cpu_usage,
                "memory_usage": latest_metrics.memory_usage,
                "disk_usage": latest_metrics.disk_usage,
                "network_latency": latest_metrics.network_latency,
                "error_rate": latest_metrics.error_rate,
                "response_time": latest_metrics.response_time,
                "active_users": latest_metrics.active_users
            },
            "protection_metrics": {
                "effectiveness": latest_metrics.protection_effectiveness,
                "recovery_success_rate": latest_metrics.recovery_success_rate,
                "user_satisfaction": latest_metrics.user_satisfaction_score
            },
            "active_alerts": len(self.active_alerts),
            "recent_recovery_actions": len([
                action for action in self.recovery_actions
                if datetime.fromisoformat(action["timestamp"]) > datetime.utcnow() - timedelta(hours=1)
            ]),
            "protection_systems_status": await self._get_protection_systems_status(),
            "uptime": str(datetime.utcnow() - self.start_time)
        }
        
        await self.dashboard_manager.update_dashboard(dashboard_data)
    
    async def _get_protection_systems_status(self) -> List[ProtectionSystemStatus]:
        """Get status of all protection systems."""
        systems_status = []
        
        for system_name, system_info in self.protection_systems.items():
            try:
                health_check = system_info["health_check"]
                health_score = await health_check()
                
                # Get system metrics
                metrics_collector = system_info["metrics_collector"]
                metrics = await metrics_collector()
                
                status = ProtectionSystemStatus(
                    system_name=system_name,
                    is_active=True,
                    health_score=health_score,
                    last_action=metrics.get("last_action"),
                    last_action_time=metrics.get("last_action_time"),
                    metrics=metrics
                )
                
                systems_status.append(status)
                
            except Exception as e:
                logger.error(f"Failed to get status for {system_name}: {e}")
                status = ProtectionSystemStatus(
                    system_name=system_name,
                    is_active=False,
                    health_score=0.0,
                    last_action=None,
                    last_action_time=None,
                    metrics={"error": str(e)},
                    alerts=[f"Health check failed: {str(e)}"]
                )
                systems_status.append(status)
        
        return systems_status
    
    # System-specific health checks and metrics collectors
    async def _check_failure_prevention_health(self) -> float:
        """Check failure prevention system health."""
        try:
            status = self.failure_prevention.get_system_status()
            
            # Calculate health score based on various factors
            health_score = 1.0
            
            # Reduce score for recent failures
            if status["recent_failures"] > 10:
                health_score -= 0.3
            elif status["recent_failures"] > 5:
                health_score -= 0.1
            
            # Reduce score for open circuit breakers
            open_breakers = sum(1 for cb in status["circuit_breakers"].values() 
                              if cb["state"] == "open")
            if open_breakers > 0:
                health_score -= 0.2 * open_breakers
            
            # Reduce score for high resource usage
            resources = status["system_resources"]
            if resources["cpu_percent"] > 90:
                health_score -= 0.2
            if resources["memory_percent"] > 90:
                health_score -= 0.2
            
            return max(0.0, health_score)
            
        except Exception as e:
            logger.error(f"Failed to check failure prevention health: {e}")
            return 0.0
    
    async def _collect_failure_prevention_metrics(self) -> Dict[str, Any]:
        """Collect failure prevention system metrics."""
        try:
            status = self.failure_prevention.get_system_status()
            patterns = self.failure_prevention.get_failure_patterns()
            
            return {
                "recent_failures": status["recent_failures"],
                "circuit_breakers": status["circuit_breakers"],
                "system_resources": status["system_resources"],
                "failure_patterns": patterns,
                "monitoring_active": status["monitoring_active"],
                "last_action": "system_monitoring",
                "last_action_time": datetime.utcnow()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _check_degradation_manager_health(self) -> float:
        """Check degradation manager health."""
        try:
            # Check if degradation manager is responsive
            current_level = self.degradation_manager.current_degradation_level
            degraded_services = len(self.degradation_manager.degraded_services)
            
            health_score = 1.0
            
            # Reduce score based on degradation level
            if current_level == DegradationLevel.EMERGENCY_MODE:
                health_score = 0.3
            elif current_level == DegradationLevel.MAJOR_DEGRADATION:
                health_score = 0.5
            elif current_level == DegradationLevel.MINOR_DEGRADATION:
                health_score = 0.8
            
            # Reduce score for many degraded services
            if degraded_services > 5:
                health_score -= 0.2
            elif degraded_services > 2:
                health_score -= 0.1
            
            return max(0.0, health_score)
            
        except Exception as e:
            logger.error(f"Failed to check degradation manager health: {e}")
            return 0.0
    
    async def _collect_degradation_manager_metrics(self) -> Dict[str, Any]:
        """Collect degradation manager metrics."""
        try:
            return {
                "current_degradation_level": self.degradation_manager.current_degradation_level.value,
                "degraded_services": dict(self.degradation_manager.degraded_services),
                "degradation_strategies": len(self.degradation_manager.degradation_strategies),
                "system_metrics_history": len(self.degradation_manager.system_metrics_history),
                "user_preferences": len(self.degradation_manager.user_preferences),
                "last_action": "degradation_monitoring",
                "last_action_time": datetime.utcnow()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _check_ux_protector_health(self) -> float:
        """Check UX protector health."""
        try:
            experience_level = self.ux_protector.get_current_experience_level()
            metrics = self.ux_protector.get_experience_metrics()
            
            # Map experience level to health score
            level_scores = {
                UserExperienceLevel.EXCELLENT: 1.0,
                UserExperienceLevel.GOOD: 0.8,
                UserExperienceLevel.ACCEPTABLE: 0.6,
                UserExperienceLevel.DEGRADED: 0.4,
                UserExperienceLevel.MINIMAL: 0.2
            }
            
            base_score = level_scores.get(experience_level, 0.5)
            
            # Adjust based on metrics
            if metrics.success_rate < 0.8:
                base_score -= 0.2
            if metrics.avg_response_time > 3.0:
                base_score -= 0.1
            
            return max(0.0, base_score)
            
        except Exception as e:
            logger.error(f"Failed to check UX protector health: {e}")
            return 0.0
    
    async def _collect_ux_protector_metrics(self) -> Dict[str, Any]:
        """Collect UX protector metrics."""
        try:
            metrics = self.ux_protector.get_experience_metrics()
            loading_states = self.ux_protector.get_loading_states()
            
            return {
                "experience_level": metrics.experience_level.value,
                "avg_response_time": metrics.avg_response_time,
                "success_rate": metrics.success_rate,
                "error_count": metrics.error_count,
                "user_satisfaction_score": metrics.user_satisfaction_score,
                "active_loading_states": len(loading_states),
                "total_user_actions": len(self.ux_protector.user_actions),
                "last_action": "experience_monitoring",
                "last_action_time": datetime.utcnow()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _check_intelligent_retry_health(self) -> float:
        """Check intelligent retry system health."""
        try:
            # Check if retry system is functioning
            failure_history_size = len(self.intelligent_retry.failure_history)
            circuit_breakers = len(self.intelligent_retry.circuit_breakers)
            
            health_score = 1.0
            
            # Reduce score for too many failures
            if failure_history_size > 100:
                health_score -= 0.2
            elif failure_history_size > 50:
                health_score -= 0.1
            
            # Reduce score for open circuit breakers
            open_breakers = sum(1 for cb in self.intelligent_retry.circuit_breakers.values()
                              if cb.get('state') == 'open')
            if open_breakers > 0:
                health_score -= 0.1 * open_breakers
            
            return max(0.0, health_score)
            
        except Exception as e:
            logger.error(f"Failed to check intelligent retry health: {e}")
            return 0.0
    
    async def _collect_intelligent_retry_metrics(self) -> Dict[str, Any]:
        """Collect intelligent retry metrics."""
        try:
            return {
                "failure_history_size": len(self.intelligent_retry.failure_history),
                "circuit_breakers": len(self.intelligent_retry.circuit_breakers),
                "config": {
                    "max_attempts": self.intelligent_retry.config.max_attempts,
                    "base_delay": self.intelligent_retry.config.base_delay,
                    "max_delay": self.intelligent_retry.config.max_delay
                },
                "last_action": "retry_monitoring",
                "last_action_time": datetime.utcnow()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _check_fallback_generator_health(self) -> float:
        """Check fallback generator health."""
        try:
            # Check if fallback generator is functioning
            templates_count = len(self.fallback_generator.fallback_templates)
            usage_patterns = len(self.fallback_generator.usage_patterns)
            
            health_score = 1.0
            
            # Ensure we have sufficient templates
            if templates_count < 5:
                health_score -= 0.2
            
            return max(0.0, health_score)
            
        except Exception as e:
            logger.error(f"Failed to check fallback generator health: {e}")
            return 0.0
    
    async def _collect_fallback_generator_metrics(self) -> Dict[str, Any]:
        """Collect fallback generator metrics."""
        try:
            return {
                "fallback_templates": len(self.fallback_generator.fallback_templates),
                "usage_patterns": len(self.fallback_generator.usage_patterns),
                "template_types": list(self.fallback_generator.fallback_templates.keys()),
                "last_action": "fallback_monitoring",
                "last_action_time": datetime.utcnow()
            }
        except Exception as e:
            return {"error": str(e)}
    
    # Callback handlers for cross-system coordination
    async def _handle_system_failure(self, failure_event: FailureEvent):
        """Handle system failure events for coordination."""
        logger.warning(f"System failure detected: {failure_event.failure_type.value}")
        
        # Coordinate response across systems
        if failure_event.failure_type.value in ["memory_error", "cpu_overload"]:
            # Trigger performance optimization
            await self._optimize_performance()
        
        elif failure_event.failure_type.value in ["network_error", "timeout_error"]:
            # Enhance retry mechanisms
            await self._enhance_error_recovery()
        
        # Notify user experience protector
        if hasattr(self.ux_protector, 'handle_system_failure'):
            await self.ux_protector.handle_system_failure(failure_event)
    
    async def _handle_experience_change(self, experience_data: Dict[str, Any]):
        """Handle user experience changes."""
        experience_level = experience_data.get("experience_level")
        
        if experience_level in ["degraded", "minimal"]:
            logger.warning(f"User experience degraded: {experience_level}")
            
            # Trigger user experience improvement
            await self._improve_user_experience()
            
            # Notify degradation manager
            if hasattr(self.degradation_manager, 'handle_experience_degradation'):
                await self.degradation_manager.handle_experience_degradation(experience_data)
    
    async def _handle_degradation_event(self, degradation_data: Dict[str, Any]):
        """Handle degradation events."""
        service_name = degradation_data.get("service_name")
        degradation_level = degradation_data.get("level")
        
        logger.info(f"Degradation event: {service_name} -> {degradation_level}")
        
        # Coordinate with other systems
        if degradation_level in ["major_degradation", "emergency_mode"]:
            # Enhance failure prevention
            if hasattr(self.failure_prevention, 'increase_protection_level'):
                await self.failure_prevention.increase_protection_level()
    
    # Utility methods
    def _calculate_system_error_rate(self) -> float:
        """Calculate system-wide error rate."""
        try:
            if hasattr(self.ux_protector, 'user_actions') and self.ux_protector.user_actions:
                recent_actions = [
                    action for action in self.ux_protector.user_actions
                    if action.timestamp > datetime.utcnow() - timedelta(minutes=10)
                ]
                
                if recent_actions:
                    failed_actions = sum(1 for action in recent_actions if not action.success)
                    return failed_actions / len(recent_actions)
            
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time."""
        try:
            if hasattr(self.ux_protector, 'user_actions') and self.ux_protector.user_actions:
                recent_actions = [
                    action for action in self.ux_protector.user_actions
                    if action.timestamp > datetime.utcnow() - timedelta(minutes=10)
                ]
                
                if recent_actions:
                    return sum(action.response_time for action in recent_actions) / len(recent_actions)
            
            return 0.0
        except Exception:
            return 0.0
    
    def _estimate_active_users(self) -> int:
        """Estimate number of active users."""
        try:
            if hasattr(self.ux_protector, 'user_actions') and self.ux_protector.user_actions:
                recent_actions = [
                    action for action in self.ux_protector.user_actions
                    if action.timestamp > datetime.utcnow() - timedelta(minutes=30)
                ]
                
                unique_users = set(
                    action.user_id for action in recent_actions 
                    if action.user_id
                )
                return len(unique_users)
            
            return 1  # At least one user (anonymous)
        except Exception:
            return 1
    
    def _calculate_protection_effectiveness(self) -> float:
        """Calculate overall protection effectiveness."""
        try:
            # Combine metrics from all protection systems
            failure_prevention_health = asyncio.run(self._check_failure_prevention_health())
            degradation_health = asyncio.run(self._check_degradation_manager_health())
            ux_health = asyncio.run(self._check_ux_protector_health())
            retry_health = asyncio.run(self._check_intelligent_retry_health())
            fallback_health = asyncio.run(self._check_fallback_generator_health())
            
            # Weighted average
            weights = [0.25, 0.25, 0.25, 0.15, 0.10]
            healths = [failure_prevention_health, degradation_health, ux_health, 
                      retry_health, fallback_health]
            
            return sum(w * h for w, h in zip(weights, healths))
        except Exception:
            return 0.5
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate."""
        try:
            if not self.recovery_actions:
                return 1.0
            
            recent_actions = [
                action for action in self.recovery_actions
                if datetime.fromisoformat(action["timestamp"]) > datetime.utcnow() - timedelta(hours=24)
            ]
            
            if not recent_actions:
                return 1.0
            
            successful_actions = sum(1 for action in recent_actions if action["success"])
            return successful_actions / len(recent_actions)
        except Exception:
            return 0.5
    
    def _calculate_user_satisfaction(self) -> float:
        """Calculate user satisfaction score."""
        try:
            metrics = self.ux_protector.get_experience_metrics()
            return metrics.user_satisfaction_score
        except Exception:
            return 0.5
    
    def _estimate_network_latency(self) -> float:
        """Estimate network latency."""
        # In a real implementation, this would measure actual network latency
        # For now, return a simulated value
        import random
        return random.uniform(10, 100)  # 10-100ms
    
    async def _send_system_notification(self, notification_type: str, data: Dict[str, Any]):
        """Send system notification to users."""
        try:
            notification = {
                "type": notification_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
            
            # In a real implementation, this would send notifications through
            # WebSocket, push notifications, or other channels
            logger.info(f"System notification: {notification}")
            
        except Exception as e:
            logger.error(f"Failed to send system notification: {e}")
    
    async def _emergency_startup_recovery(self):
        """Emergency recovery during startup failure."""
        logger.error("Attempting emergency startup recovery")
        
        try:
            # Try to start with minimal configuration
            minimal_config = UnifiedConfiguration(
                protection_mode=ProtectionMode.EMERGENCY_MODE,
                auto_recovery_enabled=False,
                predictive_prevention_enabled=False,
                user_experience_optimization=False,
                intelligent_routing_enabled=False,
                fallback_generation_enabled=True,
                degradation_learning_enabled=False,
                real_time_monitoring=False,
                alert_thresholds={},
                recovery_timeouts={},
                user_notification_settings={}
            )
            
            self.config = minimal_config
            
            # Try to start essential systems only
            essential_systems = ["failure_prevention", "ux_protector"]
            for system_name in essential_systems:
                if system_name in self.protection_systems:
                    try:
                        system_instance = self.protection_systems[system_name]["instance"]
                        if hasattr(system_instance, 'start'):
                            if asyncio.iscoroutinefunction(system_instance.start):
                                await system_instance.start()
                            else:
                                system_instance.start()
                        logger.info(f"Emergency startup: {system_name} started")
                    except Exception as e:
                        logger.error(f"Emergency startup failed for {system_name}: {e}")
            
            self.is_active = True
            logger.warning("Emergency startup recovery completed with minimal functionality")
            
        except Exception as e:
            logger.critical(f"Emergency startup recovery failed: {e}")
            raise
    
    # Public API methods
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information."""
        if not self.system_metrics_history:
            return {"status": "initializing"}
        
        latest_metrics = self.system_metrics_history[-1]
        
        return {
            "overall_status": latest_metrics.overall_status.value,
            "timestamp": latest_metrics.timestamp.isoformat(),
            "metrics": {
                "cpu_usage": latest_metrics.cpu_usage,
                "memory_usage": latest_metrics.memory_usage,
                "disk_usage": latest_metrics.disk_usage,
                "network_latency": latest_metrics.network_latency,
                "error_rate": latest_metrics.error_rate,
                "response_time": latest_metrics.response_time,
                "active_users": latest_metrics.active_users,
                "protection_effectiveness": latest_metrics.protection_effectiveness,
                "recovery_success_rate": latest_metrics.recovery_success_rate,
                "user_satisfaction_score": latest_metrics.user_satisfaction_score
            },
            "active_alerts": len(self.active_alerts),
            "recent_recovery_actions": len([
                action for action in self.recovery_actions
                if datetime.fromisoformat(action["timestamp"]) > datetime.utcnow() - timedelta(hours=1)
            ]),
            "uptime": str(datetime.utcnow() - self.start_time),
            "protection_mode": self.config.protection_mode.value,
            "is_active": self.is_active
        }
    
    def get_protection_systems_status(self) -> Dict[str, Any]:
        """Get status of all protection systems."""
        try:
            return asyncio.run(self._get_protection_systems_status())
        except Exception as e:
            logger.error(f"Failed to get protection systems status: {e}")
            return {"error": str(e)}
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        return self.active_alerts.copy()
    
    def get_recent_recovery_actions(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent recovery actions."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            action for action in self.recovery_actions
            if datetime.fromisoformat(action["timestamp"]) > cutoff_time
        ]
    
    def update_configuration(self, new_config: UnifiedConfiguration):
        """Update unified configuration."""
        self.config = new_config
        self.configuration_manager.update_config(new_config)
        logger.info("Bulletproof orchestrator configuration updated")
    
    def get_configuration(self) -> UnifiedConfiguration:
        """Get current configuration."""
        return self.config


# Supporting classes for orchestrator components
class SystemHealthMonitor:
    """Monitors overall system health."""
    
    def __init__(self):
        self.is_active = False
    
    def start(self):
        """Start health monitoring."""
        self.is_active = True
        logger.info("System health monitor started")
    
    def stop(self):
        """Stop health monitoring."""
        self.is_active = False
        logger.info("System health monitor stopped")


class RecoveryCoordinator:
    """Coordinates recovery actions across systems."""
    
    def __init__(self):
        self.is_active = False
    
    def start(self):
        """Start recovery coordination."""
        self.is_active = True
        logger.info("Recovery coordinator started")
    
    def stop(self):
        """Stop recovery coordination."""
        self.is_active = False
        logger.info("Recovery coordinator stopped")


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self.is_active = False
        self.alerts: List[Dict[str, Any]] = []
    
    def start(self):
        """Start alert management."""
        self.is_active = True
        logger.info("Alert manager started")
    
    def stop(self):
        """Stop alert management."""
        self.is_active = False
        logger.info("Alert manager stopped")
    
    async def add_alert(self, alert: Dict[str, Any]):
        """Add a new alert."""
        self.alerts.append(alert)
        logger.warning(f"Alert added: {alert['type']} - {alert['message']}")


class DashboardManager:
    """Manages the system health dashboard."""
    
    def __init__(self):
        self.is_active = False
        self.dashboard_data: Dict[str, Any] = {}
    
    def start(self):
        """Start dashboard management."""
        self.is_active = True
        logger.info("Dashboard manager started")
    
    def stop(self):
        """Stop dashboard management."""
        self.is_active = False
        logger.info("Dashboard manager stopped")
    
    async def update_dashboard(self, data: Dict[str, Any]):
        """Update dashboard data."""
        self.dashboard_data = data
        # In a real implementation, this would update the actual dashboard


class ConfigurationManager:
    """Manages unified configuration."""
    
    def __init__(self, config: UnifiedConfiguration):
        self.config = config
    
    def update_config(self, new_config: UnifiedConfiguration):
        """Update configuration."""
        self.config = new_config
        logger.info("Configuration updated")
    
    def get_config(self) -> UnifiedConfiguration:
        """Get current configuration."""
        return self.config


# Global orchestrator instance
bulletproof_orchestrator = BulletproofOrchestrator()


# Convenience functions
async def start_bulletproof_system(config: Optional[UnifiedConfiguration] = None):
    """Start the bulletproof system with optional configuration."""
    if config:
        bulletproof_orchestrator.update_configuration(config)
    
    await bulletproof_orchestrator.start()


async def stop_bulletproof_system():
    """Stop the bulletproof system."""
    await bulletproof_orchestrator.stop()


def get_system_health() -> Dict[str, Any]:
    """Get current system health."""
    return bulletproof_orchestrator.get_system_health()


def get_system_dashboard() -> Dict[str, Any]:
    """Get system dashboard data."""
    return bulletproof_orchestrator.dashboard_manager.dashboard_data


@asynccontextmanager
async def bulletproof_protection():
    """Context manager for bulletproof protection."""
    try:
        if not bulletproof_orchestrator.is_active:
            await bulletproof_orchestrator.start()
        yield bulletproof_orchestrator
    finally:
        # Keep orchestrator running - don't stop on context exit
        pass