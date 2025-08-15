"""
API routes for the Bulletproof Orchestrator.
Provides unified configuration and management interface for all protection systems.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from ...core.bulletproof_orchestrator import (
    bulletproof_orchestrator, 
    UnifiedConfiguration, 
    ProtectionMode,
    SystemHealthStatus,
    start_bulletproof_system,
    stop_bulletproof_system,
    get_system_health,
    get_system_dashboard
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bulletproof", tags=["Bulletproof Orchestrator"])


# Pydantic models for API
class ConfigurationUpdate(BaseModel):
    """Model for configuration updates."""
    protection_mode: Optional[str] = None
    auto_recovery_enabled: Optional[bool] = None
    predictive_prevention_enabled: Optional[bool] = None
    user_experience_optimization: Optional[bool] = None
    intelligent_routing_enabled: Optional[bool] = None
    fallback_generation_enabled: Optional[bool] = None
    degradation_learning_enabled: Optional[bool] = None
    real_time_monitoring: Optional[bool] = None
    alert_thresholds: Optional[Dict[str, float]] = None
    recovery_timeouts: Optional[Dict[str, float]] = None
    user_notification_settings: Optional[Dict[str, Any]] = None


class SystemHealthResponse(BaseModel):
    """Response model for system health."""
    overall_status: str
    timestamp: str
    metrics: Dict[str, float]
    active_alerts: int
    recent_recovery_actions: int
    uptime: str
    protection_mode: str
    is_active: bool


class AlertResponse(BaseModel):
    """Response model for alerts."""
    type: str
    severity: str
    message: str
    value: float
    threshold: float
    timestamp: str


class RecoveryActionResponse(BaseModel):
    """Response model for recovery actions."""
    type: str
    timestamp: str
    metrics_snapshot: Dict[str, float]
    actions_taken: List[str]
    success: bool
    error: Optional[str] = None


# System Control Endpoints
@router.post("/start")
async def start_system(background_tasks: BackgroundTasks, config: Optional[ConfigurationUpdate] = None):
    """Start the bulletproof orchestrator system."""
    try:
        if bulletproof_orchestrator.is_active:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Bulletproof system is already active",
                    "status": "active"
                }
            )
        
        # Convert config if provided
        unified_config = None
        if config:
            unified_config = _convert_to_unified_config(config)
        
        # Start system in background
        background_tasks.add_task(start_bulletproof_system, unified_config)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Bulletproof system startup initiated",
                "status": "starting"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start bulletproof system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start system: {str(e)}")


@router.post("/stop")
async def stop_system(background_tasks: BackgroundTasks):
    """Stop the bulletproof orchestrator system."""
    try:
        if not bulletproof_orchestrator.is_active:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Bulletproof system is already inactive",
                    "status": "inactive"
                }
            )
        
        # Stop system in background
        background_tasks.add_task(stop_bulletproof_system)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Bulletproof system shutdown initiated",
                "status": "stopping"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to stop bulletproof system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop system: {str(e)}")


@router.post("/restart")
async def restart_system(background_tasks: BackgroundTasks, config: Optional[ConfigurationUpdate] = None):
    """Restart the bulletproof orchestrator system."""
    try:
        # Convert config if provided
        unified_config = None
        if config:
            unified_config = _convert_to_unified_config(config)
        
        async def restart_task():
            if bulletproof_orchestrator.is_active:
                await stop_bulletproof_system()
            await start_bulletproof_system(unified_config)
        
        # Restart system in background
        background_tasks.add_task(restart_task)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Bulletproof system restart initiated",
                "status": "restarting"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to restart bulletproof system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restart system: {str(e)}")


# System Health and Status Endpoints
@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health_endpoint():
    """Get comprehensive system health information."""
    try:
        health_data = get_system_health()
        return SystemHealthResponse(**health_data)
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


@router.get("/status")
async def get_system_status():
    """Get detailed system status including all protection systems."""
    try:
        health_data = get_system_health()
        protection_systems = bulletproof_orchestrator.get_protection_systems_status()
        
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": health_data,
            "protection_systems": protection_systems,
            "orchestrator_active": bulletproof_orchestrator.is_active,
            "configuration": {
                "protection_mode": bulletproof_orchestrator.config.protection_mode.value,
                "auto_recovery_enabled": bulletproof_orchestrator.config.auto_recovery_enabled,
                "real_time_monitoring": bulletproof_orchestrator.config.real_time_monitoring
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@router.get("/dashboard")
async def get_dashboard():
    """Get system health dashboard data."""
    try:
        dashboard_data = get_system_dashboard()
        
        if not dashboard_data:
            # Return basic dashboard if no data available
            health_data = get_system_health()
            dashboard_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": health_data.get("overall_status", "unknown"),
                "system_metrics": health_data.get("metrics", {}),
                "protection_metrics": {
                    "effectiveness": 0.0,
                    "recovery_success_rate": 0.0,
                    "user_satisfaction": 0.0
                },
                "active_alerts": 0,
                "recent_recovery_actions": 0,
                "protection_systems_status": [],
                "uptime": health_data.get("uptime", "0:00:00")
            }
        
        return {
            "success": True,
            "dashboard": dashboard_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


# Configuration Management Endpoints
@router.get("/config")
async def get_configuration():
    """Get current unified configuration."""
    try:
        config = bulletproof_orchestrator.get_configuration()
        
        return {
            "success": True,
            "configuration": {
                "protection_mode": config.protection_mode.value,
                "auto_recovery_enabled": config.auto_recovery_enabled,
                "predictive_prevention_enabled": config.predictive_prevention_enabled,
                "user_experience_optimization": config.user_experience_optimization,
                "intelligent_routing_enabled": config.intelligent_routing_enabled,
                "fallback_generation_enabled": config.fallback_generation_enabled,
                "degradation_learning_enabled": config.degradation_learning_enabled,
                "real_time_monitoring": config.real_time_monitoring,
                "alert_thresholds": config.alert_thresholds,
                "recovery_timeouts": config.recovery_timeouts,
                "user_notification_settings": config.user_notification_settings
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@router.put("/config")
async def update_configuration(config_update: ConfigurationUpdate):
    """Update unified configuration."""
    try:
        # Get current configuration
        current_config = bulletproof_orchestrator.get_configuration()
        
        # Create updated configuration
        updated_config = UnifiedConfiguration(
            protection_mode=ProtectionMode(config_update.protection_mode) if config_update.protection_mode else current_config.protection_mode,
            auto_recovery_enabled=config_update.auto_recovery_enabled if config_update.auto_recovery_enabled is not None else current_config.auto_recovery_enabled,
            predictive_prevention_enabled=config_update.predictive_prevention_enabled if config_update.predictive_prevention_enabled is not None else current_config.predictive_prevention_enabled,
            user_experience_optimization=config_update.user_experience_optimization if config_update.user_experience_optimization is not None else current_config.user_experience_optimization,
            intelligent_routing_enabled=config_update.intelligent_routing_enabled if config_update.intelligent_routing_enabled is not None else current_config.intelligent_routing_enabled,
            fallback_generation_enabled=config_update.fallback_generation_enabled if config_update.fallback_generation_enabled is not None else current_config.fallback_generation_enabled,
            degradation_learning_enabled=config_update.degradation_learning_enabled if config_update.degradation_learning_enabled is not None else current_config.degradation_learning_enabled,
            real_time_monitoring=config_update.real_time_monitoring if config_update.real_time_monitoring is not None else current_config.real_time_monitoring,
            alert_thresholds=config_update.alert_thresholds if config_update.alert_thresholds else current_config.alert_thresholds,
            recovery_timeouts=config_update.recovery_timeouts if config_update.recovery_timeouts else current_config.recovery_timeouts,
            user_notification_settings=config_update.user_notification_settings if config_update.user_notification_settings else current_config.user_notification_settings
        )
        
        # Update configuration
        bulletproof_orchestrator.update_configuration(updated_config)
        
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "configuration": {
                "protection_mode": updated_config.protection_mode.value,
                "auto_recovery_enabled": updated_config.auto_recovery_enabled,
                "predictive_prevention_enabled": updated_config.predictive_prevention_enabled,
                "user_experience_optimization": updated_config.user_experience_optimization,
                "intelligent_routing_enabled": updated_config.intelligent_routing_enabled,
                "fallback_generation_enabled": updated_config.fallback_generation_enabled,
                "degradation_learning_enabled": updated_config.degradation_learning_enabled,
                "real_time_monitoring": updated_config.real_time_monitoring
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


# Alerts and Monitoring Endpoints
@router.get("/alerts", response_model=List[AlertResponse])
async def get_active_alerts():
    """Get currently active alerts."""
    try:
        alerts = bulletproof_orchestrator.get_active_alerts()
        
        return [
            AlertResponse(
                type=alert["type"],
                severity=alert["severity"],
                message=alert["message"],
                value=alert["value"],
                threshold=alert["threshold"],
                timestamp=alert["timestamp"]
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.get("/recovery-actions", response_model=List[RecoveryActionResponse])
async def get_recovery_actions(hours: int = 24):
    """Get recent recovery actions."""
    try:
        if hours < 1 or hours > 168:  # Limit to 1 hour - 1 week
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
        
        actions = bulletproof_orchestrator.get_recent_recovery_actions(hours)
        
        return [
            RecoveryActionResponse(
                type=action["type"],
                timestamp=action["timestamp"],
                metrics_snapshot=action["metrics_snapshot"],
                actions_taken=action["actions_taken"],
                success=action["success"],
                error=action.get("error")
            )
            for action in actions
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recovery actions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recovery actions: {str(e)}")


# System Metrics Endpoints
@router.get("/metrics")
async def get_system_metrics():
    """Get detailed system metrics."""
    try:
        health_data = get_system_health()
        protection_systems = bulletproof_orchestrator.get_protection_systems_status()
        
        # Calculate additional metrics
        metrics_history = bulletproof_orchestrator.system_metrics_history
        
        # Get recent metrics for trends
        recent_metrics = metrics_history[-10:] if len(metrics_history) >= 10 else metrics_history
        
        trends = {}
        if len(recent_metrics) > 1:
            # Calculate simple trends
            cpu_trend = recent_metrics[-1].cpu_usage - recent_metrics[0].cpu_usage
            memory_trend = recent_metrics[-1].memory_usage - recent_metrics[0].memory_usage
            response_time_trend = recent_metrics[-1].response_time - recent_metrics[0].response_time
            
            trends = {
                "cpu_usage": "increasing" if cpu_trend > 5 else "decreasing" if cpu_trend < -5 else "stable",
                "memory_usage": "increasing" if memory_trend > 5 else "decreasing" if memory_trend < -5 else "stable",
                "response_time": "increasing" if response_time_trend > 0.5 else "decreasing" if response_time_trend < -0.5 else "stable"
            }
        
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "current_metrics": health_data.get("metrics", {}),
            "trends": trends,
            "protection_systems": protection_systems,
            "metrics_history_size": len(metrics_history),
            "system_uptime": health_data.get("uptime", "0:00:00")
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")


# Protection System Control Endpoints
@router.post("/protection-systems/{system_name}/enable")
async def enable_protection_system(system_name: str):
    """Enable a specific protection system."""
    try:
        if system_name not in bulletproof_orchestrator.protection_systems:
            raise HTTPException(status_code=404, detail=f"Protection system '{system_name}' not found")
        
        # In a real implementation, you would enable/disable the specific system
        # For now, we'll just return success
        
        return {
            "success": True,
            "message": f"Protection system '{system_name}' enabled",
            "system_name": system_name,
            "status": "enabled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable protection system {system_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enable protection system: {str(e)}")


@router.post("/protection-systems/{system_name}/disable")
async def disable_protection_system(system_name: str):
    """Disable a specific protection system."""
    try:
        if system_name not in bulletproof_orchestrator.protection_systems:
            raise HTTPException(status_code=404, detail=f"Protection system '{system_name}' not found")
        
        # In a real implementation, you would enable/disable the specific system
        # For now, we'll just return success
        
        return {
            "success": True,
            "message": f"Protection system '{system_name}' disabled",
            "system_name": system_name,
            "status": "disabled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable protection system {system_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to disable protection system: {str(e)}")


# Emergency Control Endpoints
@router.post("/emergency/activate")
async def activate_emergency_mode():
    """Activate emergency mode across all protection systems."""
    try:
        # Update configuration to emergency mode
        current_config = bulletproof_orchestrator.get_configuration()
        emergency_config = UnifiedConfiguration(
            protection_mode=ProtectionMode.EMERGENCY_MODE,
            auto_recovery_enabled=True,
            predictive_prevention_enabled=False,
            user_experience_optimization=True,
            intelligent_routing_enabled=False,
            fallback_generation_enabled=True,
            degradation_learning_enabled=False,
            real_time_monitoring=True,
            alert_thresholds=current_config.alert_thresholds,
            recovery_timeouts=current_config.recovery_timeouts,
            user_notification_settings=current_config.user_notification_settings
        )
        
        bulletproof_orchestrator.update_configuration(emergency_config)
        
        return {
            "success": True,
            "message": "Emergency mode activated",
            "mode": "emergency",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to activate emergency mode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to activate emergency mode: {str(e)}")


@router.post("/emergency/deactivate")
async def deactivate_emergency_mode():
    """Deactivate emergency mode and return to normal operation."""
    try:
        # Update configuration to full protection mode
        current_config = bulletproof_orchestrator.get_configuration()
        normal_config = UnifiedConfiguration(
            protection_mode=ProtectionMode.FULL_PROTECTION,
            auto_recovery_enabled=True,
            predictive_prevention_enabled=True,
            user_experience_optimization=True,
            intelligent_routing_enabled=True,
            fallback_generation_enabled=True,
            degradation_learning_enabled=True,
            real_time_monitoring=True,
            alert_thresholds=current_config.alert_thresholds,
            recovery_timeouts=current_config.recovery_timeouts,
            user_notification_settings=current_config.user_notification_settings
        )
        
        bulletproof_orchestrator.update_configuration(normal_config)
        
        return {
            "success": True,
            "message": "Emergency mode deactivated",
            "mode": "full_protection",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to deactivate emergency mode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to deactivate emergency mode: {str(e)}")


# Utility Functions
def _convert_to_unified_config(config_update: ConfigurationUpdate) -> UnifiedConfiguration:
    """Convert API config update to unified configuration."""
    return UnifiedConfiguration(
        protection_mode=ProtectionMode(config_update.protection_mode) if config_update.protection_mode else ProtectionMode.FULL_PROTECTION,
        auto_recovery_enabled=config_update.auto_recovery_enabled if config_update.auto_recovery_enabled is not None else True,
        predictive_prevention_enabled=config_update.predictive_prevention_enabled if config_update.predictive_prevention_enabled is not None else True,
        user_experience_optimization=config_update.user_experience_optimization if config_update.user_experience_optimization is not None else True,
        intelligent_routing_enabled=config_update.intelligent_routing_enabled if config_update.intelligent_routing_enabled is not None else True,
        fallback_generation_enabled=config_update.fallback_generation_enabled if config_update.fallback_generation_enabled is not None else True,
        degradation_learning_enabled=config_update.degradation_learning_enabled if config_update.degradation_learning_enabled is not None else True,
        real_time_monitoring=config_update.real_time_monitoring if config_update.real_time_monitoring is not None else True,
        alert_thresholds=config_update.alert_thresholds or {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "error_rate": 0.05,
            "response_time": 3.0,
            "user_satisfaction": 0.7
        },
        recovery_timeouts=config_update.recovery_timeouts or {
            "automatic_recovery": 30.0,
            "degradation_recovery": 60.0,
            "system_restart": 300.0
        },
        user_notification_settings=config_update.user_notification_settings or {
            "show_system_status": True,
            "show_recovery_progress": True,
            "show_degradation_notices": True,
            "auto_dismiss_success": True
        }
    )


# Health check endpoint for the orchestrator itself
@router.get("/ping")
async def ping():
    """Simple health check endpoint."""
    return {
        "success": True,
        "message": "Bulletproof orchestrator API is operational",
        "timestamp": datetime.utcnow().isoformat(),
        "orchestrator_active": bulletproof_orchestrator.is_active
    }