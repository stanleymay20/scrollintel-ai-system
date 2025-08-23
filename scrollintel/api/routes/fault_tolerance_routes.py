"""
API routes for Fault Tolerance and Recovery System

Provides REST endpoints for:
- Circuit breaker management and monitoring
- Retry configuration and statistics
- Graceful degradation control
- Recovery operation management
- System health monitoring

Requirements: 4.3, 9.2
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from scrollintel.core.fault_tolerance import (
    fault_tolerance_manager,
    CircuitBreakerConfig,
    RetryConfig,
    SystemFailure,
    FailureType,
    DegradationLevel,
    CircuitBreakerState
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/fault-tolerance", tags=["fault-tolerance"])

# Pydantic models for API
class CircuitBreakerConfigModel(BaseModel):
    name: str
    failure_threshold: int = Field(default=5, ge=1, le=100)
    recovery_timeout: int = Field(default=60, ge=1, le=3600)
    success_threshold: int = Field(default=3, ge=1, le=10)
    timeout: int = Field(default=30, ge=1, le=300)

class RetryConfigModel(BaseModel):
    max_attempts: int = Field(default=3, ge=1, le=10)
    base_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    max_delay: float = Field(default=60.0, ge=1.0, le=300.0)
    exponential_base: float = Field(default=2.0, ge=1.1, le=10.0)
    jitter: bool = True

class SystemFailureModel(BaseModel):
    failure_id: str
    failure_type: FailureType
    component: str
    message: str
    severity: str = Field(pattern="^(low|medium|high|critical)$")
    context: Dict[str, Any] = {}
    stack_trace: Optional[str] = None

class ServiceDegradationModel(BaseModel):
    service_name: str
    degradation_level: DegradationLevel

class CircuitBreakerStatusModel(BaseModel):
    name: str
    state: CircuitBreakerState
    failure_count: int
    success_count: int
    last_failure_time: Optional[float]
    next_attempt_time: Optional[float]

class SystemHealthModel(BaseModel):
    timestamp: str
    overall_status: str
    circuit_breakers: Dict[str, Dict[str, Any]]
    degraded_services: Dict[str, str]
    recent_recoveries: List[Dict[str, Any]]

class RecoveryResultModel(BaseModel):
    recovery_id: str
    success: bool
    strategy: str
    duration: float
    actions_taken: List[str]
    health_status: Dict[str, Any]
    timestamp: str

# Circuit Breaker Management Endpoints

@router.post("/circuit-breakers", response_model=Dict[str, str])
async def create_circuit_breaker(config: CircuitBreakerConfigModel):
    """Create a new circuit breaker with specified configuration"""
    try:
        cb_config = CircuitBreakerConfig(
            name=config.name,
            failure_threshold=config.failure_threshold,
            recovery_timeout=config.recovery_timeout,
            success_threshold=config.success_threshold,
            timeout=config.timeout
        )
        
        circuit_breaker = fault_tolerance_manager.create_circuit_breaker(cb_config)
        
        logger.info(f"Created circuit breaker: {config.name}")
        return {
            "message": f"Circuit breaker '{config.name}' created successfully",
            "name": config.name,
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Failed to create circuit breaker {config.name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create circuit breaker: {str(e)}")

@router.get("/circuit-breakers", response_model=List[CircuitBreakerStatusModel])
async def list_circuit_breakers():
    """List all circuit breakers and their current status"""
    try:
        circuit_breakers = []
        
        for name, cb in fault_tolerance_manager.circuit_breakers.items():
            status = CircuitBreakerStatusModel(
                name=name,
                state=cb.state,
                failure_count=cb.failure_count,
                success_count=cb.success_count,
                last_failure_time=cb.last_failure_time,
                next_attempt_time=cb.next_attempt_time
            )
            circuit_breakers.append(status)
        
        return circuit_breakers
        
    except Exception as e:
        logger.error(f"Failed to list circuit breakers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list circuit breakers: {str(e)}")

@router.get("/circuit-breakers/{name}", response_model=CircuitBreakerStatusModel)
async def get_circuit_breaker_status(name: str):
    """Get status of specific circuit breaker"""
    try:
        cb = fault_tolerance_manager.get_circuit_breaker(name)
        if not cb:
            raise HTTPException(status_code=404, detail=f"Circuit breaker '{name}' not found")
        
        return CircuitBreakerStatusModel(
            name=name,
            state=cb.state,
            failure_count=cb.failure_count,
            success_count=cb.success_count,
            last_failure_time=cb.last_failure_time,
            next_attempt_time=cb.next_attempt_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get circuit breaker status for {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get circuit breaker status: {str(e)}")

@router.post("/circuit-breakers/{name}/reset", response_model=Dict[str, str])
async def reset_circuit_breaker(name: str):
    """Reset circuit breaker to closed state"""
    try:
        cb = fault_tolerance_manager.get_circuit_breaker(name)
        if not cb:
            raise HTTPException(status_code=404, detail=f"Circuit breaker '{name}' not found")
        
        cb.state = CircuitBreakerState.CLOSED
        cb.failure_count = 0
        cb.success_count = 0
        cb.last_failure_time = None
        cb.next_attempt_time = None
        
        logger.info(f"Reset circuit breaker: {name}")
        return {
            "message": f"Circuit breaker '{name}' reset successfully",
            "name": name,
            "status": "reset"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset circuit breaker {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset circuit breaker: {str(e)}")

# Graceful Degradation Endpoints

@router.post("/degradation", response_model=Dict[str, str])
async def degrade_service(degradation: ServiceDegradationModel):
    """Degrade service to specified level"""
    try:
        fault_tolerance_manager.degradation_manager.degrade_service(
            degradation.service_name,
            degradation.degradation_level
        )
        
        logger.info(f"Degraded service {degradation.service_name} to level {degradation.degradation_level.value}")
        return {
            "message": f"Service '{degradation.service_name}' degraded to {degradation.degradation_level.value}",
            "service": degradation.service_name,
            "level": degradation.degradation_level.value
        }
        
    except Exception as e:
        logger.error(f"Failed to degrade service {degradation.service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to degrade service: {str(e)}")

@router.get("/degradation", response_model=Dict[str, str])
async def get_degraded_services():
    """Get list of currently degraded services"""
    try:
        degraded_services = {}
        
        for service, level in fault_tolerance_manager.degradation_manager.degradation_levels.items():
            if level != DegradationLevel.NONE:
                degraded_services[service] = level.value
        
        return degraded_services
        
    except Exception as e:
        logger.error(f"Failed to get degraded services: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get degraded services: {str(e)}")

@router.post("/degradation/{service_name}/restore", response_model=Dict[str, str])
async def restore_service(service_name: str):
    """Restore service to normal operation"""
    try:
        fault_tolerance_manager.degradation_manager.degrade_service(
            service_name,
            DegradationLevel.NONE
        )
        
        logger.info(f"Restored service {service_name} to normal operation")
        return {
            "message": f"Service '{service_name}' restored to normal operation",
            "service": service_name,
            "status": "restored"
        }
        
    except Exception as e:
        logger.error(f"Failed to restore service {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restore service: {str(e)}")

# Recovery Management Endpoints

@router.post("/recovery/initiate", response_model=RecoveryResultModel)
async def initiate_recovery(failure: SystemFailureModel, background_tasks: BackgroundTasks):
    """Initiate automated recovery for system failure"""
    try:
        system_failure = SystemFailure(
            failure_id=failure.failure_id,
            failure_type=failure.failure_type,
            component=failure.component,
            message=failure.message,
            timestamp=datetime.now(),
            severity=failure.severity,
            context=failure.context,
            stack_trace=failure.stack_trace
        )
        
        # Initiate recovery in background
        recovery_result = await fault_tolerance_manager.handle_system_failure(system_failure)
        
        return RecoveryResultModel(
            recovery_id=recovery_result.recovery_id,
            success=recovery_result.success,
            strategy=recovery_result.strategy,
            duration=recovery_result.duration,
            actions_taken=recovery_result.actions_taken,
            health_status=recovery_result.health_status,
            timestamp=recovery_result.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to initiate recovery for failure {failure.failure_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate recovery: {str(e)}")

@router.get("/recovery/history", response_model=List[RecoveryResultModel])
async def get_recovery_history(limit: int = 50):
    """Get recovery operation history"""
    try:
        history = fault_tolerance_manager.recovery_manager.recovery_history[-limit:]
        
        return [
            RecoveryResultModel(
                recovery_id=recovery.recovery_id,
                success=recovery.success,
                strategy=recovery.strategy,
                duration=recovery.duration,
                actions_taken=recovery.actions_taken,
                health_status=recovery.health_status,
                timestamp=recovery.timestamp.isoformat()
            )
            for recovery in reversed(history)
        ]
        
    except Exception as e:
        logger.error(f"Failed to get recovery history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recovery history: {str(e)}")

@router.get("/recovery/{recovery_id}", response_model=RecoveryResultModel)
async def get_recovery_details(recovery_id: str):
    """Get details of specific recovery operation"""
    try:
        recovery = next(
            (r for r in fault_tolerance_manager.recovery_manager.recovery_history 
             if r.recovery_id == recovery_id),
            None
        )
        
        if not recovery:
            raise HTTPException(status_code=404, detail=f"Recovery operation '{recovery_id}' not found")
        
        return RecoveryResultModel(
            recovery_id=recovery.recovery_id,
            success=recovery.success,
            strategy=recovery.strategy,
            duration=recovery.duration,
            actions_taken=recovery.actions_taken,
            health_status=recovery.health_status,
            timestamp=recovery.timestamp.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recovery details for {recovery_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recovery details: {str(e)}")

# System Health Monitoring Endpoints

@router.get("/health", response_model=SystemHealthModel)
async def get_system_health():
    """Get overall system health status"""
    try:
        health = fault_tolerance_manager.get_system_health()
        
        return SystemHealthModel(
            timestamp=health['timestamp'],
            overall_status=health['overall_status'],
            circuit_breakers=health['circuit_breakers'],
            degraded_services=health['degraded_services'],
            recent_recoveries=health['recent_recoveries']
        )
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")

@router.get("/health/summary", response_model=Dict[str, Any])
async def get_health_summary():
    """Get summarized health metrics"""
    try:
        health = fault_tolerance_manager.get_system_health()
        
        # Calculate summary metrics
        total_circuit_breakers = len(health['circuit_breakers'])
        open_circuit_breakers = sum(
            1 for cb in health['circuit_breakers'].values() 
            if cb['state'] != 'closed'
        )
        
        degraded_services_count = len(health['degraded_services'])
        
        recent_recoveries = health['recent_recoveries'][:10]
        successful_recoveries = sum(1 for r in recent_recoveries if r['success'])
        
        summary = {
            'overall_status': health['overall_status'],
            'timestamp': health['timestamp'],
            'metrics': {
                'total_circuit_breakers': total_circuit_breakers,
                'open_circuit_breakers': open_circuit_breakers,
                'degraded_services': degraded_services_count,
                'recent_recovery_success_rate': (
                    successful_recoveries / len(recent_recoveries) * 100 
                    if recent_recoveries else 100
                )
            },
            'alerts': []
        }
        
        # Add alerts for issues
        if open_circuit_breakers > 0:
            summary['alerts'].append(f"{open_circuit_breakers} circuit breaker(s) are open")
        
        if degraded_services_count > 0:
            summary['alerts'].append(f"{degraded_services_count} service(s) are degraded")
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get health summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get health summary: {str(e)}")

# Configuration Endpoints

@router.get("/config/defaults", response_model=Dict[str, Any])
async def get_default_configs():
    """Get default fault tolerance configurations"""
    try:
        from scrollintel.core.fault_tolerance import DEFAULT_CIRCUIT_BREAKER_CONFIG, DEFAULT_RETRY_CONFIG
        
        return {
            'circuit_breaker': {
                'failure_threshold': DEFAULT_CIRCUIT_BREAKER_CONFIG.failure_threshold,
                'recovery_timeout': DEFAULT_CIRCUIT_BREAKER_CONFIG.recovery_timeout,
                'success_threshold': DEFAULT_CIRCUIT_BREAKER_CONFIG.success_threshold,
                'timeout': DEFAULT_CIRCUIT_BREAKER_CONFIG.timeout
            },
            'retry': {
                'max_attempts': DEFAULT_RETRY_CONFIG.max_attempts,
                'base_delay': DEFAULT_RETRY_CONFIG.base_delay,
                'max_delay': DEFAULT_RETRY_CONFIG.max_delay,
                'exponential_base': DEFAULT_RETRY_CONFIG.exponential_base,
                'jitter': DEFAULT_RETRY_CONFIG.jitter
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get default configs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get default configs: {str(e)}")

# Test Endpoints for Development

@router.post("/test/failure", response_model=Dict[str, str])
async def simulate_failure(failure_type: FailureType, component: str = "test_component"):
    """Simulate system failure for testing (development only)"""
    try:
        import uuid
        
        failure = SystemFailure(
            failure_id=str(uuid.uuid4()),
            failure_type=failure_type,
            component=component,
            message=f"Simulated {failure_type.value} failure",
            timestamp=datetime.now(),
            severity="medium",
            context={"simulated": True}
        )
        
        recovery_result = await fault_tolerance_manager.handle_system_failure(failure)
        
        return {
            "message": f"Simulated {failure_type.value} failure for {component}",
            "failure_id": failure.failure_id,
            "recovery_id": recovery_result.recovery_id,
            "recovery_success": recovery_result.success
        }
        
    except Exception as e:
        logger.error(f"Failed to simulate failure: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to simulate failure: {str(e)}")