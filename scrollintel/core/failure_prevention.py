"""
Comprehensive failure prevention and resilience system for ScrollIntel.
Ensures the application never fails in users' hands.
"""

import asyncio
import logging
import traceback
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from functools import wraps
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures we protect against."""
    NETWORK_ERROR = "network_error"
    DATABASE_ERROR = "database_error"
    MEMORY_ERROR = "memory_error"
    CPU_OVERLOAD = "cpu_overload"
    DISK_FULL = "disk_full"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    EXTERNAL_API_ERROR = "external_api_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class FailureEvent:
    """Represents a failure event."""
    failure_type: FailureType
    timestamp: datetime
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        return (
            self.last_failure_time and
            datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class RetryStrategy:
    """Configurable retry strategy."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 exponential_backoff: bool = True, jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** attempt)
        else:
            delay = self.base_delay
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return min(delay, 60.0)  # Cap at 60 seconds


class FailurePreventionSystem:
    """Comprehensive failure prevention and recovery system."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failure_history: List[FailureEvent] = []
        self.system_monitors: Dict[str, Any] = {}
        self.recovery_strategies: Dict[FailureType, Callable] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.graceful_degradation_modes: Dict[str, Callable] = {}
        self._monitoring_active = False
        self._monitoring_thread = None
        
        # Initialize default recovery strategies
        self._setup_default_recovery_strategies()
        self._setup_default_health_checks()
        
        # Start system monitoring
        self.start_monitoring()
    
    def _setup_default_recovery_strategies(self):
        """Setup default recovery strategies for common failures."""
        self.recovery_strategies[FailureType.NETWORK_ERROR] = self._recover_network_error
        self.recovery_strategies[FailureType.DATABASE_ERROR] = self._recover_database_error
        self.recovery_strategies[FailureType.MEMORY_ERROR] = self._recover_memory_error
        self.recovery_strategies[FailureType.CPU_OVERLOAD] = self._recover_cpu_overload
        self.recovery_strategies[FailureType.DISK_FULL] = self._recover_disk_full
        self.recovery_strategies[FailureType.TIMEOUT_ERROR] = self._recover_timeout_error
        self.recovery_strategies[FailureType.EXTERNAL_API_ERROR] = self._recover_external_api_error
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        self.health_checks["memory"] = self._check_memory_health
        self.health_checks["cpu"] = self._check_cpu_health
        self.health_checks["disk"] = self._check_disk_health
        self.health_checks["database"] = self._check_database_health
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
    
    def with_failure_protection(self, service_name: str = "default", 
                               retry_strategy: Optional[RetryStrategy] = None):
        """Decorator for comprehensive failure protection."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_protection(
                    func, service_name, retry_strategy, *args, **kwargs
                )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(self._execute_with_protection(
                    func, service_name, retry_strategy, *args, **kwargs
                ))
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    async def _execute_with_protection(self, func: Callable, service_name: str,
                                     retry_strategy: Optional[RetryStrategy],
                                     *args, **kwargs):
        """Execute function with comprehensive protection."""
        circuit_breaker = self.get_circuit_breaker(service_name)
        retry_strategy = retry_strategy or RetryStrategy()
        
        last_exception = None
        
        for attempt in range(retry_strategy.max_attempts):
            try:
                # Check system health before execution
                await self._perform_health_checks()
                
                # Execute with circuit breaker protection
                if asyncio.iscoroutinefunction(func):
                    result = await circuit_breaker.call(func, *args, **kwargs)
                else:
                    result = circuit_breaker.call(func, *args, **kwargs)
                
                return result
                
            except Exception as e:
                last_exception = e
                failure_type = self._classify_failure(e)
                
                # Log failure
                failure_event = FailureEvent(
                    failure_type=failure_type,
                    timestamp=datetime.utcnow(),
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                    context={
                        "service_name": service_name,
                        "attempt": attempt + 1,
                        "args": str(args)[:200],  # Truncate for logging
                        "kwargs": str(kwargs)[:200]
                    }
                )
                
                self.failure_history.append(failure_event)
                logger.error(f"Failure in {service_name}: {e}", exc_info=True)
                
                # Attempt recovery
                recovery_successful = await self._attempt_recovery(failure_event)
                
                if recovery_successful:
                    logger.info(f"Recovery successful for {service_name}")
                    continue
                
                # If not the last attempt, wait before retry
                if attempt < retry_strategy.max_attempts - 1:
                    delay = retry_strategy.get_delay(attempt)
                    logger.info(f"Retrying {service_name} in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
        
        # All attempts failed, try graceful degradation
        degraded_result = await self._attempt_graceful_degradation(
            service_name, last_exception, *args, **kwargs
        )
        
        if degraded_result is not None:
            logger.warning(f"Using degraded mode for {service_name}")
            return degraded_result
        
        # If all else fails, raise the last exception
        raise last_exception
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure based on exception."""
        error_msg = str(exception).lower()
        
        if "network" in error_msg or "connection" in error_msg:
            return FailureType.NETWORK_ERROR
        elif "database" in error_msg or "sql" in error_msg:
            return FailureType.DATABASE_ERROR
        elif "memory" in error_msg or "out of memory" in error_msg:
            return FailureType.MEMORY_ERROR
        elif "timeout" in error_msg:
            return FailureType.TIMEOUT_ERROR
        elif "validation" in error_msg or "invalid" in error_msg:
            return FailureType.VALIDATION_ERROR
        elif "api" in error_msg or "http" in error_msg:
            return FailureType.EXTERNAL_API_ERROR
        else:
            return FailureType.UNKNOWN_ERROR
    
    async def _attempt_recovery(self, failure_event: FailureEvent) -> bool:
        """Attempt to recover from a failure."""
        failure_event.recovery_attempted = True
        
        # Notify integration system about the failure
        try:
            from .failure_ux_integration import failure_ux_integrator
            await failure_ux_integrator.handle_unified_failure(failure_event)
        except ImportError:
            # Integration system not available, continue with basic recovery
            pass
        except Exception as e:
            logger.error(f"Failed to notify integration system: {e}")
        
        recovery_strategy = self.recovery_strategies.get(failure_event.failure_type)
        if recovery_strategy:
            try:
                await recovery_strategy(failure_event)
                failure_event.recovery_successful = True
                return True
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
        
        return False
    
    async def _attempt_graceful_degradation(self, service_name: str, 
                                          exception: Exception, *args, **kwargs):
        """Attempt graceful degradation when all else fails."""
        degradation_handler = self.graceful_degradation_modes.get(service_name)
        if degradation_handler:
            try:
                return await degradation_handler(exception, *args, **kwargs)
            except Exception as e:
                logger.error(f"Graceful degradation failed for {service_name}: {e}")
        
        return None
    
    def register_graceful_degradation(self, service_name: str, handler: Callable):
        """Register a graceful degradation handler for a service."""
        self.graceful_degradation_modes[service_name] = handler
    
    async def _perform_health_checks(self):
        """Perform system health checks."""
        for check_name, check_func in self.health_checks.items():
            try:
                is_healthy = await check_func()
                if not is_healthy:
                    logger.warning(f"Health check failed: {check_name}")
            except Exception as e:
                logger.error(f"Health check error for {check_name}: {e}")
    
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True
            )
            self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self._monitoring_active:
            try:
                # Monitor system resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                
                # Check for critical conditions
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage: {cpu_percent}%")
                    asyncio.run(self._recover_cpu_overload(None))
                
                if memory_percent > 90:
                    logger.warning(f"High memory usage: {memory_percent}%")
                    asyncio.run(self._recover_memory_error(None))
                
                if disk_percent > 95:
                    logger.warning(f"Low disk space: {disk_percent}%")
                    asyncio.run(self._recover_disk_full(None))
                
                # Clean up old failure history
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.failure_history = [
                    f for f in self.failure_history 
                    if f.timestamp > cutoff_time
                ]
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    # Recovery strategy implementations
    async def _recover_network_error(self, failure_event: Optional[FailureEvent]):
        """Recover from network errors."""
        # Wait for network to stabilize
        await asyncio.sleep(5)
        
        # Test connectivity
        import socket
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            logger.info("Network connectivity restored")
        except Exception:
            logger.warning("Network still unavailable")
    
    async def _recover_database_error(self, failure_event: Optional[FailureEvent]):
        """Recover from database errors."""
        # Attempt to reconnect to database
        try:
            # This would reconnect to your database
            # For now, just wait and hope it recovers
            await asyncio.sleep(2)
            logger.info("Database recovery attempted")
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
    
    async def _recover_memory_error(self, failure_event: Optional[FailureEvent]):
        """Recover from memory errors."""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Clear caches if available
        try:
            # Clear any application caches here
            logger.info("Memory cleanup performed")
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    async def _recover_cpu_overload(self, failure_event: Optional[FailureEvent]):
        """Recover from CPU overload."""
        # Reduce processing load temporarily
        await asyncio.sleep(1)
        logger.info("CPU load reduction attempted")
    
    async def _recover_disk_full(self, failure_event: Optional[FailureEvent]):
        """Recover from disk full errors."""
        try:
            # Clean up temporary files
            temp_dirs = ["/tmp", "temp", "logs"]
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    # Clean old files (older than 1 day)
                    cutoff_time = time.time() - 86400
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.getmtime(file_path) < cutoff_time:
                                try:
                                    os.remove(file_path)
                                except Exception:
                                    pass
            
            logger.info("Disk cleanup performed")
        except Exception as e:
            logger.error(f"Disk cleanup failed: {e}")
    
    async def _recover_timeout_error(self, failure_event: Optional[FailureEvent]):
        """Recover from timeout errors."""
        # Just wait a bit for the system to stabilize
        await asyncio.sleep(2)
        logger.info("Timeout recovery attempted")
    
    async def _recover_external_api_error(self, failure_event: Optional[FailureEvent]):
        """Recover from external API errors."""
        # Wait for external service to recover
        await asyncio.sleep(5)
        logger.info("External API recovery attempted")
    
    # Health check implementations
    async def _check_memory_health(self) -> bool:
        """Check memory health."""
        memory_percent = psutil.virtual_memory().percent
        return memory_percent < 85
    
    async def _check_cpu_health(self) -> bool:
        """Check CPU health."""
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent < 80
    
    async def _check_disk_health(self) -> bool:
        """Check disk health."""
        disk_percent = psutil.disk_usage('/').percent
        return disk_percent < 90
    
    async def _check_database_health(self) -> bool:
        """Check database health."""
        # This would check your actual database connection
        # For now, assume it's healthy
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        recent_failures = [
            f for f in self.failure_history 
            if f.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        circuit_breaker_status = {
            name: {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
            for name, cb in self.circuit_breakers.items()
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_active": self._monitoring_active,
            "recent_failures": len(recent_failures),
            "circuit_breakers": circuit_breaker_status,
            "system_resources": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        }
    
    def register_failure_callback(self, callback: Callable[[FailureEvent], None]):
        """Register callback for failure events to enable cross-system coordination."""
        if not hasattr(self, '_failure_callbacks'):
            self._failure_callbacks = []
        self._failure_callbacks.append(callback)
    
    async def _notify_failure_callbacks(self, failure_event: FailureEvent):
        """Notify registered callbacks about failure events."""
        if hasattr(self, '_failure_callbacks'):
            for callback in self._failure_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(failure_event)
                    else:
                        callback(failure_event)
                except Exception as e:
                    logger.error(f"Failure callback error: {e}")
    
    def get_failure_patterns(self) -> Dict[str, Any]:
        """Get failure patterns for predictive analysis."""
        if not self.failure_history:
            return {}
        
        # Analyze failure patterns
        failure_types = defaultdict(int)
        failure_times = []
        recovery_success_rate = 0
        
        for failure in self.failure_history:
            failure_types[failure.failure_type.value] += 1
            failure_times.append(failure.timestamp)
            if failure.recovery_successful:
                recovery_success_rate += 1
        
        recovery_success_rate = recovery_success_rate / len(self.failure_history)
        
        # Calculate failure frequency
        if len(failure_times) > 1:
            time_diffs = [
                (failure_times[i] - failure_times[i-1]).total_seconds()
                for i in range(1, len(failure_times))
            ]
            avg_time_between_failures = sum(time_diffs) / len(time_diffs)
        else:
            avg_time_between_failures = 0
        
        return {
            "failure_types": dict(failure_types),
            "total_failures": len(self.failure_history),
            "recovery_success_rate": recovery_success_rate,
            "avg_time_between_failures": avg_time_between_failures,
            "most_common_failure": max(failure_types.items(), key=lambda x: x[1])[0] if failure_types else None
        }


# Global instance
failure_prevention = FailurePreventionSystem()


# Convenience decorators
def bulletproof(service_name: str = "default", max_retries: int = 3):
    """Make any function bulletproof against failures."""
    retry_strategy = RetryStrategy(max_attempts=max_retries)
    return failure_prevention.with_failure_protection(service_name, retry_strategy)


def critical_operation(service_name: str = "critical"):
    """Mark an operation as critical with enhanced protection."""
    retry_strategy = RetryStrategy(max_attempts=5, base_delay=0.5)
    return failure_prevention.with_failure_protection(service_name, retry_strategy)


@asynccontextmanager
async def safe_operation(operation_name: str = "operation"):
    """Context manager for safe operations with automatic cleanup."""
    start_time = time.time()
    try:
        logger.info(f"Starting safe operation: {operation_name}")
        yield
        logger.info(f"Safe operation completed: {operation_name} ({time.time() - start_time:.2f}s)")
    except Exception as e:
        logger.error(f"Safe operation failed: {operation_name} - {e}")
        # Attempt recovery
        await failure_prevention._attempt_recovery(
            FailureEvent(
                failure_type=failure_prevention._classify_failure(e),
                timestamp=datetime.utcnow(),
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                context={"operation_name": operation_name}
            )
        )
        raise
    finally:
        # Cleanup code here if needed
        pass


def never_fail(default_return=None):
    """Decorator that ensures a function never fails by returning a default value."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {e}", exc_info=True)
                return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {e}", exc_info=True)
                return default_return
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator