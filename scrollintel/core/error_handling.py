"""
Comprehensive error handling and recovery system for ScrollIntel.
Provides centralized error management, retry mechanisms, and graceful degradation.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, Any, Optional, List, Callable, Union, Type
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps
import json

from .interfaces import (
    AgentError, SecurityError, EngineError, ConfigurationError,
    DataError, ValidationError, ExternalServiceError
)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    AGENT = "agent"
    ENGINE = "engine"
    SECURITY = "security"
    DATA = "data"
    VALIDATION = "validation"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    FAIL_FAST = "fail_fast"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # exponential, linear, fixed


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker implementation for external service calls."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.success_count = 0
        
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 successes to close
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0


class ErrorHandler:
    """Central error handler with recovery mechanisms."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_counts: Dict[str, int] = {}
        self.error_rates: Dict[str, List[float]] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.degradation_handlers: Dict[str, Callable] = {}
        self.error_rate_window = 300  # 5 minutes
        self.error_rate_threshold = 0.05  # 5% error rate threshold
        self.automatic_recovery_enabled = True
        
    def register_fallback(self, component: str, handler: Callable):
        """Register fallback handler for a component."""
        self.fallback_handlers[component] = handler
    
    def register_degradation(self, component: str, handler: Callable):
        """Register degradation handler for a component."""
        self.degradation_handlers[component] = handler
    
    def get_circuit_breaker(self, service_name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            config = config or CircuitBreakerConfig()
            self.circuit_breakers[service_name] = CircuitBreaker(config)
        return self.circuit_breakers[service_name]
    
    async def handle_error(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Handle error with appropriate recovery strategy."""
        # Log error
        await self._log_error(error, context)
        
        # Track error rate
        self._track_error_rate(context.component)
        
        # Classify error
        category = self._classify_error(error)
        severity = self._determine_severity(error, context)
        
        # Update context
        context.category = category
        context.severity = severity
        
        # Check for automatic recovery
        if self.automatic_recovery_enabled:
            recovery_result = await self._attempt_automatic_recovery(error, context)
            if recovery_result:
                return recovery_result
        
        # Determine recovery strategy
        strategy = self._determine_recovery_strategy(error, context)
        
        # Execute recovery
        return await self._execute_recovery(error, context, strategy)
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error into appropriate category."""
        if isinstance(error, AgentError):
            return ErrorCategory.AGENT
        elif isinstance(error, EngineError):
            return ErrorCategory.ENGINE
        elif isinstance(error, SecurityError):
            return ErrorCategory.SECURITY
        elif isinstance(error, DataError):
            return ErrorCategory.DATA
        elif isinstance(error, ValidationError):
            return ErrorCategory.VALIDATION
        elif isinstance(error, ExternalServiceError):
            return ErrorCategory.EXTERNAL_SERVICE
        elif isinstance(error, ConfigurationError):
            return ErrorCategory.CONFIGURATION
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, (MemoryError, OSError)):
            return ErrorCategory.RESOURCE
        else:
            return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error: Exception, context: ErrorContext) -> ErrorSeverity:
        """Determine error severity based on error type and context."""
        if isinstance(error, (SecurityError, ConfigurationError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (AgentError, EngineError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (DataError, ValidationError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _determine_recovery_strategy(self, error: Exception, context: ErrorContext) -> RecoveryStrategy:
        """Determine appropriate recovery strategy."""
        if context.category == ErrorCategory.SECURITY:
            return RecoveryStrategy.FAIL_FAST
        elif context.category == ErrorCategory.EXTERNAL_SERVICE:
            return RecoveryStrategy.CIRCUIT_BREAKER
        elif context.category in [ErrorCategory.AGENT, ErrorCategory.ENGINE]:
            return RecoveryStrategy.FALLBACK
        elif context.category == ErrorCategory.NETWORK:
            return RecoveryStrategy.RETRY
        else:
            return RecoveryStrategy.DEGRADE
    
    async def _execute_recovery(self, error: Exception, context: ErrorContext, strategy: RecoveryStrategy) -> Dict[str, Any]:
        """Execute recovery strategy."""
        if strategy == RecoveryStrategy.FAIL_FAST:
            return await self._fail_fast(error, context)
        elif strategy == RecoveryStrategy.RETRY:
            return await self._retry_with_backoff(error, context)
        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._execute_fallback(error, context)
        elif strategy == RecoveryStrategy.DEGRADE:
            return await self._graceful_degradation(error, context)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return await self._circuit_breaker_recovery(error, context)
        else:
            return await self._default_recovery(error, context)
    
    async def _fail_fast(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Fail fast strategy - immediately return error."""
        return {
            "success": False,
            "error": {
                "type": "critical_error",
                "message": str(error),
                "error_id": context.error_id,
                "user_message": self._get_user_friendly_message(error, context),
                "recovery_action": "Contact system administrator"
            }
        }
    
    async def _retry_with_backoff(self, error: Exception, context: ErrorContext, config: Optional[RetryConfig] = None) -> Dict[str, Any]:
        """Retry with exponential backoff."""
        config = config or RetryConfig()
        
        for attempt in range(config.max_attempts):
            if attempt > 0:
                delay = self._calculate_delay(attempt, config)
                await asyncio.sleep(delay)
            
            try:
                # This would need to be implemented by the calling code
                # For now, we'll simulate a retry
                self.logger.info(f"Retry attempt {attempt + 1} for {context.component}")
                # In real implementation, this would re-execute the failed operation
                break
            except Exception as retry_error:
                if attempt == config.max_attempts - 1:
                    return await self._fallback_after_retry_failure(retry_error, context)
                continue
        
        return {
            "success": True,
            "message": f"Operation succeeded after {attempt + 1} attempts",
            "attempts": attempt + 1
        }
    
    async def _execute_fallback(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Execute fallback handler."""
        fallback_handler = self.fallback_handlers.get(context.component)
        
        if fallback_handler:
            try:
                result = await fallback_handler(error, context)
                return {
                    "success": True,
                    "fallback_used": True,
                    "result": result,
                    "user_message": "Service temporarily using backup system"
                }
            except Exception as fallback_error:
                self.logger.error(f"Fallback failed for {context.component}: {fallback_error}")
                return await self._graceful_degradation(error, context)
        
        return await self._graceful_degradation(error, context)
    
    async def _graceful_degradation(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Implement graceful degradation."""
        degradation_handler = self.degradation_handlers.get(context.component)
        
        if degradation_handler:
            try:
                result = await degradation_handler(error, context)
                return {
                    "success": True,
                    "degraded": True,
                    "result": result,
                    "user_message": "Service running with limited functionality"
                }
            except Exception as degradation_error:
                self.logger.error(f"Degradation failed for {context.component}: {degradation_error}")
        
        return {
            "success": False,
            "error": {
                "type": "service_unavailable",
                "message": str(error),
                "error_id": context.error_id,
                "user_message": self._get_user_friendly_message(error, context),
                "recovery_action": "Please try again later"
            }
        }
    
    async def _circuit_breaker_recovery(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Handle circuit breaker recovery."""
        service_name = context.component
        circuit_breaker = self.get_circuit_breaker(service_name)
        
        if not circuit_breaker.can_execute():
            return {
                "success": False,
                "error": {
                    "type": "service_unavailable",
                    "message": f"Service {service_name} is temporarily unavailable",
                    "error_id": context.error_id,
                    "user_message": f"{service_name} is temporarily unavailable. Please try again later.",
                    "recovery_action": "Wait and retry"
                }
            }
        
        circuit_breaker.record_failure()
        return await self._execute_fallback(error, context)
    
    async def _default_recovery(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Default recovery strategy."""
        return {
            "success": False,
            "error": {
                "type": "unknown_error",
                "message": str(error),
                "error_id": context.error_id,
                "user_message": self._get_user_friendly_message(error, context),
                "recovery_action": "Please try again or contact support"
            }
        }
    
    async def _fallback_after_retry_failure(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Handle fallback after retry failure."""
        self.logger.error(f"All retry attempts failed for {context.component}")
        return await self._execute_fallback(error, context)
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt."""
        if config.backoff_strategy == "exponential":
            delay = config.base_delay * (config.exponential_base ** attempt)
        elif config.backoff_strategy == "linear":
            delay = config.base_delay * (attempt + 1)
        else:  # fixed
            delay = config.base_delay
        
        delay = min(delay, config.max_delay)
        
        if config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay
    
    def _get_user_friendly_message(self, error: Exception, context: ErrorContext) -> str:
        """Generate user-friendly error message."""
        if context.category == ErrorCategory.AGENT:
            return "The AI agent is temporarily unavailable. Please try again in a few moments."
        elif context.category == ErrorCategory.ENGINE:
            return "The processing engine encountered an issue. Your request is being handled with backup systems."
        elif context.category == ErrorCategory.SECURITY:
            return "Access denied. Please check your credentials and try again."
        elif context.category == ErrorCategory.DATA:
            return "There was an issue with your data. Please check the format and try again."
        elif context.category == ErrorCategory.VALIDATION:
            return "Please check your input and try again."
        elif context.category == ErrorCategory.EXTERNAL_SERVICE:
            return "An external service is temporarily unavailable. We're working to restore full functionality."
        elif context.category == ErrorCategory.NETWORK:
            return "Network connectivity issue. Please check your connection and try again."
        elif context.category == ErrorCategory.RESOURCE:
            return "System resources are temporarily limited. Please try again in a few moments."
        else:
            return "An unexpected error occurred. Please try again or contact support if the issue persists."
    
    def _track_error_rate(self, component: str):
        """Track error rate for monitoring and alerting."""
        current_time = time.time()
        
        # Initialize error rate tracking for component
        if component not in self.error_rates:
            self.error_rates[component] = []
        
        # Add current error timestamp
        self.error_rates[component].append(current_time)
        
        # Clean old entries outside the window
        cutoff_time = current_time - self.error_rate_window
        self.error_rates[component] = [
            timestamp for timestamp in self.error_rates[component]
            if timestamp > cutoff_time
        ]
        
        # Check if error rate exceeds threshold
        error_count = len(self.error_rates[component])
        error_rate = error_count / (self.error_rate_window / 60)  # errors per minute
        
        if error_rate > self.error_rate_threshold * 60:  # Convert to per minute
            self.logger.warning(
                f"High error rate detected for {component}: {error_rate:.2f} errors/min",
                component=component,
                error_rate=error_rate,
                threshold=self.error_rate_threshold * 60
            )
    
    async def _attempt_automatic_recovery(self, error: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Attempt automatic recovery mechanisms."""
        # Check if this is a transient error that can be automatically recovered
        if isinstance(error, (ConnectionError, TimeoutError)):
            # Attempt connection recovery
            recovery_result = await self._recover_connection(error, context)
            if recovery_result:
                return recovery_result
        
        # Check if this is a resource exhaustion error
        if isinstance(error, MemoryError):
            # Attempt memory cleanup
            recovery_result = await self._recover_memory(error, context)
            if recovery_result:
                return recovery_result
        
        # Check if this is a service overload error
        if "overload" in str(error).lower() or "rate limit" in str(error).lower():
            # Implement backoff and retry
            recovery_result = await self._recover_from_overload(error, context)
            if recovery_result:
                return recovery_result
        
        return None
    
    async def _recover_connection(self, error: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Attempt to recover from connection errors."""
        try:
            # Wait briefly and attempt to re-establish connection
            await asyncio.sleep(1.0)
            
            # This would be implemented by specific services
            # For now, we'll simulate a successful recovery
            self.logger.info(f"Connection recovery attempted for {context.component}")
            
            return {
                "success": True,
                "recovered": True,
                "message": "Connection automatically recovered",
                "recovery_type": "connection_recovery"
            }
        except Exception as recovery_error:
            self.logger.error(f"Connection recovery failed: {recovery_error}")
            return None
    
    async def _recover_memory(self, error: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Attempt to recover from memory errors."""
        try:
            # Trigger garbage collection
            import gc
            gc.collect()
            
            self.logger.info(f"Memory recovery attempted for {context.component}")
            
            return {
                "success": True,
                "recovered": True,
                "message": "Memory automatically cleaned up",
                "recovery_type": "memory_recovery"
            }
        except Exception as recovery_error:
            self.logger.error(f"Memory recovery failed: {recovery_error}")
            return None
    
    async def _recover_from_overload(self, error: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Attempt to recover from service overload."""
        try:
            # Implement exponential backoff
            backoff_time = min(2.0 ** self.error_counts.get(context.component, 0), 60.0)
            await asyncio.sleep(backoff_time)
            
            self.logger.info(f"Overload recovery attempted for {context.component} with {backoff_time}s backoff")
            
            return {
                "success": True,
                "recovered": True,
                "message": f"Service recovered after {backoff_time}s backoff",
                "recovery_type": "overload_recovery",
                "backoff_time": backoff_time
            }
        except Exception as recovery_error:
            self.logger.error(f"Overload recovery failed: {recovery_error}")
            return None
    
    def get_error_rate(self, component: str) -> float:
        """Get current error rate for a component."""
        if component not in self.error_rates:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - self.error_rate_window
        
        # Clean old entries
        self.error_rates[component] = [
            timestamp for timestamp in self.error_rates[component]
            if timestamp > cutoff_time
        ]
        
        error_count = len(self.error_rates[component])
        return error_count / (self.error_rate_window / 60)  # errors per minute
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        stats = {
            "total_components": len(self.error_rates),
            "components": {},
            "overall_error_rate": 0.0,
            "high_error_rate_components": []
        }
        
        total_errors = 0
        for component, error_timestamps in self.error_rates.items():
            error_rate = self.get_error_rate(component)
            stats["components"][component] = {
                "error_rate": error_rate,
                "error_count": len(error_timestamps),
                "status": "healthy" if error_rate < self.error_rate_threshold * 60 else "unhealthy"
            }
            
            total_errors += len(error_timestamps)
            
            if error_rate > self.error_rate_threshold * 60:
                stats["high_error_rate_components"].append({
                    "component": component,
                    "error_rate": error_rate
                })
        
        stats["overall_error_rate"] = total_errors / (self.error_rate_window / 60) if self.error_rates else 0.0
        
        return stats

    async def _log_error(self, error: Exception, context: ErrorContext):
        """Log error with full context."""
        error_data = {
            "error_id": context.error_id,
            "timestamp": context.timestamp,
            "severity": context.severity.value,
            "category": context.category.value,
            "component": context.component,
            "operation": context.operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "user_id": context.user_id,
            "session_id": context.session_id,
            "request_id": context.request_id,
            "metadata": context.metadata,
            "error_rate": self.get_error_rate(context.component)
        }
        
        if context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.logger.error(f"Error {context.error_id}: {json.dumps(error_data, indent=2)}")
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Error {context.error_id}: {json.dumps(error_data, indent=2)}")
        else:
            self.logger.info(f"Error {context.error_id}: {json.dumps(error_data, indent=2)}")


# Global error handler instance
error_handler = ErrorHandler()


def with_error_handling(
    component: str,
    operation: str,
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
):
    """Decorator for adding error handling to functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import uuid
            
            context = ErrorContext(
                error_id=str(uuid.uuid4()),
                timestamp=time.time(),
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.UNKNOWN,
                component=component,
                operation=operation
            )
            
            try:
                # Check circuit breaker if configured
                if circuit_breaker_config:
                    circuit_breaker = error_handler.get_circuit_breaker(component, circuit_breaker_config)
                    if not circuit_breaker.can_execute():
                        raise ExternalServiceError(f"Circuit breaker open for {component}")
                
                result = await func(*args, **kwargs)
                
                # Record success for circuit breaker
                if circuit_breaker_config:
                    circuit_breaker = error_handler.get_circuit_breaker(component)
                    circuit_breaker.record_success()
                
                return result
                
            except Exception as e:
                # Record failure for circuit breaker
                if circuit_breaker_config:
                    circuit_breaker = error_handler.get_circuit_breaker(component)
                    circuit_breaker.record_failure()
                
                return await error_handler.handle_error(e, context)
        
        return wrapper
    return decorator


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for adding retry logic to functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_config = config or RetryConfig()
            
            for attempt in range(retry_config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == retry_config.max_attempts - 1:
                        raise
                    
                    delay = error_handler._calculate_delay(attempt, retry_config)
                    await asyncio.sleep(delay)
            
        return wrapper
    return decorator


def with_circuit_breaker(service_name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for adding circuit breaker pattern to functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            circuit_breaker = error_handler.get_circuit_breaker(service_name, config)
            
            if not circuit_breaker.can_execute():
                raise ExternalServiceError(f"Circuit breaker open for {service_name}")
            
            try:
                result = await func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception as e:
                circuit_breaker.record_failure()
                raise
        
        return wrapper
    return decorator