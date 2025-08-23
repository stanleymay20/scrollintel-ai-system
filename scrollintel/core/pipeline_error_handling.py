"""
Pipeline Error Handling and Recovery System
Provides comprehensive error handling, recovery mechanisms, and circuit breakers.
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import traceback
import json

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    MANUAL = "manual"

@dataclass
class PipelineError:
    """Pipeline error information"""
    error_id: str
    pipeline_id: str
    node_id: Optional[str]
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime
    stack_trace: str
    context: Dict[str, Any]
    recovery_strategy: RecoveryStrategy
    retry_count: int = 0
    resolved: bool = False

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker for pipeline operations"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN

class RetryPolicy:
    """Retry policy configuration"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay

class PipelineErrorHandler:
    """Comprehensive pipeline error handling system"""
    
    def __init__(self):
        self.errors: Dict[str, PipelineError] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        
        # Default retry policy
        self.default_retry_policy = RetryPolicy()
        
        # Error classification rules
        self.error_classifiers = {
            'connection_error': {
                'patterns': ['connection', 'timeout', 'network'],
                'severity': ErrorSeverity.MEDIUM,
                'strategy': RecoveryStrategy.RETRY
            },
            'data_validation_error': {
                'patterns': ['validation', 'schema', 'format'],
                'severity': ErrorSeverity.HIGH,
                'strategy': RecoveryStrategy.SKIP
            },
            'resource_error': {
                'patterns': ['memory', 'disk', 'cpu', 'resource'],
                'severity': ErrorSeverity.CRITICAL,
                'strategy': RecoveryStrategy.ABORT
            },
            'authentication_error': {
                'patterns': ['auth', 'permission', 'unauthorized'],
                'severity': ErrorSeverity.HIGH,
                'strategy': RecoveryStrategy.MANUAL
            }
        }
    
    def classify_error(self, error: Exception, context: Dict[str, Any]) -> tuple:
        """Classify error and determine recovery strategy"""
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Check classification rules
        for error_class, config in self.error_classifiers.items():
            if any(pattern in error_message for pattern in config['patterns']):
                return config['severity'], config['strategy']
        
        # Default classification
        return ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY
    
    def handle_error(self, error: Exception, pipeline_id: str, node_id: Optional[str] = None,
                    context: Dict[str, Any] = None) -> PipelineError:
        """Handle pipeline error with classification and recovery"""
        import uuid
        
        error_id = str(uuid.uuid4())
        severity, strategy = self.classify_error(error, context or {})
        
        pipeline_error = PipelineError(
            error_id=error_id,
            pipeline_id=pipeline_id,
            node_id=node_id,
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            timestamp=datetime.utcnow(),
            stack_trace=traceback.format_exc(),
            context=context or {},
            recovery_strategy=strategy
        )
        
        self.errors[error_id] = pipeline_error
        
        # Log error
        logger.error(f"Pipeline error {error_id}: {error}", extra={
            'pipeline_id': pipeline_id,
            'node_id': node_id,
            'error_type': pipeline_error.error_type,
            'severity': severity.value,
            'strategy': strategy.value
        })
        
        return pipeline_error
    
    def get_circuit_breaker(self, operation_key: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation"""
        if operation_key not in self.circuit_breakers:
            self.circuit_breakers[operation_key] = CircuitBreaker()
        return self.circuit_breakers[operation_key]
    
    def get_retry_policy(self, operation_key: str) -> RetryPolicy:
        """Get retry policy for operation"""
        return self.retry_policies.get(operation_key, self.default_retry_policy)
    
    def register_fallback_handler(self, operation_key: str, handler: Callable):
        """Register fallback handler for operation"""
        self.fallback_handlers[operation_key] = handler
    
    async def execute_with_recovery(self, func: Callable, operation_key: str,
                                  pipeline_id: str, node_id: Optional[str] = None,
                                  context: Dict[str, Any] = None) -> Any:
        """Execute function with comprehensive error handling and recovery"""
        circuit_breaker = self.get_circuit_breaker(operation_key)
        retry_policy = self.get_retry_policy(operation_key)
        
        last_error = None
        
        for attempt in range(retry_policy.max_attempts):
            try:
                # Execute with circuit breaker protection
                if asyncio.iscoroutinefunction(func):
                    result = await circuit_breaker.call(func)
                else:
                    result = circuit_breaker.call(func)
                
                return result
                
            except Exception as e:
                last_error = e
                pipeline_error = self.handle_error(e, pipeline_id, node_id, context)
                
                # Check if we should retry
                if (attempt < retry_policy.max_attempts - 1 and 
                    pipeline_error.recovery_strategy == RecoveryStrategy.RETRY):
                    
                    delay = retry_policy.get_delay(attempt)
                    logger.info(f"Retrying operation {operation_key} in {delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    continue
                
                # Try fallback if available
                if (pipeline_error.recovery_strategy == RecoveryStrategy.FALLBACK and 
                    operation_key in self.fallback_handlers):
                    
                    try:
                        fallback_handler = self.fallback_handlers[operation_key]
                        if asyncio.iscoroutinefunction(fallback_handler):
                            return await fallback_handler(pipeline_error, context)
                        else:
                            return fallback_handler(pipeline_error, context)
                    except Exception as fallback_error:
                        logger.error(f"Fallback handler failed: {fallback_error}")
                
                # Handle other recovery strategies
                if pipeline_error.recovery_strategy == RecoveryStrategy.SKIP:
                    logger.warning(f"Skipping operation {operation_key} due to error")
                    return None
                elif pipeline_error.recovery_strategy == RecoveryStrategy.ABORT:
                    logger.error(f"Aborting pipeline {pipeline_id} due to critical error")
                    raise e
                elif pipeline_error.recovery_strategy == RecoveryStrategy.MANUAL:
                    logger.error(f"Manual intervention required for pipeline {pipeline_id}")
                    raise e
                else:
                    # Default: re-raise the error
                    raise e
        
        # All retries exhausted
        if last_error:
            raise last_error

def with_error_handling(operation_key: str, pipeline_id: str = None, node_id: str = None):
    """Decorator for automatic error handling"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract pipeline_id from kwargs if not provided
            actual_pipeline_id = pipeline_id or kwargs.get('pipeline_id', 'unknown')
            actual_node_id = node_id or kwargs.get('node_id')
            
            context = {
                'function': func.__name__,
                'args': str(args)[:200],  # Limit context size
                'kwargs': {k: str(v)[:100] for k, v in kwargs.items()}
            }
            
            return await error_handler.execute_with_recovery(
                func, operation_key, actual_pipeline_id, actual_node_id, context
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, create async wrapper
            async def async_func():
                return func(*args, **kwargs)
            
            actual_pipeline_id = pipeline_id or kwargs.get('pipeline_id', 'unknown')
            actual_node_id = node_id or kwargs.get('node_id')
            
            context = {
                'function': func.__name__,
                'args': str(args)[:200],
                'kwargs': {k: str(v)[:100] for k, v in kwargs.items()}
            }
            
            return asyncio.run(error_handler.execute_with_recovery(
                async_func, operation_key, actual_pipeline_id, actual_node_id, context
            ))
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Global error handler instance
error_handler = PipelineErrorHandler()

# Convenience functions
def register_retry_policy(operation_key: str, policy: RetryPolicy):
    """Register custom retry policy"""
    error_handler.retry_policies[operation_key] = policy

def register_fallback(operation_key: str, handler: Callable):
    """Register fallback handler"""
    error_handler.register_fallback_handler(operation_key, handler)

def get_error_summary(pipeline_id: str) -> Dict[str, Any]:
    """Get error summary for pipeline"""
    pipeline_errors = [e for e in error_handler.errors.values() 
                      if e.pipeline_id == pipeline_id]
    
    return {
        'total_errors': len(pipeline_errors),
        'by_severity': {
            severity.value: len([e for e in pipeline_errors if e.severity == severity])
            for severity in ErrorSeverity
        },
        'by_type': {},
        'recent_errors': [
            {
                'error_id': e.error_id,
                'message': e.message,
                'severity': e.severity.value,
                'timestamp': e.timestamp.isoformat()
            }
            for e in sorted(pipeline_errors, key=lambda x: x.timestamp, reverse=True)[:10]
        ]
    }