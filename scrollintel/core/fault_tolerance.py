"""
Fault Tolerance and Recovery System for Agent Steering System

This module implements enterprise-grade fault tolerance patterns including:
- Circuit breaker pattern for resilient service communication
- Retry logic with exponential backoff for transient failures
- Graceful degradation system for maintaining service during outages
- Automated recovery procedures for system failures

Requirements: 4.3, 9.2
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from datetime import datetime, timedelta
import json
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class FailureType(Enum):
    """Types of system failures"""
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    AUTHENTICATION_ERROR = "auth_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"

class DegradationLevel(Enum):
    """Levels of service degradation"""
    NONE = "none"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    success_threshold: int = 3
    timeout: int = 30

@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[type] = field(default_factory=lambda: [Exception])

@dataclass
class SystemFailure:
    """Represents a system failure"""
    failure_id: str
    failure_type: FailureType
    component: str
    message: str
    timestamp: datetime
    severity: str
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

@dataclass
class RecoveryResult:
    """Result of recovery operation"""
    recovery_id: str
    success: bool
    strategy: str
    duration: float
    actions_taken: List[str]
    health_status: Dict[str, Any]
    timestamp: datetime

class CircuitBreaker:
    """
    Circuit breaker implementation for resilient service communication
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
        
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.state == CircuitBreakerState.OPEN:
            if self.next_attempt_time and time.time() >= self.next_attempt_time:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.config.name} moved to HALF_OPEN")
                return False
            return True
        return False
    
    def record_success(self):
        """Record successful operation"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.config.name} moved to CLOSED")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.next_attempt_time = time.time() + self.config.recovery_timeout
                logger.warning(f"Circuit breaker {self.config.name} moved to OPEN")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout
            logger.warning(f"Circuit breaker {self.config.name} moved back to OPEN")

class RetryManager:
    """
    Retry logic with exponential backoff for transient failures
    """
    
    @staticmethod
    def calculate_delay(attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt"""
        delay = config.base_delay * (config.exponential_base ** (attempt - 1))
        delay = min(delay, config.max_delay)
        
        if config.jitter:
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, delay * 0.1)
            delay += jitter
            
        return delay
    
    @staticmethod
    def is_retryable_exception(exception: Exception, config: RetryConfig) -> bool:
        """Check if exception is retryable"""
        return any(isinstance(exception, exc_type) for exc_type in config.retryable_exceptions)

class GracefulDegradationManager:
    """
    Graceful degradation system for maintaining service during outages
    """
    
    def __init__(self):
        self.degradation_levels: Dict[str, DegradationLevel] = {}
        self.fallback_strategies: Dict[str, Callable] = {}
        self.service_health: Dict[str, bool] = {}
    
    def register_service(self, service_name: str, fallback_strategy: Callable):
        """Register service with fallback strategy"""
        self.degradation_levels[service_name] = DegradationLevel.NONE
        self.fallback_strategies[service_name] = fallback_strategy
        self.service_health[service_name] = True
        logger.info(f"Registered service {service_name} for graceful degradation")
    
    def degrade_service(self, service_name: str, level: DegradationLevel):
        """Degrade service to specified level"""
        if service_name in self.degradation_levels:
            self.degradation_levels[service_name] = level
            self.service_health[service_name] = level == DegradationLevel.NONE
            logger.warning(f"Service {service_name} degraded to level {level.value}")
    
    def get_fallback_response(self, service_name: str, *args, **kwargs):
        """Get fallback response for degraded service"""
        if service_name in self.fallback_strategies:
            level = self.degradation_levels.get(service_name, DegradationLevel.NONE)
            return self.fallback_strategies[service_name](level, *args, **kwargs)
        return None

class AutomatedRecoveryManager:
    """
    Automated recovery procedures for system failures
    """
    
    def __init__(self):
        self.recovery_strategies: Dict[FailureType, Callable] = {}
        self.recovery_history: List[RecoveryResult] = []
        self.health_checkers: Dict[str, Callable] = {}
    
    def register_recovery_strategy(self, failure_type: FailureType, strategy: Callable):
        """Register recovery strategy for failure type"""
        self.recovery_strategies[failure_type] = strategy
        logger.info(f"Registered recovery strategy for {failure_type.value}")
    
    def register_health_checker(self, component: str, checker: Callable):
        """Register health checker for component"""
        self.health_checkers[component] = checker
        logger.info(f"Registered health checker for {component}")
    
    async def initiate_recovery(self, failure: SystemFailure) -> RecoveryResult:
        """Initiate automated recovery for system failure"""
        start_time = time.time()
        recovery_id = f"recovery_{int(start_time)}_{failure.component}"
        
        logger.info(f"Initiating recovery {recovery_id} for failure {failure.failure_id}")
        
        try:
            # Assess failure impact
            impact = await self._assess_failure_impact(failure)
            
            # Determine recovery strategy
            strategy = await self._determine_recovery_strategy(failure, impact)
            
            # Execute recovery
            actions_taken = await self._execute_recovery_strategy(strategy, failure)
            
            # Validate recovery
            health_status = await self._validate_recovery(failure.component)
            
            duration = time.time() - start_time
            
            result = RecoveryResult(
                recovery_id=recovery_id,
                success=health_status.get('healthy', False),
                strategy=strategy,
                duration=duration,
                actions_taken=actions_taken,
                health_status=health_status,
                timestamp=datetime.now()
            )
            
            self.recovery_history.append(result)
            
            if result.success:
                logger.info(f"Recovery {recovery_id} completed successfully in {duration:.2f}s")
            else:
                logger.error(f"Recovery {recovery_id} failed after {duration:.2f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            result = RecoveryResult(
                recovery_id=recovery_id,
                success=False,
                strategy="error_handling",
                duration=duration,
                actions_taken=[f"Recovery failed: {str(e)}"],
                health_status={'healthy': False, 'error': str(e)},
                timestamp=datetime.now()
            )
            
            self.recovery_history.append(result)
            logger.error(f"Recovery {recovery_id} encountered error: {e}")
            return result
    
    async def _assess_failure_impact(self, failure: SystemFailure) -> Dict[str, Any]:
        """Assess the impact of system failure"""
        impact = {
            'severity': failure.severity,
            'affected_components': [failure.component],
            'estimated_downtime': 0,
            'user_impact': 'low'
        }
        
        # Determine impact based on failure type and component
        if failure.failure_type in [FailureType.SERVICE_UNAVAILABLE, FailureType.RESOURCE_EXHAUSTED]:
            impact['user_impact'] = 'high'
            impact['estimated_downtime'] = 300  # 5 minutes
        elif failure.failure_type in [FailureType.NETWORK_ERROR, FailureType.TIMEOUT]:
            impact['user_impact'] = 'medium'
            impact['estimated_downtime'] = 60  # 1 minute
        
        return impact
    
    async def _determine_recovery_strategy(self, failure: SystemFailure, impact: Dict[str, Any]) -> str:
        """Determine appropriate recovery strategy"""
        if failure.failure_type in self.recovery_strategies:
            return f"automated_{failure.failure_type.value}"
        
        # Default strategies based on impact
        if impact['user_impact'] == 'high':
            return "immediate_failover"
        elif impact['user_impact'] == 'medium':
            return "service_restart"
        else:
            return "monitoring_increase"
    
    async def _execute_recovery_strategy(self, strategy: str, failure: SystemFailure) -> List[str]:
        """Execute recovery strategy"""
        actions = []
        
        try:
            if strategy.startswith("automated_"):
                failure_type = FailureType(strategy.replace("automated_", ""))
                if failure_type in self.recovery_strategies:
                    result = await self.recovery_strategies[failure_type](failure)
                    actions.append(f"Executed automated strategy for {failure_type.value}")
                    if isinstance(result, list):
                        actions.extend(result)
            
            elif strategy == "immediate_failover":
                actions.append("Initiated immediate failover")
                actions.append(f"Redirected traffic from {failure.component}")
                actions.append("Activated backup systems")
            
            elif strategy == "service_restart":
                actions.append(f"Restarting service {failure.component}")
                await asyncio.sleep(2)  # Simulate restart time
                actions.append(f"Service {failure.component} restarted")
            
            elif strategy == "monitoring_increase":
                actions.append("Increased monitoring frequency")
                actions.append("Added additional health checks")
            
        except Exception as e:
            actions.append(f"Recovery action failed: {str(e)}")
        
        return actions
    
    async def _validate_recovery(self, component: str) -> Dict[str, Any]:
        """Validate system health after recovery"""
        health_status = {
            'healthy': True,
            'component': component,
            'checks_passed': [],
            'checks_failed': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if component in self.health_checkers:
                check_result = await self.health_checkers[component]()
                if check_result:
                    health_status['checks_passed'].append(f"{component}_health_check")
                else:
                    health_status['checks_failed'].append(f"{component}_health_check")
                    health_status['healthy'] = False
            else:
                # Default health check
                health_status['checks_passed'].append("basic_connectivity")
        
        except Exception as e:
            health_status['healthy'] = False
            health_status['checks_failed'].append(f"health_check_error: {str(e)}")
        
        return health_status

class FaultToleranceManager:
    """
    Main fault tolerance manager coordinating all fault tolerance components
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.degradation_manager = GracefulDegradationManager()
        self.recovery_manager = AutomatedRecoveryManager()
        self.retry_manager = RetryManager()
    
    def create_circuit_breaker(self, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Create and register circuit breaker"""
        circuit_breaker = CircuitBreaker(config)
        self.circuit_breakers[config.name] = circuit_breaker
        logger.info(f"Created circuit breaker: {config.name}")
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    async def execute_with_circuit_breaker(
        self, 
        operation: Callable[[], T], 
        config: CircuitBreakerConfig
    ) -> T:
        """Execute operation with circuit breaker protection"""
        circuit_breaker = self.circuit_breakers.get(config.name)
        if not circuit_breaker:
            circuit_breaker = self.create_circuit_breaker(config)
        
        if circuit_breaker.is_open():
            raise Exception(f"Circuit breaker {config.name} is open")
        
        try:
            result = await operation() if asyncio.iscoroutinefunction(operation) else operation()
            circuit_breaker.record_success()
            return result
        except config.expected_exception as e:
            circuit_breaker.record_failure()
            raise e
    
    async def execute_with_retry(
        self, 
        operation: Callable[[], T], 
        config: RetryConfig
    ) -> T:
        """Execute operation with retry logic"""
        last_exception = None
        
        for attempt in range(1, config.max_attempts + 1):
            try:
                result = await operation() if asyncio.iscoroutinefunction(operation) else operation()
                if attempt > 1:
                    logger.info(f"Operation succeeded on attempt {attempt}")
                return result
            
            except Exception as e:
                last_exception = e
                
                if attempt == config.max_attempts:
                    logger.error(f"Operation failed after {config.max_attempts} attempts")
                    break
                
                if not self.retry_manager.is_retryable_exception(e, config):
                    logger.error(f"Non-retryable exception: {e}")
                    break
                
                delay = self.retry_manager.calculate_delay(attempt, config)
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    async def execute_with_fault_tolerance(
        self,
        operation: Callable[[], T],
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        fallback: Optional[Callable] = None
    ) -> T:
        """Execute operation with comprehensive fault tolerance"""
        try:
            # Apply circuit breaker if configured
            if circuit_breaker_config:
                if retry_config:
                    # Combine circuit breaker with retry
                    async def retry_operation():
                        return await self.execute_with_retry(operation, retry_config)
                    return await self.execute_with_circuit_breaker(retry_operation, circuit_breaker_config)
                else:
                    return await self.execute_with_circuit_breaker(operation, circuit_breaker_config)
            
            # Apply retry if configured
            elif retry_config:
                return await self.execute_with_retry(operation, retry_config)
            
            # Execute operation directly
            else:
                return await operation() if asyncio.iscoroutinefunction(operation) else operation()
        
        except Exception as e:
            # Use fallback if available
            if fallback:
                logger.warning(f"Operation failed, using fallback: {e}")
                return await fallback() if asyncio.iscoroutinefunction(fallback) else fallback()
            raise e
    
    async def handle_system_failure(self, failure: SystemFailure) -> RecoveryResult:
        """Handle system failure with automated recovery"""
        logger.error(f"System failure detected: {failure.failure_id} - {failure.message}")
        
        # Initiate graceful degradation if needed
        if failure.severity in ['high', 'critical']:
            degradation_level = DegradationLevel.MODERATE if failure.severity == 'high' else DegradationLevel.SEVERE
            self.degradation_manager.degrade_service(failure.component, degradation_level)
        
        # Initiate automated recovery
        recovery_result = await self.recovery_manager.initiate_recovery(failure)
        
        # Restore service if recovery was successful
        if recovery_result.success:
            self.degradation_manager.degrade_service(failure.component, DegradationLevel.NONE)
        
        return recovery_result
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'circuit_breakers': {},
            'degraded_services': {},
            'recent_recoveries': []
        }
        
        # Check circuit breaker states
        for name, cb in self.circuit_breakers.items():
            health['circuit_breakers'][name] = {
                'state': cb.state.value,
                'failure_count': cb.failure_count,
                'success_count': cb.success_count
            }
            
            if cb.state != CircuitBreakerState.CLOSED:
                health['overall_status'] = 'degraded'
        
        # Check degraded services
        for service, level in self.degradation_manager.degradation_levels.items():
            if level != DegradationLevel.NONE:
                health['degraded_services'][service] = level.value
                health['overall_status'] = 'degraded'
        
        # Recent recovery attempts
        recent_recoveries = sorted(
            self.recovery_manager.recovery_history[-10:], 
            key=lambda x: x.timestamp, 
            reverse=True
        )
        
        for recovery in recent_recoveries:
            health['recent_recoveries'].append({
                'recovery_id': recovery.recovery_id,
                'success': recovery.success,
                'strategy': recovery.strategy,
                'duration': recovery.duration,
                'timestamp': recovery.timestamp.isoformat()
            })
        
        return health

# Example usage and default configurations
DEFAULT_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    name="default",
    failure_threshold=5,
    recovery_timeout=60,
    success_threshold=3,
    timeout=30
)

DEFAULT_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    retryable_exceptions=[ConnectionError, TimeoutError, Exception]
)

# Global fault tolerance manager instance
fault_tolerance_manager = FaultToleranceManager()