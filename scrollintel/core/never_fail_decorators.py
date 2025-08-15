"""
Never-fail decorators for ScrollIntel.
Ensures critical functions never fail and always return something useful.
Enhanced with context-aware fallbacks, intelligent retry strategies, and progressive timeout handling.
"""

import asyncio
import logging
import functools
import time
import random
import hashlib
from typing import Any, Callable, Optional, Dict, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import traceback
import json
import inspect
from contextlib import asynccontextmanager

from .failure_prevention import failure_prevention, bulletproof, critical_operation
from .graceful_degradation import degradation_manager, with_graceful_degradation
from .user_experience_protection import ux_protector, track_user_action

logger = logging.getLogger(__name__)


class FailurePattern(Enum):
    """Types of failure patterns for intelligent retry strategies."""
    TRANSIENT = "transient"          # Temporary network/service issues
    RESOURCE_EXHAUSTION = "resource" # Memory, CPU, disk issues
    RATE_LIMITED = "rate_limited"    # API rate limiting
    DEPENDENCY_FAILURE = "dependency" # External service failures
    DATA_CORRUPTION = "data_corruption" # Data integrity issues
    TIMEOUT = "timeout"              # Operation timeouts
    AUTHENTICATION = "auth"          # Authentication/authorization issues
    UNKNOWN = "unknown"              # Unclassified failures


@dataclass
class FailureContext:
    """Context information about a failure for intelligent handling."""
    function_name: str
    failure_pattern: FailurePattern
    attempt_count: int
    total_elapsed_time: float
    last_error: Exception
    user_context: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    historical_failures: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RetryConfig:
    """Configuration for intelligent retry strategies."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.1
    pattern_specific_delays: Dict[FailurePattern, float] = field(default_factory=dict)
    adaptive_backoff: bool = True
    circuit_breaker_threshold: int = 5


class IntelligentRetryStrategy:
    """Intelligent retry strategy that adapts based on failure patterns."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.failure_history: Dict[str, List[FailureContext]] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Default pattern-specific delays
        self.config.pattern_specific_delays.update({
            FailurePattern.TRANSIENT: 0.5,
            FailurePattern.RESOURCE_EXHAUSTION: 5.0,
            FailurePattern.RATE_LIMITED: 10.0,
            FailurePattern.DEPENDENCY_FAILURE: 2.0,
            FailurePattern.DATA_CORRUPTION: 0.1,
            FailurePattern.TIMEOUT: 1.0,
            FailurePattern.AUTHENTICATION: 30.0,
            FailurePattern.UNKNOWN: 1.0
        })
    
    def classify_failure(self, error: Exception, context: Dict[str, Any] = None) -> FailurePattern:
        """Classify failure pattern based on error and context."""
        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Network and connection issues
        if any(term in error_msg for term in ['connection', 'network', 'unreachable', 'dns']):
            return FailurePattern.TRANSIENT
        
        # Resource exhaustion
        if any(term in error_msg for term in ['memory', 'disk', 'cpu', 'resource', 'quota']):
            return FailurePattern.RESOURCE_EXHAUSTION
        
        # Rate limiting
        if any(term in error_msg for term in ['rate limit', 'too many requests', '429', 'throttle']):
            return FailurePattern.RATE_LIMITED
        
        # Timeout issues
        if any(term in error_msg for term in ['timeout', 'timed out', 'deadline']):
            return FailurePattern.TIMEOUT
        
        # Authentication issues
        if any(term in error_msg for term in ['auth', 'unauthorized', '401', '403', 'forbidden']):
            return FailurePattern.AUTHENTICATION
        
        # Data corruption
        if any(term in error_msg for term in ['corrupt', 'invalid data', 'checksum', 'integrity']):
            return FailurePattern.DATA_CORRUPTION
        
        # Dependency failures
        if any(term in error_msg for term in ['service unavailable', '503', '502', '504', 'bad gateway']):
            return FailurePattern.DEPENDENCY_FAILURE
        
        return FailurePattern.UNKNOWN
    
    def calculate_delay(self, failure_context: FailureContext) -> float:
        """Calculate intelligent delay based on failure pattern and history."""
        pattern = failure_context.failure_pattern
        attempt = failure_context.attempt_count
        
        # Base delay from pattern-specific configuration
        base_delay = self.config.pattern_specific_delays.get(pattern, self.config.base_delay)
        
        # Exponential backoff
        if self.config.adaptive_backoff:
            # Analyze historical failures for this function
            func_history = self.failure_history.get(failure_context.function_name, [])
            recent_failures = [f for f in func_history 
                             if f.failure_pattern == pattern and 
                             (datetime.now() - datetime.fromtimestamp(time.time() - 300)).total_seconds() < 300]
            
            # Increase delay if there are many recent failures of the same pattern
            if len(recent_failures) > 3:
                base_delay *= (1 + len(recent_failures) * 0.5)
        
        # Exponential backoff with jitter
        delay = min(
            base_delay * (self.config.exponential_base ** (attempt - 1)),
            self.config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        jitter = delay * self.config.jitter_factor * (random.random() - 0.5)
        delay += jitter
        
        return max(0.1, delay)  # Minimum 100ms delay
    
    def should_retry(self, failure_context: FailureContext) -> bool:
        """Determine if retry should be attempted based on context."""
        # Check circuit breaker
        func_name = failure_context.function_name
        if func_name in self.circuit_breakers:
            cb = self.circuit_breakers[func_name]
            if cb['state'] == 'open':
                if time.time() - cb['last_failure'] < cb['timeout']:
                    return False
                else:
                    # Try to close circuit breaker
                    cb['state'] = 'half_open'
        
        # Check max attempts
        if failure_context.attempt_count >= self.config.max_attempts:
            return False
        
        # Pattern-specific retry logic
        pattern = failure_context.failure_pattern
        
        # Don't retry authentication errors
        if pattern == FailurePattern.AUTHENTICATION:
            return False
        
        # Don't retry data corruption errors
        if pattern == FailurePattern.DATA_CORRUPTION:
            return False
        
        # Limit retries for resource exhaustion
        if pattern == FailurePattern.RESOURCE_EXHAUSTION and failure_context.attempt_count >= 2:
            return False
        
        return True
    
    def record_failure(self, failure_context: FailureContext):
        """Record failure for learning and circuit breaker logic."""
        func_name = failure_context.function_name
        
        # Update failure history
        if func_name not in self.failure_history:
            self.failure_history[func_name] = []
        
        self.failure_history[func_name].append(failure_context)
        
        # Keep only recent failures (last 1 hour)
        cutoff_time = time.time() - 3600
        self.failure_history[func_name] = [
            f for f in self.failure_history[func_name]
            if time.time() - f.total_elapsed_time < cutoff_time
        ]
        
        # Update circuit breaker
        if func_name not in self.circuit_breakers:
            self.circuit_breakers[func_name] = {
                'state': 'closed',
                'failure_count': 0,
                'last_failure': 0,
                'timeout': 60
            }
        
        cb = self.circuit_breakers[func_name]
        cb['failure_count'] += 1
        cb['last_failure'] = time.time()
        
        # Open circuit breaker if too many failures
        if cb['failure_count'] >= self.config.circuit_breaker_threshold:
            cb['state'] = 'open'
            logger.warning(f"Circuit breaker opened for {func_name}")
    
    def record_success(self, func_name: str):
        """Record successful execution."""
        if func_name in self.circuit_breakers:
            cb = self.circuit_breakers[func_name]
            cb['failure_count'] = 0
            cb['state'] = 'closed'


class ContextAwareFallbackGenerator:
    """Generates context-aware fallback data based on function signature and usage patterns."""
    
    def __init__(self):
        self.fallback_templates: Dict[str, Any] = {}
        self.usage_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize fallback data templates."""
        self.fallback_templates = {
            # Data structure templates
            'list': [],
            'dict': {},
            'dataframe': {'columns': [], 'data': [], 'empty': True},
            'chart_data': {
                'type': 'placeholder',
                'data': [{'x': 'No Data', 'y': 0}],
                'title': 'Data Temporarily Unavailable',
                'message': 'Please try again in a moment'
            },
            'api_response': {
                'success': True,
                'data': None,
                'message': 'Service temporarily unavailable',
                'fallback': True,
                'retry_after': 30
            },
            'user_data': {
                'id': 'anonymous',
                'name': 'Guest User',
                'preferences': {},
                'temporary': True
            },
            'file_content': {
                'content': '',
                'size': 0,
                'type': 'text/plain',
                'error': 'File temporarily unavailable'
            },
            'analysis_result': {
                'summary': 'Analysis temporarily unavailable',
                'insights': ['Please try again later'],
                'confidence': 0.0,
                'fallback': True
            }
        }
    
    def generate_fallback_data(self, func: Callable, args: tuple, kwargs: dict, 
                             error: Exception, context: Dict[str, Any] = None) -> Any:
        """Generate context-aware fallback data."""
        func_name = func.__name__
        
        # Analyze function signature
        sig = inspect.signature(func)
        return_annotation = sig.return_annotation
        
        # Record usage pattern
        self._record_usage_pattern(func_name, args, kwargs, context)
        
        # Generate fallback based on function characteristics
        fallback_data = self._generate_by_function_analysis(func, args, kwargs, error, context)
        
        if fallback_data is not None:
            return fallback_data
        
        # Generate fallback based on return type annotation
        if return_annotation != inspect.Signature.empty:
            fallback_data = self._generate_by_return_type(return_annotation, func_name, context)
            if fallback_data is not None:
                return fallback_data
        
        # Generate fallback based on function name patterns
        return self._generate_by_name_pattern(func_name, args, kwargs, context)
    
    def _record_usage_pattern(self, func_name: str, args: tuple, kwargs: dict, 
                            context: Dict[str, Any] = None):
        """Record usage patterns for learning."""
        if func_name not in self.usage_patterns:
            self.usage_patterns[func_name] = []
        
        pattern = {
            'timestamp': time.time(),
            'args_types': [type(arg).__name__ for arg in args],
            'kwargs_keys': list(kwargs.keys()),
            'context': context or {}
        }
        
        self.usage_patterns[func_name].append(pattern)
        
        # Keep only recent patterns
        cutoff_time = time.time() - 3600  # 1 hour
        self.usage_patterns[func_name] = [
            p for p in self.usage_patterns[func_name]
            if p['timestamp'] > cutoff_time
        ]
    
    def _generate_by_function_analysis(self, func: Callable, args: tuple, kwargs: dict,
                                     error: Exception, context: Dict[str, Any] = None) -> Any:
        """Generate fallback based on function analysis."""
        func_name = func.__name__.lower()
        
        # Database query functions
        if any(term in func_name for term in ['get', 'fetch', 'find', 'query', 'select']):
            if 'list' in func_name or 'all' in func_name:
                return []
            elif 'count' in func_name:
                return 0
            elif 'exists' in func_name or 'check' in func_name:
                return False
            else:
                return None
        
        # Chart/visualization functions
        if any(term in func_name for term in ['chart', 'plot', 'graph', 'visualize']):
            return self._generate_chart_fallback(args, kwargs, context)
        
        # Analysis functions
        if any(term in func_name for term in ['analyze', 'process', 'calculate', 'compute']):
            return self._generate_analysis_fallback(args, kwargs, context)
        
        # File operations
        if any(term in func_name for term in ['read', 'load', 'import', 'upload']):
            return self._generate_file_fallback(args, kwargs, context)
        
        # API calls
        if any(term in func_name for term in ['api', 'request', 'call', 'send']):
            return self._generate_api_fallback(args, kwargs, context)
        
        return None
    
    def _generate_by_return_type(self, return_type: type, func_name: str, 
                               context: Dict[str, Any] = None) -> Any:
        """Generate fallback based on return type annotation."""
        if return_type == list or str(return_type).startswith('typing.List'):
            return []
        elif return_type == dict or str(return_type).startswith('typing.Dict'):
            return {}
        elif return_type == str:
            return f"Data temporarily unavailable for {func_name}"
        elif return_type == int:
            return 0
        elif return_type == float:
            return 0.0
        elif return_type == bool:
            return False
        
        return None
    
    def _generate_by_name_pattern(self, func_name: str, args: tuple, kwargs: dict,
                                context: Dict[str, Any] = None) -> Any:
        """Generate fallback based on function name patterns."""
        func_name_lower = func_name.lower()
        
        # Default patterns based on common naming conventions
        if func_name_lower.endswith('_list') or func_name_lower.startswith('list_'):
            return []
        elif func_name_lower.endswith('_count') or func_name_lower.startswith('count_'):
            return 0
        elif func_name_lower.endswith('_exists') or func_name_lower.startswith('exists_'):
            return False
        elif func_name_lower.endswith('_data') or func_name_lower.startswith('get_'):
            return self.fallback_templates['dict'].copy()
        
        # Return generic fallback
        return {
            'error': 'Function temporarily unavailable',
            'function': func_name,
            'fallback': True,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _generate_chart_fallback(self, args: tuple, kwargs: dict, 
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate fallback for chart/visualization functions."""
        chart_data = self.fallback_templates['chart_data'].copy()
        
        # Try to extract data from arguments
        for arg in args:
            if isinstance(arg, (list, dict)) and arg:
                if isinstance(arg, list) and len(arg) > 0:
                    # Use first few items as sample data
                    sample_data = arg[:3] if len(arg) > 3 else arg
                    chart_data['data'] = [
                        {'x': f'Sample {i+1}', 'y': hash(str(item)) % 100}
                        for i, item in enumerate(sample_data)
                    ]
                    chart_data['title'] = 'Sample Data (Fallback)'
                    break
        
        return chart_data
    
    def _generate_analysis_fallback(self, args: tuple, kwargs: dict,
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate fallback for analysis functions."""
        analysis_result = self.fallback_templates['analysis_result'].copy()
        
        # Customize based on context
        if context and 'user_id' in context:
            analysis_result['summary'] = f"Analysis for user {context['user_id']} is temporarily unavailable"
        
        return analysis_result
    
    def _generate_file_fallback(self, args: tuple, kwargs: dict,
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate fallback for file operations."""
        file_content = self.fallback_templates['file_content'].copy()
        
        # Try to extract filename from arguments
        for arg in args:
            if isinstance(arg, str) and ('.' in arg or '/' in arg):
                file_content['filename'] = arg
                break
        
        return file_content
    
    def _generate_api_fallback(self, args: tuple, kwargs: dict,
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate fallback for API calls."""
        api_response = self.fallback_templates['api_response'].copy()
        
        # Customize based on HTTP method or endpoint
        if kwargs.get('method') == 'POST':
            api_response['message'] = 'Data submission temporarily unavailable'
        elif kwargs.get('method') == 'GET':
            api_response['message'] = 'Data retrieval temporarily unavailable'
        
        return api_response


class ProgressiveTimeoutHandler:
    """Handles progressive timeout with user feedback."""
    
    def __init__(self):
        self.timeout_thresholds = [1.0, 3.0, 10.0, 30.0]  # Progressive timeout levels
        self.user_messages = {
            1.0: "Processing...",
            3.0: "This is taking a bit longer than usual...",
            10.0: "Still working on your request. Thank you for your patience.",
            30.0: "This is taking longer than expected. We're still processing your request."
        }
    
    @asynccontextmanager
    async def progressive_timeout(self, func_name: str, user_id: str = None, 
                                max_timeout: float = 60.0):
        """Context manager for progressive timeout handling."""
        start_time = time.time()
        current_threshold_index = 0
        
        # Start background task for user feedback
        feedback_task = asyncio.create_task(
            self._provide_progressive_feedback(func_name, user_id, start_time)
        )
        
        try:
            yield
        finally:
            feedback_task.cancel()
            
            # Final user feedback
            elapsed = time.time() - start_time
            if elapsed > max_timeout:
                logger.warning(f"Function {func_name} exceeded maximum timeout: {elapsed:.2f}s")
            
            # Clean up any user notifications
            if user_id:
                await self._clear_user_notifications(user_id, func_name)
    
    async def _provide_progressive_feedback(self, func_name: str, user_id: str, start_time: float):
        """Provide progressive feedback to user."""
        threshold_index = 0
        
        try:
            while threshold_index < len(self.timeout_thresholds):
                threshold = self.timeout_thresholds[threshold_index]
                elapsed = time.time() - start_time
                
                if elapsed >= threshold:
                    message = self.user_messages[threshold]
                    await self._send_user_notification(user_id, func_name, message, elapsed)
                    threshold_index += 1
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
        except asyncio.CancelledError:
            pass
    
    async def _send_user_notification(self, user_id: str, func_name: str, 
                                    message: str, elapsed: float):
        """Send notification to user about operation progress."""
        if user_id:
            # This would integrate with your notification system
            logger.info(f"User {user_id} notification for {func_name}: {message} (elapsed: {elapsed:.1f}s)")
            
            # Update user experience protector
            if hasattr(ux_protector, 'update_loading_message'):
                await ux_protector.update_loading_message(user_id, func_name, message)
    
    async def _clear_user_notifications(self, user_id: str, func_name: str):
        """Clear user notifications for completed operation."""
        if user_id:
            logger.debug(f"Clearing notifications for user {user_id}, function {func_name}")


# Global instances
intelligent_retry = IntelligentRetryStrategy()
fallback_generator = ContextAwareFallbackGenerator()
timeout_handler = ProgressiveTimeoutHandler()


def never_fail_api_endpoint(
    fallback_response: Optional[Dict[str, Any]] = None,
    degradation_service: str = "api",
    user_action: str = "api_call",
    context_aware: bool = True,
    progressive_timeout: bool = True,
    max_timeout: float = 30.0
):
    """
    Enhanced decorator that ensures API endpoints never fail.
    Features context-aware fallbacks, intelligent retries, and progressive timeout handling.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            user_id = kwargs.get('user_id') or (args[0].get('user_id') if args and isinstance(args[0], dict) else None)
            
            # Progressive timeout handling
            timeout_context = timeout_handler.progressive_timeout(
                func.__name__, user_id, max_timeout
            ) if progressive_timeout else None
            
            try:
                if timeout_context:
                    async with timeout_context:
                        result = await _execute_with_intelligent_retry(
                            func, args, kwargs, degradation_service, context_aware
                        )
                else:
                    result = await _execute_with_intelligent_retry(
                        func, args, kwargs, degradation_service, context_aware
                    )
                
                # Ensure result is always a valid response
                if result is None:
                    if context_aware:
                        result = fallback_generator.generate_fallback_data(
                            func, args, kwargs, None, {'service': degradation_service}
                        )
                    else:
                        result = fallback_response or {
                            "success": True,
                            "data": None,
                            "message": "Operation completed"
                        }
                
                # Add enhanced metadata
                if isinstance(result, dict):
                    result.update({
                        "response_time": time.time() - start_time,
                        "timestamp": datetime.utcnow().isoformat(),
                        "service_health": "operational",
                        "cache_used": False
                    })
                
                # Record success for learning
                intelligent_retry.record_success(func.__name__)
                
                return result
                
            except Exception as e:
                logger.error(f"API endpoint {func.__name__} failed: {e}", exc_info=True)
                
                # Create failure context
                failure_context = FailureContext(
                    function_name=func.__name__,
                    failure_pattern=intelligent_retry.classify_failure(e),
                    attempt_count=1,
                    total_elapsed_time=time.time() - start_time,
                    last_error=e,
                    user_context={'user_id': user_id, 'service': degradation_service}
                )
                
                # Try graceful degradation
                degraded_result = await degradation_manager.apply_degradation(
                    degradation_service, [failure_context.failure_pattern.value, "api_failure"]
                )
                
                if degraded_result:
                    return {
                        "success": True,
                        "data": degraded_result,
                        "degraded": True,
                        "degradation_reason": failure_context.failure_pattern.value,
                        "message": "Service operating in reduced capacity",
                        "response_time": time.time() - start_time,
                        "timestamp": datetime.utcnow().isoformat(),
                        "retry_recommended": intelligent_retry.should_retry(failure_context)
                    }
                
                # Context-aware fallback generation
                if context_aware:
                    context_fallback = fallback_generator.generate_fallback_data(
                        func, args, kwargs, e, {'service': degradation_service, 'user_id': user_id}
                    )
                    if context_fallback:
                        return {
                            "success": True,
                            "data": context_fallback,
                            "fallback": True,
                            "fallback_reason": str(e),
                            "response_time": time.time() - start_time,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                
                # Ultimate fallback with enhanced error information
                return fallback_response or {
                    "success": False,
                    "error": "Service temporarily unavailable",
                    "error_type": failure_context.failure_pattern.value,
                    "message": _get_user_friendly_error_message(failure_context.failure_pattern),
                    "retry_after": _calculate_retry_delay(failure_context.failure_pattern),
                    "support_available": True,
                    "response_time": time.time() - start_time,
                    "timestamp": datetime.utcnow().isoformat(),
                    "troubleshooting": _get_troubleshooting_tips(failure_context.failure_pattern)
                }
        
        return wrapper
    return decorator


async def _execute_with_intelligent_retry(func: Callable, args: tuple, kwargs: dict,
                                        service_name: str, context_aware: bool) -> Any:
    """Execute function with intelligent retry strategy."""
    start_time = time.time()
    last_exception = None
    
    for attempt in range(intelligent_retry.config.max_attempts):
        try:
            # Execute with failure prevention
            result = await failure_prevention._execute_with_protection(
                func, service_name, None, *args, **kwargs
            )
            
            # Record success
            intelligent_retry.record_success(func.__name__)
            return result
            
        except Exception as e:
            last_exception = e
            
            # Create failure context
            failure_context = FailureContext(
                function_name=func.__name__,
                failure_pattern=intelligent_retry.classify_failure(e),
                attempt_count=attempt + 1,
                total_elapsed_time=time.time() - start_time,
                last_error=e,
                system_state={'service': service_name}
            )
            
            # Record failure
            intelligent_retry.record_failure(failure_context)
            
            # Check if we should retry
            if not intelligent_retry.should_retry(failure_context):
                break
            
            # Calculate intelligent delay
            delay = intelligent_retry.calculate_delay(failure_context)
            
            logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1})")
            await asyncio.sleep(delay)
    
    # All retries exhausted
    raise last_exception


def _get_user_friendly_error_message(pattern: FailurePattern) -> str:
    """Get user-friendly error message based on failure pattern."""
    messages = {
        FailurePattern.TRANSIENT: "We're experiencing temporary connectivity issues. Please try again in a moment.",
        FailurePattern.RESOURCE_EXHAUSTION: "Our servers are currently busy. We're working to resolve this quickly.",
        FailurePattern.RATE_LIMITED: "You're making requests too quickly. Please wait a moment before trying again.",
        FailurePattern.DEPENDENCY_FAILURE: "An external service is temporarily unavailable. We're monitoring the situation.",
        FailurePattern.TIMEOUT: "Your request is taking longer than expected. Please try again or contact support.",
        FailurePattern.AUTHENTICATION: "There's an issue with your authentication. Please log in again.",
        FailurePattern.DATA_CORRUPTION: "We detected a data integrity issue. Please contact support.",
        FailurePattern.UNKNOWN: "We encountered an unexpected issue. Our team has been notified."
    }
    return messages.get(pattern, "Service temporarily unavailable. Please try again later.")


def _calculate_retry_delay(pattern: FailurePattern) -> int:
    """Calculate recommended retry delay based on failure pattern."""
    delays = {
        FailurePattern.TRANSIENT: 5,
        FailurePattern.RESOURCE_EXHAUSTION: 30,
        FailurePattern.RATE_LIMITED: 60,
        FailurePattern.DEPENDENCY_FAILURE: 15,
        FailurePattern.TIMEOUT: 10,
        FailurePattern.AUTHENTICATION: 0,  # Don't retry auth errors
        FailurePattern.DATA_CORRUPTION: 0,  # Don't retry data corruption
        FailurePattern.UNKNOWN: 30
    }
    return delays.get(pattern, 30)


def _get_troubleshooting_tips(pattern: FailurePattern) -> List[str]:
    """Get troubleshooting tips based on failure pattern."""
    tips = {
        FailurePattern.TRANSIENT: [
            "Check your internet connection",
            "Try refreshing the page",
            "Wait a few moments and try again"
        ],
        FailurePattern.RESOURCE_EXHAUSTION: [
            "Try again during off-peak hours",
            "Reduce the complexity of your request",
            "Contact support if the issue persists"
        ],
        FailurePattern.RATE_LIMITED: [
            "Wait before making another request",
            "Reduce the frequency of your requests",
            "Consider upgrading your plan for higher limits"
        ],
        FailurePattern.DEPENDENCY_FAILURE: [
            "This is a temporary external service issue",
            "Try again in a few minutes",
            "Check our status page for updates"
        ],
        FailurePattern.TIMEOUT: [
            "Try breaking your request into smaller parts",
            "Check your internet connection",
            "Contact support if timeouts persist"
        ],
        FailurePattern.AUTHENTICATION: [
            "Log out and log back in",
            "Clear your browser cache",
            "Reset your password if needed"
        ],
        FailurePattern.DATA_CORRUPTION: [
            "Contact support immediately",
            "Do not retry the operation",
            "Provide details about what you were doing"
        ],
        FailurePattern.UNKNOWN: [
            "Try refreshing the page",
            "Clear your browser cache",
            "Contact support with error details"
        ]
    }
    return tips.get(pattern, ["Contact support for assistance"])


def never_fail_data_operation(
    fallback_data: Any = None,
    cache_key: Optional[str] = None,
    max_retries: int = 3,
    context_aware: bool = True,
    auto_generate_fallback: bool = True,
    cache_duration: int = 300,
    stale_cache_acceptable: bool = True
):
    """
    Enhanced decorator for data operations that must never fail.
    Features intelligent caching, context-aware fallbacks, and automatic data generation.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Generate dynamic cache key if not provided
            effective_cache_key = cache_key or _generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache first
            if effective_cache_key:
                cached_data = await _get_from_cache(effective_cache_key)
                if cached_data is not None:
                    logger.info(f"Using cached data for {func.__name__}")
                    # Add cache metadata
                    if isinstance(cached_data, dict):
                        cached_data['_cache_hit'] = True
                        cached_data['_cached_at'] = cached_data.get('_cached_at', 'unknown')
                    return cached_data
            
            # Execute with intelligent retry
            last_exception = None
            failure_context = None
            
            for attempt in range(max_retries):
                try:
                    result = await func(*args, **kwargs)
                    
                    # Cache successful result with metadata
                    if effective_cache_key and result is not None:
                        cache_data = result
                        if isinstance(result, dict):
                            cache_data = result.copy()
                            cache_data['_cached_at'] = datetime.utcnow().isoformat()
                            cache_data['_cache_hit'] = False
                        
                        await _store_in_cache(effective_cache_key, cache_data, cache_duration)
                    
                    # Record success
                    intelligent_retry.record_success(func.__name__)
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Create failure context
                    failure_context = FailureContext(
                        function_name=func.__name__,
                        failure_pattern=intelligent_retry.classify_failure(e),
                        attempt_count=attempt + 1,
                        total_elapsed_time=time.time() - start_time,
                        last_error=e,
                        user_context={'cache_key': effective_cache_key}
                    )
                    
                    logger.warning(f"Data operation {func.__name__} failed (attempt {attempt + 1}): {e}")
                    
                    # Check if we should retry
                    if attempt < max_retries - 1 and intelligent_retry.should_retry(failure_context):
                        delay = intelligent_retry.calculate_delay(failure_context)
                        await asyncio.sleep(delay)
                    else:
                        break
            
            # All attempts failed, try fallback strategies
            logger.error(f"All attempts failed for {func.__name__}: {last_exception}")
            
            # Record failure for learning
            if failure_context:
                intelligent_retry.record_failure(failure_context)
            
            # Strategy 1: Try stale cache data
            if effective_cache_key and stale_cache_acceptable:
                stale_data = await _get_from_cache(effective_cache_key, allow_stale=True)
                if stale_data is not None:
                    logger.info(f"Using stale cached data for {func.__name__}")
                    if isinstance(stale_data, dict):
                        stale_data['_stale_cache'] = True
                        stale_data['_fallback_reason'] = 'stale_cache'
                    return stale_data
            
            # Strategy 2: Use provided fallback data
            if fallback_data is not None:
                logger.info(f"Using provided fallback data for {func.__name__}")
                if isinstance(fallback_data, dict):
                    fallback_copy = fallback_data.copy()
                    fallback_copy['_fallback_reason'] = 'provided_fallback'
                    return fallback_copy
                return fallback_data
            
            # Strategy 3: Context-aware fallback generation
            if context_aware and auto_generate_fallback:
                generated_fallback = fallback_generator.generate_fallback_data(
                    func, args, kwargs, last_exception, 
                    {'cache_key': effective_cache_key, 'attempt_count': max_retries}
                )
                if generated_fallback is not None:
                    logger.info(f"Using generated fallback data for {func.__name__}")
                    if isinstance(generated_fallback, dict):
                        generated_fallback['_fallback_reason'] = 'context_aware_generation'
                        generated_fallback['_generated_at'] = datetime.utcnow().isoformat()
                    return generated_fallback
            
            # Strategy 4: Pattern-based fallback
            return _generate_pattern_based_fallback(func.__name__, args, kwargs, last_exception)
        
        return wrapper
    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a cache key based on function name and arguments."""
    # Create a hash of the arguments
    arg_str = str(args) + str(sorted(kwargs.items()))
    arg_hash = hashlib.md5(arg_str.encode()).hexdigest()[:8]
    return f"{func_name}_{arg_hash}"


def _generate_pattern_based_fallback(func_name: str, args: tuple, kwargs: dict, 
                                   error: Exception) -> Any:
    """Generate fallback data based on function name patterns."""
    func_name_lower = func_name.lower()
    
    # List operations
    if any(term in func_name_lower for term in ['list', 'get_all', 'fetch_all', 'find_all']):
        return {
            'data': [],
            'total': 0,
            'page': 1,
            'per_page': 10,
            '_fallback_reason': 'empty_list_pattern',
            '_error': str(error)
        }
    
    # Count operations
    if any(term in func_name_lower for term in ['count', 'total', 'sum']):
        return {
            'count': 0,
            '_fallback_reason': 'zero_count_pattern',
            '_error': str(error)
        }
    
    # Boolean operations
    if any(term in func_name_lower for term in ['exists', 'has_', 'is_', 'can_', 'check']):
        return {
            'result': False,
            '_fallback_reason': 'false_boolean_pattern',
            '_error': str(error)
        }
    
    # Get single item operations
    if any(term in func_name_lower for term in ['get_', 'find_', 'fetch_']):
        return {
            'data': None,
            'found': False,
            '_fallback_reason': 'null_item_pattern',
            '_error': str(error)
        }
    
    # Analysis/calculation operations
    if any(term in func_name_lower for term in ['analyze', 'calculate', 'compute', 'process']):
        return {
            'result': 'Analysis temporarily unavailable',
            'status': 'failed',
            'confidence': 0.0,
            '_fallback_reason': 'analysis_failure_pattern',
            '_error': str(error)
        }
    
    # Default fallback
    return {
        'error': 'Data temporarily unavailable',
        'function': func_name,
        'fallback': True,
        'timestamp': datetime.utcnow().isoformat(),
        '_fallback_reason': 'generic_pattern',
        '_error': str(error)
    }


def never_fail_visualization(
    fallback_chart_type: str = "table",
    sample_data: Optional[List[Dict[str, Any]]] = None,
    context_aware: bool = True,
    progressive_degradation: bool = True,
    auto_simplify: bool = True
):
    """
    Enhanced decorator for visualization functions that must never fail.
    Features progressive degradation, context-aware fallbacks, and automatic simplification.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Try original function first
                result = await func(*args, **kwargs)
                
                # Validate result structure
                if _is_valid_visualization_result(result):
                    return result
                else:
                    logger.warning(f"Invalid visualization result from {func.__name__}, applying fallback")
                    raise ValueError("Invalid visualization result structure")
                
            except Exception as e:
                logger.error(f"Visualization {func.__name__} failed: {e}", exc_info=True)
                
                # Create failure context
                failure_context = FailureContext(
                    function_name=func.__name__,
                    failure_pattern=intelligent_retry.classify_failure(e),
                    attempt_count=1,
                    total_elapsed_time=time.time() - start_time,
                    last_error=e
                )
                
                # Progressive degradation strategy
                if progressive_degradation:
                    degraded_result = await _apply_visualization_degradation(
                        func, args, kwargs, failure_context, fallback_chart_type
                    )
                    if degraded_result:
                        return degraded_result
                
                # Context-aware fallback generation
                if context_aware:
                    context_result = await _generate_context_aware_visualization(
                        func, args, kwargs, e, fallback_chart_type, sample_data
                    )
                    if context_result:
                        return context_result
                
                # Ultimate fallback
                return _generate_ultimate_visualization_fallback(
                    func.__name__, args, kwargs, e, fallback_chart_type, sample_data
                )
        
        return wrapper
    return decorator


def _is_valid_visualization_result(result: Any) -> bool:
    """Validate that a visualization result has the expected structure."""
    if not isinstance(result, dict):
        return False
    
    # Check for required fields
    required_fields = ['data']
    optional_fields = ['chart_type', 'title', 'config', 'success']
    
    # Must have data field
    if 'data' not in result:
        return False
    
    # Data should be a list or dict
    if not isinstance(result['data'], (list, dict)):
        return False
    
    return True


async def _apply_visualization_degradation(func: Callable, args: tuple, kwargs: dict,
                                         failure_context: FailureContext, 
                                         fallback_chart_type: str) -> Optional[Dict[str, Any]]:
    """Apply progressive degradation to visualization."""
    pattern = failure_context.failure_pattern
    
    # Level 1: Simplify chart type
    if pattern in [FailurePattern.RESOURCE_EXHAUSTION, FailurePattern.TIMEOUT]:
        try:
            # Try with simplified parameters
            simplified_kwargs = kwargs.copy()
            simplified_kwargs['complexity'] = 'low'
            simplified_kwargs['animation'] = False
            simplified_kwargs['max_points'] = 50
            
            result = await func(*args, **simplified_kwargs)
            if _is_valid_visualization_result(result):
                result['degraded'] = True
                result['degradation_level'] = 'simplified'
                result['message'] = 'Showing simplified visualization for better performance'
                return result
        except Exception:
            pass
    
    # Level 2: Use basic chart type
    try:
        basic_data = _extract_data_from_args(args, kwargs)
        if basic_data:
            return {
                "success": True,
                "chart_type": "bar" if fallback_chart_type == "table" else fallback_chart_type,
                "data": basic_data[:20],  # Limit to 20 points
                "title": f"Simplified {func.__name__.replace('_', ' ').title()}",
                "degraded": True,
                "degradation_level": "basic_chart",
                "message": "Showing basic chart due to system constraints",
                "config": {
                    "width": 400,
                    "height": 300,
                    "responsive": True,
                    "animation": False
                }
            }
    except Exception:
        pass
    
    return None


async def _generate_context_aware_visualization(func: Callable, args: tuple, kwargs: dict,
                                              error: Exception, fallback_chart_type: str,
                                              sample_data: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Generate context-aware visualization fallback."""
    # Extract data from arguments
    extracted_data = _extract_data_from_args(args, kwargs)
    
    # Analyze data structure to determine best fallback
    if extracted_data:
        data_analysis = _analyze_data_structure(extracted_data)
        
        # Choose appropriate fallback based on data
        if data_analysis['type'] == 'time_series':
            chart_type = 'line'
            processed_data = _process_time_series_data(extracted_data)
        elif data_analysis['type'] == 'categorical':
            chart_type = 'bar'
            processed_data = _process_categorical_data(extracted_data)
        elif data_analysis['type'] == 'numerical':
            chart_type = 'scatter'
            processed_data = _process_numerical_data(extracted_data)
        else:
            chart_type = fallback_chart_type
            processed_data = extracted_data[:10]  # Limit data points
        
        return {
            "success": True,
            "chart_type": chart_type,
            "data": processed_data,
            "title": f"Fallback Visualization - {func.__name__.replace('_', ' ').title()}",
            "fallback": True,
            "context_aware": True,
            "data_analysis": data_analysis,
            "message": f"Generated {chart_type} chart based on your data structure",
            "config": {
                "width": 500,
                "height": 350,
                "responsive": True,
                "animation": False,
                "simplified": True
            },
            "error_info": {
                "original_error": str(error),
                "fallback_reason": "context_aware_generation"
            }
        }
    
    return None


def _extract_data_from_args(args: tuple, kwargs: dict) -> Optional[List[Dict[str, Any]]]:
    """Extract data from function arguments."""
    # Check kwargs first
    for key in ['data', 'dataset', 'values', 'records']:
        if key in kwargs and isinstance(kwargs[key], (list, dict)):
            data = kwargs[key]
            if isinstance(data, dict):
                # Convert dict to list of key-value pairs
                return [{'key': k, 'value': v} for k, v in data.items()]
            return data if isinstance(data, list) else []
    
    # Check args
    for arg in args:
        if isinstance(arg, list) and arg:
            # Ensure it's a list of dicts
            if all(isinstance(item, dict) for item in arg):
                return arg
            elif all(isinstance(item, (int, float)) for item in arg):
                # Convert list of numbers to chart data
                return [{'index': i, 'value': val} for i, val in enumerate(arg)]
        elif isinstance(arg, dict) and arg:
            # Convert dict to list
            return [{'key': k, 'value': v} for k, v in arg.items()]
    
    return None


def _analyze_data_structure(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze data structure to determine best visualization approach."""
    if not data:
        return {'type': 'empty', 'fields': [], 'sample_size': 0}
    
    sample_item = data[0]
    fields = list(sample_item.keys())
    
    # Check for time series data
    time_fields = ['date', 'time', 'timestamp', 'created_at', 'updated_at']
    has_time_field = any(field.lower() in time_fields for field in fields)
    
    # Check for numerical data
    numerical_fields = []
    categorical_fields = []
    
    for field in fields:
        sample_values = [item.get(field) for item in data[:5] if item.get(field) is not None]
        if sample_values:
            if all(isinstance(val, (int, float)) for val in sample_values):
                numerical_fields.append(field)
            else:
                categorical_fields.append(field)
    
    # Determine data type
    if has_time_field and numerical_fields:
        data_type = 'time_series'
    elif len(categorical_fields) > 0 and len(numerical_fields) > 0:
        data_type = 'categorical'
    elif len(numerical_fields) >= 2:
        data_type = 'numerical'
    else:
        data_type = 'mixed'
    
    return {
        'type': data_type,
        'fields': fields,
        'numerical_fields': numerical_fields,
        'categorical_fields': categorical_fields,
        'has_time_field': has_time_field,
        'sample_size': len(data)
    }


def _process_time_series_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process data for time series visualization."""
    # Limit to 50 points for performance
    limited_data = data[:50]
    
    # Ensure consistent structure
    processed = []
    for item in limited_data:
        processed_item = {}
        
        # Find time field
        for key, value in item.items():
            if key.lower() in ['date', 'time', 'timestamp', 'created_at']:
                processed_item['x'] = str(value)
                break
        
        # Find value field
        for key, value in item.items():
            if isinstance(value, (int, float)):
                processed_item['y'] = value
                break
        
        if 'x' in processed_item and 'y' in processed_item:
            processed.append(processed_item)
    
    return processed


def _process_categorical_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process data for categorical visualization."""
    # Group by category and sum values
    category_sums = {}
    
    for item in data:
        category = None
        value = 0
        
        # Find category field
        for key, val in item.items():
            if isinstance(val, str):
                category = val
                break
        
        # Find value field
        for key, val in item.items():
            if isinstance(val, (int, float)):
                value = val
                break
        
        if category:
            category_sums[category] = category_sums.get(category, 0) + value
    
    # Convert to chart format
    return [
        {'category': cat, 'value': val}
        for cat, val in list(category_sums.items())[:10]  # Limit to 10 categories
    ]


def _process_numerical_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process data for numerical visualization."""
    processed = []
    
    for item in data[:30]:  # Limit to 30 points
        numerical_values = [val for val in item.values() if isinstance(val, (int, float))]
        
        if len(numerical_values) >= 2:
            processed.append({
                'x': numerical_values[0],
                'y': numerical_values[1]
            })
    
    return processed


def _generate_ultimate_visualization_fallback(func_name: str, args: tuple, kwargs: dict,
                                            error: Exception, fallback_chart_type: str,
                                            sample_data: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Generate ultimate fallback visualization when all else fails."""
    # Use provided sample data or generate generic data
    if sample_data:
        data = sample_data
    else:
        data = [
            {"category": "Sample Data", "value": 100},
            {"category": "Fallback Mode", "value": 150},
            {"category": "System Recovery", "value": 75}
        ]
    
    return {
        "success": True,
        "chart_type": fallback_chart_type,
        "data": data,
        "title": f"Visualization Unavailable - {func_name.replace('_', ' ').title()}",
        "fallback": True,
        "ultimate_fallback": True,
        "message": "Visualization system temporarily unavailable. Showing sample data.",
        "config": {
            "width": 400,
            "height": 300,
            "responsive": True,
            "animation": False,
            "theme": "minimal"
        },
        "error_info": {
            "original_error": str(error),
            "error_type": type(error).__name__,
            "fallback_reason": "ultimate_fallback",
            "timestamp": datetime.utcnow().isoformat()
        },
        "user_actions": [
            "Try refreshing the page",
            "Reduce data complexity",
            "Contact support if issue persists"
        ]
    }


def never_fail_ai_operation(
    fallback_response: str = "I'm currently operating with limited capabilities. Please try again or contact support.",
    confidence_threshold: float = 0.3,
    context_aware: bool = True,
    intelligent_fallback: bool = True,
    response_templates: Optional[Dict[str, str]] = None
):
    """
    Enhanced decorator for AI operations that must never fail.
    Features context-aware responses, intelligent fallbacks, and response templates.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Validate and normalize result structure
                normalized_result = _normalize_ai_result(result, func.__name__)
                
                # Record success
                intelligent_retry.record_success(func.__name__)
                
                return normalized_result
                
            except Exception as e:
                logger.error(f"AI operation {func.__name__} failed: {e}", exc_info=True)
                
                # Create failure context
                failure_context = FailureContext(
                    function_name=func.__name__,
                    failure_pattern=intelligent_retry.classify_failure(e),
                    attempt_count=1,
                    total_elapsed_time=time.time() - start_time,
                    last_error=e,
                    user_context=_extract_user_context_from_args(args, kwargs)
                )
                
                # Record failure
                intelligent_retry.record_failure(failure_context)
                
                # Generate intelligent fallback response
                if intelligent_fallback:
                    intelligent_response = await _generate_intelligent_ai_fallback(
                        func, args, kwargs, failure_context, response_templates
                    )
                    if intelligent_response:
                        return intelligent_response
                
                # Context-aware fallback
                if context_aware:
                    context_response = _generate_context_aware_ai_response(
                        func.__name__, args, kwargs, e, fallback_response
                    )
                    if context_response:
                        return context_response
                
                # Ultimate fallback
                return _generate_ultimate_ai_fallback(
                    func.__name__, args, kwargs, e, fallback_response, confidence_threshold
                )
        
        return wrapper
    return decorator


def _normalize_ai_result(result: Any, func_name: str) -> Dict[str, Any]:
    """Normalize AI result to consistent structure."""
    if isinstance(result, str):
        return {
            "response": result,
            "confidence": 0.8,
            "success": True,
            "source": func_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    elif isinstance(result, dict):
        normalized = result.copy()
        
        # Ensure required fields
        if "response" not in normalized:
            normalized["response"] = str(normalized.get("content", normalized.get("text", "")))
        if "confidence" not in normalized:
            normalized["confidence"] = 0.8
        if "success" not in normalized:
            normalized["success"] = True
        
        normalized["source"] = func_name
        normalized["timestamp"] = datetime.utcnow().isoformat()
        
        return normalized
    else:
        return {
            "response": str(result),
            "confidence": 0.5,
            "success": True,
            "source": func_name,
            "timestamp": datetime.utcnow().isoformat(),
            "note": "Result converted to string"
        }


def _extract_user_context_from_args(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract user context from function arguments."""
    context = {}
    
    # Look for common context fields
    context_fields = ['user_id', 'session_id', 'query', 'prompt', 'input', 'question']
    
    for field in context_fields:
        if field in kwargs:
            context[field] = str(kwargs[field])[:200]  # Truncate for logging
    
    # Extract from positional arguments
    for i, arg in enumerate(args):
        if isinstance(arg, str) and len(arg) < 500:
            context[f'arg_{i}'] = arg[:200]
        elif isinstance(arg, dict):
            for key in context_fields:
                if key in arg:
                    context[key] = str(arg[key])[:200]
    
    return context


async def _generate_intelligent_ai_fallback(func: Callable, args: tuple, kwargs: dict,
                                          failure_context: FailureContext,
                                          response_templates: Optional[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """Generate intelligent AI fallback based on context and patterns."""
    func_name = func.__name__.lower()
    pattern = failure_context.failure_pattern
    user_context = failure_context.user_context or {}
    
    # Use response templates if provided
    if response_templates:
        template_key = None
        
        # Match template based on function name
        for key in response_templates.keys():
            if key.lower() in func_name:
                template_key = key
                break
        
        if template_key:
            template_response = response_templates[template_key]
            
            # Substitute context variables
            if 'query' in user_context:
                template_response = template_response.replace('{query}', user_context['query'])
            if 'user_id' in user_context:
                template_response = template_response.replace('{user}', user_context['user_id'])
            
            return {
                "response": template_response,
                "confidence": 0.6,
                "success": True,
                "fallback": True,
                "fallback_type": "template_based",
                "pattern": pattern.value,
                "suggestions": _get_ai_suggestions_for_pattern(pattern)
            }
    
    # Function-specific intelligent responses
    if 'chat' in func_name or 'conversation' in func_name:
        return await _generate_chat_fallback(user_context, pattern)
    elif 'analyze' in func_name or 'analysis' in func_name:
        return await _generate_analysis_fallback(user_context, pattern)
    elif 'summarize' in func_name or 'summary' in func_name:
        return await _generate_summary_fallback(user_context, pattern)
    elif 'translate' in func_name:
        return await _generate_translation_fallback(user_context, pattern)
    elif 'generate' in func_name:
        return await _generate_generation_fallback(user_context, pattern)
    
    return None


async def _generate_chat_fallback(user_context: Dict[str, Any], pattern: FailurePattern) -> Dict[str, Any]:
    """Generate fallback for chat/conversation functions."""
    query = user_context.get('query', user_context.get('input', ''))
    
    responses = {
        FailurePattern.RATE_LIMITED: f"I'm receiving a lot of requests right now. Regarding your question about '{query[:50]}...', I'd be happy to help once the system catches up.",
        FailurePattern.TIMEOUT: f"Your question about '{query[:50]}...' is quite complex and taking longer to process. Let me provide a quick response while I work on a more detailed answer.",
        FailurePattern.RESOURCE_EXHAUSTION: "I'm operating with limited resources right now, but I can still help with simpler questions or provide general guidance.",
        FailurePattern.DEPENDENCY_FAILURE: "Some of my advanced features are temporarily unavailable, but I can still assist with basic questions and provide general information."
    }
    
    response = responses.get(pattern, f"I'm having some technical difficulties, but I'm still here to help with your question about '{query[:50]}...'")
    
    return {
        "response": response,
        "confidence": 0.5,
        "success": True,
        "fallback": True,
        "fallback_type": "chat_specific",
        "pattern": pattern.value,
        "suggestions": [
            "Try asking a simpler question",
            "Break your question into smaller parts",
            "Try again in a few minutes"
        ]
    }


async def _generate_analysis_fallback(user_context: Dict[str, Any], pattern: FailurePattern) -> Dict[str, Any]:
    """Generate fallback for analysis functions."""
    return {
        "response": "I'm unable to perform the full analysis right now, but I can provide some general insights based on common patterns in similar data.",
        "confidence": 0.3,
        "success": True,
        "fallback": True,
        "fallback_type": "analysis_specific",
        "pattern": pattern.value,
        "general_insights": [
            "Consider checking data quality and completeness",
            "Look for trends and patterns in your dataset",
            "Verify data sources and collection methods"
        ],
        "suggestions": [
            "Try with a smaller dataset",
            "Check data format and structure",
            "Contact support for complex analysis needs"
        ]
    }


async def _generate_summary_fallback(user_context: Dict[str, Any], pattern: FailurePattern) -> Dict[str, Any]:
    """Generate fallback for summary functions."""
    input_text = user_context.get('input', user_context.get('text', ''))
    
    # Provide basic summary if we have input text
    if input_text and len(input_text) > 100:
        word_count = len(input_text.split())
        char_count = len(input_text)
        
        basic_summary = f"Document contains approximately {word_count} words and {char_count} characters. "
        
        # Add basic content hints
        if any(word in input_text.lower() for word in ['data', 'analysis', 'report']):
            basic_summary += "Content appears to be analytical or data-focused."
        elif any(word in input_text.lower() for word in ['meeting', 'discussion', 'agenda']):
            basic_summary += "Content appears to be meeting or discussion-related."
        
        return {
            "response": f"I can't provide a full summary right now, but here's what I can tell you: {basic_summary}",
            "confidence": 0.4,
            "success": True,
            "fallback": True,
            "fallback_type": "summary_basic",
            "word_count": word_count,
            "char_count": char_count
        }
    
    return {
        "response": "I'm unable to generate a summary at the moment. Please try again with your content, or contact support for assistance.",
        "confidence": 0.2,
        "success": True,
        "fallback": True,
        "fallback_type": "summary_unavailable"
    }


async def _generate_translation_fallback(user_context: Dict[str, Any], pattern: FailurePattern) -> Dict[str, Any]:
    """Generate fallback for translation functions."""
    return {
        "response": "Translation services are temporarily unavailable. You might try using an alternative translation service or contact support for assistance.",
        "confidence": 0.1,
        "success": True,
        "fallback": True,
        "fallback_type": "translation_unavailable",
        "alternatives": [
            "Google Translate",
            "DeepL",
            "Microsoft Translator"
        ]
    }


async def _generate_generation_fallback(user_context: Dict[str, Any], pattern: FailurePattern) -> Dict[str, Any]:
    """Generate fallback for content generation functions."""
    return {
        "response": "Content generation is temporarily limited. I can provide templates, examples, or guidance instead of generating new content.",
        "confidence": 0.3,
        "success": True,
        "fallback": True,
        "fallback_type": "generation_limited",
        "alternatives": [
            "I can provide templates for your content type",
            "I can suggest structure and key points",
            "I can review and improve existing content"
        ]
    }


def _generate_context_aware_ai_response(func_name: str, args: tuple, kwargs: dict,
                                      error: Exception, fallback_response: str) -> Dict[str, Any]:
    """Generate context-aware AI response."""
    # Extract context from arguments
    context_info = []
    user_input = None
    
    for arg in args:
        if isinstance(arg, str) and len(arg) < 200:
            user_input = arg
            context_info.append(f"regarding '{arg[:50]}...'")
            break
    
    # Look for context in kwargs
    for key in ['query', 'prompt', 'input', 'question']:
        if key in kwargs and isinstance(kwargs[key], str):
            user_input = kwargs[key]
            context_info.append(f"about your {key}")
            break
    
    # Generate contextual response
    if context_info:
        contextual_response = f"{fallback_response} {context_info[0]}"
    else:
        contextual_response = fallback_response
    
    return {
        "response": contextual_response,
        "confidence": 0.4,
        "success": True,
        "fallback": True,
        "fallback_type": "context_aware",
        "user_input": user_input[:100] if user_input else None,
        "error_type": type(error).__name__,
        "suggestions": [
            "Try rephrasing your request",
            "Break complex requests into simpler parts",
            "Check your internet connection",
            "Contact support if the issue persists"
        ]
    }


def _generate_ultimate_ai_fallback(func_name: str, args: tuple, kwargs: dict,
                                 error: Exception, fallback_response: str,
                                 confidence_threshold: float) -> Dict[str, Any]:
    """Generate ultimate AI fallback when all else fails."""
    return {
        "response": fallback_response,
        "confidence": confidence_threshold,
        "success": True,
        "fallback": True,
        "fallback_type": "ultimate",
        "error_handled": True,
        "function": func_name,
        "error_type": type(error).__name__,
        "timestamp": datetime.utcnow().isoformat(),
        "suggestions": [
            "Try rephrasing your request",
            "Check your internet connection",
            "Try again in a few minutes",
            "Contact support if the issue persists"
        ],
        "support_info": {
            "error_id": hashlib.md5(f"{func_name}_{str(error)}_{time.time()}".encode()).hexdigest()[:8],
            "function": func_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    }


def _get_ai_suggestions_for_pattern(pattern: FailurePattern) -> List[str]:
    """Get AI-specific suggestions based on failure pattern."""
    suggestions = {
        FailurePattern.RATE_LIMITED: [
            "Wait a moment before making another request",
            "Try breaking your request into smaller parts",
            "Consider upgrading for higher rate limits"
        ],
        FailurePattern.TIMEOUT: [
            "Try asking a simpler question",
            "Break complex requests into parts",
            "Reduce the amount of text being processed"
        ],
        FailurePattern.RESOURCE_EXHAUSTION: [
            "Try during off-peak hours",
            "Simplify your request",
            "Use shorter input text"
        ],
        FailurePattern.DEPENDENCY_FAILURE: [
            "Try again in a few minutes",
            "Check our status page for updates",
            "Use basic features while we restore full service"
        ]
    }
    
    return suggestions.get(pattern, [
        "Try again in a few minutes",
        "Contact support if the issue persists",
        "Check your internet connection"
    ])


def never_fail_file_operation(
    fallback_content: str = "",
    create_if_missing: bool = True
):
    """
    Decorator for file operations that must never fail.
    Creates files if missing, returns fallback content on read errors.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
                
            except FileNotFoundError as e:
                if create_if_missing and "write" in func.__name__.lower():
                    # Try to create the file
                    try:
                        import os
                        file_path = args[0] if args else kwargs.get("path", "")
                        if file_path:
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                            return await func(*args, **kwargs)
                    except Exception:
                        pass
                
                logger.warning(f"File operation {func.__name__} failed - file not found: {e}")
                
                if "read" in func.__name__.lower():
                    return fallback_content
                else:
                    return {"success": False, "error": "File not found", "created": False}
                
            except PermissionError as e:
                logger.error(f"File operation {func.__name__} failed - permission denied: {e}")
                return {"success": False, "error": "Permission denied", "suggestion": "Check file permissions"}
                
            except Exception as e:
                logger.error(f"File operation {func.__name__} failed: {e}", exc_info=True)
                
                if "read" in func.__name__.lower():
                    return fallback_content
                else:
                    return {"success": False, "error": "File operation failed", "retry_possible": True}
        
        return wrapper
    return decorator


def never_fail_database_operation(
    fallback_data: Any = None,
    use_cache: bool = True,
    cache_duration: int = 300
):
    """
    Decorator for database operations that must never fail.
    Uses cache or returns fallback data if database fails.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"db_{func.__name__}_{hash(str(args) + str(kwargs))}" if use_cache else None
            
            try:
                # Try cache first for read operations
                if cache_key and "get" in func.__name__.lower():
                    cached_result = await _get_from_cache(cache_key)
                    if cached_result is not None:
                        return cached_result
                
                # Execute database operation
                result = await func(*args, **kwargs)
                
                # Cache successful result
                if cache_key and result is not None:
                    await _store_in_cache(cache_key, result, duration=cache_duration)
                
                return result
                
            except Exception as e:
                logger.error(f"Database operation {func.__name__} failed: {e}", exc_info=True)
                
                # Try stale cache
                if cache_key:
                    stale_result = await _get_from_cache(cache_key, allow_stale=True)
                    if stale_result is not None:
                        logger.info(f"Using stale cache for {func.__name__}")
                        return stale_result
                
                # Return fallback data
                if fallback_data is not None:
                    return fallback_data
                
                # Return appropriate empty structure
                if "list" in func.__name__.lower() or "all" in func.__name__.lower():
                    return []
                elif "count" in func.__name__.lower():
                    return 0
                elif "exists" in func.__name__.lower():
                    return False
                else:
                    return None
        
        return wrapper
    return decorator


# Enhanced cache implementation with metadata and intelligent management
_cache: Dict[str, Dict[str, Any]] = {}
_cache_stats = {
    'hits': 0,
    'misses': 0,
    'evictions': 0,
    'stale_hits': 0
}

async def _get_from_cache(key: str, allow_stale: bool = False) -> Any:
    """Get data from cache with enhanced metadata handling."""
    if key not in _cache:
        _cache_stats['misses'] += 1
        return None
    
    cache_entry = _cache[key]
    current_time = time.time()
    
    # Check if expired
    if current_time > cache_entry["expires"]:
        if allow_stale:
            # Return stale data with metadata
            _cache_stats['stale_hits'] += 1
            data = cache_entry["data"]
            
            # Add stale metadata if it's a dict
            if isinstance(data, dict):
                data = data.copy()
                data['_cache_stale'] = True
                data['_cache_age'] = current_time - cache_entry["created"]
                data['_cache_expired_by'] = current_time - cache_entry["expires"]
            
            return data
        else:
            # Remove expired entry
            del _cache[key]
            _cache_stats['evictions'] += 1
            _cache_stats['misses'] += 1
            return None
    
    # Return fresh data
    _cache_stats['hits'] += 1
    data = cache_entry["data"]
    
    # Add cache metadata if it's a dict
    if isinstance(data, dict):
        data = data.copy()
        data['_cache_fresh'] = True
        data['_cache_age'] = current_time - cache_entry["created"]
        data['_cache_ttl'] = cache_entry["expires"] - current_time
    
    # Update access time for LRU
    cache_entry["last_accessed"] = current_time
    cache_entry["access_count"] = cache_entry.get("access_count", 0) + 1
    
    return data

async def _store_in_cache(key: str, data: Any, duration: int = 300):
    """Store data in cache with enhanced metadata."""
    current_time = time.time()
    
    _cache[key] = {
        "data": data,
        "expires": current_time + duration,
        "created": current_time,
        "last_accessed": current_time,
        "access_count": 1,
        "size": _estimate_size(data),
        "duration": duration
    }
    
    # Intelligent cache cleanup
    await _cleanup_cache()

async def _cleanup_cache():
    """Intelligent cache cleanup based on size, age, and access patterns."""
    current_time = time.time()
    
    # Clean up expired entries first
    expired_keys = [
        k for k, v in _cache.items() 
        if current_time > v["expires"]
    ]
    
    for k in expired_keys:
        del _cache[k]
        _cache_stats['evictions'] += 1
    
    # If still too many entries, use LRU eviction
    max_cache_size = 1000
    if len(_cache) > max_cache_size:
        # Sort by last accessed time (LRU)
        sorted_entries = sorted(
            _cache.items(),
            key=lambda x: (x[1]["last_accessed"], x[1]["access_count"])
        )
        
        # Remove least recently used entries
        entries_to_remove = len(_cache) - max_cache_size + 100  # Remove extra for buffer
        
        for i in range(min(entries_to_remove, len(sorted_entries))):
            key = sorted_entries[i][0]
            del _cache[key]
            _cache_stats['evictions'] += 1

def _estimate_size(data: Any) -> int:
    """Estimate the size of cached data."""
    try:
        if isinstance(data, str):
            return len(data)
        elif isinstance(data, (list, dict)):
            return len(str(data))
        else:
            return len(str(data))
    except:
        return 100  # Default estimate

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring."""
    total_requests = _cache_stats['hits'] + _cache_stats['misses']
    hit_rate = _cache_stats['hits'] / total_requests if total_requests > 0 else 0
    
    return {
        'total_entries': len(_cache),
        'total_requests': total_requests,
        'hits': _cache_stats['hits'],
        'misses': _cache_stats['misses'],
        'stale_hits': _cache_stats['stale_hits'],
        'evictions': _cache_stats['evictions'],
        'hit_rate': hit_rate,
        'cache_size_mb': sum(entry.get('size', 0) for entry in _cache.values()) / (1024 * 1024),
        'oldest_entry_age': max(
            (time.time() - entry['created'] for entry in _cache.values()),
            default=0
        )
    }

async def clear_cache(pattern: Optional[str] = None):
    """Clear cache entries, optionally matching a pattern."""
    if pattern:
        keys_to_remove = [k for k in _cache.keys() if pattern in k]
        for k in keys_to_remove:
            del _cache[k]
    else:
        _cache.clear()
        
    # Reset stats
    _cache_stats.update({
        'hits': 0,
        'misses': 0,
        'evictions': 0,
        'stale_hits': 0
    })


# Enhanced convenience decorators utilizing advanced patterns
def bulletproof_endpoint(fallback_response: Optional[Dict[str, Any]] = None, 
                        max_timeout: float = 30.0):
    """Make an API endpoint completely bulletproof with advanced features."""
    def decorator(func: Callable) -> Callable:
        return never_fail_api_endpoint(
            fallback_response=fallback_response,
            context_aware=True,
            progressive_timeout=True,
            max_timeout=max_timeout
        )(critical_operation()(func))
    return decorator


def safe_data_fetch(fallback_data: Any = None, cache_duration: int = 300):
    """Make a data fetching function safe with intelligent caching and fallbacks."""
    def decorator(func: Callable) -> Callable:
        return never_fail_data_operation(
            fallback_data=fallback_data,
            context_aware=True,
            auto_generate_fallback=True,
            cache_duration=cache_duration,
            stale_cache_acceptable=True
        )(bulletproof()(func))
    return decorator


def resilient_ai_call(fallback_response: str = "AI service temporarily unavailable",
                     response_templates: Optional[Dict[str, str]] = None):
    """Make an AI call resilient with intelligent fallbacks."""
    def decorator(func: Callable) -> Callable:
        return never_fail_ai_operation(
            fallback_response=fallback_response,
            context_aware=True,
            intelligent_fallback=True,
            response_templates=response_templates
        )(with_graceful_degradation("ai_services")(func))
    return decorator


def smart_visualization(fallback_chart_type: str = "table"):
    """Make visualization functions smart with progressive degradation."""
    def decorator(func: Callable) -> Callable:
        return never_fail_visualization(
            fallback_chart_type=fallback_chart_type,
            context_aware=True,
            progressive_degradation=True,
            auto_simplify=True
        )(with_graceful_degradation("visualization")(func))
    return decorator


def adaptive_operation(service_name: str = "default", 
                      fallback_data: Any = None,
                      max_timeout: float = 60.0):
    """Make any operation adaptive with full bulletproof protection."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = kwargs.get('user_id')
            
            # Use progressive timeout
            async with timeout_handler.progressive_timeout(func.__name__, user_id, max_timeout):
                # Execute with intelligent retry and context-aware fallbacks
                try:
                    return await _execute_with_intelligent_retry(
                        func, args, kwargs, service_name, context_aware=True
                    )
                except Exception as e:
                    # Generate context-aware fallback
                    fallback = fallback_generator.generate_fallback_data(
                        func, args, kwargs, e, {'service': service_name}
                    )
                    
                    if fallback is not None:
                        return fallback
                    elif fallback_data is not None:
                        return fallback_data
                    else:
                        # Re-raise if no fallback available
                        raise e
        
        return wrapper
    return decorator


def context_aware_cache(cache_duration: int = 300, 
                       generate_fallback: bool = True):
    """Add context-aware caching with intelligent fallback generation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Try cache first
            cached_result = await _get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                await _store_in_cache(cache_key, result, cache_duration)
                
                return result
                
            except Exception as e:
                # Try stale cache
                stale_result = await _get_from_cache(cache_key, allow_stale=True)
                if stale_result is not None:
                    return stale_result
                
                # Generate fallback if enabled
                if generate_fallback:
                    fallback = fallback_generator.generate_fallback_data(
                        func, args, kwargs, e, {'cache_key': cache_key}
                    )
                    if fallback is not None:
                        return fallback
                
                # Re-raise if no fallback
                raise e
        
        return wrapper
    return decorator


def progressive_timeout_operation(max_timeout: float = 60.0):
    """Add progressive timeout handling with user feedback."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = kwargs.get('user_id', 'anonymous')
            
            async with timeout_handler.progressive_timeout(func.__name__, user_id, max_timeout):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Utility functions for monitoring and debugging
def get_never_fail_stats() -> Dict[str, Any]:
    """Get comprehensive statistics about never-fail decorators."""
    return {
        'cache_stats': get_cache_stats(),
        'retry_stats': {
            'total_functions': len(intelligent_retry.failure_history),
            'circuit_breakers': {
                name: {
                    'state': cb['state'],
                    'failure_count': cb['failure_count']
                }
                for name, cb in intelligent_retry.circuit_breakers.items()
            },
            'recent_failures': sum(
                len(failures) for failures in intelligent_retry.failure_history.values()
            )
        },
        'fallback_stats': {
            'usage_patterns': len(fallback_generator.usage_patterns),
            'template_count': len(fallback_generator.fallback_templates)
        }
    }


async def reset_never_fail_state():
    """Reset all never-fail decorator state (useful for testing)."""
    await clear_cache()
    intelligent_retry.failure_history.clear()
    intelligent_retry.circuit_breakers.clear()
    fallback_generator.usage_patterns.clear()
    
    logger.info("Never-fail decorator state reset")