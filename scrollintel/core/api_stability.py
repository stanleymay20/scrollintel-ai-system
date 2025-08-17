"""
API Stability and Performance Optimization System
Handles rate limiting, validation, error handling, and performance monitoring
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
from functools import wraps
import redis
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import traceback
from datetime import datetime, timedelta
import psutil

logger = logging.getLogger(__name__)

class RateLimitType(Enum):
    PER_SECOND = "per_second"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RateLimitConfig:
    requests_per_second: int = 10
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_allowance: int = 20

@dataclass
class APIMetrics:
    endpoint: str
    method: str
    response_time: float
    status_code: int
    timestamp: datetime
    user_id: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class ErrorReport:
    error_id: str
    endpoint: str
    method: str
    error_type: str
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    timestamp: datetime
    user_id: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None

class APIStabilitySystem:
    """Comprehensive API stability and performance system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.rate_limiter = RateLimiter(config.get('rate_limiting', {}))
        self.validator = RequestValidator()
        self.error_handler = ErrorHandler()
        self.performance_monitor = PerformanceMonitor()
        self.circuit_breaker = CircuitBreaker()
        
    async def initialize(self):
        """Initialize API stability system"""
        try:
            # Initialize Redis for rate limiting and caching
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                decode_responses=True,
                max_connections=20
            )
            
            # Initialize components
            await self.rate_limiter.initialize(self.redis_client)
            await self.performance_monitor.initialize()
            
            logger.info("API stability system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize API stability system: {e}")
            raise
    
    async def process_request(self, request: Request, call_next: Callable) -> Response:
        """Process incoming request with all stability checks"""
        start_time = time.time()
        
        try:
            # Extract request info
            endpoint = str(request.url.path)
            method = request.method
            user_id = self._extract_user_id(request)
            
            # Rate limiting check
            rate_limit_result = await self.rate_limiter.check_rate_limit(
                user_id or self._get_client_ip(request), endpoint, method
            )
            
            if not rate_limit_result['allowed']:
                return JSONResponse(
                    status_code=429,
                    content={
                        'error': 'Rate limit exceeded',
                        'retry_after': rate_limit_result['retry_after']
                    }
                )
            
            # Circuit breaker check
            if not await self.circuit_breaker.is_available(endpoint):
                return JSONResponse(
                    status_code=503,
                    content={'error': 'Service temporarily unavailable'}
                )
            
            # Request validation
            validation_result = await self.validator.validate_request(request)
            if not validation_result['valid']:
                return JSONResponse(
                    status_code=400,
                    content={'error': validation_result['error']}
                )
            
            # Process request
            response = await call_next(request)
            
            # Record success metrics
            response_time = time.time() - start_time
            await self.performance_monitor.record_request(
                endpoint, method, response_time, response.status_code, user_id
            )
            
            # Update circuit breaker
            await self.circuit_breaker.record_success(endpoint)
            
            return response
            
        except Exception as e:
            # Handle errors
            response_time = time.time() - start_time
            error_response = await self.error_handler.handle_error(
                e, request, response_time
            )
            
            # Update circuit breaker
            await self.circuit_breaker.record_failure(endpoint)
            
            return error_response
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        # Implementation would extract from JWT token or session
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            # Decode JWT and extract user_id
            pass
        return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        return request.client.host if request.client else 'unknown'
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'rate_limiter': await self.rate_limiter.get_status(),
            'performance': await self.performance_monitor.get_status(),
            'circuit_breaker': await self.circuit_breaker.get_status(),
            'error_handler': await self.error_handler.get_status(),
            'timestamp': datetime.utcnow().isoformat()
        }

class RateLimiter:
    """Advanced rate limiting with multiple time windows"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.default_limits = RateLimitConfig(**config.get('default_limits', {}))
        self.endpoint_limits = config.get('endpoint_limits', {})
        self.redis_client = None
        
    async def initialize(self, redis_client):
        """Initialize rate limiter"""
        self.redis_client = redis_client
    
    async def check_rate_limit(self, identifier: str, endpoint: str, method: str) -> Dict[str, Any]:
        """Check if request is within rate limits"""
        try:
            # Get limits for this endpoint
            limits = self._get_limits_for_endpoint(endpoint, method)
            
            # Check each time window
            for limit_type, limit_value in [
                (RateLimitType.PER_SECOND, limits.requests_per_second),
                (RateLimitType.PER_MINUTE, limits.requests_per_minute),
                (RateLimitType.PER_HOUR, limits.requests_per_hour),
                (RateLimitType.PER_DAY, limits.requests_per_day)
            ]:
                if not await self._check_window_limit(identifier, endpoint, limit_type, limit_value):
                    retry_after = self._calculate_retry_after(limit_type)
                    return {
                        'allowed': False,
                        'limit_type': limit_type.value,
                        'retry_after': retry_after
                    }
            
            # Record the request
            await self._record_request(identifier, endpoint)
            
            return {'allowed': True}
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fail open - allow request if rate limiting fails
            return {'allowed': True}
    
    async def _check_window_limit(self, identifier: str, endpoint: str, 
                                limit_type: RateLimitType, limit_value: int) -> bool:
        """Check rate limit for specific time window"""
        window_size = self._get_window_size(limit_type)
        current_time = int(time.time())
        window_start = current_time - window_size
        
        key = f"rate_limit:{identifier}:{endpoint}:{limit_type.value}"
        
        # Use Redis sorted set to track requests in time window
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(key, window_size + 60)
        
        results = pipe.execute()
        current_count = results[1]
        
        return current_count < limit_value
    
    async def _record_request(self, identifier: str, endpoint: str):
        """Record request for rate limiting"""
        current_time = int(time.time())
        
        for limit_type in RateLimitType:
            key = f"rate_limit:{identifier}:{endpoint}:{limit_type.value}"
            self.redis_client.zadd(key, {str(current_time): current_time})
    
    def _get_limits_for_endpoint(self, endpoint: str, method: str) -> RateLimitConfig:
        """Get rate limits for specific endpoint"""
        endpoint_key = f"{method}:{endpoint}"
        if endpoint_key in self.endpoint_limits:
            return RateLimitConfig(**self.endpoint_limits[endpoint_key])
        return self.default_limits
    
    def _get_window_size(self, limit_type: RateLimitType) -> int:
        """Get window size in seconds for limit type"""
        return {
            RateLimitType.PER_SECOND: 1,
            RateLimitType.PER_MINUTE: 60,
            RateLimitType.PER_HOUR: 3600,
            RateLimitType.PER_DAY: 86400
        }[limit_type]
    
    def _calculate_retry_after(self, limit_type: RateLimitType) -> int:
        """Calculate retry after seconds"""
        return self._get_window_size(limit_type)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status"""
        return {
            'default_limits': asdict(self.default_limits),
            'endpoint_limits_count': len(self.endpoint_limits),
            'redis_connected': self.redis_client is not None
        }

class RequestValidator:
    """Request validation and sanitization"""
    
    def __init__(self):
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.required_headers = ['Content-Type']
        
    async def validate_request(self, request: Request) -> Dict[str, Any]:
        """Validate incoming request"""
        try:
            # Check request size
            content_length = request.headers.get('content-length')
            if content_length and int(content_length) > self.max_request_size:
                return {
                    'valid': False,
                    'error': 'Request too large'
                }
            
            # Validate content type for POST/PUT requests
            if request.method in ['POST', 'PUT', 'PATCH']:
                content_type = request.headers.get('content-type', '')
                if not content_type.startswith(('application/json', 'multipart/form-data')):
                    return {
                        'valid': False,
                        'error': 'Invalid content type'
                    }
            
            # Check for suspicious patterns
            url_path = str(request.url.path)
            if self._contains_suspicious_patterns(url_path):
                return {
                    'valid': False,
                    'error': 'Invalid request pattern'
                }
            
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"Request validation error: {e}")
            return {
                'valid': False,
                'error': 'Validation failed'
            }
    
    def _contains_suspicious_patterns(self, path: str) -> bool:
        """Check for suspicious patterns in request path"""
        suspicious_patterns = [
            '../', '..\\', '<script', 'javascript:', 'vbscript:',
            'onload=', 'onerror=', 'eval(', 'exec('
        ]
        
        path_lower = path.lower()
        return any(pattern in path_lower for pattern in suspicious_patterns)

class ErrorHandler:
    """Comprehensive error handling and reporting"""
    
    def __init__(self):
        self.error_counts = {}
        self.recent_errors = []
        
    async def handle_error(self, error: Exception, request: Request, 
                         response_time: float) -> JSONResponse:
        """Handle and report errors"""
        try:
            # Generate error ID
            error_id = self._generate_error_id(error, request)
            
            # Determine error severity
            severity = self._determine_severity(error)
            
            # Create error report
            error_report = ErrorReport(
                error_id=error_id,
                endpoint=str(request.url.path),
                method=request.method,
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                severity=severity,
                timestamp=datetime.utcnow(),
                user_id=self._extract_user_id(request)
            )
            
            # Store error report
            await self._store_error_report(error_report)
            
            # Update error counts
            self._update_error_counts(error_report)
            
            # Log error
            logger.error(f"API Error [{error_id}]: {error}", exc_info=True)
            
            # Return appropriate response
            return self._create_error_response(error, error_id, severity)
            
        except Exception as e:
            logger.error(f"Error handler failed: {e}")
            return JSONResponse(
                status_code=500,
                content={'error': 'Internal server error'}
            )
    
    def _generate_error_id(self, error: Exception, request: Request) -> str:
        """Generate unique error ID"""
        error_string = f"{type(error).__name__}:{str(error)}:{request.url.path}:{time.time()}"
        return hashlib.md5(error_string.encode()).hexdigest()[:12]
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity"""
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, HTTPException):
            if error.status_code >= 500:
                return ErrorSeverity.HIGH
            else:
                return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        # Implementation would extract from JWT token or session
        return None
    
    async def _store_error_report(self, error_report: ErrorReport):
        """Store error report for analysis"""
        self.recent_errors.append(error_report)
        
        # Keep only recent errors (last 1000)
        if len(self.recent_errors) > 1000:
            self.recent_errors = self.recent_errors[-1000:]
    
    def _update_error_counts(self, error_report: ErrorReport):
        """Update error statistics"""
        error_key = f"{error_report.endpoint}:{error_report.error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def _create_error_response(self, error: Exception, error_id: str, 
                             severity: ErrorSeverity) -> JSONResponse:
        """Create appropriate error response"""
        if isinstance(error, HTTPException):
            status_code = error.status_code
            message = error.detail
        elif severity == ErrorSeverity.CRITICAL:
            status_code = 500
            message = "Critical system error"
        else:
            status_code = 500
            message = "Internal server error"
        
        return JSONResponse(
            status_code=status_code,
            content={
                'error': message,
                'error_id': error_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get error handler status"""
        total_errors = len(self.recent_errors)
        recent_errors = [
            error for error in self.recent_errors
            if error.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        return {
            'total_errors_tracked': total_errors,
            'errors_last_hour': len(recent_errors),
            'top_error_types': self._get_top_error_types(),
            'error_rate_trend': self._calculate_error_rate_trend()
        }
    
    def _get_top_error_types(self) -> List[Dict[str, Any]]:
        """Get most common error types"""
        sorted_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'error_type': error_type, 'count': count}
            for error_type, count in sorted_errors[:10]
        ]
    
    def _calculate_error_rate_trend(self) -> str:
        """Calculate error rate trend"""
        # Simple implementation - would be more sophisticated in production
        recent_errors = [
            error for error in self.recent_errors
            if error.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        if len(recent_errors) > 10:
            return "increasing"
        elif len(recent_errors) < 5:
            return "decreasing"
        else:
            return "stable"

class PerformanceMonitor:
    """API performance monitoring and optimization"""
    
    def __init__(self):
        self.metrics = []
        self.endpoint_stats = {}
        
    async def initialize(self):
        """Initialize performance monitor"""
        # Start background tasks
        asyncio.create_task(self._cleanup_old_metrics())
    
    async def record_request(self, endpoint: str, method: str, response_time: float,
                           status_code: int, user_id: Optional[str] = None):
        """Record request metrics"""
        metric = APIMetrics(
            endpoint=endpoint,
            method=method,
            response_time=response_time,
            status_code=status_code,
            timestamp=datetime.utcnow(),
            user_id=user_id
        )
        
        self.metrics.append(metric)
        self._update_endpoint_stats(endpoint, method, response_time, status_code)
    
    def _update_endpoint_stats(self, endpoint: str, method: str, 
                             response_time: float, status_code: int):
        """Update endpoint statistics"""
        key = f"{method}:{endpoint}"
        
        if key not in self.endpoint_stats:
            self.endpoint_stats[key] = {
                'total_requests': 0,
                'total_response_time': 0.0,
                'error_count': 0,
                'min_response_time': float('inf'),
                'max_response_time': 0.0
            }
        
        stats = self.endpoint_stats[key]
        stats['total_requests'] += 1
        stats['total_response_time'] += response_time
        stats['min_response_time'] = min(stats['min_response_time'], response_time)
        stats['max_response_time'] = max(stats['max_response_time'], response_time)
        
        if status_code >= 400:
            stats['error_count'] += 1
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics periodically"""
        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.metrics = [
                    metric for metric in self.metrics
                    if metric.timestamp > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get performance monitoring status"""
        if not self.metrics:
            return {
                'total_requests': 0,
                'average_response_time': 0.0,
                'error_rate': 0.0,
                'slowest_endpoints': []
            }
        
        total_requests = len(self.metrics)
        total_response_time = sum(metric.response_time for metric in self.metrics)
        error_count = sum(1 for metric in self.metrics if metric.status_code >= 400)
        
        # Calculate slowest endpoints
        slowest_endpoints = []
        for key, stats in self.endpoint_stats.items():
            if stats['total_requests'] > 0:
                avg_response_time = stats['total_response_time'] / stats['total_requests']
                slowest_endpoints.append({
                    'endpoint': key,
                    'avg_response_time': avg_response_time,
                    'total_requests': stats['total_requests']
                })
        
        slowest_endpoints.sort(key=lambda x: x['avg_response_time'], reverse=True)
        
        return {
            'total_requests': total_requests,
            'average_response_time': total_response_time / total_requests,
            'error_rate': error_count / total_requests * 100,
            'slowest_endpoints': slowest_endpoints[:10]
        }

class CircuitBreaker:
    """Circuit breaker pattern for service resilience"""
    
    def __init__(self):
        self.failure_threshold = 5
        self.recovery_timeout = 60
        self.circuit_states = {}  # endpoint -> state info
    
    async def is_available(self, endpoint: str) -> bool:
        """Check if endpoint is available"""
        state = self.circuit_states.get(endpoint, {
            'state': 'closed',
            'failure_count': 0,
            'last_failure_time': None
        })
        
        if state['state'] == 'closed':
            return True
        elif state['state'] == 'open':
            # Check if recovery timeout has passed
            if (time.time() - state['last_failure_time']) > self.recovery_timeout:
                state['state'] = 'half_open'
                self.circuit_states[endpoint] = state
                return True
            return False
        elif state['state'] == 'half_open':
            return True
        
        return False
    
    async def record_success(self, endpoint: str):
        """Record successful request"""
        if endpoint in self.circuit_states:
            state = self.circuit_states[endpoint]
            if state['state'] == 'half_open':
                # Reset to closed state
                state['state'] = 'closed'
                state['failure_count'] = 0
                state['last_failure_time'] = None
            elif state['state'] == 'closed':
                # Reset failure count
                state['failure_count'] = 0
    
    async def record_failure(self, endpoint: str):
        """Record failed request"""
        state = self.circuit_states.get(endpoint, {
            'state': 'closed',
            'failure_count': 0,
            'last_failure_time': None
        })
        
        state['failure_count'] += 1
        state['last_failure_time'] = time.time()
        
        if state['failure_count'] >= self.failure_threshold:
            state['state'] = 'open'
        
        self.circuit_states[endpoint] = state
    
    async def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'monitored_endpoints': len(self.circuit_states),
            'open_circuits': sum(1 for state in self.circuit_states.values() 
                               if state['state'] == 'open'),
            'half_open_circuits': sum(1 for state in self.circuit_states.values() 
                                    if state['state'] == 'half_open'),
            'circuit_states': dict(self.circuit_states)
        }

# Global API stability system instance
api_stability_system = None

async def initialize_api_stability_system(config: Dict[str, Any]) -> APIStabilitySystem:
    """Initialize global API stability system"""
    global api_stability_system
    api_stability_system = APIStabilitySystem(config)
    await api_stability_system.initialize()
    return api_stability_system

def get_api_stability_system() -> APIStabilitySystem:
    """Get global API stability system instance"""
    if api_stability_system is None:
        raise RuntimeError("API stability system not initialized")
    return api_stability_system