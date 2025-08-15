"""
Performance Monitoring Middleware for ScrollIntel.
Tracks request performance, database queries, and system metrics.
"""

import time
import logging
import asyncio
from typing import Callable, Dict, Any, Optional
from datetime import datetime
import json
import psutil
import os

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ...core.database_pool import get_optimized_db_pool
from ...core.config import get_settings

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Store and manage performance metrics."""
    
    def __init__(self):
        self.request_metrics: Dict[str, Any] = {}
        self.system_metrics: Dict[str, Any] = {}
        self.database_metrics: Dict[str, Any] = {}
        self.slow_requests: list = []
        self.error_requests: list = []
        
        # Thresholds
        self.slow_request_threshold = 2.0  # seconds
        self.max_stored_metrics = 1000
    
    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        request_size: int = 0,
        response_size: int = 0,
        user_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """Record request metrics."""
        
        timestamp = datetime.utcnow()
        
        # Create request record
        request_record = {
            'timestamp': timestamp.isoformat(),
            'method': method,
            'path': path,
            'status_code': status_code,
            'duration': duration,
            'request_size': request_size,
            'response_size': response_size,
            'user_id': user_id,
            'error': error
        }
        
        # Store in appropriate collections
        endpoint_key = f"{method}:{path}"
        
        if endpoint_key not in self.request_metrics:
            self.request_metrics[endpoint_key] = {
                'count': 0,
                'total_duration': 0.0,
                'avg_duration': 0.0,
                'min_duration': float('inf'),
                'max_duration': 0.0,
                'error_count': 0,
                'last_request': None,
                'status_codes': {}
            }
        
        metrics = self.request_metrics[endpoint_key]
        metrics['count'] += 1
        metrics['total_duration'] += duration
        metrics['avg_duration'] = metrics['total_duration'] / metrics['count']
        metrics['min_duration'] = min(metrics['min_duration'], duration)
        metrics['max_duration'] = max(metrics['max_duration'], duration)
        metrics['last_request'] = timestamp.isoformat()
        
        # Track status codes
        if status_code not in metrics['status_codes']:
            metrics['status_codes'][status_code] = 0
        metrics['status_codes'][status_code] += 1
        
        # Track errors
        if status_code >= 400:
            metrics['error_count'] += 1
            self.error_requests.append(request_record)
        
        # Track slow requests
        if duration > self.slow_request_threshold:
            self.slow_requests.append(request_record)
        
        # Limit stored records
        if len(self.slow_requests) > self.max_stored_metrics:
            self.slow_requests = self.slow_requests[-self.max_stored_metrics:]
        
        if len(self.error_requests) > self.max_stored_metrics:
            self.error_requests = self.error_requests[-self.max_stored_metrics:]
    
    def record_system_metrics(self) -> None:
        """Record current system metrics."""
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Process metrics
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            
            self.system_metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                },
                'process': {
                    'memory_rss': process_memory.rss,
                    'memory_vms': process_memory.vms,
                    'cpu_percent': process.cpu_percent(),
                    'num_threads': process.num_threads(),
                    'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    async def record_database_metrics(self) -> None:
        """Record database performance metrics."""
        
        try:
            db_pool = await get_optimized_db_pool()
            pool_stats = db_pool.get_pool_status()
            performance_stats = db_pool.get_performance_stats()
            
            self.database_metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'pool_stats': pool_stats,
                'performance_stats': performance_stats
            }
            
        except Exception as e:
            logger.warning(f"Failed to collect database metrics: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        
        total_requests = sum(metrics['count'] for metrics in self.request_metrics.values())
        total_errors = sum(metrics['error_count'] for metrics in self.request_metrics.values())
        
        # Calculate average response time across all endpoints
        if total_requests > 0:
            total_duration = sum(metrics['total_duration'] for metrics in self.request_metrics.values())
            avg_response_time = total_duration / total_requests
        else:
            avg_response_time = 0.0
        
        return {
            'summary': {
                'total_requests': total_requests,
                'total_errors': total_errors,
                'error_rate': (total_errors / total_requests * 100) if total_requests > 0 else 0.0,
                'avg_response_time': avg_response_time,
                'slow_requests': len(self.slow_requests),
                'endpoints_tracked': len(self.request_metrics)
            },
            'system_metrics': self.system_metrics,
            'database_metrics': self.database_metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_slow_requests(self, limit: int = 10) -> list:
        """Get slowest requests."""
        return sorted(
            self.slow_requests,
            key=lambda x: x['duration'],
            reverse=True
        )[:limit]
    
    def get_error_requests(self, limit: int = 10) -> list:
        """Get recent error requests."""
        return self.error_requests[-limit:] if self.error_requests else []
    
    def get_endpoint_metrics(self, limit: int = 20) -> Dict[str, Any]:
        """Get metrics for top endpoints."""
        
        # Sort endpoints by request count
        sorted_endpoints = sorted(
            self.request_metrics.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        return dict(sorted_endpoints[:limit])


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor request performance and system metrics."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
        self.metrics = PerformanceMetrics()
        
        # Start background metrics collection
        asyncio.create_task(self._start_metrics_collection())
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect performance metrics."""
        
        start_time = time.time()
        
        # Get request info
        method = request.method
        path = str(request.url.path)
        user_id = None
        request_size = 0
        
        # Try to get user ID from request
        try:
            if hasattr(request.state, 'user'):
                user_id = getattr(request.state.user, 'id', None)
        except:
            pass
        
        # Get request size
        try:
            if hasattr(request, 'headers'):
                content_length = request.headers.get('content-length')
                if content_length:
                    request_size = int(content_length)
        except:
            pass
        
        # Process request
        response = None
        error = None
        
        try:
            response = await call_next(request)
        except Exception as e:
            error = str(e)
            logger.error(f"Request failed: {method} {path} - {error}")
            
            # Return error response
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "detail": str(e)}
            )
        
        # Calculate metrics
        duration = time.time() - start_time
        status_code = response.status_code if response else 500
        response_size = 0
        
        # Get response size
        try:
            if hasattr(response, 'headers'):
                content_length = response.headers.get('content-length')
                if content_length:
                    response_size = int(content_length)
        except:
            pass
        
        # Record metrics
        self.metrics.record_request(
            method=method,
            path=path,
            status_code=status_code,
            duration=duration,
            request_size=request_size,
            response_size=response_size,
            user_id=user_id,
            error=error
        )
        
        # Add performance headers
        if response:
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            response.headers["X-Request-ID"] = str(id(request))
        
        return response
    
    async def _start_metrics_collection(self) -> None:
        """Start background metrics collection."""
        
        while True:
            try:
                # Collect system metrics every 30 seconds
                self.metrics.record_system_metrics()
                
                # Collect database metrics every 60 seconds
                await self.metrics.record_database_metrics()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get the metrics collector."""
        return self.metrics


# Global metrics instance
_performance_metrics: Optional[PerformanceMetrics] = None


def get_performance_metrics() -> PerformanceMetrics:
    """Get the global performance metrics instance."""
    global _performance_metrics
    
    if _performance_metrics is None:
        _performance_metrics = PerformanceMetrics()
    
    return _performance_metrics


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request logging."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request details."""
        
        start_time = time.time()
        
        # Log request start
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log successful request
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"- {response.status_code} in {duration:.3f}s"
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log failed request
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"- {str(e)} in {duration:.3f}s"
            )
            
            raise


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware for health check endpoints."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.health_endpoints = ['/health', '/api/health', '/status']
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle health check requests."""
        
        if request.url.path in self.health_endpoints:
            try:
                # Quick health check
                db_pool = await get_optimized_db_pool()
                pool_status = db_pool.get_pool_status()
                
                health_status = {
                    'status': 'healthy',
                    'timestamp': datetime.utcnow().isoformat(),
                    'database': {
                        'connected': True,
                        'active_connections': pool_status.get('checked_out', 0)
                    },
                    'version': '1.0.0'
                }
                
                return JSONResponse(content=health_status)
                
            except Exception as e:
                health_status = {
                    'status': 'unhealthy',
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e),
                    'version': '1.0.0'
                }
                
                return JSONResponse(
                    content=health_status,
                    status_code=503
                )
        
        return await call_next(request)