"""
ScrollIntel Performance Monitoring and Optimization System
Comprehensive performance tracking with response time monitoring, database optimization, and caching
"""

import time
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager, contextmanager
from collections import defaultdict, deque
import statistics
import psutil
import logging
from functools import wraps
import hashlib
import pickle

from sqlalchemy import text, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ..core.config import get_settings
from ..core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)

@dataclass
class ResponseTimeMetric:
    """Response time metric for endpoints"""
    endpoint: str
    method: str
    response_time: float
    status_code: int
    timestamp: datetime
    user_id: Optional[str] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None

@dataclass
class DatabaseQueryMetric:
    """Database query performance metric"""
    query_hash: str
    query_type: str
    execution_time: float
    rows_affected: Optional[int]
    timestamp: datetime
    connection_id: Optional[str] = None
    query_text: Optional[str] = None

@dataclass
class CacheMetric:
    """Cache performance metric"""
    cache_key: str
    operation: str  # hit, miss, set, delete
    execution_time: float
    data_size: Optional[int]
    timestamp: datetime
    ttl: Optional[int] = None

@dataclass
class PerformanceSummary:
    """Performance summary statistics"""
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    cache_hit_rate: float
    slow_queries_count: int
    avg_db_query_time: float
    active_connections: int
    memory_usage_percent: float
    cpu_usage_percent: float

class ResponseTimeTracker:
    """Tracks response times for all endpoints"""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics: deque = deque(maxlen=max_metrics)
        self.endpoint_stats: Dict[str, List[float]] = defaultdict(list)
        self.logger = get_logger(__name__)
        
    @asynccontextmanager
    async def track_request(self, endpoint: str, method: str, user_id: Optional[str] = None):
        """Context manager to track request response time"""
        start_time = time.time()
        status_code = 200
        request_size = None
        response_size = None
        
        try:
            yield
        except Exception as e:
            status_code = 500
            self.logger.error(f"Request error in {method} {endpoint}: {e}")
            raise
        finally:
            end_time = time.time()
            response_time = end_time - start_time
            
            # Record metric
            metric = ResponseTimeMetric(
                endpoint=endpoint,
                method=method,
                response_time=response_time,
                status_code=status_code,
                timestamp=datetime.utcnow(),
                user_id=user_id,
                request_size=request_size,
                response_size=response_size
            )
            
            self.metrics.append(metric)
            self.endpoint_stats[f"{method} {endpoint}"].append(response_time)
            
            # Keep endpoint stats manageable
            if len(self.endpoint_stats[f"{method} {endpoint}"]) > 1000:
                self.endpoint_stats[f"{method} {endpoint}"] = \
                    self.endpoint_stats[f"{method} {endpoint}"][-500:]
                    
            # Log slow requests
            if response_time > 2.0:
                self.logger.warning(
                    f"Slow request: {method} {endpoint} took {response_time:.2f}s",
                    endpoint=endpoint,
                    method=method,
                    response_time=response_time,
                    status_code=status_code
                )
                
    def get_endpoint_stats(self, endpoint: str, method: str) -> Dict[str, float]:
        """Get statistics for a specific endpoint"""
        key = f"{method} {endpoint}"
        times = self.endpoint_stats.get(key, [])
        
        if not times:
            return {}
            
        return {
            "avg_response_time": statistics.mean(times),
            "median_response_time": statistics.median(times),
            "p95_response_time": self._percentile(times, 95),
            "p99_response_time": self._percentile(times, 99),
            "min_response_time": min(times),
            "max_response_time": max(times),
            "request_count": len(times)
        }
        
    def get_all_endpoint_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all endpoints"""
        return {
            endpoint: self.get_endpoint_stats(*endpoint.split(' ', 1))
            for endpoint in self.endpoint_stats.keys()
        }
        
    def get_slow_endpoints(self, threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Get endpoints with average response time above threshold"""
        slow_endpoints = []
        
        for endpoint, times in self.endpoint_stats.items():
            avg_time = statistics.mean(times)
            if avg_time > threshold:
                method, path = endpoint.split(' ', 1)
                slow_endpoints.append({
                    "endpoint": path,
                    "method": method,
                    "avg_response_time": avg_time,
                    "request_count": len(times),
                    "p95_response_time": self._percentile(times, 95)
                })
                
        return sorted(slow_endpoints, key=lambda x: x["avg_response_time"], reverse=True)
        
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

class DatabaseQueryMonitor:
    """Monitors database query performance"""
    
    def __init__(self, max_metrics: int = 5000):
        self.metrics: deque = deque(maxlen=max_metrics)
        self.query_stats: Dict[str, List[float]] = defaultdict(list)
        self.slow_query_threshold = 1.0  # seconds
        self.logger = get_logger(__name__)
        
    @contextmanager
    def track_query(self, query_text: str, query_type: str = "unknown"):
        """Context manager to track database query performance"""
        start_time = time.time()
        query_hash = hashlib.md5(query_text.encode()).hexdigest()[:16]
        rows_affected = None
        
        try:
            result = yield
            if hasattr(result, 'rowcount'):
                rows_affected = result.rowcount
        except Exception as e:
            self.logger.error(f"Database query error: {e}")
            raise
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Record metric
            metric = DatabaseQueryMetric(
                query_hash=query_hash,
                query_type=query_type,
                execution_time=execution_time,
                rows_affected=rows_affected,
                timestamp=datetime.utcnow(),
                query_text=query_text[:500] if len(query_text) > 500 else query_text
            )
            
            self.metrics.append(metric)
            self.query_stats[query_hash].append(execution_time)
            
            # Log slow queries
            if execution_time > self.slow_query_threshold:
                self.logger.warning(
                    f"Slow query detected: {execution_time:.2f}s",
                    query_hash=query_hash,
                    query_type=query_type,
                    execution_time=execution_time,
                    query_text=query_text[:200]
                )
                
    def get_slow_queries(self, threshold: float = None) -> List[Dict[str, Any]]:
        """Get slow queries above threshold"""
        threshold = threshold or self.slow_query_threshold
        slow_queries = []
        
        for query_hash, times in self.query_stats.items():
            avg_time = statistics.mean(times)
            if avg_time > threshold:
                # Find a recent metric for this query
                recent_metric = None
                for metric in reversed(self.metrics):
                    if metric.query_hash == query_hash:
                        recent_metric = metric
                        break
                        
                slow_queries.append({
                    "query_hash": query_hash,
                    "avg_execution_time": avg_time,
                    "max_execution_time": max(times),
                    "execution_count": len(times),
                    "query_type": recent_metric.query_type if recent_metric else "unknown",
                    "query_text": recent_metric.query_text if recent_metric else None
                })
                
        return sorted(slow_queries, key=lambda x: x["avg_execution_time"], reverse=True)
        
    def get_query_stats_summary(self) -> Dict[str, Any]:
        """Get summary of query performance statistics"""
        if not self.metrics:
            return {}
            
        all_times = [m.execution_time for m in self.metrics]
        slow_queries = [m for m in self.metrics if m.execution_time > self.slow_query_threshold]
        
        return {
            "total_queries": len(self.metrics),
            "avg_query_time": statistics.mean(all_times),
            "median_query_time": statistics.median(all_times),
            "p95_query_time": self._percentile(all_times, 95),
            "p99_query_time": self._percentile(all_times, 99),
            "slow_queries_count": len(slow_queries),
            "slow_queries_percentage": (len(slow_queries) / len(self.metrics)) * 100,
            "unique_queries": len(self.query_stats)
        }
        
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

class CacheManager:
    """Manages caching layer with performance monitoring"""
    
    def __init__(self):
        self.redis_client = None
        self.local_cache: Dict[str, Any] = {}
        self.cache_metrics: deque = deque(maxlen=5000)
        self.hit_count = 0
        self.miss_count = 0
        self.logger = get_logger(__name__)
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            self.logger.info("Redis cache initialized successfully")
        except Exception as e:
            self.logger.warning(f"Redis not available, using local cache: {e}")
            
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        start_time = time.time()
        
        try:
            # Try Redis first
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value is not None:
                    self._record_cache_metric(key, "hit", time.time() - start_time)
                    self.hit_count += 1
                    return json.loads(value)
                    
            # Try local cache
            if key in self.local_cache:
                self._record_cache_metric(key, "hit", time.time() - start_time)
                self.hit_count += 1
                return self.local_cache[key]
                
            # Cache miss
            self._record_cache_metric(key, "miss", time.time() - start_time)
            self.miss_count += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            self.miss_count += 1
            return None
            
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache"""
        start_time = time.time()
        
        try:
            # Set in Redis
            if self.redis_client:
                serialized = json.dumps(value, default=str)
                await self.redis_client.setex(key, ttl, serialized)
                
            # Set in local cache
            self.local_cache[key] = value
            
            # Keep local cache size manageable
            if len(self.local_cache) > 1000:
                # Remove oldest 20% of entries
                keys_to_remove = list(self.local_cache.keys())[:200]
                for k in keys_to_remove:
                    del self.local_cache[k]
                    
            self._record_cache_metric(key, "set", time.time() - start_time, ttl=ttl)
            
        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
            
    async def delete(self, key: str):
        """Delete value from cache"""
        start_time = time.time()
        
        try:
            # Delete from Redis
            if self.redis_client:
                await self.redis_client.delete(key)
                
            # Delete from local cache
            if key in self.local_cache:
                del self.local_cache[key]
                
            self._record_cache_metric(key, "delete", time.time() - start_time)
            
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            
    def _record_cache_metric(self, key: str, operation: str, execution_time: float, ttl: Optional[int] = None):
        """Record cache operation metric"""
        metric = CacheMetric(
            cache_key=key,
            operation=operation,
            execution_time=execution_time,
            data_size=None,
            timestamp=datetime.utcnow(),
            ttl=ttl
        )
        self.cache_metrics.append(metric)
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "local_cache_size": len(self.local_cache),
            "redis_available": self.redis_client is not None
        }

class PerformanceOptimizer:
    """Optimizes system performance based on metrics"""
    
    def __init__(self, response_tracker: ResponseTimeTracker, 
                 db_monitor: DatabaseQueryMonitor, cache_manager: CacheManager):
        self.response_tracker = response_tracker
        self.db_monitor = db_monitor
        self.cache_manager = cache_manager
        self.logger = get_logger(__name__)
        
    async def optimize_slow_endpoints(self) -> List[Dict[str, Any]]:
        """Identify and suggest optimizations for slow endpoints"""
        slow_endpoints = self.response_tracker.get_slow_endpoints(threshold=1.0)
        optimizations = []
        
        for endpoint in slow_endpoints:
            suggestions = []
            
            # Suggest caching for GET endpoints
            if endpoint["method"] == "GET" and endpoint["avg_response_time"] > 2.0:
                suggestions.append("Add response caching")
                
            # Suggest database optimization for very slow endpoints
            if endpoint["avg_response_time"] > 5.0:
                suggestions.append("Review database queries")
                suggestions.append("Add database indexes")
                
            # Suggest pagination for endpoints with many requests
            if endpoint["request_count"] > 1000 and endpoint["avg_response_time"] > 1.0:
                suggestions.append("Implement pagination")
                
            optimizations.append({
                **endpoint,
                "suggestions": suggestions
            })
            
        return optimizations
        
    async def optimize_database_queries(self) -> List[Dict[str, Any]]:
        """Suggest database query optimizations"""
        slow_queries = self.db_monitor.get_slow_queries(threshold=0.5)
        optimizations = []
        
        for query in slow_queries:
            suggestions = []
            
            # Common optimization suggestions
            if "SELECT" in query.get("query_text", "").upper():
                suggestions.append("Add appropriate indexes")
                suggestions.append("Consider query result caching")
                
            if "ORDER BY" in query.get("query_text", "").upper():
                suggestions.append("Add index on ORDER BY columns")
                
            if "GROUP BY" in query.get("query_text", "").upper():
                suggestions.append("Add composite index for GROUP BY")
                
            if query["execution_count"] > 100:
                suggestions.append("High frequency query - prioritize optimization")
                
            optimizations.append({
                **query,
                "suggestions": suggestions
            })
            
        return optimizations
        
    async def get_performance_recommendations(self) -> Dict[str, Any]:
        """Get comprehensive performance recommendations"""
        cache_stats = self.cache_manager.get_cache_stats()
        
        recommendations = {
            "endpoint_optimizations": await self.optimize_slow_endpoints(),
            "database_optimizations": await self.optimize_database_queries(),
            "cache_recommendations": [],
            "system_recommendations": []
        }
        
        # Cache recommendations
        if cache_stats["hit_rate"] < 80:
            recommendations["cache_recommendations"].append({
                "type": "cache_hit_rate",
                "message": f"Cache hit rate is {cache_stats['hit_rate']:.1f}%. Consider caching more frequently accessed data.",
                "priority": "high"
            })
            
        # System recommendations
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 80:
            recommendations["system_recommendations"].append({
                "type": "high_cpu",
                "message": f"CPU usage is {cpu_percent:.1f}%. Consider scaling horizontally.",
                "priority": "high"
            })
            
        if memory_percent > 85:
            recommendations["system_recommendations"].append({
                "type": "high_memory",
                "message": f"Memory usage is {memory_percent:.1f}%. Consider increasing memory or optimizing memory usage.",
                "priority": "high"
            })
            
        return recommendations

class PerformanceDashboard:
    """Real-time performance dashboard"""
    
    def __init__(self, response_tracker: ResponseTimeTracker, 
                 db_monitor: DatabaseQueryMonitor, cache_manager: CacheManager):
        self.response_tracker = response_tracker
        self.db_monitor = db_monitor
        self.cache_manager = cache_manager
        self.logger = get_logger(__name__)
        
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Response time metrics
        all_endpoint_stats = self.response_tracker.get_all_endpoint_stats()
        if all_endpoint_stats:
            all_response_times = []
            for stats in all_endpoint_stats.values():
                all_response_times.extend([stats.get("avg_response_time", 0)])
            avg_response_time = statistics.mean(all_response_times) if all_response_times else 0
            p95_response_time = self.response_tracker._percentile(all_response_times, 95) if all_response_times else 0
        else:
            avg_response_time = 0
            p95_response_time = 0
            
        # Database metrics
        db_stats = self.db_monitor.get_query_stats_summary()
        
        # Cache metrics
        cache_stats = self.cache_manager.get_cache_stats()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2)
            },
            "response_times": {
                "avg_response_time": avg_response_time,
                "p95_response_time": p95_response_time,
                "slow_endpoints_count": len(self.response_tracker.get_slow_endpoints())
            },
            "database": {
                "avg_query_time": db_stats.get("avg_query_time", 0),
                "slow_queries_count": db_stats.get("slow_queries_count", 0),
                "total_queries": db_stats.get("total_queries", 0)
            },
            "cache": {
                "hit_rate": cache_stats["hit_rate"],
                "total_requests": cache_stats["total_requests"],
                "local_cache_size": cache_stats["local_cache_size"]
            },
            "endpoints": all_endpoint_stats
        }

# Global instances
response_tracker = ResponseTimeTracker()
db_monitor = DatabaseQueryMonitor()
cache_manager = CacheManager()
performance_optimizer = PerformanceOptimizer(response_tracker, db_monitor, cache_manager)
performance_dashboard = PerformanceDashboard(response_tracker, db_monitor, cache_manager)

# Decorator for automatic response time tracking
def track_performance(endpoint: str, method: str = "GET"):
    """Decorator to automatically track endpoint performance"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with response_tracker.track_request(endpoint, method):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, we'll use a simple time tracking
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    response_time = time.time() - start_time
                    
                    # Record metric
                    metric = ResponseTimeMetric(
                        endpoint=endpoint,
                        method=method,
                        response_time=response_time,
                        status_code=200,
                        timestamp=datetime.utcnow()
                    )
                    response_tracker.metrics.append(metric)
                    response_tracker.endpoint_stats[f"{method} {endpoint}"].append(response_time)
                    
                    return result
                except Exception as e:
                    response_time = time.time() - start_time
                    metric = ResponseTimeMetric(
                        endpoint=endpoint,
                        method=method,
                        response_time=response_time,
                        status_code=500,
                        timestamp=datetime.utcnow()
                    )
                    response_tracker.metrics.append(metric)
                    response_tracker.endpoint_stats[f"{method} {endpoint}"].append(response_time)
                    raise
            return sync_wrapper
    return decorator

# Initialize cache manager
async def initialize_performance_monitoring():
    """Initialize performance monitoring components"""
    await cache_manager.initialize()
    logger.info("Performance monitoring system initialized")