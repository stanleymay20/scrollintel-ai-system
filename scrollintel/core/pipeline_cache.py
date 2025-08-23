"""
Pipeline Caching and Performance Optimization System
Provides intelligent caching, query optimization, and performance monitoring.
"""
import asyncio
import hashlib
import json
import pickle
import time
from typing import Any, Dict, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from functools import wraps
import redis
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache strategy types"""
    LRU = "lru"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    REFRESH_AHEAD = "refresh_ahead"

@dataclass
class CacheEntry:
    """Cache entry metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    size_bytes: int
    tags: List[str]

@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit: bool
    timestamp: datetime
    metadata: Dict[str, Any]

class IntelligentCache:
    """Intelligent caching system with multiple strategies"""
    
    def __init__(self, max_size_mb: int = 1024, default_ttl: int = 3600):
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.size_bytes = 0
        
        # Redis connection for distributed caching
        try:
            self.redis_client = redis.Redis(
                host='localhost', 
                port=6379, 
                decode_responses=False,
                socket_connect_timeout=5
            )
            self.redis_available = True
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory cache only: {e}")
            self.redis_client = None
            self.redis_available = False
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
    
    def _generate_key(self, operation: str, params: Dict[str, Any]) -> str:
        """Generate cache key from operation and parameters"""
        # Create deterministic hash from operation and sorted parameters
        param_str = json.dumps(params, sort_keys=True, default=str)
        key_data = f"{operation}:{param_str}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage"""
        return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        return pickle.loads(data)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            return len(self._serialize_value(value))
        except Exception:
            return 1024  # Default estimate
    
    def _evict_lru(self):
        """Evict least recently used entries"""
        if not self.cache:
            return
        
        # Sort by last accessed time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest 25% of entries
        evict_count = max(1, len(sorted_entries) // 4)
        
        for i in range(evict_count):
            key, entry = sorted_entries[i]
            self.size_bytes -= entry.size_bytes
            del self.cache[key]
            self.eviction_count += 1
            
            # Also remove from Redis if available
            if self.redis_available:
                try:
                    self.redis_client.delete(f"pipeline_cache:{key}")
                except Exception as e:
                    logger.warning(f"Failed to evict from Redis: {e}")
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        if entry.ttl_seconds is None:
            return False
        
        age = (datetime.utcnow() - entry.created_at).total_seconds()
        return age > entry.ttl_seconds
    
    async def get(self, operation: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get value from cache"""
        key = self._generate_key(operation, params)
        
        # Check local cache first
        if key in self.cache:
            entry = self.cache[key]
            
            if self._is_expired(entry):
                del self.cache[key]
                self.size_bytes -= entry.size_bytes
            else:
                entry.last_accessed = datetime.utcnow()
                entry.access_count += 1
                self.hit_count += 1
                return entry.value
        
        # Check Redis cache
        if self.redis_available:
            try:
                redis_key = f"pipeline_cache:{key}"
                cached_data = self.redis_client.get(redis_key)
                
                if cached_data:
                    value = self._deserialize_value(cached_data)
                    
                    # Store in local cache for faster access
                    await self.set(operation, params, value, ttl=self.default_ttl)
                    
                    self.hit_count += 1
                    return value
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")
        
        self.miss_count += 1
        return None
    
    async def set(self, operation: str, params: Dict[str, Any], value: Any, 
                 ttl: Optional[int] = None, tags: List[str] = None):
        """Set value in cache"""
        key = self._generate_key(operation, params)
        ttl = ttl or self.default_ttl
        tags = tags or []
        
        # Calculate size
        size_bytes = self._calculate_size(value)
        
        # Check if we need to evict
        max_size_bytes = self.max_size_mb * 1024 * 1024
        while self.size_bytes + size_bytes > max_size_bytes and self.cache:
            self._evict_lru()
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1,
            ttl_seconds=ttl,
            size_bytes=size_bytes,
            tags=tags
        )
        
        # Store in local cache
        if key in self.cache:
            self.size_bytes -= self.cache[key].size_bytes
        
        self.cache[key] = entry
        self.size_bytes += size_bytes
        
        # Store in Redis
        if self.redis_available:
            try:
                redis_key = f"pipeline_cache:{key}"
                serialized_value = self._serialize_value(value)
                self.redis_client.setex(redis_key, ttl, serialized_value)
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
    
    async def invalidate(self, operation: str = None, tags: List[str] = None):
        """Invalidate cache entries by operation or tags"""
        keys_to_remove = []
        
        for key, entry in self.cache.items():
            should_remove = False
            
            if operation and key.startswith(self._generate_key(operation, {})[:16]):
                should_remove = True
            
            if tags and any(tag in entry.tags for tag in tags):
                should_remove = True
            
            if should_remove:
                keys_to_remove.append(key)
        
        # Remove from local cache
        for key in keys_to_remove:
            entry = self.cache[key]
            self.size_bytes -= entry.size_bytes
            del self.cache[key]
            
            # Remove from Redis
            if self.redis_available:
                try:
                    self.redis_client.delete(f"pipeline_cache:{key}")
                except Exception as e:
                    logger.warning(f"Failed to invalidate Redis key: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate_percent': round(hit_rate, 2),
            'eviction_count': self.eviction_count,
            'cache_size_entries': len(self.cache),
            'cache_size_mb': round(self.size_bytes / (1024 * 1024), 2),
            'max_size_mb': self.max_size_mb,
            'redis_available': self.redis_available
        }

class QueryOptimizer:
    """Database query optimization system"""
    
    def __init__(self):
        self.query_cache = IntelligentCache(max_size_mb=512)
        self.slow_query_threshold = 1.0  # seconds
        self.query_stats = {}
    
    async def execute_optimized_query(self, db: Session, query: str, 
                                    params: Dict[str, Any] = None,
                                    cache_ttl: int = 300) -> List[Dict[str, Any]]:
        """Execute query with caching and optimization"""
        params = params or {}
        
        # Try cache first
        cached_result = await self.query_cache.get("sql_query", {
            'query': query,
            'params': params
        })
        
        if cached_result is not None:
            return cached_result
        
        # Execute query with timing
        start_time = time.time()
        
        try:
            result = db.execute(text(query), params)
            rows = [dict(row) for row in result.fetchall()]
            
            duration = time.time() - start_time
            
            # Log slow queries
            if duration > self.slow_query_threshold:
                logger.warning(f"Slow query detected ({duration:.2f}s): {query[:200]}...")
            
            # Update query statistics
            query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
            if query_hash not in self.query_stats:
                self.query_stats[query_hash] = {
                    'query': query[:200],
                    'execution_count': 0,
                    'total_duration': 0,
                    'avg_duration': 0,
                    'max_duration': 0
                }
            
            stats = self.query_stats[query_hash]
            stats['execution_count'] += 1
            stats['total_duration'] += duration
            stats['avg_duration'] = stats['total_duration'] / stats['execution_count']
            stats['max_duration'] = max(stats['max_duration'], duration)
            
            # Cache result
            await self.query_cache.set("sql_query", {
                'query': query,
                'params': params
            }, rows, ttl=cache_ttl)
            
            return rows
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_query_stats(self) -> List[Dict[str, Any]]:
        """Get query performance statistics"""
        return sorted(
            self.query_stats.values(),
            key=lambda x: x['avg_duration'],
            reverse=True
        )

class PerformanceMonitor:
    """Performance monitoring and optimization"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.max_metrics = 10000
    
    def record_performance(self, operation: str, duration_ms: float,
                         memory_usage_mb: float = 0, cpu_usage_percent: float = 0,
                         cache_hit: bool = False, metadata: Dict[str, Any] = None):
        """Record performance metrics"""
        metric = PerformanceMetrics(
            operation=operation,
            duration_ms=duration_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
            cache_hit=cache_hit,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Keep only recent metrics
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def get_performance_summary(self, operation: str = None, 
                              hours: int = 24) -> Dict[str, Any]:
        """Get performance summary"""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        relevant_metrics = [
            m for m in self.metrics
            if m.timestamp >= since and (not operation or m.operation == operation)
        ]
        
        if not relevant_metrics:
            return {}
        
        durations = [m.duration_ms for m in relevant_metrics]
        cache_hits = [m for m in relevant_metrics if m.cache_hit]
        
        return {
            'operation': operation or 'all',
            'total_operations': len(relevant_metrics),
            'avg_duration_ms': sum(durations) / len(durations),
            'min_duration_ms': min(durations),
            'max_duration_ms': max(durations),
            'cache_hit_rate': len(cache_hits) / len(relevant_metrics) * 100,
            'operations_per_hour': len(relevant_metrics) / hours
        }

def cached(ttl: int = 3600, tags: List[str] = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            operation = f"{func.__module__}.{func.__name__}"
            params = {
                'args': str(args),
                'kwargs': kwargs
            }
            
            # Try cache first
            cached_result = await intelligent_cache.get(operation, params)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            
            # Cache result
            await intelligent_cache.set(operation, params, result, ttl=ttl, tags=tags)
            
            # Record performance
            performance_monitor.record_performance(
                operation=operation,
                duration_ms=duration,
                cache_hit=False
            )
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, create async wrapper
            async def async_func():
                return func(*args, **kwargs)
            
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Global instances
intelligent_cache = IntelligentCache()
query_optimizer = QueryOptimizer()
performance_monitor = PerformanceMonitor()

# Convenience functions
async def cache_pipeline_result(pipeline_id: str, result: Any, ttl: int = 3600):
    """Cache pipeline execution result"""
    await intelligent_cache.set("pipeline_result", {'pipeline_id': pipeline_id}, result, ttl=ttl)

async def get_cached_pipeline_result(pipeline_id: str) -> Optional[Any]:
    """Get cached pipeline result"""
    return await intelligent_cache.get("pipeline_result", {'pipeline_id': pipeline_id})

async def invalidate_pipeline_cache(pipeline_id: str):
    """Invalidate all cache entries for a pipeline"""
    await intelligent_cache.invalidate(tags=[f"pipeline:{pipeline_id}"])

def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics"""
    return {
        'cache': intelligent_cache.get_stats(),
        'query_optimizer': query_optimizer.get_query_stats()[:10],  # Top 10 slowest queries
        'performance': performance_monitor.get_performance_summary()
    }