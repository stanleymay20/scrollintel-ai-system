"""
Smart Caching System with Staleness Indicators for ScrollIntel.
Provides intelligent caching with automatic staleness detection,
cache warming, and adaptive expiration policies.
"""

import asyncio
import logging
import time
import hashlib
import json
import pickle
import os
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies for different content types."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # Adaptive based on usage patterns
    WRITE_THROUGH = "write_through" # Write through to backing store
    WRITE_BACK = "write_back"      # Write back to backing store


class StalenessLevel(Enum):
    """Levels of content staleness."""
    FRESH = "fresh"               # Content is current
    SLIGHTLY_STALE = "slightly_stale"  # Minor staleness, still usable
    MODERATELY_STALE = "moderately_stale"  # Noticeable staleness
    VERY_STALE = "very_stale"     # Significantly stale
    EXPIRED = "expired"           # Content has expired


@dataclass
class StalenessIndicator:
    """Indicators for determining content staleness."""
    data_source_modified: Optional[datetime] = None
    user_preferences_changed: bool = False
    system_state_changed: bool = False
    dependency_updated: bool = False
    manual_invalidation: bool = False
    usage_pattern_changed: bool = False
    external_trigger: Optional[str] = None
    confidence_score: float = 1.0  # 0.0 to 1.0


@dataclass
class CacheEntry:
    """Enhanced cache entry with staleness tracking."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    last_modified: datetime
    access_count: int
    hit_count: int
    miss_count: int
    size_bytes: int
    ttl_seconds: Optional[float] = None
    staleness_indicators: StalenessIndicator = field(default_factory=StalenessIndicator)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    compression_enabled: bool = False
    encrypted: bool = False
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def time_since_access_seconds(self) -> float:
        """Get time since last access in seconds."""
        return (datetime.utcnow() - self.last_accessed).total_seconds()
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return self.age_seconds > self.ttl_seconds


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    staleness_detection_count: int = 0
    cache_warming_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
    
    @property
    def average_entry_size(self) -> float:
        """Calculate average entry size."""
        return self.total_size_bytes / self.total_entries if self.total_entries > 0 else 0.0


class SmartCacheManager:
    """Smart cache manager with staleness detection and adaptive policies."""
    
    def __init__(self, max_size_mb: int = 100, default_ttl_seconds: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.cache_lock = threading.RLock()
        
        # Staleness detection
        self.staleness_detectors: Dict[str, Callable] = {}
        self.staleness_thresholds: Dict[StalenessLevel, float] = {
            StalenessLevel.FRESH: 0.0,
            StalenessLevel.SLIGHTLY_STALE: 0.2,
            StalenessLevel.MODERATELY_STALE: 0.5,
            StalenessLevel.VERY_STALE: 0.8,
            StalenessLevel.EXPIRED: 1.0
        }
        
        # Cache warming
        self.warming_functions: Dict[str, Callable] = {}
        self.warming_schedule: Dict[str, datetime] = {}
        
        # Statistics and monitoring
        self.stats = CacheStats()
        self.performance_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.enable_compression = True
        self.enable_encryption = False
        self.enable_persistence = True
        self.persistence_file = Path("data/smart_cache.pkl")
        
        # Background tasks
        self._cleanup_task = None
        self._warming_task = None
        self._monitoring_task = None
        
        # Initialize
        self._initialize_staleness_detectors()
        self._load_persistent_cache()
        self._start_background_tasks()
    
    def _initialize_staleness_detectors(self):
        """Initialize staleness detection functions."""
        self.staleness_detectors = {
            "time_based": self._detect_time_based_staleness,
            "dependency_based": self._detect_dependency_staleness,
            "usage_pattern": self._detect_usage_pattern_staleness,
            "data_source": self._detect_data_source_staleness,
            "user_context": self._detect_user_context_staleness
        }
    
    async def get(self, key: str, default: Any = None, 
                  staleness_tolerance: StalenessLevel = StalenessLevel.MODERATELY_STALE) -> Tuple[Any, StalenessLevel]:
        """Get value from cache with staleness information."""
        with self.cache_lock:
            if key not in self.cache:
                self.stats.miss_count += 1
                return default, StalenessLevel.EXPIRED
            
            entry = self.cache[key]
            
            # Update access statistics
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            entry.hit_count += 1
            self.stats.hit_count += 1
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            
            # Check staleness
            staleness_level = await self._assess_staleness(entry)
            
            # Return value if within tolerance
            if staleness_level.value <= staleness_tolerance.value:
                return entry.value, staleness_level
            else:
                # Value is too stale
                self.stats.miss_count += 1
                return default, staleness_level
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None,
                  tags: List[str] = None, dependencies: List[str] = None,
                  metadata: Dict[str, Any] = None) -> bool:
        """Set value in cache with metadata."""
        try:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                last_modified=datetime.utcnow(),
                access_count=0,
                hit_count=0,
                miss_count=0,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds or self.default_ttl_seconds,
                dependencies=dependencies or [],
                tags=tags or [],
                metadata=metadata or {}
            )
            
            with self.cache_lock:
                # Check if we need to evict entries
                await self._ensure_capacity(size_bytes)
                
                # Add to cache
                self.cache[key] = entry
                self.stats.total_entries += 1
                self.stats.total_size_bytes += size_bytes
                
                # Schedule cache warming for dependencies
                await self._schedule_dependency_warming(entry)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache entry {key}: {e}")
            return False
    
    async def invalidate(self, key: str, reason: str = "manual") -> bool:
        """Invalidate a cache entry."""
        with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.staleness_indicators.manual_invalidation = True
                entry.staleness_indicators.external_trigger = reason
                
                # Remove from cache
                del self.cache[key]
                self.stats.total_entries -= 1
                self.stats.total_size_bytes -= entry.size_bytes
                
                logger.info(f"Invalidated cache entry {key} (reason: {reason})")
                return True
        
        return False
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all entries with specified tags."""
        invalidated_count = 0
        
        with self.cache_lock:
            keys_to_remove = []
            
            for key, entry in self.cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                await self.invalidate(key, f"tag_invalidation: {tags}")
                invalidated_count += 1
        
        return invalidated_count
    
    async def invalidate_dependencies(self, dependency_key: str) -> int:
        """Invalidate all entries that depend on a specific key."""
        invalidated_count = 0
        
        with self.cache_lock:
            keys_to_remove = []
            
            for key, entry in self.cache.items():
                if dependency_key in entry.dependencies:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                await self.invalidate(key, f"dependency_invalidation: {dependency_key}")
                invalidated_count += 1
        
        return invalidated_count
    
    async def warm_cache(self, key: str, warming_function: Callable) -> bool:
        """Warm cache with fresh data."""
        try:
            # Execute warming function
            fresh_value = await warming_function()
            
            # Update cache
            success = await self.set(key, fresh_value)
            
            if success:
                self.stats.cache_warming_count += 1
                logger.info(f"Cache warmed for key: {key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Cache warming failed for key {key}: {e}")
            return False
    
    def register_warming_function(self, key_pattern: str, warming_function: Callable):
        """Register a function for cache warming."""
        self.warming_functions[key_pattern] = warming_function
    
    async def _assess_staleness(self, entry: CacheEntry) -> StalenessLevel:
        """Assess the staleness level of a cache entry."""
        staleness_scores = []
        
        # Run all staleness detectors
        for detector_name, detector_func in self.staleness_detectors.items():
            try:
                score = await detector_func(entry)
                staleness_scores.append(score)
            except Exception as e:
                logger.warning(f"Staleness detector {detector_name} failed: {e}")
        
        # Calculate overall staleness score
        if staleness_scores:
            avg_score = sum(staleness_scores) / len(staleness_scores)
        else:
            avg_score = 0.0
        
        # Determine staleness level
        for level in reversed(list(StalenessLevel)):
            if avg_score >= self.staleness_thresholds[level]:
                self.stats.staleness_detection_count += 1
                return level
        
        return StalenessLevel.FRESH
    
    async def _detect_time_based_staleness(self, entry: CacheEntry) -> float:
        """Detect staleness based on time."""
        if entry.ttl_seconds is None:
            return 0.0
        
        age_ratio = entry.age_seconds / entry.ttl_seconds
        return min(age_ratio, 1.0)
    
    async def _detect_dependency_staleness(self, entry: CacheEntry) -> float:
        """Detect staleness based on dependencies."""
        if not entry.dependencies:
            return 0.0
        
        stale_dependencies = 0
        
        for dep_key in entry.dependencies:
            if dep_key in self.cache:
                dep_entry = self.cache[dep_key]
                dep_staleness = await self._assess_staleness(dep_entry)
                if dep_staleness != StalenessLevel.FRESH:
                    stale_dependencies += 1
            else:
                # Dependency not in cache - consider stale
                stale_dependencies += 1
        
        return stale_dependencies / len(entry.dependencies)
    
    async def _detect_usage_pattern_staleness(self, entry: CacheEntry) -> float:
        """Detect staleness based on usage patterns."""
        # Simple heuristic: if not accessed recently, might be stale
        time_since_access = entry.time_since_access_seconds
        
        # Consider stale if not accessed in last hour
        staleness_threshold = 3600  # 1 hour
        
        if time_since_access > staleness_threshold:
            return min(time_since_access / staleness_threshold, 1.0)
        
        return 0.0
    
    async def _detect_data_source_staleness(self, entry: CacheEntry) -> float:
        """Detect staleness based on data source changes."""
        indicators = entry.staleness_indicators
        
        if indicators.data_source_modified:
            # Check if data source was modified after cache entry
            if indicators.data_source_modified > entry.created_at:
                return 1.0
        
        return 0.0
    
    async def _detect_user_context_staleness(self, entry: CacheEntry) -> float:
        """Detect staleness based on user context changes."""
        indicators = entry.staleness_indicators
        staleness_score = 0.0
        
        if indicators.user_preferences_changed:
            staleness_score += 0.3
        
        if indicators.system_state_changed:
            staleness_score += 0.2
        
        if indicators.manual_invalidation:
            staleness_score = 1.0
        
        return min(staleness_score, 1.0)
    
    async def _ensure_capacity(self, required_bytes: int):
        """Ensure cache has capacity for new entry."""
        current_size = self.stats.total_size_bytes
        
        while current_size + required_bytes > self.max_size_bytes and self.cache:
            # Evict least recently used entry
            oldest_key = next(iter(self.cache))
            oldest_entry = self.cache[oldest_key]
            
            del self.cache[oldest_key]
            current_size -= oldest_entry.size_bytes
            self.stats.total_entries -= 1
            self.stats.total_size_bytes -= oldest_entry.size_bytes
            self.stats.eviction_count += 1
            
            logger.debug(f"Evicted cache entry {oldest_key} to free space")
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            # Use pickle to estimate size
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._calculate_size(k) + self._calculate_size(v)
                    for k, v in value.items()
                )
            else:
                return 1024  # Default estimate
    
    async def _schedule_dependency_warming(self, entry: CacheEntry):
        """Schedule cache warming for entry dependencies."""
        for dep_key in entry.dependencies:
            if dep_key in self.warming_functions:
                # Schedule warming in 5 minutes
                self.warming_schedule[dep_key] = datetime.utcnow() + timedelta(minutes=5)
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._warming_task = asyncio.create_task(self._warming_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _cleanup_loop(self):
        """Background task for cache cleanup."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                with self.cache_lock:
                    expired_keys = []
                    
                    for key, entry in self.cache.items():
                        if entry.is_expired:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        await self.invalidate(key, "ttl_expired")
                
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _warming_loop(self):
        """Background task for cache warming."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.utcnow()
                keys_to_warm = []
                
                for key, scheduled_time in self.warming_schedule.items():
                    if current_time >= scheduled_time:
                        keys_to_warm.append(key)
                
                for key in keys_to_warm:
                    if key in self.warming_functions:
                        await self.warm_cache(key, self.warming_functions[key])
                    del self.warming_schedule[key]
                
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                await asyncio.sleep(60)
    
    async def _monitoring_loop(self):
        """Background task for performance monitoring."""
        while True:
            try:
                await asyncio.sleep(600)  # Monitor every 10 minutes
                
                # Record performance snapshot
                snapshot = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "stats": {
                        "total_entries": self.stats.total_entries,
                        "total_size_mb": self.stats.total_size_bytes / (1024 * 1024),
                        "hit_rate": self.stats.hit_rate,
                        "average_entry_size": self.stats.average_entry_size
                    }
                }
                
                self.performance_history.append(snapshot)
                
                # Keep only recent history
                if len(self.performance_history) > 144:  # 24 hours of 10-minute intervals
                    self.performance_history = self.performance_history[-144:]
                
                # Save to persistent storage
                if self.enable_persistence:
                    await self._save_persistent_cache()
                
            except Exception as e:
                logger.error(f"Cache monitoring error: {e}")
                await asyncio.sleep(60)
    
    def _load_persistent_cache(self):
        """Load cache from persistent storage."""
        if not self.enable_persistence or not self.persistence_file.exists():
            return
        
        try:
            with open(self.persistence_file, 'rb') as f:
                data = pickle.load(f)
                
                # Restore cache entries
                for key, entry_data in data.get('cache', {}).items():
                    entry = CacheEntry(**entry_data)
                    
                    # Check if entry is still valid
                    if not entry.is_expired:
                        self.cache[key] = entry
                        self.stats.total_entries += 1
                        self.stats.total_size_bytes += entry.size_bytes
                
                # Restore statistics
                if 'stats' in data:
                    stats_data = data['stats']
                    self.stats.hit_count = stats_data.get('hit_count', 0)
                    self.stats.miss_count = stats_data.get('miss_count', 0)
                    self.stats.eviction_count = stats_data.get('eviction_count', 0)
                
                logger.info(f"Loaded {len(self.cache)} cache entries from persistent storage")
                
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")
    
    async def _save_persistent_cache(self):
        """Save cache to persistent storage."""
        if not self.enable_persistence:
            return
        
        try:
            # Prepare data for serialization
            cache_data = {}
            
            with self.cache_lock:
                for key, entry in self.cache.items():
                    # Convert entry to dict for serialization
                    entry_dict = {
                        'key': entry.key,
                        'value': entry.value,
                        'created_at': entry.created_at,
                        'last_accessed': entry.last_accessed,
                        'last_modified': entry.last_modified,
                        'access_count': entry.access_count,
                        'hit_count': entry.hit_count,
                        'miss_count': entry.miss_count,
                        'size_bytes': entry.size_bytes,
                        'ttl_seconds': entry.ttl_seconds,
                        'dependencies': entry.dependencies,
                        'tags': entry.tags,
                        'metadata': entry.metadata
                    }
                    cache_data[key] = entry_dict
            
            data = {
                'cache': cache_data,
                'stats': {
                    'hit_count': self.stats.hit_count,
                    'miss_count': self.stats.miss_count,
                    'eviction_count': self.stats.eviction_count
                },
                'saved_at': datetime.utcnow().isoformat()
            }
            
            # Ensure directory exists
            self.persistence_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.persistence_file, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"Failed to save persistent cache: {e}")
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        return self.stats
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance history."""
        return self.performance_history.copy()
    
    async def clear(self):
        """Clear all cache entries."""
        with self.cache_lock:
            self.cache.clear()
            self.stats = CacheStats()
        
        logger.info("Cache cleared")
    
    def shutdown(self):
        """Shutdown cache manager and cleanup resources."""
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._warming_task:
            self._warming_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        # Save persistent cache
        if self.enable_persistence:
            asyncio.create_task(self._save_persistent_cache())
        
        logger.info("Cache manager shutdown")


# Global instance
smart_cache_manager = SmartCacheManager()


# Convenience functions
async def cache_get(key: str, default: Any = None, 
                   staleness_tolerance: StalenessLevel = StalenessLevel.MODERATELY_STALE) -> Tuple[Any, StalenessLevel]:
    """Get value from smart cache."""
    return await smart_cache_manager.get(key, default, staleness_tolerance)


async def cache_set(key: str, value: Any, ttl_seconds: Optional[float] = None,
                   tags: List[str] = None, dependencies: List[str] = None) -> bool:
    """Set value in smart cache."""
    return await smart_cache_manager.set(key, value, ttl_seconds, tags, dependencies)


async def cache_invalidate(key: str, reason: str = "manual") -> bool:
    """Invalidate cache entry."""
    return await smart_cache_manager.invalidate(key, reason)


def cache_with_staleness(ttl_seconds: Optional[float] = None, 
                        tags: List[str] = None,
                        dependencies: List[str] = None):
    """Decorator for caching function results with staleness detection."""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            key_data = {
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            cache_key = hashlib.md5(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Try to get from cache
            cached_value, staleness = await cache_get(cache_key)
            
            if cached_value is not None and staleness != StalenessLevel.EXPIRED:
                return cached_value
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_set(cache_key, result, ttl_seconds, tags, dependencies)
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator