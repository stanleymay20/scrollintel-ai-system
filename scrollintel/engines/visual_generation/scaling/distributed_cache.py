"""
Distributed caching system for visual generation.
Implements intelligent caching with Redis Cluster and semantic similarity.
"""

import asyncio
import logging
import hashlib
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import RedisCluster
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    LRU = "lru"
    LFU = "lfu"
    TTL_BASED = "ttl_based"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """Represents a cached generation result."""
    key: str
    content_hash: str
    content_type: str  # 'image' or 'video'
    prompt: str
    parameters: Dict[str, Any]
    result_data: bytes
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int = 3600
    size_bytes: int = 0
    quality_score: float = 0.0
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def age_seconds(self) -> int:
        """Get age of cache entry in seconds."""
        return int((datetime.now() - self.created_at).total_seconds())


class SemanticSimilarityEngine:
    """Engine for computing semantic similarity between prompts."""
    
    def __init__(self):
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.similarity_threshold = 0.85
    
    async def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text (simplified implementation)."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Simplified embedding computation (in production, use proper embedding model)
        # This is a placeholder - replace with actual embedding model
        words = text.lower().split()
        embedding = np.random.rand(384)  # Simulate 384-dimensional embedding
        
        # Add some deterministic component based on text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        np.random.seed(seed)
        embedding += np.random.rand(384) * 0.1
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        self.embedding_cache[text] = embedding
        return embedding
    
    async def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        embedding1 = await self.compute_embedding(text1)
        embedding2 = await self.compute_embedding(text2)
        
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    async def find_similar_prompts(self, prompt: str, cached_prompts: List[str]) -> List[Tuple[str, float]]:
        """Find similar prompts from cache with similarity scores."""
        similarities = []
        
        for cached_prompt in cached_prompts:
            similarity = await self.compute_similarity(prompt, cached_prompt)
            if similarity >= self.similarity_threshold:
                similarities.append((cached_prompt, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities


class DistributedVisualCache:
    """
    Distributed caching system for visual generation results.
    Supports multiple caching strategies and semantic similarity matching.
    """
    
    def __init__(self, 
                 redis_nodes: List[Dict[str, Any]] = None,
                 strategy: CacheStrategy = CacheStrategy.HYBRID,
                 max_memory_mb: int = 1024):
        self.strategy = strategy
        self.max_memory_mb = max_memory_mb
        self.redis_cluster: Optional[RedisCluster] = None
        self.local_cache: Dict[str, CacheEntry] = {}
        self.semantic_engine = SemanticSimilarityEngine()
        
        # Default Redis cluster configuration
        if redis_nodes is None:
            redis_nodes = [
                {"host": "localhost", "port": 7000},
                {"host": "localhost", "port": 7001},
                {"host": "localhost", "port": 7002}
            ]
        self.redis_nodes = redis_nodes
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size_bytes': 0,
            'semantic_hits': 0
        }
    
    async def initialize(self):
        """Initialize the distributed cache system."""
        try:
            # Initialize Redis cluster
            self.redis_cluster = RedisCluster(
                startup_nodes=self.redis_nodes,
                decode_responses=False,
                skip_full_coverage_check=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_cluster.ping()
            logger.info("Connected to Redis cluster for visual generation cache")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis cluster: {e}. Using local cache only.")
            self.redis_cluster = None
    
    async def close(self):
        """Close cache connections."""
        if self.redis_cluster:
            await self.redis_cluster.close()
    
    def _generate_cache_key(self, prompt: str, parameters: Dict[str, Any], content_type: str) -> str:
        """Generate a unique cache key for the request."""
        # Create deterministic key from prompt and parameters
        key_data = {
            'prompt': prompt.strip().lower(),
            'parameters': parameters,
            'content_type': content_type
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        
        return f"visual_gen:{content_type}:{key_hash[:16]}"
    
    def _generate_content_hash(self, content: bytes) -> str:
        """Generate hash for content data."""
        return hashlib.md5(content).hexdigest()
    
    async def get(self, prompt: str, parameters: Dict[str, Any], content_type: str) -> Optional[CacheEntry]:
        """
        Retrieve cached result for the given request.
        
        Args:
            prompt: Generation prompt
            parameters: Generation parameters
            content_type: 'image' or 'video'
        
        Returns:
            Cached entry if found, None otherwise
        """
        cache_key = self._generate_cache_key(prompt, parameters, content_type)
        
        # Try exact match first
        entry = await self._get_exact_match(cache_key)
        if entry:
            self.stats['hits'] += 1
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            return entry
        
        # Try semantic similarity match if enabled
        if self.strategy in [CacheStrategy.SEMANTIC_SIMILARITY, CacheStrategy.HYBRID]:
            entry = await self._get_semantic_match(prompt, parameters, content_type)
            if entry:
                self.stats['semantic_hits'] += 1
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                return entry
        
        self.stats['misses'] += 1
        return None
    
    async def _get_exact_match(self, cache_key: str) -> Optional[CacheEntry]:
        """Get exact cache match."""
        # Try local cache first
        if cache_key in self.local_cache:
            entry = self.local_cache[cache_key]
            if not entry.is_expired:
                return entry
            else:
                # Remove expired entry
                del self.local_cache[cache_key]
        
        # Try Redis cluster
        if self.redis_cluster:
            try:
                cached_data = await self.redis_cluster.get(cache_key)
                if cached_data:
                    entry = pickle.loads(cached_data)
                    if not entry.is_expired:
                        # Store in local cache for faster access
                        self.local_cache[cache_key] = entry
                        return entry
                    else:
                        # Remove expired entry from Redis
                        await self.redis_cluster.delete(cache_key)
            except Exception as e:
                logger.error(f"Error retrieving from Redis cache: {e}")
        
        return None
    
    async def _get_semantic_match(self, prompt: str, parameters: Dict[str, Any], 
                                 content_type: str) -> Optional[CacheEntry]:
        """Find semantically similar cached results."""
        try:
            # Get all cached prompts for this content type
            cached_prompts = await self._get_cached_prompts(content_type)
            
            if not cached_prompts:
                return None
            
            # Find similar prompts
            similar_prompts = await self.semantic_engine.find_similar_prompts(prompt, cached_prompts)
            
            # Try to retrieve results for similar prompts
            for similar_prompt, similarity_score in similar_prompts:
                similar_key = self._generate_cache_key(similar_prompt, parameters, content_type)
                entry = await self._get_exact_match(similar_key)
                
                if entry:
                    # Adjust quality score based on similarity
                    entry.quality_score *= similarity_score
                    logger.info(f"Found semantic match with similarity {similarity_score:.3f}")
                    return entry
            
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
        
        return None
    
    async def _get_cached_prompts(self, content_type: str) -> List[str]:
        """Get list of cached prompts for semantic matching."""
        prompts = []
        
        # From local cache
        for entry in self.local_cache.values():
            if entry.content_type == content_type and not entry.is_expired:
                prompts.append(entry.prompt)
        
        # From Redis (simplified - in production, maintain separate prompt index)
        if self.redis_cluster:
            try:
                pattern = f"visual_gen:{content_type}:*"
                keys = await self.redis_cluster.keys(pattern)
                
                # Limit to avoid performance issues
                for key in keys[:100]:  # Limit to 100 keys
                    try:
                        cached_data = await self.redis_cluster.get(key)
                        if cached_data:
                            entry = pickle.loads(cached_data)
                            if not entry.is_expired:
                                prompts.append(entry.prompt)
                    except Exception:
                        continue
            except Exception as e:
                logger.error(f"Error getting cached prompts from Redis: {e}")
        
        return list(set(prompts))  # Remove duplicates
    
    async def put(self, prompt: str, parameters: Dict[str, Any], content_type: str,
                  result_data: bytes, metadata: Dict[str, Any] = None,
                  ttl_seconds: int = 3600, quality_score: float = 1.0):
        """
        Store result in cache.
        
        Args:
            prompt: Generation prompt
            parameters: Generation parameters
            content_type: 'image' or 'video'
            result_data: Generated content data
            metadata: Additional metadata
            ttl_seconds: Time to live in seconds
            quality_score: Quality score of the result
        """
        cache_key = self._generate_cache_key(prompt, parameters, content_type)
        content_hash = self._generate_content_hash(result_data)
        
        entry = CacheEntry(
            key=cache_key,
            content_hash=content_hash,
            content_type=content_type,
            prompt=prompt,
            parameters=parameters,
            result_data=result_data,
            metadata=metadata or {},
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=ttl_seconds,
            size_bytes=len(result_data),
            quality_score=quality_score
        )
        
        # Store in local cache
        await self._store_local(cache_key, entry)
        
        # Store in Redis cluster
        if self.redis_cluster:
            await self._store_redis(cache_key, entry)
        
        # Update statistics
        self.stats['total_size_bytes'] += entry.size_bytes
        
        # Trigger eviction if needed
        await self._evict_if_needed()
    
    async def _store_local(self, cache_key: str, entry: CacheEntry):
        """Store entry in local cache."""
        self.local_cache[cache_key] = entry
    
    async def _store_redis(self, cache_key: str, entry: CacheEntry):
        """Store entry in Redis cluster."""
        try:
            serialized_entry = pickle.dumps(entry)
            await self.redis_cluster.setex(
                cache_key,
                entry.ttl_seconds,
                serialized_entry
            )
        except Exception as e:
            logger.error(f"Error storing in Redis cache: {e}")
    
    async def _evict_if_needed(self):
        """Evict entries if cache size exceeds limits."""
        current_size_mb = self.stats['total_size_bytes'] / (1024 * 1024)
        
        if current_size_mb > self.max_memory_mb:
            await self._perform_eviction()
    
    async def _perform_eviction(self):
        """Perform cache eviction based on strategy."""
        if self.strategy == CacheStrategy.LRU:
            await self._evict_lru()
        elif self.strategy == CacheStrategy.LFU:
            await self._evict_lfu()
        elif self.strategy == CacheStrategy.TTL_BASED:
            await self._evict_expired()
        elif self.strategy == CacheStrategy.HYBRID:
            await self._evict_hybrid()
    
    async def _evict_lru(self):
        """Evict least recently used entries."""
        # Sort by last accessed time
        sorted_entries = sorted(
            self.local_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest 25% of entries
        evict_count = max(1, len(sorted_entries) // 4)
        
        for i in range(evict_count):
            key, entry = sorted_entries[i]
            await self._remove_entry(key, entry)
    
    async def _evict_lfu(self):
        """Evict least frequently used entries."""
        # Sort by access count
        sorted_entries = sorted(
            self.local_cache.items(),
            key=lambda x: x[1].access_count
        )
        
        # Remove least used 25% of entries
        evict_count = max(1, len(sorted_entries) // 4)
        
        for i in range(evict_count):
            key, entry = sorted_entries[i]
            await self._remove_entry(key, entry)
    
    async def _evict_expired(self):
        """Remove expired entries."""
        expired_keys = []
        
        for key, entry in self.local_cache.items():
            if entry.is_expired:
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self.local_cache[key]
            await self._remove_entry(key, entry)
    
    async def _evict_hybrid(self):
        """Hybrid eviction strategy combining multiple factors."""
        # First remove expired entries
        await self._evict_expired()
        
        # If still over limit, use weighted scoring
        current_size_mb = self.stats['total_size_bytes'] / (1024 * 1024)
        
        if current_size_mb > self.max_memory_mb:
            # Score entries based on multiple factors
            scored_entries = []
            
            for key, entry in self.local_cache.items():
                # Lower score = higher priority for eviction
                score = (
                    entry.quality_score * 0.4 +  # Quality
                    (entry.access_count / 10.0) * 0.3 +  # Frequency
                    (1.0 / max(entry.age_seconds, 1)) * 0.3  # Recency
                )
                scored_entries.append((score, key, entry))
            
            # Sort by score (lowest first)
            scored_entries.sort(key=lambda x: x[0])
            
            # Remove lowest scoring entries
            evict_count = max(1, len(scored_entries) // 4)
            
            for i in range(evict_count):
                _, key, entry = scored_entries[i]
                await self._remove_entry(key, entry)
    
    async def _remove_entry(self, key: str, entry: CacheEntry):
        """Remove entry from cache."""
        # Remove from local cache
        if key in self.local_cache:
            del self.local_cache[key]
        
        # Remove from Redis
        if self.redis_cluster:
            try:
                await self.redis_cluster.delete(key)
            except Exception as e:
                logger.error(f"Error removing from Redis cache: {e}")
        
        # Update statistics
        self.stats['evictions'] += 1
        self.stats['total_size_bytes'] -= entry.size_bytes
    
    async def invalidate(self, prompt: str = None, content_type: str = None):
        """Invalidate cache entries matching criteria."""
        keys_to_remove = []
        
        for key, entry in self.local_cache.items():
            should_remove = True
            
            if prompt and entry.prompt != prompt:
                should_remove = False
            
            if content_type and entry.content_type != content_type:
                should_remove = False
            
            if should_remove:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            entry = self.local_cache[key]
            await self._remove_entry(key, entry)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1)
        semantic_hit_rate = self.stats['semantic_hits'] / max(self.stats['misses'], 1)
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'semantic_hit_rate': semantic_hit_rate,
            'local_cache_size': len(self.local_cache),
            'memory_usage_mb': self.stats['total_size_bytes'] / (1024 * 1024),
            'memory_limit_mb': self.max_memory_mb,
            'strategy': self.strategy.value
        }
    
    async def cleanup_expired(self):
        """Clean up expired entries."""
        await self._evict_expired()


# Global distributed cache instance
distributed_cache = DistributedVisualCache()