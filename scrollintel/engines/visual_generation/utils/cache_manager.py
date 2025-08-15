"""
Intelligent caching system for generation results with Redis backend.
"""

import hashlib
import json
import time
import pickle
from typing import Optional, Dict, Any, List
import asyncio
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..base import GenerationRequest, GenerationResult
from ..config import InfrastructureConfig


class GenerationCacheManager:
    """Intelligent caching for generation results with Redis backend and semantic similarity."""
    
    def __init__(self, config: InfrastructureConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.similarity_threshold = 0.85
        self.cleanup_task = None
        self.semantic_model = None
        
        # Initialize Redis connection
        if config.cache_enabled and config.redis_url:
            self._init_redis_connection()
        
        # Initialize semantic similarity model
        if config.semantic_similarity_enabled:
            self._init_semantic_model()
        
        if config.cache_enabled:
            self._start_cleanup_task()
    
    def _init_redis_connection(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.config.redis_url,
                decode_responses=False,  # We'll handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
        except Exception as e:
            print(f"Failed to initialize Redis connection: {e}")
            self.redis_client = None
    
    def _init_semantic_model(self):
        """Initialize semantic similarity model."""
        try:
            # Use a lightweight sentence transformer model
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Failed to initialize semantic model: {e}")
            self.semantic_model = None
    
    def _start_cleanup_task(self):
        """Start background task for cache cleanup."""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired cache entries."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue cleanup task
                print(f"Cache cleanup error: {e}")
    
    async def _cleanup_expired_entries(self):
        """Remove expired entries from cache."""
        try:
            if self.redis_client:
                # Redis handles TTL automatically, but we can clean up orphaned embeddings
                async for key in self.redis_client.scan_iter(match="embedding:*"):
                    cache_key = key.decode().replace('embedding:', '')
                    if not await self.redis_client.exists(f"cache:{cache_key}"):
                        await self.redis_client.delete(key)
            else:
                # Clean up local cache
                current_time = time.time()
                expired_keys = []
                
                for key, entry in self.local_cache.items():
                    if current_time - entry['timestamp'] > entry.get('ttl', self.config.cache_ttl):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.local_cache[key]
        except Exception as e:
            print(f"Cache cleanup error: {e}")
    
    def generate_cache_key(self, request: GenerationRequest) -> str:
        """Generate a cache key for the request."""
        # Create a hash based on key request parameters
        key_data = {
            'prompt': request.prompt,
            'negative_prompt': request.negative_prompt,
            'content_type': request.content_type.value,
            'style': request.style,
            'quality': request.quality,
            'seed': request.seed
        }
        
        # Add type-specific parameters
        if hasattr(request, 'resolution'):
            key_data['resolution'] = request.resolution
        if hasattr(request, 'duration'):
            key_data['duration'] = request.duration
        if hasattr(request, 'fps'):
            key_data['fps'] = request.fps
        
        # Create hash
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def get_cached_result(self, request: GenerationRequest) -> Optional[GenerationResult]:
        """Check for cached results with semantic similarity."""
        if not self.config.cache_enabled:
            return None
        
        # Exact match check
        cache_key = self.generate_cache_key(request)
        cached_result = await self._get_from_cache(cache_key)
        
        if cached_result:
            # Update access time
            await self._update_access_time(cache_key)
            return cached_result
        
        # Semantic similarity check
        if self.config.semantic_similarity_enabled and self.semantic_model:
            similar_result = await self._find_similar_cached_result(request)
            if similar_result:
                return similar_result
        
        return None
    
    async def _get_from_cache(self, cache_key: str) -> Optional[GenerationResult]:
        """Get result from cache (Redis or local)."""
        try:
            if self.redis_client:
                # Try Redis first
                cached_data = await self.redis_client.get(f"cache:{cache_key}")
                if cached_data:
                    cache_entry = pickle.loads(cached_data)
                    if not self._is_expired(cache_entry):
                        return self._deserialize_result(cache_entry['result'])
                    else:
                        # Remove expired entry
                        await self.redis_client.delete(f"cache:{cache_key}")
            else:
                # Fallback to local cache
                cached_entry = self.local_cache.get(cache_key)
                if cached_entry and not self._is_expired(cached_entry):
                    return self._deserialize_result(cached_entry['result'])
        except Exception as e:
            print(f"Cache retrieval error: {e}")
        
        return None
    
    async def _update_access_time(self, cache_key: str):
        """Update last access time for cache entry."""
        try:
            if self.redis_client:
                cached_data = await self.redis_client.get(f"cache:{cache_key}")
                if cached_data:
                    cache_entry = pickle.loads(cached_data)
                    cache_entry['last_accessed'] = time.time()
                    await self.redis_client.set(
                        f"cache:{cache_key}",
                        pickle.dumps(cache_entry),
                        ex=cache_entry.get('ttl', self.config.cache_ttl)
                    )
            else:
                cached_entry = self.local_cache.get(cache_key)
                if cached_entry:
                    cached_entry['last_accessed'] = time.time()
        except Exception as e:
            print(f"Access time update error: {e}")
    
    async def _find_similar_cached_result(self, request: GenerationRequest) -> Optional[GenerationResult]:
        """Find similar cached results based on semantic similarity."""
        if not self.semantic_model:
            return None
        
        try:
            # Get request embedding
            request_embedding = self.semantic_model.encode([request.prompt])
            
            # Search for similar prompts
            similar_keys = await self._search_similar_prompts(request_embedding[0], request)
            
            for cache_key in similar_keys:
                cached_result = await self._get_from_cache(cache_key)
                if cached_result:
                    await self._update_access_time(cache_key)
                    return cached_result
            
        except Exception as e:
            print(f"Semantic similarity search error: {e}")
        
        return None
    
    async def _search_similar_prompts(self, request_embedding: np.ndarray, request: GenerationRequest) -> List[str]:
        """Search for similar prompts using semantic embeddings."""
        similar_keys = []
        
        try:
            if self.redis_client:
                # Get all prompt embeddings from Redis
                pattern = "embedding:*"
                async for key in self.redis_client.scan_iter(match=pattern):
                    embedding_data = await self.redis_client.get(key)
                    if embedding_data:
                        stored_data = pickle.loads(embedding_data)
                        stored_embedding = stored_data['embedding']
                        stored_request = stored_data['request']
                        
                        # Calculate cosine similarity
                        similarity = cosine_similarity(
                            [request_embedding], 
                            [stored_embedding]
                        )[0][0]
                        
                        if similarity >= self.similarity_threshold:
                            # Check parameter compatibility
                            if self._are_parameters_compatible(request, stored_request):
                                cache_key = key.decode().replace('embedding:', '')
                                similar_keys.append(cache_key)
            
        except Exception as e:
            print(f"Similarity search error: {e}")
        
        return similar_keys
    
    def _calculate_prompt_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity between two prompts (simplified)."""
        if prompt1 == prompt2:
            return 1.0
        
        # Simple word-based similarity
        words1 = set(prompt1.split())
        words2 = set(prompt2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _are_parameters_compatible(self, request: GenerationRequest, cached_request: Dict[str, Any]) -> bool:
        """Check if request parameters are compatible with cached request."""
        # Check content type
        if request.content_type.value != cached_request.get('content_type'):
            return False
        
        # Check style compatibility
        if request.style != cached_request.get('style'):
            return False
        
        # Check resolution compatibility (for images/videos)
        if hasattr(request, 'resolution'):
            cached_resolution = cached_request.get('resolution')
            if cached_resolution and request.resolution != tuple(cached_resolution):
                return False
        
        # Check quality compatibility
        quality_levels = ['low', 'medium', 'high', 'ultra']
        request_quality_idx = quality_levels.index(request.quality) if request.quality in quality_levels else 2
        cached_quality_idx = quality_levels.index(cached_request.get('quality', 'high')) if cached_request.get('quality') in quality_levels else 2
        
        # Allow using higher quality cached results
        if cached_quality_idx < request_quality_idx:
            return False
        
        return True
    
    async def cache_result(self, request: GenerationRequest, result: GenerationResult):
        """Cache generation result with appropriate TTL."""
        if not self.config.cache_enabled:
            return
        
        cache_key = self.generate_cache_key(request)
        ttl = self._calculate_cache_ttl(request, result)
        
        cache_entry = {
            'request': self._serialize_request(request),
            'result': self._serialize_result(result),
            'timestamp': time.time(),
            'last_accessed': time.time(),
            'ttl': ttl
        }
        
        try:
            if self.redis_client:
                # Store in Redis
                await self.redis_client.set(
                    f"cache:{cache_key}",
                    pickle.dumps(cache_entry),
                    ex=ttl
                )
                
                # Store semantic embedding if enabled
                if self.config.semantic_similarity_enabled and self.semantic_model:
                    await self._store_prompt_embedding(cache_key, request)
                
            else:
                # Fallback to local cache
                self.local_cache[cache_key] = cache_entry
                await self._enforce_cache_size_limit()
                
        except Exception as e:
            print(f"Cache storage error: {e}")
    
    async def _store_prompt_embedding(self, cache_key: str, request: GenerationRequest):
        """Store prompt embedding for semantic similarity search."""
        try:
            if self.semantic_model:
                embedding = self.semantic_model.encode([request.prompt])[0]
                embedding_data = {
                    'embedding': embedding,
                    'request': self._serialize_request(request),
                    'timestamp': time.time()
                }
                
                await self.redis_client.set(
                    f"embedding:{cache_key}",
                    pickle.dumps(embedding_data),
                    ex=self.config.cache_ttl
                )
        except Exception as e:
            print(f"Embedding storage error: {e}")
    
    def _calculate_cache_ttl(self, request: GenerationRequest, result: GenerationResult) -> int:
        """Calculate appropriate TTL for the cache entry."""
        base_ttl = self.config.cache_ttl
        
        # Longer TTL for high-quality results
        if result.quality_metrics and result.quality_metrics.overall_score > 0.9:
            return base_ttl * 2
        
        # Longer TTL for expensive operations (videos)
        if hasattr(request, 'duration') and request.duration > 10:
            return base_ttl * 3
        
        # Shorter TTL for low-quality results
        if result.quality_metrics and result.quality_metrics.overall_score < 0.7:
            return base_ttl // 2
        
        return base_ttl
    
    async def _enforce_cache_size_limit(self):
        """Enforce cache size limits by removing least recently used entries."""
        max_entries = 1000  # Maximum number of cache entries
        
        if self.redis_client:
            # Redis handles memory management automatically
            return
        
        if len(self.local_cache) <= max_entries:
            return
        
        # Sort by last accessed time
        sorted_entries = sorted(
            self.local_cache.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Remove oldest entries
        entries_to_remove = len(self.local_cache) - max_entries
        for i in range(entries_to_remove):
            key = sorted_entries[i][0]
            del self.local_cache[key]
    
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if a cache entry is expired."""
        current_time = time.time()
        entry_age = current_time - cache_entry['timestamp']
        return entry_age > cache_entry.get('ttl', self.config.cache_ttl)
    
    def _serialize_request(self, request: GenerationRequest) -> Dict[str, Any]:
        """Serialize request for caching."""
        return {
            'prompt': request.prompt,
            'negative_prompt': request.negative_prompt,
            'content_type': request.content_type.value,
            'style': request.style,
            'quality': request.quality,
            'seed': request.seed,
            'resolution': getattr(request, 'resolution', None),
            'duration': getattr(request, 'duration', None),
            'fps': getattr(request, 'fps', None)
        }
    
    def _serialize_result(self, result: GenerationResult) -> Dict[str, Any]:
        """Serialize result for caching."""
        return {
            'id': result.id,
            'request_id': result.request_id,
            'status': result.status.value,
            'content_urls': result.content_urls,
            'content_paths': result.content_paths,
            'metadata': result.metadata,
            'generation_time': result.generation_time,
            'cost': result.cost,
            'model_used': result.model_used,
            'quality_metrics': result.quality_metrics.__dict__ if result.quality_metrics else None
        }
    
    def _deserialize_result(self, result_data: Dict[str, Any]) -> GenerationResult:
        """Deserialize result from cache."""
        from ..base import GenerationStatus, QualityMetrics
        
        quality_metrics = None
        if result_data.get('quality_metrics'):
            quality_metrics = QualityMetrics(**result_data['quality_metrics'])
        
        return GenerationResult(
            id=result_data['id'],
            request_id=result_data['request_id'],
            status=GenerationStatus(result_data['status']),
            content_urls=result_data['content_urls'],
            content_paths=result_data['content_paths'],
            metadata=result_data['metadata'],
            generation_time=result_data['generation_time'],
            cost=result_data['cost'],
            model_used=result_data['model_used'],
            quality_metrics=quality_metrics
        )
    
    async def invalidate_cache(self, pattern: Optional[str] = None):
        """Invalidate cache entries matching a pattern."""
        try:
            if self.redis_client:
                if pattern is None:
                    # Clear all cache entries
                    async for key in self.redis_client.scan_iter(match="cache:*"):
                        await self.redis_client.delete(key)
                    async for key in self.redis_client.scan_iter(match="embedding:*"):
                        await self.redis_client.delete(key)
                else:
                    # Remove entries matching pattern
                    async for key in self.redis_client.scan_iter(match=f"cache:*{pattern}*"):
                        await self.redis_client.delete(key)
                    async for key in self.redis_client.scan_iter(match=f"embedding:*{pattern}*"):
                        await self.redis_client.delete(key)
            else:
                if pattern is None:
                    # Clear all cache
                    self.local_cache.clear()
                else:
                    # Remove entries matching pattern
                    keys_to_remove = [
                        key for key in self.local_cache.keys()
                        if pattern in key
                    ]
                    for key in keys_to_remove:
                        del self.local_cache[key]
        except Exception as e:
            print(f"Cache invalidation error: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'cache_enabled': self.config.cache_enabled,
            'cache_ttl': self.config.cache_ttl,
            'cleanup_interval': self.config.cleanup_interval,
            'semantic_similarity_enabled': self.config.semantic_similarity_enabled,
            'redis_enabled': self.redis_client is not None
        }
        
        try:
            if self.redis_client:
                # Get Redis stats
                info = await self.redis_client.info('memory')
                cache_keys = 0
                embedding_keys = 0
                
                async for key in self.redis_client.scan_iter(match="cache:*"):
                    cache_keys += 1
                async for key in self.redis_client.scan_iter(match="embedding:*"):
                    embedding_keys += 1
                
                stats.update({
                    'total_entries': cache_keys,
                    'embedding_entries': embedding_keys,
                    'redis_memory_used': info.get('used_memory_human', 'N/A'),
                    'redis_memory_peak': info.get('used_memory_peak_human', 'N/A')
                })
            else:
                # Local cache stats
                current_time = time.time()
                total_entries = len(self.local_cache)
                expired_entries = sum(
                    1 for entry in self.local_cache.values()
                    if self._is_expired(entry)
                )
                
                stats.update({
                    'total_entries': total_entries,
                    'active_entries': total_entries - expired_entries,
                    'expired_entries': expired_entries
                })
        except Exception as e:
            stats['error'] = str(e)
        
        return stats
    
    async def cleanup(self):
        """Clean up cache manager resources."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
        
        self.local_cache.clear()