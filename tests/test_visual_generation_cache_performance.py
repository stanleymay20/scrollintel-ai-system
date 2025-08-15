"""
Performance tests for visual generation cache system.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from scrollintel.engines.visual_generation.utils.cache_manager import GenerationCacheManager
from scrollintel.engines.visual_generation.config import InfrastructureConfig
from scrollintel.engines.visual_generation.base import (
    GenerationRequest, GenerationResult, GenerationStatus, 
    QualityMetrics, ContentType
)


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = InfrastructureConfig()
    config.cache_enabled = True
    config.cache_ttl = 3600
    config.redis_url = None  # Use local cache for tests
    config.semantic_similarity_enabled = True
    config.similarity_threshold = 0.85
    config.cleanup_interval = 60
    return config


@pytest.fixture
def sample_request():
    """Create sample generation request."""
    return GenerationRequest(
        prompt="A beautiful sunset over mountains",
        content_type=ContentType.IMAGE,
        style="photorealistic",
        quality="high",
        resolution=(1024, 1024)
    )


@pytest.fixture
def sample_result():
    """Create sample generation result."""
    return GenerationResult(
        id="test-result-1",
        request_id="test-request-1",
        status=GenerationStatus.COMPLETED,
        content_urls=["https://example.com/image1.jpg"],
        content_paths=["/path/to/image1.jpg"],
        metadata={"model": "test-model"},
        generation_time=5.0,
        cost=0.01,
        model_used="test-model",
        quality_metrics=QualityMetrics(
            overall_score=0.9,
            technical_quality=0.85,
            aesthetic_score=0.95,
            prompt_adherence=0.88
        )
    )


class TestCachePerformance:
    """Test cache performance and effectiveness."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, mock_config, sample_request, sample_result):
        """Test cache hit performance."""
        cache_manager = GenerationCacheManager(mock_config)
        
        # Cache the result
        await cache_manager.cache_result(sample_request, sample_result)
        
        # Measure cache hit time
        start_time = time.time()
        cached_result = await cache_manager.get_cached_result(sample_request)
        hit_time = time.time() - start_time
        
        assert cached_result is not None
        assert cached_result.id == sample_result.id
        assert hit_time < 0.01  # Should be very fast (< 10ms)
        
        await cache_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_cache_miss_performance(self, mock_config, sample_request):
        """Test cache miss performance."""
        cache_manager = GenerationCacheManager(mock_config)
        
        # Measure cache miss time
        start_time = time.time()
        cached_result = await cache_manager.get_cached_result(sample_request)
        miss_time = time.time() - start_time
        
        assert cached_result is None
        assert miss_time < 0.05  # Should be fast even for miss (< 50ms)
        
        await cache_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_performance(self, mock_config, sample_result):
        """Test semantic similarity search performance."""
        cache_manager = GenerationCacheManager(mock_config)
        
        # Mock the semantic model to avoid loading actual model
        with patch.object(cache_manager, '_init_semantic_model'):
            cache_manager.semantic_model = Mock()
            cache_manager.semantic_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        
        # Cache multiple similar requests
        similar_prompts = [
            "A beautiful sunset over mountains",
            "A gorgeous sunset above mountain peaks",
            "Beautiful mountain sunset scene",
            "Sunset over rocky mountains",
            "Mountain landscape at sunset"
        ]
        
        for i, prompt in enumerate(similar_prompts):
            request = GenerationRequest(
                prompt=prompt,
                content_type=ContentType.IMAGE,
                style="photorealistic",
                quality="high",
                resolution=(1024, 1024)
            )
            result = GenerationResult(
                id=f"test-result-{i}",
                request_id=f"test-request-{i}",
                status=GenerationStatus.COMPLETED,
                content_urls=[f"https://example.com/image{i}.jpg"],
                content_paths=[f"/path/to/image{i}.jpg"],
                metadata={"model": "test-model"},
                generation_time=5.0,
                cost=0.01,
                model_used="test-model",
                quality_metrics=QualityMetrics(overall_score=0.9)
            )
            await cache_manager.cache_result(request, result)
        
        # Test similarity search performance
        test_request = GenerationRequest(
            prompt="Sunset over mountain range",
            content_type=ContentType.IMAGE,
            style="photorealistic",
            quality="high",
            resolution=(1024, 1024)
        )
        
        start_time = time.time()
        cached_result = await cache_manager.get_cached_result(test_request)
        similarity_time = time.time() - start_time
        
        # Should find similar result quickly
        assert similarity_time < 0.1  # Should be fast (< 100ms)
        
        await cache_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self, mock_config, sample_result):
        """Test concurrent cache access performance."""
        cache_manager = GenerationCacheManager(mock_config)
        
        # Create multiple requests
        requests = []
        for i in range(10):
            request = GenerationRequest(
                prompt=f"Test prompt {i}",
                content_type=ContentType.IMAGE,
                style="photorealistic",
                quality="high",
                resolution=(1024, 1024)
            )
            requests.append(request)
        
        # Cache all results
        for i, request in enumerate(requests):
            result = GenerationResult(
                id=f"test-result-{i}",
                request_id=f"test-request-{i}",
                status=GenerationStatus.COMPLETED,
                content_urls=[f"https://example.com/image{i}.jpg"],
                content_paths=[f"/path/to/image{i}.jpg"],
                metadata={"model": "test-model"},
                generation_time=5.0,
                cost=0.01,
                model_used="test-model",
                quality_metrics=QualityMetrics(overall_score=0.9)
            )
            await cache_manager.cache_result(request, result)
        
        # Test concurrent access
        async def get_cached_result(request):
            return await cache_manager.get_cached_result(request)
        
        start_time = time.time()
        tasks = [get_cached_result(request) for request in requests]
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        # All requests should return cached results
        assert all(result is not None for result in results)
        assert concurrent_time < 0.5  # Should handle concurrent access efficiently
        
        await cache_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_cache_size_limit_performance(self, mock_config, sample_result):
        """Test cache size limit enforcement performance."""
        cache_manager = GenerationCacheManager(mock_config)
        
        # Fill cache beyond limit
        num_entries = 1200  # More than the 1000 limit
        
        start_time = time.time()
        for i in range(num_entries):
            request = GenerationRequest(
                prompt=f"Test prompt {i}",
                content_type=ContentType.IMAGE,
                style="photorealistic",
                quality="high",
                resolution=(1024, 1024)
            )
            result = GenerationResult(
                id=f"test-result-{i}",
                request_id=f"test-request-{i}",
                status=GenerationStatus.COMPLETED,
                content_urls=[f"https://example.com/image{i}.jpg"],
                content_paths=[f"/path/to/image{i}.jpg"],
                metadata={"model": "test-model"},
                generation_time=5.0,
                cost=0.01,
                model_used="test-model",
                quality_metrics=QualityMetrics(overall_score=0.9)
            )
            await cache_manager.cache_result(request, result)
        
        cache_fill_time = time.time() - start_time
        
        # Cache should be limited to reasonable size
        assert len(cache_manager.local_cache) <= 1000
        assert cache_fill_time < 10.0  # Should handle large cache efficiently
        
        await cache_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_cache_cleanup_performance(self, mock_config, sample_result):
        """Test cache cleanup performance."""
        # Set short TTL for testing
        mock_config.cache_ttl = 1
        cache_manager = GenerationCacheManager(mock_config)
        
        # Add entries to cache
        for i in range(100):
            request = GenerationRequest(
                prompt=f"Test prompt {i}",
                content_type=ContentType.IMAGE,
                style="photorealistic",
                quality="high",
                resolution=(1024, 1024)
            )
            result = GenerationResult(
                id=f"test-result-{i}",
                request_id=f"test-request-{i}",
                status=GenerationStatus.COMPLETED,
                content_urls=[f"https://example.com/image{i}.jpg"],
                content_paths=[f"/path/to/image{i}.jpg"],
                metadata={"model": "test-model"},
                generation_time=5.0,
                cost=0.01,
                model_used="test-model",
                quality_metrics=QualityMetrics(overall_score=0.9)
            )
            await cache_manager.cache_result(request, result)
        
        # Wait for entries to expire
        await asyncio.sleep(2)
        
        # Test cleanup performance
        start_time = time.time()
        await cache_manager._cleanup_expired_entries()
        cleanup_time = time.time() - start_time
        
        assert cleanup_time < 1.0  # Cleanup should be fast
        
        await cache_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_cache_statistics_performance(self, mock_config, sample_result):
        """Test cache statistics retrieval performance."""
        cache_manager = GenerationCacheManager(mock_config)
        
        # Add some entries
        for i in range(50):
            request = GenerationRequest(
                prompt=f"Test prompt {i}",
                content_type=ContentType.IMAGE,
                style="photorealistic",
                quality="high",
                resolution=(1024, 1024)
            )
            result = GenerationResult(
                id=f"test-result-{i}",
                request_id=f"test-request-{i}",
                status=GenerationStatus.COMPLETED,
                content_urls=[f"https://example.com/image{i}.jpg"],
                content_paths=[f"/path/to/image{i}.jpg"],
                metadata={"model": "test-model"},
                generation_time=5.0,
                cost=0.01,
                model_used="test-model",
                quality_metrics=QualityMetrics(overall_score=0.9)
            )
            await cache_manager.cache_result(request, result)
        
        # Test statistics performance
        start_time = time.time()
        stats = await cache_manager.get_cache_stats()
        stats_time = time.time() - start_time
        
        assert stats_time < 0.1  # Statistics should be very fast
        assert 'total_entries' in stats
        assert stats['total_entries'] == 50
        
        await cache_manager.cleanup()


class TestCacheEffectiveness:
    """Test cache effectiveness and hit rates."""
    
    @pytest.mark.asyncio
    async def test_exact_match_effectiveness(self, mock_config, sample_request, sample_result):
        """Test exact match cache effectiveness."""
        cache_manager = GenerationCacheManager(mock_config)
        
        # Cache result
        await cache_manager.cache_result(sample_request, sample_result)
        
        # Test exact match
        cached_result = await cache_manager.get_cached_result(sample_request)
        assert cached_result is not None
        assert cached_result.id == sample_result.id
        
        await cache_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_parameter_compatibility_effectiveness(self, mock_config, sample_result):
        """Test parameter compatibility in cache matching."""
        cache_manager = GenerationCacheManager(mock_config)
        
        # Cache high-quality result
        high_quality_request = GenerationRequest(
            prompt="Test prompt",
            content_type=ContentType.IMAGE,
            style="photorealistic",
            quality="high",
            resolution=(1024, 1024)
        )
        await cache_manager.cache_result(high_quality_request, sample_result)
        
        # Request lower quality (should use cached high-quality result)
        low_quality_request = GenerationRequest(
            prompt="Test prompt",
            content_type=ContentType.IMAGE,
            style="photorealistic",
            quality="medium",
            resolution=(1024, 1024)
        )
        
        cached_result = await cache_manager.get_cached_result(low_quality_request)
        assert cached_result is not None  # Should reuse higher quality result
        
        # Request higher quality (should not use cached lower-quality result)
        cache_manager.local_cache.clear()
        
        low_quality_result = GenerationResult(
            id="low-quality-result",
            request_id="low-quality-request",
            status=GenerationStatus.COMPLETED,
            content_urls=["https://example.com/low_quality.jpg"],
            content_paths=["/path/to/low_quality.jpg"],
            metadata={"model": "test-model"},
            generation_time=3.0,
            cost=0.005,
            model_used="test-model",
            quality_metrics=QualityMetrics(overall_score=0.6)
        )
        
        medium_quality_request = GenerationRequest(
            prompt="Test prompt",
            content_type=ContentType.IMAGE,
            style="photorealistic",
            quality="medium",
            resolution=(1024, 1024)
        )
        await cache_manager.cache_result(medium_quality_request, low_quality_result)
        
        cached_result = await cache_manager.get_cached_result(high_quality_request)
        assert cached_result is None  # Should not use lower quality result
        
        await cache_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_ttl_effectiveness(self, mock_config, sample_request, sample_result):
        """Test TTL-based cache expiration effectiveness."""
        # Set very short TTL
        mock_config.cache_ttl = 1
        cache_manager = GenerationCacheManager(mock_config)
        
        # Cache result
        await cache_manager.cache_result(sample_request, sample_result)
        
        # Should be available immediately
        cached_result = await cache_manager.get_cached_result(sample_request)
        assert cached_result is not None
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        # Should be expired
        cached_result = await cache_manager.get_cached_result(sample_request)
        assert cached_result is None
        
        await cache_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_quality_based_ttl_effectiveness(self, mock_config, sample_request):
        """Test quality-based TTL calculation effectiveness."""
        cache_manager = GenerationCacheManager(mock_config)
        
        # High-quality result should get longer TTL
        high_quality_result = GenerationResult(
            id="high-quality-result",
            request_id="high-quality-request",
            status=GenerationStatus.COMPLETED,
            content_urls=["https://example.com/high_quality.jpg"],
            content_paths=["/path/to/high_quality.jpg"],
            metadata={"model": "test-model"},
            generation_time=10.0,
            cost=0.02,
            model_used="test-model",
            quality_metrics=QualityMetrics(overall_score=0.95)
        )
        
        high_quality_ttl = cache_manager._calculate_cache_ttl(sample_request, high_quality_result)
        
        # Low-quality result should get shorter TTL
        low_quality_result = GenerationResult(
            id="low-quality-result",
            request_id="low-quality-request",
            status=GenerationStatus.COMPLETED,
            content_urls=["https://example.com/low_quality.jpg"],
            content_paths=["/path/to/low_quality.jpg"],
            metadata={"model": "test-model"},
            generation_time=3.0,
            cost=0.005,
            model_used="test-model",
            quality_metrics=QualityMetrics(overall_score=0.6)
        )
        
        low_quality_ttl = cache_manager._calculate_cache_ttl(sample_request, low_quality_result)
        
        assert high_quality_ttl > low_quality_ttl
        assert high_quality_ttl == mock_config.cache_ttl * 2  # Should be doubled
        assert low_quality_ttl == mock_config.cache_ttl // 2  # Should be halved
        
        await cache_manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])