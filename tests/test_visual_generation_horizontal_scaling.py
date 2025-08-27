"""
Tests for visual generation horizontal scaling capabilities.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from scrollintel.engines.visual_generation.scaling import (
    VisualGenerationLoadBalancer,
    WorkerNode,
    WorkerStatus,
    LoadBalancingStrategy,
    DistributedVisualCache,
    CacheStrategy,
    DistributedSessionManager,
    UserSession,
    GenerationRequest,
    RequestStatus,
    DatabaseOptimizer,
    HorizontalScalingCoordinator
)


class TestVisualGenerationLoadBalancer:
    """Test load balancer functionality."""
    
    @pytest.fixture
    async def load_balancer(self):
        """Create load balancer for testing."""
        lb = VisualGenerationLoadBalancer(LoadBalancingStrategy.HEALTH_BASED)
        await lb.start()
        yield lb
        await lb.stop()
    
    @pytest.fixture
    def sample_workers(self):
        """Create sample worker nodes."""
        return [
            WorkerNode(
                id="worker1",
                endpoint="http://worker1:8000",
                capabilities=["image_generation"],
                max_concurrent_jobs=4,
                current_jobs=1
            ),
            WorkerNode(
                id="worker2",
                endpoint="http://worker2:8000",
                capabilities=["image_generation", "video_generation"],
                max_concurrent_jobs=4,
                current_jobs=3
            ),
            WorkerNode(
                id="worker3",
                endpoint="http://worker3:8000",
                capabilities=["video_generation"],
                max_concurrent_jobs=2,
                current_jobs=0
            )
        ]
    
    async def test_worker_registration(self, load_balancer, sample_workers):
        """Test worker registration and unregistration."""
        # Register workers
        for worker in sample_workers:
            load_balancer.register_worker(worker)
        
        assert len(load_balancer.workers) == 3
        assert "worker1" in load_balancer.workers
        
        # Unregister worker
        load_balancer.unregister_worker("worker1")
        assert len(load_balancer.workers) == 2
        assert "worker1" not in load_balancer.workers
    
    async def test_worker_selection_by_capability(self, load_balancer, sample_workers):
        """Test worker selection based on capabilities."""
        for worker in sample_workers:
            load_balancer.register_worker(worker)
        
        # Test image generation selection
        selected = await load_balancer.select_worker("image_generation")
        assert selected is not None
        assert "image_generation" in selected.capabilities
        
        # Test video generation selection
        selected = await load_balancer.select_worker("video_generation")
        assert selected is not None
        assert "video_generation" in selected.capabilities
        
        # Test non-existent capability
        selected = await load_balancer.select_worker("audio_generation")
        assert selected is None
    
    async def test_health_based_selection(self, load_balancer, sample_workers):
        """Test health-based worker selection."""
        # Set different health scores
        sample_workers[0].status = WorkerStatus.HEALTHY
        sample_workers[0].current_jobs = 1
        sample_workers[1].status = WorkerStatus.BUSY
        sample_workers[1].current_jobs = 3
        sample_workers[2].status = WorkerStatus.HEALTHY
        sample_workers[2].current_jobs = 0
        
        for worker in sample_workers:
            load_balancer.register_worker(worker)
        
        # Should select worker3 (healthiest for video generation)
        selected = await load_balancer.select_worker("video_generation")
        assert selected.id == "worker3"
    
    async def test_worker_metrics_update(self, load_balancer, sample_workers):
        """Test worker metrics updating."""
        worker = sample_workers[0]
        load_balancer.register_worker(worker)
        
        # Update metrics
        await load_balancer.update_worker_metrics("worker1", {
            "current_jobs": 2,
            "average_response_time": 15.0,
            "gpu_memory_usage": 0.7,
            "cpu_usage": 0.5
        })
        
        updated_worker = load_balancer.workers["worker1"]
        assert updated_worker.current_jobs == 2
        assert updated_worker.average_response_time == 15.0
        assert updated_worker.gpu_memory_usage == 0.7
        assert updated_worker.cpu_usage == 0.5
    
    async def test_job_completion_reporting(self, load_balancer, sample_workers):
        """Test job completion reporting."""
        worker = sample_workers[0]
        worker.current_jobs = 2
        load_balancer.register_worker(worker)
        
        # Report successful completion
        await load_balancer.report_job_completion("worker1", True, 10.0)
        
        updated_worker = load_balancer.workers["worker1"]
        assert updated_worker.current_jobs == 1
        assert updated_worker.total_requests == 1
        assert updated_worker.failed_requests == 0
        
        # Report failed completion
        await load_balancer.report_job_completion("worker1", False, 5.0)
        
        assert updated_worker.total_requests == 2
        assert updated_worker.failed_requests == 1
    
    async def test_cluster_status(self, load_balancer, sample_workers):
        """Test cluster status reporting."""
        for worker in sample_workers:
            load_balancer.register_worker(worker)
        
        status = load_balancer.get_cluster_status()
        
        assert status["total_workers"] == 3
        assert status["healthy_workers"] == 3
        assert status["total_capacity"] == 10  # 4 + 4 + 2
        assert status["current_load"] == 4  # 1 + 3 + 0


class TestDistributedVisualCache:
    """Test distributed caching functionality."""
    
    @pytest.fixture
    async def cache(self):
        """Create cache for testing."""
        cache = DistributedVisualCache(
            redis_nodes=None,  # Use local cache only for testing
            strategy=CacheStrategy.HYBRID
        )
        await cache.initialize()
        yield cache
        await cache.close()
    
    async def test_cache_put_and_get(self, cache):
        """Test basic cache operations."""
        prompt = "A beautiful sunset"
        parameters = {"style": "photorealistic", "resolution": "1024x1024"}
        content_type = "image"
        result_data = b"fake_image_data"
        
        # Store in cache
        await cache.put(prompt, parameters, content_type, result_data)
        
        # Retrieve from cache
        entry = await cache.get(prompt, parameters, content_type)
        
        assert entry is not None
        assert entry.prompt == prompt
        assert entry.content_type == content_type
        assert entry.result_data == result_data
    
    async def test_cache_miss(self, cache):
        """Test cache miss scenario."""
        prompt = "Non-existent prompt"
        parameters = {"style": "cartoon"}
        content_type = "image"
        
        entry = await cache.get(prompt, parameters, content_type)
        assert entry is None
    
    async def test_cache_expiration(self, cache):
        """Test cache entry expiration."""
        prompt = "Expiring prompt"
        parameters = {"style": "abstract"}
        content_type = "image"
        result_data = b"expiring_data"
        
        # Store with short TTL
        await cache.put(prompt, parameters, content_type, result_data, ttl_seconds=1)
        
        # Should be available immediately
        entry = await cache.get(prompt, parameters, content_type)
        assert entry is not None
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        # Should be expired now
        entry = await cache.get(prompt, parameters, content_type)
        assert entry is None
    
    async def test_cache_invalidation(self, cache):
        """Test cache invalidation."""
        # Store multiple entries
        await cache.put("prompt1", {}, "image", b"data1")
        await cache.put("prompt2", {}, "video", b"data2")
        await cache.put("prompt3", {}, "image", b"data3")
        
        # Invalidate image entries
        await cache.invalidate(content_type="image")
        
        # Image entries should be gone
        assert await cache.get("prompt1", {}, "image") is None
        assert await cache.get("prompt3", {}, "image") is None
        
        # Video entry should remain
        assert await cache.get("prompt2", {}, "video") is not None
    
    async def test_cache_statistics(self, cache):
        """Test cache statistics."""
        # Perform some operations
        await cache.put("test", {}, "image", b"data")
        await cache.get("test", {}, "image")  # Hit
        await cache.get("missing", {}, "image")  # Miss
        
        stats = await cache.get_cache_stats()
        
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["hit_rate"] > 0


class TestDistributedSessionManager:
    """Test session management functionality."""
    
    @pytest.fixture
    async def session_manager(self):
        """Create session manager for testing."""
        sm = DistributedSessionManager(redis_nodes=None)  # Local only for testing
        await sm.initialize()
        yield sm
        await sm.close()
    
    async def test_session_creation(self, session_manager):
        """Test session creation."""
        user_id = "test_user_123"
        preferences = {"quality": "high", "style": "realistic"}
        
        session = await session_manager.create_session(user_id, preferences)
        
        assert session.user_id == user_id
        assert session.preferences == preferences
        assert session.status.value == "active"
    
    async def test_session_retrieval(self, session_manager):
        """Test session retrieval."""
        user_id = "test_user_456"
        session = await session_manager.create_session(user_id)
        
        # Retrieve session
        retrieved = await session_manager.get_session(session.id)
        
        assert retrieved is not None
        assert retrieved.id == session.id
        assert retrieved.user_id == user_id
    
    async def test_request_creation(self, session_manager):
        """Test generation request creation."""
        user_id = "test_user_789"
        session = await session_manager.create_session(user_id)
        
        request = await session_manager.create_request(
            session.id,
            "image",
            "A beautiful landscape",
            {"style": "photorealistic"},
            priority=5
        )
        
        assert request is not None
        assert request.user_id == user_id
        assert request.session_id == session.id
        assert request.request_type == "image"
        assert request.priority == 5
        assert request.status == RequestStatus.QUEUED
    
    async def test_request_rate_limiting(self, session_manager):
        """Test request rate limiting."""
        user_id = "rate_limited_user"
        session = await session_manager.create_session(user_id)
        
        # Set low rate limit for testing
        session.rate_limit_per_minute = 2
        await session_manager.update_session(session)
        
        # Create requests up to limit
        request1 = await session_manager.create_request(session.id, "image", "prompt1", {})
        request2 = await session_manager.create_request(session.id, "image", "prompt2", {})
        
        assert request1 is not None
        assert request2 is not None
        
        # Third request should be rejected
        request3 = await session_manager.create_request(session.id, "image", "prompt3", {})
        assert request3 is None
    
    async def test_request_status_updates(self, session_manager):
        """Test request status updates."""
        user_id = "status_test_user"
        session = await session_manager.create_session(user_id)
        request = await session_manager.create_request(session.id, "image", "test prompt", {})
        
        # Start processing
        await session_manager.start_request_processing(request.id, "worker1")
        updated_request = await session_manager.get_request(request.id)
        assert updated_request.status == RequestStatus.PROCESSING
        assert updated_request.worker_id == "worker1"
        
        # Update progress
        await session_manager.update_request_progress(request.id, 0.5)
        updated_request = await session_manager.get_request(request.id)
        assert updated_request.progress == 0.5
        
        # Complete request
        await session_manager.complete_request(request.id, ["result_url1", "result_url2"])
        updated_request = await session_manager.get_request(request.id)
        assert updated_request.status == RequestStatus.COMPLETED
        assert updated_request.result_urls == ["result_url1", "result_url2"]
    
    async def test_request_failure_and_retry(self, session_manager):
        """Test request failure and retry logic."""
        user_id = "retry_test_user"
        session = await session_manager.create_session(user_id)
        request = await session_manager.create_request(session.id, "image", "test prompt", {})
        
        # Fail request with retry
        await session_manager.fail_request(request.id, "Worker error", retry=True)
        updated_request = await session_manager.get_request(request.id)
        
        assert updated_request.status == RequestStatus.QUEUED  # Should be retried
        assert updated_request.retry_count == 1
        assert updated_request.error_message == "Worker error"
        
        # Fail again without retry
        await session_manager.fail_request(request.id, "Final error", retry=False)
        updated_request = await session_manager.get_request(request.id)
        
        assert updated_request.status == RequestStatus.FAILED
    
    async def test_session_termination(self, session_manager):
        """Test session termination."""
        user_id = "termination_test_user"
        session = await session_manager.create_session(user_id)
        request = await session_manager.create_request(session.id, "image", "test prompt", {})
        
        # Terminate session
        await session_manager.terminate_session(session.id)
        
        # Session should be terminated
        updated_session = await session_manager.get_session(session.id)
        assert updated_session.status.value == "terminated"
        
        # Request should be cancelled
        updated_request = await session_manager.get_request(request.id)
        assert updated_request.status == RequestStatus.CANCELLED


class TestDatabaseOptimizer:
    """Test database optimization functionality."""
    
    @pytest.fixture
    def db_optimizer(self):
        """Create database optimizer for testing."""
        # Use mock database URL for testing
        return DatabaseOptimizer("postgresql://test:test@localhost/test")
    
    def test_cache_key_generation(self, db_optimizer):
        """Test cache key generation."""
        prompt = "Test prompt"
        parameters = {"style": "realistic", "size": "1024x1024"}
        content_type = "image"
        
        key1 = db_optimizer._generate_cache_key(prompt, parameters, content_type)
        key2 = db_optimizer._generate_cache_key(prompt, parameters, content_type)
        
        # Same inputs should generate same key
        assert key1 == key2
        
        # Different inputs should generate different keys
        key3 = db_optimizer._generate_cache_key("Different prompt", parameters, content_type)
        assert key1 != key3
    
    def test_query_metrics_recording(self, db_optimizer):
        """Test query metrics recording."""
        from scrollintel.engines.visual_generation.scaling.database_optimizer import QueryType
        
        # Record some metrics
        db_optimizer._record_query_metrics(QueryType.SELECT, 0.5, 10)
        db_optimizer._record_query_metrics(QueryType.INSERT, 1.2, 5)
        
        assert len(db_optimizer.query_metrics) == 2
        
        # Check first metric
        metric1 = db_optimizer.query_metrics[0]
        assert metric1.query_type == QueryType.SELECT
        assert metric1.execution_time == 0.5
        assert metric1.rows_affected == 10
        assert metric1.error is None
        
        # Check second metric
        metric2 = db_optimizer.query_metrics[1]
        assert metric2.query_type == QueryType.INSERT
        assert metric2.execution_time == 1.2
        assert metric2.rows_affected == 5


class TestHorizontalScalingCoordinator:
    """Test horizontal scaling coordinator."""
    
    @pytest.fixture
    async def coordinator(self):
        """Create coordinator for testing."""
        with patch('scrollintel.engines.visual_generation.scaling.horizontal_scaler.DatabaseOptimizer'):
            coordinator = HorizontalScalingCoordinator(
                database_url="postgresql://test:test@localhost/test",
                redis_nodes=None
            )
            
            # Mock the initialize methods to avoid actual connections
            coordinator.load_balancer.start = AsyncMock()
            coordinator.distributed_cache.initialize = AsyncMock()
            coordinator.session_manager.initialize = AsyncMock()
            coordinator.database_optimizer.initialize = AsyncMock()
            
            await coordinator.initialize()
            yield coordinator
            await coordinator.shutdown()
    
    async def test_worker_registration(self, coordinator):
        """Test worker registration through coordinator."""
        await coordinator.register_worker(
            "test_worker",
            "http://test:8000",
            ["image_generation"],
            max_concurrent_jobs=4
        )
        
        # Verify worker was registered with load balancer
        assert "test_worker" in coordinator.load_balancer.workers
    
    async def test_request_submission(self, coordinator):
        """Test request submission through coordinator."""
        # Mock session manager methods
        coordinator.session_manager.create_session = AsyncMock(return_value=Mock(id="session1"))
        coordinator.session_manager.create_request = AsyncMock(return_value=Mock(id="request1"))
        coordinator.distributed_cache.get = AsyncMock(return_value=None)  # Cache miss
        
        request_id = await coordinator.submit_generation_request(
            "user123",
            "image",
            "Test prompt",
            {"style": "realistic"}
        )
        
        assert request_id == "request1"
    
    async def test_cached_request_handling(self, coordinator):
        """Test handling of cached requests."""
        # Mock cache hit
        cached_entry = Mock()
        cached_entry.content_hash = "abc123"
        coordinator.distributed_cache.get = AsyncMock(return_value=cached_entry)
        
        # Mock session manager
        mock_session = Mock(id="session1")
        mock_request = Mock(id="request1")
        coordinator.session_manager.create_session = AsyncMock(return_value=mock_session)
        coordinator.session_manager.create_request = AsyncMock(return_value=mock_request)
        coordinator.session_manager.complete_request = AsyncMock()
        
        request_id = await coordinator.submit_generation_request(
            "user123",
            "image",
            "Cached prompt",
            {"style": "realistic"}
        )
        
        # Should complete request immediately with cached result
        coordinator.session_manager.complete_request.assert_called_once()
        assert request_id == "request1"
    
    async def test_system_metrics_collection(self, coordinator):
        """Test system metrics collection."""
        # Mock component metrics
        coordinator.load_balancer.get_cluster_status = Mock(return_value={"total_workers": 3})
        coordinator.distributed_cache.get_cache_stats = AsyncMock(return_value={"hit_rate": 0.8})
        coordinator.session_manager.get_queue_status = AsyncMock(return_value={"queued_requests": 5})
        coordinator.database_optimizer.get_performance_metrics = AsyncMock(return_value={"queries_per_second": 100})
        
        metrics = await coordinator.get_system_metrics()
        
        assert "cluster" in metrics
        assert "cache" in metrics
        assert "queue" in metrics
        assert "database" in metrics
        assert "scaling" in metrics


@pytest.mark.asyncio
async def test_integration_scenario():
    """Test a complete integration scenario."""
    # This test would require actual Redis and PostgreSQL instances
    # For now, we'll test the component interactions with mocks
    
    with patch('redis.asyncio.RedisCluster'), \
         patch('asyncpg.create_pool'):
        
        coordinator = HorizontalScalingCoordinator(
            database_url="postgresql://test:test@localhost/test"
        )
        
        # Mock initialization
        coordinator.load_balancer.start = AsyncMock()
        coordinator.distributed_cache.initialize = AsyncMock()
        coordinator.session_manager.initialize = AsyncMock()
        coordinator.database_optimizer.initialize = AsyncMock()
        
        await coordinator.initialize()
        
        # Register workers
        await coordinator.register_worker("worker1", "http://worker1:8000", ["image_generation"])
        await coordinator.register_worker("worker2", "http://worker2:8000", ["video_generation"])
        
        # Mock session and request creation
        coordinator.session_manager.create_session = AsyncMock(return_value=Mock(id="session1"))
        coordinator.session_manager.create_request = AsyncMock(return_value=Mock(id="request1"))
        coordinator.distributed_cache.get = AsyncMock(return_value=None)
        
        # Submit request
        request_id = await coordinator.submit_generation_request(
            "user123",
            "image",
            "A beautiful landscape",
            {"style": "photorealistic", "resolution": "1024x1024"}
        )
        
        assert request_id == "request1"
        
        await coordinator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])