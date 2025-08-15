"""
Comprehensive tests for the Intelligent Fallback Content Generation System.
Tests all components: fallback manager, progressive loader, smart cache, workflow alternatives, and integration.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from scrollintel.core.intelligent_fallback_manager import (
    IntelligentFallbackManager, ContentContext, ContentType, FallbackContent, FallbackQuality
)
from scrollintel.core.progressive_content_loader import (
    ProgressiveContentLoader, ContentChunk, ContentPriority, LoadingStage
)
from scrollintel.core.smart_cache_manager import (
    SmartCacheManager, StalenessLevel, CacheEntry
)
from scrollintel.core.workflow_alternative_engine import (
    WorkflowAlternativeEngine, WorkflowContext, WorkflowCategory, DifficultyLevel
)
from scrollintel.core.intelligent_fallback_integration import (
    IntelligentFallbackIntegration, IntegratedFallbackRequest, FallbackStrategy
)


class TestIntelligentFallbackManager:
    """Test the intelligent fallback manager."""
    
    @pytest.fixture
    def fallback_manager(self):
        return IntelligentFallbackManager()
    
    @pytest.mark.asyncio
    async def test_generate_chart_fallback(self, fallback_manager):
        """Test chart fallback generation."""
        context = ContentContext(
            user_id="test_user",
            content_type=ContentType.CHART,
            original_request={"chart_type": "bar", "title": "Test Chart"}
        )
        
        fallback = await fallback_manager.generate_fallback_content(context)
        
        assert isinstance(fallback, FallbackContent)
        assert fallback.content_type == ContentType.CHART
        assert fallback.quality in [FallbackQuality.LOW, FallbackQuality.MEDIUM]
        assert "data" in fallback.content
        assert fallback.user_message is not None
        assert len(fallback.suggested_actions) > 0
    
    @pytest.mark.asyncio
    async def test_generate_table_fallback(self, fallback_manager):
        """Test table fallback generation."""
        context = ContentContext(
            user_id="test_user",
            content_type=ContentType.TABLE,
            original_request={"columns": ["Name", "Value", "Status"]}
        )
        
        fallback = await fallback_manager.generate_fallback_content(context)
        
        assert fallback.content_type == ContentType.TABLE
        assert "columns" in fallback.content
        assert "rows" in fallback.content
        assert len(fallback.content["columns"]) > 0
        assert len(fallback.content["rows"]) > 0
    
    @pytest.mark.asyncio
    async def test_generate_analysis_fallback(self, fallback_manager):
        """Test analysis fallback generation."""
        context = ContentContext(
            user_id="test_user",
            content_type=ContentType.ANALYSIS,
            original_request={"analysis_type": "trend"}
        )
        
        fallback = await fallback_manager.generate_fallback_content(context)
        
        assert fallback.content_type == ContentType.ANALYSIS
        assert "insights" in fallback.content
        assert "summary" in fallback.content
        assert isinstance(fallback.content["insights"], list)
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, fallback_manager):
        """Test caching of fallback content."""
        context = ContentContext(
            user_id="test_user",
            content_type=ContentType.TEXT,
            original_request={"test": "data"}
        )
        
        # Generate fallback twice
        fallback1 = await fallback_manager.generate_fallback_content(context)
        fallback2 = await fallback_manager.generate_fallback_content(context)
        
        # Should use cache for second request
        assert fallback2.cache_key is not None
    
    def test_cache_stats(self, fallback_manager):
        """Test cache statistics."""
        stats = fallback_manager.get_cache_stats()
        
        assert "total_entries" in stats
        assert isinstance(stats["total_entries"], int)


class TestProgressiveContentLoader:
    """Test the progressive content loader."""
    
    @pytest.fixture
    def content_loader(self):
        return ProgressiveContentLoader()
    
    @pytest.mark.asyncio
    async def test_load_single_chunk(self, content_loader):
        """Test loading a single content chunk."""
        async def mock_loader():
            await asyncio.sleep(0.1)
            return {"data": "test_content"}
        
        chunk = ContentChunk(
            chunk_id="test_chunk",
            priority=ContentPriority.HIGH,
            content_type=ContentType.TEXT,
            loader_function=mock_loader
        )
        
        request = content_loader.create_loading_request(
            user_id="test_user",
            content_chunks=[chunk]
        )
        
        progress_updates = []
        async for progress in content_loader.load_content_progressively(request):
            progress_updates.append(progress)
            if progress.stage == LoadingStage.COMPLETE:
                break
        
        assert len(progress_updates) > 0
        final_progress = progress_updates[-1]
        assert final_progress.stage == LoadingStage.COMPLETE
        assert final_progress.partial_results is not None
        assert "test_chunk" in final_progress.partial_results
    
    @pytest.mark.asyncio
    async def test_multiple_chunks_with_priorities(self, content_loader):
        """Test loading multiple chunks with different priorities."""
        async def fast_loader():
            await asyncio.sleep(0.05)
            return {"data": "fast_content"}
        
        async def slow_loader():
            await asyncio.sleep(0.2)
            return {"data": "slow_content"}
        
        chunks = [
            ContentChunk(
                chunk_id="high_priority",
                priority=ContentPriority.CRITICAL,
                content_type=ContentType.TEXT,
                loader_function=fast_loader
            ),
            ContentChunk(
                chunk_id="low_priority",
                priority=ContentPriority.LOW,
                content_type=ContentType.TEXT,
                loader_function=slow_loader
            )
        ]
        
        request = content_loader.create_loading_request(
            user_id="test_user",
            content_chunks=chunks
        )
        
        progress_updates = []
        async for progress in content_loader.load_content_progressively(request):
            progress_updates.append(progress)
            if progress.stage == LoadingStage.COMPLETE:
                break
        
        final_progress = progress_updates[-1]
        assert "high_priority" in final_progress.partial_results
        assert "low_priority" in final_progress.partial_results
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, content_loader):
        """Test timeout handling in progressive loading."""
        async def timeout_loader():
            await asyncio.sleep(10)  # Long delay to trigger timeout
            return {"data": "timeout_content"}
        
        chunk = ContentChunk(
            chunk_id="timeout_chunk",
            priority=ContentPriority.MEDIUM,
            content_type=ContentType.TEXT,
            loader_function=timeout_loader
        )
        
        request = content_loader.create_loading_request(
            user_id="test_user",
            content_chunks=[chunk],
            timeout_seconds=0.5  # Short timeout
        )
        
        progress_updates = []
        async for progress in content_loader.load_content_progressively(request):
            progress_updates.append(progress)
            if progress.stage in [LoadingStage.COMPLETE, LoadingStage.FAILED]:
                break
        
        # Should complete with fallback content due to timeout
        final_progress = progress_updates[-1]
        assert final_progress.stage == LoadingStage.COMPLETE
        assert final_progress.metadata.get("timeout_occurred") is True


class TestSmartCacheManager:
    """Test the smart cache manager."""
    
    @pytest.fixture
    def cache_manager(self):
        return SmartCacheManager(max_size_mb=1, default_ttl_seconds=60)
    
    @pytest.mark.asyncio
    async def test_basic_cache_operations(self, cache_manager):
        """Test basic cache set and get operations."""
        test_data = {"key": "value", "number": 42}
        
        # Set cache entry
        success = await cache_manager.set("test_key", test_data, ttl_seconds=30)
        assert success is True
        
        # Get cache entry
        cached_value, staleness = await cache_manager.get("test_key")
        assert cached_value == test_data
        assert staleness == StalenessLevel.FRESH
    
    @pytest.mark.asyncio
    async def test_staleness_detection(self, cache_manager):
        """Test staleness detection."""
        test_data = {"timestamp": datetime.utcnow().isoformat()}
        
        # Set with very short TTL
        await cache_manager.set("stale_key", test_data, ttl_seconds=0.1)
        
        # Wait for staleness
        await asyncio.sleep(0.2)
        
        # Should be stale now
        cached_value, staleness = await cache_manager.get("stale_key")
        assert staleness != StalenessLevel.FRESH
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache_manager):
        """Test cache invalidation."""
        await cache_manager.set("invalid_key", {"data": "test"})
        
        # Verify it's cached
        cached_value, _ = await cache_manager.get("invalid_key")
        assert cached_value is not None
        
        # Invalidate
        success = await cache_manager.invalidate("invalid_key", "test_invalidation")
        assert success is True
        
        # Should be gone
        cached_value, _ = await cache_manager.get("invalid_key")
        assert cached_value is None
    
    @pytest.mark.asyncio
    async def test_tag_based_invalidation(self, cache_manager):
        """Test tag-based cache invalidation."""
        await cache_manager.set("tagged_key1", {"data": "1"}, tags=["group_a", "test"])
        await cache_manager.set("tagged_key2", {"data": "2"}, tags=["group_a"])
        await cache_manager.set("tagged_key3", {"data": "3"}, tags=["group_b"])
        
        # Invalidate by tag
        invalidated_count = await cache_manager.invalidate_by_tags(["group_a"])
        assert invalidated_count == 2
        
        # Check results
        cached_value1, _ = await cache_manager.get("tagged_key1")
        cached_value2, _ = await cache_manager.get("tagged_key2")
        cached_value3, _ = await cache_manager.get("tagged_key3")
        
        assert cached_value1 is None
        assert cached_value2 is None
        assert cached_value3 is not None
    
    def test_cache_stats(self, cache_manager):
        """Test cache statistics."""
        stats = cache_manager.get_stats()
        
        assert hasattr(stats, 'total_entries')
        assert hasattr(stats, 'hit_rate')
        assert hasattr(stats, 'total_size_bytes')


class TestWorkflowAlternativeEngine:
    """Test the workflow alternative engine."""
    
    @pytest.fixture
    def workflow_engine(self):
        return WorkflowAlternativeEngine()
    
    @pytest.mark.asyncio
    async def test_suggest_data_analysis_alternatives(self, workflow_engine):
        """Test suggesting alternatives for data analysis workflows."""
        context = WorkflowContext(
            user_id="test_user",
            original_workflow="analyze_sales_data",
            failure_reason="timeout",
            user_skill_level=DifficultyLevel.INTERMEDIATE,
            time_constraints=30
        )
        
        result = await workflow_engine.suggest_alternatives(context)
        
        assert len(result.alternatives) > 0
        assert result.confidence_score > 0.0
        assert result.reasoning is not None
        
        # Check that alternatives are appropriate
        for alt in result.alternatives:
            assert alt.estimated_total_time_minutes <= context.time_constraints or context.time_constraints is None
            assert alt.difficulty.value <= context.user_skill_level.value or context.user_skill_level == DifficultyLevel.EXPERT
    
    @pytest.mark.asyncio
    async def test_suggest_visualization_alternatives(self, workflow_engine):
        """Test suggesting alternatives for visualization workflows."""
        context = WorkflowContext(
            user_id="test_user",
            original_workflow="create_chart_visualization",
            failure_reason="chart_generation_failed",
            user_skill_level=DifficultyLevel.BEGINNER
        )
        
        result = await workflow_engine.suggest_alternatives(context)
        
        assert len(result.alternatives) > 0
        
        # Should include simple alternatives for beginners
        beginner_alternatives = [alt for alt in result.alternatives if alt.difficulty == DifficultyLevel.BEGINNER]
        assert len(beginner_alternatives) > 0
    
    @pytest.mark.asyncio
    async def test_record_alternative_outcome(self, workflow_engine):
        """Test recording alternative outcomes for learning."""
        user_id = "test_user"
        alternative_id = "manual_data_review"
        
        # Record successful outcome
        await workflow_engine.record_alternative_outcome(
            user_id=user_id,
            alternative_id=alternative_id,
            success=True,
            feedback_score=0.8,
            completion_time_minutes=25
        )
        
        # Check that it was recorded
        stats = workflow_engine.get_user_statistics(user_id)
        assert stats["total_attempts"] == 1
        assert stats["successful_attempts"] == 1
        assert stats["success_rate"] == 1.0
    
    def test_get_alternative_by_id(self, workflow_engine):
        """Test retrieving specific alternative by ID."""
        alternative = workflow_engine.get_alternative_by_id("manual_data_review")
        
        assert alternative is not None
        assert alternative.alternative_id == "manual_data_review"
        assert alternative.name is not None
        assert len(alternative.steps) > 0


class TestIntelligentFallbackIntegration:
    """Test the integrated fallback system."""
    
    @pytest.fixture
    def integration_system(self):
        return IntelligentFallbackIntegration()
    
    @pytest.mark.asyncio
    async def test_immediate_fallback_strategy(self, integration_system):
        """Test immediate fallback strategy."""
        async def failing_function():
            raise Exception("Test failure")
        
        request = IntegratedFallbackRequest(
            request_id="test_immediate",
            user_id="test_user",
            content_type=ContentType.TEXT,
            original_function=failing_function,
            original_args=(),
            original_kwargs={},
            preferred_strategy=FallbackStrategy.IMMEDIATE_FALLBACK,
            max_wait_time_seconds=5.0
        )
        
        result = await integration_system.handle_content_failure(request)
        
        assert result.success is True
        assert result.strategy_used == FallbackStrategy.IMMEDIATE_FALLBACK
        assert result.content is not None
        assert result.loading_time_seconds < 5.0
    
    @pytest.mark.asyncio
    async def test_cached_content_strategy(self, integration_system):
        """Test cached content strategy."""
        # Pre-populate cache
        from scrollintel.core.smart_cache_manager import cache_set
        cache_key = f"fallback_{ContentType.TEXT.value}_{hash(str(()))}"
        await cache_set(cache_key, {"cached": "content"}, ttl_seconds=3600)
        
        async def failing_function():
            raise Exception("Test failure")
        
        request = IntegratedFallbackRequest(
            request_id="test_cached",
            user_id="test_user",
            content_type=ContentType.TEXT,
            original_function=failing_function,
            original_args=(),
            original_kwargs={},
            preferred_strategy=FallbackStrategy.CACHED_CONTENT,
            allow_stale_cache=True
        )
        
        result = await integration_system.handle_content_failure(request)
        
        assert result.success is True
        assert result.strategy_used == FallbackStrategy.CACHED_CONTENT
        assert result.cache_hit is True
        assert result.content == {"cached": "content"}
    
    @pytest.mark.asyncio
    async def test_workflow_alternative_strategy(self, integration_system):
        """Test workflow alternative strategy."""
        async def failing_function():
            raise Exception("Analysis failed")
        
        request = IntegratedFallbackRequest(
            request_id="test_workflow",
            user_id="test_user",
            content_type=ContentType.ANALYSIS,
            original_function=failing_function,
            original_args=(),
            original_kwargs={},
            preferred_strategy=FallbackStrategy.WORKFLOW_ALTERNATIVE,
            failure_context=Exception("Analysis timeout")
        )
        
        result = await integration_system.handle_content_failure(request)
        
        assert result.success is True
        assert result.strategy_used == FallbackStrategy.WORKFLOW_ALTERNATIVE
        assert len(result.workflow_alternatives) > 0
        
        # Check workflow alternative structure
        alt = result.workflow_alternatives[0]
        assert "id" in alt
        assert "name" in alt
        assert "description" in alt
        assert "steps" in alt
    
    @pytest.mark.asyncio
    async def test_hybrid_strategy(self, integration_system):
        """Test hybrid strategy combining multiple approaches."""
        async def failing_function():
            raise Exception("Test failure")
        
        request = IntegratedFallbackRequest(
            request_id="test_hybrid",
            user_id="test_user",
            content_type=ContentType.CHART,
            original_function=failing_function,
            original_args=(),
            original_kwargs={},
            preferred_strategy=FallbackStrategy.HYBRID,
            max_wait_time_seconds=15.0
        )
        
        result = await integration_system.handle_content_failure(request)
        
        assert result.success is True
        assert result.strategy_used == FallbackStrategy.HYBRID
        assert result.content is not None
    
    @pytest.mark.asyncio
    async def test_progressive_loading_strategy(self, integration_system):
        """Test progressive loading strategy."""
        async def slow_function():
            await asyncio.sleep(0.1)
            return {"progressive": "content"}
        
        request = IntegratedFallbackRequest(
            request_id="test_progressive",
            user_id="test_user",
            content_type=ContentType.DASHBOARD,
            original_function=slow_function,
            original_args=(),
            original_kwargs={},
            preferred_strategy=FallbackStrategy.PROGRESSIVE_LOADING,
            max_wait_time_seconds=10.0
        )
        
        result = await integration_system.handle_content_failure(request)
        
        assert result.success is True
        assert result.strategy_used == FallbackStrategy.PROGRESSIVE_LOADING
    
    def test_strategy_performance_tracking(self, integration_system):
        """Test strategy performance tracking."""
        stats = integration_system.get_strategy_performance_stats()
        
        assert isinstance(stats, dict)
        for strategy_name, strategy_stats in stats.items():
            assert "success_rate" in strategy_stats
            assert "avg_response_time" in strategy_stats
            assert "total_attempts" in strategy_stats
    
    def test_user_strategy_preferences(self, integration_system):
        """Test user strategy preferences."""
        user_id = "test_user"
        preferred_strategy = FallbackStrategy.CACHED_CONTENT
        
        integration_system.set_user_strategy_preference(user_id, preferred_strategy)
        
        assert integration_system.user_strategy_preferences[user_id] == preferred_strategy


class TestIntegrationDecorator:
    """Test the integration decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_with_intelligent_fallback_decorator(self):
        """Test the intelligent fallback decorator."""
        from scrollintel.core.intelligent_fallback_integration import with_intelligent_fallback
        
        @with_intelligent_fallback(ContentType.TEXT, max_wait_time=5.0)
        async def failing_function():
            raise Exception("Function failed")
        
        # Should return fallback content instead of raising exception
        result = await failing_function()
        
        assert result is not None
        # The exact structure depends on the fallback content generated
    
    @pytest.mark.asyncio
    async def test_get_intelligent_fallback_function(self):
        """Test the get_intelligent_fallback convenience function."""
        from scrollintel.core.intelligent_fallback_integration import get_intelligent_fallback
        
        async def failing_function():
            raise Exception("Function failed")
        
        result = await get_intelligent_fallback(
            content_type=ContentType.TEXT,
            original_function=failing_function,
            args=(),
            kwargs={},
            error=Exception("Test error"),
            user_id="test_user"
        )
        
        assert result.success is True
        assert result.content is not None
        assert result.strategy_used is not None


# Integration tests
class TestFullSystemIntegration:
    """Test the full system integration."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_fallback_flow(self):
        """Test complete end-to-end fallback flow."""
        from scrollintel.core.intelligent_fallback_integration import (
            intelligent_fallback_integration, IntegratedFallbackRequest, FallbackStrategy
        )
        
        async def complex_failing_function(data_type="chart"):
            if data_type == "chart":
                raise Exception("Chart generation failed due to high system load")
            return {"success": True}
        
        # Create request
        request = IntegratedFallbackRequest(
            request_id="end_to_end_test",
            user_id="integration_test_user",
            content_type=ContentType.CHART,
            original_function=complex_failing_function,
            original_args=("chart",),
            original_kwargs={},
            failure_context=Exception("Chart generation failed due to high system load"),
            preferred_strategy=FallbackStrategy.HYBRID,
            max_wait_time_seconds=10.0,
            user_preferences={"prefer_simple": True},
            system_context={"load": "high", "available_memory": "low"}
        )
        
        # Execute fallback
        result = await intelligent_fallback_integration.handle_content_failure(request)
        
        # Verify result
        assert result.success is True
        assert result.content is not None
        assert result.strategy_used is not None
        assert result.loading_time_seconds >= 0
        assert result.user_message is not None
        assert len(result.suggested_actions) > 0
        
        # Should have some workflow alternatives for chart failures
        if result.workflow_alternatives:
            assert len(result.workflow_alternatives) > 0
            alt = result.workflow_alternatives[0]
            assert "name" in alt
            assert "steps" in alt
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under concurrent load."""
        from scrollintel.core.intelligent_fallback_integration import (
            intelligent_fallback_integration, IntegratedFallbackRequest, FallbackStrategy
        )
        
        async def load_test_function(request_id):
            await asyncio.sleep(0.01)  # Simulate some work
            raise Exception(f"Load test failure {request_id}")
        
        # Create multiple concurrent requests
        requests = []
        for i in range(10):
            request = IntegratedFallbackRequest(
                request_id=f"load_test_{i}",
                user_id=f"user_{i % 3}",  # 3 different users
                content_type=ContentType.TEXT,
                original_function=load_test_function,
                original_args=(i,),
                original_kwargs={},
                preferred_strategy=FallbackStrategy.IMMEDIATE_FALLBACK,
                max_wait_time_seconds=5.0
            )
            requests.append(request)
        
        # Execute all requests concurrently
        start_time = time.time()
        tasks = [
            intelligent_fallback_integration.handle_content_failure(req)
            for req in requests
        ]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify all succeeded
        assert len(results) == 10
        for result in results:
            assert result.success is True
        
        # Should complete reasonably quickly
        total_time = end_time - start_time
        assert total_time < 10.0  # Should complete within 10 seconds
        
        print(f"Processed {len(requests)} concurrent requests in {total_time:.2f} seconds")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])