"""
Comprehensive tests for the Intelligent Orchestrator system.
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from scrollintel.engines.visual_generation.intelligent_orchestrator import (
    IntelligentOrchestrator, ModelPerformance
)
from scrollintel.engines.visual_generation.config import VisualGenerationConfig
from scrollintel.engines.visual_generation.base import (
    BaseVisualGenerator, GenerationRequest, GenerationResult,
    ImageGenerationRequest, VideoGenerationRequest, GenerationStatus,
    QualityMetrics
)
from scrollintel.engines.visual_generation.exceptions import VisualGenerationError


class MockGenerator(BaseVisualGenerator):
    """Mock generator for testing"""
    
    def __init__(self, name: str, success_rate: float = 1.0, generation_time: float = 5.0):
        super().__init__({})
        self.name = name
        self.success_rate = success_rate
        self.generation_time = generation_time
        self.is_initialized = True
        self.call_count = 0
    
    async def validate_request(self, request: GenerationRequest) -> bool:
        return True
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.call_count += 1
        await asyncio.sleep(0.1)  # Simulate processing
        
        # Simulate failure based on success rate
        import random
        if random.random() > self.success_rate:
            return GenerationResult(
                id=f"result_{request.request_id}",
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                error_message="Mock failure",
                generation_time=self.generation_time,
                model_used=self.name
            )
        
        return GenerationResult(
            id=f"result_{request.request_id}",
            request_id=request.request_id,
            status=GenerationStatus.COMPLETED,
            content_paths=[f"/mock/path/{request.request_id}.png"],
            generation_time=self.generation_time,
            cost=0.0,
            model_used=self.name,
            quality_metrics=QualityMetrics(overall_score=0.9)
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "model_name": self.name,
            "supported_content_types": ["image"],
            "mock_model": True
        }


class TestIntelligentOrchestrator:
    """Test the intelligent orchestrator system"""
    
    @pytest.fixture
    def config(self):
        """Get test configuration"""
        return VisualGenerationConfig()
    
    @pytest.fixture
    def orchestrator(self, config):
        """Get orchestrator instance"""
        return IntelligentOrchestrator(config)
    
    @pytest.fixture
    def mock_generators(self):
        """Get mock generators with different characteristics"""
        return {
            'scrollintel_fast': MockGenerator('scrollintel_fast', success_rate=0.95, generation_time=2.0),
            'scrollintel_quality': MockGenerator('scrollintel_quality', success_rate=0.98, generation_time=8.0),
            'external_api': MockGenerator('external_api', success_rate=0.85, generation_time=15.0),
            'backup_model': MockGenerator('backup_model', success_rate=0.7, generation_time=20.0)
        }
    
    def test_orchestrator_initialization(self, orchestrator, config):
        """Test orchestrator initialization"""
        assert orchestrator.config == config
        assert orchestrator.generators == {}
        assert orchestrator.performance_metrics == {}
        assert orchestrator.fallback_chains == {}
        assert orchestrator.cost_optimization == True
        assert orchestrator.quality_preference == "balanced"
    
    def test_generator_registration(self, orchestrator, mock_generators):
        """Test generator registration"""
        for name, generator in mock_generators.items():
            orchestrator.register_generator(name, generator)
        
        # Should register all generators
        assert len(orchestrator.generators) == len(mock_generators)
        assert len(orchestrator.performance_metrics) == len(mock_generators)
        
        # Should create performance metrics for each
        for name in mock_generators:
            assert name in orchestrator.performance_metrics
            assert isinstance(orchestrator.performance_metrics[name], ModelPerformance)
            assert orchestrator.performance_metrics[name].model_name == name
    
    def test_fallback_chain_setup(self, orchestrator, mock_generators):
        """Test fallback chain setup"""
        # Register generators
        for name, generator in mock_generators.items():
            orchestrator.register_generator(name, generator)
        
        # Should create fallback chains
        assert 'image' in orchestrator.fallback_chains
        
        # ScrollIntel models should be prioritized
        image_chain = orchestrator.fallback_chains['image']
        scrollintel_models = [m for m in image_chain if m.startswith('scrollintel_')]
        if scrollintel_models:
            # First model should be ScrollIntel
            assert image_chain[0].startswith('scrollintel_')
    
    def test_model_prioritization(self, orchestrator):
        """Test model prioritization logic"""
        models = ['external_api', 'scrollintel_fast', 'backup_model', 'scrollintel_quality']
        prioritized = orchestrator._prioritize_models(models, 'image')
        
        # ScrollIntel models should come first
        scrollintel_indices = [i for i, m in enumerate(prioritized) if m.startswith('scrollintel_')]
        external_indices = [i for i, m in enumerate(prioritized) if not m.startswith('scrollintel_')]
        
        if scrollintel_indices and external_indices:
            assert min(scrollintel_indices) < min(external_indices), "ScrollIntel models should be prioritized"
    
    @pytest.mark.asyncio
    async def test_optimal_generator_selection_speed(self, orchestrator, mock_generators):
        """Test optimal generator selection for speed preference"""
        # Register generators
        for name, generator in mock_generators.items():
            orchestrator.register_generator(name, generator)
        
        # Set speed preference
        orchestrator.quality_preference = "speed"
        
        # Create test request
        request = ImageGenerationRequest(
            prompt="Test speed selection",
            user_id="test_user",
            resolution=(1024, 1024)
        )
        
        # Select generator
        selected = await orchestrator.select_optimal_generator(request)
        
        # Should select fastest model
        assert selected.name in ['scrollintel_fast', 'scrollintel_quality']  # ScrollIntel models are prioritized
    
    @pytest.mark.asyncio
    async def test_optimal_generator_selection_quality(self, orchestrator, mock_generators):
        """Test optimal generator selection for quality preference"""
        # Register generators and simulate some usage
        for name, generator in mock_generators.items():
            orchestrator.register_generator(name, generator)
            # Simulate performance metrics
            metrics = orchestrator.performance_metrics[name]
            metrics.quality_score = 0.95 if 'quality' in name else 0.8
            metrics.total_requests = 10
        
        # Set quality preference
        orchestrator.quality_preference = "quality"
        
        # Create test request
        request = ImageGenerationRequest(
            prompt="Test quality selection",
            user_id="test_user",
            resolution=(1024, 1024)
        )
        
        # Select generator
        selected = await orchestrator.select_optimal_generator(request)
        
        # Should prioritize ScrollIntel models (they come first in prioritization)
        assert selected.name.startswith('scrollintel_')
    
    @pytest.mark.asyncio
    async def test_fallback_generation_success(self, orchestrator, mock_generators):
        """Test successful generation with fallback"""
        # Register generators
        for name, generator in mock_generators.items():
            orchestrator.register_generator(name, generator)
        
        # Create test request
        request = ImageGenerationRequest(
            prompt="Test fallback success",
            user_id="test_user",
            resolution=(1024, 1024)
        )
        
        # Generate with fallback
        result = await orchestrator.generate_with_fallback(request)
        
        # Should succeed
        assert result.status == GenerationStatus.COMPLETED
        assert result.model_used in mock_generators.keys()
        assert len(result.content_paths) > 0
    
    @pytest.mark.asyncio
    async def test_fallback_generation_with_failures(self, orchestrator):
        """Test fallback when some models fail"""
        # Create generators with different failure rates
        failing_generator = MockGenerator('failing_model', success_rate=0.0)  # Always fails
        working_generator = MockGenerator('working_model', success_rate=1.0)  # Always works
        
        orchestrator.register_generator('failing_model', failing_generator)
        orchestrator.register_generator('working_model', working_generator)
        
        # Create test request
        request = ImageGenerationRequest(
            prompt="Test fallback with failures",
            user_id="test_user",
            resolution=(1024, 1024)
        )
        
        # Generate with fallback
        result = await orchestrator.generate_with_fallback(request)
        
        # Should eventually succeed with working model
        assert result.status == GenerationStatus.COMPLETED
        assert result.model_used == 'working_model'
    
    @pytest.mark.asyncio
    async def test_all_models_fail(self, orchestrator):
        """Test behavior when all models fail"""
        # Create generators that always fail
        failing_generator1 = MockGenerator('fail1', success_rate=0.0)
        failing_generator2 = MockGenerator('fail2', success_rate=0.0)
        
        orchestrator.register_generator('fail1', failing_generator1)
        orchestrator.register_generator('fail2', failing_generator2)
        
        # Create test request
        request = ImageGenerationRequest(
            prompt="Test all fail",
            user_id="test_user",
            resolution=(1024, 1024)
        )
        
        # Generate with fallback
        result = await orchestrator.generate_with_fallback(request)
        
        # Should fail gracefully
        assert result.status == GenerationStatus.FAILED
        assert "All models failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_performance_metrics_update(self, orchestrator, mock_generators):
        """Test performance metrics updating"""
        # Register generator
        generator = mock_generators['scrollintel_fast']
        orchestrator.register_generator('test_model', generator)
        
        # Create mock result
        result = GenerationResult(
            id="test_result",
            request_id="test_request",
            status=GenerationStatus.COMPLETED,
            generation_time=5.0,
            cost=2.5,
            model_used="test_model",
            quality_metrics=QualityMetrics(overall_score=0.85)
        )
        
        # Update metrics
        await orchestrator._update_performance_metrics("test_model", result, 5.0)
        
        # Check metrics were updated
        metrics = orchestrator.performance_metrics["test_model"]
        assert metrics.average_time == 5.0
        assert metrics.average_cost == 2.5
        assert metrics.quality_score == 0.85
        assert metrics.total_requests == 1
        assert metrics.success_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_performance_metrics_averaging(self, orchestrator, mock_generators):
        """Test performance metrics averaging over multiple requests"""
        # Register generator
        generator = mock_generators['scrollintel_fast']
        orchestrator.register_generator('test_model', generator)
        
        # Simulate multiple results
        results = [
            GenerationResult(
                id=f"result_{i}",
                request_id=f"request_{i}",
                status=GenerationStatus.COMPLETED,
                generation_time=float(i + 1),
                cost=float(i),
                model_used="test_model",
                quality_metrics=QualityMetrics(overall_score=0.8 + i * 0.05)
            )
            for i in range(3)
        ]
        
        # Update metrics for each result
        for i, result in enumerate(results):
            await orchestrator._update_performance_metrics("test_model", result, float(i + 1))
        
        # Check averaged metrics
        metrics = orchestrator.performance_metrics["test_model"]
        assert metrics.total_requests == 3
        assert metrics.success_rate == 1.0
        # Metrics should be weighted averages
        assert 0 < metrics.average_time < 10
        assert 0 <= metrics.average_cost < 10
        assert 0.8 <= metrics.quality_score <= 1.0
    
    def test_performance_report_generation(self, orchestrator, mock_generators):
        """Test performance report generation"""
        # Register generators
        for name, generator in mock_generators.items():
            orchestrator.register_generator(name, generator)
        
        # Generate report
        report = orchestrator.get_performance_report()
        
        # Should have comprehensive report
        assert "total_models" in report
        assert "active_models" in report
        assert "fallback_chains" in report
        assert "model_performance" in report
        
        assert report["total_models"] == len(mock_generators)
        assert len(report["model_performance"]) == len(mock_generators)
        
        # Each model should have performance data
        for name in mock_generators:
            assert name in report["model_performance"]
            model_perf = report["model_performance"][name]
            assert "success_rate" in model_perf
            assert "average_time" in model_perf
            assert "total_requests" in model_perf
    
    def test_preference_setting(self, orchestrator):
        """Test setting orchestrator preferences"""
        # Test setting preferences
        orchestrator.set_preferences(quality_preference="speed", cost_optimization=False)
        
        assert orchestrator.quality_preference == "speed"
        assert orchestrator.cost_optimization == False
        
        # Should trigger fallback chain reoptimization
        # (This would be tested by checking if _setup_fallback_chains was called)
    
    @pytest.mark.asyncio
    async def test_health_check(self, orchestrator, mock_generators):
        """Test system health check"""
        # Register generators
        for name, generator in mock_generators.items():
            orchestrator.register_generator(name, generator)
        
        # Run health check
        health = await orchestrator.health_check()
        
        # Should provide comprehensive health status
        assert "overall_health" in health
        assert "models" in health
        assert "issues" in health
        
        # Should check all models
        assert len(health["models"]) == len(mock_generators)
        
        # All mock models should be healthy
        for name in mock_generators:
            assert name in health["models"]
            assert health["models"][name]["status"] == "healthy"
        
        # Overall health should be good
        assert health["overall_health"] in ["healthy", "degraded", "critical"]
    
    @pytest.mark.asyncio
    async def test_health_check_with_unhealthy_models(self, orchestrator):
        """Test health check with some unhealthy models"""
        # Create a generator that will fail initialization
        class FailingGenerator(MockGenerator):
            def __init__(self):
                super().__init__("failing_generator")
                self.is_initialized = False
            
            def get_capabilities(self):
                raise Exception("Model failed to initialize")
        
        # Register healthy and unhealthy generators
        healthy_gen = MockGenerator("healthy_model")
        failing_gen = FailingGenerator()
        
        orchestrator.register_generator("healthy_model", healthy_gen)
        orchestrator.register_generator("failing_model", failing_gen)
        
        # Run health check
        health = await orchestrator.health_check()
        
        # Should detect issues
        assert len(health["issues"]) > 0
        assert health["models"]["healthy_model"]["status"] == "healthy"
        assert health["models"]["failing_model"]["status"] == "unhealthy"
        
        # Overall health should be degraded
        assert health["overall_health"] in ["degraded", "critical"]
    
    def test_performance_optimization(self, orchestrator, mock_generators):
        """Test performance optimization"""
        # Register generators with different performance
        for name, generator in mock_generators.items():
            orchestrator.register_generator(name, generator)
            # Simulate different success rates
            metrics = orchestrator.performance_metrics[name]
            metrics.total_requests = 20
            if 'backup' in name:
                metrics.failed_requests = 12  # 40% success rate
                metrics.success_rate = 0.4
            else:
                metrics.failed_requests = 1  # 95% success rate
                metrics.success_rate = 0.95
        
        # Run optimization
        orchestrator.optimize_performance()
        
        # Should reorder fallback chains based on performance
        # (In a real implementation, this would disable poorly performing models)
        assert len(orchestrator.fallback_chains) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_generation_requests(self, orchestrator, mock_generators):
        """Test handling concurrent generation requests"""
        # Register generators
        for name, generator in mock_generators.items():
            orchestrator.register_generator(name, generator)
        
        # Create multiple concurrent requests
        requests = [
            ImageGenerationRequest(
                prompt=f"Concurrent test {i}",
                user_id=f"user_{i}",
                resolution=(512, 512)
            )
            for i in range(5)
        ]
        
        # Process concurrently
        start_time = time.time()
        tasks = [orchestrator.generate_with_fallback(req) for req in requests]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Should handle all requests
        assert len(results) == len(requests)
        
        # Most should succeed
        successful = sum(1 for r in results if r.status == GenerationStatus.COMPLETED)
        assert successful >= len(requests) * 0.8
        
        # Should be faster than sequential processing
        assert total_time < len(requests) * 5  # Assuming 5s per request sequentially
    
    def test_content_type_determination(self, orchestrator):
        """Test content type determination from requests"""
        # Test image request
        image_request = ImageGenerationRequest(
            prompt="Test image",
            user_id="test_user",
            resolution=(1024, 1024)
        )
        assert orchestrator._determine_content_type(image_request) == 'image'
        
        # Test video request
        video_request = VideoGenerationRequest(
            prompt="Test video",
            user_id="test_user",
            duration=5.0,
            resolution=(1920, 1080)
        )
        assert orchestrator._determine_content_type(video_request) == 'video'
    
    @pytest.mark.asyncio
    async def test_model_selection_strategies(self, orchestrator, mock_generators):
        """Test different model selection strategies"""
        # Register generators with known characteristics
        for name, generator in mock_generators.items():
            orchestrator.register_generator(name, generator)
            # Set up performance metrics
            metrics = orchestrator.performance_metrics[name]
            metrics.total_requests = 10
            if 'fast' in name:
                metrics.average_time = 2.0
                metrics.quality_score = 0.8
            elif 'quality' in name:
                metrics.average_time = 8.0
                metrics.quality_score = 0.95
            else:
                metrics.average_time = 15.0
                metrics.quality_score = 0.7
        
        candidates = list(mock_generators.keys())
        
        # Test speed selection
        fastest = orchestrator._select_fastest(candidates, None)
        assert 'fast' in fastest or fastest.startswith('scrollintel_')
        
        # Test quality selection
        best_quality = orchestrator._select_highest_quality(candidates, None)
        assert 'quality' in best_quality or best_quality.startswith('scrollintel_')
        
        # Test balanced selection
        balanced = orchestrator._select_balanced(candidates, None)
        assert balanced in candidates


class TestModelPerformance:
    """Test ModelPerformance class"""
    
    def test_model_performance_initialization(self):
        """Test ModelPerformance initialization"""
        perf = ModelPerformance("test_model")
        
        assert perf.model_name == "test_model"
        assert perf.success_rate == 1.0
        assert perf.average_time == 0.0
        assert perf.average_cost == 0.0
        assert perf.quality_score == 0.0
        assert perf.total_requests == 0
        assert perf.failed_requests == 0
        assert perf.last_used == 0.0
    
    def test_model_performance_updates(self):
        """Test ModelPerformance metric updates"""
        perf = ModelPerformance("test_model")
        
        # Simulate some requests
        perf.total_requests = 10
        perf.failed_requests = 2
        perf.average_time = 5.0
        perf.average_cost = 1.5
        perf.quality_score = 0.85
        perf.last_used = time.time()
        
        # Calculate success rate
        perf.success_rate = (perf.total_requests - perf.failed_requests) / perf.total_requests
        
        assert perf.success_rate == 0.8
        assert perf.average_time == 5.0
        assert perf.average_cost == 1.5
        assert perf.quality_score == 0.85
        assert perf.last_used > 0


if __name__ == "__main__":
    pytest.main([__file__])