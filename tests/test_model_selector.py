"""
Tests for Model Selector functionality.

Tests model selection strategies, performance tracking, A/B testing,
and analytics capabilities.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
import random

from scrollintel.core.model_selector import (
    ModelSelector,
    ModelCapabilities,
    ModelPerformanceMetrics,
    GenerationRequest,
    ModelSelection,
    ModelType,
    QualityMetric,
    PerformanceBasedStrategy,
    CostOptimizedStrategy,
    QualityOptimizedStrategy,
    ABTestingFramework,
    initialize_default_models
)


class TestModelCapabilities:
    """Test ModelCapabilities class."""
    
    def test_model_capabilities_creation(self):
        """Test creating model capabilities."""
        caps = ModelCapabilities(
            model_id="test_model",
            model_type=ModelType.IMAGE_GENERATION,
            supported_resolutions=[(1024, 1024)],
            supported_formats=["jpg", "png"],
            max_prompt_length=1000,
            cost_per_generation=0.05
        )
        
        assert caps.model_id == "test_model"
        assert caps.model_type == ModelType.IMAGE_GENERATION
        assert (1024, 1024) in caps.supported_resolutions
        assert "jpg" in caps.supported_formats
        assert caps.cost_per_generation == 0.05


class TestModelPerformanceMetrics:
    """Test ModelPerformanceMetrics class."""
    
    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = ModelPerformanceMetrics(
            model_id="test_model",
            total_generations=100,
            successful_generations=95,
            average_processing_time=25.0,
            average_quality_score=0.85,
            average_cost=0.03
        )
        
        assert metrics.model_id == "test_model"
        assert metrics.success_rate == 95.0
        assert metrics.efficiency_score == 0.85 / 25.0
        assert metrics.cost_effectiveness == 0.85 / 0.03
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ModelPerformanceMetrics("test", total_generations=0)
        assert metrics.success_rate == 0.0
        
        metrics.total_generations = 10
        metrics.successful_generations = 8
        assert metrics.success_rate == 80.0
    
    def test_efficiency_score(self):
        """Test efficiency score calculation."""
        metrics = ModelPerformanceMetrics(
            "test",
            average_quality_score=0.8,
            average_processing_time=20.0
        )
        assert metrics.efficiency_score == 0.04
        
        # Test zero processing time
        metrics.average_processing_time = 0.0
        assert metrics.efficiency_score == 0.0


class TestGenerationRequest:
    """Test GenerationRequest class."""
    
    def test_request_creation(self):
        """Test creating generation request."""
        request = GenerationRequest(
            request_id="test_123",
            model_type=ModelType.IMAGE_GENERATION,
            prompt="A beautiful sunset",
            resolution=(1024, 1024),
            quality_preference="quality"
        )
        
        assert request.request_id == "test_123"
        assert request.model_type == ModelType.IMAGE_GENERATION
        assert request.prompt == "A beautiful sunset"
        assert request.quality_preference == "quality"
    
    def test_request_to_dict(self):
        """Test converting request to dictionary."""
        request = GenerationRequest(
            request_id="test_123",
            model_type=ModelType.IMAGE_GENERATION,
            prompt="Test prompt"
        )
        
        request_dict = request.to_dict()
        assert request_dict["model_type"] == "image_generation"
        assert request_dict["prompt"] == "Test prompt"
        assert "request_id" not in request_dict  # Should not include ID in dict


class TestSelectionStrategies:
    """Test different model selection strategies."""
    
    @pytest.fixture
    def sample_models(self):
        """Create sample models for testing."""
        models = ["model_a", "model_b", "model_c"]
        
        capabilities = {
            "model_a": ModelCapabilities(
                model_id="model_a",
                model_type=ModelType.IMAGE_GENERATION,
                supported_resolutions=[(1024, 1024)],
                supported_formats=["jpg"],
                max_prompt_length=1000,
                cost_per_generation=0.05
            ),
            "model_b": ModelCapabilities(
                model_id="model_b",
                model_type=ModelType.IMAGE_GENERATION,
                supported_resolutions=[(1024, 1024)],
                supported_formats=["jpg"],
                max_prompt_length=1000,
                cost_per_generation=0.03
            ),
            "model_c": ModelCapabilities(
                model_id="model_c",
                model_type=ModelType.IMAGE_GENERATION,
                supported_resolutions=[(1024, 1024)],
                supported_formats=["jpg"],
                max_prompt_length=1000,
                cost_per_generation=0.08
            )
        }
        
        metrics = {
            "model_a": ModelPerformanceMetrics(
                model_id="model_a",
                total_generations=100,
                successful_generations=95,
                average_quality_score=0.9,
                average_processing_time=20.0
            ),
            "model_b": ModelPerformanceMetrics(
                model_id="model_b",
                total_generations=80,
                successful_generations=70,
                average_quality_score=0.75,
                average_processing_time=15.0
            ),
            "model_c": ModelPerformanceMetrics(
                model_id="model_c",
                total_generations=50,
                successful_generations=48,
                average_quality_score=0.95,
                average_processing_time=30.0
            )
        }
        
        return models, capabilities, metrics
    
    @pytest.fixture
    def sample_request(self):
        """Create sample generation request."""
        return GenerationRequest(
            request_id="test_123",
            model_type=ModelType.IMAGE_GENERATION,
            prompt="Test prompt",
            resolution=(1024, 1024),
            format="jpg"
        )
    
    @pytest.mark.asyncio
    async def test_performance_based_strategy(self, sample_models, sample_request):
        """Test performance-based selection strategy."""
        models, capabilities, metrics = sample_models
        strategy = PerformanceBasedStrategy()
        
        selection = await strategy.select_model(
            sample_request, models, metrics, capabilities
        )
        
        # Should select a model with good performance (model_a or model_c both have high scores)
        assert selection.selected_model in ["model_a", "model_c"]
        assert selection.confidence_score > 0
        assert len(selection.alternative_models) <= 3
    
    @pytest.mark.asyncio
    async def test_cost_optimized_strategy(self, sample_models, sample_request):
        """Test cost-optimized selection strategy."""
        models, capabilities, metrics = sample_models
        strategy = CostOptimizedStrategy(min_quality_threshold=0.7)
        
        selection = await strategy.select_model(
            sample_request, models, metrics, capabilities
        )
        
        # Should select model_b (lowest cost above quality threshold)
        assert selection.selected_model == "model_b"
        assert "cost optimization" in selection.selection_reason.lower()
    
    @pytest.mark.asyncio
    async def test_quality_optimized_strategy(self, sample_models, sample_request):
        """Test quality-optimized selection strategy."""
        models, capabilities, metrics = sample_models
        strategy = QualityOptimizedStrategy()
        
        selection = await strategy.select_model(
            sample_request, models, metrics, capabilities
        )
        
        # Should select model_c (highest quality score)
        assert selection.selected_model == "model_c"
        assert "highest quality" in selection.selection_reason.lower()
    
    @pytest.mark.asyncio
    async def test_no_compatible_models(self, sample_models, sample_request):
        """Test handling when no compatible models are found."""
        models, capabilities, metrics = sample_models
        strategy = PerformanceBasedStrategy()
        
        # Request incompatible format
        sample_request.format = "gif"
        
        with pytest.raises(ValueError, match="No compatible models found"):
            await strategy.select_model(
                sample_request, models, metrics, capabilities
            )


class TestABTestingFramework:
    """Test A/B testing framework."""
    
    @pytest.fixture
    def ab_framework(self):
        """Create A/B testing framework."""
        return ABTestingFramework(test_duration_hours=1)
    
    @pytest.fixture
    def sample_request(self):
        """Create sample request for testing."""
        return GenerationRequest(
            request_id="test_123",
            model_type=ModelType.IMAGE_GENERATION,
            prompt="Test prompt"
        )
    
    @pytest.mark.asyncio
    async def test_create_ab_test(self, ab_framework):
        """Test creating A/B test."""
        success = await ab_framework.create_test(
            test_id="test_1",
            model_a="model_a",
            model_b="model_b",
            traffic_split=0.6
        )
        
        assert success
        assert "test_1" in ab_framework.active_tests
        
        # Test duplicate creation
        duplicate = await ab_framework.create_test(
            test_id="test_1",
            model_a="model_c",
            model_b="model_d"
        )
        assert not duplicate
    
    @pytest.mark.asyncio
    async def test_test_assignment(self, ab_framework, sample_request):
        """Test model assignment in A/B test."""
        await ab_framework.create_test(
            test_id="test_1",
            model_a="model_a",
            model_b="model_b",
            traffic_split=0.5
        )
        
        # Get multiple assignments to test distribution
        assignments = []
        for _ in range(100):
            assignment = await ab_framework.get_test_assignment("test_1", sample_request)
            assignments.append(assignment)
        
        # Should get both models
        assert "model_a" in assignments
        assert "model_b" in assignments
        
        # Test non-existent test
        no_assignment = await ab_framework.get_test_assignment("nonexistent", sample_request)
        assert no_assignment is None
    
    @pytest.mark.asyncio
    async def test_record_results(self, ab_framework):
        """Test recording test results."""
        await ab_framework.create_test(
            test_id="test_1",
            model_a="model_a",
            model_b="model_b"
        )
        
        # Record some results
        await ab_framework.record_test_result(
            "test_1",
            "model_a",
            {QualityMetric.OVERALL_SCORE: 0.8, QualityMetric.PROCESSING_TIME: 25.0}
        )
        
        await ab_framework.record_test_result(
            "test_1",
            "model_b",
            {QualityMetric.OVERALL_SCORE: 0.75, QualityMetric.PROCESSING_TIME: 20.0}
        )
        
        # Check test status
        status = await ab_framework.get_test_status("test_1")
        assert status["status"] == "active"
        assert "preliminary_results" in status
    
    @pytest.mark.asyncio
    async def test_test_expiration(self, ab_framework, sample_request):
        """Test test expiration and finalization."""
        # Create test with very short duration
        ab_framework.test_duration = timedelta(seconds=0.1)
        
        await ab_framework.create_test(
            test_id="test_1",
            model_a="model_a",
            model_b="model_b"
        )
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Assignment should return None and finalize test
        assignment = await ab_framework.get_test_assignment("test_1", sample_request)
        assert assignment is None
        assert "test_1" not in ab_framework.active_tests
        assert "test_1" in ab_framework.test_results


class TestModelSelector:
    """Test ModelSelector main class."""
    
    @pytest.fixture
    async def model_selector(self):
        """Create model selector with sample models."""
        selector = ModelSelector()
        
        # Register test models
        await selector.register_model(ModelCapabilities(
            model_id="fast_model",
            model_type=ModelType.IMAGE_GENERATION,
            supported_resolutions=[(1024, 1024)],
            supported_formats=["jpg"],
            max_prompt_length=1000,
            cost_per_generation=0.02,
            estimated_processing_time=10.0
        ))
        
        await selector.register_model(ModelCapabilities(
            model_id="quality_model",
            model_type=ModelType.IMAGE_GENERATION,
            supported_resolutions=[(1024, 1024)],
            supported_formats=["jpg"],
            max_prompt_length=1000,
            cost_per_generation=0.08,
            estimated_processing_time=45.0
        ))
        
        # Add some performance data
        await selector.update_model_metrics(
            "fast_model", 12.0, 0.7, 0.02, True, 0.75
        )
        await selector.update_model_metrics(
            "quality_model", 40.0, 0.95, 0.08, True, 0.9
        )
        
        return selector
    
    @pytest.fixture
    def sample_request(self):
        """Create sample request."""
        return GenerationRequest(
            request_id="test_123",
            model_type=ModelType.IMAGE_GENERATION,
            prompt="Test prompt",
            resolution=(1024, 1024),
            format="jpg"
        )
    
    @pytest.mark.asyncio
    async def test_model_registration(self, model_selector):
        """Test model registration."""
        assert "fast_model" in model_selector.model_capabilities
        assert "quality_model" in model_selector.model_capabilities
        assert len(model_selector.model_capabilities) == 2
    
    @pytest.mark.asyncio
    async def test_model_selection_strategies(self, model_selector, sample_request):
        """Test different selection strategies."""
        # Performance strategy
        perf_selection = await model_selector.select_model(sample_request, "performance")
        assert perf_selection.selected_model in ["fast_model", "quality_model"]
        
        # Cost strategy
        cost_selection = await model_selector.select_model(sample_request, "cost")
        assert cost_selection.selected_model == "fast_model"  # Cheaper model
        
        # Quality strategy
        quality_selection = await model_selector.select_model(sample_request, "quality")
        assert quality_selection.selected_model == "quality_model"  # Higher quality
    
    @pytest.mark.asyncio
    async def test_metrics_update(self, model_selector):
        """Test updating model metrics."""
        initial_metrics = model_selector.model_metrics["fast_model"]
        initial_generations = initial_metrics.total_generations
        
        await model_selector.update_model_metrics(
            "fast_model", 15.0, 0.8, 0.025, True, 0.85
        )
        
        updated_metrics = model_selector.model_metrics["fast_model"]
        assert updated_metrics.total_generations == initial_generations + 1
        assert updated_metrics.successful_generations == initial_metrics.successful_generations + 1
    
    @pytest.mark.asyncio
    async def test_model_rankings(self, model_selector):
        """Test getting model rankings."""
        rankings = await model_selector.get_model_rankings(
            ModelType.IMAGE_GENERATION,
            QualityMetric.OVERALL_SCORE
        )
        
        assert len(rankings) == 2
        assert rankings[0][0] == "quality_model"  # Should rank higher
        assert rankings[0][1] > rankings[1][1]  # Higher score
    
    @pytest.mark.asyncio
    async def test_ab_testing_integration(self, model_selector, sample_request):
        """Test A/B testing integration."""
        # Create A/B test
        success = await model_selector.create_ab_test(
            "test_1", "fast_model", "quality_model"
        )
        assert success
        
        # Selection should use A/B test
        selection = await model_selector.select_model(sample_request)
        assert selection.selected_model in ["fast_model", "quality_model"]
        assert "A/B test" in selection.selection_reason
        
        # Check test status
        status = await model_selector.get_ab_test_status("test_1")
        assert status["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_selection_analytics(self, model_selector, sample_request):
        """Test selection analytics."""
        # Make several selections
        for _ in range(5):
            await model_selector.select_model(sample_request, "performance")
        
        analytics = await model_selector.get_selection_analytics()
        assert analytics["total_selections"] >= 5
        assert "model_usage" in analytics
        assert "strategy_usage" in analytics
    
    @pytest.mark.asyncio
    async def test_no_models_registered(self):
        """Test behavior with no models registered."""
        empty_selector = ModelSelector()
        request = GenerationRequest(
            request_id="test",
            model_type=ModelType.IMAGE_GENERATION,
            prompt="Test"
        )
        
        with pytest.raises(ValueError, match="No models registered"):
            await empty_selector.select_model(request)


class TestUtilityFunctions:
    """Test utility functions."""
    
    @pytest.mark.asyncio
    async def test_initialize_default_models(self):
        """Test initializing default models."""
        selector = await initialize_default_models()
        
        assert len(selector.model_capabilities) >= 3
        assert "dalle3" in selector.model_capabilities
        assert "stable_diffusion_xl" in selector.model_capabilities
        assert "midjourney" in selector.model_capabilities
        
        # Check that models have metrics
        for model_id in selector.model_capabilities:
            assert model_id in selector.model_metrics
            metrics = selector.model_metrics[model_id]
            assert metrics.total_generations > 0


@pytest.mark.asyncio
async def test_model_selection_end_to_end():
    """Test complete model selection workflow."""
    # Initialize selector with default models
    selector = await initialize_default_models()
    
    # Create test request
    request = GenerationRequest(
        request_id="e2e_test",
        model_type=ModelType.IMAGE_GENERATION,
        prompt="A beautiful landscape",
        resolution=(1024, 1024),
        quality_preference="balanced"
    )
    
    # Test selection
    selection = await selector.select_model(request, "performance")
    assert selection.selected_model in selector.model_capabilities
    assert selection.confidence_score > 0
    assert selection.estimated_cost > 0
    assert selection.estimated_time > 0
    
    # Simulate generation result and update metrics
    await selector.update_model_metrics(
        model_id=selection.selected_model,
        processing_time=selection.estimated_time * 0.9,  # Slightly faster
        quality_score=0.85,
        cost=selection.estimated_cost,
        success=True,
        user_satisfaction=0.8
    )
    
    # Check updated metrics
    updated_metrics = selector.model_metrics[selection.selected_model]
    assert updated_metrics.total_generations > 0
    assert updated_metrics.successful_generations > 0


@pytest.mark.asyncio
async def test_concurrent_selections():
    """Test concurrent model selections."""
    selector = await initialize_default_models()
    
    async def make_selection(i):
        request = GenerationRequest(
            request_id=f"concurrent_{i}",
            model_type=ModelType.IMAGE_GENERATION,
            prompt=f"Test prompt {i}"
        )
        return await selector.select_model(request)
    
    # Make concurrent selections
    tasks = [make_selection(i) for i in range(10)]
    selections = await asyncio.gather(*tasks)
    
    # All should succeed
    assert len(selections) == 10
    for selection in selections:
        assert selection.selected_model in selector.model_capabilities


@pytest.mark.asyncio
async def test_model_selector_error_handling():
    """Test error handling in model selector."""
    selector = ModelSelector()
    
    # Test with incompatible request
    request = GenerationRequest(
        request_id="error_test",
        model_type=ModelType.VIDEO_GENERATION,  # No video models registered
        prompt="Test video"
    )
    
    with pytest.raises(ValueError):
        await selector.select_model(request)