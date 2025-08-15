"""
Comprehensive tests for the ImageGenerationPipeline.

Tests cover pipeline orchestration, model selection, result aggregation,
quality comparison, and error handling scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from PIL import Image
import tempfile
import os

from scrollintel.engines.visual_generation.pipeline import (
    ImageGenerationPipeline,
    ModelSelector,
    ResultAggregator,
    ModelSelectionStrategy,
    ModelCapability,
    ModelSelectionResult,
    GenerationComparison
)
from scrollintel.engines.visual_generation.base import (
    ImageGenerationRequest,
    GenerationResult,
    GenerationStatus,
    QualityMetrics
)
from scrollintel.engines.visual_generation.config import VisualGenerationConfig
from scrollintel.engines.visual_generation.exceptions import (
    ModelError,
    ResourceError,
    ValidationError,
    ConfigurationError
)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config_data = {
        'models': {
            'stable_diffusion_xl': {
                'type': 'image',
                'enabled': True,
                'api_key': 'test_key',
                'parameters': {}
            },
            'dalle3': {
                'type': 'image',
                'enabled': True,
                'api_key': 'test_key',
                'parameters': {}
            },
            'midjourney': {
                'type': 'image',
                'enabled': True,
                'api_key': 'test_key',
                'parameters': {}
            }
        },
        'infrastructure': {
            'gpu_enabled': True,
            'max_concurrent_requests': 10,
            'cache_enabled': True,
            'storage_path': './test_generated_content',
            'temp_path': './test_temp'
        },
        'safety': {
            'enabled': True,
            'nsfw_detection': True
        },
        'quality': {
            'enabled': True,
            'min_quality_score': 0.7
        },
        'cost': {
            'enabled': True,
            'cost_per_image': 0.01
        }
    }
    return VisualGenerationConfig(config_data)


@pytest.fixture
def sample_request():
    """Create a sample image generation request."""
    return ImageGenerationRequest(
        prompt="A beautiful sunset over mountains",
        user_id="test_user",
        resolution=(1024, 1024),
        num_images=1,
        style="photorealistic",
        quality="high"
    )


@pytest.fixture
def mock_image():
    """Create a mock PIL Image."""
    image = Image.new('RGB', (1024, 1024), color='red')
    return image


@pytest.fixture
def sample_generation_result(mock_image):
    """Create a sample generation result."""
    # Create temporary file for the image
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    mock_image.save(temp_file.name)
    temp_file.close()
    
    return GenerationResult(
        id="test_result_1",
        request_id="test_request_1",
        status=GenerationStatus.COMPLETED,
        content_paths=[temp_file.name],
        quality_metrics=QualityMetrics(
            overall_score=0.85,
            technical_quality=0.9,
            aesthetic_score=0.8,
            prompt_adherence=0.85
        ),
        generation_time=25.0,
        cost=0.02,
        model_used="stable_diffusion_xl"
    )


class TestModelSelector:
    """Test the ModelSelector class."""
    
    def test_model_selector_initialization(self, mock_config):
        """Test ModelSelector initialization."""
        selector = ModelSelector(mock_config)
        
        assert selector.config == mock_config
        assert len(selector.model_capabilities) > 0
        assert 'stable_diffusion_xl' in selector.model_capabilities
        assert isinstance(selector.selection_history, list)
    
    def test_model_capabilities_structure(self, mock_config):
        """Test that model capabilities are properly structured."""
        selector = ModelSelector(mock_config)
        
        for model_name, capability in selector.model_capabilities.items():
            assert isinstance(capability, ModelCapability)
            assert capability.name == model_name
            assert 0 <= capability.quality_score <= 1
            assert 0 <= capability.speed_score <= 1
            assert 0 <= capability.cost_score <= 1
            assert isinstance(capability.style_compatibility, dict)
            assert isinstance(capability.resolution_support, list)
            assert capability.max_batch_size > 0
    
    @pytest.mark.asyncio
    async def test_select_best_quality_strategy(self, mock_config, sample_request):
        """Test best quality selection strategy."""
        selector = ModelSelector(mock_config)
        
        selection = await selector.select_models(
            sample_request, 
            ModelSelectionStrategy.BEST_QUALITY
        )
        
        assert isinstance(selection, ModelSelectionResult)
        assert len(selection.selected_models) == 1
        assert "quality" in selection.selection_reason.lower()
        assert selection.confidence_score > 0
        assert len(selection.fallback_models) >= 0
    
    @pytest.mark.asyncio
    async def test_select_fastest_strategy(self, mock_config, sample_request):
        """Test fastest selection strategy."""
        selector = ModelSelector(mock_config)
        
        selection = await selector.select_models(
            sample_request, 
            ModelSelectionStrategy.FASTEST
        )
        
        assert isinstance(selection, ModelSelectionResult)
        assert len(selection.selected_models) == 1
        assert "fastest" in selection.selection_reason.lower()
    
    @pytest.mark.asyncio
    async def test_select_cost_effective_strategy(self, mock_config, sample_request):
        """Test cost-effective selection strategy."""
        selector = ModelSelector(mock_config)
        
        selection = await selector.select_models(
            sample_request, 
            ModelSelectionStrategy.MOST_COST_EFFECTIVE
        )
        
        assert isinstance(selection, ModelSelectionResult)
        assert len(selection.selected_models) == 1
        assert "cost" in selection.selection_reason.lower()
    
    @pytest.mark.asyncio
    async def test_select_balanced_strategy(self, mock_config, sample_request):
        """Test balanced selection strategy."""
        selector = ModelSelector(mock_config)
        
        selection = await selector.select_models(
            sample_request, 
            ModelSelectionStrategy.BALANCED
        )
        
        assert isinstance(selection, ModelSelectionResult)
        assert len(selection.selected_models) == 1
        assert "balanced" in selection.selection_reason.lower()
    
    @pytest.mark.asyncio
    async def test_select_all_models_strategy(self, mock_config, sample_request):
        """Test all models selection strategy."""
        selector = ModelSelector(mock_config)
        
        selection = await selector.select_models(
            sample_request, 
            ModelSelectionStrategy.ALL_MODELS
        )
        
        assert isinstance(selection, ModelSelectionResult)
        assert len(selection.selected_models) > 1
        assert "all" in selection.selection_reason.lower()
    
    @pytest.mark.asyncio
    async def test_user_preference_strategy(self, mock_config, sample_request):
        """Test user preference selection strategy."""
        sample_request.model_preference = "dalle3"
        selector = ModelSelector(mock_config)
        
        selection = await selector.select_models(
            sample_request, 
            ModelSelectionStrategy.USER_PREFERENCE
        )
        
        assert isinstance(selection, ModelSelectionResult)
        assert "dalle3" in selection.selected_models
        assert "preference" in selection.selection_reason.lower()
    
    @pytest.mark.asyncio
    async def test_no_available_models_error(self, mock_config):
        """Test error when no models are available."""
        # Create request with unsupported parameters
        request = ImageGenerationRequest(
            prompt="test",
            user_id="test",
            resolution=(10000, 10000),  # Unsupported resolution
            style="unsupported_style"
        )
        
        selector = ModelSelector(mock_config)
        
        with pytest.raises(ResourceError, match="No suitable models available"):
            await selector.select_models(request)
    
    def test_resolution_support_check(self, mock_config):
        """Test resolution support checking."""
        selector = ModelSelector(mock_config)
        capability = selector.model_capabilities['stable_diffusion_xl']
        
        # Test supported resolution
        assert selector._supports_resolution(capability, (1024, 1024))
        
        # Test close resolution (within tolerance)
        assert selector._supports_resolution(capability, (1000, 1000))
        
        # Test unsupported resolution
        assert not selector._supports_resolution(capability, (5000, 5000))
    
    @pytest.mark.asyncio
    async def test_cost_estimation(self, mock_config, sample_request):
        """Test cost estimation for selected models."""
        selector = ModelSelector(mock_config)
        
        # Test single model cost
        cost = await selector._estimate_total_cost(['stable_diffusion_xl'], sample_request)
        assert cost > 0
        
        # Test multiple models cost
        multi_cost = await selector._estimate_total_cost(
            ['stable_diffusion_xl', 'dalle3'], sample_request
        )
        assert multi_cost > cost
    
    @pytest.mark.asyncio
    async def test_time_estimation(self, mock_config, sample_request):
        """Test time estimation for selected models."""
        selector = ModelSelector(mock_config)
        
        # Test single model time
        time_estimate = await selector._estimate_total_time(['stable_diffusion_xl'], sample_request)
        assert time_estimate > 0
        
        # Test multiple models time (should be max, not sum)
        multi_time = await selector._estimate_total_time(
            ['stable_diffusion_xl', 'dalle3'], sample_request
        )
        assert multi_time >= time_estimate


class TestResultAggregator:
    """Test the ResultAggregator class."""
    
    def test_result_aggregator_initialization(self, mock_config):
        """Test ResultAggregator initialization."""
        aggregator = ResultAggregator(mock_config)
        assert aggregator.config == mock_config
    
    @pytest.mark.asyncio
    async def test_aggregate_single_result(self, mock_config, sample_generation_result, sample_request):
        """Test aggregating a single successful result."""
        aggregator = ResultAggregator(mock_config)
        results = {'stable_diffusion_xl': sample_generation_result}
        
        comparison = await aggregator.aggregate_results(results, sample_request)
        
        assert isinstance(comparison, GenerationComparison)
        assert comparison.best_result == sample_generation_result
        assert len(comparison.quality_rankings) == 1
        assert comparison.quality_rankings[0][0] == 'stable_diffusion_xl'
        assert len(comparison.cost_comparison) == 1
        assert len(comparison.time_comparison) == 1
        assert isinstance(comparison.recommendation, str)
    
    @pytest.mark.asyncio
    async def test_aggregate_multiple_results(self, mock_config, sample_request, mock_image):
        """Test aggregating multiple results."""
        aggregator = ResultAggregator(mock_config)
        
        # Create multiple results with different quality scores
        result1 = GenerationResult(
            id="result1", request_id="req1", status=GenerationStatus.COMPLETED,
            quality_metrics=QualityMetrics(overall_score=0.8),
            generation_time=20.0, cost=0.01, model_used="stable_diffusion_xl"
        )
        
        result2 = GenerationResult(
            id="result2", request_id="req1", status=GenerationStatus.COMPLETED,
            quality_metrics=QualityMetrics(overall_score=0.9),
            generation_time=30.0, cost=0.04, model_used="dalle3"
        )
        
        results = {
            'stable_diffusion_xl': result1,
            'dalle3': result2
        }
        
        comparison = await aggregator.aggregate_results(results, sample_request)
        
        assert comparison.best_result.model_used == "dalle3"  # Higher quality
        assert len(comparison.quality_rankings) == 2
        assert comparison.quality_rankings[0][0] == "dalle3"  # Best quality first
        assert comparison.quality_rankings[0][1] == 0.9
    
    @pytest.mark.asyncio
    async def test_aggregate_with_failed_results(self, mock_config, sample_request):
        """Test aggregating results with some failures."""
        aggregator = ResultAggregator(mock_config)
        
        successful_result = GenerationResult(
            id="result1", request_id="req1", status=GenerationStatus.COMPLETED,
            quality_metrics=QualityMetrics(overall_score=0.8),
            generation_time=20.0, cost=0.01, model_used="stable_diffusion_xl"
        )
        
        failed_result = GenerationResult(
            id="result2", request_id="req1", status=GenerationStatus.FAILED,
            error_message="Model failed", model_used="dalle3"
        )
        
        results = {
            'stable_diffusion_xl': successful_result,
            'dalle3': failed_result
        }
        
        comparison = await aggregator.aggregate_results(results, sample_request)
        
        # Should only include successful results
        assert len(comparison.results) == 1
        assert 'stable_diffusion_xl' in comparison.results
        assert 'dalle3' not in comparison.results
        assert comparison.best_result == successful_result
    
    @pytest.mark.asyncio
    async def test_aggregate_all_failed_results(self, mock_config, sample_request):
        """Test aggregating when all results failed."""
        aggregator = ResultAggregator(mock_config)
        
        failed_result1 = GenerationResult(
            id="result1", request_id="req1", status=GenerationStatus.FAILED,
            error_message="Model 1 failed", model_used="stable_diffusion_xl"
        )
        
        failed_result2 = GenerationResult(
            id="result2", request_id="req1", status=GenerationStatus.FAILED,
            error_message="Model 2 failed", model_used="dalle3"
        )
        
        results = {
            'stable_diffusion_xl': failed_result1,
            'dalle3': failed_result2
        }
        
        with pytest.raises(ModelError, match="All model generations failed"):
            await aggregator.aggregate_results(results, sample_request)
    
    @pytest.mark.asyncio
    async def test_aggregate_empty_results(self, mock_config, sample_request):
        """Test aggregating empty results."""
        aggregator = ResultAggregator(mock_config)
        
        with pytest.raises(ValidationError, match="No results to aggregate"):
            await aggregator.aggregate_results({}, sample_request)
    
    @pytest.mark.asyncio
    async def test_quality_estimation_without_metrics(self, mock_config, sample_request):
        """Test quality estimation when metrics are not available."""
        aggregator = ResultAggregator(mock_config)
        
        result_without_metrics = GenerationResult(
            id="result1", request_id="req1", status=GenerationStatus.COMPLETED,
            quality_metrics=None,  # No metrics
            generation_time=20.0, cost=0.01, model_used="stable_diffusion_xl"
        )
        
        estimated_score = await aggregator._estimate_quality_score(
            "stable_diffusion_xl", result_without_metrics, sample_request
        )
        
        assert 0 <= estimated_score <= 1
        assert estimated_score > 0  # Should have some base score


class TestImageGenerationPipeline:
    """Test the main ImageGenerationPipeline class."""
    
    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing."""
        mock_sdxl = AsyncMock()
        mock_sdxl.generate_image = AsyncMock()
        mock_sdxl.validate_request = AsyncMock(return_value=True)
        mock_sdxl.get_capabilities = Mock(return_value={"model_name": "stable_diffusion_xl"})
        mock_sdxl.cleanup = AsyncMock()
        
        mock_dalle3 = AsyncMock()
        mock_dalle3.generate_image = AsyncMock()
        mock_dalle3.validate_request = AsyncMock(return_value=True)
        mock_dalle3.get_capabilities = Mock(return_value={"model_name": "dalle3"})
        mock_dalle3.cleanup = AsyncMock()
        
        return {
            'stable_diffusion_xl': mock_sdxl,
            'dalle3': mock_dalle3
        }
    
    def test_pipeline_initialization(self, mock_config):
        """Test pipeline initialization."""
        pipeline = ImageGenerationPipeline(mock_config)
        
        assert pipeline.config == mock_config
        assert isinstance(pipeline.model_selector, ModelSelector)
        assert isinstance(pipeline.result_aggregator, ResultAggregator)
        assert pipeline.default_strategy == ModelSelectionStrategy.BALANCED
        assert not pipeline.is_initialized
    
    @pytest.mark.asyncio
    async def test_pipeline_initialize_success(self, mock_config):
        """Test successful pipeline initialization."""
        with patch('scrollintel.engines.visual_generation.pipeline.StableDiffusionXLModel') as mock_sdxl_class, \
             patch('scrollintel.engines.visual_generation.pipeline.DALLE3Model') as mock_dalle3_class, \
             patch('scrollintel.engines.visual_generation.pipeline.MidjourneyModel') as mock_mj_class:
            
            # Setup mocks
            mock_sdxl = AsyncMock()
            mock_sdxl.initialize = AsyncMock()
            mock_sdxl_class.return_value = mock_sdxl
            
            mock_dalle3 = AsyncMock()
            mock_dalle3_class.return_value = mock_dalle3
            
            mock_mj = AsyncMock()
            mock_mj_class.return_value = mock_mj
            
            pipeline = ImageGenerationPipeline(mock_config)
            await pipeline.initialize()
            
            assert pipeline.is_initialized
            assert len(pipeline.models) > 0
            mock_sdxl.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pipeline_initialize_no_models(self, mock_config):
        """Test pipeline initialization when no models can be initialized."""
        # Mock all models to fail initialization
        with patch('scrollintel.engines.visual_generation.pipeline.StableDiffusionXLModel', side_effect=Exception("Init failed")), \
             patch('scrollintel.engines.visual_generation.pipeline.DALLE3Model', side_effect=Exception("Init failed")), \
             patch('scrollintel.engines.visual_generation.pipeline.MidjourneyModel', side_effect=Exception("Init failed")):
            
            pipeline = ImageGenerationPipeline(mock_config)
            
            with pytest.raises(ConfigurationError, match="No image generation models could be initialized"):
                await pipeline.initialize()
    
    @pytest.mark.asyncio
    async def test_single_model_generation(self, mock_config, sample_request, sample_generation_result, mock_models):
        """Test generation with a single model."""
        pipeline = ImageGenerationPipeline(mock_config)
        pipeline.models = mock_models
        pipeline.is_initialized = True
        
        # Mock model selector to return single model
        mock_selection = ModelSelectionResult(
            selected_models=['stable_diffusion_xl'],
            selection_reason="Test selection",
            confidence_score=0.9,
            fallback_models=[],
            estimated_cost=0.02,
            estimated_time=25.0
        )
        
        with patch.object(pipeline.model_selector, 'select_models', return_value=mock_selection):
            mock_models['stable_diffusion_xl'].generate_image.return_value = sample_generation_result
            
            result = await pipeline.generate_image(sample_request)
            
            assert result.status == GenerationStatus.COMPLETED
            assert 'pipeline_version' in result.metadata
            assert 'selection_strategy' in result.metadata
            mock_models['stable_diffusion_xl'].generate_image.assert_called_once_with(sample_request)
    
    @pytest.mark.asyncio
    async def test_multi_model_generation(self, mock_config, sample_request, mock_models):
        """Test generation with multiple models."""
        pipeline = ImageGenerationPipeline(mock_config)
        pipeline.models = mock_models
        pipeline.is_initialized = True
        
        # Create different results for each model
        result1 = GenerationResult(
            id="result1", request_id=sample_request.request_id, 
            status=GenerationStatus.COMPLETED,
            quality_metrics=QualityMetrics(overall_score=0.8),
            generation_time=20.0, cost=0.01, model_used="stable_diffusion_xl"
        )
        
        result2 = GenerationResult(
            id="result2", request_id=sample_request.request_id,
            status=GenerationStatus.COMPLETED,
            quality_metrics=QualityMetrics(overall_score=0.9),
            generation_time=30.0, cost=0.04, model_used="dalle3"
        )
        
        mock_models['stable_diffusion_xl'].generate_image.return_value = result1
        mock_models['dalle3'].generate_image.return_value = result2
        
        # Mock model selector to return multiple models
        mock_selection = ModelSelectionResult(
            selected_models=['stable_diffusion_xl', 'dalle3'],
            selection_reason="Test multi-model selection",
            confidence_score=0.8,
            fallback_models=[],
            estimated_cost=0.05,
            estimated_time=30.0
        )
        
        with patch.object(pipeline.model_selector, 'select_models', return_value=mock_selection):
            result = await pipeline.generate_image(sample_request)
            
            assert result.status == GenerationStatus.COMPLETED
            assert result.model_used == "dalle3"  # Should pick the higher quality result
            assert 'multi_model_comparison' in result.metadata
            assert 'quality_rankings' in result.metadata
            
            # Both models should have been called
            mock_models['stable_diffusion_xl'].generate_image.assert_called_once()
            mock_models['dalle3'].generate_image.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_model_usage(self, mock_config, sample_request, sample_generation_result, mock_models):
        """Test fallback model usage when primary model fails."""
        pipeline = ImageGenerationPipeline(mock_config)
        pipeline.models = mock_models
        pipeline.is_initialized = True
        
        # Mock primary model to fail, fallback to succeed
        mock_models['stable_diffusion_xl'].generate_image.side_effect = Exception("Primary model failed")
        mock_models['dalle3'].generate_image.return_value = sample_generation_result
        
        mock_selection = ModelSelectionResult(
            selected_models=['stable_diffusion_xl'],
            selection_reason="Test selection",
            confidence_score=0.9,
            fallback_models=['dalle3'],
            estimated_cost=0.02,
            estimated_time=25.0
        )
        
        with patch.object(pipeline.model_selector, 'select_models', return_value=mock_selection):
            result = await pipeline.generate_image(sample_request)
            
            assert result.status == GenerationStatus.COMPLETED
            assert result.metadata.get('fallback_used') is True
            assert result.metadata.get('original_model') == 'stable_diffusion_xl'
            assert result.metadata.get('fallback_model') == 'dalle3'
    
    @pytest.mark.asyncio
    async def test_all_models_fail(self, mock_config, sample_request, mock_models):
        """Test behavior when all models fail."""
        pipeline = ImageGenerationPipeline(mock_config)
        pipeline.models = mock_models
        pipeline.is_initialized = True
        
        # Mock all models to fail
        for model in mock_models.values():
            model.generate_image.side_effect = Exception("Model failed")
        
        mock_selection = ModelSelectionResult(
            selected_models=['stable_diffusion_xl'],
            selection_reason="Test selection",
            confidence_score=0.9,
            fallback_models=['dalle3'],
            estimated_cost=0.02,
            estimated_time=25.0
        )
        
        with patch.object(pipeline.model_selector, 'select_models', return_value=mock_selection):
            result = await pipeline.generate_image(sample_request)
            
            assert result.status == GenerationStatus.FAILED
            assert "All models failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_validate_request(self, mock_config, sample_request, mock_models):
        """Test request validation."""
        pipeline = ImageGenerationPipeline(mock_config)
        pipeline.models = mock_models
        pipeline.is_initialized = True
        
        # Test valid request
        is_valid = await pipeline.validate_request(sample_request)
        assert is_valid
        
        # Test invalid request type
        invalid_request = "not a request object"
        is_valid = await pipeline.validate_request(invalid_request)
        assert not is_valid
    
    @pytest.mark.asyncio
    async def test_cost_estimation(self, mock_config, sample_request, mock_models):
        """Test cost estimation."""
        pipeline = ImageGenerationPipeline(mock_config)
        pipeline.models = mock_models
        pipeline.is_initialized = True
        
        mock_selection = ModelSelectionResult(
            selected_models=['stable_diffusion_xl'],
            selection_reason="Test selection",
            confidence_score=0.9,
            fallback_models=[],
            estimated_cost=0.025,
            estimated_time=25.0
        )
        
        with patch.object(pipeline.model_selector, 'select_models', return_value=mock_selection):
            cost = await pipeline.estimate_cost(sample_request)
            assert cost == 0.025
    
    @pytest.mark.asyncio
    async def test_time_estimation(self, mock_config, sample_request, mock_models):
        """Test time estimation."""
        pipeline = ImageGenerationPipeline(mock_config)
        pipeline.models = mock_models
        pipeline.is_initialized = True
        
        mock_selection = ModelSelectionResult(
            selected_models=['stable_diffusion_xl'],
            selection_reason="Test selection",
            confidence_score=0.9,
            fallback_models=[],
            estimated_cost=0.025,
            estimated_time=30.0
        )
        
        with patch.object(pipeline.model_selector, 'select_models', return_value=mock_selection):
            time_estimate = await pipeline.estimate_time(sample_request)
            assert time_estimate == 30.0
    
    def test_get_capabilities(self, mock_config, mock_models):
        """Test getting pipeline capabilities."""
        pipeline = ImageGenerationPipeline(mock_config)
        pipeline.models = mock_models
        pipeline.is_initialized = True
        
        capabilities = pipeline.get_capabilities()
        
        assert capabilities['pipeline_name'] == 'ImageGenerationPipeline'
        assert 'available_models' in capabilities
        assert 'model_capabilities' in capabilities
        assert 'selection_strategies' in capabilities
        assert 'features' in capabilities
        assert isinstance(capabilities['features'], list)
    
    @pytest.mark.asyncio
    async def test_get_model_status(self, mock_config, mock_models):
        """Test getting model status."""
        pipeline = ImageGenerationPipeline(mock_config)
        pipeline.models = mock_models
        pipeline.is_initialized = True
        
        # Mock get_model_info for models
        mock_models['stable_diffusion_xl'].get_model_info = AsyncMock(
            return_value={"name": "Stable Diffusion XL", "version": "1.0"}
        )
        mock_models['dalle3'].get_model_info = AsyncMock(
            return_value={"name": "DALL-E 3", "version": "1.0"}
        )
        
        status = await pipeline.get_model_status()
        
        assert 'stable_diffusion_xl' in status
        assert 'dalle3' in status
        assert status['stable_diffusion_xl']['available'] is True
        assert status['dalle3']['available'] is True
    
    @pytest.mark.asyncio
    async def test_cleanup(self, mock_config, mock_models):
        """Test pipeline cleanup."""
        pipeline = ImageGenerationPipeline(mock_config)
        pipeline.models = mock_models
        pipeline.is_initialized = True
        
        await pipeline.cleanup()
        
        assert not pipeline.is_initialized
        assert len(pipeline.models) == 0
        
        # Verify cleanup was called on all models
        for model in mock_models.values():
            model.cleanup.assert_called_once()
    
    def test_strategy_determination(self, mock_config):
        """Test strategy determination logic."""
        pipeline = ImageGenerationPipeline(mock_config)
        
        # Test user preference
        request_with_preference = ImageGenerationRequest(
            prompt="test", user_id="test", model_preference="dalle3"
        )
        strategy = pipeline._determine_strategy(request_with_preference)
        assert strategy == ModelSelectionStrategy.USER_PREFERENCE
        
        # Test quality-focused
        request_ultra_quality = ImageGenerationRequest(
            prompt="test", user_id="test", quality="ultra"
        )
        strategy = pipeline._determine_strategy(request_ultra_quality)
        assert strategy == ModelSelectionStrategy.BEST_QUALITY
        
        # Test speed priority
        request_speed = ImageGenerationRequest(
            prompt="test", user_id="test", metadata={"priority": "speed"}
        )
        strategy = pipeline._determine_strategy(request_speed)
        assert strategy == ModelSelectionStrategy.FASTEST
        
        # Test cost priority
        request_cost = ImageGenerationRequest(
            prompt="test", user_id="test", metadata={"priority": "cost"}
        )
        strategy = pipeline._determine_strategy(request_cost)
        assert strategy == ModelSelectionStrategy.MOST_COST_EFFECTIVE
        
        # Test comparison request
        request_compare = ImageGenerationRequest(
            prompt="test", user_id="test", metadata={"compare_models": True}
        )
        strategy = pipeline._determine_strategy(request_compare)
        assert strategy == ModelSelectionStrategy.ALL_MODELS
        
        # Test default
        request_default = ImageGenerationRequest(
            prompt="test", user_id="test"
        )
        strategy = pipeline._determine_strategy(request_default)
        assert strategy == ModelSelectionStrategy.BALANCED
    
    @pytest.mark.asyncio
    async def test_benchmark_models(self, mock_config, mock_models):
        """Test model benchmarking functionality."""
        pipeline = ImageGenerationPipeline(mock_config)
        pipeline.models = mock_models
        pipeline.is_initialized = True
        
        # Mock successful results
        mock_result = GenerationResult(
            id="benchmark_result", request_id="benchmark",
            status=GenerationStatus.COMPLETED,
            quality_metrics=QualityMetrics(overall_score=0.8),
            generation_time=20.0, cost=0.01
        )
        
        for model in mock_models.values():
            model.generate_image.return_value = mock_result
        
        test_prompts = ["A cat", "A dog"]
        benchmark_results = await pipeline.benchmark_models(test_prompts)
        
        assert len(benchmark_results) == 2
        for prompt in test_prompts:
            assert prompt in benchmark_results
            assert 'stable_diffusion_xl' in benchmark_results[prompt]
            assert 'dalle3' in benchmark_results[prompt]
            assert benchmark_results[prompt]['stable_diffusion_xl']['success'] is True


# Integration tests
class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_single_model_flow(self, mock_config, sample_request):
        """Test complete end-to-end flow with single model."""
        with patch('scrollintel.engines.visual_generation.pipeline.StableDiffusionXLModel') as mock_sdxl_class:
            # Setup mock model
            mock_sdxl = AsyncMock()
            mock_sdxl.initialize = AsyncMock()
            mock_sdxl.generate_image = AsyncMock()
            mock_sdxl.validate_request = AsyncMock(return_value=True)
            mock_sdxl.get_capabilities = Mock(return_value={"model_name": "stable_diffusion_xl"})
            mock_sdxl.cleanup = AsyncMock()
            mock_sdxl_class.return_value = mock_sdxl
            
            # Create successful result
            mock_result = GenerationResult(
                id="integration_test", request_id=sample_request.request_id,
                status=GenerationStatus.COMPLETED,
                quality_metrics=QualityMetrics(overall_score=0.85),
                generation_time=25.0, cost=0.02, model_used="stable_diffusion_xl"
            )
            mock_sdxl.generate_image.return_value = mock_result
            
            # Test complete flow
            pipeline = ImageGenerationPipeline(mock_config)
            await pipeline.initialize()
            
            result = await pipeline.generate_image(sample_request)
            
            assert result.status == GenerationStatus.COMPLETED
            assert result.model_used == "stable_diffusion_xl"
            assert 'pipeline_version' in result.metadata
            
            await pipeline.cleanup()
    
    @pytest.mark.asyncio
    async def test_end_to_end_multi_model_comparison(self, mock_config, sample_request):
        """Test complete end-to-end flow with multi-model comparison."""
        with patch('scrollintel.engines.visual_generation.pipeline.StableDiffusionXLModel') as mock_sdxl_class, \
             patch('scrollintel.engines.visual_generation.pipeline.DALLE3Model') as mock_dalle3_class:
            
            # Setup mock models
            mock_sdxl = AsyncMock()
            mock_sdxl.initialize = AsyncMock()
            mock_sdxl.generate_image = AsyncMock()
            mock_sdxl.validate_request = AsyncMock(return_value=True)
            mock_sdxl.get_capabilities = Mock(return_value={"model_name": "stable_diffusion_xl"})
            mock_sdxl.cleanup = AsyncMock()
            mock_sdxl_class.return_value = mock_sdxl
            
            mock_dalle3 = AsyncMock()
            mock_dalle3.generate_image = AsyncMock()
            mock_dalle3.validate_request = AsyncMock(return_value=True)
            mock_dalle3.get_capabilities = Mock(return_value={"model_name": "dalle3"})
            mock_dalle3.cleanup = AsyncMock()
            mock_dalle3_class.return_value = mock_dalle3
            
            # Create different quality results
            result1 = GenerationResult(
                id="result1", request_id=sample_request.request_id,
                status=GenerationStatus.COMPLETED,
                quality_metrics=QualityMetrics(overall_score=0.8),
                generation_time=20.0, cost=0.01, model_used="stable_diffusion_xl"
            )
            
            result2 = GenerationResult(
                id="result2", request_id=sample_request.request_id,
                status=GenerationStatus.COMPLETED,
                quality_metrics=QualityMetrics(overall_score=0.9),
                generation_time=30.0, cost=0.04, model_used="dalle3"
            )
            
            mock_sdxl.generate_image.return_value = result1
            mock_dalle3.generate_image.return_value = result2
            
            # Request comparison
            sample_request.metadata = {"compare_models": True}
            
            pipeline = ImageGenerationPipeline(mock_config)
            await pipeline.initialize()
            
            result = await pipeline.generate_image(sample_request)
            
            assert result.status == GenerationStatus.COMPLETED
            assert result.model_used == "dalle3"  # Higher quality should win
            assert result.metadata.get('multi_model_comparison') is True
            assert 'quality_rankings' in result.metadata
            assert 'recommendation' in result.metadata
            
            await pipeline.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])