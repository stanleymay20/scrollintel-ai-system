"""
Unit tests for Stable Diffusion XL Model Integration.

Tests cover model integration, prompt preprocessing, parameter optimization,
resolution handling, and output validation.
"""

import pytest
import torch
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import numpy as np

from scrollintel.engines.visual_generation.models.stable_diffusion_xl import (
    StableDiffusionXLModel,
    StableDiffusionXLRequest,
    StableDiffusionXLResult,
    PromptPreprocessor,
    ResolutionOptimizer
)
from scrollintel.engines.visual_generation.exceptions import ModelError, PromptError
from scrollintel.engines.visual_generation.config import VisualGenerationConfig


class TestPromptPreprocessor:
    """Test prompt preprocessing functionality."""
    
    def setup_method(self):
        self.preprocessor = PromptPreprocessor()
    
    def test_enhance_prompt_photorealistic(self):
        """Test prompt enhancement for photorealistic style."""
        prompt = "a beautiful landscape"
        enhanced = self.preprocessor.enhance_prompt(prompt, "photorealistic")
        
        assert "photorealistic" in enhanced
        assert "high resolution" in enhanced
        assert "detailed" in enhanced
        assert prompt in enhanced
    
    def test_enhance_prompt_artistic(self):
        """Test prompt enhancement for artistic style."""
        prompt = "abstract painting"
        enhanced = self.preprocessor.enhance_prompt(prompt, "artistic")
        
        assert "artistic" in enhanced
        assert "creative" in enhanced
        assert "beautiful composition" in enhanced
    
    def test_enhance_prompt_professional(self):
        """Test prompt enhancement for professional style."""
        prompt = "corporate headshot"
        enhanced = self.preprocessor.enhance_prompt(prompt, "professional")
        
        assert "professional photography" in enhanced
        assert "studio lighting" in enhanced
    
    def test_optimize_negative_prompt_with_existing(self):
        """Test negative prompt optimization with existing content."""
        existing = "bad lighting"
        optimized = self.preprocessor.optimize_negative_prompt(existing)
        
        assert "bad lighting" in optimized
        assert "blurry" in optimized
        assert "low quality" in optimized
    
    def test_optimize_negative_prompt_empty(self):
        """Test negative prompt optimization with no existing content."""
        optimized = self.preprocessor.optimize_negative_prompt()
        
        assert "blurry" in optimized
        assert "low quality" in optimized
        assert "pixelated" in optimized
    
    def test_validate_prompt_valid(self):
        """Test prompt validation with valid input."""
        valid_prompt = "a beautiful sunset over mountains"
        assert self.preprocessor.validate_prompt(valid_prompt) is True
    
    def test_validate_prompt_too_short(self):
        """Test prompt validation with too short input."""
        with pytest.raises(PromptError, match="at least 3 characters"):
            self.preprocessor.validate_prompt("hi")
    
    def test_validate_prompt_too_long(self):
        """Test prompt validation with too long input."""
        long_prompt = "a" * 1001
        with pytest.raises(PromptError, match="too long"):
            self.preprocessor.validate_prompt(long_prompt)
    
    def test_validate_prompt_unsafe_content(self):
        """Test prompt validation with unsafe content."""
        with pytest.raises(PromptError, match="Unsafe content detected"):
            self.preprocessor.validate_prompt("create nsfw content")


class TestResolutionOptimizer:
    """Test resolution optimization functionality."""
    
    def setup_method(self):
        self.optimizer = ResolutionOptimizer()
    
    def test_calculate_aspect_ratio_square(self):
        """Test aspect ratio calculation for square images."""
        ratio = self.optimizer.calculate_aspect_ratio(1024, 1024)
        assert ratio == "1:1"
    
    def test_calculate_aspect_ratio_widescreen(self):
        """Test aspect ratio calculation for widescreen images."""
        ratio = self.optimizer.calculate_aspect_ratio(1920, 1080)
        assert ratio == "16:9"
    
    def test_calculate_aspect_ratio_portrait(self):
        """Test aspect ratio calculation for portrait images."""
        ratio = self.optimizer.calculate_aspect_ratio(1080, 1920)
        assert ratio == "9:16"
    
    def test_optimize_resolution_supported_aspect(self):
        """Test resolution optimization for supported aspect ratio."""
        width, height = self.optimizer.optimize_resolution(1000, 1000)
        assert (width, height) in [(1024, 1024), (512, 512)]
    
    def test_optimize_resolution_unsupported_aspect(self):
        """Test resolution optimization for unsupported aspect ratio."""
        width, height = self.optimizer.optimize_resolution(800, 600)
        
        # Should be multiples of 64
        assert width % 64 == 0
        assert height % 64 == 0
        assert width >= 800
        assert height >= 600
    
    def test_supported_resolutions_exist(self):
        """Test that supported resolutions are properly defined."""
        assert "1:1" in self.optimizer.SUPPORTED_RESOLUTIONS
        assert "16:9" in self.optimizer.SUPPORTED_RESOLUTIONS
        assert "9:16" in self.optimizer.SUPPORTED_RESOLUTIONS
        
        # Check that resolutions are tuples of integers
        for aspect, resolutions in self.optimizer.SUPPORTED_RESOLUTIONS.items():
            assert isinstance(resolutions, list)
            for res in resolutions:
                assert isinstance(res, tuple)
                assert len(res) == 2
                assert all(isinstance(dim, int) for dim in res)


class TestStableDiffusionXLModel:
    """Test Stable Diffusion XL model integration."""
    
    def setup_method(self):
        self.config = VisualGenerationConfig({
            "sdxl_model_id": "test/model",
            "sdxl_use_refiner": False,
            "sdxl_cpu_offload": True
        })
        self.model = StableDiffusionXLModel(self.config)
    
    def test_model_initialization(self):
        """Test model initialization parameters."""
        assert self.model.model_name == "stable-diffusion-xl"
        assert self.model.pipeline is None
        assert not self.model.is_initialized
        assert isinstance(self.model.preprocessor, PromptPreprocessor)
        assert isinstance(self.model.resolution_optimizer, ResolutionOptimizer)
    
    @patch('torch.cuda.is_available')
    def test_device_selection_cuda(self, mock_cuda):
        """Test device selection when CUDA is available."""
        mock_cuda.return_value = True
        model = StableDiffusionXLModel(self.config)
        assert model.device == "cuda"
    
    @patch('torch.cuda.is_available')
    def test_device_selection_cpu(self, mock_cuda):
        """Test device selection when CUDA is not available."""
        mock_cuda.return_value = False
        model = StableDiffusionXLModel(self.config)
        assert model.device == "cpu"
    
    @patch('scrollintel.engines.visual_generation.models.stable_diffusion_xl.DPMSolverMultistepScheduler')
    @patch('scrollintel.engines.visual_generation.models.stable_diffusion_xl.StableDiffusionXLPipeline')
    @patch('torch.cuda.is_available')
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_cuda, mock_pipeline_class, mock_scheduler_class):
        """Test successful model initialization."""
        mock_cuda.return_value = True
        
        # Setup pipeline mock
        mock_pipeline = Mock()
        mock_pipeline.scheduler = Mock()
        mock_pipeline.scheduler.config = {"test": "config"}
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        # Setup scheduler mock
        mock_scheduler = Mock()
        mock_scheduler_class.from_config.return_value = mock_scheduler
        
        await self.model.initialize()
        
        assert self.model.is_initialized
        assert self.model.pipeline is not None
        mock_pipeline_class.from_pretrained.assert_called_once()
        mock_scheduler_class.from_config.assert_called_once_with({"test": "config"})
    
    @patch('scrollintel.engines.visual_generation.models.stable_diffusion_xl.StableDiffusionXLPipeline')
    @pytest.mark.asyncio
    async def test_initialize_failure(self, mock_pipeline_class):
        """Test model initialization failure."""
        mock_pipeline_class.from_pretrained.side_effect = Exception("Model load failed")
        
        with pytest.raises(ModelError, match="SDXL initialization failed"):
            await self.model.initialize()
        
        assert not self.model.is_initialized
    
    @patch('torch.manual_seed')
    @patch('torch.randint')
    @pytest.mark.asyncio
    async def test_generate_with_seed(self, mock_randint, mock_manual_seed):
        """Test image generation with specified seed."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.images = [Mock(spec=Image.Image)]
        mock_pipeline.return_value = mock_result
        self.model.pipeline = mock_pipeline
        self.model.is_initialized = True
        
        request = StableDiffusionXLRequest(
            prompt="test prompt",
            seed=12345
        )
        
        result = await self.model.generate(request)
        
        mock_manual_seed.assert_called_with(12345)
        assert isinstance(result, StableDiffusionXLResult)
        assert result.seed_used == 12345
    
    @patch('torch.manual_seed')
    @patch('torch.randint')
    @pytest.mark.asyncio
    async def test_generate_without_seed(self, mock_randint, mock_manual_seed):
        """Test image generation without specified seed."""
        mock_randint.return_value.item.return_value = 67890
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.images = [Mock(spec=Image.Image)]
        mock_pipeline.return_value = mock_result
        self.model.pipeline = mock_pipeline
        self.model.is_initialized = True
        
        request = StableDiffusionXLRequest(prompt="test prompt")
        
        result = await self.model.generate(request)
        
        assert result.seed_used == 67890
        mock_manual_seed.assert_called_with(67890)
    
    @pytest.mark.asyncio
    async def test_generate_not_initialized(self):
        """Test generation when model is not initialized."""
        request = StableDiffusionXLRequest(prompt="test prompt")
        
        with patch.object(self.model, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_pipeline = Mock()
            mock_result = Mock()
            mock_result.images = [Mock(spec=Image.Image)]
            mock_pipeline.return_value = mock_result
            
            async def set_initialized():
                self.model.pipeline = mock_pipeline
                self.model.is_initialized = True
            
            mock_init.side_effect = set_initialized
            
            result = await self.model.generate(request)
            
            mock_init.assert_called_once()
            assert isinstance(result, StableDiffusionXLResult)
    
    @pytest.mark.asyncio
    async def test_generate_invalid_prompt(self):
        """Test generation with invalid prompt."""
        self.model.is_initialized = True
        
        request = StableDiffusionXLRequest(prompt="")
        
        with pytest.raises(ModelError, match="Generation failed"):
            await self.model.generate(request)
    
    @pytest.mark.asyncio
    async def test_generate_pipeline_error(self):
        """Test generation when pipeline raises error."""
        mock_pipeline = Mock()
        mock_pipeline.side_effect = Exception("Pipeline error")
        self.model.pipeline = mock_pipeline
        self.model.is_initialized = True
        
        request = StableDiffusionXLRequest(prompt="test prompt")
        
        with pytest.raises(ModelError, match="Generation failed"):
            await self.model.generate(request)
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        info = self.model.get_model_info()
        
        assert info["name"] == "Stable Diffusion XL"
        assert info["device"] == self.model.device
        assert "max_resolution" in info
        assert "supported_aspects" in info
        assert "features" in info
        assert info["initialized"] == self.model.is_initialized
    
    @patch('torch.cuda.empty_cache')
    @pytest.mark.asyncio
    async def test_cleanup(self, mock_empty_cache):
        """Test model cleanup."""
        # Set up model with pipeline
        self.model.pipeline = Mock()
        self.model.is_initialized = True
        
        await self.model.cleanup()
        
        assert self.model.pipeline is None
        assert not self.model.is_initialized


class TestStableDiffusionXLRequest:
    """Test request data structure."""
    
    def test_default_values(self):
        """Test default request values."""
        request = StableDiffusionXLRequest(prompt="test")
        
        assert request.prompt == "test"
        assert request.negative_prompt is None
        assert request.width == 1024
        assert request.height == 1024
        assert request.num_inference_steps == 50
        assert request.guidance_scale == 7.5
        assert request.num_images == 1
        assert request.seed is None
        assert request.scheduler == "DPMSolverMultistep"
    
    def test_custom_values(self):
        """Test custom request values."""
        request = StableDiffusionXLRequest(
            prompt="custom prompt",
            negative_prompt="bad quality",
            width=512,
            height=768,
            num_inference_steps=30,
            guidance_scale=10.0,
            num_images=4,
            seed=12345
        )
        
        assert request.prompt == "custom prompt"
        assert request.negative_prompt == "bad quality"
        assert request.width == 512
        assert request.height == 768
        assert request.num_inference_steps == 30
        assert request.guidance_scale == 10.0
        assert request.num_images == 4
        assert request.seed == 12345


class TestStableDiffusionXLResult:
    """Test result data structure."""
    
    def test_result_structure(self):
        """Test result data structure."""
        mock_images = [Mock(spec=Image.Image), Mock(spec=Image.Image)]
        
        result = StableDiffusionXLResult(
            images=mock_images,
            prompt="enhanced prompt",
            negative_prompt="optimized negative",
            parameters={"width": 1024, "height": 1024},
            generation_time=5.2,
            seed_used=12345,
            model_version="test/model"
        )
        
        assert len(result.images) == 2
        assert result.prompt == "enhanced prompt"
        assert result.negative_prompt == "optimized negative"
        assert result.parameters["width"] == 1024
        assert result.generation_time == 5.2
        assert result.seed_used == 12345
        assert result.model_version == "test/model"


@pytest.mark.integration
class TestStableDiffusionXLIntegration:
    """Integration tests for Stable Diffusion XL model."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.asyncio
    async def test_full_generation_pipeline(self):
        """Test complete generation pipeline (requires GPU)."""
        config = VisualGenerationConfig({
            "sdxl_model_id": "stabilityai/stable-diffusion-xl-base-1.0"
        })
        model = StableDiffusionXLModel(config)
        
        try:
            await model.initialize()
            
            request = StableDiffusionXLRequest(
                prompt="a beautiful landscape with mountains",
                width=512,
                height=512,
                num_inference_steps=20,  # Reduced for faster testing
                num_images=1
            )
            
            result = await model.generate(request)
            
            assert len(result.images) == 1
            assert isinstance(result.images[0], Image.Image)
            assert result.generation_time > 0
            assert result.seed_used is not None
            
        finally:
            await model.cleanup()
    
    def test_model_info_accuracy(self):
        """Test that model info reflects actual capabilities."""
        config = VisualGenerationConfig({})
        model = StableDiffusionXLModel(config)
        
        info = model.get_model_info()
        
        # Verify info structure
        required_keys = ["name", "version", "device", "max_resolution", 
                        "supported_aspects", "features", "initialized"]
        for key in required_keys:
            assert key in info
        
        # Verify supported aspects match optimizer
        expected_aspects = list(model.resolution_optimizer.SUPPORTED_RESOLUTIONS.keys())
        assert info["supported_aspects"] == expected_aspects