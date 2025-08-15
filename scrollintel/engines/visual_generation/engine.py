"""
Main visual generation engine that orchestrates all components.
"""

import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base import (
    GenerationPipeline, ModelOrchestrator, GenerationRequest, 
    GenerationResult, GenerationStatus, ImageGenerationRequest, 
    VideoGenerationRequest
)
from .config import VisualGenerationConfig
from . import models
from .utils import (
    ContentSafetyFilter, QualityAssessor, 
    GenerationCacheManager, PromptEnhancer
)
from .exceptions import VisualGenerationError, ConfigurationError


class VisualGenerationEngine:
    """Main engine for visual content generation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = VisualGenerationConfig(config_path)
        self.orchestrator = ModelOrchestrator()
        self.pipeline = None
        self.safety_filter = None
        self.quality_assessor = None
        self.cache_manager = None
        self.prompt_enhancer = None
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the visual generation engine."""
        try:
            # Validate configuration
            self.config.validate_config()
            
            # Initialize utility components
            await self._initialize_utilities()
            
            # Initialize and register models
            await self._initialize_models()
            
            # Create generation pipeline
            self.pipeline = GenerationPipeline(self.orchestrator)
            self.pipeline.safety_filter = self.safety_filter
            self.pipeline.quality_assessor = self.quality_assessor
            self.pipeline.cache_manager = self.cache_manager
            
            # Create storage directories
            self._create_storage_directories()
            
            self.is_initialized = True
            
        except Exception as e:
            raise VisualGenerationError(f"Failed to initialize visual generation engine: {str(e)}")
    
    async def _initialize_utilities(self):
        """Initialize utility components."""
        # Initialize safety filter
        if self.config.safety.enabled:
            self.safety_filter = ContentSafetyFilter(self.config.safety)
        
        # Initialize quality assessor
        if self.config.quality.enabled:
            self.quality_assessor = QualityAssessor(self.config.quality)
        
        # Initialize cache manager
        if self.config.infrastructure.cache_enabled:
            self.cache_manager = GenerationCacheManager(self.config.infrastructure)
        
        # Initialize prompt enhancer
        self.prompt_enhancer = PromptEnhancer()
    
    async def _initialize_models(self):
        """Initialize and register all available models."""
        enabled_models = self.config.get_enabled_models()
        
        for model_name, model_config in enabled_models.items():
            try:
                # Initialize model based on type
                if model_config.type == 'image':
                    model = await self._create_image_model(model_name, model_config)
                elif model_config.type == 'video':
                    model = await self._create_video_model(model_name, model_config)
                elif model_config.type == 'enhancement':
                    model = await self._create_enhancement_model(model_name, model_config)
                else:
                    continue
                
                # Initialize the model
                await model.initialize()
                
                # Register with orchestrator
                self.orchestrator.register_generator(model_name, model)
                
            except Exception as e:
                print(f"Warning: Failed to initialize model {model_name}: {str(e)}")
                # Continue with other models
    
    async def _create_image_model(self, model_name: str, model_config):
        """Create an image generation model."""
        StableDiffusionXLModel, DALLE3Model, MidjourneyModel = models.get_image_models()
        
        if model_name == 'stable_diffusion_xl':
            return StableDiffusionXLModel(model_config)
        elif model_name == 'dalle3':
            return DALLE3Model(model_config)
        elif model_name == 'midjourney':
            return MidjourneyModel(model_config)
        else:
            raise ConfigurationError(f"Unknown image model: {model_name}")
    
    async def _create_video_model(self, model_name: str, model_config):
        """Create a video generation model."""
        ProprietaryNeuralRenderer, UltraRealisticVideoGenerator = models.get_video_models()
        
        if model_name == 'proprietary_neural_renderer':
            return ProprietaryNeuralRenderer(model_config)
        elif model_name == 'ultra_realistic_video_generator':
            return UltraRealisticVideoGenerator(model_config)
        else:
            raise ConfigurationError(f"Unknown video model: {model_name}")
    
    async def _create_enhancement_model(self, model_name: str, model_config):
        """Create an enhancement model."""
        ImageEnhancer, VideoEnhancer = models.get_enhancement_models()
        
        if model_name == 'image_enhancer':
            return ImageEnhancer(model_config)
        elif model_name == 'video_enhancer':
            return VideoEnhancer(model_config)
        else:
            raise ConfigurationError(f"Unknown enhancement model: {model_name}")
    
    def _create_storage_directories(self):
        """Create necessary storage directories."""
        directories = [
            self.config.infrastructure.storage_path,
            self.config.infrastructure.temp_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def generate_image(self, request: ImageGenerationRequest) -> GenerationResult:
        """Generate images based on the request."""
        if not self.is_initialized:
            await self.initialize()
        
        # Enhance prompt if needed
        if self.prompt_enhancer:
            enhancement = await self.prompt_enhancer.enhance_prompt(request)
            request.prompt = enhancement.enhanced_prompt
            
            # Generate negative prompt if not provided
            if not request.negative_prompt:
                request.negative_prompt = await self.prompt_enhancer.generate_negative_prompt(request)
        
        # Process through pipeline
        return await self.pipeline.process_request(request)
    
    async def generate_video(self, request: VideoGenerationRequest) -> GenerationResult:
        """Generate ultra-realistic videos based on the request."""
        if not self.is_initialized:
            await self.initialize()
        
        # Enhance prompt for video generation
        if self.prompt_enhancer:
            enhancement = await self.prompt_enhancer.enhance_prompt(request)
            request.prompt = enhancement.enhanced_prompt
        
        # Process through pipeline
        return await self.pipeline.process_request(request)
    
    async def enhance_content(self, content_path: str, enhancement_type: str) -> GenerationResult:
        """Enhance existing visual content."""
        if not self.is_initialized:
            await self.initialize()
        
        # Find appropriate enhancement model
        enhancement_models = self.config.get_enabled_models('enhancement')
        
        if not enhancement_models:
            raise VisualGenerationError("No enhancement models available")
        
        # Use the first available enhancement model
        model_name = list(enhancement_models.keys())[0]
        model = self.orchestrator.generators.get(model_name)
        
        if not model:
            raise VisualGenerationError(f"Enhancement model {model_name} not initialized")
        
        return await model.enhance_content(content_path, enhancement_type)
    
    async def batch_generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        """Process multiple generation requests in batch."""
        if not self.is_initialized:
            await self.initialize()
        
        # Process requests concurrently with limit
        max_concurrent = self.config.infrastructure.max_concurrent_requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_request(request):
            async with semaphore:
                return await self.pipeline.process_request(request)
        
        tasks = [process_single_request(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(GenerationResult(
                    id=f"batch_result_{i}",
                    request_id=requests[i].request_id,
                    status=GenerationStatus.FAILED,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def get_model_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of all available models."""
        capabilities = {}
        
        for model_name, model in self.orchestrator.generators.items():
            capabilities[model_name] = model.get_capabilities()
        
        return capabilities
    
    async def estimate_cost(self, request: GenerationRequest) -> float:
        """Estimate the cost of processing a request."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            generator = await self.orchestrator.select_optimal_generator(request)
            return await generator.estimate_cost(request)
        except Exception:
            # Return default cost estimate
            if isinstance(request, ImageGenerationRequest):
                return self.config.cost.cost_per_image
            elif isinstance(request, VideoGenerationRequest):
                return self.config.cost.cost_per_video_second * request.duration
            else:
                return 0.01
    
    async def estimate_time(self, request: GenerationRequest) -> float:
        """Estimate the processing time for a request."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            generator = await self.orchestrator.select_optimal_generator(request)
            return await generator.estimate_time(request)
        except Exception:
            # Return default time estimate
            if isinstance(request, ImageGenerationRequest):
                return 30.0  # 30 seconds
            elif isinstance(request, VideoGenerationRequest):
                return 60.0 * request.duration  # 1 minute per second of video
            else:
                return 10.0
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health information."""
        status = {
            'initialized': self.is_initialized,
            'models': {},
            'cache_stats': None,
            'config': {
                'safety_enabled': self.config.safety.enabled,
                'quality_enabled': self.config.quality.enabled,
                'cache_enabled': self.config.infrastructure.cache_enabled
            }
        }
        
        # Get model status
        for model_name, model in self.orchestrator.generators.items():
            status['models'][model_name] = {
                'initialized': model.is_initialized,
                'capabilities': model.get_capabilities()
            }
        
        # Get cache statistics
        if self.cache_manager:
            status['cache_stats'] = self.cache_manager.get_cache_stats()
        
        return status
    
    async def cleanup(self):
        """Clean up resources used by the engine."""
        # Cleanup models
        for model in self.orchestrator.generators.values():
            await model.cleanup()
        
        # Cleanup cache manager
        if self.cache_manager:
            await self.cache_manager.cleanup()
        
        self.is_initialized = False