"""
Unified image generation pipeline that orchestrates multiple models.

This module implements the ImageGenerationPipeline class that provides a unified
interface for image generation across multiple models (Stable Diffusion XL, DALL-E 3, Midjourney).
It includes model selection logic, result aggregation, and quality comparison functionality.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import tempfile
import os
from PIL import Image

from .base import (
    ImageGenerator, 
    ImageGenerationRequest, 
    GenerationRequest,
    GenerationResult, 
    GenerationStatus,
    QualityMetrics
)
from .models.stable_diffusion_xl import StableDiffusionXLModel
from .models.dalle3 import DALLE3Model
from .models.midjourney import MidjourneyModel
from .models.ultra_performance_pipeline import UltraRealisticVideoGenerationPipeline, ProcessingMode
from .exceptions import (
    ModelError, 
    ResourceError, 
    ValidationError,
    ConfigurationError
)
from .config import VisualGenerationConfig

logger = logging.getLogger(__name__)


class ModelSelectionStrategy(Enum):
    """Strategy for selecting models."""
    BEST_QUALITY = "best_quality"
    FASTEST = "fastest"
    MOST_COST_EFFECTIVE = "most_cost_effective"
    BALANCED = "balanced"
    USER_PREFERENCE = "user_preference"
    ALL_MODELS = "all_models"


@dataclass
class ModelCapability:
    """Represents the capabilities of a model."""
    name: str
    quality_score: float  # 0-1
    speed_score: float  # 0-1 (higher is faster)
    cost_score: float  # 0-1 (higher is more cost effective)
    style_compatibility: Dict[str, float]  # style -> compatibility score
    resolution_support: List[Tuple[int, int]]
    max_batch_size: int
    supports_negative_prompts: bool
    supports_seeds: bool
    availability_score: float  # 0-1 (based on rate limits, etc.)


@dataclass
class ModelSelectionResult:
    """Result of model selection process."""
    selected_models: List[str]
    selection_reason: str
    confidence_score: float
    fallback_models: List[str]
    estimated_cost: float
    estimated_time: float


@dataclass
class GenerationComparison:
    """Comparison of results from multiple models."""
    results: Dict[str, GenerationResult]
    best_result: GenerationResult
    quality_rankings: List[Tuple[str, float]]  # (model_name, quality_score)
    cost_comparison: Dict[str, float]
    time_comparison: Dict[str, float]
    recommendation: str


class ModelSelector:
    """Intelligent model selection based on request parameters and model capabilities."""
    
    def __init__(self, config: VisualGenerationConfig):
        self.config = config
        self.model_capabilities = self._initialize_model_capabilities()
        self.selection_history: List[Dict[str, Any]] = []
        
    def _initialize_model_capabilities(self) -> Dict[str, ModelCapability]:
        """Initialize model capabilities based on configuration and benchmarks."""
        capabilities = {}
        
        # Stable Diffusion XL capabilities
        if 'stable_diffusion_xl' in self.config.models:
            capabilities['stable_diffusion_xl'] = ModelCapability(
                name='stable_diffusion_xl',
                quality_score=0.85,
                speed_score=0.7,
                cost_score=0.9,  # Very cost effective
                style_compatibility={
                    'photorealistic': 0.9,
                    'artistic': 0.8,
                    'professional': 0.85,
                    'creative': 0.75,
                    'abstract': 0.7
                },
                resolution_support=[
                    (512, 512), (768, 768), (1024, 1024),
                    (1152, 896), (896, 1152), (1344, 768), (768, 1344)
                ],
                max_batch_size=4,
                supports_negative_prompts=True,
                supports_seeds=True,
                availability_score=0.95
            )
        
        # DALL-E 3 capabilities
        if 'dalle3' in self.config.models:
            capabilities['dalle3'] = ModelCapability(
                name='dalle3',
                quality_score=0.95,
                speed_score=0.6,
                cost_score=0.4,  # More expensive
                style_compatibility={
                    'photorealistic': 0.95,
                    'artistic': 0.9,
                    'professional': 0.9,
                    'creative': 0.95,
                    'abstract': 0.85
                },
                resolution_support=[
                    (1024, 1024), (1024, 1792), (1792, 1024)
                ],
                max_batch_size=1,
                supports_negative_prompts=False,
                supports_seeds=False,
                availability_score=0.8  # Rate limited
            )
        
        # Midjourney capabilities
        if 'midjourney' in self.config.models:
            capabilities['midjourney'] = ModelCapability(
                name='midjourney',
                quality_score=0.9,
                speed_score=0.4,  # Slower due to queue
                cost_score=0.6,
                style_compatibility={
                    'photorealistic': 0.8,
                    'artistic': 0.95,
                    'professional': 0.75,
                    'creative': 0.9,
                    'abstract': 0.85
                },
                resolution_support=[
                    (1024, 1024), (1792, 1024), (1024, 1792)
                ],
                max_batch_size=4,
                supports_negative_prompts=True,
                supports_seeds=True,
                availability_score=0.7  # Queue dependent
            )
        
        return capabilities
    
    async def select_models(
        self, 
        request: ImageGenerationRequest, 
        strategy: ModelSelectionStrategy = ModelSelectionStrategy.BALANCED
    ) -> ModelSelectionResult:
        """Select the best model(s) for the given request and strategy."""
        
        # Filter available models
        available_models = self._filter_available_models(request)
        
        if not available_models:
            raise ResourceError("No suitable models available for this request")
        
        # Apply selection strategy
        if strategy == ModelSelectionStrategy.BEST_QUALITY:
            selected = self._select_best_quality(available_models, request)
        elif strategy == ModelSelectionStrategy.FASTEST:
            selected = self._select_fastest(available_models, request)
        elif strategy == ModelSelectionStrategy.MOST_COST_EFFECTIVE:
            selected = self._select_most_cost_effective(available_models, request)
        elif strategy == ModelSelectionStrategy.USER_PREFERENCE:
            selected = self._select_user_preference(available_models, request)
        elif strategy == ModelSelectionStrategy.ALL_MODELS:
            selected = self._select_all_models(available_models, request)
        else:  # BALANCED
            selected = self._select_balanced(available_models, request)
        
        # Calculate estimates
        estimated_cost = await self._estimate_total_cost(selected.selected_models, request)
        estimated_time = await self._estimate_total_time(selected.selected_models, request)
        
        selected.estimated_cost = estimated_cost
        selected.estimated_time = estimated_time
        
        # Record selection for learning
        self._record_selection(request, selected)
        
        return selected
    
    def _filter_available_models(self, request: ImageGenerationRequest) -> List[str]:
        """Filter models based on request requirements and availability."""
        available = []
        
        for model_name, capability in self.model_capabilities.items():
            # Check if model is enabled
            model_config = self.config.get_model_config(model_name)
            if not model_config or not model_config.enabled:
                continue
            
            # Check resolution support
            if not self._supports_resolution(capability, request.resolution):
                continue
            
            # Check style compatibility
            style_score = capability.style_compatibility.get(request.style, 0.5)
            if style_score < 0.3:  # Minimum compatibility threshold
                continue
            
            # Check availability
            if capability.availability_score < 0.5:
                continue
            
            available.append(model_name)
        
        return available
    
    def _supports_resolution(self, capability: ModelCapability, resolution: Tuple[int, int]) -> bool:
        """Check if model supports the requested resolution."""
        width, height = resolution
        
        for supported_width, supported_height in capability.resolution_support:
            # Allow some flexibility in resolution matching
            if (abs(width - supported_width) <= 128 and 
                abs(height - supported_height) <= 128):
                return True
        
        return False
    
    def _select_best_quality(self, available_models: List[str], request: ImageGenerationRequest) -> ModelSelectionResult:
        """Select model with highest quality score."""
        best_model = max(available_models, key=lambda m: self.model_capabilities[m].quality_score)
        
        return ModelSelectionResult(
            selected_models=[best_model],
            selection_reason=f"Selected {best_model} for highest quality (score: {self.model_capabilities[best_model].quality_score})",
            confidence_score=0.9,
            fallback_models=[m for m in available_models if m != best_model][:2],
            estimated_cost=0.0,
            estimated_time=0.0
        )
    
    def _select_fastest(self, available_models: List[str], request: ImageGenerationRequest) -> ModelSelectionResult:
        """Select fastest model."""
        fastest_model = max(available_models, key=lambda m: self.model_capabilities[m].speed_score)
        
        return ModelSelectionResult(
            selected_models=[fastest_model],
            selection_reason=f"Selected {fastest_model} for fastest generation (speed score: {self.model_capabilities[fastest_model].speed_score})",
            confidence_score=0.8,
            fallback_models=[m for m in available_models if m != fastest_model][:2],
            estimated_cost=0.0,
            estimated_time=0.0
        )
    
    def _select_most_cost_effective(self, available_models: List[str], request: ImageGenerationRequest) -> ModelSelectionResult:
        """Select most cost-effective model."""
        most_cost_effective = max(available_models, key=lambda m: self.model_capabilities[m].cost_score)
        
        return ModelSelectionResult(
            selected_models=[most_cost_effective],
            selection_reason=f"Selected {most_cost_effective} for cost effectiveness (cost score: {self.model_capabilities[most_cost_effective].cost_score})",
            confidence_score=0.85,
            fallback_models=[m for m in available_models if m != most_cost_effective][:2],
            estimated_cost=0.0,
            estimated_time=0.0
        )
    
    def _select_user_preference(self, available_models: List[str], request: ImageGenerationRequest) -> ModelSelectionResult:
        """Select based on user preference if specified."""
        if request.model_preference and request.model_preference in available_models:
            return ModelSelectionResult(
                selected_models=[request.model_preference],
                selection_reason=f"Selected {request.model_preference} based on user preference",
                confidence_score=1.0,
                fallback_models=[m for m in available_models if m != request.model_preference][:2],
                estimated_cost=0.0,
                estimated_time=0.0
            )
        else:
            # Fall back to balanced selection
            return self._select_balanced(available_models, request)
    
    def _select_all_models(self, available_models: List[str], request: ImageGenerationRequest) -> ModelSelectionResult:
        """Select all available models for comparison."""
        return ModelSelectionResult(
            selected_models=available_models,
            selection_reason="Selected all available models for comparison",
            confidence_score=0.7,
            fallback_models=[],
            estimated_cost=0.0,
            estimated_time=0.0
        )
    
    def _select_balanced(self, available_models: List[str], request: ImageGenerationRequest) -> ModelSelectionResult:
        """Select model with best balanced score."""
        def calculate_balanced_score(model_name: str) -> float:
            capability = self.model_capabilities[model_name]
            style_score = capability.style_compatibility.get(request.style, 0.5)
            
            # Weighted combination of factors
            score = (
                capability.quality_score * 0.4 +
                capability.speed_score * 0.2 +
                capability.cost_score * 0.2 +
                style_score * 0.15 +
                capability.availability_score * 0.05
            )
            return score
        
        best_model = max(available_models, key=calculate_balanced_score)
        best_score = calculate_balanced_score(best_model)
        
        return ModelSelectionResult(
            selected_models=[best_model],
            selection_reason=f"Selected {best_model} for balanced performance (score: {best_score:.3f})",
            confidence_score=0.85,
            fallback_models=sorted(
                [m for m in available_models if m != best_model],
                key=calculate_balanced_score,
                reverse=True
            )[:2],
            estimated_cost=0.0,
            estimated_time=0.0
        )
    
    async def _estimate_total_cost(self, model_names: List[str], request: ImageGenerationRequest) -> float:
        """Estimate total cost for selected models."""
        total_cost = 0.0
        
        for model_name in model_names:
            # This would be replaced with actual cost estimation from models
            base_cost = 0.01  # Base cost per image
            
            if model_name == 'dalle3':
                base_cost = 0.04  # DALL-E 3 is more expensive
            elif model_name == 'midjourney':
                base_cost = 0.02
            
            # Factor in resolution and number of images
            pixels = request.resolution[0] * request.resolution[1]
            resolution_multiplier = pixels / (1024 * 1024)
            
            model_cost = base_cost * resolution_multiplier * request.num_images
            total_cost += model_cost
        
        return total_cost
    
    async def _estimate_total_time(self, model_names: List[str], request: ImageGenerationRequest) -> float:
        """Estimate total time for selected models."""
        if len(model_names) == 1:
            # Single model - sequential processing
            return await self._estimate_single_model_time(model_names[0], request)
        else:
            # Multiple models - parallel processing, return max time
            times = []
            for model_name in model_names:
                time_estimate = await self._estimate_single_model_time(model_name, request)
                times.append(time_estimate)
            return max(times)
    
    async def _estimate_single_model_time(self, model_name: str, request: ImageGenerationRequest) -> float:
        """Estimate time for a single model."""
        capability = self.model_capabilities[model_name]
        
        # Base time estimates (seconds)
        base_times = {
            'stable_diffusion_xl': 20.0,
            'dalle3': 30.0,
            'midjourney': 60.0
        }
        
        base_time = base_times.get(model_name, 30.0)
        
        # Factor in resolution
        pixels = request.resolution[0] * request.resolution[1]
        resolution_factor = pixels / (1024 * 1024)
        
        # Factor in number of images
        batch_factor = request.num_images / capability.max_batch_size
        if batch_factor > 1:
            batch_factor = 1 + (batch_factor - 1) * 0.8  # Batch processing efficiency
        
        estimated_time = base_time * resolution_factor * batch_factor
        return estimated_time
    
    def _record_selection(self, request: ImageGenerationRequest, selection: ModelSelectionResult):
        """Record selection for learning and optimization."""
        record = {
            'timestamp': datetime.now(),
            'request_style': request.style,
            'request_resolution': request.resolution,
            'request_quality': request.quality,
            'selected_models': selection.selected_models,
            'selection_reason': selection.selection_reason,
            'confidence_score': selection.confidence_score
        }
        
        self.selection_history.append(record)
        
        # Keep only recent history (last 1000 selections)
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]


class ResultAggregator:
    """Aggregates and compares results from multiple models."""
    
    def __init__(self, config: VisualGenerationConfig):
        self.config = config
    
    async def aggregate_results(
        self, 
        results: Dict[str, GenerationResult], 
        request: ImageGenerationRequest
    ) -> GenerationComparison:
        """Aggregate results from multiple models and determine the best one."""
        
        if not results:
            raise ValidationError("No results to aggregate")
        
        # Filter successful results
        successful_results = {
            model: result for model, result in results.items()
            if result.status == GenerationStatus.COMPLETED
        }
        
        if not successful_results:
            raise ModelError("All model generations failed")
        
        # Calculate quality rankings
        quality_rankings = await self._rank_by_quality(successful_results, request)
        
        # Determine best result
        best_model, best_score = quality_rankings[0]
        best_result = successful_results[best_model]
        
        # Calculate cost and time comparisons
        cost_comparison = {
            model: result.cost for model, result in successful_results.items()
        }
        
        time_comparison = {
            model: result.generation_time for model, result in successful_results.items()
        }
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            quality_rankings, cost_comparison, time_comparison
        )
        
        return GenerationComparison(
            results=successful_results,
            best_result=best_result,
            quality_rankings=quality_rankings,
            cost_comparison=cost_comparison,
            time_comparison=time_comparison,
            recommendation=recommendation
        )
    
    async def _rank_by_quality(
        self, 
        results: Dict[str, GenerationResult], 
        request: ImageGenerationRequest
    ) -> List[Tuple[str, float]]:
        """Rank results by quality score."""
        rankings = []
        
        for model_name, result in results.items():
            if result.quality_metrics:
                quality_score = result.quality_metrics.overall_score
            else:
                # Fallback quality estimation based on model capabilities
                quality_score = await self._estimate_quality_score(model_name, result, request)
            
            rankings.append((model_name, quality_score))
        
        # Sort by quality score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    async def _estimate_quality_score(
        self, 
        model_name: str, 
        result: GenerationResult, 
        request: ImageGenerationRequest
    ) -> float:
        """Estimate quality score when metrics are not available."""
        # Base quality scores by model
        base_scores = {
            'stable_diffusion_xl': 0.8,
            'dalle3': 0.9,
            'midjourney': 0.85
        }
        
        base_score = base_scores.get(model_name, 0.75)
        
        # Adjust based on generation time (longer might indicate higher quality)
        time_factor = min(1.0, result.generation_time / 30.0)  # Normalize to 30 seconds
        
        # Adjust based on cost (higher cost might indicate higher quality)
        cost_factor = min(1.0, result.cost / 0.05)  # Normalize to $0.05
        
        estimated_score = base_score * (0.7 + 0.2 * time_factor + 0.1 * cost_factor)
        
        return min(1.0, estimated_score)
    
    def _generate_recommendation(
        self, 
        quality_rankings: List[Tuple[str, float]], 
        cost_comparison: Dict[str, float], 
        time_comparison: Dict[str, float]
    ) -> str:
        """Generate a recommendation based on the comparison."""
        best_model, best_quality = quality_rankings[0]
        
        # Find most cost-effective
        most_cost_effective = min(cost_comparison.items(), key=lambda x: x[1])
        
        # Find fastest
        fastest = min(time_comparison.items(), key=lambda x: x[1])
        
        recommendation = f"Best quality: {best_model} (score: {best_quality:.3f}). "
        
        if most_cost_effective[0] != best_model:
            recommendation += f"Most cost-effective: {most_cost_effective[0]} (${most_cost_effective[1]:.3f}). "
        
        if fastest[0] != best_model:
            recommendation += f"Fastest: {fastest[0]} ({fastest[1]:.1f}s). "
        
        # Overall recommendation
        if best_model == most_cost_effective[0] == fastest[0]:
            recommendation += f"{best_model} is the clear winner across all metrics."
        elif best_quality > 0.9:
            recommendation += f"Recommend {best_model} for its superior quality."
        elif best_quality - quality_rankings[1][1] < 0.05:  # Close quality scores
            recommendation += f"Quality is similar - consider {most_cost_effective[0]} for cost savings or {fastest[0]} for speed."
        else:
            recommendation += f"Recommend {best_model} for the best balance of quality and performance."
        
        return recommendation


class ImageGenerationPipeline(ImageGenerator):
    """
    Unified image generation pipeline that orchestrates multiple models.
    
    This pipeline provides intelligent model selection, parallel generation,
    result aggregation, and quality comparison across multiple image generation models.
    """
    
    def __init__(self, config: VisualGenerationConfig):
        super().__init__(config.infrastructure.__dict__)
        self.config = config
        self.models: Dict[str, Any] = {}
        self.model_selector = ModelSelector(config)
        self.result_aggregator = ResultAggregator(config)
        
        # Pipeline configuration
        self.default_strategy = ModelSelectionStrategy.BALANCED
        self.enable_parallel_generation = True
        self.max_concurrent_models = 3
        self.enable_result_comparison = True
        
        logger.info("ImageGenerationPipeline initialized")
    
    async def initialize(self) -> None:
        """Initialize all available models."""
        logger.info("Initializing image generation models...")
        
        # Initialize Stable Diffusion XL
        if self.config.get_model_config('stable_diffusion_xl'):
            try:
                self.models['stable_diffusion_xl'] = StableDiffusionXLModel(self.config)
                await self.models['stable_diffusion_xl'].initialize()
                logger.info("Stable Diffusion XL model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Stable Diffusion XL: {e}")
        
        # Initialize DALL-E 3
        if self.config.get_model_config('dalle3'):
            try:
                self.models['dalle3'] = DALLE3Model(self.config)
                logger.info("DALL-E 3 model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize DALL-E 3: {e}")
        
        # Initialize Midjourney
        if self.config.get_model_config('midjourney'):
            try:
                self.models['midjourney'] = MidjourneyModel(self.config)
                logger.info("Midjourney model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Midjourney: {e}")
        
        # Initialize Ultra-Performance Video Pipeline
        try:
            self.models['ultra_performance_video'] = UltraRealisticVideoGenerationPipeline()
            logger.info("Ultra-Performance Video Generation Pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Ultra-Performance Video Pipeline: {e}")
        
        if not self.models:
            raise ConfigurationError("No image generation models could be initialized")
        
        self.is_initialized = True
        logger.info(f"Pipeline initialized with {len(self.models)} models: {list(self.models.keys())}")
    
    async def generate_image(self, request: ImageGenerationRequest) -> GenerationResult:
        """
        Generate images using the optimal model selection strategy.
        
        Args:
            request: Image generation request
            
        Returns:
            GenerationResult with the best generated images
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Determine selection strategy
            strategy = self._determine_strategy(request)
            
            # Select models
            selection = await self.model_selector.select_models(request, strategy)
            logger.info(f"Selected models: {selection.selected_models} - {selection.selection_reason}")
            
            # Generate with selected models
            if len(selection.selected_models) == 1:
                # Single model generation
                result = await self._generate_single_model(
                    selection.selected_models[0], request
                )
            else:
                # Multi-model generation and comparison
                result = await self._generate_multi_model(
                    selection.selected_models, request
                )
            
            # Update result with pipeline metadata
            result.metadata.update({
                'pipeline_version': '1.0',
                'selection_strategy': strategy.value,
                'selected_models': selection.selected_models,
                'selection_confidence': selection.confidence_score,
                'total_pipeline_time': time.time() - start_time
            })
            
            logger.info(f"Pipeline generation completed in {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline generation failed: {e}")
            return GenerationResult(
                id=f"pipeline_{request.request_id}",
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                error_message=str(e),
                generation_time=time.time() - start_time
            )
    
    async def generate_ultra_realistic_video(
        self,
        prompt: str,
        duration: float = 5.0,
        resolution: Tuple[int, int] = (3840, 2160),
        fps: int = 60,
        mode: ProcessingMode = ProcessingMode.ULTRA_FAST
    ) -> Dict[str, Any]:
        """
        Generate ultra-realistic video with 10x performance advantage.
        
        Args:
            prompt: Text description for video generation
            duration: Video duration in seconds
            resolution: Output resolution (width, height)
            fps: Frames per second
            mode: Processing mode for performance/quality balance
            
        Returns:
            Dictionary containing video path, metrics, and performance data
        """
        if not self.is_initialized:
            await self.initialize()
        
        if 'ultra_performance_video' not in self.models:
            raise ModelError("Ultra-Performance Video Generation Pipeline not available")
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting ultra-realistic video generation: {resolution[0]}x{resolution[1]} @ {fps}fps for {duration}s")
            
            # Use the ultra-performance pipeline
            ultra_pipeline = self.models['ultra_performance_video']
            
            result = await ultra_pipeline.generate_ultra_realistic_video(
                prompt=prompt,
                duration=duration,
                resolution=resolution,
                fps=fps,
                mode=mode
            )
            
            # Add pipeline metadata
            result['pipeline_metadata'] = {
                'pipeline_version': '1.0',
                'total_pipeline_time': time.time() - start_time,
                'mode': mode.value,
                'resolution': resolution,
                'fps': fps,
                'duration': duration
            }
            
            logger.info(f"Ultra-realistic video generation completed in {time.time() - start_time:.2f}s "
                       f"with {result['speed_advantage']:.1f}x speed advantage")
            
            return result
            
        except Exception as e:
            logger.error(f"Ultra-realistic video generation failed: {e}")
            raise ModelError(f"Ultra-performance video generation failed: {e}")
    
    def _determine_strategy(self, request: ImageGenerationRequest) -> ModelSelectionStrategy:
        """Determine the best selection strategy based on request parameters."""
        
        # User specified preference
        if request.model_preference:
            return ModelSelectionStrategy.USER_PREFERENCE
        
        # Quality-focused requests
        if request.quality in ['ultra', 'maximum']:
            return ModelSelectionStrategy.BEST_QUALITY
        
        # Speed-focused requests (based on metadata)
        if request.metadata.get('priority') == 'speed':
            return ModelSelectionStrategy.FASTEST
        
        # Cost-focused requests
        if request.metadata.get('priority') == 'cost':
            return ModelSelectionStrategy.MOST_COST_EFFECTIVE
        
        # Comparison requests
        if request.metadata.get('compare_models', False):
            return ModelSelectionStrategy.ALL_MODELS
        
        # Default to balanced
        return self.default_strategy
    
    async def _generate_single_model(
        self, 
        model_name: str, 
        request: ImageGenerationRequest
    ) -> GenerationResult:
        """Generate using a single model."""
        
        if model_name not in self.models:
            raise ModelError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        try:
            # Generate using the specific model
            result = await model.generate_image(request)
            
            logger.info(f"Single model generation completed: {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Single model generation failed for {model_name}: {e}")
            
            # Try fallback models
            fallback_models = [name for name in self.models.keys() if name != model_name]
            
            for fallback_model in fallback_models[:2]:  # Try up to 2 fallbacks
                try:
                    logger.info(f"Trying fallback model: {fallback_model}")
                    result = await self.models[fallback_model].generate_image(request)
                    
                    # Update metadata to indicate fallback was used
                    result.metadata['fallback_used'] = True
                    result.metadata['original_model'] = model_name
                    result.metadata['fallback_model'] = fallback_model
                    
                    return result
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback model {fallback_model} also failed: {fallback_error}")
                    continue
            
            # All models failed
            raise ModelError(f"All models failed. Original error: {e}")
    
    async def _generate_multi_model(
        self, 
        model_names: List[str], 
        request: ImageGenerationRequest
    ) -> GenerationResult:
        """Generate using multiple models and compare results."""
        
        # Limit concurrent models
        if len(model_names) > self.max_concurrent_models:
            model_names = model_names[:self.max_concurrent_models]
            logger.info(f"Limited to {self.max_concurrent_models} concurrent models")
        
        # Create tasks for parallel generation
        tasks = {}
        for model_name in model_names:
            if model_name in self.models:
                task = asyncio.create_task(
                    self._safe_model_generation(model_name, request)
                )
                tasks[model_name] = task
        
        if not tasks:
            raise ModelError("No valid models available for generation")
        
        # Wait for all tasks to complete
        logger.info(f"Starting parallel generation with {len(tasks)} models")
        results = {}
        
        for model_name, task in tasks.items():
            try:
                result = await task
                results[model_name] = result
                logger.info(f"Model {model_name} completed successfully")
            except Exception as e:
                logger.error(f"Model {model_name} failed: {e}")
                # Create failed result
                results[model_name] = GenerationResult(
                    id=f"{model_name}_{request.request_id}",
                    request_id=request.request_id,
                    status=GenerationStatus.FAILED,
                    error_message=str(e),
                    model_used=model_name
                )
        
        # Aggregate and compare results
        if self.enable_result_comparison:
            comparison = await self.result_aggregator.aggregate_results(results, request)
            
            # Use the best result as the primary result
            best_result = comparison.best_result
            
            # Add comparison metadata
            best_result.metadata.update({
                'multi_model_comparison': True,
                'quality_rankings': comparison.quality_rankings,
                'cost_comparison': comparison.cost_comparison,
                'time_comparison': comparison.time_comparison,
                'recommendation': comparison.recommendation,
                'all_results': {
                    model: {
                        'status': result.status.value,
                        'quality_score': result.quality_metrics.overall_score if result.quality_metrics else 0.0,
                        'generation_time': result.generation_time,
                        'cost': result.cost
                    }
                    for model, result in results.items()
                }
            })
            
            logger.info(f"Multi-model comparison completed. Best: {comparison.best_result.model_used}")
            return best_result
        
        else:
            # Return first successful result
            for result in results.values():
                if result.status == GenerationStatus.COMPLETED:
                    return result
            
            # All failed
            raise ModelError("All model generations failed")
    
    async def _safe_model_generation(
        self, 
        model_name: str, 
        request: ImageGenerationRequest
    ) -> GenerationResult:
        """Safely generate with a model, handling errors gracefully."""
        try:
            model = self.models[model_name]
            return await model.generate_image(request)
        except Exception as e:
            logger.error(f"Safe generation failed for {model_name}: {e}")
            raise
    
    async def validate_request(self, request: GenerationRequest) -> bool:
        """Validate if the request can be processed by the pipeline."""
        if not isinstance(request, ImageGenerationRequest):
            return False
        
        # Check if at least one model can handle the request
        for model in self.models.values():
            try:
                if await model.validate_request(request):
                    return True
            except Exception:
                continue
        
        return False
    
    async def estimate_cost(self, request: GenerationRequest) -> float:
        """Estimate the cost of processing this request."""
        if not isinstance(request, ImageGenerationRequest):
            return 0.0
        
        # Use model selector to get cost estimate
        try:
            selection = await self.model_selector.select_models(request)
            return selection.estimated_cost
        except Exception:
            # Fallback estimation
            return 0.02 * request.num_images
    
    async def estimate_time(self, request: GenerationRequest) -> float:
        """Estimate the processing time for this request."""
        if not isinstance(request, ImageGenerationRequest):
            return 0.0
        
        # Use model selector to get time estimate
        try:
            selection = await self.model_selector.select_models(request)
            return selection.estimated_time
        except Exception:
            # Fallback estimation
            return 30.0 * request.num_images
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of the pipeline."""
        model_capabilities = {}
        for model_name, model in self.models.items():
            try:
                model_capabilities[model_name] = model.get_capabilities()
            except Exception:
                model_capabilities[model_name] = {"error": "Failed to get capabilities"}
        
        return {
            "pipeline_name": "ImageGenerationPipeline",
            "version": "1.0",
            "supported_content_types": ["image"],
            "available_models": list(self.models.keys()),
            "model_capabilities": model_capabilities,
            "selection_strategies": [strategy.value for strategy in ModelSelectionStrategy],
            "features": [
                "Intelligent model selection",
                "Parallel generation",
                "Result comparison",
                "Quality assessment",
                "Cost optimization",
                "Automatic fallback",
                "Multi-model orchestration"
            ],
            "max_concurrent_models": self.max_concurrent_models,
            "supports_comparison": self.enable_result_comparison
        }
    
    async def cleanup(self) -> None:
        """Clean up all models and resources."""
        logger.info("Cleaning up image generation pipeline...")
        
        for model_name, model in self.models.items():
            try:
                await model.cleanup()
                logger.info(f"Cleaned up {model_name}")
            except Exception as e:
                logger.error(f"Error cleaning up {model_name}: {e}")
        
        self.models.clear()
        self.is_initialized = False
        logger.info("Pipeline cleanup completed")
    
    # Additional utility methods
    
    async def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models in the pipeline."""
        status = {}
        
        for model_name, model in self.models.items():
            try:
                model_info = await model.get_model_info()
                status[model_name] = {
                    "available": True,
                    "initialized": getattr(model, 'is_initialized', True),
                    "info": model_info
                }
            except Exception as e:
                status[model_name] = {
                    "available": False,
                    "error": str(e)
                }
        
        return status
    
    async def benchmark_models(self, test_prompts: List[str]) -> Dict[str, Any]:
        """Benchmark all models with test prompts."""
        benchmark_results = {}
        
        for prompt in test_prompts:
            request = ImageGenerationRequest(
                prompt=prompt,
                user_id="benchmark",
                resolution=(1024, 1024),
                num_images=1
            )
            
            prompt_results = {}
            
            for model_name, model in self.models.items():
                try:
                    start_time = time.time()
                    result = await model.generate_image(request)
                    generation_time = time.time() - start_time
                    
                    prompt_results[model_name] = {
                        "success": result.status == GenerationStatus.COMPLETED,
                        "generation_time": generation_time,
                        "cost": result.cost,
                        "quality_score": result.quality_metrics.overall_score if result.quality_metrics else 0.0
                    }
                    
                except Exception as e:
                    prompt_results[model_name] = {
                        "success": False,
                        "error": str(e)
                    }
            
            benchmark_results[prompt] = prompt_results
        
        return benchmark_results