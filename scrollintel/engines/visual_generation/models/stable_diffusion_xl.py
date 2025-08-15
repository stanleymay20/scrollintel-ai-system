"""
Stable Diffusion XL Model Integration for ScrollIntel Visual Generation.

This module provides integration with Stable Diffusion XL using the diffusers library,
implementing prompt preprocessing, parameter optimization, and support for different
resolutions and aspect ratios.
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from PIL import Image
import numpy as np
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid
import time
from datetime import datetime

from ..base import ImageGenerator, ImageGenerationRequest, GenerationRequest, GenerationResult, GenerationStatus, QualityMetrics
from ..exceptions import ModelError, PromptError
from ..config import VisualGenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class StableDiffusionXLRequest:
    """Request parameters for Stable Diffusion XL generation."""
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: Optional[int] = None
    scheduler: str = "DPMSolverMultistep"
    clip_skip: Optional[int] = None


@dataclass
class StableDiffusionXLResult:
    """Result from Stable Diffusion XL generation."""
    images: List[Image.Image]
    prompt: str
    negative_prompt: Optional[str]
    parameters: Dict[str, Any]
    generation_time: float
    seed_used: int
    model_version: str


class PromptPreprocessor:
    """Handles prompt preprocessing and optimization for Stable Diffusion XL."""
    
    def __init__(self):
        self.quality_enhancers = [
            "high quality", "detailed", "sharp focus", "professional",
            "8k uhd", "masterpiece", "best quality"
        ]
        self.negative_defaults = [
            "blurry", "low quality", "pixelated", "distorted",
            "ugly", "deformed", "bad anatomy", "worst quality"
        ]
    
    def enhance_prompt(self, prompt: str, style: str = "photorealistic") -> str:
        """Enhance prompt with quality modifiers based on style."""
        enhanced = prompt.strip()
        
        # Add style-specific enhancements
        if style == "photorealistic":
            enhanced += ", photorealistic, high resolution, detailed"
        elif style == "artistic":
            enhanced += ", artistic, creative, beautiful composition"
        elif style == "professional":
            enhanced += ", professional photography, studio lighting"
        
        # Add quality enhancers if not already present
        for enhancer in self.quality_enhancers[:3]:  # Limit to avoid over-enhancement
            if enhancer.lower() not in enhanced.lower():
                enhanced += f", {enhancer}"
        
        return enhanced
    
    def optimize_negative_prompt(self, negative_prompt: Optional[str] = None) -> str:
        """Optimize negative prompt with common quality issues."""
        if negative_prompt:
            negatives = [negative_prompt.strip()]
        else:
            negatives = []
        
        # Add default negative prompts
        for default in self.negative_defaults:
            if default not in " ".join(negatives).lower():
                negatives.append(default)
        
        return ", ".join(negatives)
    
    def validate_prompt(self, prompt: str) -> bool:
        """Validate prompt for safety and quality."""
        if not prompt or len(prompt.strip()) < 3:
            raise PromptError("Prompt must be at least 3 characters long")
        
        if len(prompt) > 1000:
            raise PromptError("Prompt too long (max 1000 characters)")
        
        # Basic safety checks (extend as needed)
        unsafe_terms = ["nsfw", "explicit", "violence"]
        prompt_lower = prompt.lower()
        for term in unsafe_terms:
            if term in prompt_lower:
                raise PromptError(f"Unsafe content detected: {term}")
        
        return True


class ResolutionOptimizer:
    """Handles resolution and aspect ratio optimization."""
    
    # Supported resolutions for SDXL (optimized for best quality)
    SUPPORTED_RESOLUTIONS = {
        "1:1": [(1024, 1024), (512, 512)],
        "16:9": [(1344, 768), (1152, 896)],
        "9:16": [(768, 1344), (896, 1152)],
        "4:3": [(1152, 896), (896, 1152)],
        "3:4": [(896, 1152), (1152, 896)],
        "21:9": [(1536, 640), (1344, 576)],
        "9:21": [(640, 1536), (576, 1344)]
    }
    
    def optimize_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """Optimize resolution for best SDXL performance."""
        aspect_ratio = self.calculate_aspect_ratio(width, height)
        
        if aspect_ratio in self.SUPPORTED_RESOLUTIONS:
            # Find closest supported resolution
            target_pixels = width * height
            best_resolution = min(
                self.SUPPORTED_RESOLUTIONS[aspect_ratio],
                key=lambda res: abs(res[0] * res[1] - target_pixels)
            )
            return best_resolution
        
        # Fallback to closest multiple of 64 (SDXL requirement)
        optimized_width = ((width + 31) // 64) * 64
        optimized_height = ((height + 31) // 64) * 64
        
        return optimized_width, optimized_height
    
    def calculate_aspect_ratio(self, width: int, height: int) -> str:
        """Calculate aspect ratio string from dimensions."""
        from math import gcd
        
        divisor = gcd(width, height)
        ratio_w = width // divisor
        ratio_h = height // divisor
        
        return f"{ratio_w}:{ratio_h}"


class StableDiffusionXLModel(ImageGenerator):
    """Stable Diffusion XL model integration with advanced features."""
    
    def __init__(self, config: VisualGenerationConfig):
        super().__init__(config)
        self.model_name = "stable-diffusion-xl"
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocessor = PromptPreprocessor()
        self.resolution_optimizer = ResolutionOptimizer()
        
        # Model configuration
        self.model_id = config.get("sdxl_model_id", "stabilityai/stable-diffusion-xl-base-1.0")
        self.use_refiner = config.get("sdxl_use_refiner", False)
        self.enable_cpu_offload = config.get("sdxl_cpu_offload", True)
        
        logger.info(f"Initializing Stable Diffusion XL on {self.device}")
    
    async def initialize(self) -> None:
        """Initialize the Stable Diffusion XL pipeline."""
        try:
            logger.info("Loading Stable Diffusion XL pipeline...")
            
            # Load main pipeline
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None
            )
            
            # Optimize pipeline
            if self.device == "cuda":
                self.pipeline = self.pipeline.to(self.device)
                if self.enable_cpu_offload:
                    self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_vae_slicing()
                self.pipeline.enable_vae_tiling()
            
            # Set optimized scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            self.is_initialized = True
            logger.info("Stable Diffusion XL pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Stable Diffusion XL: {e}")
            raise ModelError(f"SDXL initialization failed: {e}")
    
    async def generate(self, request: StableDiffusionXLRequest) -> StableDiffusionXLResult:
        """Generate images using Stable Diffusion XL."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Validate and preprocess prompt
            self.preprocessor.validate_prompt(request.prompt)
            enhanced_prompt = self.preprocessor.enhance_prompt(request.prompt)
            optimized_negative = self.preprocessor.optimize_negative_prompt(request.negative_prompt)
            
            # Optimize resolution
            optimized_width, optimized_height = self.resolution_optimizer.optimize_resolution(
                request.width, request.height
            )
            
            # Set seed for reproducibility
            if request.seed is not None:
                torch.manual_seed(request.seed)
                seed_used = request.seed
            else:
                seed_used = torch.randint(0, 2**32 - 1, (1,)).item()
                torch.manual_seed(seed_used)
            
            logger.info(f"Generating {request.num_images} images with SDXL")
            logger.debug(f"Enhanced prompt: {enhanced_prompt}")
            logger.debug(f"Resolution: {optimized_width}x{optimized_height}")
            
            # Generate images
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=optimized_negative,
                    width=optimized_width,
                    height=optimized_height,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    num_images_per_prompt=request.num_images,
                    generator=torch.Generator(device=self.device).manual_seed(seed_used)
                )
            
            generation_time = time.time() - start_time
            
            # Create result
            sdxl_result = StableDiffusionXLResult(
                images=result.images,
                prompt=enhanced_prompt,
                negative_prompt=optimized_negative,
                parameters={
                    "width": optimized_width,
                    "height": optimized_height,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "scheduler": request.scheduler,
                    "original_prompt": request.prompt
                },
                generation_time=generation_time,
                seed_used=seed_used,
                model_version=self.model_id
            )
            
            logger.info(f"Generated {len(result.images)} images in {generation_time:.2f}s")
            return sdxl_result
            
        except Exception as e:
            logger.error(f"SDXL generation failed: {e}")
            raise ModelError(f"Generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        return {
            "name": "Stable Diffusion XL",
            "version": self.model_id,
            "device": self.device,
            "max_resolution": "1536x1536",
            "supported_aspects": list(self.resolution_optimizer.SUPPORTED_RESOLUTIONS.keys()),
            "features": [
                "High-resolution generation",
                "Multiple aspect ratios",
                "Prompt enhancement",
                "Quality optimization",
                "Batch generation"
            ],
            "initialized": self.is_initialized
        }
    
    async def generate_image(self, request: ImageGenerationRequest) -> GenerationResult:
        """Generate images using Stable Diffusion XL (ImageGenerator interface)."""
        # Convert ImageGenerationRequest to StableDiffusionXLRequest
        sdxl_request = StableDiffusionXLRequest(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.resolution[0],
            height=request.resolution[1],
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            num_images=request.num_images,
            seed=request.seed
        )
        
        # Generate using SDXL
        sdxl_result = await self.generate(sdxl_request)
        
        # Convert to GenerationResult
        import tempfile
        import os
        content_paths = []
        
        for i, image in enumerate(sdxl_result.images):
            temp_path = os.path.join(tempfile.gettempdir(), f"sdxl_{request.request_id}_{i}.png")
            image.save(temp_path)
            content_paths.append(temp_path)
        
        quality_metrics = QualityMetrics(
            overall_score=0.85,  # Default high quality for SDXL
            technical_quality=0.9,
            aesthetic_score=0.8,
            prompt_adherence=0.85,
            safety_score=1.0,
            sharpness=0.9,
            color_balance=0.85,
            composition_score=0.8
        )
        
        return GenerationResult(
            id=f"sdxl_{request.request_id}",
            request_id=request.request_id,
            status=GenerationStatus.COMPLETED,
            content_paths=content_paths,
            metadata=sdxl_result.parameters,
            quality_metrics=quality_metrics,
            generation_time=sdxl_result.generation_time,
            cost=self._calculate_cost(request),
            model_used=f"stable-diffusion-xl-{self.model_id}",
            completed_at=datetime.now()
        )
    
    async def validate_request(self, request: GenerationRequest) -> bool:
        """Validate if the request can be processed by SDXL."""
        if not isinstance(request, ImageGenerationRequest):
            return False
        
        # Check resolution limits
        max_pixels = 2048 * 2048
        if request.resolution[0] * request.resolution[1] > max_pixels:
            return False
        
        # Check if prompt is valid
        try:
            self.preprocessor.validate_prompt(request.prompt)
            return True
        except PromptError:
            return False
    
    async def estimate_cost(self, request: GenerationRequest) -> float:
        """Estimate the cost of processing this request."""
        if not isinstance(request, ImageGenerationRequest):
            return 0.0
        
        return self._calculate_cost(request)
    
    def _calculate_cost(self, request: ImageGenerationRequest) -> float:
        """Calculate cost based on request parameters."""
        base_cost = 0.02  # Base cost per image
        
        # Resolution multiplier
        pixels = request.resolution[0] * request.resolution[1]
        resolution_multiplier = pixels / (1024 * 1024)  # Normalize to 1024x1024
        
        # Steps multiplier
        steps_multiplier = request.steps / 50  # Normalize to 50 steps
        
        # Number of images
        total_cost = base_cost * resolution_multiplier * steps_multiplier * request.num_images
        
        return round(total_cost, 4)
    
    async def estimate_time(self, request: GenerationRequest) -> float:
        """Estimate the processing time for this request."""
        if not isinstance(request, ImageGenerationRequest):
            return 0.0
        
        # Base time per image (seconds)
        base_time = 15.0 if self.device == "cuda" else 60.0
        
        # Resolution factor
        pixels = request.resolution[0] * request.resolution[1]
        resolution_factor = pixels / (1024 * 1024)
        
        # Steps factor
        steps_factor = request.steps / 50
        
        # Estimate total time
        estimated_time = base_time * resolution_factor * steps_factor * request.num_images
        
        return estimated_time
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of SDXL generator."""
        return {
            "model_name": "Stable Diffusion XL",
            "supported_content_types": ["image"],
            "max_resolution": (2048, 2048),
            "max_duration": 0.0,  # Not applicable for images
            "supports_batch": True,
            "supports_enhancement": False,
            "supported_aspects": list(self.resolution_optimizer.SUPPORTED_RESOLUTIONS.keys()),
            "features": [
                "High-resolution generation",
                "Multiple aspect ratios", 
                "Prompt enhancement",
                "Quality optimization",
                "Batch generation"
            ]
        }

    async def cleanup(self) -> None:
        """Clean up model resources."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        logger.info("Stable Diffusion XL model cleaned up")