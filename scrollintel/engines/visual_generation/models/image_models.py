"""
Image generation model implementations.
"""

import asyncio
import base64
import io
from typing import Dict, Any, List
import time

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from PIL import Image
except ImportError:
    Image = None

from ..base import ImageGenerator, ImageGenerationRequest, GenerationResult, GenerationStatus, QualityMetrics
from ..exceptions import ModelError, APIError, ValidationError
from ..config import ModelConfig


class StableDiffusionXLModel(ImageGenerator):
    """Stable Diffusion XL model implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config.__dict__)
        self.model_config = config
        self.session = None
    
    async def initialize(self) -> None:
        """Initialize the Stable Diffusion XL model."""
        if not self.model_config.api_key:
            raise ModelError("Stability API key not configured")
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.model_config.timeout)
        )
        self.is_initialized = True
    
    async def validate_request(self, request: ImageGenerationRequest) -> bool:
        """Validate if the request can be processed."""
        if not isinstance(request, ImageGenerationRequest):
            return False
        
        max_res = self.model_config.max_resolution
        if request.resolution[0] > max_res[0] or request.resolution[1] > max_res[1]:
            return False
        
        return True
    
    async def estimate_cost(self, request: ImageGenerationRequest) -> float:
        """Estimate the cost of processing this request."""
        base_cost = 0.01  # Base cost per image
        resolution_multiplier = (request.resolution[0] * request.resolution[1]) / (1024 * 1024)
        return base_cost * request.num_images * resolution_multiplier
    
    async def estimate_time(self, request: ImageGenerationRequest) -> float:
        """Estimate the processing time."""
        base_time = 15.0  # Base time in seconds
        resolution_multiplier = (request.resolution[0] * request.resolution[1]) / (1024 * 1024)
        return base_time * resolution_multiplier * request.num_images
    
    async def generate_image(self, request: ImageGenerationRequest) -> GenerationResult:
        """Generate images using Stable Diffusion XL."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "text_prompts": [
                    {"text": request.prompt, "weight": 1.0}
                ],
                "cfg_scale": request.guidance_scale,
                "height": request.resolution[1],
                "width": request.resolution[0],
                "samples": request.num_images,
                "steps": request.steps,
                "seed": request.seed or 0,
                "style_preset": request.style if request.style != "photorealistic" else None
            }
            
            if request.negative_prompt:
                payload["text_prompts"].append({
                    "text": request.negative_prompt,
                    "weight": -1.0
                })
            
            # Make API request
            async with self.session.post(
                self.model_config.api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise APIError(
                        f"Stability API error: {error_text}",
                        api_name="stability_ai",
                        status_code=response.status
                    )
                
                result_data = await response.json()
            
            # Process results
            content_paths = []
            for i, artifact in enumerate(result_data.get("artifacts", [])):
                if artifact.get("finishReason") == "SUCCESS":
                    # Decode base64 image
                    image_data = base64.b64decode(artifact["base64"])
                    image = Image.open(io.BytesIO(image_data))
                    
                    # Save image
                    filename = f"sdxl_{request.request_id}_{i}.png"
                    filepath = f"./generated_content/{filename}"
                    image.save(filepath)
                    content_paths.append(filepath)
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                id=f"result_{request.request_id}",
                request_id=request.request_id,
                status=GenerationStatus.COMPLETED,
                content_paths=content_paths,
                content_urls=[f"/api/content/{path.split('/')[-1]}" for path in content_paths],
                generation_time=generation_time,
                cost=await self.estimate_cost(request),
                model_used="stable_diffusion_xl",
                quality_metrics=QualityMetrics(
                    overall_score=0.85,  # Placeholder - would be calculated by quality assessor
                    technical_quality=0.9,
                    aesthetic_score=0.8,
                    prompt_adherence=0.85
                )
            )
            
        except Exception as e:
            return GenerationResult(
                id=f"result_{request.request_id}",
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                error_message=str(e),
                generation_time=time.time() - start_time,
                model_used="stable_diffusion_xl"
            )
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return model capabilities."""
        return {
            "model_name": "stable_diffusion_xl",
            "supported_content_types": ["image"],
            "max_resolution": self.model_config.max_resolution,
            "max_batch_size": self.model_config.batch_size,
            "supports_negative_prompts": True,
            "supports_style_presets": True,
            "supports_seed_control": True
        }


class DALLE3Model(ImageGenerator):
    """DALL-E 3 model implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config.__dict__)
        self.model_config = config
        self.session = None
    
    async def initialize(self) -> None:
        """Initialize the DALL-E 3 model."""
        if not self.model_config.api_key:
            raise ModelError("OpenAI API key not configured")
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.model_config.timeout)
        )
        self.is_initialized = True
    
    async def validate_request(self, request: ImageGenerationRequest) -> bool:
        """Validate if the request can be processed."""
        if not isinstance(request, ImageGenerationRequest):
            return False
        
        # DALL-E 3 has specific resolution requirements
        supported_resolutions = [(1024, 1024), (1792, 1024), (1024, 1792)]
        if request.resolution not in supported_resolutions:
            return False
        
        # DALL-E 3 only supports 1 image at a time
        if request.num_images > 1:
            return False
        
        return True
    
    async def estimate_cost(self, request: ImageGenerationRequest) -> float:
        """Estimate the cost of processing this request."""
        # DALL-E 3 pricing based on resolution
        if request.resolution == (1024, 1024):
            return 0.04
        else:  # HD resolutions
            return 0.08
    
    async def estimate_time(self, request: ImageGenerationRequest) -> float:
        """Estimate the processing time."""
        return 20.0  # DALL-E 3 typically takes around 20 seconds
    
    async def generate_image(self, request: ImageGenerationRequest) -> GenerationResult:
        """Generate images using DALL-E 3."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json"
            }
            
            # Determine quality and size
            size = f"{request.resolution[0]}x{request.resolution[1]}"
            quality = "hd" if request.resolution != (1024, 1024) else "standard"
            
            payload = {
                "model": "dall-e-3",
                "prompt": request.prompt,
                "n": 1,  # DALL-E 3 only supports 1 image
                "size": size,
                "quality": quality,
                "style": self.model_config.parameters.get("style", "vivid")
            }
            
            # Make API request
            async with self.session.post(
                self.model_config.api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise APIError(
                        f"OpenAI API error: {error_text}",
                        api_name="openai",
                        status_code=response.status
                    )
                
                result_data = await response.json()
            
            # Process results
            content_paths = []
            for i, image_data in enumerate(result_data.get("data", [])):
                image_url = image_data["url"]
                
                # Download image
                async with self.session.get(image_url) as img_response:
                    if img_response.status == 200:
                        image_bytes = await img_response.read()
                        
                        # Save image
                        filename = f"dalle3_{request.request_id}_{i}.png"
                        filepath = f"./generated_content/{filename}"
                        
                        with open(filepath, 'wb') as f:
                            f.write(image_bytes)
                        
                        content_paths.append(filepath)
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                id=f"result_{request.request_id}",
                request_id=request.request_id,
                status=GenerationStatus.COMPLETED,
                content_paths=content_paths,
                content_urls=[f"/api/content/{path.split('/')[-1]}" for path in content_paths],
                generation_time=generation_time,
                cost=await self.estimate_cost(request),
                model_used="dalle3",
                quality_metrics=QualityMetrics(
                    overall_score=0.9,  # DALL-E 3 typically produces high quality
                    technical_quality=0.95,
                    aesthetic_score=0.9,
                    prompt_adherence=0.85
                )
            )
            
        except Exception as e:
            return GenerationResult(
                id=f"result_{request.request_id}",
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                error_message=str(e),
                generation_time=time.time() - start_time,
                model_used="dalle3"
            )
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return model capabilities."""
        return {
            "model_name": "dalle3",
            "supported_content_types": ["image"],
            "supported_resolutions": [(1024, 1024), (1792, 1024), (1024, 1792)],
            "max_batch_size": 1,
            "supports_negative_prompts": False,
            "supports_style_presets": True,
            "supports_seed_control": False
        }


class MidjourneyModel(ImageGenerator):
    """Midjourney model implementation (placeholder for future API)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config.__dict__)
        self.model_config = config
        self.session = None
    
    async def initialize(self) -> None:
        """Initialize the Midjourney model."""
        if not self.model_config.api_key:
            raise ModelError("Midjourney API key not configured")
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.model_config.timeout)
        )
        self.is_initialized = True
    
    async def validate_request(self, request: ImageGenerationRequest) -> bool:
        """Validate if the request can be processed."""
        # Placeholder validation - Midjourney API not yet available
        return False
    
    async def estimate_cost(self, request: ImageGenerationRequest) -> float:
        """Estimate the cost of processing this request."""
        return 0.02 * request.num_images
    
    async def estimate_time(self, request: ImageGenerationRequest) -> float:
        """Estimate the processing time."""
        return 60.0  # Midjourney typically takes longer
    
    async def generate_image(self, request: ImageGenerationRequest) -> GenerationResult:
        """Generate images using Midjourney (placeholder)."""
        # Placeholder implementation - Midjourney API not yet available
        return GenerationResult(
            id=f"result_{request.request_id}",
            request_id=request.request_id,
            status=GenerationStatus.FAILED,
            error_message="Midjourney API not yet available",
            model_used="midjourney"
        )
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return model capabilities."""
        return {
            "model_name": "midjourney",
            "supported_content_types": ["image"],
            "max_resolution": (2048, 2048),
            "max_batch_size": 4,
            "supports_negative_prompts": False,
            "supports_style_presets": True,
            "supports_seed_control": True,
            "status": "coming_soon"
        }