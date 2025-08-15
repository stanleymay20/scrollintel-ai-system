"""
DALL-E 3 API integration for high-quality image generation.
Implements OpenAI DALL-E 3 API with rate limiting, error handling, and quality optimization.
"""

import asyncio
import base64
import io
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import backoff
from PIL import Image
import openai
from openai import AsyncOpenAI

from ..base import BaseImageModel, ImageGenerationRequest, ImageGenerationResult
from ..exceptions import (
    ModelError, 
    RateLimitError, 
    SafetyError, 
    InvalidRequestError,
    APIConnectionError
)
from ..config import VisualGenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class DALLE3Parameters:
    """DALL-E 3 specific parameters."""
    model: str = "dall-e-3"
    size: str = "1024x1024"  # 1024x1024, 1024x1792, 1792x1024
    quality: str = "standard"  # standard, hd
    style: str = "vivid"  # vivid, natural
    response_format: str = "url"  # url, b64_json
    user: Optional[str] = None


class DALLE3Model(BaseImageModel):
    """
    DALL-E 3 model integration with OpenAI API.
    
    Features:
    - High-quality image generation
    - Advanced prompt handling
    - Rate limiting and retry logic
    - Error handling and recovery
    - Image format conversion
    - Quality optimization
    """
    
    def __init__(self, config: VisualGenerationConfig):
        super().__init__(config)
        
        # Get DALL-E 3 model config
        dalle3_config = config.get_model_config('dalle3')
        if not dalle3_config:
            raise ModelError("DALL-E 3 model configuration not found")
        
        # Initialize OpenAI client
        api_key = dalle3_config.api_key or config.get('openai_api_key')
        if not api_key:
            raise ModelError("OpenAI API key not configured for DALL-E 3")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = "dall-e-3"
        
        # Initialize rate limiter
        rpm_limit = dalle3_config.parameters.get('dalle3_rpm_limit', 5)
        rpd_limit = dalle3_config.parameters.get('dalle3_rpd_limit', 200)
        self.rate_limiter = RateLimiter(
            requests_per_minute=rpm_limit,
            requests_per_day=rpd_limit
        )
        
        # Supported sizes for DALL-E 3
        self.supported_sizes = {
            "1024x1024": "1024x1024",
            "1024x1792": "1024x1792", 
            "1792x1024": "1792x1024"
        }
        
        # Quality settings
        self.quality_settings = {
            "standard": "standard",
            "high": "hd",
            "hd": "hd"
        }
        
        # Style settings
        self.style_settings = {
            "vivid": "vivid",
            "natural": "natural",
            "photorealistic": "natural",
            "artistic": "vivid"
        }
    
    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """
        Generate images using DALL-E 3 API.
        
        Args:
            request: Image generation request
            
        Returns:
            ImageGenerationResult with generated images
            
        Raises:
            ModelError: If generation fails
            RateLimitError: If rate limit exceeded
            SafetyError: If content policy violated
        """
        start_time = datetime.now()
        
        try:
            # Check rate limits
            await self.rate_limiter.acquire()
            
            # Prepare parameters
            params = await self._prepare_parameters(request)
            
            # Enhance prompt for DALL-E 3
            enhanced_prompt = await self._enhance_prompt(request.prompt, request.style)
            
            logger.info(f"Generating image with DALL-E 3: {enhanced_prompt[:100]}...")
            
            # Generate image
            response = await self._generate_with_retry(enhanced_prompt, params)
            
            # Process response
            images = await self._process_response(response, request)
            
            # Calculate metrics
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return ImageGenerationResult(
                images=images,
                model_used=self.model_name,
                generation_time=generation_time,
                parameters=params.__dict__,
                prompt_used=enhanced_prompt,
                original_prompt=request.prompt,
                metadata={
                    "revised_prompt": getattr(response.data[0], 'revised_prompt', None),
                    "size": params.size,
                    "quality": params.quality,
                    "style": params.style
                }
            )
            
        except openai.RateLimitError as e:
            logger.warning(f"DALL-E 3 rate limit exceeded: {e}")
            raise RateLimitError(f"Rate limit exceeded: {e}")
            
        except openai.BadRequestError as e:
            logger.error(f"DALL-E 3 bad request: {e}")
            if "content_policy_violation" in str(e).lower():
                raise SafetyError(f"Content policy violation: {e}")
            raise InvalidRequestError(f"Invalid request: {e}")
            
        except openai.APIConnectionError as e:
            logger.error(f"DALL-E 3 connection error: {e}")
            raise APIConnectionError(f"API connection failed: {e}")
            
        except Exception as e:
            logger.error(f"DALL-E 3 generation failed: {e}")
            raise ModelError(f"Generation failed: {e}")
    
    async def _prepare_parameters(self, request: ImageGenerationRequest) -> DALLE3Parameters:
        """Prepare DALL-E 3 specific parameters."""
        # Map resolution to supported size
        size = self._map_resolution_to_size(request.resolution)
        
        # Map quality setting
        quality = self.quality_settings.get(request.quality, "standard")
        
        # Map style setting
        style = self.style_settings.get(request.style, "vivid")
        
        return DALLE3Parameters(
            size=size,
            quality=quality,
            style=style,
            response_format="url",  # We'll download and process
            user=getattr(request, 'user_id', None)
        )
    
    def _map_resolution_to_size(self, resolution: Tuple[int, int]) -> str:
        """Map requested resolution to supported DALL-E 3 size."""
        width, height = resolution
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        if abs(aspect_ratio - 1.0) < 0.1:  # Square
            return "1024x1024"
        elif aspect_ratio > 1.2:  # Landscape
            return "1792x1024"
        else:  # Portrait
            return "1024x1792"
    
    async def _enhance_prompt(self, prompt: str, style: str) -> str:
        """
        Enhance prompt for better DALL-E 3 results.
        
        DALL-E 3 works best with detailed, descriptive prompts.
        """
        enhanced = prompt
        
        # Add style-specific enhancements
        if style == "photorealistic":
            if "photo" not in enhanced.lower():
                enhanced = f"A high-quality photograph of {enhanced}"
        elif style == "artistic":
            if "art" not in enhanced.lower():
                enhanced = f"An artistic rendering of {enhanced}"
        
        # Ensure prompt is within limits (4000 characters for DALL-E 3)
        if len(enhanced) > 4000:
            enhanced = enhanced[:3997] + "..."
        
        return enhanced
    
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIConnectionError),
        max_tries=3,
        max_time=300
    )
    async def _generate_with_retry(self, prompt: str, params: DALLE3Parameters):
        """Generate image with retry logic."""
        try:
            response = await self.client.images.generate(
                model=params.model,
                prompt=prompt,
                size=params.size,
                quality=params.quality,
                style=params.style,
                response_format=params.response_format,
                n=1,  # DALL-E 3 only supports n=1
                user=params.user
            )
            return response
            
        except Exception as e:
            logger.error(f"DALL-E 3 API call failed: {e}")
            raise
    
    async def _process_response(self, response, request: ImageGenerationRequest) -> List[Image.Image]:
        """Process API response and convert to PIL Images."""
        images = []
        
        for image_data in response.data:
            try:
                if hasattr(image_data, 'url') and image_data.url:
                    # Download image from URL
                    image = await self._download_image(image_data.url)
                elif hasattr(image_data, 'b64_json') and image_data.b64_json:
                    # Decode base64 image
                    image = self._decode_base64_image(image_data.b64_json)
                else:
                    raise ModelError("No image data in response")
                
                # Apply quality optimization
                optimized_image = await self._optimize_image(image, request)
                images.append(optimized_image)
                
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                raise ModelError(f"Image processing failed: {e}")
        
        return images
    
    async def _download_image(self, url: str) -> Image.Image:
        """Download image from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        return Image.open(io.BytesIO(image_data))
                    else:
                        raise APIConnectionError(f"Failed to download image: {response.status}")
        except Exception as e:
            logger.error(f"Image download failed: {e}")
            raise APIConnectionError(f"Image download failed: {e}")
    
    def _decode_base64_image(self, b64_data: str) -> Image.Image:
        """Decode base64 image data."""
        try:
            image_data = base64.b64decode(b64_data)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error(f"Base64 decode failed: {e}")
            raise ModelError(f"Base64 decode failed: {e}")
    
    async def _optimize_image(self, image: Image.Image, request: ImageGenerationRequest) -> Image.Image:
        """Apply quality optimization to generated image."""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if needed (DALL-E 3 generates at fixed sizes)
            target_width, target_height = request.resolution
            current_width, current_height = image.size
            
            if (current_width, current_height) != (target_width, target_height):
                # Use high-quality resampling
                image = image.resize(
                    (target_width, target_height),
                    Image.Resampling.LANCZOS
                )
            
            # Apply enhancement based on quality setting
            if request.quality in ["high", "hd"]:
                image = await self._enhance_image_quality(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Image optimization failed: {e}")
            return image  # Return original if optimization fails
    
    async def _enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """Apply quality enhancements to image."""
        try:
            # Apply sharpening filter
            from PIL import ImageFilter, ImageEnhance
            
            # Subtle sharpening
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
            
            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
            
            # Enhance color saturation slightly
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.02)
            
            return image
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        return {
            "name": self.model_name,
            "provider": "OpenAI",
            "version": "dall-e-3",
            "supported_sizes": list(self.supported_sizes.keys()),
            "supported_qualities": list(self.quality_settings.keys()),
            "supported_styles": list(self.style_settings.keys()),
            "max_prompt_length": 4000,
            "max_images_per_request": 1,
            "rate_limits": {
                "requests_per_minute": self.rate_limiter.requests_per_minute,
                "requests_per_day": self.rate_limiter.requests_per_day
            }
        }
    
    async def validate_request(self, request: ImageGenerationRequest) -> bool:
        """Validate if request is compatible with DALL-E 3."""
        # Check prompt length
        if len(request.prompt) > 4000:
            raise InvalidRequestError("Prompt too long for DALL-E 3 (max 4000 characters)")
        
        # Check if we can map the resolution
        try:
            self._map_resolution_to_size(request.resolution)
        except Exception:
            raise InvalidRequestError(f"Unsupported resolution: {request.resolution}")
        
        # DALL-E 3 only supports 1 image per request
        if request.num_images > 1:
            logger.warning("DALL-E 3 only supports 1 image per request, adjusting...")
        
        return True


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int, requests_per_day: int):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.minute_requests = []
        self.day_requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self._lock:
            now = datetime.now()
            
            # Clean old requests
            self._clean_old_requests(now)
            
            # Check limits
            if len(self.minute_requests) >= self.requests_per_minute:
                wait_time = 60 - (now - self.minute_requests[0]).total_seconds()
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    return await self.acquire()
            
            if len(self.day_requests) >= self.requests_per_day:
                wait_time = 86400 - (now - self.day_requests[0]).total_seconds()
                raise RateLimitError(f"Daily rate limit exceeded, wait {wait_time:.0f}s")
            
            # Record request
            self.minute_requests.append(now)
            self.day_requests.append(now)
    
    def _clean_old_requests(self, now: datetime):
        """Remove old request timestamps."""
        minute_ago = now - timedelta(minutes=1)
        day_ago = now - timedelta(days=1)
        
        self.minute_requests = [req for req in self.minute_requests if req > minute_ago]
        self.day_requests = [req for req in self.day_requests if req > day_ago]