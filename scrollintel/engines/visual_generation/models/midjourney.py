"""
Midjourney API integration for high-quality image generation.
Implements Discord bot API integration with job queuing, status polling, and retry logic.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import backoff
from PIL import Image
import io
import base64

from ..base import BaseImageModel, ImageGenerationRequest, ImageGenerationResult, QualityMetrics
from ..exceptions import (
    ModelError, 
    RateLimitError, 
    SafetyError, 
    InvalidRequestError,
    APIConnectionError,
    TimeoutError,
    ValidationError
)
from ..config import VisualGenerationConfig

logger = logging.getLogger(__name__)


class MidjourneyJobStatus(Enum):
    """Status of a Midjourney generation job."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class MidjourneyJob:
    """Represents a Midjourney generation job."""
    job_id: str
    prompt: str
    status: MidjourneyJobStatus = MidjourneyJobStatus.PENDING
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    image_urls: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MidjourneyParameters:
    """Midjourney-specific generation parameters."""
    aspect_ratio: str = "1:1"  # 1:1, 16:9, 9:16, 4:3, 3:4, etc.
    quality: str = "1"  # 0.25, 0.5, 1, 2
    stylize: int = 100  # 0-1000
    chaos: int = 0  # 0-100
    version: str = "6"  # Midjourney version
    style: Optional[str] = None  # raw, cute, expressive, original, scenic
    seed: Optional[int] = None
    stop: Optional[int] = None  # 10-100
    weird: Optional[int] = None  # 0-3000
    tile: bool = False
    no_text: bool = False
    upscale: bool = False
    variation: bool = False


class MidjourneyPromptFormatter:
    """Handles Midjourney-specific prompt formatting and optimization."""
    
    def __init__(self):
        self.parameter_map = {
            "aspect_ratio": "--ar",
            "quality": "--q",
            "stylize": "--stylize",
            "chaos": "--chaos",
            "version": "--v",
            "style": "--style",
            "seed": "--seed",
            "stop": "--stop",
            "weird": "--weird",
            "tile": "--tile",
            "no_text": "--no"
        }
        
        # Midjourney style keywords
        self.style_enhancers = {
            "photorealistic": ["photorealistic", "hyperrealistic", "8k", "detailed"],
            "artistic": ["artistic", "creative", "beautiful", "masterpiece"],
            "cinematic": ["cinematic", "dramatic lighting", "film photography"],
            "anime": ["anime style", "manga", "japanese art"],
            "oil_painting": ["oil painting", "classical art", "renaissance"],
            "watercolor": ["watercolor", "soft colors", "artistic medium"],
            "sketch": ["pencil sketch", "line art", "drawing"],
            "digital_art": ["digital art", "concept art", "digital painting"]
        }
    
    def format_prompt(self, prompt: str, params: MidjourneyParameters) -> str:
        """Format prompt with Midjourney-specific parameters."""
        formatted_prompt = prompt.strip()
        
        # Add parameter flags
        parameter_string = self._build_parameter_string(params)
        if parameter_string:
            formatted_prompt += f" {parameter_string}"
        
        # Validate prompt length (Midjourney has limits)
        if len(formatted_prompt) > 4000:
            logger.warning("Prompt too long, truncating...")
            formatted_prompt = formatted_prompt[:3997] + "..."
        
        return formatted_prompt
    
    def _build_parameter_string(self, params: MidjourneyParameters) -> str:
        """Build parameter string from MidjourneyParameters."""
        param_parts = []
        
        # Add aspect ratio
        if params.aspect_ratio and params.aspect_ratio != "1:1":
            param_parts.append(f"--ar {params.aspect_ratio}")
        
        # Add quality
        if params.quality != "1":
            param_parts.append(f"--q {params.quality}")
        
        # Add stylize
        if params.stylize != 100:
            param_parts.append(f"--stylize {params.stylize}")
        
        # Add chaos
        if params.chaos > 0:
            param_parts.append(f"--chaos {params.chaos}")
        
        # Add version
        if params.version != "6":
            param_parts.append(f"--v {params.version}")
        
        # Add style
        if params.style:
            param_parts.append(f"--style {params.style}")
        
        # Add seed
        if params.seed is not None:
            param_parts.append(f"--seed {params.seed}")
        
        # Add stop
        if params.stop is not None:
            param_parts.append(f"--stop {params.stop}")
        
        # Add weird
        if params.weird is not None:
            param_parts.append(f"--weird {params.weird}")
        
        # Add tile
        if params.tile:
            param_parts.append("--tile")
        
        # Add no text
        if params.no_text:
            param_parts.append("--no text")
        
        return " ".join(param_parts)
    
    def enhance_prompt_for_style(self, prompt: str, style: str) -> str:
        """Enhance prompt with style-specific keywords."""
        if style in self.style_enhancers:
            enhancers = self.style_enhancers[style]
            # Add enhancers that aren't already in the prompt
            for enhancer in enhancers:
                if enhancer.lower() not in prompt.lower():
                    prompt += f", {enhancer}"
        
        return prompt
    
    def validate_prompt(self, prompt: str) -> bool:
        """Validate prompt for Midjourney compatibility."""
        if not prompt or len(prompt.strip()) < 3:
            raise ValidationError("Prompt must be at least 3 characters long")
        
        if len(prompt) > 4000:
            raise ValidationError("Prompt too long (max 4000 characters)")
        
        # Check for banned content (basic safety)
        banned_terms = ["nsfw", "explicit", "nude", "sexual", "violence", "gore"]
        prompt_lower = prompt.lower()
        for term in banned_terms:
            if term in prompt_lower:
                raise SafetyError(f"Potentially unsafe content detected: {term}")
        
        return True


class MidjourneyJobQueue:
    """Manages Midjourney job queue and status polling."""
    
    def __init__(self, max_concurrent_jobs: int = 3, poll_interval: float = 5.0):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.poll_interval = poll_interval
        self.jobs: Dict[str, MidjourneyJob] = {}
        self.active_jobs: Dict[str, MidjourneyJob] = {}
        self.completed_jobs: Dict[str, MidjourneyJob] = {}
        self._lock = asyncio.Lock()
        self._polling_task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    async def submit_job(self, job: MidjourneyJob) -> str:
        """Submit a new job to the queue."""
        async with self._lock:
            self.jobs[job.job_id] = job
            logger.info(f"Job {job.job_id} submitted to queue")
            
            # Start polling if not already running
            if not self._polling_task or self._polling_task.done():
                self._polling_task = asyncio.create_task(self._poll_jobs())
            
            return job.job_id
    
    async def get_job_status(self, job_id: str) -> Optional[MidjourneyJob]:
        """Get the current status of a job."""
        return self.jobs.get(job_id) or self.completed_jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's still pending or processing."""
        async with self._lock:
            job = self.jobs.get(job_id)
            if job and job.status in [MidjourneyJobStatus.PENDING, MidjourneyJobStatus.QUEUED]:
                job.status = MidjourneyJobStatus.CANCELLED
                logger.info(f"Job {job_id} cancelled")
                return True
            return False
    
    async def _poll_jobs(self):
        """Poll active jobs for status updates."""
        while not self._shutdown:
            try:
                async with self._lock:
                    # Move pending jobs to active if we have capacity
                    pending_jobs = [job for job in self.jobs.values() 
                                  if job.status == MidjourneyJobStatus.PENDING]
                    
                    available_slots = self.max_concurrent_jobs - len(self.active_jobs)
                    
                    for job in pending_jobs[:available_slots]:
                        job.status = MidjourneyJobStatus.QUEUED
                        job.started_at = datetime.now()
                        self.active_jobs[job.job_id] = job
                        logger.info(f"Job {job.job_id} moved to active queue")
                
                # Poll active jobs (outside lock to avoid blocking)
                active_job_ids = list(self.active_jobs.keys())
                for job_id in active_job_ids:
                    await self._poll_single_job(job_id)
                
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Error in job polling: {e}")
                await asyncio.sleep(self.poll_interval)
    
    async def _poll_single_job(self, job_id: str):
        """Poll a single job for status updates."""
        job = self.active_jobs.get(job_id)
        if not job:
            return
        
        try:
            # This would be replaced with actual Midjourney API status check
            # For now, simulate job progression
            await self._simulate_job_progress(job)
            
        except Exception as e:
            logger.error(f"Error polling job {job_id}: {e}")
            job.status = MidjourneyJobStatus.FAILED
            job.error_message = str(e)
            await self._complete_job(job)
    
    async def _simulate_job_progress(self, job: MidjourneyJob):
        """Simulate job progress (replace with actual API calls)."""
        elapsed = (datetime.now() - job.started_at).total_seconds()
        
        if elapsed < 10:
            job.status = MidjourneyJobStatus.PROCESSING
            job.progress = min(0.9, elapsed / 10.0)
        else:
            # Simulate completion
            job.status = MidjourneyJobStatus.COMPLETED
            job.progress = 1.0
            job.completed_at = datetime.now()
            job.image_urls = [f"https://example.com/generated_{job.job_id}.png"]
            await self._complete_job(job)
    
    async def _complete_job(self, job: MidjourneyJob):
        """Move job to completed status."""
        async with self._lock:
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
            self.completed_jobs[job.job_id] = job
            
            # Clean up old completed jobs (keep last 100)
            if len(self.completed_jobs) > 100:
                oldest_jobs = sorted(
                    self.completed_jobs.values(),
                    key=lambda j: j.completed_at or j.created_at
                )[:len(self.completed_jobs) - 100]
                
                for old_job in oldest_jobs:
                    del self.completed_jobs[old_job.job_id]
    
    async def shutdown(self):
        """Shutdown the job queue."""
        self._shutdown = True
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass


class MidjourneyModel(BaseImageModel):
    """
    Midjourney model integration with Discord bot API.
    
    Features:
    - Discord bot API integration
    - Job queuing and status polling
    - Midjourney-specific prompt formatting
    - Retry logic and error handling
    - Parameter optimization
    - Quality assessment
    """
    
    def __init__(self, config: VisualGenerationConfig):
        super().__init__(config)
        
        # Get Midjourney model config
        midjourney_config = config.get_model_config('midjourney')
        if not midjourney_config:
            raise ModelError("Midjourney model configuration not found")
        
        # Discord bot configuration
        self.bot_token = midjourney_config.api_key
        self.server_id = midjourney_config.parameters.get('server_id')
        self.channel_id = midjourney_config.parameters.get('channel_id')
        self.application_id = midjourney_config.parameters.get('application_id')
        
        if not all([self.bot_token, self.server_id, self.channel_id]):
            raise ModelError("Missing required Midjourney Discord bot configuration")
        
        # API endpoints
        self.discord_api_base = "https://discord.com/api/v10"
        self.midjourney_api_base = midjourney_config.parameters.get(
            'api_base', 'https://api.midjourney.com/v1'
        )
        
        # Initialize components
        self.prompt_formatter = MidjourneyPromptFormatter()
        self.job_queue = MidjourneyJobQueue(
            max_concurrent_jobs=midjourney_config.parameters.get('max_concurrent_jobs', 3),
            poll_interval=midjourney_config.parameters.get('poll_interval', 5.0)
        )
        
        # Rate limiting
        self.max_requests_per_minute = midjourney_config.parameters.get('max_rpm', 10)
        self.request_timestamps = []
        
        # Retry configuration
        self.max_retries = midjourney_config.parameters.get('max_retries', 3)
        self.retry_delay = midjourney_config.parameters.get('retry_delay', 5.0)
        
        # Timeout configuration
        self.generation_timeout = midjourney_config.parameters.get('generation_timeout', 300)  # 5 minutes
        
        self.model_name = "midjourney"
        logger.info("Midjourney model initialized")
    
    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """
        Generate images using Midjourney API.
        
        Args:
            request: Image generation request
            
        Returns:
            ImageGenerationResult with generated images
            
        Raises:
            ModelError: If generation fails
            RateLimitError: If rate limit exceeded
            SafetyError: If content policy violated
            TimeoutError: If generation times out
        """
        start_time = datetime.now()
        
        try:
            # Validate request
            await self.validate_request(request)
            
            # Check rate limits
            await self._check_rate_limit()
            
            # Prepare parameters
            params = await self._prepare_parameters(request)
            
            # Format prompt
            formatted_prompt = self.prompt_formatter.format_prompt(request.prompt, params)
            
            logger.info(f"Generating image with Midjourney: {formatted_prompt[:100]}...")
            
            # Submit job with retry logic
            job = await self._submit_job_with_retry(formatted_prompt, params, request)
            
            # Wait for completion
            result_job = await self._wait_for_completion(job.job_id)
            
            # Process results
            images = await self._process_job_result(result_job, request)
            
            # Calculate metrics
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return ImageGenerationResult(
                images=images,
                model_used=self.model_name,
                generation_time=generation_time,
                parameters=params.__dict__,
                prompt_used=formatted_prompt,
                original_prompt=request.prompt,
                metadata={
                    "job_id": result_job.job_id,
                    "midjourney_version": params.version,
                    "aspect_ratio": params.aspect_ratio,
                    "quality": params.quality,
                    "stylize": params.stylize,
                    "chaos": params.chaos
                }
            )
            
        except (SafetyError, ValidationError, RateLimitError, TimeoutError):
            # Re-raise these specific errors
            raise
            
        except Exception as e:
            logger.error(f"Midjourney generation failed: {e}")
            raise ModelError(f"Generation failed: {e}")
    
    async def _prepare_parameters(self, request: ImageGenerationRequest) -> MidjourneyParameters:
        """Prepare Midjourney-specific parameters from request."""
        # Map resolution to aspect ratio
        aspect_ratio = self._calculate_aspect_ratio(request.resolution)
        
        # Map quality setting
        quality_map = {
            "low": "0.25",
            "standard": "0.5", 
            "high": "1",
            "ultra": "2"
        }
        quality = quality_map.get(request.quality, "1")
        
        # Map style to Midjourney parameters
        stylize = 100  # Default
        chaos = 0
        style = None
        
        if request.style == "photorealistic":
            stylize = 50
            style = "raw"
        elif request.style == "artistic":
            stylize = 250
        elif request.style == "creative":
            stylize = 500
            chaos = 25
        elif request.style == "abstract":
            stylize = 750
            chaos = 50
        
        return MidjourneyParameters(
            aspect_ratio=aspect_ratio,
            quality=quality,
            stylize=stylize,
            chaos=chaos,
            style=style,
            seed=request.seed,
            version="6"  # Use latest version
        )
    
    def _calculate_aspect_ratio(self, resolution: Tuple[int, int]) -> str:
        """Calculate aspect ratio string from resolution."""
        width, height = resolution
        
        # Calculate GCD for simplification
        from math import gcd
        divisor = gcd(width, height)
        simplified_width = width // divisor
        simplified_height = height // divisor
        
        # Map to common Midjourney aspect ratios
        ratio = simplified_width / simplified_height
        
        if abs(ratio - 1.0) < 0.1:
            return "1:1"
        elif abs(ratio - 16/9) < 0.1:
            return "16:9"
        elif abs(ratio - 9/16) < 0.1:
            return "9:16"
        elif abs(ratio - 4/3) < 0.1:
            return "4:3"
        elif abs(ratio - 3/4) < 0.1:
            return "3:4"
        elif abs(ratio - 21/9) < 0.1:
            return "21:9"
        else:
            return f"{simplified_width}:{simplified_height}"
    
    async def _check_rate_limit(self):
        """Check and enforce rate limits."""
        now = datetime.now()
        
        # Clean old timestamps
        cutoff = now - timedelta(minutes=1)
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]
        
        # Check limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.request_timestamps[0]).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                raise RateLimitError(f"Rate limit exceeded, retry after {wait_time:.0f}s")
        
        # Record request
        self.request_timestamps.append(now)
    
    @backoff.on_exception(
        backoff.expo,
        (APIConnectionError, ModelError),
        max_tries=3,
        max_time=60
    )
    async def _submit_job_with_retry(
        self, 
        prompt: str, 
        params: MidjourneyParameters,
        request: ImageGenerationRequest
    ) -> MidjourneyJob:
        """Submit job to Midjourney with retry logic."""
        try:
            # Create job
            job_id = f"mj_{int(time.time() * 1000)}_{hash(prompt) % 10000}"
            job = MidjourneyJob(
                job_id=job_id,
                prompt=prompt,
                metadata={
                    "request_id": request.request_id,
                    "user_id": request.user_id,
                    "parameters": params.__dict__
                }
            )
            
            # Submit to Discord API (simulated for now)
            await self._submit_to_discord_api(job, params)
            
            # Add to job queue
            await self.job_queue.submit_job(job)
            
            logger.info(f"Job {job_id} submitted successfully")
            return job
            
        except Exception as e:
            logger.error(f"Failed to submit job: {e}")
            raise APIConnectionError(f"Failed to submit job: {e}")
    
    async def _submit_to_discord_api(self, job: MidjourneyJob, params: MidjourneyParameters):
        """Submit generation request to Discord API."""
        headers = {
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json"
        }
        
        # Prepare Discord slash command payload
        payload = {
            "type": 2,  # APPLICATION_COMMAND
            "application_id": self.application_id,
            "guild_id": self.server_id,
            "channel_id": self.channel_id,
            "session_id": f"session_{job.job_id}",
            "data": {
                "version": "1166847114203123795",  # Midjourney command version
                "id": "938956540159881230",  # Midjourney command ID
                "name": "imagine",
                "type": 1,
                "options": [
                    {
                        "type": 3,
                        "name": "prompt",
                        "value": job.prompt
                    }
                ]
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.discord_api_base}/interactions"
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 204:
                        logger.info(f"Successfully submitted job {job.job_id} to Discord")
                        job.status = MidjourneyJobStatus.QUEUED
                    else:
                        error_text = await response.text()
                        logger.error(f"Discord API error: {response.status} - {error_text}")
                        raise APIConnectionError(f"Discord API error: {response.status}")
                        
        except aiohttp.ClientError as e:
            logger.error(f"Discord API connection error: {e}")
            raise APIConnectionError(f"Discord API connection failed: {e}")
    
    async def _wait_for_completion(self, job_id: str) -> MidjourneyJob:
        """Wait for job completion with timeout."""
        start_time = datetime.now()
        timeout = timedelta(seconds=self.generation_timeout)
        
        while datetime.now() - start_time < timeout:
            job = await self.job_queue.get_job_status(job_id)
            
            if not job:
                raise ModelError(f"Job {job_id} not found")
            
            if job.status == MidjourneyJobStatus.COMPLETED:
                logger.info(f"Job {job_id} completed successfully")
                return job
            elif job.status == MidjourneyJobStatus.FAILED:
                raise ModelError(f"Job {job_id} failed: {job.error_message}")
            elif job.status == MidjourneyJobStatus.CANCELLED:
                raise ModelError(f"Job {job_id} was cancelled")
            
            # Log progress
            if job.progress > 0:
                logger.info(f"Job {job_id} progress: {job.progress:.1%}")
            
            await asyncio.sleep(2.0)  # Poll every 2 seconds
        
        # Timeout reached
        await self.job_queue.cancel_job(job_id)
        raise TimeoutError(f"Job {job_id} timed out after {self.generation_timeout}s")
    
    async def _process_job_result(self, job: MidjourneyJob, request: ImageGenerationRequest) -> List[Image.Image]:
        """Process completed job result and download images."""
        if not job.image_urls:
            raise ModelError("No images generated")
        
        images = []
        
        for url in job.image_urls:
            try:
                image = await self._download_image(url)
                
                # Apply post-processing
                processed_image = await self._post_process_image(image, request)
                images.append(processed_image)
                
            except Exception as e:
                logger.error(f"Failed to process image from {url}: {e}")
                # Continue with other images if available
                continue
        
        if not images:
            raise ModelError("Failed to process any generated images")
        
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
                        
        except aiohttp.ClientError as e:
            logger.error(f"Image download failed: {e}")
            raise APIConnectionError(f"Image download failed: {e}")
    
    async def _post_process_image(self, image: Image.Image, request: ImageGenerationRequest) -> Image.Image:
        """Apply post-processing to generated image."""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to requested resolution if needed
            target_width, target_height = request.resolution
            current_width, current_height = image.size
            
            if (current_width, current_height) != (target_width, target_height):
                # Use high-quality resampling
                image = image.resize(
                    (target_width, target_height),
                    Image.Resampling.LANCZOS
                )
            
            # Apply quality enhancements for high-quality requests
            if request.quality in ["high", "ultra"]:
                image = await self._enhance_image_quality(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Image post-processing failed: {e}")
            return image  # Return original if processing fails
    
    async def _enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """Apply quality enhancements to image."""
        try:
            from PIL import ImageFilter, ImageEnhance
            
            # Subtle sharpening
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=105, threshold=3))
            
            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.02)
            
            return image
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        return {
            "name": self.model_name,
            "provider": "Midjourney",
            "version": "6",
            "supported_aspects": ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"],
            "supported_qualities": ["0.25", "0.5", "1", "2"],
            "max_prompt_length": 4000,
            "max_concurrent_jobs": self.job_queue.max_concurrent_jobs,
            "generation_timeout": self.generation_timeout,
            "rate_limits": {
                "requests_per_minute": self.max_requests_per_minute
            },
            "features": [
                "High-quality artistic generation",
                "Multiple aspect ratios",
                "Style control",
                "Chaos and stylize parameters",
                "Seed support",
                "Job queuing",
                "Status polling"
            ]
        }
    
    async def validate_request(self, request: ImageGenerationRequest) -> bool:
        """Validate if request is compatible with Midjourney."""
        # Validate prompt
        self.prompt_formatter.validate_prompt(request.prompt)
        
        # Check if we can map the resolution to an aspect ratio
        aspect_ratio = self._calculate_aspect_ratio(request.resolution)
        if not aspect_ratio:
            raise ValidationError(f"Unsupported resolution: {request.resolution}")
        
        # Midjourney generates one image at a time
        if request.num_images > 1:
            logger.warning("Midjourney generates one image per request, will need multiple jobs")
        
        return True
    
    async def cleanup(self):
        """Clean up model resources."""
        await self.job_queue.shutdown()
        logger.info("Midjourney model cleaned up")