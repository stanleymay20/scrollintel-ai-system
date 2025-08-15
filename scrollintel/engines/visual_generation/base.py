"""
Base classes and interfaces for visual generation engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from enum import Enum
import asyncio


class GenerationStatus(Enum):
    """Status of a generation request."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContentType(Enum):
    """Type of content being generated."""
    IMAGE = "image"
    VIDEO = "video"
    ENHANCED_IMAGE = "enhanced_image"
    ENHANCED_VIDEO = "enhanced_video"


@dataclass
class QualityMetrics:
    """Quality assessment metrics for generated content."""
    overall_score: float = 0.0  # 0-1
    technical_quality: float = 0.0
    aesthetic_score: float = 0.0
    prompt_adherence: float = 0.0
    safety_score: float = 0.0
    uniqueness_score: float = 0.0
    
    # Image-specific metrics
    sharpness: Optional[float] = None
    color_balance: Optional[float] = None
    composition_score: Optional[float] = None
    
    # Video-specific metrics
    temporal_consistency: Optional[float] = None
    motion_smoothness: Optional[float] = None
    frame_quality: Optional[float] = None
    
    # Ultra-realistic metrics
    realism_score: Optional[float] = None
    humanoid_accuracy: Optional[float] = None
    physics_accuracy: Optional[float] = None


@dataclass
class GenerationRequest:
    """Base request for visual content generation."""
    prompt: str
    user_id: str
    content_type: Optional[ContentType] = None
    request_id: str = field(default_factory=lambda: f"req_{datetime.now().timestamp()}")
    negative_prompt: Optional[str] = None
    style: str = "photorealistic"
    quality: str = "high"
    seed: Optional[int] = None
    model_preference: Optional[str] = None
    enhancement_level: str = "standard"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ImageGenerationRequest(GenerationRequest):
    """Request for image generation."""
    resolution: Tuple[int, int] = (1024, 1024)
    aspect_ratio: str = "1:1"
    num_images: int = 1
    guidance_scale: float = 7.5
    steps: int = 50
    
    def __post_init__(self):
        self.content_type = ContentType.IMAGE


@dataclass
class VideoGenerationRequest(GenerationRequest):
    """Request for ultra-realistic video generation."""
    duration: float = 5.0  # seconds
    resolution: Tuple[int, int] = (3840, 2160)  # 4K by default
    fps: int = 60  # 60fps for ultra-realistic quality
    motion_intensity: str = "medium"
    camera_movement: Optional[str] = None
    source_image: Optional[str] = None
    audio_sync: bool = False
    
    # Ultra-realistic specific parameters
    humanoid_generation: bool = False
    physics_simulation: bool = True
    temporal_consistency_level: str = "ultra_high"
    neural_rendering_quality: str = "photorealistic_plus"
    
    def __post_init__(self):
        self.content_type = ContentType.VIDEO


@dataclass
class GenerationResult:
    """Result of a visual generation request."""
    id: str
    request_id: str
    status: GenerationStatus
    content_urls: List[str] = field(default_factory=list)
    content_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Optional[QualityMetrics] = None
    generation_time: float = 0.0
    cost: float = 0.0
    model_used: str = ""
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class BaseVisualGenerator(ABC):
    """Abstract base class for all visual generation engines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = self.__class__.__name__
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the generation engine."""
        pass
    
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate visual content based on the request."""
        pass
    
    @abstractmethod
    async def validate_request(self, request: GenerationRequest) -> bool:
        """Validate if the request can be processed by this generator."""
        pass
    
    @abstractmethod
    async def estimate_cost(self, request: GenerationRequest) -> float:
        """Estimate the cost of processing this request."""
        pass
    
    @abstractmethod
    async def estimate_time(self, request: GenerationRequest) -> float:
        """Estimate the processing time for this request."""
        pass
    
    async def cleanup(self) -> None:
        """Clean up resources used by the generator."""
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of this generator."""
        return {
            "model_name": self.model_name,
            "supported_content_types": [],
            "max_resolution": (1024, 1024),
            "max_duration": 30.0,
            "supports_batch": False,
            "supports_enhancement": False
        }


class ImageGenerator(BaseVisualGenerator):
    """Base class for image generation engines."""
    
    @abstractmethod
    async def generate_image(self, request: ImageGenerationRequest) -> GenerationResult:
        """Generate images based on the request."""
        pass
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Route to appropriate generation method."""
        if isinstance(request, ImageGenerationRequest):
            return await self.generate_image(request)
        else:
            raise ValueError(f"Unsupported request type: {type(request)}")


@dataclass
class ImageGenerationResult:
    """Result of image generation with PIL Images."""
    images: List[Any] = field(default_factory=list)  # List of PIL Images
    model_used: str = ""
    generation_time: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    prompt_used: str = ""
    original_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Optional[QualityMetrics] = None


class BaseImageModel(ABC):
    """Base class for specific image generation models."""
    
    def __init__(self, config):
        self.config = config
        self.model_name = self.__class__.__name__
    
    @abstractmethod
    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Generate images using this specific model."""
        pass
    
    @abstractmethod
    async def validate_request(self, request: ImageGenerationRequest) -> bool:
        """Validate if request is compatible with this model."""
        pass
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about this model."""
        return {
            "name": self.model_name,
            "provider": "Unknown",
            "version": "1.0"
        }


class VideoGenerator(BaseVisualGenerator):
    """Base class for ultra-realistic video generation engines."""
    
    @abstractmethod
    async def generate_video(self, request: VideoGenerationRequest) -> GenerationResult:
        """Generate ultra-realistic videos based on the request."""
        pass
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Route to appropriate generation method."""
        if isinstance(request, VideoGenerationRequest):
            return await self.generate_video(request)
        else:
            raise ValueError(f"Unsupported request type: {type(request)}")


class EnhancementGenerator(BaseVisualGenerator):
    """Base class for content enhancement engines."""
    
    @abstractmethod
    async def enhance_content(self, content_path: str, enhancement_type: str) -> GenerationResult:
        """Enhance existing visual content."""
        pass


class ModelOrchestrator:
    """Orchestrates multiple generation models for optimal results."""
    
    def __init__(self):
        self.generators: Dict[str, BaseVisualGenerator] = {}
        self.model_selector = None
    
    def register_generator(self, name: str, generator: BaseVisualGenerator):
        """Register a new generator."""
        self.generators[name] = generator
    
    async def select_optimal_generator(self, request: GenerationRequest) -> BaseVisualGenerator:
        """Select the best generator for the given request."""
        # Simple selection logic - can be enhanced with ML-based selection
        for generator in self.generators.values():
            if await generator.validate_request(request):
                return generator
        
        raise ValueError("No suitable generator found for request")
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate content using the optimal generator."""
        generator = await self.select_optimal_generator(request)
        return await generator.generate(request)


class GenerationPipeline:
    """Main pipeline for visual content generation."""
    
    def __init__(self, orchestrator: ModelOrchestrator):
        self.orchestrator = orchestrator
        self.safety_filter = None
        self.quality_assessor = None
        self.cache_manager = None
    
    async def process_request(self, request: GenerationRequest) -> GenerationResult:
        """Process a generation request through the complete pipeline."""
        try:
            # 1. Validate request safety
            if self.safety_filter:
                safety_result = await self.safety_filter.validate_request(request)
                if not safety_result.is_safe:
                    return GenerationResult(
                        id=f"result_{request.request_id}",
                        request_id=request.request_id,
                        status=GenerationStatus.FAILED,
                        error_message=f"Safety violation: {safety_result.reason}"
                    )
            
            # 2. Check cache
            if self.cache_manager:
                cached_result = await self.cache_manager.get_cached_result(request)
                if cached_result:
                    return cached_result
            
            # 3. Generate content
            result = await self.orchestrator.generate(request)
            
            # 4. Assess quality
            if self.quality_assessor and result.status == GenerationStatus.COMPLETED:
                quality_metrics = await self.quality_assessor.assess_quality(
                    result.content_paths, request
                )
                result.quality_metrics = quality_metrics
            
            # 5. Cache result
            if self.cache_manager and result.status == GenerationStatus.COMPLETED:
                await self.cache_manager.cache_result(request, result)
            
            return result
            
        except Exception as e:
            return GenerationResult(
                id=f"result_{request.request_id}",
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                error_message=str(e)
            )