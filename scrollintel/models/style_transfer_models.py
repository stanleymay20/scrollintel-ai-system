"""
Data models for style transfer operations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from datetime import datetime


class StyleTransferStatus(Enum):
    """Status of style transfer operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtisticStyle(Enum):
    """Available artistic styles for transfer."""
    IMPRESSIONIST = "impressionist"
    CUBIST = "cubist"
    ABSTRACT = "abstract"
    WATERCOLOR = "watercolor"
    OIL_PAINTING = "oil_painting"
    PENCIL_SKETCH = "pencil_sketch"
    CARTOON = "cartoon"
    ANIME = "anime"
    VINTAGE = "vintage"
    MODERN_ART = "modern_art"
    RENAISSANCE = "renaissance"
    POP_ART = "pop_art"
    CUSTOM = "custom"


class ContentPreservationLevel(Enum):
    """Level of content preservation during style transfer."""
    LOW = "low"          # Heavy stylization, less content preservation
    MEDIUM = "medium"    # Balanced stylization and content preservation
    HIGH = "high"        # Light stylization, strong content preservation
    MAXIMUM = "maximum"  # Minimal stylization, maximum content preservation


@dataclass
class StyleTransferConfig:
    """Configuration for style transfer operations."""
    content_weight: float = 1.0
    style_weight: float = 1000.0
    total_variation_weight: float = 1.0
    num_iterations: int = 1000
    learning_rate: float = 0.01
    max_image_size: int = 512
    preserve_colors: bool = False
    blend_ratio: float = 1.0
    content_preservation_level: ContentPreservationLevel = ContentPreservationLevel.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "content_weight": self.content_weight,
            "style_weight": self.style_weight,
            "total_variation_weight": self.total_variation_weight,
            "num_iterations": self.num_iterations,
            "learning_rate": self.learning_rate,
            "max_image_size": self.max_image_size,
            "preserve_colors": self.preserve_colors,
            "blend_ratio": self.blend_ratio,
            "content_preservation_level": self.content_preservation_level.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StyleTransferConfig':
        """Create config from dictionary."""
        config = cls()
        for key, value in data.items():
            if key == "content_preservation_level":
                setattr(config, key, ContentPreservationLevel(value))
            elif hasattr(config, key):
                setattr(config, key, value)
        return config


@dataclass
class StyleTransferRequest:
    """Request for style transfer operation."""
    id: str
    content_paths: List[str]
    style_path: Optional[str] = None
    style_type: Optional[ArtisticStyle] = None
    style_types: Optional[List[ArtisticStyle]] = None
    config: Optional[StyleTransferConfig] = None
    batch_processing: bool = False
    multiple_styles: bool = False
    preserve_original_colors: bool = False
    output_format: str = "png"
    quality: int = 95
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate request after initialization."""
        if not self.content_paths:
            raise ValueError("At least one content path is required")
        
        if not self.style_path and not self.style_type and not self.style_types:
            raise ValueError("Either style_path, style_type, or style_types must be provided")
        
        if self.multiple_styles and not self.style_types:
            raise ValueError("style_types must be provided for multiple styles processing")
        
        if self.batch_processing and len(self.content_paths) < 2:
            raise ValueError("Batch processing requires at least 2 content paths")
        
        # Convert string enums to enum objects if needed
        if isinstance(self.style_type, str):
            self.style_type = ArtisticStyle(self.style_type)
        
        if self.style_types and isinstance(self.style_types[0], str):
            self.style_types = [ArtisticStyle(style) for style in self.style_types]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "id": self.id,
            "content_paths": self.content_paths,
            "style_path": self.style_path,
            "style_type": self.style_type.value if self.style_type else None,
            "style_types": [s.value for s in self.style_types] if self.style_types else None,
            "config": self.config.to_dict() if self.config else None,
            "batch_processing": self.batch_processing,
            "multiple_styles": self.multiple_styles,
            "preserve_original_colors": self.preserve_original_colors,
            "output_format": self.output_format,
            "quality": self.quality,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class StyleTransferResult:
    """Result of style transfer operation."""
    id: str
    status: StyleTransferStatus
    result_paths: List[str] = field(default_factory=list)
    result_urls: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    style_consistency_score: float = 0.0
    content_preservation_score: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Set completion time if status is completed."""
        if self.status == StyleTransferStatus.COMPLETED and not self.completed_at:
            self.completed_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "id": self.id,
            "status": self.status.value,
            "result_paths": self.result_paths,
            "result_urls": self.result_urls,
            "processing_time": self.processing_time,
            "style_consistency_score": self.style_consistency_score,
            "content_preservation_score": self.content_preservation_score,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class BatchProcessingRequest:
    """Request for batch style transfer processing."""
    content_paths: List[str]
    style_path: Optional[str] = None
    style_type: Optional[ArtisticStyle] = None
    config: Optional[StyleTransferConfig] = None
    parallel_processing: bool = True
    max_concurrent: int = 3
    
    def __post_init__(self):
        """Validate batch request."""
        if len(self.content_paths) < 2:
            raise ValueError("Batch processing requires at least 2 content paths")
        
        if not self.style_path and not self.style_type:
            raise ValueError("Either style_path or style_type must be provided")
        
        if isinstance(self.style_type, str):
            self.style_type = ArtisticStyle(self.style_type)


@dataclass
class StyleConsistencyMetrics:
    """Metrics for measuring style consistency across multiple images."""
    overall_consistency: float
    color_consistency: float
    texture_consistency: float
    pattern_consistency: float
    style_strength: float
    content_preservation: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "overall_consistency": self.overall_consistency,
            "color_consistency": self.color_consistency,
            "texture_consistency": self.texture_consistency,
            "pattern_consistency": self.pattern_consistency,
            "style_strength": self.style_strength,
            "content_preservation": self.content_preservation
        }


@dataclass
class StylePreset:
    """Predefined style configuration preset."""
    name: str
    style_type: ArtisticStyle
    config: StyleTransferConfig
    description: str
    example_image_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preset to dictionary."""
        return {
            "name": self.name,
            "style_type": self.style_type.value,
            "config": self.config.to_dict(),
            "description": self.description,
            "example_image_path": self.example_image_path,
            "tags": self.tags
        }


@dataclass
class StyleTransferJob:
    """Job tracking for style transfer operations."""
    id: str
    request: StyleTransferRequest
    status: StyleTransferStatus
    progress: float = 0.0
    current_step: str = ""
    estimated_completion_time: Optional[datetime] = None
    result: Optional[StyleTransferResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def update_progress(self, progress: float, step: str = ""):
        """Update job progress."""
        self.progress = min(100.0, max(0.0, progress))
        self.current_step = step
        
        if self.progress >= 100.0 and self.status == StyleTransferStatus.PROCESSING:
            self.status = StyleTransferStatus.COMPLETED
            self.completed_at = datetime.now()
    
    def start_processing(self):
        """Mark job as started."""
        self.status = StyleTransferStatus.PROCESSING
        self.started_at = datetime.now()
    
    def complete_with_result(self, result: StyleTransferResult):
        """Complete job with result."""
        self.result = result
        self.status = StyleTransferStatus.COMPLETED
        self.progress = 100.0
        self.completed_at = datetime.now()
    
    def fail_with_error(self, error_message: str):
        """Mark job as failed."""
        self.status = StyleTransferStatus.FAILED
        self.result = StyleTransferResult(
            id=f"failed_{self.id}",
            status=StyleTransferStatus.FAILED,
            error_message=error_message
        )
        self.completed_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "id": self.id,
            "request": self.request.to_dict(),
            "status": self.status.value,
            "progress": self.progress,
            "current_step": self.current_step,
            "estimated_completion_time": self.estimated_completion_time.isoformat() if self.estimated_completion_time else None,
            "result": self.result.to_dict() if self.result else None,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


# Database models (if using SQLAlchemy)
try:
    from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.sql import func
    
    Base = declarative_base()
    
    class StyleTransferJobDB(Base):
        """Database model for style transfer jobs."""
        __tablename__ = "style_transfer_jobs"
        
        id = Column(String, primary_key=True)
        request_data = Column(JSON, nullable=False)
        status = Column(String, nullable=False, default="pending")
        progress = Column(Float, default=0.0)
        current_step = Column(String, default="")
        result_data = Column(JSON)
        error_message = Column(Text)
        created_at = Column(DateTime, default=func.now())
        started_at = Column(DateTime)
        completed_at = Column(DateTime)
        processing_time = Column(Float)
        
        def to_job(self) -> StyleTransferJob:
            """Convert database record to job object."""
            request = StyleTransferRequest(**self.request_data)
            
            job = StyleTransferJob(
                id=self.id,
                request=request,
                status=StyleTransferStatus(self.status),
                progress=self.progress,
                current_step=self.current_step,
                created_at=self.created_at,
                started_at=self.started_at,
                completed_at=self.completed_at
            )
            
            if self.result_data:
                job.result = StyleTransferResult(**self.result_data)
            
            return job
    
    class StylePresetDB(Base):
        """Database model for style presets."""
        __tablename__ = "style_presets"
        
        id = Column(Integer, primary_key=True)
        name = Column(String, unique=True, nullable=False)
        style_type = Column(String, nullable=False)
        config_data = Column(JSON, nullable=False)
        description = Column(Text)
        example_image_path = Column(String)
        tags = Column(JSON)
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime, default=func.now())
        updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
        
        def to_preset(self) -> StylePreset:
            """Convert database record to preset object."""
            return StylePreset(
                name=self.name,
                style_type=ArtisticStyle(self.style_type),
                config=StyleTransferConfig.from_dict(self.config_data),
                description=self.description or "",
                example_image_path=self.example_image_path,
                tags=self.tags or []
            )

except ImportError:
    # SQLAlchemy not available
    Base = None
    StyleTransferJobDB = None
    StylePresetDB = None


# Utility functions
def create_default_presets() -> List[StylePreset]:
    """Create default style presets."""
    presets = [
        StylePreset(
            name="Classic Impressionist",
            style_type=ArtisticStyle.IMPRESSIONIST,
            config=StyleTransferConfig(
                style_weight=10000.0,
                content_weight=1.0,
                num_iterations=800,
                preserve_colors=False,
                content_preservation_level=ContentPreservationLevel.MEDIUM
            ),
            description="Classic impressionist style with soft brushstrokes and vibrant colors",
            tags=["classic", "artistic", "painterly"]
        ),
        StylePreset(
            name="Bold Cubist",
            style_type=ArtisticStyle.CUBIST,
            config=StyleTransferConfig(
                style_weight=50000.0,
                content_weight=1.0,
                num_iterations=1200,
                preserve_colors=False,
                content_preservation_level=ContentPreservationLevel.LOW
            ),
            description="Bold cubist style with geometric shapes and fragmented forms",
            tags=["geometric", "abstract", "bold"]
        ),
        StylePreset(
            name="Soft Watercolor",
            style_type=ArtisticStyle.WATERCOLOR,
            config=StyleTransferConfig(
                style_weight=5000.0,
                content_weight=1.5,
                num_iterations=600,
                preserve_colors=True,
                content_preservation_level=ContentPreservationLevel.HIGH
            ),
            description="Gentle watercolor effect with soft edges and flowing colors",
            tags=["soft", "gentle", "flowing"]
        ),
        StylePreset(
            name="Vibrant Pop Art",
            style_type=ArtisticStyle.POP_ART,
            config=StyleTransferConfig(
                style_weight=25000.0,
                content_weight=0.8,
                num_iterations=1200,
                preserve_colors=False,
                content_preservation_level=ContentPreservationLevel.LOW
            ),
            description="Bold pop art style with high contrast and vibrant colors",
            tags=["bold", "vibrant", "modern"]
        ),
        StylePreset(
            name="Anime Style",
            style_type=ArtisticStyle.ANIME,
            config=StyleTransferConfig(
                style_weight=20000.0,
                content_weight=1.0,
                num_iterations=1000,
                preserve_colors=False,
                content_preservation_level=ContentPreservationLevel.MEDIUM
            ),
            description="Japanese anime/manga style with clean lines and cel shading",
            tags=["anime", "manga", "clean"]
        ),
        StylePreset(
            name="Vintage Sepia",
            style_type=ArtisticStyle.VINTAGE,
            config=StyleTransferConfig(
                style_weight=6000.0,
                content_weight=1.2,
                num_iterations=700,
                preserve_colors=True,
                content_preservation_level=ContentPreservationLevel.HIGH
            ),
            description="Vintage sepia tone effect with aged appearance",
            tags=["vintage", "sepia", "aged"]
        )
    ]
    
    return presets


def validate_style_transfer_request(request_data: Dict[str, Any]) -> StyleTransferRequest:
    """Validate and create style transfer request from dictionary."""
    try:
        # Convert string enums to enum objects
        if "style_type" in request_data and isinstance(request_data["style_type"], str):
            request_data["style_type"] = ArtisticStyle(request_data["style_type"])
        
        if "style_types" in request_data and request_data["style_types"]:
            request_data["style_types"] = [
                ArtisticStyle(style) if isinstance(style, str) else style
                for style in request_data["style_types"]
            ]
        
        # Convert config dictionary to config object
        if "config" in request_data and isinstance(request_data["config"], dict):
            request_data["config"] = StyleTransferConfig.from_dict(request_data["config"])
        
        # Convert datetime strings
        if "created_at" in request_data and isinstance(request_data["created_at"], str):
            request_data["created_at"] = datetime.fromisoformat(request_data["created_at"])
        
        return StyleTransferRequest(**request_data)
        
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid style transfer request: {str(e)}")


def calculate_estimated_processing_time(request: StyleTransferRequest) -> float:
    """Calculate estimated processing time for a style transfer request."""
    base_time = 30.0  # Base time in seconds
    
    # Factor in number of images
    num_images = len(request.content_paths)
    if request.multiple_styles and request.style_types:
        num_images *= len(request.style_types)
    
    # Factor in configuration complexity
    config_multiplier = 1.0
    if request.config:
        # More iterations = more time
        iteration_factor = request.config.num_iterations / 1000.0
        config_multiplier = max(0.5, min(3.0, iteration_factor))
    
    # Factor in batch processing efficiency
    batch_efficiency = 0.8 if request.batch_processing and num_images > 1 else 1.0
    
    estimated_time = base_time * num_images * config_multiplier * batch_efficiency
    
    return estimated_time