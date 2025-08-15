"""
Data models for content export and format conversion functionality.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

from scrollintel.models.database import Base


class ExportStatus(str, Enum):
    """Export operation status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportJob(Base):
    """Database model for export jobs."""
    __tablename__ = "export_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(String(255), index=True)
    
    # Input details
    input_path = Column(String(500), nullable=False)
    content_type = Column(String(50), nullable=False)
    
    # Export settings
    target_format = Column(String(50), nullable=False)
    quality_level = Column(String(50), nullable=False)
    compression_type = Column(String(50), nullable=False)
    resolution = Column(String(50))  # "1920x1080"
    frame_rate = Column(Integer)
    bitrate = Column(String(50))
    preserve_metadata = Column(Boolean, default=True)
    watermark = Column(String(255))
    
    # Output details
    output_path = Column(String(500))
    output_filename = Column(String(255))
    file_size = Column(Integer)
    
    # Status and progress
    status = Column(String(50), default=ExportStatus.PENDING)
    progress = Column(Float, default=0.0)
    error_message = Column(Text)
    warnings = Column(JSON)
    
    # Quality metrics
    quality_metrics = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Processing details
    processing_time = Column(Float)
    worker_id = Column(String(255))


class ExportTemplate(Base):
    """Database model for export templates."""
    __tablename__ = "export_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    template_id = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Template settings
    target_format = Column(String(50), nullable=False)
    quality_level = Column(String(50), nullable=False)
    compression_type = Column(String(50), nullable=False)
    resolution = Column(String(50))
    frame_rate = Column(Integer)
    bitrate = Column(String(50))
    preserve_metadata = Column(Boolean, default=True)
    watermark = Column(String(255))
    
    # Custom parameters
    custom_params = Column(JSON)
    
    # Metadata
    created_by = Column(String(255))
    is_public = Column(Boolean, default=False)
    usage_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ContentMetadataRecord(Base):
    """Database model for content metadata."""
    __tablename__ = "content_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(String(255), unique=True, index=True, nullable=False)
    
    # Basic metadata
    title = Column(String(255))
    description = Column(Text)
    creator = Column(String(255))
    copyright = Column(String(255))
    
    # Generation metadata
    generation_prompt = Column(Text)
    model_used = Column(String(255))
    generation_params = Column(JSON)
    
    # Tags and categories
    tags = Column(JSON)  # List of strings
    categories = Column(JSON)  # List of strings
    
    # Custom fields
    custom_fields = Column(JSON)
    
    # Timestamps
    creation_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ExportStatistics(Base):
    """Database model for export statistics."""
    __tablename__ = "export_statistics"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, index=True, nullable=False)
    
    # Export counts by format
    jpeg_exports = Column(Integer, default=0)
    png_exports = Column(Integer, default=0)
    webp_exports = Column(Integer, default=0)
    mp4_exports = Column(Integer, default=0)
    webm_exports = Column(Integer, default=0)
    other_exports = Column(Integer, default=0)
    
    # Quality distribution
    low_quality_exports = Column(Integer, default=0)
    medium_quality_exports = Column(Integer, default=0)
    high_quality_exports = Column(Integer, default=0)
    lossless_exports = Column(Integer, default=0)
    
    # Performance metrics
    total_processing_time = Column(Float, default=0.0)
    average_processing_time = Column(Float, default=0.0)
    total_file_size = Column(Integer, default=0)
    
    # Error statistics
    failed_exports = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)


# Pydantic models for API responses

class ExportJobResponse(BaseModel):
    """Response model for export job."""
    id: int
    job_id: str
    user_id: Optional[str]
    input_path: str
    content_type: str
    target_format: str
    quality_level: str
    compression_type: str
    resolution: Optional[str]
    frame_rate: Optional[int]
    bitrate: Optional[str]
    preserve_metadata: bool
    watermark: Optional[str]
    output_path: Optional[str]
    output_filename: Optional[str]
    file_size: Optional[int]
    status: str
    progress: float
    error_message: Optional[str]
    warnings: Optional[List[str]]
    quality_metrics: Optional[Dict[str, Any]]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    processing_time: Optional[float]
    worker_id: Optional[str]
    
    class Config:
        from_attributes = True


class ExportTemplateResponse(BaseModel):
    """Response model for export template."""
    id: int
    template_id: str
    name: str
    description: Optional[str]
    target_format: str
    quality_level: str
    compression_type: str
    resolution: Optional[str]
    frame_rate: Optional[int]
    bitrate: Optional[str]
    preserve_metadata: bool
    watermark: Optional[str]
    custom_params: Optional[Dict[str, Any]]
    created_by: Optional[str]
    is_public: bool
    usage_count: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ContentMetadataResponse(BaseModel):
    """Response model for content metadata."""
    id: int
    content_id: str
    title: Optional[str]
    description: Optional[str]
    creator: Optional[str]
    copyright: Optional[str]
    generation_prompt: Optional[str]
    model_used: Optional[str]
    generation_params: Optional[Dict[str, Any]]
    tags: Optional[List[str]]
    categories: Optional[List[str]]
    custom_fields: Optional[Dict[str, Any]]
    creation_date: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ExportStatisticsResponse(BaseModel):
    """Response model for export statistics."""
    id: int
    date: datetime
    jpeg_exports: int
    png_exports: int
    webp_exports: int
    mp4_exports: int
    webm_exports: int
    other_exports: int
    low_quality_exports: int
    medium_quality_exports: int
    high_quality_exports: int
    lossless_exports: int
    total_processing_time: float
    average_processing_time: float
    total_file_size: int
    failed_exports: int
    success_rate: float
    
    class Config:
        from_attributes = True


# Request models

class CreateExportJobRequest(BaseModel):
    """Request model for creating export job."""
    input_path: str = Field(..., description="Path to input content")
    target_format: str = Field(..., description="Target export format")
    quality_level: str = Field("high", description="Quality level")
    compression_type: str = Field("balanced", description="Compression type")
    resolution: Optional[str] = Field(None, description="Target resolution")
    frame_rate: Optional[int] = Field(None, description="Target frame rate")
    bitrate: Optional[str] = Field(None, description="Target bitrate")
    preserve_metadata: bool = Field(True, description="Preserve metadata")
    watermark: Optional[str] = Field(None, description="Watermark text")
    output_filename: Optional[str] = Field(None, description="Custom output filename")


class CreateExportTemplateRequest(BaseModel):
    """Request model for creating export template."""
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    target_format: str = Field(..., description="Target export format")
    quality_level: str = Field("high", description="Quality level")
    compression_type: str = Field("balanced", description="Compression type")
    resolution: Optional[str] = Field(None, description="Target resolution")
    frame_rate: Optional[int] = Field(None, description="Target frame rate")
    bitrate: Optional[str] = Field(None, description="Target bitrate")
    preserve_metadata: bool = Field(True, description="Preserve metadata")
    watermark: Optional[str] = Field(None, description="Watermark text")
    custom_params: Optional[Dict[str, Any]] = Field(None, description="Custom parameters")
    is_public: bool = Field(False, description="Make template public")


class UpdateContentMetadataRequest(BaseModel):
    """Request model for updating content metadata."""
    title: Optional[str] = None
    description: Optional[str] = None
    creator: Optional[str] = None
    copyright: Optional[str] = None
    generation_prompt: Optional[str] = None
    model_used: Optional[str] = None
    generation_params: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class ExportQueueStatus(BaseModel):
    """Model for export queue status."""
    total_jobs: int
    pending_jobs: int
    processing_jobs: int
    completed_jobs: int
    failed_jobs: int
    average_wait_time: float
    estimated_completion_time: Optional[datetime]


class FormatCapabilities(BaseModel):
    """Model for format capabilities."""
    format: str
    supports_transparency: bool
    supports_animation: bool
    supports_audio: bool
    supports_subtitles: bool
    max_resolution: Optional[str]
    typical_use_cases: List[str]
    compression_types: List[str]
    quality_levels: List[str]


class ExportPreview(BaseModel):
    """Model for export preview."""
    estimated_file_size: int
    estimated_processing_time: float
    quality_prediction: float
    compression_ratio: float
    format_recommendations: List[str]
    warnings: List[str]