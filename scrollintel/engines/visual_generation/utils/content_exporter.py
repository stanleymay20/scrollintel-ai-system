"""
Content Exporter for Visual Generation System

This module provides comprehensive export and format conversion capabilities
for generated visual content, supporting multiple formats, quality settings,
compression options, and metadata preservation.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime

# Image processing imports
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS
import cv2
import numpy as np

# Video processing imports
import ffmpeg

# Metadata handling
from mutagen.mp4 import MP4
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""
    # Image formats
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    TIFF = "tiff"
    BMP = "bmp"
    
    # Video formats
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"
    GIF = "gif"
    
    # Archive formats
    ZIP = "zip"
    TAR = "tar"


class QualityLevel(Enum):
    """Quality levels for export."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    LOSSLESS = "lossless"


class CompressionType(Enum):
    """Compression types."""
    NONE = "none"
    FAST = "fast"
    BALANCED = "balanced"
    MAXIMUM = "maximum"


@dataclass
class ExportSettings:
    """Export configuration settings."""
    format: ExportFormat
    quality: QualityLevel = QualityLevel.HIGH
    compression: CompressionType = CompressionType.BALANCED
    resolution: Optional[Tuple[int, int]] = None
    frame_rate: Optional[int] = None
    bitrate: Optional[str] = None
    preserve_metadata: bool = True
    watermark: Optional[str] = None
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class ContentMetadata:
    """Metadata for generated content."""
    title: Optional[str] = None
    description: Optional[str] = None
    creator: Optional[str] = None
    creation_date: Optional[datetime] = None
    generation_prompt: Optional[str] = None
    model_used: Optional[str] = None
    generation_params: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    copyright: Optional[str] = None
    custom_fields: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.creation_date is None:
            self.creation_date = datetime.now()
        if self.tags is None:
            self.tags = []
        if self.custom_fields is None:
            self.custom_fields = {}


@dataclass
class ExportResult:
    """Result of export operation."""
    success: bool
    output_path: str
    file_size: int
    format: ExportFormat
    quality_metrics: Dict[str, float]
    processing_time: float
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class ContentExporter:
    """
    Advanced content exporter with multiple format support,
    quality settings, compression options, and metadata preservation.
    """
    
    def __init__(self, output_directory: str = "exports"):
        """
        Initialize the ContentExporter.
        
        Args:
            output_directory: Base directory for exported files
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Quality settings for different formats
        self.quality_settings = {
            QualityLevel.LOW: {"jpeg": 60, "webp": 50, "video_crf": 28},
            QualityLevel.MEDIUM: {"jpeg": 80, "webp": 70, "video_crf": 23},
            QualityLevel.HIGH: {"jpeg": 95, "webp": 90, "video_crf": 18},
            QualityLevel.LOSSLESS: {"jpeg": 100, "webp": 100, "video_crf": 0}
        }
        
        # Compression presets
        self.compression_presets = {
            CompressionType.NONE: {"preset": "ultrafast", "crf": 0},
            CompressionType.FAST: {"preset": "fast", "crf": 18},
            CompressionType.BALANCED: {"preset": "medium", "crf": 23},
            CompressionType.MAXIMUM: {"preset": "slow", "crf": 28}
        }
    
    async def export_content(
        self,
        content_path: str,
        settings: ExportSettings,
        metadata: Optional[ContentMetadata] = None,
        output_filename: Optional[str] = None
    ) -> ExportResult:
        """
        Export content with specified settings.
        
        Args:
            content_path: Path to the content file to export
            settings: Export settings
            metadata: Content metadata
            output_filename: Custom output filename
            
        Returns:
            ExportResult with export details
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate input file
            if not os.path.exists(content_path):
                return ExportResult(
                    success=False,
                    output_path="",
                    file_size=0,
                    format=settings.format,
                    quality_metrics={},
                    processing_time=0,
                    error_message=f"Input file not found: {content_path}"
                )
            
            # Determine content type
            content_type = self._detect_content_type(content_path)
            
            # Generate output filename
            if output_filename is None:
                base_name = Path(content_path).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{base_name}_{timestamp}.{settings.format.value}"
            
            output_path = self.output_directory / output_filename
            
            # Export based on content type
            if content_type == "image":
                result = await self._export_image(
                    content_path, output_path, settings, metadata
                )
            elif content_type == "video":
                result = await self._export_video(
                    content_path, output_path, settings, metadata
                )
            else:
                return ExportResult(
                    success=False,
                    output_path="",
                    file_size=0,
                    format=settings.format,
                    quality_metrics={},
                    processing_time=0,
                    error_message=f"Unsupported content type: {content_type}"
                )
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            
            logger.info(f"Export completed: {output_path} ({processing_time:.2f}s)")
            return result
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Export failed: {str(e)}")
            
            return ExportResult(
                success=False,
                output_path="",
                file_size=0,
                format=settings.format,
                quality_metrics={},
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _export_image(
        self,
        input_path: str,
        output_path: Path,
        settings: ExportSettings,
        metadata: Optional[ContentMetadata]
    ) -> ExportResult:
        """Export image with specified settings."""
        try:
            # Load image
            with Image.open(input_path) as img:
                # Apply resolution changes if specified
                if settings.resolution:
                    img = img.resize(settings.resolution, Image.Resampling.LANCZOS)
                
                # Apply watermark if specified
                if settings.watermark:
                    img = await self._apply_watermark(img, settings.watermark)
                
                # Prepare save parameters
                save_params = {}
                
                if settings.format == ExportFormat.JPEG:
                    save_params = self._get_jpeg_params(settings)
                elif settings.format == ExportFormat.PNG:
                    save_params = self._get_png_params(settings)
                elif settings.format == ExportFormat.WEBP:
                    save_params = self._get_webp_params(settings)
                elif settings.format == ExportFormat.TIFF:
                    save_params = self._get_tiff_params(settings)
                
                # Add metadata if preserving
                if settings.preserve_metadata and metadata:
                    save_params.update(self._prepare_image_metadata(metadata))
                
                # Save image
                img.save(str(output_path), **save_params)
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_image_quality_metrics(
                input_path, str(output_path)
            )
            
            # Get file size
            file_size = output_path.stat().st_size
            
            return ExportResult(
                success=True,
                output_path=str(output_path),
                file_size=file_size,
                format=settings.format,
                quality_metrics=quality_metrics,
                processing_time=0  # Will be set by caller
            )
            
        except Exception as e:
            raise Exception(f"Image export failed: {str(e)}")
    
    async def _export_video(
        self,
        input_path: str,
        output_path: Path,
        settings: ExportSettings,
        metadata: Optional[ContentMetadata]
    ) -> ExportResult:
        """Export video with specified settings."""
        try:
            # Build ffmpeg command
            input_stream = ffmpeg.input(input_path)
            
            # Apply video filters
            video_filters = []
            
            # Resolution scaling
            if settings.resolution:
                video_filters.append(f"scale={settings.resolution[0]}:{settings.resolution[1]}")
            
            # Apply watermark
            if settings.watermark:
                video_filters.append(f"drawtext=text='{settings.watermark}':x=10:y=10:fontsize=24:fontcolor=white")
            
            # Apply filters
            if video_filters:
                input_stream = input_stream.filter('vf', ','.join(video_filters))
            
            # Get encoding parameters
            encoding_params = self._get_video_encoding_params(settings)
            
            # Build output stream
            output_stream = input_stream.output(str(output_path), **encoding_params)
            
            # Run ffmpeg
            await asyncio.create_subprocess_exec(
                *ffmpeg.compile(output_stream, overwrite_output=True),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Add metadata if preserving
            if settings.preserve_metadata and metadata:
                await self._add_video_metadata(str(output_path), metadata)
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_video_quality_metrics(
                input_path, str(output_path)
            )
            
            # Get file size
            file_size = output_path.stat().st_size
            
            return ExportResult(
                success=True,
                output_path=str(output_path),
                file_size=file_size,
                format=settings.format,
                quality_metrics=quality_metrics,
                processing_time=0  # Will be set by caller
            )
            
        except Exception as e:
            raise Exception(f"Video export failed: {str(e)}")
    
    def _detect_content_type(self, file_path: str) -> str:
        """Detect content type from file extension."""
        extension = Path(file_path).suffix.lower()
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp', '.gif'}
        video_extensions = {'.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv'}
        
        if extension in image_extensions:
            return "image"
        elif extension in video_extensions:
            return "video"
        else:
            return "unknown"
    
    def _get_jpeg_params(self, settings: ExportSettings) -> Dict[str, Any]:
        """Get JPEG export parameters."""
        quality = self.quality_settings[settings.quality]["jpeg"]
        
        params = {
            "format": "JPEG",
            "quality": quality,
            "optimize": True
        }
        
        if settings.compression == CompressionType.MAXIMUM:
            params["progressive"] = True
        
        return params
    
    def _get_png_params(self, settings: ExportSettings) -> Dict[str, Any]:
        """Get PNG export parameters."""
        params = {"format": "PNG"}
        
        if settings.compression == CompressionType.MAXIMUM:
            params["compress_level"] = 9
        elif settings.compression == CompressionType.BALANCED:
            params["compress_level"] = 6
        else:
            params["compress_level"] = 1
        
        return params
    
    def _get_webp_params(self, settings: ExportSettings) -> Dict[str, Any]:
        """Get WebP export parameters."""
        quality = self.quality_settings[settings.quality]["webp"]
        
        params = {
            "format": "WEBP",
            "quality": quality,
            "method": 6 if settings.compression == CompressionType.MAXIMUM else 4
        }
        
        if settings.quality == QualityLevel.LOSSLESS:
            params["lossless"] = True
        
        return params
    
    def _get_tiff_params(self, settings: ExportSettings) -> Dict[str, Any]:
        """Get TIFF export parameters."""
        params = {"format": "TIFF"}
        
        if settings.compression != CompressionType.NONE:
            params["compression"] = "lzw"
        
        return params
    
    def _get_video_encoding_params(self, settings: ExportSettings) -> Dict[str, Any]:
        """Get video encoding parameters."""
        preset = self.compression_presets[settings.compression]
        crf = self.quality_settings[settings.quality]["video_crf"]
        
        params = {
            "vcodec": "libx264",
            "preset": preset["preset"],
            "crf": crf
        }
        
        if settings.frame_rate:
            params["r"] = settings.frame_rate
        
        if settings.bitrate:
            params["b:v"] = settings.bitrate
        
        # Format-specific settings
        if settings.format == ExportFormat.WEBM:
            params["vcodec"] = "libvpx-vp9"
            params["acodec"] = "libopus"
        elif settings.format == ExportFormat.AVI:
            params["vcodec"] = "libxvid"
        
        return params
    
    def _prepare_image_metadata(self, metadata: ContentMetadata) -> Dict[str, Any]:
        """Prepare image metadata for embedding."""
        exif_dict = {}
        
        if metadata.title:
            exif_dict["0th"] = {270: metadata.title}  # ImageDescription
        
        if metadata.creator:
            exif_dict["0th"] = exif_dict.get("0th", {})
            exif_dict["0th"][315] = metadata.creator  # Artist
        
        return {"exif": exif_dict} if exif_dict else {}
    
    async def _add_video_metadata(self, video_path: str, metadata: ContentMetadata):
        """Add metadata to video file."""
        try:
            video_file = MP4(video_path)
            
            if metadata.title:
                video_file["\xa9nam"] = [metadata.title]
            
            if metadata.creator:
                video_file["\xa9ART"] = [metadata.creator]
            
            if metadata.description:
                video_file["\xa9cmt"] = [metadata.description]
            
            if metadata.creation_date:
                video_file["\xa9day"] = [metadata.creation_date.strftime("%Y-%m-%d")]
            
            video_file.save()
            
        except Exception as e:
            logger.warning(f"Failed to add video metadata: {str(e)}")
    
    async def _apply_watermark(self, image: Image.Image, watermark_text: str) -> Image.Image:
        """Apply watermark to image."""
        from PIL import ImageDraw, ImageFont
        
        # Create a copy to avoid modifying original
        watermarked = image.copy()
        draw = ImageDraw.Draw(watermarked)
        
        # Try to use a nice font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            font = ImageFont.load_default()
        
        # Calculate position (bottom right)
        text_bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = image.width - text_width - 20
        y = image.height - text_height - 20
        
        # Draw text with shadow
        draw.text((x+2, y+2), watermark_text, font=font, fill=(0, 0, 0, 128))
        draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 200))
        
        return watermarked
    
    async def _calculate_image_quality_metrics(
        self, 
        original_path: str, 
        exported_path: str
    ) -> Dict[str, float]:
        """Calculate quality metrics for exported image."""
        try:
            # Load images
            original = cv2.imread(original_path)
            exported = cv2.imread(exported_path)
            
            if original is None or exported is None:
                return {"error": 1.0}
            
            # Resize exported to match original if needed
            if original.shape != exported.shape:
                exported = cv2.resize(exported, (original.shape[1], original.shape[0]))
            
            # Calculate PSNR
            mse = np.mean((original - exported) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            
            # Calculate SSIM
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(
                cv2.cvtColor(original, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(exported, cv2.COLOR_BGR2GRAY)
            )
            
            return {
                "psnr": float(psnr),
                "ssim": float(ssim_score),
                "mse": float(mse)
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate image quality metrics: {str(e)}")
            return {"error": 1.0}
    
    async def _calculate_video_quality_metrics(
        self, 
        original_path: str, 
        exported_path: str
    ) -> Dict[str, float]:
        """Calculate quality metrics for exported video."""
        try:
            # Get video properties
            original_probe = ffmpeg.probe(original_path)
            exported_probe = ffmpeg.probe(exported_path)
            
            original_stream = next(s for s in original_probe['streams'] if s['codec_type'] == 'video')
            exported_stream = next(s for s in exported_probe['streams'] if s['codec_type'] == 'video')
            
            # Calculate compression ratio
            original_size = os.path.getsize(original_path)
            exported_size = os.path.getsize(exported_path)
            compression_ratio = original_size / exported_size if exported_size > 0 else 0
            
            # Calculate bitrate ratio
            original_bitrate = int(original_stream.get('bit_rate', 0))
            exported_bitrate = int(exported_stream.get('bit_rate', 0))
            bitrate_ratio = original_bitrate / exported_bitrate if exported_bitrate > 0 else 0
            
            return {
                "compression_ratio": float(compression_ratio),
                "bitrate_ratio": float(bitrate_ratio),
                "original_size": float(original_size),
                "exported_size": float(exported_size)
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate video quality metrics: {str(e)}")
            return {"error": 1.0}
    
    async def batch_export(
        self,
        content_paths: List[str],
        settings: ExportSettings,
        metadata_list: Optional[List[ContentMetadata]] = None,
        max_concurrent: int = 3
    ) -> List[ExportResult]:
        """
        Export multiple content files concurrently.
        
        Args:
            content_paths: List of content file paths
            settings: Export settings to apply to all files
            metadata_list: Optional list of metadata for each file
            max_concurrent: Maximum concurrent exports
            
        Returns:
            List of ExportResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def export_single(i: int, path: str) -> ExportResult:
            async with semaphore:
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
                return await self.export_content(path, settings, metadata)
        
        tasks = [export_single(i, path) for i, path in enumerate(content_paths)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ExportResult(
                    success=False,
                    output_path="",
                    file_size=0,
                    format=settings.format,
                    quality_metrics={},
                    processing_time=0,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def create_archive(
        self,
        content_paths: List[str],
        archive_format: ExportFormat = ExportFormat.ZIP,
        archive_name: Optional[str] = None
    ) -> ExportResult:
        """
        Create an archive of multiple content files.
        
        Args:
            content_paths: List of content file paths to archive
            archive_format: Archive format (ZIP or TAR)
            archive_name: Custom archive name
            
        Returns:
            ExportResult for the archive
        """
        import zipfile
        import tarfile
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            if archive_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_name = f"content_archive_{timestamp}.{archive_format.value}"
            
            archive_path = self.output_directory / archive_name
            
            if archive_format == ExportFormat.ZIP:
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for content_path in content_paths:
                        if os.path.exists(content_path):
                            arcname = os.path.basename(content_path)
                            zipf.write(content_path, arcname)
            
            elif archive_format == ExportFormat.TAR:
                with tarfile.open(archive_path, 'w:gz') as tarf:
                    for content_path in content_paths:
                        if os.path.exists(content_path):
                            arcname = os.path.basename(content_path)
                            tarf.add(content_path, arcname)
            
            else:
                raise ValueError(f"Unsupported archive format: {archive_format}")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            file_size = archive_path.stat().st_size
            
            return ExportResult(
                success=True,
                output_path=str(archive_path),
                file_size=file_size,
                format=archive_format,
                quality_metrics={"files_archived": len(content_paths)},
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            return ExportResult(
                success=False,
                output_path="",
                file_size=0,
                format=archive_format,
                quality_metrics={},
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def get_supported_formats(self, content_type: str) -> List[ExportFormat]:
        """Get supported export formats for content type."""
        if content_type == "image":
            return [
                ExportFormat.JPEG,
                ExportFormat.PNG,
                ExportFormat.WEBP,
                ExportFormat.TIFF,
                ExportFormat.BMP
            ]
        elif content_type == "video":
            return [
                ExportFormat.MP4,
                ExportFormat.AVI,
                ExportFormat.MOV,
                ExportFormat.WEBM,
                ExportFormat.GIF
            ]
        else:
            return []
    
    def get_format_info(self, format: ExportFormat) -> Dict[str, Any]:
        """Get information about a specific format."""
        format_info = {
            ExportFormat.JPEG: {
                "description": "JPEG image format with lossy compression",
                "supports_transparency": False,
                "supports_animation": False,
                "typical_use": "Photos and complex images"
            },
            ExportFormat.PNG: {
                "description": "PNG image format with lossless compression",
                "supports_transparency": True,
                "supports_animation": False,
                "typical_use": "Graphics with transparency"
            },
            ExportFormat.WEBP: {
                "description": "Modern web image format",
                "supports_transparency": True,
                "supports_animation": True,
                "typical_use": "Web images with small file sizes"
            },
            ExportFormat.MP4: {
                "description": "MP4 video format",
                "supports_audio": True,
                "supports_subtitles": True,
                "typical_use": "General video content"
            },
            ExportFormat.WEBM: {
                "description": "WebM video format for web",
                "supports_audio": True,
                "supports_subtitles": True,
                "typical_use": "Web video content"
            }
        }
        
        return format_info.get(format, {"description": "Unknown format"})


class FormatConverter:
    """Utility class for format conversion operations."""
    
    def __init__(self, exporter: ContentExporter):
        self.exporter = exporter
    
    async def convert_format(
        self,
        input_path: str,
        target_format: ExportFormat,
        quality: QualityLevel = QualityLevel.HIGH,
        preserve_metadata: bool = True
    ) -> ExportResult:
        """
        Convert content to a different format.
        
        Args:
            input_path: Path to input file
            target_format: Target export format
            quality: Quality level for conversion
            preserve_metadata: Whether to preserve metadata
            
        Returns:
            ExportResult with conversion details
        """
        settings = ExportSettings(
            format=target_format,
            quality=quality,
            preserve_metadata=preserve_metadata
        )
        
        return await self.exporter.export_content(input_path, settings)
    
    async def batch_convert(
        self,
        input_paths: List[str],
        target_format: ExportFormat,
        quality: QualityLevel = QualityLevel.HIGH
    ) -> List[ExportResult]:
        """Convert multiple files to the same format."""
        settings = ExportSettings(format=target_format, quality=quality)
        return await self.exporter.batch_export(input_paths, settings)
    
    async def optimize_for_web(self, input_path: str) -> ExportResult:
        """Optimize content for web delivery."""
        content_type = self.exporter._detect_content_type(input_path)
        
        if content_type == "image":
            settings = ExportSettings(
                format=ExportFormat.WEBP,
                quality=QualityLevel.HIGH,
                compression=CompressionType.BALANCED,
                resolution=(1920, 1080)  # Max web resolution
            )
        elif content_type == "video":
            settings = ExportSettings(
                format=ExportFormat.WEBM,
                quality=QualityLevel.MEDIUM,
                compression=CompressionType.BALANCED,
                resolution=(1920, 1080),
                frame_rate=30
            )
        else:
            raise ValueError(f"Unsupported content type for web optimization: {content_type}")
        
        return await self.exporter.export_content(input_path, settings)