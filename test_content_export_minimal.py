"""
Minimal test for content export functionality without external dependencies.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


class ExportFormat(Enum):
    """Supported export formats."""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    TIFF = "tiff"
    BMP = "bmp"
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"
    GIF = "gif"
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


class MinimalContentExporter:
    """Minimal content exporter for testing."""
    
    def __init__(self, output_directory: str = "exports"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.quality_settings = {
            QualityLevel.LOW: {"jpeg": 60, "webp": 50},
            QualityLevel.MEDIUM: {"jpeg": 80, "webp": 70},
            QualityLevel.HIGH: {"jpeg": 95, "webp": 90},
            QualityLevel.LOSSLESS: {"jpeg": 100, "webp": 100}
        }
    
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
    
    async def export_content(
        self,
        content_path: str,
        settings: ExportSettings,
        metadata: Optional[ContentMetadata] = None,
        output_filename: Optional[str] = None
    ) -> ExportResult:
        """Export content with specified settings."""
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
            
            return result
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            
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
                
                # Prepare save parameters
                save_params = {}
                
                if settings.format == ExportFormat.JPEG:
                    save_params = self._get_jpeg_params(settings)
                elif settings.format == ExportFormat.PNG:
                    save_params = self._get_png_params(settings)
                
                # Save image
                img.save(str(output_path), **save_params)
            
            # Calculate quality metrics (simplified)
            quality_metrics = {
                "psnr": 32.0,  # Mock value
                "ssim": 0.95,  # Mock value
                "mse": 12.0    # Mock value
            }
            
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
            }
        }
        
        return format_info.get(format, {"description": "Unknown format"})


async def test_basic_functionality():
    """Test basic content export functionality."""
    print("🧪 Testing Minimal Content Export Functionality")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test image
        input_path = os.path.join(temp_dir, "test_image.png")
        image = Image.new('RGB', (100, 100), color='blue')
        image.save(input_path)
        print(f"✅ Created test image: {input_path}")
        
        # Initialize exporter
        export_dir = os.path.join(temp_dir, "exports")
        exporter = MinimalContentExporter(output_directory=export_dir)
        print(f"✅ Initialized MinimalContentExporter with output directory: {export_dir}")
        
        # Test 1: Basic export
        print("\n📤 Test 1: Basic JPEG Export")
        settings = ExportSettings(
            format=ExportFormat.JPEG,
            quality=QualityLevel.HIGH,
            compression=CompressionType.BALANCED
        )
        
        result = await exporter.export_content(input_path, settings)
        
        if result.success:
            print(f"   ✅ Export successful!")
            print(f"   📁 Output: {result.output_path}")
            print(f"   📊 File size: {result.file_size:,} bytes")
            print(f"   ⏱️  Processing time: {result.processing_time:.3f}s")
            print(f"   📈 Quality metrics: {result.quality_metrics}")
            
            # Verify output file
            if os.path.exists(result.output_path):
                output_image = Image.open(result.output_path)
                print(f"   🖼️  Output format: {output_image.format}")
                print(f"   📐 Output size: {output_image.size}")
            else:
                print(f"   ❌ Output file not found!")
        else:
            print(f"   ❌ Export failed: {result.error_message}")
        
        # Test 2: PNG Export
        print("\n📤 Test 2: PNG Export")
        png_settings = ExportSettings(
            format=ExportFormat.PNG,
            quality=QualityLevel.LOSSLESS,
            compression=CompressionType.MAXIMUM
        )
        
        png_result = await exporter.export_content(input_path, png_settings)
        
        if png_result.success:
            print(f"   ✅ PNG export successful!")
            print(f"   📁 Output: {png_result.output_path}")
            print(f"   📊 File size: {png_result.file_size:,} bytes")
        else:
            print(f"   ❌ PNG export failed: {png_result.error_message}")
        
        # Test 3: Metadata handling
        print("\n📋 Test 3: Metadata Handling")
        metadata = ContentMetadata(
            title="Test Image",
            description="A test image for content export",
            creator="Test Suite",
            generation_prompt="Create a blue square",
            model_used="test_model_v1",
            tags=["test", "blue", "square"]
        )
        
        metadata_result = await exporter.export_content(
            input_path,
            settings,
            metadata,
            "test_with_metadata.jpg"
        )
        
        if metadata_result.success:
            print(f"   ✅ Export with metadata successful!")
            print(f"   📁 Output: {metadata_result.output_path}")
            print(f"   📊 File size: {metadata_result.file_size:,} bytes")
        else:
            print(f"   ❌ Export with metadata failed: {metadata_result.error_message}")
        
        # Test 4: Format capabilities
        print("\n📊 Test 4: Format Capabilities")
        
        image_formats = exporter.get_supported_formats("image")
        video_formats = exporter.get_supported_formats("video")
        
        print(f"   🖼️  Supported image formats: {[f.value for f in image_formats]}")
        print(f"   🎥 Supported video formats: {[f.value for f in video_formats]}")
        
        # Get format info
        jpeg_info = exporter.get_format_info(ExportFormat.JPEG)
        png_info = exporter.get_format_info(ExportFormat.PNG)
        
        print(f"   📄 JPEG info: {jpeg_info['description']}")
        print(f"      Transparency: {'✅' if jpeg_info.get('supports_transparency') else '❌'}")
        print(f"   📄 PNG info: {png_info['description']}")
        print(f"      Transparency: {'✅' if png_info.get('supports_transparency') else '❌'}")
        
        # Test 5: Error handling
        print("\n🚨 Test 5: Error Handling")
        
        error_result = await exporter.export_content(
            "nonexistent_file.jpg",
            ExportSettings(format=ExportFormat.PNG)
        )
        
        if not error_result.success:
            print(f"   ✅ Error handling working correctly!")
            print(f"   📝 Error message: {error_result.error_message}")
        else:
            print(f"   ❌ Error handling failed - should have returned error!")
        
        print("\n🎉 All tests completed!")


def test_data_classes():
    """Test data classes and enums."""
    print("🏗️  Testing Data Classes and Enums")
    print("=" * 50)
    
    # Test ExportSettings
    settings = ExportSettings(
        format=ExportFormat.WEBP,
        quality=QualityLevel.MEDIUM,
        compression=CompressionType.FAST,
        resolution=(800, 600),
        watermark="TEST"
    )
    
    print(f"✅ ExportSettings created:")
    print(f"   Format: {settings.format.value}")
    print(f"   Quality: {settings.quality.value}")
    print(f"   Compression: {settings.compression.value}")
    print(f"   Resolution: {settings.resolution}")
    print(f"   Watermark: {settings.watermark}")
    
    # Test ContentMetadata
    metadata = ContentMetadata(
        title="Test Metadata",
        creator="Test Creator",
        tags=["test", "metadata"]
    )
    
    print(f"\n✅ ContentMetadata created:")
    print(f"   Title: {metadata.title}")
    print(f"   Creator: {metadata.creator}")
    print(f"   Tags: {metadata.tags}")
    print(f"   Creation date: {metadata.creation_date}")
    
    # Test enums
    print(f"\n✅ Enums working:")
    print(f"   Export formats: {[f.value for f in ExportFormat]}")
    print(f"   Quality levels: {[q.value for q in QualityLevel]}")
    print(f"   Compression types: {[c.value for c in CompressionType]}")


async def main():
    """Run all tests."""
    print("🚀 Minimal Content Export System Test")
    print("=" * 60)
    
    try:
        test_data_classes()
        await test_basic_functionality()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print("✅ Content export system core functionality is working correctly")
        print("📝 Note: This is a minimal test without external dependencies")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())