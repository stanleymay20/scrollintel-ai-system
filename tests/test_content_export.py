"""
Tests for content export and format conversion functionality.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import numpy as np

from scrollintel.engines.visual_generation.utils.content_exporter import (
    ContentExporter,
    FormatConverter,
    ExportSettings,
    ContentMetadata,
    ExportFormat,
    QualityLevel,
    CompressionType,
    ExportResult
)


class TestContentExporter:
    """Test cases for ContentExporter class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a sample image for testing."""
        image_path = os.path.join(temp_dir, "test_image.png")
        
        # Create a simple test image
        image = Image.new('RGB', (100, 100), color='red')
        image.save(image_path)
        
        return image_path
    
    @pytest.fixture
    def content_exporter(self, temp_dir):
        """Create ContentExporter instance for testing."""
        return ContentExporter(output_directory=temp_dir)
    
    @pytest.fixture
    def export_settings(self):
        """Create sample export settings."""
        return ExportSettings(
            format=ExportFormat.JPEG,
            quality=QualityLevel.HIGH,
            compression=CompressionType.BALANCED
        )
    
    @pytest.fixture
    def content_metadata(self):
        """Create sample content metadata."""
        return ContentMetadata(
            title="Test Image",
            description="A test image for unit testing",
            creator="Test Suite",
            generation_prompt="Create a red square",
            model_used="test_model",
            tags=["test", "red", "square"]
        )
    
    def test_content_exporter_initialization(self, temp_dir):
        """Test ContentExporter initialization."""
        exporter = ContentExporter(output_directory=temp_dir)
        
        assert exporter.output_directory == Path(temp_dir)
        assert exporter.output_directory.exists()
        assert len(exporter.quality_settings) > 0
        assert len(exporter.compression_presets) > 0
    
    def test_detect_content_type(self, content_exporter):
        """Test content type detection."""
        assert content_exporter._detect_content_type("test.jpg") == "image"
        assert content_exporter._detect_content_type("test.png") == "image"
        assert content_exporter._detect_content_type("test.mp4") == "video"
        assert content_exporter._detect_content_type("test.avi") == "video"
        assert content_exporter._detect_content_type("test.txt") == "unknown"
    
    def test_get_jpeg_params(self, content_exporter, export_settings):
        """Test JPEG parameter generation."""
        params = content_exporter._get_jpeg_params(export_settings)
        
        assert params["format"] == "JPEG"
        assert "quality" in params
        assert params["optimize"] is True
    
    def test_get_png_params(self, content_exporter):
        """Test PNG parameter generation."""
        settings = ExportSettings(
            format=ExportFormat.PNG,
            compression=CompressionType.MAXIMUM
        )
        params = content_exporter._get_png_params(settings)
        
        assert params["format"] == "PNG"
        assert params["compress_level"] == 9
    
    def test_get_webp_params(self, content_exporter):
        """Test WebP parameter generation."""
        settings = ExportSettings(
            format=ExportFormat.WEBP,
            quality=QualityLevel.LOSSLESS
        )
        params = content_exporter._get_webp_params(settings)
        
        assert params["format"] == "WEBP"
        assert params["lossless"] is True
    
    def test_get_video_encoding_params(self, content_exporter):
        """Test video encoding parameter generation."""
        settings = ExportSettings(
            format=ExportFormat.MP4,
            quality=QualityLevel.HIGH,
            compression=CompressionType.FAST,
            frame_rate=30,
            bitrate="2M"
        )
        params = content_exporter._get_video_encoding_params(settings)
        
        assert params["vcodec"] == "libx264"
        assert params["r"] == 30
        assert params["b:v"] == "2M"
    
    @pytest.mark.asyncio
    async def test_export_image_success(self, content_exporter, sample_image, export_settings, content_metadata):
        """Test successful image export."""
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 30.0, "ssim": 0.95}
            
            result = await content_exporter.export_content(
                sample_image,
                export_settings,
                content_metadata
            )
            
            assert result.success is True
            assert result.format == ExportFormat.JPEG
            assert result.file_size > 0
            assert os.path.exists(result.output_path)
            assert "psnr" in result.quality_metrics
    
    @pytest.mark.asyncio
    async def test_export_nonexistent_file(self, content_exporter, export_settings):
        """Test export with nonexistent input file."""
        result = await content_exporter.export_content(
            "nonexistent_file.jpg",
            export_settings
        )
        
        assert result.success is False
        assert "not found" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_batch_export(self, content_exporter, temp_dir, export_settings):
        """Test batch export functionality."""
        # Create multiple test images
        image_paths = []
        for i in range(3):
            image_path = os.path.join(temp_dir, f"test_image_{i}.png")
            image = Image.new('RGB', (50, 50), color=['red', 'green', 'blue'][i])
            image.save(image_path)
            image_paths.append(image_path)
        
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 30.0, "ssim": 0.95}
            
            results = await content_exporter.batch_export(
                image_paths,
                export_settings,
                max_concurrent=2
            )
            
            assert len(results) == 3
            assert all(result.success for result in results)
            assert all(os.path.exists(result.output_path) for result in results)
    
    @pytest.mark.asyncio
    async def test_create_zip_archive(self, content_exporter, temp_dir):
        """Test ZIP archive creation."""
        # Create test files
        file_paths = []
        for i in range(2):
            file_path = os.path.join(temp_dir, f"test_file_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"Test content {i}")
            file_paths.append(file_path)
        
        result = await content_exporter.create_archive(
            file_paths,
            ExportFormat.ZIP,
            "test_archive.zip"
        )
        
        assert result.success is True
        assert result.format == ExportFormat.ZIP
        assert result.file_size > 0
        assert os.path.exists(result.output_path)
        assert result.quality_metrics["files_archived"] == 2
    
    @pytest.mark.asyncio
    async def test_create_tar_archive(self, content_exporter, temp_dir):
        """Test TAR archive creation."""
        # Create test files
        file_paths = []
        for i in range(2):
            file_path = os.path.join(temp_dir, f"test_file_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"Test content {i}")
            file_paths.append(file_path)
        
        result = await content_exporter.create_archive(
            file_paths,
            ExportFormat.TAR,
            "test_archive.tar.gz"
        )
        
        assert result.success is True
        assert result.format == ExportFormat.TAR
        assert result.file_size > 0
        assert os.path.exists(result.output_path)
    
    def test_get_supported_formats(self, content_exporter):
        """Test getting supported formats."""
        image_formats = content_exporter.get_supported_formats("image")
        video_formats = content_exporter.get_supported_formats("video")
        unknown_formats = content_exporter.get_supported_formats("unknown")
        
        assert ExportFormat.JPEG in image_formats
        assert ExportFormat.PNG in image_formats
        assert ExportFormat.MP4 in video_formats
        assert ExportFormat.WEBM in video_formats
        assert len(unknown_formats) == 0
    
    def test_get_format_info(self, content_exporter):
        """Test getting format information."""
        jpeg_info = content_exporter.get_format_info(ExportFormat.JPEG)
        png_info = content_exporter.get_format_info(ExportFormat.PNG)
        
        assert "description" in jpeg_info
        assert jpeg_info["supports_transparency"] is False
        assert png_info["supports_transparency"] is True
    
    @pytest.mark.asyncio
    async def test_apply_watermark(self, content_exporter):
        """Test watermark application."""
        # Create test image
        image = Image.new('RGB', (200, 200), color='white')
        
        watermarked = await content_exporter._apply_watermark(image, "TEST WATERMARK")
        
        assert watermarked.size == image.size
        assert watermarked.mode == image.mode
        # Watermarked image should be different from original
        assert not np.array_equal(np.array(image), np.array(watermarked))
    
    @pytest.mark.asyncio
    async def test_calculate_image_quality_metrics_error_handling(self, content_exporter):
        """Test quality metrics calculation error handling."""
        # Test with invalid paths
        metrics = await content_exporter._calculate_image_quality_metrics(
            "nonexistent1.jpg",
            "nonexistent2.jpg"
        )
        
        assert "error" in metrics
        assert metrics["error"] == 1.0


class TestFormatConverter:
    """Test cases for FormatConverter class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a sample image for testing."""
        image_path = os.path.join(temp_dir, "test_image.png")
        image = Image.new('RGB', (100, 100), color='blue')
        image.save(image_path)
        return image_path
    
    @pytest.fixture
    def format_converter(self, temp_dir):
        """Create FormatConverter instance for testing."""
        exporter = ContentExporter(output_directory=temp_dir)
        return FormatConverter(exporter)
    
    @pytest.mark.asyncio
    async def test_convert_format(self, format_converter, sample_image):
        """Test format conversion."""
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 30.0, "ssim": 0.95}
            
            result = await format_converter.convert_format(
                sample_image,
                ExportFormat.JPEG,
                QualityLevel.HIGH,
                preserve_metadata=True
            )
            
            assert result.success is True
            assert result.format == ExportFormat.JPEG
            assert os.path.exists(result.output_path)
    
    @pytest.mark.asyncio
    async def test_batch_convert(self, format_converter, temp_dir):
        """Test batch format conversion."""
        # Create multiple test images
        image_paths = []
        for i in range(3):
            image_path = os.path.join(temp_dir, f"test_image_{i}.png")
            image = Image.new('RGB', (50, 50), color=['red', 'green', 'blue'][i])
            image.save(image_path)
            image_paths.append(image_path)
        
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 30.0, "ssim": 0.95}
            
            results = await format_converter.batch_convert(
                image_paths,
                ExportFormat.WEBP,
                QualityLevel.MEDIUM
            )
            
            assert len(results) == 3
            assert all(result.success for result in results)
            assert all(result.format == ExportFormat.WEBP for result in results)
    
    @pytest.mark.asyncio
    async def test_optimize_for_web_image(self, format_converter, sample_image):
        """Test web optimization for images."""
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 30.0, "ssim": 0.95}
            
            result = await format_converter.optimize_for_web(sample_image)
            
            assert result.success is True
            assert result.format == ExportFormat.WEBP
            assert os.path.exists(result.output_path)
    
    @pytest.mark.asyncio
    async def test_optimize_for_web_unsupported_type(self, format_converter, temp_dir):
        """Test web optimization with unsupported content type."""
        # Create a text file
        text_path = os.path.join(temp_dir, "test.txt")
        with open(text_path, 'w') as f:
            f.write("test content")
        
        with pytest.raises(ValueError, match="Unsupported content type"):
            await format_converter.optimize_for_web(text_path)


class TestExportSettings:
    """Test cases for ExportSettings class."""
    
    def test_export_settings_creation(self):
        """Test ExportSettings creation with defaults."""
        settings = ExportSettings(format=ExportFormat.PNG)
        
        assert settings.format == ExportFormat.PNG
        assert settings.quality == QualityLevel.HIGH
        assert settings.compression == CompressionType.BALANCED
        assert settings.preserve_metadata is True
        assert settings.custom_params == {}
    
    def test_export_settings_with_custom_params(self):
        """Test ExportSettings with custom parameters."""
        custom_params = {"custom_option": "value"}
        settings = ExportSettings(
            format=ExportFormat.JPEG,
            quality=QualityLevel.LOW,
            compression=CompressionType.MAXIMUM,
            resolution=(800, 600),
            custom_params=custom_params
        )
        
        assert settings.format == ExportFormat.JPEG
        assert settings.quality == QualityLevel.LOW
        assert settings.compression == CompressionType.MAXIMUM
        assert settings.resolution == (800, 600)
        assert settings.custom_params == custom_params


class TestContentMetadata:
    """Test cases for ContentMetadata class."""
    
    def test_content_metadata_creation(self):
        """Test ContentMetadata creation with defaults."""
        metadata = ContentMetadata(title="Test Title")
        
        assert metadata.title == "Test Title"
        assert metadata.tags == []
        assert metadata.custom_fields == {}
        assert metadata.creation_date is not None
    
    def test_content_metadata_with_all_fields(self):
        """Test ContentMetadata with all fields."""
        from datetime import datetime
        
        creation_date = datetime.now()
        tags = ["tag1", "tag2"]
        custom_fields = {"field1": "value1"}
        
        metadata = ContentMetadata(
            title="Test Title",
            description="Test Description",
            creator="Test Creator",
            creation_date=creation_date,
            generation_prompt="Test prompt",
            model_used="test_model",
            tags=tags,
            copyright="Test Copyright",
            custom_fields=custom_fields
        )
        
        assert metadata.title == "Test Title"
        assert metadata.description == "Test Description"
        assert metadata.creator == "Test Creator"
        assert metadata.creation_date == creation_date
        assert metadata.generation_prompt == "Test prompt"
        assert metadata.model_used == "test_model"
        assert metadata.tags == tags
        assert metadata.copyright == "Test Copyright"
        assert metadata.custom_fields == custom_fields


class TestExportResult:
    """Test cases for ExportResult class."""
    
    def test_export_result_creation(self):
        """Test ExportResult creation."""
        result = ExportResult(
            success=True,
            output_path="/path/to/output.jpg",
            file_size=1024,
            format=ExportFormat.JPEG,
            quality_metrics={"psnr": 30.0},
            processing_time=2.5
        )
        
        assert result.success is True
        assert result.output_path == "/path/to/output.jpg"
        assert result.file_size == 1024
        assert result.format == ExportFormat.JPEG
        assert result.quality_metrics == {"psnr": 30.0}
        assert result.processing_time == 2.5
        assert result.warnings == []
    
    def test_export_result_with_error(self):
        """Test ExportResult with error."""
        result = ExportResult(
            success=False,
            output_path="",
            file_size=0,
            format=ExportFormat.PNG,
            quality_metrics={},
            processing_time=0.1,
            error_message="Test error",
            warnings=["Warning 1", "Warning 2"]
        )
        
        assert result.success is False
        assert result.error_message == "Test error"
        assert len(result.warnings) == 2


@pytest.mark.integration
class TestContentExportIntegration:
    """Integration tests for content export functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def content_exporter(self, temp_dir):
        """Create ContentExporter instance for testing."""
        return ContentExporter(output_directory=temp_dir)
    
    @pytest.mark.asyncio
    async def test_full_export_workflow(self, content_exporter, temp_dir):
        """Test complete export workflow."""
        # Create test image
        input_path = os.path.join(temp_dir, "input.png")
        image = Image.new('RGB', (200, 200), color='red')
        image.save(input_path)
        
        # Create export settings
        settings = ExportSettings(
            format=ExportFormat.JPEG,
            quality=QualityLevel.HIGH,
            compression=CompressionType.BALANCED,
            watermark="TEST"
        )
        
        # Create metadata
        metadata = ContentMetadata(
            title="Integration Test Image",
            creator="Test Suite",
            tags=["integration", "test"]
        )
        
        # Perform export
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 30.0, "ssim": 0.95}
            
            result = await content_exporter.export_content(
                input_path,
                settings,
                metadata
            )
        
        # Verify results
        assert result.success is True
        assert result.format == ExportFormat.JPEG
        assert result.file_size > 0
        assert os.path.exists(result.output_path)
        assert result.processing_time > 0
        
        # Verify output file
        output_image = Image.open(result.output_path)
        assert output_image.format == 'JPEG'
        assert output_image.size == (200, 200)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, content_exporter, temp_dir):
        """Test error handling and recovery mechanisms."""
        # Test with invalid input
        settings = ExportSettings(format=ExportFormat.PNG)
        
        result = await content_exporter.export_content(
            "nonexistent_file.jpg",
            settings
        )
        
        assert result.success is False
        assert result.error_message is not None
        assert result.processing_time >= 0