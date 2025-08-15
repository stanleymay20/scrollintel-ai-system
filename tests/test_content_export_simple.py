"""
Simple tests for content export functionality without external dependencies.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image

from scrollintel.engines.visual_generation.utils.content_exporter import (
    ExportSettings,
    ContentMetadata,
    ExportFormat,
    QualityLevel,
    CompressionType,
    ExportResult
)


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


class TestContentExporterBasic:
    """Basic test cases for ContentExporter class without external dependencies."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_content_exporter_initialization(self, temp_dir):
        """Test ContentExporter initialization."""
        # Import here to avoid dependency issues during collection
        from scrollintel.engines.visual_generation.utils.content_exporter import ContentExporter
        
        exporter = ContentExporter(output_directory=temp_dir)
        
        assert exporter.output_directory == Path(temp_dir)
        assert exporter.output_directory.exists()
        assert len(exporter.quality_settings) > 0
        assert len(exporter.compression_presets) > 0
    
    def test_detect_content_type(self, temp_dir):
        """Test content type detection."""
        from scrollintel.engines.visual_generation.utils.content_exporter import ContentExporter
        
        exporter = ContentExporter(output_directory=temp_dir)
        
        assert exporter._detect_content_type("test.jpg") == "image"
        assert exporter._detect_content_type("test.png") == "image"
        assert exporter._detect_content_type("test.mp4") == "video"
        assert exporter._detect_content_type("test.avi") == "video"
        assert exporter._detect_content_type("test.txt") == "unknown"
    
    def test_get_jpeg_params(self, temp_dir):
        """Test JPEG parameter generation."""
        from scrollintel.engines.visual_generation.utils.content_exporter import ContentExporter
        
        exporter = ContentExporter(output_directory=temp_dir)
        export_settings = ExportSettings(format=ExportFormat.JPEG)
        
        params = exporter._get_jpeg_params(export_settings)
        
        assert params["format"] == "JPEG"
        assert "quality" in params
        assert params["optimize"] is True
    
    def test_get_png_params(self, temp_dir):
        """Test PNG parameter generation."""
        from scrollintel.engines.visual_generation.utils.content_exporter import ContentExporter
        
        exporter = ContentExporter(output_directory=temp_dir)
        settings = ExportSettings(
            format=ExportFormat.PNG,
            compression=CompressionType.MAXIMUM
        )
        params = exporter._get_png_params(settings)
        
        assert params["format"] == "PNG"
        assert params["compress_level"] == 9
    
    def test_get_supported_formats(self, temp_dir):
        """Test getting supported formats."""
        from scrollintel.engines.visual_generation.utils.content_exporter import ContentExporter
        
        exporter = ContentExporter(output_directory=temp_dir)
        
        image_formats = exporter.get_supported_formats("image")
        video_formats = exporter.get_supported_formats("video")
        unknown_formats = exporter.get_supported_formats("unknown")
        
        assert ExportFormat.JPEG in image_formats
        assert ExportFormat.PNG in image_formats
        assert ExportFormat.MP4 in video_formats
        assert ExportFormat.WEBM in video_formats
        assert len(unknown_formats) == 0
    
    def test_get_format_info(self, temp_dir):
        """Test getting format information."""
        from scrollintel.engines.visual_generation.utils.content_exporter import ContentExporter
        
        exporter = ContentExporter(output_directory=temp_dir)
        
        jpeg_info = exporter.get_format_info(ExportFormat.JPEG)
        png_info = exporter.get_format_info(ExportFormat.PNG)
        
        assert "description" in jpeg_info
        assert jpeg_info["supports_transparency"] is False
        assert png_info["supports_transparency"] is True
    
    @pytest.mark.asyncio
    async def test_export_nonexistent_file(self, temp_dir):
        """Test export with nonexistent input file."""
        from scrollintel.engines.visual_generation.utils.content_exporter import ContentExporter
        
        exporter = ContentExporter(output_directory=temp_dir)
        export_settings = ExportSettings(format=ExportFormat.JPEG)
        
        result = await exporter.export_content(
            "nonexistent_file.jpg",
            export_settings
        )
        
        assert result.success is False
        assert "not found" in result.error_message.lower()


class TestFormatConverterBasic:
    """Basic test cases for FormatConverter class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_format_converter_initialization(self, temp_dir):
        """Test FormatConverter initialization."""
        from scrollintel.engines.visual_generation.utils.content_exporter import ContentExporter, FormatConverter
        
        exporter = ContentExporter(output_directory=temp_dir)
        converter = FormatConverter(exporter)
        
        assert converter.exporter == exporter
    
    @pytest.mark.asyncio
    async def test_optimize_for_web_unsupported_type(self, temp_dir):
        """Test web optimization with unsupported content type."""
        from scrollintel.engines.visual_generation.utils.content_exporter import ContentExporter, FormatConverter
        
        exporter = ContentExporter(output_directory=temp_dir)
        converter = FormatConverter(exporter)
        
        # Create a text file
        text_path = os.path.join(temp_dir, "test.txt")
        with open(text_path, 'w') as f:
            f.write("test content")
        
        with pytest.raises(ValueError, match="Unsupported content type"):
            await converter.optimize_for_web(text_path)


@pytest.mark.integration
class TestContentExportIntegrationSimple:
    """Simple integration tests for content export functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_basic_image_export_workflow(self, temp_dir):
        """Test basic image export workflow."""
        from scrollintel.engines.visual_generation.utils.content_exporter import ContentExporter
        
        # Create test image
        input_path = os.path.join(temp_dir, "input.png")
        image = Image.new('RGB', (100, 100), color='red')
        image.save(input_path)
        
        # Initialize exporter
        export_dir = os.path.join(temp_dir, "exports")
        exporter = ContentExporter(output_directory=export_dir)
        
        # Create export settings
        settings = ExportSettings(
            format=ExportFormat.JPEG,
            quality=QualityLevel.HIGH,
            compression=CompressionType.BALANCED
        )
        
        # Mock the quality metrics calculation to avoid external dependencies
        with patch.object(exporter, '_calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 30.0, "ssim": 0.95}
            
            # Perform export
            result = await exporter.export_content(input_path, settings)
        
        # Verify results
        assert result.success is True
        assert result.format == ExportFormat.JPEG
        assert result.file_size > 0
        assert os.path.exists(result.output_path)
        assert result.processing_time >= 0
        
        # Verify output file
        output_image = Image.open(result.output_path)
        assert output_image.format == 'JPEG'
        assert output_image.size == (100, 100)
    
    @pytest.mark.asyncio
    async def test_archive_creation_simple(self, temp_dir):
        """Test simple archive creation."""
        from scrollintel.engines.visual_generation.utils.content_exporter import ContentExporter
        
        exporter = ContentExporter(output_directory=temp_dir)
        
        # Create test files
        file_paths = []
        for i in range(2):
            file_path = os.path.join(temp_dir, f"test_file_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"Test content {i}")
            file_paths.append(file_path)
        
        result = await exporter.create_archive(
            file_paths,
            ExportFormat.ZIP,
            "test_archive.zip"
        )
        
        assert result.success is True
        assert result.format == ExportFormat.ZIP
        assert result.file_size > 0
        assert os.path.exists(result.output_path)
        assert result.quality_metrics["files_archived"] == 2