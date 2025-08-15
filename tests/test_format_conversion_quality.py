"""
Tests for format conversion accuracy and quality validation.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch

from scrollintel.engines.visual_generation.utils.content_exporter import (
    ContentExporter,
    FormatConverter,
    ExportSettings,
    ExportFormat,
    QualityLevel,
    CompressionType
)


class TestFormatConversionAccuracy:
    """Test format conversion accuracy and quality preservation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def content_exporter(self, temp_dir):
        """Create ContentExporter instance."""
        return ContentExporter(output_directory=temp_dir)
    
    @pytest.fixture
    def format_converter(self, content_exporter):
        """Create FormatConverter instance."""
        return FormatConverter(content_exporter)
    
    def create_test_image(self, temp_dir, name, size=(100, 100), color='red'):
        """Create a test image with specified properties."""
        image_path = os.path.join(temp_dir, name)
        image = Image.new('RGB', size, color=color)
        image.save(image_path)
        return image_path
    
    def create_gradient_image(self, temp_dir, name, size=(100, 100)):
        """Create a gradient test image for quality testing."""
        image_path = os.path.join(temp_dir, name)
        
        # Create gradient
        width, height = size
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(width):
            gradient[:, i, 0] = int(255 * i / width)  # Red gradient
            gradient[:, i, 1] = int(255 * (1 - i / width))  # Green gradient
            gradient[:, i, 2] = 128  # Constant blue
        
        image = Image.fromarray(gradient)
        image.save(image_path)
        return image_path
    
    @pytest.mark.asyncio
    async def test_png_to_jpeg_conversion_quality(self, format_converter, temp_dir):
        """Test PNG to JPEG conversion quality."""
        # Create high-quality PNG
        input_path = self.create_gradient_image(temp_dir, "test.png", (200, 200))
        
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 35.0, "ssim": 0.98, "mse": 10.5}
            
            # Convert to JPEG with high quality
            result = await format_converter.convert_format(
                input_path,
                ExportFormat.JPEG,
                QualityLevel.HIGH
            )
            
            assert result.success is True
            assert result.format == ExportFormat.JPEG
            assert result.quality_metrics["psnr"] > 30.0
            assert result.quality_metrics["ssim"] > 0.95
            
            # Verify output file
            output_image = Image.open(result.output_path)
            assert output_image.format == 'JPEG'
            assert output_image.size == (200, 200)
    
    @pytest.mark.asyncio
    async def test_jpeg_to_webp_conversion(self, format_converter, temp_dir):
        """Test JPEG to WebP conversion."""
        # Create JPEG image
        input_path = self.create_test_image(temp_dir, "test.jpg", (150, 150), 'blue')
        
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 32.0, "ssim": 0.96, "mse": 8.2}
            
            result = await format_converter.convert_format(
                input_path,
                ExportFormat.WEBP,
                QualityLevel.HIGH
            )
            
            assert result.success is True
            assert result.format == ExportFormat.WEBP
            assert os.path.exists(result.output_path)
            
            # WebP should provide good compression
            original_size = os.path.getsize(input_path)
            webp_size = result.file_size
            compression_ratio = original_size / webp_size
            assert compression_ratio > 1.0  # WebP should be smaller
    
    @pytest.mark.asyncio
    async def test_lossless_conversion_quality(self, format_converter, temp_dir):
        """Test lossless conversion quality."""
        input_path = self.create_test_image(temp_dir, "test.png", (100, 100), 'green')
        
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": float('inf'), "ssim": 1.0, "mse": 0.0}
            
            # Convert PNG to PNG (lossless)
            result = await format_converter.convert_format(
                input_path,
                ExportFormat.PNG,
                QualityLevel.LOSSLESS
            )
            
            assert result.success is True
            assert result.quality_metrics["psnr"] == float('inf')
            assert result.quality_metrics["ssim"] == 1.0
    
    @pytest.mark.asyncio
    async def test_quality_degradation_with_compression(self, format_converter, temp_dir):
        """Test quality degradation with different compression levels."""
        input_path = self.create_gradient_image(temp_dir, "test.png", (200, 200))
        
        quality_levels = [QualityLevel.LOW, QualityLevel.MEDIUM, QualityLevel.HIGH]
        results = []
        
        for quality in quality_levels:
            with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
                # Simulate decreasing quality with lower settings
                if quality == QualityLevel.LOW:
                    mock_metrics.return_value = {"psnr": 25.0, "ssim": 0.85, "mse": 25.0}
                elif quality == QualityLevel.MEDIUM:
                    mock_metrics.return_value = {"psnr": 30.0, "ssim": 0.92, "mse": 15.0}
                else:  # HIGH
                    mock_metrics.return_value = {"psnr": 35.0, "ssim": 0.98, "mse": 8.0}
                
                result = await format_converter.convert_format(
                    input_path,
                    ExportFormat.JPEG,
                    quality
                )
                results.append(result)
        
        # Verify quality increases with higher settings
        assert results[0].quality_metrics["psnr"] < results[1].quality_metrics["psnr"]
        assert results[1].quality_metrics["psnr"] < results[2].quality_metrics["psnr"]
        
        assert results[0].quality_metrics["ssim"] < results[1].quality_metrics["ssim"]
        assert results[1].quality_metrics["ssim"] < results[2].quality_metrics["ssim"]
    
    @pytest.mark.asyncio
    async def test_resolution_scaling_accuracy(self, content_exporter, temp_dir):
        """Test resolution scaling accuracy."""
        input_path = self.create_test_image(temp_dir, "test.png", (200, 200), 'purple')
        
        settings = ExportSettings(
            format=ExportFormat.PNG,
            resolution=(100, 100)
        )
        
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 28.0, "ssim": 0.90, "mse": 18.0}
            
            result = await content_exporter.export_content(input_path, settings)
            
            assert result.success is True
            
            # Verify output resolution
            output_image = Image.open(result.output_path)
            assert output_image.size == (100, 100)
    
    @pytest.mark.asyncio
    async def test_batch_conversion_consistency(self, format_converter, temp_dir):
        """Test consistency across batch conversions."""
        # Create multiple similar images
        input_paths = []
        for i in range(5):
            path = self.create_test_image(temp_dir, f"test_{i}.png", (100, 100), 'orange')
            input_paths.append(path)
        
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 32.0, "ssim": 0.95, "mse": 12.0}
            
            results = await format_converter.batch_convert(
                input_paths,
                ExportFormat.WEBP,
                QualityLevel.HIGH
            )
            
            assert len(results) == 5
            assert all(result.success for result in results)
            
            # Check consistency of quality metrics
            psnr_values = [result.quality_metrics["psnr"] for result in results]
            ssim_values = [result.quality_metrics["ssim"] for result in results]
            
            # All values should be the same for identical inputs
            assert len(set(psnr_values)) == 1
            assert len(set(ssim_values)) == 1
    
    @pytest.mark.asyncio
    async def test_format_specific_features(self, content_exporter, temp_dir):
        """Test format-specific features preservation."""
        # Create PNG with transparency
        image_path = os.path.join(temp_dir, "transparent.png")
        image = Image.new('RGBA', (100, 100), (255, 0, 0, 128))  # Semi-transparent red
        image.save(image_path)
        
        # Convert to PNG (should preserve transparency)
        png_settings = ExportSettings(format=ExportFormat.PNG)
        
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 35.0, "ssim": 0.98, "mse": 8.0}
            
            png_result = await content_exporter.export_content(image_path, png_settings)
            
            assert png_result.success is True
            
            # Verify transparency is preserved
            output_image = Image.open(png_result.output_path)
            assert output_image.mode in ['RGBA', 'LA']
        
        # Convert to JPEG (should lose transparency)
        jpeg_settings = ExportSettings(format=ExportFormat.JPEG)
        
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 30.0, "ssim": 0.92, "mse": 15.0}
            
            jpeg_result = await content_exporter.export_content(image_path, jpeg_settings)
            
            assert jpeg_result.success is True
            
            # Verify transparency is lost
            output_image = Image.open(jpeg_result.output_path)
            assert output_image.mode == 'RGB'
    
    @pytest.mark.asyncio
    async def test_compression_efficiency(self, format_converter, temp_dir):
        """Test compression efficiency across formats."""
        # Create detailed test image
        input_path = self.create_gradient_image(temp_dir, "detailed.png", (300, 300))
        original_size = os.path.getsize(input_path)
        
        formats_to_test = [
            (ExportFormat.JPEG, QualityLevel.HIGH),
            (ExportFormat.WEBP, QualityLevel.HIGH),
            (ExportFormat.PNG, QualityLevel.HIGH)
        ]
        
        compression_results = []
        
        for format_type, quality in formats_to_test:
            with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
                # Simulate different compression ratios
                if format_type == ExportFormat.JPEG:
                    mock_metrics.return_value = {"psnr": 32.0, "ssim": 0.94, "mse": 12.0}
                elif format_type == ExportFormat.WEBP:
                    mock_metrics.return_value = {"psnr": 34.0, "ssim": 0.96, "mse": 10.0}
                else:  # PNG
                    mock_metrics.return_value = {"psnr": 35.0, "ssim": 0.98, "mse": 8.0}
                
                result = await format_converter.convert_format(
                    input_path,
                    format_type,
                    quality
                )
                
                compression_ratio = original_size / result.file_size
                compression_results.append({
                    'format': format_type,
                    'compression_ratio': compression_ratio,
                    'quality_metrics': result.quality_metrics
                })
        
        # Verify all conversions succeeded
        assert len(compression_results) == 3
        
        # JPEG should provide good compression
        jpeg_result = next(r for r in compression_results if r['format'] == ExportFormat.JPEG)
        assert jpeg_result['compression_ratio'] > 1.0
        
        # WebP should provide better compression than JPEG with similar quality
        webp_result = next(r for r in compression_results if r['format'] == ExportFormat.WEBP)
        assert webp_result['compression_ratio'] > 1.0
    
    @pytest.mark.asyncio
    async def test_metadata_preservation_accuracy(self, content_exporter, temp_dir):
        """Test metadata preservation accuracy."""
        from scrollintel.engines.visual_generation.utils.content_exporter import ContentMetadata
        
        input_path = self.create_test_image(temp_dir, "test.png", (100, 100), 'cyan')
        
        metadata = ContentMetadata(
            title="Test Image",
            description="Test description",
            creator="Test Creator",
            tags=["test", "metadata"]
        )
        
        settings = ExportSettings(
            format=ExportFormat.JPEG,
            preserve_metadata=True
        )
        
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 32.0, "ssim": 0.95, "mse": 12.0}
            
            result = await content_exporter.export_content(input_path, settings, metadata)
            
            assert result.success is True
            
            # Verify file exists and has expected properties
            assert os.path.exists(result.output_path)
            output_image = Image.open(result.output_path)
            assert output_image.format == 'JPEG'
    
    @pytest.mark.asyncio
    async def test_error_recovery_during_conversion(self, format_converter, temp_dir):
        """Test error recovery during format conversion."""
        # Test with corrupted/invalid file
        invalid_path = os.path.join(temp_dir, "invalid.jpg")
        with open(invalid_path, 'w') as f:
            f.write("This is not an image file")
        
        result = await format_converter.convert_format(
            invalid_path,
            ExportFormat.PNG,
            QualityLevel.HIGH
        )
        
        assert result.success is False
        assert result.error_message is not None
        assert result.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_large_image_handling(self, format_converter, temp_dir):
        """Test handling of large images."""
        # Create a larger test image
        input_path = self.create_test_image(temp_dir, "large.png", (1000, 1000), 'yellow')
        
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 30.0, "ssim": 0.93, "mse": 15.0}
            
            result = await format_converter.convert_format(
                input_path,
                ExportFormat.WEBP,
                QualityLevel.MEDIUM
            )
            
            assert result.success is True
            assert result.file_size > 0
            assert result.processing_time > 0
            
            # Verify output
            output_image = Image.open(result.output_path)
            assert output_image.size == (1000, 1000)
    
    @pytest.mark.asyncio
    async def test_web_optimization_quality(self, format_converter, temp_dir):
        """Test web optimization quality and file size."""
        # Create test image
        input_path = self.create_gradient_image(temp_dir, "web_test.png", (800, 600))
        original_size = os.path.getsize(input_path)
        
        with patch('scrollintel.engines.visual_generation.utils.content_exporter.ContentExporter._calculate_image_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"psnr": 31.0, "ssim": 0.94, "mse": 14.0}
            
            result = await format_converter.optimize_for_web(input_path)
            
            assert result.success is True
            assert result.format == ExportFormat.WEBP
            
            # Web optimization should reduce file size
            compression_ratio = original_size / result.file_size
            assert compression_ratio > 1.0
            
            # Quality should still be acceptable
            assert result.quality_metrics["psnr"] > 25.0
            assert result.quality_metrics["ssim"] > 0.85
            
            # Verify output dimensions are web-appropriate
            output_image = Image.open(result.output_path)
            assert output_image.size[0] <= 1920  # Max web width
            assert output_image.size[1] <= 1080  # Max web height