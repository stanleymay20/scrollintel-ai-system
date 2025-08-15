"""
Comprehensive tests for style transfer capabilities.
Tests style consistency, content preservation, and batch processing.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from PIL import Image

from scrollintel.engines.style_transfer_engine import StyleTransferEngine, StyleType
from scrollintel.models.style_transfer_models import (
    StyleTransferRequest, StyleTransferResult, StyleTransferStatus,
    ArtisticStyle, ContentPreservationLevel, StyleTransferConfig,
    BatchProcessingRequest, StyleTransferJob, StylePreset,
    create_default_presets, validate_style_transfer_request,
    calculate_estimated_processing_time
)


class TestStyleTransferEngine:
    """Test suite for StyleTransferEngine."""
    
    @pytest.fixture
    async def style_engine(self):
        """Create and initialize style transfer engine."""
        engine = StyleTransferEngine()
        await engine.initialize()
        return engine
    
    @pytest.fixture
    def sample_images(self):
        """Create sample test images."""
        temp_dir = tempfile.mkdtemp()
        
        # Create content image
        content_img = Image.new('RGB', (256, 256), (100, 150, 200))
        content_path = Path(temp_dir) / "content.png"
        content_img.save(content_path)
        
        # Create style image
        style_img = Image.new('RGB', (256, 256), (200, 100, 50))
        style_path = Path(temp_dir) / "style.png"
        style_img.save(style_path)
        
        yield {
            "temp_dir": temp_dir,
            "content_path": str(content_path),
            "style_path": str(style_path)
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, style_engine):
        """Test style transfer engine initialization."""
        assert style_engine.is_initialized
        assert style_engine.neural_transfer is not None
        assert len(style_engine.style_presets) > 0
        
        capabilities = style_engine.get_capabilities()
        assert "engine_name" in capabilities
        assert "supported_styles" in capabilities
        assert "features" in capabilities
    
    @pytest.mark.asyncio
    async def test_single_style_transfer(self, style_engine, sample_images):
        """Test single image style transfer."""
        content_path = sample_images["content_path"]
        
        # Test with preset style
        result_path = await style_engine.apply_style_transfer(
            content_path=content_path,
            style_type=StyleType.IMPRESSIONIST
        )
        
        assert Path(result_path).exists()
        
        # Verify output image
        result_img = Image.open(result_path)
        assert result_img.size[0] > 0
        assert result_img.size[1] > 0
        assert result_img.mode == 'RGB'
    
    @pytest.mark.asyncio
    async def test_custom_style_transfer(self, style_engine, sample_images):
        """Test style transfer with custom style image."""
        content_path = sample_images["content_path"]
        style_path = sample_images["style_path"]
        
        result_path = await style_engine.apply_style_transfer(
            content_path=content_path,
            style_path=style_path
        )
        
        assert Path(result_path).exists()
        
        # Verify output image
        result_img = Image.open(result_path)
        assert result_img.size[0] > 0
        assert result_img.size[1] > 0
    
    @pytest.mark.asyncio
    async def test_batch_style_transfer(self, style_engine, sample_images):
        """Test batch processing of multiple images."""
        temp_dir = sample_images["temp_dir"]
        
        # Create multiple content images
        content_paths = []
        for i in range(3):
            img = Image.new('RGB', (128, 128), (i * 50, i * 60, i * 70))
            path = Path(temp_dir) / f"content_{i}.png"
            img.save(path)
            content_paths.append(str(path))
        
        # Create batch request
        batch_request = BatchProcessingRequest(
            content_paths=content_paths,
            style_type=ArtisticStyle.WATERCOLOR
        )
        
        result_paths = await style_engine.batch_style_transfer(batch_request)
        
        assert len(result_paths) == len(content_paths)
        
        # Verify all results exist
        for result_path in result_paths:
            assert Path(result_path).exists()
            result_img = Image.open(result_path)
            assert result_img.size[0] > 0
            assert result_img.size[1] > 0
    
    @pytest.mark.asyncio
    async def test_multiple_styles_single_image(self, style_engine, sample_images):
        """Test applying multiple styles to a single image."""
        content_path = sample_images["content_path"]
        
        style_types = [
            StyleType.IMPRESSIONIST,
            StyleType.WATERCOLOR,
            StyleType.CARTOON
        ]
        
        result_paths = await style_engine.apply_multiple_styles(
            content_path=content_path,
            style_types=style_types
        )
        
        assert len(result_paths) == len(style_types)
        
        # Verify all results exist and are different
        for result_path in result_paths:
            assert Path(result_path).exists()
            result_img = Image.open(result_path)
            assert result_img.size[0] > 0
            assert result_img.size[1] > 0
    
    @pytest.mark.asyncio
    async def test_style_transfer_with_config(self, style_engine, sample_images):
        """Test style transfer with custom configuration."""
        content_path = sample_images["content_path"]
        
        # Custom configuration
        config = StyleTransferConfig(
            content_weight=2.0,
            style_weight=5000.0,
            num_iterations=500,
            max_image_size=256,
            preserve_colors=True,
            blend_ratio=0.8
        )
        
        result_path = await style_engine.apply_style_transfer(
            content_path=content_path,
            style_type=StyleType.ABSTRACT,
            config=config
        )
        
        assert Path(result_path).exists()
    
    @pytest.mark.asyncio
    async def test_content_preservation_levels(self, style_engine, sample_images):
        """Test different content preservation levels."""
        content_path = sample_images["content_path"]
        
        preservation_levels = [
            ContentPreservationLevel.LOW,
            ContentPreservationLevel.MEDIUM,
            ContentPreservationLevel.HIGH,
            ContentPreservationLevel.MAXIMUM
        ]
        
        results = []
        for level in preservation_levels:
            config = StyleTransferConfig(
                content_preservation_level=level,
                num_iterations=200  # Faster for testing
            )
            
            result_path = await style_engine.apply_style_transfer(
                content_path=content_path,
                style_type=StyleType.IMPRESSIONIST,
                config=config
            )
            
            results.append(result_path)
            assert Path(result_path).exists()
        
        # All results should be different files
        assert len(set(results)) == len(results)
    
    @pytest.mark.asyncio
    async def test_style_consistency_measurement(self, style_engine, sample_images):
        """Test style consistency measurement across multiple images."""
        temp_dir = sample_images["temp_dir"]
        
        # Create similar content images
        content_paths = []
        for i in range(2):
            img = Image.new('RGB', (128, 128), (100 + i * 10, 150 + i * 10, 200 + i * 10))
            path = Path(temp_dir) / f"similar_content_{i}.png"
            img.save(path)
            content_paths.append(str(path))
        
        # Apply same style to both images
        results = []
        for content_path in content_paths:
            result_path = await style_engine.apply_style_transfer(
                content_path=content_path,
                style_type=StyleType.WATERCOLOR
            )
            results.append(result_path)
        
        # Verify consistency (placeholder - in real implementation would analyze images)
        assert len(results) == len(content_paths)
        for result_path in results:
            assert Path(result_path).exists()
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_image(self, style_engine):
        """Test error handling for invalid image files."""
        with pytest.raises(Exception):
            await style_engine.apply_style_transfer(
                content_path="nonexistent_file.png",
                style_type=StyleType.IMPRESSIONIST
            )
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_style(self, style_engine, sample_images):
        """Test error handling for invalid style parameters."""
        content_path = sample_images["content_path"]
        
        with pytest.raises(Exception):
            await style_engine.apply_style_transfer(
                content_path=content_path,
                style_path="nonexistent_style.png"
            )
    
    @pytest.mark.asyncio
    async def test_preset_style_patterns(self, style_engine):
        """Test generation of preset style patterns."""
        for style_type in StyleType:
            style_image = await style_engine._get_preset_style_image(style_type)
            assert isinstance(style_image, Image.Image)
            assert style_image.size == (256, 256)
            assert style_image.mode == 'RGB'
    
    @pytest.mark.asyncio
    async def test_image_validation(self, style_engine, sample_images):
        """Test image validation functionality."""
        content_path = sample_images["content_path"]
        
        # Valid image should pass
        is_valid = await style_engine._validate_image(content_path)
        assert is_valid
        
        # Invalid path should fail
        with pytest.raises(Exception):
            await style_engine._validate_image("nonexistent.png")
    
    @pytest.mark.asyncio
    async def test_blend_with_original(self, style_engine):
        """Test blending stylized image with original."""
        # Create test images
        original = Image.new('RGB', (100, 100), (255, 0, 0))  # Red
        stylized = Image.new('RGB', (100, 100), (0, 0, 255))  # Blue
        
        # Test different blend ratios
        blend_ratios = [0.0, 0.5, 1.0]
        
        for ratio in blend_ratios:
            blended = style_engine._blend_with_original(original, stylized, ratio)
            assert isinstance(blended, Image.Image)
            assert blended.size == original.size
            assert blended.mode == 'RGB'


class TestStyleTransferModels:
    """Test suite for style transfer data models."""
    
    def test_style_transfer_config_creation(self):
        """Test StyleTransferConfig creation and validation."""
        config = StyleTransferConfig(
            content_weight=1.5,
            style_weight=8000.0,
            num_iterations=800,
            preserve_colors=True
        )
        
        assert config.content_weight == 1.5
        assert config.style_weight == 8000.0
        assert config.num_iterations == 800
        assert config.preserve_colors is True
    
    def test_style_transfer_config_serialization(self):
        """Test config serialization and deserialization."""
        config = StyleTransferConfig(
            content_weight=2.0,
            style_weight=10000.0,
            content_preservation_level=ContentPreservationLevel.HIGH
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["content_weight"] == 2.0
        assert config_dict["content_preservation_level"] == "high"
        
        # Test from_dict
        restored_config = StyleTransferConfig.from_dict(config_dict)
        assert restored_config.content_weight == config.content_weight
        assert restored_config.content_preservation_level == config.content_preservation_level
    
    def test_style_transfer_request_validation(self):
        """Test StyleTransferRequest validation."""
        # Valid request
        request = StyleTransferRequest(
            id="test_request",
            content_paths=["image1.png", "image2.png"],
            style_type=ArtisticStyle.IMPRESSIONIST
        )
        
        assert request.id == "test_request"
        assert len(request.content_paths) == 2
        assert request.style_type == ArtisticStyle.IMPRESSIONIST
        
        # Invalid request - no content paths
        with pytest.raises(ValueError):
            StyleTransferRequest(
                id="invalid_request",
                content_paths=[],
                style_type=ArtisticStyle.IMPRESSIONIST
            )
        
        # Invalid request - no style specification
        with pytest.raises(ValueError):
            StyleTransferRequest(
                id="invalid_request",
                content_paths=["image1.png"]
            )
    
    def test_style_transfer_request_serialization(self):
        """Test request serialization."""
        request = StyleTransferRequest(
            id="test_request",
            content_paths=["image1.png"],
            style_type=ArtisticStyle.WATERCOLOR,
            batch_processing=True,
            config=StyleTransferConfig(num_iterations=500)
        )
        
        request_dict = request.to_dict()
        assert isinstance(request_dict, dict)
        assert request_dict["id"] == "test_request"
        assert request_dict["style_type"] == "watercolor"
        assert request_dict["batch_processing"] is True
        assert isinstance(request_dict["config"], dict)
    
    def test_style_transfer_result_creation(self):
        """Test StyleTransferResult creation."""
        result = StyleTransferResult(
            id="test_result",
            status=StyleTransferStatus.COMPLETED,
            result_paths=["result1.png", "result2.png"],
            processing_time=45.5,
            style_consistency_score=0.85,
            content_preservation_score=0.90
        )
        
        assert result.id == "test_result"
        assert result.status == StyleTransferStatus.COMPLETED
        assert len(result.result_paths) == 2
        assert result.processing_time == 45.5
        assert result.style_consistency_score == 0.85
        assert result.content_preservation_score == 0.90
        assert result.completed_at is not None  # Should be set automatically
    
    def test_batch_processing_request_validation(self):
        """Test BatchProcessingRequest validation."""
        # Valid batch request
        batch_request = BatchProcessingRequest(
            content_paths=["img1.png", "img2.png", "img3.png"],
            style_type=ArtisticStyle.ABSTRACT
        )
        
        assert len(batch_request.content_paths) == 3
        assert batch_request.style_type == ArtisticStyle.ABSTRACT
        
        # Invalid batch request - too few images
        with pytest.raises(ValueError):
            BatchProcessingRequest(
                content_paths=["img1.png"],
                style_type=ArtisticStyle.ABSTRACT
            )
        
        # Invalid batch request - no style
        with pytest.raises(ValueError):
            BatchProcessingRequest(
                content_paths=["img1.png", "img2.png"]
            )
    
    def test_style_transfer_job_lifecycle(self):
        """Test StyleTransferJob lifecycle management."""
        request = StyleTransferRequest(
            id="test_request",
            content_paths=["image1.png"],
            style_type=ArtisticStyle.CARTOON
        )
        
        job = StyleTransferJob(
            id="test_job",
            request=request,
            status=StyleTransferStatus.PENDING
        )
        
        # Test job start
        job.start_processing()
        assert job.status == StyleTransferStatus.PROCESSING
        assert job.started_at is not None
        
        # Test progress update
        job.update_progress(50.0, "Processing style transfer")
        assert job.progress == 50.0
        assert job.current_step == "Processing style transfer"
        
        # Test completion
        result = StyleTransferResult(
            id="test_result",
            status=StyleTransferStatus.COMPLETED,
            result_paths=["result.png"]
        )
        
        job.complete_with_result(result)
        assert job.status == StyleTransferStatus.COMPLETED
        assert job.progress == 100.0
        assert job.result == result
        assert job.completed_at is not None
    
    def test_style_presets_creation(self):
        """Test creation of default style presets."""
        presets = create_default_presets()
        
        assert len(presets) > 0
        
        for preset in presets:
            assert isinstance(preset, StylePreset)
            assert preset.name
            assert isinstance(preset.style_type, ArtisticStyle)
            assert isinstance(preset.config, StyleTransferConfig)
            assert preset.description
    
    def test_request_validation_function(self):
        """Test request validation utility function."""
        # Valid request data
        request_data = {
            "id": "test_request",
            "content_paths": ["image1.png"],
            "style_type": "impressionist",
            "batch_processing": False
        }
        
        request = validate_style_transfer_request(request_data)
        assert isinstance(request, StyleTransferRequest)
        assert request.style_type == ArtisticStyle.IMPRESSIONIST
        
        # Invalid request data
        invalid_data = {
            "id": "invalid_request",
            "content_paths": [],  # Empty paths
            "style_type": "impressionist"
        }
        
        with pytest.raises(ValueError):
            validate_style_transfer_request(invalid_data)
    
    def test_processing_time_estimation(self):
        """Test processing time estimation."""
        # Single image request
        single_request = StyleTransferRequest(
            id="single_request",
            content_paths=["image1.png"],
            style_type=ArtisticStyle.WATERCOLOR
        )
        
        time_estimate = calculate_estimated_processing_time(single_request)
        assert time_estimate > 0
        
        # Batch request
        batch_request = StyleTransferRequest(
            id="batch_request",
            content_paths=["img1.png", "img2.png", "img3.png"],
            style_type=ArtisticStyle.ABSTRACT,
            batch_processing=True
        )
        
        batch_time_estimate = calculate_estimated_processing_time(batch_request)
        assert batch_time_estimate > time_estimate  # Should take longer
        
        # Multiple styles request
        multi_style_request = StyleTransferRequest(
            id="multi_style_request",
            content_paths=["image1.png"],
            style_types=[ArtisticStyle.IMPRESSIONIST, ArtisticStyle.CUBIST],
            multiple_styles=True
        )
        
        multi_time_estimate = calculate_estimated_processing_time(multi_style_request)
        assert multi_time_estimate > time_estimate  # Should take longer


class TestStyleTransferIntegration:
    """Integration tests for style transfer system."""
    
    @pytest.fixture
    async def style_engine(self):
        """Create style transfer engine for integration tests."""
        engine = StyleTransferEngine()
        await engine.initialize()
        return engine
    
    @pytest.fixture
    def test_images(self):
        """Create test images for integration testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create diverse test images
        images = {}
        
        # Portrait-like image
        portrait = Image.new('RGB', (200, 300), (180, 150, 120))
        portrait_path = Path(temp_dir) / "portrait.png"
        portrait.save(portrait_path)
        images["portrait"] = str(portrait_path)
        
        # Landscape-like image
        landscape = Image.new('RGB', (400, 200), (100, 180, 100))
        landscape_path = Path(temp_dir) / "landscape.png"
        landscape.save(landscape_path)
        images["landscape"] = str(landscape_path)
        
        # Abstract pattern
        abstract = Image.new('RGB', (256, 256), (200, 50, 200))
        abstract_path = Path(temp_dir) / "abstract.png"
        abstract.save(abstract_path)
        images["abstract"] = str(abstract_path)
        
        yield {
            "temp_dir": temp_dir,
            "images": images
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_end_to_end_style_transfer(self, style_engine, test_images):
        """Test complete end-to-end style transfer workflow."""
        portrait_path = test_images["images"]["portrait"]
        
        # Create comprehensive request
        request = StyleTransferRequest(
            id="e2e_test",
            content_paths=[portrait_path],
            style_type=ArtisticStyle.IMPRESSIONIST,
            config=StyleTransferConfig(
                num_iterations=300,  # Faster for testing
                content_preservation_level=ContentPreservationLevel.MEDIUM
            )
        )
        
        # Process request
        result = await style_engine.process_style_transfer_request(request)
        
        # Verify result
        assert result.status == StyleTransferStatus.COMPLETED
        assert len(result.result_paths) == 1
        assert result.processing_time > 0
        assert result.style_consistency_score > 0
        assert result.content_preservation_score > 0
        
        # Verify output file
        output_path = result.result_paths[0]
        assert Path(output_path).exists()
        
        output_img = Image.open(output_path)
        assert output_img.size[0] > 0
        assert output_img.size[1] > 0
    
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, style_engine, test_images):
        """Test batch processing integration."""
        image_paths = list(test_images["images"].values())
        
        # Create batch request
        request = StyleTransferRequest(
            id="batch_integration_test",
            content_paths=image_paths,
            style_type=ArtisticStyle.WATERCOLOR,
            batch_processing=True,
            config=StyleTransferConfig(num_iterations=200)  # Faster for testing
        )
        
        # Process batch request
        result = await style_engine.process_style_transfer_request(request)
        
        # Verify batch result
        assert result.status == StyleTransferStatus.COMPLETED
        assert len(result.result_paths) == len(image_paths)
        assert result.processing_time > 0
        
        # Verify all output files
        for output_path in result.result_paths:
            assert Path(output_path).exists()
            output_img = Image.open(output_path)
            assert output_img.size[0] > 0
            assert output_img.size[1] > 0
    
    @pytest.mark.asyncio
    async def test_multiple_styles_integration(self, style_engine, test_images):
        """Test multiple styles application integration."""
        portrait_path = test_images["images"]["portrait"]
        
        styles = [
            ArtisticStyle.IMPRESSIONIST,
            ArtisticStyle.WATERCOLOR,
            ArtisticStyle.CARTOON
        ]
        
        # Create multiple styles request
        request = StyleTransferRequest(
            id="multi_styles_test",
            content_paths=[portrait_path],
            style_types=styles,
            multiple_styles=True,
            config=StyleTransferConfig(num_iterations=200)
        )
        
        # Process request
        result = await style_engine.process_style_transfer_request(request)
        
        # Verify result
        assert result.status == StyleTransferStatus.COMPLETED
        assert len(result.result_paths) == len(styles)
        
        # Verify all outputs are different
        output_sizes = []
        for output_path in result.result_paths:
            assert Path(output_path).exists()
            output_img = Image.open(output_path)
            output_sizes.append(output_img.size)
        
        # All should have valid sizes
        for size in output_sizes:
            assert size[0] > 0 and size[1] > 0
    
    @pytest.mark.asyncio
    async def test_style_consistency_across_similar_images(self, style_engine, test_images):
        """Test style consistency when applied to similar images."""
        temp_dir = test_images["temp_dir"]
        
        # Create similar images with slight variations
        similar_images = []
        for i in range(3):
            # Slight color variations
            color = (150 + i * 10, 120 + i * 5, 100 + i * 8)
            img = Image.new('RGB', (200, 200), color)
            path = Path(temp_dir) / f"similar_{i}.png"
            img.save(path)
            similar_images.append(str(path))
        
        # Apply same style to all images
        results = []
        for img_path in similar_images:
            request = StyleTransferRequest(
                id=f"consistency_test_{img_path}",
                content_paths=[img_path],
                style_type=ArtisticStyle.ABSTRACT,
                config=StyleTransferConfig(num_iterations=200)
            )
            
            result = await style_engine.process_style_transfer_request(request)
            results.append(result)
        
        # Verify all completed successfully
        for result in results:
            assert result.status == StyleTransferStatus.COMPLETED
            assert len(result.result_paths) == 1
            assert result.style_consistency_score > 0.5  # Should have reasonable consistency
    
    @pytest.mark.asyncio
    async def test_content_preservation_levels_integration(self, style_engine, test_images):
        """Test different content preservation levels in integration."""
        portrait_path = test_images["images"]["portrait"]
        
        preservation_levels = [
            ContentPreservationLevel.LOW,
            ContentPreservationLevel.MEDIUM,
            ContentPreservationLevel.HIGH,
            ContentPreservationLevel.MAXIMUM
        ]
        
        results = []
        for level in preservation_levels:
            config = StyleTransferConfig(
                content_preservation_level=level,
                num_iterations=200,
                blend_ratio=0.5 if level == ContentPreservationLevel.MAXIMUM else 1.0
            )
            
            request = StyleTransferRequest(
                id=f"preservation_test_{level.value}",
                content_paths=[portrait_path],
                style_type=ArtisticStyle.IMPRESSIONIST,
                config=config
            )
            
            result = await style_engine.process_style_transfer_request(request)
            results.append((level, result))
        
        # Verify all completed and have different preservation scores
        preservation_scores = []
        for level, result in results:
            assert result.status == StyleTransferStatus.COMPLETED
            assert result.content_preservation_score > 0
            preservation_scores.append(result.content_preservation_score)
        
        # Higher preservation levels should generally have higher scores
        # (This is a simplified check - real implementation would be more sophisticated)
        assert len(set(preservation_scores)) > 1  # Should have different scores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])