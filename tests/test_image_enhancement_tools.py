"""
Tests for image enhancement tools including Real-ESRGAN and GFPGAN integration.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import numpy as np

from scrollintel.engines.visual_generation.models.enhancement_models import (
    ImageEnhancer,
    RealESRGANUpscaler,
    GFPGANFaceRestorer
)
from scrollintel.engines.visual_generation.base import GenerationRequest, GenerationStatus
from scrollintel.engines.visual_generation.config import ModelConfig
from scrollintel.engines.visual_generation.exceptions import ModelError, ValidationError


class TestRealESRGANUpscaler:
    """Test Real-ESRGAN upscaler functionality."""
    
    @pytest.fixture
    def upscaler(self):
        return RealESRGANUpscaler()
    
    @pytest.fixture
    def test_image(self):
        # Create a test image
        image = Image.new('RGB', (256, 256), color='red')
        return image
    
    @pytest.mark.asyncio
    async def test_upscaler_initialization(self, upscaler):
        """Test upscaler initialization."""
        await upscaler.initialize()
        
        assert upscaler.model is not None
        assert upscaler.model["loaded"] is True
        assert upscaler.model["scale_factor"] == 4
        assert upscaler.device in ["cuda", "cpu"]
    
    @pytest.mark.asyncio
    async def test_upscale_image(self, upscaler, test_image):
        """Test image upscaling functionality."""
        await upscaler.initialize()
        
        original_size = test_image.size
        upscaled_image = await upscaler.upscale_image(test_image)
        
        # Check that image was upscaled
        assert upscaled_image.size[0] == original_size[0] * upscaler.scale_factor
        assert upscaled_image.size[1] == original_size[1] * upscaler.scale_factor
        assert isinstance(upscaled_image, Image.Image)
    
    @pytest.mark.asyncio
    async def test_upscale_without_initialization(self, upscaler, test_image):
        """Test upscaling without initialization auto-initializes."""
        upscaled_image = await upscaler.upscale_image(test_image)
        
        assert upscaler.model is not None
        assert isinstance(upscaled_image, Image.Image)


class TestGFPGANFaceRestorer:
    """Test GFPGAN face restoration functionality."""
    
    @pytest.fixture
    def face_restorer(self):
        return GFPGANFaceRestorer()
    
    @pytest.fixture
    def test_image(self):
        # Create a test image with face-like content
        image = Image.new('RGB', (512, 512), color='white')
        return image
    
    @pytest.mark.asyncio
    async def test_face_restorer_initialization(self, face_restorer):
        """Test face restorer initialization."""
        await face_restorer.initialize()
        
        assert face_restorer.model is not None
        assert face_restorer.face_detector is not None
        assert face_restorer.model["loaded"] is True
        assert face_restorer.face_detector["loaded"] is True
    
    @pytest.mark.asyncio
    async def test_detect_faces(self, face_restorer, test_image):
        """Test face detection functionality."""
        await face_restorer.initialize()
        
        faces = await face_restorer.detect_faces(test_image)
        
        assert isinstance(faces, list)
        if faces:  # Mock returns one face
            face = faces[0]
            assert "bbox" in face
            assert "confidence" in face
            assert "landmarks" in face
            assert face["confidence"] > 0.5
    
    @pytest.mark.asyncio
    async def test_restore_faces(self, face_restorer, test_image):
        """Test face restoration functionality."""
        await face_restorer.initialize()
        
        restored_image = await face_restorer.restore_faces(test_image)
        
        assert isinstance(restored_image, Image.Image)
        assert restored_image.size == test_image.size
    
    @pytest.mark.asyncio
    async def test_restore_faces_no_faces(self, face_restorer):
        """Test face restoration when no faces are detected."""
        # Create image that won't have faces detected
        empty_image = Image.new('RGB', (100, 100), color='black')
        
        with patch.object(face_restorer, 'detect_faces', return_value=[]):
            restored_image = await face_restorer.restore_faces(empty_image)
            
            # Should return original image when no faces detected
            assert restored_image.size == empty_image.size


class TestImageEnhancer:
    """Test comprehensive image enhancement functionality."""
    
    @pytest.fixture
    def config(self):
        return ModelConfig(
            model_name="test_enhancer",
            device="cpu",
            batch_size=1
        )
    
    @pytest.fixture
    def enhancer(self, config):
        return ImageEnhancer(config)
    
    @pytest.fixture
    def test_image_path(self):
        # Create a temporary test image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image = Image.new('RGB', (256, 256), color='blue')
            image.save(tmp.name, 'PNG')
            yield tmp.name
        
        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    
    @pytest.mark.asyncio
    async def test_enhancer_initialization(self, enhancer):
        """Test image enhancer initialization."""
        await enhancer.initialize()
        
        assert enhancer.is_initialized is True
        assert enhancer.upscaler is not None
        assert enhancer.face_restorer is not None
    
    @pytest.mark.asyncio
    async def test_validate_image_valid(self, enhancer, test_image_path):
        """Test image validation with valid image."""
        result = await enhancer.validate_image(test_image_path)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_image_not_found(self, enhancer):
        """Test image validation with non-existent file."""
        with pytest.raises(ValidationError, match="Image file not found"):
            await enhancer.validate_image("nonexistent.png")
    
    @pytest.mark.asyncio
    async def test_validate_image_unsupported_format(self, enhancer):
        """Test image validation with unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"not an image")
            tmp.flush()
            
            try:
                with pytest.raises(ValidationError, match="Unsupported image format"):
                    await enhancer.validate_image(tmp.name)
            finally:
                os.unlink(tmp.name)
    
    @pytest.mark.asyncio
    async def test_upscale_image(self, enhancer, test_image_path):
        """Test image upscaling functionality."""
        await enhancer.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        output_path = await enhancer.upscale_image(test_image_path, scale_factor=2)
        
        assert os.path.exists(output_path)
        assert "upscaled" in output_path
        
        # Verify upscaled image
        with Image.open(output_path) as upscaled_img:
            with Image.open(test_image_path) as original_img:
                # Should be larger (mock implementation uses 4x scale)
                assert upscaled_img.size[0] >= original_img.size[0]
                assert upscaled_img.size[1] >= original_img.size[1]
        
        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)
    
    @pytest.mark.asyncio
    async def test_restore_faces(self, enhancer, test_image_path):
        """Test face restoration functionality."""
        await enhancer.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        output_path = await enhancer.restore_faces(test_image_path)
        
        assert os.path.exists(output_path)
        assert "face_restored" in output_path
        
        # Verify restored image exists and has correct format
        with Image.open(output_path) as restored_img:
            assert restored_img.mode == 'RGB'
        
        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)
    
    @pytest.mark.asyncio
    async def test_enhance_quality(self, enhancer, test_image_path):
        """Test general quality enhancement."""
        await enhancer.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        output_path = await enhancer.enhance_quality(test_image_path)
        
        assert os.path.exists(output_path)
        assert "quality_enhanced" in output_path
        
        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)
    
    @pytest.mark.asyncio
    async def test_denoise_image(self, enhancer, test_image_path):
        """Test image denoising functionality."""
        await enhancer.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        output_path = await enhancer.denoise_image(test_image_path)
        
        assert os.path.exists(output_path)
        assert "denoised" in output_path
        
        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)
    
    @pytest.mark.asyncio
    async def test_sharpen_image(self, enhancer, test_image_path):
        """Test image sharpening functionality."""
        await enhancer.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        output_path = await enhancer.sharpen_image(test_image_path)
        
        assert os.path.exists(output_path)
        assert "sharpened" in output_path
        
        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)
    
    @pytest.mark.asyncio
    async def test_enhance_content_upscale(self, enhancer, test_image_path):
        """Test enhance_content method with upscale type."""
        await enhancer.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        result = await enhancer.enhance_content(test_image_path, "upscale", scale_factor=2)
        
        assert result.status == GenerationStatus.COMPLETED
        assert len(result.content_paths) == 1
        assert os.path.exists(result.content_paths[0])
        assert result.model_used == "image_enhancer"
        assert result.metadata["enhancement_type"] == "upscale"
        
        # Cleanup
        if os.path.exists(result.content_paths[0]):
            os.unlink(result.content_paths[0])
    
    @pytest.mark.asyncio
    async def test_enhance_content_invalid_type(self, enhancer, test_image_path):
        """Test enhance_content with invalid enhancement type."""
        await enhancer.initialize()
        
        result = await enhancer.enhance_content(test_image_path, "invalid_type")
        
        assert result.status == GenerationStatus.FAILED
        assert "Unsupported enhancement type" in result.error_message
    
    @pytest.mark.asyncio
    async def test_generate_method(self, enhancer, test_image_path):
        """Test main generate method."""
        await enhancer.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        # Create mock request
        request = Mock()
        request.source_image_path = test_image_path
        request.enhancement_type = "quality_enhance"
        
        result = await enhancer.generate(request)
        
        assert result.status == GenerationStatus.COMPLETED
        assert len(result.content_paths) == 1
        
        # Cleanup
        if result.content_paths and os.path.exists(result.content_paths[0]):
            os.unlink(result.content_paths[0])
    
    @pytest.mark.asyncio
    async def test_generate_no_source_path(self, enhancer):
        """Test generate method without source path."""
        request = Mock()
        request.source_image_path = None
        
        result = await enhancer.generate(request)
        
        assert result.status == GenerationStatus.FAILED
        assert "Source image path is required" in result.error_message
    
    def test_get_capabilities(self, enhancer):
        """Test capabilities reporting."""
        capabilities = enhancer.get_capabilities()
        
        assert capabilities["model_name"] == "advanced_image_enhancer"
        assert "upscale" in capabilities["enhancement_types"]
        assert "face_restore" in capabilities["enhancement_types"]
        assert "quality_enhance" in capabilities["enhancement_types"]
        assert capabilities["max_upscale_factor"] == 4
        assert capabilities["features"]["real_esrgan_upscaling"] is True
        assert capabilities["features"]["gfpgan_face_restoration"] is True
    
    @pytest.mark.asyncio
    async def test_estimate_cost(self, enhancer):
        """Test cost estimation."""
        request = Mock()
        request.enhancement_type = "upscale"
        
        cost = await enhancer.estimate_cost(request)
        assert cost > 0
        assert isinstance(cost, float)
    
    @pytest.mark.asyncio
    async def test_estimate_time(self, enhancer):
        """Test time estimation."""
        request = Mock()
        request.enhancement_type = "face_restore"
        
        time_estimate = await enhancer.estimate_time(request)
        assert time_estimate > 0
        assert isinstance(time_estimate, float)


class TestImageEnhancementIntegration:
    """Integration tests for image enhancement functionality."""
    
    @pytest.fixture
    def config(self):
        return ModelConfig(
            model_name="integration_test_enhancer",
            device="cpu",
            batch_size=1
        )
    
    @pytest.mark.asyncio
    async def test_full_enhancement_workflow(self, config):
        """Test complete enhancement workflow."""
        enhancer = ImageEnhancer(config)
        
        # Create test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image = Image.new('RGB', (128, 128), color='green')
            test_image.save(tmp.name, 'PNG')
            test_image_path = tmp.name
        
        try:
            # Create output directory
            Path("./generated_content").mkdir(exist_ok=True)
            
            # Test initialization
            await enhancer.initialize()
            assert enhancer.is_initialized
            
            # Test validation
            is_valid = await enhancer.validate_image(test_image_path)
            assert is_valid
            
            # Test enhancement
            result = await enhancer.enhance_content(test_image_path, "quality_enhance")
            assert result.status == GenerationStatus.COMPLETED
            assert os.path.exists(result.content_paths[0])
            
            # Test quality metrics
            assert result.quality_metrics is not None
            assert result.quality_metrics.overall_score > 0
            
            # Cleanup
            if os.path.exists(result.content_paths[0]):
                os.unlink(result.content_paths[0])
                
        finally:
            # Cleanup test image
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, config):
        """Test performance benchmarks for enhancement operations."""
        enhancer = ImageEnhancer(config)
        await enhancer.initialize()
        
        # Create test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image = Image.new('RGB', (256, 256), color='red')
            test_image.save(tmp.name, 'PNG')
            test_image_path = tmp.name
        
        try:
            # Create output directory
            Path("./generated_content").mkdir(exist_ok=True)
            
            import time
            
            # Test upscaling performance
            start_time = time.time()
            result = await enhancer.enhance_content(test_image_path, "upscale")
            upscale_time = time.time() - start_time
            
            assert result.status == GenerationStatus.COMPLETED
            assert upscale_time < 30.0  # Should complete within 30 seconds
            
            # Cleanup
            if os.path.exists(result.content_paths[0]):
                os.unlink(result.content_paths[0])
                
        finally:
            # Cleanup test image
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)


if __name__ == "__main__":
    pytest.main([__file__])