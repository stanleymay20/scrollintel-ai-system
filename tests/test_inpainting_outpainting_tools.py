"""
Tests for inpainting and outpainting tools for object removal and image extension.
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
    InpaintingEngine,
    OutpaintingEngine,
    EditingToolsEngine
)
from scrollintel.engines.visual_generation.base import GenerationRequest, GenerationStatus
from scrollintel.engines.visual_generation.config import ModelConfig
from scrollintel.engines.visual_generation.exceptions import ModelError, ValidationError


class TestInpaintingEngine:
    """Test inpainting engine functionality."""
    
    @pytest.fixture
    def inpainting_engine(self):
        return InpaintingEngine()
    
    @pytest.fixture
    def test_image(self):
        # Create a test image with some content
        image = Image.new('RGB', (256, 256), color='white')
        # Add some colored regions
        pixels = image.load()
        for i in range(100, 150):
            for j in range(100, 150):
                pixels[i, j] = (255, 0, 0)  # Red square
        return image
    
    @pytest.mark.asyncio
    async def test_inpainting_initialization(self, inpainting_engine):
        """Test inpainting engine initialization."""
        await inpainting_engine.initialize()
        
        assert inpainting_engine.model is not None
        assert inpainting_engine.mask_generator is not None
        assert inpainting_engine.model["loaded"] is True
        assert inpainting_engine.mask_generator["loaded"] is True
        assert inpainting_engine.device in ["cuda", "cpu"]
    
    @pytest.mark.asyncio
    async def test_generate_mask_with_coordinates(self, inpainting_engine, test_image):
        """Test mask generation with provided coordinates."""
        await inpainting_engine.initialize()
        
        coordinates = (50, 50, 150, 150)  # x1, y1, x2, y2
        mask = await inpainting_engine.generate_mask(test_image, coordinates=coordinates)
        
        assert isinstance(mask, Image.Image)
        assert mask.mode == 'L'  # Grayscale mask
        assert mask.size == test_image.size
        
        # Check that mask has white region in specified coordinates
        mask_array = np.array(mask)
        assert np.any(mask_array[50:150, 50:150] == 255)  # White area to inpaint
    
    @pytest.mark.asyncio
    async def test_generate_mask_with_target_object(self, inpainting_engine, test_image):
        """Test mask generation with target object description."""
        await inpainting_engine.initialize()
        
        mask = await inpainting_engine.generate_mask(test_image, target_object="red square")
        
        assert isinstance(mask, Image.Image)
        assert mask.mode == 'L'
        assert mask.size == test_image.size
        
        # Should generate some mask (mock creates circular mask)
        mask_array = np.array(mask)
        assert np.any(mask_array > 0)  # Should have some white pixels
    
    @pytest.mark.asyncio
    async def test_inpaint_image(self, inpainting_engine, test_image):
        """Test image inpainting functionality."""
        await inpainting_engine.initialize()
        
        # Generate a mask
        coordinates = (100, 100, 150, 150)
        mask = await inpainting_engine.generate_mask(test_image, coordinates=coordinates)
        
        # Apply inpainting
        inpainted_image = await inpainting_engine.inpaint_image(test_image, mask)
        
        assert isinstance(inpainted_image, Image.Image)
        assert inpainted_image.size == test_image.size
        assert inpainted_image.mode == 'RGB'
    
    @pytest.mark.asyncio
    async def test_inpaint_with_prompt(self, inpainting_engine, test_image):
        """Test inpainting with prompt guidance."""
        await inpainting_engine.initialize()
        
        coordinates = (100, 100, 150, 150)
        mask = await inpainting_engine.generate_mask(test_image, coordinates=coordinates)
        
        # Apply inpainting with prompt
        inpainted_image = await inpainting_engine.inpaint_image(
            test_image, mask, prompt="blue flower"
        )
        
        assert isinstance(inpainted_image, Image.Image)
        assert inpainted_image.size == test_image.size


class TestOutpaintingEngine:
    """Test outpainting engine functionality."""
    
    @pytest.fixture
    def outpainting_engine(self):
        return OutpaintingEngine()
    
    @pytest.fixture
    def test_image(self):
        # Create a test image
        image = Image.new('RGB', (128, 128), color='blue')
        return image
    
    @pytest.mark.asyncio
    async def test_outpainting_initialization(self, outpainting_engine):
        """Test outpainting engine initialization."""
        await outpainting_engine.initialize()
        
        assert outpainting_engine.model is not None
        assert outpainting_engine.model["loaded"] is True
        assert outpainting_engine.device in ["cuda", "cpu"]
    
    @pytest.mark.asyncio
    async def test_extend_image_right(self, outpainting_engine, test_image):
        """Test extending image to the right."""
        await outpainting_engine.initialize()
        
        original_size = test_image.size
        extension_pixels = 64
        
        extended_image = await outpainting_engine.extend_image(
            test_image, "right", extension_pixels
        )
        
        assert isinstance(extended_image, Image.Image)
        assert extended_image.size[0] == original_size[0] + extension_pixels
        assert extended_image.size[1] == original_size[1]
    
    @pytest.mark.asyncio
    async def test_extend_image_left(self, outpainting_engine, test_image):
        """Test extending image to the left."""
        await outpainting_engine.initialize()
        
        original_size = test_image.size
        extension_pixels = 64
        
        extended_image = await outpainting_engine.extend_image(
            test_image, "left", extension_pixels
        )
        
        assert isinstance(extended_image, Image.Image)
        assert extended_image.size[0] == original_size[0] + extension_pixels
        assert extended_image.size[1] == original_size[1]
    
    @pytest.mark.asyncio
    async def test_extend_image_all_directions(self, outpainting_engine, test_image):
        """Test extending image in all directions."""
        await outpainting_engine.initialize()
        
        original_size = test_image.size
        extension_pixels = 32
        
        extended_image = await outpainting_engine.extend_image(
            test_image, "all", extension_pixels
        )
        
        assert isinstance(extended_image, Image.Image)
        assert extended_image.size[0] == original_size[0] + 2 * extension_pixels
        assert extended_image.size[1] == original_size[1] + 2 * extension_pixels
    
    @pytest.mark.asyncio
    async def test_extend_image_invalid_direction(self, outpainting_engine, test_image):
        """Test extending image with invalid direction."""
        await outpainting_engine.initialize()
        
        with pytest.raises(ValidationError, match="Invalid direction"):
            await outpainting_engine.extend_image(test_image, "invalid", 64)
    
    @pytest.mark.asyncio
    async def test_extend_image_with_prompt(self, outpainting_engine, test_image):
        """Test extending image with prompt guidance."""
        await outpainting_engine.initialize()
        
        extended_image = await outpainting_engine.extend_image(
            test_image, "bottom", 64, prompt="grass field"
        )
        
        assert isinstance(extended_image, Image.Image)
        assert extended_image.size[1] == test_image.size[1] + 64


class TestEditingToolsEngine:
    """Test comprehensive editing tools engine."""
    
    @pytest.fixture
    def config(self):
        return ModelConfig(
            model_name="test_editing_tools",
            device="cpu",
            batch_size=1
        )
    
    @pytest.fixture
    def editing_engine(self, config):
        return EditingToolsEngine(config)
    
    @pytest.fixture
    def test_image_path(self):
        # Create a temporary test image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image = Image.new('RGB', (200, 200), color='white')
            # Add a red square to remove
            pixels = image.load()
            for i in range(80, 120):
                for j in range(80, 120):
                    pixels[i, j] = (255, 0, 0)
            image.save(tmp.name, 'PNG')
            yield tmp.name
        
        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    
    @pytest.mark.asyncio
    async def test_editing_tools_initialization(self, editing_engine):
        """Test editing tools engine initialization."""
        await editing_engine.initialize()
        
        assert editing_engine.is_initialized is True
        assert editing_engine.inpainting_engine is not None
        assert editing_engine.outpainting_engine is not None
    
    @pytest.mark.asyncio
    async def test_remove_object_with_coordinates(self, editing_engine, test_image_path):
        """Test object removal with coordinates."""
        await editing_engine.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        coordinates = (80, 80, 120, 120)  # Red square coordinates
        output_path = await editing_engine.remove_object(
            test_image_path, coordinates=coordinates
        )
        
        assert os.path.exists(output_path)
        assert "object_removed" in output_path
        
        # Verify output image
        with Image.open(output_path) as result_img:
            assert result_img.mode == 'RGB'
            assert result_img.size == (200, 200)
        
        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)
    
    @pytest.mark.asyncio
    async def test_remove_object_with_description(self, editing_engine, test_image_path):
        """Test object removal with object description."""
        await editing_engine.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        output_path = await editing_engine.remove_object(
            test_image_path, target_object="red square"
        )
        
        assert os.path.exists(output_path)
        assert "object_removed" in output_path
        
        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)
    
    @pytest.mark.asyncio
    async def test_replace_object(self, editing_engine, test_image_path):
        """Test object replacement functionality."""
        await editing_engine.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        output_path = await editing_engine.replace_object(
            test_image_path, 
            target_object="red square",
            replacement_prompt="blue circle"
        )
        
        assert os.path.exists(output_path)
        assert "object_replaced" in output_path
        
        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)
    
    @pytest.mark.asyncio
    async def test_extend_image(self, editing_engine, test_image_path):
        """Test image extension functionality."""
        await editing_engine.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        output_path = await editing_engine.extend_image(
            test_image_path, "right", 100
        )
        
        assert os.path.exists(output_path)
        assert "extended_right" in output_path
        
        # Verify extended image
        with Image.open(output_path) as extended_img:
            assert extended_img.size[0] > 200  # Should be wider
            assert extended_img.size[1] == 200  # Height unchanged
        
        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)
    
    @pytest.mark.asyncio
    async def test_generate_remove_object(self, editing_engine, test_image_path):
        """Test generate method for object removal."""
        await editing_engine.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        # Create mock request
        request = Mock()
        request.source_image_path = test_image_path
        request.editing_type = "remove_object"
        request.target_object = "red square"
        request.coordinates = None
        
        result = await editing_engine.generate(request)
        
        assert result.status == GenerationStatus.COMPLETED
        assert len(result.content_paths) == 1
        assert os.path.exists(result.content_paths[0])
        assert result.model_used == "editing_tools_engine"
        assert result.metadata["editing_type"] == "remove_object"
        
        # Cleanup
        if os.path.exists(result.content_paths[0]):
            os.unlink(result.content_paths[0])
    
    @pytest.mark.asyncio
    async def test_generate_replace_object(self, editing_engine, test_image_path):
        """Test generate method for object replacement."""
        await editing_engine.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        # Create mock request
        request = Mock()
        request.source_image_path = test_image_path
        request.editing_type = "replace_object"
        request.target_object = "red square"
        request.replacement_prompt = "green circle"
        request.coordinates = None
        
        result = await editing_engine.generate(request)
        
        assert result.status == GenerationStatus.COMPLETED
        assert len(result.content_paths) == 1
        assert result.metadata["editing_type"] == "replace_object"
        
        # Cleanup
        if os.path.exists(result.content_paths[0]):
            os.unlink(result.content_paths[0])
    
    @pytest.mark.asyncio
    async def test_generate_extend_image(self, editing_engine, test_image_path):
        """Test generate method for image extension."""
        await editing_engine.initialize()
        
        # Create output directory
        Path("./generated_content").mkdir(exist_ok=True)
        
        # Create mock request
        request = Mock()
        request.source_image_path = test_image_path
        request.editing_type = "extend_image"
        request.direction = "bottom"
        request.extension_pixels = 128
        request.prompt = "grass field"
        
        result = await editing_engine.generate(request)
        
        assert result.status == GenerationStatus.COMPLETED
        assert len(result.content_paths) == 1
        assert result.metadata["editing_type"] == "extend_image"
        
        # Cleanup
        if os.path.exists(result.content_paths[0]):
            os.unlink(result.content_paths[0])
    
    @pytest.mark.asyncio
    async def test_generate_no_source_path(self, editing_engine):
        """Test generate method without source path."""
        request = Mock()
        request.source_image_path = None
        request.editing_type = "remove_object"
        
        result = await editing_engine.generate(request)
        
        assert result.status == GenerationStatus.FAILED
        assert "Source image path is required" in result.error_message
    
    @pytest.mark.asyncio
    async def test_generate_invalid_editing_type(self, editing_engine, test_image_path):
        """Test generate method with invalid editing type."""
        request = Mock()
        request.source_image_path = test_image_path
        request.editing_type = "invalid_type"
        
        result = await editing_engine.generate(request)
        
        assert result.status == GenerationStatus.FAILED
        assert "Unsupported editing type" in result.error_message
    
    def test_get_capabilities(self, editing_engine):
        """Test capabilities reporting."""
        capabilities = editing_engine.get_capabilities()
        
        assert capabilities["model_name"] == "editing_tools_engine"
        assert "remove_object" in capabilities["editing_types"]
        assert "replace_object" in capabilities["editing_types"]
        assert "extend_image" in capabilities["editing_types"]
        assert capabilities["features"]["automatic_mask_generation"] is True
        assert capabilities["features"]["context_aware_inpainting"] is True
        assert capabilities["features"]["seamless_outpainting"] is True
        assert "left" in capabilities["supported_directions"]
        assert "right" in capabilities["supported_directions"]
        assert capabilities["max_extension_pixels"] == 1024
    
    @pytest.mark.asyncio
    async def test_estimate_cost(self, editing_engine):
        """Test cost estimation for different editing types."""
        # Test remove object cost
        request = Mock()
        request.editing_type = "remove_object"
        cost = await editing_engine.estimate_cost(request)
        assert cost > 0
        
        # Test replace object cost (should be higher)
        request.editing_type = "replace_object"
        replace_cost = await editing_engine.estimate_cost(request)
        assert replace_cost > cost
    
    @pytest.mark.asyncio
    async def test_estimate_time(self, editing_engine):
        """Test time estimation for different editing types."""
        # Test extend image time
        request = Mock()
        request.editing_type = "extend_image"
        time_estimate = await editing_engine.estimate_time(request)
        assert time_estimate > 0
        
        # Test replace object time (should be higher)
        request.editing_type = "replace_object"
        replace_time = await editing_engine.estimate_time(request)
        assert replace_time > time_estimate


class TestEditingToolsIntegration:
    """Integration tests for editing tools functionality."""
    
    @pytest.fixture
    def config(self):
        return ModelConfig(
            model_name="integration_editing_tools",
            device="cpu",
            batch_size=1
        )
    
    @pytest.mark.asyncio
    async def test_full_editing_workflow(self, config):
        """Test complete editing workflow."""
        editing_engine = EditingToolsEngine(config)
        
        # Create test image with object to edit
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image = Image.new('RGB', (256, 256), color='white')
            # Add colored regions
            pixels = test_image.load()
            for i in range(100, 150):
                for j in range(100, 150):
                    pixels[i, j] = (255, 0, 0)  # Red square
            test_image.save(tmp.name, 'PNG')
            test_image_path = tmp.name
        
        try:
            # Create output directory
            Path("./generated_content").mkdir(exist_ok=True)
            
            # Test initialization
            await editing_engine.initialize()
            assert editing_engine.is_initialized
            
            # Test object removal
            remove_result = await editing_engine.remove_object(
                test_image_path, coordinates=(100, 100, 150, 150)
            )
            assert os.path.exists(remove_result)
            
            # Test object replacement
            replace_result = await editing_engine.replace_object(
                test_image_path, "red square", "blue circle"
            )
            assert os.path.exists(replace_result)
            
            # Test image extension
            extend_result = await editing_engine.extend_image(
                test_image_path, "right", 128
            )
            assert os.path.exists(extend_result)
            
            # Verify extended image dimensions
            with Image.open(extend_result) as extended_img:
                assert extended_img.size[0] > 256
                assert extended_img.size[1] == 256
            
            # Cleanup
            for result_path in [remove_result, replace_result, extend_result]:
                if os.path.exists(result_path):
                    os.unlink(result_path)
                    
        finally:
            # Cleanup test image
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)
    
    @pytest.mark.asyncio
    async def test_editing_accuracy_validation(self, config):
        """Test editing accuracy and quality validation."""
        editing_engine = EditingToolsEngine(config)
        await editing_engine.initialize()
        
        # Create test image with specific pattern
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image = Image.new('RGB', (200, 200), color='blue')
            # Add pattern to remove
            pixels = test_image.load()
            for i in range(90, 110):
                for j in range(90, 110):
                    pixels[i, j] = (255, 255, 0)  # Yellow square
            test_image.save(tmp.name, 'PNG')
            test_image_path = tmp.name
        
        try:
            # Create output directory
            Path("./generated_content").mkdir(exist_ok=True)
            
            # Test removal accuracy
            result_path = await editing_engine.remove_object(
                test_image_path, coordinates=(90, 90, 110, 110)
            )
            
            # Verify result exists and has correct format
            assert os.path.exists(result_path)
            
            with Image.open(result_path) as result_img:
                assert result_img.mode == 'RGB'
                assert result_img.size == (200, 200)
                
                # Check that the area has been modified (inpainted)
                result_pixels = result_img.load()
                original_pixels = test_image.load()
                
                # At least some pixels in the target area should be different
                differences = 0
                for i in range(90, 110):
                    for j in range(90, 110):
                        if result_pixels[i, j] != original_pixels[i, j]:
                            differences += 1
                
                assert differences > 0  # Should have some changes in target area
            
            # Cleanup
            if os.path.exists(result_path):
                os.unlink(result_path)
                
        finally:
            # Cleanup test image
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)


if __name__ == "__main__":
    pytest.main([__file__])