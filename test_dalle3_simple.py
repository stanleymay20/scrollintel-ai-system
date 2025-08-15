"""
Simple test to verify DALL-E 3 integration works.
"""

import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
from PIL import Image

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from scrollintel.engines.visual_generation.models.dalle3 import DALLE3Model, DALLE3Parameters
from scrollintel.engines.visual_generation.base import ImageGenerationRequest
from scrollintel.engines.visual_generation.config import VisualGenerationConfig


async def test_dalle3_basic_functionality():
    """Test basic DALL-E 3 functionality."""
    print("Testing DALL-E 3 basic functionality...")
    
    # Create config with proper structure
    config_data = {
        'models': {
            'dalle3': {
                'type': 'image',
                'api_key': 'test-key',
                'enabled': True,
                'parameters': {
                    'dalle3_rpm_limit': 5,
                    'dalle3_rpd_limit': 200
                }
            }
        }
    }
    config = VisualGenerationConfig(config_data)
    
    # Create model
    model = DALLE3Model(config)
    
    # Test initialization
    assert model.model_name == "dall-e-3"
    assert model.rate_limiter.requests_per_minute == 5
    print("✓ Model initialization successful")
    
    # Test parameter preparation
    request = ImageGenerationRequest(
        prompt="A beautiful sunset",
        user_id="test-user",
        resolution=(1024, 1024),
        style="photorealistic",
        quality="high"
    )
    
    params = await model._prepare_parameters(request)
    assert isinstance(params, DALLE3Parameters)
    assert params.size == "1024x1024"
    assert params.quality == "hd"
    print("✓ Parameter preparation successful")
    
    # Test resolution mapping
    assert model._map_resolution_to_size((1024, 1024)) == "1024x1024"
    assert model._map_resolution_to_size((1792, 1024)) == "1792x1024"
    assert model._map_resolution_to_size((1024, 1792)) == "1024x1792"
    print("✓ Resolution mapping successful")
    
    # Test prompt enhancement
    enhanced = await model._enhance_prompt("a cat", "photorealistic")
    assert "photograph" in enhanced.lower()
    print("✓ Prompt enhancement successful")
    
    # Test model info
    info = await model.get_model_info()
    assert info["name"] == "dall-e-3"
    assert info["provider"] == "OpenAI"
    print("✓ Model info retrieval successful")
    
    # Test request validation
    assert await model.validate_request(request) == True
    print("✓ Request validation successful")
    
    print("All basic functionality tests passed! ✅")


async def test_dalle3_mocked_generation():
    """Test DALL-E 3 generation with mocked API calls."""
    print("\nTesting DALL-E 3 mocked generation...")
    
    config_data = {
        'models': {
            'dalle3': {
                'type': 'image',
                'api_key': 'test-key',
                'enabled': True,
                'parameters': {
                    'dalle3_rpm_limit': 5,
                    'dalle3_rpd_limit': 200
                }
            }
        }
    }
    config = VisualGenerationConfig(config_data)
    
    model = DALLE3Model(config)
    
    request = ImageGenerationRequest(
        prompt="A beautiful landscape",
        user_id="test-user",
        resolution=(1024, 1024),
        style="natural",
        quality="standard"
    )
    
    # Create mock response
    mock_response = Mock()
    mock_response.data = [Mock()]
    mock_response.data[0].url = "https://example.com/image.png"
    mock_response.data[0].revised_prompt = "A beautiful landscape with enhanced details"
    
    # Create sample image
    sample_image = Image.new('RGB', (1024, 1024), color='blue')
    
    # Mock the API calls
    with patch.object(model.rate_limiter, 'acquire', new_callable=AsyncMock) as mock_acquire, \
         patch.object(model, '_generate_with_retry', return_value=mock_response) as mock_generate, \
         patch.object(model, '_download_image', return_value=sample_image) as mock_download, \
         patch.object(model, '_optimize_image', return_value=sample_image) as mock_optimize:
        
        result = await model.generate(request)
        
        # Verify the result
        assert result.model_used == "dall-e-3"
        assert len(result.images) == 1
        assert isinstance(result.images[0], Image.Image)
        assert result.generation_time > 0
        assert "revised_prompt" in result.metadata
        
        # Verify mocks were called
        mock_acquire.assert_called_once()
        mock_generate.assert_called_once()
        mock_download.assert_called_once()
        mock_optimize.assert_called_once()
    
    print("✓ Mocked generation test successful")
    print("DALL-E 3 integration test completed successfully! ✅")


async def main():
    """Run all tests."""
    try:
        await test_dalle3_basic_functionality()
        await test_dalle3_mocked_generation()
        print("\n🎉 All DALL-E 3 integration tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)