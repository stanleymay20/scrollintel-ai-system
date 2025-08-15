"""
Integration tests for DALL-E 3 API integration.
Tests API communication, response handling, error scenarios, and rate limiting.
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from PIL import Image
import io
import base64
from datetime import datetime, timedelta

from scrollintel.engines.visual_generation.models.dalle3 import (
    DALLE3Model, 
    DALLE3Parameters, 
    RateLimiter
)
from scrollintel.engines.visual_generation.base import ImageGenerationRequest
from scrollintel.engines.visual_generation.config import VisualGenerationConfig
from scrollintel.engines.visual_generation.exceptions import (
    ModelError,
    RateLimitError,
    SafetyError,
    InvalidRequestError,
    APIConnectionError
)


@pytest.fixture
def config():
    """Test configuration."""
    config_data = {
        'models': {
            'dalle3': {
                'type': 'image',
                'api_key': 'test-api-key',
                'enabled': True,
                'parameters': {
                    'dalle3_rpm_limit': 5,
                    'dalle3_rpd_limit': 200
                }
            }
        }
    }
    return VisualGenerationConfig(config_data)


@pytest.fixture
def dalle3_model(config):
    """DALL-E 3 model instance."""
    return DALLE3Model(config)


@pytest.fixture
def sample_request():
    """Sample image generation request."""
    return ImageGenerationRequest(
        prompt="A beautiful sunset over mountains",
        user_id="test-user",
        resolution=(1024, 1024),
        style="photorealistic",
        quality="high",
        num_images=1
    )


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    response = Mock()
    response.data = [Mock()]
    response.data[0].url = "https://example.com/image.png"
    response.data[0].revised_prompt = "A beautiful sunset over mountains with enhanced details"
    return response


@pytest.fixture
def sample_image():
    """Sample PIL Image."""
    image = Image.new('RGB', (1024, 1024), color='red')
    return image


class TestDALLE3Model:
    """Test DALL-E 3 model functionality."""
    
    @pytest.mark.asyncio
    async def test_model_initialization(self, config):
        """Test model initialization."""
        model = DALLE3Model(config)
        
        assert model.model_name == "dall-e-3"
        assert model.rate_limiter.requests_per_minute == 5
        assert model.rate_limiter.requests_per_day == 200
        assert "1024x1024" in model.supported_sizes
        assert "standard" in model.quality_settings
        assert "vivid" in model.style_settings
    
    @pytest.mark.asyncio
    async def test_parameter_preparation(self, dalle3_model, sample_request):
        """Test parameter preparation for DALL-E 3."""
        params = await dalle3_model._prepare_parameters(sample_request)
        
        assert isinstance(params, DALLE3Parameters)
        assert params.model == "dall-e-3"
        assert params.size == "1024x1024"
        assert params.quality == "hd"  # high maps to hd
        assert params.style == "natural"  # photorealistic maps to natural
        assert params.response_format == "url"
    
    def test_resolution_mapping(self, dalle3_model):
        """Test resolution to size mapping."""
        # Square
        assert dalle3_model._map_resolution_to_size((1024, 1024)) == "1024x1024"
        assert dalle3_model._map_resolution_to_size((512, 512)) == "1024x1024"
        
        # Landscape
        assert dalle3_model._map_resolution_to_size((1792, 1024)) == "1792x1024"
        assert dalle3_model._map_resolution_to_size((1600, 900)) == "1792x1024"
        
        # Portrait
        assert dalle3_model._map_resolution_to_size((1024, 1792)) == "1024x1792"
        assert dalle3_model._map_resolution_to_size((900, 1600)) == "1024x1792"
    
    @pytest.mark.asyncio
    async def test_prompt_enhancement(self, dalle3_model):
        """Test prompt enhancement."""
        # Photorealistic style
        enhanced = await dalle3_model._enhance_prompt("a cat", "photorealistic")
        assert "photograph" in enhanced.lower()
        
        # Artistic style
        enhanced = await dalle3_model._enhance_prompt("a cat", "artistic")
        assert "artistic" in enhanced.lower()
        
        # Long prompt truncation
        long_prompt = "a" * 5000
        enhanced = await dalle3_model._enhance_prompt(long_prompt, "natural")
        assert len(enhanced) <= 4000
        assert enhanced.endswith("...")
    
    @pytest.mark.asyncio
    async def test_successful_generation(self, dalle3_model, sample_request, mock_openai_response, sample_image):
        """Test successful image generation."""
        with patch.object(dalle3_model, 'rate_limiter') as mock_limiter, \
             patch.object(dalle3_model, '_generate_with_retry', return_value=mock_openai_response) as mock_generate, \
             patch.object(dalle3_model, '_download_image', return_value=sample_image) as mock_download, \
             patch.object(dalle3_model, '_optimize_image', return_value=sample_image) as mock_optimize:
            
            mock_limiter.acquire = AsyncMock()
            
            result = await dalle3_model.generate(sample_request)
            
            assert result.model_used == "dall-e-3"
            assert len(result.images) == 1
            assert isinstance(result.images[0], Image.Image)
            assert result.generation_time > 0
            assert "revised_prompt" in result.metadata
            
            mock_limiter.acquire.assert_called_once()
            mock_generate.assert_called_once()
            mock_download.assert_called_once()
            mock_optimize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, dalle3_model, sample_request):
        """Test rate limit error handling."""
        import openai
        
        with patch.object(dalle3_model, 'rate_limiter') as mock_limiter, \
             patch.object(dalle3_model, '_generate_with_retry') as mock_generate:
            
            mock_limiter.acquire = AsyncMock()
            
            # Create a proper mock response for OpenAI exception
            mock_response = Mock()
            mock_response.request = Mock()
            mock_generate.side_effect = openai.RateLimitError("Rate limit exceeded", response=mock_response, body=None)
            
            with pytest.raises(RateLimitError):
                await dalle3_model.generate(sample_request)
    
    @pytest.mark.asyncio
    async def test_content_policy_violation(self, dalle3_model, sample_request):
        """Test content policy violation handling."""
        import openai
        
        with patch.object(dalle3_model, 'rate_limiter') as mock_limiter, \
             patch.object(dalle3_model, '_generate_with_retry') as mock_generate:
            
            mock_limiter.acquire = AsyncMock()
            
            # Create a proper mock response for OpenAI exception
            mock_response = Mock()
            mock_response.request = Mock()
            mock_generate.side_effect = openai.BadRequestError(
                "content_policy_violation", response=mock_response, body=None
            )
            
            with pytest.raises(SafetyError):
                await dalle3_model.generate(sample_request)
    
    @pytest.mark.asyncio
    async def test_api_connection_error(self, dalle3_model, sample_request):
        """Test API connection error handling."""
        import openai
        
        with patch.object(dalle3_model, 'rate_limiter') as mock_limiter, \
             patch.object(dalle3_model, '_generate_with_retry') as mock_generate:
            
            mock_limiter.acquire = AsyncMock()
            mock_generate.side_effect = openai.APIConnectionError("Connection failed", request=Mock())
            
            with pytest.raises(APIConnectionError):
                await dalle3_model.generate(sample_request)
    
    @pytest.mark.asyncio
    async def test_image_download(self, dalle3_model, sample_image):
        """Test image download from URL."""
        # Create mock response data
        img_buffer = io.BytesIO()
        sample_image.save(img_buffer, format='PNG')
        img_data = img_buffer.getvalue()
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=img_data)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await dalle3_model._download_image("https://example.com/image.png")
            
            assert isinstance(result, Image.Image)
            assert result.size == (1024, 1024)
    
    @pytest.mark.asyncio
    async def test_image_download_failure(self, dalle3_model):
        """Test image download failure handling."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 404
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(APIConnectionError):
                await dalle3_model._download_image("https://example.com/nonexistent.png")
    
    def test_base64_image_decode(self, dalle3_model, sample_image):
        """Test base64 image decoding."""
        # Create base64 encoded image
        img_buffer = io.BytesIO()
        sample_image.save(img_buffer, format='PNG')
        img_data = img_buffer.getvalue()
        b64_data = base64.b64encode(img_data).decode('utf-8')
        
        result = dalle3_model._decode_base64_image(b64_data)
        
        assert isinstance(result, Image.Image)
        assert result.size == (1024, 1024)
    
    def test_base64_decode_failure(self, dalle3_model):
        """Test base64 decode failure handling."""
        with pytest.raises(ModelError):
            dalle3_model._decode_base64_image("invalid_base64_data")
    
    @pytest.mark.asyncio
    async def test_image_optimization(self, dalle3_model, sample_request, sample_image):
        """Test image optimization."""
        # Test with different target resolution
        sample_request.resolution = (512, 512)
        
        result = await dalle3_model._optimize_image(sample_image, sample_request)
        
        assert isinstance(result, Image.Image)
        assert result.size == (512, 512)
        assert result.mode == 'RGB'
    
    @pytest.mark.asyncio
    async def test_image_quality_enhancement(self, dalle3_model, sample_image):
        """Test image quality enhancement."""
        result = await dalle3_model._enhance_image_quality(sample_image)
        
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size
        assert result.mode == sample_image.mode
    
    @pytest.mark.asyncio
    async def test_model_info(self, dalle3_model):
        """Test model information retrieval."""
        info = await dalle3_model.get_model_info()
        
        assert info["name"] == "dall-e-3"
        assert info["provider"] == "OpenAI"
        assert "supported_sizes" in info
        assert "supported_qualities" in info
        assert "supported_styles" in info
        assert "rate_limits" in info
    
    @pytest.mark.asyncio
    async def test_request_validation(self, dalle3_model, sample_request):
        """Test request validation."""
        # Valid request
        assert await dalle3_model.validate_request(sample_request) == True
        
        # Invalid prompt length
        sample_request.prompt = "a" * 5000
        with pytest.raises(InvalidRequestError):
            await dalle3_model.validate_request(sample_request)
        
        # Multiple images (should log warning but not fail)
        sample_request.prompt = "valid prompt"
        sample_request.num_images = 4
        assert await dalle3_model.validate_request(sample_request) == True


class TestRateLimiter:
    """Test rate limiter functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(requests_per_minute=5, requests_per_day=100)
        
        assert limiter.requests_per_minute == 5
        assert limiter.requests_per_day == 100
        assert len(limiter.minute_requests) == 0
        assert len(limiter.day_requests) == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        """Test rate limiter acquire."""
        limiter = RateLimiter(requests_per_minute=5, requests_per_day=100)
        
        # Should allow first request
        await limiter.acquire()
        assert len(limiter.minute_requests) == 1
        assert len(limiter.day_requests) == 1
    
    @pytest.mark.asyncio
    async def test_rate_limiter_minute_limit(self):
        """Test minute rate limit enforcement."""
        limiter = RateLimiter(requests_per_minute=2, requests_per_day=100)
        
        # Fill up minute limit
        await limiter.acquire()
        await limiter.acquire()
        
        # Next request should be delayed
        start_time = datetime.now()
        
        # Mock sleep to avoid actual waiting
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await limiter.acquire()
            mock_sleep.assert_called()
    
    @pytest.mark.asyncio
    async def test_rate_limiter_day_limit(self):
        """Test daily rate limit enforcement."""
        limiter = RateLimiter(requests_per_minute=100, requests_per_day=2)
        
        # Fill up daily limit
        await limiter.acquire()
        await limiter.acquire()
        
        # Next request should raise error
        with pytest.raises(RateLimitError):
            await limiter.acquire()
    
    def test_clean_old_requests(self):
        """Test cleaning of old request timestamps."""
        limiter = RateLimiter(requests_per_minute=5, requests_per_day=100)
        
        now = datetime.now()
        old_time = now - timedelta(minutes=2)
        
        # Add old requests
        limiter.minute_requests = [old_time, now]
        limiter.day_requests = [old_time, now]
        
        # Clean old requests
        limiter._clean_old_requests(now)
        
        # Only recent requests should remain
        assert len(limiter.minute_requests) == 1
        assert len(limiter.day_requests) == 2  # Day requests should still be there


class TestDALLE3Integration:
    """Integration tests for DALL-E 3 with external dependencies."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_generation_flow(self, dalle3_model, sample_request):
        """Test complete generation flow with mocked external calls."""
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].url = "https://example.com/image.png"
        mock_response.data[0].revised_prompt = "Enhanced prompt"
        
        sample_image = Image.new('RGB', (1024, 1024), color='blue')
        
        with patch.object(dalle3_model.client.images, 'generate', new_callable=AsyncMock) as mock_generate, \
             patch.object(dalle3_model, '_download_image', return_value=sample_image) as mock_download, \
             patch.object(dalle3_model.rate_limiter, 'acquire', new_callable=AsyncMock) as mock_acquire:
            
            mock_generate.return_value = mock_response
            
            result = await dalle3_model.generate(sample_request)
            
            # Verify all components were called
            mock_acquire.assert_called_once()
            mock_generate.assert_called_once()
            mock_download.assert_called_once_with("https://example.com/image.png")
            
            # Verify result
            assert result.model_used == "dall-e-3"
            assert len(result.images) == 1
            assert isinstance(result.images[0], Image.Image)
            assert result.metadata["revised_prompt"] == "Enhanced prompt"
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, dalle3_model, sample_request):
        """Test retry mechanism for transient failures."""
        import openai
        
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].url = "https://example.com/image.png"
        
        sample_image = Image.new('RGB', (1024, 1024), color='green')
        
        with patch.object(dalle3_model.client.images, 'generate', new_callable=AsyncMock) as mock_generate, \
             patch.object(dalle3_model, '_download_image', return_value=sample_image), \
             patch.object(dalle3_model.rate_limiter, 'acquire', new_callable=AsyncMock), \
             patch('asyncio.sleep', new_callable=AsyncMock):
            
            # First call fails, second succeeds
            mock_response_error = Mock()
            mock_response_error.request = Mock()
            mock_generate.side_effect = [
                openai.RateLimitError("Rate limit", response=mock_response_error, body=None),
                mock_response
            ]
            
            result = await dalle3_model.generate(sample_request)
            
            # Should have retried and succeeded
            assert mock_generate.call_count == 2
            assert result.model_used == "dall-e-3"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, dalle3_model):
        """Test handling of concurrent requests."""
        requests = [
            ImageGenerationRequest(
                prompt=f"Test prompt {i}",
                user_id="test-user",
                resolution=(1024, 1024),
                style="natural",
                quality="standard"
            )
            for i in range(3)
        ]
        
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].url = "https://example.com/image.png"
        
        sample_image = Image.new('RGB', (1024, 1024), color='yellow')
        
        with patch.object(dalle3_model.client.images, 'generate', new_callable=AsyncMock) as mock_generate, \
             patch.object(dalle3_model, '_download_image', return_value=sample_image), \
             patch.object(dalle3_model.rate_limiter, 'acquire', new_callable=AsyncMock):
            
            mock_generate.return_value = mock_response
            
            # Execute concurrent requests
            results = await asyncio.gather(*[
                dalle3_model.generate(req) for req in requests
            ])
            
            # All should succeed
            assert len(results) == 3
            assert all(r.model_used == "dall-e-3" for r in results)
            assert mock_generate.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])