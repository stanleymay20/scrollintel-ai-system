"""
Unit tests for Midjourney model integration.
Tests Discord bot API integration, job queuing, status polling, and error handling.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from PIL import Image
import io

from scrollintel.engines.visual_generation.models.midjourney import (
    MidjourneyModel,
    MidjourneyJob,
    MidjourneyJobStatus,
    MidjourneyParameters,
    MidjourneyPromptFormatter,
    MidjourneyJobQueue
)
from scrollintel.engines.visual_generation.base import ImageGenerationRequest
from scrollintel.engines.visual_generation.exceptions import (
    ModelError,
    RateLimitError,
    SafetyError,
    ValidationError,
    APIConnectionError,
    TimeoutError
)
from scrollintel.engines.visual_generation.config import VisualGenerationConfig


class TestMidjourneyPromptFormatter:
    """Test Midjourney prompt formatting functionality."""
    
    def setup_method(self):
        self.formatter = MidjourneyPromptFormatter()
    
    def test_basic_prompt_formatting(self):
        """Test basic prompt formatting with parameters."""
        params = MidjourneyParameters(
            aspect_ratio="16:9",
            quality="2",
            stylize=200,
            chaos=25
        )
        
        prompt = "A beautiful landscape"
        formatted = self.formatter.format_prompt(prompt, params)
        
        assert "A beautiful landscape" in formatted
        assert "--ar 16:9" in formatted
        assert "--q 2" in formatted
        assert "--stylize 200" in formatted
        assert "--chaos 25" in formatted
    
    def test_default_parameters_not_included(self):
        """Test that default parameters are not included in formatted prompt."""
        params = MidjourneyParameters()  # All defaults
        
        prompt = "A simple test"
        formatted = self.formatter.format_prompt(prompt, params)
        
        # Should only contain the prompt, no parameters
        assert formatted == "A simple test"
    
    def test_style_enhancement(self):
        """Test style-specific prompt enhancement."""
        prompt = "A portrait"
        
        enhanced = self.formatter.enhance_prompt_for_style(prompt, "photorealistic")
        assert "photorealistic" in enhanced.lower()
        assert "hyperrealistic" in enhanced.lower()
        
        enhanced = self.formatter.enhance_prompt_for_style(prompt, "artistic")
        assert "artistic" in enhanced.lower()
        assert "creative" in enhanced.lower()
    
    def test_prompt_validation_success(self):
        """Test successful prompt validation."""
        valid_prompt = "A beautiful sunset over mountains"
        assert self.formatter.validate_prompt(valid_prompt) is True
    
    def test_prompt_validation_too_short(self):
        """Test prompt validation with too short prompt."""
        with pytest.raises(ValidationError, match="at least 3 characters"):
            self.formatter.validate_prompt("hi")
    
    def test_prompt_validation_too_long(self):
        """Test prompt validation with too long prompt."""
        long_prompt = "x" * 4001
        with pytest.raises(ValidationError, match="too long"):
            self.formatter.validate_prompt(long_prompt)
    
    def test_prompt_validation_unsafe_content(self):
        """Test prompt validation with unsafe content."""
        with pytest.raises(SafetyError, match="unsafe content"):
            self.formatter.validate_prompt("Create nsfw content")
    
    def test_parameter_string_building(self):
        """Test parameter string building."""
        params = MidjourneyParameters(
            aspect_ratio="4:3",
            quality="0.5",
            stylize=500,
            seed=12345,
            tile=True,
            no_text=True
        )
        
        param_string = self.formatter._build_parameter_string(params)
        
        assert "--ar 4:3" in param_string
        assert "--q 0.5" in param_string
        assert "--stylize 500" in param_string
        assert "--seed 12345" in param_string
        assert "--tile" in param_string
        assert "--no text" in param_string


class TestMidjourneyJobQueue:
    """Test Midjourney job queue functionality."""
    
    def setup_method(self):
        self.queue = MidjourneyJobQueue(max_concurrent_jobs=2, poll_interval=0.1)
    
    @pytest.mark.asyncio
    async def test_job_submission(self):
        """Test job submission to queue."""
        job = MidjourneyJob(job_id="test_job_1", prompt="Test prompt")
        
        job_id = await self.queue.submit_job(job)
        assert job_id == "test_job_1"
        assert job_id in self.queue.jobs
    
    @pytest.mark.asyncio
    async def test_job_status_retrieval(self):
        """Test job status retrieval."""
        job = MidjourneyJob(job_id="test_job_2", prompt="Test prompt")
        await self.queue.submit_job(job)
        
        retrieved_job = await self.queue.get_job_status("test_job_2")
        assert retrieved_job is not None
        assert retrieved_job.job_id == "test_job_2"
        assert retrieved_job.prompt == "Test prompt"
    
    @pytest.mark.asyncio
    async def test_job_cancellation(self):
        """Test job cancellation."""
        job = MidjourneyJob(job_id="test_job_3", prompt="Test prompt")
        await self.queue.submit_job(job)
        
        cancelled = await self.queue.cancel_job("test_job_3")
        assert cancelled is True
        
        retrieved_job = await self.queue.get_job_status("test_job_3")
        assert retrieved_job.status == MidjourneyJobStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_concurrent_job_limit(self):
        """Test that concurrent job limit is enforced."""
        # Submit 3 jobs to a queue with max 2 concurrent
        jobs = [
            MidjourneyJob(job_id=f"test_job_{i}", prompt=f"Test prompt {i}")
            for i in range(3)
        ]
        
        for job in jobs:
            await self.queue.submit_job(job)
        
        # Wait a bit for queue processing
        await asyncio.sleep(0.2)
        
        # Should have at most 2 active jobs
        assert len(self.queue.active_jobs) <= 2
    
    @pytest.mark.asyncio
    async def test_queue_cleanup(self):
        """Test queue cleanup on shutdown."""
        job = MidjourneyJob(job_id="test_job_4", prompt="Test prompt")
        await self.queue.submit_job(job)
        
        await self.queue.shutdown()
        assert self.queue._shutdown is True


class TestMidjourneyModel:
    """Test Midjourney model integration."""
    
    def setup_method(self):
        # Mock configuration
        self.mock_config = Mock(spec=VisualGenerationConfig)
        self.mock_config.get_model_config.return_value = Mock(
            api_key="test_bot_token",
            parameters={
                'server_id': '123456789',
                'channel_id': '987654321',
                'application_id': '111222333',
                'max_concurrent_jobs': 3,
                'poll_interval': 1.0,
                'max_rpm': 10,
                'max_retries': 3,
                'generation_timeout': 60
            }
        )
        
        self.model = MidjourneyModel(self.mock_config)
    
    def test_model_initialization(self):
        """Test model initialization with configuration."""
        assert self.model.model_name == "midjourney"
        assert self.model.bot_token == "test_bot_token"
        assert self.model.server_id == '123456789'
        assert self.model.channel_id == '987654321'
        assert self.model.max_requests_per_minute == 10
    
    def test_model_initialization_missing_config(self):
        """Test model initialization with missing configuration."""
        mock_config = Mock(spec=VisualGenerationConfig)
        mock_config.get_model_config.return_value = None
        
        with pytest.raises(ModelError, match="configuration not found"):
            MidjourneyModel(mock_config)
    
    def test_model_initialization_missing_credentials(self):
        """Test model initialization with missing credentials."""
        mock_config = Mock(spec=VisualGenerationConfig)
        mock_config.get_model_config.return_value = Mock(
            api_key=None,
            parameters={'server_id': '123', 'channel_id': '456'}
        )
        
        with pytest.raises(ModelError, match="Missing required"):
            MidjourneyModel(mock_config)
    
    def test_aspect_ratio_calculation(self):
        """Test aspect ratio calculation from resolution."""
        # Square
        assert self.model._calculate_aspect_ratio((1024, 1024)) == "1:1"
        
        # 16:9
        assert self.model._calculate_aspect_ratio((1920, 1080)) == "16:9"
        
        # 9:16
        assert self.model._calculate_aspect_ratio((1080, 1920)) == "9:16"
        
        # 4:3
        assert self.model._calculate_aspect_ratio((1600, 1200)) == "4:3"
        
        # Custom ratio
        assert self.model._calculate_aspect_ratio((1500, 1000)) == "3:2"
    
    @pytest.mark.asyncio
    async def test_parameter_preparation(self):
        """Test parameter preparation from request."""
        request = ImageGenerationRequest(
            prompt="Test prompt",
            user_id="test_user",
            resolution=(1920, 1080),
            quality="high",
            style="photorealistic",
            seed=12345
        )
        
        params = await self.model._prepare_parameters(request)
        
        assert params.aspect_ratio == "16:9"
        assert params.quality == "1"
        assert params.stylize == 50  # Photorealistic style
        assert params.style == "raw"
        assert params.seed == 12345
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self):
        """Test rate limit enforcement."""
        # Fill up the rate limit
        for _ in range(self.model.max_requests_per_minute):
            self.model.request_timestamps.append(datetime.now())
        
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            await self.model._check_rate_limit()
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_discord_api_submission_success(self, mock_post):
        """Test successful Discord API submission."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 204
        mock_post.return_value.__aenter__.return_value = mock_response
        
        job = MidjourneyJob(job_id="test_job", prompt="Test prompt")
        params = MidjourneyParameters()
        
        await self.model._submit_to_discord_api(job, params)
        
        assert job.status == MidjourneyJobStatus.QUEUED
        mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_discord_api_submission_failure(self, mock_post):
        """Test Discord API submission failure."""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text.return_value = "Bad Request"
        mock_post.return_value.__aenter__.return_value = mock_response
        
        job = MidjourneyJob(job_id="test_job", prompt="Test prompt")
        params = MidjourneyParameters()
        
        with pytest.raises(APIConnectionError, match="Discord API error"):
            await self.model._submit_to_discord_api(job, params)
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_image_download_success(self, mock_get):
        """Test successful image download."""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = img_bytes.getvalue()
        mock_get.return_value.__aenter__.return_value = mock_response
        
        downloaded_image = await self.model._download_image("https://example.com/test.png")
        
        assert isinstance(downloaded_image, Image.Image)
        assert downloaded_image.size == (100, 100)
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_image_download_failure(self, mock_get):
        """Test image download failure."""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_get.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(APIConnectionError, match="Failed to download"):
            await self.model._download_image("https://example.com/nonexistent.png")
    
    @pytest.mark.asyncio
    async def test_image_post_processing(self):
        """Test image post-processing."""
        # Create a test image
        test_image = Image.new('RGB', (512, 512), color='blue')
        
        request = ImageGenerationRequest(
            prompt="Test",
            user_id="test_user",
            resolution=(1024, 1024),
            quality="high"
        )
        
        processed_image = await self.model._post_process_image(test_image, request)
        
        assert isinstance(processed_image, Image.Image)
        assert processed_image.size == (1024, 1024)  # Should be resized
        assert processed_image.mode == 'RGB'
    
    @pytest.mark.asyncio
    async def test_request_validation_success(self):
        """Test successful request validation."""
        request = ImageGenerationRequest(
            prompt="A beautiful landscape with mountains and trees",
            user_id="test_user",
            resolution=(1024, 1024)
        )
        
        is_valid = await self.model.validate_request(request)
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_request_validation_invalid_prompt(self):
        """Test request validation with invalid prompt."""
        request = ImageGenerationRequest(
            prompt="xx",  # Too short
            user_id="test_user",
            resolution=(1024, 1024)
        )
        
        with pytest.raises(ValidationError):
            await self.model.validate_request(request)
    
    @pytest.mark.asyncio
    async def test_get_model_info(self):
        """Test model info retrieval."""
        info = await self.model.get_model_info()
        
        assert info["name"] == "midjourney"
        assert info["provider"] == "Midjourney"
        assert info["version"] == "6"
        assert "1:1" in info["supported_aspects"]
        assert "16:9" in info["supported_aspects"]
        assert info["max_prompt_length"] == 4000
        assert "Job queuing" in info["features"]
    
    @pytest.mark.asyncio
    @patch.object(MidjourneyModel, '_submit_job_with_retry')
    @patch.object(MidjourneyModel, '_wait_for_completion')
    @patch.object(MidjourneyModel, '_process_job_result')
    async def test_generate_success(self, mock_process, mock_wait, mock_submit):
        """Test successful image generation."""
        # Mock the generation pipeline
        test_job = MidjourneyJob(
            job_id="test_job",
            prompt="Test prompt",
            status=MidjourneyJobStatus.COMPLETED
        )
        
        test_image = Image.new('RGB', (1024, 1024), color='green')
        
        mock_submit.return_value = test_job
        mock_wait.return_value = test_job
        mock_process.return_value = [test_image]
        
        request = ImageGenerationRequest(
            prompt="A test image",
            user_id="test_user",
            resolution=(1024, 1024)
        )
        
        result = await self.model.generate(request)
        
        assert len(result.images) == 1
        assert result.model_used == "midjourney"
        assert result.original_prompt == "A test image"
        assert result.generation_time > 0
    
    @pytest.mark.asyncio
    @patch.object(MidjourneyModel, '_submit_job_with_retry')
    async def test_generate_submission_failure(self, mock_submit):
        """Test generation failure during job submission."""
        mock_submit.side_effect = APIConnectionError("Connection failed")
        
        request = ImageGenerationRequest(
            prompt="A test image",
            user_id="test_user",
            resolution=(1024, 1024)
        )
        
        with pytest.raises(APIConnectionError):
            await self.model.generate(request)
    
    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self):
        """Test timeout during job completion wait."""
        # Create a job that never completes
        job = MidjourneyJob(job_id="timeout_job", prompt="Test")
        await self.model.job_queue.submit_job(job)
        
        # Set a very short timeout for testing
        original_timeout = self.model.generation_timeout
        self.model.generation_timeout = 0.1
        
        try:
            with pytest.raises(TimeoutError, match="timed out"):
                await self.model._wait_for_completion("timeout_job")
        finally:
            self.model.generation_timeout = original_timeout
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test model cleanup."""
        await self.model.cleanup()
        assert self.model.job_queue._shutdown is True


class TestMidjourneyJob:
    """Test MidjourneyJob data class."""
    
    def test_job_creation(self):
        """Test job creation with default values."""
        job = MidjourneyJob(job_id="test_job", prompt="Test prompt")
        
        assert job.job_id == "test_job"
        assert job.prompt == "Test prompt"
        assert job.status == MidjourneyJobStatus.PENDING
        assert job.progress == 0.0
        assert job.retry_count == 0
        assert isinstance(job.created_at, datetime)
        assert job.image_urls == []
    
    def test_job_with_metadata(self):
        """Test job creation with metadata."""
        metadata = {"user_id": "test_user", "request_id": "req_123"}
        job = MidjourneyJob(
            job_id="test_job",
            prompt="Test prompt",
            metadata=metadata
        )
        
        assert job.metadata == metadata
        assert job.metadata["user_id"] == "test_user"


class TestMidjourneyParameters:
    """Test MidjourneyParameters data class."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = MidjourneyParameters()
        
        assert params.aspect_ratio == "1:1"
        assert params.quality == "1"
        assert params.stylize == 100
        assert params.chaos == 0
        assert params.version == "6"
        assert params.style is None
        assert params.seed is None
        assert params.tile is False
        assert params.no_text is False
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = MidjourneyParameters(
            aspect_ratio="16:9",
            quality="2",
            stylize=500,
            chaos=50,
            style="raw",
            seed=12345,
            tile=True
        )
        
        assert params.aspect_ratio == "16:9"
        assert params.quality == "2"
        assert params.stylize == 500
        assert params.chaos == 50
        assert params.style == "raw"
        assert params.seed == 12345
        assert params.tile is True


@pytest.mark.asyncio
class TestMidjourneyIntegration:
    """Integration tests for Midjourney model."""
    
    async def test_full_generation_pipeline_mock(self):
        """Test the full generation pipeline with mocked external calls."""
        # This test would require extensive mocking of Discord API
        # and is more suitable for integration testing
        pass
    
    async def test_error_handling_chain(self):
        """Test error handling through the generation chain."""
        # Test various error scenarios and ensure proper error propagation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])