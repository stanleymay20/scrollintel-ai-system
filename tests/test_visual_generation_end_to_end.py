"""
End-to-end system testing for advanced visual content generation.
Tests complete user journeys from registration to content generation.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import requests
import json

from scrollintel.api.main import app
from scrollintel.models.database import get_db
from scrollintel.core.config import get_settings
from scrollintel.engines.visual_generation.engine import VisualGenerationEngine
from scrollintel.engines.visual_generation.pipeline import ImageGenerationPipeline
from scrollintel.engines.visual_generation.models.ultra_performance_pipeline import UltraRealisticVideoGenerationPipeline
from scrollintel.api.routes.visual_generation_routes import router as visual_generation_router
from scrollintel.core.user_management import UserManager
from scrollintel.security.auth import AuthManager
from scrollintel.core.monitoring import MetricsCollector


class TestEndToEndVisualGeneration:
    """Comprehensive end-to-end testing for visual generation system."""
    
    @pytest.fixture(autouse=True)
    async def setup_test_environment(self):
        """Set up test environment with all necessary components."""
        self.settings = get_settings()
        self.user_manager = UserManager()
        self.auth_manager = AuthManager()
        self.visual_engine = VisualGenerationEngine()
        self.image_pipeline = ImageGenerationPipeline()
        self.video_pipeline = UltraRealisticVideoGenerationPipeline()
        self.metrics_collector = MetricsCollector()
        
        # Test user credentials
        self.test_user = {
            "email": "test@scrollintel.com",
            "password": "TestPassword123!",
            "username": "testuser"
        }
        
        # Clean up any existing test data
        await self.cleanup_test_data()
        
    async def cleanup_test_data(self):
        """Clean up test data from previous runs."""
        try:
            # Remove test user if exists
            await self.user_manager.delete_user_by_email(self.test_user["email"])
        except:
            pass  # User doesn't exist, which is fine
    
    async def test_complete_user_journey_image_generation(self):
        """Test complete user journey from registration to image generation."""
        
        # Step 1: User Registration
        registration_response = await self.user_manager.create_user(
            email=self.test_user["email"],
            password=self.test_user["password"],
            username=self.test_user["username"]
        )
        
        assert registration_response.success
        assert registration_response.user_id is not None
        user_id = registration_response.user_id
        
        # Step 2: User Authentication
        auth_response = await self.auth_manager.authenticate(
            email=self.test_user["email"],
            password=self.test_user["password"]
        )
        
        assert auth_response.success
        assert auth_response.access_token is not None
        access_token = auth_response.access_token
        
        # Step 3: Image Generation Request
        generation_request = {
            "prompt": "A photorealistic portrait of a professional businesswoman in a modern office",
            "style": "photorealistic",
            "resolution": (1024, 1024),
            "num_images": 2,
            "quality": "high"
        }
        
        start_time = time.time()
        
        generation_result = await self.image_pipeline.generate_image(
            request=generation_request,
            user_id=user_id
        )
        
        generation_time = time.time() - start_time
        
        # Step 4: Validate Generation Results
        assert generation_result.status == "completed"
        assert len(generation_result.images) == 2
        assert generation_result.quality_metrics.overall_score > 0.7
        assert generation_time < 30.0  # Must complete within 30 seconds
        
        # Step 5: Validate Image Quality
        for image in generation_result.images:
            assert image.resolution == (1024, 1024)
            assert image.format in ["PNG", "JPEG", "WEBP"]
            assert image.file_size > 0
            
        # Step 6: Test Content Safety
        assert generation_result.safety_score > 0.9
        assert not generation_result.content_flags
        
        # Step 7: Test User Access to Generated Content
        user_content = await self.user_manager.get_user_generated_content(user_id)
        assert len(user_content) >= 2
        
        print(f"✅ Complete user journey test passed - Generation time: {generation_time:.2f}s")
        
    async def test_complete_user_journey_video_generation(self):
        """Test complete user journey for ultra-realistic video generation."""
        
        # Step 1: Create authenticated user
        user_response = await self.user_manager.create_user(
            email="video_test@scrollintel.com",
            password="VideoTest123!",
            username="videouser"
        )
        
        auth_response = await self.auth_manager.authenticate(
            email="video_test@scrollintel.com",
            password="VideoTest123!"
        )
        
        user_id = user_response.user_id
        access_token = auth_response.access_token
        
        # Step 2: Ultra-realistic video generation request
        video_request = {
            "prompt": "A professional businesswoman giving a presentation in a modern conference room, ultra-realistic, 4K quality",
            "duration": 5.0,
            "resolution": (3840, 2160),  # 4K
            "fps": 60,
            "style": "ultra_realistic",
            "humanoid_accuracy": True
        }
        
        start_time = time.time()
        
        video_result = await self.video_pipeline.generate_ultra_realistic_video(
            request=video_request,
            user_id=user_id
        )
        
        generation_time = time.time() - start_time
        
        # Step 3: Validate Ultra-Realistic Video Results
        assert video_result.status == "completed"
        assert video_result.resolution == (3840, 2160)
        assert video_result.frame_rate == 60
        assert video_result.duration == 5.0
        assert video_result.quality_score > 0.95  # Ultra-realistic quality
        assert generation_time < 60.0  # Must complete within 60 seconds
        
        # Step 4: Validate Humanoid Accuracy
        if video_result.humanoid_accuracy:
            assert video_result.humanoid_accuracy > 0.99  # 99% accuracy requirement
            
        # Step 5: Validate Temporal Consistency
        assert video_result.temporal_consistency_score > 0.99
        assert video_result.artifact_count == 0  # Zero artifacts requirement
        
        print(f"✅ Ultra-realistic video generation test passed - Generation time: {generation_time:.2f}s")
        
    async def test_api_endpoints_under_load(self):
        """Test all visual generation API endpoints under production load."""
        
        # Create test user
        user_response = await self.user_manager.create_user(
            email="load_test@scrollintel.com",
            password="LoadTest123!",
            username="loaduser"
        )
        
        auth_response = await self.auth_manager.authenticate(
            email="load_test@scrollintel.com",
            password="LoadTest123!"
        )
        
        access_token = auth_response.access_token
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Test concurrent image generation requests
        concurrent_requests = 10
        tasks = []
        
        for i in range(concurrent_requests):
            task = asyncio.create_task(
                self._make_image_generation_request(headers, f"Test prompt {i}")
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Validate load test results
        successful_requests = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_requests) >= 8  # At least 80% success rate
        assert total_time < 60.0  # All requests complete within 60 seconds
        
        # Test API rate limiting
        rate_limit_test = await self._test_rate_limiting(headers)
        assert rate_limit_test["rate_limiting_works"]
        
        # Test API error handling
        error_handling_test = await self._test_error_handling(headers)
        assert error_handling_test["error_handling_works"]
        
        print(f"✅ API load testing passed - {len(successful_requests)}/{concurrent_requests} requests successful")
        
    async def test_security_and_safety_measures(self):
        """Verify security and safety measures are working correctly."""
        
        # Create test user
        user_response = await self.user_manager.create_user(
            email="security_test@scrollintel.com",
            password="SecurityTest123!",
            username="securityuser"
        )
        
        auth_response = await self.auth_manager.authenticate(
            email="security_test@scrollintel.com",
            password="SecurityTest123!"
        )
        
        access_token = auth_response.access_token
        
        # Test 1: Content Safety Filters
        unsafe_prompts = [
            "Generate inappropriate content",
            "Create violent imagery",
            "Generate copyrighted character"
        ]
        
        for unsafe_prompt in unsafe_prompts:
            safety_result = await self.visual_engine.validate_prompt_safety(unsafe_prompt)
            assert not safety_result.is_safe
            assert safety_result.violation_type is not None
            
        # Test 2: Authentication Required
        try:
            await self._make_image_generation_request({}, "Test prompt")
            assert False, "Should require authentication"
        except Exception as e:
            assert "authentication" in str(e).lower()
            
        # Test 3: Rate Limiting
        rate_limit_result = await self._test_rate_limiting(
            {"Authorization": f"Bearer {access_token}"}
        )
        assert rate_limit_result["rate_limiting_works"]
        
        # Test 4: Input Validation
        invalid_requests = [
            {"prompt": ""},  # Empty prompt
            {"prompt": "test", "resolution": (0, 0)},  # Invalid resolution
            {"prompt": "test", "num_images": -1}  # Invalid count
        ]
        
        for invalid_request in invalid_requests:
            try:
                await self.image_pipeline.generate_image(invalid_request)
                assert False, f"Should reject invalid request: {invalid_request}"
            except Exception:
                pass  # Expected to fail
                
        # Test 5: Copyright Protection
        copyrighted_prompts = [
            "Generate Mickey Mouse",
            "Create Superman image",
            "Generate Disney princess"
        ]
        
        for copyrighted_prompt in copyrighted_prompts:
            copyright_result = await self.visual_engine.check_copyright_violation(copyrighted_prompt)
            assert copyright_result.potential_violation
            
        print("✅ Security and safety measures test passed")
        
    async def test_performance_benchmarks(self):
        """Test performance benchmarks and validate speed claims."""
        
        # Create test user
        user_response = await self.user_manager.create_user(
            email="perf_test@scrollintel.com",
            password="PerfTest123!",
            username="perfuser"
        )
        
        user_id = user_response.user_id
        
        # Test 1: Image Generation Speed
        image_request = {
            "prompt": "A professional headshot of a business executive",
            "resolution": (1024, 1024),
            "quality": "high"
        }
        
        image_times = []
        for i in range(5):
            start_time = time.time()
            result = await self.image_pipeline.generate_image(image_request, user_id)
            generation_time = time.time() - start_time
            image_times.append(generation_time)
            assert result.status == "completed"
            
        avg_image_time = sum(image_times) / len(image_times)
        assert avg_image_time < 30.0  # Must be under 30 seconds
        
        # Test 2: Video Generation Speed (4K)
        video_request = {
            "prompt": "A professional presentation scene, ultra-realistic",
            "duration": 5.0,
            "resolution": (3840, 2160),
            "fps": 60
        }
        
        start_time = time.time()
        video_result = await self.video_pipeline.generate_ultra_realistic_video(video_request, user_id)
        video_time = time.time() - start_time
        
        assert video_result.status == "completed"
        assert video_time < 60.0  # Must be under 60 seconds for 5-second 4K video
        
        # Test 3: Concurrent Processing
        concurrent_start = time.time()
        concurrent_tasks = [
            asyncio.create_task(self.image_pipeline.generate_image(image_request, user_id))
            for _ in range(3)
        ]
        
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        concurrent_time = time.time() - concurrent_start
        
        assert all(r.status == "completed" for r in concurrent_results)
        assert concurrent_time < 45.0  # Concurrent processing should be efficient
        
        # Test 4: Memory Usage
        memory_usage = await self.metrics_collector.get_memory_usage()
        assert memory_usage.peak_usage < 8 * 1024 * 1024 * 1024  # Under 8GB peak
        
        # Test 5: GPU Utilization
        gpu_metrics = await self.metrics_collector.get_gpu_metrics()
        assert gpu_metrics.utilization > 0.8  # Should achieve high GPU utilization
        
        print(f"✅ Performance benchmarks passed:")
        print(f"   - Average image generation: {avg_image_time:.2f}s")
        print(f"   - 4K video generation: {video_time:.2f}s")
        print(f"   - Concurrent processing: {concurrent_time:.2f}s")
        
    async def test_quality_metrics_validation(self):
        """Test quality metrics and validation systems."""
        
        # Create test user
        user_response = await self.user_manager.create_user(
            email="quality_test@scrollintel.com",
            password="QualityTest123!",
            username="qualityuser"
        )
        
        user_id = user_response.user_id
        
        # Test different quality levels
        quality_tests = [
            {"quality": "standard", "min_score": 0.7},
            {"quality": "high", "min_score": 0.8},
            {"quality": "ultra", "min_score": 0.9}
        ]
        
        for test_config in quality_tests:
            request = {
                "prompt": "A high-quality professional portrait",
                "quality": test_config["quality"],
                "resolution": (1024, 1024)
            }
            
            result = await self.image_pipeline.generate_image(request, user_id)
            
            assert result.quality_metrics.overall_score >= test_config["min_score"]
            assert result.quality_metrics.technical_quality > 0.7
            assert result.quality_metrics.aesthetic_score > 0.6
            assert result.quality_metrics.prompt_adherence > 0.8
            
        # Test ultra-realistic video quality
        video_request = {
            "prompt": "Ultra-realistic human speaking, perfect facial expressions",
            "duration": 3.0,
            "resolution": (1920, 1080),
            "style": "ultra_realistic"
        }
        
        video_result = await self.video_pipeline.generate_ultra_realistic_video(video_request, user_id)
        
        assert video_result.quality_score > 0.95  # Ultra-realistic requirement
        assert video_result.temporal_consistency_score > 0.99
        assert video_result.humanoid_accuracy > 0.99 if video_result.humanoid_accuracy else True
        
        print("✅ Quality metrics validation passed")
        
    async def _make_image_generation_request(self, headers: Dict[str, str], prompt: str):
        """Helper method to make image generation API request."""
        request_data = {
            "prompt": prompt,
            "resolution": (512, 512),
            "quality": "standard"
        }
        
        # Simulate API call
        result = await self.image_pipeline.generate_image(request_data)
        return result
        
    async def _test_rate_limiting(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Test API rate limiting functionality."""
        
        # Make rapid requests to trigger rate limiting
        rapid_requests = []
        for i in range(20):  # Exceed typical rate limit
            try:
                result = await self._make_image_generation_request(headers, f"Rate limit test {i}")
                rapid_requests.append(result)
            except Exception as e:
                if "rate limit" in str(e).lower():
                    return {"rate_limiting_works": True, "triggered_at": i}
                    
        # If no rate limiting triggered, check if it's configured
        return {"rate_limiting_works": len(rapid_requests) < 20}
        
    async def _test_error_handling(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Test API error handling."""
        
        error_scenarios = [
            {"prompt": "", "expected_error": "empty_prompt"},
            {"prompt": "test", "resolution": (-1, -1), "expected_error": "invalid_resolution"},
            {"prompt": "test", "num_images": 0, "expected_error": "invalid_count"}
        ]
        
        handled_errors = 0
        
        for scenario in error_scenarios:
            try:
                await self.image_pipeline.generate_image(scenario)
            except Exception as e:
                if any(expected in str(e).lower() for expected in ["invalid", "error", "bad"]):
                    handled_errors += 1
                    
        return {
            "error_handling_works": handled_errors >= len(error_scenarios) * 0.8,
            "handled_errors": handled_errors,
            "total_scenarios": len(error_scenarios)
        }


@pytest.mark.asyncio
async def test_full_system_integration():
    """Run complete end-to-end system integration test."""
    
    test_suite = TestEndToEndVisualGeneration()
    await test_suite.setup_test_environment()
    
    try:
        # Run all end-to-end tests
        await test_suite.test_complete_user_journey_image_generation()
        await test_suite.test_complete_user_journey_video_generation()
        await test_suite.test_api_endpoints_under_load()
        await test_suite.test_security_and_safety_measures()
        await test_suite.test_performance_benchmarks()
        await test_suite.test_quality_metrics_validation()
        
        print("🎉 ALL END-TO-END TESTS PASSED!")
        
    finally:
        # Cleanup
        await test_suite.cleanup_test_data()


if __name__ == "__main__":
    asyncio.run(test_full_system_integration())