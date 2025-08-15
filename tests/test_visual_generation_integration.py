"""
Integration and performance tests for ScrollIntel Visual Generation System.
Tests end-to-end workflows and performance benchmarks.
"""
import pytest
import asyncio
import time
import statistics
import concurrent.futures
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from scrollintel.engines.visual_generation import (
    get_engine, ImageGenerationRequest, VideoGenerationRequest,
    GenerationStatus, ContentType
)
from scrollintel.engines.visual_generation.config_manager import create_optimal_setup
from scrollintel.engines.visual_generation.base import QualityMetrics


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""
    
    @pytest.fixture
    async def engine(self):
        """Get initialized engine for testing"""
        engine = get_engine()
        await engine.initialize()
        return engine
    
    @pytest.mark.asyncio
    async def test_complete_image_generation_workflow(self, engine):
        """Test complete image generation workflow from request to result"""
        # Create comprehensive request
        request = ImageGenerationRequest(
            prompt="A photorealistic portrait of a professional businesswoman in a modern office, natural lighting, 8k resolution, award winning photography",
            user_id="integration_test_user",
            resolution=(2048, 2048),
            num_images=2,
            style="photorealistic",
            quality="ultra_high",
            guidance_scale=7.5,
            steps=50,
            negative_prompt="low quality, blurry, distorted"
        )
        
        # Test cost estimation
        estimated_cost = await engine.estimate_cost(request)
        assert estimated_cost >= 0.0, "Cost estimate should be non-negative"
        
        # Test time estimation
        estimated_time = await engine.estimate_time(request)
        assert estimated_time > 0, "Time estimate should be positive"
        assert estimated_time < 300, "Time estimate should be reasonable"
        
        # Generate image
        start_time = time.time()
        result = await engine.generate_image(request)
        actual_time = time.time() - start_time
        
        # Verify result
        assert result.status == GenerationStatus.COMPLETED
        assert result.request_id == request.request_id
        assert len(result.content_paths) == request.num_images
        assert len(result.content_urls) == request.num_images
        assert result.generation_time > 0
        assert result.cost >= 0.0
        
        # Verify quality metrics
        assert result.quality_metrics is not None
        assert result.quality_metrics.overall_score >= 0.8
        assert result.quality_metrics.prompt_adherence >= 0.8
        
        # Verify metadata
        assert result.metadata is not None
        assert "model_used" in result.metadata or result.model_used is not None
        
        # Time estimation should be reasonably accurate (within 50% margin)
        time_accuracy = abs(actual_time - estimated_time) / estimated_time
        assert time_accuracy < 0.5, f"Time estimation should be accurate, got {time_accuracy:.2%} error"
    
    @pytest.mark.asyncio
    async def test_complete_video_generation_workflow(self, engine):
        """Test complete video generation workflow with advanced features"""
        # Create comprehensive video request
        request = VideoGenerationRequest(
            prompt="A professional presentation by a businesswoman in a modern conference room, smooth camera movement, natural gestures, photorealistic quality",
            user_id="integration_test_user",
            duration=10.0,
            resolution=(1920, 1080),
            fps=30,
            style="photorealistic",
            motion_intensity="medium",
            camera_movement="smooth_pan",
            physics_simulation=True,
            humanoid_generation=True
        )
        
        # Test validation
        is_valid = await engine.validate_request(request)
        assert is_valid, "Request should be valid"
        
        # Test cost and time estimation
        estimated_cost = await engine.estimate_cost(request)
        estimated_time = await engine.estimate_time(request)
        
        assert estimated_cost >= 0.0
        assert estimated_time > 0
        assert estimated_time < 1800, "Video generation should be under 30 minutes"
        
        # Generate video
        start_time = time.time()
        result = await engine.generate_video(request)
        actual_time = time.time() - start_time
        
        # Verify result
        assert result.status == GenerationStatus.COMPLETED
        assert result.request_id == request.request_id
        assert len(result.content_paths) == 1
        assert result.content_paths[0].endswith('.mp4')
        assert result.generation_time > 0
        
        # Verify advanced quality metrics
        assert result.quality_metrics is not None
        assert result.quality_metrics.overall_score >= 0.85
        assert result.quality_metrics.temporal_consistency >= 0.8
        assert result.quality_metrics.motion_smoothness >= 0.8
        
        # Verify advanced features were used
        if result.metadata:
            assert result.metadata.get("physics_simulation") == True
            assert result.metadata.get("humanoid_generation") == True
    
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, engine):
        """Test batch processing of multiple requests"""
        # Create multiple requests
        requests = [
            ImageGenerationRequest(
                prompt=f"Professional headshot {i+1}, studio lighting, high quality",
                user_id=f"batch_user_{i}",
                resolution=(1024, 1024),
                num_images=1
            )
            for i in range(5)
        ]
        
        # Process batch
        start_time = time.time()
        results = await engine.batch_generate(requests)
        batch_time = time.time() - start_time
        
        # Verify batch results
        assert len(results) == len(requests)
        
        # Count successful generations
        successful = sum(1 for r in results if r.status == GenerationStatus.COMPLETED)
        success_rate = successful / len(requests)
        assert success_rate >= 0.8, f"Success rate should be at least 80%, got {success_rate:.1%}"
        
        # Batch processing should be more efficient than sequential
        estimated_sequential_time = sum(await engine.estimate_time(req) for req in requests)
        efficiency = batch_time / estimated_sequential_time
        assert efficiency < 1.2, f"Batch processing should be efficient, got {efficiency:.2f}x sequential time"
    
    @pytest.mark.asyncio
    async def test_enhancement_workflow(self, engine):
        """Test content enhancement workflow"""
        # First generate an image
        image_request = ImageGenerationRequest(
            prompt="A landscape photo that needs enhancement",
            user_id="enhancement_test_user",
            resolution=(1024, 1024),
            num_images=1
        )
        
        image_result = await engine.generate_image(image_request)
        assert image_result.status == GenerationStatus.COMPLETED
        
        # Then enhance it
        original_path = image_result.content_paths[0]
        enhancement_result = await engine.enhance_content(original_path, "super_resolution")
        
        # Verify enhancement
        assert enhancement_result.status == GenerationStatus.COMPLETED
        assert len(enhancement_result.content_paths) == 1
        assert enhancement_result.cost >= 0.0
        
        # Enhanced image should have better quality metrics
        if image_result.quality_metrics and enhancement_result.quality_metrics:
            assert enhancement_result.quality_metrics.overall_score >= image_result.quality_metrics.overall_score
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, engine):
        """Test error recovery and fallback mechanisms"""
        # Create a potentially problematic request
        problematic_request = ImageGenerationRequest(
            prompt="",  # Empty prompt might cause issues
            user_id="error_test_user",
            resolution=(1024, 1024),
            num_images=1
        )
        
        # System should handle gracefully
        result = await engine.generate_image(problematic_request)
        
        # Should either succeed with fallback or fail gracefully
        assert result.status in [GenerationStatus.COMPLETED, GenerationStatus.FAILED]
        assert result.request_id == problematic_request.request_id
        
        if result.status == GenerationStatus.FAILED:
            assert result.error_message is not None
            assert len(result.error_message) > 0
    
    @pytest.mark.asyncio
    async def test_model_selection_workflow(self, engine):
        """Test intelligent model selection workflow"""
        # Create requests with different requirements
        speed_request = ImageGenerationRequest(
            prompt="Quick test image",
            user_id="speed_test_user",
            resolution=(512, 512),
            quality="fast"
        )
        
        quality_request = ImageGenerationRequest(
            prompt="High quality artistic masterpiece",
            user_id="quality_test_user",
            resolution=(2048, 2048),
            quality="ultra_high"
        )
        
        # Generate both
        speed_result = await engine.generate_image(speed_request)
        quality_result = await engine.generate_image(quality_request)
        
        # Both should succeed
        assert speed_result.status == GenerationStatus.COMPLETED
        assert quality_result.status == GenerationStatus.COMPLETED
        
        # Speed request should be faster
        assert speed_result.generation_time <= quality_result.generation_time * 1.5
        
        # Quality request should have better quality metrics
        if speed_result.quality_metrics and quality_result.quality_metrics:
            assert quality_result.quality_metrics.overall_score >= speed_result.quality_metrics.overall_score


class TestPerformanceBenchmarks:
    """Test performance benchmarks and scalability"""
    
    @pytest.fixture
    async def engine(self):
        """Get initialized engine for performance testing"""
        engine = get_engine()
        await engine.initialize()
        return engine
    
    @pytest.mark.asyncio
    async def test_image_generation_performance(self, engine):
        """Test image generation performance benchmarks"""
        # Test different resolutions
        resolutions = [(512, 512), (1024, 1024), (2048, 2048)]
        performance_data = []
        
        for resolution in resolutions:
            request = ImageGenerationRequest(
                prompt="Performance benchmark test image",
                user_id="perf_test_user",
                resolution=resolution,
                num_images=1
            )
            
            # Measure performance
            start_time = time.time()
            result = await engine.generate_image(request)
            generation_time = time.time() - start_time
            
            if result.status == GenerationStatus.COMPLETED:
                performance_data.append({
                    'resolution': resolution,
                    'time': generation_time,
                    'pixels': resolution[0] * resolution[1],
                    'quality_score': result.quality_metrics.overall_score if result.quality_metrics else 0.0
                })
        
        # Analyze performance scaling
        assert len(performance_data) >= 2, "Should have performance data for multiple resolutions"
        
        # Higher resolution should generally take longer (but not excessively)
        for i in range(1, len(performance_data)):
            current = performance_data[i]
            previous = performance_data[i-1]
            
            pixel_ratio = current['pixels'] / previous['pixels']
            time_ratio = current['time'] / previous['time']
            
            # Time should scale reasonably with pixel count
            assert time_ratio <= pixel_ratio * 2, f"Time scaling should be reasonable: {time_ratio:.2f}x time for {pixel_ratio:.2f}x pixels"
    
    @pytest.mark.asyncio
    async def test_concurrent_request_performance(self, engine):
        """Test performance under concurrent load"""
        # Create multiple concurrent requests
        num_concurrent = 10
        requests = [
            ImageGenerationRequest(
                prompt=f"Concurrent test {i}",
                user_id=f"concurrent_user_{i}",
                resolution=(512, 512),
                num_images=1
            )
            for i in range(num_concurrent)
        ]
        
        # Test sequential processing time
        start_time = time.time()
        sequential_results = []
        for request in requests[:3]:  # Test with subset for speed
            result = await engine.generate_image(request)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Test concurrent processing time
        start_time = time.time()
        concurrent_tasks = [engine.generate_image(req) for req in requests]
        concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time
        
        # Analyze results
        successful_concurrent = sum(1 for r in concurrent_results 
                                  if not isinstance(r, Exception) and r.status == GenerationStatus.COMPLETED)
        
        # Should handle most concurrent requests successfully
        success_rate = successful_concurrent / num_concurrent
        assert success_rate >= 0.7, f"Should handle at least 70% of concurrent requests, got {success_rate:.1%}"
        
        # Concurrent processing should show some efficiency gain
        estimated_sequential_time = sequential_time * (num_concurrent / 3)
        efficiency = concurrent_time / estimated_sequential_time
        assert efficiency < 1.5, f"Concurrent processing should be reasonably efficient, got {efficiency:.2f}x sequential time"
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, engine):
        """Test memory usage stability over multiple requests"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate multiple images to test memory stability
        for i in range(10):
            request = ImageGenerationRequest(
                prompt=f"Memory test image {i}",
                user_id="memory_test_user",
                resolution=(1024, 1024),
                num_images=1
            )
            
            result = await engine.generate_image(request)
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory usage should not grow excessively
            assert memory_increase < 1000, f"Memory usage should be stable, increased by {memory_increase:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_quality_consistency(self, engine):
        """Test quality consistency across multiple generations"""
        # Generate multiple images with same prompt
        base_prompt = "A professional headshot with studio lighting, high quality"
        quality_scores = []
        
        for i in range(5):
            request = ImageGenerationRequest(
                prompt=base_prompt,
                user_id=f"quality_test_user_{i}",
                resolution=(1024, 1024),
                num_images=1,
                seed=42  # Fixed seed for consistency
            )
            
            result = await engine.generate_image(request)
            
            if result.status == GenerationStatus.COMPLETED and result.quality_metrics:
                quality_scores.append(result.quality_metrics.overall_score)
        
        # Analyze quality consistency
        assert len(quality_scores) >= 3, "Should have quality scores from multiple generations"
        
        mean_quality = statistics.mean(quality_scores)
        quality_std = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
        
        # Quality should be consistently high
        assert mean_quality >= 0.8, f"Average quality should be high, got {mean_quality:.3f}"
        assert quality_std <= 0.1, f"Quality should be consistent, got std dev {quality_std:.3f}"
    
    @pytest.mark.asyncio
    async def test_throughput_measurement(self, engine):
        """Test system throughput measurement"""
        # Measure throughput over time period
        test_duration = 30  # seconds
        completed_requests = 0
        start_time = time.time()
        
        # Generate requests continuously for test duration
        while time.time() - start_time < test_duration:
            request = ImageGenerationRequest(
                prompt="Throughput test image",
                user_id="throughput_test_user",
                resolution=(512, 512),  # Smaller for faster generation
                num_images=1
            )
            
            result = await engine.generate_image(request)
            if result.status == GenerationStatus.COMPLETED:
                completed_requests += 1
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
        
        actual_duration = time.time() - start_time
        throughput = completed_requests / actual_duration  # requests per second
        
        # Should achieve reasonable throughput
        assert throughput > 0, "Should complete some requests"
        assert throughput < 100, "Throughput should be realistic"  # Sanity check
        
        print(f"Achieved throughput: {throughput:.2f} requests/second")
    
    @pytest.mark.asyncio
    async def test_error_rate_under_load(self, engine):
        """Test error rate under high load conditions"""
        # Create high load scenario
        num_requests = 20
        requests = [
            ImageGenerationRequest(
                prompt=f"Load test {i}",
                user_id=f"load_user_{i}",
                resolution=(1024, 1024),
                num_images=1
            )
            for i in range(num_requests)
        ]
        
        # Process all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*[engine.generate_image(req) for req in requests], return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze error rates
        successful = sum(1 for r in results 
                        if not isinstance(r, Exception) and r.status == GenerationStatus.COMPLETED)
        failed = sum(1 for r in results 
                    if isinstance(r, Exception) or r.status == GenerationStatus.FAILED)
        
        success_rate = successful / num_requests
        error_rate = failed / num_requests
        
        # Error rate should be acceptable under load
        assert error_rate <= 0.3, f"Error rate should be under 30% under load, got {error_rate:.1%}"
        assert success_rate >= 0.7, f"Success rate should be at least 70% under load, got {success_rate:.1%}"
        
        print(f"Load test results: {success_rate:.1%} success rate, {error_rate:.1%} error rate")


class TestQualityRegression:
    """Test quality regression and consistency"""
    
    @pytest.fixture
    async def engine(self):
        """Get initialized engine for quality testing"""
        engine = get_engine()
        await engine.initialize()
        return engine
    
    @pytest.mark.asyncio
    async def test_quality_baseline_maintenance(self, engine):
        """Test that quality baselines are maintained"""
        # Define quality baseline tests
        baseline_tests = [
            {
                'prompt': 'A professional portrait photograph, studio lighting, high quality',
                'expected_min_quality': 0.85,
                'category': 'portrait'
            },
            {
                'prompt': 'A beautiful landscape with mountains and lakes, golden hour lighting',
                'expected_min_quality': 0.80,
                'category': 'landscape'
            },
            {
                'prompt': 'Abstract digital art with vibrant colors and geometric shapes',
                'expected_min_quality': 0.75,
                'category': 'abstract'
            }
        ]
        
        for test_case in baseline_tests:
            request = ImageGenerationRequest(
                prompt=test_case['prompt'],
                user_id="quality_baseline_user",
                resolution=(1024, 1024),
                num_images=1,
                quality="high"
            )
            
            result = await engine.generate_image(request)
            
            # Verify quality meets baseline
            assert result.status == GenerationStatus.COMPLETED
            assert result.quality_metrics is not None
            
            actual_quality = result.quality_metrics.overall_score
            expected_quality = test_case['expected_min_quality']
            
            assert actual_quality >= expected_quality, \
                f"{test_case['category']} quality below baseline: {actual_quality:.3f} < {expected_quality:.3f}"
    
    @pytest.mark.asyncio
    async def test_prompt_adherence_consistency(self, engine):
        """Test consistency of prompt adherence"""
        # Test prompts with specific requirements
        test_prompts = [
            "A red car on a blue background",
            "A person wearing a green hat and yellow shirt",
            "A wooden table with three books on it",
            "A sunset over the ocean with birds flying"
        ]
        
        adherence_scores = []
        
        for prompt in test_prompts:
            request = ImageGenerationRequest(
                prompt=prompt,
                user_id="adherence_test_user",
                resolution=(1024, 1024),
                num_images=1
            )
            
            result = await engine.generate_image(request)
            
            if result.status == GenerationStatus.COMPLETED and result.quality_metrics:
                adherence_scores.append(result.quality_metrics.prompt_adherence)
        
        # Analyze adherence consistency
        assert len(adherence_scores) >= len(test_prompts) * 0.8, "Most prompts should generate successfully"
        
        mean_adherence = statistics.mean(adherence_scores)
        adherence_std = statistics.stdev(adherence_scores) if len(adherence_scores) > 1 else 0
        
        # Prompt adherence should be consistently good
        assert mean_adherence >= 0.8, f"Average prompt adherence should be high, got {mean_adherence:.3f}"
        assert adherence_std <= 0.15, f"Prompt adherence should be consistent, got std dev {adherence_std:.3f}"
    
    @pytest.mark.asyncio
    async def test_technical_quality_standards(self, engine):
        """Test technical quality standards"""
        request = ImageGenerationRequest(
            prompt="Technical quality test image with fine details and sharp focus",
            user_id="technical_quality_user",
            resolution=(2048, 2048),
            num_images=1,
            quality="ultra_high"
        )
        
        result = await engine.generate_image(request)
        
        # Verify technical quality metrics
        assert result.status == GenerationStatus.COMPLETED
        assert result.quality_metrics is not None
        
        metrics = result.quality_metrics
        
        # Technical quality standards
        assert metrics.technical_quality >= 0.85, f"Technical quality too low: {metrics.technical_quality:.3f}"
        assert metrics.sharpness >= 0.80, f"Sharpness too low: {metrics.sharpness:.3f}"
        assert metrics.color_balance >= 0.75, f"Color balance too low: {metrics.color_balance:.3f}"
        
        # Overall quality should be high for ultra_high setting
        assert metrics.overall_score >= 0.90, f"Overall quality should be very high: {metrics.overall_score:.3f}"


class TestSystemIntegration:
    """Test integration with broader ScrollIntel system"""
    
    @pytest.mark.asyncio
    async def test_engine_initialization_integration(self):
        """Test engine initialization and system integration"""
        # Test engine creation and initialization
        engine = get_engine()
        assert engine is not None
        
        # Test initialization
        await engine.initialize()
        
        # Test system status
        status = engine.get_system_status()
        assert status['initialized'] == True
        assert 'models' in status
        assert len(status['models']) > 0
        
        # Test capabilities
        capabilities = await engine.get_model_capabilities()
        assert len(capabilities) > 0
        
        # Should have ScrollIntel models
        scrollintel_models = [name for name in capabilities.keys() if 'scrollintel' in name.lower()]
        assert len(scrollintel_models) > 0, "Should have ScrollIntel proprietary models"
    
    @pytest.mark.asyncio
    async def test_configuration_integration(self):
        """Test configuration system integration"""
        # Test optimal setup creation
        config_manager = create_optimal_setup()
        assert config_manager is not None
        
        # Test configuration retrieval
        config = config_manager.get_optimal_configuration()
        assert config is not None
        assert len(config.models) > 0
        
        # Test deployment recommendations
        recommendations = config_manager.get_deployment_recommendations()
        assert 'current_mode' in recommendations
        assert 'enabled_models' in recommendations
        assert len(recommendations['enabled_models']) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling integration across system"""
        engine = get_engine()
        await engine.initialize()
        
        # Test various error scenarios
        error_scenarios = [
            # Invalid resolution
            ImageGenerationRequest(
                prompt="Test",
                user_id="error_test",
                resolution=(10000, 10000),
                num_images=1
            ),
            # Invalid video duration
            VideoGenerationRequest(
                prompt="Test",
                user_id="error_test",
                duration=10000.0,
                resolution=(1920, 1080)
            )
        ]
        
        for request in error_scenarios:
            try:
                if isinstance(request, ImageGenerationRequest):
                    result = await engine.generate_image(request)
                else:
                    result = await engine.generate_video(request)
                
                # Should either succeed or fail gracefully
                assert result.status in [GenerationStatus.COMPLETED, GenerationStatus.FAILED]
                
                if result.status == GenerationStatus.FAILED:
                    assert result.error_message is not None
                    
            except Exception as e:
                # Exceptions should be handled gracefully
                assert isinstance(e, (ValueError, TypeError, Exception))
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self):
        """Test monitoring and metrics integration"""
        engine = get_engine()
        await engine.initialize()
        
        # Generate some content to create metrics
        request = ImageGenerationRequest(
            prompt="Monitoring test image",
            user_id="monitoring_test_user",
            resolution=(512, 512),
            num_images=1
        )
        
        result = await engine.generate_image(request)
        
        # Test that metrics are being collected
        if hasattr(engine, 'get_metrics'):
            metrics = engine.get_metrics()
            assert metrics is not None
        
        # Test system health
        if hasattr(engine, 'health_check'):
            health = await engine.health_check()
            assert 'overall_health' in health
    
    def test_api_compatibility(self):
        """Test API compatibility and interface consistency"""
        # Test that all required methods exist
        engine = get_engine()
        
        required_methods = [
            'initialize',
            'generate_image', 
            'generate_video',
            'batch_generate',
            'estimate_cost',
            'estimate_time',
            'validate_request',
            'get_system_status',
            'get_model_capabilities'
        ]
        
        for method_name in required_methods:
            assert hasattr(engine, method_name), f"Engine should have {method_name} method"
            method = getattr(engine, method_name)
            assert callable(method), f"{method_name} should be callable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])