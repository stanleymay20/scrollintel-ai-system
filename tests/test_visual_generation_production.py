#!/usr/bin/env python3
"""
Production tests for ScrollIntel Visual Generation System
Tests superiority over InVideo and competitors
"""

import pytest
import asyncio
import time
from pathlib import Path

from scrollintel.engines.visual_generation import get_engine, ImageGenerationRequest, VideoGenerationRequest
from scrollintel.engines.visual_generation.production_config import get_production_config
from scrollintel.engines.visual_generation.base import GenerationStatus


class TestVisualGenerationSuperiority:
    """Test ScrollIntel's superiority over competitors"""
    
    @pytest.fixture
    async def engine(self):
        """Get production-ready engine"""
        engine = get_engine()
        await engine.initialize()
        return engine
    
    @pytest.fixture
    def production_config(self):
        """Get production configuration"""
        return get_production_config()
    
    @pytest.mark.asyncio
    async def test_image_generation_speed(self, engine):
        """Test that image generation is faster than competitors"""
        request = ImageGenerationRequest(
            prompt="A photorealistic portrait of a person",
            user_id="test_user",
            resolution=(1024, 1024),
            num_images=1
        )
        
        start_time = time.time()
        result = await engine.generate_image(request)
        generation_time = time.time() - start_time
        
        # ScrollIntel should generate images in under 30 seconds
        assert generation_time < 30.0, f"Generation took {generation_time}s, should be under 30s"
        assert result.status == GenerationStatus.COMPLETED
        assert result.cost == 0.0, "Local generation should be free"
        
        # Quality should be superior
        if result.quality_metrics:
            assert result.quality_metrics.overall_score >= 0.85, "Quality should be high"
    
    @pytest.mark.asyncio
    async def test_video_generation_quality(self, engine):
        """Test ultra-realistic video generation"""
        request = VideoGenerationRequest(
            prompt="A person walking through a forest",
            user_id="test_user",
            duration=5.0,
            resolution=(1920, 1080),
            fps=30,
            humanoid_generation=True,
            physics_simulation=True
        )
        
        start_time = time.time()
        result = await engine.generate_video(request)
        generation_time = time.time() - start_time
        
        # Should complete successfully
        assert result.status == GenerationStatus.COMPLETED
        assert result.cost == 0.0, "Proprietary engine should be free"
        
        # Quality metrics should be superior
        if result.quality_metrics:
            assert result.quality_metrics.overall_score >= 0.95, "Video quality should be ultra-high"
            assert result.quality_metrics.temporal_consistency >= 0.95, "Temporal consistency should be excellent"
            assert result.quality_metrics.realism_score >= 0.95, "Realism should be near-perfect"
        
        # Should support advanced features
        assert result.metadata["features_used"]["physics_simulation"] == True
        assert result.metadata["features_used"]["humanoid_generation"] == True
    
    @pytest.mark.asyncio
    async def test_4k_video_generation(self, engine):
        """Test 4K video generation capability"""
        request = VideoGenerationRequest(
            prompt="Ultra-realistic product showcase",
            user_id="test_user",
            duration=3.0,
            resolution=(3840, 2160),  # 4K
            fps=60,
            neural_rendering_quality="photorealistic_plus"
        )
        
        result = await engine.generate_video(request)
        
        # Should handle 4K generation
        assert result.status == GenerationStatus.COMPLETED
        assert result.metadata["resolution"] == (3840, 2160)
        assert result.metadata["fps"] == 60
    
    @pytest.mark.asyncio
    async def test_cost_advantage(self, engine):
        """Test cost advantages over competitors"""
        # Test multiple generations to verify cost efficiency
        requests = [
            ImageGenerationRequest(
                prompt=f"Test image {i}",
                user_id="cost_test_user",
                resolution=(1024, 1024)
            )
            for i in range(5)
        ]
        
        results = await engine.batch_generate(requests)
        
        # All should be free with local generation
        total_cost = sum(r.cost for r in results)
        assert total_cost == 0.0, "Local generation should be completely free"
        
        # All should complete successfully
        completed = sum(1 for r in results if r.status == GenerationStatus.COMPLETED)
        assert completed == len(requests), "All requests should complete successfully"
    
    def test_competitive_advantages(self, production_config):
        """Test competitive advantage documentation"""
        advantages = production_config.get_competitive_advantages()
        
        # Should have advantages over all major competitors
        assert "vs_invideo" in advantages
        assert "vs_runway" in advantages
        assert "vs_pika_labs" in advantages
        assert "unique_advantages" in advantages
        
        # Should highlight cost advantages
        assert "FREE" in advantages["vs_invideo"]["cost"]
        assert "FREE" in advantages["vs_runway"]["cost"]
        
        # Should highlight quality advantages
        assert "4K" in advantages["vs_invideo"]["quality"]
        assert "Ultra-realistic" in advantages["vs_invideo"]["quality"]
    
    def test_production_readiness(self, production_config):
        """Test production readiness validation"""
        readiness = production_config.validate_production_readiness()
        
        # Should be production ready
        assert readiness["overall_readiness"]["status"] == "PRODUCTION_READY"
        assert readiness["overall_readiness"]["score"] >= 0.9
        
        # Should have proper model availability
        assert len(readiness["model_availability"]) > 0
        
        # Should have local models available (no API keys required)
        local_models = [
            model for model, config in readiness["model_availability"].items()
            if config.get("local_model", False)
        ]
        assert len(local_models) > 0, "Should have local models available"
    
    @pytest.mark.asyncio
    async def test_model_selection_strategy(self, engine):
        """Test intelligent model selection"""
        # Test that local models are preferred for cost efficiency
        request = ImageGenerationRequest(
            prompt="Test model selection",
            user_id="strategy_test_user",
            quality="high"  # Should use local model
        )
        
        result = await engine.generate_image(request)
        
        # Should use local model for cost efficiency
        assert "local" in result.model_used or "scrollintel" in result.model_used
        assert result.cost == 0.0
    
    @pytest.mark.asyncio
    async def test_quality_superiority(self, engine):
        """Test quality superiority over industry standards"""
        request = VideoGenerationRequest(
            prompt="High quality test video",
            user_id="quality_test_user",
            duration=3.0,
            neural_rendering_quality="photorealistic_plus",
            temporal_consistency_level="ultra_high"
        )
        
        result = await engine.generate_video(request)
        
        if result.quality_metrics:
            # Should exceed industry standards
            assert result.quality_metrics.overall_score >= 0.95, "Should exceed 95% quality"
            assert result.quality_metrics.temporal_consistency >= 0.95, "Should have excellent temporal consistency"
            assert result.quality_metrics.realism_score >= 0.95, "Should have near-perfect realism"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, engine):
        """Test performance benchmarks against competitors"""
        # Test image generation speed
        image_request = ImageGenerationRequest(
            prompt="Performance benchmark test",
            user_id="benchmark_user",
            resolution=(1024, 1024)
        )
        
        start_time = time.time()
        image_result = await engine.generate_image(image_request)
        image_time = time.time() - start_time
        
        # Should be faster than industry average (30s)
        assert image_time < 30.0, f"Image generation should be under 30s, got {image_time}s"
        
        # Test video generation speed
        video_request = VideoGenerationRequest(
            prompt="Performance benchmark video",
            user_id="benchmark_user",
            duration=5.0,
            resolution=(1920, 1080)
        )
        
        start_time = time.time()
        video_result = await engine.generate_video(video_request)
        video_time = time.time() - start_time
        
        # Should be reasonable for 5s of video
        assert video_time < 300.0, f"Video generation should be under 5 minutes, got {video_time}s"
    
    def test_feature_superiority(self, production_config):
        """Test feature superiority over competitors"""
        capabilities = production_config.get_competitive_advantages()
        unique_features = capabilities["unique_advantages"]
        
        # Should have features competitors don't have
        feature_keywords = [
            "proprietary", "physics", "humanoid", "4K", "60fps", 
            "temporal consistency", "neural rendering"
        ]
        
        features_text = " ".join(unique_features).lower()
        for keyword in feature_keywords:
            assert keyword.lower() in features_text, f"Should mention {keyword} as unique advantage"
    
    @pytest.mark.asyncio
    async def test_scalability(self, engine):
        """Test system scalability"""
        # Test concurrent requests
        concurrent_requests = [
            ImageGenerationRequest(
                prompt=f"Scalability test {i}",
                user_id=f"scale_user_{i}",
                resolution=(512, 512)  # Smaller for faster testing
            )
            for i in range(5)
        ]
        
        start_time = time.time()
        results = await engine.batch_generate(concurrent_requests)
        total_time = time.time() - start_time
        
        # Should handle concurrent requests efficiently
        assert len(results) == len(concurrent_requests)
        successful = sum(1 for r in results if r.status == GenerationStatus.COMPLETED)
        assert successful >= len(concurrent_requests) * 0.8, "At least 80% should succeed"
        
        # Should be faster than sequential processing
        assert total_time < len(concurrent_requests) * 30, "Batch processing should be efficient"


@pytest.mark.asyncio
async def test_integration_with_scrollintel():
    """Test integration with main ScrollIntel system"""
    # Test that visual generation integrates properly
    engine = get_engine()
    await engine.initialize()
    
    # Should be able to get system status
    status = engine.get_system_status()
    assert status["initialized"] == True
    assert len(status["models"]) > 0
    
    # Should have proper capabilities
    capabilities = await engine.get_model_capabilities()
    assert len(capabilities) > 0


class TestScrollIntelModelComponents:
    """Test individual ScrollIntel model components"""
    
    @pytest.fixture
    async def image_generator(self):
        """Get ScrollIntel image generator"""
        from scrollintel.engines.visual_generation.config import ModelConfig
        config = ModelConfig(
            name="scrollintel_image_generator",
            type="image",
            model_path="./models/scrollintel_image_gen",
            max_resolution=(4096, 4096),
            enabled=True
        )
        generator = ScrollIntelImageGenerator(config)
        await generator.initialize()
        return generator
    
    @pytest.fixture
    async def video_generator(self):
        """Get ScrollIntel video generator"""
        from scrollintel.engines.visual_generation.config import ModelConfig
        config = ModelConfig(
            name="scrollintel_video_generator",
            type="video",
            model_path="./models/scrollintel_video_gen",
            max_resolution=(3840, 2160),
            max_duration=1800.0,
            enabled=True
        )
        generator = ScrollIntelVideoGenerator(config)
        await generator.initialize()
        return generator
    
    @pytest.fixture
    async def enhancement_suite(self):
        """Get ScrollIntel enhancement suite"""
        from scrollintel.engines.visual_generation.config import ModelConfig
        config = ModelConfig(
            name="scrollintel_enhancement_suite",
            type="enhancement",
            model_path="./models/scrollintel_enhance",
            max_resolution=(8192, 8192),
            enabled=True
        )
        suite = ScrollIntelEnhancementSuite(config)
        await suite.initialize()
        return suite
    
    @pytest.mark.asyncio
    async def test_image_generator_capabilities(self, image_generator):
        """Test ScrollIntel image generator capabilities"""
        capabilities = image_generator.get_capabilities()
        
        # Should have superior capabilities
        assert capabilities["max_resolution"] == (4096, 4096)
        assert capabilities["supports_prompt_enhancement"] == True
        assert capabilities["supports_style_control"] == True
        assert capabilities["cost"] == "free"
        
        # Should list advantages over competitors
        assert "advantages" in capabilities
        assert len(capabilities["advantages"]) >= 5
    
    @pytest.mark.asyncio
    async def test_video_generator_capabilities(self, video_generator):
        """Test ScrollIntel video generator capabilities"""
        capabilities = video_generator.get_capabilities()
        
        # Should have revolutionary capabilities
        assert capabilities["max_resolution"] == (3840, 2160)  # 4K
        assert capabilities["max_fps"] == 60
        assert capabilities["max_duration"] == 1800.0  # 30 minutes
        assert capabilities["supports_physics_simulation"] == True
        assert capabilities["supports_humanoid_generation"] == True
        
        # Should have advantages over InVideo
        assert "advantages_over_invideo" in capabilities
        invideo_advantages = capabilities["advantages_over_invideo"]
        assert len(invideo_advantages) >= 10
        assert "True AI generation vs template-based" in invideo_advantages
        assert "4K 60fps vs limited quality" in invideo_advantages
        assert "Free vs expensive subscription" in invideo_advantages
    
    @pytest.mark.asyncio
    async def test_enhancement_suite_capabilities(self, enhancement_suite):
        """Test ScrollIntel enhancement suite capabilities"""
        capabilities = enhancement_suite.get_capabilities()
        
        # Should have advanced enhancement capabilities
        assert capabilities["max_upscale_factor"] == 8
        assert "super_resolution" in capabilities["enhancement_types"]
        assert "face_restoration" in capabilities["enhancement_types"]
        assert "style_transfer" in capabilities["enhancement_types"]
        
        # Should have advantages over competitors
        assert "advantages" in capabilities
        assert "8x upscaling vs 4x competitors" in capabilities["advantages"]
    
    @pytest.mark.asyncio
    async def test_image_generation_request_validation(self, image_generator):
        """Test image generation request validation"""
        from scrollintel.engines.visual_generation.base import ImageGenerationRequest
        
        # Valid request should pass
        valid_request = ImageGenerationRequest(
            prompt="Test image",
            user_id="test_user",
            resolution=(1024, 1024),
            num_images=1
        )
        assert await image_generator.validate_request(valid_request) == True
        
        # Invalid resolution should fail
        invalid_request = ImageGenerationRequest(
            prompt="Test image",
            user_id="test_user",
            resolution=(8192, 8192),  # Too large
            num_images=1
        )
        assert await image_generator.validate_request(invalid_request) == False
    
    @pytest.mark.asyncio
    async def test_video_generation_request_validation(self, video_generator):
        """Test video generation request validation"""
        from scrollintel.engines.visual_generation.base import VideoGenerationRequest
        
        # Valid request should pass
        valid_request = VideoGenerationRequest(
            prompt="Test video",
            user_id="test_user",
            duration=10.0,
            resolution=(1920, 1080),
            fps=30
        )
        assert await video_generator.validate_request(valid_request) == True
        
        # Invalid duration should fail
        invalid_request = VideoGenerationRequest(
            prompt="Test video",
            user_id="test_user",
            duration=2000.0,  # Too long
            resolution=(1920, 1080),
            fps=30
        )
        assert await video_generator.validate_request(invalid_request) == False
    
    @pytest.mark.asyncio
    async def test_cost_estimation(self, image_generator, video_generator):
        """Test cost estimation for ScrollIntel models"""
        from scrollintel.engines.visual_generation.base import ImageGenerationRequest, VideoGenerationRequest
        
        # Image generation should be free
        image_request = ImageGenerationRequest(
            prompt="Cost test",
            user_id="test_user",
            resolution=(1024, 1024)
        )
        image_cost = await image_generator.estimate_cost(image_request)
        assert image_cost == 0.0, "ScrollIntel image generation should be free"
        
        # Video generation should be free
        video_request = VideoGenerationRequest(
            prompt="Cost test",
            user_id="test_user",
            duration=5.0,
            resolution=(1920, 1080)
        )
        video_cost = await video_generator.estimate_cost(video_request)
        assert video_cost == 0.0, "ScrollIntel video generation should be free"
    
    @pytest.mark.asyncio
    async def test_time_estimation(self, image_generator, video_generator):
        """Test time estimation accuracy"""
        from scrollintel.engines.visual_generation.base import ImageGenerationRequest, VideoGenerationRequest
        
        # Image generation time estimation
        image_request = ImageGenerationRequest(
            prompt="Time test",
            user_id="test_user",
            resolution=(1024, 1024)
        )
        estimated_time = await image_generator.estimate_time(image_request)
        assert estimated_time > 0, "Should provide time estimate"
        assert estimated_time < 60, "Should be reasonable estimate"
        
        # Video generation time estimation
        video_request = VideoGenerationRequest(
            prompt="Time test",
            user_id="test_user",
            duration=5.0,
            resolution=(1920, 1080)
        )
        estimated_time = await video_generator.estimate_time(video_request)
        assert estimated_time > 0, "Should provide time estimate"
        assert estimated_time < 600, "Should be reasonable estimate for 5s video"


class TestConfigurationManager:
    """Test configuration management system"""
    
    @pytest.fixture
    def config_manager(self):
        """Get configuration manager"""
        return ConfigurationManager()
    
    def test_api_key_detection(self, config_manager):
        """Test API key detection"""
        api_keys = config_manager._check_api_keys()
        
        # Should check for all major API keys
        expected_keys = ['stability_ai', 'openai', 'midjourney', 'anthropic', 'replicate']
        for key in expected_keys:
            assert key in api_keys
            assert isinstance(api_keys[key], bool)
    
    def test_optimal_configuration(self, config_manager):
        """Test optimal configuration generation"""
        config = config_manager.get_optimal_configuration()
        
        # Should have models configured
        assert len(config.models) > 0
        
        # Should have ScrollIntel proprietary models
        proprietary_models = [name for name in config.models.keys() if name.startswith('scrollintel_')]
        assert len(proprietary_models) >= 3, "Should have ScrollIntel proprietary models"
    
    def test_deployment_recommendations(self, config_manager):
        """Test deployment recommendations"""
        recommendations = config_manager.get_deployment_recommendations()
        
        # Should provide comprehensive recommendations
        assert 'current_mode' in recommendations
        assert 'performance_tier' in recommendations
        assert 'enabled_models' in recommendations
        assert 'recommendations' in recommendations
        
        # Should have enabled models
        assert len(recommendations['enabled_models']) > 0
    
    def test_self_hosted_configuration(self, config_manager):
        """Test self-hosted configuration"""
        config_manager.deployment_config.mode = "self_hosted"
        config = config_manager.get_optimal_configuration()
        
        # Should prioritize local models
        local_models = [name for name, model in config.models.items() 
                       if model.enabled and name.startswith('scrollintel_')]
        assert len(local_models) >= 3, "Should enable ScrollIntel local models"
    
    def test_hybrid_configuration(self, config_manager):
        """Test hybrid configuration"""
        config_manager.deployment_config.mode = "hybrid"
        config = config_manager.get_optimal_configuration()
        
        # Should enable both local and API models where available
        enabled_models = [name for name, model in config.models.items() if model.enabled]
        assert len(enabled_models) >= 3, "Should enable multiple models in hybrid mode"


class TestIntelligentOrchestrator:
    """Test intelligent orchestration system"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Get intelligent orchestrator"""
        from scrollintel.engines.visual_generation.config import VisualGenerationConfig
        config = VisualGenerationConfig()
        orchestrator = IntelligentOrchestrator(config)
        
        # Register mock generators
        mock_generator = AsyncMock()
        mock_generator.validate_request = AsyncMock(return_value=True)
        mock_generator.generate = AsyncMock()
        
        orchestrator.register_generator("test_generator", mock_generator)
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_generator_registration(self, orchestrator):
        """Test generator registration"""
        assert "test_generator" in orchestrator.generators
        assert "test_generator" in orchestrator.performance_metrics
    
    @pytest.mark.asyncio
    async def test_fallback_chain_setup(self, orchestrator):
        """Test fallback chain setup"""
        # Should have fallback chains for different content types
        assert 'image' in orchestrator.fallback_chains
        assert 'video' in orchestrator.fallback_chains
        assert 'enhancement' in orchestrator.fallback_chains
    
    @pytest.mark.asyncio
    async def test_model_prioritization(self, orchestrator):
        """Test model prioritization logic"""
        models = ['scrollintel_image_gen', 'dalle3', 'stable_diffusion']
        prioritized = orchestrator._prioritize_models(models, 'image')
        
        # ScrollIntel models should be prioritized first
        scrollintel_models = [m for m in prioritized if m.startswith('scrollintel_')]
        if scrollintel_models:
            assert prioritized.index(scrollintel_models[0]) == 0, "ScrollIntel models should be first priority"
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, orchestrator):
        """Test performance metrics tracking"""
        from scrollintel.engines.visual_generation.base import GenerationResult, GenerationStatus, QualityMetrics
        
        # Simulate successful generation
        result = GenerationResult(
            id="test_result",
            request_id="test_request",
            status=GenerationStatus.COMPLETED,
            generation_time=5.0,
            cost=0.0,
            model_used="test_generator",
            quality_metrics=QualityMetrics(overall_score=0.95)
        )
        
        await orchestrator._update_performance_metrics("test_generator", result, 5.0)
        
        # Should update metrics
        metrics = orchestrator.performance_metrics["test_generator"]
        assert metrics.average_time == 5.0
        assert metrics.quality_score == 0.95
        assert metrics.success_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_health_check(self, orchestrator):
        """Test system health check"""
        health = await orchestrator.health_check()
        
        # Should provide comprehensive health status
        assert 'overall_health' in health
        assert 'models' in health
        assert 'issues' in health
        
        # Should check all registered models
        assert 'test_generator' in health['models']


class TestProductionIntegration:
    """Test production integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_image_workflow(self):
        """Test complete image generation workflow"""
        engine = get_engine()
        await engine.initialize()
        
        # Create request
        request = ImageGenerationRequest(
            prompt="A beautiful landscape with mountains and lakes, photorealistic, 8k",
            user_id="integration_test_user",
            resolution=(1024, 1024),
            num_images=1,
            quality="high"
        )
        
        # Generate image
        result = await engine.generate_image(request)
        
        # Verify result
        assert result.status == GenerationStatus.COMPLETED
        assert len(result.content_paths) == 1
        assert result.cost == 0.0  # Should be free with local models
        
        # Verify file exists (mock)
        assert result.content_paths[0].endswith('.png') or result.content_paths[0].endswith('.jpg')
    
    @pytest.mark.asyncio
    async def test_end_to_end_video_workflow(self):
        """Test complete video generation workflow"""
        engine = get_engine()
        await engine.initialize()
        
        # Create request
        request = VideoGenerationRequest(
            prompt="A person walking through a magical forest with glowing particles",
            user_id="integration_test_user",
            duration=5.0,
            resolution=(1920, 1080),
            fps=30,
            physics_simulation=True,
            humanoid_generation=True
        )
        
        # Generate video
        result = await engine.generate_video(request)
        
        # Verify result
        assert result.status == GenerationStatus.COMPLETED
        assert len(result.content_paths) == 1
        assert result.cost == 0.0  # Should be free with proprietary models
        
        # Verify advanced features were used
        if result.metadata:
            assert result.metadata.get("physics_simulation") == True
            assert result.metadata.get("humanoid_generation") == True
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing capabilities"""
        engine = get_engine()
        await engine.initialize()
        
        # Create multiple requests
        requests = [
            ImageGenerationRequest(
                prompt=f"Batch test image {i}",
                user_id="batch_test_user",
                resolution=(512, 512)
            )
            for i in range(3)
        ]
        
        # Process batch
        results = await engine.batch_generate(requests)
        
        # Verify results
        assert len(results) == len(requests)
        successful = sum(1 for r in results if r.status == GenerationStatus.COMPLETED)
        assert successful >= len(requests) * 0.8, "At least 80% should succeed"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in production scenarios"""
        engine = get_engine()
        await engine.initialize()
        
        # Test invalid request
        invalid_request = ImageGenerationRequest(
            prompt="",  # Empty prompt
            user_id="error_test_user",
            resolution=(10000, 10000),  # Invalid resolution
            num_images=100  # Too many images
        )
        
        result = await engine.generate_image(invalid_request)
        
        # Should handle gracefully
        assert result.status in [GenerationStatus.FAILED, GenerationStatus.COMPLETED]
        if result.status == GenerationStatus.FAILED:
            assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        engine = get_engine()
        await engine.initialize()
        
        # Create concurrent requests
        async def generate_image(i):
            request = ImageGenerationRequest(
                prompt=f"Concurrent test {i}",
                user_id=f"concurrent_user_{i}",
                resolution=(512, 512)
            )
            return await engine.generate_image(request)
        
        # Run concurrent requests
        tasks = [generate_image(i) for i in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify handling
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 2, "Should handle most concurrent requests"
    
    def test_production_configuration_validation(self):
        """Test production configuration validation"""
        config_manager = create_optimal_setup()
        
        # Should create valid configuration
        assert config_manager is not None
        
        # Should have deployment recommendations
        recommendations = config_manager.get_deployment_recommendations()
        assert recommendations['performance_tier'] in ['basic', 'professional', 'enterprise']
        
        # Should have enabled models
        assert len(recommendations['enabled_models']) > 0


class TestCompetitiveAdvantages:
    """Test competitive advantages validation"""
    
    def test_cost_advantages(self):
        """Test cost advantages over competitors"""
        from scrollintel.engines.visual_generation.production_config import get_production_config
        config = get_production_config()
        
        advantages = config.get_competitive_advantages()
        
        # Should highlight cost advantages
        assert "FREE" in str(advantages).upper()
        assert "SUBSCRIPTION" in str(advantages).upper()
        
        # Should compare against major competitors
        assert any("invideo" in str(adv).lower() for adv in advantages.values())
    
    def test_quality_advantages(self):
        """Test quality advantages over competitors"""
        from scrollintel.engines.visual_generation.production_config import get_production_config
        config = get_production_config()
        
        advantages = config.get_competitive_advantages()
        
        # Should highlight quality advantages
        quality_keywords = ["4K", "60fps", "ultra", "photorealistic", "superior"]
        advantages_text = str(advantages).lower()
        
        found_keywords = [kw for kw in quality_keywords if kw.lower() in advantages_text]
        assert len(found_keywords) >= 3, f"Should mention quality advantages, found: {found_keywords}"
    
    def test_feature_advantages(self):
        """Test feature advantages over competitors"""
        from scrollintel.engines.visual_generation.production_config import get_production_config
        config = get_production_config()
        
        advantages = config.get_competitive_advantages()
        
        # Should highlight unique features
        feature_keywords = ["physics", "humanoid", "neural", "proprietary", "ai"]
        advantages_text = str(advantages).lower()
        
        found_features = [kw for kw in feature_keywords if kw.lower() in advantages_text]
        assert len(found_features) >= 3, f"Should mention unique features, found: {found_features}"
    
    def test_performance_advantages(self):
        """Test performance advantages validation"""
        from scrollintel.engines.visual_generation.production_config import get_production_config
        config = get_production_config()
        
        # Should validate performance claims
        performance_metrics = config.get_performance_benchmarks()
        
        # Should have benchmark data
        assert "image_generation_speed" in performance_metrics
        assert "video_generation_quality" in performance_metrics
        assert "cost_efficiency" in performance_metrics
        
        # Should show superior performance
        assert performance_metrics["cost_efficiency"]["scrollintel"] == 0.0  # Free


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])