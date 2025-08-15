"""
Demo script for Model Selector functionality.

Demonstrates model registration, selection strategies, performance tracking,
and A/B testing capabilities.
"""

import asyncio
import random
from datetime import datetime
from typing import List

from scrollintel.core.model_selector import (
    ModelSelector,
    ModelCapabilities,
    GenerationRequest,
    ModelType,
    QualityMetric,
    initialize_default_models
)


async def demo_basic_model_selection():
    """Demonstrate basic model selection functionality."""
    print("\n" + "="*60)
    print("🎯 DEMO: Basic Model Selection")
    print("="*60)
    
    # Initialize with default models
    selector = await initialize_default_models()
    
    print(f"📋 Registered models: {list(selector.model_capabilities.keys())}")
    
    # Create sample requests
    requests = [
        GenerationRequest(
            request_id="req_1",
            model_type=ModelType.IMAGE_GENERATION,
            prompt="A beautiful sunset over mountains",
            quality_preference="quality"
        ),
        GenerationRequest(
            request_id="req_2",
            model_type=ModelType.IMAGE_GENERATION,
            prompt="A cute cat playing with yarn",
            quality_preference="speed"
        ),
        GenerationRequest(
            request_id="req_3",
            model_type=ModelType.IMAGE_GENERATION,
            prompt="Abstract art with vibrant colors",
            quality_preference="balanced"
        )
    ]
    
    # Test different selection strategies
    strategies = ["performance", "cost", "quality"]
    
    for strategy in strategies:
        print(f"\n🔍 Testing {strategy.upper()} strategy:")
        
        for request in requests:
            selection = await selector.select_model(request, strategy)
            print(f"   Request: {request.prompt[:30]}...")
            print(f"   Selected: {selection.selected_model}")
            print(f"   Reason: {selection.selection_reason}")
            print(f"   Confidence: {selection.confidence_score:.2f}")
            print(f"   Est. Cost: ${selection.estimated_cost:.3f}")
            print(f"   Est. Time: {selection.estimated_time:.1f}s")
            print()


async def demo_performance_tracking():
    """Demonstrate performance metrics tracking."""
    print("\n" + "="*60)
    print("📊 DEMO: Performance Tracking")
    print("="*60)
    
    selector = await initialize_default_models()
    
    # Simulate generations and track performance
    models = list(selector.model_capabilities.keys())
    
    print("🔄 Simulating generations and tracking performance...")
    
    for i in range(20):
        # Random model and metrics
        model_id = random.choice(models)
        processing_time = random.uniform(10, 60)
        quality_score = random.uniform(0.6, 0.95)
        cost = random.uniform(0.02, 0.08)
        success = random.random() > 0.1  # 90% success rate
        user_satisfaction = random.uniform(0.5, 1.0) if success else 0.0
        
        await selector.update_model_metrics(
            model_id=model_id,
            processing_time=processing_time,
            quality_score=quality_score,
            cost=cost,
            success=success,
            user_satisfaction=user_satisfaction
        )
        
        if (i + 1) % 5 == 0:
            print(f"   Processed {i + 1} generations...")
    
    # Show updated metrics
    print("\n📈 Updated Model Performance:")
    for model_id in models:
        metrics = selector.model_metrics[model_id]
        print(f"\n   {model_id.upper()}:")
        print(f"     Total generations: {metrics.total_generations}")
        print(f"     Success rate: {metrics.success_rate:.1f}%")
        print(f"     Avg quality: {metrics.average_quality_score:.2f}")
        print(f"     Avg time: {metrics.average_processing_time:.1f}s")
        print(f"     Avg cost: ${metrics.average_cost:.3f}")
        print(f"     User satisfaction: {metrics.user_satisfaction_score:.2f}")
        print(f"     Efficiency score: {metrics.efficiency_score:.3f}")


async def demo_model_rankings():
    """Demonstrate model ranking functionality."""
    print("\n" + "="*60)
    print("🏆 DEMO: Model Rankings")
    print("="*60)
    
    selector = await initialize_default_models()
    
    # Add some varied performance data
    performance_data = {
        "dalle3": {"quality": 0.9, "time": 15.0, "satisfaction": 0.85},
        "stable_diffusion_xl": {"quality": 0.8, "time": 25.0, "satisfaction": 0.75},
        "midjourney": {"quality": 0.95, "time": 45.0, "satisfaction": 0.9}
    }
    
    for model_id, data in performance_data.items():
        for _ in range(10):  # Multiple data points
            await selector.update_model_metrics(
                model_id=model_id,
                processing_time=data["time"] + random.uniform(-5, 5),
                quality_score=data["quality"] + random.uniform(-0.1, 0.1),
                cost=random.uniform(0.02, 0.05),
                success=True,
                user_satisfaction=data["satisfaction"] + random.uniform(-0.1, 0.1)
            )
    
    # Get rankings for different metrics
    metrics = [
        QualityMetric.OVERALL_SCORE,
        QualityMetric.PROCESSING_TIME,
        QualityMetric.USER_SATISFACTION
    ]
    
    for metric in metrics:
        rankings = await selector.get_model_rankings(ModelType.IMAGE_GENERATION, metric)
        print(f"\n🏅 Rankings by {metric.value.replace('_', ' ').title()}:")
        
        for i, (model_id, score) in enumerate(rankings, 1):
            print(f"   {i}. {model_id}: {score:.3f}")


async def demo_ab_testing():
    """Demonstrate A/B testing functionality."""
    print("\n" + "="*60)
    print("🧪 DEMO: A/B Testing")
    print("="*60)
    
    selector = await initialize_default_models()
    
    # Create A/B test
    test_id = "dalle3_vs_sdxl"
    success = await selector.create_ab_test(
        test_id=test_id,
        model_a="dalle3",
        model_b="stable_diffusion_xl",
        traffic_split=0.6  # 60% to dalle3, 40% to stable_diffusion_xl
    )
    
    if success:
        print(f"✅ Created A/B test: {test_id}")
    else:
        print(f"❌ Failed to create A/B test: {test_id}")
        return
    
    # Simulate requests during A/B test
    print("\n🔄 Simulating requests during A/B test...")
    
    assignments = {"dalle3": 0, "stable_diffusion_xl": 0}
    
    for i in range(20):
        request = GenerationRequest(
            request_id=f"ab_test_req_{i}",
            model_type=ModelType.IMAGE_GENERATION,
            prompt=f"Test prompt {i}"
        )
        
        # Selection should use A/B test assignment
        selection = await selector.select_model(request)
        assignments[selection.selected_model] += 1
        
        # Simulate generation results
        processing_time = random.uniform(15, 35)
        quality_score = random.uniform(0.7, 0.9)
        
        await selector.update_model_metrics(
            model_id=selection.selected_model,
            processing_time=processing_time,
            quality_score=quality_score,
            cost=random.uniform(0.02, 0.05),
            success=True,
            user_satisfaction=random.uniform(0.6, 0.9)
        )
        
        if (i + 1) % 5 == 0:
            print(f"   Processed {i + 1} A/B test requests...")
    
    print(f"\n📊 A/B Test Assignments:")
    print(f"   DALL-E 3: {assignments['dalle3']} requests")
    print(f"   Stable Diffusion XL: {assignments['stable_diffusion_xl']} requests")
    
    # Get test status
    test_status = await selector.get_ab_test_status(test_id)
    if test_status:
        print(f"\n📈 A/B Test Status:")
        print(f"   Status: {test_status['status']}")
        print(f"   Model A requests: {test_status['request_count_a']}")
        print(f"   Model B requests: {test_status['request_count_b']}")


async def demo_custom_model_registration():
    """Demonstrate registering custom models."""
    print("\n" + "="*60)
    print("🔧 DEMO: Custom Model Registration")
    print("="*60)
    
    selector = ModelSelector()
    
    # Register custom models
    custom_models = [
        ModelCapabilities(
            model_id="fast_model",
            model_type=ModelType.IMAGE_GENERATION,
            supported_resolutions=[(512, 512), (1024, 1024)],
            supported_formats=["jpg", "png"],
            max_prompt_length=500,
            supports_negative_prompts=True,
            gpu_memory_required=2.0,
            estimated_processing_time=8.0,
            cost_per_generation=0.01
        ),
        ModelCapabilities(
            model_id="premium_model",
            model_type=ModelType.IMAGE_GENERATION,
            supported_resolutions=[(1024, 1024), (2048, 2048), (4096, 4096)],
            supported_formats=["jpg", "png", "tiff"],
            max_prompt_length=5000,
            supports_negative_prompts=True,
            supports_style_control=True,
            supports_batch_processing=True,
            gpu_memory_required=16.0,
            estimated_processing_time=120.0,
            cost_per_generation=0.50
        )
    ]
    
    for model in custom_models:
        await selector.register_model(model)
        print(f"✅ Registered: {model.model_id}")
        print(f"   Type: {model.model_type.value}")
        print(f"   Resolutions: {model.supported_resolutions}")
        print(f"   Cost: ${model.cost_per_generation:.3f}")
        print(f"   Est. Time: {model.estimated_processing_time:.1f}s")
        print()
    
    # Test selection with custom models
    request = GenerationRequest(
        request_id="custom_test",
        model_type=ModelType.IMAGE_GENERATION,
        prompt="High quality artistic image",
        quality_preference="speed"
    )
    
    selection = await selector.select_model(request, "cost")
    print(f"🎯 Selected for speed preference: {selection.selected_model}")
    print(f"   Reason: {selection.selection_reason}")
    
    # Test with quality preference
    request.quality_preference = "quality"
    selection = await selector.select_model(request, "quality")
    print(f"🎯 Selected for quality preference: {selection.selected_model}")
    print(f"   Reason: {selection.selection_reason}")


async def demo_selection_analytics():
    """Demonstrate selection analytics."""
    print("\n" + "="*60)
    print("📊 DEMO: Selection Analytics")
    print("="*60)
    
    selector = await initialize_default_models()
    
    # Make various selections
    strategies = ["performance", "cost", "quality"]
    
    print("🔄 Making various model selections...")
    
    for i in range(30):
        request = GenerationRequest(
            request_id=f"analytics_req_{i}",
            model_type=ModelType.IMAGE_GENERATION,
            prompt=f"Analytics test prompt {i}",
            quality_preference=random.choice(["speed", "balanced", "quality"])
        )
        
        strategy = random.choice(strategies)
        selection = await selector.select_model(request, strategy)
        
        if (i + 1) % 10 == 0:
            print(f"   Made {i + 1} selections...")
    
    # Get analytics
    analytics = await selector.get_selection_analytics()
    
    print(f"\n📈 Selection Analytics:")
    print(f"   Total selections: {analytics['total_selections']}")
    print(f"   Most selected model: {analytics.get('most_selected_model', 'N/A')}")
    print(f"   Most used strategy: {analytics.get('most_used_strategy', 'N/A')}")
    
    print(f"\n📊 Model Usage:")
    for model, count in analytics.get('model_usage', {}).items():
        percentage = (count / analytics['total_selections']) * 100
        print(f"   {model}: {count} times ({percentage:.1f}%)")
    
    print(f"\n🎯 Strategy Usage:")
    for strategy, count in analytics.get('strategy_usage', {}).items():
        percentage = (count / analytics['total_selections']) * 100
        print(f"   {strategy}: {count} times ({percentage:.1f}%)")


async def main():
    """Run all demos."""
    print("🎭 Model Selector Demo Suite")
    print("=" * 60)
    
    demos = [
        demo_basic_model_selection,
        demo_performance_tracking,
        demo_model_rankings,
        demo_ab_testing,
        demo_custom_model_registration,
        demo_selection_analytics
    ]
    
    for i, demo in enumerate(demos, 1):
        try:
            await demo()
            print(f"\n✅ Demo {i}/{len(demos)} completed successfully!")
        except Exception as e:
            print(f"\n❌ Demo {i}/{len(demos)} failed: {e}")
        
        if i < len(demos):
            print("\n" + "⏳ Waiting 2 seconds before next demo...")
            await asyncio.sleep(2)
    
    print("\n" + "="*60)
    print("🎉 All demos completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())