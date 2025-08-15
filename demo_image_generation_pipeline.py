"""
Demo script for the unified image generation pipeline.

This script demonstrates the ImageGenerationPipeline's capabilities including:
- Model selection strategies
- Single and multi-model generation
- Result comparison and quality assessment
- Cost and time estimation
"""

import asyncio
import logging
from datetime import datetime
from scrollintel.engines.visual_generation.pipeline import (
    ImageGenerationPipeline,
    ModelSelectionStrategy
)
from scrollintel.engines.visual_generation.base import ImageGenerationRequest
from scrollintel.engines.visual_generation.config import VisualGenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_pipeline_capabilities():
    """Demonstrate the pipeline's key capabilities."""
    
    print("🎨 ScrollIntel Image Generation Pipeline Demo")
    print("=" * 50)
    
    # Create configuration
    config_data = {
        'models': {
            'stable_diffusion_xl': {
                'type': 'image',
                'enabled': True,
                'api_key': 'demo_key',
                'parameters': {
                    'cfg_scale': 7.5,
                    'steps': 50
                }
            },
            'dalle3': {
                'type': 'image',
                'enabled': True,
                'api_key': 'demo_key',
                'parameters': {
                    'quality': 'hd',
                    'style': 'vivid'
                }
            },
            'midjourney': {
                'type': 'image',
                'enabled': True,
                'api_key': 'demo_key',
                'parameters': {
                    'version': '6',
                    'quality': '1'
                }
            }
        },
        'infrastructure': {
            'gpu_enabled': True,
            'max_concurrent_requests': 10,
            'cache_enabled': True,
            'storage_path': './demo_generated_content',
            'temp_path': './demo_temp'
        },
        'safety': {
            'enabled': True,
            'nsfw_detection': True
        },
        'quality': {
            'enabled': True,
            'min_quality_score': 0.7
        },
        'cost': {
            'enabled': True,
            'cost_per_image': 0.01
        }
    }
    
    config = VisualGenerationConfig(config_data)
    
    # Initialize pipeline
    print("\n🚀 Initializing Image Generation Pipeline...")
    pipeline = ImageGenerationPipeline(config)
    
    # Note: In a real implementation, this would initialize actual models
    # For demo purposes, we'll show the pipeline structure
    print(f"✅ Pipeline initialized with configuration")
    print(f"   Available models: {list(config.models.keys())}")
    
    # Show pipeline capabilities
    print("\n📋 Pipeline Capabilities:")
    capabilities = pipeline.get_capabilities()
    for key, value in capabilities.items():
        if isinstance(value, list):
            print(f"   {key}: {', '.join(value) if value else 'None'}")
        elif isinstance(value, dict):
            print(f"   {key}: {len(value)} items")
        else:
            print(f"   {key}: {value}")
    
    # Demo different selection strategies
    print("\n🎯 Model Selection Strategies Demo:")
    
    sample_request = ImageGenerationRequest(
        prompt="A majestic mountain landscape at sunset with dramatic clouds",
        user_id="demo_user",
        resolution=(1024, 1024),
        num_images=1,
        style="photorealistic",
        quality="high"
    )
    
    strategies = [
        (ModelSelectionStrategy.BALANCED, "Balanced performance"),
        (ModelSelectionStrategy.BEST_QUALITY, "Highest quality"),
        (ModelSelectionStrategy.FASTEST, "Fastest generation"),
        (ModelSelectionStrategy.MOST_COST_EFFECTIVE, "Most cost-effective"),
        (ModelSelectionStrategy.ALL_MODELS, "Compare all models")
    ]
    
    for strategy, description in strategies:
        try:
            selection = await pipeline.model_selector.select_models(sample_request, strategy)
            print(f"\n   📌 {description}:")
            print(f"      Selected: {', '.join(selection.selected_models)}")
            print(f"      Reason: {selection.selection_reason}")
            print(f"      Confidence: {selection.confidence_score:.2f}")
            print(f"      Estimated cost: ${selection.estimated_cost:.3f}")
            print(f"      Estimated time: {selection.estimated_time:.1f}s")
            
            if selection.fallback_models:
                print(f"      Fallbacks: {', '.join(selection.fallback_models)}")
                
        except Exception as e:
            print(f"   ❌ {description}: {e}")
    
    # Demo request validation
    print("\n✅ Request Validation Demo:")
    
    test_requests = [
        ImageGenerationRequest(
            prompt="A beautiful sunset",
            user_id="demo_user",
            resolution=(1024, 1024)
        ),
        ImageGenerationRequest(
            prompt="Test with extreme resolution",
            user_id="demo_user",
            resolution=(10000, 10000)  # Unsupported
        ),
        ImageGenerationRequest(
            prompt="",  # Empty prompt
            user_id="demo_user",
            resolution=(1024, 1024)
        )
    ]
    
    for i, request in enumerate(test_requests, 1):
        try:
            is_valid = await pipeline.validate_request(request)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"   Request {i}: {status}")
            print(f"      Prompt: '{request.prompt[:50]}{'...' if len(request.prompt) > 50 else ''}'")
            print(f"      Resolution: {request.resolution}")
            
        except Exception as e:
            print(f"   Request {i}: ❌ Error - {e}")
    
    # Demo cost and time estimation
    print("\n💰 Cost & Time Estimation Demo:")
    
    estimation_requests = [
        ("Single image, standard quality", ImageGenerationRequest(
            prompt="A cat", user_id="demo", resolution=(1024, 1024), num_images=1, quality="standard"
        )),
        ("Multiple images, high quality", ImageGenerationRequest(
            prompt="A dog", user_id="demo", resolution=(1024, 1024), num_images=4, quality="high"
        )),
        ("Large resolution", ImageGenerationRequest(
            prompt="A landscape", user_id="demo", resolution=(1792, 1024), num_images=1, quality="high"
        ))
    ]
    
    for description, request in estimation_requests:
        try:
            cost = await pipeline.estimate_cost(request)
            time_est = await pipeline.estimate_time(request)
            
            print(f"\n   📊 {description}:")
            print(f"      Estimated cost: ${cost:.3f}")
            print(f"      Estimated time: {time_est:.1f}s")
            print(f"      Images: {request.num_images}")
            print(f"      Resolution: {request.resolution}")
            print(f"      Quality: {request.quality}")
            
        except Exception as e:
            print(f"   ❌ {description}: {e}")
    
    # Demo model capabilities comparison
    print("\n🔍 Model Capabilities Comparison:")
    
    model_capabilities = pipeline.model_selector.model_capabilities
    
    print(f"\n   {'Model':<20} {'Quality':<8} {'Speed':<8} {'Cost':<8} {'Availability':<12}")
    print(f"   {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    
    for model_name, capability in model_capabilities.items():
        print(f"   {model_name:<20} {capability.quality_score:<8.2f} "
              f"{capability.speed_score:<8.2f} {capability.cost_score:<8.2f} "
              f"{capability.availability_score:<12.2f}")
    
    # Demo style compatibility
    print("\n🎨 Style Compatibility Matrix:")
    
    styles = ['photorealistic', 'artistic', 'professional', 'creative', 'abstract']
    
    print(f"\n   {'Model':<20}", end="")
    for style in styles:
        print(f"{style:<12}", end="")
    print()
    
    print(f"   {'-'*20}", end="")
    for _ in styles:
        print(f"{'-'*12}", end="")
    print()
    
    for model_name, capability in model_capabilities.items():
        print(f"   {model_name:<20}", end="")
        for style in styles:
            score = capability.style_compatibility.get(style, 0.0)
            print(f"{score:<12.2f}", end="")
        print()
    
    # Demo user preference handling
    print("\n👤 User Preference Demo:")
    
    preference_request = ImageGenerationRequest(
        prompt="A futuristic cityscape",
        user_id="demo_user",
        resolution=(1024, 1024),
        model_preference="dalle3"  # User prefers DALL-E 3
    )
    
    try:
        selection = await pipeline.model_selector.select_models(
            preference_request, 
            ModelSelectionStrategy.USER_PREFERENCE
        )
        
        print(f"   User requested: {preference_request.model_preference}")
        print(f"   Selected: {', '.join(selection.selected_models)}")
        print(f"   Reason: {selection.selection_reason}")
        
    except Exception as e:
        print(f"   ❌ User preference handling failed: {e}")
    
    # Demo metadata-based strategy selection
    print("\n🎛️ Metadata-Based Strategy Selection:")
    
    metadata_tests = [
        ({"priority": "speed"}, "Speed priority"),
        ({"priority": "cost"}, "Cost priority"),
        ({"compare_models": True}, "Model comparison"),
        ({}, "Default strategy")
    ]
    
    for metadata, description in metadata_tests:
        request = ImageGenerationRequest(
            prompt="Test prompt",
            user_id="demo_user",
            resolution=(1024, 1024),
            metadata=metadata
        )
        
        strategy = pipeline._determine_strategy(request)
        print(f"   {description}: {strategy.value}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nThe ImageGenerationPipeline provides:")
    print("✅ Intelligent model selection based on request parameters")
    print("✅ Multiple selection strategies (quality, speed, cost, balanced)")
    print("✅ Automatic fallback handling when models fail")
    print("✅ Multi-model comparison and result aggregation")
    print("✅ Cost and time estimation")
    print("✅ Request validation and error handling")
    print("✅ User preference support")
    print("✅ Comprehensive logging and monitoring")


async def demo_generation_workflow():
    """Demonstrate a complete generation workflow (simulated)."""
    
    print("\n" + "="*60)
    print("🔄 Complete Generation Workflow Demo (Simulated)")
    print("="*60)
    
    # This would be a real generation in production
    print("\n📝 Workflow Steps:")
    print("1. ✅ Request received and validated")
    print("2. ✅ Model selection strategy determined")
    print("3. ✅ Optimal model(s) selected based on request parameters")
    print("4. ✅ Cost and time estimates calculated")
    print("5. ✅ Safety filters applied to prompt")
    print("6. ✅ Generation request sent to selected model(s)")
    print("7. ✅ Results received and quality assessed")
    print("8. ✅ Best result selected (if multiple models used)")
    print("9. ✅ Result cached for future similar requests")
    print("10. ✅ Final result returned to user")
    
    print("\n🎯 Key Benefits:")
    print("• Automatic model selection optimizes for quality, speed, or cost")
    print("• Fallback handling ensures high availability")
    print("• Multi-model comparison provides best possible results")
    print("• Intelligent caching reduces costs and improves response times")
    print("• Comprehensive error handling and logging")
    print("• Scalable architecture supports multiple concurrent requests")


if __name__ == "__main__":
    print("Starting Image Generation Pipeline Demo...")
    
    try:
        asyncio.run(demo_pipeline_capabilities())
        asyncio.run(demo_generation_workflow())
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Thank you for trying the ScrollIntel Image Generation Pipeline!")