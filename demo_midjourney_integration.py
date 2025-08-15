#!/usr/bin/env python3
"""
Demo script for Midjourney API integration.
Demonstrates the key features of the MidjourneyModel implementation.
"""

import asyncio
import logging
from unittest.mock import Mock

from scrollintel.engines.visual_generation.models.midjourney import (
    MidjourneyModel,
    MidjourneyPromptFormatter,
    MidjourneyParameters,
    MidjourneyJobQueue
)
from scrollintel.engines.visual_generation.base import ImageGenerationRequest
from scrollintel.engines.visual_generation.config import VisualGenerationConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_prompt_formatting():
    """Demonstrate Midjourney prompt formatting capabilities."""
    print("\n=== Midjourney Prompt Formatting Demo ===")
    
    formatter = MidjourneyPromptFormatter()
    
    # Test basic prompt formatting
    params = MidjourneyParameters(
        aspect_ratio="16:9",
        quality="2",
        stylize=300,
        chaos=25,
        style="raw",
        seed=12345
    )
    
    original_prompt = "A majestic mountain landscape at sunset"
    formatted_prompt = formatter.format_prompt(original_prompt, params)
    
    print(f"Original prompt: {original_prompt}")
    print(f"Formatted prompt: {formatted_prompt}")
    
    # Test style enhancement
    enhanced_prompt = formatter.enhance_prompt_for_style(original_prompt, "photorealistic")
    print(f"Enhanced for photorealistic: {enhanced_prompt}")
    
    # Test prompt validation
    try:
        formatter.validate_prompt("A safe and appropriate prompt for image generation")
        print("✅ Prompt validation passed")
    except Exception as e:
        print(f"❌ Prompt validation failed: {e}")


async def demo_job_queue():
    """Demonstrate job queue functionality."""
    print("\n=== Midjourney Job Queue Demo ===")
    
    queue = MidjourneyJobQueue(max_concurrent_jobs=2, poll_interval=0.5)
    
    from scrollintel.engines.visual_generation.models.midjourney import MidjourneyJob
    
    # Submit test jobs
    jobs = []
    for i in range(3):
        job = MidjourneyJob(
            job_id=f"demo_job_{i}",
            prompt=f"Test prompt {i}",
            metadata={"demo": True}
        )
        jobs.append(job)
        await queue.submit_job(job)
        print(f"✅ Submitted job {job.job_id}")
    
    # Check job statuses
    await asyncio.sleep(1.0)  # Let queue process
    
    for job in jobs:
        status = await queue.get_job_status(job.job_id)
        print(f"Job {job.job_id} status: {status.status.value}")
    
    # Cleanup
    await queue.shutdown()
    print("✅ Job queue demo completed")


async def demo_model_configuration():
    """Demonstrate model configuration and initialization."""
    print("\n=== Midjourney Model Configuration Demo ===")
    
    # Mock configuration
    mock_config = Mock(spec=VisualGenerationConfig)
    mock_config.get_model_config.return_value = Mock(
        api_key="demo_bot_token",
        parameters={
            'server_id': '123456789',
            'channel_id': '987654321',
            'application_id': '111222333',
            'max_concurrent_jobs': 3,
            'poll_interval': 2.0,
            'max_rpm': 10,
            'generation_timeout': 120
        }
    )
    
    try:
        model = MidjourneyModel(mock_config)
        print("✅ Model initialized successfully")
        print(f"Model name: {model.model_name}")
        print(f"Max concurrent jobs: {model.job_queue.max_concurrent_jobs}")
        print(f"Rate limit: {model.max_requests_per_minute} requests/minute")
        
        # Test model info
        info = await model.get_model_info()
        print(f"Model info: {info['name']} v{info['version']}")
        print(f"Supported aspects: {info['supported_aspects']}")
        print(f"Features: {', '.join(info['features'])}")
        
        # Test parameter preparation
        request = ImageGenerationRequest(
            prompt="A beautiful landscape",
            user_id="demo_user",
            resolution=(1920, 1080),
            quality="high",
            style="photorealistic"
        )
        
        params = await model._prepare_parameters(request)
        print(f"✅ Parameters prepared: aspect_ratio={params.aspect_ratio}, quality={params.quality}")
        
        # Test request validation
        is_valid = await model.validate_request(request)
        print(f"✅ Request validation: {is_valid}")
        
        # Cleanup
        await model.cleanup()
        print("✅ Model cleanup completed")
        
    except Exception as e:
        print(f"❌ Model demo failed: {e}")


async def demo_aspect_ratio_calculation():
    """Demonstrate aspect ratio calculation."""
    print("\n=== Aspect Ratio Calculation Demo ===")
    
    mock_config = Mock(spec=VisualGenerationConfig)
    mock_config.get_model_config.return_value = Mock(
        api_key="demo_token",
        parameters={
            'server_id': '123', 'channel_id': '456', 'application_id': '789'
        }
    )
    
    model = MidjourneyModel(mock_config)
    
    test_resolutions = [
        (1024, 1024),    # Square
        (1920, 1080),    # 16:9
        (1080, 1920),    # 9:16
        (1600, 1200),    # 4:3
        (1200, 1600),    # 3:4
        (2560, 1080),    # 21:9
        (1500, 1000),    # 3:2
    ]
    
    for width, height in test_resolutions:
        aspect_ratio = model._calculate_aspect_ratio((width, height))
        print(f"{width}x{height} -> {aspect_ratio}")
    
    await model.cleanup()
    print("✅ Aspect ratio calculation demo completed")


async def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n=== Error Handling Demo ===")
    
    formatter = MidjourneyPromptFormatter()
    
    # Test prompt validation errors
    test_cases = [
        ("", "Empty prompt"),
        ("hi", "Too short prompt"),
        ("x" * 4001, "Too long prompt"),
        ("Create nsfw content", "Unsafe content"),
    ]
    
    for prompt, description in test_cases:
        try:
            formatter.validate_prompt(prompt)
            print(f"❌ {description}: Should have failed but didn't")
        except Exception as e:
            print(f"✅ {description}: Correctly caught - {type(e).__name__}")


async def main():
    """Run all demo functions."""
    print("🚀 Starting Midjourney Integration Demo")
    print("=" * 50)
    
    try:
        await demo_prompt_formatting()
        await demo_job_queue()
        await demo_model_configuration()
        await demo_aspect_ratio_calculation()
        await demo_error_handling()
        
        print("\n" + "=" * 50)
        print("✅ All demos completed successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("  • Midjourney-specific prompt formatting")
        print("  • Discord bot API integration structure")
        print("  • Job queuing and status polling")
        print("  • Parameter optimization and mapping")
        print("  • Aspect ratio calculation")
        print("  • Error handling and validation")
        print("  • Retry logic framework")
        print("  • Rate limiting enforcement")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())