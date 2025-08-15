"""
Demo script for usage tracking system in visual content generation.
"""

import asyncio
import json
from datetime import datetime, timedelta
from scrollintel.engines.usage_tracker import UsageTracker
from scrollintel.models.usage_tracking_models import GenerationType, ResourceType


async def demo_basic_usage_tracking():
    """Demonstrate basic usage tracking functionality."""
    print("=== Basic Usage Tracking Demo ===")
    
    tracker = UsageTracker()
    user_id = "demo_user_001"
    
    # Start tracking an image generation
    print(f"\n1. Starting image generation tracking for user: {user_id}")
    session_id = await tracker.start_generation_tracking(
        user_id=user_id,
        generation_type=GenerationType.IMAGE,
        model_used="stable_diffusion_xl",
        prompt="A photorealistic portrait of a person in natural lighting",
        parameters={
            "resolution": (1024, 1024),
            "steps": 50,
            "guidance_scale": 7.5,
            "style": "photorealistic"
        }
    )
    print(f"   Session ID: {session_id}")
    
    # Track various resources during generation
    print("\n2. Tracking resource usage during generation...")
    
    # API call to start generation
    await tracker.track_resource_usage(
        session_id=session_id,
        resource_type=ResourceType.API_CALLS,
        amount=1,
        metadata={"endpoint": "/generate", "method": "POST"}
    )
    print("   ✓ Tracked API call")
    
    # GPU computation time
    await tracker.track_resource_usage(
        session_id=session_id,
        resource_type=ResourceType.GPU_SECONDS,
        amount=8.5,
        metadata={"gpu_type": "A100", "model_loading_time": 2.0}
    )
    print("   ✓ Tracked GPU usage: 8.5 seconds")
    
    # CPU processing time
    await tracker.track_resource_usage(
        session_id=session_id,
        resource_type=ResourceType.CPU_SECONDS,
        amount=15.2,
        metadata={"preprocessing": 3.2, "postprocessing": 12.0}
    )
    print("   ✓ Tracked CPU usage: 15.2 seconds")
    
    # Storage for generated image
    await tracker.track_resource_usage(
        session_id=session_id,
        resource_type=ResourceType.STORAGE_GB,
        amount=0.05,
        metadata={"format": "PNG", "compression": "lossless"}
    )
    print("   ✓ Tracked storage: 0.05 GB")
    
    # End tracking with success
    print("\n3. Completing generation tracking...")
    completed_session = await tracker.end_generation_tracking(
        session_id=session_id,
        success=True,
        quality_score=0.92,
        error_message=None
    )
    
    print(f"   ✓ Generation completed successfully")
    print(f"   Duration: {completed_session.duration_seconds:.2f} seconds")
    print(f"   Total cost: ${completed_session.total_cost:.4f}")
    print(f"   Quality score: {completed_session.quality_score}")
    
    return tracker, user_id


async def demo_video_generation_tracking():
    """Demonstrate video generation tracking with higher costs."""
    print("\n=== Video Generation Tracking Demo ===")
    
    tracker = UsageTracker()
    user_id = "demo_user_002"
    
    # Start tracking video generation
    print(f"\n1. Starting video generation tracking for user: {user_id}")
    session_id = await tracker.start_generation_tracking(
        user_id=user_id,
        generation_type=GenerationType.VIDEO,
        model_used="runway_ml",
        prompt="A cinematic shot of a person walking through a futuristic city",
        parameters={
            "duration": 5.0,
            "fps": 24,
            "resolution": (1280, 720),
            "style": "cinematic"
        }
    )
    
    # Track extensive resource usage for video
    print("\n2. Tracking intensive video generation resources...")
    
    # Multiple API calls for video frames
    await tracker.track_resource_usage(
        session_id=session_id,
        resource_type=ResourceType.API_CALLS,
        amount=120,  # 5 seconds * 24 fps
        metadata={"frame_generation": True}
    )
    
    # Heavy GPU usage for video generation
    await tracker.track_resource_usage(
        session_id=session_id,
        resource_type=ResourceType.GPU_SECONDS,
        amount=45.0,
        metadata={"gpu_type": "A100", "video_processing": True}
    )
    
    # CPU for video encoding
    await tracker.track_resource_usage(
        session_id=session_id,
        resource_type=ResourceType.CPU_SECONDS,
        amount=30.0,
        metadata={"video_encoding": True, "codec": "H.264"}
    )
    
    # Large storage for video file
    await tracker.track_resource_usage(
        session_id=session_id,
        resource_type=ResourceType.STORAGE_GB,
        amount=0.8,
        metadata={"format": "MP4", "bitrate": "10Mbps"}
    )
    
    # Bandwidth for delivery
    await tracker.track_resource_usage(
        session_id=session_id,
        resource_type=ResourceType.BANDWIDTH_GB,
        amount=0.8,
        metadata={"delivery": "CDN"}
    )
    
    # Complete video generation
    completed_session = await tracker.end_generation_tracking(
        session_id=session_id,
        success=True,
        quality_score=0.88
    )
    
    print(f"   ✓ Video generation completed")
    print(f"   Duration: {completed_session.duration_seconds:.2f} seconds")
    print(f"   Total cost: ${completed_session.total_cost:.4f}")
    print(f"   Quality score: {completed_session.quality_score}")
    
    return tracker, user_id


async def demo_failed_generation():
    """Demonstrate tracking of failed generation."""
    print("\n=== Failed Generation Tracking Demo ===")
    
    tracker = UsageTracker()
    user_id = "demo_user_003"
    
    session_id = await tracker.start_generation_tracking(
        user_id=user_id,
        generation_type=GenerationType.IMAGE,
        model_used="unstable_model",
        prompt="An impossible geometric shape that defies physics"
    )
    
    # Track some resource usage before failure
    await tracker.track_resource_usage(
        session_id=session_id,
        resource_type=ResourceType.GPU_SECONDS,
        amount=3.0
    )
    
    # End with failure
    completed_session = await tracker.end_generation_tracking(
        session_id=session_id,
        success=False,
        quality_score=None,
        error_message="Model failed to generate coherent output"
    )
    
    print(f"   ✗ Generation failed")
    print(f"   Error: {completed_session.error_message}")
    print(f"   Wasted cost: ${completed_session.total_cost:.4f}")
    
    return tracker, user_id


async def demo_usage_analytics():
    """Demonstrate usage analytics and reporting."""
    print("\n=== Usage Analytics Demo ===")
    
    # Create tracker with some historical data
    tracker = UsageTracker()
    user_id = "analytics_demo_user"
    
    # Generate multiple sessions with different patterns
    print("\n1. Generating sample usage data...")
    
    session_data = [
        (GenerationType.IMAGE, "stable_diffusion_xl", 5.0, True, 0.9),
        (GenerationType.IMAGE, "dalle3", 3.0, True, 0.85),
        (GenerationType.VIDEO, "runway_ml", 25.0, True, 0.88),
        (GenerationType.IMAGE, "midjourney", 8.0, False, None),
        (GenerationType.ENHANCEMENT, "real_esrgan", 2.0, True, 0.75),
        (GenerationType.IMAGE, "stable_diffusion_xl", 6.0, True, 0.92),
        (GenerationType.VIDEO, "pika_labs", 20.0, True, 0.80),
        (GenerationType.IMAGE, "dalle3", 4.0, False, None),
    ]
    
    for gen_type, model, gpu_time, success, quality in session_data:
        session_id = await tracker.start_generation_tracking(
            user_id=user_id,
            generation_type=gen_type,
            model_used=model,
            prompt=f"Test prompt for {model}"
        )
        
        await tracker.track_resource_usage(
            session_id=session_id,
            resource_type=ResourceType.GPU_SECONDS,
            amount=gpu_time
        )
        
        if gen_type == GenerationType.VIDEO:
            await tracker.track_resource_usage(
                session_id=session_id,
                resource_type=ResourceType.STORAGE_GB,
                amount=0.5
            )
        
        await tracker.end_generation_tracking(
            session_id=session_id,
            success=success,
            quality_score=quality
        )
    
    # Get usage summary
    print("\n2. Generating usage summary...")
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    
    summary = await tracker.get_user_usage_summary(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"   Total generations: {summary.total_generations}")
    print(f"   Successful: {summary.successful_generations}")
    print(f"   Failed: {summary.failed_generations}")
    print(f"   Success rate: {(summary.successful_generations/summary.total_generations)*100:.1f}%")
    print(f"   Image generations: {summary.image_generations}")
    print(f"   Video generations: {summary.video_generations}")
    print(f"   Enhancement operations: {summary.enhancement_operations}")
    print(f"   Total GPU seconds: {summary.total_gpu_seconds}")
    print(f"   Total cost: ${summary.total_cost:.4f}")
    print(f"   Average cost per generation: ${summary.average_cost_per_generation:.4f}")
    print(f"   Average quality score: {summary.average_quality_score:.2f}")
    print(f"   Average generation time: {summary.average_generation_time:.2f}s")
    
    return tracker, user_id


async def demo_budget_monitoring():
    """Demonstrate budget monitoring and alerts."""
    print("\n=== Budget Monitoring Demo ===")
    
    tracker = UsageTracker()
    user_id = "budget_demo_user"
    budget_limit = 5.00  # $5 budget
    
    print(f"\n1. Setting up budget monitoring (${budget_limit} limit)")
    
    # Create sessions that will exceed budget thresholds
    expensive_sessions = [
        ("High-res portrait", 30.0),  # $1.50
        ("Cinematic landscape", 35.0),  # $1.75
        ("Abstract art piece", 40.0),   # $2.00
    ]
    
    total_spent = 0
    for prompt, gpu_time in expensive_sessions:
        session_id = await tracker.start_generation_tracking(
            user_id=user_id,
            generation_type=GenerationType.IMAGE,
            model_used="premium_model",
            prompt=prompt
        )
        
        await tracker.track_resource_usage(
            session_id=session_id,
            resource_type=ResourceType.GPU_SECONDS,
            amount=gpu_time
        )
        
        session = await tracker.end_generation_tracking(session_id=session_id)
        total_spent += session.total_cost
        
        print(f"   Generated: {prompt} - Cost: ${session.total_cost:.2f}")
    
    print(f"\n2. Total spent: ${total_spent:.2f} of ${budget_limit:.2f} budget")
    
    # Check budget alerts
    alerts = await tracker.check_budget_alerts(
        user_id=user_id,
        budget_limit=budget_limit,
        period_days=30
    )
    
    print(f"\n3. Budget alerts triggered: {len(alerts)}")
    for alert in alerts:
        usage_pct = (alert.current_usage / alert.budget_limit) * 100
        print(f"   🚨 {alert.alert_type}: {usage_pct:.1f}% of budget used")
        print(f"      Current usage: ${alert.current_usage:.2f}")
        print(f"      Budget limit: ${alert.budget_limit:.2f}")


async def demo_cost_optimization():
    """Demonstrate cost optimization recommendations."""
    print("\n=== Cost Optimization Demo ===")
    
    tracker = UsageTracker()
    user_id = "optimization_demo_user"
    
    print("\n1. Creating suboptimal usage patterns...")
    
    # Create patterns that will trigger optimization recommendations
    
    # High-cost, low-quality model usage
    for i in range(6):
        session_id = await tracker.start_generation_tracking(
            user_id=user_id,
            generation_type=GenerationType.IMAGE,
            model_used="expensive_low_quality_model",
            prompt=f"Test image {i}"
        )
        
        await tracker.track_resource_usage(
            session_id=session_id,
            resource_type=ResourceType.GPU_SECONDS,
            amount=25.0  # Expensive
        )
        
        await tracker.end_generation_tracking(
            session_id=session_id,
            success=True,
            quality_score=0.4  # Low quality
        )
    
    # Failed generations (wasted resources)
    for i in range(3):
        session_id = await tracker.start_generation_tracking(
            user_id=user_id,
            generation_type=GenerationType.IMAGE,
            model_used="unstable_model",
            prompt=f"Failed generation {i}"
        )
        
        await tracker.track_resource_usage(
            session_id=session_id,
            resource_type=ResourceType.GPU_SECONDS,
            amount=10.0
        )
        
        await tracker.end_generation_tracking(
            session_id=session_id,
            success=False,
            error_message="Generation failed"
        )
    
    # GPU-heavy usage pattern
    for i in range(4):
        session_id = await tracker.start_generation_tracking(
            user_id=user_id,
            generation_type=GenerationType.VIDEO,
            model_used="gpu_intensive_model",
            prompt=f"GPU heavy video {i}"
        )
        
        await tracker.track_resource_usage(
            session_id=session_id,
            resource_type=ResourceType.GPU_SECONDS,
            amount=60.0  # Very expensive
        )
        
        await tracker.end_generation_tracking(session_id=session_id)
    
    print("\n2. Analyzing usage patterns and generating recommendations...")
    
    recommendations = await tracker.generate_cost_optimization_recommendations(
        user_id=user_id,
        analysis_days=30
    )
    
    print(f"\n3. Generated {len(recommendations)} optimization recommendations:")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n   Recommendation {i}: {rec.title}")
        print(f"   Type: {rec.recommendation_type}")
        print(f"   Priority: {rec.priority.upper()}")
        print(f"   Description: {rec.description}")
        print(f"   Potential savings: ${rec.potential_savings:.2f}")
        print(f"   Implementation effort: {rec.implementation_effort}")
        
        if rec.current_cost > 0:
            savings_pct = (rec.potential_savings / rec.current_cost) * 100
            print(f"   Savings percentage: {savings_pct:.1f}%")


async def demo_real_time_cost_estimation():
    """Demonstrate real-time cost estimation."""
    print("\n=== Real-Time Cost Estimation Demo ===")
    
    tracker = UsageTracker()
    
    # Test different generation scenarios
    scenarios = [
        {
            "name": "Standard Image",
            "type": GenerationType.IMAGE,
            "model": "stable_diffusion_xl",
            "params": {"resolution": (1024, 1024), "steps": 50}
        },
        {
            "name": "High-Res Image",
            "type": GenerationType.IMAGE,
            "model": "stable_diffusion_xl",
            "params": {"resolution": (2048, 2048), "steps": 100}
        },
        {
            "name": "Premium Image",
            "type": GenerationType.IMAGE,
            "model": "dalle3",
            "params": {"resolution": (1024, 1024)}
        },
        {
            "name": "Short Video",
            "type": GenerationType.VIDEO,
            "model": "runway_ml",
            "params": {"duration": 3.0, "fps": 24, "resolution": (1280, 720)}
        },
        {
            "name": "Long HD Video",
            "type": GenerationType.VIDEO,
            "model": "runway_ml",
            "params": {"duration": 10.0, "fps": 30, "resolution": (1920, 1080)}
        },
        {
            "name": "Image Enhancement",
            "type": GenerationType.ENHANCEMENT,
            "model": "real_esrgan",
            "params": {"upscale_factor": 4}
        }
    ]
    
    print("\n1. Cost estimates for different generation scenarios:")
    
    for scenario in scenarios:
        estimate = await tracker.get_real_time_cost_calculation(
            generation_type=scenario["type"],
            model_name=scenario["model"],
            parameters=scenario["params"]
        )
        
        print(f"\n   {scenario['name']}:")
        print(f"   Model: {scenario['model']}")
        print(f"   Base cost: ${estimate['base_cost']:.4f}")
        print(f"   Complexity multiplier: {estimate['multiplier']:.2f}x")
        print(f"   Estimated cost: ${estimate['estimated_cost']:.4f}")
        
        # Show parameter impact
        if estimate['multiplier'] > 1.0:
            print(f"   💡 Cost increased by {(estimate['multiplier']-1)*100:.0f}% due to parameters")


async def demo_usage_forecasting():
    """Demonstrate usage forecasting."""
    print("\n=== Usage Forecasting Demo ===")
    
    tracker = UsageTracker()
    user_id = "forecasting_demo_user"
    
    # Create historical usage pattern (simulate daily usage over time)
    print("\n1. Creating historical usage pattern...")
    
    base_daily_cost = 2.0
    for day in range(14):  # 2 weeks of data
        # Simulate varying daily usage
        daily_multiplier = 1.0 + (day * 0.1)  # Increasing trend
        daily_cost = base_daily_cost * daily_multiplier
        
        # Create sessions for this day
        sessions_per_day = max(1, int(daily_cost / 0.5))  # ~$0.50 per session
        
        for session in range(sessions_per_day):
            session_id = await tracker.start_generation_tracking(
                user_id=user_id,
                generation_type=GenerationType.IMAGE,
                model_used="stable_diffusion_xl",
                prompt=f"Daily generation day {day} session {session}"
            )
            
            gpu_time = (daily_cost / sessions_per_day) / 0.05  # Convert cost to GPU time
            await tracker.track_resource_usage(
                session_id=session_id,
                resource_type=ResourceType.GPU_SECONDS,
                amount=gpu_time
            )
            
            await tracker.end_generation_tracking(session_id=session_id)
    
    print("   ✓ Created 14 days of historical usage data")
    
    # Generate forecast
    print("\n2. Generating usage forecast...")
    
    forecast = await tracker.generate_usage_forecast(
        user_id=user_id,
        forecast_days=30,
        historical_days=14
    )
    
    print(f"   Forecast period: {forecast.forecast_period_days} days")
    print(f"   Historical data points: {len(forecast.historical_usage)}")
    print(f"   Usage trend: {forecast.usage_trend}")
    print(f"   Predicted cost: ${forecast.predicted_cost:.2f}")
    print(f"   Confidence interval: ${forecast.confidence_interval[0]:.2f} - ${forecast.confidence_interval[1]:.2f}")
    
    if forecast.usage_trend == "increasing":
        print("   📈 Usage is trending upward - consider budget adjustments")
    elif forecast.usage_trend == "decreasing":
        print("   📉 Usage is trending downward - potential for cost savings")
    else:
        print("   📊 Usage is stable - predictable costs expected")


async def main():
    """Run all usage tracking demos."""
    print("🎬 Visual Content Generation - Usage Tracking System Demo")
    print("=" * 60)
    
    try:
        # Run all demo scenarios
        await demo_basic_usage_tracking()
        await demo_video_generation_tracking()
        await demo_failed_generation()
        await demo_usage_analytics()
        await demo_budget_monitoring()
        await demo_cost_optimization()
        await demo_real_time_cost_estimation()
        await demo_usage_forecasting()
        
        print("\n" + "=" * 60)
        print("✅ All usage tracking demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• Real-time resource tracking and cost calculation")
        print("• Comprehensive usage analytics and reporting")
        print("• Budget monitoring with automated alerts")
        print("• Cost optimization recommendations")
        print("• Usage forecasting and trend analysis")
        print("• Support for multiple generation types and models")
        print("• Failed generation tracking and cost recovery")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())