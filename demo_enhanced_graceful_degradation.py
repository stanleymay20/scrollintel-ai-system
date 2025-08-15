#!/usr/bin/env python3
"""
Demo script showcasing the enhanced graceful degradation system.
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrollintel.core.graceful_degradation import (
    IntelligentDegradationManager,
    DegradationLevel,
    UserPreference,
    SystemMetrics,
    with_intelligent_degradation,
    degradation_manager,
    learn_from_feedback,
    adjust_degradation_dynamically,
    get_degradation_analytics
)


async def demo_ml_based_selection():
    """Demonstrate ML-based degradation selection."""
    print("🧠 ML-Based Degradation Selection Demo")
    print("-" * 50)
    
    # Create different user profiles
    users = [
        UserPreference(
            user_id="performance_user",
            preferred_degradation_level=DegradationLevel.MINOR_DEGRADATION,
            feature_priorities={"speed": 0.9, "functionality": 0.3},
            tolerance_for_delays=0.2,
            prefers_functionality_over_speed=False
        ),
        UserPreference(
            user_id="functionality_user",
            preferred_degradation_level=DegradationLevel.MAJOR_DEGRADATION,
            feature_priorities={"functionality": 0.9, "speed": 0.3},
            tolerance_for_delays=0.8,
            prefers_functionality_over_speed=True
        )
    ]
    
    for user in users:
        degradation_manager.update_user_preference(user.user_id, user)
        print(f"👤 Created user profile: {user.user_id}")
        print(f"   Preferred level: {user.preferred_degradation_level.value}")
        print(f"   Delay tolerance: {user.tolerance_for_delays}")
    
    # Test degradation selection for different users
    conditions = ["high_cpu", "memory_pressure"]
    
    for user in users:
        strategy = await degradation_manager.evaluate_degradation_needed(
            "visualization", conditions, user.user_id
        )
        
        if strategy:
            print(f"\n🎯 Selected for {user.user_id}: {strategy.level.value}")
            print(f"   Description: {strategy.description}")
        else:
            print(f"\n❌ No strategy selected for {user.user_id}")
    
    print()


async def demo_dynamic_adjustment():
    """Demonstrate dynamic degradation level adjustment."""
    print("⚡ Dynamic Degradation Adjustment Demo")
    print("-" * 50)
    
    # Simulate a service starting in emergency mode
    service_name = "ai_services"
    degradation_manager.degraded_services[service_name] = DegradationLevel.EMERGENCY_MODE
    
    print(f"🚨 Starting {service_name} in: {DegradationLevel.EMERGENCY_MODE.value}")
    
    # Simulate system recovery over time
    recovery_steps = [
        ("System stabilizing...", DegradationLevel.MAJOR_DEGRADATION),
        ("Performance improving...", DegradationLevel.MINOR_DEGRADATION),
        ("System fully recovered!", DegradationLevel.FULL_SERVICE)
    ]
    
    for step_msg, expected_level in recovery_steps:
        await asyncio.sleep(1)  # Simulate time passing
        print(f"📈 {step_msg}")
        
        await adjust_degradation_dynamically(service_name)
        
        current_level = degradation_manager.degraded_services.get(
            service_name, DegradationLevel.FULL_SERVICE
        )
        print(f"   Current level: {current_level.value}")
        
        # Simulate improving conditions by adjusting thresholds
        if service_name in degradation_manager.dynamic_thresholds:
            for level_name in degradation_manager.dynamic_thresholds[service_name]:
                for metric in degradation_manager.dynamic_thresholds[service_name][level_name]:
                    degradation_manager.dynamic_thresholds[service_name][level_name][metric] *= 1.2
    
    print()


async def demo_user_feedback_learning():
    """Demonstrate learning from user feedback."""
    print("📚 User Feedback Learning Demo")
    print("-" * 50)
    
    user_id = "learning_user"
    service_name = "database"
    
    # Create initial user preference
    initial_pref = UserPreference(
        user_id=user_id,
        preferred_degradation_level=DegradationLevel.MINOR_DEGRADATION,
        feature_priorities={},
        tolerance_for_delays=0.5,
        prefers_functionality_over_speed=True
    )
    
    degradation_manager.update_user_preference(user_id, initial_pref)
    print(f"👤 Initial preference for {user_id}:")
    print(f"   Preferred level: {initial_pref.preferred_degradation_level.value}")
    print(f"   Delay tolerance: {initial_pref.tolerance_for_delays}")
    
    # Simulate user feedback scenarios
    feedback_scenarios = [
        (DegradationLevel.MINOR_DEGRADATION, 0.9, "User very satisfied with minor degradation"),
        (DegradationLevel.MAJOR_DEGRADATION, 0.2, "User dissatisfied with major degradation"),
        (DegradationLevel.MINOR_DEGRADATION, 0.8, "User satisfied with minor degradation again")
    ]
    
    for level, satisfaction, description in feedback_scenarios:
        print(f"\n📝 Feedback: {description}")
        print(f"   Level: {level.value}, Satisfaction: {satisfaction}")
        
        await learn_from_feedback(user_id, service_name, level, satisfaction)
        
        # Show updated preferences
        updated_pref = degradation_manager.user_preferences[user_id]
        print(f"   Updated preference: {updated_pref.preferred_degradation_level.value}")
        print(f"   Updated tolerance: {updated_pref.tolerance_for_delays:.2f}")
    
    print()


async def demo_impact_assessment():
    """Demonstrate degradation impact assessment."""
    print("📊 Impact Assessment Demo")
    print("-" * 50)
    
    services = ["visualization", "ai_services", "database"]
    
    for service in services:
        print(f"\n🔍 Assessing impact for {service}:")
        
        # Get strategies for this service
        strategies = degradation_manager.degradation_strategies.get(service, [])
        
        for strategy in strategies[:2]:  # Show first 2 strategies
            impact = await degradation_manager._assess_degradation_impact(
                service, strategy, "functionality_user"
            )
            
            print(f"   {strategy.level.value}:")
            print(f"     User satisfaction: {impact.user_satisfaction_score:.2f}")
            print(f"     Functionality retained: {impact.functionality_retained:.2f}")
            print(f"     Performance improvement: {impact.performance_improvement:.2f}")
            print(f"     Resource savings: {impact.resource_savings:.2f}")
            print(f"     Recovery time: {impact.recovery_time_estimate}")
    
    print()


@with_intelligent_degradation("demo_service", user_id_param="user_id")
async def demo_function_with_failures(operation: str, user_id: str = None):
    """Demo function that can fail in different ways."""
    failure_types = {
        "timeout": "Connection timeout occurred",
        "memory": "Out of memory error",
        "database": "Database connection failed",
        "api": "API service unavailable"
    }
    
    if operation in failure_types:
        raise Exception(failure_types[operation])
    
    return {"result": f"Success for operation: {operation}", "user_id": user_id}


async def demo_decorator_functionality():
    """Demonstrate the intelligent degradation decorator."""
    print("🎭 Decorator Functionality Demo")
    print("-" * 50)
    
    operations = ["timeout", "memory", "database", "api", "success"]
    
    for operation in operations:
        print(f"\n🔧 Testing operation: {operation}")
        
        try:
            result = await demo_function_with_failures(operation, user_id="demo_user")
            
            if isinstance(result, dict) and result.get("degraded"):
                print(f"   ✅ Degradation applied: {result.get('degradation_level', 'unknown')}")
                print(f"   📝 Message: {result.get('message', 'No message')}")
            else:
                print(f"   ✅ Success: {result}")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print()


async def demo_analytics():
    """Demonstrate degradation analytics."""
    print("📈 Degradation Analytics Demo")
    print("-" * 50)
    
    analytics = get_degradation_analytics()
    
    print("📊 System Analytics:")
    print(f"   Total degradations: {analytics['total_degradations']}")
    print(f"   Services analyzed: {len(analytics['services_analyzed'])}")
    print(f"   User preferences tracked: {len(analytics.get('user_preference_distribution', {}))}")
    
    if analytics['average_satisfaction_by_service']:
        print("\n😊 Average Satisfaction by Service:")
        for service, satisfaction in analytics['average_satisfaction_by_service'].items():
            print(f"   {service}: {satisfaction:.2f}")
    
    if analytics['most_used_degradation_levels']:
        print("\n📊 Most Used Degradation Levels:")
        for level, count in analytics['most_used_degradation_levels'].items():
            print(f"   {level}: {count} times")
    
    if analytics['user_preference_distribution']:
        print("\n👥 User Preference Distribution:")
        for level, count in analytics['user_preference_distribution'].items():
            print(f"   {level}: {count} users")
    
    print("\n🧠 ML Model Performance:")
    for service, performance in analytics['ml_model_performance'].items():
        print(f"   {service}:")
        print(f"     Weight magnitude: {performance['weight_magnitude']:.3f}")
        top_features = sorted(
            performance['feature_importance'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        print(f"     Top features: {', '.join([f'{k}({v:.3f})' for k, v in top_features])}")
    
    print()


async def main():
    """Run the comprehensive demo."""
    print("🚀 Enhanced Graceful Degradation System Demo")
    print("=" * 60)
    print("This demo showcases the intelligent features of the enhanced")
    print("graceful degradation system including ML-based selection,")
    print("dynamic adjustment, user preference learning, and impact assessment.")
    print("=" * 60)
    
    demos = [
        demo_ml_based_selection,
        demo_dynamic_adjustment,
        demo_user_feedback_learning,
        demo_impact_assessment,
        demo_decorator_functionality,
        demo_analytics
    ]
    
    for i, demo_func in enumerate(demos, 1):
        print(f"\n[{i}/{len(demos)}] ", end="")
        await demo_func()
        
        if i < len(demos):
            print("⏳ Waiting 2 seconds before next demo...")
            await asyncio.sleep(2)
    
    print("🎉 Demo completed! The enhanced graceful degradation system")
    print("   provides intelligent, adaptive, and user-aware degradation")
    print("   strategies that learn and improve over time.")


if __name__ == "__main__":
    asyncio.run(main())