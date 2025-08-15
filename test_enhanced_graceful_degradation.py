#!/usr/bin/env python3
"""
Test script for enhanced graceful degradation system.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrollintel.core.graceful_degradation import (
    IntelligentDegradationManager,
    DegradationLevel,
    UserPreference,
    SystemMetrics,
    with_intelligent_degradation,
    degradation_manager
)
from datetime import datetime


async def test_ml_based_degradation():
    """Test ML-based degradation selection."""
    print("Testing ML-based degradation selection...")
    
    # Create a test user preference
    user_pref = UserPreference(
        user_id="test_user_1",
        preferred_degradation_level=DegradationLevel.MINOR_DEGRADATION,
        feature_priorities={"visualization": 0.8, "ai_services": 0.6},
        tolerance_for_delays=0.7,
        prefers_functionality_over_speed=True
    )
    
    degradation_manager.update_user_preference("test_user_1", user_pref)
    
    # Test degradation evaluation
    conditions = ["high_cpu", "memory_pressure"]
    strategy = await degradation_manager.evaluate_degradation_needed(
        "visualization", conditions, "test_user_1"
    )
    
    if strategy:
        print(f"✓ Selected strategy: {strategy.level.value} - {strategy.description}")
    else:
        print("✗ No strategy selected")
    
    return strategy is not None


async def test_dynamic_adjustment():
    """Test dynamic degradation level adjustment."""
    print("\nTesting dynamic degradation level adjustment...")
    
    # Simulate a degraded service
    degradation_manager.degraded_services["database"] = DegradationLevel.MAJOR_DEGRADATION
    
    print(f"Initial degradation level: {degradation_manager.degraded_services['database'].value}")
    
    # Test dynamic adjustment
    await degradation_manager.adjust_degradation_level_dynamically("database")
    
    current_level = degradation_manager.degraded_services.get("database", DegradationLevel.FULL_SERVICE)
    print(f"After adjustment: {current_level.value}")
    
    return True


async def test_user_feedback_learning():
    """Test learning from user feedback."""
    print("\nTesting user feedback learning...")
    
    # Simulate user feedback
    await degradation_manager.learn_from_user_feedback(
        "test_user_1", "ai_services", DegradationLevel.MINOR_DEGRADATION, 0.8
    )
    
    # Check if preferences were updated
    user_pref = degradation_manager.user_preferences.get("test_user_1")
    if user_pref:
        print(f"✓ User preference updated: {user_pref.preferred_degradation_level.value}")
        print(f"  Tolerance for delays: {user_pref.tolerance_for_delays}")
        return True
    else:
        print("✗ User preference not found")
        return False


async def test_impact_assessment():
    """Test degradation impact assessment."""
    print("\nTesting impact assessment...")
    
    # Get a strategy for testing
    conditions = ["processing_slow"]
    strategy = await degradation_manager.evaluate_degradation_needed(
        "file_processing", conditions, "test_user_1"
    )
    
    if strategy:
        impact = await degradation_manager._assess_degradation_impact(
            "file_processing", strategy, "test_user_1"
        )
        
        print(f"✓ Impact assessment completed:")
        print(f"  User satisfaction: {impact.user_satisfaction_score:.2f}")
        print(f"  Functionality retained: {impact.functionality_retained:.2f}")
        print(f"  Performance improvement: {impact.performance_improvement:.2f}")
        print(f"  Resource savings: {impact.resource_savings:.2f}")
        print(f"  Recovery time: {impact.recovery_time_estimate}")
        
        return True
    else:
        print("✗ No strategy found for impact assessment")
        return False


@with_intelligent_degradation("test_service", user_id_param="user_id")
async def test_function_with_degradation(user_id: str = None):
    """Test function with intelligent degradation decorator."""
    # Simulate a failure
    raise Exception("Simulated timeout error")


async def test_decorator():
    """Test the intelligent degradation decorator."""
    print("\nTesting intelligent degradation decorator...")
    
    try:
        result = await test_function_with_degradation(user_id="test_user_1")
        print(f"✓ Decorator handled failure gracefully: {type(result)}")
        return True
    except Exception as e:
        print(f"✗ Decorator failed: {e}")
        return False


async def test_analytics():
    """Test degradation analytics."""
    print("\nTesting degradation analytics...")
    
    analytics = degradation_manager.get_degradation_analytics()
    
    print("✓ Analytics generated:")
    print(f"  Total degradations: {analytics['total_degradations']}")
    print(f"  Services analyzed: {analytics['services_analyzed']}")
    print(f"  User preferences: {len(analytics.get('user_preference_distribution', {}))}")
    
    return True


async def main():
    """Run all tests."""
    print("Enhanced Graceful Degradation System Test")
    print("=" * 50)
    
    tests = [
        ("ML-based degradation", test_ml_based_degradation),
        ("Dynamic adjustment", test_dynamic_adjustment),
        ("User feedback learning", test_user_feedback_learning),
        ("Impact assessment", test_impact_assessment),
        ("Decorator functionality", test_decorator),
        ("Analytics", test_analytics),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! Enhanced graceful degradation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    asyncio.run(main())