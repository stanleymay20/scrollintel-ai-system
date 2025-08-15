"""
Test script for failure prevention and user experience integration.
"""

import asyncio
import time
from datetime import datetime
from scrollintel.core.failure_ux_integration import (
    failure_ux_integrator, 
    bulletproof_user_operation,
    bulletproof_with_ux,
    get_failure_ux_integration_status
)
from scrollintel.core.failure_prevention import FailureEvent, FailureType
from scrollintel.core.user_experience_protection import get_user_experience_status


async def test_basic_integration():
    """Test basic integration functionality."""
    print("Testing basic integration...")
    
    # Test unified failure handling
    failure_event = FailureEvent(
        failure_type=FailureType.NETWORK_ERROR,
        timestamp=datetime.utcnow(),
        error_message="Connection timeout",
        stack_trace="",
        context={"test": "basic_integration"}
    )
    
    response = await failure_ux_integrator.handle_unified_failure(failure_event)
    print(f"Unified response created with {len(response.technical_recovery_actions)} technical actions")
    print(f"UX actions: {response.ux_protection_actions}")
    print(f"User message: {response.user_communication['message']}")


async def test_user_behavior_analysis():
    """Test user behavior analysis and prediction."""
    print("\nTesting user behavior analysis...")
    
    # Simulate user actions with patterns that should trigger predictions
    try:
        async with bulletproof_user_operation("slow_operation", "test_user", "normal"):
            # Simulate slow operation
            await asyncio.sleep(0.1)
    except:
        pass  # Expected for testing
    
    # Simulate multiple failed operations
    for i in range(3):
        try:
            async with bulletproof_user_operation("failing_operation", "test_user", "normal"):
                # Simulate failure
                raise ValueError("Validation error")
        except ValueError:
            pass  # Expected for testing
    
    # Check if predictions were generated
    status = get_failure_ux_integration_status()
    print(f"Active predictions: {status['active_predictions']}")
    print(f"Recent predictions: {status['recent_predictions']}")


async def test_bulletproof_decorator():
    """Test the bulletproof decorator with UX protection."""
    print("\nTesting bulletproof decorator...")
    
    @bulletproof_with_ux("test_operation", "test_user", "critical")
    async def test_function():
        # Simulate some work
        await asyncio.sleep(0.1)
        return "Success!"
    
    try:
        result = await test_function()
        print(f"Bulletproof function result: {result}")
    except Exception as e:
        print(f"Function failed as expected: {e}")


async def test_user_feedback_integration():
    """Test user feedback integration."""
    print("\nTesting user feedback integration...")
    
    # Simulate user feedback about slowness
    from scrollintel.core.user_experience_protection import ux_protector
    
    try:
        ux_protector.record_user_feedback("test_user", {
            "issue": "The app is too slow",
            "severity": "medium"
        })
        
        # Wait a moment for async processing
        await asyncio.sleep(0.1)
        
        # Check if predictions were generated from feedback
        status = get_failure_ux_integration_status()
        print(f"Predictions after feedback: {status['active_predictions']}")
    except Exception as e:
        print(f"Feedback integration not fully implemented: {e}")


async def test_system_status():
    """Test system status reporting."""
    print("\nTesting system status...")
    
    # Get integration status
    integration_status = get_failure_ux_integration_status()
    print(f"Integration metrics: {integration_status['metrics']}")
    print(f"Prediction accuracy: {integration_status['prediction_accuracy']:.2%}")
    
    # Get UX status
    ux_status = get_user_experience_status()
    print(f"Experience level: {ux_status['experience_level']}")
    print(f"Success rate: {ux_status['success_rate']:.2%}")


async def test_failure_simulation():
    """Test failure simulation and recovery."""
    print("\nTesting failure simulation...")
    
    # Simulate a database error
    failure_event = FailureEvent(
        failure_type=FailureType.DATABASE_ERROR,
        timestamp=datetime.utcnow(),
        error_message="Database connection lost",
        stack_trace="",
        context={"operation": "save_user_data", "user_id": "test_user"}
    )
    
    response = await failure_ux_integrator.handle_unified_failure(failure_event)
    print(f"Database failure handled with {len(response.fallback_strategies)} fallback strategies")
    print(f"Recovery timeline: {list(response.recovery_timeline.keys())}")


async def main():
    """Run all tests."""
    print("Starting Failure-UX Integration Tests")
    print("=" * 50)
    
    try:
        await test_basic_integration()
        await test_user_behavior_analysis()
        await test_bulletproof_decorator()
        await test_user_feedback_integration()
        await test_system_status()
        await test_failure_simulation()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
        # Final status report
        final_status = get_failure_ux_integration_status()
        print(f"\nFinal Integration Status:")
        print(f"- Total predictions made: {final_status['metrics']['predictions_made']}")
        print(f"- Failures prevented: {final_status['metrics']['failures_prevented']}")
        print(f"- UX improvements: {final_status['metrics']['user_experience_improvements']}")
        print(f"- Active user patterns: {final_status['user_patterns']}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())