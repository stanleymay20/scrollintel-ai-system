#!/usr/bin/env python3
"""
Test script for enhanced bulletproof middleware functionality.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrollintel.core.bulletproof_middleware import (
    BulletproofMiddleware,
    IntelligentRequestRouter,
    DynamicTimeoutManager,
    RequestPrioritizer,
    EnhancedErrorResponseSystem,
    RequestPriority,
    RequestContext
)


async def test_intelligent_request_router():
    """Test the intelligent request router."""
    print("Testing Intelligent Request Router...")
    
    router = IntelligentRequestRouter()
    
    # Create test context
    context = RequestContext(
        request_id="test_001",
        priority=RequestPriority.HIGH,
        complexity_score=0.7,
        estimated_duration=5.0,
        user_id="test_user",
        action_type="ai_interaction"
    )
    
    # Test routing
    target_node = router.route_request(context)
    print(f"✓ Request routed to: {target_node}")
    
    # Test node metrics update
    router.update_node_metrics(target_node, 2.5, True)
    print(f"✓ Node metrics updated successfully")
    
    return True


async def test_dynamic_timeout_manager():
    """Test the dynamic timeout manager."""
    print("Testing Dynamic Timeout Manager...")
    
    timeout_manager = DynamicTimeoutManager()
    
    # Create test contexts with different complexities
    contexts = [
        RequestContext("test_001", RequestPriority.CRITICAL, 0.2, 1.0, "user1", "authentication"),
        RequestContext("test_002", RequestPriority.HIGH, 0.8, 10.0, "user2", "ai_interaction"),
        RequestContext("test_003", RequestPriority.BACKGROUND, 0.9, 30.0, "user3", "file_upload")
    ]
    
    for context in contexts:
        timeout = timeout_manager.calculate_timeout(context)
        print(f"✓ {context.action_type} (complexity: {context.complexity_score:.1f}) -> timeout: {timeout:.1f}s")
        
        # Record response time
        timeout_manager.record_response_time(context.action_type, timeout * 0.8)
    
    return True


async def test_request_prioritizer():
    """Test the request prioritizer."""
    print("Testing Request Prioritizer...")
    
    prioritizer = RequestPrioritizer()
    
    # Create test contexts
    contexts = [
        RequestContext("test_001", RequestPriority.CRITICAL, 0.3, 2.0, "user1", "authentication"),
        RequestContext("test_002", RequestPriority.HIGH, 0.6, 5.0, "user2", "ai_interaction"),
        RequestContext("test_003", RequestPriority.NORMAL, 0.4, 3.0, "user3", "data_retrieval"),
        RequestContext("test_004", RequestPriority.BACKGROUND, 0.8, 15.0, "user4", "file_upload")
    ]
    
    # Test queuing logic
    for context in contexts:
        should_queue = prioritizer.should_queue_request(context)
        if should_queue:
            prioritizer.queue_request(context)
            print(f"✓ {context.request_id} ({context.priority.value}) queued")
        else:
            prioritizer.start_request(context)
            print(f"✓ {context.request_id} ({context.priority.value}) started")
    
    # Get queue status
    status = prioritizer.get_queue_status()
    print(f"✓ Queue status: {status}")
    
    # Complete some requests
    for context in contexts[:2]:
        prioritizer.complete_request(context.request_id)
        print(f"✓ {context.request_id} completed")
    
    return True


async def test_enhanced_error_response_system():
    """Test the enhanced error response system."""
    print("Testing Enhanced Error Response System...")
    
    error_system = EnhancedErrorResponseSystem()
    
    # Create test context
    context = RequestContext(
        request_id="test_error_001",
        priority=RequestPriority.HIGH,
        complexity_score=0.6,
        estimated_duration=5.0,
        user_id="test_user",
        action_type="visualization"
    )
    
    # Test different error types
    test_errors = [
        TimeoutError("Request timed out"),
        ConnectionError("Network connection failed"),
        ValueError("Invalid input data"),
        Exception("Unexpected server error")
    ]
    
    for error in test_errors:
        enhanced_response = error_system.enhance_error_response(error, context)
        print(f"✓ Enhanced response for {error.__class__.__name__}:")
        print(f"  - Message: {enhanced_response['message']}")
        print(f"  - Recovery options: {len(enhanced_response['recovery_options'])}")
        print(f"  - User actions: {len(enhanced_response['what_you_can_do'])}")
    
    return True


async def test_middleware_integration():
    """Test middleware integration and stats."""
    print("Testing Middleware Integration...")
    
    # Create a mock ASGI app
    class MockApp:
        def __init__(self):
            self.middleware_stack = []
    
    app = MockApp()
    middleware = BulletproofMiddleware(app)
    app.middleware_stack.append(middleware)
    
    # Test stats generation
    stats = middleware.get_enhanced_middleware_stats()
    print(f"✓ Enhanced stats generated with {len(stats)} metrics")
    
    # Test backward compatibility
    basic_stats = middleware.get_middleware_stats()
    print(f"✓ Basic stats (backward compatibility) generated with {len(basic_stats)} metrics")
    
    # Test complexity calculation
    avg_complexity = middleware._calculate_avg_complexity()
    print(f"✓ Average complexity calculated: {avg_complexity:.2f}")
    
    # Test health status calculation
    health_status = middleware._calculate_health_status(0.02, 1.5)
    print(f"✓ Health status calculated: {health_status}")
    
    return True


async def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Bulletproof Middleware Components\n")
    
    tests = [
        test_intelligent_request_router,
        test_dynamic_timeout_manager,
        test_request_prioritizer,
        test_enhanced_error_response_system,
        test_middleware_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n--- {test.__name__.replace('test_', '').replace('_', ' ').title()} ---")
            result = await test()
            if result:
                print("✅ PASSED\n")
                passed += 1
            else:
                print("❌ FAILED\n")
                failed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}\n")
            failed += 1
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Enhanced bulletproof middleware is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)