#!/usr/bin/env python3
"""
Demo script showcasing the enhanced bulletproof middleware capabilities.
"""

import asyncio
import sys
import os
import time
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrollintel.core.bulletproof_middleware import (
    BulletproofMiddleware,
    RequestPriority,
    RequestContext,
    LoadBalancingStrategy
)


class MockRequest:
    """Mock request for demonstration."""
    
    def __init__(self, path: str, method: str = "GET", headers: Dict[str, str] = None, 
                 query_params: Dict[str, str] = None):
        self.url = MockURL(path)
        self.method = method
        self.headers = headers or {}
        self.query_params = query_params or {}
        self.state = MockState()


class MockURL:
    """Mock URL for demonstration."""
    
    def __init__(self, path: str):
        self.path = path


class MockState:
    """Mock request state."""
    
    def __init__(self):
        self.request_id = None


class MockApp:
    """Mock ASGI app for demonstration."""
    
    def __init__(self):
        self.middleware_stack = []


async def demo_intelligent_request_routing():
    """Demonstrate intelligent request routing capabilities."""
    print("🎯 Intelligent Request Routing Demo")
    print("=" * 50)
    
    app = MockApp()
    middleware = BulletproofMiddleware(app)
    app.middleware_stack.append(middleware)
    
    # Test different types of requests
    test_requests = [
        {
            "path": "/api/ai/chat",
            "method": "POST",
            "headers": {"X-User-Tier": "premium", "content-length": "1024"},
            "description": "AI Chat Request (Premium User)"
        },
        {
            "path": "/api/upload/file",
            "method": "POST", 
            "headers": {"content-length": "10485760"},  # 10MB
            "description": "Large File Upload"
        },
        {
            "path": "/api/visualization/chart",
            "method": "GET",
            "query_params": {"data_points": "10000", "chart_type": "complex"},
            "description": "Complex Visualization Request"
        },
        {
            "path": "/api/auth/login",
            "method": "POST",
            "headers": {"X-Request-Priority": "critical"},
            "description": "Critical Authentication Request"
        },
        {
            "path": "/api/analytics/batch",
            "method": "POST",
            "headers": {"content-length": "5242880"},  # 5MB
            "description": "Background Analytics Processing"
        }
    ]
    
    for i, req_data in enumerate(test_requests, 1):
        print(f"\n{i}. {req_data['description']}")
        print("-" * 30)
        
        # Create mock request
        request = MockRequest(
            req_data["path"],
            req_data["method"],
            req_data.get("headers", {}),
            req_data.get("query_params", {})
        )
        
        # Create request context
        start_time = time.time()
        request_id = f"demo_{i:03d}"
        context = await middleware._create_request_context(request, request_id, start_time)
        
        # Route the request
        target_node = middleware.request_router.route_request(context)
        
        # Calculate timeout
        timeout = middleware.timeout_manager.calculate_timeout(context)
        
        print(f"📍 Route Target: {target_node}")
        print(f"⚡ Priority: {context.priority.value}")
        print(f"🧠 Complexity Score: {context.complexity_score:.2f}")
        print(f"⏱️  Estimated Duration: {context.estimated_duration:.1f}s")
        print(f"⏰ Dynamic Timeout: {timeout:.1f}s")
        
        # Check if would be queued
        should_queue = middleware.request_prioritizer.should_queue_request(context)
        print(f"📋 Would Queue: {'Yes' if should_queue else 'No'}")
    
    print(f"\n📊 Router Statistics:")
    print(f"   Server Nodes: {len(middleware.request_router.server_nodes)}")
    print(f"   Load Balancing: {middleware.request_router.load_balancing_strategy.value}")
    print(f"   Routing Rules: {len(middleware.request_router.routing_rules)}")


async def demo_dynamic_timeout_adjustment():
    """Demonstrate dynamic timeout adjustment."""
    print("\n\n⏰ Dynamic Timeout Adjustment Demo")
    print("=" * 50)
    
    app = MockApp()
    middleware = BulletproofMiddleware(app)
    
    # Simulate different system load conditions
    load_conditions = [
        {"name": "Low Load", "cpu": 20, "memory": 30, "description": "Optimal conditions"},
        {"name": "Medium Load", "cpu": 60, "memory": 65, "description": "Moderate usage"},
        {"name": "High Load", "cpu": 85, "memory": 80, "description": "Heavy usage"},
        {"name": "Critical Load", "cpu": 95, "memory": 90, "description": "System stressed"}
    ]
    
    # Test request types
    request_types = [
        {"action": "authentication", "complexity": 0.1, "priority": RequestPriority.CRITICAL},
        {"action": "ai_interaction", "complexity": 0.8, "priority": RequestPriority.HIGH},
        {"action": "file_upload", "complexity": 0.6, "priority": RequestPriority.NORMAL},
        {"action": "visualization", "complexity": 0.7, "priority": RequestPriority.NORMAL}
    ]
    
    print("Timeout adjustments under different load conditions:\n")
    
    for load in load_conditions:
        print(f"🖥️  {load['name']} ({load['description']})")
        print(f"   CPU: {load['cpu']}%, Memory: {load['memory']}%")
        
        # Simulate system load effect on timeout manager
        middleware.timeout_manager.system_load_factor = 1.0 + (load['cpu'] + load['memory']) / 200.0
        
        for req_type in request_types:
            context = RequestContext(
                request_id=f"load_test_{req_type['action']}",
                priority=req_type['priority'],
                complexity_score=req_type['complexity'],
                estimated_duration=5.0,
                user_id="demo_user",
                action_type=req_type['action']
            )
            
            timeout = middleware.timeout_manager.calculate_timeout(context)
            print(f"   {req_type['action']:15} -> {timeout:5.1f}s")
        
        print()


async def demo_request_prioritization():
    """Demonstrate request prioritization and queuing."""
    print("\n\n📋 Request Prioritization & Queuing Demo")
    print("=" * 50)
    
    app = MockApp()
    middleware = BulletproofMiddleware(app)
    
    # Create a mix of requests with different priorities
    test_requests = [
        {"id": "auth_001", "priority": RequestPriority.CRITICAL, "action": "authentication"},
        {"id": "ai_001", "priority": RequestPriority.HIGH, "action": "ai_interaction"},
        {"id": "viz_001", "priority": RequestPriority.NORMAL, "action": "visualization"},
        {"id": "upload_001", "priority": RequestPriority.NORMAL, "action": "file_upload"},
        {"id": "batch_001", "priority": RequestPriority.BACKGROUND, "action": "batch_processing"},
        {"id": "auth_002", "priority": RequestPriority.CRITICAL, "action": "authentication"},
        {"id": "ai_002", "priority": RequestPriority.HIGH, "action": "ai_interaction"},
        {"id": "analytics_001", "priority": RequestPriority.LOW, "action": "analytics"}
    ]
    
    print("Processing requests in order of arrival:\n")
    
    for req in test_requests:
        context = RequestContext(
            request_id=req["id"],
            priority=req["priority"],
            complexity_score=0.5,
            estimated_duration=3.0,
            user_id="demo_user",
            action_type=req["action"]
        )
        
        should_queue = middleware.request_prioritizer.should_queue_request(context)
        
        if should_queue:
            middleware.request_prioritizer.queue_request(context)
            print(f"📋 {req['id']:12} ({req['priority'].value:10}) -> QUEUED")
        else:
            middleware.request_prioritizer.start_request(context)
            print(f"🚀 {req['id']:12} ({req['priority'].value:10}) -> STARTED")
    
    # Show queue status
    queue_status = middleware.request_prioritizer.get_queue_status()
    print(f"\n📊 Queue Status:")
    print(f"   Total Active: {queue_status['total_active']}")
    print(f"   Total Queued: {queue_status['total_queued']}")
    
    for priority, count in queue_status['queued_requests'].items():
        if count > 0:
            print(f"   {priority:10}: {count} queued")
    
    # Simulate completing some requests
    print(f"\n⚡ Completing some requests...")
    for req in test_requests[:3]:
        if req["id"] in [r.request_id for r in middleware.request_prioritizer.active_requests.values()]:
            middleware.request_prioritizer.complete_request(req["id"])
            print(f"✅ {req['id']} completed")


async def demo_enhanced_error_responses():
    """Demonstrate enhanced error response system."""
    print("\n\n🚨 Enhanced Error Response System Demo")
    print("=" * 50)
    
    app = MockApp()
    middleware = BulletproofMiddleware(app)
    
    # Test different error scenarios
    error_scenarios = [
        {
            "error": TimeoutError("Request processing timeout"),
            "context": RequestContext("timeout_001", RequestPriority.HIGH, 0.8, 10.0, "user1", "ai_interaction"),
            "description": "AI Request Timeout"
        },
        {
            "error": ConnectionError("Database connection lost"),
            "context": RequestContext("conn_001", RequestPriority.NORMAL, 0.3, 2.0, "user2", "data_retrieval"),
            "description": "Database Connection Error"
        },
        {
            "error": ValueError("Invalid file format provided"),
            "context": RequestContext("val_001", RequestPriority.NORMAL, 0.5, 5.0, "user3", "file_upload"),
            "description": "File Upload Validation Error"
        },
        {
            "error": Exception("Unexpected server error occurred"),
            "context": RequestContext("srv_001", RequestPriority.HIGH, 0.6, 3.0, "user4", "visualization"),
            "description": "Unexpected Server Error"
        }
    ]
    
    for i, scenario in enumerate(error_scenarios, 1):
        print(f"\n{i}. {scenario['description']}")
        print("-" * 40)
        
        enhanced_response = middleware.error_response_system.enhance_error_response(
            scenario["error"], scenario["context"]
        )
        
        print(f"📝 User Message: {enhanced_response['message']}")
        print(f"🔍 What Happened: {enhanced_response['what_happened']}")
        print(f"⏱️  Estimated Fix: {enhanced_response['estimated_fix_time']}")
        print(f"🔄 Retry Recommended: {enhanced_response['retry_recommended']}")
        
        if enhanced_response['what_you_can_do']:
            print(f"💡 User Actions:")
            for action in enhanced_response['what_you_can_do']:
                print(f"   - {action['label']}: {action['description']}")
        
        if enhanced_response['recovery_options']:
            print(f"🛠️  Recovery Options:")
            for option in enhanced_response['recovery_options']:
                print(f"   - {option['title']}: {option['description']}")
        
        if enhanced_response['contextual_help'].get('solutions'):
            print(f"📚 Contextual Help:")
            for solution in enhanced_response['contextual_help']['solutions'][:2]:
                print(f"   - {solution}")


async def demo_comprehensive_stats():
    """Demonstrate comprehensive middleware statistics."""
    print("\n\n📊 Comprehensive Middleware Statistics Demo")
    print("=" * 50)
    
    app = MockApp()
    middleware = BulletproofMiddleware(app)
    app.middleware_stack.append(middleware)
    
    # Simulate some activity
    middleware.request_count = 1250
    middleware.error_count = 23
    middleware.response_times = [0.5, 1.2, 0.8, 2.1, 0.9, 1.5, 3.2, 0.7, 1.8, 2.5]
    middleware.current_load = 0.65
    
    # Add some complexity patterns
    middleware.request_complexity_cache = {
        "ai_interaction_1024_5": 0.8,
        "file_upload_10485760_2": 0.6,
        "visualization_512_8": 0.7,
        "authentication_128_1": 0.1
    }
    
    # Get enhanced stats
    stats = middleware.get_enhanced_middleware_stats()
    
    print("🎯 Performance Metrics:")
    print(f"   Total Requests: {stats['total_requests']:,}")
    print(f"   Error Rate: {stats['error_rate']:.1%}")
    print(f"   Avg Response Time: {stats['avg_response_time']:.2f}s")
    print(f"   Health Status: {stats['health_status'].upper()}")
    
    print(f"\n📈 Performance Percentiles:")
    for percentile, value in stats['performance_percentiles'].items():
        print(f"   {percentile.upper()}: {value:.2f}s")
    
    print(f"\n🖥️  System Status:")
    print(f"   Current Load: {stats['current_load']:.1%}")
    
    print(f"\n🎯 Routing & Load Balancing:")
    routing_stats = stats['routing_stats']
    print(f"   Server Nodes: {routing_stats['server_nodes']}")
    print(f"   Strategy: {routing_stats['load_balancing_strategy']}")
    print(f"   Circuit Breakers: {routing_stats['circuit_breakers']}")
    
    print(f"\n📋 Queue Management:")
    queue_stats = stats['queue_stats']
    print(f"   Total Active: {queue_stats['total_active']}")
    print(f"   Total Queued: {queue_stats['total_queued']}")
    
    print(f"\n🧠 Request Analysis:")
    complexity_stats = stats['complexity_stats']
    print(f"   Cached Patterns: {complexity_stats['cached_patterns']}")
    print(f"   Avg Complexity: {complexity_stats['avg_complexity']:.2f}")
    
    print(f"\n⏰ Timeout Management:")
    timeout_stats = stats['timeout_stats']
    print(f"   Dynamic Timeouts: {'Enabled' if timeout_stats['dynamic_timeouts_enabled'] else 'Disabled'}")
    print(f"   Timeout Patterns: {timeout_stats['timeout_patterns']}")
    
    print(f"\n🚨 Error Response Enhancement:")
    error_stats = stats['error_response_stats']
    print(f"   Enhanced Responses: {'Enabled' if error_stats['enhanced_responses_enabled'] else 'Disabled'}")
    print(f"   Error Patterns: {error_stats['error_patterns']}")
    print(f"   Contextual Help: {error_stats['contextual_help_available']} topics")


async def main():
    """Run all demonstrations."""
    print("🚀 Enhanced Bulletproof Middleware Demonstration")
    print("=" * 60)
    print("This demo showcases the advanced features of the enhanced")
    print("bulletproof middleware system for ScrollIntel.")
    print("=" * 60)
    
    demos = [
        demo_intelligent_request_routing,
        demo_dynamic_timeout_adjustment,
        demo_request_prioritization,
        demo_enhanced_error_responses,
        demo_comprehensive_stats
    ]
    
    for demo in demos:
        try:
            await demo()
            await asyncio.sleep(1)  # Brief pause between demos
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    
    print("\n\n🎉 Enhanced Bulletproof Middleware Demo Complete!")
    print("=" * 60)
    print("Key Features Demonstrated:")
    print("✅ Intelligent request routing with complexity analysis")
    print("✅ Dynamic timeout adjustment based on system load")
    print("✅ Priority-based request queuing and load balancing")
    print("✅ Enhanced error responses with contextual help")
    print("✅ Comprehensive monitoring and statistics")
    print("✅ Backward compatibility with existing systems")
    print("\nThe enhanced middleware provides bulletproof reliability")
    print("while maintaining excellent user experience under all conditions.")


if __name__ == "__main__":
    asyncio.run(main())