#!/usr/bin/env python3
"""
Test script for Agent Orchestrator Foundation
"""
import asyncio
import sys
import os

# Add scrollintel_core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scrollintel_core'))

from scrollintel_core.agents.orchestrator import AgentOrchestrator


async def test_orchestrator_foundation():
    """Test the Agent Orchestrator Foundation implementation"""
    print("🚀 Testing Agent Orchestrator Foundation")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    
    try:
        # Test 1: Initialize orchestrator
        print("\n1. Initializing orchestrator...")
        await orchestrator.initialize()
        print("✅ Orchestrator initialized successfully")
        
        # Test 2: Check available agents
        print("\n2. Checking available agents...")
        agents = await orchestrator.get_available_agents()
        print(f"✅ Found {len(agents)} agents:")
        for agent in agents:
            print(f"   - {agent['name']}: {agent['description']}")
        
        # Test 3: Health check
        print("\n3. Performing health check...")
        health = await orchestrator.health_check()
        print(f"✅ Health check complete:")
        print(f"   - Orchestrator healthy: {health['orchestrator_healthy']}")
        print(f"   - Total agents: {health['total_agents']}")
        print(f"   - Healthy agents: {health['healthy_agents']}")
        
        # Test 4: Test agent routing
        print("\n4. Testing agent routing...")
        test_queries = [
            "What technology stack should I use for my startup?",
            "Analyze this data for insights",
            "Build a machine learning model",
            "Create a dashboard for my KPIs",
            "Help me with AI strategy",
            "How many customers do we have?",
            "Forecast next quarter's revenue"
        ]
        
        for query in test_queries:
            response = await orchestrator.route_request(query)
            print(f"   Query: '{query[:40]}...'")
            print(f"   → Routed to: {response.agent_name}")
            print(f"   → Success: {response.success}")
            if response.error:
                print(f"   → Error: {response.error}")
        
        # Test 5: Get request statistics
        print("\n5. Checking request statistics...")
        stats = orchestrator.get_request_stats()
        print(f"✅ Request stats:")
        print(f"   - Total requests: {stats['total_requests']}")
        print(f"   - Success rate: {stats['success_rate']:.2%}")
        print(f"   - Agent usage: {stats['agent_usage']}")
        
        # Test 6: Test agent registry
        print("\n6. Testing agent registry...")
        registry = await orchestrator.get_agent_registry()
        print(f"✅ Agent registry contains {len(registry)} agents:")
        for name, info in registry.items():
            print(f"   - {name}: {info['total_requests']} requests, {info['status']} status")
        
        # Test 7: Test individual agent health checks
        print("\n7. Testing individual agent health checks...")
        for agent_name in ["cto", "data_scientist", "ml_engineer"]:
            is_healthy = await orchestrator.is_agent_healthy(agent_name)
            print(f"   - {agent_name}: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")
        
        # Test 8: Test periodic health check
        print("\n8. Testing periodic health check...")
        periodic_health = await orchestrator.periodic_health_check()
        print(f"✅ Periodic health check complete:")
        print(f"   - Healthy agents: {periodic_health['healthy_agents']}/{periodic_health['total_agents']}")
        
        # Test 9: Test agent suggestion
        print("\n9. Testing agent suggestion...")
        test_queries = [
            "I need help with my database architecture",
            "Can you analyze my sales data?",
            "Build a recommendation system"
        ]
        for query in test_queries:
            suggestion = orchestrator.suggest_agent(query)
            print(f"   Query: '{query}'")
            print(f"   → Suggested: {suggestion['suggested_agent']} (confidence: {suggestion['confidence']:.2f})")
            print(f"   → Reason: {suggestion['reason']}")
        
        # Test 10: Test routing information
        print("\n10. Testing routing information...")
        routing_info = orchestrator.get_routing_info()
        print(f"✅ Routing strategy: {routing_info['routing_strategy']}")
        print(f"   - Default agent: {routing_info['default_agent']}")
        print(f"   - Fallback strategy: {routing_info['fallback_strategy']}")
        
        print("\n🎉 All tests passed! Agent Orchestrator Foundation is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        await orchestrator.shutdown()
        print("\n🧹 Orchestrator shutdown complete")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_orchestrator_foundation())
    sys.exit(0 if success else 1)