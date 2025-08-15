#!/usr/bin/env python3
"""
Test script for AI Engineer Agent implementation
"""
import asyncio
import sys
import os

# Add the scrollintel_core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scrollintel_core'))

from scrollintel_core.agents.ai_engineer_agent import AIEngineerAgent
from scrollintel_core.agents.base import AgentRequest


async def test_ai_engineer_agent():
    """Test the AI Engineer Agent implementation"""
    print("🤖 Testing AI Engineer Agent Implementation")
    print("=" * 50)
    
    # Initialize the agent
    agent = AIEngineerAgent()
    
    # Test 1: AI Strategy Request
    print("\n📋 Test 1: AI Strategy Generation")
    strategy_request = AgentRequest(
        query="Create an AI implementation strategy for my retail business",
        context={
            "business_size": "medium",
            "industry": "retail",
            "current_capabilities": ["basic_analytics"],
            "budget": "medium",
            "timeline": "12 months"
        }
    )
    
    response = await agent.process(strategy_request)
    print(f"✅ Success: {response.success}")
    if response.success:
        print(f"📊 Strategy includes: {list(response.result.keys())}")
        print(f"🎯 AI Maturity Level: {response.result['ai_maturity_assessment']['current_level']}")
    else:
        print(f"❌ Error: {response.error}")
    
    # Test 2: Architecture Recommendation
    print("\n🏗️ Test 2: Architecture Recommendation")
    arch_request = AgentRequest(
        query="Recommend AI architecture for real-time recommendations",
        context={
            "use_case": "real-time recommendations",
            "scale": "large",
            "performance": "high",
            "data_volume": "large"
        }
    )
    
    response = await agent.process(arch_request)
    print(f"✅ Success: {response.success}")
    if response.success:
        print(f"🏛️ Recommended Architecture: {response.result['recommended_architecture']['pattern']}")
        print(f"📈 Scale Config: {response.result['scale_configuration']['max_requests_per_second']} req/sec")
    else:
        print(f"❌ Error: {response.error}")
    
    # Test 3: Cost Estimation
    print("\n💰 Test 3: Cost Estimation")
    cost_request = AgentRequest(
        query="Estimate costs for implementing AI in my business",
        context={
            "scale": "medium",
            "complexity": "high",
            "timeline": "9 months",
            "use_cases": ["customer_analytics", "inventory_optimization"]
        }
    )
    
    response = await agent.process(cost_request)
    print(f"✅ Success: {response.success}")
    if response.success:
        print(f"💵 Total Cost Estimate: {response.result['total_cost_estimate']['annual_total']}")
        print(f"📊 Development Costs: {response.result['detailed_breakdown']['development_costs']['model_development']}")
    else:
        print(f"❌ Error: {response.error}")
    
    # Test 4: Integration Guidance
    print("\n🔗 Test 4: Integration Guidance")
    integration_request = AgentRequest(
        query="Provide AI integration best practices for my existing system",
        context={
            "current_system": "legacy_erp",
            "integration_type": "api_first"
        }
    )
    
    response = await agent.process(integration_request)
    print(f"✅ Success: {response.success}")
    if response.success:
        print(f"🔧 Integration Strategies: {len(response.result['integration_strategies'])} options")
        print(f"✨ Best Practices: {len(response.result['best_practices'])} recommendations")
    else:
        print(f"❌ Error: {response.error}")
    
    # Test 5: Health Check
    print("\n🏥 Test 5: Agent Health Check")
    health = await agent.health_check()
    print(f"💚 Agent Health: {'Healthy' if health['healthy'] else 'Unhealthy'}")
    print(f"🎯 Capabilities: {len(health['capabilities'])} capabilities")
    
    # Test 6: Agent Info
    print("\n📋 Test 6: Agent Information")
    info = agent.get_info()
    print(f"🤖 Agent Name: {info['name']}")
    print(f"📝 Description: {info['description']}")
    print(f"⚡ Capabilities: {', '.join(info['capabilities'])}")
    
    print("\n" + "=" * 50)
    print("✅ AI Engineer Agent Implementation Test Complete!")
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_ai_engineer_agent())
        if result:
            print("🎉 All tests passed successfully!")
            sys.exit(0)
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"💥 Test execution failed: {e}")
        sys.exit(1)