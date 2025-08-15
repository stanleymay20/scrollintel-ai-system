#!/usr/bin/env python3
"""
Integration test for ML Engineer Agent with the orchestrator
"""
import asyncio
import sys
import os

# Add the scrollintel_core directory to the path
sys.path.append('scrollintel_core')

from agents.orchestrator import AgentOrchestrator
from agents.base import AgentRequest

async def test_ml_engineer_integration():
    """Test ML Engineer Agent through the orchestrator"""
    print("🔗 Testing ML Engineer Agent Integration")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    
    # Test 1: Check if ML Engineer Agent is registered
    print("\n1. Checking agent registration...")
    agents = orchestrator.get_available_agents()
    ml_agent_found = any(agent['name'] == 'ML Engineer Agent' for agent in agents)
    print(f"ML Engineer Agent registered: {'✅ Yes' if ml_agent_found else '❌ No'}")
    
    if ml_agent_found:
        # Test 2: Route ML query to ML Engineer Agent
        print("\n2. Testing query routing...")
        request = AgentRequest(
            query="I want to build a machine learning model",
            context={}
        )
        
        response = await orchestrator.process_request(request)
        print(f"Query routed successfully: {'✅ Yes' if response.success else '❌ No'}")
        print(f"Handled by: {response.agent_name}")
        
        # Test 3: Test with sample data
        print("\n3. Testing with sample data...")
        sample_data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [0, 0, 1, 1, 1]
        }
        
        request = AgentRequest(
            query="Build a classification model",
            context={
                "data": sample_data,
                "target_column": "target"
            }
        )
        
        response = await orchestrator.process_request(request)
        print(f"Model building: {'✅ Success' if response.success else '❌ Failed'}")
        if response.success and 'model_id' in response.result:
            print(f"Model ID: {response.result['model_id']}")
    
    print("\n" + "=" * 50)
    print("✅ ML Engineer Agent Integration Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_ml_engineer_integration())