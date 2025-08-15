#!/usr/bin/env python3
"""
Simple test for ScrollMLEngineer agent.
"""

import asyncio
from datetime import datetime
from scrollintel.core.interfaces import AgentRequest
from scrollintel.agents.scroll_ml_engineer import ScrollMLEngineer

async def test_ml_engineer():
    """Test ScrollMLEngineer agent."""
    print("Testing ScrollMLEngineer agent...")
    
    # Create agent
    agent = ScrollMLEngineer()
    print(f"Agent created: {agent.name}")
    
    # Test health check
    is_healthy = await agent.health_check()
    print(f"Health check: {is_healthy}")
    
    # Test capabilities
    capabilities = agent.get_capabilities()
    print(f"Capabilities: {len(capabilities)} available")
    for cap in capabilities:
        print(f"  - {cap.name}: {cap.description}")
    
    # Test pipeline setup request
    request = AgentRequest(
        id="test-1",
        user_id="test-user",
        agent_id="scroll-ml-engineer",
        prompt="Set up ML pipeline for customer churn prediction",
        context={
            "action": "setup_pipeline",
            "dataset_path": "data/customers.csv",
            "target_column": "churn",
            "framework": "scikit-learn"
        },
        priority=1,
        created_at=datetime.now()
    )
    
    print("\nTesting pipeline setup...")
    response = await agent.process_request(request)
    print(f"Response status: {response.status}")
    print(f"Response length: {len(response.content)} characters")
    print(f"Execution time: {response.execution_time:.2f}s")
    print(f"Response content: {response.content[:200]}...")
    
    if "ML Pipeline Setup Report" in response.content:
        print("✅ Pipeline setup test passed")
    else:
        print("❌ Pipeline setup test failed")
    
    # Test general advice
    general_request = AgentRequest(
        id="test-2",
        user_id="test-user",
        agent_id="scroll-ml-engineer",
        prompt="What's the best approach for model versioning?",
        context={},
        priority=1,
        created_at=datetime.now()
    )
    
    print("\nTesting general advice...")
    response = await agent.process_request(general_request)
    print(f"Response status: {response.status}")
    
    if "ML Engineering Consultation" in response.content:
        print("✅ General advice test passed")
    else:
        print("❌ General advice test failed")
    
    print("\n✅ ScrollMLEngineer agent tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_ml_engineer())