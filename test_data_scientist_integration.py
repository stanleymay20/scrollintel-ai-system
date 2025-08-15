#!/usr/bin/env python3
"""
Integration test for Data Scientist Agent with the orchestrator
"""

import asyncio
import pandas as pd
import numpy as np
import sys
import os

# Add the scrollintel_core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scrollintel_core'))

from agents.orchestrator import AgentOrchestrator

async def test_integration():
    """Test Data Scientist Agent integration with orchestrator"""
    print("🔗 Testing Data Scientist Agent Integration")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    await orchestrator.initialize()
    
    print("✅ Orchestrator initialized")
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'sales': np.random.normal(1000, 200, 100),
        'marketing': np.random.normal(500, 100, 100),
        'customers': np.random.poisson(50, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })
    
    # Add correlation
    sample_data['sales'] += sample_data['marketing'] * 0.5
    
    print(f"📊 Created sample data: {sample_data.shape}")
    
    # Test queries that should route to Data Scientist Agent
    test_queries = [
        "Analyze my sales data",
        "Check data quality",
        "Find correlations between variables",
        "Detect patterns in the data",
        "Generate insights from my dataset"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing: '{query}'")
        
        try:
            # Process through orchestrator
            response = await orchestrator.route_request(
                user_input=query,
                context={"dataframe": sample_data}
            )
            
            if response.success:
                print(f"   ✅ Routed to: {response.agent_name}")
                print(f"   ⏱️  Processing time: {response.processing_time:.3f}s")
                print(f"   🎯 Confidence: {response.confidence_score:.2f}")
                
                if response.suggestions:
                    print(f"   💭 Suggestions: {len(response.suggestions)}")
                    
            else:
                print(f"   ❌ Failed: {response.error}")
                
        except Exception as e:
            print(f"   💥 Exception: {str(e)}")
    
    # Test agent suggestion
    print(f"\n🎯 Testing Agent Suggestion")
    suggestion = orchestrator.suggest_agent("analyze my customer data")
    print(f"   Suggested agent: {suggestion.get('suggested_agent', 'unknown')}")
    print(f"   Confidence: {suggestion.get('confidence', 0):.2f}")
    
    # Cleanup
    await orchestrator.shutdown()
    print(f"\n🎉 Integration test complete!")

if __name__ == "__main__":
    asyncio.run(test_integration())