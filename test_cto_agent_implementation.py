#!/usr/bin/env python3
"""
Test script for the enhanced CTO Agent implementation
"""
import asyncio
import sys
import os

# Add the scrollintel_core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scrollintel_core'))

from agents.cto_agent import CTOAgent
from agents.base import AgentRequest


async def test_cto_agent():
    """Test the CTO Agent with various request types"""
    
    print("🚀 Testing Enhanced CTO Agent Implementation")
    print("=" * 50)
    
    # Initialize the CTO Agent
    cto_agent = CTOAgent()
    
    # Test cases
    test_cases = [
        {
            "name": "Technology Stack Recommendation",
            "query": "What technology stack should I use for my e-commerce startup?",
            "context": {
                "business_type": "ecommerce",
                "scale": "small",
                "budget": "medium",
                "team_size": "small",
                "timeline": "fast"
            }
        },
        {
            "name": "Technology Comparison",
            "query": "Compare React vs Vue for my frontend",
            "context": {
                "business_type": "saas",
                "team_size": "medium"
            }
        },
        {
            "name": "Architecture Guidance",
            "query": "How should I design my system architecture?",
            "context": {
                "scale": "medium",
                "complexity": "high"
            }
        },
        {
            "name": "Scaling Strategy",
            "query": "How do I scale my application from 1000 to 50000 users?",
            "context": {
                "current_users": 1000,
                "target_users": 50000,
                "growth_rate": "rapid"
            }
        },
        {
            "name": "Technology Trends",
            "query": "What are the latest technology trends I should consider?",
            "context": {}
        },
        {
            "name": "Team Guidance",
            "query": "What skills should I look for when hiring developers?",
            "context": {
                "current_team_size": 2,
                "target_team_size": 8,
                "budget": "high"
            }
        }
    ]
    
    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 30)
        print(f"Query: {test_case['query']}")
        
        # Create request
        request = AgentRequest(
            query=test_case['query'],
            context=test_case['context'],
            user_id="test_user",
            session_id="test_session"
        )
        
        try:
            # Process request
            response = await cto_agent.process(request)
            
            if response.success:
                print(f"✅ Success (Confidence: {response.confidence_score:.2f})")
                print(f"Request Type: {response.metadata.get('request_type', 'unknown')}")
                print(f"Processing Time: {response.processing_time:.3f}s")
                
                # Show key results
                if isinstance(response.result, dict):
                    if "recommendations" in response.result:
                        print("📋 Key Recommendations:")
                        recommendations = response.result["recommendations"]
                        if isinstance(recommendations, dict):
                            for category, details in list(recommendations.items())[:2]:
                                print(f"  • {category}: {details}")
                    
                    if "guidance" in response.result:
                        print(f"💡 Guidance: {response.result['guidance']}")
                    
                    if "comparison" in response.result:
                        comp = response.result["comparison"]
                        print(f"⚖️  Comparing: {comp.get('technology_1')} vs {comp.get('technology_2')}")
                
                # Show suggestions
                if response.suggestions:
                    print("💭 Follow-up suggestions:")
                    for suggestion in response.suggestions[:2]:
                        print(f"  • {suggestion}")
                        
            else:
                print(f"❌ Failed: {response.error}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    # Test health check
    print(f"\n🏥 Health Check")
    print("-" * 30)
    health = await cto_agent.health_check()
    print(f"Status: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
    print(f"Capabilities: {len(cto_agent.get_capabilities())} features")
    
    print(f"\n🎉 CTO Agent Implementation Test Complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_cto_agent())