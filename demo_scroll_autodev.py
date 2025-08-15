#!/usr/bin/env python3
"""
Demo script for ScrollAutoDev Agent - Advanced Prompt Engineering
Demonstrates prompt optimization, A/B testing, template generation, and chain management.
"""

import asyncio
import os
import sys
from uuid import uuid4
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrollintel.agents.scroll_autodev_agent import ScrollAutoDevAgent, PromptOptimizationStrategy, PromptCategory
from scrollintel.core.interfaces import AgentRequest


async def demo_prompt_optimization():
    """Demonstrate prompt optimization capabilities."""
    print("🚀 ScrollAutoDev Agent - Prompt Optimization Demo")
    print("=" * 60)
    
    agent = ScrollAutoDevAgent()
    
    # Test prompt optimization
    print("\n1. PROMPT OPTIMIZATION")
    print("-" * 30)
    
    original_prompt = "Analyze the sales data"
    print(f"Original Prompt: '{original_prompt}'")
    
    request = AgentRequest(
        id=str(uuid4()),
        user_id=str(uuid4()),
        agent_id=agent.agent_id,
        prompt=f"optimize {original_prompt}",
        context={
            "strategy": PromptOptimizationStrategy.A_B_TESTING.value,
            "test_data": [
                "Q1 2023 sales: $1.2M revenue, 15% growth",
                "Q2 2023 sales: $1.4M revenue, 20% growth",
                "Q3 2023 sales: $1.1M revenue, -5% decline"
            ],
            "target_metric": "performance_score",
            "max_variations": 5
        }
    )
    
    try:
        print("Processing optimization request...")
        response = await agent.process_request(request)
        
        if response.status.value == "success":
            print("✅ Optimization completed successfully!")
            print(f"Execution time: {response.execution_time:.2f}s")
            print("\nOptimization Results:")
            print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
        else:
            print(f"❌ Optimization failed: {response.error_message}")
    
    except Exception as e:
        print(f"❌ Error during optimization: {str(e)}")


async def demo_prompt_testing():
    """Demonstrate prompt variation testing."""
    print("\n\n2. PROMPT VARIATION TESTING")
    print("-" * 30)
    
    agent = ScrollAutoDevAgent()
    
    variations = [
        "Analyze the quarterly sales data and provide insights",
        "Examine Q3 sales performance with detailed breakdown",
        "Review sales metrics and identify trends and patterns",
        "Conduct comprehensive analysis of sales data with recommendations"
    ]
    
    test_cases = [
        "Q3 2023: Revenue $1.1M, Units 2,500, Avg deal $440",
        "Q2 2023: Revenue $1.4M, Units 3,200, Avg deal $437",
        "Q1 2023: Revenue $1.2M, Units 2,800, Avg deal $428"
    ]
    
    print(f"Testing {len(variations)} prompt variations with {len(test_cases)} test cases")
    
    request = AgentRequest(
        id=str(uuid4()),
        user_id=str(uuid4()),
        agent_id=agent.agent_id,
        prompt="test prompt variations",
        context={
            "variations": variations,
            "test_cases": test_cases,
            "evaluation_criteria": ["accuracy", "relevance", "clarity", "actionability"],
            "statistical_significance": 0.05
        }
    )
    
    try:
        print("Running A/B tests...")
        response = await agent.process_request(request)
        
        if response.status.value == "success":
            print("✅ Testing completed successfully!")
            print(f"Execution time: {response.execution_time:.2f}s")
            print("\nTesting Results:")
            print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
        else:
            print(f"❌ Testing failed: {response.error_message}")
    
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")


async def demo_prompt_chain():
    """Demonstrate prompt chain management."""
    print("\n\n3. PROMPT CHAIN MANAGEMENT")
    print("-" * 30)
    
    agent = ScrollAutoDevAgent()
    
    chain_config = {
        "name": "Sales Analysis Workflow",
        "description": "Multi-step sales data analysis with dependency management",
        "prompts": [
            {
                "id": "data_validation",
                "prompt": "Validate and clean the sales data: {raw_data}. Check for missing values, outliers, and data quality issues.",
                "dependencies": []
            },
            {
                "id": "trend_analysis",
                "prompt": "Analyze sales trends from the validated data: {result_data_validation}. Identify patterns, seasonality, and growth rates.",
                "dependencies": ["data_validation"]
            },
            {
                "id": "performance_metrics",
                "prompt": "Calculate key performance metrics from the trend analysis: {result_trend_analysis}. Include KPIs, conversion rates, and efficiency metrics.",
                "dependencies": ["trend_analysis"]
            },
            {
                "id": "recommendations",
                "prompt": "Generate actionable recommendations based on performance metrics: {result_performance_metrics} and trends: {result_trend_analysis}.",
                "dependencies": ["performance_metrics", "trend_analysis"]
            }
        ],
        "dependencies": {
            "trend_analysis": ["data_validation"],
            "performance_metrics": ["trend_analysis"],
            "recommendations": ["performance_metrics", "trend_analysis"]
        }
    }
    
    execution_context = {
        "raw_data": "sales_q3_2023.csv with 10,000 records, 15 columns including date, product, revenue, quantity, customer_id"
    }
    
    print(f"Executing chain: '{chain_config['name']}'")
    print(f"Steps: {len(chain_config['prompts'])}")
    
    request = AgentRequest(
        id=str(uuid4()),
        user_id=str(uuid4()),
        agent_id=agent.agent_id,
        prompt="execute prompt chain",
        context={
            "chain": chain_config,
            "execution_context": execution_context
        }
    )
    
    try:
        print("Executing chain workflow...")
        response = await agent.process_request(request)
        
        if response.status.value == "success":
            print("✅ Chain execution completed successfully!")
            print(f"Execution time: {response.execution_time:.2f}s")
            print("\nChain Results:")
            print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
        else:
            print(f"❌ Chain execution failed: {response.error_message}")
    
    except Exception as e:
        print(f"❌ Error during chain execution: {str(e)}")


async def demo_template_generation():
    """Demonstrate industry-specific template generation."""
    print("\n\n4. TEMPLATE GENERATION")
    print("-" * 30)
    
    agent = ScrollAutoDevAgent()
    
    industries = [
        {
            "industry": "healthcare",
            "use_case": "patient_data_analysis",
            "requirements": ["HIPAA compliant", "medical terminology", "clinical insights"]
        },
        {
            "industry": "finance",
            "use_case": "risk_assessment",
            "requirements": ["regulatory compliance", "quantitative analysis", "risk metrics"]
        },
        {
            "industry": "retail",
            "use_case": "customer_behavior_analysis",
            "requirements": ["personalization", "conversion optimization", "seasonal trends"]
        }
    ]
    
    for industry_config in industries:
        print(f"\nGenerating templates for {industry_config['industry']} - {industry_config['use_case']}")
        
        request = AgentRequest(
            id=str(uuid4()),
            user_id=str(uuid4()),
            agent_id=agent.agent_id,
            prompt="generate templates",
            context={
                "industry": industry_config["industry"],
                "use_case": industry_config["use_case"],
                "requirements": industry_config["requirements"],
                "target_audience": "business analysts",
                "complexity_level": "intermediate"
            }
        )
        
        try:
            response = await agent.process_request(request)
            
            if response.status.value == "success":
                print(f"✅ Templates generated for {industry_config['industry']}")
                print(response.content[:300] + "..." if len(response.content) > 300 else response.content)
            else:
                print(f"❌ Template generation failed: {response.error_message}")
        
        except Exception as e:
            print(f"❌ Error generating templates: {str(e)}")


async def demo_prompt_analysis():
    """Demonstrate prompt quality analysis."""
    print("\n\n5. PROMPT QUALITY ANALYSIS")
    print("-" * 30)
    
    agent = ScrollAutoDevAgent()
    
    test_prompts = [
        "Analyze data",  # Poor quality - too vague
        "Analyze the quarterly sales data and provide insights with recommendations",  # Good quality
        "Please examine the Q3 2023 sales performance data including revenue, units sold, customer segments, and geographic distribution. Provide a comprehensive analysis with trend identification, performance metrics, and actionable business recommendations formatted as an executive summary.",  # Excellent quality
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nAnalyzing Prompt {i}: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        request = AgentRequest(
            id=str(uuid4()),
            user_id=str(uuid4()),
            agent_id=agent.agent_id,
            prompt=prompt,
            context={"analysis_mode": True}
        )
        
        try:
            response = await agent.process_request(request)
            
            if response.status.value == "success":
                print("✅ Analysis completed")
                # Extract quality score from response (simplified)
                if "Quality Score:" in response.content:
                    score_line = [line for line in response.content.split('\n') if 'Quality Score:' in line][0]
                    print(f"   {score_line}")
                print(f"   Analysis preview: {response.content[:200]}...")
            else:
                print(f"❌ Analysis failed: {response.error_message}")
        
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")


async def demo_agent_capabilities():
    """Demonstrate agent capabilities and health check."""
    print("\n\n6. AGENT CAPABILITIES & HEALTH")
    print("-" * 30)
    
    agent = ScrollAutoDevAgent()
    
    # Show capabilities
    print("Agent Capabilities:")
    capabilities = agent.get_capabilities()
    for cap in capabilities:
        print(f"  • {cap.name}: {cap.description}")
        print(f"    Input types: {', '.join(cap.input_types)}")
        print(f"    Output types: {', '.join(cap.output_types)}")
        print()
    
    # Health check
    print("Performing health check...")
    try:
        is_healthy = await agent.health_check()
        if is_healthy:
            print("✅ Agent is healthy and ready")
        else:
            print("⚠️ Agent health check failed")
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
    
    # Show agent info
    print(f"\nAgent Information:")
    print(f"  ID: {agent.agent_id}")
    print(f"  Name: {agent.name}")
    print(f"  Type: {agent.agent_type.value}")
    print(f"  Capabilities: {len(capabilities)}")
    print(f"  Template Categories: {len(agent.prompt_templates)}")


async def main():
    """Run all demos."""
    print("🎯 ScrollAutoDev Agent - Comprehensive Demo")
    print("Advanced Prompt Engineering, A/B Testing, and Template Generation")
    print("=" * 80)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not set. Some features may not work properly.")
        print("Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        print()
    
    try:
        # Run all demos
        await demo_prompt_optimization()
        await demo_prompt_testing()
        await demo_prompt_chain()
        await demo_template_generation()
        await demo_prompt_analysis()
        await demo_agent_capabilities()
        
        print("\n\n🎉 Demo completed successfully!")
        print("ScrollAutoDev Agent is ready for production use.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())