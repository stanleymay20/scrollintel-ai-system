#!/usr/bin/env python3
"""
Test script for Data Scientist Agent implementation
Tests all major capabilities with sample data
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the scrollintel_core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scrollintel_core'))

from agents.data_scientist_agent import DataScientistAgent
from agents.base import AgentRequest

def create_sample_data():
    """Create sample datasets for testing"""
    np.random.seed(42)
    
    # Sample 1: Sales data with correlations and patterns
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    sales_data = pd.DataFrame({
        'date': dates,
        'revenue': np.random.normal(10000, 2000, 1000) + np.sin(np.arange(1000) * 2 * np.pi / 365) * 1000,
        'marketing_spend': np.random.normal(2000, 500, 1000),
        'customer_count': np.random.poisson(100, 1000),
        'product_category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'temperature': np.random.normal(20, 10, 1000)
    })
    
    # Add correlation between marketing spend and revenue
    sales_data['revenue'] += sales_data['marketing_spend'] * 0.3 + np.random.normal(0, 500, 1000)
    
    # Add some missing values
    sales_data.loc[np.random.choice(1000, 50, replace=False), 'marketing_spend'] = np.nan
    sales_data.loc[np.random.choice(1000, 30, replace=False), 'customer_count'] = np.nan
    
    # Add some duplicates
    sales_data = pd.concat([sales_data, sales_data.iloc[:20]], ignore_index=True)
    
    # Sample 2: Customer data with quality issues
    customer_data = pd.DataFrame({
        'customer_id': range(1, 501),
        'age': np.random.normal(35, 12, 500),
        'income': np.random.lognormal(10, 0.5, 500),
        'satisfaction_score': np.random.uniform(1, 10, 500),
        'purchase_frequency': np.random.poisson(5, 500),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], 500),
        'subscription_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], 500)
    })
    
    # Add outliers
    customer_data.loc[np.random.choice(500, 10, replace=False), 'income'] *= 10
    customer_data.loc[np.random.choice(500, 5, replace=False), 'age'] = np.random.uniform(100, 120, 5)
    
    return sales_data, customer_data

async def test_data_scientist_agent():
    """Test the Data Scientist Agent with various scenarios"""
    print("🧪 Testing Data Scientist Agent Implementation")
    print("=" * 60)
    
    # Initialize agent
    agent = DataScientistAgent()
    print(f"✅ Agent initialized: {agent.name}")
    print(f"📋 Capabilities: {len(agent.get_capabilities())} capabilities")
    
    # Create sample data
    sales_data, customer_data = create_sample_data()
    print(f"📊 Created sample datasets:")
    print(f"   - Sales data: {sales_data.shape}")
    print(f"   - Customer data: {customer_data.shape}")
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Comprehensive Analysis",
            "query": "Analyze my sales data comprehensively",
            "data": sales_data,
            "description": "Full exploratory data analysis"
        },
        {
            "name": "Data Quality Assessment",
            "query": "Assess the quality of my customer data",
            "data": customer_data,
            "description": "Data quality metrics and recommendations"
        },
        {
            "name": "Correlation Analysis",
            "query": "Find correlations in my sales data",
            "data": sales_data,
            "description": "Relationship analysis between variables"
        },
        {
            "name": "Pattern Detection",
            "query": "Detect patterns in my customer data",
            "data": customer_data,
            "description": "Clustering and pattern identification"
        },
        {
            "name": "Outlier Detection",
            "query": "Find outliers and anomalies in my data",
            "data": customer_data,
            "description": "Anomaly detection analysis"
        },
        {
            "name": "Insights Generation",
            "query": "Generate insights from my sales data",
            "data": sales_data,
            "description": "Automated insights extraction"
        },
        {
            "name": "Data Profiling",
            "query": "Profile my customer dataset",
            "data": customer_data,
            "description": "Comprehensive data profiling"
        },
        {
            "name": "No Data Guidance",
            "query": "How do I analyze my data?",
            "data": None,
            "description": "Guidance when no data is provided"
        }
    ]
    
    print("\n🔬 Running Test Scenarios")
    print("-" * 40)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Query: '{scenario['query']}'")
        print(f"   Description: {scenario['description']}")
        
        try:
            # Create request
            context = {"dataframe": scenario["data"]} if scenario["data"] is not None else {}
            request = AgentRequest(
                query=scenario["query"],
                context=context,
                parameters={}
            )
            
            # Process request
            response = await agent.process(request)
            
            # Display results
            if response.success:
                print(f"   ✅ Success (confidence: {response.confidence_score:.2f})")
                print(f"   ⏱️  Processing time: {response.processing_time:.3f}s")
                
                # Show key results
                if isinstance(response.result, dict):
                    result_keys = list(response.result.keys())
                    print(f"   📊 Result sections: {', '.join(result_keys[:5])}")
                    
                    # Show specific insights for different analysis types
                    if "insights" in response.result:
                        insights = response.result["insights"]
                        if isinstance(insights, list) and insights:
                            print(f"   💡 Key insight: {insights[0]}")
                    elif "quality_score" in response.result:
                        score = response.result["quality_score"]
                        print(f"   📈 Quality score: {score:.1f}/100")
                    elif "correlation_analysis" in response.result:
                        corr_data = response.result["correlation_analysis"]
                        if "strong_correlations" in corr_data:
                            strong_count = len(corr_data["strong_correlations"])
                            print(f"   🔗 Strong correlations found: {strong_count}")
                
                # Show suggestions
                if response.suggestions:
                    print(f"   💭 Suggestions: {len(response.suggestions)} provided")
                
                # Show visualization recommendations
                if "visualization_recommendations" in response.metadata:
                    viz_count = len(response.metadata["visualization_recommendations"])
                    print(f"   📈 Visualization recommendations: {viz_count}")
                
            else:
                print(f"   ❌ Failed: {response.error}")
                
        except Exception as e:
            print(f"   💥 Exception: {str(e)}")
    
    # Test health check
    print(f"\n🏥 Health Check")
    print("-" * 20)
    health = await agent.health_check()
    print(f"   Status: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
    print(f"   Capabilities: {len(health['capabilities'])}")
    
    # Test agent info
    print(f"\n📋 Agent Information")
    print("-" * 20)
    info = agent.get_info()
    print(f"   Name: {info['name']}")
    print(f"   Description: {info['description']}")
    print(f"   Capabilities: {len(info['capabilities'])}")
    print(f"   Status: {'✅ Healthy' if info['healthy'] else '❌ Unhealthy'}")
    
    print(f"\n🎉 Data Scientist Agent Testing Complete!")
    print("=" * 60)

def test_specific_capabilities():
    """Test specific data science capabilities"""
    print("\n🔬 Testing Specific Capabilities")
    print("-" * 40)
    
    # Test data extraction
    agent = DataScientistAgent()
    
    # Test with different data formats
    test_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['x', 'y', 'z', 'x', 'y'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    # Test data extraction from different context formats
    contexts = [
        {"dataframe": test_data},
        {"data": test_data},
        {"data": test_data.to_dict()},
        {"data": test_data.values.tolist()},
        {}  # No data
    ]
    
    for i, context in enumerate(contexts):
        extracted = agent._extract_data_from_context(context)
        status = "✅ Success" if extracted is not None else "❌ No data"
        print(f"   Context {i+1}: {status}")
    
    # Test analysis type classification
    queries = [
        "analyze my data",
        "check data quality",
        "find correlations",
        "detect patterns",
        "find outliers",
        "generate insights",
        "profile my dataset",
        "what can you do?"
    ]
    
    print(f"\n🏷️  Query Classification Test")
    for query in queries:
        analysis_type = agent._classify_analysis_type(query)
        print(f"   '{query}' → {analysis_type}")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_data_scientist_agent())
    test_specific_capabilities()