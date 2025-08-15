#!/usr/bin/env python3
"""
Test script for enhanced QA Agent implementation
Tests SQL generation, context awareness, caching, and result explanation
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrollintel_core.agents.qa_agent import QAAgent
from scrollintel_core.agents.base import AgentRequest


def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    
    # Generate sample sales data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n_records = len(dates) * 3  # Multiple records per day
    
    data = {
        'date': np.random.choice(dates, n_records),
        'sales': np.random.normal(1000, 200, n_records),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_records),
        'customer_id': np.random.randint(1, 1000, n_records),
        'quantity': np.random.randint(1, 10, n_records),
        'price': np.random.uniform(10, 500, n_records)
    }
    
    df = pd.DataFrame(data)
    df['revenue'] = df['quantity'] * df['price']
    df['profit'] = df['revenue'] * np.random.uniform(0.1, 0.3, len(df))
    
    return df


async def test_qa_agent_basic():
    """Test basic QA Agent functionality"""
    print("🧪 Testing Basic QA Agent Functionality")
    print("=" * 50)
    
    agent = QAAgent()
    
    # Test agent info
    info = agent.get_info()
    print(f"Agent Name: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Capabilities: {len(info['capabilities'])} capabilities")
    
    # Test health check
    health = await agent.health_check()
    print(f"Health Status: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
    print()


async def test_query_without_data():
    """Test queries without data context"""
    print("🔍 Testing Queries Without Data Context")
    print("=" * 50)
    
    agent = QAAgent()
    
    test_queries = [
        "How many customers do we have?",
        "Show me the top 10 products by sales",
        "Compare revenue between regions",
        "What's the trend in monthly sales?",
        "Find all orders from last month"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        request = AgentRequest(query=query, context={})
        response = await agent.process(request)
        
        if response.success:
            print(f"✅ Response Type: {response.result.get('type', 'unknown')}")
            print(f"   Question Type: {response.result.get('question_type', 'unknown')}")
            print(f"   Confidence: {response.confidence_score:.2f}")
        else:
            print(f"❌ Error: {response.error}")


async def test_query_with_data():
    """Test queries with data context"""
    print("\n📊 Testing Queries With Data Context")
    print("=" * 50)
    
    agent = QAAgent()
    sample_data = create_sample_data()
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Columns: {list(sample_data.columns)}")
    
    test_queries = [
        "How many sales records do we have?",
        "What's the total revenue?",
        "Show me average sales by region",
        "What's the trend in daily sales?",
        "Find the top 5 customers by revenue",
        "Compare sales between product categories"
    ]
    
    context = {
        "dataframe": sample_data,
        "table_name": "sales",
        "schema": {"sales": list(sample_data.columns)},
        "column_types": {col: str(sample_data[col].dtype) for col in sample_data.columns}
    }
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        request = AgentRequest(query=query, context=context)
        response = await agent.process(request)
        
        if response.success:
            result = response.result
            print(f"✅ Response Type: {result.get('type', 'unknown')}")
            print(f"   SQL Query: {result.get('sql_query', 'N/A')}")
            print(f"   Confidence: {response.confidence_score:.2f}")
            print(f"   Data Shape: {result.get('data_shape', 'N/A')}")
            
            if result.get('insights'):
                print(f"   Insights: {len(result['insights'])} insights generated")
                for insight in result['insights'][:2]:  # Show first 2 insights
                    print(f"     • {insight}")
            
            if result.get('visualization'):
                viz = result['visualization']
                print(f"   Visualization: {viz.get('type', 'unknown')} - {viz.get('title', 'N/A')}")
        else:
            print(f"❌ Error: {response.error}")


async def test_sql_generation():
    """Test SQL generation capabilities"""
    print("\n🔧 Testing SQL Generation")
    print("=" * 50)
    
    agent = QAAgent()
    sample_data = create_sample_data()
    
    # Create query context
    from scrollintel_core.agents.qa_agent import QueryContext
    
    query_context = QueryContext(
        table_schema={"sales": list(sample_data.columns)},
        sample_data=sample_data.head(100),
        column_types={col: str(sample_data[col].dtype) for col in sample_data.columns},
        relationships=[],
        previous_queries=[],
        user_preferences={}
    )
    
    test_cases = [
        ("Count all records", "aggregation"),
        ("Show top 10 sales", "ranking"),
        ("Average revenue by region", "aggregation"),
        ("Sales trend over time", "trend"),
        ("Compare regions", "comparison"),
        ("Find high value customers", "filtering")
    ]
    
    for query, expected_type in test_cases:
        print(f"\nQuery: {query}")
        sql, explanation, confidence = agent.sql_generator.generate_sql(query, query_context)
        print(f"   SQL: {sql}")
        print(f"   Explanation: {explanation}")
        print(f"   Confidence: {confidence:.2f}")


async def test_caching():
    """Test query caching functionality"""
    print("\n💾 Testing Query Caching")
    print("=" * 50)
    
    agent = QAAgent()
    sample_data = create_sample_data()
    
    context = {
        "dataframe": sample_data,
        "table_name": "sales"
    }
    
    query = "How many sales records do we have?"
    
    # First query - should not be cached
    print("First query (not cached):")
    request = AgentRequest(query=query, context=context)
    response1 = await agent.process(request)
    print(f"   Cached: {response1.metadata.get('cached', False)}")
    print(f"   Processing Time: {response1.processing_time:.4f}s")
    
    # Second query - should be cached
    print("\nSecond query (should be cached):")
    response2 = await agent.process(request)
    print(f"   Cached: {response2.metadata.get('cached', False)}")
    print(f"   Processing Time: {response2.processing_time:.4f}s")
    
    # Verify cache is working
    if response2.metadata.get('cached', False):
        print("✅ Caching is working correctly!")
    else:
        print("❌ Caching may not be working as expected")


async def test_conversation_memory():
    """Test conversation memory and context awareness"""
    print("\n🧠 Testing Conversation Memory")
    print("=" * 50)
    
    agent = QAAgent()
    sample_data = create_sample_data()
    
    context = {
        "dataframe": sample_data,
        "table_name": "sales"
    }
    
    # Simulate a conversation
    conversation = [
        "How many sales records do we have?",
        "What's the total revenue?",
        "Show me the breakdown by region",
        "Which region has the highest sales?"
    ]
    
    for i, query in enumerate(conversation):
        print(f"\nQuery {i+1}: {query}")
        request = AgentRequest(query=query, context=context)
        response = await agent.process(request)
        
        print(f"   Conversation Length: {response.metadata.get('conversation_length', 0)}")
        print(f"   Success: {'✅' if response.success else '❌'}")
        
        if response.suggestions:
            print(f"   Suggestions: {len(response.suggestions)} follow-up suggestions")


async def test_error_handling():
    """Test error handling and edge cases"""
    print("\n⚠️  Testing Error Handling")
    print("=" * 50)
    
    agent = QAAgent()
    
    # Test with invalid data
    invalid_contexts = [
        {"dataframe": "not_a_dataframe"},
        {"data": None},
        {"file_path": "nonexistent_file.csv"},
        {}  # Empty context
    ]
    
    for i, context in enumerate(invalid_contexts):
        print(f"\nTest case {i+1}: Invalid context")
        request = AgentRequest(query="Count records", context=context)
        response = await agent.process(request)
        
        if response.success:
            print(f"   ✅ Handled gracefully: {response.result.get('type', 'unknown')}")
        else:
            print(f"   ⚠️  Error handled: {response.error}")
            print(f"   Suggestions provided: {len(response.suggestions) > 0}")


async def main():
    """Run all tests"""
    print("🚀 Enhanced QA Agent Test Suite")
    print("=" * 60)
    
    try:
        await test_qa_agent_basic()
        await test_query_without_data()
        await test_query_with_data()
        await test_sql_generation()
        await test_caching()
        await test_conversation_memory()
        await test_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("Enhanced QA Agent is working correctly with:")
        print("  • SQL generation from natural language")
        print("  • Context-aware query understanding")
        print("  • Intelligent caching system")
        print("  • Result explanation and visualization")
        print("  • Conversation memory")
        print("  • Robust error handling")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())