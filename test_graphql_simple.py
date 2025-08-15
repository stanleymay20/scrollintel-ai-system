"""Simple test to verify GraphQL implementation."""

import asyncio
from ai_data_readiness.api.graphql.schema import schema


async def test_simple_query():
    """Test a simple GraphQL query."""
    query = """
    query {
        systemMetrics
    }
    """
    
    result = await schema.execute(query)
    print("Query result:", result)
    print("Data:", result.data)
    print("Errors:", result.errors)
    
    return result


async def test_dataset_query():
    """Test dataset query."""
    query = """
    query {
        dataset(datasetId: "test-id") {
            id
            name
            qualityScore
            status
        }
    }
    """
    
    result = await schema.execute(query)
    print("Dataset query result:", result)
    print("Data:", result.data)
    print("Errors:", result.errors)
    
    return result


async def main():
    """Run tests."""
    print("Testing GraphQL schema...")
    
    print("\n1. Testing system metrics query:")
    await test_simple_query()
    
    print("\n2. Testing dataset query:")
    await test_dataset_query()
    
    print("\nGraphQL tests completed!")


if __name__ == "__main__":
    asyncio.run(main())