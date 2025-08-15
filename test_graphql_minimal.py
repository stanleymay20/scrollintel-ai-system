"""Minimal test to verify GraphQL schema structure."""

import strawberry
from typing import List, Optional, Dict, Any
from datetime import datetime


@strawberry.type
class SimpleDataset:
    """Simple dataset type for testing."""
    id: str
    name: str
    quality_score: float


@strawberry.type
class Query:
    """Simple query for testing."""
    
    @strawberry.field
    def hello(self) -> str:
        return "Hello GraphQL!"
    
    @strawberry.field
    def dataset(self, dataset_id: str) -> Optional[SimpleDataset]:
        return SimpleDataset(
            id=dataset_id,
            name="Test Dataset",
            quality_score=0.85
        )
    
    @strawberry.field
    def datasets(self, limit: int = 10) -> List[SimpleDataset]:
        return [
            SimpleDataset(
                id=f"dataset_{i}",
                name=f"Dataset {i}",
                quality_score=0.8 + (i * 0.01)
            )
            for i in range(1, min(limit + 1, 6))
        ]


@strawberry.input
class DatasetInput:
    """Input for creating datasets."""
    name: str
    description: str = ""


@strawberry.type
class Mutation:
    """Simple mutation for testing."""
    
    @strawberry.field
    def create_dataset(self, input: DatasetInput) -> SimpleDataset:
        return SimpleDataset(
            id="new_dataset_id",
            name=input.name,
            quality_score=0.0
        )


# Create simple schema
simple_schema = strawberry.Schema(query=Query, mutation=Mutation)


async def test_simple_graphql():
    """Test simple GraphQL operations."""
    print("Testing simple GraphQL schema...")
    
    # Test query
    query = """
    query {
        hello
        dataset(datasetId: "test123") {
            id
            name
            qualityScore
        }
        datasets(limit: 3) {
            id
            name
            qualityScore
        }
    }
    """
    
    result = await simple_schema.execute(query)
    print("Query result:")
    print("Data:", result.data)
    print("Errors:", result.errors)
    
    # Test mutation
    mutation = """
    mutation CreateDataset($input: DatasetInput!) {
        createDataset(input: $input) {
            id
            name
            qualityScore
        }
    }
    """
    
    variables = {
        "input": {
            "name": "New Test Dataset",
            "description": "Created via GraphQL"
        }
    }
    
    result = await simple_schema.execute(mutation, variable_values=variables)
    print("\nMutation result:")
    print("Data:", result.data)
    print("Errors:", result.errors)
    
    print("\nGraphQL schema test completed successfully!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_simple_graphql())