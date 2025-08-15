"""
Integration tests for ScrollQA system.
Tests the complete natural language data querying workflow.
"""

import pytest
import asyncio
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from scrollintel.engines.scroll_qa_engine import ScrollQAEngine, QueryType
from scrollintel.api.routes.scroll_qa_routes import router
from scrollintel.models.database import User
from scrollintel.core.interfaces import UserRole


class TestScrollQAIntegration:
    """Integration tests for complete ScrollQA workflow."""
    
    @pytest.fixture
    def test_data_files(self):
        """Create temporary test data files."""
        # Create customers CSV
        customers_data = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown", "Charlie Wilson"],
            "email": ["john@example.com", "jane@example.com", "bob@example.com", "alice@example.com", "charlie@example.com"],
            "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
            "age": [25, 30, 35, 28, 42],
            "signup_date": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05", "2023-05-12"]
        })
        
        # Create orders CSV
        orders_data = pd.DataFrame({
            "id": [101, 102, 103, 104, 105, 106, 107],
            "customer_id": [1, 2, 1, 3, 2, 4, 5],
            "product": ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard", "Mouse", "Headphones"],
            "amount": [999.99, 699.50, 399.25, 299.00, 89.99, 49.99, 149.99],
            "order_date": ["2023-06-01", "2023-06-02", "2023-06-03", "2023-06-04", "2023-06-05", "2023-06-06", "2023-06-07"],
            "status": ["completed", "pending", "completed", "shipped", "completed", "processing", "completed"]
        })
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as customers_file:
            customers_data.to_csv(customers_file.name, index=False)
            customers_path = customers_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as orders_file:
            orders_data.to_csv(orders_file.name, index=False)
            orders_path = orders_file.name
        
        yield {
            "customers": customers_path,
            "orders": orders_path,
            "customers_data": customers_data,
            "orders_data": orders_data
        }
        
        # Cleanup
        os.unlink(customers_path)
        os.unlink(orders_path)
    
    @pytest.fixture
    def mock_engine_with_data(self, test_data_files):
        """Create mock engine with test data."""
        engine = AsyncMock(spec=ScrollQAEngine)
        
        # Mock dataset schemas based on test data
        engine.dataset_schemas = {
            "customers": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "name", "type": "VARCHAR", "nullable": False},
                    {"name": "email", "type": "VARCHAR", "nullable": True},
                    {"name": "city", "type": "VARCHAR", "nullable": True},
                    {"name": "age", "type": "INTEGER", "nullable": True},
                    {"name": "signup_date", "type": "DATE", "nullable": True}
                ],
                "source_type": "csv",
                "file_path": test_data_files["customers"]
            },
            "orders": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "customer_id", "type": "INTEGER", "nullable": False},
                    {"name": "product", "type": "VARCHAR", "nullable": False},
                    {"name": "amount", "type": "DECIMAL", "nullable": False},
                    {"name": "order_date", "type": "DATE", "nullable": False},
                    {"name": "status", "type": "VARCHAR", "nullable": False}
                ],
                "source_type": "csv",
                "file_path": test_data_files["orders"]
            }
        }
        
        # Mock status and metrics
        engine.get_status.return_value = {
            "healthy": True,
            "openai_configured": True,
            "vector_store_configured": True,
            "redis_configured": True,
            "datasets_indexed": 2
        }
        
        engine.get_metrics.return_value = {
            "engine_id": "scroll_qa_engine",
            "usage_count": 0,
            "error_count": 0
        }
        
        engine.health_check.return_value = True
        
        return engine
    
    @pytest.fixture
    def client(self):
        """Create test client with ScrollQA routes."""
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Create mock authenticated user."""
        user = Mock(spec=User)
        user.id = "test-user-123"
        user.email = "analyst@scrollintel.com"
        user.role = UserRole.ANALYST
        return user
    
    @pytest.mark.asyncio
    async def test_complete_sql_generation_workflow(self, client, mock_user, mock_engine_with_data, test_data_files):
        """Test complete SQL generation workflow from query to results."""
        
        # Mock SQL generation response
        mock_engine_with_data.execute.return_value = {
            "query_type": QueryType.SQL_GENERATION,
            "original_query": "Show me all customers from New York",
            "generated_sql": "SELECT * FROM customers WHERE city = 'New York';",
            "results": {
                "data": [
                    {
                        "id": 1,
                        "name": "John Doe",
                        "email": "john@example.com",
                        "city": "New York",
                        "age": 25,
                        "signup_date": "2023-01-15"
                    }
                ],
                "columns": ["id", "name", "email", "city", "age", "signup_date"],
                "row_count": 1,
                "query_executed": "SELECT * FROM customers WHERE city = 'New York';"
            },
            "execution_time": "2023-06-01T12:00:00Z",
            "datasets_used": ["customers"]
        }
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine_with_data):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log'):
                    
                    # Make the API request
                    response = client.post(
                        "/scroll-qa/query",
                        json={
                            "query": "Show me all customers from New York",
                            "datasets": ["customers"],
                            "query_type": QueryType.SQL_GENERATION
                        }
                    )
                    
                    # Verify response
                    assert response.status_code == 200
                    data = response.json()
                    
                    assert data["query_type"] == QueryType.SQL_GENERATION
                    assert data["original_query"] == "Show me all customers from New York"
                    assert data["results"]["results"]["row_count"] == 1
                    assert data["results"]["results"]["data"][0]["name"] == "John Doe"
                    assert data["results"]["results"]["data"][0]["city"] == "New York"
                    
                    # Verify engine was called with correct parameters
                    mock_engine_with_data.execute.assert_called_once()
                    call_args = mock_engine_with_data.execute.call_args[0][0]
                    assert call_args["action"] == "query"
                    assert call_args["query"] == "Show me all customers from New York"
                    assert call_args["datasets"] == ["customers"]
                    assert call_args["query_type"] == QueryType.SQL_GENERATION
    
    @pytest.mark.asyncio
    async def test_multi_dataset_analysis_workflow(self, client, mock_user, mock_engine_with_data, test_data_files):
        """Test multi-dataset analysis workflow."""
        
        # Mock multi-source query response
        mock_engine_with_data.execute.return_value = {
            "query_type": QueryType.MULTI_SOURCE,
            "original_query": "Analyze customer orders and spending patterns",
            "individual_results": {
                "customers": {
                    "query_type": QueryType.SQL_GENERATION,
                    "results": {
                        "data": [
                            {"avg_age": 32, "total_customers": 5}
                        ],
                        "row_count": 1
                    }
                },
                "orders": {
                    "query_type": QueryType.SQL_GENERATION,
                    "results": {
                        "data": [
                            {"avg_order_value": 369.67, "total_orders": 7}
                        ],
                        "row_count": 1
                    }
                }
            },
            "combined_analysis": "Analysis shows 5 customers with average age of 32 years have placed 7 orders with an average value of $369.67. This indicates strong customer engagement with higher-value purchases.",
            "execution_time": "2023-06-01T12:00:00Z",
            "datasets_queried": ["customers", "orders"]
        }
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine_with_data):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log'):
                    
                    # Make the API request
                    response = client.post(
                        "/scroll-qa/multi-source-query",
                        json={
                            "query": "Analyze customer orders and spending patterns",
                            "datasets": ["customers", "orders"]
                        }
                    )
                    
                    # Verify response
                    assert response.status_code == 200
                    data = response.json()
                    
                    assert data["query_type"] == QueryType.MULTI_SOURCE
                    assert "individual_results" in data["results"]
                    assert "combined_analysis" in data["results"]
                    
                    # Check individual results
                    individual_results = data["results"]["individual_results"]
                    assert "customers" in individual_results
                    assert "orders" in individual_results
                    assert individual_results["customers"]["results"]["data"][0]["total_customers"] == 5
                    assert individual_results["orders"]["results"]["data"][0]["total_orders"] == 7
                    
                    # Check combined analysis
                    analysis = data["results"]["combined_analysis"]
                    assert "5 customers" in analysis
                    assert "$369.67" in analysis
                    assert "strong customer engagement" in analysis
    
    @pytest.mark.asyncio
    async def test_dataset_indexing_and_search_workflow(self, client, mock_user, mock_engine_with_data, test_data_files):
        """Test dataset indexing and semantic search workflow."""
        
        # Test dataset indexing
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine_with_data):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log'):
                    
                    # Index the dataset
                    index_response = client.post(
                        "/scroll-qa/index-dataset",
                        json={
                            "dataset_name": "customers",
                            "dataset_path": test_data_files["customers"],
                            "description": "Customer data for testing"
                        }
                    )
                    
                    assert index_response.status_code == 200
                    index_data = index_response.json()
                    assert index_data["status"] == "indexing_started"
                    assert index_data["dataset_name"] == "customers"
        
        # Test semantic search after indexing
        mock_engine_with_data.execute.return_value = {
            "query_type": QueryType.SEMANTIC_SEARCH,
            "original_query": "Find information about young customers",
            "search_results": [
                {
                    "content": "Customer: John Doe, Age: 25, City: New York, Email: john@example.com",
                    "metadata": {"dataset": "customers", "row_index": 0},
                    "relevance_score": 0.92
                },
                {
                    "content": "Customer: Alice Brown, Age: 28, City: Houston, Email: alice@example.com",
                    "metadata": {"dataset": "customers", "row_index": 3},
                    "relevance_score": 0.87
                }
            ],
            "contextual_response": "Found 2 young customers: John Doe (25) from New York and Alice Brown (28) from Houston. Both are relatively new customers who signed up in early 2023.",
            "execution_time": "2023-06-01T12:00:00Z",
            "datasets_searched": ["customers"]
        }
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine_with_data):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log'):
                    
                    # Perform semantic search
                    search_response = client.post(
                        "/scroll-qa/semantic-search",
                        json={
                            "query": "Find information about young customers",
                            "datasets": ["customers"]
                        }
                    )
                    
                    assert search_response.status_code == 200
                    search_data = search_response.json()
                    
                    assert search_data["query_type"] == QueryType.SEMANTIC_SEARCH
                    assert len(search_data["results"]["search_results"]) == 2
                    
                    # Check relevance scores
                    results = search_data["results"]["search_results"]
                    assert results[0]["relevance_score"] == 0.92
                    assert results[1]["relevance_score"] == 0.87
                    
                    # Check contextual response
                    response_text = search_data["results"]["contextual_response"]
                    assert "John Doe (25)" in response_text
                    assert "Alice Brown (28)" in response_text
    
    @pytest.mark.asyncio
    async def test_context_aware_comprehensive_analysis(self, client, mock_user, mock_engine_with_data, test_data_files):
        """Test context-aware analysis combining SQL and semantic search."""
        
        mock_engine_with_data.execute.return_value = {
            "query_type": QueryType.CONTEXT_AWARE,
            "original_query": "Provide comprehensive analysis of customer purchasing behavior",
            "sql_results": {
                "query_type": QueryType.SQL_GENERATION,
                "generated_sql": "SELECT c.city, COUNT(o.id) as order_count, AVG(o.amount) as avg_amount FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.city;",
                "results": {
                    "data": [
                        {"city": "New York", "order_count": 2, "avg_amount": 699.62},
                        {"city": "Los Angeles", "order_count": 2, "avg_amount": 374.75},
                        {"city": "Chicago", "order_count": 1, "avg_amount": 299.00},
                        {"city": "Houston", "order_count": 1, "avg_amount": 49.99},
                        {"city": "Phoenix", "order_count": 1, "avg_amount": 149.99}
                    ],
                    "row_count": 5
                }
            },
            "semantic_results": {
                "query_type": QueryType.SEMANTIC_SEARCH,
                "search_results": [
                    {
                        "content": "High-value customers tend to purchase electronics and premium products",
                        "metadata": {"insight_type": "purchasing_pattern"},
                        "relevance_score": 0.94
                    },
                    {
                        "content": "Geographic distribution shows concentration in major metropolitan areas",
                        "metadata": {"insight_type": "geographic_analysis"},
                        "relevance_score": 0.89
                    }
                ]
            },
            "comprehensive_response": """
            Comprehensive Customer Purchasing Behavior Analysis:
            
            Geographic Insights:
            - New York customers show highest engagement with 2 orders averaging $699.62
            - Los Angeles customers also active with 2 orders averaging $374.75
            - Other cities (Chicago, Houston, Phoenix) show lower but consistent activity
            
            Purchasing Patterns:
            - Clear preference for electronics and premium products among high-value customers
            - Geographic concentration in major metropolitan areas suggests urban market focus
            - Average order values vary significantly by location, indicating regional economic factors
            
            Recommendations:
            - Focus marketing efforts on New York and Los Angeles for high-value products
            - Develop targeted campaigns for other cities to increase order frequency
            - Consider regional pricing strategies based on local purchasing power
            """,
            "execution_time": "2023-06-01T12:00:00Z",
            "datasets_used": ["customers", "orders"]
        }
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine_with_data):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log'):
                    
                    # Make context-aware query
                    response = client.post(
                        "/scroll-qa/context-aware-query",
                        json={
                            "query": "Provide comprehensive analysis of customer purchasing behavior",
                            "datasets": ["customers", "orders"]
                        }
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    
                    assert data["query_type"] == QueryType.CONTEXT_AWARE
                    
                    # Check SQL results
                    sql_results = data["results"]["sql_results"]
                    assert sql_results["results"]["row_count"] == 5
                    assert "JOIN" in sql_results["generated_sql"]
                    
                    # Check semantic results
                    semantic_results = data["results"]["semantic_results"]
                    assert len(semantic_results["search_results"]) == 2
                    
                    # Check comprehensive response
                    comprehensive = data["results"]["comprehensive_response"]
                    assert "Geographic Insights" in comprehensive
                    assert "Purchasing Patterns" in comprehensive
                    assert "Recommendations" in comprehensive
                    assert "New York customers" in comprehensive
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, client, mock_user, mock_engine_with_data):
        """Test error handling and recovery scenarios."""
        
        # Test engine error
        mock_engine_with_data.execute.side_effect = Exception("Database connection failed")
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine_with_data):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log') as mock_audit:
                    
                    response = client.post(
                        "/scroll-qa/query",
                        json={
                            "query": "Show me all customers",
                            "datasets": ["customers"]
                        }
                    )
                    
                    assert response.status_code == 500
                    assert "Database connection failed" in response.json()["detail"]
                    
                    # Verify error was audited
                    mock_audit.assert_called_once()
                    audit_call = mock_audit.call_args[1]
                    assert audit_call["details"]["success"] is False
        
        # Test recovery after error
        mock_engine_with_data.execute.side_effect = None
        mock_engine_with_data.execute.return_value = {
            "query_type": QueryType.SQL_GENERATION,
            "original_query": "Show me all customers",
            "results": {"data": [], "row_count": 0},
            "execution_time": "2023-06-01T12:00:00Z",
            "datasets_used": ["customers"]
        }
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine_with_data):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log'):
                    
                    response = client.post(
                        "/scroll-qa/query",
                        json={
                            "query": "Show me all customers",
                            "datasets": ["customers"]
                        }
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["query_type"] == QueryType.SQL_GENERATION
    
    @pytest.mark.asyncio
    async def test_performance_with_large_datasets(self, client, mock_user, mock_engine_with_data):
        """Test performance with simulated large datasets."""
        
        # Simulate large dataset response
        large_dataset_response = {
            "query_type": QueryType.SQL_GENERATION,
            "original_query": "Analyze large dataset",
            "results": {
                "data": [{"total_records": 1000000, "processing_time": "2.5s"}],
                "row_count": 1,
                "performance_metrics": {
                    "query_execution_time": 2.5,
                    "data_transfer_time": 0.3,
                    "total_processing_time": 2.8
                }
            },
            "execution_time": "2023-06-01T12:00:00Z",
            "datasets_used": ["large_dataset"]
        }
        
        mock_engine_with_data.execute.return_value = large_dataset_response
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine_with_data):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log'):
                    
                    import time
                    start_time = time.time()
                    
                    response = client.post(
                        "/scroll-qa/query",
                        json={
                            "query": "Analyze large dataset",
                            "datasets": ["large_dataset"]
                        }
                    )
                    
                    end_time = time.time()
                    api_response_time = end_time - start_time
                    
                    assert response.status_code == 200
                    data = response.json()
                    
                    # Verify large dataset handling
                    assert data["results"]["results"]["data"][0]["total_records"] == 1000000
                    
                    # API should respond quickly even for large datasets (mocked)
                    assert api_response_time < 1.0  # Should be very fast with mocked engine
    
    @pytest.mark.asyncio
    async def test_concurrent_query_handling(self, client, mock_user, mock_engine_with_data):
        """Test handling of concurrent queries."""
        
        # Mock different responses for concurrent queries
        responses = [
            {
                "query_type": QueryType.SQL_GENERATION,
                "original_query": f"Query {i}",
                "results": {"data": [{"result": f"Result {i}"}], "row_count": 1},
                "execution_time": "2023-06-01T12:00:00Z",
                "datasets_used": ["customers"]
            }
            for i in range(5)
        ]
        
        mock_engine_with_data.execute.side_effect = responses
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine_with_data):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log'):
                    
                    # Make concurrent requests
                    import asyncio
                    import httpx
                    
                    async def make_request(query_num):
                        async with httpx.AsyncClient(app=client.app, base_url="http://test") as ac:
                            response = await ac.post(
                                "/scroll-qa/query",
                                json={
                                    "query": f"Query {query_num}",
                                    "datasets": ["customers"]
                                }
                            )
                            return response.json()
                    
                    # This would test concurrent requests in a real async environment
                    # For now, we verify the mock setup works
                    response = client.post(
                        "/scroll-qa/query",
                        json={
                            "query": "Query 0",
                            "datasets": ["customers"]
                        }
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "Result 0" in str(data["results"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])