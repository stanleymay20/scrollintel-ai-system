"""
Tests for ScrollQA API routes.
Tests the FastAPI endpoints for natural language data querying.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import status

from scrollintel.api.routes.scroll_qa_routes import router, get_scroll_qa_engine
from scrollintel.engines.scroll_qa_engine import QueryType
from scrollintel.models.database import User
from scrollintel.core.interfaces import UserRole


class TestScrollQARoutes:
    """Test cases for ScrollQA API routes."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user for authentication."""
        user = Mock(spec=User)
        user.id = "test-user-id"
        user.email = "test@example.com"
        user.role = UserRole.ANALYST
        return user
    
    @pytest.fixture
    def mock_engine(self):
        """Create mock ScrollQA engine."""
        engine = AsyncMock()
        engine.execute = AsyncMock()
        engine.get_status = Mock(return_value={
            "healthy": True,
            "openai_configured": True,
            "vector_store_configured": True,
            "redis_configured": True,
            "datasets_indexed": 2
        })
        engine.get_metrics = Mock(return_value={
            "engine_id": "scroll_qa_engine",
            "usage_count": 10,
            "error_count": 0
        })
        engine.health_check = AsyncMock(return_value=True)
        engine.dataset_schemas = {"customers": {}, "orders": {}}
        return engine
    
    @pytest.mark.asyncio
    async def test_process_natural_language_query(self, client, mock_user, mock_engine):
        """Test natural language query processing endpoint."""
        
        # Mock engine response
        mock_engine.execute.return_value = {
            "query_type": QueryType.SQL_GENERATION,
            "original_query": "Show me all customers",
            "generated_sql": "SELECT * FROM customers;",
            "results": {
                "data": [{"id": 1, "name": "John Doe"}],
                "row_count": 1
            },
            "execution_time": "2023-06-01T12:00:00",
            "datasets_used": ["customers"]
        }
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log') as mock_audit:
                    
                    response = client.post(
                        "/scroll-qa/query",
                        json={
                            "query": "Show me all customers",
                            "datasets": ["customers"],
                            "query_type": QueryType.SQL_GENERATION
                        }
                    )
                    
                    assert response.status_code == status.HTTP_200_OK
                    
                    data = response.json()
                    assert data["query_type"] == QueryType.SQL_GENERATION
                    assert data["original_query"] == "Show me all customers"
                    assert "results" in data
                    
                    # Verify engine was called correctly
                    mock_engine.execute.assert_called_once()
                    call_args = mock_engine.execute.call_args[0][0]
                    assert call_args["action"] == "query"
                    assert call_args["query"] == "Show me all customers"
                    assert call_args["datasets"] == ["customers"]
                    
                    # Verify audit log was called
                    mock_audit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_sql_query(self, client, mock_user, mock_engine):
        """Test SQL query generation endpoint."""
        
        mock_engine.execute.return_value = {
            "query_type": QueryType.SQL_GENERATION,
            "original_query": "Count customers",
            "generated_sql": "SELECT COUNT(*) FROM customers;",
            "results": {"data": [{"count": 100}]},
            "execution_time": "2023-06-01T12:00:00",
            "datasets_used": ["customers"]
        }
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log'):
                    
                    response = client.post(
                        "/scroll-qa/sql-query",
                        json={
                            "query": "Count customers",
                            "datasets": ["customers"]
                        }
                    )
                    
                    assert response.status_code == status.HTTP_200_OK
                    
                    data = response.json()
                    assert data["query_type"] == QueryType.SQL_GENERATION
                    assert "SELECT COUNT(*)" in data["results"]["generated_sql"]
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, client, mock_user, mock_engine):
        """Test semantic search endpoint."""
        
        mock_engine.execute.return_value = {
            "query_type": QueryType.SEMANTIC_SEARCH,
            "original_query": "customer satisfaction",
            "search_results": [
                {
                    "content": "Customer satisfaction is high",
                    "metadata": {"dataset": "reviews"},
                    "relevance_score": 0.95
                }
            ],
            "contextual_response": "Customer satisfaction appears to be high based on recent data.",
            "execution_time": "2023-06-01T12:00:00",
            "datasets_searched": ["reviews"]
        }
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log'):
                    
                    response = client.post(
                        "/scroll-qa/semantic-search",
                        json={
                            "query": "customer satisfaction",
                            "datasets": ["reviews"]
                        }
                    )
                    
                    assert response.status_code == status.HTTP_200_OK
                    
                    data = response.json()
                    assert data["query_type"] == QueryType.SEMANTIC_SEARCH
                    assert len(data["results"]["search_results"]) == 1
                    assert data["results"]["search_results"][0]["relevance_score"] == 0.95
    
    @pytest.mark.asyncio
    async def test_context_aware_query(self, client, mock_user, mock_engine):
        """Test context-aware query endpoint."""
        
        mock_engine.execute.return_value = {
            "query_type": QueryType.CONTEXT_AWARE,
            "original_query": "analyze customer trends",
            "sql_results": {"data": [{"trend": "increasing"}]},
            "semantic_results": {"insights": ["positive sentiment"]},
            "comprehensive_response": "Customer trends show positive growth with increasing satisfaction.",
            "execution_time": "2023-06-01T12:00:00",
            "datasets_used": ["customers", "reviews"]
        }
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log'):
                    
                    response = client.post(
                        "/scroll-qa/context-aware-query",
                        json={
                            "query": "analyze customer trends",
                            "datasets": ["customers", "reviews"]
                        }
                    )
                    
                    assert response.status_code == status.HTTP_200_OK
                    
                    data = response.json()
                    assert data["query_type"] == QueryType.CONTEXT_AWARE
                    assert "sql_results" in data["results"]
                    assert "semantic_results" in data["results"]
                    assert "comprehensive_response" in data["results"]
    
    @pytest.mark.asyncio
    async def test_multi_source_query(self, client, mock_user, mock_engine):
        """Test multi-source query endpoint."""
        
        mock_engine.execute.return_value = {
            "query_type": QueryType.MULTI_SOURCE,
            "original_query": "compare all data sources",
            "individual_results": {
                "customers": {"data": [{"count": 100}]},
                "orders": {"data": [{"count": 500}]}
            },
            "combined_analysis": "Strong correlation between customer count and order volume.",
            "execution_time": "2023-06-01T12:00:00",
            "datasets_queried": ["customers", "orders"]
        }
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log'):
                    
                    response = client.post(
                        "/scroll-qa/multi-source-query",
                        json={
                            "query": "compare all data sources",
                            "datasets": ["customers", "orders"]
                        }
                    )
                    
                    assert response.status_code == status.HTTP_200_OK
                    
                    data = response.json()
                    assert data["query_type"] == QueryType.MULTI_SOURCE
                    assert "individual_results" in data["results"]
                    assert "combined_analysis" in data["results"]
    
    @pytest.mark.asyncio
    async def test_multi_source_query_no_datasets(self, client, mock_user, mock_engine):
        """Test multi-source query with no datasets specified."""
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                
                response = client.post(
                    "/scroll-qa/multi-source-query",
                    json={
                        "query": "compare all data sources",
                        "datasets": []
                    }
                )
                
                assert response.status_code == status.HTTP_400_BAD_REQUEST
                assert "At least one dataset must be specified" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_index_dataset(self, client, mock_user, mock_engine):
        """Test dataset indexing endpoint."""
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log') as mock_audit:
                    
                    response = client.post(
                        "/scroll-qa/index-dataset",
                        json={
                            "dataset_name": "products",
                            "dataset_path": "/path/to/products.csv",
                            "description": "Product catalog data"
                        }
                    )
                    
                    assert response.status_code == status.HTTP_200_OK
                    
                    data = response.json()
                    assert data["status"] == "indexing_started"
                    assert data["dataset_name"] == "products"
                    assert "background" in data["message"]
                    
                    # Verify audit log was called
                    mock_audit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_dataset_schema(self, client, mock_user, mock_engine):
        """Test dataset schema retrieval endpoint."""
        
        mock_engine.execute.return_value = {
            "dataset_name": "customers",
            "schema": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "name", "type": "VARCHAR", "nullable": False}
                ],
                "source_type": "database"
            }
        }
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                
                response = client.get("/scroll-qa/schema/customers")
                
                assert response.status_code == status.HTTP_200_OK
                
                data = response.json()
                assert data["dataset_name"] == "customers"
                assert "schema" in data
                assert len(data["schema"]["columns"]) == 2
    
    @pytest.mark.asyncio
    async def test_get_dataset_schema_not_found(self, client, mock_user, mock_engine):
        """Test dataset schema retrieval for non-existent dataset."""
        
        mock_engine.execute.return_value = {
            "dataset_name": "nonexistent",
            "error": "Dataset not found"
        }
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                
                response = client.get("/scroll-qa/schema/nonexistent")
                
                assert response.status_code == status.HTTP_404_NOT_FOUND
                assert "Dataset not found" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_list_available_datasets(self, client, mock_user, mock_engine):
        """Test listing available datasets endpoint."""
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                
                response = client.get("/scroll-qa/datasets")
                
                assert response.status_code == status.HTTP_200_OK
                
                data = response.json()
                assert "datasets" in data
                assert "total_datasets" in data
                assert "engine_status" in data
                assert data["total_datasets"] == 2
    
    @pytest.mark.asyncio
    async def test_clear_query_cache(self, client, mock_user, mock_engine):
        """Test cache clearing endpoint."""
        
        mock_engine.execute.return_value = {
            "status": "success",
            "keys_cleared": 5
        }
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log') as mock_audit:
                    
                    response = client.delete("/scroll-qa/cache")
                    
                    assert response.status_code == status.HTTP_200_OK
                    
                    data = response.json()
                    assert data["status"] == "success"
                    assert data["keys_cleared"] == 5
                    
                    # Verify audit log was called
                    mock_audit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_engine_status(self, client, mock_user, mock_engine):
        """Test engine status endpoint."""
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                
                response = client.get("/scroll-qa/status")
                
                assert response.status_code == status.HTTP_200_OK
                
                data = response.json()
                assert "status" in data
                assert "metrics" in data
                assert "health_check" in data
                assert data["status"]["healthy"] is True
                assert data["health_check"] is True
    
    @pytest.mark.asyncio
    async def test_get_supported_query_types(self, client):
        """Test supported query types endpoint."""
        
        response = client.get("/scroll-qa/query-types")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "query_types" in data
        assert len(data["query_types"]) == 4
        
        query_types = [qt["type"] for qt in data["query_types"]]
        assert QueryType.SQL_GENERATION in query_types
        assert QueryType.SEMANTIC_SEARCH in query_types
        assert QueryType.CONTEXT_AWARE in query_types
        assert QueryType.MULTI_SOURCE in query_types
    
    @pytest.mark.asyncio
    async def test_get_example_queries(self, client):
        """Test example queries endpoint."""
        
        response = client.get("/scroll-qa/examples")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "examples" in data
        assert QueryType.SQL_GENERATION in data["examples"]
        assert QueryType.SEMANTIC_SEARCH in data["examples"]
        assert QueryType.CONTEXT_AWARE in data["examples"]
        assert QueryType.MULTI_SOURCE in data["examples"]
        
        # Check that examples are provided for each query type
        for query_type, examples in data["examples"].items():
            assert isinstance(examples, list)
            assert len(examples) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client, mock_user, mock_engine):
        """Test error handling in API routes."""
        
        # Test engine error
        mock_engine.execute.side_effect = Exception("Engine error")
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                with patch('scrollintel.api.routes.scroll_qa_routes.audit_log') as mock_audit:
                    
                    response = client.post(
                        "/scroll-qa/query",
                        json={
                            "query": "test query",
                            "datasets": ["customers"]
                        }
                    )
                    
                    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
                    assert "Engine error" in response.json()["detail"]
                    
                    # Verify error was logged in audit
                    mock_audit.assert_called_once()
                    audit_call = mock_audit.call_args[1]
                    assert audit_call["details"]["success"] is False
    
    @pytest.mark.asyncio
    async def test_authentication_required(self, client, mock_engine):
        """Test that authentication is required for protected endpoints."""
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
            # This would normally fail with authentication error
            # The actual authentication testing depends on the auth middleware setup
            pass
    
    @pytest.mark.asyncio
    async def test_request_validation(self, client, mock_user, mock_engine):
        """Test request validation."""
        
        with patch('scrollintel.api.routes.scroll_qa_routes.get_current_user', return_value=mock_user):
            with patch('scrollintel.api.routes.scroll_qa_routes.get_scroll_qa_engine', return_value=mock_engine):
                
                # Test missing required field
                response = client.post(
                    "/scroll-qa/query",
                    json={
                        "datasets": ["customers"]
                        # Missing "query" field
                    }
                )
                
                assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    async def test_engine_dependency_injection(self):
        """Test engine dependency injection."""
        
        # Test that engine is properly created and cached
        engine1 = await get_scroll_qa_engine()
        engine2 = await get_scroll_qa_engine()
        
        # Should return the same instance (singleton pattern)
        assert engine1 is engine2


class TestScrollQARoutesIntegration:
    """Integration tests for ScrollQA routes with real dependencies."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_query_flow(self):
        """Test complete query flow with real engine."""
        # This would test the full flow with real engine instance
        # Implementation depends on test environment setup
        pass
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        # This would test concurrent request handling
        # Implementation depends on test environment setup
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])