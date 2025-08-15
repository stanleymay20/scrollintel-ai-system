"""
Unit tests for ScrollQA Engine.
Tests natural language to SQL conversion and semantic search functionality.
"""

import pytest
import pytest_asyncio
import asyncio
import pandas as pd
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from scrollintel.engines.scroll_qa_engine import ScrollQAEngine, QueryType
from scrollintel.engines.base_engine import EngineStatus


class TestScrollQAEngine:
    """Test cases for ScrollQA Engine."""
    
    @pytest_asyncio.fixture
    async def engine(self):
        """Create a ScrollQA engine instance for testing."""
        engine = ScrollQAEngine()
        
        # Mock external dependencies
        with patch('scrollintel.engines.scroll_qa_engine.get_config') as mock_config:
            mock_config.return_value = Mock(
                ai_services=Mock(
                    openai_api_key="test-key",
                    openai_model="gpt-4"
                ),
                database=Mock(
                    redis_host="localhost",
                    redis_port=6379,
                    redis_password=None,
                    redis_db=0,
                    postgres_url="postgresql://test:test@localhost/test",
                    pinecone_api_key="test-pinecone-key",
                    pinecone_environment="test-env"
                )
            )
            
            # Mock OpenAI client
            engine.openai_client = AsyncMock()
            
            # Mock Redis client
            engine.redis_client = AsyncMock()
            engine.redis_client.ping = AsyncMock()
            
            # Mock vector store
            engine.vector_store = Mock()
            
            # Mock embeddings
            engine.embeddings = Mock()
            
            # Set up test schemas
            engine.dataset_schemas = {
                "customers": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "nullable": False},
                        {"name": "name", "type": "VARCHAR", "nullable": False},
                        {"name": "email", "type": "VARCHAR", "nullable": True},
                        {"name": "city", "type": "VARCHAR", "nullable": True}
                    ],
                    "source_type": "database",
                    "table_name": "customers"
                },
                "orders": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "nullable": False},
                        {"name": "customer_id", "type": "INTEGER", "nullable": False},
                        {"name": "amount", "type": "DECIMAL", "nullable": False},
                        {"name": "order_date", "type": "DATE", "nullable": False}
                    ],
                    "source_type": "database",
                    "table_name": "orders"
                }
            }
            
            await engine.initialize()
            
        yield engine
        
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.engine_id == "scroll_qa_engine"
        assert engine.name == "ScrollQA Engine"
        assert engine.status == EngineStatus.READY
        assert len(engine.dataset_schemas) == 2
    
    @pytest.mark.asyncio
    async def test_get_status(self, engine):
        """Test engine status reporting."""
        status = engine.get_status()
        
        assert status["healthy"] is True
        assert status["openai_configured"] is True
        assert status["vector_store_configured"] is True
        assert status["redis_configured"] is True
        assert status["datasets_indexed"] == 2
    
    @pytest.mark.asyncio
    async def test_sql_generation_query(self, engine):
        """Test natural language to SQL conversion."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "SELECT * FROM customers WHERE city = 'New York';"
        
        engine.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Mock SQL execution
        with patch.object(engine, '_execute_sql_query') as mock_execute:
            mock_execute.return_value = {
                "data": [{"id": 1, "name": "John Doe", "city": "New York"}],
                "columns": ["id", "name", "city"],
                "row_count": 1,
                "query_executed": "SELECT * FROM customers WHERE city = 'New York';"
            }
            
            input_data = {
                "action": "query",
                "query": "Show me all customers from New York",
                "datasets": ["customers"],
                "query_type": QueryType.SQL_GENERATION
            }
            
            result = await engine.process(input_data)
            
            assert result["query_type"] == QueryType.SQL_GENERATION
            assert result["original_query"] == "Show me all customers from New York"
            assert "SELECT * FROM customers" in result["generated_sql"]
            assert result["results"]["row_count"] == 1
    
    @pytest.mark.asyncio
    async def test_semantic_search_query(self, engine):
        """Test semantic search functionality."""
        # Mock vector store search
        mock_docs = [
            Mock(
                page_content="Customer satisfaction data shows high ratings",
                metadata={"dataset": "customers", "row_index": 1},
                score=0.95
            ),
            Mock(
                page_content="Product quality metrics indicate improvement",
                metadata={"dataset": "products", "row_index": 2},
                score=0.87
            )
        ]
        
        engine.vector_store.similarity_search = Mock(return_value=mock_docs)
        
        # Mock contextual response generation
        engine.openai_client.chat.completions.create = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Based on the data, customer satisfaction is high with positive trends."
        engine.openai_client.chat.completions.create.return_value = mock_response
        
        input_data = {
            "action": "query",
            "query": "What do we know about customer satisfaction?",
            "datasets": ["customers"],
            "query_type": QueryType.SEMANTIC_SEARCH
        }
        
        with patch('asyncio.to_thread', return_value=mock_docs):
            result = await engine.process(input_data)
        
        assert result["query_type"] == QueryType.SEMANTIC_SEARCH
        assert len(result["search_results"]) == 2
        assert result["search_results"][0]["relevance_score"] == 0.95
        assert "customer satisfaction" in result["contextual_response"].lower()
    
    @pytest.mark.asyncio
    async def test_context_aware_query(self, engine):
        """Test context-aware query processing."""
        # Mock SQL generation
        with patch.object(engine, '_generate_sql_query') as mock_sql:
            mock_sql.return_value = {
                "query_type": QueryType.SQL_GENERATION,
                "results": {"data": [{"customer_count": 100}]}
            }
            
            # Mock semantic search
            with patch.object(engine, '_semantic_search') as mock_semantic:
                mock_semantic.return_value = {
                    "search_results": [{"content": "Customer insights"}]
                }
                
                # Mock comprehensive response
                engine.openai_client.chat.completions.create = AsyncMock()
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "Comprehensive analysis shows 100 customers with positive insights."
                engine.openai_client.chat.completions.create.return_value = mock_response
                
                input_data = {
                    "action": "query",
                    "query": "Analyze our customer base",
                    "datasets": ["customers"],
                    "query_type": QueryType.CONTEXT_AWARE
                }
                
                result = await engine.process(input_data)
                
                assert result["query_type"] == QueryType.CONTEXT_AWARE
                assert "sql_results" in result
                assert "semantic_results" in result
                assert "comprehensive_response" in result
    
    @pytest.mark.asyncio
    async def test_multi_source_query(self, engine):
        """Test multi-source data querying."""
        # Mock individual dataset queries
        with patch.object(engine, '_generate_sql_query') as mock_sql:
            mock_sql.side_effect = [
                {
                    "query_type": QueryType.SQL_GENERATION,
                    "results": {"data": [{"customer_count": 100}]}
                },
                {
                    "query_type": QueryType.SQL_GENERATION,
                    "results": {"data": [{"order_count": 500}]}
                }
            ]
            
            # Mock multi-source analysis
            engine.openai_client.chat.completions.create = AsyncMock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Analysis shows 100 customers with 500 orders, indicating good engagement."
            engine.openai_client.chat.completions.create.return_value = mock_response
            
            input_data = {
                "action": "query",
                "query": "Compare customers and orders",
                "datasets": ["customers", "orders"],
                "query_type": QueryType.MULTI_SOURCE
            }
            
            result = await engine.process(input_data)
            
            assert result["query_type"] == QueryType.MULTI_SOURCE
            assert len(result["individual_results"]) == 2
            assert "customers" in result["individual_results"]
            assert "orders" in result["individual_results"]
            assert "combined_analysis" in result
    
    @pytest.mark.asyncio
    async def test_dataset_indexing(self, engine):
        """Test dataset indexing for semantic search."""
        # Create test CSV data
        test_data = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "description": ["Great customer", "Regular buyer", "New customer"]
        })
        
        # Mock pandas read_csv
        with patch('pandas.read_csv', return_value=test_data):
            # Mock vector store add_documents
            engine.vector_store.add_documents = Mock()
            
            input_data = {
                "action": "index_dataset",
                "dataset_name": "test_customers",
                "dataset_path": "/path/to/test.csv"
            }
            
            with patch('asyncio.to_thread'):
                result = await engine.process(input_data)
            
            assert result["dataset_name"] == "test_customers"
            assert result["rows_processed"] == 3
            assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_get_dataset_schema(self, engine):
        """Test dataset schema retrieval."""
        input_data = {
            "action": "get_schema",
            "dataset_name": "customers"
        }
        
        result = await engine.process(input_data)
        
        assert result["dataset_name"] == "customers"
        assert "schema" in result
        assert len(result["schema"]["columns"]) == 4
        assert result["schema"]["columns"][0]["name"] == "id"
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, engine):
        """Test query result caching."""
        # Test cache key generation
        cache_key = engine._generate_cache_key(
            "test query",
            ["customers"],
            QueryType.SQL_GENERATION
        )
        
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length
        
        # Test cache storage and retrieval
        test_result = {"test": "data"}
        
        await engine._cache_result(cache_key, test_result)
        engine.redis_client.setex.assert_called_once()
        
        # Mock cached data retrieval
        import pickle
        engine.redis_client.get = AsyncMock(return_value=pickle.dumps(test_result))
        
        cached_result = await engine._get_cached_result(cache_key)
        assert cached_result == test_result
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, engine):
        """Test cache clearing functionality."""
        # Mock Redis keys and delete
        engine.redis_client.keys = AsyncMock(return_value=["scrollqa:key1", "scrollqa:key2"])
        engine.redis_client.delete = AsyncMock()
        
        input_data = {"action": "clear_cache"}
        result = await engine.process(input_data)
        
        assert result["status"] == "success"
        assert result["keys_cleared"] == 2
        engine.redis_client.delete.assert_called_once_with("scrollqa:key1", "scrollqa:key2")
    
    @pytest.mark.asyncio
    async def test_sql_query_cleaning(self, engine):
        """Test SQL query cleaning and validation."""
        # Test markdown removal
        dirty_sql = "```sql\nSELECT * FROM customers;\n```"
        clean_sql = engine._clean_sql_query(dirty_sql)
        assert clean_sql == "SELECT * FROM customers;"
        
        # Test comment removal
        commented_sql = "SELECT * FROM customers; -- This is a comment\n-- Another comment\nWHERE id = 1"
        clean_sql = engine._clean_sql_query(commented_sql)
        assert "-- This is a comment" not in clean_sql
        assert "WHERE id = 1;" in clean_sql
    
    @pytest.mark.asyncio
    async def test_schema_context_creation(self, engine):
        """Test schema context formatting for AI."""
        schemas = {
            "customers": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "name", "type": "VARCHAR", "nullable": True}
                ]
            }
        }
        
        context = engine._create_schema_context(schemas)
        
        assert "Table: customers" in context
        assert "id (INTEGER) NOT NULL" in context
        assert "name (VARCHAR) NULL" in context
    
    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling in various scenarios."""
        # Test invalid action
        with pytest.raises(ValueError, match="Unknown action"):
            await engine.process({"action": "invalid_action"})
        
        # Test missing query
        with pytest.raises(ValueError, match="Query is required"):
            await engine.process({
                "action": "query",
                "datasets": ["customers"],
                "query_type": QueryType.SQL_GENERATION
            })
        
        # Test missing dataset for schema request
        result = await engine.process({
            "action": "get_schema",
            "dataset_name": "nonexistent"
        })
        assert "error" in result
        assert result["error"] == "Dataset not found"
    
    @pytest.mark.asyncio
    async def test_sql_execution_error_handling(self, engine):
        """Test SQL execution error handling."""
        with patch('sqlalchemy.create_engine') as mock_engine:
            mock_conn = Mock()
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
            mock_conn.execute.side_effect = Exception("SQL Error")
            
            result = await engine._execute_sql_query("SELECT * FROM invalid_table;")
            
            assert "error" in result
            assert "SQL Error" in result["error"]
    
    def test_query_type_constants(self):
        """Test QueryType constants."""
        assert QueryType.SQL_GENERATION == "sql_generation"
        assert QueryType.SEMANTIC_SEARCH == "semantic_search"
        assert QueryType.CONTEXT_AWARE == "context_aware"
        assert QueryType.MULTI_SOURCE == "multi_source"
    
    @pytest.mark.asyncio
    async def test_health_check(self, engine):
        """Test engine health check."""
        health_status = await engine.health_check()
        assert health_status is True
    
    @pytest.mark.asyncio
    async def test_engine_metrics(self, engine):
        """Test engine metrics reporting."""
        metrics = engine.get_metrics()
        
        assert "engine_id" in metrics
        assert metrics["engine_id"] == "scroll_qa_engine"
        assert "status" in metrics
        assert "capabilities" in metrics
        assert "usage_count" in metrics
        assert "error_count" in metrics


@pytest.mark.asyncio
async def test_engine_lifecycle():
    """Test complete engine lifecycle."""
    engine = ScrollQAEngine()
    
    # Mock dependencies
    with patch('scrollintel.engines.scroll_qa_engine.get_config') as mock_config:
        mock_config.return_value = Mock(
            ai_services=Mock(openai_api_key=None),
            database=Mock(
                redis_host="localhost",
                redis_port=6379,
                postgres_url="postgresql://test:test@localhost/test"
            )
        )
        
        # Test initialization without OpenAI
        await engine.initialize()
        assert engine.status == EngineStatus.READY
        
        # Test cleanup
        await engine.cleanup()
        assert len(engine.query_context) == 0
        assert len(engine.dataset_schemas) == 0


if __name__ == "__main__":
    pytest.main([__file__])