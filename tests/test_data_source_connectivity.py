"""
Integration tests for data source connectivity framework.
"""
import pytest
import asyncio
import tempfile
import json
import csv
import os
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scrollintel.models.data_source_models import (
    DataSourceConfig, ConnectionTest, DataSchema, 
    DataSourceType, ConnectionStatus, Base
)
from scrollintel.core.connection_manager import ConnectionManager
from scrollintel.connectors.database_connectors import DatabaseConnector
from scrollintel.connectors.api_connectors import RestApiConnector, GraphQLConnector
from scrollintel.connectors.file_connectors import FileSystemConnector
from scrollintel.connectors.streaming_connectors import StreamingConnector

@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture
def connection_manager(db_session):
    """Create connection manager instance."""
    return ConnectionManager(db_session)

@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', 'age', 'email'])
        writer.writerow([1, 'John Doe', 30, 'john@example.com'])
        writer.writerow([2, 'Jane Smith', 25, 'jane@example.com'])
        writer.writerow([3, 'Bob Johnson', 35, 'bob@example.com'])
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

@pytest.fixture
def sample_json_file():
    """Create a sample JSON file for testing."""
    data = [
        {"id": 1, "name": "John Doe", "age": 30, "email": "john@example.com"},
        {"id": 2, "name": "Jane Smith", "age": 25, "email": "jane@example.com"},
        {"id": 3, "name": "Bob Johnson", "age": 35, "email": "bob@example.com"}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

class TestConnectionManager:
    """Test connection manager functionality."""
    
    @pytest.mark.asyncio
    async def test_create_data_source(self, connection_manager):
        """Test creating a new data source configuration."""
        config_data = {
            "name": "Test Database",
            "description": "Test PostgreSQL database",
            "source_type": "database",
            "connection_config": {
                "database_type": "postgresql",
                "host": "localhost",
                "port": 5432,
                "database": "test_db"
            },
            "auth_config": {
                "username": "test_user",
                "password": "test_pass"
            },
            "created_by": "test_user"
        }
        
        with patch.object(connection_manager, 'test_connection', return_value=True):
            config = await connection_manager.create_data_source(config_data)
            
            assert config.name == "Test Database"
            assert config.source_type == DataSourceType.DATABASE
            assert config.connection_config["database_type"] == "postgresql"
    
    @pytest.mark.asyncio
    async def test_health_check(self, connection_manager, db_session):
        """Test health check functionality."""
        # Create a test data source
        config = DataSourceConfig(
            id="test-id",
            name="Test Source",
            source_type=DataSourceType.DATABASE,
            connection_config={"database_type": "postgresql"},
            status=ConnectionStatus.ACTIVE
        )
        db_session.add(config)
        db_session.commit()
        
        with patch.object(connection_manager, 'test_connection', return_value=True):
            results = await connection_manager.health_check()
            
            assert "test-id" in results
            assert results["test-id"]["status"] == "healthy"

class TestDatabaseConnector:
    """Test database connector functionality."""
    
    def test_validate_config(self):
        """Test database configuration validation."""
        connector = DatabaseConnector()
        
        # Valid config
        valid_config = {
            "database_type": "postgresql",
            "host": "localhost",
            "database": "test_db"
        }
        errors = connector.validate_config(valid_config)
        assert len(errors) == 0
        
        # Invalid config - missing required fields
        invalid_config = {
            "database_type": "postgresql"
        }
        errors = connector.validate_config(invalid_config)
        assert len(errors) > 0
        assert any("host" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_postgresql_test_connection_mock(self):
        """Test PostgreSQL connection test with mocking."""
        connector = DatabaseConnector()
        
        connection_config = {
            "database_type": "postgresql",
            "host": "localhost",
            "port": 5432,
            "database": "test_db"
        }
        auth_config = {
            "username": "test_user",
            "password": "test_pass"
        }
        
        with patch('asyncpg.connect') as mock_connect:
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = "PostgreSQL 13.0"
            mock_connect.return_value = mock_conn
            
            success, error, details = await connector.test_connection(connection_config, auth_config)
            
            assert success is True
            assert error is None
            assert "version" in details

class TestRestApiConnector:
    """Test REST API connector functionality."""
    
    def test_validate_config(self):
        """Test REST API configuration validation."""
        connector = RestApiConnector()
        
        # Valid config
        valid_config = {
            "base_url": "https://api.example.com"
        }
        errors = connector.validate_config(valid_config)
        assert len(errors) == 0
        
        # Invalid config - missing protocol
        invalid_config = {
            "base_url": "api.example.com"
        }
        errors = connector.validate_config(invalid_config)
        assert len(errors) > 0
    
    @pytest.mark.asyncio
    async def test_rest_api_test_connection_mock(self):
        """Test REST API connection test with mocking."""
        connector = RestApiConnector()
        
        connection_config = {
            "base_url": "https://api.example.com",
            "health_endpoint": "/health"
        }
        auth_config = {
            "type": "bearer",
            "token": "test_token"
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"status": "ok"}
            mock_response.content_type = "application/json"
            mock_response.headers = {}
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            success, error, details = await connector.test_connection(connection_config, auth_config)
            
            assert success is True
            assert error is None
            assert details["status_code"] == 200

class TestGraphQLConnector:
    """Test GraphQL connector functionality."""
    
    def test_validate_config(self):
        """Test GraphQL configuration validation."""
        connector = GraphQLConnector()
        
        # Valid config
        valid_config = {
            "endpoint": "https://api.example.com/graphql"
        }
        errors = connector.validate_config(valid_config)
        assert len(errors) == 0
        
        # Invalid config - missing protocol
        invalid_config = {
            "endpoint": "api.example.com/graphql"
        }
        errors = connector.validate_config(invalid_config)
        assert len(errors) > 0
    
    @pytest.mark.asyncio
    async def test_graphql_test_connection_mock(self):
        """Test GraphQL connection test with mocking."""
        connector = GraphQLConnector()
        
        connection_config = {
            "endpoint": "https://api.example.com/graphql"
        }
        auth_config = {
            "type": "api_key",
            "api_key": "test_key"
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "data": {
                    "__schema": {
                        "queryType": {"name": "Query"}
                    }
                }
            }
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            success, error, details = await connector.test_connection(connection_config, auth_config)
            
            assert success is True
            assert error is None
            assert "schema_info" in details

class TestFileSystemConnector:
    """Test file system connector functionality."""
    
    def test_validate_config(self):
        """Test file system configuration validation."""
        connector = FileSystemConnector()
        
        # Valid config
        valid_config = {
            "file_path": "/path/to/file.csv",
            "file_format": "csv"
        }
        errors = connector.validate_config(valid_config)
        assert len(errors) == 0
        
        # Invalid config - missing required fields
        invalid_config = {
            "file_path": "/path/to/file.csv"
        }
        errors = connector.validate_config(invalid_config)
        assert len(errors) > 0
    
    @pytest.mark.asyncio
    async def test_csv_file_operations(self, sample_csv_file):
        """Test CSV file operations."""
        connector = FileSystemConnector()
        
        connection_config = {
            "file_path": sample_csv_file,
            "file_format": "csv",
            "has_header": True
        }
        
        # Test connection
        success, error, details = await connector.test_connection(connection_config, {})
        assert success is True
        assert error is None
        assert details["file_format"] == "csv"
        
        # Create connection
        connection = await connector.create_connection(connection_config, {})
        assert connection["file_format"] == "csv"
        
        # Discover schema
        schemas = await connector.discover_schema(connection)
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["table_name"] == os.path.splitext(os.path.basename(sample_csv_file))[0]
        assert len(schema["columns"]) == 4  # id, name, age, email
        
        # Read data
        data = await connector.read_data(connection, {"limit": 10})
        assert len(data) == 3
        assert data[0]["name"] == "John Doe"
    
    @pytest.mark.asyncio
    async def test_json_file_operations(self, sample_json_file):
        """Test JSON file operations."""
        connector = FileSystemConnector()
        
        connection_config = {
            "file_path": sample_json_file,
            "file_format": "json"
        }
        
        # Test connection
        success, error, details = await connector.test_connection(connection_config, {})
        assert success is True
        assert error is None
        assert details["data_type"] == "array"
        
        # Create connection
        connection = await connector.create_connection(connection_config, {})
        
        # Discover schema
        schemas = await connector.discover_schema(connection)
        assert len(schemas) == 1
        schema = schemas[0]
        assert len(schema["columns"]) == 4  # id, name, age, email
        
        # Read data
        data = await connector.read_data(connection, {"limit": 10})
        assert len(data) == 3
        assert data[0]["name"] == "John Doe"

class TestStreamingConnector:
    """Test streaming connector functionality."""
    
    def test_validate_config(self):
        """Test streaming configuration validation."""
        connector = StreamingConnector()
        
        # Valid Kafka config
        valid_kafka_config = {
            "stream_type": "kafka",
            "topic": "test_topic"
        }
        errors = connector.validate_config(valid_kafka_config)
        assert len(errors) == 0
        
        # Invalid config - missing required fields
        invalid_config = {
            "stream_type": "kafka"
        }
        errors = connector.validate_config(invalid_config)
        assert len(errors) > 0
        assert any("topic" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_kafka_test_connection_mock(self):
        """Test Kafka connection test with mocking."""
        connector = StreamingConnector()
        
        connection_config = {
            "stream_type": "kafka",
            "bootstrap_servers": ["localhost:9092"],
            "topic": "test_topic"
        }
        auth_config = {}
        
        with patch('aiokafka.AIOKafkaConsumer') as mock_consumer_class:
            mock_consumer = AsyncMock()
            mock_consumer._client.cluster.topics.return_value = {"test_topic", "other_topic"}
            mock_consumer_class.return_value = mock_consumer
            
            success, error, details = await connector.test_connection(connection_config, auth_config)
            
            assert success is True
            assert error is None
            assert "available_topics" in details
            assert "test_topic" in details["available_topics"]

class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_data_source_lifecycle(self, connection_manager, sample_csv_file):
        """Test complete data source lifecycle from creation to data reading."""
        
        # Create data source
        config_data = {
            "name": "Test CSV File",
            "description": "Test CSV data source",
            "source_type": "file_system",
            "connection_config": {
                "file_path": sample_csv_file,
                "file_format": "csv",
                "has_header": True
            },
            "auth_config": {},
            "created_by": "test_user"
        }
        
        config = await connection_manager.create_data_source(config_data)
        assert config.status == ConnectionStatus.ACTIVE
        
        # Discover schema
        schemas = await connection_manager.discover_schema(config.id)
        assert len(schemas) > 0
        
        # Get connection and read data
        connection = await connection_manager.get_connection(config.id)
        connector = connection_manager.connectors[DataSourceType.FILE_SYSTEM]
        data = await connector.read_data(connection, {"limit": 10})
        
        assert len(data) == 3
        assert all(isinstance(row, dict) for row in data)
    
    @pytest.mark.asyncio
    async def test_multiple_data_sources_health_check(self, connection_manager, sample_csv_file, sample_json_file):
        """Test health check with multiple data sources."""
        
        # Create CSV data source
        csv_config = await connection_manager.create_data_source({
            "name": "CSV Source",
            "source_type": "file_system",
            "connection_config": {
                "file_path": sample_csv_file,
                "file_format": "csv"
            },
            "auth_config": {}
        })
        
        # Create JSON data source
        json_config = await connection_manager.create_data_source({
            "name": "JSON Source",
            "source_type": "file_system",
            "connection_config": {
                "file_path": sample_json_file,
                "file_format": "json"
            },
            "auth_config": {}
        })
        
        # Perform health check
        health_results = await connection_manager.health_check()
        
        assert len(health_results) == 2
        assert csv_config.id in health_results
        assert json_config.id in health_results
        assert all(result["status"] == "healthy" for result in health_results.values())

if __name__ == "__main__":
    pytest.main([__file__, "-v"])