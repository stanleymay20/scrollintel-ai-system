"""
Simple integration tests for data source connectivity framework.
Tests core functionality without requiring optional dependencies.
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
from scrollintel.connectors.base_connector import BaseConnector

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

class MockConnector(BaseConnector):
    """Mock connector for testing."""
    
    async def test_connection(self, connection_config, auth_config):
        return True, None, {"mock": "test_success"}
    
    async def create_connection(self, connection_config, auth_config):
        return {"mock": "connection", "config": connection_config}
    
    async def discover_schema(self, connection):
        return [{
            "schema_name": "mock_schema",
            "table_name": "mock_table",
            "columns": [
                {"name": "id", "type": "integer", "nullable": False, "primary_key": True},
                {"name": "name", "type": "string", "nullable": True, "primary_key": False}
            ]
        }]
    
    async def read_data(self, connection, query_config):
        return [
            {"id": 1, "name": "Test 1"},
            {"id": 2, "name": "Test 2"}
        ]

class TestConnectionManagerCore:
    """Test core connection manager functionality."""
    
    @pytest.mark.asyncio
    async def test_create_data_source_with_mock(self, connection_manager):
        """Test creating a data source with mock connector."""
        # Replace connector with mock
        mock_connector = MockConnector()
        connection_manager.connectors[DataSourceType.DATABASE] = mock_connector
        
        config_data = {
            "name": "Test Database",
            "description": "Test database connection",
            "source_type": "database",
            "connection_config": {
                "host": "localhost",
                "database": "test_db"
            },
            "auth_config": {
                "username": "test_user",
                "password": "test_pass"
            },
            "created_by": "test_user"
        }
        
        config = await connection_manager.create_data_source(config_data)
        
        assert config.name == "Test Database"
        assert config.source_type == DataSourceType.DATABASE
        assert config.status == ConnectionStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_get_connection(self, connection_manager, db_session):
        """Test getting a connection."""
        # Create test config
        config = DataSourceConfig(
            id="test-id",
            name="Test Source",
            source_type=DataSourceType.DATABASE,
            connection_config={"host": "localhost"},
            status=ConnectionStatus.ACTIVE
        )
        db_session.add(config)
        db_session.commit()
        
        # Replace connector with mock
        mock_connector = MockConnector()
        connection_manager.connectors[DataSourceType.DATABASE] = mock_connector
        
        connection = await connection_manager.get_connection("test-id")
        
        assert connection["mock"] == "connection"
        assert connection["config"]["host"] == "localhost"
    
    @pytest.mark.asyncio
    async def test_discover_schema_with_mock(self, connection_manager, db_session):
        """Test schema discovery with mock connector."""
        # Create test config
        config = DataSourceConfig(
            id="test-id",
            name="Test Source",
            source_type=DataSourceType.DATABASE,
            connection_config={"host": "localhost"},
            status=ConnectionStatus.ACTIVE
        )
        db_session.add(config)
        db_session.commit()
        
        # Replace connector with mock
        mock_connector = MockConnector()
        connection_manager.connectors[DataSourceType.DATABASE] = mock_connector
        
        schemas = await connection_manager.discover_schema("test-id")
        
        assert len(schemas) == 1
        assert schemas[0].table_name == "mock_table"
        assert len(schemas[0].columns) == 2
    
    @pytest.mark.asyncio
    async def test_health_check_with_mock(self, connection_manager, db_session):
        """Test health check with mock connector."""
        # Create test config
        config = DataSourceConfig(
            id="test-id",
            name="Test Source",
            source_type=DataSourceType.DATABASE,
            connection_config={"host": "localhost"},
            status=ConnectionStatus.ACTIVE
        )
        db_session.add(config)
        db_session.commit()
        
        # Replace connector with mock
        mock_connector = MockConnector()
        connection_manager.connectors[DataSourceType.DATABASE] = mock_connector
        
        results = await connection_manager.health_check()
        
        assert "test-id" in results
        assert results["test-id"]["status"] == "healthy"
        assert results["test-id"]["name"] == "Test Source"

class TestDataSourceModels:
    """Test data source model functionality."""
    
    def test_data_source_config_creation(self, db_session):
        """Test creating DataSourceConfig model."""
        config = DataSourceConfig(
            id="test-id",
            name="Test Source",
            description="Test description",
            source_type=DataSourceType.REST_API,
            connection_config={"base_url": "https://api.example.com"},
            auth_config={"type": "bearer", "token": "test_token"},
            created_by="test_user"
        )
        
        db_session.add(config)
        db_session.commit()
        
        # Retrieve and verify
        retrieved = db_session.query(DataSourceConfig).filter_by(id="test-id").first()
        assert retrieved.name == "Test Source"
        assert retrieved.source_type == DataSourceType.REST_API
        assert retrieved.connection_config["base_url"] == "https://api.example.com"
    
    def test_connection_test_creation(self, db_session):
        """Test creating ConnectionTest model."""
        test_result = ConnectionTest(
            id="test-result-id",
            data_source_id="test-source-id",
            success=True,
            response_time_ms=150,
            test_details={"version": "1.0"}
        )
        
        db_session.add(test_result)
        db_session.commit()
        
        # Retrieve and verify
        retrieved = db_session.query(ConnectionTest).filter_by(id="test-result-id").first()
        assert retrieved.success is True
        assert retrieved.response_time_ms == 150
        assert retrieved.test_details["version"] == "1.0"
    
    def test_data_schema_creation(self, db_session):
        """Test creating DataSchema model."""
        schema = DataSchema(
            id="schema-id",
            data_source_id="test-source-id",
            schema_name="public",
            table_name="users",
            columns=[
                {"name": "id", "type": "integer", "nullable": False, "primary_key": True},
                {"name": "email", "type": "string", "nullable": False, "primary_key": False}
            ]
        )
        
        db_session.add(schema)
        db_session.commit()
        
        # Retrieve and verify
        retrieved = db_session.query(DataSchema).filter_by(id="schema-id").first()
        assert retrieved.table_name == "users"
        assert len(retrieved.columns) == 2
        assert retrieved.columns[0]["name"] == "id"

class TestConnectorValidation:
    """Test connector configuration validation."""
    
    def test_base_connector_interface(self):
        """Test that base connector defines the required interface."""
        # BaseConnector is abstract, so we can't instantiate it directly
        # Instead, test that it has the required abstract methods
        from abc import ABC
        assert issubclass(BaseConnector, ABC)
        
        # Check that required methods are defined as abstract
        abstract_methods = BaseConnector.__abstractmethods__
        expected_methods = {'test_connection', 'create_connection', 'discover_schema', 'read_data'}
        assert expected_methods.issubset(abstract_methods)
    
    def test_mock_connector_implementation(self):
        """Test that mock connector implements all required methods."""
        connector = MockConnector()
        
        # Test that all methods are implemented and callable
        assert asyncio.run(connector.test_connection({}, {})) == (True, None, {"mock": "test_success"})
        assert asyncio.run(connector.create_connection({}, {}))["mock"] == "connection"
        
        schemas = asyncio.run(connector.discover_schema({}))
        assert len(schemas) == 1
        assert schemas[0]["table_name"] == "mock_table"
        
        data = asyncio.run(connector.read_data({}, {}))
        assert len(data) == 2
        assert data[0]["name"] == "Test 1"

class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_data_source_id(self, connection_manager):
        """Test handling of invalid data source ID."""
        # The connection manager logs the error but returns False instead of raising
        result = await connection_manager.test_connection("invalid-id")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_connection_manager_cleanup(self, connection_manager):
        """Test connection cleanup functionality."""
        # Add a mock connection
        connection_manager.active_connections["test-id"] = Mock()
        connection_manager.active_connections["test-id"].close = Mock()
        
        # Test single connection cleanup
        connection_manager.close_connection("test-id")
        assert "test-id" not in connection_manager.active_connections
        
        # Test cleanup all connections
        connection_manager.active_connections["test-id-1"] = Mock()
        connection_manager.active_connections["test-id-2"] = Mock()
        connection_manager.active_connections["test-id-1"].close = Mock()
        connection_manager.active_connections["test-id-2"].close = Mock()
        
        connection_manager.close_all_connections()
        assert len(connection_manager.active_connections) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])