"""
Integration tests for database connectivity system
Tests EnterpriseConnection, ConnectionConfig models and database connectors
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import uuid

from scrollintel.models.enterprise_connection_models import (
    ConnectionConfig, ConnectionType, ConnectionStatus, 
    EnterpriseConnection, DataSchema, SyncConfig, SyncResult
)
from scrollintel.connectors.database_connectors import (
    DatabaseConnectorFactory, PostgreSQLConnector, MySQLConnector
)
from scrollintel.core.connection_pool_manager import ConnectionPoolManager
from scrollintel.core.schema_discovery import SchemaDiscoveryEngine
from scrollintel.core.data_source_manager import DataSourceManager

class TestEnterpriseConnectionModels:
    """Test enterprise connection models"""
    
    def test_connection_config_creation(self):
        """Test ConnectionConfig model creation"""
        config = ConnectionConfig(
            host="localhost",
            port=5432,
            database="test_db",
            schema="public",
            connection_timeout=30,
            max_connections=10
        )
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "test_db"
        assert config.schema == "public"
        assert config.max_connections == 10
    
    def test_enterprise_connection_creation(self):
        """Test EnterpriseConnection model creation"""
        connection = EnterpriseConnection(
            id=str(uuid.uuid4()),
            name="Test Connection",
            type=ConnectionType.POSTGRESQL.value,
            config={"host": "localhost"},
            credentials={"username": "test"},
            status=ConnectionStatus.ACTIVE.value
        )
        
        assert connection.name == "Test Connection"
        assert connection.type == ConnectionType.POSTGRESQL.value
        assert connection.status == ConnectionStatus.ACTIVE.value
    
    def test_data_schema_creation(self):
        """Test DataSchema model creation"""
        schema = DataSchema(
            connection_id="test-conn-1",
            schema_name="public",
            tables=[{"table_name": "users", "table_type": "BASE TABLE"}],
            views=[],
            procedures=[],
            functions=[],
            extracted_at=datetime.utcnow()
        )
        
        assert schema.connection_id == "test-conn-1"
        assert schema.schema_name == "public"
        assert len(schema.tables) == 1
        assert schema.tables[0]["table_name"] == "users"

class TestDatabaseConnectors:
    """Test database connector implementations"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock connection configuration"""
        return ConnectionConfig(
            host="localhost",
            port=5432,
            database="test_db",
            connection_timeout=30,
            max_connections=5
        )
    
    @pytest.fixture
    def mock_credentials(self):
        """Mock credentials"""
        return {
            "username": "test_user",
            "password": "test_pass"
        }
    
    def test_connector_factory_postgresql(self, mock_config, mock_credentials):
        """Test PostgreSQL connector creation"""
        connector = DatabaseConnectorFactory.create_connector(
            ConnectionType.POSTGRESQL, mock_config, mock_credentials
        )
        
        assert isinstance(connector, PostgreSQLConnector)
        assert connector.config == mock_config
        assert connector.credentials == mock_credentials
    
    def test_connector_factory_mysql(self, mock_config, mock_credentials):
        """Test MySQL connector creation"""
        mock_config.port = 3306
        connector = DatabaseConnectorFactory.create_connector(
            ConnectionType.MYSQL, mock_config, mock_credentials
        )
        
        assert isinstance(connector, MySQLConnector)
    
    def test_connector_factory_unsupported_type(self, mock_config, mock_credentials):
        """Test unsupported connection type"""
        with pytest.raises(ValueError, match="Unsupported connection type"):
            DatabaseConnectorFactory.create_connector(
                "unsupported_type", mock_config, mock_credentials
            )
    
    @pytest.mark.asyncio
    async def test_postgresql_connection_string_building(self, mock_config, mock_credentials):
        """Test PostgreSQL connection string building"""
        connector = PostgreSQLConnector(mock_config, mock_credentials)
        connection_string = connector._build_connection_string()
        
        expected = "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db"
        assert connection_string == expected
    
    @pytest.mark.asyncio
    async def test_mysql_connection_string_building(self, mock_config, mock_credentials):
        """Test MySQL connection string building"""
        mock_config.port = 3306
        connector = MySQLConnector(mock_config, mock_credentials)
        connection_string = connector._build_connection_string()
        
        expected = "mysql+aiomysql://test_user:test_pass@localhost:3306/test_db"
        assert connection_string == expected

class TestConnectionPoolManager:
    """Test connection pool manager"""
    
    @pytest.fixture
    def pool_manager(self):
        """Create connection pool manager"""
        return ConnectionPoolManager(health_check_interval=1)
    
    @pytest.fixture
    def mock_connection(self):
        """Mock enterprise connection"""
        return EnterpriseConnection(
            id="test-conn-1",
            name="Test Connection",
            type=ConnectionType.POSTGRESQL.value,
            config={},
            credentials={},
            status=ConnectionStatus.ACTIVE.value
        )
    
    @pytest.mark.asyncio
    async def test_pool_manager_start_stop(self, pool_manager):
        """Test pool manager lifecycle"""
        await pool_manager.start()
        assert pool_manager._running is True
        
        await pool_manager.stop()
        assert pool_manager._running is False
    
    @pytest.mark.asyncio
    async def test_add_connection_success(self, pool_manager, mock_connection):
        """Test successful connection addition"""
        config = ConnectionConfig(
            host="localhost",
            port=5432,
            database="test_db"
        )
        credentials = {"username": "test", "password": "test"}
        
        with patch.object(DatabaseConnectorFactory, 'create_connector') as mock_factory:
            mock_connector = AsyncMock()
            mock_connector.connect.return_value = True
            mock_factory.return_value = mock_connector
            
            success = await pool_manager.add_connection(mock_connection, config, credentials)
            
            assert success is True
            assert mock_connection.id in pool_manager._connections
            assert mock_connection.id in pool_manager._health_metrics
    
    @pytest.mark.asyncio
    async def test_add_connection_failure(self, pool_manager, mock_connection):
        """Test failed connection addition"""
        config = ConnectionConfig(
            host="localhost",
            port=5432,
            database="test_db"
        )
        credentials = {"username": "test", "password": "test"}
        
        with patch.object(DatabaseConnectorFactory, 'create_connector') as mock_factory:
            mock_connector = AsyncMock()
            mock_connector.connect.return_value = False
            mock_factory.return_value = mock_connector
            
            success = await pool_manager.add_connection(mock_connection, config, credentials)
            
            assert success is False
            assert mock_connection.id not in pool_manager._connections
    
    @pytest.mark.asyncio
    async def test_remove_connection(self, pool_manager, mock_connection):
        """Test connection removal"""
        # First add a connection
        config = ConnectionConfig(host="localhost", port=5432, database="test_db")
        credentials = {"username": "test", "password": "test"}
        
        with patch.object(DatabaseConnectorFactory, 'create_connector') as mock_factory:
            mock_connector = AsyncMock()
            mock_connector.connect.return_value = True
            mock_connector.disconnect = AsyncMock()
            mock_factory.return_value = mock_connector
            
            await pool_manager.add_connection(mock_connection, config, credentials)
            
            # Now remove it
            success = await pool_manager.remove_connection(mock_connection.id)
            
            assert success is True
            assert mock_connection.id not in pool_manager._connections
            assert mock_connection.id not in pool_manager._health_metrics
    
    @pytest.mark.asyncio
    async def test_failover_group_setup(self, pool_manager):
        """Test failover group configuration"""
        primary_id = "primary-conn"
        failover_ids = ["failover-1", "failover-2"]
        
        pool_manager.add_failover_group(primary_id, failover_ids)
        
        assert primary_id in pool_manager._failover_groups
        assert pool_manager._failover_groups[primary_id] == failover_ids

class TestSchemaDiscovery:
    """Test schema discovery functionality"""
    
    @pytest.fixture
    def mock_connector(self):
        """Mock database connector"""
        connector = AsyncMock()
        connector.execute_query = AsyncMock()
        return connector
    
    @pytest.mark.asyncio
    async def test_schema_discovery_engine_creation(self, mock_connector):
        """Test schema discovery engine creation"""
        engine = SchemaDiscoveryEngine(mock_connector, ConnectionType.POSTGRESQL)
        
        assert engine.connector == mock_connector
        assert engine.connection_type == ConnectionType.POSTGRESQL
    
    @pytest.mark.asyncio
    async def test_postgresql_column_discovery(self, mock_connector):
        """Test PostgreSQL column information discovery"""
        # Mock query result
        mock_connector.execute_query.return_value = [
            {
                'column_name': 'id',
                'data_type': 'integer',
                'is_nullable': 'NO',
                'column_default': 'nextval(\'users_id_seq\'::regclass)',
                'character_maximum_length': None,
                'numeric_precision': 32,
                'numeric_scale': 0,
                'is_primary_key': True,
                'is_foreign_key': False,
                'foreign_table_name': None,
                'foreign_column_name': None
            },
            {
                'column_name': 'username',
                'data_type': 'character varying',
                'is_nullable': 'NO',
                'column_default': None,
                'character_maximum_length': 255,
                'numeric_precision': None,
                'numeric_scale': None,
                'is_primary_key': False,
                'is_foreign_key': False,
                'foreign_table_name': None,
                'foreign_column_name': None
            }
        ]
        
        engine = SchemaDiscoveryEngine(mock_connector, ConnectionType.POSTGRESQL)
        columns = await engine._get_postgresql_columns("users", "public")
        
        assert len(columns) == 2
        assert columns[0].name == 'id'
        assert columns[0].data_type == 'integer'
        assert columns[0].is_primary_key is True
        assert columns[1].name == 'username'
        assert columns[1].data_type == 'character varying'
        assert columns[1].max_length == 255

class TestDataSourceManager:
    """Test data source manager functionality"""
    
    @pytest.fixture
    def data_source_manager(self):
        """Create data source manager"""
        return DataSourceManager()
    
    @pytest.mark.asyncio
    async def test_manager_lifecycle(self, data_source_manager):
        """Test manager start/stop lifecycle"""
        await data_source_manager.start()
        assert data_source_manager._running is True
        
        await data_source_manager.stop()
        assert data_source_manager._running is False
    
    @pytest.mark.asyncio
    async def test_create_connection_success(self, data_source_manager):
        """Test successful connection creation"""
        config = ConnectionConfig(
            host="localhost",
            port=5432,
            database="test_db",
            connection_params={"type": "postgresql"}
        )
        credentials = {"username": "test", "password": "test"}
        
        with patch.object(data_source_manager.pool_manager, 'add_connection') as mock_add:
            with patch.object(data_source_manager, '_save_connection') as mock_save:
                mock_add.return_value = True
                mock_save.return_value = None
                
                connection_id = await data_source_manager.create_connection(config, credentials)
                
                assert connection_id is not None
                assert len(connection_id) > 0
                mock_add.assert_called_once()
                mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self, data_source_manager):
        """Test connection testing"""
        connection_id = "test-conn-1"
        
        mock_connector = AsyncMock()
        mock_connector.test_connection.return_value = {
            "status": ConnectionStatus.ACTIVE.value,
            "server_info": {"version": "PostgreSQL 13.0"}
        }
        
        with patch.object(data_source_manager.pool_manager, 'get_connection') as mock_get:
            mock_get.return_value = mock_connector
            
            result = await data_source_manager.test_connection(connection_id)
            
            assert result.connection_id == connection_id
            assert result.status == ConnectionStatus.ACTIVE
            assert result.response_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_get_schema_success(self, data_source_manager):
        """Test schema discovery"""
        connection_id = "test-conn-1"
        
        mock_connector = AsyncMock()
        mock_connection = Mock()
        mock_connection.type = ConnectionType.POSTGRESQL
        
        mock_schema = DataSchema(
            connection_id="",
            schema_name="public",
            tables=[{"table_name": "users"}],
            views=[],
            procedures=[],
            functions=[],
            extracted_at=datetime.utcnow()
        )
        
        with patch.object(data_source_manager.pool_manager, 'get_connection') as mock_get_conn:
            with patch.object(data_source_manager, '_get_connection') as mock_get_meta:
                with patch.object(SchemaDiscoveryEngine, 'discover_full_schema') as mock_discover:
                    with patch.object(data_source_manager, '_save_schema') as mock_save:
                        mock_get_conn.return_value = mock_connector
                        mock_get_meta.return_value = mock_connection
                        mock_discover.return_value = mock_schema
                        mock_save.return_value = None
                        
                        result = await data_source_manager.get_schema(connection_id)
                        
                        assert result.connection_id == connection_id
                        assert result.schema_name == "public"
                        assert len(result.tables) == 1

@pytest.mark.asyncio
async def test_end_to_end_database_connectivity():
    """End-to-end test of database connectivity system"""
    # This test would require actual database connections
    # For now, we'll test the integration with mocks
    
    manager = DataSourceManager()
    await manager.start()
    
    try:
        # Test configuration
        config = ConnectionConfig(
            host="localhost",
            port=5432,
            database="test_db",
            connection_params={"type": "postgresql"}
        )
        credentials = {"username": "test", "password": "test"}
        
        with patch.object(manager.pool_manager, 'add_connection') as mock_add:
            with patch.object(manager, '_save_connection') as mock_save:
                mock_add.return_value = True
                mock_save.return_value = None
                
                # Create connection
                connection_id = await manager.create_connection(config, credentials)
                assert connection_id is not None
                
                # Test connection
                mock_connector = AsyncMock()
                mock_connector.test_connection.return_value = {
                    "status": ConnectionStatus.ACTIVE.value
                }
                
                with patch.object(manager.pool_manager, 'get_connection') as mock_get:
                    mock_get.return_value = mock_connector
                    
                    test_result = await manager.test_connection(connection_id)
                    assert test_result.status == ConnectionStatus.ACTIVE
                
                # Remove connection
                with patch.object(manager.pool_manager, 'remove_connection') as mock_remove:
                    with patch.object(manager, '_delete_connection') as mock_delete:
                        mock_remove.return_value = True
                        mock_delete.return_value = None
                        
                        success = await manager.remove_connection(connection_id)
                        assert success is True
    
    finally:
        await manager.stop()

if __name__ == "__main__":
    pytest.main([__file__])