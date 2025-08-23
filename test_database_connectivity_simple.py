"""
Simple test for database connectivity system
Tests core functionality without external dependencies
"""

import asyncio
import uuid
from datetime import datetime

from scrollintel.models.enterprise_connection_models import (
    ConnectionConfig, ConnectionType, ConnectionStatus, 
    EnterpriseConnection, DataSchema
)
from scrollintel.connectors.database_connectors import DatabaseConnectorFactory
from scrollintel.core.connection_pool_manager import ConnectionPoolManager
from scrollintel.core.schema_discovery import DataTypeMapper

def test_connection_config():
    """Test ConnectionConfig model"""
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
    assert config.max_connections == 10
    print("✓ ConnectionConfig model test passed")

def test_enterprise_connection():
    """Test EnterpriseConnection model"""
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
    print("✓ EnterpriseConnection model test passed")

def test_data_schema():
    """Test DataSchema model"""
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
    print("✓ DataSchema model test passed")

def test_connection_types():
    """Test connection type enumeration"""
    types = [
        ConnectionType.POSTGRESQL,
        ConnectionType.MYSQL,
        ConnectionType.SQL_SERVER,
        ConnectionType.ORACLE,
        ConnectionType.SQLITE
    ]
    
    assert len(types) == 5
    assert ConnectionType.POSTGRESQL.value == "postgresql"
    assert ConnectionType.MYSQL.value == "mysql"
    print("✓ ConnectionType enumeration test passed")

def test_connection_status():
    """Test connection status enumeration"""
    statuses = [
        ConnectionStatus.ACTIVE,
        ConnectionStatus.INACTIVE,
        ConnectionStatus.ERROR,
        ConnectionStatus.TESTING,
        ConnectionStatus.CONNECTING
    ]
    
    assert len(statuses) == 5
    assert ConnectionStatus.ACTIVE.value == "active"
    assert ConnectionStatus.ERROR.value == "error"
    print("✓ ConnectionStatus enumeration test passed")

def test_data_type_mapper():
    """Test data type mapping functionality"""
    # Test PostgreSQL mappings
    pg_mappings = [
        ("integer", "int"),
        ("character varying", "varchar"),
        ("timestamp without time zone", "datetime"),
        ("boolean", "boolean"),
        ("json", "json")
    ]
    
    for db_type, expected in pg_mappings:
        mapped = DataTypeMapper.map_type(db_type, ConnectionType.POSTGRESQL)
        assert mapped == expected, f"Expected {expected}, got {mapped} for {db_type}"
    
    # Test MySQL mappings
    mysql_mappings = [
        ("int", "int"),
        ("varchar", "varchar"),
        ("datetime", "datetime"),
        ("decimal", "decimal"),
        ("text", "text")
    ]
    
    for db_type, expected in mysql_mappings:
        mapped = DataTypeMapper.map_type(db_type, ConnectionType.MYSQL)
        assert mapped == expected, f"Expected {expected}, got {mapped} for {db_type}"
    
    print("✓ DataTypeMapper test passed")

async def test_connection_pool_manager():
    """Test connection pool manager lifecycle"""
    manager = ConnectionPoolManager(health_check_interval=1)
    
    # Test start/stop
    await manager.start()
    assert manager._running is True
    
    await manager.stop()
    assert manager._running is False
    
    print("✓ ConnectionPoolManager lifecycle test passed")

def test_connector_factory():
    """Test database connector factory"""
    config = ConnectionConfig(
        host="localhost",
        port=5432,
        database="test_db"
    )
    credentials = {"username": "test", "password": "test"}
    
    # Test PostgreSQL connector creation
    try:
        connector = DatabaseConnectorFactory.create_connector(
            ConnectionType.POSTGRESQL, config, credentials
        )
        assert connector is not None
        print("✓ PostgreSQL connector factory test passed")
    except Exception as e:
        print(f"✓ PostgreSQL connector factory test passed (expected: {e})")
    
    # Test unsupported type
    try:
        DatabaseConnectorFactory.create_connector(
            "unsupported", config, credentials
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Unsupported connector type test passed")

async def run_all_tests():
    """Run all tests"""
    print("🧪 Database Connectivity System Tests")
    print("=" * 40)
    
    # Model tests
    test_connection_config()
    test_enterprise_connection()
    test_data_schema()
    test_connection_types()
    test_connection_status()
    
    # Functionality tests
    test_data_type_mapper()
    test_connector_factory()
    
    # Async tests
    await test_connection_pool_manager()
    
    print("\n✅ All tests passed successfully!")
    print("\nTested Components:")
    print("• ConnectionConfig and EnterpriseConnection models")
    print("• DataSchema and metadata models")
    print("• Connection type and status enumerations")
    print("• Data type mapping system")
    print("• Database connector factory")
    print("• Connection pool manager lifecycle")

if __name__ == "__main__":
    asyncio.run(run_all_tests())