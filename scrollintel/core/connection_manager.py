"""
Connection Manager for Data Pipeline Automation System
Manages connections to various data sources with connection pooling and health monitoring.
"""
import uuid
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from scrollintel.models.data_source_models import (
    DataSourceConfig, ConnectionTest, DataSchema, 
    DataSourceType, ConnectionStatus
)
from scrollintel.connectors.database_connectors import DatabaseConnector
from scrollintel.connectors.api_connectors import RestApiConnector, GraphQLConnector
from scrollintel.connectors.file_connectors import FileSystemConnector
from scrollintel.connectors.streaming_connectors import StreamingConnector

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages data source connections with pooling and health monitoring."""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.active_connections: Dict[str, Any] = {}
        self.connection_pools: Dict[str, Any] = {}
        self.connectors = {
            DataSourceType.DATABASE: DatabaseConnector(),
            DataSourceType.REST_API: RestApiConnector(),
            DataSourceType.GRAPHQL: GraphQLConnector(),
            DataSourceType.FILE_SYSTEM: FileSystemConnector(),
            DataSourceType.STREAMING: StreamingConnector()
        }
    
    async def create_data_source(self, config_data: Dict[str, Any]) -> DataSourceConfig:
        """Create a new data source configuration."""
        try:
            config = DataSourceConfig(
                id=str(uuid.uuid4()),
                name=config_data["name"],
                description=config_data.get("description", ""),
                source_type=DataSourceType(config_data["source_type"]),
                connection_config=config_data["connection_config"],
                auth_config=config_data.get("auth_config", {}),
                created_by=config_data.get("created_by", "system")
            )
            
            self.db_session.add(config)
            self.db_session.commit()
            
            # Test the connection
            await self.test_connection(config.id)
            
            logger.info(f"Created data source: {config.name} ({config.id})")
            return config
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to create data source: {str(e)}")
            raise
    
    async def test_connection(self, data_source_id: str) -> bool:
        """Test connection to a data source."""
        try:
            config = self.db_session.query(DataSourceConfig).filter_by(id=data_source_id).first()
            if not config:
                raise ValueError(f"Data source {data_source_id} not found")
            
            start_time = datetime.utcnow()
            connector = self.connectors[config.source_type]
            
            # Test the connection
            success, error_message, test_details = await connector.test_connection(
                config.connection_config, 
                config.auth_config
            )
            
            response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Record test result
            test_result = ConnectionTest(
                id=str(uuid.uuid4()),
                data_source_id=data_source_id,
                success=success,
                error_message=error_message,
                response_time_ms=response_time,
                test_details=test_details
            )
            
            self.db_session.add(test_result)
            
            # Update data source status
            config.status = ConnectionStatus.ACTIVE if success else ConnectionStatus.ERROR
            config.last_tested = datetime.utcnow()
            if success:
                config.last_connected = datetime.utcnow()
            
            self.db_session.commit()
            
            logger.info(f"Connection test for {config.name}: {'SUCCESS' if success else 'FAILED'}")
            return success
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    async def get_connection(self, data_source_id: str) -> Any:
        """Get an active connection to a data source."""
        try:
            if data_source_id in self.active_connections:
                return self.active_connections[data_source_id]
            
            config = self.db_session.query(DataSourceConfig).filter_by(id=data_source_id).first()
            if not config:
                raise ValueError(f"Data source {data_source_id} not found")
            
            connector = self.connectors[config.source_type]
            connection = await connector.create_connection(
                config.connection_config,
                config.auth_config
            )
            
            self.active_connections[data_source_id] = connection
            return connection
            
        except Exception as e:
            logger.error(f"Failed to get connection: {str(e)}")
            raise
    
    async def discover_schema(self, data_source_id: str) -> List[DataSchema]:
        """Discover schema information from a data source."""
        try:
            config = self.db_session.query(DataSourceConfig).filter_by(id=data_source_id).first()
            if not config:
                raise ValueError(f"Data source {data_source_id} not found")
            
            connector = self.connectors[config.source_type]
            connection = await self.get_connection(data_source_id)
            
            schemas = await connector.discover_schema(connection)
            
            # Store discovered schemas
            schema_objects = []
            for schema_info in schemas:
                schema_obj = DataSchema(
                    id=str(uuid.uuid4()),
                    data_source_id=data_source_id,
                    schema_name=schema_info.get("schema_name"),
                    table_name=schema_info.get("table_name"),
                    columns=schema_info["columns"]
                )
                schema_objects.append(schema_obj)
                self.db_session.add(schema_obj)
            
            # Update config with schema info
            config.schema_info = {
                "schemas": [s.schema_name for s in schema_objects if s.schema_name],
                "tables": [s.table_name for s in schema_objects if s.table_name],
                "last_discovered": datetime.utcnow().isoformat()
            }
            
            self.db_session.commit()
            
            logger.info(f"Discovered {len(schema_objects)} schemas for {config.name}")
            return schema_objects
            
        except Exception as e:
            logger.error(f"Schema discovery failed: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all active data sources."""
        results = {}
        
        active_configs = self.db_session.query(DataSourceConfig).filter_by(
            status=ConnectionStatus.ACTIVE
        ).all()
        
        for config in active_configs:
            try:
                success = await self.test_connection(config.id)
                results[config.id] = {
                    "name": config.name,
                    "status": "healthy" if success else "unhealthy",
                    "last_tested": datetime.utcnow().isoformat()
                }
            except Exception as e:
                results[config.id] = {
                    "name": config.name,
                    "status": "error",
                    "error": str(e),
                    "last_tested": datetime.utcnow().isoformat()
                }
        
        return results
    
    def close_connection(self, data_source_id: str):
        """Close connection to a data source."""
        if data_source_id in self.active_connections:
            connection = self.active_connections[data_source_id]
            try:
                if hasattr(connection, 'close'):
                    connection.close()
                elif hasattr(connection, 'disconnect'):
                    connection.disconnect()
            except Exception as e:
                logger.warning(f"Error closing connection: {str(e)}")
            finally:
                del self.active_connections[data_source_id]
    
    def close_all_connections(self):
        """Close all active connections."""
        for data_source_id in list(self.active_connections.keys()):
            self.close_connection(data_source_id)