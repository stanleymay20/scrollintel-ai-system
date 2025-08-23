"""
Data Source Manager - Main interface for database connectivity system
Manages connections, schema discovery, and data synchronization
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from ..models.enterprise_connection_models import (
    EnterpriseConnection, ConnectionConfig, ConnectionStatus, 
    DataSchema, SyncConfig, SyncResult, ConnectionTest, EncryptedCredentials
)
from ..connectors.database_connectors import DatabaseConnectorFactory
from ..core.connection_pool_manager import ConnectionPoolManager, ConnectionFailoverManager
from ..core.schema_discovery import SchemaDiscoveryEngine, DataTypeMapper
try:
    from ..core.config import get_database_session
except ImportError:
    # Mock for testing
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def get_database_session():
        yield None

logger = logging.getLogger(__name__)

class DataSourceManager:
    """Main manager for enterprise database connections"""
    
    def __init__(self):
        self.pool_manager = ConnectionPoolManager()
        self.failover_manager = ConnectionFailoverManager(self.pool_manager)
        self._running = False
    
    async def start(self):
        """Start the data source manager"""
        await self.pool_manager.start()
        self._running = True
        logger.info("Data source manager started")
    
    async def stop(self):
        """Stop the data source manager"""
        await self.pool_manager.stop()
        self._running = False
        logger.info("Data source manager stopped")
    
    async def create_connection(self, config: ConnectionConfig, credentials: Dict[str, Any]) -> str:
        """Create a new enterprise database connection"""
        try:
            # Generate connection ID
            connection_id = str(uuid.uuid4())
            
            # Encrypt credentials (simplified - should use proper encryption)
            encrypted_creds = self._encrypt_credentials(credentials)
            
            # Create connection entity
            connection = EnterpriseConnection(
                id=connection_id,
                name=f"{config.host}:{config.port}/{config.database}",
                type=config.connection_params.get('type', 'postgresql'),
                config=config.dict(),
                credentials=encrypted_creds,
                status=ConnectionStatus.TESTING.value,
                created_at=datetime.utcnow()
            )
            
            # Test connection
            success = await self.pool_manager.add_connection(connection, config, credentials)
            if success:
                connection.status = ConnectionStatus.ACTIVE.value
                connection.last_success = datetime.utcnow()
                
                # Save to database
                await self._save_connection(connection)
                
                logger.info(f"Created connection {connection_id}")
                return connection_id
            else:
                connection.status = ConnectionStatus.ERROR.value
                raise Exception("Failed to establish database connection")
                
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            raise
    
    async def test_connection(self, connection_id: str) -> ConnectionTest:
        """Test an existing database connection"""
        try:
            start_time = datetime.utcnow()
            
            connector = await self.pool_manager.get_connection(connection_id)
            if not connector:
                return ConnectionTest(
                    connection_id=connection_id,
                    status=ConnectionStatus.ERROR,
                    response_time_ms=0.0,
                    error_message="Connection not found",
                    test_timestamp=start_time
                )
            
            # Perform connection test
            test_result = await connector.test_connection()
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ConnectionTest(
                connection_id=connection_id,
                status=ConnectionStatus(test_result.get("status", "error")),
                response_time_ms=response_time,
                error_message=test_result.get("error"),
                test_timestamp=start_time,
                metadata=test_result.get("server_info", {})
            )
            
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return ConnectionTest(
                connection_id=connection_id,
                status=ConnectionStatus.ERROR,
                response_time_ms=response_time,
                error_message=str(e),
                test_timestamp=start_time
            )
    
    async def get_schema(self, connection_id: str, schema_name: Optional[str] = None) -> DataSchema:
        """Discover and return database schema information"""
        try:
            connector = await self.pool_manager.get_connection(connection_id)
            if not connector:
                raise Exception(f"Connection {connection_id} not found")
            
            # Get connection info to determine type
            connection = await self._get_connection(connection_id)
            if not connection:
                raise Exception(f"Connection metadata not found for {connection_id}")
            
            # Create schema discovery engine
            discovery_engine = SchemaDiscoveryEngine(connector, connection.type)
            
            # Discover schema
            schema = await discovery_engine.discover_full_schema(schema_name)
            schema.connection_id = connection_id
            
            # Save schema information
            await self._save_schema(schema)
            
            return schema
            
        except Exception as e:
            logger.error(f"Schema discovery failed for {connection_id}: {e}")
            raise
    
    async def sync_data(self, connection_id: str, sync_config: SyncConfig) -> SyncResult:
        """Synchronize data from external database"""
        try:
            sync_id = str(uuid.uuid4())
            start_time = datetime.utcnow()
            
            result = SyncResult(
                sync_id=sync_id,
                connection_id=connection_id,
                status="running",
                records_processed=0,
                records_success=0,
                records_failed=0,
                start_time=start_time
            )
            
            # Execute sync operation with failover
            await self.pool_manager.execute_with_failover(
                connection_id,
                "_perform_data_sync",
                sync_config,
                result
            )
            
            result.end_time = datetime.utcnow()
            result.status = "completed"
            
            # Save sync result
            await self._save_sync_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Data sync failed for {connection_id}: {e}")
            result.status = "failed"
            result.error_details.append(str(e))
            result.end_time = datetime.utcnow()
            raise
    
    async def remove_connection(self, connection_id: str) -> bool:
        """Remove a database connection"""
        try:
            # Remove from pool
            success = await self.pool_manager.remove_connection(connection_id)
            
            if success:
                # Remove from database
                await self._delete_connection(connection_id)
                logger.info(f"Removed connection {connection_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to remove connection {connection_id}: {e}")
            return False
    
    async def list_connections(self) -> List[Dict[str, Any]]:
        """List all database connections with health status"""
        try:
            connections = await self._get_all_connections()
            result = []
            
            for conn in connections:
                health = self.pool_manager.get_health_status(conn.id)
                result.append({
                    "id": conn.id,
                    "name": conn.name,
                    "type": conn.type,
                    "status": conn.status,
                    "created_at": conn.created_at,
                    "last_sync": conn.last_sync,
                    "health": health.__dict__ if health else None
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list connections: {e}")
            return []
    
    async def setup_failover(
        self, 
        primary_id: str, 
        failover_ids: List[str], 
        strategy: str = "priority"
    ):
        """Setup failover configuration for connections"""
        try:
            if strategy == "priority":
                await self.failover_manager.setup_priority_failover(primary_id, failover_ids)
            elif strategy == "round_robin":
                await self.failover_manager.setup_round_robin_failover([primary_id] + failover_ids)
            elif strategy == "master_slave":
                await self.failover_manager.setup_master_slave_failover(primary_id, failover_ids)
            else:
                raise ValueError(f"Unknown failover strategy: {strategy}")
            
            logger.info(f"Setup {strategy} failover for {primary_id}")
            
        except Exception as e:
            logger.error(f"Failed to setup failover: {e}")
            raise
    
    def _encrypt_credentials(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive credential information"""
        # TODO: Implement proper encryption
        # For now, just return as-is (should use encryption in production)
        return credentials
    
    def _decrypt_credentials(self, encrypted_creds: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt credential information"""
        # TODO: Implement proper decryption
        return encrypted_creds
    
    async def _save_connection(self, connection: EnterpriseConnection):
        """Save connection to database"""
        try:
            async with get_database_session() as session:
                session.add(connection)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to save connection: {e}")
            raise
    
    async def _get_connection(self, connection_id: str) -> Optional[EnterpriseConnection]:
        """Get connection from database"""
        try:
            async with get_database_session() as session:
                result = await session.get(EnterpriseConnection, connection_id)
                return result
        except Exception as e:
            logger.error(f"Failed to get connection: {e}")
            return None
    
    async def _get_all_connections(self) -> List[EnterpriseConnection]:
        """Get all connections from database"""
        try:
            async with get_database_session() as session:
                from sqlalchemy import select
                result = await session.execute(select(EnterpriseConnection))
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Failed to get connections: {e}")
            return []
    
    async def _delete_connection(self, connection_id: str):
        """Delete connection from database"""
        try:
            async with get_database_session() as session:
                connection = await session.get(EnterpriseConnection, connection_id)
                if connection:
                    await session.delete(connection)
                    await session.commit()
        except Exception as e:
            logger.error(f"Failed to delete connection: {e}")
            raise
    
    async def _save_schema(self, schema: DataSchema):
        """Save schema information to database"""
        # TODO: Implement schema storage
        pass
    
    async def _save_sync_result(self, result: SyncResult):
        """Save sync result to database"""
        # TODO: Implement sync result storage
        pass
    
    async def _perform_data_sync(self, sync_config: SyncConfig, result: SyncResult):
        """Perform actual data synchronization"""
        # TODO: Implement data synchronization logic
        # This would involve:
        # 1. Reading data from source
        # 2. Applying transformations
        # 3. Loading into target
        # 4. Updating result metrics
        pass

# Global instance
data_source_manager = DataSourceManager()