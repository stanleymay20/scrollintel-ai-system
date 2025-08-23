"""
Connection Pool Manager with Failover Mechanisms
Handles connection pooling, health monitoring, and automatic failover
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

from ..models.enterprise_connection_models import (
    ConnectionConfig, ConnectionStatus, EnterpriseConnection
)
from ..connectors.database_connectors import DatabaseConnector, DatabaseConnectorFactory

logger = logging.getLogger(__name__)

@dataclass
class ConnectionHealth:
    """Connection health metrics"""
    connection_id: str
    status: ConnectionStatus
    last_check: datetime
    response_time_ms: float
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_details: List[str] = field(default_factory=list)

class ConnectionPoolManager:
    """Manages database connection pools with failover capabilities"""
    
    def __init__(self, health_check_interval: int = 30):
        self._connections: Dict[str, DatabaseConnector] = {}
        self._health_metrics: Dict[str, ConnectionHealth] = {}
        self._failover_groups: Dict[str, List[str]] = {}
        self._health_check_interval = health_check_interval
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the connection pool manager"""
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Connection pool manager started")
    
    async def stop(self):
        """Stop the connection pool manager"""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for connector in self._connections.values():
            try:
                await connector.disconnect()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        
        self._connections.clear()
        self._health_metrics.clear()
        logger.info("Connection pool manager stopped")
    
    async def add_connection(
        self, 
        connection: EnterpriseConnection,
        config: ConnectionConfig,
        credentials: Dict[str, Any]
    ) -> bool:
        """Add a new database connection to the pool"""
        try:
            connector = DatabaseConnectorFactory.create_connector(
                connection.type, config, credentials
            )
            
            # Test connection before adding
            success = await connector.connect()
            if not success:
                logger.error(f"Failed to establish connection {connection.id}")
                return False
            
            self._connections[connection.id] = connector
            self._health_metrics[connection.id] = ConnectionHealth(
                connection_id=connection.id,
                status=ConnectionStatus.ACTIVE,
                last_check=datetime.utcnow(),
                response_time_ms=0.0
            )
            
            logger.info(f"Added connection {connection.id} to pool")
            return True
            
        except Exception as e:
            logger.error(f"Error adding connection {connection.id}: {e}")
            return False
    
    async def remove_connection(self, connection_id: str) -> bool:
        """Remove a connection from the pool"""
        try:
            if connection_id in self._connections:
                connector = self._connections[connection_id]
                await connector.disconnect()
                del self._connections[connection_id]
                
            if connection_id in self._health_metrics:
                del self._health_metrics[connection_id]
            
            # Remove from failover groups
            for group_name, connections in self._failover_groups.items():
                if connection_id in connections:
                    connections.remove(connection_id)
            
            logger.info(f"Removed connection {connection_id} from pool")
            return True
            
        except Exception as e:
            logger.error(f"Error removing connection {connection_id}: {e}")
            return False
    
    async def get_connection(self, connection_id: str) -> Optional[DatabaseConnector]:
        """Get a healthy connection, with failover if needed"""
        # Try primary connection first
        if connection_id in self._connections:
            health = self._health_metrics.get(connection_id)
            if health and health.status == ConnectionStatus.ACTIVE:
                return self._connections[connection_id]
        
        # Try failover connections
        failover_connections = self._get_failover_connections(connection_id)
        for failover_id in failover_connections:
            if failover_id in self._connections:
                health = self._health_metrics.get(failover_id)
                if health and health.status == ConnectionStatus.ACTIVE:
                    logger.info(f"Using failover connection {failover_id} for {connection_id}")
                    return self._connections[failover_id]
        
        logger.warning(f"No healthy connections available for {connection_id}")
        return None
    
    async def execute_with_failover(
        self, 
        connection_id: str, 
        operation: str,
        *args, 
        **kwargs
    ) -> Any:
        """Execute operation with automatic failover"""
        connection_ids = [connection_id] + self._get_failover_connections(connection_id)
        
        last_error = None
        for conn_id in connection_ids:
            try:
                connector = await self.get_connection(conn_id)
                if not connector:
                    continue
                
                # Execute the operation
                method = getattr(connector, operation)
                result = await method(*args, **kwargs)
                
                # Update success metrics
                await self._update_success_metrics(conn_id)
                return result
                
            except Exception as e:
                last_error = e
                await self._update_failure_metrics(conn_id, str(e))
                logger.warning(f"Operation failed on {conn_id}: {e}")
                continue
        
        # All connections failed
        raise Exception(f"All connections failed. Last error: {last_error}")
    
    def add_failover_group(self, primary_id: str, failover_ids: List[str]):
        """Add failover group configuration"""
        self._failover_groups[primary_id] = failover_ids
        logger.info(f"Added failover group for {primary_id}: {failover_ids}")
    
    def get_health_status(self, connection_id: str) -> Optional[ConnectionHealth]:
        """Get connection health status"""
        return self._health_metrics.get(connection_id)
    
    def get_all_health_status(self) -> Dict[str, ConnectionHealth]:
        """Get health status for all connections"""
        return self._health_metrics.copy()
    
    async def _health_check_loop(self):
        """Continuous health checking loop"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self._health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)  # Short delay on error
    
    async def _perform_health_checks(self):
        """Perform health checks on all connections"""
        tasks = []
        for connection_id in list(self._connections.keys()):
            task = asyncio.create_task(self._check_connection_health(connection_id))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_connection_health(self, connection_id: str):
        """Check health of a specific connection"""
        try:
            connector = self._connections.get(connection_id)
            if not connector:
                return
            
            health = self._health_metrics.get(connection_id)
            if not health:
                return
            
            # Perform health check
            start_time = datetime.utcnow()
            test_result = await connector.test_connection()
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update health metrics
            health.last_check = datetime.utcnow()
            health.response_time_ms = response_time
            health.total_requests += 1
            
            if test_result.get("status") == ConnectionStatus.ACTIVE.value:
                health.status = ConnectionStatus.ACTIVE
                health.consecutive_failures = 0
                health.successful_requests += 1
            else:
                health.status = ConnectionStatus.ERROR
                health.consecutive_failures += 1
                health.failed_requests += 1
                error_msg = test_result.get("error", "Unknown error")
                health.error_details.append(f"{datetime.utcnow()}: {error_msg}")
                
                # Keep only last 10 errors
                if len(health.error_details) > 10:
                    health.error_details = health.error_details[-10:]
            
            # Mark as inactive if too many consecutive failures
            if health.consecutive_failures >= 3:
                health.status = ConnectionStatus.INACTIVE
                logger.warning(f"Connection {connection_id} marked as inactive due to failures")
            
        except Exception as e:
            logger.error(f"Health check failed for {connection_id}: {e}")
            health = self._health_metrics.get(connection_id)
            if health:
                health.status = ConnectionStatus.ERROR
                health.consecutive_failures += 1
    
    def _get_failover_connections(self, connection_id: str) -> List[str]:
        """Get failover connections for a primary connection"""
        return self._failover_groups.get(connection_id, [])
    
    async def _update_success_metrics(self, connection_id: str):
        """Update success metrics for a connection"""
        health = self._health_metrics.get(connection_id)
        if health:
            health.successful_requests += 1
            health.total_requests += 1
            health.consecutive_failures = 0
    
    async def _update_failure_metrics(self, connection_id: str, error: str):
        """Update failure metrics for a connection"""
        health = self._health_metrics.get(connection_id)
        if health:
            health.failed_requests += 1
            health.total_requests += 1
            health.consecutive_failures += 1
            health.error_details.append(f"{datetime.utcnow()}: {error}")
            
            # Keep only last 10 errors
            if len(health.error_details) > 10:
                health.error_details = health.error_details[-10:]

class ConnectionFailoverManager:
    """Manages connection failover strategies"""
    
    def __init__(self, pool_manager: ConnectionPoolManager):
        self.pool_manager = pool_manager
    
    async def setup_master_slave_failover(
        self, 
        master_id: str, 
        slave_ids: List[str]
    ):
        """Setup master-slave failover configuration"""
        self.pool_manager.add_failover_group(master_id, slave_ids)
        logger.info(f"Configured master-slave failover: {master_id} -> {slave_ids}")
    
    async def setup_round_robin_failover(
        self, 
        connection_ids: List[str]
    ):
        """Setup round-robin failover configuration"""
        for i, conn_id in enumerate(connection_ids):
            failover_list = connection_ids[:i] + connection_ids[i+1:]
            self.pool_manager.add_failover_group(conn_id, failover_list)
        
        logger.info(f"Configured round-robin failover for: {connection_ids}")
    
    async def setup_priority_failover(
        self, 
        primary_id: str, 
        priority_list: List[str]
    ):
        """Setup priority-based failover configuration"""
        self.pool_manager.add_failover_group(primary_id, priority_list)
        logger.info(f"Configured priority failover: {primary_id} -> {priority_list}")