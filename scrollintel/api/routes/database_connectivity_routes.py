"""
API routes for database connectivity system
Provides REST endpoints for managing enterprise database connections
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ...models.enterprise_connection_models import (
    ConnectionConfig, ConnectionTest, DataSchema, SyncConfig, SyncResult
)
from ...core.data_source_manager import data_source_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/database", tags=["Database Connectivity"])

@router.post("/connections", response_model=Dict[str, str])
async def create_connection(
    config: ConnectionConfig,
    credentials: Dict[str, Any]
):
    """Create a new database connection"""
    try:
        connection_id = await data_source_manager.create_connection(config, credentials)
        return {
            "connection_id": connection_id,
            "status": "created",
            "message": "Database connection created successfully"
        }
    except Exception as e:
        logger.error(f"Failed to create connection: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/connections", response_model=List[Dict[str, Any]])
async def list_connections():
    """List all database connections"""
    try:
        connections = await data_source_manager.list_connections()
        return connections
    except Exception as e:
        logger.error(f"Failed to list connections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connections/{connection_id}/test", response_model=ConnectionTest)
async def test_connection(connection_id: str):
    """Test a database connection"""
    try:
        result = await data_source_manager.test_connection(connection_id)
        return result
    except Exception as e:
        logger.error(f"Failed to test connection {connection_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connections/{connection_id}/schema", response_model=DataSchema)
async def get_schema(
    connection_id: str,
    schema_name: Optional[str] = None
):
    """Get database schema information"""
    try:
        schema = await data_source_manager.get_schema(connection_id, schema_name)
        return schema
    except Exception as e:
        logger.error(f"Failed to get schema for {connection_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connections/{connection_id}/sync", response_model=SyncResult)
async def sync_data(
    connection_id: str,
    sync_config: SyncConfig,
    background_tasks: BackgroundTasks
):
    """Synchronize data from database connection"""
    try:
        # Start sync in background
        result = await data_source_manager.sync_data(connection_id, sync_config)
        return result
    except Exception as e:
        logger.error(f"Failed to sync data for {connection_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/connections/{connection_id}")
async def remove_connection(connection_id: str):
    """Remove a database connection"""
    try:
        success = await data_source_manager.remove_connection(connection_id)
        if success:
            return {"status": "removed", "message": "Connection removed successfully"}
        else:
            raise HTTPException(status_code=404, detail="Connection not found")
    except Exception as e:
        logger.error(f"Failed to remove connection {connection_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connections/{connection_id}/failover")
async def setup_failover(
    connection_id: str,
    failover_config: Dict[str, Any]
):
    """Setup failover configuration for a connection"""
    try:
        failover_ids = failover_config.get("failover_connections", [])
        strategy = failover_config.get("strategy", "priority")
        
        await data_source_manager.setup_failover(connection_id, failover_ids, strategy)
        
        return {
            "status": "configured",
            "message": f"Failover configured with {strategy} strategy"
        }
    except Exception as e:
        logger.error(f"Failed to setup failover for {connection_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connections/{connection_id}/health")
async def get_connection_health(connection_id: str):
    """Get connection health metrics"""
    try:
        health = data_source_manager.pool_manager.get_health_status(connection_id)
        if health:
            return {
                "connection_id": health.connection_id,
                "status": health.status.value,
                "last_check": health.last_check,
                "response_time_ms": health.response_time_ms,
                "consecutive_failures": health.consecutive_failures,
                "total_requests": health.total_requests,
                "successful_requests": health.successful_requests,
                "failed_requests": health.failed_requests,
                "success_rate": (
                    health.successful_requests / health.total_requests * 100
                    if health.total_requests > 0 else 0
                ),
                "error_details": health.error_details[-5:]  # Last 5 errors
            }
        else:
            raise HTTPException(status_code=404, detail="Connection health data not found")
    except Exception as e:
        logger.error(f"Failed to get health for {connection_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_system_health():
    """Get overall system health"""
    try:
        all_health = data_source_manager.pool_manager.get_all_health_status()
        
        total_connections = len(all_health)
        active_connections = sum(1 for h in all_health.values() if h.status.value == "active")
        error_connections = sum(1 for h in all_health.values() if h.status.value == "error")
        
        return {
            "system_status": "healthy" if error_connections == 0 else "degraded",
            "total_connections": total_connections,
            "active_connections": active_connections,
            "error_connections": error_connections,
            "uptime_percentage": (
                active_connections / total_connections * 100
                if total_connections > 0 else 100
            ),
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types")
async def get_supported_types():
    """Get supported database types and their capabilities"""
    return {
        "supported_types": [
            {
                "type": "postgresql",
                "name": "PostgreSQL",
                "default_port": 5432,
                "features": ["async", "pooling", "ssl", "schema_discovery", "failover"],
                "driver_required": "asyncpg"
            },
            {
                "type": "mysql",
                "name": "MySQL",
                "default_port": 3306,
                "features": ["async", "pooling", "ssl", "schema_discovery", "failover"],
                "driver_required": "aiomysql"
            },
            {
                "type": "sql_server",
                "name": "Microsoft SQL Server",
                "default_port": 1433,
                "features": ["async", "pooling", "ssl", "schema_discovery", "failover"],
                "driver_required": "pyodbc"
            },
            {
                "type": "oracle",
                "name": "Oracle Database",
                "default_port": 1521,
                "features": ["async", "pooling", "ssl", "schema_discovery", "failover"],
                "driver_required": "cx_Oracle_async"
            },
            {
                "type": "sqlite",
                "name": "SQLite",
                "default_port": None,
                "features": ["local", "file_based"],
                "driver_required": "aiosqlite"
            }
        ]
    }

# Startup and shutdown events
@router.on_event("startup")
async def startup_database_manager():
    """Start the database manager on application startup"""
    await data_source_manager.start()
    logger.info("Database connectivity system started")

@router.on_event("shutdown")
async def shutdown_database_manager():
    """Stop the database manager on application shutdown"""
    await data_source_manager.stop()
    logger.info("Database connectivity system stopped")