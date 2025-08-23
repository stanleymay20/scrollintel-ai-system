"""
Database Connectors for Enterprise Integration
Supports SQL Server, Oracle, MySQL, PostgreSQL with connection pooling and failover
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager
import json
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy import text, inspect, MetaData, Table

# Optional database driver imports
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

try:
    import aiomysql
    HAS_AIOMYSQL = True
except ImportError:
    HAS_AIOMYSQL = False

try:
    import cx_Oracle_async
    HAS_ORACLE = True
except ImportError:
    HAS_ORACLE = False

try:
    import pyodbc
    HAS_PYODBC = True
except ImportError:
    HAS_PYODBC = False

from ..models.enterprise_connection_models import (
    ConnectionConfig, ConnectionType, ConnectionStatus, 
    EnterpriseConnection, DataSchema, TableMetadata, SyncResult
)

logger = logging.getLogger(__name__)

class DatabaseConnector(ABC):
    """Abstract base class for database connectors"""
    
    def __init__(self, config: ConnectionConfig, credentials: Dict[str, Any]):
        self.config = config
        self.credentials = credentials
        self.engine = None
        self.session_factory = None
        self._connection_pool = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish database connection"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """Test database connectivity"""
        pass
    
    @abstractmethod
    async def get_schema_info(self) -> DataSchema:
        """Extract database schema information"""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute SQL query"""
        pass

class PostgreSQLConnector(DatabaseConnector):
    """PostgreSQL database connector with async support"""
    
    async def connect(self) -> bool:
        """Establish PostgreSQL connection with pooling"""
        if not HAS_ASYNCPG:
            logger.error("asyncpg driver not available for PostgreSQL connections")
            return False
            
        try:
            connection_string = self._build_connection_string()
            
            self.engine = create_async_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self.config.max_connections,
                max_overflow=self.config.max_connections * 2,
                pool_timeout=self.config.connection_timeout,
                pool_recycle=3600,
                echo=False
            )
            
            self.session_factory = async_sessionmaker(
                self.engine, 
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test the connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            logger.info("PostgreSQL connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Close PostgreSQL connection"""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.session_factory = None
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test PostgreSQL connection"""
        start_time = datetime.utcnow()
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT version(), current_database(), current_user"))
                row = result.fetchone()
                
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "status": ConnectionStatus.ACTIVE.value,
                "response_time_ms": response_time,
                "server_info": {
                    "version": row[0] if row else None,
                    "database": row[1] if row else None,
                    "user": row[2] if row else None
                }
            }
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return {
                "status": ConnectionStatus.ERROR.value,
                "response_time_ms": response_time,
                "error": str(e)
            }
    
    async def get_schema_info(self) -> DataSchema:
        """Extract PostgreSQL schema information"""
        try:
            async with self.engine.begin() as conn:
                # Get tables
                tables_query = text("""
                    SELECT table_name, table_type, table_schema
                    FROM information_schema.tables 
                    WHERE table_schema = :schema_name
                    ORDER BY table_name
                """)
                
                schema_name = self.config.schema or 'public'
                tables_result = await conn.execute(tables_query, {"schema_name": schema_name})
                tables = [dict(row._mapping) for row in tables_result]
                
                # Get views
                views_query = text("""
                    SELECT table_name as view_name, view_definition
                    FROM information_schema.views 
                    WHERE table_schema = :schema_name
                """)
                views_result = await conn.execute(views_query, {"schema_name": schema_name})
                views = [dict(row._mapping) for row in views_result]
                
                return DataSchema(
                    connection_id="",  # Will be set by caller
                    schema_name=schema_name,
                    tables=tables,
                    views=views,
                    procedures=[],  # TODO: Add stored procedures
                    functions=[],   # TODO: Add functions
                    extracted_at=datetime.utcnow()
                )
                
        except Exception as e:
            logger.error(f"Failed to extract PostgreSQL schema: {str(e)}")
            raise
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute PostgreSQL query"""
        try:
            async with self.session_factory() as session:
                result = await session.execute(text(query), params or {})
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string"""
        username = self.credentials.get("username", "")
        password = self.credentials.get("password", "")
        
        return (
            f"postgresql+asyncpg://{username}:{password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )

class MySQLConnector(DatabaseConnector):
    """MySQL database connector with async support"""
    
    async def connect(self) -> bool:
        """Establish MySQL connection"""
        if not HAS_AIOMYSQL:
            logger.error("aiomysql driver not available for MySQL connections")
            return False
            
        try:
            connection_string = self._build_connection_string()
            
            self.engine = create_async_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self.config.max_connections,
                max_overflow=self.config.max_connections * 2,
                pool_timeout=self.config.connection_timeout,
                echo=False
            )
            
            self.session_factory = async_sessionmaker(self.engine, class_=AsyncSession)
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            logger.info("MySQL connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Close MySQL connection"""
        if self.engine:
            await self.engine.dispose()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test MySQL connection"""
        start_time = datetime.utcnow()
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT VERSION(), DATABASE(), USER()"))
                row = result.fetchone()
                
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "status": ConnectionStatus.ACTIVE.value,
                "response_time_ms": response_time,
                "server_info": {
                    "version": row[0] if row else None,
                    "database": row[1] if row else None,
                    "user": row[2] if row else None
                }
            }
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return {
                "status": ConnectionStatus.ERROR.value,
                "response_time_ms": response_time,
                "error": str(e)
            }
    
    async def get_schema_info(self) -> DataSchema:
        """Extract MySQL schema information"""
        try:
            async with self.engine.begin() as conn:
                # Get tables
                tables_query = text("""
                    SELECT TABLE_NAME, TABLE_TYPE, TABLE_SCHEMA
                    FROM information_schema.TABLES 
                    WHERE TABLE_SCHEMA = :schema_name
                    ORDER BY TABLE_NAME
                """)
                
                schema_name = self.config.database
                tables_result = await conn.execute(tables_query, {"schema_name": schema_name})
                tables = [dict(row._mapping) for row in tables_result]
                
                return DataSchema(
                    connection_id="",
                    schema_name=schema_name,
                    tables=tables,
                    views=[],
                    procedures=[],
                    functions=[],
                    extracted_at=datetime.utcnow()
                )
                
        except Exception as e:
            logger.error(f"Failed to extract MySQL schema: {str(e)}")
            raise
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute MySQL query"""
        try:
            async with self.session_factory() as session:
                result = await session.execute(text(query), params or {})
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error(f"MySQL query execution failed: {str(e)}")
            raise
    
    def _build_connection_string(self) -> str:
        """Build MySQL connection string"""
        username = self.credentials.get("username", "")
        password = self.credentials.get("password", "")
        
        return (
            f"mysql+aiomysql://{username}:{password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )

class SQLServerConnector(DatabaseConnector):
    """SQL Server database connector"""
    
    async def connect(self) -> bool:
        """Establish SQL Server connection"""
        if not HAS_PYODBC:
            logger.error("pyodbc driver not available for SQL Server connections")
            return False
            
        try:
            connection_string = self._build_connection_string()
            
            self.engine = create_async_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self.config.max_connections,
                echo=False
            )
            
            self.session_factory = async_sessionmaker(self.engine, class_=AsyncSession)
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            logger.info("SQL Server connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to SQL Server: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Close SQL Server connection"""
        if self.engine:
            await self.engine.dispose()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test SQL Server connection"""
        start_time = datetime.utcnow()
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT @@VERSION, DB_NAME(), SYSTEM_USER"))
                row = result.fetchone()
                
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "status": ConnectionStatus.ACTIVE.value,
                "response_time_ms": response_time,
                "server_info": {
                    "version": row[0] if row else None,
                    "database": row[1] if row else None,
                    "user": row[2] if row else None
                }
            }
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return {
                "status": ConnectionStatus.ERROR.value,
                "response_time_ms": response_time,
                "error": str(e)
            }
    
    async def get_schema_info(self) -> DataSchema:
        """Extract SQL Server schema information"""
        try:
            async with self.engine.begin() as conn:
                # Get tables
                tables_query = text("""
                    SELECT TABLE_NAME, TABLE_TYPE, TABLE_SCHEMA
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_CATALOG = :database_name
                    ORDER BY TABLE_NAME
                """)
                
                tables_result = await conn.execute(tables_query, {"database_name": self.config.database})
                tables = [dict(row._mapping) for row in tables_result]
                
                return DataSchema(
                    connection_id="",
                    schema_name=self.config.schema or "dbo",
                    tables=tables,
                    views=[],
                    procedures=[],
                    functions=[],
                    extracted_at=datetime.utcnow()
                )
                
        except Exception as e:
            logger.error(f"Failed to extract SQL Server schema: {str(e)}")
            raise
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute SQL Server query"""
        try:
            async with self.session_factory() as session:
                result = await session.execute(text(query), params or {})
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error(f"SQL Server query execution failed: {str(e)}")
            raise
    
    def _build_connection_string(self) -> str:
        """Build SQL Server connection string"""
        username = self.credentials.get("username", "")
        password = self.credentials.get("password", "")
        
        return (
            f"mssql+pyodbc://{username}:{password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
            f"?driver=ODBC+Driver+17+for+SQL+Server"
        )

class OracleConnector(DatabaseConnector):
    """Oracle database connector"""
    
    async def connect(self) -> bool:
        """Establish Oracle connection"""
        if not HAS_ORACLE:
            logger.error("cx_Oracle_async driver not available for Oracle connections")
            return False
            
        try:
            connection_string = self._build_connection_string()
            
            self.engine = create_async_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self.config.max_connections,
                echo=False
            )
            
            self.session_factory = async_sessionmaker(self.engine, class_=AsyncSession)
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1 FROM DUAL"))
            
            logger.info("Oracle connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Oracle: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Close Oracle connection"""
        if self.engine:
            await self.engine.dispose()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Oracle connection"""
        start_time = datetime.utcnow()
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT * FROM V$VERSION WHERE ROWNUM = 1"))
                row = result.fetchone()
                
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "status": ConnectionStatus.ACTIVE.value,
                "response_time_ms": response_time,
                "server_info": {
                    "version": row[0] if row else None
                }
            }
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return {
                "status": ConnectionStatus.ERROR.value,
                "response_time_ms": response_time,
                "error": str(e)
            }
    
    async def get_schema_info(self) -> DataSchema:
        """Extract Oracle schema information"""
        try:
            async with self.engine.begin() as conn:
                # Get tables
                tables_query = text("""
                    SELECT TABLE_NAME, 'BASE TABLE' as TABLE_TYPE, OWNER as TABLE_SCHEMA
                    FROM ALL_TABLES 
                    WHERE OWNER = :schema_name
                    ORDER BY TABLE_NAME
                """)
                
                schema_name = self.config.schema or self.credentials.get("username", "").upper()
                tables_result = await conn.execute(tables_query, {"schema_name": schema_name})
                tables = [dict(row._mapping) for row in tables_result]
                
                return DataSchema(
                    connection_id="",
                    schema_name=schema_name,
                    tables=tables,
                    views=[],
                    procedures=[],
                    functions=[],
                    extracted_at=datetime.utcnow()
                )
                
        except Exception as e:
            logger.error(f"Failed to extract Oracle schema: {str(e)}")
            raise
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute Oracle query"""
        try:
            async with self.session_factory() as session:
                result = await session.execute(text(query), params or {})
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error(f"Oracle query execution failed: {str(e)}")
            raise
    
    def _build_connection_string(self) -> str:
        """Build Oracle connection string"""
        username = self.credentials.get("username", "")
        password = self.credentials.get("password", "")
        
        return (
            f"oracle+cx_oracle://{username}:{password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )

class DatabaseConnectorFactory:
    """Factory for creating database connectors"""
    
    _connectors = {
        ConnectionType.POSTGRESQL: PostgreSQLConnector,
        ConnectionType.MYSQL: MySQLConnector,
        ConnectionType.SQL_SERVER: SQLServerConnector,
        ConnectionType.ORACLE: OracleConnector,
    }
    
    @classmethod
    def create_connector(
        self, 
        connection_type: ConnectionType, 
        config: ConnectionConfig, 
        credentials: Dict[str, Any]
    ) -> DatabaseConnector:
        """Create appropriate database connector"""
        
        connector_class = self._connectors.get(connection_type)
        if not connector_class:
            raise ValueError(f"Unsupported connection type: {connection_type}")
        
        return connector_class(config, credentials)