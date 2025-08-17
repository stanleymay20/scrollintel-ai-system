"""
Database Connection Manager for ScrollIntel
Handles PostgreSQL primary connection with SQLite fallback
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, AsyncGenerator, Union
from sqlalchemy import create_engine, text, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DatabaseError
import sqlite3
import asyncpg
from pathlib import Path

from .configuration_manager import get_config, DatabaseType, ConfigurationError
from ..models.database import Base

logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class DatabaseConnectionManager:
    """Manages database connections with PostgreSQL primary and SQLite fallback."""
    
    def __init__(self):
        """Initialize database connection manager."""
        self.config = get_config()
        self.current_database_type: Optional[DatabaseType] = None
        self.current_database_url: Optional[str] = None
        
        # Connection objects
        self.async_engine = None
        self.sync_engine = None
        self.async_session_factory = None
        self.sync_session_factory = None
        
        # Connection status
        self.is_connected = False
        self.connection_attempts = 0
        self.last_connection_attempt = 0
        self.fallback_active = False
        
        # Health check
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
    
    async def initialize(self) -> bool:
        """Initialize database connection with fallback logic."""
        try:
            logger.info("Initializing database connection...")
            
            # Try primary database first
            if await self._connect_to_primary():
                logger.info("Connected to primary database (PostgreSQL)")
                return True
            
            # Fallback to SQLite
            logger.warning("Primary database unavailable, falling back to SQLite")
            if await self._connect_to_fallback():
                logger.info("Connected to fallback database (SQLite)")
                self.fallback_active = True
                return True
            
            # Both connections failed
            raise DatabaseConnectionError("Failed to connect to both primary and fallback databases")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise DatabaseConnectionError(f"Database initialization failed: {e}")
    
    async def _connect_to_primary(self) -> bool:
        """Attempt to connect to primary PostgreSQL database."""
        try:
            primary_url = self.config.database.primary_url
            logger.info(f"Attempting to connect to primary database...")
            
            # Test connection first (use sync version for initialization)
            if not self._test_postgresql_connection_sync(primary_url):
                return False
            
            # Create engines
            await self._create_postgresql_engines(primary_url)
            
            # Test the engines
            if await self._test_engines():
                self.current_database_type = DatabaseType.POSTGRESQL
                self.current_database_url = primary_url
                self.is_connected = True
                self.fallback_active = False
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to connect to primary database: {e}")
            return False
    
    async def _connect_to_fallback(self) -> bool:
        """Attempt to connect to fallback SQLite database."""
        try:
            fallback_url = self.config.database.fallback_url
            logger.info(f"Attempting to connect to fallback database...")
            
            # Ensure SQLite directory exists
            self._ensure_sqlite_directory(fallback_url)
            
            # Create engines
            await self._create_sqlite_engines(fallback_url)
            
            # Test the engines
            if await self._test_engines():
                self.current_database_type = DatabaseType.SQLITE
                self.current_database_url = fallback_url
                self.is_connected = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to fallback database: {e}")
            return False
    
    def _test_postgresql_connection_sync(self, database_url: str) -> bool:
        """Test PostgreSQL connection availability synchronously."""
        try:
            # Parse connection URL to get connection parameters
            import urllib.parse
            parsed = urllib.parse.urlparse(database_url)
            
            # Extract connection parameters
            host = parsed.hostname or 'localhost'
            port = parsed.port or 5432
            database = parsed.path.lstrip('/') or 'postgres'
            username = parsed.username or 'postgres'
            password = parsed.password or ''
            
            # Test connection with psycopg2 (sync)
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=username,
                    password=password,
                    connect_timeout=self.config.database.timeout
                )
                
                # Test basic query
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                
                return result and result[0] == 1
                
            except ImportError:
                # psycopg2 not available, skip test
                logger.debug("psycopg2 not available, skipping PostgreSQL connection test")
                return True  # Assume it will work
            
        except Exception as e:
            logger.debug(f"PostgreSQL connection test failed: {e}")
            return False
    
    async def _test_postgresql_connection(self, database_url: str) -> bool:
        """Test PostgreSQL connection availability."""
        try:
            # Parse connection URL to get connection parameters
            import urllib.parse
            parsed = urllib.parse.urlparse(database_url)
            
            # Extract connection parameters
            host = parsed.hostname or 'localhost'
            port = parsed.port or 5432
            database = parsed.path.lstrip('/') or 'postgres'
            username = parsed.username or 'postgres'
            password = parsed.password or ''
            
            # Test connection with asyncpg
            conn = await asyncpg.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password,
                timeout=self.config.database.timeout
            )
            
            # Test basic query
            result = await conn.fetchval('SELECT 1')
            await conn.close()
            
            return result == 1
            
        except Exception as e:
            logger.debug(f"PostgreSQL connection test failed: {e}")
            return False
    
    def _ensure_sqlite_directory(self, database_url: str) -> None:
        """Ensure SQLite database directory exists."""
        try:
            if database_url.startswith('sqlite:///'):
                # Extract file path
                file_path = database_url.replace('sqlite:///', '')
                if file_path != ':memory:':
                    db_path = Path(file_path)
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Ensured SQLite directory exists: {db_path.parent}")
        except Exception as e:
            logger.warning(f"Failed to create SQLite directory: {e}")
    
    async def _create_postgresql_engines(self, database_url: str) -> None:
        """Create PostgreSQL engines."""
        try:
            # Convert URL to use appropriate drivers
            sync_url = database_url.replace('postgresql://', 'postgresql+psycopg2://')
            async_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')
            
            # Create async engine
            self.async_engine = create_async_engine(
                async_url,
                echo=self.config.debug,
                pool_size=self.config.database.pool_size,
                max_overflow=self.config.database.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={
                    "server_settings": {
                        "application_name": "ScrollIntel",
                    }
                }
            )
            
            # Create sync engine for migrations
            self.sync_engine = create_engine(
                sync_url,
                echo=self.config.debug,
                pool_size=self.config.database.pool_size,
                max_overflow=self.config.database.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
                poolclass=QueuePool
            )
            
            # Create session factories
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            self.sync_session_factory = sessionmaker(
                bind=self.sync_engine,
                autocommit=False,
                autoflush=False
            )
            
            logger.debug("PostgreSQL engines created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL engines: {e}")
            raise
    
    async def _create_sqlite_engines(self, database_url: str) -> None:
        """Create SQLite engines."""
        try:
            # Convert to async SQLite URL if needed
            if not database_url.startswith('sqlite+aiosqlite://'):
                if database_url.startswith('sqlite:///'):
                    async_url = database_url.replace('sqlite:///', 'sqlite+aiosqlite:///')
                else:
                    async_url = database_url.replace('sqlite://', 'sqlite+aiosqlite://')
            else:
                async_url = database_url
            
            # Create async engine
            self.async_engine = create_async_engine(
                async_url,
                echo=self.config.debug,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": self.config.database.timeout
                }
            )
            
            # Create sync engine
            sync_url = database_url.replace('sqlite+aiosqlite://', 'sqlite://')
            self.sync_engine = create_engine(
                sync_url,
                echo=self.config.debug,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": self.config.database.timeout
                }
            )
            
            # Enable foreign key constraints for SQLite
            @event.listens_for(self.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
            
            # Create session factories
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            self.sync_session_factory = sessionmaker(
                bind=self.sync_engine,
                autocommit=False,
                autoflush=False
            )
            
            logger.debug("SQLite engines created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create SQLite engines: {e}")
            raise
    
    async def _test_engines(self) -> bool:
        """Test database engines."""
        try:
            # Test async engine
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                if result.scalar() != 1:
                    return False
            
            # Test sync engine
            with self.sync_engine.begin() as conn:
                result = conn.execute(text("SELECT 1"))
                if result.scalar() != 1:
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Engine test failed: {e}")
            return False
    
    async def create_tables(self) -> None:
        """Create database tables."""
        try:
            if not self.is_connected:
                raise DatabaseConnectionError("Not connected to database")
            
            logger.info("Creating database tables...")
            
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise DatabaseConnectionError(f"Table creation failed: {e}")
    
    async def drop_tables(self) -> None:
        """Drop database tables."""
        try:
            if not self.is_connected:
                raise DatabaseConnectionError("Not connected to database")
            
            logger.warning("Dropping database tables...")
            
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            
            logger.info("Database tables dropped successfully")
            
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise DatabaseConnectionError(f"Table drop failed: {e}")
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session."""
        if not self.is_connected or not self.async_session_factory:
            raise DatabaseConnectionError("Database not connected")
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    def get_sync_session(self):
        """Get a sync database session context manager."""
        if not self.is_connected or not self.sync_session_factory:
            raise DatabaseConnectionError("Database not connected")
        
        from contextlib import contextmanager
        
        @contextmanager
        def session_context():
            session = self.sync_session_factory()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
        
        return session_context()
    
    async def check_health(self) -> Dict[str, Any]:
        """Check database connection health."""
        current_time = time.time()
        
        # Skip if recently checked
        if current_time - self.last_health_check < self.health_check_interval:
            return {
                "healthy": self.is_connected,
                "database_type": self.current_database_type.value if self.current_database_type else None,
                "fallback_active": self.fallback_active,
                "last_check": self.last_health_check
            }
        
        health_status = {
            "healthy": False,
            "database_type": None,
            "fallback_active": self.fallback_active,
            "last_check": current_time,
            "error": None
        }
        
        try:
            if self.async_engine:
                async with self.async_engine.begin() as conn:
                    result = await conn.execute(text("SELECT 1"))
                    if result.scalar() == 1:
                        health_status["healthy"] = True
                        health_status["database_type"] = self.current_database_type.value
            
            self.last_health_check = current_time
            
        except Exception as e:
            health_status["error"] = str(e)
            logger.warning(f"Database health check failed: {e}")
            
            # Try to reconnect if health check fails
            if self.is_connected:
                logger.info("Attempting to reconnect to database...")
                await self._attempt_reconnection()
        
        return health_status
    
    async def _attempt_reconnection(self) -> bool:
        """Attempt to reconnect to database."""
        try:
            # Mark as disconnected
            self.is_connected = False
            
            # Close existing connections
            if self.async_engine:
                await self.async_engine.dispose()
            if self.sync_engine:
                self.sync_engine.dispose()
            
            # Try to reconnect
            return await self.initialize()
            
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False
    
    async def switch_to_fallback(self) -> bool:
        """Manually switch to fallback database."""
        try:
            logger.info("Switching to fallback database...")
            
            # Close current connections
            if self.async_engine:
                await self.async_engine.dispose()
            if self.sync_engine:
                self.sync_engine.dispose()
            
            # Connect to fallback
            if await self._connect_to_fallback():
                logger.info("Successfully switched to fallback database")
                return True
            
            logger.error("Failed to switch to fallback database")
            return False
            
        except Exception as e:
            logger.error(f"Error switching to fallback database: {e}")
            return False
    
    async def switch_to_primary(self) -> bool:
        """Attempt to switch back to primary database."""
        try:
            if not self.fallback_active:
                logger.info("Already using primary database")
                return True
            
            logger.info("Attempting to switch back to primary database...")
            
            # Test primary connection
            if not await self._test_postgresql_connection(self.config.database.primary_url):
                logger.info("Primary database still unavailable")
                return False
            
            # Close current connections
            if self.async_engine:
                await self.async_engine.dispose()
            if self.sync_engine:
                self.sync_engine.dispose()
            
            # Connect to primary
            if await self._connect_to_primary():
                logger.info("Successfully switched back to primary database")
                return True
            
            # If primary fails, reconnect to fallback
            logger.warning("Failed to switch to primary, reconnecting to fallback")
            await self._connect_to_fallback()
            return False
            
        except Exception as e:
            logger.error(f"Error switching to primary database: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get current connection information."""
        return {
            "connected": self.is_connected,
            "database_type": self.current_database_type.value if self.current_database_type else None,
            "database_url": self.current_database_url,
            "fallback_active": self.fallback_active,
            "connection_attempts": self.connection_attempts,
            "last_connection_attempt": self.last_connection_attempt
        }
    
    async def close(self) -> None:
        """Close database connections."""
        try:
            logger.info("Closing database connections...")
            
            if self.async_engine:
                await self.async_engine.dispose()
            
            if self.sync_engine:
                self.sync_engine.dispose()
            
            self.is_connected = False
            self.async_engine = None
            self.sync_engine = None
            self.async_session_factory = None
            self.sync_session_factory = None
            
            logger.info("Database connections closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Global database connection manager
_db_connection_manager: Optional[DatabaseConnectionManager] = None


async def get_database_manager() -> DatabaseConnectionManager:
    """Get the global database connection manager."""
    global _db_connection_manager
    
    if _db_connection_manager is None:
        _db_connection_manager = DatabaseConnectionManager()
        await _db_connection_manager.initialize()
    
    return _db_connection_manager


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    manager = await get_database_manager()
    async with manager.get_async_session() as session:
        yield session


def get_sync_session():
    """Get a sync database session context manager."""
    import asyncio
    
    # Get the manager synchronously (this might need adjustment)
    loop = asyncio.get_event_loop()
    manager = loop.run_until_complete(get_database_manager())
    return manager.get_sync_session()


async def check_database_health() -> Dict[str, Any]:
    """Check database health."""
    try:
        manager = await get_database_manager()
        return await manager.check_health()
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "database_type": None,
            "fallback_active": False
        }


async def ensure_database_ready() -> bool:
    """Ensure database is ready and create tables if needed."""
    try:
        manager = await get_database_manager()
        
        # Check if tables exist
        try:
            async with manager.get_async_session() as session:
                # Try to query a table to see if schema exists
                from ..models.database import User
                from sqlalchemy import select
                result = await session.execute(select(User).limit(1))
                result.fetchall()
                logger.info("Database tables already exist")
                return True
        except Exception:
            # Tables don't exist, create them
            logger.info("Database tables don't exist, creating them...")
            await manager.create_tables()
            return True
            
    except Exception as e:
        logger.error(f"Failed to ensure database ready: {e}")
        return False


async def cleanup_database() -> None:
    """Cleanup database connections."""
    global _db_connection_manager
    
    if _db_connection_manager:
        await _db_connection_manager.close()
        _db_connection_manager = None