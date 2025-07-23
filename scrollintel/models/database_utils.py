"""
Database utilities for connection management, migrations, and testing.
Provides database session management and health checking.
"""

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional, Dict, Any
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import redis.asyncio as redis
import redis as sync_redis
from alembic import command
from alembic.config import Config
import os

from .database import Base
from ..core.config import get_config
from ..core.interfaces import ConfigurationError

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: Optional[str] = None, echo: bool = False):
        """Initialize database manager."""
        self.config = get_config()
        self.database_url = database_url or self.config.database_url
        self.echo = echo or self.config.debug
        
        # Create engines
        self.engine = create_engine(
            self.database_url,
            echo=self.echo,
            pool_size=self.config.db_pool_size,
            max_overflow=self.config.db_max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
        # Create async engine for async operations
        async_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        self.async_engine = create_async_engine(
            async_url,
            echo=self.echo,
            pool_size=self.config.db_pool_size,
            max_overflow=self.config.db_max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
        # Create session factories
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )
        
        self.AsyncSessionLocal = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
        )
        
        # Redis connection
        self.redis_client: Optional[redis.Redis] = None
        self.sync_redis_client: Optional[sync_redis.Redis] = None
        
    async def initialize(self) -> None:
        """Initialize database connections and create tables."""
        try:
            # Test database connection
            async with self.async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            # Initialize Redis
            await self._initialize_redis()
            
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise ConfigurationError(f"Database initialization failed: {e}")
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis connections."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                db=self.config.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            
            self.sync_redis_client = sync_redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                db=self.config.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            
            logger.info("Redis connection initialized successfully")
            
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            self.redis_client = None
            self.sync_redis_client = None
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a synchronous database session."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an asynchronous database session."""
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    async def drop_tables(self) -> None:
        """Drop all database tables."""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    async def check_health(self) -> Dict[str, Any]:
        """Check database and Redis health."""
        health = {
            "database": False,
            "redis": False,
            "details": {}
        }
        
        # Check database
        try:
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                health["database"] = result.scalar() == 1
                health["details"]["database"] = "Connected"
        except Exception as e:
            health["details"]["database"] = f"Error: {str(e)}"
        
        # Check Redis
        if self.redis_client:
            try:
                await self.redis_client.ping()
                health["redis"] = True
                health["details"]["redis"] = "Connected"
            except Exception as e:
                health["details"]["redis"] = f"Error: {str(e)}"
        else:
            health["details"]["redis"] = "Not configured"
        
        return health
    
    async def get_redis(self) -> Optional[redis.Redis]:
        """Get Redis client."""
        return self.redis_client
    
    def get_sync_redis(self) -> Optional[sync_redis.Redis]:
        """Get synchronous Redis client."""
        return self.sync_redis_client
    
    async def close(self) -> None:
        """Close all connections."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            await self.async_engine.dispose()
            self.engine.dispose()
            
            logger.info("Database connections closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


class TestDatabaseManager(DatabaseManager):
    """Database manager for testing with in-memory database."""
    
    def __init__(self):
        """Initialize test database manager with SQLite in-memory database."""
        self.config = get_config()
        self.database_url = "sqlite:///:memory:"
        self.echo = False
        
        # Create engines with SQLite-specific settings
        self.engine = create_engine(
            self.database_url,
            echo=self.echo,
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,
            },
        )
        
        # SQLite doesn't support async, so we use the same engine
        self.async_engine = self.engine
        
        # Create session factories
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )
        
        # For testing, we'll use sync sessions
        self.AsyncSessionLocal = self.SessionLocal
        
        # No Redis for testing
        self.redis_client = None
        self.sync_redis_client = None
    
    async def initialize(self) -> None:
        """Initialize test database."""
        try:
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Test database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize test database: {e}")
            raise
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[Session, None]:
        """Get a session (sync for SQLite testing)."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def check_health(self) -> Dict[str, Any]:
        """Check test database health."""
        return {
            "database": True,
            "redis": False,
            "details": {
                "database": "Test database (SQLite in-memory)",
                "redis": "Not available in test mode"
            }
        }


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


async def init_database(database_url: Optional[str] = None, echo: bool = False) -> DatabaseManager:
    """Initialize the global database manager."""
    global db_manager
    
    if db_manager is None:
        db_manager = DatabaseManager(database_url, echo)
        await db_manager.initialize()
    
    return db_manager


async def get_db_manager() -> DatabaseManager:
    """Get the global database manager."""
    global db_manager
    
    if db_manager is None:
        db_manager = await init_database()
    
    return db_manager


async def cleanup_database() -> None:
    """Cleanup database connections."""
    global db_manager
    
    if db_manager:
        await db_manager.close()
        db_manager = None


# Dependency functions for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency to get database session."""
    manager = await get_db_manager()
    async with manager.get_async_session() as session:
        yield session


async def get_redis() -> Optional[redis.Redis]:
    """FastAPI dependency to get Redis client."""
    manager = await get_db_manager()
    return await manager.get_redis()


async def check_database_health() -> Dict[str, Any]:
    """Check database health status."""
    try:
        manager = await get_db_manager()
        return await manager.check_health()
    except Exception as e:
        return {
            "database": False,
            "redis": False,
            "details": {
                "error": str(e)
            }
        }


def run_migrations(alembic_cfg_path: str = "alembic.ini") -> None:
    """Run database migrations using Alembic."""
    try:
        if not os.path.exists(alembic_cfg_path):
            raise FileNotFoundError(f"Alembic configuration file not found: {alembic_cfg_path}")
        
        alembic_cfg = Config(alembic_cfg_path)
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}")
        raise


def create_migration(message: str, alembic_cfg_path: str = "alembic.ini") -> None:
    """Create a new database migration."""
    try:
        if not os.path.exists(alembic_cfg_path):
            raise FileNotFoundError(f"Alembic configuration file not found: {alembic_cfg_path}")
        
        alembic_cfg = Config(alembic_cfg_path)
        command.revision(alembic_cfg, message=message, autogenerate=True)
        logger.info(f"Migration '{message}' created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create migration: {e}")
        raise


def get_migration_history(alembic_cfg_path: str = "alembic.ini") -> list:
    """Get migration history."""
    try:
        if not os.path.exists(alembic_cfg_path):
            raise FileNotFoundError(f"Alembic configuration file not found: {alembic_cfg_path}")
        
        alembic_cfg = Config(alembic_cfg_path)
        # This would need to be implemented based on Alembic's API
        # For now, return empty list
        return []
        
    except Exception as e:
        logger.error(f"Failed to get migration history: {e}")
        return []


# Test database utilities
_test_db_manager: Optional[TestDatabaseManager] = None


async def get_test_db_manager() -> TestDatabaseManager:
    """Get test database manager."""
    global _test_db_manager
    
    if _test_db_manager is None:
        _test_db_manager = TestDatabaseManager()
        await _test_db_manager.initialize()
    
    return _test_db_manager


async def cleanup_test_database() -> None:
    """Cleanup test database."""
    global _test_db_manager
    
    if _test_db_manager:
        await _test_db_manager.close()
        _test_db_manager = None


@asynccontextmanager
async def get_test_session() -> AsyncGenerator[Session, None]:
    """Get test database session."""
    manager = await get_test_db_manager()
    async with manager.get_async_session() as session:
        yield session


# Database inspection utilities
def get_table_info(table_name: str) -> Dict[str, Any]:
    """Get information about a database table."""
    try:
        config = get_config()
        engine = create_engine(config.database_url)
        inspector = inspect(engine)
        
        if table_name not in inspector.get_table_names():
            raise ValueError(f"Table '{table_name}' does not exist")
        
        columns = inspector.get_columns(table_name)
        indexes = inspector.get_indexes(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        
        return {
            "table_name": table_name,
            "columns": columns,
            "indexes": indexes,
            "foreign_keys": foreign_keys,
        }
        
    except Exception as e:
        logger.error(f"Failed to get table info for '{table_name}': {e}")
        raise


def get_database_schema() -> Dict[str, Any]:
    """Get complete database schema information."""
    try:
        config = get_config()
        engine = create_engine(config.database_url)
        inspector = inspect(engine)
        
        tables = {}
        for table_name in inspector.get_table_names():
            tables[table_name] = get_table_info(table_name)
        
        return {
            "database_url": config.database_url,
            "tables": tables,
            "table_count": len(tables),
        }
        
    except Exception as e:
        logger.error(f"Failed to get database schema: {e}")
        raise