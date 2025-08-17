"""
Enhanced Database Connection Pool for Heavy Volume Support
Implements connection pooling, async operations, and performance optimizations.
"""

import os
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.orm import sessionmaker
import psutil

logger = logging.getLogger(__name__)


class OptimizedDatabasePool:
    """Optimized database connection pool for heavy volume operations."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///scrollintel.db')
        self.is_sqlite = 'sqlite' in self.database_url.lower()
        
        # Heavy volume connection pool settings
        self.pool_config = self._get_pool_config()
        
        # Initialize engines
        self.sync_engine = None
        self.async_engine = None
        self.async_session_factory = None
        
        # Performance monitoring
        self.connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'query_count': 0,
            'avg_query_time': 0.0
        }
    
    def _get_pool_config(self) -> Dict[str, Any]:
        """Get optimized pool configuration based on database type."""
        
        if self.is_sqlite:
            # SQLite configuration (limited pooling)
            return {
                'poolclass': NullPool,  # SQLite doesn't support connection pooling well
                'pool_pre_ping': False,
                'echo': False,
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 30
                }
            }
        else:
            # PostgreSQL/MySQL configuration for heavy volume
            return {
                'poolclass': QueuePool,
                'pool_size': 20,           # Base connections (increased from default 5)
                'max_overflow': 30,        # Additional connections (increased from default 10)
                'pool_timeout': 30,        # Wait time for connection
                'pool_recycle': 3600,      # Recycle connections after 1 hour
                'pool_pre_ping': True,     # Validate connections before use
                'echo': False,             # Set to True for SQL debugging
                'connect_args': {
                    'connect_timeout': 10,
                    'application_name': 'ScrollIntel_HeavyVolume'
                }
            }
    
    async def initialize(self) -> None:
        """Initialize database engines and connection pools."""
        
        logger.info(f"Initializing database pool for: {self.database_url}")
        
        try:
            # Create synchronous engine
            self.sync_engine = create_engine(
                self.database_url,
                **self.pool_config
            )
            
            # Create asynchronous engine for heavy volume operations
            if not self.is_sqlite:
                # Convert to async URL for PostgreSQL/MySQL
                async_url = self.database_url.replace('postgresql://', 'postgresql+asyncpg://')
                async_url = async_url.replace('mysql://', 'mysql+aiomysql://')
                
                self.async_engine = create_async_engine(
                    async_url,
                    **{k: v for k, v in self.pool_config.items() 
                       if k not in ['connect_args']}  # Remove sync-specific args
                )
                
                # Create async session factory
                self.async_session_factory = async_sessionmaker(
                    self.async_engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
            
            logger.info("Database pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {str(e)}")
            raise
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup."""
        
        if not self.async_engine:
            raise RuntimeError("Async engine not initialized. Call initialize() first.")
        
        session = self.async_session_factory()
        try:
            self.connection_stats['active_connections'] += 1
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            self.connection_stats['failed_connections'] += 1
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            await session.close()
            self.connection_stats['active_connections'] -= 1
    
    def get_sync_session(self):
        """Get synchronous database session."""
        
        if not self.sync_engine:
            raise RuntimeError("Sync engine not initialized. Call initialize() first.")
        
        Session = sessionmaker(bind=self.sync_engine)
        return Session()
    
    async def execute_batch_query(
        self, 
        query: str, 
        params_list: list,
        batch_size: int = 1000
    ) -> None:
        """Execute batch queries efficiently for heavy volume operations."""
        
        if not self.async_engine:
            raise RuntimeError("Async engine required for batch operations")
        
        total_batches = len(params_list) // batch_size + (1 if len(params_list) % batch_size else 0)
        
        async with self.get_async_session() as session:
            for i in range(0, len(params_list), batch_size):
                batch = params_list[i:i + batch_size]
                
                try:
                    await session.execute(text(query), batch)
                    
                    # Commit every batch to avoid long transactions
                    await session.commit()
                    
                    batch_num = (i // batch_size) + 1
                    logger.debug(f"Executed batch {batch_num}/{total_batches} ({len(batch)} records)")
                    
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Batch execution failed at batch {batch_num}: {str(e)}")
                    raise
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection pool statistics."""
        
        stats = self.connection_stats.copy()
        
        if self.sync_engine and hasattr(self.sync_engine.pool, 'size'):
            pool = self.sync_engine.pool
            stats.update({
                'pool_size': pool.size(),
                'checked_in': pool.checkedin(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'invalid': pool.invalid()
            })
        
        # Add system resource usage
        memory_info = psutil.virtual_memory()
        stats.update({
            'system_memory_percent': memory_info.percent,
            'system_memory_available_gb': memory_info.available / (1024**3)
        })
        
        return stats
    
    async def optimize_for_heavy_volume(self) -> None:
        """Apply database optimizations for heavy volume processing."""
        
        if self.is_sqlite:
            # SQLite optimizations
            optimizations = [
                "PRAGMA journal_mode = WAL;",
                "PRAGMA synchronous = NORMAL;",
                "PRAGMA cache_size = 10000;",
                "PRAGMA temp_store = MEMORY;",
                "PRAGMA mmap_size = 268435456;"  # 256MB
            ]
        else:
            # PostgreSQL optimizations
            optimizations = [
                "SET work_mem = '256MB';",
                "SET maintenance_work_mem = '1GB';",
                "SET effective_cache_size = '4GB';",
                "SET random_page_cost = 1.1;",
                "SET checkpoint_completion_target = 0.9;"
            ]
        
        try:
            if self.async_engine:
                async with self.get_async_session() as session:
                    for optimization in optimizations:
                        await session.execute(text(optimization))
                        logger.debug(f"Applied optimization: {optimization}")
            else:
                with self.get_sync_session() as session:
                    for optimization in optimizations:
                        session.execute(text(optimization))
                        logger.debug(f"Applied optimization: {optimization}")
            
            logger.info("Database optimizations applied for heavy volume processing")
            
        except Exception as e:
            logger.warning(f"Some database optimizations failed: {str(e)}")
    
    async def close(self) -> None:
        """Close all database connections and clean up resources."""
        
        logger.info("Closing database connection pool...")
        
        try:
            if self.async_engine:
                await self.async_engine.dispose()
            
            if self.sync_engine:
                self.sync_engine.dispose()
            
            logger.info("Database pool closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing database pool: {str(e)}")


# Global database pool instance
_db_pool: Optional[OptimizedDatabasePool] = None


async def get_optimized_db_pool() -> OptimizedDatabasePool:
    """Get or create the global optimized database pool."""
    
    global _db_pool
    
    if _db_pool is None:
        _db_pool = OptimizedDatabasePool()
        await _db_pool.initialize()
        await _db_pool.optimize_for_heavy_volume()
    
    return _db_pool


async def close_db_pool() -> None:
    """Close the global database pool."""
    
    global _db_pool
    
    if _db_pool:
        await _db_pool.close()
        _db_pool = None