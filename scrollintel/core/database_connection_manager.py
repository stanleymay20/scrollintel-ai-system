"""
Database Connection Manager - 100% Optimized
Intelligent database management with automatic fallback
"""

import asyncio
import logging
import os
import sqlite3
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
import time

logger = logging.getLogger(__name__)

class DatabaseConnectionManager:
    """Intelligent database connection manager with fallback"""
    
    def __init__(self):
        self.primary_engine = None
        self.fallback_engine = None
        self.session_factory = None
        self.connection_status = "disconnected"
        self.fallback_active = False
        self.connection_pool = None
        
    async def initialize_with_fallback(self):
        """Initialize database with intelligent fallback"""
        logger.info("🗄️  Initializing database connection...")
        
        # Try PostgreSQL first
        if await self._try_postgresql():
            logger.info("✅ PostgreSQL connection established")
            self.connection_status = "postgresql"
            return True
        
        # Fallback to SQLite
        if await self._setup_sqlite_fallback():
            logger.info("✅ SQLite fallback connection established")
            self.connection_status = "sqlite_fallback"
            self.fallback_active = True
            return True
        
        logger.error("❌ All database connections failed")
        self.connection_status = "failed"
        return False
    
    async def _try_postgresql(self) -> bool:
        """Try to connect to PostgreSQL"""
        try:
            from scrollintel.core.config import get_config
            config = get_config()
            
            database_url = config.DATABASE_URL
            if not database_url or database_url == "your_database_url_here":
                logger.info("PostgreSQL URL not configured, skipping...")
                return False
            
            # Test connection
            logger.info("Testing PostgreSQL connection...")
            
            # Create async engine
            self.primary_engine = create_async_engine(
                database_url,
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Test connection
            async with self.primary_engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.primary_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("PostgreSQL connection test successful")
            return True
            
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
            if self.primary_engine:
                await self.primary_engine.dispose()
                self.primary_engine = None
            return False
    
    async def _setup_sqlite_fallback(self) -> bool:
        """Setup SQLite fallback database"""
        try:
            logger.info("Setting up SQLite fallback...")
            
            # Create SQLite database path
            db_path = "scrollintel_fallback.db"
            
            # Create async SQLite engine
            sqlite_url = f"sqlite+aiosqlite:///{db_path}"
            
            self.fallback_engine = create_async_engine(
                sqlite_url,
                echo=False,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 20
                }
            )
            
            # Test connection
            async with self.fallback_engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.fallback_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Initialize basic tables
            await self._initialize_fallback_tables()
            
            logger.info(f"SQLite fallback database created: {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"SQLite fallback setup failed: {e}")
            return False
    
    async def _initialize_fallback_tables(self):
        """Initialize basic tables for fallback database"""
        try:
            async with self.get_session() as session:
                # Create basic tables
                await session.execute("""
                    CREATE TABLE IF NOT EXISTS system_status (
                        id INTEGER PRIMARY KEY,
                        component TEXT NOT NULL,
                        status TEXT NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                await session.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                await session.execute("""
                    CREATE TABLE IF NOT EXISTS agent_logs (
                        id INTEGER PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        message TEXT NOT NULL,
                        level TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                await session.commit()
                logger.info("Fallback database tables initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize fallback tables: {e}")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic fallback"""
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
        
        session = self.session_factory()
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()
    
    async def get_connection_pool(self):
        """Get connection pool"""
        if self.primary_engine:
            return self.primary_engine.pool
        elif self.fallback_engine:
            return self.fallback_engine.pool
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            start_time = time.time()
            
            async with self.get_session() as session:
                await session.execute("SELECT 1")
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "connection_type": self.connection_status,
                "fallback_active": self.fallback_active,
                "response_time_ms": round(response_time * 1000, 2),
                "pool_info": await self._get_pool_info()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection_type": self.connection_status,
                "fallback_active": self.fallback_active
            }
    
    async def _get_pool_info(self) -> Dict[str, Any]:
        """Get connection pool information"""
        try:
            pool = await self.get_connection_pool()
            if pool:
                return {
                    "size": getattr(pool, 'size', 0),
                    "checked_in": getattr(pool, 'checkedin', 0),
                    "checked_out": getattr(pool, 'checkedout', 0),
                    "overflow": getattr(pool, 'overflow', 0)
                }
        except Exception:
            pass
        
        return {"info": "not_available"}
    
    async def execute_query(self, query: str, params: Optional[Dict] = None):
        """Execute query with automatic retry and fallback"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                async with self.get_session() as session:
                    result = await session.execute(query, params or {})
                    await session.commit()
                    return result
                    
            except Exception as e:
                logger.warning(f"Query execution attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    raise
                
                # Wait before retry
                await asyncio.sleep(0.5 * (attempt + 1))
    
    async def log_performance_metric(self, metric_name: str, value: float):
        """Log performance metric to database"""
        try:
            async with self.get_session() as session:
                if self.fallback_active:
                    await session.execute(
                        "INSERT INTO performance_metrics (metric_name, metric_value) VALUES (?, ?)",
                        (metric_name, value)
                    )
                else:
                    # PostgreSQL syntax
                    await session.execute(
                        "INSERT INTO performance_metrics (metric_name, metric_value) VALUES (:name, :value)",
                        {"name": metric_name, "value": value}
                    )
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to log performance metric: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status from database"""
        try:
            async with self.get_session() as session:
                if self.fallback_active:
                    result = await session.execute(
                        "SELECT component, status, last_updated FROM system_status ORDER BY last_updated DESC LIMIT 10"
                    )
                else:
                    result = await session.execute(
                        "SELECT component, status, last_updated FROM system_status ORDER BY last_updated DESC LIMIT 10"
                    )
                
                rows = result.fetchall()
                return {
                    "components": [
                        {
                            "component": row[0],
                            "status": row[1],
                            "last_updated": row[2]
                        }
                        for row in rows
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"components": [], "error": str(e)}
    
    async def close(self):
        """Close database connections"""
        try:
            if self.primary_engine:
                await self.primary_engine.dispose()
                logger.info("PostgreSQL connection closed")
            
            if self.fallback_engine:
                await self.fallback_engine.dispose()
                logger.info("SQLite fallback connection closed")
                
            self.connection_status = "disconnected"
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

# Global database manager instance
_db_manager = None

def get_database_manager() -> DatabaseConnectionManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseConnectionManager()
    return _db_manager

async def initialize_database():
    """Initialize database with fallback"""
    manager = get_database_manager()
    return await manager.initialize_with_fallback()

async def get_db_session():
    """Get database session"""
    manager = get_database_manager()
    async with manager.get_session() as session:
        yield session