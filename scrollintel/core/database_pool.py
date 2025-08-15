"""
Optimized Database Connection Pool Manager for ScrollIntel.
Provides high-performance database connections with connection pooling,
query optimization, and performance monitoring.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import create_engine, text, event, pool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.engine import Engine
from sqlalchemy.sql import ClauseElement
import redis.asyncio as redis

from ..core.config import get_settings
from ..core.interfaces import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolStats:
    """Statistics for database connection pool."""
    pool_size: int
    checked_in: int
    checked_out: int
    overflow: int
    invalid: int
    total_connections: int
    peak_connections: int
    connection_requests: int
    failed_connections: int
    avg_connection_time: float
    query_count: int
    slow_query_count: int
    cache_hits: int
    cache_misses: int


@dataclass
class QueryStats:
    """Statistics for database queries."""
    query_hash: str
    query_text: str
    execution_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    last_executed: datetime
    is_slow: bool


class QueryCache:
    """Simple in-memory query cache."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result."""
        if key not in self.cache:
            return None
        
        # Check TTL
        if datetime.utcnow() - self.access_times[key] > timedelta(seconds=self.ttl_seconds):
            self.delete(key)
            return None
        
        self.access_times[key] = datetime.utcnow()
        return self.cache[key]['result']
    
    def set(self, key: str, value: Any) -> None:
        """Set cached result."""
        # Evict old entries if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = {
            'result': value,
            'created_at': datetime.utcnow()
        }
        self.access_times[key] = datetime.utcnow()
    
    def delete(self, key: str) -> None:
        """Delete cached result."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self.access_times.clear()
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self.delete(oldest_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'hit_rate': 0.0  # Would need to track hits/misses
        }


class PerformanceMonitor:
    """Monitor database performance metrics."""
    
    def __init__(self):
        self.query_stats: Dict[str, QueryStats] = {}
        self.connection_stats = ConnectionPoolStats(
            pool_size=0, checked_in=0, checked_out=0, overflow=0,
            invalid=0, total_connections=0, peak_connections=0,
            connection_requests=0, failed_connections=0,
            avg_connection_time=0.0, query_count=0, slow_query_count=0,
            cache_hits=0, cache_misses=0
        )
        self.slow_query_threshold = 1.0  # 1 second
    
    def record_query(self, query_text: str, execution_time: float) -> None:
        """Record query execution statistics."""
        query_hash = str(hash(query_text))
        
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = QueryStats(
                query_hash=query_hash,
                query_text=query_text[:500],  # Truncate long queries
                execution_count=0,
                total_time=0.0,
                avg_time=0.0,
                min_time=float('inf'),
                max_time=0.0,
                last_executed=datetime.utcnow(),
                is_slow=False
            )
        
        stats = self.query_stats[query_hash]
        stats.execution_count += 1
        stats.total_time += execution_time
        stats.avg_time = stats.total_time / stats.execution_count
        stats.min_time = min(stats.min_time, execution_time)
        stats.max_time = max(stats.max_time, execution_time)
        stats.last_executed = datetime.utcnow()
        stats.is_slow = execution_time > self.slow_query_threshold
        
        self.connection_stats.query_count += 1
        if stats.is_slow:
            self.connection_stats.slow_query_count += 1
    
    def record_cache_hit(self) -> None:
        """Record cache hit."""
        self.connection_stats.cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record cache miss."""
        self.connection_stats.cache_misses += 1
    
    def get_slow_queries(self, limit: int = 10) -> List[QueryStats]:
        """Get slowest queries."""
        slow_queries = [q for q in self.query_stats.values() if q.is_slow]
        return sorted(slow_queries, key=lambda q: q.avg_time, reverse=True)[:limit]
    
    def get_frequent_queries(self, limit: int = 10) -> List[QueryStats]:
        """Get most frequent queries."""
        return sorted(
            self.query_stats.values(),
            key=lambda q: q.execution_count,
            reverse=True
        )[:limit]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'connection_stats': self.connection_stats.__dict__,
            'query_stats': {
                'total_queries': len(self.query_stats),
                'slow_queries': len([q for q in self.query_stats.values() if q.is_slow]),
                'avg_query_time': sum(q.avg_time for q in self.query_stats.values()) / len(self.query_stats) if self.query_stats else 0,
            },
            'slow_queries': [q.__dict__ for q in self.get_slow_queries()],
            'frequent_queries': [q.__dict__ for q in self.get_frequent_queries()],
            'generated_at': datetime.utcnow().isoformat()
        }


class OptimizedDatabasePool:
    """High-performance database connection pool with monitoring and caching."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize optimized database pool."""
        self.settings = get_settings()
        self.database_url = database_url or self.settings.database_url
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        self.query_cache = QueryCache(
            max_size=1000,
            ttl_seconds=self.settings.cache_ttl_seconds
        ) if self.settings.enable_query_cache else None
        
        # Create optimized engines
        self._create_engines()
        
        # Session factories
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False  # Optimize for performance
        )
        
        self.AsyncSessionLocal = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
        
        # Redis for distributed caching
        self.redis_client: Optional[redis.Redis] = None
        
    def _create_engines(self) -> None:
        """Create optimized database engines."""
        # Sync engine with optimized pool
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=self.settings.db_pool_size,
            max_overflow=self.settings.db_max_overflow,
            pool_timeout=self.settings.db_pool_timeout,
            pool_recycle=self.settings.db_pool_recycle,
            pool_pre_ping=self.settings.db_pool_pre_ping,
            echo=False,  # Disable SQL logging for performance
            future=True,  # Use SQLAlchemy 2.0 style
            # Performance optimizations
            connect_args={
                "application_name": "scrollintel",
                "options": "-c default_transaction_isolation=read_committed"
            }
        )
        
        # Async engine
        async_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        self.async_engine = create_async_engine(
            async_url,
            poolclass=pool.QueuePool,
            pool_size=self.settings.db_pool_size,
            max_overflow=self.settings.db_max_overflow,
            pool_timeout=self.settings.db_pool_timeout,
            pool_recycle=self.settings.db_pool_recycle,
            pool_pre_ping=self.settings.db_pool_pre_ping,
            echo=False,
            future=True,
            connect_args={
                "application_name": "scrollintel_async",
                "command_timeout": 60
            }
        )
        
        # Add performance monitoring events
        self._setup_monitoring()
    
    def _setup_monitoring(self) -> None:
        """Setup performance monitoring events."""
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total_time = time.time() - context._query_start_time
            self.monitor.record_query(statement, total_time)
        
        @event.listens_for(self.engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            self.monitor.connection_stats.connection_requests += 1
            
            # Set connection-level optimizations
            with dbapi_connection.cursor() as cursor:
                # Optimize PostgreSQL settings for performance
                cursor.execute("SET synchronous_commit = off")
                cursor.execute("SET wal_buffers = '16MB'")
                cursor.execute("SET checkpoint_completion_target = 0.9")
                cursor.execute("SET random_page_cost = 1.1")
    
    async def initialize(self) -> None:
        """Initialize the database pool."""
        try:
            # Test database connection
            async with self.async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            # Initialize Redis if configured
            if self.settings.redis_host:
                await self._initialize_redis()
            
            logger.info("Optimized database pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise ConfigurationError(f"Database pool initialization failed: {e}")
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis for distributed caching."""
        try:
            self.redis_client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                password=self.settings.redis_password,
                db=self.settings.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                max_connections=20
            )
            
            await self.redis_client.ping()
            logger.info("Redis initialized for distributed caching")
            
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            self.redis_client = None
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get optimized synchronous database session."""
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
        """Get optimized asynchronous database session."""
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def execute_cached_query(
        self,
        query: ClauseElement,
        parameters: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None
    ) -> Any:
        """Execute query with caching support."""
        if not self.query_cache:
            # No caching, execute directly
            async with self.get_async_session() as session:
                result = await session.execute(query, parameters or {})
                return result.fetchall()
        
        # Generate cache key
        if not cache_key:
            cache_key = f"query:{hash(str(query))}:{hash(str(parameters))}"
        
        # Check cache first
        cached_result = self.query_cache.get(cache_key)
        if cached_result is not None:
            self.monitor.record_cache_hit()
            return cached_result
        
        # Execute query
        async with self.get_async_session() as session:
            result = await session.execute(query, parameters or {})
            data = result.fetchall()
        
        # Cache result
        self.query_cache.set(cache_key, data)
        self.monitor.record_cache_miss()
        
        return data
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current connection pool status."""
        pool = self.engine.pool
        
        return {
            'pool_size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'invalid': pool.invalid(),
            'total_connections': pool.size() + pool.overflow(),
            'pool_class': pool.__class__.__name__,
            'engine_url': str(self.engine.url).replace(self.engine.url.password or '', '***')
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        pool_status = self.get_pool_status()
        performance_report = self.monitor.get_performance_report()
        
        cache_stats = {}
        if self.query_cache:
            cache_stats = self.query_cache.get_stats()
        
        return {
            'pool_status': pool_status,
            'performance_report': performance_report,
            'cache_stats': cache_stats,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Run database optimization tasks."""
        optimizations = []
        
        try:
            async with self.get_async_session() as session:
                # Analyze tables for better query planning
                await session.execute(text("ANALYZE"))
                optimizations.append("Table statistics updated")
                
                # Vacuum to reclaim space (if needed)
                # Note: VACUUM cannot run inside a transaction
                pass
                
                # Update connection pool statistics
                self.monitor.connection_stats.total_connections = self.get_pool_status()['total_connections']
                
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            optimizations.append(f"Optimization failed: {e}")
        
        return {
            'optimizations_applied': optimizations,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def close(self) -> None:
        """Close all connections and cleanup."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            await self.async_engine.dispose()
            self.engine.dispose()
            
            if self.query_cache:
                self.query_cache.clear()
            
            logger.info("Database pool closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing database pool: {e}")


# Global optimized database pool instance
_db_pool: Optional[OptimizedDatabasePool] = None


async def get_optimized_db_pool() -> OptimizedDatabasePool:
    """Get the global optimized database pool."""
    global _db_pool
    
    if _db_pool is None:
        _db_pool = OptimizedDatabasePool()
        await _db_pool.initialize()
    
    return _db_pool


async def cleanup_db_pool() -> None:
    """Cleanup the global database pool."""
    global _db_pool
    
    if _db_pool:
        await _db_pool.close()
        _db_pool = None


# FastAPI dependencies
async def get_optimized_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for optimized database session."""
    pool = await get_optimized_db_pool()
    async with pool.get_async_session() as session:
        yield session


def get_optimized_sync_db() -> Generator[Session, None, None]:
    """FastAPI dependency for optimized synchronous database session."""
    # For sync operations, we'll create a simple optimized connection
    from ..core.config import get_settings
    settings = get_settings()
    
    engine = create_engine(
        settings.database_url,
        poolclass=QueuePool,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=settings.db_pool_pre_ping,
        echo=False
    )
    
    SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )
    
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()