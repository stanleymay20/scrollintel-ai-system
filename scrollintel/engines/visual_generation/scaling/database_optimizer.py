"""
Database optimization for high-throughput visual generation operations.
Implements connection pooling, query optimization, and performance monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool, Connection
import json
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    BATCH_INSERT = "batch_insert"
    BATCH_UPDATE = "batch_update"


@dataclass
class QueryMetrics:
    """Metrics for database query performance."""
    query_type: QueryType
    execution_time: float
    rows_affected: int
    timestamp: datetime = field(default_factory=datetime.now)
    query_hash: str = ""
    error: Optional[str] = None


@dataclass
class ConnectionPoolConfig:
    """Configuration for database connection pool."""
    min_size: int = 10
    max_size: int = 50
    max_queries: int = 50000
    max_inactive_connection_lifetime: float = 300.0
    timeout: float = 60.0
    command_timeout: float = 30.0
    server_settings: Dict[str, str] = field(default_factory=lambda: {
        'application_name': 'visual_generation',
        'tcp_keepalives_idle': '600',
        'tcp_keepalives_interval': '30',
        'tcp_keepalives_count': '3'
    })


class DatabaseOptimizer:
    """
    High-performance database optimizer for visual generation operations.
    Provides connection pooling, query optimization, and performance monitoring.
    """
    
    def __init__(self, database_url: str, config: ConnectionPoolConfig = None):
        self.database_url = database_url
        self.config = config or ConnectionPoolConfig()
        self.pool: Optional[Pool] = None
        
        # Performance monitoring
        self.query_metrics: List[QueryMetrics] = []
        self.slow_query_threshold = 1.0  # seconds
        self.metrics_retention_hours = 24
        
        # Query cache for prepared statements
        self.prepared_statements: Dict[str, str] = {}
        
        # Batch operation settings
        self.batch_size = 1000
        self.batch_timeout = 5.0  # seconds
        
        # Background tasks
        self._metrics_cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self):
        """Initialize the database optimizer."""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                max_queries=self.config.max_queries,
                max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
                timeout=self.config.timeout,
                command_timeout=self.config.command_timeout,
                server_settings=self.config.server_settings
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            logger.info("Database connection pool initialized successfully")
            
            # Start background tasks
            self._running = True
            self._metrics_cleanup_task = asyncio.create_task(self._metrics_cleanup_loop())
            
            # Create optimized indexes and tables
            await self._setup_database_optimizations()
            
        except Exception as e:
            logger.error(f"Failed to initialize database optimizer: {e}")
            raise
    
    async def close(self):
        """Close the database optimizer."""
        self._running = False
        
        if self._metrics_cleanup_task:
            self._metrics_cleanup_task.cancel()
            try:
                await self._metrics_cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute_query(self, query: str, *args, query_type: QueryType = QueryType.SELECT) -> Any:
        """Execute a single query with performance monitoring."""
        start_time = time.time()
        query_hash = str(hash(query))
        
        try:
            async with self.get_connection() as conn:
                if query_type == QueryType.SELECT:
                    result = await conn.fetch(query, *args)
                    rows_affected = len(result)
                else:
                    result = await conn.execute(query, *args)
                    # Parse rows affected from result string (e.g., "INSERT 0 5")
                    rows_affected = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
                
                execution_time = time.time() - start_time
                
                # Record metrics
                self._record_query_metrics(
                    query_type, execution_time, rows_affected, query_hash
                )
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_query_metrics(
                query_type, execution_time, 0, query_hash, str(e)
            )
            raise
    
    async def execute_batch_insert(self, table: str, columns: List[str], 
                                  data: List[Tuple], conflict_action: str = "NOTHING") -> int:
        """Execute optimized batch insert operation."""
        if not data:
            return 0
        
        start_time = time.time()
        total_inserted = 0
        
        try:
            # Prepare the query
            placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])
            column_names = ", ".join(columns)
            
            query = f"""
                INSERT INTO {table} ({column_names})
                VALUES ({placeholders})
                ON CONFLICT DO {conflict_action}
            """
            
            async with self.get_connection() as conn:
                # Process in batches
                for i in range(0, len(data), self.batch_size):
                    batch = data[i:i + self.batch_size]
                    
                    # Use executemany for better performance
                    result = await conn.executemany(query, batch)
                    total_inserted += len(batch)
            
            execution_time = time.time() - start_time
            self._record_query_metrics(
                QueryType.BATCH_INSERT, execution_time, total_inserted
            )
            
            return total_inserted
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_query_metrics(
                QueryType.BATCH_INSERT, execution_time, 0, error=str(e)
            )
            raise
    
    async def execute_batch_update(self, table: str, updates: List[Dict[str, Any]], 
                                  key_column: str) -> int:
        """Execute optimized batch update operation."""
        if not updates:
            return 0
        
        start_time = time.time()
        total_updated = 0
        
        try:
            async with self.get_connection() as conn:
                # Process in batches
                for i in range(0, len(updates), self.batch_size):
                    batch = updates[i:i + self.batch_size]
                    
                    # Build dynamic update query
                    if batch:
                        sample_update = batch[0]
                        set_columns = [col for col in sample_update.keys() if col != key_column]
                        
                        # Create CASE statements for each column
                        case_statements = []
                        for col in set_columns:
                            cases = []
                            for update in batch:
                                key_val = update[key_column]
                                col_val = update[col]
                                cases.append(f"WHEN {key_column} = '{key_val}' THEN '{col_val}'")
                            
                            case_stmt = f"{col} = CASE " + " ".join(cases) + f" ELSE {col} END"
                            case_statements.append(case_stmt)
                        
                        # Build WHERE clause
                        key_values = [str(update[key_column]) for update in batch]
                        where_clause = f"{key_column} IN ({', '.join([f\"'{val}'\" for val in key_values])})"
                        
                        query = f"""
                            UPDATE {table}
                            SET {', '.join(case_statements)}
                            WHERE {where_clause}
                        """
                        
                        result = await conn.execute(query)
                        rows_affected = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
                        total_updated += rows_affected
            
            execution_time = time.time() - start_time
            self._record_query_metrics(
                QueryType.BATCH_UPDATE, execution_time, total_updated
            )
            
            return total_updated
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_query_metrics(
                QueryType.BATCH_UPDATE, execution_time, 0, error=str(e)
            )
            raise
    
    async def execute_transaction(self, operations: List[Tuple[str, tuple]]) -> List[Any]:
        """Execute multiple operations in a transaction."""
        start_time = time.time()
        results = []
        
        try:
            async with self.get_connection() as conn:
                async with conn.transaction():
                    for query, args in operations:
                        result = await conn.execute(query, *args)
                        results.append(result)
            
            execution_time = time.time() - start_time
            self._record_query_metrics(
                QueryType.UPDATE, execution_time, len(operations)
            )
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_query_metrics(
                QueryType.UPDATE, execution_time, 0, error=str(e)
            )
            raise
    
    async def get_generation_requests_batch(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Optimized query to get generation requests by status."""
        query = """
            SELECT id, user_id, session_id, request_type, prompt, parameters,
                   status, worker_id, created_at, started_at, completed_at,
                   progress, result_urls, error_message, retry_count, priority
            FROM visual_generation_requests
            WHERE status = $1
            ORDER BY priority DESC, created_at ASC
            LIMIT $2
        """
        
        result = await self.execute_query(query, status, limit, QueryType.SELECT)
        
        # Convert to dictionaries
        requests = []
        for row in result:
            request_dict = dict(row)
            # Parse JSON fields
            if request_dict.get('parameters'):
                request_dict['parameters'] = json.loads(request_dict['parameters'])
            if request_dict.get('result_urls'):
                request_dict['result_urls'] = json.loads(request_dict['result_urls'])
            requests.append(request_dict)
        
        return requests
    
    async def update_request_status_batch(self, request_updates: List[Dict[str, Any]]) -> int:
        """Batch update request statuses."""
        if not request_updates:
            return 0
        
        # Prepare batch data for update
        updates = []
        for update in request_updates:
            updates.append({
                'id': update['id'],
                'status': update.get('status'),
                'worker_id': update.get('worker_id'),
                'started_at': update.get('started_at'),
                'completed_at': update.get('completed_at'),
                'progress': update.get('progress', 0.0),
                'result_urls': json.dumps(update.get('result_urls', [])),
                'error_message': update.get('error_message')
            })
        
        return await self.execute_batch_update(
            'visual_generation_requests', updates, 'id'
        )
    
    async def insert_generation_results_batch(self, results: List[Dict[str, Any]]) -> int:
        """Batch insert generation results."""
        if not results:
            return 0
        
        columns = ['request_id', 'content_type', 'file_path', 'file_size', 
                  'quality_score', 'metadata', 'created_at']
        
        data = []
        for result in results:
            data.append((
                result['request_id'],
                result['content_type'],
                result['file_path'],
                result.get('file_size', 0),
                result.get('quality_score', 0.0),
                json.dumps(result.get('metadata', {})),
                result.get('created_at', datetime.now())
            ))
        
        return await self.execute_batch_insert(
            'visual_generation_results', columns, data
        )
    
    async def cleanup_old_requests(self, days_old: int = 30) -> int:
        """Clean up old completed/failed requests."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        query = """
            DELETE FROM visual_generation_requests
            WHERE status IN ('completed', 'failed', 'cancelled')
            AND completed_at < $1
        """
        
        result = await self.execute_query(query, cutoff_date, QueryType.DELETE)
        rows_affected = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
        
        logger.info(f"Cleaned up {rows_affected} old requests")
        return rows_affected
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        # Calculate metrics from recent queries
        recent_metrics = [
            m for m in self.query_metrics
            if m.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        if not recent_metrics:
            return {
                'total_queries': 0,
                'average_execution_time': 0.0,
                'slow_queries': 0,
                'error_rate': 0.0,
                'queries_per_second': 0.0
            }
        
        total_queries = len(recent_metrics)
        total_execution_time = sum(m.execution_time for m in recent_metrics)
        slow_queries = len([m for m in recent_metrics if m.execution_time > self.slow_query_threshold])
        errors = len([m for m in recent_metrics if m.error])
        
        # Calculate queries per second (last hour)
        time_span_hours = 1.0
        queries_per_second = total_queries / (time_span_hours * 3600)
        
        return {
            'total_queries': total_queries,
            'average_execution_time': total_execution_time / total_queries,
            'slow_queries': slow_queries,
            'error_rate': errors / total_queries if total_queries > 0 else 0.0,
            'queries_per_second': queries_per_second,
            'pool_size': self.pool.get_size() if self.pool else 0,
            'pool_idle_connections': self.pool.get_idle_size() if self.pool else 0
        }
    
    async def optimize_database_performance(self):
        """Run database optimization tasks."""
        try:
            async with self.get_connection() as conn:
                # Update table statistics
                await conn.execute("ANALYZE visual_generation_requests")
                await conn.execute("ANALYZE visual_generation_results")
                await conn.execute("ANALYZE user_sessions")
                
                # Vacuum if needed (check table bloat)
                await conn.execute("VACUUM (ANALYZE) visual_generation_requests")
                
                logger.info("Database optimization completed")
                
        except Exception as e:
            logger.error(f"Error during database optimization: {e}")
    
    def _record_query_metrics(self, query_type: QueryType, execution_time: float,
                             rows_affected: int, query_hash: str = "", error: str = None):
        """Record query performance metrics."""
        metric = QueryMetrics(
            query_type=query_type,
            execution_time=execution_time,
            rows_affected=rows_affected,
            query_hash=query_hash,
            error=error
        )
        
        self.query_metrics.append(metric)
        
        # Log slow queries
        if execution_time > self.slow_query_threshold:
            logger.warning(f"Slow query detected: {query_type.value} took {execution_time:.2f}s")
        
        # Log errors
        if error:
            logger.error(f"Query error: {query_type.value} - {error}")
    
    async def _setup_database_optimizations(self):
        """Set up database indexes and optimizations."""
        try:
            async with self.get_connection() as conn:
                # Create indexes for visual generation tables
                indexes = [
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vg_requests_status ON visual_generation_requests(status)",
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vg_requests_user_id ON visual_generation_requests(user_id)",
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vg_requests_created_at ON visual_generation_requests(created_at)",
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vg_requests_priority_created ON visual_generation_requests(priority DESC, created_at ASC)",
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vg_results_request_id ON visual_generation_results(request_id)",
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id)",
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_status ON user_sessions(status)"
                ]
                
                for index_sql in indexes:
                    try:
                        await conn.execute(index_sql)
                    except Exception as e:
                        # Index might already exist
                        logger.debug(f"Index creation skipped: {e}")
                
                logger.info("Database indexes created/verified")
                
        except Exception as e:
            logger.error(f"Error setting up database optimizations: {e}")
    
    async def _metrics_cleanup_loop(self):
        """Background task to clean up old metrics."""
        while self._running:
            try:
                # Clean up old metrics
                cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
                self.query_metrics = [
                    m for m in self.query_metrics
                    if m.timestamp > cutoff_time
                ]
                
                # Run database optimization periodically
                await self.optimize_database_performance()
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics cleanup loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry


# Global database optimizer instance
database_optimizer = DatabaseOptimizer("postgresql://user:password@localhost/visual_generation")