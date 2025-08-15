"""
Database Optimizer for ScrollIntel.
Provides database performance optimization, index recommendations, and maintenance tasks.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import AsyncSession

from .database_pool import get_optimized_db_pool
from .config import get_settings
from ..models.database import Base

logger = logging.getLogger(__name__)


@dataclass
class TableStatistics:
    """Statistics for a database table."""
    table_name: str
    schema_name: str
    row_count: int
    table_size_bytes: int
    index_size_bytes: int
    vacuum_needed: bool
    analyze_needed: bool
    last_vacuum: Optional[datetime]
    last_analyze: Optional[datetime]
    bloat_ratio: float
    seq_scan_count: int
    seq_tup_read: int
    idx_scan_count: int
    idx_tup_fetch: int


@dataclass
class IndexRecommendation:
    """Recommendation for creating a database index."""
    table_name: str
    column_names: List[str]
    index_type: str
    create_statement: str
    estimated_benefit: float
    reason: str
    priority: str


@dataclass
class OptimizationResult:
    """Result of database optimization operation."""
    operation: str
    success: bool
    execution_time: float
    details: Dict[str, Any]
    error: Optional[str] = None


class DatabaseOptimizer:
    """Database performance optimizer with maintenance and monitoring."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logger
        
    async def initialize(self) -> None:
        """Initialize the database optimizer."""
        try:
            # Test database connection
            db_pool = await get_optimized_db_pool()
            async with db_pool.get_async_session() as session:
                await session.execute(text("SELECT 1"))
            
            self.logger.info("Database optimizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database optimizer: {e}")
            raise  
  
    async def _analyze_table_statistics(self) -> List[TableStatistics]:
        """Analyze database table statistics."""
        
        db_pool = await get_optimized_db_pool()
        table_stats = []
        
        async with db_pool.get_async_session() as session:
            # Get table statistics from PostgreSQL system catalogs
            query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins + n_tup_upd + n_tup_del as total_writes,
                    n_tup_ins,
                    n_tup_upd,
                    n_tup_del,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    idx_tup_fetch,
                    n_dead_tup,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze
                FROM pg_stat_user_tables
                ORDER BY schemaname, tablename
            """)
            
            result = await session.execute(query)
            stats_rows = result.fetchall()
            
            for row in stats_rows:
                # Get table size information
                size_query = text("""
                    SELECT 
                        pg_total_relation_size(:table_name) as total_size,
                        pg_relation_size(:table_name) as table_size,
                        pg_total_relation_size(:table_name) - pg_relation_size(:table_name) as index_size,
                        (SELECT reltuples::bigint FROM pg_class WHERE relname = :table_only) as row_count
                """)
                
                table_full_name = f"{row.schemaname}.{row.tablename}"
                size_result = await session.execute(
                    size_query, 
                    {"table_name": table_full_name, "table_only": row.tablename}
                )
                size_row = size_result.fetchone()
                
                # Calculate bloat ratio (simplified)
                bloat_ratio = 0.0
                if row.n_dead_tup and size_row.row_count:
                    bloat_ratio = row.n_dead_tup / max(size_row.row_count, 1)
                
                # Determine if vacuum/analyze is needed
                vacuum_needed = (
                    bloat_ratio > 0.1 or  # More than 10% dead tuples
                    row.n_dead_tup > 1000 or  # More than 1000 dead tuples
                    (row.last_vacuum is None and row.last_autovacuum is None)
                )
                
                analyze_needed = (
                    row.total_writes > 1000 or  # Significant writes since last analyze
                    (row.last_analyze is None and row.last_autoanalyze is None)
                )
                
                table_stat = TableStatistics(
                    table_name=row.tablename,
                    schema_name=row.schemaname,
                    row_count=int(size_row.row_count or 0),
                    table_size_bytes=int(size_row.table_size or 0),
                    index_size_bytes=int(size_row.index_size or 0),
                    vacuum_needed=vacuum_needed,
                    analyze_needed=analyze_needed,
                    last_vacuum=row.last_vacuum or row.last_autovacuum,
                    last_analyze=row.last_analyze or row.last_autoanalyze,
                    bloat_ratio=bloat_ratio,
                    seq_scan_count=int(row.seq_scan or 0),
                    seq_tup_read=int(row.seq_tup_read or 0),
                    idx_scan_count=int(row.idx_scan or 0),
                    idx_tup_fetch=int(row.idx_tup_fetch or 0)
                )
                
                table_stats.append(table_stat)
        
        return table_stats
    
    async def _generate_index_recommendations(self, table_stats: List[TableStatistics]) -> List[IndexRecommendation]:
        """Generate index recommendations based on table statistics."""
        
        recommendations = []
        db_pool = await get_optimized_db_pool()
        
        async with db_pool.get_async_session() as session:
            for table_stat in table_stats:
                # Skip small tables
                if table_stat.row_count < 1000:
                    continue
                
                # Check for tables with high sequential scan ratio
                if table_stat.seq_scan_count > 0 and table_stat.idx_scan_count > 0:
                    seq_ratio = table_stat.seq_scan_count / (table_stat.seq_scan_count + table_stat.idx_scan_count)
                    
                    if seq_ratio > 0.8:  # More than 80% sequential scans
                        # Get column statistics to suggest indexes
                        column_query = text("""
                            SELECT 
                                column_name,
                                data_type,
                                is_nullable
                            FROM information_schema.columns 
                            WHERE table_schema = :schema_name 
                            AND table_name = :table_name
                            ORDER BY ordinal_position
                        """)
                        
                        column_result = await session.execute(
                            column_query,
                            {"schema_name": table_stat.schema_name, "table_name": table_stat.table_name}
                        )
                        columns = column_result.fetchall()
                        
                        # Suggest indexes for commonly filtered columns
                        for column in columns:
                            if column.column_name in ['id', 'user_id', 'created_at', 'updated_at', 'status']:
                                recommendation = IndexRecommendation(
                                    table_name=table_stat.table_name,
                                    column_names=[column.column_name],
                                    index_type='btree',
                                    create_statement=f"CREATE INDEX CONCURRENTLY idx_{table_stat.table_name}_{column.column_name} ON {table_stat.schema_name}.{table_stat.table_name} ({column.column_name});",
                                    estimated_benefit=seq_ratio * 100,
                                    reason=f"High sequential scan ratio ({seq_ratio:.1%}) suggests missing index",
                                    priority='high' if seq_ratio > 0.9 else 'medium'
                                )
                                recommendations.append(recommendation)
        
        return recommendations    
    
    async def run_maintenance_tasks(self) -> List[OptimizationResult]:
        """Run database maintenance tasks."""
        
        results = []
        table_stats = await self._analyze_table_statistics()
        
        db_pool = await get_optimized_db_pool()
        
        for table_stat in table_stats:
            # Run VACUUM if needed
            if table_stat.vacuum_needed:
                start_time = time.time()
                try:
                    # Use a separate connection for VACUUM (can't run in transaction)
                    engine = db_pool.engine
                    with engine.connect() as conn:
                        conn.execute(text(f"VACUUM {table_stat.schema_name}.{table_stat.table_name}"))
                        conn.commit()
                    
                    execution_time = time.time() - start_time
                    results.append(OptimizationResult(
                        operation=f"VACUUM {table_stat.table_name}",
                        success=True,
                        execution_time=execution_time,
                        details={
                            "table": table_stat.table_name,
                            "bloat_ratio_before": table_stat.bloat_ratio,
                            "dead_tuples": table_stat.seq_tup_read
                        }
                    ))
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    results.append(OptimizationResult(
                        operation=f"VACUUM {table_stat.table_name}",
                        success=False,
                        execution_time=execution_time,
                        details={"table": table_stat.table_name},
                        error=str(e)
                    ))
            
            # Run ANALYZE if needed
            if table_stat.analyze_needed:
                start_time = time.time()
                try:
                    async with db_pool.get_async_session() as session:
                        await session.execute(text(f"ANALYZE {table_stat.schema_name}.{table_stat.table_name}"))
                        await session.commit()
                    
                    execution_time = time.time() - start_time
                    results.append(OptimizationResult(
                        operation=f"ANALYZE {table_stat.table_name}",
                        success=True,
                        execution_time=execution_time,
                        details={"table": table_stat.table_name}
                    ))
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    results.append(OptimizationResult(
                        operation=f"ANALYZE {table_stat.table_name}",
                        success=False,
                        execution_time=execution_time,
                        details={"table": table_stat.table_name},
                        error=str(e)
                    ))
        
        return results
    
    async def get_database_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive database health report."""
        
        start_time = time.time()
        
        try:
            # Get table statistics
            table_stats = await self._analyze_table_statistics()
            
            # Get database size information
            db_pool = await get_optimized_db_pool()
            async with db_pool.get_async_session() as session:
                # Database size
                db_size_query = text("SELECT pg_database_size(current_database()) as db_size")
                db_size_result = await session.execute(db_size_query)
                db_size = db_size_result.fetchone().db_size
                
                # Connection statistics
                conn_stats_query = text("""
                    SELECT 
                        count(*) as total_connections,
                        count(*) FILTER (WHERE state = 'active') as active_connections,
                        count(*) FILTER (WHERE state = 'idle') as idle_connections
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                """)
                conn_stats_result = await session.execute(conn_stats_query)
                conn_stats = conn_stats_result.fetchone()
                
                # Lock statistics
                lock_stats_query = text("""
                    SELECT 
                        mode,
                        count(*) as lock_count
                    FROM pg_locks 
                    WHERE database = (SELECT oid FROM pg_database WHERE datname = current_database())
                    GROUP BY mode
                """)
                lock_stats_result = await session.execute(lock_stats_query)
                lock_stats = {row.mode: row.lock_count for row in lock_stats_result.fetchall()}
            
            # Calculate health metrics
            total_tables = len(table_stats)
            tables_needing_vacuum = sum(1 for t in table_stats if t.vacuum_needed)
            tables_needing_analyze = sum(1 for t in table_stats if t.analyze_needed)
            
            # Calculate health score (0-100)
            health_score = 100.0
            
            # Penalize tables needing maintenance
            if total_tables > 0:
                vacuum_penalty = (tables_needing_vacuum / total_tables) * 20
                analyze_penalty = (tables_needing_analyze / total_tables) * 15
                health_score -= (vacuum_penalty + analyze_penalty)
            
            # Penalize high connection usage
            max_connections = 100  # Default PostgreSQL max_connections
            if conn_stats.total_connections > max_connections * 0.8:
                health_score -= 15
            
            # Penalize excessive locks
            total_locks = sum(lock_stats.values())
            if total_locks > 50:
                health_score -= 10
            
            health_score = max(0.0, min(100.0, health_score))
            
            # Determine overall health status
            if health_score >= 90:
                overall_health = "excellent"
            elif health_score >= 75:
                overall_health = "good"
            elif health_score >= 60:
                overall_health = "fair"
            elif health_score >= 40:
                overall_health = "poor"
            else:
                overall_health = "critical"
            
            execution_time = time.time() - start_time
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "execution_time": execution_time,
                "overall_health": overall_health,
                "health_score": health_score,
                "metrics": {
                    "database_size_bytes": db_size,
                    "database_size_mb": round(db_size / (1024 * 1024), 2),
                    "total_tables": total_tables,
                    "tables_needing_vacuum": tables_needing_vacuum,
                    "tables_needing_analyze": tables_needing_analyze,
                    "total_connections": conn_stats.total_connections,
                    "active_connections": conn_stats.active_connections,
                    "idle_connections": conn_stats.idle_connections,
                    "lock_statistics": lock_stats
                },
                "table_statistics": [
                    {
                        "table_name": t.table_name,
                        "row_count": t.row_count,
                        "table_size_mb": round(t.table_size_bytes / (1024 * 1024), 2),
                        "vacuum_needed": t.vacuum_needed,
                        "analyze_needed": t.analyze_needed,
                        "bloat_ratio": round(t.bloat_ratio, 3)
                    }
                    for t in table_stats
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate database health report: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_health": "unknown",
                "error": str(e)
            } 
   
    async def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """Run comprehensive database optimization."""
        
        start_time = time.time()
        
        try:
            # Get table statistics
            self.logger.info("Analyzing table statistics...")
            table_stats = await self._analyze_table_statistics()
            
            # Generate index recommendations
            self.logger.info("Generating index recommendations...")
            index_recommendations = await self._generate_index_recommendations(table_stats)
            
            # Run maintenance tasks
            self.logger.info("Running maintenance tasks...")
            maintenance_results = await self.run_maintenance_tasks()
            
            # Update table statistics after maintenance
            self.logger.info("Updating table statistics...")
            db_pool = await get_optimized_db_pool()
            async with db_pool.get_async_session() as session:
                await session.execute(text("ANALYZE"))
                await session.commit()
            
            execution_time = time.time() - start_time
            
            return {
                "started_at": datetime.utcnow().isoformat(),
                "status": "completed",
                "execution_time": execution_time,
                "table_stats": {
                    "total_tables": len(table_stats),
                    "tables_analyzed": len([t for t in table_stats if not t.analyze_needed]),
                    "tables_vacuumed": len([t for t in table_stats if not t.vacuum_needed])
                },
                "index_recommendations": [
                    {
                        "table": rec.table_name,
                        "columns": rec.column_names,
                        "type": rec.index_type,
                        "priority": rec.priority,
                        "reason": rec.reason,
                        "create_statement": rec.create_statement
                    }
                    for rec in index_recommendations
                ],
                "maintenance": {
                    "total_operations": len(maintenance_results),
                    "successful_operations": len([r for r in maintenance_results if r.success]),
                    "failed_operations": len([r for r in maintenance_results if not r.success]),
                    "operations": [
                        {
                            "operation": r.operation,
                            "success": r.success,
                            "execution_time": r.execution_time,
                            "error": r.error
                        }
                        for r in maintenance_results
                    ]
                },
                "statistics_update": {
                    "status": "completed",
                    "message": "Database statistics updated successfully"
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Comprehensive optimization failed: {e}")
            
            return {
                "started_at": datetime.utcnow().isoformat(),
                "status": "failed",
                "execution_time": execution_time,
                "error": str(e)
            }
    
    async def optimize_query_performance(self, query: str) -> Dict[str, Any]:
        """Analyze and optimize a specific query."""
        
        db_pool = await get_optimized_db_pool()
        
        async with db_pool.get_async_session() as session:
            try:
                # Get query execution plan
                explain_query = text(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}")
                result = await session.execute(explain_query)
                plan_data = result.fetchone()[0]
                
                # Extract performance metrics
                plan = plan_data[0]['Plan']
                execution_time = plan_data[0]['Execution Time']
                planning_time = plan_data[0]['Planning Time']
                
                # Analyze plan for optimization opportunities
                suggestions = []
                
                def analyze_plan_node(node, depth=0):
                    node_type = node.get('Node Type', '')
                    
                    # Check for sequential scans on large tables
                    if node_type == 'Seq Scan':
                        rows = node.get('Actual Rows', 0)
                        if rows > 1000:
                            suggestions.append({
                                "type": "missing_index",
                                "message": f"Sequential scan on {node.get('Relation Name', 'unknown')} table with {rows} rows",
                                "recommendation": "Consider adding an index on the filtered columns"
                            })
                    
                    # Check for expensive sorts
                    if node_type == 'Sort' and node.get('Actual Total Time', 0) > 100:
                        suggestions.append({
                            "type": "expensive_sort",
                            "message": f"Expensive sort operation taking {node.get('Actual Total Time', 0):.2f}ms",
                            "recommendation": "Consider adding an index to avoid sorting"
                        })
                    
                    # Check for nested loops with high cost
                    if node_type == 'Nested Loop' and node.get('Total Cost', 0) > 1000:
                        suggestions.append({
                            "type": "expensive_join",
                            "message": f"Expensive nested loop join with cost {node.get('Total Cost', 0):.2f}",
                            "recommendation": "Consider adding indexes on join columns or using hash/merge join"
                        })
                    
                    # Recursively analyze child nodes
                    for child in node.get('Plans', []):
                        analyze_plan_node(child, depth + 1)
                
                analyze_plan_node(plan)
                
                return {
                    "query": query[:200] + "..." if len(query) > 200 else query,
                    "execution_time_ms": execution_time,
                    "planning_time_ms": planning_time,
                    "total_time_ms": execution_time + planning_time,
                    "execution_plan": plan_data,
                    "optimization_suggestions": suggestions,
                    "performance_rating": self._rate_query_performance(execution_time, suggestions)
                }
                
            except Exception as e:
                return {
                    "query": query[:200] + "..." if len(query) > 200 else query,
                    "error": str(e),
                    "optimization_suggestions": [
                        {
                            "type": "analysis_failed",
                            "message": f"Could not analyze query: {str(e)}",
                            "recommendation": "Check query syntax and permissions"
                        }
                    ]
                }
    
    def _rate_query_performance(self, execution_time: float, suggestions: List[Dict]) -> str:
        """Rate query performance based on execution time and suggestions."""
        
        if execution_time < 10:  # Less than 10ms
            return "excellent"
        elif execution_time < 100:  # Less than 100ms
            return "good" if len(suggestions) == 0 else "fair"
        elif execution_time < 1000:  # Less than 1 second
            return "fair" if len(suggestions) <= 2 else "poor"
        else:  # More than 1 second
            return "poor"


# Global database optimizer instance
_database_optimizer: Optional[DatabaseOptimizer] = None


async def get_database_optimizer() -> DatabaseOptimizer:
    """Get the global database optimizer instance."""
    global _database_optimizer
    
    if _database_optimizer is None:
        _database_optimizer = DatabaseOptimizer()
        await _database_optimizer.initialize()
    
    return _database_optimizer