#!/usr/bin/env python3
"""
Database Optimization Script
Analyzes database performance and applies optimizations
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from scrollintel.core.config import get_settings
from scrollintel.core.database_optimizer import database_optimizer
from scrollintel.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

async def create_database_indexes():
    """Create recommended database indexes"""
    
    # Common indexes for ScrollIntel tables
    indexes = [
        # User events table indexes
        """
        CREATE INDEX IF NOT EXISTS idx_user_events_user_id_timestamp 
        ON user_events (user_id, timestamp DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_user_events_session_id 
        ON user_events (session_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_user_events_event_type_timestamp 
        ON user_events (event_type, timestamp DESC);
        """,
        
        # Performance metrics indexes
        """
        CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp 
        ON performance_metrics (timestamp DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_performance_metrics_endpoint_method 
        ON performance_metrics (endpoint, method);
        """,
        
        # Database query metrics indexes
        """
        CREATE INDEX IF NOT EXISTS idx_db_query_metrics_query_hash 
        ON db_query_metrics (query_hash);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_db_query_metrics_timestamp 
        ON db_query_metrics (timestamp DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_db_query_metrics_execution_time 
        ON db_query_metrics (execution_time DESC);
        """,
        
        # Cache metrics indexes
        """
        CREATE INDEX IF NOT EXISTS idx_cache_metrics_timestamp 
        ON cache_metrics (timestamp DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cache_metrics_operation 
        ON cache_metrics (operation);
        """,
        
        # Agent requests indexes
        """
        CREATE INDEX IF NOT EXISTS idx_agent_requests_user_id_timestamp 
        ON agent_requests (user_id, timestamp DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_agent_requests_agent_type 
        ON agent_requests (agent_type);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_agent_requests_status 
        ON agent_requests (status);
        """,
        
        # File uploads indexes
        """
        CREATE INDEX IF NOT EXISTS idx_file_uploads_user_id_timestamp 
        ON file_uploads (user_id, timestamp DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_file_uploads_status 
        ON file_uploads (status);
        """,
        
        # Projects and workspaces indexes
        """
        CREATE INDEX IF NOT EXISTS idx_projects_user_id 
        ON projects (user_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_projects_created_at 
        ON projects (created_at DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_workspaces_organization_id 
        ON workspaces (organization_id);
        """,
        
        # Audit logs indexes
        """
        CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id_timestamp 
        ON audit_logs (user_id, timestamp DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_audit_logs_action 
        ON audit_logs (action);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_audit_logs_resource_type 
        ON audit_logs (resource_type);
        """
    ]
    
    try:
        # Create async engine
        engine = create_async_engine(settings.DATABASE_URL)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with async_session() as session:
            logger.info("Creating database indexes for performance optimization...")
            
            for i, index_sql in enumerate(indexes, 1):
                try:
                    await session.execute(text(index_sql))
                    logger.info(f"Created index {i}/{len(indexes)}")
                except Exception as e:
                    logger.warning(f"Failed to create index {i}: {e}")
                    
            await session.commit()
            logger.info("Database index creation completed")
            
    except Exception as e:
        logger.error(f"Error creating database indexes: {e}")
        raise

async def analyze_query_performance():
    """Analyze current query performance and generate recommendations"""
    
    try:
        engine = create_async_engine(settings.DATABASE_URL)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with async_session() as session:
            logger.info("Analyzing database query performance...")
            
            # Generate optimization report
            report = await database_optimizer.generate_optimization_report(session)
            
            if "error" in report:
                logger.error(f"Error generating optimization report: {report['error']}")
                return
                
            # Print summary
            summary = report.get("summary", {})
            logger.info(f"Performance Analysis Summary:")
            logger.info(f"  Total slow queries: {summary.get('total_slow_queries', 0)}")
            logger.info(f"  High priority queries: {summary.get('high_priority_queries', 0)}")
            logger.info(f"  Index recommendations: {summary.get('index_recommendations', 0)}")
            logger.info(f"  Average query time: {summary.get('avg_query_time', 0):.3f}s")
            
            # Print top recommendations
            optimizations = report.get("query_optimizations", [])
            if optimizations:
                logger.info("\nTop Query Optimizations:")
                for i, opt in enumerate(optimizations[:5], 1):
                    logger.info(f"  {i}. Query {opt['query_hash'][:8]}...")
                    for suggestion in opt["recommendations"][:2]:
                        logger.info(f"     - {suggestion}")
                        
            # Print index recommendations
            indexes = report.get("index_recommendations", [])
            if indexes:
                logger.info("\nIndex Recommendations:")
                for i, idx in enumerate(indexes[:5], 1):
                    logger.info(f"  {i}. {idx['table_name']}.{', '.join(idx['columns'])}")
                    logger.info(f"     Reason: {idx['reason']}")
                    logger.info(f"     SQL: {idx['sql_command']}")
                    
            # Save report to file
            import json
            report_file = project_root / "database_optimization_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Detailed report saved to: {report_file}")
            
    except Exception as e:
        logger.error(f"Error analyzing query performance: {e}")
        raise

async def optimize_database_settings():
    """Apply database-level optimizations"""
    
    try:
        engine = create_async_engine(settings.DATABASE_URL)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with async_session() as session:
            logger.info("Applying database configuration optimizations...")
            
            # PostgreSQL optimization settings
            optimizations = [
                # Increase shared buffers for better caching
                "ALTER SYSTEM SET shared_buffers = '256MB';",
                
                # Optimize checkpoint settings
                "ALTER SYSTEM SET checkpoint_completion_target = 0.9;",
                "ALTER SYSTEM SET wal_buffers = '16MB';",
                
                # Optimize query planner
                "ALTER SYSTEM SET random_page_cost = 1.1;",
                "ALTER SYSTEM SET effective_cache_size = '1GB';",
                
                # Optimize connection settings
                "ALTER SYSTEM SET max_connections = 200;",
                
                # Enable query statistics
                "ALTER SYSTEM SET track_activities = on;",
                "ALTER SYSTEM SET track_counts = on;",
                "ALTER SYSTEM SET track_io_timing = on;",
                "ALTER SYSTEM SET track_functions = 'all';",
                
                # Optimize logging for performance monitoring
                "ALTER SYSTEM SET log_min_duration_statement = 1000;",  # Log queries > 1s
                "ALTER SYSTEM SET log_checkpoints = on;",
                "ALTER SYSTEM SET log_connections = on;",
                "ALTER SYSTEM SET log_disconnections = on;",
                "ALTER SYSTEM SET log_lock_waits = on;",
            ]
            
            for optimization in optimizations:
                try:
                    await session.execute(text(optimization))
                    logger.info(f"Applied: {optimization}")
                except Exception as e:
                    logger.warning(f"Failed to apply optimization: {e}")
                    
            await session.commit()
            
            # Reload configuration
            await session.execute(text("SELECT pg_reload_conf();"))
            await session.commit()
            
            logger.info("Database optimization settings applied")
            logger.info("Note: Some settings may require a database restart to take effect")
            
    except Exception as e:
        logger.error(f"Error applying database optimizations: {e}")
        raise

async def vacuum_and_analyze():
    """Run VACUUM and ANALYZE on all tables"""
    
    try:
        engine = create_async_engine(settings.DATABASE_URL)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with async_session() as session:
            logger.info("Running VACUUM and ANALYZE on database tables...")
            
            # Get all table names
            result = await session.execute(text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
            """))
            tables = [row[0] for row in result.fetchall()]
            
            for table in tables:
                try:
                    # VACUUM and ANALYZE each table
                    await session.execute(text(f"VACUUM ANALYZE {table};"))
                    logger.info(f"Vacuumed and analyzed table: {table}")
                except Exception as e:
                    logger.warning(f"Failed to vacuum table {table}: {e}")
                    
            logger.info("Database maintenance completed")
            
    except Exception as e:
        logger.error(f"Error during database maintenance: {e}")
        raise

async def main():
    """Main optimization function"""
    
    if len(sys.argv) < 2:
        print("Usage: python optimize-database.py <command>")
        print("Commands:")
        print("  indexes    - Create performance indexes")
        print("  analyze    - Analyze query performance")
        print("  settings   - Optimize database settings")
        print("  vacuum     - Run VACUUM and ANALYZE")
        print("  all        - Run all optimizations")
        return
        
    command = sys.argv[1].lower()
    
    try:
        if command == "indexes":
            await create_database_indexes()
        elif command == "analyze":
            await analyze_query_performance()
        elif command == "settings":
            await optimize_database_settings()
        elif command == "vacuum":
            await vacuum_and_analyze()
        elif command == "all":
            logger.info("Running comprehensive database optimization...")
            await create_database_indexes()
            await optimize_database_settings()
            await vacuum_and_analyze()
            await analyze_query_performance()
            logger.info("Database optimization completed successfully!")
        else:
            print(f"Unknown command: {command}")
            return
            
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())