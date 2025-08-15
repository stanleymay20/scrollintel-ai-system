#!/usr/bin/env python3
"""
Create database migration for monitoring and analytics tables
"""

import asyncio
import os
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_database_url():
    """Get database URL from environment"""
    postgres_user = os.getenv('POSTGRES_USER', 'postgres')
    postgres_password = os.getenv('POSTGRES_PASSWORD', 'password')
    postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
    postgres_port = os.getenv('POSTGRES_PORT', '5432')
    postgres_db = os.getenv('POSTGRES_DB', 'scrollintel')
    
    print(f"Database connection details:")
    print(f"  Host: {postgres_host}")
    print(f"  Port: {postgres_port}")
    print(f"  Database: {postgres_db}")
    print(f"  User: {postgres_user}")
    print(f"  Password: {'*' * len(postgres_password) if postgres_password else 'None'}")
    
    return f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

async def create_monitoring_tables():
    """Create monitoring and analytics tables"""
    
    # SQL for creating monitoring tables
    monitoring_tables_sql = """
    -- User events table for analytics
    CREATE TABLE IF NOT EXISTS user_events (
        id SERIAL PRIMARY KEY,
        event_id VARCHAR(255) UNIQUE NOT NULL,
        user_id VARCHAR(255),
        session_id VARCHAR(255) NOT NULL,
        event_type VARCHAR(100) NOT NULL,
        event_name VARCHAR(255) NOT NULL,
        properties JSONB DEFAULT '{}',
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        ip_address INET,
        user_agent TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- System metrics table
    CREATE TABLE IF NOT EXISTS system_metrics (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        cpu_percent FLOAT NOT NULL,
        memory_percent FLOAT NOT NULL,
        disk_percent FLOAT NOT NULL,
        active_connections INTEGER DEFAULT 0,
        request_rate FLOAT DEFAULT 0,
        error_rate FLOAT DEFAULT 0,
        avg_response_time FLOAT DEFAULT 0,
        agent_count INTEGER DEFAULT 0,
        metadata JSONB DEFAULT '{}'
    );

    -- Alert history table
    CREATE TABLE IF NOT EXISTS alert_history (
        id SERIAL PRIMARY KEY,
        alert_id VARCHAR(255) NOT NULL,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        severity VARCHAR(50) NOT NULL,
        status VARCHAR(50) NOT NULL,
        metric_name VARCHAR(255) NOT NULL,
        current_value FLOAT NOT NULL,
        threshold FLOAT NOT NULL,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        resolved_at TIMESTAMP WITH TIME ZONE,
        acknowledged_at TIMESTAMP WITH TIME ZONE,
        acknowledged_by VARCHAR(255),
        tags JSONB DEFAULT '{}',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Performance logs table
    CREATE TABLE IF NOT EXISTS performance_logs (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        operation VARCHAR(255) NOT NULL,
        duration FLOAT NOT NULL,
        success BOOLEAN NOT NULL,
        user_id VARCHAR(255),
        agent_type VARCHAR(100),
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Agent metrics table
    CREATE TABLE IF NOT EXISTS agent_metrics (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        agent_type VARCHAR(100) NOT NULL,
        agent_id VARCHAR(255),
        requests_count INTEGER DEFAULT 0,
        success_count INTEGER DEFAULT 0,
        error_count INTEGER DEFAULT 0,
        avg_processing_time FLOAT DEFAULT 0,
        status VARCHAR(50) DEFAULT 'active',
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Database metrics table
    CREATE TABLE IF NOT EXISTS database_metrics (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        active_connections INTEGER NOT NULL,
        idle_connections INTEGER NOT NULL,
        total_connections INTEGER NOT NULL,
        max_connections INTEGER NOT NULL,
        database_size BIGINT NOT NULL,
        queries_per_second FLOAT DEFAULT 0,
        slow_queries INTEGER DEFAULT 0,
        cache_hit_ratio FLOAT DEFAULT 0,
        deadlocks INTEGER DEFAULT 0,
        temp_files INTEGER DEFAULT 0,
        temp_bytes BIGINT DEFAULT 0,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Create indexes for better performance
    CREATE INDEX IF NOT EXISTS idx_user_events_user_id ON user_events(user_id);
    CREATE INDEX IF NOT EXISTS idx_user_events_session_id ON user_events(session_id);
    CREATE INDEX IF NOT EXISTS idx_user_events_timestamp ON user_events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_user_events_event_type ON user_events(event_type);
    CREATE INDEX IF NOT EXISTS idx_user_events_event_name ON user_events(event_name);

    CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);
    CREATE INDEX IF NOT EXISTS idx_alert_history_timestamp ON alert_history(timestamp);
    CREATE INDEX IF NOT EXISTS idx_alert_history_status ON alert_history(status);
    CREATE INDEX IF NOT EXISTS idx_alert_history_severity ON alert_history(severity);

    CREATE INDEX IF NOT EXISTS idx_performance_logs_timestamp ON performance_logs(timestamp);
    CREATE INDEX IF NOT EXISTS idx_performance_logs_operation ON performance_logs(operation);
    CREATE INDEX IF NOT EXISTS idx_performance_logs_user_id ON performance_logs(user_id);

    CREATE INDEX IF NOT EXISTS idx_agent_metrics_timestamp ON agent_metrics(timestamp);
    CREATE INDEX IF NOT EXISTS idx_agent_metrics_agent_type ON agent_metrics(agent_type);

    CREATE INDEX IF NOT EXISTS idx_database_metrics_timestamp ON database_metrics(timestamp);
    """
    
    try:
        # Create async engine
        database_url = get_database_url()
        async_database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')
        
        engine = create_async_engine(async_database_url)
        
        async with engine.begin() as conn:
            # Execute the SQL to create tables
            await conn.execute(text(monitoring_tables_sql))
            
            print("✅ Monitoring tables created successfully!")
            
            # Verify tables were created
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('user_events', 'system_metrics', 'alert_history', 
                                  'performance_logs', 'agent_metrics', 'database_metrics')
                ORDER BY table_name
            """))
            
            tables = result.fetchall()
            print(f"📊 Created {len(tables)} monitoring tables:")
            for table in tables:
                print(f"   - {table[0]}")
                
        await engine.dispose()
                
    except Exception as e:
        print(f"❌ Error creating monitoring tables: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(create_monitoring_tables())