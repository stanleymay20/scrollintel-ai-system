#!/usr/bin/env python3
"""
Agent Steering System Database Migration

Creates the foundational database schemas for enterprise-grade agent orchestration
including agent registry, task management, and performance tracking tables.
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.models.agent_steering_models import Base
from scrollintel.core.config import get_settings

def create_agent_steering_tables():
    """Create all agent steering system tables"""
    try:
        # Use SQLite for development if PostgreSQL is not available
        try:
            settings = get_settings()
            database_url = settings.database_url
            # Test PostgreSQL connection
            test_engine = create_engine(database_url)
            test_engine.connect().close()
        except Exception as e:
            print(f"⚠️  PostgreSQL not available ({e}), using SQLite for development")
            database_url = "sqlite:///./scrollintel_dev.db"
        
        # Create engine
        engine = create_engine(database_url)
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        print("✅ Agent Steering System tables created successfully")
        
        # Create indexes for performance optimization
        with engine.connect() as conn:
            # Agent performance indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_agents_status_load 
                ON agents(status, current_load);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_agents_type_capabilities 
                ON agents(type) WHERE status = 'active';
            """))
            
            # Task performance indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_tasks_status_priority 
                ON tasks(status, priority);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_tasks_assigned_agent 
                ON tasks(assigned_agent_id) WHERE status IN ('pending', 'running');
            """))
            
            # Performance metrics indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_agent_performance_recorded_at 
                ON agent_performance_metrics(agent_id, recorded_at DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_task_performance_recorded_at 
                ON task_performance_metrics(task_id, recorded_at DESC);
            """))
            
            # System events indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_system_events_type_severity 
                ON system_events(event_type, severity, occurred_at DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_system_events_source_occurred 
                ON system_events(source, occurred_at DESC);
            """))
            
            conn.commit()
            
        print("✅ Performance indexes created successfully")
        
        # Insert initial system configuration
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Insert system events for migration
            from scrollintel.models.agent_steering_models import SystemEvent
            
            migration_event = SystemEvent(
                event_type="system_migration",
                severity="INFO",
                source="migration_script",
                message="Agent Steering System database migration completed",
                details={
                    "migration_timestamp": datetime.utcnow().isoformat(),
                    "tables_created": [
                        "agents",
                        "tasks", 
                        "agent_performance_metrics",
                        "task_performance_metrics",
                        "orchestration_sessions",
                        "system_events"
                    ],
                    "indexes_created": 8
                }
            )
            
            session.add(migration_event)
            session.commit()
            
            print("✅ Initial system configuration inserted")
            
        except Exception as e:
            session.rollback()
            print(f"⚠️  Warning: Could not insert initial configuration: {e}")
        finally:
            session.close()
            
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

def verify_migration():
    """Verify that all tables were created correctly"""
    try:
        # Use same logic as create_agent_steering_tables for database URL
        try:
            settings = get_settings()
            database_url = settings.database_url
            # Test PostgreSQL connection
            test_engine = create_engine(database_url)
            test_engine.connect().close()
        except Exception:
            database_url = "sqlite:///./scrollintel_dev.db"
        engine = create_engine(database_url)
        
        # Check if all tables exist (works for both PostgreSQL and SQLite)
        with engine.connect() as conn:
            if "sqlite" in database_url:
                result = conn.execute(text("""
                    SELECT name as table_name 
                    FROM sqlite_master 
                    WHERE type='table' 
                    AND name IN (
                        'agents', 'tasks', 'agent_performance_metrics',
                        'task_performance_metrics', 'orchestration_sessions', 'system_events'
                    )
                    ORDER BY name;
                """))
            else:
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN (
                        'agents', 'tasks', 'agent_performance_metrics',
                        'task_performance_metrics', 'orchestration_sessions', 'system_events'
                    )
                    ORDER BY table_name;
                """))
            
            tables = [row[0] for row in result]
            expected_tables = [
                'agent_performance_metrics',
                'agents', 
                'orchestration_sessions',
                'system_events',
                'task_performance_metrics',
                'tasks'
            ]
            
            missing_tables = set(expected_tables) - set(tables)
            
            if missing_tables:
                print(f"❌ Missing tables: {missing_tables}")
                return False
            
            print(f"✅ All {len(tables)} tables verified successfully")
            
            # Check table row counts
            for table in tables:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                print(f"   - {table}: {count} rows")
            
            return True
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Agent Steering System Database Migration...")
    print("=" * 60)
    
    # Create tables
    if create_agent_steering_tables():
        print("\n🔍 Verifying migration...")
        if verify_migration():
            print("\n✅ Agent Steering System migration completed successfully!")
            print("\nNext steps:")
            print("1. Start the real-time messaging system")
            print("2. Initialize the agent registry")
            print("3. Configure secure communication protocols")
            print("4. Begin agent registration")
        else:
            print("\n❌ Migration verification failed!")
            sys.exit(1)
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)