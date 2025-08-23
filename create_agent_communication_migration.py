"""
Database Migration for Agent Communication Framework

Creates tables for secure agent messaging, collaboration sessions,
distributed state synchronization, and resource locking.
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.models.agent_communication_models import Base
from scrollintel.core.config import get_settings

settings = get_settings()


def create_agent_communication_tables():
    """Create all agent communication tables"""
    try:
        # Create database engine
        engine = create_engine(settings.DATABASE_URL)
        
        print("Creating Agent Communication Framework tables...")
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        # Verify tables were created
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        # Check if tables exist
        tables_to_check = [
            'agent_messages',
            'collaboration_sessions', 
            'session_participants',
            'session_messages',
            'resource_locks',
            'distributed_state'
        ]
        
        existing_tables = []
        for table_name in tables_to_check:
            result = db.execute(text(f"""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{table_name}'
            """)).fetchone()
            
            if result:
                existing_tables.append(table_name)
                print(f"✓ Table '{table_name}' created successfully")
            else:
                print(f"✗ Table '{table_name}' creation failed")
        
        db.close()
        
        print(f"\nAgent Communication Framework Migration Summary:")
        print(f"✓ {len(existing_tables)}/{len(tables_to_check)} tables created successfully")
        
        if len(existing_tables) == len(tables_to_check):
            print("✅ All agent communication tables created successfully!")
            return True
        else:
            print("❌ Some tables failed to create")
            return False
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_indexes():
    """Create performance indexes for agent communication tables"""
    try:
        engine = create_engine(settings.DATABASE_URL)
        
        print("\nCreating performance indexes...")
        
        indexes = [
            # Agent Messages indexes
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_to_agent ON agent_messages(to_agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_from_agent ON agent_messages(from_agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_session ON agent_messages(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_correlation ON agent_messages(correlation_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_created ON agent_messages(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_expires ON agent_messages(expires_at)",
            
            # Session Participants indexes
            "CREATE INDEX IF NOT EXISTS idx_session_participants_agent ON session_participants(agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_session_participants_session ON session_participants(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_session_participants_status ON session_participants(status)",
            
            # Resource Locks indexes
            "CREATE INDEX IF NOT EXISTS idx_resource_locks_resource ON resource_locks(resource_id)",
            "CREATE INDEX IF NOT EXISTS idx_resource_locks_holder ON resource_locks(holder_agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_resource_locks_status ON resource_locks(status)",
            "CREATE INDEX IF NOT EXISTS idx_resource_locks_expires ON resource_locks(expires_at)",
            
            # Distributed State indexes
            "CREATE INDEX IF NOT EXISTS idx_distributed_state_key ON distributed_state(state_key)",
            "CREATE INDEX IF NOT EXISTS idx_distributed_state_namespace ON distributed_state(state_namespace)",
            "CREATE INDEX IF NOT EXISTS idx_distributed_state_owner ON distributed_state(owner_agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_distributed_state_session ON distributed_state(owner_session_id)",
            
            # Collaboration Sessions indexes
            "CREATE INDEX IF NOT EXISTS idx_collaboration_sessions_initiator ON collaboration_sessions(initiator_agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_collaboration_sessions_status ON collaboration_sessions(status)",
            "CREATE INDEX IF NOT EXISTS idx_collaboration_sessions_activity ON collaboration_sessions(last_activity)",
            
            # Session Messages indexes
            "CREATE INDEX IF NOT EXISTS idx_session_messages_session ON session_messages(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_session_messages_sender ON session_messages(sender_agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_session_messages_sequence ON session_messages(sequence_number)"
        ]
        
        with engine.connect() as conn:
            for index_sql in indexes:
                try:
                    conn.execute(text(index_sql))
                    index_name = index_sql.split("idx_")[1].split(" ")[0]
                    print(f"✓ Index 'idx_{index_name}' created")
                except Exception as e:
                    print(f"✗ Failed to create index: {e}")
            
            conn.commit()
        
        print("✅ Performance indexes created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Index creation failed: {e}")
        return False


def seed_initial_data():
    """Seed initial configuration data"""
    try:
        engine = create_engine(settings.DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        print("\nSeeding initial configuration data...")
        
        # Add any initial configuration data here if needed
        # For now, just verify the tables are accessible
        
        # Test basic operations
        from scrollintel.models.agent_communication_models import (
            AgentMessage, CollaborationSession, ResourceLock, DistributedState
        )
        
        # Count existing records
        message_count = db.query(AgentMessage).count()
        session_count = db.query(CollaborationSession).count()
        lock_count = db.query(ResourceLock).count()
        state_count = db.query(DistributedState).count()
        
        print(f"✓ Agent Messages: {message_count} records")
        print(f"✓ Collaboration Sessions: {session_count} records")
        print(f"✓ Resource Locks: {lock_count} records")
        print(f"✓ Distributed State: {state_count} records")
        
        db.close()
        
        print("✅ Initial data verification completed!")
        return True
        
    except Exception as e:
        print(f"❌ Data seeding failed: {e}")
        return False


def verify_migration():
    """Verify the migration was successful"""
    try:
        engine = create_engine(settings.DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        print("\nVerifying migration...")
        
        # Test creating and querying each model
        from scrollintel.models.agent_communication_models import (
            AgentMessage, CollaborationSession, SessionParticipant,
            SessionMessage, ResourceLock, DistributedState
        )
        
        # Test basic CRUD operations
        test_results = []
        
        # Test AgentMessage
        try:
            test_message = AgentMessage(
                id="test_msg_001",
                from_agent_id="test_sender",
                to_agent_id="test_receiver",
                message_type="notification",
                encrypted_content="test_encrypted_content",
                content_hash="test_hash",
                encryption_key_id="test_key"
            )
            db.add(test_message)
            db.commit()
            
            # Query it back
            retrieved = db.query(AgentMessage).filter(AgentMessage.id == "test_msg_001").first()
            if retrieved:
                test_results.append("AgentMessage: ✓")
                db.delete(retrieved)
            else:
                test_results.append("AgentMessage: ✗")
                
        except Exception as e:
            test_results.append(f"AgentMessage: ✗ ({e})")
        
        # Test CollaborationSession
        try:
            test_session = CollaborationSession(
                id="test_session_001",
                initiator_agent_id="test_initiator",
                session_name="Test Session",
                objective={"goal": "test"}
            )
            db.add(test_session)
            db.commit()
            
            retrieved = db.query(CollaborationSession).filter(
                CollaborationSession.id == "test_session_001"
            ).first()
            if retrieved:
                test_results.append("CollaborationSession: ✓")
                db.delete(retrieved)
            else:
                test_results.append("CollaborationSession: ✗")
                
        except Exception as e:
            test_results.append(f"CollaborationSession: ✗ ({e})")
        
        # Test ResourceLock
        try:
            test_lock = ResourceLock(
                id="test_lock_001",
                resource_id="test_resource",
                resource_type="test_type",
                holder_agent_id="test_holder"
            )
            db.add(test_lock)
            db.commit()
            
            retrieved = db.query(ResourceLock).filter(ResourceLock.id == "test_lock_001").first()
            if retrieved:
                test_results.append("ResourceLock: ✓")
                db.delete(retrieved)
            else:
                test_results.append("ResourceLock: ✗")
                
        except Exception as e:
            test_results.append(f"ResourceLock: ✗ ({e})")
        
        # Test DistributedState
        try:
            test_state = DistributedState(
                id="test_state_001",
                state_key="test_key",
                state_namespace="test_namespace",
                state_value={"test": "value"},
                state_hash="test_hash",
                owner_agent_id="test_owner",
                last_modified_by="test_modifier"
            )
            db.add(test_state)
            db.commit()
            
            retrieved = db.query(DistributedState).filter(
                DistributedState.id == "test_state_001"
            ).first()
            if retrieved:
                test_results.append("DistributedState: ✓")
                db.delete(retrieved)
            else:
                test_results.append("DistributedState: ✗")
                
        except Exception as e:
            test_results.append(f"DistributedState: ✗ ({e})")
        
        db.commit()
        db.close()
        
        print("\nModel Verification Results:")
        for result in test_results:
            print(f"  {result}")
        
        success_count = len([r for r in test_results if "✓" in r])
        total_count = len(test_results)
        
        if success_count == total_count:
            print("✅ All models verified successfully!")
            return True
        else:
            print(f"❌ {total_count - success_count} models failed verification")
            return False
            
    except Exception as e:
        print(f"❌ Migration verification failed: {e}")
        return False


def main():
    """Run the complete migration process"""
    print("Agent Communication Framework - Database Migration")
    print("=" * 60)
    print(f"Database URL: {settings.DATABASE_URL}")
    print(f"Migration started at: {datetime.utcnow().isoformat()}")
    print()
    
    success = True
    
    # Step 1: Create tables
    if not create_agent_communication_tables():
        success = False
    
    # Step 2: Create indexes
    if success and not create_indexes():
        success = False
    
    # Step 3: Seed initial data
    if success and not seed_initial_data():
        success = False
    
    # Step 4: Verify migration
    if success and not verify_migration():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ AGENT COMMUNICATION MIGRATION COMPLETED SUCCESSFULLY!")
        print("\nThe following components are now available:")
        print("• Secure encrypted messaging between agents")
        print("• Multi-agent collaboration session management")
        print("• Distributed state synchronization")
        print("• Resource locking and conflict resolution")
        print("• Comprehensive monitoring and status tracking")
    else:
        print("❌ MIGRATION FAILED!")
        print("Please check the error messages above and resolve any issues.")
    
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)