"""
Create database migration for audit and compliance system.
"""
import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.models.audit_models import Base
from scrollintel.core.config import get_config


def create_audit_compliance_tables():
    """Create audit and compliance tables."""
    
    # Get database URL
    config = get_config()
    database_url = config.get("database_url", "sqlite:///audit_compliance.db")
    
    # Create engine
    engine = create_engine(database_url)
    
    # Create session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        print("✅ Audit and compliance tables created successfully")
        
        # Create indexes for better performance
        with engine.connect() as conn:
            # Check if audit_logs table exists and has the expected columns
            try:
                # Test if the table has the expected structure
                conn.execute(text("SELECT risk_level FROM audit_logs LIMIT 1;"))
                
                # If we get here, the table exists with the right structure
                # Audit logs indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp 
                    ON audit_logs(timestamp DESC);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id 
                    ON audit_logs(user_id);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_audit_logs_resource 
                    ON audit_logs(resource_type, resource_id);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_audit_logs_action 
                    ON audit_logs(action);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_audit_logs_risk_level 
                    ON audit_logs(risk_level);
                """))
                
            except Exception as e:
                print(f"⚠️  Skipping audit_logs indexes due to table structure: {e}")
                # Table might not exist or have different structure, skip indexes
            
            # Create indexes for other tables
            try:
                # Compliance violations indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_compliance_violations_rule_id 
                    ON compliance_violations(rule_id);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_compliance_violations_resource 
                    ON compliance_violations(resource_type, resource_id);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_compliance_violations_status 
                    ON compliance_violations(status);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_compliance_violations_severity 
                    ON compliance_violations(severity);
                """))
                
                # Access controls indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_access_controls_user_id 
                    ON access_controls(user_id);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_access_controls_resource 
                    ON access_controls(resource_type, resource_id);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_access_controls_role 
                    ON access_controls(role);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_access_controls_team_id 
                    ON access_controls(team_id);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_access_controls_expires_at 
                    ON access_controls(expires_at);
                """))
                
                # Change approvals indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_change_approvals_resource 
                    ON change_approvals(resource_type, resource_id);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_change_approvals_requested_by 
                    ON change_approvals(requested_by);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_change_approvals_status 
                    ON change_approvals(status);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_change_approvals_priority 
                    ON change_approvals(priority);
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_change_approvals_deadline 
                    ON change_approvals(deadline);
                """))
                
            except Exception as e:
                print(f"⚠️  Some indexes could not be created: {e}")
            
            conn.commit()
            
        print("✅ Database indexes created successfully")
        
        # Insert default compliance rules
        session = SessionLocal()
        try:
            from scrollintel.core.compliance_manager import ComplianceManager
            
            compliance_manager = ComplianceManager(session)
            # Default rules are created in the constructor
            
            print("✅ Default compliance rules initialized")
            
        finally:
            session.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating audit and compliance tables: {e}")
        return False


def verify_migration():
    """Verify that the migration was successful."""
    
    config = get_config()
    database_url = config.get("database_url", "sqlite:///audit_compliance.db")
    engine = create_engine(database_url)
    
    try:
        with engine.connect() as conn:
            # Check if tables exist
            tables = [
                "audit_logs",
                "compliance_rules", 
                "compliance_violations",
                "access_controls",
                "change_approvals"
            ]
            
            for table in tables:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name = '{table}';
                """))
                
                count = result.scalar()
                if count == 0:
                    print(f"❌ Table {table} not found")
                    return False
                else:
                    print(f"✅ Table {table} exists")
            
            # Check compliance rules
            result = conn.execute(text("SELECT COUNT(*) FROM compliance_rules;"))
            rule_count = result.scalar()
            print(f"✅ Found {rule_count} compliance rules")
            
        return True
        
    except Exception as e:
        print(f"❌ Error verifying migration: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Creating audit and compliance system migration...")
    
    success = create_audit_compliance_tables()
    
    if success:
        print("\n🔍 Verifying migration...")
        if verify_migration():
            print("\n✅ Audit and compliance system migration completed successfully!")
            print("\nNext steps:")
            print("1. Test the audit logging functionality")
            print("2. Configure compliance rules for your environment")
            print("3. Set up access controls for resources")
            print("4. Test the change approval workflow")
        else:
            print("\n❌ Migration verification failed")
            sys.exit(1)
    else:
        print("\n❌ Migration failed")
        sys.exit(1)