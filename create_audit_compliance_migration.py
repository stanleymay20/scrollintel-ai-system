"""
Database Migration for Audit and Compliance System

This script creates the necessary database tables for the audit and compliance
system including audit logs, compliance checks, access control, and change approvals.
"""

import sys
import os
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.core.config import get_settings
from scrollintel.models.audit_models import Base


def create_audit_compliance_tables():
    """Create all audit and compliance related tables"""
    
    # Get database URL
    settings = get_settings()
    database_url = settings.database_url
    
    # Create engine
    engine = create_engine(database_url)
    
    # Create all tables
    print("Creating audit and compliance tables...")
    Base.metadata.create_all(engine)
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Create indexes for better performance
        print("Creating performance indexes...")
        
        # Audit logs indexes
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp 
            ON audit_logs(timestamp DESC);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id 
            ON audit_logs(user_id);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_resource 
            ON audit_logs(resource_type, resource_id);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_action 
            ON audit_logs(action);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_risk_level 
            ON audit_logs(risk_level);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_compliance_status 
            ON audit_logs(compliance_status);
        """))
        
        # Compliance checks indexes
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_compliance_checks_audit_log_id 
            ON compliance_checks(audit_log_id);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_compliance_checks_status 
            ON compliance_checks(status);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_compliance_checks_check_name 
            ON compliance_checks(check_name);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_compliance_checks_checked_at 
            ON compliance_checks(checked_at DESC);
        """))
        
        # Change approvals indexes
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_change_approvals_prompt_id 
            ON change_approvals(prompt_id);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_change_approvals_requester_id 
            ON change_approvals(requester_id);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_change_approvals_status 
            ON change_approvals(status);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_change_approvals_requested_at 
            ON change_approvals(requested_at DESC);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_change_approvals_approver_id 
            ON change_approvals(approver_id);
        """))
        
        # Access control indexes
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_access_controls_user_id 
            ON access_controls(user_id);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_access_controls_role 
            ON access_controls(role);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_access_controls_is_active 
            ON access_controls(is_active);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_access_controls_expires_at 
            ON access_controls(expires_at);
        """))
        
        # Compliance reports indexes
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_compliance_reports_report_type 
            ON compliance_reports(report_type);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_compliance_reports_generated_at 
            ON compliance_reports(generated_at DESC);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_compliance_reports_date_range 
            ON compliance_reports(date_range_start, date_range_end);
        """))
        
        # Full-text search indexes for audit logs (PostgreSQL specific)
        try:
            session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_audit_logs_search 
                ON audit_logs USING gin(to_tsvector('english', 
                    COALESCE(resource_name, '') || ' ' || 
                    COALESCE(changes_summary, '') || ' ' || 
                    COALESCE(user_email, '')
                ));
            """))
            print("Created full-text search index for audit logs")
        except Exception as e:
            print(f"Note: Could not create full-text search index (likely not PostgreSQL): {e}")
        
        session.commit()
        print("Successfully created all indexes")
        
        # Insert initial data
        print("Inserting initial configuration data...")
        
        # Insert system user access control entry
        session.execute(text("""
            INSERT INTO access_controls (
                id, user_id, user_email, role, permissions, 
                granted_by, granted_at, is_active, access_count
            ) VALUES (
                'system-admin-access', 'system', 'system@scrollintel.com', 'admin',
                '["admin:user_manage", "admin:audit_read", "admin:compliance_manage", "admin:system_config"]',
                'system', :now, true, 0
            ) ON CONFLICT (id) DO NOTHING;
        """), {"now": datetime.utcnow()})
        
        # Insert default compliance rules configuration
        session.execute(text("""
            INSERT INTO compliance_reports (
                id, report_type, report_name, date_range_start, date_range_end,
                filters, summary, detailed_findings, generated_by, generated_at, file_format
            ) VALUES (
                'initial-system-report', 'compliance_status', 'Initial System Status',
                :start_date, :end_date, '{}', 
                '{"message": "Initial system setup completed", "compliance_rate": 100}',
                '[]', 'system', :now, 'json'
            ) ON CONFLICT (id) DO NOTHING;
        """), {
            "start_date": datetime.utcnow(),
            "end_date": datetime.utcnow(),
            "now": datetime.utcnow()
        })
        
        session.commit()
        print("Successfully inserted initial data")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        session.rollback()
        raise
    finally:
        session.close()
    
    print("Audit and compliance migration completed successfully!")


def verify_tables():
    """Verify that all tables were created correctly"""
    
    settings = get_settings()
    database_url = settings.database_url
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check if tables exist
        tables_to_check = [
            'audit_logs',
            'compliance_checks', 
            'change_approvals',
            'access_controls',
            'compliance_reports'
        ]
        
        print("Verifying table creation...")
        
        for table_name in tables_to_check:
            result = session.execute(text(f"""
                SELECT COUNT(*) as count 
                FROM information_schema.tables 
                WHERE table_name = '{table_name}';
            """))
            
            count = result.fetchone()[0]
            if count > 0:
                print(f"✓ Table '{table_name}' exists")
            else:
                print(f"✗ Table '{table_name}' missing")
        
        # Check indexes
        print("\nVerifying index creation...")
        
        result = session.execute(text("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename IN ('audit_logs', 'compliance_checks', 'change_approvals', 'access_controls', 'compliance_reports')
            ORDER BY indexname;
        """))
        
        indexes = result.fetchall()
        print(f"Created {len(indexes)} indexes:")
        for index in indexes:
            print(f"  - {index[0]}")
        
        # Test basic functionality
        print("\nTesting basic functionality...")
        
        # Test audit log insertion
        session.execute(text("""
            INSERT INTO audit_logs (
                id, timestamp, user_id, user_email, action, resource_type, 
                resource_id, compliance_status, risk_level
            ) VALUES (
                'test-audit-log', :now, 'test-user', 'test@example.com', 
                'create', 'test', 'test-resource', 'pending_review', 'low'
            );
        """), {"now": datetime.utcnow()})
        
        # Test access control insertion
        session.execute(text("""
            INSERT INTO access_controls (
                id, user_id, user_email, role, permissions, granted_by, 
                granted_at, is_active, access_count
            ) VALUES (
                'test-access-control', 'test-user', 'test@example.com', 'viewer',
                '["prompt:read"]', 'admin', :now, true, 0
            );
        """), {"now": datetime.utcnow()})
        
        session.commit()
        
        # Verify data was inserted
        result = session.execute(text("SELECT COUNT(*) FROM audit_logs WHERE id = 'test-audit-log';"))
        audit_count = result.fetchone()[0]
        
        result = session.execute(text("SELECT COUNT(*) FROM access_controls WHERE id = 'test-access-control';"))
        access_count = result.fetchone()[0]
        
        if audit_count > 0 and access_count > 0:
            print("✓ Basic functionality test passed")
        else:
            print("✗ Basic functionality test failed")
        
        # Clean up test data
        session.execute(text("DELETE FROM audit_logs WHERE id = 'test-audit-log';"))
        session.execute(text("DELETE FROM access_controls WHERE id = 'test-access-control';"))
        session.commit()
        
        print("\nAll verification checks completed successfully!")
        
    except Exception as e:
        print(f"Error during verification: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    print("Starting audit and compliance system migration...")
    print("=" * 50)
    
    try:
        # Create tables and indexes
        create_audit_compliance_tables()
        
        # Verify everything was created correctly
        verify_tables()
        
        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print("\nNext steps:")
        print("1. Run the test suite: python -m pytest tests/test_audit_compliance_system.py")
        print("2. Start using the audit and compliance APIs")
        print("3. Configure compliance rules as needed")
        print("4. Set up regular compliance reporting")
        
    except Exception as e:
        print(f"\nMigration failed: {e}")
        print("Please check the error message and try again.")
        sys.exit(1)