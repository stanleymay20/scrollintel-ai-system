"""
Create database migration for user management and role-based access control.
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.core.config import get_settings
from scrollintel.models.database import Base
from scrollintel.models.user_management_models import (
    Organization, OrganizationUser, Workspace, WorkspaceMember,
    Project, UserInvitation, UserSession, UserAuditLog, APIKey
)


def create_user_management_tables():
    """Create user management tables in the database."""
    settings = get_settings()
    
    # Create database engine
    engine = create_engine(settings.database_url)
    
    # Create all tables
    print("Creating user management tables...")
    Base.metadata.create_all(engine, tables=[
        Organization.__table__,
        OrganizationUser.__table__,
        Workspace.__table__,
        WorkspaceMember.__table__,
        Project.__table__,
        UserInvitation.__table__,
        UserSession.__table__,
        UserAuditLog.__table__,
        APIKey.__table__
    ])
    
    # Create session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Create default organization for existing users
        print("Creating default organization...")
        
        # Check if any organizations exist
        existing_orgs = db.execute(text("SELECT COUNT(*) FROM organizations")).scalar()
        
        if existing_orgs == 0:
            # Create default organization
            default_org_sql = """
            INSERT INTO organizations (
                id, name, display_name, description, subscription_plan, 
                subscription_status, max_users, max_workspaces, max_storage_gb,
                settings, features, is_active, created_at, updated_at
            ) VALUES (
                gen_random_uuid(), 'default', 'Default Organization', 
                'Default organization for existing users', 'free', 'active',
                10, 5, 10,
                '{"allow_user_registration": false, "require_email_verification": true, "session_timeout_hours": 24}',
                '["basic_analytics", "file_upload", "api_access"]',
                true, NOW(), NOW()
            ) RETURNING id;
            """
            
            result = db.execute(text(default_org_sql))
            default_org_id = result.fetchone()[0]
            
            print(f"Created default organization with ID: {default_org_id}")
            
            # Add existing users to default organization as admins
            existing_users_sql = """
            INSERT INTO organization_users (
                id, organization_id, user_id, role, permissions, status, joined_at
            )
            SELECT 
                gen_random_uuid(), :org_id, id, 'admin', '["*"]', 'active', NOW()
            FROM users 
            WHERE is_active = true;
            """
            
            db.execute(text(existing_users_sql), {"org_id": default_org_id})
            
            # Create default workspace
            default_workspace_sql = """
            INSERT INTO workspaces (
                id, name, description, organization_id, owner_id, visibility,
                settings, is_active, created_at, updated_at
            )
            SELECT 
                gen_random_uuid(), 'Default Workspace', 
                'Default workspace for the organization',
                :org_id, u.id, 'organization',
                '{"default_workspace": true}', true, NOW(), NOW()
            FROM users u
            WHERE u.is_active = true
            LIMIT 1
            RETURNING id;
            """
            
            result = db.execute(text(default_workspace_sql), {"org_id": default_org_id})
            if result.rowcount > 0:
                default_workspace_id = result.fetchone()[0]
                
                # Add all organization users as workspace members
                workspace_members_sql = """
                INSERT INTO workspace_members (
                    id, workspace_id, user_id, role, permissions, status, added_at
                )
                SELECT 
                    gen_random_uuid(), :workspace_id, ou.user_id, 
                    CASE 
                        WHEN ou.role = 'admin' THEN 'owner'
                        WHEN ou.role = 'analyst' THEN 'admin'
                        ELSE 'member'
                    END,
                    '["*"]', 'active', NOW()
                FROM organization_users ou
                WHERE ou.organization_id = :org_id;
                """
                
                db.execute(text(workspace_members_sql), {
                    "workspace_id": default_workspace_id,
                    "org_id": default_org_id
                })
                
                print(f"Created default workspace with ID: {default_workspace_id}")
        
        # Create indexes for better performance
        print("Creating additional indexes...")
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_org_user_composite ON organization_users(organization_id, user_id, status);",
            "CREATE INDEX IF NOT EXISTS idx_workspace_member_composite ON workspace_members(workspace_id, user_id, status);",
            "CREATE INDEX IF NOT EXISTS idx_user_audit_log_composite ON user_audit_logs(organization_id, timestamp, action);",
            "CREATE INDEX IF NOT EXISTS idx_user_session_composite ON user_sessions(user_id, is_active, expires_at);",
            "CREATE INDEX IF NOT EXISTS idx_api_key_composite ON api_keys(organization_id, is_active, expires_at);",
            "CREATE INDEX IF NOT EXISTS idx_invitation_composite ON user_invitations(organization_id, status, expires_at);"
        ]
        
        for index_sql in indexes:
            try:
                db.execute(text(index_sql))
                print(f"Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                print(f"Warning: Could not create index - {e}")
        
        db.commit()
        print("User management migration completed successfully!")
        
    except Exception as e:
        db.rollback()
        print(f"Error during migration: {e}")
        raise
    finally:
        db.close()


def verify_migration():
    """Verify that the migration was successful."""
    settings = get_settings()
    engine = create_engine(settings.database_url)
    
    with engine.connect() as conn:
        # Check if tables exist
        tables_to_check = [
            'organizations', 'organization_users', 'workspaces', 
            'workspace_members', 'projects', 'user_invitations',
            'user_sessions', 'user_audit_logs', 'api_keys'
        ]
        
        for table in tables_to_check:
            result = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{table}'
                );
            """))
            
            exists = result.scalar()
            if exists:
                print(f"✓ Table '{table}' created successfully")
            else:
                print(f"✗ Table '{table}' not found")
        
        # Check if default organization was created
        result = conn.execute(text("SELECT COUNT(*) FROM organizations WHERE name = 'default';"))
        default_org_count = result.scalar()
        
        if default_org_count > 0:
            print("✓ Default organization created successfully")
        else:
            print("✗ Default organization not found")
        
        # Check if users were added to default organization
        result = conn.execute(text("""
            SELECT COUNT(*) FROM organization_users ou
            JOIN organizations o ON ou.organization_id = o.id
            WHERE o.name = 'default';
        """))
        
        org_user_count = result.scalar()
        print(f"✓ {org_user_count} users added to default organization")


if __name__ == "__main__":
    print("Starting user management migration...")
    print("=" * 50)
    
    try:
        create_user_management_tables()
        print("\n" + "=" * 50)
        print("Verifying migration...")
        verify_migration()
        print("\n" + "=" * 50)
        print("Migration completed successfully! 🎉")
        
    except Exception as e:
        print(f"\nMigration failed: {e}")
        sys.exit(1)