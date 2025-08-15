"""
Database migration script for ScrollIntel Core
Creates tables and initial data
"""
import asyncio
from sqlalchemy import text
from database import engine, SessionLocal, create_tables
from models import User, Workspace, WorkspaceMember
import uuid
from datetime import datetime


def create_initial_data():
    """Create initial data for development"""
    db = SessionLocal()
    
    try:
        # Check if data already exists
        existing_user = db.query(User).first()
        if existing_user:
            print("✅ Initial data already exists")
            return
        
        print("📝 Creating initial data...")
        
        # Create admin user
        admin_user = User(
            email="admin@scrollintel.com",
            name="ScrollIntel Admin",
            hashed_password="$2b$12$dummy_hash_for_development",  # In production, use proper hashing
            role="admin",
            is_active=True
        )
        db.add(admin_user)
        db.flush()  # Get the ID
        
        # Create default workspace
        default_workspace = Workspace(
            name="Default Workspace",
            description="Default workspace for getting started",
            owner_id=admin_user.id,
            settings={"theme": "default", "auto_save": True}
        )
        db.add(default_workspace)
        db.flush()
        
        # Add admin as workspace member
        workspace_member = WorkspaceMember(
            workspace_id=default_workspace.id,
            user_id=admin_user.id,
            role="owner"
        )
        db.add(workspace_member)
        
        # Create demo user
        demo_user = User(
            email="demo@scrollintel.com",
            name="Demo User",
            hashed_password="$2b$12$dummy_hash_for_development",
            role="user",
            is_active=True
        )
        db.add(demo_user)
        db.flush()
        
        # Add demo user to default workspace
        demo_member = WorkspaceMember(
            workspace_id=default_workspace.id,
            user_id=demo_user.id,
            role="member"
        )
        db.add(demo_member)
        
        db.commit()
        
        print(f"✅ Created admin user: {admin_user.email}")
        print(f"✅ Created demo user: {demo_user.email}")
        print(f"✅ Created default workspace: {default_workspace.name}")
        
    except Exception as e:
        print(f"❌ Error creating initial data: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def main():
    """Main migration function"""
    print("🚀 Starting ScrollIntel Core database migration...")
    
    try:
        # Test database connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ Connected to PostgreSQL: {version}")
        
        # Create all tables
        print("📊 Creating database tables...")
        create_tables()
        print("✅ Database tables created successfully")
        
        # Create initial data
        create_initial_data()
        
        print("🎉 Migration completed successfully!")
        print("\n📋 Next steps:")
        print("1. Start the application: ./start.sh dev")
        print("2. Access API docs: http://localhost:8001/docs")
        print("3. Use demo credentials:")
        print("   - Admin: admin@scrollintel.com")
        print("   - Demo: demo@scrollintel.com")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()