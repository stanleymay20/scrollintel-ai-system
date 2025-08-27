"""
Create security and compliance database migration
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.models.database import Base, get_database_url
from scrollintel.models.security_models import (
    User, Role, Permission, UserRole, RolePermission,
    UserSession, AuditEvent, SecurityConfiguration,
    EncryptionKey, ComplianceRule, SecurityIncident,
    DataClassification,
    create_default_roles_and_permissions,
    create_default_compliance_rules
)

def create_security_tables():
    """Create all security-related database tables"""
    
    # Get database URL
    database_url = get_database_url()
    
    # Create engine
    engine = create_engine(database_url)
    
    print("Creating security and compliance database tables...")
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine, checkfirst=True)
        print("✓ Security tables created successfully")
        
        # Create default roles and permissions
        print("Creating default roles and permissions...")
        create_default_roles_and_permissions()
        print("✓ Default roles and permissions created")
        
        # Create default compliance rules
        print("Creating default compliance rules...")
        create_default_compliance_rules()
        print("✓ Default compliance rules created")
        
        # Create default admin user
        print("Creating default admin user...")
        create_default_admin_user(engine)
        print("✓ Default admin user created")
        
        # Create security configuration entries
        print("Creating security configuration...")
        create_security_configuration(engine)
        print("✓ Security configuration created")
        
        print("\n🔒 Security and compliance framework setup completed successfully!")
        print("\nDefault admin user created.")
        print("Username: admin")
        print("Password: Generated securely - check environment variables or logs")
        print("\nPlease set ADMIN_PASSWORD environment variable or change password after first login.")
        
    except Exception as e:
        print(f"❌ Error creating security tables: {str(e)}")
        raise

def create_default_admin_user(engine):
    """Create default admin user"""
    from scrollintel.security.security_framework import security_framework
    
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        # Check if admin user already exists
        admin_user = db.query(User).filter(User.username == "admin").first()
        if admin_user:
            print("Admin user already exists, skipping creation")
            return
        
        # Get password from environment or generate secure random password
        import secrets
        import string
        
        password = os.getenv("ADMIN_PASSWORD")
        if not password:
            # Generate secure random password
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
            password = ''.join(secrets.choice(alphabet) for _ in range(16))
            print(f"Generated admin password: {password}")
            print("IMPORTANT: Save this password securely!")
        
        password_hash, salt = security_framework.encryption_manager.hash_password(password)
        
        # Create admin user
        admin_user = User(
            username="admin",
            email="admin@scrollintel.com",
            password_hash=password_hash,
            password_salt=salt,
            is_active=True,
            is_verified=True,
            security_level="top_secret",
            mfa_enabled=False
        )
        
        db.add(admin_user)
        db.commit()
        
        # Assign admin role
        admin_role = db.query(Role).filter(Role.name == "admin").first()
        if admin_role:
            user_role = UserRole(user_id=admin_user.id, role_id=admin_role.id)
            db.add(user_role)
            db.commit()
        
        # Generate MFA secret for admin
        mfa_secret = security_framework.mfa.generate_totp_secret("admin")
        admin_user.mfa_secret = mfa_secret
        db.commit()
        
    finally:
        db.close()

def create_security_configuration(engine):
    """Create default security configuration"""
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        # Default security configurations
        configs = [
            {
                "key": "session_timeout_hours",
                "value": "8",
                "description": "Default session timeout in hours"
            },
            {
                "key": "max_failed_login_attempts",
                "value": "5",
                "description": "Maximum failed login attempts before lockout"
            },
            {
                "key": "lockout_duration_minutes",
                "value": "15",
                "description": "Account lockout duration in minutes"
            },
            {
                "key": "password_min_length",
                "value": "8",
                "description": "Minimum password length"
            },
            {
                "key": "password_require_special_chars",
                "value": "true",
                "description": "Require special characters in passwords"
            },
            {
                "key": "mfa_required_for_admins",
                "value": "true",
                "description": "Require MFA for admin users"
            },
            {
                "key": "audit_retention_days",
                "value": "2555",
                "description": "Audit log retention period in days (7 years)"
            },
            {
                "key": "encryption_algorithm",
                "value": "AES-256",
                "description": "Default encryption algorithm"
            }
        ]
        
        for config_data in configs:
            # Check if config already exists
            existing_config = db.query(SecurityConfiguration).filter(
                SecurityConfiguration.key == config_data["key"]
            ).first()
            
            if not existing_config:
                config = SecurityConfiguration(**config_data)
                db.add(config)
        
        db.commit()
        
    finally:
        db.close()

def verify_security_setup():
    """Verify security setup is working correctly"""
    print("\nVerifying security setup...")
    
    try:
        # Test database connection
        database_url = get_database_url()
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            # Check if tables exist
            tables_to_check = [
                'users', 'roles', 'permissions', 'user_roles', 'role_permissions',
                'user_sessions', 'audit_events', 'security_configurations',
                'encryption_keys', 'compliance_rules', 'security_incidents',
                'data_classifications'
            ]
            
            for table in tables_to_check:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                print(f"✓ Table '{table}': {count} records")
        
        # Test security framework
        from scrollintel.security.security_framework import security_framework
        
        # Test encryption
        test_data = "Test encryption data"
        encrypted = security_framework.encrypt_sensitive_data(test_data)
        decrypted = security_framework.decrypt_sensitive_data(encrypted)
        
        if decrypted == test_data:
            print("✓ Encryption/decryption test passed")
        else:
            print("❌ Encryption/decryption test failed")
        
        # Test RBAC
        permissions = security_framework.rbac.get_user_permissions("admin")
        if permissions:
            print(f"✓ RBAC test passed - Admin has {len(permissions)} permissions")
        else:
            print("❌ RBAC test failed - No permissions found for admin")
        
        print("\n🔒 Security verification completed successfully!")
        
    except Exception as e:
        print(f"❌ Security verification failed: {str(e)}")
        raise

if __name__ == "__main__":
    print("🔒 ScrollIntel Security and Compliance Framework Setup")
    print("=" * 60)
    
    try:
        # Create security tables
        create_security_tables()
        
        # Verify setup
        verify_security_setup()
        
        print("\n" + "=" * 60)
        print("Security framework is ready for use!")
        print("\nNext steps:")
        print("1. Change the default admin password")
        print("2. Set up MFA for admin users")
        print("3. Configure compliance requirements")
        print("4. Review security policies")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        sys.exit(1)