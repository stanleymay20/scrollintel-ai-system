"""
Database migration script for API Key Management system.
Creates tables for API keys, usage tracking, quotas, and rate limiting.
"""

import sys
import os
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.models.database import Base
from scrollintel.models.api_key_models import APIKey, APIUsage, APIQuota, RateLimitRecord
from scrollintel.core.config import get_settings


def create_api_key_tables():
    """Create API key management tables."""
    settings = get_settings()
    
    # Create database engine
    engine = create_engine(settings.database_url)
    
    # Create session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    try:
        print("Creating API Key Management tables...")
        
        # Create all tables defined in the models
        Base.metadata.create_all(bind=engine, tables=[
            APIKey.__table__,
            APIUsage.__table__,
            APIQuota.__table__,
            RateLimitRecord.__table__
        ])
        
        print("✅ API Key Management tables created successfully!")
        
        # Create indexes for better performance
        print("Creating additional indexes...")
        
        # API Usage indexes for analytics
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp_endpoint 
            ON api_usage (timestamp, endpoint);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_api_usage_user_timestamp 
            ON api_usage (user_id, timestamp);
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_api_usage_key_timestamp 
            ON api_usage (api_key_id, timestamp);
        """))
        
        # API Quota indexes
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_api_quota_period_user 
            ON api_quotas (user_id, period_start, period_end);
        """))
        
        # Rate limit indexes
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_rate_limit_active_window 
            ON rate_limit_records (api_key_id, window_start, window_duration_seconds)
            WHERE window_start + INTERVAL '1 second' * window_duration_seconds > NOW();
        """))
        
        session.commit()
        print("✅ Additional indexes created successfully!")
        
        # Insert sample data for testing (optional)
        if os.getenv('CREATE_SAMPLE_DATA', 'false').lower() == 'true':
            print("Creating sample API key data...")
            create_sample_data(session)
        
        print("\n🎉 API Key Management system setup completed!")
        print("\nNext steps:")
        print("1. Update your main application to include the API key routes")
        print("2. Add the API key middleware to your FastAPI app")
        print("3. Configure rate limiting and billing settings")
        print("4. Test the API key creation and usage endpoints")
        
    except Exception as e:
        print(f"❌ Error creating tables: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()


def create_sample_data(session):
    """Create sample data for testing."""
    try:
        # Check if we already have sample data
        existing_key = session.query(APIKey).filter(
            APIKey.name == "Sample Development Key"
        ).first()
        
        if existing_key:
            print("Sample data already exists, skipping...")
            return
        
        # Create a sample API key
        from scrollintel.models.database import User
        
        # Find or create a test user
        test_user = session.query(User).filter(
            User.email == "test@scrollintel.com"
        ).first()
        
        if not test_user:
            print("No test user found, skipping sample data creation...")
            return
        
        # Generate sample API key
        raw_key, key_hash = APIKey.generate_key()
        key_prefix = raw_key[:8]
        
        sample_key = APIKey(
            user_id=str(test_user.id),
            name="Sample Development Key",
            description="Sample API key for development and testing",
            key_hash=key_hash,
            key_prefix=key_prefix,
            permissions=[
                "agents:read",
                "agents:execute",
                "data:read",
                "models:read",
                "models:execute"
            ],
            rate_limit_per_minute=100,
            rate_limit_per_hour=1000,
            rate_limit_per_day=10000,
            quota_requests_per_month=50000,
            is_active=True
        )
        
        session.add(sample_key)
        session.commit()
        session.refresh(sample_key)
        
        # Create sample usage data
        now = datetime.utcnow()
        sample_usage = [
            APIUsage(
                api_key_id=sample_key.id,
                user_id=sample_key.user_id,
                endpoint="/api/v1/agents/cto",
                method="POST",
                status_code=200,
                response_time_ms=150.5,
                request_size_bytes=1024,
                response_size_bytes=2048,
                ip_address="127.0.0.1",
                user_agent="TestClient/1.0",
                request_metadata={"test": True},
                timestamp=now
            ),
            APIUsage(
                api_key_id=sample_key.id,
                user_id=sample_key.user_id,
                endpoint="/api/v1/data/upload",
                method="POST",
                status_code=201,
                response_time_ms=250.0,
                request_size_bytes=5120,
                response_size_bytes=512,
                ip_address="127.0.0.1",
                user_agent="TestClient/1.0",
                request_metadata={"file_type": "csv"},
                timestamp=now
            )
        ]
        
        for usage in sample_usage:
            session.add(usage)
        
        # Create sample quota
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 12:
            period_end = period_start.replace(year=now.year + 1, month=1)
        else:
            period_end = period_start.replace(month=now.month + 1)
        
        sample_quota = APIQuota(
            api_key_id=sample_key.id,
            user_id=sample_key.user_id,
            period_start=period_start,
            period_end=period_end,
            requests_count=2,
            requests_limit=50000,
            data_transfer_bytes=7680,
            compute_time_seconds=0.4,
            cost_usd=0.002
        )
        
        session.add(sample_quota)
        session.commit()
        
        print(f"✅ Sample data created!")
        print(f"   Sample API Key: {raw_key}")
        print(f"   Key ID: {sample_key.id}")
        print("   ⚠️  Save this key - it won't be shown again!")
        
    except Exception as e:
        print(f"❌ Error creating sample data: {str(e)}")
        session.rollback()


def drop_api_key_tables():
    """Drop API key management tables (for cleanup)."""
    settings = get_settings()
    engine = create_engine(settings.database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    try:
        print("Dropping API Key Management tables...")
        
        # Drop tables in reverse order due to foreign key constraints
        tables_to_drop = [
            RateLimitRecord.__table__,
            APIQuota.__table__,
            APIUsage.__table__,
            APIKey.__table__
        ]
        
        for table in tables_to_drop:
            table.drop(bind=engine, checkfirst=True)
        
        session.commit()
        print("✅ API Key Management tables dropped successfully!")
        
    except Exception as e:
        print(f"❌ Error dropping tables: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="API Key Management Database Migration")
    parser.add_argument(
        "--action",
        choices=["create", "drop"],
        default="create",
        help="Action to perform (create or drop tables)"
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Create sample data for testing"
    )
    
    args = parser.parse_args()
    
    if args.sample_data:
        os.environ['CREATE_SAMPLE_DATA'] = 'true'
    
    if args.action == "create":
        create_api_key_tables()
    elif args.action == "drop":
        if input("Are you sure you want to drop all API key tables? (yes/no): ").lower() == "yes":
            drop_api_key_tables()
        else:
            print("Operation cancelled.")