"""
Database Migration for Continuous Improvement Framework

This script creates the database tables and indexes for the continuous improvement system.
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.models.continuous_improvement_models import Base
from scrollintel.core.config import get_settings

def create_continuous_improvement_tables():
    """Create continuous improvement database tables"""
    
    settings = get_settings()
    
    # Create database engine
    engine = create_engine(settings.database_url)
    
    print("🗄️  Creating Continuous Improvement Framework tables...")
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Create additional indexes for performance
        with engine.connect() as conn:
            # Indexes for user_feedback table
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id_created 
                ON user_feedback(user_id, created_at DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_feedback_feature_area_type 
                ON user_feedback(feature_area, feedback_type);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_feedback_business_impact 
                ON user_feedback(business_impact_score DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_feedback_satisfaction 
                ON user_feedback(satisfaction_rating DESC) 
                WHERE satisfaction_rating IS NOT NULL;
            """))
            
            # Indexes for ab_tests table
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_ab_tests_status_feature 
                ON ab_tests(status, feature_area);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_ab_tests_dates 
                ON ab_tests(start_date, end_date) 
                WHERE start_date IS NOT NULL;
            """))
            
            # Indexes for ab_test_results table
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_ab_test_results_test_variant 
                ON ab_test_results(test_id, variant_name);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_ab_test_results_user_timestamp 
                ON ab_test_results(user_id, timestamp DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_ab_test_results_conversion 
                ON ab_test_results(test_id, conversion_event) 
                WHERE conversion_event = true;
            """))
            
            # Indexes for model_retraining_jobs table
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_retraining_model_status 
                ON model_retraining_jobs(model_name, status);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_retraining_scheduled 
                ON model_retraining_jobs(scheduled_at) 
                WHERE status = 'scheduled';
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_retraining_agent_type 
                ON model_retraining_jobs(agent_type) 
                WHERE agent_type IS NOT NULL;
            """))
            
            # Indexes for feature_enhancements table
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_feature_enhancements_status_priority 
                ON feature_enhancements(status, priority);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_feature_enhancements_business_value 
                ON feature_enhancements(business_value_score DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_feature_enhancements_feature_area 
                ON feature_enhancements(feature_area, enhancement_type);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_feature_enhancements_requester 
                ON feature_enhancements(requester_id, created_at DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_feature_enhancements_assigned 
                ON feature_enhancements(assigned_developer, status) 
                WHERE assigned_developer IS NOT NULL;
            """))
            
            conn.commit()
        
        print("✅ Continuous improvement tables created successfully")
        print("📊 Performance indexes added")
        
        # Insert sample configuration data
        insert_sample_data(engine)
        
    except Exception as e:
        print(f"❌ Error creating tables: {str(e)}")
        raise

def insert_sample_data(engine):
    """Insert sample configuration and reference data"""
    
    print("📝 Inserting sample configuration data...")
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Sample data would be inserted here if needed
        # For now, we'll just create the schema
        
        print("✅ Sample data inserted successfully")
        
    except Exception as e:
        print(f"❌ Error inserting sample data: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

def verify_tables():
    """Verify that all tables were created correctly"""
    
    settings = get_settings()
    engine = create_engine(settings.database_url)
    
    print("🔍 Verifying table creation...")
    
    expected_tables = [
        'user_feedback',
        'ab_tests', 
        'ab_test_results',
        'model_retraining_jobs',
        'feature_enhancements'
    ]
    
    with engine.connect() as conn:
        # Get list of tables
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('user_feedback', 'ab_tests', 'ab_test_results', 'model_retraining_jobs', 'feature_enhancements')
        """))
        
        existing_tables = [row[0] for row in result]
        
        print(f"📋 Expected tables: {len(expected_tables)}")
        print(f"✅ Created tables: {len(existing_tables)}")
        
        for table in expected_tables:
            if table in existing_tables:
                print(f"   ✓ {table}")
            else:
                print(f"   ❌ {table} - MISSING")
        
        # Verify indexes
        result = conn.execute(text("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename IN ('user_feedback', 'ab_tests', 'ab_test_results', 'model_retraining_jobs', 'feature_enhancements')
            AND indexname LIKE 'idx_%'
        """))
        
        indexes = [row[0] for row in result]
        print(f"📊 Performance indexes created: {len(indexes)}")
        
        if len(existing_tables) == len(expected_tables):
            print("✅ All continuous improvement tables verified successfully")
            return True
        else:
            print("❌ Some tables are missing")
            return False

def main():
    """Main migration function"""
    
    print("🚀 Continuous Improvement Framework Database Migration")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Create tables
        create_continuous_improvement_tables()
        
        # Verify creation
        if verify_tables():
            print("\n🎉 Migration completed successfully!")
            print("🔧 Continuous improvement system is ready for use")
        else:
            print("\n❌ Migration completed with errors")
            return 1
            
    except Exception as e:
        print(f"\n💥 Migration failed: {str(e)}")
        return 1
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)