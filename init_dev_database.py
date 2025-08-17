#!/usr/bin/env python3
"""
Simple database initialization script for development
Uses SQLite to avoid PostgreSQL dependency issues
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment to development
os.environ['ENVIRONMENT'] = 'development'
os.environ['USE_SQLITE'] = 'true'
os.environ['SKIP_REDIS'] = 'true'
os.environ['DATABASE_URL'] = 'sqlite:///./scrollintel_dev.db'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def init_development_database():
    """Initialize development database with SQLite"""
    try:
        logger.info("🚀 Initializing ScrollIntel development database...")
        
        # Import after setting environment variables
        from scrollintel.models.database_utils import DatabaseManager
        from scrollintel.models.database import Base
        from scrollintel.core.config import get_settings
        
        # Get settings
        settings = get_settings()
        logger.info(f"📊 Using database: {settings.database_url}")
        
        # Create database manager
        db_manager = DatabaseManager()
        
        # Initialize (this will create the SQLite file)
        await db_manager.initialize()
        logger.info("✅ Database manager initialized")
        
        # Create all tables
        await db_manager.create_tables()
        logger.info("✅ Database tables created")
        
        # Check health
        health = await db_manager.check_health()
        logger.info(f"🏥 Database health: {health}")
        
        # Close connections
        await db_manager.close()
        
        logger.info("🎉 Development database initialization complete!")
        logger.info("📁 Database file: scrollintel_dev.db")
        logger.info("🚀 You can now start the backend with: python -m uvicorn scrollintel.api.main:app --reload")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = asyncio.run(init_development_database())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()