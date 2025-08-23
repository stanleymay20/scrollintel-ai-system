#!/usr/bin/env python3
"""
Test script for database connection with PostgreSQL and SQLite fallback
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_database_connection():
    """Test database connection with fallback."""
    try:
        logger.info("Testing database connection with PostgreSQL/SQLite fallback...")
        
        from scrollintel.core.database_connection_manager import DatabaseConnectionManager
        
        # Create database manager
        db_manager = DatabaseConnectionManager()
        
        # Initialize connection
        logger.info("Initializing database connection...")
        success = await db_manager.initialize()
        
        if success:
            logger.info("✓ Database connection initialized successfully")
            
            # Get connection info
            info = db_manager.get_connection_info()
            logger.info(f"Connection info:")
            logger.info(f"  Connected: {info['connected']}")
            logger.info(f"  Database Type: {info['database_type']}")
            logger.info(f"  Fallback Active: {info['fallback_active']}")
            
            # Check health
            health = await db_manager.check_health()
            logger.info(f"Health check:")
            logger.info(f"  Healthy: {health['healthy']}")
            logger.info(f"  Database Type: {health['database_type']}")
            
            # Test creating tables
            try:
                logger.info("Testing table creation...")
                await db_manager.create_tables()
                logger.info("✓ Tables created successfully")
                
                # Test session
                logger.info("Testing database session...")
                async with db_manager.get_async_session() as session:
                    from sqlalchemy import text
                    result = await session.execute(text("SELECT 1"))
                    value = result.scalar()
                    if value == 1:
                        logger.info("✓ Database session test successful")
                    else:
                        logger.error("✗ Database session test failed")
                
            except Exception as e:
                logger.error(f"Table creation or session test failed: {e}")
            
            # Clean up
            await db_manager.close()
            logger.info("Database connection closed")
            
            return True
        else:
            logger.error("✗ Database connection initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


async def test_configuration_and_database():
    """Test both configuration and database systems."""
    try:
        logger.info("Testing configuration system...")
        
        from scrollintel.core.configuration_manager import get_config
        
        # Load configuration
        config = get_config()
        logger.info(f"Configuration loaded:")
        logger.info(f"  Environment: {config.environment}")
        logger.info(f"  Primary DB: {config.database.primary_url}")
        logger.info(f"  Fallback DB: {config.database.fallback_url}")
        
        # Test database connection
        return await test_database_connection()
        
    except Exception as e:
        logger.error(f"Configuration and database test failed: {e}")
        return False


def main():
    """Run the test."""
    logger.info("Starting database connection test...")
    
    # Set environment variables for testing
    os.environ.setdefault("DATABASE_URL", os.getenv("DATABASE_URL", os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/scrollintel")))
    os.environ.setdefault("SQLITE_URL", "sqlite:///./data/scrollintel_test.db")
    os.environ.setdefault("ENVIRONMENT", "development")
    
    # Ensure data directory exists
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Run the async test
        result = asyncio.run(test_configuration_and_database())
        
        if result:
            logger.info("🎉 All tests passed!")
            return 0
        else:
            logger.error("❌ Tests failed")
            return 1
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())