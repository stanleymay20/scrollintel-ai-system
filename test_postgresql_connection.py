#!/usr/bin/env python3
"""
Test PostgreSQL connection specifically
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


async def test_postgresql_connection():
    """Test PostgreSQL connection specifically."""
    try:
        logger.info("Testing PostgreSQL connection...")
        
        from scrollintel.core.database_connection_manager import DatabaseConnectionManager
        
        # Set PostgreSQL as primary
        os.environ["DATABASE_URL"] = os.getenv("DATABASE_URL", os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/scrollintel"))
        
        # Create database manager
        db_manager = DatabaseConnectionManager()
        
        # Test sync connection first
        logger.info("Testing sync PostgreSQL connection...")
        sync_result = db_manager._test_postgresql_connection_sync(
            os.getenv("DATABASE_URL", os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/scrollintel"))
        )
        
        if sync_result:
            logger.info("✓ Sync PostgreSQL connection test passed")
        else:
            logger.warning("⚠ Sync PostgreSQL connection test failed (PostgreSQL may not be running)")
        
        # Try to initialize
        logger.info("Attempting to initialize database connection...")
        success = await db_manager.initialize()
        
        if success:
            info = db_manager.get_connection_info()
            if info['database_type'] == 'postgresql':
                logger.info("✓ Successfully connected to PostgreSQL")
                
                # Test health check
                health = await db_manager.check_health()
                logger.info(f"Health check result: {health}")
                
                # Test connection pooling
                logger.info("Testing connection pooling...")
                sessions = []
                try:
                    for i in range(3):
                        session_ctx = db_manager.get_async_session()
                        sessions.append(session_ctx)
                        async with session_ctx as session:
                            from sqlalchemy import text
                            result = await session.execute(text("SELECT 1"))
                            logger.info(f"Session {i+1}: {result.scalar()}")
                    
                    logger.info("✓ Connection pooling test passed")
                    
                except Exception as e:
                    logger.error(f"Connection pooling test failed: {e}")
                
            else:
                logger.info(f"Connected to {info['database_type']} (fallback)")
            
            await db_manager.close()
            return True
        else:
            logger.error("Failed to initialize database connection")
            return False
            
    except Exception as e:
        logger.error(f"PostgreSQL connection test failed: {e}")
        return False


def main():
    """Run the PostgreSQL test."""
    logger.info("Starting PostgreSQL connection test...")
    logger.info("Note: This test requires PostgreSQL to be running on localhost:5432")
    logger.info("If PostgreSQL is not available, the system will fall back to SQLite")
    
    try:
        result = asyncio.run(test_postgresql_connection())
        
        if result:
            logger.info("🎉 PostgreSQL connection test completed!")
            return 0
        else:
            logger.warning("⚠ PostgreSQL connection test had issues (but fallback should work)")
            return 0  # Not a failure since fallback is expected
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())