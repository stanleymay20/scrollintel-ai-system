#!/usr/bin/env python3
"""
Railway startup script for ScrollIntel
Handles database initialization and application startup
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def initialize_database():
    """Initialize database if needed"""
    try:
        # Check if we have a database URL
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            logger.info("Database URL found, initializing database...")
            
            # Try to run database migrations
            try:
                from alembic.config import Config
                from alembic import command
                
                alembic_cfg = Config("alembic.ini")
                command.upgrade(alembic_cfg, "head")
                logger.info("Database migrations completed")
            except Exception as e:
                logger.warning(f"Database migration failed: {e}")
                
                # Try basic database initialization
                try:
                    from scrollintel.models.database import init_db
                    await init_db()
                    logger.info("Basic database initialization completed")
                except Exception as e2:
                    logger.warning(f"Basic database initialization failed: {e2}")
        else:
            logger.info("No database URL found, using SQLite fallback")
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

def main():
    """Main startup function"""
    logger.info("Starting ScrollIntel on Railway...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Railway Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'unknown')}")
    logger.info(f"Port: {os.getenv('PORT', '8000')}")
    
    # Initialize database
    try:
        asyncio.run(initialize_database())
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
    
    # Import and run the application
    try:
        import uvicorn
        from scrollintel.api.main import app
        
        port = int(os.getenv("PORT", 8000))
        host = "0.0.0.0"
        
        logger.info(f"Starting ScrollIntel server on {host}:{port}")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            loop="asyncio"
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()