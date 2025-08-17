#!/usr/bin/env python3
"""
Simple startup script for ScrollIntel that handles async database and memory optimization.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def init_database():
    """Initialize database with async support."""
    try:
        from scrollintel.models.database_utils import DatabaseManager
        from scrollintel.core.config import get_settings
        
        settings = get_settings()
        logger.info(f"Using database: {settings.get('database_url', 'sqlite:///./scrollintel.db')}")
        
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Create tables if they don't exist
        await db_manager.create_tables()
        logger.info("Database initialized successfully")
        
        return db_manager
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return None

async def start_api_server():
    """Start the FastAPI server with proper async setup."""
    try:
        import uvicorn
        
        # Initialize database first
        db_manager = await init_database()
        if not db_manager:
            logger.error("Cannot start server without database")
            return
        
        logger.info("Starting ScrollIntel API server...")
        
        # Import app after database is initialized
        from scrollintel.api.main import app
        
        # Start server
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload to prevent memory issues
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """Main entry point."""
    logger.info("🚀 Starting ScrollIntel...")
    
    try:
        # Run the async startup
        asyncio.run(start_api_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()