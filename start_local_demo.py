#!/usr/bin/env python3
"""
Local demo startup script for ScrollIntel with SQLite database.
"""

import asyncio
import logging
import os
import sys
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_local_env():
    """Setup local environment with SQLite."""
    # Override database URL for local development
    os.environ['DATABASE_URL'] = 'sqlite:///./scrollintel_local.db'
    os.environ['ENVIRONMENT'] = 'development'
    os.environ['DEBUG'] = 'true'
    
    logger.info("Using SQLite database for local development")

async def init_database():
    """Initialize SQLite database."""
    try:
        from scrollintel.models.database_utils import DatabaseManager
        
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Create tables if they don't exist
        await db_manager.create_tables()
        logger.info("SQLite database initialized successfully")
        
        return db_manager
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Continue anyway for demo purposes
        return None

def start_frontend():
    """Start the Next.js frontend in a separate process."""
    try:
        logger.info("Starting frontend server...")
        
        # Change to frontend directory and start Next.js
        frontend_dir = project_root / "frontend"
        
        if not frontend_dir.exists():
            logger.error("Frontend directory not found")
            return None
            
        # Start Next.js dev server
        process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        logger.info("Frontend server starting on http://localhost:3000")
        return process
        
    except Exception as e:
        logger.error(f"Failed to start frontend: {e}")
        return None

async def start_api_server():
    """Start the FastAPI server with SQLite."""
    try:
        # Initialize database first
        db_manager = await init_database()
        
        logger.info("Starting ScrollIntel API server with SQLite...")
        
        # Import app after database is initialized
        from scrollintel.api.main import app
        
        # Start server
        import uvicorn
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point."""
    print("🚀 Starting ScrollIntel Local Demo...")
    print("=" * 50)
    
    # Setup local environment
    setup_local_env()
    
    # Start frontend
    frontend_process = start_frontend()
    
    if frontend_process:
        print("✅ Frontend starting at: http://localhost:3000")
    else:
        print("❌ Frontend failed to start")
    
    print("✅ API starting at: http://localhost:8000")
    print("=" * 50)
    print("🌐 Open http://localhost:3000 in your browser to access ScrollIntel!")
    print("📚 API docs available at: http://localhost:8000/docs")
    print("=" * 50)
    
    try:
        # Run the API server
        asyncio.run(start_api_server())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if frontend_process:
            frontend_process.terminate()
    except Exception as e:
        logger.error(f"Server failed: {e}")
        if frontend_process:
            frontend_process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()