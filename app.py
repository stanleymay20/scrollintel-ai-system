#!/usr/bin/env python3
"""
ScrollIntel Application Entry Point for Railway Deployment
This file serves as the main entry point for the application.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables for production
os.environ.setdefault("ENVIRONMENT", "production")
os.environ.setdefault("LOG_LEVEL", "info")

def main():
    """Main application entry point"""
    try:
        # Import the FastAPI app
        from scrollintel.api.main import app
        
        # Get port from environment (Railway sets this)
        port = int(os.environ.get("PORT", 8000))
        host = os.environ.get("HOST", "0.0.0.0")
        
        print(f"Starting ScrollIntel on {host}:{port}")
        
        # Run the application
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()