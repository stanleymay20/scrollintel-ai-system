#!/usr/bin/env python3
"""
Script to start the ScrollIntel frontend development server
"""

import subprocess
import sys
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_frontend():
    """Start the Next.js frontend development server."""
    try:
        logger.info("🚀 Starting ScrollIntel Frontend Development Server...")
        
        # Change to frontend directory
        frontend_dir = os.path.join(os.getcwd(), 'frontend')
        
        if not os.path.exists(frontend_dir):
            logger.error("Frontend directory not found!")
            return False
        
        # Start the development server
        logger.info("Starting Next.js development server...")
        
        # Use subprocess to start the server
        process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info("Frontend server starting...")
        logger.info("Frontend will be available at: http://localhost:3000")
        logger.info("Press Ctrl+C to stop the server")
        
        # Wait a bit and check if process is running
        time.sleep(3)
        
        if process.poll() is None:
            logger.info("✅ Frontend server started successfully!")
            logger.info("🔗 Frontend URL: http://localhost:3000")
            return True
        else:
            stdout, stderr = process.communicate()
            logger.error(f"Frontend server failed to start:")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to start frontend server: {e}")
        return False

if __name__ == "__main__":
    if start_frontend():
        logger.info("Frontend server is running!")
    else:
        logger.error("Failed to start frontend server")
        sys.exit(1)