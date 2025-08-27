#!/usr/bin/env python3
"""
Railway startup script for ScrollIntel
"""

import os
import sys
import uvicorn

def main():
    """Start the ScrollIntel API for Railway deployment"""
    
    # Get port from environment (Railway sets this)
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting ScrollIntel API on port {port}")
    print(f"Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'production')}")
    
    # Start the server
    uvicorn.run(
        "scrollintel.api.railway_main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()