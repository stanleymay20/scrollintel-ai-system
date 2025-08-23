#!/usr/bin/env python3
"""
Simple ScrollIntel startup script
Runs the simplified API without complex dependencies
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Start the simple ScrollIntel API"""
    
    # Add the project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    print("Starting ScrollIntel Simple API...")
    print("API will be available at: http://localhost:8000")
    print("Dashboard will be available at: http://localhost:3000")
    print("This is a simplified version for development")
    print()
    
    try:
        # Start the simple API server
        import uvicorn
        
        # Try different ports if 8000 is busy
        ports_to_try = [8000, 8001, 8002, 8003, 8080]
        
        for port in ports_to_try:
            try:
                print(f"Trying to start server on port {port}...")
                uvicorn.run(
                    "scrollintel.api.simple_main:app",
                    host="127.0.0.1",  # Use localhost instead of 0.0.0.0 for Windows
                    port=port,
                    reload=True,
                    log_level="info"
                )
                break
            except OSError as e:
                if "WinError 10013" in str(e) or "Address already in use" in str(e):
                    print(f"Port {port} is busy, trying next port...")
                    continue
                else:
                    raise e
        else:
            print("Could not find an available port. Please close other applications using ports 8000-8003, 8080")
    except ImportError:
        print("uvicorn not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn[standard]"])
        
        # Try again with port fallback
        import uvicorn
        
        ports_to_try = [8000, 8001, 8002, 8003, 8080]
        
        for port in ports_to_try:
            try:
                print(f"Trying to start server on port {port}...")
                uvicorn.run(
                    "scrollintel.api.simple_main:app",
                    host="127.0.0.1",
                    port=port,
                    reload=True,
                    log_level="info"
                )
                break
            except OSError as e:
                if "WinError 10013" in str(e) or "Address already in use" in str(e):
                    print(f"Port {port} is busy, trying next port...")
                    continue
                else:
                    raise e
        else:
            print("Could not find an available port.")
    except Exception as e:
        print(f"Error starting API: {e}")
        print("Try running: pip install -r requirements.txt")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())