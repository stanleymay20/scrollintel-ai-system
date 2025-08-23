#!/usr/bin/env python3
"""
Start ScrollIntel with both backend and frontend
Simple development setup
"""

import sys
import os
import subprocess
import time
import threading
from pathlib import Path

def start_backend():
    """Start the backend API"""
    print("🔧 Starting ScrollIntel Backend API...")
    
    try:
        # Change to project root
        os.chdir(Path(__file__).parent)
        
        # Start the simple API
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "scrollintel.api.simple_main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except Exception as e:
        print(f"❌ Backend error: {e}")

def start_frontend():
    """Start the frontend development server"""
    print("🎨 Starting ScrollIntel Frontend...")
    
    try:
        # Change to frontend directory
        frontend_dir = Path(__file__).parent / "frontend"
        os.chdir(frontend_dir)
        
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], check=True)
        
        # Start the development server
        subprocess.run(["npm", "run", "dev"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend error: {e}")
        print("💡 Make sure Node.js and npm are installed")
    except Exception as e:
        print(f"❌ Frontend error: {e}")

def main():
    """Main function to start both services"""
    print("🚀 Starting ScrollIntel Development Environment")
    print("=" * 50)
    print("📍 Backend API: http://localhost:8000")
    print("📍 Frontend App: http://localhost:3000")
    print("=" * 50)
    print()
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait a moment for backend to start
    print("⏳ Waiting for backend to start...")
    time.sleep(3)
    
    # Start frontend (this will block)
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ScrollIntel...")
        return 0

if __name__ == "__main__":
    sys.exit(main())