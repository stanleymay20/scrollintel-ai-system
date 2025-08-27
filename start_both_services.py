#!/usr/bin/env python3
"""
Start both frontend and backend services for ScrollIntel
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def start_backend():
    """Start the backend API server"""
    print("🚀 Starting ScrollIntel Backend...")
    
    # Change to project root
    os.chdir(Path(__file__).parent)
    
    try:
        # Start the simple backend
        subprocess.run([
            sys.executable, "run_simple.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
    except Exception as e:
        print(f"❌ Backend error: {e}")

def start_frontend():
    """Start the frontend development server"""
    print("🎨 Starting ScrollIntel Frontend...")
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    
    try:
        # Check if node_modules exists, if not install dependencies
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], check=True)
        
        # Start the frontend
        subprocess.run(["npm", "run", "dev"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Frontend error: {e}")

def main():
    """Main function to start both services"""
    print("🌟 ScrollIntel - Starting Frontend & Backend")
    print("=" * 50)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Give backend time to start
    print("⏳ Waiting for backend to initialize...")
    time.sleep(3)
    
    # Start frontend in main thread
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Services stopped by user")
    
    print("\n✅ ScrollIntel services stopped")

if __name__ == "__main__":
    main()