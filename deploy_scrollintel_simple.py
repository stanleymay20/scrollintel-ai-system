#!/usr/bin/env python3
"""
ScrollIntel Simple Deployment Script
Deploy ScrollIntel without Docker for immediate testing
"""

import os
import subprocess
import time
import sys
import threading
from pathlib import Path

def run_backend():
    """Run the backend server"""
    print("🚀 Starting ScrollIntel Backend...")
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        'ENVIRONMENT': 'development',
        'DEBUG': 'true',
        'API_HOST': '0.0.0.0',
        'API_PORT': '8000',
        'SKIP_REDIS': 'true',  # Skip Redis for simple deployment
        'DATABASE_URL': 'sqlite:///./scrollintel.db'  # Use SQLite for simplicity
    })
    
    try:
        # Start the backend
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "scrollintel.api.gateway:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], env=env, cwd=".")
    except KeyboardInterrupt:
        print("🛑 Backend stopped")

def run_frontend():
    """Run the frontend server"""
    print("🎨 Starting ScrollIntel Frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return
    
    try:
        # Install dependencies if needed
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
        
        # Start the frontend
        subprocess.run(["npm", "run", "dev"], cwd=frontend_dir)
    except KeyboardInterrupt:
        print("🛑 Frontend stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend failed to start: {e}")

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check Python
    try:
        import uvicorn
        print("✅ Python and uvicorn available")
    except ImportError:
        print("❌ uvicorn not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "uvicorn"], check=True)
    
    # Check Node.js
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js available: {result.stdout.strip()}")
        else:
            print("❌ Node.js not found. Please install Node.js from https://nodejs.org/")
            return False
    except FileNotFoundError:
        print("❌ Node.js not found. Please install Node.js from https://nodejs.org/")
        return False
    
    return True

def create_simple_database():
    """Create a simple SQLite database"""
    print("🗄️ Setting up database...")
    
    try:
        # Create database initialization script
        init_script = """
import os
import sys
sys.path.append('.')

from scrollintel.models.database import Base
from sqlalchemy import create_engine

# Create SQLite database
engine = create_engine('sqlite:///./scrollintel.db')
Base.metadata.create_all(engine)
print("✅ Database created successfully")
"""
        
        with open("init_simple_db.py", "w") as f:
            f.write(init_script)
        
        # Run the initialization
        subprocess.run([sys.executable, "init_simple_db.py"], check=True)
        
        # Clean up
        os.remove("init_simple_db.py")
        
        return True
    except Exception as e:
        print(f"⚠️ Database setup failed: {e}")
        print("📝 Will use in-memory database")
        return False

def main():
    """Main deployment function"""
    
    print("🚀 ScrollIntel Simple Deployment")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Setup database
    create_simple_database()
    
    print("\n🎯 Deployment Options:")
    print("1. Backend only (API server)")
    print("2. Frontend only (Web interface)")
    print("3. Full stack (Both backend and frontend)")
    
    choice = input("\nChoose deployment option (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting Backend Only...")
        run_backend()
    
    elif choice == "2":
        print("\n🎨 Starting Frontend Only...")
        print("⚠️ Make sure backend is running on http://localhost:8000")
        run_frontend()
    
    elif choice == "3":
        print("\n🚀 Starting Full Stack...")
        
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # Wait a bit for backend to start
        print("⏳ Waiting for backend to start...")
        time.sleep(5)
        
        # Start frontend
        try:
            run_frontend()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
    
    else:
        print("❌ Invalid choice")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Deployment stopped by user")
        sys.exit(0)