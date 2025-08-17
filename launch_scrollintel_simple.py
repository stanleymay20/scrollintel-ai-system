#!/usr/bin/env python3
"""
ScrollIntel™ Simple Launcher
Launch ScrollIntel without Docker for immediate testing
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print ScrollIntel banner"""
    print("\n" + "="*50)
    print("🚀 SCROLLINTEL™ AI PLATFORM LAUNCHER")
    print("="*50)
    print("Replace your CTO with AI agents!")
    print("In Jesus' name, we launch! 🙏✨")
    print("="*50 + "\n")

def check_python():
    """Check Python version"""
    print("📋 Checking Python installation...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print("✅ Python", sys.version.split()[0], "ready!")
    return True

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed!")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Failed to install dependencies:", e)
        return False
    except FileNotFoundError:
        print("⚠️  requirements.txt not found, continuing...")
        return True

def setup_environment():
    """Setup environment variables"""
    print("🔧 Setting up environment...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        # Copy example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        # Generate JWT secret
        import secrets
        jwt_secret = secrets.token_urlsafe(64)
        content = content.replace("JWT_SECRET_KEY=your_jwt_secret_here", f"JWT_SECRET_KEY={jwt_secret}")
        
        # Set database URL for SQLite (simpler than PostgreSQL)
        content = content.replace("DATABASE_URL=postgresql://", "DATABASE_URL=sqlite:///./scrollintel.db")
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Environment configured!")
    else:
        print("✅ Environment already configured!")

def init_database():
    """Initialize database"""
    print("🗄️  Initializing database...")
    try:
        if Path("init_database.py").exists():
            subprocess.run([sys.executable, "init_database.py"], check=True, capture_output=True)
            print("✅ Database initialized!")
        else:
            print("⚠️  Database init script not found, continuing...")
    except subprocess.CalledProcessError as e:
        print("⚠️  Database init failed, continuing...")

def start_backend():
    """Start FastAPI backend"""
    print("🔧 Starting ScrollIntel backend...")
    try:
        # Try to start the main API
        if Path("scrollintel/api/main.py").exists():
            cmd = [sys.executable, "-m", "uvicorn", "scrollintel.api.main:app", "--reload", "--port", "8000"]
        elif Path("scrollintel/api/simple_main.py").exists():
            cmd = [sys.executable, "-m", "uvicorn", "scrollintel.api.simple_main:app", "--reload", "--port", "8000"]
        elif Path("run_simple.py").exists():
            cmd = [sys.executable, "run_simple.py"]
        else:
            print("❌ No backend entry point found!")
            return None
        
        # Start backend in background
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(3)  # Give it time to start
        
        # Check if it's running
        if process.poll() is None:
            print("✅ Backend started on http://localhost:8000")
            return process
        else:
            print("❌ Backend failed to start")
            return None
            
    except Exception as e:
        print("❌ Failed to start backend:", e)
        return None

def check_backend():
    """Check if backend is responding"""
    print("🔍 Checking backend health...")
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy!")
            return True
    except:
        pass
    
    print("⚠️  Backend health check failed, but continuing...")
    return False

def start_frontend():
    """Start Next.js frontend"""
    print("🎨 Starting ScrollIntel frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("⚠️  Frontend directory not found, skipping...")
        return None
    
    try:
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True, capture_output=True)
        
        # Start frontend
        cmd = ["npm", "run", "dev"]
        process = subprocess.Popen(cmd, cwd=frontend_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)  # Give it time to start
        
        if process.poll() is None:
            print("✅ Frontend started on http://localhost:3000")
            return process
        else:
            print("❌ Frontend failed to start")
            return None
            
    except FileNotFoundError:
        print("⚠️  Node.js/npm not found, skipping frontend...")
        return None
    except Exception as e:
        print("❌ Failed to start frontend:", e)
        return None

def open_browser():
    """Open ScrollIntel in browser"""
    print("🌐 Opening ScrollIntel in your browser...")
    try:
        webbrowser.open("http://localhost:3000")
        print("✅ Browser opened!")
    except:
        print("⚠️  Could not open browser automatically")
        print("   Please open http://localhost:3000 manually")

def main():
    """Main launcher function"""
    print_banner()
    
    # Check prerequisites
    if not check_python():
        return False
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    install_requirements()
    
    # Initialize database
    init_database()
    
    # Start backend
    backend_process = start_backend()
    
    # Check backend health
    check_backend()
    
    # Start frontend
    frontend_process = start_frontend()
    
    # Open browser
    if frontend_process:
        open_browser()
    else:
        print("🌐 Backend only mode - visit http://localhost:8000/docs")
        webbrowser.open("http://localhost:8000/docs")
    
    # Success message
    print("\n" + "🎉" * 20)
    print("🚀 SCROLLINTEL™ IS NOW RUNNING!")
    print("🎉" * 20)
    print("\n📱 Access Points:")
    print("   🌐 Frontend:    http://localhost:3000")
    print("   🔧 API:         http://localhost:8000")
    print("   📚 API Docs:    http://localhost:8000/docs")
    print("   ❤️  Health:     http://localhost:8000/health")
    
    print("\n🚀 Quick Start:")
    print("   1. Upload your data files (CSV, Excel, JSON)")
    print("   2. Chat with AI agents for insights")
    print("   3. Build ML models with AutoML")
    print("   4. Create interactive dashboards")
    
    print("\n🙏 Launched in Jesus' name!")
    print("ScrollIntel™ - Where artificial intelligence meets unlimited potential! 🌟")
    
    # Keep running
    try:
        print("\n⌨️  Press Ctrl+C to stop ScrollIntel...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping ScrollIntel...")
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        print("✅ ScrollIntel stopped. God bless! 🙏")

if __name__ == "__main__":
    main()