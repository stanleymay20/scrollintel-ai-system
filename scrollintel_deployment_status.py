#!/usr/bin/env python3
"""
ScrollIntel Deployment Status and Management
"""

import requests
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def check_backend_status():
    """Check if backend is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data
        return False, None
    except:
        return False, None

def check_frontend_status():
    """Check if frontend is accessible"""
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_backend():
    """Start the backend server"""
    print("🚀 Starting ScrollIntel Backend...")
    
    try:
        # Start backend in background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "scrollintel.api.gateway:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], env={**dict(os.environ), 
                'ENVIRONMENT': 'development',
                'DEBUG': 'true',
                'SKIP_REDIS': 'true',
                'DATABASE_URL': 'sqlite:///./scrollintel.db'})
        
        print("⏳ Waiting for backend to start...")
        time.sleep(5)
        
        # Check if it started successfully
        backend_running, _ = check_backend_status()
        if backend_running:
            print("✅ Backend started successfully!")
            return process
        else:
            print("❌ Backend failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the frontend server"""
    print("🌐 Starting ScrollIntel Frontend...")
    
    try:
        process = subprocess.Popen([
            sys.executable, "start_frontend_simple.py"
        ])
        
        print("⏳ Waiting for frontend to start...")
        time.sleep(3)
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def show_status():
    """Show current deployment status"""
    print("📊 ScrollIntel Deployment Status")
    print("=" * 40)
    
    # Check backend
    backend_running, health_data = check_backend_status()
    if backend_running:
        print("✅ Backend: Running (http://localhost:8000)")
        if health_data:
            print(f"   Services: {health_data.get('services', {})}")
            print(f"   Metrics: {health_data.get('metrics', {})}")
    else:
        print("❌ Backend: Not running")
    
    # Check frontend
    frontend_running = check_frontend_status()
    if frontend_running:
        print("✅ Frontend: Running (http://localhost:3000)")
    else:
        print("❌ Frontend: Not running")
    
    print("\n🌐 Access URLs:")
    if backend_running:
        print("   • API Health: http://localhost:8000/health")
        print("   • API Docs: http://localhost:8000/docs")
        print("   • Interactive API: http://localhost:8000/redoc")
    
    if frontend_running:
        print("   • Web Interface: http://localhost:3000/simple_frontend.html")
    
    return backend_running, frontend_running

def main():
    """Main deployment management function"""
    
    print("🚀 ScrollIntel Deployment Manager")
    print("=" * 50)
    
    while True:
        backend_running, frontend_running = show_status()
        
        print("\n🎯 Options:")
        print("1. Start Backend")
        print("2. Start Frontend") 
        print("3. Start Both")
        print("4. Open Web Interface")
        print("5. Open API Docs")
        print("6. Refresh Status")
        print("7. Exit")
        
        choice = input("\nChoose option (1-7): ").strip()
        
        if choice == "1":
            if not backend_running:
                start_backend()
            else:
                print("✅ Backend is already running")
        
        elif choice == "2":
            if not frontend_running:
                start_frontend()
            else:
                print("✅ Frontend is already running")
        
        elif choice == "3":
            if not backend_running:
                backend_process = start_backend()
            if not frontend_running:
                frontend_process = start_frontend()
            
            print("\n🎉 ScrollIntel is starting up!")
            print("⏳ Please wait a moment for all services to initialize...")
            time.sleep(5)
        
        elif choice == "4":
            if frontend_running:
                webbrowser.open("http://localhost:3000/simple_frontend.html")
                print("🌐 Opening web interface...")
            else:
                print("❌ Frontend is not running. Start it first.")
        
        elif choice == "5":
            if backend_running:
                webbrowser.open("http://localhost:8000/docs")
                print("📚 Opening API documentation...")
            else:
                print("❌ Backend is not running. Start it first.")
        
        elif choice == "6":
            print("🔄 Refreshing status...")
            continue
        
        elif choice == "7":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")
        
        print("\n" + "-" * 50)

if __name__ == "__main__":
    import os
    main()