#!/usr/bin/env python3
"""
ScrollIntel Application Starter
Starts the ScrollIntel application with proper database setup
"""

import os
import sys
import time
import subprocess
import sqlite3
from pathlib import Path

def setup_sqlite_database():
    """Set up SQLite database for local development"""
    print("🔧 Setting up SQLite database for local development...")
    
    # Update environment to use SQLite
    db_path = Path("scrollintel.db")
    sqlite_url = f"sqlite:///{db_path.absolute()}"
    
    # Update .env file
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Replace PostgreSQL URL with SQLite
        content = content.replace(
            'DATABASE_URL=postgresql://postgres:scrollintel_secure_2024@postgres:5432/scrollintel',
            f'DATABASE_URL={sqlite_url}'
        )
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated DATABASE_URL to use SQLite: {sqlite_url}")
    
    # Create basic SQLite database
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create a simple test table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert test data
        cursor.execute("INSERT OR IGNORE INTO test_table (id, name) VALUES (1, 'ScrollIntel Test')")
        
        conn.commit()
        conn.close()
        
        print("✅ SQLite database created and initialized")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create SQLite database: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'pydantic',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    return True

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting ScrollIntel backend server...")
    
    try:
        # Try to start the main application
        cmd = [
            sys.executable, '-m', 'uvicorn',
            'scrollintel.api.main:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--reload'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Start the server
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor the output for a few seconds
        start_time = time.time()
        while time.time() - start_time < 10:  # Wait up to 10 seconds
            if process.poll() is not None:
                # Process has terminated
                output, _ = process.communicate()
                print(f"❌ Server failed to start:\n{output}")
                return False
            
            time.sleep(0.5)
        
        print("✅ Backend server started successfully!")
        print("🌐 API available at: http://localhost:8000")
        print("📚 API docs available at: http://localhost:8000/docs")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start backend server: {e}")
        return False

def start_frontend():
    """Start the Next.js frontend server"""
    print("🎨 Starting ScrollIntel frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("⚠️ Frontend directory not found, skipping frontend startup")
        return None
    
    try:
        # Check if node_modules exists
        node_modules = frontend_dir / "node_modules"
        if not node_modules.exists():
            print("📦 Installing frontend dependencies...")
            subprocess.check_call(['npm', 'install'], cwd=frontend_dir)
        
        # Start the frontend
        print("🚀 Starting frontend development server...")
        process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        print("✅ Frontend server starting...")
        print("🌐 Frontend available at: http://localhost:3000")
        
        return process
        
    except FileNotFoundError:
        print("⚠️ Node.js/npm not found, skipping frontend startup")
        return None
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def test_api_endpoints():
    """Test basic API endpoints"""
    print("🧪 Testing API endpoints...")
    
    import requests
    import time
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    base_url = "http://localhost:8000"
    
    endpoints_to_test = [
        "/",
        "/health",
        "/docs"
    ]
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint} - OK")
            else:
                print(f"⚠️ {endpoint} - Status {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint} - Failed: {e}")

def main():
    """Main function to start ScrollIntel"""
    print("=" * 60)
    print("🚀 ScrollIntel Application Starter")
    print("=" * 60)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return False
    
    # Step 2: Setup database
    if not setup_sqlite_database():
        print("❌ Database setup failed")
        return False
    
    # Step 3: Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Backend startup failed")
        return False
    
    # Step 4: Test API
    try:
        test_api_endpoints()
    except Exception as e:
        print(f"⚠️ API testing failed: {e}")
    
    # Step 5: Start frontend (optional)
    frontend_process = start_frontend()
    
    print("\n" + "=" * 60)
    print("🎉 ScrollIntel is now running!")
    print("=" * 60)
    print("📊 Backend API: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    if frontend_process:
        print("🎨 Frontend UI: http://localhost:3000")
    print("\n💡 Press Ctrl+C to stop the servers")
    print("=" * 60)
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ScrollIntel...")
        
        if backend_process:
            backend_process.terminate()
            backend_process.wait()
            print("✅ Backend server stopped")
        
        if frontend_process:
            frontend_process.terminate()
            frontend_process.wait()
            print("✅ Frontend server stopped")
        
        print("👋 ScrollIntel stopped successfully!")

if __name__ == "__main__":
    main()