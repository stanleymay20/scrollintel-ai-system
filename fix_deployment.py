#!/usr/bin/env python3
"""
ScrollIntel Deployment Fix Script
Automatically fixes common deployment issues
"""

import subprocess
import os
import time
import sys
from pathlib import Path

def run_command(cmd, description, check_output=False):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        if check_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"✅ {description} - Success")
                return result.stdout
            else:
                print(f"❌ {description} - Failed: {result.stderr}")
                return None
        else:
            result = subprocess.run(cmd, shell=True, timeout=60)
            if result.returncode == 0:
                print(f"✅ {description} - Success")
                return True
            else:
                print(f"❌ {description} - Failed")
                return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - Timeout")
        return False
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        return False

def check_and_create_env():
    """Check and create environment file"""
    if not os.path.exists('.env'):
        print("🔧 Creating .env file...")
        env_content = """# ScrollIntel Environment Configuration
DATABASE_URL=postgresql://scrollintel:scrollintel123@localhost:5432/scrollintel
SECRET_KEY=your-secret-key-here-change-in-production
API_KEY=your-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
ENVIRONMENT=development
DEBUG=true
PORT=8000
HOST=0.0.0.0
FRONTEND_PORT=3000
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("✅ .env file exists")

def fix_port_mapping():
    """Fix port mapping in docker-compose"""
    print("🔧 Checking Docker services...")
    
    # Check if services are running on wrong ports
    result = run_command("docker ps --format 'table {{.Names}}\t{{.Ports}}'", "Checking running containers", check_output=True)
    if result:
        print("Current container ports:")
        print(result)
    
    # Stop and restart with correct ports
    run_command("docker-compose down", "Stopping services")
    time.sleep(2)
    run_command("docker-compose up -d", "Starting services with correct ports")

def build_frontend():
    """Build frontend if needed"""
    if os.path.exists("frontend"):
        print("🔧 Building frontend...")
        os.chdir("frontend")
        
        # Install dependencies
        if os.path.exists("package.json"):
            run_command("npm install", "Installing frontend dependencies")
            run_command("npm run build", "Building frontend")
        
        os.chdir("..")
    else:
        print("⚠️  Frontend directory not found")

def start_services():
    """Start all services"""
    print("🚀 Starting ScrollIntel services...")
    
    # Start backend
    if os.path.exists("scrollintel"):
        print("🔧 Starting backend service...")
        # Kill any existing processes on port 8000
        run_command("pkill -f 'uvicorn.*8000'", "Stopping existing backend", check_output=True)
        time.sleep(2)
        
        # Start backend in background
        backend_cmd = "python -m uvicorn scrollintel.api.main:app --host 0.0.0.0 --port 8000 --reload"
        subprocess.Popen(backend_cmd, shell=True)
        print("✅ Backend starting on port 8000")
    
    # Start frontend
    if os.path.exists("frontend"):
        print("🔧 Starting frontend service...")
        os.chdir("frontend")
        
        # Kill any existing processes on port 3000
        run_command("pkill -f 'node.*3000'", "Stopping existing frontend", check_output=True)
        time.sleep(2)
        
        # Start frontend in background
        if os.path.exists("package.json"):
            subprocess.Popen("npm start", shell=True)
            print("✅ Frontend starting on port 3000")
        
        os.chdir("..")

def wait_for_services():
    """Wait for services to start"""
    print("⏳ Waiting for services to start...")
    time.sleep(10)

def main():
    """Main fix routine"""
    print("🔧 ScrollIntel Deployment Fix")
    print("=" * 50)
    
    # Step 1: Create environment file
    check_and_create_env()
    
    # Step 2: Fix Docker services
    fix_port_mapping()
    
    # Step 3: Build frontend
    build_frontend()
    
    # Step 4: Start services
    start_services()
    
    # Step 5: Wait for services
    wait_for_services()
    
    # Step 6: Run deployment check
    print("\n🔍 Running deployment check...")
    result = subprocess.run([sys.executable, "check_deployment_status.py"], capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode == 0:
        print("\n🎉 Deployment fixed successfully!")
        print("\n🔗 Access your application:")
        print("🌐 Frontend: http://localhost:3000")
        print("🔧 Backend API: http://localhost:8000/docs")
        print("💾 Database: localhost:5432")
    else:
        print("\n⚠️  Some issues remain. Check the output above.")
        print("\n🔧 Manual steps you can try:")
        print("1. Check Docker: docker-compose logs -f")
        print("2. Restart services: docker-compose restart")
        print("3. Rebuild: docker-compose up --build")

if __name__ == "__main__":
    main()