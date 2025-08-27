#!/usr/bin/env python3
"""
Quick Fix Deployment Script for ScrollIntel
Fixes common deployment issues and gets ScrollIntel running fast.
"""

import os
import sys
import subprocess
import time
import json
import requests
from pathlib import Path

def run_command(cmd, check=True, capture_output=False):
    """Run a command and handle errors."""
    print(f"Running: {cmd}")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)
            return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        if capture_output and e.stdout:
            print(f"stdout: {e.stdout}")
        if capture_output and e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def check_docker():
    """Check if Docker is running."""
    try:
        result = run_command("docker --version", capture_output=True)
        if result:
            print(f"✅ Docker found: {result}")
            return True
    except:
        pass
    
    print("❌ Docker not found or not running")
    print("Please install Docker and make sure it's running")
    return False

def check_port(port):
    """Check if a port is available."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def stop_existing_containers():
    """Stop any existing ScrollIntel containers."""
    print("🛑 Stopping existing containers...")
    run_command("docker-compose down", check=False)
    run_command("docker stop $(docker ps -q --filter name=scrollintel)", check=False)

def create_minimal_env():
    """Create a minimal .env file for quick deployment."""
    env_content = """# ScrollIntel Quick Deploy Configuration
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=quick-deploy-secret-key-change-in-production
DATABASE_URL=postgresql://scrollintel:scrollintel123@db:5432/scrollintel_dev
REDIS_URL=redis://redis:6379/0

# API Keys (add your own)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Database
POSTGRES_DB=scrollintel_dev
POSTGRES_USER=scrollintel
POSTGRES_PASSWORD=scrollintel123

# Security
CORS_ORIGINS=*
ALLOWED_HOSTS=*

# Performance
WORKERS=2
MAX_CONNECTIONS=100
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print("✅ Created minimal .env file")

def create_quick_docker_compose():
    """Create a simplified docker-compose for quick deployment."""
    compose_content = """version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://scrollintel:scrollintel123@db:5432/scrollintel_dev
      - REDIS_URL=redis://redis:6379/0
      - ENVIRONMENT=development
      - DEBUG=true
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: scrollintel_dev
      POSTGRES_USER: scrollintel
      POSTGRES_PASSWORD: scrollintel123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U scrollintel -d scrollintel_dev"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  postgres_data:
"""
    
    with open('docker-compose.quick.yml', 'w') as f:
        f.write(compose_content)
    print("✅ Created quick docker-compose.yml")

def wait_for_service(url, timeout=120):
    """Wait for a service to be ready."""
    print(f"⏳ Waiting for service at {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Service ready at {url}")
                return True
        except:
            pass
        
        time.sleep(2)
        print(".", end="", flush=True)
    
    print(f"\n❌ Service at {url} not ready after {timeout}s")
    return False

def main():
    """Main deployment function."""
    print("🚀 ScrollIntel Quick Fix Deployment")
    print("=" * 50)
    
    # Check prerequisites
    if not check_docker():
        sys.exit(1)
    
    # Check ports
    if not check_port(8000):
        print("❌ Port 8000 is already in use")
        print("Please stop the service using port 8000 or use a different port")
        sys.exit(1)
    
    # Stop existing containers
    stop_existing_containers()
    
    # Create configuration files
    create_minimal_env()
    create_quick_docker_compose()
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    print("\n🔨 Building and starting services...")
    
    # Build and start services
    if not run_command("docker-compose -f docker-compose.quick.yml up --build -d"):
        print("❌ Failed to start services")
        sys.exit(1)
    
    print("\n⏳ Waiting for services to be ready...")
    
    # Wait for database
    print("Waiting for database...")
    time.sleep(10)
    
    # Wait for backend
    if wait_for_service("http://localhost:8000/"):
        print("\n🎉 ScrollIntel is now running!")
        print("\n📍 Access Points:")
        print("   • Main API: http://localhost:8000/")
        print("   • API Docs: http://localhost:8000/docs")
        print("   • Health Check: http://localhost:8000/health")
        print("   • System Status: http://localhost:8000/status")
        
        print("\n🔧 Management Commands:")
        print("   • View logs: docker-compose -f docker-compose.quick.yml logs -f")
        print("   • Stop services: docker-compose -f docker-compose.quick.yml down")
        print("   • Restart: docker-compose -f docker-compose.quick.yml restart")
        
        print("\n⚠️  Next Steps:")
        print("   1. Add your API keys to .env file")
        print("   2. Test the endpoints at http://localhost:8000/docs")
        print("   3. Check logs if you encounter issues")
        
        # Test the API
        try:
            response = requests.get("http://localhost:8000/")
            if response.status_code == 200:
                print("\n✅ API test successful!")
                data = response.json()
                print(f"   Service: {data.get('data', {}).get('name', 'Unknown')}")
                print(f"   Version: {data.get('data', {}).get('version', 'Unknown')}")
            else:
                print(f"\n⚠️  API test returned status {response.status_code}")
        except Exception as e:
            print(f"\n⚠️  API test failed: {e}")
    
    else:
        print("\n❌ Services failed to start properly")
        print("\n🔍 Troubleshooting:")
        print("   • Check logs: docker-compose -f docker-compose.quick.yml logs")
        print("   • Check container status: docker-compose -f docker-compose.quick.yml ps")
        print("   • Try rebuilding: docker-compose -f docker-compose.quick.yml up --build --force-recreate")

if __name__ == "__main__":
    main()