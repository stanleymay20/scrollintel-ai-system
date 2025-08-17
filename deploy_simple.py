#!/usr/bin/env python3
"""
Simple ScrollIntel Deployment Script
Deploy ScrollIntel using Docker Compose on Windows
"""

import os
import sys
import subprocess
import time
import json

def run_command(command, shell=True):
    """Run command and return result"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(
            command,
            shell=shell,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ Success: {command}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {command}")
        print(f"Error: {e.stderr}")
        return None

def create_production_env():
    """Create production environment file"""
    env_content = """# ScrollIntel Production Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=scrollintel
POSTGRES_USER=postgres
POSTGRES_PASSWORD=scrollintel_secure_2024

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Security
JWT_SECRET_KEY=super_secure_jwt_secret_key_2024_scrollintel_production

# API Keys
OPENAI_API_KEY=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

# Application
API_HOST=0.0.0.0
API_PORT=8000
BUILD_TARGET=production
NODE_ENV=production
"""
    
    with open(".env.production", "w") as f:
        f.write(env_content)
    
    # Also update main .env
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Production environment configured")

def deploy_with_docker():
    """Deploy using Docker Compose"""
    print("🚀 Starting ScrollIntel Docker Deployment...")
    
    # Create production environment
    create_production_env()
    
    # Stop any existing containers
    print("🛑 Stopping existing containers...")
    run_command("docker-compose down")
    
    # Build images
    print("🔨 Building Docker images...")
    if not run_command("docker-compose build"):
        print("❌ Failed to build images")
        return False
    
    # Start services
    print("🚀 Starting services...")
    if not run_command("docker-compose up -d"):
        print("❌ Failed to start services")
        return False
    
    # Wait for services
    print("⏳ Waiting for services to start...")
    time.sleep(30)
    
    # Check status
    print("📊 Checking service status...")
    run_command("docker-compose ps")
    
    return True

def deploy_simple_backend():
    """Deploy just the backend without Docker"""
    print("🚀 Starting Simple Backend Deployment...")
    
    # Create production environment
    create_production_env()
    
    # Install dependencies
    print("📦 Installing dependencies...")
    if not run_command("pip install -r requirements.txt"):
        print("❌ Failed to install dependencies")
        return False
    
    # Start the backend
    print("🚀 Starting ScrollIntel backend...")
    
    # Use the simple main that we know works
    try:
        print("✅ Backend starting on http://localhost:8000")
        print("📚 API docs available at http://localhost:8000/docs")
        print("🏥 Health check at http://localhost:8000/health")
        print("\n🎉 ScrollIntel is ready!")
        print("Press Ctrl+C to stop")
        
        # Start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "scrollintel.api.simple_main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 ScrollIntel stopped")
    
    return True

def main():
    """Main deployment function"""
    print("🌟 ScrollIntel Simple Deployment")
    print("=" * 50)
    
    # Check if Docker is available
    docker_available = run_command("docker --version") is not None
    
    if docker_available:
        print("🐳 Docker detected - attempting Docker deployment...")
        if deploy_with_docker():
            print("\n🎉 ScrollIntel deployed successfully with Docker!")
            print_access_info()
            return True
        else:
            print("⚠️ Docker deployment failed, falling back to simple deployment...")
    
    print("🔧 Starting simple backend deployment...")
    if deploy_simple_backend():
        print("\n🎉 ScrollIntel deployed successfully!")
        return True
    else:
        print("\n❌ Deployment failed")
        return False

def print_access_info():
    """Print access information"""
    print("\n🎉 ScrollIntel is now running!")
    print("=" * 50)
    print("🔗 Access Points:")
    print("   📱 Frontend:     http://localhost:3000")
    print("   🔧 Backend API:  http://localhost:8000")
    print("   📚 API Docs:     http://localhost:8000/docs")
    print("   ❤️  Health:      http://localhost:8000/health")
    print("\n📊 Management:")
    print("   🐳 View logs:    docker-compose logs -f")
    print("   🛑 Stop:         docker-compose down")
    print("   🔄 Restart:      docker-compose restart")
    print("\n🚀 ScrollIntel is ready for production use!")

if __name__ == "__main__":
    main()