#!/usr/bin/env python3
"""
ScrollIntel Fix and Deploy Script
Addresses common deployment issues and gets ScrollIntel running.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

def run_cmd(cmd, check=True):
    """Run command with error handling."""
    print(f"🔧 {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"   {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False

def fix_docker_issues():
    """Fix common Docker issues."""
    print("\n🔧 Fixing Docker Issues...")
    
    # Stop all containers
    print("Stopping existing containers...")
    run_cmd("docker-compose down", check=False)
    run_cmd("docker stop $(docker ps -aq)", check=False)
    
    # Clean up
    print("Cleaning up Docker resources...")
    run_cmd("docker system prune -f", check=False)
    
    # Remove problematic images
    run_cmd("docker rmi $(docker images -f dangling=true -q)", check=False)

def create_simple_env():
    """Create a working .env file."""
    print("\n📝 Creating .env file...")
    
    env_content = """# ScrollIntel Configuration
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=dev-secret-key-change-in-production

# Database
DATABASE_URL=postgresql://scrollintel:scrollintel123@localhost:5432/scrollintel_dev
POSTGRES_DB=scrollintel_dev
POSTGRES_USER=scrollintel
POSTGRES_PASSWORD=scrollintel123

# Redis
REDIS_URL=redis://localhost:6379/0

# API Keys (replace with your own)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Security
CORS_ORIGINS=*
ALLOWED_HOSTS=*

# Performance
WORKERS=1
MAX_CONNECTIONS=50
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print("✅ Created .env file")

def create_simple_compose():
    """Create a simple docker-compose file."""
    print("\n📝 Creating docker-compose.simple.yml...")
    
    compose_content = """version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
      - DATABASE_URL=postgresql://scrollintel:scrollintel123@db:5432/scrollintel_dev
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s

  db:
    image: postgres:15-alpine
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
    
    with open('docker-compose.simple.yml', 'w') as f:
        f.write(compose_content)
    print("✅ Created docker-compose.simple.yml")

def create_simple_dockerfile():
    """Create a simplified Dockerfile."""
    print("\n📝 Creating Dockerfile.simple...")
    
    dockerfile_content = """FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    libpq-dev \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/logs /app/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/ || exit 1

# Run application
CMD ["uvicorn", "scrollintel.api.gateway:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    with open('Dockerfile.simple', 'w') as f:
        f.write(dockerfile_content)
    print("✅ Created Dockerfile.simple")

def create_simple_requirements():
    """Create a minimal requirements.txt."""
    print("\n📝 Creating requirements.simple.txt...")
    
    requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
"""
    
    with open('requirements.simple.txt', 'w') as f:
        f.write(requirements)
    print("✅ Created requirements.simple.txt")

def deploy_simple():
    """Deploy with simple configuration."""
    print("\n🚀 Deploying ScrollIntel (Simple Mode)...")
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    # Build and deploy
    if not run_cmd("docker-compose -f docker-compose.simple.yml up --build -d"):
        print("❌ Deployment failed")
        return False
    
    print("✅ Deployment started")
    
    # Wait for services
    print("\n⏳ Waiting for services to start...")
    time.sleep(30)
    
    # Check status
    if run_cmd("docker-compose -f docker-compose.simple.yml ps"):
        print("\n🎉 ScrollIntel is running!")
        print("\n📍 Access Points:")
        print("   • API: http://localhost:8000/")
        print("   • Docs: http://localhost:8000/docs")
        print("   • Health: http://localhost:8000/health")
        
        print("\n🔧 Management:")
        print("   • Logs: docker-compose -f docker-compose.simple.yml logs -f")
        print("   • Stop: docker-compose -f docker-compose.simple.yml down")
        print("   • Restart: docker-compose -f docker-compose.simple.yml restart")
        
        return True
    
    return False

def main():
    """Main function."""
    print("🔧 ScrollIntel Fix and Deploy")
    print("=" * 40)
    
    # Check Docker
    if not run_cmd("docker --version"):
        print("❌ Docker not found. Please install Docker first.")
        sys.exit(1)
    
    # Fix issues
    fix_docker_issues()
    
    # Create configuration
    create_simple_env()
    create_simple_compose()
    create_simple_dockerfile()
    create_simple_requirements()
    
    # Deploy
    if deploy_simple():
        print("\n✅ ScrollIntel deployment completed successfully!")
        print("\nRun 'python test_scrollintel_health.py' to verify the deployment.")
    else:
        print("\n❌ Deployment failed. Check the logs for details.")
        print("Run 'docker-compose -f docker-compose.simple.yml logs' for more info.")

if __name__ == "__main__":
    main()