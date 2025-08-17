#!/usr/bin/env python3
"""
Fix Docker and System Issues for ScrollIntel
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_docker():
    """Check if Docker is running"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker not found")
            return False
    except FileNotFoundError:
        print("❌ Docker not installed")
        return False

def fix_docker_compose():
    """Fix Docker Compose configuration"""
    print("🔧 Fixing Docker Compose configuration...")
    
    # Read current docker-compose.yml
    compose_file = Path("docker-compose.yml")
    if not compose_file.exists():
        print("❌ docker-compose.yml not found")
        return False
    
    content = compose_file.read_text()
    
    # Remove version line if present
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if not line.strip().startswith('version:'):
            fixed_lines.append(line)
    
    # Write fixed content
    compose_file.write_text('\n'.join(fixed_lines))
    print("✅ Docker Compose configuration fixed")
    return True

def create_minimal_compose():
    """Create minimal Docker Compose for development"""
    print("📝 Creating minimal Docker Compose...")
    
    minimal_compose = """# ScrollIntel Minimal Docker Compose
services:
  postgres:
    image: postgres:15-alpine
    container_name: scrollintel-postgres
    environment:
      POSTGRES_DB: scrollintel
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: scrollintel_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: scrollintel-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    name: scrollintel-network
"""
    
    Path("docker-compose.minimal.yml").write_text(minimal_compose)
    print("✅ Minimal Docker Compose created")

def fix_environment():
    """Fix environment configuration"""
    print("🔧 Fixing environment configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file...")
        env_content = """# ScrollIntel Environment Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=scrollintel
POSTGRES_USER=postgres
POSTGRES_PASSWORD=scrollintel_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Security
JWT_SECRET_KEY=your_jwt_secret_key_change_in_production
JWT_ALGORITHM=HS256

# AI Services (Add your keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional Services
PINECONE_API_KEY=your_pinecone_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
"""
        env_file.write_text(env_content)
        print("✅ Environment file created")
    else:
        print("✅ Environment file exists")

def create_startup_scripts():
    """Create startup scripts for different scenarios"""
    print("📝 Creating startup scripts...")
    
    # Windows batch script
    windows_script = """@echo off
echo Starting ScrollIntel...

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker not available, starting simple mode...
    python run_simple.py
    goto :end
)

REM Try Docker Compose
echo Starting with Docker Compose...
docker-compose -f docker-compose.minimal.yml up -d postgres redis
timeout /t 10 /nobreak >nul

REM Start Python backend
echo Starting ScrollIntel backend...
python run_simple.py

:end
pause
"""
    
    Path("start_scrollintel.bat").write_text(windows_script)
    
    # Unix shell script
    unix_script = """#!/bin/bash
echo "Starting ScrollIntel..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Docker not available, starting simple mode..."
    python3 run_simple.py
    exit 0
fi

# Try Docker Compose
echo "Starting with Docker Compose..."
docker-compose -f docker-compose.minimal.yml up -d postgres redis
sleep 10

# Start Python backend
echo "Starting ScrollIntel backend..."
python3 run_simple.py
"""
    
    unix_script_path = Path("start_scrollintel.sh")
    unix_script_path.write_text(unix_script)
    unix_script_path.chmod(0o755)
    
    print("✅ Startup scripts created")

def fix_requirements():
    """Fix requirements.txt to include only essential packages"""
    print("🔧 Fixing requirements.txt...")
    
    essential_requirements = """# ScrollIntel Essential Requirements
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
sqlalchemy>=2.0.0
alembic>=1.12.0
psycopg2-binary>=2.9.7
redis>=5.0.0
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
pydantic>=2.4.0
pydantic-settings>=2.0.0
requests>=2.31.0
aiofiles>=23.2.0
jinja2>=3.1.2
"""
    
    Path("requirements.core.txt").write_text(essential_requirements)
    print("✅ Core requirements file created")

def create_health_check():
    """Create simple health check script"""
    print("📝 Creating health check script...")
    
    health_check = """#!/usr/bin/env python3
import requests
import sys
from datetime import datetime

def check_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ScrollIntel is healthy")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Timestamp: {data.get('timestamp', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to ScrollIntel (not running?)")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🏥 ScrollIntel Health Check")
    print("=" * 30)
    
    if check_health():
        print("\\n🎉 ScrollIntel is running properly!")
        sys.exit(0)
    else:
        print("\\n💡 Try starting ScrollIntel with:")
        print("   python run_simple.py")
        print("   or")
        print("   ./start_scrollintel.sh")
        sys.exit(1)
"""
    
    health_script = Path("health_check.py")
    health_script.write_text(health_check, encoding='utf-8')
    try:
        health_script.chmod(0o755)
    except:
        pass  # Windows doesn't support chmod
    print("✅ Health check script created")

def main():
    """Main fix function"""
    print("🔧 ScrollIntel Issue Fixer")
    print("=" * 40)
    
    # Check Docker
    docker_available = check_docker()
    
    # Fix configurations
    fix_docker_compose()
    create_minimal_compose()
    fix_environment()
    fix_requirements()
    create_startup_scripts()
    create_health_check()
    
    print("\n✅ All fixes applied!")
    print("\n🚀 To start ScrollIntel:")
    print("   Option 1: python run_simple.py")
    print("   Option 2: ./start_scrollintel.sh (Unix/Linux)")
    print("   Option 3: start_scrollintel.bat (Windows)")
    
    print("\n🏥 To check health:")
    print("   python health_check.py")
    
    if not docker_available:
        print("\n⚠️  Docker not available - using simple mode")
        print("   Install Docker Desktop for full features")

if __name__ == "__main__":
    main()