#!/usr/bin/env python3
"""
ScrollIntel Deployment Configuration Script
Ensures all hardcoding is properly addressed for production deployment
"""

import os
import json
import shutil
from pathlib import Path

def create_production_ready_configs():
    """Create production-ready configuration files"""
    
    print("🚀 Creating Production-Ready Configurations...")
    
    # 1. Create Docker environment file
    docker_env = """# Docker Production Environment
# This file is used by docker-compose.prod.yml

# Application
ENVIRONMENT=production
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000

# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=scrollintel
POSTGRES_USER=scrollintel_user
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Security
JWT_SECRET_KEY=${JWT_SECRET_KEY}

# AI Services
OPENAI_API_KEY=${OPENAI_API_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

# Email
SMTP_SERVER=${SMTP_SERVER}
SMTP_PORT=587
EMAIL_USERNAME=${EMAIL_USERNAME}
EMAIL_PASSWORD=${EMAIL_PASSWORD}
FROM_EMAIL=${FROM_EMAIL}
"""
    
    with open(".env.docker", "w") as f:
        f.write(docker_env)
    print("  ✅ Created .env.docker")
    
    # 2. Create Kubernetes ConfigMap template
    k8s_config = """apiVersion: v1
kind: ConfigMap
metadata:
  name: scrollintel-config
  namespace: scrollintel
data:
  ENVIRONMENT: "production"
  DEBUG: "false"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  LOG_LEVEL: "INFO"
  
  # Database
  POSTGRES_HOST: "postgres-service"
  POSTGRES_PORT: "5432"
  POSTGRES_DB: "scrollintel"
  
  # Redis
  REDIS_HOST: "redis-service"
  REDIS_PORT: "6379"
  
  # Email
  SMTP_PORT: "587"
  FROM_EMAIL: "noreply@yourdomain.com"

---
apiVersion: v1
kind: Secret
metadata:
  name: scrollintel-secrets
  namespace: scrollintel
type: Opaque
stringData:
  POSTGRES_PASSWORD: "your-secure-db-password"
  JWT_SECRET_KEY: "your-jwt-secret-key"
  OPENAI_API_KEY: "sk-your-openai-key"
  ANTHROPIC_API_KEY: "your-anthropic-key"
  EMAIL_PASSWORD: "your-email-password"
"""
    
    os.makedirs("k8s", exist_ok=True)
    with open("k8s/config.yaml", "w") as f:
        f.write(k8s_config)
    print("  ✅ Created k8s/config.yaml")
    
    # 3. Create Railway deployment config
    railway_config = {
        "build": {
            "builder": "DOCKERFILE",
            "dockerfilePath": "Dockerfile.prod"
        },
        "deploy": {
            "startCommand": "python -m uvicorn scrollintel.api.main:app --host 0.0.0.0 --port $PORT",
            "healthcheckPath": "/health",
            "healthcheckTimeout": 100,
            "restartPolicyType": "ON_FAILURE",
            "restartPolicyMaxRetries": 10
        }
    }
    
    with open("railway.json", "w") as f:
        json.dump(railway_config, f, indent=2)
    print("  ✅ Created railway.json")
    
    # 4. Create Render deployment config
    render_config = {
        "services": [
            {
                "type": "web",
                "name": "scrollintel-api",
                "env": "python",
                "buildCommand": "pip install -r requirements.txt",
                "startCommand": "python -m uvicorn scrollintel.api.main:app --host 0.0.0.0 --port $PORT",
                "healthCheckPath": "/health",
                "envVars": [
                    {"key": "ENVIRONMENT", "value": "production"},
                    {"key": "DEBUG", "value": "false"},
                    {"key": "API_HOST", "value": "0.0.0.0"}
                ]
            },
            {
                "type": "web",
                "name": "scrollintel-frontend",
                "env": "node",
                "buildCommand": "cd frontend && npm install && npm run build",
                "startCommand": "cd frontend && npm start",
                "rootDir": "frontend"
            }
        ],
        "databases": [
            {
                "name": "scrollintel-db",
                "databaseName": "scrollintel",
                "user": "scrollintel_user"
            }
        ]
    }
    
    with open("render.yaml", "w") as f:
        json.dump(render_config, f, indent=2)
    print("  ✅ Created render.yaml")
    
    # 5. Create Vercel deployment config for frontend
    vercel_config = {
        "version": 2,
        "name": "scrollintel-frontend",
        "builds": [
            {
                "src": "frontend/package.json",
                "use": "@vercel/next"
            }
        ],
        "routes": [
            {
                "src": "/api/(.*)",
                "dest": "https://your-api-domain.com/api/$1"
            },
            {
                "src": "/(.*)",
                "dest": "frontend/$1"
            }
        ],
        "env": {
            "NEXT_PUBLIC_API_URL": "https://your-api-domain.com",
            "NEXT_PUBLIC_WS_URL": "wss://your-api-domain.com"
        }
    }
    
    with open("vercel.json", "w") as f:
        json.dump(vercel_config, f, indent=2)
    print("  ✅ Created vercel.json")

def create_deployment_scripts():
    """Create deployment scripts for different platforms"""
    
    print("📜 Creating Deployment Scripts...")
    
    # 1. Docker deployment script
    docker_deploy = """#!/bin/bash
# Docker Production Deployment Script

set -e

echo "🚀 Starting ScrollIntel Docker Deployment..."

# Check if required environment variables are set
required_vars=("POSTGRES_PASSWORD" "JWT_SECRET_KEY" "OPENAI_API_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Error: $var environment variable is not set"
        exit 1
    fi
done

# Build and deploy with Docker Compose
echo "📦 Building Docker images..."
docker-compose -f docker-compose.prod.yml build

echo "🚀 Starting services..."
docker-compose -f docker-compose.prod.yml up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

echo "🔍 Checking service health..."
docker-compose -f docker-compose.prod.yml ps

echo "✅ ScrollIntel deployed successfully!"
echo "🌐 Access your application at: http://localhost:8000"
"""
    
    with open("scripts/deploy-docker.sh", "w", encoding='utf-8') as f:
        f.write(docker_deploy)
    os.chmod("scripts/deploy-docker.sh", 0o755)
    print("  ✅ Created scripts/deploy-docker.sh")
    
    # 2. Railway deployment script
    railway_deploy = """#!/bin/bash
# Railway Deployment Script

set -e

echo "🚀 Deploying ScrollIntel to Railway..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI is not installed. Install it from: https://railway.app/cli"
    exit 1
fi

# Login to Railway (if not already logged in)
railway login

# Set environment variables
echo "🔧 Setting environment variables..."
railway variables set ENVIRONMENT=production
railway variables set DEBUG=false
railway variables set API_HOST=0.0.0.0
railway variables set API_PORT=8000

# Deploy
echo "🚀 Deploying to Railway..."
railway up

echo "✅ Deployment complete!"
echo "🌐 Your app will be available at the Railway-provided URL"
"""
    
    with open("scripts/deploy-railway.sh", "w", encoding='utf-8') as f:
        f.write(railway_deploy)
    os.chmod("scripts/deploy-railway.sh", 0o755)
    print("  ✅ Created scripts/deploy-railway.sh")
    
    # 3. Render deployment script
    render_deploy = """#!/bin/bash
# Render Deployment Script

set -e

echo "🚀 Deploying ScrollIntel to Render..."

# Check if render.yaml exists
if [ ! -f "render.yaml" ]; then
    echo "❌ render.yaml not found. Run configure_for_deployment.py first."
    exit 1
fi

echo "📝 render.yaml configuration found"
echo "🌐 Go to https://render.com and create a new service using this repository"
echo "📋 Use the render.yaml file for automatic configuration"
echo "🔧 Don't forget to set your environment variables in the Render dashboard"

echo "Required environment variables:"
echo "  - POSTGRES_PASSWORD"
echo "  - JWT_SECRET_KEY"
echo "  - OPENAI_API_KEY"
echo "  - ANTHROPIC_API_KEY"
echo "  - EMAIL_PASSWORD"

echo "✅ Render configuration ready!"
"""
    
    with open("scripts/deploy-render.sh", "w", encoding='utf-8') as f:
        f.write(render_deploy)
    os.chmod("scripts/deploy-render.sh", 0o755)
    print("  ✅ Created scripts/deploy-render.sh")

def create_environment_validation_script():
    """Create script to validate environment configuration"""
    
    validation_script = """#!/usr/bin/env python3
\"\"\"
Environment Configuration Validation Script
Validates that all required environment variables are properly set
\"\"\"

import os
import sys
from typing import Dict, List, Tuple

def validate_environment() -> Tuple[bool, List[str]]:
    \"\"\"Validate environment configuration\"\"\"
    
    issues = []
    
    # Required variables for production
    required_vars = {
        'ENVIRONMENT': 'Environment type (production/staging/development)',
        'POSTGRES_PASSWORD': 'Database password',
        'JWT_SECRET_KEY': 'JWT secret key for authentication',
        'OPENAI_API_KEY': 'OpenAI API key for AI features'
    }
    
    # Optional but recommended variables
    recommended_vars = {
        'ANTHROPIC_API_KEY': 'Anthropic API key for Claude AI',
        'SMTP_SERVER': 'SMTP server for email notifications',
        'EMAIL_PASSWORD': 'Email password for SMTP authentication',
        'REDIS_HOST': 'Redis host for caching'
    }
    
    # Check required variables
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            issues.append(f"❌ MISSING: {var} - {description}")
        elif value in ['your-openai-api-key-here', 'your-secret-key-change-in-production']:
            issues.append(f"⚠️  PLACEHOLDER: {var} - {description}")
        else:
            print(f"✅ {var}: Configured")
    
    # Check recommended variables
    for var, description in recommended_vars.items():
        value = os.getenv(var)
        if not value:
            print(f"ℹ️  OPTIONAL: {var} - {description}")
        else:
            print(f"✅ {var}: Configured")
    
    # Validate specific formats
    openai_key = os.getenv('OPENAI_API_KEY', '')
    if openai_key and not openai_key.startswith('sk-'):
        issues.append(f"⚠️  INVALID FORMAT: OPENAI_API_KEY should start with 'sk-'")
    
    jwt_secret = os.getenv('JWT_SECRET_KEY', '')
    if jwt_secret and len(jwt_secret) < 32:
        issues.append(f"⚠️  WEAK: JWT_SECRET_KEY should be at least 32 characters long")
    
    return len(issues) == 0, issues

def main():
    \"\"\"Main validation function\"\"\"
    
    print("🔍 ScrollIntel Environment Validation")
    print("=" * 50)
    
    # Load .env file if it exists
    env_file = '.env'
    if os.path.exists(env_file):
        print(f"📁 Loading environment from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    try:
                        key, value = line.strip().split('=', 1)
                        if key not in os.environ:  # Don't override existing env vars
                            os.environ[key] = value
                    except ValueError:
                        continue
    
    # Validate configuration
    is_valid, issues = validate_environment()
    
    if issues:
        print("\\n🚨 Configuration Issues:")
        for issue in issues:
            print(f"   {issue}")
    
    if is_valid:
        print("\\n🎉 Environment configuration is valid!")
        print("✅ Ready for production deployment")
        return True
    else:
        print("\\n❌ Environment configuration has issues")
        print("💡 Fix the issues above before deploying to production")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""
    
    with open("scripts/validate-environment.py", "w", encoding='utf-8') as f:
        f.write(validation_script)
    os.chmod("scripts/validate-environment.py", 0o755)
    print("  ✅ Created scripts/validate-environment.py")

def main():
    """Main configuration function"""
    
    print("🔧 ScrollIntel Deployment Configuration")
    print("=" * 50)
    
    # Create scripts directory if it doesn't exist
    os.makedirs("scripts", exist_ok=True)
    
    # Create all configuration files
    create_production_ready_configs()
    create_deployment_scripts()
    create_environment_validation_script()
    
    print("\n" + "=" * 50)
    print("✅ Deployment Configuration Complete!")
    print("\n📋 What was created:")
    print("   • .env.docker - Docker environment configuration")
    print("   • k8s/config.yaml - Kubernetes configuration")
    print("   • railway.json - Railway deployment configuration")
    print("   • render.yaml - Render deployment configuration")
    print("   • vercel.json - Vercel frontend deployment configuration")
    print("   • scripts/deploy-docker.sh - Docker deployment script")
    print("   • scripts/deploy-railway.sh - Railway deployment script")
    print("   • scripts/deploy-render.sh - Render deployment script")
    print("   • scripts/validate-environment.py - Environment validation script")
    
    print("\n🚀 Next Steps:")
    print("   1. Run: python scripts/validate-environment.py")
    print("   2. Set your real API keys in environment variables")
    print("   3. Choose your deployment platform:")
    print("      • Docker: ./scripts/deploy-docker.sh")
    print("      • Railway: ./scripts/deploy-railway.sh")
    print("      • Render: ./scripts/deploy-render.sh")
    print("   4. Test your deployment")
    
    print("\n💡 Important:")
    print("   • Never commit real API keys to version control")
    print("   • Use environment variables for all sensitive data")
    print("   • Test in staging before production deployment")

if __name__ == "__main__":
    main()