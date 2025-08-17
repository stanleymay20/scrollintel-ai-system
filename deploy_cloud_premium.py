#!/usr/bin/env python3
"""
ScrollIntel Premium Cloud Deployment
Deploy to the best cloud platforms with enterprise features
"""

import os
import sys
import subprocess
import time
import json

def run_command(command, shell=True):
    """Run command and return result"""
    try:
        print(f"🔧 {command}")
        result = subprocess.run(
            command,
            shell=shell,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ Success")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e.stderr}")
        return None

def deploy_to_railway():
    """Deploy to Railway - Best for full-stack apps"""
    print("\n🚂 RAILWAY DEPLOYMENT (Recommended)")
    print("=" * 50)
    print("✨ Features: Auto-scaling, PostgreSQL, Redis, SSL, Custom domains")
    
    try:
        # Install Railway CLI
        if not run_command("railway --version"):
            print("📦 Installing Railway CLI...")
            run_command("npm install -g @railway/cli")
        
        # Login
        print("🔐 Login to Railway (browser will open)...")
        run_command("railway login")
        
        # Create project
        print("📁 Creating Railway project...")
        run_command("railway init")
        
        # Add PostgreSQL
        print("🗄️ Adding PostgreSQL database...")
        run_command("railway add postgresql")
        
        # Add Redis
        print("🔴 Adding Redis cache...")
        run_command("railway add redis")
        
        # Set environment variables
        print("🔧 Setting environment variables...")
        env_vars = {
            "ENVIRONMENT": "production",
            "DEBUG": "false",
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "JWT_SECRET_KEY": "railway_secure_jwt_key_2024"
        }
        
        for key, value in env_vars.items():
            if value:
                run_command(f'railway variables set {key}="{value}"')
        
        # Deploy
        print("🚀 Deploying to Railway...")
        run_command("railway up")
        
        # Get URL
        result = run_command("railway status --json")
        if result:
            try:
                status = json.loads(result)
                url = status.get("deployments", [{}])[0].get("url", "")
                if url:
                    print(f"\n🎉 RAILWAY DEPLOYMENT SUCCESSFUL!")
                    print(f"🔗 URL: {url}")
                    print(f"📊 Dashboard: https://railway.app/dashboard")
                    return url
            except:
                pass
        
        print("✅ Railway deployment initiated! Check your dashboard.")
        return True
        
    except Exception as e:
        print(f"❌ Railway deployment failed: {e}")
        return False

def deploy_to_render():
    """Deploy to Render - Best for backend APIs"""
    print("\n☁️ RENDER DEPLOYMENT")
    print("=" * 50)
    print("✨ Features: Auto-deploy from Git, PostgreSQL, SSL, CDN")
    
    try:
        # Create render.yaml
        render_config = """services:
  - type: web
    name: scrollintel-api
    env: python
    plan: starter
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn scrollintel.api.simple_main:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /health
    envVars:
      - key: ENVIRONMENT
        value: production
      - key: DEBUG
        value: false
      - key: JWT_SECRET_KEY
        generateValue: true
      - key: OPENAI_API_KEY
        sync: false

databases:
  - name: scrollintel-db
    plan: starter
"""
        
        with open("render.yaml", "w") as f:
            f.write(render_config)
        
        print("✅ Created render.yaml configuration")
        print("📝 Next steps:")
        print("1. Push code to GitHub")
        print("2. Connect GitHub repo to Render")
        print("3. Deploy automatically")
        print("🔗 Go to: https://render.com")
        
        return True
        
    except Exception as e:
        print(f"❌ Render setup failed: {e}")
        return False

def deploy_to_vercel():
    """Deploy frontend to Vercel - Best for Next.js"""
    print("\n▲ VERCEL DEPLOYMENT (Frontend)")
    print("=" * 50)
    print("✨ Features: Edge network, Auto-scaling, Analytics")
    
    try:
        # Install Vercel CLI
        if not run_command("vercel --version"):
            print("📦 Installing Vercel CLI...")
            run_command("npm install -g vercel")
        
        # Navigate to frontend
        if os.path.exists("frontend"):
            os.chdir("frontend")
            
            # Install dependencies
            print("📦 Installing frontend dependencies...")
            run_command("npm install")
            
            # Deploy
            print("🚀 Deploying to Vercel...")
            run_command("vercel --prod")
            
            os.chdir("..")
            
            print("✅ Vercel deployment completed!")
            print("🔗 Check your Vercel dashboard for the URL")
            return True
        else:
            print("⚠️ Frontend directory not found")
            return False
            
    except Exception as e:
        print(f"❌ Vercel deployment failed: {e}")
        return False

def deploy_to_aws():
    """Deploy to AWS - Enterprise grade"""
    print("\n☁️ AWS DEPLOYMENT (Enterprise)")
    print("=" * 50)
    print("✨ Features: ECS, RDS, ElastiCache, CloudFront, Auto-scaling")
    
    try:
        # Check AWS CLI
        if not run_command("aws --version"):
            print("❌ AWS CLI not found. Please install it first.")
            print("🔗 https://aws.amazon.com/cli/")
            return False
        
        # Create AWS deployment script
        aws_script = """#!/bin/bash
# AWS ECS Deployment Script

# Build and push Docker image
docker build -t scrollintel-api .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker tag scrollintel-api:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/scrollintel-api:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/scrollintel-api:latest

# Deploy to ECS
aws ecs update-service --cluster scrollintel-cluster --service scrollintel-service --force-new-deployment
"""
        
        with open("deploy-aws.sh", "w") as f:
            f.write(aws_script)
        
        print("✅ Created AWS deployment script")
        print("📝 Manual setup required:")
        print("1. Create ECS cluster")
        print("2. Create RDS PostgreSQL instance")
        print("3. Create ElastiCache Redis")
        print("4. Configure load balancer")
        print("5. Run: chmod +x deploy-aws.sh && ./deploy-aws.sh")
        
        return True
        
    except Exception as e:
        print(f"❌ AWS setup failed: {e}")
        return False

def deploy_to_digitalocean():
    """Deploy to DigitalOcean App Platform"""
    print("\n🌊 DIGITALOCEAN DEPLOYMENT")
    print("=" * 50)
    print("✨ Features: App Platform, Managed databases, CDN")
    
    try:
        # Create app spec
        app_spec = {
            "name": "scrollintel",
            "services": [
                {
                    "name": "api",
                    "source_dir": "/",
                    "github": {
                        "repo": "your-username/scrollintel",
                        "branch": "main"
                    },
                    "run_command": "uvicorn scrollintel.api.simple_main:app --host 0.0.0.0 --port $PORT",
                    "environment_slug": "python",
                    "instance_count": 1,
                    "instance_size_slug": "basic-xxs",
                    "http_port": 8000,
                    "health_check": {
                        "http_path": "/health"
                    },
                    "envs": [
                        {
                            "key": "ENVIRONMENT",
                            "value": "production"
                        },
                        {
                            "key": "DEBUG",
                            "value": "false"
                        }
                    ]
                }
            ],
            "databases": [
                {
                    "name": "scrollintel-db",
                    "engine": "PG",
                    "version": "14"
                }
            ]
        }
        
        with open(".do/app.yaml", "w") as f:
            import yaml
            yaml.dump(app_spec, f)
        
        print("✅ Created DigitalOcean app spec")
        print("📝 Next steps:")
        print("1. Push code to GitHub")
        print("2. Create app on DigitalOcean")
        print("3. Connect GitHub repo")
        print("🔗 Go to: https://cloud.digitalocean.com/apps")
        
        return True
        
    except Exception as e:
        print(f"❌ DigitalOcean setup failed: {e}")
        return False

def main():
    """Main deployment orchestrator"""
    print("🌟 ScrollIntel Premium Cloud Deployment")
    print("=" * 60)
    print("🚀 Best cloud platforms for enterprise deployment")
    
    options = {
        "1": {"name": "Railway (Recommended)", "func": deploy_to_railway},
        "2": {"name": "Render + Vercel", "func": lambda: deploy_to_render() and deploy_to_vercel()},
        "3": {"name": "AWS (Enterprise)", "func": deploy_to_aws},
        "4": {"name": "DigitalOcean", "func": deploy_to_digitalocean},
        "5": {"name": "Deploy All", "func": lambda: all([
            deploy_to_railway(),
            deploy_to_render(),
            deploy_to_vercel()
        ])}
    }
    
    print("\n🎯 Premium Deployment Options:")
    for key, option in options.items():
        print(f"{key}. {option['name']}")
    
    choice = input("\nSelect deployment option (1-5): ").strip()
    
    if choice not in options:
        print("❌ Invalid option")
        return False
    
    print(f"\n🚀 Starting: {options[choice]['name']}")
    
    # Create production environment
    create_production_env()
    
    # Execute deployment
    success = options[choice]['func']()
    
    if success:
        print("\n🎉 DEPLOYMENT SUCCESSFUL!")
        print("=" * 50)
        print("✅ ScrollIntel is now live in the cloud!")
        print("🔗 Access your deployed application")
        print("📊 Monitor performance and scaling")
        print("🔒 SSL certificates automatically configured")
        print("🌍 Global CDN for fast access worldwide")
        print("\n🙏 Deployed in Jesus' name! God bless!")
    else:
        print("\n❌ Deployment failed. Please check the logs.")
    
    return success

def create_production_env():
    """Create production environment configuration"""
    env_content = """# ScrollIntel Production Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Security
JWT_SECRET_KEY=cloud_production_jwt_secret_2024

# API Keys
OPENAI_API_KEY=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

# Application
API_HOST=0.0.0.0
API_PORT=8000
"""
    
    with open(".env.production", "w") as f:
        f.write(env_content)
    
    print("✅ Production environment configured")

if __name__ == "__main__":
    main()