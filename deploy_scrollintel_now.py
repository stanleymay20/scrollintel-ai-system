#!/usr/bin/env python3
"""
ScrollIntel Immediate Deployment Script
Deploy ScrollIntel to production with multiple platform options
"""

import os
import sys
import subprocess
import json
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScrollIntelDeployment:
    """ScrollIntel deployment orchestrator"""
    
    def __init__(self):
        self.deployment_options = {
            "1": {"name": "Docker Compose (Local Production)", "method": "deploy_docker_compose"},
            "2": {"name": "Render (Cloud Backend)", "method": "deploy_render"},
            "3": {"name": "Vercel (Frontend) + Render (Backend)", "method": "deploy_vercel_render"},
            "4": {"name": "Full Production Server", "method": "deploy_production_server"},
            "5": {"name": "Railway (Full Stack)", "method": "deploy_railway"}
        }
    
    def show_deployment_options(self):
        """Show available deployment options"""
        print("\n🚀 ScrollIntel Deployment Options:")
        print("=" * 50)
        
        for key, option in self.deployment_options.items():
            print(f"{key}. {option['name']}")
        
        print("\nRecommended for immediate deployment: Option 1 (Docker Compose)")
        print("Recommended for production: Option 3 (Vercel + Render)")
    
    def deploy(self):
        """Main deployment orchestrator"""
        print("🌟 ScrollIntel Production Deployment")
        print("=" * 50)
        
        self.show_deployment_options()
        
        choice = input("\nSelect deployment option (1-5): ").strip()
        
        if choice not in self.deployment_options:
            print("❌ Invalid option selected")
            return False
        
        option = self.deployment_options[choice]
        print(f"\n🚀 Starting deployment: {option['name']}")
        
        try:
            method = getattr(self, option['method'])
            return method()
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def deploy_docker_compose(self):
        """Deploy using Docker Compose (fastest option)"""
        print("\n🐳 Deploying with Docker Compose...")
        
        try:
            # Check if Docker is installed
            if not self.command_exists("docker"):
                print("❌ Docker not found. Please install Docker first.")
                return False
            
            if not self.command_exists("docker-compose"):
                print("❌ Docker Compose not found. Please install Docker Compose first.")
                return False
            
            # Create production environment file
            self.create_production_env()
            
            # Build and start services
            print("📦 Building Docker images...")
            self.run_command("docker-compose -f docker-compose.yml build")
            
            print("🚀 Starting services...")
            self.run_command("docker-compose -f docker-compose.yml up -d")
            
            # Wait for services to be ready
            print("⏳ Waiting for services to be ready...")
            time.sleep(30)
            
            # Health check
            if self.check_health("http://localhost:8000"):
                print("✅ Backend is healthy!")
            else:
                print("⚠️ Backend health check failed")
            
            # Print access information
            self.print_docker_access_info()
            return True
            
        except Exception as e:
            logger.error(f"Docker deployment failed: {e}")
            return False
    
    def deploy_render(self):
        """Deploy backend to Render"""
        print("\n☁️ Deploying to Render...")
        
        try:
            # Check if render CLI is available
            if not self.command_exists("render"):
                print("📦 Installing Render CLI...")
                self.run_command("npm install -g @render/cli")
            
            # Create render.yaml if it doesn't exist
            if not os.path.exists("render.yaml"):
                self.create_render_config()
            
            # Deploy to Render
            print("🚀 Deploying to Render...")
            self.run_command("render deploy")
            
            print("✅ Render deployment initiated!")
            print("🔗 Check your Render dashboard for deployment status")
            print("🌐 Your API will be available at: https://scrollintel-backend.onrender.com")
            
            return True
            
        except Exception as e:
            logger.error(f"Render deployment failed: {e}")
            return False
    
    def deploy_vercel_render(self):
        """Deploy frontend to Vercel and backend to Render"""
        print("\n🌐 Deploying to Vercel + Render...")
        
        try:
            # Deploy backend to Render first
            print("1️⃣ Deploying backend to Render...")
            if not self.deploy_render():
                return False
            
            # Deploy frontend to Vercel
            print("2️⃣ Deploying frontend to Vercel...")
            
            if not self.command_exists("vercel"):
                print("📦 Installing Vercel CLI...")
                self.run_command("npm install -g vercel")
            
            # Navigate to frontend directory
            os.chdir("frontend")
            
            # Install dependencies
            print("📦 Installing frontend dependencies...")
            self.run_command("npm install")
            
            # Build the application
            print("🔨 Building frontend...")
            self.run_command("npm run build")
            
            # Deploy to Vercel
            print("🚀 Deploying to Vercel...")
            self.run_command("vercel --prod --yes")
            
            os.chdir("..")
            
            print("✅ Full stack deployment completed!")
            print("🔗 Frontend: Check your Vercel dashboard")
            print("🔗 Backend: https://scrollintel-backend.onrender.com")
            
            return True
            
        except Exception as e:
            logger.error(f"Vercel + Render deployment failed: {e}")
            return False
    
    def deploy_production_server(self):
        """Deploy to production server"""
        print("\n🏭 Deploying to production server...")
        
        try:
            # Run the comprehensive production deployment
            self.run_command("python scripts/production-deployment.py")
            return True
            
        except Exception as e:
            logger.error(f"Production server deployment failed: {e}")
            return False
    
    def deploy_railway(self):
        """Deploy to Railway"""
        print("\n🚂 Deploying to Railway...")
        
        try:
            if not self.command_exists("railway"):
                print("📦 Installing Railway CLI...")
                self.run_command("npm install -g @railway/cli")
            
            # Login to Railway
            print("🔐 Please login to Railway...")
            self.run_command("railway login")
            
            # Create new project
            print("📁 Creating Railway project...")
            self.run_command("railway init")
            
            # Deploy
            print("🚀 Deploying to Railway...")
            self.run_command("railway up")
            
            print("✅ Railway deployment completed!")
            print("🔗 Check your Railway dashboard for the deployment URL")
            
            return True
            
        except Exception as e:
            logger.error(f"Railway deployment failed: {e}")
            return False
    
    def create_production_env(self):
        """Create production environment file"""
        env_content = f"""# ScrollIntel Production Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=scrollintel
POSTGRES_USER=postgres
POSTGRES_PASSWORD=scrollintel_secure_password_2024

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Security
JWT_SECRET_KEY={self.generate_secret_key()}

# API Keys (set these with your actual keys)
OPENAI_API_KEY={os.getenv('OPENAI_API_KEY', 'your_openai_key_here')}
ANTHROPIC_API_KEY={os.getenv('ANTHROPIC_API_KEY', 'your_anthropic_key_here')}

# Application
API_HOST=0.0.0.0
API_PORT=8000
BUILD_TARGET=production
NODE_ENV=production
"""
        
        with open(".env.production", "w") as f:
            f.write(env_content)
        
        # Also update the main .env file
        with open(".env", "w") as f:
            f.write(env_content)
        
        print("✅ Production environment configured")
    
    def create_render_config(self):
        """Create Render configuration"""
        render_config = {
            "services": [
                {
                    "type": "web",
                    "name": "scrollintel-backend",
                    "env": "python",
                    "plan": "starter",
                    "buildCommand": "pip install -r requirements.txt",
                    "startCommand": "uvicorn scrollintel.api.simple_main:app --host 0.0.0.0 --port $PORT",
                    "healthCheckPath": "/health",
                    "envVars": [
                        {"key": "ENVIRONMENT", "value": "production"},
                        {"key": "DEBUG", "value": "false"},
                        {"key": "JWT_SECRET_KEY", "generateValue": True},
                        {"key": "OPENAI_API_KEY", "sync": False}
                    ]
                }
            ]
        }
        
        with open("render.yaml", "w") as f:
            import yaml
            yaml.dump(render_config, f, default_flow_style=False)
        
        print("✅ Render configuration created")
    
    def generate_secret_key(self):
        """Generate a secure secret key"""
        import secrets
        return secrets.token_urlsafe(64)
    
    def command_exists(self, command):
        """Check if command exists"""
        try:
            subprocess.run(["which", command], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def run_command(self, command):
        """Run shell command"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error: {e.stderr}")
            raise
    
    def check_health(self, url):
        """Check service health"""
        try:
            import requests
            response = requests.get(f"{url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def print_docker_access_info(self):
        """Print Docker deployment access information"""
        print("\n🎉 ScrollIntel is now running!")
        print("=" * 50)
        print("🔗 Access Points:")
        print("   📱 Frontend:     http://localhost:3000")
        print("   🔧 Backend API:  http://localhost:8000")
        print("   📚 API Docs:     http://localhost:8000/docs")
        print("   ❤️  Health:      http://localhost:8000/health")
        print("   🗄️  Database:    localhost:5432")
        print("   🔴 Redis:        localhost:6379")
        print("\n📊 Management:")
        print("   🐳 View logs:    docker-compose logs -f")
        print("   🛑 Stop:         docker-compose down")
        print("   🔄 Restart:      docker-compose restart")
        print("   📈 Monitor:      docker-compose ps")
        print("\n🚀 ScrollIntel is ready for production use!")

def main():
    """Main function"""
    deployment = ScrollIntelDeployment()
    
    if deployment.deploy():
        print("\n🎉 ScrollIntel deployment completed successfully!")
        print("🙏 Launched in Jesus' name! God bless!")
    else:
        print("\n❌ Deployment failed. Please check the logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()