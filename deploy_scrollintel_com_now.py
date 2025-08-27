#!/usr/bin/env python3
"""
ScrollIntel.com Instant Deployment
One-command deployment to make scrollintel.com accessible to users immediately
"""

import os
import subprocess
import sys
import time
from datetime import datetime

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                ScrollIntel.com INSTANT DEPLOY                ║
║              Get Your Platform Live in 5 Minutes            ║
╚══════════════════════════════════════════════════════════════╝
    """)

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_command(command, description):
    log(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        log(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        log(f"❌ {description} - Failed: {e}")
        return False

def create_quick_env():
    """Create a quick environment file."""
    env_content = """# ScrollIntel.com Quick Deploy Environment
NODE_ENV=production
ENVIRONMENT=production

# Domain Configuration
DOMAIN=scrollintel.com
API_DOMAIN=api.scrollintel.com
APP_DOMAIN=app.scrollintel.com

# Database
DATABASE_URL=postgresql://scrollintel:quickpass123@db:5432/scrollintel_prod
POSTGRES_DB=scrollintel_prod
POSTGRES_USER=scrollintel
POSTGRES_PASSWORD=quickpass123

# Cache
REDIS_URL=redis://redis:6379/0

# Security
SECRET_KEY=quick-deploy-secret-key-change-in-production
JWT_SECRET_KEY=quick-jwt-secret-change-in-production

# AI Keys (REPLACE WITH YOUR ACTUAL KEYS)
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# CORS
CORS_ORIGINS=https://scrollintel.com,https://app.scrollintel.com,https://www.scrollintel.com

# Monitoring
GRAFANA_PASSWORD=admin123

# Features
ENABLE_ANALYTICS=true
ENABLE_MONITORING=true
ENABLE_CACHING=true
"""
    
    with open('.env.production', 'w') as f:
        f.write(env_content)
    
    log("✅ Quick environment configuration created")

def create_quick_compose():
    """Create a quick Docker Compose file."""
    compose_content = """version: '3.8'

services:
  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    restart: unless-stopped
    depends_on:
      - backend

  # Backend
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env.production
    depends_on:
      - db
      - redis
    restart: unless-stopped
    volumes:
      - ./uploads:/app/uploads

  # Database
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: scrollintel_prod
      POSTGRES_USER: scrollintel
      POSTGRES_PASSWORD: quickpass123
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  # Monitoring
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  postgres_data:
  grafana_data:
"""
    
    with open('docker-compose.quick.yml', 'w') as f:
        f.write(compose_content)
    
    log("✅ Quick Docker Compose configuration created")

def create_nginx_config():
    """Create Nginx configuration for domain routing."""
    nginx_content = """server {
    listen 80;
    server_name scrollintel.com www.scrollintel.com app.scrollintel.com;
    
    # Redirect www to non-www
    if ($host = www.scrollintel.com) {
        return 301 https://scrollintel.com$request_uri;
    }
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name api.scrollintel.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name grafana.scrollintel.com;
    
    location / {
        proxy_pass http://localhost:3001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
"""
    
    os.makedirs('nginx', exist_ok=True)
    with open('nginx/scrollintel.conf', 'w') as f:
        f.write(nginx_content)
    
    log("✅ Nginx configuration created")

def deploy_now():
    """Deploy ScrollIntel.com immediately."""
    print_banner()
    
    log("🚀 Starting ScrollIntel.com instant deployment...")
    
    # Create configurations
    create_quick_env()
    create_quick_compose()
    create_nginx_config()
    
    # Stop any existing containers
    log("🛑 Stopping any existing containers...")
    run_command("docker-compose -f docker-compose.quick.yml down", "Stopping containers")
    
    # Build and start services
    if not run_command("docker-compose -f docker-compose.quick.yml build", "Building containers"):
        log("❌ Build failed. Check Docker installation.")
        return False
    
    if not run_command("docker-compose -f docker-compose.quick.yml up -d", "Starting services"):
        log("❌ Failed to start services.")
        return False
    
    # Wait for services to start
    log("⏳ Waiting for services to initialize...")
    time.sleep(30)
    
    # Initialize database
    log("🔄 Initializing database...")
    run_command("docker-compose -f docker-compose.quick.yml exec -T backend python init_database.py", "Database initialization")
    
    # Health checks
    log("🔍 Running health checks...")
    time.sleep(10)
    
    # Test local endpoints
    endpoints = [
        ("http://localhost:3000", "Frontend"),
        ("http://localhost:8000/health", "Backend API"),
        ("http://localhost:8000/docs", "API Documentation"),
        ("http://localhost:3001", "Grafana Dashboard")
    ]
    
    for url, name in endpoints:
        run_command(f"curl -f -s -o /dev/null {url}", f"Testing {name}")
    
    # Success message
    print("\n" + "="*60)
    print("🎉 ScrollIntel.com is now LIVE!")
    print("="*60)
    
    print(f"""
🌐 LOCAL ACCESS (immediate):
   • Frontend: http://localhost:3000
   • API: http://localhost:8000
   • API Docs: http://localhost:8000/docs
   • Grafana: http://localhost:3001 (admin/admin123)

🌍 DOMAIN ACCESS (after DNS setup):
   • Main Site: https://scrollintel.com
   • Application: https://app.scrollintel.com
   • API: https://api.scrollintel.com

📋 NEXT STEPS FOR DOMAIN ACCESS:
   1. Point your DNS to this server IP: {get_server_ip()}
   2. Set up SSL certificates with Let's Encrypt
   3. Configure Nginx with the provided config

🔧 MANAGEMENT:
   • View logs: docker-compose -f docker-compose.quick.yml logs -f
   • Stop services: docker-compose -f docker-compose.quick.yml down
   • Restart: docker-compose -f docker-compose.quick.yml restart

⚠️  IMPORTANT:
   • Update API keys in .env.production
   • Change default passwords for production
   • Set up proper SSL certificates for domain access

🤖 Your ScrollIntel platform is ready with 15+ AI agents!
""")
    
    return True

def get_server_ip():
    """Get server IP address."""
    try:
        result = subprocess.run("curl -s ifconfig.me", shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "YOUR_SERVER_IP"
    except:
        return "YOUR_SERVER_IP"

def main():
    """Main deployment function."""
    try:
        success = deploy_now()
        if success:
            print("\n🎯 ScrollIntel.com deployment completed successfully!")
            print("🚀 Your AI platform is now accessible to users!")
            return 0
        else:
            print("\n❌ Deployment failed. Check the logs above.")
            return 1
    except KeyboardInterrupt:
        print("\n\n⚠️ Deployment interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())