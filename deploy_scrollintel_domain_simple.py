#!/usr/bin/env python3
"""
Simple ScrollIntel.com Domain Deployment
Deploy ScrollIntel to scrollintel.com domain quickly
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def print_banner():
    print("""
=======================================================
        ScrollIntel.com Domain Deployment
        Deploy to Production in Minutes
=======================================================
    """)

def create_production_env():
    """Create production environment file"""
    print("📝 Creating production environment...")
    
    env_content = """# ScrollIntel.com Production Environment
NODE_ENV=production
ENVIRONMENT=production

# Domain Configuration
DOMAIN=scrollintel.com
API_DOMAIN=api.scrollintel.com
APP_DOMAIN=app.scrollintel.com

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# Database (Update with your actual database URL)
DATABASE_URL=postgresql://scrollintel:your_password@localhost:5432/scrollintel_prod
REDIS_URL=redis://localhost:6379/0

# Security (IMPORTANT: Change these!)
SECRET_KEY=your-super-secure-secret-key-change-this-now
JWT_SECRET_KEY=your-jwt-secret-key-change-this-now

# AI Services (Add your actual API keys)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# CORS
CORS_ORIGINS=https://scrollintel.com,https://app.scrollintel.com

# Features
ENABLE_MONITORING=true
ENABLE_ANALYTICS=true
ENABLE_CACHING=true

# Email (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=noreply@scrollintel.com
SMTP_PASSWORD=your-email-password
"""
    
    with open('.env.scrollintel.com', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env.scrollintel.com")
    return True

def create_docker_compose():
    """Create Docker Compose for scrollintel.com"""
    print("🐳 Creating Docker Compose configuration...")
    
    compose_content = """version: '3.8'

services:
  # ScrollIntel Backend
  scrollintel-backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env.scrollintel.com
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    restart: unless-stopped
    depends_on:
      - postgres
      - redis

  # Frontend (if you have one)
  scrollintel-frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=https://api.scrollintel.com
    restart: unless-stopped
    depends_on:
      - scrollintel-backend

  # Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: scrollintel_prod
      POSTGRES_USER: scrollintel
      POSTGRES_PASSWORD: your_password
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
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Reverse Proxy (Nginx)
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - scrollintel-backend
      - scrollintel-frontend
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    name: scrollintel-network
"""
    
    with open('docker-compose.scrollintel.yml', 'w') as f:
        f.write(compose_content)
    
    print("✅ Created docker-compose.scrollintel.yml")
    return True

def create_nginx_config():
    """Create Nginx configuration"""
    print("🌐 Creating Nginx configuration...")
    
    os.makedirs('nginx', exist_ok=True)
    
    nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream backend {
        server scrollintel-backend:8000;
    }
    
    upstream frontend {
        server scrollintel-frontend:3000;
    }

    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name scrollintel.com app.scrollintel.com api.scrollintel.com;
        return 301 https://$server_name$request_uri;
    }

    # Main domain - Frontend
    server {
        listen 443 ssl;
        server_name scrollintel.com app.scrollintel.com;
        
        ssl_certificate /etc/ssl/certs/scrollintel.com.crt;
        ssl_certificate_key /etc/ssl/certs/scrollintel.com.key;
        
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }

    # API subdomain - Backend
    server {
        listen 443 ssl;
        server_name api.scrollintel.com;
        
        ssl_certificate /etc/ssl/certs/scrollintel.com.crt;
        ssl_certificate_key /etc/ssl/certs/scrollintel.com.key;
        
        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
"""
    
    with open('nginx/nginx.conf', 'w') as f:
        f.write(nginx_config)
    
    print("✅ Created nginx configuration")
    return True

def create_deployment_scripts():
    """Create deployment scripts"""
    print("📜 Creating deployment scripts...")
    
    # Deploy script
    deploy_script = """#!/bin/bash
# ScrollIntel.com Deployment Script

echo "🚀 Deploying ScrollIntel to scrollintel.com..."

# Load environment
export $(cat .env.scrollintel.com | grep -v '^#' | xargs)

# Build and start services
echo "📦 Building containers..."
docker-compose -f docker-compose.scrollintel.yml build

echo "🚀 Starting services..."
docker-compose -f docker-compose.scrollintel.yml up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 30

# Initialize database
echo "🗄️ Initializing database..."
docker-compose -f docker-compose.scrollintel.yml exec scrollintel-backend python init_database.py

echo "✅ Deployment complete!"
echo "🌐 ScrollIntel is now live at:"
echo "   Main: https://scrollintel.com"
echo "   API: https://api.scrollintel.com"
echo "   Docs: https://api.scrollintel.com/docs"
"""
    
    with open('deploy_scrollintel.sh', 'w') as f:
        f.write(deploy_script)
    
    os.chmod('deploy_scrollintel.sh', 0o755)
    
    # Status script
    status_script = """#!/bin/bash
# ScrollIntel Status Check

echo "📊 ScrollIntel.com Status"
echo "========================"

# Container status
docker-compose -f docker-compose.scrollintel.yml ps

# Health checks
echo ""
echo "🏥 Health Checks:"
curl -s https://api.scrollintel.com/health || echo "API not responding"
curl -s https://scrollintel.com || echo "Frontend not responding"

echo ""
echo "💾 System Resources:"
df -h
free -h
"""
    
    with open('status_scrollintel.sh', 'w') as f:
        f.write(status_script)
    
    os.chmod('status_scrollintel.sh', 0o755)
    
    print("✅ Created deployment scripts")
    return True

def create_ssl_setup():
    """Create SSL certificate setup"""
    print("🔒 Creating SSL setup...")
    
    ssl_script = """#!/bin/bash
# SSL Certificate Setup for ScrollIntel.com

echo "🔒 Setting up SSL certificates for scrollintel.com..."

# Create SSL directory
mkdir -p ssl

# Option 1: Let's Encrypt (Recommended)
echo "Installing certbot..."
sudo apt-get update
sudo apt-get install -y certbot

# Get certificates
certbot certonly --standalone \\
    -d scrollintel.com \\
    -d app.scrollintel.com \\
    -d api.scrollintel.com \\
    --email admin@scrollintel.com \\
    --agree-tos \\
    --non-interactive

# Copy certificates to ssl directory
sudo cp /etc/letsencrypt/live/scrollintel.com/fullchain.pem ssl/scrollintel.com.crt
sudo cp /etc/letsencrypt/live/scrollintel.com/privkey.pem ssl/scrollintel.com.key

# Set permissions
sudo chown $USER:$USER ssl/*
chmod 644 ssl/scrollintel.com.crt
chmod 600 ssl/scrollintel.com.key

echo "✅ SSL certificates installed!"
"""
    
    with open('setup_ssl_scrollintel.sh', 'w') as f:
        f.write(ssl_script)
    
    os.chmod('setup_ssl_scrollintel.sh', 0o755)
    
    print("✅ Created SSL setup script")
    return True

def create_quick_deploy():
    """Create one-command deployment"""
    print("⚡ Creating quick deployment script...")
    
    quick_deploy = """#!/bin/bash
# ScrollIntel.com One-Command Deployment

set -e

echo "🚀 ScrollIntel.com Quick Deployment"
echo "=================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "❌ Don't run as root. Run as regular user with sudo access."
    exit 1
fi

# Update system
echo "📦 Updating system..."
sudo apt-get update

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "🐳 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Setup SSL certificates
echo "🔒 Setting up SSL..."
./setup_ssl_scrollintel.sh

# Deploy application
echo "🚀 Deploying application..."
./deploy_scrollintel.sh

echo ""
echo "🎉 ScrollIntel.com deployment complete!"
echo ""
echo "🌐 Your platform is now live at:"
echo "   Main Site: https://scrollintel.com"
echo "   API: https://api.scrollintel.com"
echo "   Documentation: https://api.scrollintel.com/docs"
echo ""
echo "📋 Next steps:"
echo "1. Update .env.scrollintel.com with your API keys"
echo "2. Point your DNS to this server"
echo "3. Test the platform"
echo ""
echo "🔧 Management commands:"
echo "   Status: ./status_scrollintel.sh"
echo "   Logs: docker-compose -f docker-compose.scrollintel.yml logs -f"
echo "   Restart: docker-compose -f docker-compose.scrollintel.yml restart"
"""
    
    with open('quick_deploy_scrollintel.sh', 'w') as f:
        f.write(quick_deploy)
    
    os.chmod('quick_deploy_scrollintel.sh', 0o755)
    
    print("✅ Created quick deployment script")
    return True

def create_readme():
    """Create deployment README"""
    print("📖 Creating deployment README...")
    
    readme_content = """# ScrollIntel.com Production Deployment

## 🚀 Quick Start

### One-Command Deployment
```bash
./quick_deploy_scrollintel.sh
```

### Manual Deployment
1. **Setup SSL certificates:**
   ```bash
   ./setup_ssl_scrollintel.sh
   ```

2. **Deploy application:**
   ```bash
   ./deploy_scrollintel.sh
   ```

3. **Check status:**
   ```bash
   ./status_scrollintel.sh
   ```

## 🌐 Domain Configuration

### DNS Records
Point these DNS records to your server IP:
```
A     scrollintel.com           -> YOUR_SERVER_IP
A     app.scrollintel.com       -> YOUR_SERVER_IP
A     api.scrollintel.com       -> YOUR_SERVER_IP
CNAME www.scrollintel.com       -> scrollintel.com
```

### Environment Variables
Update `.env.scrollintel.com` with your actual values:
- `OPENAI_API_KEY`: Your OpenAI API key
- `JWT_SECRET_KEY`: Secure JWT secret
- `DATABASE_URL`: PostgreSQL connection string
- `SECRET_KEY`: Application secret key

## 🔧 Management

### View Logs
```bash
docker-compose -f docker-compose.scrollintel.yml logs -f
```

### Restart Services
```bash
docker-compose -f docker-compose.scrollintel.yml restart
```

### Update Application
```bash
git pull origin main
./deploy_scrollintel.sh
```

### Backup Database
```bash
docker-compose -f docker-compose.scrollintel.yml exec postgres pg_dump -U scrollintel scrollintel_prod > backup.sql
```

## 🌐 Access Points

After deployment, your platform will be available at:
- **Main Site**: https://scrollintel.com
- **API**: https://api.scrollintel.com
- **API Documentation**: https://api.scrollintel.com/docs
- **Health Check**: https://api.scrollintel.com/health

## 🆘 Troubleshooting

### Check Container Status
```bash
docker-compose -f docker-compose.scrollintel.yml ps
```

### View Container Logs
```bash
docker-compose -f docker-compose.scrollintel.yml logs [service-name]
```

### Restart All Services
```bash
docker-compose -f docker-compose.scrollintel.yml down
docker-compose -f docker-compose.scrollintel.yml up -d
```

## 📞 Support

- Check logs for errors
- Verify DNS configuration
- Ensure SSL certificates are valid
- Confirm environment variables are set

---

**ScrollIntel.com** - Your AI-powered platform is ready! 🚀
"""
    
    with open('SCROLLINTEL_DEPLOYMENT_README.md', 'w') as f:
        f.write(readme_content)
    
    print("✅ Created deployment README")
    return True

def main():
    """Main deployment setup"""
    print_banner()
    
    print("🎯 Setting up ScrollIntel.com domain deployment...")
    print("This will create all necessary files for production deployment.")
    print()
    
    # Create all deployment files
    steps = [
        ("Creating production environment", create_production_env),
        ("Creating Docker Compose", create_docker_compose),
        ("Creating Nginx configuration", create_nginx_config),
        ("Creating deployment scripts", create_deployment_scripts),
        ("Creating SSL setup", create_ssl_setup),
        ("Creating quick deployment", create_quick_deploy),
        ("Creating README", create_readme)
    ]
    
    for description, func in steps:
        print(f"📋 {description}...")
        if not func():
            print(f"❌ Failed: {description}")
            return False
        print(f"✅ Completed: {description}")
    
    print()
    print("🎉 ScrollIntel.com deployment setup complete!")
    print()
    print("📋 Next Steps:")
    print("1. Update .env.scrollintel.com with your actual API keys and secrets")
    print("2. Point your DNS records to this server")
    print("3. Run: ./quick_deploy_scrollintel.sh")
    print("4. Test your platform at https://scrollintel.com")
    print()
    print("📖 Full instructions: SCROLLINTEL_DEPLOYMENT_README.md")
    print()
    print("🚀 Ready to deploy ScrollIntel to scrollintel.com!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)