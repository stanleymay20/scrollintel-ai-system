#!/usr/bin/env python3
"""
ScrollIntel Domain Deployment Script
Deploy ScrollIntel to scrollintel.com domain
"""

import os
import sys
import json
import subprocess
import requests
from pathlib import Path

class ScrollIntelDomainDeployer:
    def __init__(self):
        self.domain = "scrollintel.com"
        self.subdomain = "www.scrollintel.com"
        self.api_subdomain = "api.scrollintel.com"
        self.project_root = Path(__file__).parent
        
    def print_banner(self):
        print("""
🚀 ScrollIntel Domain Deployment
================================
Domain: scrollintel.com
Target: Production deployment with custom domain
Platform: Multi-cloud deployment options
        """)
    
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        print("📋 Checking prerequisites...")
        
        # Check if domain is accessible
        try:
            response = requests.get(f"https://{self.domain}", timeout=5)
            print(f"✅ Domain {self.domain} is accessible")
        except:
            print(f"⚠️  Domain {self.domain} not yet configured (expected for new domain)")
        
        # Check environment files
        env_files = [".env", ".env.production"]
        for env_file in env_files:
            if os.path.exists(env_file):
                print(f"✅ {env_file} exists")
            else:
                print(f"⚠️  {env_file} not found")
        
        # Check if OpenAI API key is set
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            print("✅ OpenAI API key configured")
        else:
            print("⚠️  OpenAI API key not found in environment")
        
        return True
    
    def create_domain_configs(self):
        """Create domain-specific configuration files"""
        print("📝 Creating domain configuration files...")
        
        # Create Vercel configuration for custom domain
        vercel_config = {
            "version": 2,
            "name": "scrollintel",
            "builds": [
                {
                    "src": "frontend/package.json",
                    "use": "@vercel/next"
                }
            ],
            "routes": [
                {
                    "src": "/api/(.*)",
                    "dest": "https://api.scrollintel.com/api/$1"
                },
                {
                    "src": "/(.*)",
                    "dest": "frontend/$1"
                }
            ],
            "env": {
                "NEXT_PUBLIC_API_URL": "https://api.scrollintel.com",
                "NEXT_PUBLIC_DOMAIN": "scrollintel.com"
            },
            "domains": [
                "scrollintel.com",
                "www.scrollintel.com"
            ]
        }
        
        with open("vercel.json", "w") as f:
            json.dump(vercel_config, f, indent=2)
        print("✅ Created vercel.json with custom domain")
        
        # Create Render configuration for API
        render_config = {
            "services": [
                {
                    "type": "web",
                    "name": "scrollintel-api",
                    "env": "python",
                    "plan": "starter",
                    "buildCommand": "pip install -r requirements.txt",
                    "startCommand": "uvicorn scrollintel.api.simple_main:app --host 0.0.0.0 --port $PORT",
                    "healthCheckPath": "/health",
                    "domains": [
                        "api.scrollintel.com"
                    ],
                    "envVars": [
                        {
                            "key": "ENVIRONMENT",
                            "value": "production"
                        },
                        {
                            "key": "DEBUG",
                            "value": "false"
                        },
                        {
                            "key": "ALLOWED_ORIGINS",
                            "value": "https://scrollintel.com,https://www.scrollintel.com"
                        },
                        {
                            "key": "JWT_SECRET_KEY",
                            "generateValue": True
                        }
                    ]
                }
            ],
            "databases": [
                {
                    "name": "scrollintel-db",
                    "plan": "starter"
                }
            ]
        }
        
        with open("render.yaml", "w") as f:
            import yaml
            yaml.dump(render_config, f, default_flow_style=False)
        print("✅ Created render.yaml with custom domain")
        
        # Create Nginx configuration for custom domain
        nginx_config = f"""
server {{
    listen 80;
    server_name {self.domain} {self.subdomain};
    return 301 https://$server_name$request_uri;
}}

server {{
    listen 443 ssl http2;
    server_name {self.domain} {self.subdomain};
    
    ssl_certificate /etc/ssl/certs/scrollintel.com.crt;
    ssl_certificate_key /etc/ssl/private/scrollintel.com.key;
    
    location / {{
        proxy_pass http://frontend:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
    
    location /api/ {{
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}

server {{
    listen 443 ssl http2;
    server_name {self.api_subdomain};
    
    ssl_certificate /etc/ssl/certs/scrollintel.com.crt;
    ssl_certificate_key /etc/ssl/private/scrollintel.com.key;
    
    location / {{
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
"""
        
        os.makedirs("nginx", exist_ok=True)
        with open("nginx/scrollintel.com.conf", "w") as f:
            f.write(nginx_config)
        print("✅ Created nginx configuration for custom domain")
    
    def create_deployment_instructions(self):
        """Create step-by-step deployment instructions"""
        instructions = f"""
# 🚀 ScrollIntel.com Deployment Instructions

## 🎯 Domain Setup Complete!
Your domain: **{self.domain}**
API endpoint: **{self.api_subdomain}**

## 📋 Deployment Options

### Option 1: Vercel + Render (RECOMMENDED)

#### Step 1: Deploy Frontend to Vercel
1. Go to https://vercel.com
2. Import from GitHub: your ScrollIntel repository
3. Set root directory to `frontend`
4. Add custom domain: `{self.domain}` and `{self.subdomain}`
5. Deploy!

#### Step 2: Deploy Backend to Render
1. Go to https://render.com
2. Create new Web Service from GitHub
3. Use the `render.yaml` configuration
4. Add custom domain: `{self.api_subdomain}`
5. Deploy!

### Option 2: Railway (All-in-One)
1. Go to https://railway.app
2. Deploy from GitHub repository
3. Add custom domain in Railway dashboard
4. Configure environment variables

### Option 3: DigitalOcean App Platform
1. Go to https://cloud.digitalocean.com/apps
2. Create app from GitHub
3. Add custom domain in settings
4. Deploy with managed database

## 🔧 DNS Configuration

Add these DNS records to your domain registrar:

```
Type    Name    Value                           TTL
A       @       [Your deployment IP]            300
CNAME   www     scrollintel.com                 300
CNAME   api     [Your API deployment URL]      300
```

## 🌐 Expected URLs After Deployment

- **Main Site**: https://{self.domain}
- **WWW**: https://{self.subdomain}
- **API**: https://{self.api_subdomain}
- **API Docs**: https://{self.api_subdomain}/docs
- **Health Check**: https://{self.api_subdomain}/health

## 🔑 Environment Variables to Set

```
ENVIRONMENT=production
DEBUG=false
OPENAI_API_KEY=your_openai_api_key
JWT_SECRET_KEY=your_jwt_secret_key
ALLOWED_ORIGINS=https://{self.domain},https://{self.subdomain}
DATABASE_URL=postgresql://... (provided by hosting platform)
```

## 🎉 Post-Deployment Checklist

- [ ] Domain resolves to your deployment
- [ ] SSL certificate is active (HTTPS)
- [ ] API endpoints respond correctly
- [ ] Frontend loads and connects to API
- [ ] Health check passes
- [ ] All AI agents are accessible

## 🚀 Launch Commands

### Quick Deploy to Vercel:
```bash
cd frontend
npx vercel --prod --yes
```

### Quick Deploy to Render:
```bash
# Push to GitHub first, then deploy via Render dashboard
git add . && git commit -m "Deploy to scrollintel.com" && git push
```

### Test Deployment:
```bash
python test_domain_deployment.py
```

---

**🎯 ScrollIntel.com is ready to go live!**
**Choose your deployment platform and launch in minutes!**
"""
        
        with open("DOMAIN_DEPLOYMENT_GUIDE.md", "w") as f:
            f.write(instructions)
        print("✅ Created domain deployment guide")
    
    def create_test_script(self):
        """Create a script to test the domain deployment"""
        test_script = f'''#!/usr/bin/env python3
"""
Test ScrollIntel domain deployment
"""

import requests
import time

def test_domain_deployment():
    """Test if ScrollIntel is properly deployed to the domain"""
    
    urls_to_test = [
        "https://{self.domain}",
        "https://{self.subdomain}",
        "https://{self.api_subdomain}/health",
        "https://{self.api_subdomain}/docs"
    ]
    
    print("🧪 Testing ScrollIntel domain deployment...")
    
    for url in urls_to_test:
        try:
            print(f"Testing {{url}}...")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {{url}} - OK ({{response.status_code}})")
            else:
                print(f"⚠️  {{url}} - {{response.status_code}}")
        except requests.exceptions.RequestException as e:
            print(f"❌ {{url}} - Error: {{e}}")
    
    print("\\n🎉 Domain deployment test complete!")

if __name__ == "__main__":
    test_domain_deployment()
'''
        
        with open("test_domain_deployment.py", "w") as f:
            f.write(test_script)
        print("✅ Created domain deployment test script")
    
    def update_frontend_config(self):
        """Update frontend configuration for custom domain"""
        print("🔧 Updating frontend configuration...")
        
        # Update Next.js config
        next_config_path = "frontend/next.config.js"
        if os.path.exists(next_config_path):
            with open(next_config_path, "r") as f:
                content = f.read()
            
            # Add domain-specific configuration
            domain_config = f"""
// Domain configuration for scrollintel.com
const isDomain = process.env.NODE_ENV === 'production' && process.env.VERCEL_URL?.includes('scrollintel.com');
const apiUrl = isDomain ? 'https://api.scrollintel.com' : process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

"""
            
            # Insert at the beginning of the file
            updated_content = domain_config + content
            
            with open(next_config_path, "w") as f:
                f.write(updated_content)
            
            print("✅ Updated Next.js configuration for custom domain")
    
    def show_deployment_summary(self):
        """Show deployment summary and next steps"""
        print(f"""
🎉 ScrollIntel Domain Setup Complete!

📋 Files Created:
✅ vercel.json - Vercel deployment with custom domain
✅ render.yaml - Render deployment with API subdomain  
✅ nginx/scrollintel.com.conf - Nginx configuration
✅ DOMAIN_DEPLOYMENT_GUIDE.md - Step-by-step instructions
✅ test_domain_deployment.py - Deployment testing script

🌐 Your Domain Setup:
• Main Site: https://{self.domain}
• WWW: https://{self.subdomain}  
• API: https://{self.api_subdomain}

🚀 Next Steps:
1. Choose deployment platform (Vercel + Render recommended)
2. Follow DOMAIN_DEPLOYMENT_GUIDE.md instructions
3. Configure DNS records with your domain registrar
4. Deploy and test with test_domain_deployment.py

🎯 Ready to launch ScrollIntel on your custom domain!
""")
    
    def run(self):
        """Run the complete domain deployment setup"""
        self.print_banner()
        
        if not self.check_prerequisites():
            print("❌ Prerequisites check failed")
            return False
        
        self.create_domain_configs()
        self.create_deployment_instructions()
        self.create_test_script()
        self.update_frontend_config()
        self.show_deployment_summary()
        
        return True

if __name__ == "__main__":
    deployer = ScrollIntelDomainDeployer()
    success = deployer.run()
    
    if success:
        print("✅ Domain setup complete! Check DOMAIN_DEPLOYMENT_GUIDE.md for next steps.")
        sys.exit(0)
    else:
        print("❌ Domain setup failed!")
        sys.exit(1)