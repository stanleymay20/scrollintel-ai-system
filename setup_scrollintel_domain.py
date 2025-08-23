#!/usr/bin/env python3
"""
ScrollIntel Domain Setup for scrollintel.com
"""

import os
import json
from pathlib import Path

def create_vercel_config():
    """Create Vercel configuration for scrollintel.com"""
    config = {
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
        }
    }
    
    with open("vercel.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Created vercel.json for scrollintel.com")

def create_render_config():
    """Create Render configuration"""
    config = {
        "services": [
            {
                "type": "web",
                "name": "scrollintel-api",
                "env": "python",
                "plan": "starter",
                "buildCommand": "pip install -r requirements.txt",
                "startCommand": "uvicorn scrollintel.api.simple_main:app --host 0.0.0.0 --port $PORT",
                "healthCheckPath": "/health",
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
        yaml.dump(config, f, default_flow_style=False)
    print("Created render.yaml for API deployment")

def create_deployment_guide():
    """Create deployment guide"""
    guide = """# ScrollIntel.com Deployment Guide

## Domain Setup Complete!
Your domain: scrollintel.com
API endpoint: api.scrollintel.com

## Quick Deployment Options

### Option 1: Vercel + Render (RECOMMENDED)

#### Deploy Frontend to Vercel:
1. Go to https://vercel.com
2. Import from GitHub: your ScrollIntel repository
3. Set root directory to 'frontend'
4. Add custom domain: scrollintel.com
5. Deploy!

#### Deploy Backend to Render:
1. Go to https://render.com
2. Create new Web Service from GitHub
3. Use the render.yaml configuration
4. Add custom domain: api.scrollintel.com
5. Deploy!

### Option 2: Railway (All-in-One)
1. Go to https://railway.app
2. Deploy from GitHub repository
3. Add custom domain in Railway dashboard
4. Configure environment variables

## DNS Configuration

Add these DNS records to your domain registrar:

```
Type    Name    Value                           TTL
A       @       [Your deployment IP]            300
CNAME   www     scrollintel.com                 300
CNAME   api     [Your API deployment URL]      300
```

## Environment Variables

Set these in your deployment platform:

```
ENVIRONMENT=production
DEBUG=false
OPENAI_API_KEY=your_openai_api_key
JWT_SECRET_KEY=your_jwt_secret_key
ALLOWED_ORIGINS=https://scrollintel.com,https://www.scrollintel.com
```

## Expected URLs After Deployment

- Main Site: https://scrollintel.com
- API: https://api.scrollintel.com
- API Docs: https://api.scrollintel.com/docs
- Health Check: https://api.scrollintel.com/health

## Test Your Deployment

After deployment, test these URLs to ensure everything works:
- https://scrollintel.com (should load the frontend)
- https://api.scrollintel.com/health (should return {"status": "healthy"})
- https://api.scrollintel.com/docs (should show API documentation)

ScrollIntel.com is ready to go live!
"""
    
    with open("SCROLLINTEL_DOMAIN_GUIDE.md", "w", encoding='utf-8') as f:
        f.write(guide)
    print("Created SCROLLINTEL_DOMAIN_GUIDE.md")

def create_test_script():
    """Create domain test script"""
    script = '''#!/usr/bin/env python3
"""Test ScrollIntel domain deployment"""

import requests

def test_scrollintel_domain():
    """Test if ScrollIntel is deployed correctly"""
    
    urls = [
        "https://scrollintel.com",
        "https://www.scrollintel.com", 
        "https://api.scrollintel.com/health",
        "https://api.scrollintel.com/docs"
    ]
    
    print("Testing ScrollIntel domain deployment...")
    
    for url in urls:
        try:
            print(f"Testing {url}...")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"SUCCESS: {url} - {response.status_code}")
            else:
                print(f"WARNING: {url} - {response.status_code}")
        except Exception as e:
            print(f"ERROR: {url} - {e}")
    
    print("Domain test complete!")

if __name__ == "__main__":
    test_scrollintel_domain()
'''
    
    with open("test_scrollintel_domain.py", "w") as f:
        f.write(script)
    print("Created test_scrollintel_domain.py")

def main():
    """Main setup function"""
    print("Setting up ScrollIntel for scrollintel.com domain...")
    
    create_vercel_config()
    create_render_config()
    create_deployment_guide()
    create_test_script()
    
    print("\nScrollIntel Domain Setup Complete!")
    print("\nFiles created:")
    print("- vercel.json (Vercel deployment config)")
    print("- render.yaml (Render deployment config)")
    print("- SCROLLINTEL_DOMAIN_GUIDE.md (deployment instructions)")
    print("- test_scrollintel_domain.py (test script)")
    print("\nNext steps:")
    print("1. Read SCROLLINTEL_DOMAIN_GUIDE.md for deployment instructions")
    print("2. Choose Vercel + Render for best results")
    print("3. Configure DNS records with your domain registrar")
    print("4. Deploy and test!")

if __name__ == "__main__":
    main()