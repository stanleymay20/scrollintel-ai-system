#!/usr/bin/env python3
"""
Deploy ScrollIntel to scrollintel.com Domain - Quick Setup
Choose your deployment platform and get live in minutes!
"""

import os
import sys
import webbrowser
import subprocess

def print_banner():
    print("""
========================================================
    Deploy ScrollIntel to scrollintel.com NOW!
    Choose Your Platform - Live in Minutes
========================================================
    """)

def check_git_status():
    """Check if code is committed to git"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("⚠️  You have uncommitted changes. Consider committing them first.")
            return False
        return True
    except:
        print("⚠️  Git not found or not a git repository")
        return False

def deploy_railway():
    """Deploy to Railway"""
    print("""
🚂 Railway Deployment (Recommended)
===================================

Railway provides:
✅ Automatic SSL certificates
✅ Global CDN
✅ Managed PostgreSQL database
✅ Auto-scaling
✅ Custom domain support

Steps:
1. Go to https://railway.app
2. Login with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your ScrollIntel repository
5. Railway will auto-detect the configuration
6. Add these environment variables:
   - OPENAI_API_KEY: your_openai_api_key
   - JWT_SECRET_KEY: your_secure_jwt_secret
   - SECRET_KEY: your_app_secret_key
7. Add custom domain: scrollintel.com
8. Deploy!

Your app will be live at scrollintel.com in ~5 minutes!
    """)
    
    if input("Open Railway now? (y/n): ").lower() == 'y':
        webbrowser.open('https://railway.app')
    
    return True

def deploy_render():
    """Deploy to Render"""
    print("""
🎨 Render Deployment
===================

Render provides:
✅ Free SSL certificates
✅ Automatic deployments
✅ Managed PostgreSQL
✅ Custom domains
✅ Health checks

Steps:
1. Go to https://render.com
2. Connect your GitHub account
3. Create "New Web Service"
4. Select your ScrollIntel repository
5. Render will use the render.yaml configuration
6. Add environment variables:
   - OPENAI_API_KEY: your_openai_api_key
   - JWT_SECRET_KEY: your_secure_jwt_secret
   - SECRET_KEY: your_app_secret_key
7. Add custom domain: scrollintel.com
8. Deploy!

Your app will be live at scrollintel.com in ~10 minutes!
    """)
    
    if input("Open Render now? (y/n): ").lower() == 'y':
        webbrowser.open('https://render.com')
    
    return True

def deploy_vercel():
    """Deploy to Vercel"""
    print("""
▲ Vercel Deployment (Frontend + Backend)
=======================================

Vercel provides:
✅ Global edge network
✅ Automatic SSL
✅ Serverless functions
✅ Custom domains

Steps:
1. Frontend (Vercel):
   - Go to https://vercel.com
   - Import your GitHub repository
   - Set root directory to "frontend"
   - Add domain: scrollintel.com

2. Backend (Railway/Render):
   - Deploy backend separately
   - Use subdomain: api.scrollintel.com

Your app will be live at scrollintel.com in ~15 minutes!
    """)
    
    if input("Open Vercel now? (y/n): ").lower() == 'y':
        webbrowser.open('https://vercel.com')
    
    return True

def deploy_docker():
    """Deploy with Docker"""
    print("""
🐳 Docker Deployment (Self-Hosted)
==================================

For full control and customization:

1. Get a server (DigitalOcean, AWS, etc.)
2. Install Docker and Docker Compose
3. Clone your repository
4. Run deployment:

   ./deploy_scrollintel_simple.sh

5. Setup SSL certificates:
   
   sudo certbot --nginx -d scrollintel.com

6. Point DNS to your server IP

Your app will be live at scrollintel.com in ~30 minutes!
    """)
    
    return True

def test_local_first():
    """Test locally before deployment"""
    print("""
🧪 Test Locally First (Recommended)
===================================

Before deploying to production, test locally:

1. Start the application:
   python run_simple.py

2. Open: http://localhost:8000

3. Test features:
   - Upload a CSV file
   - Chat with AI agents
   - Check API docs at /docs

4. Verify everything works, then deploy!
    """)
    
    if input("Start local test now? (y/n): ").lower() == 'y':
        try:
            subprocess.run(['python', 'run_simple.py'])
        except KeyboardInterrupt:
            print("\n✅ Local test completed!")
    
    return True

def show_dns_setup():
    """Show DNS configuration"""
    print("""
🌐 DNS Configuration for scrollintel.com
========================================

After deployment, configure these DNS records:

A     scrollintel.com           -> DEPLOYMENT_IP
A     api.scrollintel.com       -> DEPLOYMENT_IP
A     app.scrollintel.com       -> DEPLOYMENT_IP
CNAME www.scrollintel.com       -> scrollintel.com

Most deployment platforms will provide you with:
- An IP address (for A records)
- Or a CNAME target (for CNAME records)

DNS propagation takes 5-60 minutes.
    """)

def show_environment_variables():
    """Show required environment variables"""
    print("""
🔑 Required Environment Variables
================================

Set these in your deployment platform:

OPENAI_API_KEY=sk-your-openai-api-key-here
JWT_SECRET_KEY=your-super-secure-jwt-secret-key
SECRET_KEY=your-application-secret-key
DATABASE_URL=postgresql://... (auto-generated by platform)

Optional:
ANTHROPIC_API_KEY=your-anthropic-api-key
SMTP_HOST=smtp.gmail.com
SMTP_USER=noreply@scrollintel.com
SMTP_PASSWORD=your-email-password
    """)

def main():
    """Main deployment menu"""
    print_banner()
    
    print("🎯 Ready to deploy ScrollIntel to scrollintel.com!")
    print("Choose your deployment platform:\n")
    
    while True:
        print("Deployment Options:")
        print("1. 🚂 Railway (Recommended - 5 minutes)")
        print("2. 🎨 Render (10 minutes)")
        print("3. ▲ Vercel (15 minutes)")
        print("4. 🐳 Docker Self-Hosted (30 minutes)")
        print("5. 🧪 Test Locally First")
        print("6. 🌐 DNS Setup Guide")
        print("7. 🔑 Environment Variables")
        print("8. ❌ Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            deploy_railway()
        elif choice == '2':
            deploy_render()
        elif choice == '3':
            deploy_vercel()
        elif choice == '4':
            deploy_docker()
        elif choice == '5':
            test_local_first()
        elif choice == '6':
            show_dns_setup()
        elif choice == '7':
            show_environment_variables()
        elif choice == '8':
            print("\n👋 Happy deploying! Your ScrollIntel platform awaits at scrollintel.com!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()