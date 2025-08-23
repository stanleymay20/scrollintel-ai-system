#!/usr/bin/env python3
"""
Quick deployment script for ScrollIntel.com
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

def check_git_status():
    """Check if code is committed to git"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, check=True)
        if result.stdout.strip():
            print("⚠️  You have uncommitted changes. Committing them now...")
            subprocess.run(['git', 'add', '.'], check=True)
            subprocess.run(['git', 'commit', '-m', 'Deploy ScrollIntel to scrollintel.com'], check=True)
            print("✅ Changes committed")
        else:
            print("✅ Git status clean")
        return True
    except subprocess.CalledProcessError:
        print("❌ Git not initialized or error occurred")
        return False

def deploy_to_railway():
    """Deploy to Railway with custom domain"""
    print("🚂 Deploying to Railway...")
    
    # Check if Railway CLI is installed
    try:
        subprocess.run(['railway', '--version'], capture_output=True, check=True)
        print("✅ Railway CLI found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Railway CLI not found. Installing...")
        print("Please install Railway CLI:")
        print("npm install -g @railway/cli")
        webbrowser.open("https://docs.railway.app/develop/cli")
        return False
    
    try:
        # Login to Railway
        print("🔐 Logging into Railway...")
        subprocess.run(['railway', 'login'], check=True)
        
        # Initialize project
        print("🚀 Initializing Railway project...")
        subprocess.run(['railway', 'init'], check=True)
        
        # Deploy
        print("📦 Deploying to Railway...")
        subprocess.run(['railway', 'up'], check=True)
        
        print("✅ Deployed to Railway!")
        print("🌐 Add your custom domain 'scrollintel.com' in the Railway dashboard")
        webbrowser.open("https://railway.app/dashboard")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Railway deployment failed: {e}")
        return False

def deploy_to_vercel():
    """Deploy frontend to Vercel"""
    print("▲ Deploying frontend to Vercel...")
    
    try:
        # Check if Vercel CLI is installed
        subprocess.run(['vercel', '--version'], capture_output=True, check=True)
        print("✅ Vercel CLI found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Vercel CLI not found. Installing...")
        subprocess.run(['npm', 'install', '-g', 'vercel'], check=True)
    
    try:
        # Change to frontend directory
        os.chdir('frontend')
        
        # Deploy to Vercel
        print("🚀 Deploying to Vercel...")
        subprocess.run(['vercel', '--prod', '--yes'], check=True)
        
        print("✅ Frontend deployed to Vercel!")
        print("🌐 Add your custom domain 'scrollintel.com' in the Vercel dashboard")
        webbrowser.open("https://vercel.com/dashboard")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Vercel deployment failed: {e}")
        return False
    finally:
        os.chdir('..')

def show_deployment_options():
    """Show deployment options to user"""
    print("""
🚀 ScrollIntel.com Deployment Options

1. Railway (Recommended - All-in-one)
   - Deploys both frontend and backend
   - Includes database
   - Easy custom domain setup
   
2. Vercel (Frontend only)
   - Deploy frontend to Vercel
   - You'll need separate backend deployment
   
3. Manual deployment
   - Follow SCROLLINTEL_DOMAIN_GUIDE.md
   
Which option would you like? (1/2/3): """)
    
    choice = input().strip()
    
    if choice == '1':
        return deploy_to_railway()
    elif choice == '2':
        return deploy_to_vercel()
    elif choice == '3':
        print("📖 Please follow the instructions in SCROLLINTEL_DOMAIN_GUIDE.md")
        return True
    else:
        print("❌ Invalid choice")
        return False

def main():
    """Main deployment function"""
    print("🌐 ScrollIntel.com Deployment Script")
    print("====================================")
    
    # Check prerequisites
    if not check_git_status():
        print("❌ Git setup required")
        return False
    
    # Show deployment options
    success = show_deployment_options()
    
    if success:
        print("""
🎉 Deployment initiated!

📋 Next Steps:
1. Wait for deployment to complete
2. Add custom domain 'scrollintel.com' in your platform dashboard
3. Configure DNS records with your domain registrar:
   - A record: @ -> [deployment IP]
   - CNAME: www -> scrollintel.com
   - CNAME: api -> [API deployment URL]
4. Test your deployment with: python test_scrollintel_domain.py

🌟 ScrollIntel will be live at https://scrollintel.com!
""")
    else:
        print("❌ Deployment failed. Check SCROLLINTEL_DOMAIN_GUIDE.md for manual instructions.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)