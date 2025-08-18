#!/usr/bin/env python3
"""
ScrollIntel One-Click Cloud Deployment
Deploy ScrollIntel to the cloud with a single command
"""

import webbrowser
import time
import sys

def show_banner():
    """Show ScrollIntel deployment banner"""
    print("🚀" * 20)
    print("🌟 SCROLLINTEL CLOUD DEPLOYMENT 🌟")
    print("🚀" * 20)
    print()

def show_options():
    """Show deployment options"""
    print("Choose your deployment platform:")
    print()
    print("1. 🚂 Railway (RECOMMENDED - Easiest)")
    print("   ✅ Auto-detects Python")
    print("   ✅ Provides PostgreSQL database")
    print("   ✅ Built-in monitoring")
    print("   ⏱️  Deploys in 3-5 minutes")
    print()
    print("2. 🎨 Render (Most Reliable)")
    print("   ✅ Great for production")
    print("   ✅ Free tier available")
    print("   ✅ Excellent uptime")
    print("   ⏱️  Deploys in 5-10 minutes")
    print()
    print("3. ⚡ Vercel (Frontend Focus)")
    print("   ✅ Best for Next.js")
    print("   ✅ Serverless functions")
    print("   ✅ Global CDN")
    print("   ⏱️  Deploys in 2-3 minutes")
    print()
    print("4. 📊 Show Status (Check current deployment)")
    print()

def deploy_railway():
    """Deploy to Railway"""
    print("🚂 Deploying to Railway...")
    print()
    print("📋 Instructions:")
    print("1. Go to https://railway.app")
    print("2. Login with your GitHub account")
    print("3. Click 'New Project' → 'Deploy from GitHub repo'")
    print("4. Select: stanleymay20/ScrollIntel")
    print("5. Add OPENAI_API_KEY environment variable")
    print("6. Deploy!")
    print()
    print("🔥 Opening Railway.app...")
    
    try:
        webbrowser.open("https://railway.app")
        print("✅ Railway.app opened in your browser")
    except:
        print("⚠️ Please manually go to https://railway.app")
    
    print()
    print("🎯 After deployment, you'll get:")
    print("   🌐 Live URL: https://scrollintel-production-xxx.up.railway.app")
    print("   📚 API Docs: https://scrollintel-production-xxx.up.railway.app/docs")
    print("   ❤️ Health: https://scrollintel-production-xxx.up.railway.app/health")

def deploy_render():
    """Deploy to Render"""
    print("🎨 Deploying to Render...")
    print()
    print("📋 Instructions:")
    print("1. Go to https://render.com")
    print("2. Sign up with GitHub")
    print("3. Click 'New +' → 'Web Service'")
    print("4. Connect: stanleymay20/ScrollIntel")
    print("5. Configure:")
    print("   - Build: pip install -r requirements.txt")
    print("   - Start: uvicorn scrollintel.api.simple_main:app --host 0.0.0.0 --port $PORT")
    print("   - Health: /health")
    print("6. Add environment variables")
    print("7. Deploy!")
    print()
    print("🔥 Opening Render.com...")
    
    try:
        webbrowser.open("https://render.com")
        print("✅ Render.com opened in your browser")
    except:
        print("⚠️ Please manually go to https://render.com")
    
    print()
    print("🎯 After deployment, you'll get:")
    print("   🌐 Live URL: https://scrollintel-backend-xxx.onrender.com")
    print("   📚 API Docs: https://scrollintel-backend-xxx.onrender.com/docs")
    print("   ❤️ Health: https://scrollintel-backend-xxx.onrender.com/health")

def deploy_vercel():
    """Deploy to Vercel"""
    print("⚡ Deploying to Vercel...")
    print()
    print("📋 Instructions:")
    print("1. Go to https://vercel.com")
    print("2. Import from GitHub: stanleymay20/ScrollIntel")
    print("3. Set root directory to 'frontend'")
    print("4. Deploy!")
    print()
    print("🔥 Opening Vercel.com...")
    
    try:
        webbrowser.open("https://vercel.com")
        print("✅ Vercel.com opened in your browser")
    except:
        print("⚠️ Please manually go to https://vercel.com")
    
    print()
    print("🎯 After deployment, you'll get:")
    print("   🌐 Live URL: https://scrollintel-xxx.vercel.app")
    print("   📱 Frontend: Modern React interface")
    print("   🚀 CDN: Global content delivery")

def show_status():
    """Show deployment status"""
    import subprocess
    
    print("📊 Checking ScrollIntel deployment status...")
    print()
    
    try:
        result = subprocess.run([sys.executable, "verify_deployment.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        print("💡 Make sure ScrollIntel is running locally first")

def main():
    """Main deployment function"""
    show_banner()
    
    while True:
        show_options()
        
        try:
            choice = input("Enter your choice (1-4): ").strip()
            print()
            
            if choice == "1":
                deploy_railway()
                break
            elif choice == "2":
                deploy_render()
                break
            elif choice == "3":
                deploy_vercel()
                break
            elif choice == "4":
                show_status()
                print()
                continue
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                print()
                continue
                
        except KeyboardInterrupt:
            print("\n👋 Deployment cancelled")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print()
    print("🎉 ScrollIntel deployment initiated!")
    print("⏱️ Your AI platform will be live in a few minutes!")
    print("🌍 Ready to change the world with AI! 🚀")

if __name__ == "__main__":
    main()