#!/usr/bin/env python3
"""
Deploy ScrollIntel to Railway Cloud Platform
"""

import webbrowser
import time

def deploy_to_railway():
    """Deploy ScrollIntel to Railway"""
    
    print("🚂 ScrollIntel Railway Cloud Deployment")
    print("=" * 50)
    
    print("📋 Super Easy Deployment Instructions:")
    print("1. Go to https://railway.app")
    print("2. Login with your GitHub account")
    print("3. Click 'New Project'")
    print("4. Select 'Deploy from GitHub repo'")
    print("5. Choose: stanleymay20/ScrollIntel")
    print("6. Railway will auto-detect Python and deploy!")
    
    print("\n🔧 Optional Environment Variables (Railway auto-configures most):")
    env_vars = {
        "OPENAI_API_KEY": "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",
        "PORT": "8000"
    }
    
    for key, value in env_vars.items():
        if key == "OPENAI_API_KEY":
            print(f"   {key} = {value[:20]}...")
        else:
            print(f"   {key} = {value}")
    
    print("\n🎯 After deployment, you'll get:")
    print("   🌐 Live URL: https://scrollintel-production-xxx.up.railway.app")
    print("   📚 API Docs: https://scrollintel-production-xxx.up.railway.app/docs")
    print("   ❤️ Health: https://scrollintel-production-xxx.up.railway.app/health")
    print("   🗄️ Auto PostgreSQL database")
    print("   📊 Built-in monitoring")
    
    print("\n🔥 Opening Railway.app...")
    time.sleep(2)
    
    try:
        webbrowser.open("https://railway.app")
        print("✅ Railway.app opened in your browser")
    except:
        print("⚠️ Please manually go to https://railway.app")
    
    print("\n🎉 Railway is the EASIEST deployment option!")
    print("⏱️ Deployment typically takes 3-5 minutes")
    print("🚀 Just connect GitHub and Railway does the rest!")
    
    return True

if __name__ == "__main__":
    deploy_to_railway()