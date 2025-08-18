#!/usr/bin/env python3
"""
Deploy ScrollIntel to Render Cloud Platform
"""

import webbrowser
import time

def deploy_to_render():
    """Deploy ScrollIntel to Render"""
    
    print("🚀 ScrollIntel Render Cloud Deployment")
    print("=" * 50)
    
    print("📋 Deployment Instructions:")
    print("1. Go to https://render.com")
    print("2. Sign up/Login with your GitHub account")
    print("3. Click 'New +' → 'Web Service'")
    print("4. Connect your GitHub repository: stanleymay20/ScrollIntel")
    print("5. Configure the service:")
    print("   - Name: scrollintel-backend")
    print("   - Environment: Python 3")
    print("   - Build Command: pip install -r requirements.txt")
    print("   - Start Command: uvicorn scrollintel.api.simple_main:app --host 0.0.0.0 --port $PORT")
    print("   - Health Check Path: /health")
    
    print("\n🔧 Environment Variables to add:")
    env_vars = {
        "ENVIRONMENT": "production",
        "DEBUG": "false",
        "JWT_SECRET_KEY": "render_secure_jwt_2024_scrollintel",
        "OPENAI_API_KEY": "your-openai-api-key-here"
    }
    
    for key, value in env_vars.items():
        print(f"   {key} = {value}")
    
    print("\n🎯 After deployment, you'll get:")
    print("   🌐 Live URL: https://scrollintel-backend-xxx.onrender.com")
    print("   📚 API Docs: https://scrollintel-backend-xxx.onrender.com/docs")
    print("   ❤️ Health: https://scrollintel-backend-xxx.onrender.com/health")
    
    print("\n🔥 Opening Render.com...")
    time.sleep(2)
    
    try:
        webbrowser.open("https://render.com")
        print("✅ Render.com opened in your browser")
    except:
        print("⚠️ Please manually go to https://render.com")
    
    print("\n🎉 Follow the instructions above to deploy ScrollIntel!")
    print("⏱️ Deployment typically takes 5-10 minutes")
    
    return True

if __name__ == "__main__":
    deploy_to_render()