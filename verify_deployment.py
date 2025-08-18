#!/usr/bin/env python3
"""
Verify ScrollIntel Deployment Status
"""

import requests
import json
import time

def test_local_deployment():
    """Test local deployment"""
    print("🔍 Testing Local Deployment...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Local Health Check: PASSED")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Services: {health_data.get('services', {})}")
            return True
        else:
            print(f"❌ Local Health Check: FAILED ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Local Health Check: FAILED ({str(e)})")
        return False

def test_api_endpoints():
    """Test key API endpoints"""
    print("\n🔍 Testing API Endpoints...")
    
    endpoints = [
        "/health",
        "/docs",
        "/openapi.json"
    ]
    
    base_url = "http://localhost:8000"
    passed = 0
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint}: PASSED")
                passed += 1
            else:
                print(f"❌ {endpoint}: FAILED ({response.status_code})")
        except Exception as e:
            print(f"❌ {endpoint}: FAILED ({str(e)})")
    
    print(f"\n📊 API Test Results: {passed}/{len(endpoints)} endpoints working")
    return passed == len(endpoints)

def show_deployment_status():
    """Show comprehensive deployment status"""
    print("\n" + "="*60)
    print("🎉 SCROLLINTEL DEPLOYMENT STATUS")
    print("="*60)
    
    # Test local deployment
    local_ok = test_local_deployment()
    
    # Test API endpoints
    api_ok = test_api_endpoints()
    
    print("\n📋 Deployment Summary:")
    print(f"   🏠 Local Backend: {'✅ RUNNING' if local_ok else '❌ STOPPED'}")
    print(f"   🔗 API Endpoints: {'✅ WORKING' if api_ok else '❌ ISSUES'}")
    print(f"   📚 Documentation: http://localhost:8000/docs")
    print(f"   ❤️  Health Check: http://localhost:8000/health")
    
    print("\n🚀 Cloud Deployment Options:")
    print("   🚂 Railway: python deploy_railway_now.py")
    print("   🎨 Render: python deploy_render_now.py")
    print("   ⚡ Vercel: Deploy frontend to vercel.com")
    
    print("\n🌟 GitHub Repository:")
    print("   📦 https://github.com/stanleymay20/ScrollIntel.git")
    
    if local_ok and api_ok:
        print("\n🎉 ScrollIntel is READY for cloud deployment!")
        print("   Choose your platform and deploy to the world! 🌍")
    else:
        print("\n⚠️  Fix local issues before cloud deployment")
    
    return local_ok and api_ok

if __name__ == "__main__":
    show_deployment_status()