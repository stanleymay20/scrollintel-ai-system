#!/usr/bin/env python3
"""
ScrollIntel.com Domain Test Script
Test your scrollintel.com deployment
"""

import requests
import time
import sys

def test_domain_deployment():
    """Test if scrollintel.com is properly deployed"""
    
    print("🧪 Testing ScrollIntel.com Deployment")
    print("=" * 50)
    
    domains = {
        "Main Site": "https://scrollintel.com",
        "App": "https://app.scrollintel.com", 
        "API": "https://api.scrollintel.com",
        "API Health": "https://api.scrollintel.com/health",
        "API Docs": "https://api.scrollintel.com/docs"
    }
    
    results = {}
    
    for name, url in domains.items():
        print(f"\n🔍 Testing {name}: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"  ✅ {name}: OK (Status: {response.status_code})")
                results[name] = True
            else:
                print(f"  ⚠️  {name}: Warning (Status: {response.status_code})")
                results[name] = False
        except requests.exceptions.ConnectionError:
            print(f"  ❌ {name}: Connection failed")
            results[name] = False
        except requests.exceptions.Timeout:
            print(f"  ⏰ {name}: Timeout")
            results[name] = False
        except Exception as e:
            print(f"  ❌ {name}: Error - {str(e)}")
            results[name] = False
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")
    
    print(f"\n🎯 Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your ScrollIntel.com deployment is working perfectly!")
        return True
    elif passed > 0:
        print("⚠️  Partial deployment detected. Some services may still be starting.")
        return False
    else:
        print("❌ Deployment not detected. Please check your setup.")
        return False

def test_local_deployment():
    """Test local deployment"""
    
    print("\n🏠 Testing Local Deployment")
    print("=" * 30)
    
    local_endpoints = {
        "Backend API": "http://localhost:8000/health",
        "Frontend": "http://localhost:3000",
        "API Docs": "http://localhost:8000/docs",
        "Grafana": "http://localhost:3001"
    }
    
    results = {}
    
    for name, url in local_endpoints.items():
        print(f"\n🔍 Testing {name}: {url}")
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"  ✅ {name}: OK")
                results[name] = True
            else:
                print(f"  ⚠️  {name}: Status {response.status_code}")
                results[name] = False
        except:
            print(f"  ❌ {name}: Not accessible")
            results[name] = False
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n🎯 Local Score: {passed}/{total} services running")
    
    return passed > 0

if __name__ == "__main__":
    print("🚀 ScrollIntel.com Deployment Tester")
    print()
    
    # Test domain deployment
    domain_success = test_domain_deployment()
    
    # Test local deployment if domain fails
    if not domain_success:
        local_success = test_local_deployment()
        
        if local_success:
            print("\n💡 Local deployment detected!")
            print("To deploy to scrollintel.com:")
            print("1. Configure DNS records")
            print("2. Run: ./deploy_production.sh")
            print("3. Run: ./setup_ssl.sh")
        else:
            print("\n🚀 To get started:")
            print("1. Run: python run_simple.py (for local testing)")
            print("2. Run: python setup_scrollintel_com.py (for production)")
    
    print("\n📖 For full setup guide, see: SCROLLINTEL_COM_SETUP_GUIDE.md")