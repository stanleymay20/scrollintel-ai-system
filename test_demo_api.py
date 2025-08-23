#!/usr/bin/env python3
"""
Quick test script to verify the demo API endpoints are working
"""

import requests
import json
import time
from datetime import datetime

def test_api_endpoints():
    """Test the main API endpoints"""
    
    # Try different ports to find the running API
    ports_to_try = [8000, 8001, 8002, 8003, 8080]
    base_url = None
    
    print("🔍 Looking for running ScrollIntel API...")
    
    for port in ports_to_try:
        try:
            url = f"http://localhost:{port}"
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                base_url = url
                print(f"✅ Found API running at {base_url}")
                break
        except requests.exceptions.RequestException:
            continue
    
    if not base_url:
        print("❌ No running ScrollIntel API found")
        print("💡 Please start the backend first with: python start_backend_demo.py")
        return False
    
    print(f"\n🧪 Testing API endpoints at {base_url}...")
    
    # Test endpoints
    endpoints = [
        ("/", "Root endpoint"),
        ("/health", "Health check"),
        ("/api/agents", "Agents list"),
        ("/api/monitoring/metrics", "System metrics"),
        ("/api/dashboard", "Dashboard data")
    ]
    
    results = []
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                results.append((endpoint, "✅ PASS", description, data))
                print(f"✅ {endpoint} - {description}")
            else:
                results.append((endpoint, f"❌ FAIL ({response.status_code})", description, None))
                print(f"❌ {endpoint} - {description} (Status: {response.status_code})")
        except Exception as e:
            results.append((endpoint, f"❌ ERROR", description, str(e)))
            print(f"❌ {endpoint} - {description} (Error: {e})")
    
    # Test chat endpoint
    print(f"\n💬 Testing chat functionality...")
    try:
        chat_data = {
            "message": "Hello, can you help me with data analysis?",
            "agent_id": "data-scientist"
        }
        response = requests.post(f"{base_url}/api/agents/chat", 
                               json=chat_data, timeout=5)
        if response.status_code == 200:
            chat_response = response.json()
            print(f"✅ Chat endpoint working")
            print(f"   Agent response: {chat_response.get('content', 'No content')[:100]}...")
        else:
            print(f"❌ Chat endpoint failed (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"API Base URL: {base_url}")
    print(f"Total endpoints tested: {len(endpoints) + 1}")
    
    passed = sum(1 for _, status, _, _ in results if "PASS" in status)
    print(f"Passed: {passed}")
    print(f"Failed: {len(results) - passed}")
    
    if passed == len(endpoints):
        print(f"🎉 All tests passed! ScrollIntel API is working correctly.")
        print(f"\n🌐 You can now:")
        print(f"   - Open {base_url}/docs for interactive API documentation")
        print(f"   - Open simple_frontend.html in your browser for a web interface")
        print(f"   - Use the API endpoints in your applications")
    else:
        print(f"⚠️  Some tests failed. Check the API server logs.")
    
    return passed == len(endpoints)

if __name__ == "__main__":
    print("🚀 ScrollIntel API Test Suite")
    print("=" * 50)
    
    success = test_api_endpoints()
    
    if success:
        print(f"\n✨ Ready to demo ScrollIntel!")
    else:
        print(f"\n🔧 Please check the API server and try again.")