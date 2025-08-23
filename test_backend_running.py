#!/usr/bin/env python3
"""
Test if ScrollIntel backend is running and accessible
"""

import requests
import time
import json

def test_backend():
    """Test backend endpoints"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing ScrollIntel Backend")
    print("=" * 40)
    
    # Test health endpoint
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test API docs
    print("\n📚 Testing API documentation...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API docs accessible")
        else:
            print(f"❌ API docs failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API docs failed: {e}")
    
    # Test agents endpoint
    print("\n🤖 Testing agents endpoint...")
    try:
        response = requests.get(f"{base_url}/api/agents", timeout=10)
        if response.status_code == 200:
            agents = response.json()
            print(f"✅ Agents endpoint working")
            print(f"   Found {len(agents)} agents")
        else:
            print(f"❌ Agents endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Agents endpoint failed: {e}")
    
    # Test a simple agent interaction
    print("\n💬 Testing agent interaction...")
    try:
        payload = {
            "message": "Hello, can you help me analyze some data?",
            "agent_id": "cto"
        }
        response = requests.post(f"{base_url}/api/agents/chat", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("✅ Agent interaction working")
            print(f"   Response: {result.get('response', 'No response')[:100]}...")
        else:
            print(f"❌ Agent interaction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Agent interaction failed: {e}")
    
    print("\n" + "=" * 40)
    print("🎉 Backend testing complete!")
    print("\n🌐 Access URLs:")
    print(f"   • API Health: {base_url}/health")
    print(f"   • API Docs: {base_url}/docs")
    print(f"   • Interactive API: {base_url}/redoc")
    
    return True

if __name__ == "__main__":
    test_backend()