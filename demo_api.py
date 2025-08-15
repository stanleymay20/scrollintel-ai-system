#!/usr/bin/env python3
"""
Demo script to test the ScrollIntel FastAPI Gateway.
Run this to start the API server and test basic functionality.
"""

import asyncio
import requests
import time
from scrollintel.api.main import app
import uvicorn
from threading import Thread
import sys

def start_server():
    """Start the FastAPI server in a separate thread."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

def test_api_endpoints():
    """Test various API endpoints."""
    base_url = "http://127.0.0.1:8000"
    
    print("🚀 Testing ScrollIntel™ API Gateway")
    print("=" * 50)
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    try:
        # Test root endpoint
        print("\n1. Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Service: {data['name']}")
            print(f"   Version: {data['version']}")
            print(f"   Status: {data['status']}")
        
        # Test health check
        print("\n2. Testing health check...")
        response = requests.get(f"{base_url}/health/")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Health: {data['status']}")
        
        # Test system status
        print("\n3. Testing system status...")
        response = requests.get(f"{base_url}/status")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   System: {data['system']}")
            print(f"   Environment: {data['environment']}")
            print(f"   Agents: {data['agents']['total_agents']}")
        
        # Test authentication
        print("\n4. Testing authentication...")
        auth_data = {
            "email": "admin@scrollintel.com",
            "password": "admin123",
            "remember_me": False
        }
        response = requests.post(f"{base_url}/auth/login", json=auth_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Token Type: {data['token_type']}")
            print(f"   User Role: {data['user_info']['role']}")
            token = data['access_token']
            
            # Test authenticated endpoint
            print("\n5. Testing authenticated endpoint...")
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(f"{base_url}/agents/", headers=headers)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                agents = response.json()
                print(f"   Agents found: {len(agents)}")
        
        # Test password requirements
        print("\n6. Testing password requirements...")
        response = requests.get(f"{base_url}/auth/password-requirements")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Requirements available: {len(data['requirements'])}")
        
        # Test protected endpoint without auth
        print("\n7. Testing protected endpoint without auth...")
        response = requests.get(f"{base_url}/admin/users")
        print(f"   Status: {response.status_code} (Expected: 401)")
        
        print("\n✅ API Gateway testing completed successfully!")
        print("\n📚 Available endpoints:")
        print("   GET  /                    - Root endpoint")
        print("   GET  /status             - System status")
        print("   GET  /health/            - Basic health check")
        print("   GET  /health/detailed    - Detailed health check")
        print("   POST /auth/login         - User authentication")
        print("   GET  /auth/password-requirements - Password requirements")
        print("   GET  /agents/            - List agents (requires auth)")
        print("   POST /agents/execute     - Execute agent (requires auth)")
        print("   GET  /admin/users        - Admin endpoints (requires admin auth)")
        print("   GET  /docs               - API documentation (dev mode)")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on port 8000.")
    except Exception as e:
        print(f"❌ Error testing API: {str(e)}")

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Just run tests without starting server
        test_api_endpoints()
    else:
        print("🚀 Starting ScrollIntel™ API Gateway Demo")
        print("=" * 50)
        print("Starting server on http://127.0.0.1:8000")
        print("Press Ctrl+C to stop the server")
        print("Add 'test' argument to run API tests")
        print("=" * 50)
        
        # Start server in background thread
        server_thread = Thread(target=start_server, daemon=True)
        server_thread.start()
        
        # Run tests
        test_api_endpoints()
        
        print("\n🌐 Server is running at http://127.0.0.1:8000")
        print("📖 API docs available at http://127.0.0.1:8000/docs")
        print("Press Ctrl+C to stop...")
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")

if __name__ == "__main__":
    main()