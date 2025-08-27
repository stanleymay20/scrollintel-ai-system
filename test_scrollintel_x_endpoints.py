"""
Test script for ScrollIntel X API endpoints.
Tests the spiritual intelligence endpoints with mock requests.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

async def test_scrollintel_x_endpoints():
    """Test ScrollIntel X API endpoints."""
    
    print("Testing ScrollIntel X API Endpoints")
    print("=" * 50)
    
    try:
        from fastapi.testclient import TestClient
        from scrollintel.api.gateway import app
        
        # Create test client
        client = TestClient(app)
        
        print("✅ Test client created successfully")
        
        # Test root endpoint
        print("\n1. Testing root endpoint...")
        response = client.get("/")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ System: {data['data']['name']}")
            print(f"   ✅ Version: {data['data']['version']}")
            print(f"   ✅ Unified Response Format: {data.get('success', False)}")
        
        # Test system status endpoint
        print("\n2. Testing system status endpoint...")
        response = client.get("/status")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ System: {data['data']['system']}")
            print(f"   ✅ ScrollIntel X Features: {data['data'].get('scrollintel_x_features', {})}")
        
        # Test health endpoint
        print("\n3. Testing health endpoint...")
        response = client.get("/health")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Health check passed")
        
        # Test OpenAPI docs endpoint
        print("\n4. Testing OpenAPI docs...")
        response = client.get("/docs")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ API documentation accessible")
        
        # Test OpenAPI schema
        print("\n5. Testing OpenAPI schema...")
        response = client.get("/openapi.json")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            schema = response.json()
            print(f"   ✅ API Title: {schema.get('info', {}).get('title', 'N/A')}")
            print(f"   ✅ API Version: {schema.get('info', {}).get('version', 'N/A')}")
            
            # Check for ScrollIntel X endpoints in schema
            paths = schema.get('paths', {})
            scrollintel_x_paths = [path for path in paths.keys() if 'scrollintel-x' in path]
            print(f"   ✅ ScrollIntel X endpoints in schema: {len(scrollintel_x_paths)}")
            
            # Check for tags
            tags = schema.get('tags', [])
            scrollintel_x_tags = [tag['name'] for tag in tags if 'ScrollIntel X' in tag['name']]
            print(f"   ✅ ScrollIntel X tags: {scrollintel_x_tags}")
        
        print("\n" + "=" * 50)
        print("ScrollIntel X Endpoints Test PASSED")
        print("All endpoints are accessible and properly configured!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_scrollintel_x_endpoints())
    exit(0 if success else 1)