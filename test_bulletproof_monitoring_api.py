#!/usr/bin/env python3
"""
Quick test to verify bulletproof monitoring API routes work correctly.
Tests the key endpoints to ensure they respond properly.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from fastapi import FastAPI
import pytest

# Import the routes
from scrollintel.api.routes.bulletproof_monitoring_routes import router

def test_bulletproof_monitoring_api_routes():
    """Test that the bulletproof monitoring API routes work correctly."""
    
    # Create a test FastAPI app
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    
    # Create test client
    client = TestClient(app)
    
    print("Testing Bulletproof Monitoring API Routes...")
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    response = client.get("/api/v1/monitoring/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200
    assert "status" in response.json()
    
    # Test 2: Get dashboard data
    print("\n2. Testing dashboard data endpoint...")
    response = client.get("/api/v1/monitoring/dashboard")
    print(f"   Status: {response.status_code}")
    print(f"   Response keys: {list(response.json().keys())}")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "alerts" in data
    
    # Test 3: Record user action
    print("\n3. Testing record user action endpoint...")
    user_action_data = {
        "user_id": "test_user_123",
        "action": "test_action",
        "success": True,
        "duration_ms": 250.0,
        "context": {"test": "data"}
    }
    response = client.post("/api/v1/monitoring/user-action", json=user_action_data)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200
    
    # Test 4: Record user satisfaction
    print("\n4. Testing record user satisfaction endpoint...")
    satisfaction_data = {
        "user_id": "test_user_123",
        "satisfaction_score": 4.5,
        "context": {"feature": "test_feature"}
    }
    response = client.post("/api/v1/monitoring/user-satisfaction", json=satisfaction_data)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200
    
    # Test 5: Record system health
    print("\n5. Testing record system health endpoint...")
    health_data = {
        "component": "test_component",
        "health_score": 95.0,
        "metrics": {"cpu_usage": 45.2, "memory_usage": 60.1}
    }
    response = client.post("/api/v1/monitoring/system-health", json=health_data)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200
    
    # Test 6: Get health report
    print("\n6. Testing health report endpoint...")
    response = client.get("/api/v1/monitoring/health-report")
    print(f"   Status: {response.status_code}")
    print(f"   Response keys: {list(response.json().keys())}")
    assert response.status_code == 200
    data = response.json()
    assert "system_health_score" in data
    assert "data_points_analyzed" in data
    
    # Test 7: Get failure patterns
    print("\n7. Testing failure patterns endpoint...")
    response = client.get("/api/v1/monitoring/failure-patterns")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    
    print("\n✅ All API route tests passed!")
    return True

def test_error_handling():
    """Test error handling in API routes."""
    
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)
    
    print("\nTesting Error Handling...")
    
    # Test invalid user action data
    print("1. Testing invalid user action data...")
    invalid_data = {"invalid": "data"}
    response = client.post("/api/v1/monitoring/user-action", json=invalid_data)
    print(f"   Status: {response.status_code}")
    # Should handle gracefully (either 422 validation error or 200 with error handling)
    assert response.status_code in [200, 422]
    
    # Test invalid satisfaction data
    print("2. Testing invalid satisfaction data...")
    invalid_satisfaction = {"user_id": "test", "satisfaction_score": "invalid"}
    response = client.post("/api/v1/monitoring/user-satisfaction", json=invalid_satisfaction)
    print(f"   Status: {response.status_code}")
    assert response.status_code in [200, 422]
    
    print("✅ Error handling tests passed!")

if __name__ == "__main__":
    try:
        # Run the main API tests
        test_bulletproof_monitoring_api_routes()
        
        # Run error handling tests
        test_error_handling()
        
        print("\n🎉 All bulletproof monitoring API tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)