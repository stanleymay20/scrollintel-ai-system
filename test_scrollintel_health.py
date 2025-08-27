#!/usr/bin/env python3
"""
ScrollIntel Health Check Script
Tests all endpoints and provides deployment status.
"""

import requests
import json
import time
import sys
from typing import Dict, Any

def test_endpoint(url: str, name: str, timeout: int = 10) -> Dict[str, Any]:
    """Test a single endpoint."""
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        response_time = (time.time() - start_time) * 1000
        
        return {
            "name": name,
            "url": url,
            "status": "✅ PASS" if response.status_code == 200 else f"❌ FAIL ({response.status_code})",
            "response_time": f"{response_time:.2f}ms",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else None
        }
    except requests.exceptions.Timeout:
        return {
            "name": name,
            "url": url,
            "status": "❌ TIMEOUT",
            "response_time": f">{timeout}s",
            "status_code": None,
            "success": False,
            "error": "Request timeout"
        }
    except requests.exceptions.ConnectionError:
        return {
            "name": name,
            "url": url,
            "status": "❌ CONNECTION ERROR",
            "response_time": "N/A",
            "status_code": None,
            "success": False,
            "error": "Connection refused"
        }
    except Exception as e:
        return {
            "name": name,
            "url": url,
            "status": f"❌ ERROR",
            "response_time": "N/A",
            "status_code": None,
            "success": False,
            "error": str(e)
        }

def main():
    """Run health checks."""
    print("🏥 ScrollIntel Health Check")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Define endpoints to test
    endpoints = [
        (f"{base_url}/", "Root Endpoint"),
        (f"{base_url}/health", "Health Check"),
        (f"{base_url}/status", "System Status"),
        (f"{base_url}/docs", "API Documentation"),
        (f"{base_url}/health/detailed", "Detailed Health"),
        (f"{base_url}/health/agents", "Agent Health"),
        (f"{base_url}/health/readiness", "Readiness Probe"),
        (f"{base_url}/health/liveness", "Liveness Probe"),
    ]
    
    results = []
    
    print(f"\n🔍 Testing {len(endpoints)} endpoints...\n")
    
    for url, name in endpoints:
        result = test_endpoint(url, name)
        results.append(result)
        
        print(f"{result['status']:<20} {name:<20} {result['response_time']:<10} {url}")
        
        if result.get('error'):
            print(f"   Error: {result['error']}")
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    success_rate = (successful / total) * 100
    
    print(f"\n📊 Summary:")
    print(f"   Total Tests: {total}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {total - successful}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print(f"\n🎉 All tests passed! ScrollIntel is healthy.")
        
        # Show some system info if available
        root_result = next((r for r in results if r['name'] == 'Root Endpoint'), None)
        if root_result and root_result.get('data'):
            data = root_result['data']
            if isinstance(data, dict) and 'data' in data:
                system_data = data['data']
                print(f"\n📋 System Information:")
                print(f"   Service: {system_data.get('name', 'Unknown')}")
                print(f"   Version: {system_data.get('version', 'Unknown')}")
                print(f"   Environment: {system_data.get('environment', 'Unknown')}")
                print(f"   Status: {system_data.get('status', 'Unknown')}")
        
        return 0
    
    elif success_rate >= 50:
        print(f"\n⚠️  Some tests failed, but core services are running.")
        print(f"   Check the failed endpoints and logs for details.")
        return 1
    
    else:
        print(f"\n❌ Most tests failed. ScrollIntel may not be running properly.")
        print(f"\n🔧 Troubleshooting Steps:")
        print(f"   1. Check if Docker containers are running:")
        print(f"      docker-compose ps")
        print(f"   2. Check application logs:")
        print(f"      docker-compose logs backend")
        print(f"   3. Check if port 8000 is accessible:")
        print(f"      curl http://localhost:8000/")
        print(f"   4. Restart the services:")
        print(f"      docker-compose down && docker-compose up -d")
        return 2

if __name__ == "__main__":
    sys.exit(main())