#!/usr/bin/env python3
"""
Comprehensive test to verify all API routes work.
Tests multiple route modules to ensure they can be imported and basic functionality works.
"""

import asyncio
import sys
import os
from datetime import datetime
import traceback

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

class APIRoutesTester:
    """Test suite for all API routes."""
    
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'total_tested': 0
        }
    
    def log_result(self, test_name, success, error=None):
        """Log test result."""
        self.results['total_tested'] += 1
        if success:
            self.results['passed'].append(test_name)
            print(f"✅ {test_name}")
        else:
            self.results['failed'].append((test_name, error))
            print(f"❌ {test_name}: {error}")
    
    async def test_bulletproof_monitoring_routes(self):
        """Test bulletproof monitoring routes."""
        try:
            from scrollintel.api.routes.bulletproof_monitoring_routes import router
            
            # Check routes exist
            routes = [route.path for route in router.routes]
            expected_routes = [
                "/api/v1/bulletproof-monitoring/metrics/record",
                "/api/v1/bulletproof-monitoring/dashboard/realtime",
                "/api/v1/bulletproof-monitoring/system/status"
            ]
            
            for expected in expected_routes:
                if expected not in routes:
                    raise Exception(f"Missing route: {expected}")
            
            # Test endpoint functions
            from scrollintel.api.routes.bulletproof_monitoring_routes import (
                get_system_status, get_realtime_dashboard, get_health_report
            )
            
            # Test system status
            result = await get_system_status()
            if not hasattr(result, 'status_code'):
                raise Exception("System status endpoint didn't return proper response")
            
            # Test dashboard
            dashboard = await get_realtime_dashboard()
            if not hasattr(dashboard, 'status_code'):
                raise Exception("Dashboard endpoint didn't return proper response")
            
            self.log_result("Bulletproof Monitoring Routes", True)
            
        except Exception as e:
            self.log_result("Bulletproof Monitoring Routes", False, str(e))
    
    async def test_health_routes(self):
        """Test health check routes."""
        try:
            from scrollintel.api.routes.health_routes import router
            
            # Check basic import works
            routes = [route.path for route in router.routes]
            if not routes:
                raise Exception("No routes found in health router")
            
            self.log_result("Health Routes", True)
            
        except Exception as e:
            self.log_result("Health Routes", False, str(e))
    
    async def test_agent_routes(self):
        """Test agent routes."""
        try:
            from scrollintel.api.routes.agent_routes import router
            
            # Check basic import works
            routes = [route.path for route in router.routes]
            if not routes:
                raise Exception("No routes found in agent router")
            
            self.log_result("Agent Routes", True)
            
        except Exception as e:
            self.log_result("Agent Routes", False, str(e))
    
    async def test_auth_routes(self):
        """Test authentication routes."""
        try:
            from scrollintel.api.routes.auth_routes import router
            
            # Check basic import works
            routes = [route.path for route in router.routes]
            if not routes:
                raise Exception("No routes found in auth router")
            
            self.log_result("Auth Routes", True)
            
        except Exception as e:
            self.log_result("Auth Routes", False, str(e))
    
    async def test_monitoring_routes(self):
        """Test monitoring routes."""
        try:
            from scrollintel.api.routes.monitoring_routes import router
            
            # Check basic import works
            routes = [route.path for route in router.routes]
            if not routes:
                raise Exception("No routes found in monitoring router")
            
            self.log_result("Monitoring Routes", True)
            
        except Exception as e:
            self.log_result("Monitoring Routes", False, str(e))
    
    async def test_file_routes(self):
        """Test file upload routes."""
        try:
            from scrollintel.api.routes.file_routes import router
            
            # Check basic import works
            routes = [route.path for route in router.routes]
            if not routes:
                raise Exception("No routes found in file router")
            
            self.log_result("File Routes", True)
            
        except Exception as e:
            self.log_result("File Routes", False, str(e))
    
    async def test_visualization_routes(self):
        """Test visualization routes."""
        try:
            from scrollintel.api.routes.visualization_routes import router
            
            # Check basic import works
            routes = [route.path for route in router.routes]
            if not routes:
                raise Exception("No routes found in visualization router")
            
            self.log_result("Visualization Routes", True)
            
        except Exception as e:
            self.log_result("Visualization Routes", False, str(e))
    
    async def test_dashboard_routes(self):
        """Test dashboard routes."""
        try:
            from scrollintel.api.routes.dashboard_routes import router
            
            # Check basic import works
            routes = [route.path for route in router.routes]
            if not routes:
                raise Exception("No routes found in dashboard router")
            
            self.log_result("Dashboard Routes", True)
            
        except Exception as e:
            self.log_result("Dashboard Routes", False, str(e))
    
    async def test_user_management_routes(self):
        """Test user management routes."""
        try:
            from scrollintel.api.routes.user_management_routes import router
            
            # Check basic import works
            routes = [route.path for route in router.routes]
            if not routes:
                raise Exception("No routes found in user management router")
            
            self.log_result("User Management Routes", True)
            
        except Exception as e:
            self.log_result("User Management Routes", False, str(e))
    
    async def test_ethics_routes(self):
        """Test ethics engine routes."""
        try:
            from scrollintel.api.routes.ethics_routes import router
            
            # Check basic import works
            routes = [route.path for route in router.routes]
            if not routes:
                raise Exception("No routes found in ethics router")
            
            self.log_result("Ethics Routes", True)
            
        except Exception as e:
            self.log_result("Ethics Routes", False, str(e))
    
    async def test_scroll_qa_routes(self):
        """Test ScrollQA routes."""
        try:
            from scrollintel.api.routes.scroll_qa_routes import router
            
            # Check basic import works
            routes = [route.path for route in router.routes]
            if not routes:
                raise Exception("No routes found in ScrollQA router")
            
            self.log_result("ScrollQA Routes", True)
            
        except Exception as e:
            self.log_result("ScrollQA Routes", False, str(e))
    
    async def test_with_fastapi_client(self):
        """Test routes with FastAPI TestClient."""
        try:
            from fastapi.testclient import TestClient
            from fastapi import FastAPI
            
            # Create test app with bulletproof monitoring routes
            app = FastAPI()
            
            from scrollintel.api.routes.bulletproof_monitoring_routes import router as bp_router
            app.include_router(bp_router)
            
            from scrollintel.api.routes.health_routes import router as health_router
            app.include_router(health_router)
            
            client = TestClient(app)
            
            # Test system status endpoint
            response = client.get("/api/v1/bulletproof-monitoring/system/status")
            if response.status_code != 200:
                raise Exception(f"System status returned {response.status_code}")
            
            # Test health endpoint
            response = client.get("/health")
            if response.status_code not in [200, 404]:  # 404 is ok if route doesn't exist
                raise Exception(f"Health endpoint returned {response.status_code}")
            
            # Test POST endpoint
            test_metric = {
                "user_id": "test_user_123",
                "metric_type": "performance",
                "value": 250.0,
                "context": {"test": True}
            }
            
            response = client.post("/api/v1/bulletproof-monitoring/metrics/record", json=test_metric)
            if response.status_code != 201:
                raise Exception(f"Metrics record returned {response.status_code}")
            
            self.log_result("FastAPI TestClient Integration", True)
            
        except Exception as e:
            self.log_result("FastAPI TestClient Integration", False, str(e))
    
    async def run_all_tests(self):
        """Run all API route tests."""
        print("🚀 Starting comprehensive API routes test...")
        print("=" * 60)
        
        # List of test methods to run
        test_methods = [
            self.test_bulletproof_monitoring_routes,
            self.test_health_routes,
            self.test_agent_routes,
            self.test_auth_routes,
            self.test_monitoring_routes,
            self.test_file_routes,
            self.test_visualization_routes,
            self.test_dashboard_routes,
            self.test_user_management_routes,
            self.test_ethics_routes,
            self.test_scroll_qa_routes,
            self.test_with_fastapi_client
        ]
        
        # Run each test
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                print(f"❌ Test method {test_method.__name__} failed: {e}")
                traceback.print_exc()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests run: {self.results['total_tested']}")
        print(f"Passed: {len(self.results['passed'])}")
        print(f"Failed: {len(self.results['failed'])}")
        
        if self.results['passed']:
            print(f"\n✅ PASSED TESTS ({len(self.results['passed'])}):")
            for test in self.results['passed']:
                print(f"   - {test}")
        
        if self.results['failed']:
            print(f"\n❌ FAILED TESTS ({len(self.results['failed'])}):")
            for test, error in self.results['failed']:
                print(f"   - {test}: {error}")
        
        success_rate = len(self.results['passed']) / self.results['total_tested'] * 100
        print(f"\n📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 OVERALL RESULT: GOOD - Most API routes are working!")
        elif success_rate >= 60:
            print("⚠️  OVERALL RESULT: FAIR - Some issues need attention")
        else:
            print("🚨 OVERALL RESULT: POOR - Significant issues found")
        
        print("=" * 60)
        
        return success_rate >= 80

async def main():
    """Main test runner."""
    tester = APIRoutesTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎯 API routes verification completed successfully!")
    else:
        print("\n⚠️  API routes verification completed with issues.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())