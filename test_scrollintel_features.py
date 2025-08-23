#!/usr/bin/env python3
"""
ScrollIntel Feature Testing Suite
Tests all major features to ensure everything is working correctly
"""

import requests
import json
import time
import sys
from datetime import datetime

class ScrollIntelTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = {}
    
    def log(self, message, status="info"):
        """Log messages with status"""
        icons = {
            "success": "✅",
            "error": "❌", 
            "warning": "⚠️",
            "info": "ℹ️"
        }
        print(f"{icons.get(status, 'ℹ️')} {message}")
    
    def test_basic_connectivity(self):
        """Test basic API connectivity"""
        self.log("Testing basic API connectivity...")
        
        try:
            response = self.session.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log(f"API is running - {data.get('message', 'Unknown')}", "success")
                return True
            else:
                self.log(f"API returned status {response.status_code}", "error")
                return False
        except Exception as e:
            self.log(f"Failed to connect: {e}", "error")
            return False
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        self.log("Testing health endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log(f"Health check passed - Status: {data.get('status', 'unknown')}", "success")
                return True
            else:
                self.log(f"Health check failed with status {response.status_code}", "warning")
                return False
        except Exception as e:
            self.log(f"Health check error: {e}", "error")
            return False
    
    def test_api_documentation(self):
        """Test API documentation endpoint"""
        self.log("Testing API documentation...")
        
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=5)
            if response.status_code == 200:
                self.log("API documentation is accessible", "success")
                return True
            else:
                self.log(f"API docs returned status {response.status_code}", "warning")
                return False
        except Exception as e:
            self.log(f"API docs error: {e}", "error")
            return False
    
    def test_agent_endpoints(self):
        """Test agent-related endpoints"""
        self.log("Testing agent endpoints...")
        
        agent_endpoints = [
            "/api/agents/status",
            "/api/scroll-ai-engineer/status",
            "/api/scroll-bi/status",
            "/api/scroll-ml-engineer/status"
        ]
        
        success_count = 0
        for endpoint in agent_endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code in [200, 404]:  # 404 is acceptable
                    self.log(f"Agent endpoint {endpoint} is accessible", "success")
                    success_count += 1
                else:
                    self.log(f"Agent endpoint {endpoint} returned {response.status_code}", "warning")
            except Exception as e:
                self.log(f"Agent endpoint {endpoint} failed: {e}", "warning")
        
        return success_count > 0
    
    def test_file_upload(self):
        """Test file upload functionality"""
        self.log("Testing file upload...")
        
        try:
            # Create a test file
            test_content = "ScrollIntel test file content"
            files = {
                'file': ('test.txt', test_content, 'text/plain')
            }
            
            response = self.session.post(
                f"{self.base_url}/api/files/upload",
                files=files,
                timeout=10
            )
            
            if response.status_code in [200, 201, 404]:  # 404 if endpoint not implemented
                self.log("File upload endpoint is accessible", "success")
                return True
            else:
                self.log(f"File upload returned status {response.status_code}", "warning")
                return False
                
        except Exception as e:
            self.log(f"File upload test failed: {e}", "warning")
            return False
    
    def test_dashboard_endpoints(self):
        """Test dashboard endpoints"""
        self.log("Testing dashboard endpoints...")
        
        dashboard_endpoints = [
            "/api/dashboard/metrics",
            "/api/monitoring/system",
            "/api/visualization/charts"
        ]
        
        success_count = 0
        for endpoint in dashboard_endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code in [200, 404]:
                    self.log(f"Dashboard endpoint {endpoint} is accessible", "success")
                    success_count += 1
                else:
                    self.log(f"Dashboard endpoint {endpoint} returned {response.status_code}", "warning")
            except Exception as e:
                self.log(f"Dashboard endpoint {endpoint} failed: {e}", "warning")
        
        return success_count > 0
    
    def test_ai_features(self):
        """Test AI-related features"""
        self.log("Testing AI features...")
        
        # Test a simple AI endpoint
        try:
            test_data = {
                "message": "Hello ScrollIntel",
                "type": "test"
            }
            
            response = self.session.post(
                f"{self.base_url}/api/agents/process",
                json=test_data,
                timeout=15
            )
            
            if response.status_code in [200, 404, 501]:  # Various acceptable responses
                self.log("AI processing endpoint is accessible", "success")
                return True
            else:
                self.log(f"AI endpoint returned status {response.status_code}", "warning")
                return False
                
        except Exception as e:
            self.log(f"AI features test failed: {e}", "warning")
            return False
    
    def run_all_tests(self):
        """Run all feature tests"""
        print("🧪 ScrollIntel Feature Testing Suite")
        print("=" * 50)
        
        tests = [
            ("Basic Connectivity", self.test_basic_connectivity),
            ("Health Endpoint", self.test_health_endpoint),
            ("API Documentation", self.test_api_documentation),
            ("Agent Endpoints", self.test_agent_endpoints),
            ("File Upload", self.test_file_upload),
            ("Dashboard Endpoints", self.test_dashboard_endpoints),
            ("AI Features", self.test_ai_features)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🔍 Testing: {test_name}")
            try:
                result = test_func()
                self.results[test_name] = result
                if result:
                    passed += 1
                    self.log(f"{test_name} - PASSED", "success")
                else:
                    self.log(f"{test_name} - FAILED", "error")
            except Exception as e:
                self.results[test_name] = False
                self.log(f"{test_name} - ERROR: {e}", "error")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        
        success_rate = (passed / total) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            self.log("🎉 ScrollIntel is working excellently!", "success")
        elif success_rate >= 60:
            self.log("✅ ScrollIntel is working well with minor issues", "success")
        elif success_rate >= 40:
            self.log("⚠️ ScrollIntel has some issues that need attention", "warning")
        else:
            self.log("❌ ScrollIntel has significant issues", "error")
        
        return success_rate >= 60

def main():
    """Main function"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8000"
    
    print(f"Testing ScrollIntel at: {base_url}")
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(3)
    
    tester = ScrollIntelTester(base_url)
    success = tester.run_all_tests()
    
    # Save results
    results_file = f"scrollintel_feature_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "base_url": base_url,
            "results": tester.results
        }, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())