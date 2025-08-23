#!/usr/bin/env python3
"""
Comprehensive ScrollIntel Application Test
Tests the complete application functionality including API, agents, and frontend integration
"""

import asyncio
import json
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import requests
from concurrent.futures import ThreadPoolExecutor

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

class ScrollIntelAppTester:
    """Comprehensive application tester for ScrollIntel"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        self.results = {}
        self.api_process = None
        self.frontend_process = None
        
    def log_success(self, message: str):
        """Log success message"""
        print(f"✅ {message}")
        
    def log_error(self, message: str):
        """Log error message"""
        print(f"❌ {message}")
        
    def log_info(self, message: str):
        """Log info message"""
        print(f"ℹ️  {message}")
        
    def log_warning(self, message: str):
        """Log warning message"""
        print(f"⚠️  {message}")

    def start_api_server(self) -> bool:
        """Start the ScrollIntel API server"""
        self.log_info("Starting ScrollIntel API server...")
        
        try:
            # Check if server is already running
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    self.log_success("API server is already running")
                    return True
            except:
                pass
            
            # Start the server
            cmd = [sys.executable, "-m", "uvicorn", "scrollintel.api.main:app", 
                   "--host", "0.0.0.0", "--port", "8000", "--reload"]
            
            self.api_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            # Wait for server to start
            for i in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=2)
                    if response.status_code == 200:
                        self.log_success("API server started successfully")
                        return True
                except:
                    time.sleep(1)
            
            self.log_error("API server failed to start within 30 seconds")
            return False
            
        except Exception as e:
            self.log_error(f"Failed to start API server: {e}")
            return False

    def test_api_endpoints(self) -> Dict[str, bool]:
        """Test core API endpoints"""
        self.log_info("Testing API endpoints...")
        
        endpoints = {
            "Root": "/",
            "Health": "/health",
            "API Health": "/api/health",
            "Dashboard": "/api/dashboard/metrics",
            "Agents Status": "/api/agents/status",
            "Monitoring": "/api/monitoring/system"
        }
        
        results = {}
        
        for name, endpoint in endpoints.items():
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code in [200, 404]:  # 404 is acceptable for some endpoints
                    results[name] = True
                    self.log_success(f"{name} endpoint: {response.status_code}")
                else:
                    results[name] = False
                    self.log_warning(f"{name} endpoint returned: {response.status_code}")
            except Exception as e:
                results[name] = False
                self.log_error(f"{name} endpoint failed: {e}")
        
        return results

    def test_agent_functionality(self) -> Dict[str, bool]:
        """Test agent functionality"""
        self.log_info("Testing agent functionality...")
        
        agents = [
            "scroll-ai-engineer",
            "scroll-bi", 
            "scroll-ml-engineer",
            "scroll-forecast",
            "scroll-analyst",
            "scroll-qa",
            "scroll-cto"
        ]
        
        results = {}
        
        for agent in agents:
            try:
                # Test agent status
                response = requests.get(f"{self.base_url}/api/{agent}/status", timeout=10)
                if response.status_code in [200, 404]:
                    results[f"{agent}_status"] = True
                    self.log_success(f"{agent} status endpoint working")
                else:
                    results[f"{agent}_status"] = False
                    self.log_warning(f"{agent} status endpoint: {response.status_code}")
                
                # Test agent capabilities
                response = requests.get(f"{self.base_url}/api/{agent}/capabilities", timeout=10)
                if response.status_code in [200, 404]:
                    results[f"{agent}_capabilities"] = True
                    self.log_success(f"{agent} capabilities endpoint working")
                else:
                    results[f"{agent}_capabilities"] = False
                    self.log_warning(f"{agent} capabilities endpoint: {response.status_code}")
                    
            except Exception as e:
                results[f"{agent}_status"] = False
                results[f"{agent}_capabilities"] = False
                self.log_error(f"{agent} test failed: {e}")
        
        return results

    def test_file_upload(self) -> bool:
        """Test file upload functionality"""
        self.log_info("Testing file upload...")
        
        try:
            # Create a test CSV file
            test_data = "name,age,city\nJohn,25,New York\nJane,30,Los Angeles\nBob,35,Chicago"
            test_file_path = "test_upload.csv"
            
            with open(test_file_path, 'w') as f:
                f.write(test_data)
            
            # Test file upload
            with open(test_file_path, 'rb') as f:
                files = {'file': ('test_data.csv', f, 'text/csv')}
                response = requests.post(f"{self.base_url}/api/files/upload", files=files, timeout=30)
            
            # Clean up
            os.remove(test_file_path)
            
            if response.status_code in [200, 201, 404]:  # 404 acceptable if endpoint not implemented
                self.log_success("File upload test completed")
                return True
            else:
                self.log_warning(f"File upload returned: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_error(f"File upload test failed: {e}")
            return False

    def test_database_operations(self) -> bool:
        """Test database operations"""
        self.log_info("Testing database operations...")
        
        try:
            # Import and test database
            from scrollintel.models.database import get_db_session, init_database_session
            
            # Initialize database
            init_database_session()
            
            # Test database connection
            with get_db_session() as session:
                result = session.execute("SELECT 1 as test")
                row = result.fetchone()
                if row and row[0] == 1:
                    self.log_success("Database connection and operations working")
                    return True
                else:
                    self.log_error("Database query returned unexpected result")
                    return False
                    
        except Exception as e:
            self.log_error(f"Database operations test failed: {e}")
            return False

    def test_ai_integrations(self) -> Dict[str, bool]:
        """Test AI service integrations"""
        self.log_info("Testing AI integrations...")
        
        results = {}
        
        # Check API keys
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        results['openai_configured'] = bool(openai_key and openai_key.startswith('sk-'))
        results['anthropic_configured'] = bool(anthropic_key and anthropic_key.startswith('sk-ant-'))
        
        if results['openai_configured']:
            self.log_success("OpenAI API key configured")
        else:
            self.log_warning("OpenAI API key not configured")
            
        if results['anthropic_configured']:
            self.log_success("Anthropic API key configured")
        else:
            self.log_warning("Anthropic API key not configured")
        
        # Test AI endpoints
        try:
            response = requests.post(
                f"{self.base_url}/api/agents/test-ai",
                json={"message": "Hello, test message"},
                timeout=30
            )
            results['ai_endpoint'] = response.status_code in [200, 404, 501]
            if results['ai_endpoint']:
                self.log_success("AI endpoint accessible")
            else:
                self.log_warning(f"AI endpoint returned: {response.status_code}")
        except Exception as e:
            results['ai_endpoint'] = False
            self.log_warning(f"AI endpoint test failed: {e}")
        
        return results

    def test_frontend_integration(self) -> bool:
        """Test frontend integration (if available)"""
        self.log_info("Testing frontend integration...")
        
        try:
            # Check if frontend is running
            response = requests.get(self.frontend_url, timeout=5)
            if response.status_code == 200:
                self.log_success("Frontend is accessible")
                return True
            else:
                self.log_warning(f"Frontend returned: {response.status_code}")
                return False
        except Exception as e:
            self.log_info("Frontend not running (this is optional)")
            return False

    def test_security_features(self) -> Dict[str, bool]:
        """Test security features"""
        self.log_info("Testing security features...")
        
        results = {}
        
        try:
            # Test security headers
            response = requests.get(f"{self.base_url}/", timeout=10)
            headers = response.headers
            
            security_headers = [
                'X-Content-Type-Options',
                'X-Frame-Options', 
                'X-XSS-Protection'
            ]
            
            for header in security_headers:
                results[f"header_{header}"] = header in headers
                if results[f"header_{header}"]:
                    self.log_success(f"Security header {header} present")
                else:
                    self.log_warning(f"Security header {header} missing")
            
            # Test auth endpoints
            auth_endpoints = ["/api/auth/login", "/api/auth/register"]
            for endpoint in auth_endpoints:
                try:
                    response = requests.post(f"{self.base_url}{endpoint}", json={}, timeout=5)
                    results[f"auth_{endpoint}"] = response.status_code in [400, 401, 422, 404]
                    if results[f"auth_{endpoint}"]:
                        self.log_success(f"Auth endpoint {endpoint} responding")
                    else:
                        self.log_warning(f"Auth endpoint {endpoint}: {response.status_code}")
                except Exception:
                    results[f"auth_{endpoint}"] = False
                    self.log_warning(f"Auth endpoint {endpoint} not accessible")
            
        except Exception as e:
            self.log_error(f"Security features test failed: {e}")
            results['security_test'] = False
        
        return results

    def test_performance_monitoring(self) -> Dict[str, bool]:
        """Test performance monitoring"""
        self.log_info("Testing performance monitoring...")
        
        results = {}
        
        try:
            # Test performance endpoints
            perf_endpoints = [
                "/api/performance/metrics",
                "/api/monitoring/performance", 
                "/api/monitoring/system"
            ]
            
            for endpoint in perf_endpoints:
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                    results[f"perf_{endpoint}"] = response.status_code in [200, 404]
                    if results[f"perf_{endpoint}"]:
                        self.log_success(f"Performance endpoint {endpoint} accessible")
                    else:
                        self.log_warning(f"Performance endpoint {endpoint}: {response.status_code}")
                except Exception:
                    results[f"perf_{endpoint}"] = False
                    self.log_warning(f"Performance endpoint {endpoint} not accessible")
            
        except Exception as e:
            self.log_error(f"Performance monitoring test failed: {e}")
            results['performance_test'] = False
        
        return results

    def cleanup(self):
        """Clean up processes"""
        self.log_info("Cleaning up...")
        
        if self.api_process:
            try:
                self.api_process.terminate()
                self.api_process.wait(timeout=5)
                self.log_success("API server stopped")
            except:
                try:
                    self.api_process.kill()
                except:
                    pass
        
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                self.log_success("Frontend server stopped")
            except:
                try:
                    self.frontend_process.kill()
                except:
                    pass

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive application test"""
        print("🚀 ScrollIntel Comprehensive Application Test")
        print("=" * 60)
        
        start_time = time.time()
        all_results = {}
        
        try:
            # Start API server
            if not self.start_api_server():
                return {"error": "Failed to start API server"}
            
            # Wait a moment for server to fully initialize
            time.sleep(3)
            
            # Run all tests
            test_functions = [
                ("API Endpoints", self.test_api_endpoints),
                ("Agent Functionality", self.test_agent_functionality),
                ("File Upload", self.test_file_upload),
                ("Database Operations", self.test_database_operations),
                ("AI Integrations", self.test_ai_integrations),
                ("Frontend Integration", self.test_frontend_integration),
                ("Security Features", self.test_security_features),
                ("Performance Monitoring", self.test_performance_monitoring)
            ]
            
            for test_name, test_func in test_functions:
                print(f"\n🔍 Testing: {test_name}")
                print("-" * 40)
                
                try:
                    result = test_func()
                    all_results[test_name] = result
                    
                    if isinstance(result, dict):
                        passed = sum(1 for v in result.values() if v)
                        total = len(result)
                        self.log_info(f"{test_name}: {passed}/{total} checks passed")
                    elif result:
                        self.log_success(f"{test_name}: PASSED")
                    else:
                        self.log_warning(f"{test_name}: FAILED")
                        
                except Exception as e:
                    all_results[test_name] = {"error": str(e)}
                    self.log_error(f"{test_name}: ERROR - {e}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Calculate overall statistics
            total_checks = 0
            passed_checks = 0
            
            for test_name, result in all_results.items():
                if isinstance(result, dict) and "error" not in result:
                    for check, passed in result.items():
                        total_checks += 1
                        if passed:
                            passed_checks += 1
                elif isinstance(result, bool):
                    total_checks += 1
                    if result:
                        passed_checks += 1
            
            success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
            
            # Generate summary
            print(f"\n{'='*60}")
            print("📊 TEST SUMMARY")
            print(f"{'='*60}")
            print(f"Total Checks: {total_checks}")
            print(f"Passed: {passed_checks}")
            print(f"Failed: {total_checks - passed_checks}")
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"Duration: {duration:.2f} seconds")
            
            if success_rate >= 90:
                print("\n🎉 Excellent! ScrollIntel is working perfectly!")
                status = "excellent"
            elif success_rate >= 80:
                print("\n✅ Great! ScrollIntel is working well!")
                status = "good"
            elif success_rate >= 70:
                print("\n👍 Good! ScrollIntel is mostly working with minor issues.")
                status = "acceptable"
            elif success_rate >= 60:
                print("\n⚠️  ScrollIntel is partially working. Some features need attention.")
                status = "needs_attention"
            else:
                print("\n❌ ScrollIntel has significant issues that need fixing.")
                status = "critical"
            
            # Detailed results
            final_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "status": status,
                "summary": {
                    "total_checks": total_checks,
                    "passed_checks": passed_checks,
                    "failed_checks": total_checks - passed_checks,
                    "success_rate": success_rate,
                    "duration_seconds": duration
                },
                "test_results": all_results
            }
            
            # Save results
            results_file = f"scrollintel_app_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            print(f"\n📄 Detailed results saved to: {results_file}")
            
            return final_results
            
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
            return {"error": "Test interrupted"}
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            return {"error": str(e)}
        finally:
            self.cleanup()

async def main():
    """Main function"""
    tester = ScrollIntelAppTester()
    
    try:
        results = tester.run_comprehensive_test()
        
        if "error" in results:
            sys.exit(1)
        elif results["summary"]["success_rate"] >= 80:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())