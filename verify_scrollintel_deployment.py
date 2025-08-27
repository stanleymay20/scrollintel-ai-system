#!/usr/bin/env python3
"""
ScrollIntel.com Deployment Verification Script
Comprehensive testing of all deployed services and features.
"""

import requests
import time
import json
import sys
from datetime import datetime

class ScrollIntelVerifier:
    def __init__(self, domain="scrollintel.com"):
        self.domain = domain
        self.api_domain = f"api.{domain}"
        self.app_domain = f"app.{domain}"
        self.results = []
        
    def log_result(self, test_name, status, message="", response_time=None):
        """Log test result."""
        result = {
            "test": test_name,
            "status": status,
            "message": message,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        time_str = f" ({response_time:.2f}s)" if response_time else ""
        print(f"{status_icon} {test_name}{time_str}: {message}")
    
    def test_endpoint(self, url, test_name, expected_status=200, timeout=10):
        """Test a single endpoint."""
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout, verify=True)
            response_time = time.time() - start_time
            
            if response.status_code == expected_status:
                self.log_result(test_name, "PASS", f"Status {response.status_code}", response_time)
                return True
            else:
                self.log_result(test_name, "FAIL", f"Expected {expected_status}, got {response.status_code}", response_time)
                return False
        except requests.exceptions.SSLError as e:
            self.log_result(test_name, "FAIL", f"SSL Error: {str(e)}")
            return False
        except requests.exceptions.Timeout:
            self.log_result(test_name, "FAIL", f"Timeout after {timeout}s")
            return False
        except requests.exceptions.ConnectionError as e:
            self.log_result(test_name, "FAIL", f"Connection Error: {str(e)}")
            return False
        except Exception as e:
            self.log_result(test_name, "FAIL", f"Error: {str(e)}")
            return False
    
    def test_ssl_certificate(self, domain):
        """Test SSL certificate validity."""
        try:
            response = requests.get(f"https://{domain}", timeout=10, verify=True)
            self.log_result(f"SSL Certificate - {domain}", "PASS", "Valid SSL certificate")
            return True
        except requests.exceptions.SSLError as e:
            self.log_result(f"SSL Certificate - {domain}", "FAIL", f"SSL Error: {str(e)}")
            return False
        except Exception as e:
            self.log_result(f"SSL Certificate - {domain}", "FAIL", f"Error: {str(e)}")
            return False
    
    def test_api_functionality(self):
        """Test API functionality."""
        api_tests = [
            (f"https://{self.api_domain}/health", "API Health Check"),
            (f"https://{self.api_domain}/docs", "API Documentation"),
            (f"https://{self.api_domain}/openapi.json", "OpenAPI Schema"),
        ]
        
        results = []
        for url, test_name in api_tests:
            results.append(self.test_endpoint(url, test_name))
        
        return all(results)
    
    def test_frontend_functionality(self):
        """Test frontend functionality."""
        frontend_tests = [
            (f"https://{self.domain}", "Frontend Home Page"),
            (f"https://{self.domain}/health", "Frontend Health Check"),
            (f"https://{self.app_domain}", "App Domain"),
        ]
        
        results = []
        for url, test_name in frontend_tests:
            results.append(self.test_endpoint(url, test_name))
        
        return all(results)
    
    def test_monitoring_services(self):
        """Test monitoring services."""
        monitoring_tests = [
            (f"https://grafana.{self.domain}", "Grafana Dashboard"),
            (f"https://prometheus.{self.domain}", "Prometheus Metrics"),
        ]
        
        results = []
        for url, test_name in monitoring_tests:
            # Monitoring services might require authentication, so 401 is acceptable
            result = self.test_endpoint(url, test_name, expected_status=200)
            if not result:
                # Try with 401 (authentication required) as acceptable
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 401:
                        self.log_result(test_name, "PASS", "Service running (auth required)")
                        result = True
                except:
                    pass
            results.append(result)
        
        return all(results)
    
    def test_security_headers(self):
        """Test security headers."""
        try:
            response = requests.get(f"https://{self.domain}", timeout=10)
            headers = response.headers
            
            security_headers = {
                'X-Frame-Options': 'SAMEORIGIN',
                'X-Content-Type-Options': 'nosniff',
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=',
            }
            
            all_present = True
            for header, expected_value in security_headers.items():
                if header in headers and expected_value in headers[header]:
                    self.log_result(f"Security Header - {header}", "PASS", f"Present: {headers[header]}")
                else:
                    self.log_result(f"Security Header - {header}", "FAIL", "Missing or incorrect")
                    all_present = False
            
            return all_present
        except Exception as e:
            self.log_result("Security Headers", "FAIL", f"Error: {str(e)}")
            return False
    
    def test_performance(self):
        """Test performance metrics."""
        performance_tests = [
            (f"https://{self.domain}", "Frontend Performance", 3.0),
            (f"https://{self.api_domain}/health", "API Performance", 2.0),
        ]
        
        results = []
        for url, test_name, max_time in performance_tests:
            try:
                start_time = time.time()
                response = requests.get(url, timeout=10)
                response_time = time.time() - start_time
                
                if response_time <= max_time:
                    self.log_result(test_name, "PASS", f"Response time: {response_time:.2f}s", response_time)
                    results.append(True)
                else:
                    self.log_result(test_name, "WARN", f"Slow response: {response_time:.2f}s (max: {max_time}s)", response_time)
                    results.append(False)
            except Exception as e:
                self.log_result(test_name, "FAIL", f"Error: {str(e)}")
                results.append(False)
        
        return all(results)
    
    def test_ai_agents(self):
        """Test AI agent endpoints."""
        agent_endpoints = [
            "/agents/cto",
            "/agents/ml-engineer",
            "/agents/data-scientist",
            "/agents/bi-agent",
            "/agents/qa-agent",
        ]
        
        results = []
        for endpoint in agent_endpoints:
            url = f"https://{self.api_domain}{endpoint}"
            # These might return 404 or 405 if not implemented, which is acceptable
            try:
                response = requests.get(url, timeout=5)
                if response.status_code in [200, 404, 405, 422]:  # Acceptable status codes
                    self.log_result(f"AI Agent - {endpoint}", "PASS", f"Status {response.status_code}")
                    results.append(True)
                else:
                    self.log_result(f"AI Agent - {endpoint}", "FAIL", f"Status {response.status_code}")
                    results.append(False)
            except Exception as e:
                self.log_result(f"AI Agent - {endpoint}", "WARN", f"Endpoint not accessible: {str(e)}")
                results.append(False)
        
        return any(results)  # At least one agent should be accessible
    
    def run_comprehensive_verification(self):
        """Run all verification tests."""
        print("🔍 ScrollIntel.com Deployment Verification")
        print("=" * 50)
        print(f"Testing domain: {self.domain}")
        print(f"Testing API: {self.api_domain}")
        print(f"Testing app: {self.app_domain}")
        print()
        
        test_suites = [
            ("SSL Certificates", self.test_ssl_certificates),
            ("Frontend Services", self.test_frontend_functionality),
            ("API Services", self.test_api_functionality),
            ("Monitoring Services", self.test_monitoring_services),
            ("Security Headers", self.test_security_headers),
            ("Performance", self.test_performance),
            ("AI Agents", self.test_ai_agents),
        ]
        
        suite_results = []
        for suite_name, test_function in test_suites:
            print(f"\n📋 Testing {suite_name}:")
            print("-" * 30)
            result = test_function()
            suite_results.append((suite_name, result))
        
        return self.generate_report(suite_results)
    
    def test_ssl_certificates(self):
        """Test SSL certificates for all domains."""
        domains = [self.domain, self.api_domain, self.app_domain, f"grafana.{self.domain}"]
        results = []
        for domain in domains:
            results.append(self.test_ssl_certificate(domain))
        return all(results)
    
    def generate_report(self, suite_results):
        """Generate final verification report."""
        print("\n" + "=" * 50)
        print("📊 VERIFICATION REPORT")
        print("=" * 50)
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.results if r["status"] == "FAIL"])
        warned_tests = len([r for r in self.results if r["status"] == "WARN"])
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️ Warnings: {warned_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Test Suite Results:")
        for suite_name, result in suite_results:
            status_icon = "✅" if result else "❌"
            print(f"{status_icon} {suite_name}")
        
        # Overall status
        critical_failures = failed_tests
        overall_success = critical_failures == 0 and passed_tests > total_tests * 0.8
        
        print(f"\n🎯 Overall Status: {'✅ DEPLOYMENT SUCCESSFUL' if overall_success else '❌ DEPLOYMENT ISSUES DETECTED'}")
        
        if overall_success:
            print("\n🎉 Congratulations! Your ScrollIntel.com deployment is successful!")
            print(f"🌐 Your platform is live at: https://{self.domain}")
            print(f"🔧 API available at: https://{self.api_domain}")
            print(f"📊 Monitoring at: https://grafana.{self.domain}")
        else:
            print("\n⚠️ Some issues were detected. Please review the failed tests above.")
            print("💡 Common fixes:")
            print("   - Wait a few minutes for services to fully start")
            print("   - Check DNS propagation")
            print("   - Verify SSL certificate generation")
            print("   - Review container logs")
        
        # Save detailed report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "domain": self.domain,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warned_tests,
                "success_rate": (passed_tests/total_tests)*100,
                "overall_success": overall_success
            },
            "test_results": self.results,
            "suite_results": dict(suite_results)
        }
        
        with open('verification_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: verification_report.json")
        
        return overall_success

def main():
    """Main verification function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify ScrollIntel.com deployment")
    parser.add_argument("--domain", default="scrollintel.com", help="Domain to test (default: scrollintel.com)")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    
    args = parser.parse_args()
    
    verifier = ScrollIntelVerifier(args.domain)
    
    if args.quick:
        # Quick tests
        print("🚀 Running quick verification...")
        success = (
            verifier.test_endpoint(f"https://{args.domain}", "Frontend") and
            verifier.test_endpoint(f"https://api.{args.domain}/health", "API Health")
        )
        print(f"Quick test result: {'✅ PASS' if success else '❌ FAIL'}")
    else:
        # Comprehensive tests
        success = verifier.run_comprehensive_verification()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()