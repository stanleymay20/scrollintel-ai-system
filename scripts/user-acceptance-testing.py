#!/usr/bin/env python3
"""
ScrollIntel Agent Steering System - User Acceptance Testing Framework
Comprehensive UAT framework for production deployment validation
"""

import os
import sys
import json
import time
import logging
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class UATTestCase:
    """User Acceptance Test Case definition"""
    test_id: str
    name: str
    description: str
    category: str
    priority: str  # critical, high, medium, low
    steps: List[str]
    expected_result: str
    actual_result: Optional[str] = None
    status: str = "pending"  # pending, running, passed, failed, skipped
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    screenshots: List[str] = None
    
    def __post_init__(self):
        if self.screenshots is None:
            self.screenshots = []

@dataclass
class UATSuite:
    """User Acceptance Test Suite"""
    suite_id: str
    name: str
    description: str
    test_cases: List[UATTestCase]
    environment: str = "production"
    base_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:3000"

class UserAcceptanceTestingFramework:
    """Comprehensive User Acceptance Testing Framework"""
    
    def __init__(self, base_url: str = "http://localhost:8000", frontend_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.frontend_url = frontend_url
        self.test_results = {}
        self.session = requests.Session()
        self.driver = None
        self.test_data = {}
        
        # Setup test environment
        self._setup_test_environment()
        
        logger.info(f"UAT Framework initialized for {base_url}")
    
    def _setup_test_environment(self):
        """Setup test environment and data"""
        # Setup Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("✅ Selenium WebDriver initialized")
        except Exception as e:
            logger.warning(f"⚠️ Selenium WebDriver setup failed: {str(e)}")
            self.driver = None
        
        # Setup test data
        self.test_data = {
            "test_user": {
                "email": "test.user@scrollintel.com",
                "password": "TestPassword123!",
                "name": "Test User"
            },
            "test_admin": {
                "email": "admin@scrollintel.com",
                "password": "AdminPassword123!",
                "name": "Admin User"
            },
            "sample_data": {
                "csv_file": "test_data.csv",
                "json_data": {"test": "data"},
                "large_dataset": "large_test_data.csv"
            }
        }
    
    def run_comprehensive_uat(self) -> Dict[str, Any]:
        """Run comprehensive user acceptance testing"""
        logger.info("🧪 Starting comprehensive user acceptance testing...")
        
        start_time = datetime.now()
        
        try:
            # Initialize test suites
            test_suites = self._initialize_test_suites()
            
            # Execute test suites
            suite_results = {}
            for suite_name, suite in test_suites.items():
                logger.info(f"Running test suite: {suite_name}")
                suite_results[suite_name] = self._execute_test_suite(suite)
            
            # Generate comprehensive report
            uat_report = self._generate_uat_report(suite_results, start_time)
            
            # Save results
            self._save_uat_results(uat_report)
            
            logger.info("✅ Comprehensive UAT completed")
            return uat_report
            
        except Exception as e:
            logger.error(f"❌ UAT execution failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
        
        finally:
            if self.driver:
                self.driver.quit()
    
    def _initialize_test_suites(self) -> Dict[str, UATSuite]:
        """Initialize all test suites"""
        return {
            "authentication": self._create_authentication_suite(),
            "agent_interactions": self._create_agent_interaction_suite(),
            "data_processing": self._create_data_processing_suite(),
            "dashboard_functionality": self._create_dashboard_suite(),
            "api_endpoints": self._create_api_suite(),
            "security_compliance": self._create_security_suite(),
            "performance_load": self._create_performance_suite(),
            "integration_points": self._create_integration_suite(),
            "mobile_responsive": self._create_mobile_suite(),
            "accessibility": self._create_accessibility_suite()
        }
    
    def _create_authentication_suite(self) -> UATSuite:
        """Create authentication test suite"""
        test_cases = [
            UATTestCase(
                test_id="AUTH_001",
                name="User Registration",
                description="Test user registration functionality",
                category="authentication",
                priority="critical",
                steps=[
                    "Navigate to registration page",
                    "Fill in valid user details",
                    "Submit registration form",
                    "Verify email confirmation"
                ],
                expected_result="User successfully registered and email confirmation sent"
            ),
            UATTestCase(
                test_id="AUTH_002",
                name="User Login",
                description="Test user login functionality",
                category="authentication",
                priority="critical",
                steps=[
                    "Navigate to login page",
                    "Enter valid credentials",
                    "Submit login form",
                    "Verify successful login"
                ],
                expected_result="User successfully logged in and redirected to dashboard"
            ),
            UATTestCase(
                test_id="AUTH_003",
                name="Password Reset",
                description="Test password reset functionality",
                category="authentication",
                priority="high",
                steps=[
                    "Navigate to password reset page",
                    "Enter email address",
                    "Submit reset request",
                    "Check email for reset link"
                ],
                expected_result="Password reset email sent successfully"
            ),
            UATTestCase(
                test_id="AUTH_004",
                name="Multi-Factor Authentication",
                description="Test MFA functionality",
                category="authentication",
                priority="high",
                steps=[
                    "Login with MFA-enabled account",
                    "Enter primary credentials",
                    "Enter MFA code",
                    "Verify successful authentication"
                ],
                expected_result="MFA authentication successful"
            ),
            UATTestCase(
                test_id="AUTH_005",
                name="Session Management",
                description="Test session timeout and management",
                category="authentication",
                priority="medium",
                steps=[
                    "Login to application",
                    "Wait for session timeout",
                    "Attempt to access protected resource",
                    "Verify redirect to login"
                ],
                expected_result="Session expires and user redirected to login"
            )
        ]
        
        return UATSuite(
            suite_id="AUTH_SUITE",
            name="Authentication Test Suite",
            description="Comprehensive authentication functionality testing",
            test_cases=test_cases
        )
    
    def _create_agent_interaction_suite(self) -> UATSuite:
        """Create agent interaction test suite"""
        test_cases = [
            UATTestCase(
                test_id="AGENT_001",
                name="Agent Chat Interface",
                description="Test basic agent chat functionality",
                category="agent_interaction",
                priority="critical",
                steps=[
                    "Navigate to chat interface",
                    "Send message to agent",
                    "Wait for agent response",
                    "Verify response quality"
                ],
                expected_result="Agent responds appropriately to user message"
            ),
            UATTestCase(
                test_id="AGENT_002",
                name="Multi-Agent Coordination",
                description="Test coordination between multiple agents",
                category="agent_interaction",
                priority="critical",
                steps=[
                    "Initiate complex task requiring multiple agents",
                    "Monitor agent coordination",
                    "Verify task completion",
                    "Check result quality"
                ],
                expected_result="Multiple agents coordinate successfully to complete task"
            ),
            UATTestCase(
                test_id="AGENT_003",
                name="Agent Performance Monitoring",
                description="Test agent performance monitoring",
                category="agent_interaction",
                priority="high",
                steps=[
                    "Access agent monitoring dashboard",
                    "Verify real-time metrics",
                    "Check performance indicators",
                    "Validate alerting system"
                ],
                expected_result="Agent performance metrics displayed accurately"
            ),
            UATTestCase(
                test_id="AGENT_004",
                name="Agent Failover",
                description="Test agent failover functionality",
                category="agent_interaction",
                priority="high",
                steps=[
                    "Simulate agent failure",
                    "Verify automatic failover",
                    "Check task continuity",
                    "Validate recovery process"
                ],
                expected_result="System automatically handles agent failure with minimal disruption"
            )
        ]
        
        return UATSuite(
            suite_id="AGENT_SUITE",
            name="Agent Interaction Test Suite",
            description="Comprehensive agent functionality testing",
            test_cases=test_cases
        )
    
    def _create_data_processing_suite(self) -> UATSuite:
        """Create data processing test suite"""
        test_cases = [
            UATTestCase(
                test_id="DATA_001",
                name="File Upload",
                description="Test file upload functionality",
                category="data_processing",
                priority="critical",
                steps=[
                    "Navigate to file upload interface",
                    "Select test file",
                    "Upload file",
                    "Verify successful upload"
                ],
                expected_result="File uploaded successfully and processed"
            ),
            UATTestCase(
                test_id="DATA_002",
                name="Data Validation",
                description="Test data validation and quality checks",
                category="data_processing",
                priority="critical",
                steps=[
                    "Upload data with known issues",
                    "Trigger validation process",
                    "Review validation results",
                    "Verify issue detection"
                ],
                expected_result="Data validation correctly identifies and reports issues"
            ),
            UATTestCase(
                test_id="DATA_003",
                name="Real-time Processing",
                description="Test real-time data processing",
                category="data_processing",
                priority="high",
                steps=[
                    "Stream data to system",
                    "Monitor processing pipeline",
                    "Verify real-time updates",
                    "Check processing latency"
                ],
                expected_result="Data processed in real-time with acceptable latency"
            ),
            UATTestCase(
                test_id="DATA_004",
                name="Large Dataset Handling",
                description="Test handling of large datasets",
                category="data_processing",
                priority="high",
                steps=[
                    "Upload large dataset",
                    "Monitor processing progress",
                    "Verify completion",
                    "Check system performance"
                ],
                expected_result="Large dataset processed successfully without system degradation"
            )
        ]
        
        return UATSuite(
            suite_id="DATA_SUITE",
            name="Data Processing Test Suite",
            description="Comprehensive data processing functionality testing",
            test_cases=test_cases
        )
    
    def _execute_test_suite(self, suite: UATSuite) -> Dict[str, Any]:
        """Execute a test suite"""
        logger.info(f"Executing test suite: {suite.name}")
        
        suite_start_time = datetime.now()
        results = {
            "suite_info": {
                "suite_id": suite.suite_id,
                "name": suite.name,
                "description": suite.description
            },
            "execution_summary": {
                "total_tests": len(suite.test_cases),
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "start_time": suite_start_time.isoformat(),
                "end_time": None,
                "duration": None
            },
            "test_results": []
        }
        
        # Execute each test case
        for test_case in suite.test_cases:
            try:
                test_result = self._execute_test_case(test_case)
                results["test_results"].append(test_result)
                
                # Update summary
                if test_result["status"] == "passed":
                    results["execution_summary"]["passed"] += 1
                elif test_result["status"] == "failed":
                    results["execution_summary"]["failed"] += 1
                else:
                    results["execution_summary"]["skipped"] += 1
                    
            except Exception as e:
                logger.error(f"Test case {test_case.test_id} execution failed: {str(e)}")
                results["test_results"].append({
                    "test_id": test_case.test_id,
                    "status": "failed",
                    "error": str(e)
                })
                results["execution_summary"]["failed"] += 1
        
        # Finalize suite results
        suite_end_time = datetime.now()
        results["execution_summary"]["end_time"] = suite_end_time.isoformat()
        results["execution_summary"]["duration"] = (suite_end_time - suite_start_time).total_seconds()
        
        return results
    
    def _execute_test_case(self, test_case: UATTestCase) -> Dict[str, Any]:
        """Execute individual test case"""
        logger.info(f"Executing test case: {test_case.test_id} - {test_case.name}")
        
        start_time = datetime.now()
        
        try:
            # Route to appropriate test execution method
            if test_case.category == "authentication":
                result = self._execute_authentication_test(test_case)
            elif test_case.category == "agent_interaction":
                result = self._execute_agent_interaction_test(test_case)
            elif test_case.category == "data_processing":
                result = self._execute_data_processing_test(test_case)
            elif test_case.category == "api_endpoints":
                result = self._execute_api_test(test_case)
            elif test_case.category == "security_compliance":
                result = self._execute_security_test(test_case)
            elif test_case.category == "performance_load":
                result = self._execute_performance_test(test_case)
            else:
                result = self._execute_generic_test(test_case)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                "test_id": test_case.test_id,
                "name": test_case.name,
                "category": test_case.category,
                "priority": test_case.priority,
                "status": result["status"],
                "execution_time": execution_time,
                "expected_result": test_case.expected_result,
                "actual_result": result.get("actual_result", ""),
                "error_message": result.get("error_message"),
                "screenshots": result.get("screenshots", []),
                "additional_data": result.get("additional_data", {})
            }
            
        except Exception as e:
            logger.error(f"Test case {test_case.test_id} failed: {str(e)}")
            return {
                "test_id": test_case.test_id,
                "name": test_case.name,
                "status": "failed",
                "error_message": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _execute_authentication_test(self, test_case: UATTestCase) -> Dict[str, Any]:
        """Execute authentication test case"""
        if test_case.test_id == "AUTH_001":
            return self._test_user_registration()
        elif test_case.test_id == "AUTH_002":
            return self._test_user_login()
        elif test_case.test_id == "AUTH_003":
            return self._test_password_reset()
        elif test_case.test_id == "AUTH_004":
            return self._test_mfa()
        elif test_case.test_id == "AUTH_005":
            return self._test_session_management()
        else:
            return {"status": "skipped", "actual_result": "Test not implemented"}
    
    def _test_user_registration(self) -> Dict[str, Any]:
        """Test user registration"""
        try:
            # API test for user registration
            registration_data = {
                "email": f"test_{int(time.time())}@example.com",
                "password": "TestPassword123!",
                "name": "Test User"
            }
            
            response = self.session.post(
                f"{self.base_url}/api/auth/register",
                json=registration_data
            )
            
            if response.status_code == 201:
                return {
                    "status": "passed",
                    "actual_result": "User registration successful",
                    "additional_data": {"response_code": response.status_code}
                }
            else:
                return {
                    "status": "failed",
                    "actual_result": f"Registration failed with status {response.status_code}",
                    "error_message": response.text
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error_message": str(e)
            }
    
    def _test_user_login(self) -> Dict[str, Any]:
        """Test user login"""
        try:
            login_data = {
                "email": self.test_data["test_user"]["email"],
                "password": self.test_data["test_user"]["password"]
            }
            
            response = self.session.post(
                f"{self.base_url}/api/auth/login",
                json=login_data
            )
            
            if response.status_code == 200:
                token = response.json().get("access_token")
                if token:
                    return {
                        "status": "passed",
                        "actual_result": "User login successful with token",
                        "additional_data": {"has_token": True}
                    }
                else:
                    return {
                        "status": "failed",
                        "actual_result": "Login successful but no token received"
                    }
            else:
                return {
                    "status": "failed",
                    "actual_result": f"Login failed with status {response.status_code}",
                    "error_message": response.text
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error_message": str(e)
            }
    
    def _execute_agent_interaction_test(self, test_case: UATTestCase) -> Dict[str, Any]:
        """Execute agent interaction test case"""
        if test_case.test_id == "AGENT_001":
            return self._test_agent_chat()
        elif test_case.test_id == "AGENT_002":
            return self._test_multi_agent_coordination()
        elif test_case.test_id == "AGENT_003":
            return self._test_agent_monitoring()
        elif test_case.test_id == "AGENT_004":
            return self._test_agent_failover()
        else:
            return {"status": "skipped", "actual_result": "Test not implemented"}
    
    def _test_agent_chat(self) -> Dict[str, Any]:
        """Test agent chat functionality"""
        try:
            # Test agent chat API
            chat_data = {
                "message": "Hello, can you help me analyze some data?",
                "agent_type": "data_scientist"
            }
            
            response = self.session.post(
                f"{self.base_url}/api/agents/chat",
                json=chat_data
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("response"):
                    return {
                        "status": "passed",
                        "actual_result": "Agent responded successfully",
                        "additional_data": {
                            "response_length": len(response_data["response"]),
                            "response_time": response.elapsed.total_seconds()
                        }
                    }
                else:
                    return {
                        "status": "failed",
                        "actual_result": "Agent response empty"
                    }
            else:
                return {
                    "status": "failed",
                    "actual_result": f"Agent chat failed with status {response.status_code}",
                    "error_message": response.text
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error_message": str(e)
            }
    
    def _execute_data_processing_test(self, test_case: UATTestCase) -> Dict[str, Any]:
        """Execute data processing test case"""
        if test_case.test_id == "DATA_001":
            return self._test_file_upload()
        elif test_case.test_id == "DATA_002":
            return self._test_data_validation()
        elif test_case.test_id == "DATA_003":
            return self._test_realtime_processing()
        elif test_case.test_id == "DATA_004":
            return self._test_large_dataset()
        else:
            return {"status": "skipped", "actual_result": "Test not implemented"}
    
    def _test_file_upload(self) -> Dict[str, Any]:
        """Test file upload functionality"""
        try:
            # Create test file
            test_file_content = "name,age,city\nJohn,30,New York\nJane,25,Los Angeles"
            test_file_path = "/tmp/test_upload.csv"
            
            with open(test_file_path, 'w') as f:
                f.write(test_file_content)
            
            # Upload file
            with open(test_file_path, 'rb') as f:
                files = {'file': ('test_upload.csv', f, 'text/csv')}
                response = self.session.post(
                    f"{self.base_url}/api/files/upload",
                    files=files
                )
            
            if response.status_code == 200:
                return {
                    "status": "passed",
                    "actual_result": "File uploaded successfully",
                    "additional_data": {"file_id": response.json().get("file_id")}
                }
            else:
                return {
                    "status": "failed",
                    "actual_result": f"File upload failed with status {response.status_code}",
                    "error_message": response.text
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error_message": str(e)
            }
    
    def _generate_uat_report(self, suite_results: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive UAT report"""
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Calculate overall statistics
        total_tests = sum(suite["execution_summary"]["total_tests"] for suite in suite_results.values())
        total_passed = sum(suite["execution_summary"]["passed"] for suite in suite_results.values())
        total_failed = sum(suite["execution_summary"]["failed"] for suite in suite_results.values())
        total_skipped = sum(suite["execution_summary"]["skipped"] for suite in suite_results.values())
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Identify critical failures
        critical_failures = []
        for suite_name, suite_result in suite_results.items():
            for test_result in suite_result["test_results"]:
                if (test_result.get("priority") == "critical" and 
                    test_result.get("status") == "failed"):
                    critical_failures.append({
                        "suite": suite_name,
                        "test_id": test_result["test_id"],
                        "test_name": test_result["name"],
                        "error": test_result.get("error_message", "Unknown error")
                    })
        
        # Generate recommendations
        recommendations = self._generate_uat_recommendations(suite_results, critical_failures)
        
        report = {
            "report_metadata": {
                "report_id": f"uat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.now().isoformat(),
                "environment": "production",
                "base_url": self.base_url,
                "frontend_url": self.frontend_url
            },
            "execution_summary": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
                "success_rate_percentage": round(success_rate, 2)
            },
            "suite_results": suite_results,
            "critical_failures": critical_failures,
            "recommendations": recommendations,
            "deployment_readiness": {
                "ready_for_production": len(critical_failures) == 0 and success_rate >= 95,
                "confidence_level": self._calculate_confidence_level(success_rate, critical_failures),
                "risk_assessment": self._assess_deployment_risk(suite_results, critical_failures)
            }
        }
        
        return report
    
    def _generate_uat_recommendations(self, suite_results: Dict[str, Any], critical_failures: List[Dict]) -> List[str]:
        """Generate UAT recommendations"""
        recommendations = []
        
        if critical_failures:
            recommendations.append("❌ CRITICAL: Address all critical test failures before production deployment")
            for failure in critical_failures:
                recommendations.append(f"  - Fix {failure['test_id']}: {failure['test_name']}")
        
        # Check success rates by suite
        for suite_name, suite_result in suite_results.items():
            summary = suite_result["execution_summary"]
            if summary["total_tests"] > 0:
                success_rate = (summary["passed"] / summary["total_tests"]) * 100
                if success_rate < 90:
                    recommendations.append(f"⚠️ {suite_name} suite has low success rate ({success_rate:.1f}%)")
        
        # Performance recommendations
        slow_tests = []
        for suite_name, suite_result in suite_results.items():
            for test_result in suite_result["test_results"]:
                if test_result.get("execution_time", 0) > 30:  # Tests taking more than 30 seconds
                    slow_tests.append(f"{suite_name}.{test_result['test_id']}")
        
        if slow_tests:
            recommendations.append(f"🐌 Consider optimizing slow tests: {', '.join(slow_tests)}")
        
        if not recommendations:
            recommendations.append("✅ All tests passed successfully - system ready for production deployment")
        
        return recommendations
    
    def _calculate_confidence_level(self, success_rate: float, critical_failures: List[Dict]) -> str:
        """Calculate deployment confidence level"""
        if critical_failures:
            return "LOW"
        elif success_rate >= 98:
            return "HIGH"
        elif success_rate >= 95:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_deployment_risk(self, suite_results: Dict[str, Any], critical_failures: List[Dict]) -> str:
        """Assess deployment risk level"""
        if critical_failures:
            return "HIGH"
        
        total_failed = sum(suite["execution_summary"]["failed"] for suite in suite_results.values())
        total_tests = sum(suite["execution_summary"]["total_tests"] for suite in suite_results.values())
        
        failure_rate = (total_failed / total_tests * 100) if total_tests > 0 else 0
        
        if failure_rate == 0:
            return "LOW"
        elif failure_rate <= 5:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _save_uat_results(self, report: Dict[str, Any]):
        """Save UAT results to file"""
        os.makedirs("reports/uat", exist_ok=True)
        
        report_file = f"reports/uat/uat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"UAT report saved to {report_file}")
        
        # Also save a summary report
        summary_file = f"reports/uat/uat_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write("ScrollIntel Agent Steering System - UAT Summary Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Report ID: {report['report_metadata']['report_id']}\n")
            f.write(f"Generated: {report['report_metadata']['generated_at']}\n")
            f.write(f"Environment: {report['report_metadata']['environment']}\n\n")
            
            f.write("Execution Summary:\n")
            f.write("-" * 20 + "\n")
            summary = report['execution_summary']
            f.write(f"Total Tests: {summary['total_tests']}\n")
            f.write(f"Passed: {summary['passed']}\n")
            f.write(f"Failed: {summary['failed']}\n")
            f.write(f"Skipped: {summary['skipped']}\n")
            f.write(f"Success Rate: {summary['success_rate_percentage']}%\n")
            f.write(f"Duration: {summary['total_duration_seconds']:.2f} seconds\n\n")
            
            f.write("Deployment Readiness:\n")
            f.write("-" * 20 + "\n")
            readiness = report['deployment_readiness']
            f.write(f"Ready for Production: {readiness['ready_for_production']}\n")
            f.write(f"Confidence Level: {readiness['confidence_level']}\n")
            f.write(f"Risk Assessment: {readiness['risk_assessment']}\n\n")
            
            if report['critical_failures']:
                f.write("Critical Failures:\n")
                f.write("-" * 20 + "\n")
                for failure in report['critical_failures']:
                    f.write(f"- {failure['test_id']}: {failure['test_name']}\n")
                    f.write(f"  Error: {failure['error']}\n")
                f.write("\n")
            
            f.write("Recommendations:\n")
            f.write("-" * 20 + "\n")
            for recommendation in report['recommendations']:
                f.write(f"- {recommendation}\n")

def main():
    """Main UAT execution function"""
    # Get configuration from environment
    base_url = os.getenv("UAT_BASE_URL", "http://localhost:8000")
    frontend_url = os.getenv("UAT_FRONTEND_URL", "http://localhost:3000")
    
    # Initialize UAT framework
    uat_framework = UserAcceptanceTestingFramework(base_url, frontend_url)
    
    # Run comprehensive UAT
    results = uat_framework.run_comprehensive_uat()
    
    # Check if deployment is ready
    if results.get("deployment_readiness", {}).get("ready_for_production", False):
        print("✅ UAT PASSED - System ready for production deployment")
        sys.exit(0)
    else:
        print("❌ UAT FAILED - System not ready for production deployment")
        print("Critical issues must be resolved before deployment")
        sys.exit(1)

if __name__ == "__main__":
    main()