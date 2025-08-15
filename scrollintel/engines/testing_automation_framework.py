"""
Testing Automation Framework for Autonomous Innovation Lab

This module provides automated testing and validation capabilities
for the ScrollIntel autonomous innovation lab system.
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import time
import random

from ..models.prototype_models import (
    Prototype, PrototypeType, PrototypeStatus, TestResult
)
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TestType(Enum):
    """Types of automated tests"""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    FUNCTIONAL_TEST = "functional_test"
    PERFORMANCE_TEST = "performance_test"
    SECURITY_TEST = "security_test"
    USABILITY_TEST = "usability_test"
    LOAD_TEST = "load_test"
    SMOKE_TEST = "smoke_test"
    REGRESSION_TEST = "regression_test"
    API_TEST = "api_test"
    UI_TEST = "ui_test"


class TestStatus(Enum):
    """Status of test execution"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class TestPriority(Enum):
    """Priority levels for tests"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestCase:
    """Individual test case definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    test_type: TestType = TestType.UNIT_TEST
    priority: TestPriority = TestPriority.MEDIUM
    
    # Test configuration
    test_code: str = ""
    test_data: Dict[str, Any] = field(default_factory=dict)
    expected_result: Any = None
    timeout_seconds: int = 30
    retry_count: int = 0
    
    # Test dependencies
    depends_on: List[str] = field(default_factory=list)  # Test IDs this test depends on
    tags: List[str] = field(default_factory=list)
    
    # Test environment
    environment_requirements: Dict[str, str] = field(default_factory=dict)
    setup_commands: List[str] = field(default_factory=list)
    teardown_commands: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    author: str = "automation"


@dataclass
class TestSuite:
    """Collection of related test cases"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    prototype_id: str = ""
    
    # Test cases
    test_cases: List[TestCase] = field(default_factory=list)
    
    # Suite configuration
    parallel_execution: bool = True
    max_parallel_tests: int = 5
    stop_on_failure: bool = False
    timeout_minutes: int = 60
    
    # Suite metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_executed: Optional[datetime] = None
    execution_count: int = 0


@dataclass
class TestExecution:
    """Test execution instance"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    test_case_id: str = ""
    test_suite_id: str = ""
    prototype_id: str = ""
    
    # Execution details
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Results
    actual_result: Any = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    output_logs: List[str] = field(default_factory=list)
    
    # Metrics
    assertions_passed: int = 0
    assertions_failed: int = 0
    coverage_percentage: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Environment
    execution_environment: Dict[str, str] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestReport:
    """Comprehensive test report"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prototype_id: str = ""
    test_suite_id: str = ""
    
    # Summary statistics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    
    # Execution metrics
    total_duration_seconds: float = 0.0
    average_test_duration: float = 0.0
    success_rate: float = 0.0
    
    # Coverage and quality
    overall_coverage: float = 0.0
    code_quality_score: float = 0.0
    performance_score: float = 0.0
    security_score: float = 0.0
    
    # Test executions
    test_executions: List[TestExecution] = field(default_factory=list)
    
    # Analysis
    failure_analysis: Dict[str, Any] = field(default_factory=dict)
    performance_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    report_version: str = "1.0"


class TestGenerator:
    """Generates automated tests for prototypes"""
    
    def __init__(self):
        self.test_templates = self._load_test_templates()
        self.test_patterns = self._load_test_patterns()
    
    def _load_test_templates(self) -> Dict[str, Dict[str, str]]:
        """Load test templates for different prototype types"""
        return {
            PrototypeType.API_SERVICE.value: {
                "unit_test": '''
import pytest
import requests
from unittest.mock import Mock, patch

class Test{class_name}:
    def test_{function_name}_success(self):
        """Test successful {function_name} operation"""
        # Arrange
        test_data = {test_data}
        expected_result = {expected_result}
        
        # Act
        result = {function_call}
        
        # Assert
        assert result == expected_result
        assert result is not None
    
    def test_{function_name}_error_handling(self):
        """Test error handling for {function_name}"""
        # Arrange
        invalid_data = {invalid_data}
        
        # Act & Assert
        with pytest.raises(ValueError):
            {function_call_with_invalid_data}
    
    def test_{function_name}_edge_cases(self):
        """Test edge cases for {function_name}"""
        # Test empty input
        result = {function_call_empty}
        assert result is not None
        
        # Test boundary values
        result = {function_call_boundary}
        assert result is not None
''',
                "integration_test": '''
import pytest
import requests
import json

class TestAPI{class_name}Integration:
    BASE_URL = "http://localhost:8000"
    
    def test_api_endpoint_availability(self):
        """Test API endpoint availability"""
        response = requests.get(f"{{self.BASE_URL}}/health")
        assert response.status_code == 200
    
    def test_create_{resource}_endpoint(self):
        """Test create {resource} endpoint"""
        test_data = {test_data}
        response = requests.post(
            f"{{self.BASE_URL}}/{endpoint}",
            json=test_data
        )
        assert response.status_code in [200, 201]
        assert response.json() is not None
    
    def test_get_{resource}_endpoint(self):
        """Test get {resource} endpoint"""
        response = requests.get(f"{{self.BASE_URL}}/{endpoint}")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_api_error_handling(self):
        """Test API error handling"""
        response = requests.post(
            f"{{self.BASE_URL}}/{endpoint}",
            json={{}}  # Invalid data
        )
        assert response.status_code in [400, 422]
''',
                "performance_test": '''
import pytest
import time
import requests
import concurrent.futures
from statistics import mean

class TestPerformance{class_name}:
    BASE_URL = "http://localhost:8000"
    
    def test_response_time(self):
        """Test API response time"""
        start_time = time.time()
        response = requests.get(f"{{self.BASE_URL}}/{endpoint}")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 2.0  # Should respond within 2 seconds
        assert response.status_code == 200
    
    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        def make_request():
            return requests.get(f"{{self.BASE_URL}}/{endpoint}")
        
        # Test with 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in results)
        
        # Calculate average response time
        response_times = []
        for _ in range(5):
            start = time.time()
            requests.get(f"{{self.BASE_URL}}/{endpoint}")
            response_times.append(time.time() - start)
        
        avg_response_time = mean(response_times)
        assert avg_response_time < 1.0  # Average should be under 1 second
    
    def test_load_handling(self):
        """Test load handling capacity"""
        def make_requests_batch():
            return [requests.get(f"{{self.BASE_URL}}/{endpoint}") for _ in range(5)]
        
        # Test with multiple batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_requests_batch) for _ in range(3)]
            results = [future.result() for future in futures]
        
        # Flatten results
        all_responses = [resp for batch in results for resp in batch]
        success_rate = sum(1 for r in all_responses if r.status_code == 200) / len(all_responses)
        
        assert success_rate >= 0.95  # 95% success rate under load
'''
            },
            PrototypeType.WEB_APP.value: {
                "ui_test": '''
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

class TestUI{class_name}:
    def setup_method(self):
        """Setup test environment"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get("http://localhost:3000")
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.driver.quit()
    
    def test_page_loads(self):
        """Test that the main page loads correctly"""
        assert "{title}" in self.driver.title
        assert self.driver.find_element(By.TAG_NAME, "body")
    
    def test_navigation_elements(self):
        """Test navigation elements are present"""
        # Check for common navigation elements
        nav_elements = self.driver.find_elements(By.TAG_NAME, "nav")
        assert len(nav_elements) > 0
        
        # Check for header
        header = self.driver.find_element(By.TAG_NAME, "header")
        assert header.is_displayed()
    
    def test_interactive_elements(self):
        """Test interactive elements work"""
        # Find and test buttons
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        for button in buttons[:3]:  # Test first 3 buttons
            if button.is_enabled():
                button.click()
                # Wait for any potential page changes
                WebDriverWait(self.driver, 2).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
    
    def test_form_functionality(self):
        """Test form functionality if present"""
        forms = self.driver.find_elements(By.TAG_NAME, "form")
        if forms:
            form = forms[0]
            inputs = form.find_elements(By.TAG_NAME, "input")
            
            # Fill out form fields
            for input_field in inputs:
                if input_field.get_attribute("type") == "text":
                    input_field.send_keys("test data")
                elif input_field.get_attribute("type") == "email":
                    input_field.send_keys("test@example.com")
    
    def test_responsive_design(self):
        """Test responsive design"""
        # Test different screen sizes
        screen_sizes = [(1920, 1080), (768, 1024), (375, 667)]
        
        for width, height in screen_sizes:
            self.driver.set_window_size(width, height)
            body = self.driver.find_element(By.TAG_NAME, "body")
            assert body.is_displayed()
            
            # Check that content is still accessible
            assert len(self.driver.find_elements(By.TAG_NAME, "div")) > 0
''',
                "functional_test": '''
import pytest
import requests
import time

class TestFunctional{class_name}:
    BASE_URL = "http://localhost:3000"
    
    def test_user_workflow(self):
        """Test complete user workflow"""
        # Step 1: Access the application
        response = requests.get(self.BASE_URL)
        assert response.status_code == 200
        
        # Step 2: Test main functionality
        # This would be customized based on the specific application
        assert "Welcome" in response.text or "Dashboard" in response.text
    
    def test_data_persistence(self):
        """Test data persistence functionality"""
        # This test would be customized based on the application's data model
        test_data = {test_data}
        
        # Create data
        create_response = requests.post(f"{{self.BASE_URL}}/api/data", json=test_data)
        assert create_response.status_code in [200, 201]
        
        # Retrieve data
        get_response = requests.get(f"{{self.BASE_URL}}/api/data")
        assert get_response.status_code == 200
        
        # Verify data exists
        data = get_response.json()
        assert isinstance(data, list)
    
    def test_error_scenarios(self):
        """Test error handling scenarios"""
        # Test 404 error
        response = requests.get(f"{{self.BASE_URL}}/nonexistent-page")
        assert response.status_code == 404
        
        # Test invalid API calls
        response = requests.post(f"{{self.BASE_URL}}/api/invalid", json={{}})
        assert response.status_code in [400, 404, 405]
'''
            }
        }
    
    def _load_test_patterns(self) -> Dict[str, List[str]]:
        """Load common test patterns"""
        return {
            "api_endpoints": [
                "GET /health",
                "GET /{resource}",
                "POST /{resource}",
                "PUT /{resource}/{id}",
                "DELETE /{resource}/{id}"
            ],
            "ui_elements": [
                "page_load",
                "navigation",
                "forms",
                "buttons",
                "links",
                "responsive_design"
            ],
            "performance_metrics": [
                "response_time",
                "throughput",
                "concurrent_users",
                "memory_usage",
                "cpu_usage"
            ],
            "security_checks": [
                "authentication",
                "authorization",
                "input_validation",
                "sql_injection",
                "xss_protection"
            ]
        }
    
    async def generate_test_suite(self, prototype: Prototype) -> TestSuite:
        """Generate comprehensive test suite for a prototype"""
        try:
            test_suite = TestSuite(
                name=f"{prototype.name} Test Suite",
                description=f"Automated test suite for {prototype.name}",
                prototype_id=prototype.id
            )
            
            # Generate different types of tests based on prototype type
            test_cases = []
            
            # Generate unit tests
            unit_tests = await self._generate_unit_tests(prototype)
            test_cases.extend(unit_tests)
            
            # Generate integration tests
            integration_tests = await self._generate_integration_tests(prototype)
            test_cases.extend(integration_tests)
            
            # Generate functional tests
            functional_tests = await self._generate_functional_tests(prototype)
            test_cases.extend(functional_tests)
            
            # Generate performance tests
            performance_tests = await self._generate_performance_tests(prototype)
            test_cases.extend(performance_tests)
            
            # Generate security tests
            security_tests = await self._generate_security_tests(prototype)
            test_cases.extend(security_tests)
            
            # Generate UI tests (if applicable)
            if prototype.prototype_type in [PrototypeType.WEB_APP, PrototypeType.MOBILE_APP]:
                ui_tests = await self._generate_ui_tests(prototype)
                test_cases.extend(ui_tests)
            
            test_suite.test_cases = test_cases
            
            logger.info(f"Generated test suite with {len(test_cases)} test cases for prototype {prototype.id}")
            return test_suite
            
        except Exception as e:
            logger.error(f"Error generating test suite: {str(e)}")
            raise
    
    async def _generate_unit_tests(self, prototype: Prototype) -> List[TestCase]:
        """Generate unit tests for the prototype"""
        test_cases = []
        
        # Extract functions/methods from generated code
        functions = self._extract_functions_from_code(prototype.generated_code)
        
        for function_name in functions[:5]:  # Limit to first 5 functions
            test_case = TestCase(
                name=f"test_{function_name}_unit",
                description=f"Unit test for {function_name} function",
                test_type=TestType.UNIT_TEST,
                priority=TestPriority.HIGH,
                test_code=self._generate_unit_test_code(prototype, function_name),
                timeout_seconds=10,
                tags=["unit", "automated"]
            )
            test_cases.append(test_case)
        
        return test_cases
    
    async def _generate_integration_tests(self, prototype: Prototype) -> List[TestCase]:
        """Generate integration tests for the prototype"""
        test_cases = []
        
        if prototype.prototype_type == PrototypeType.API_SERVICE:
            # Generate API integration tests
            endpoints = self._extract_api_endpoints(prototype.generated_code)
            
            for endpoint in endpoints[:3]:  # Limit to first 3 endpoints
                test_case = TestCase(
                    name=f"test_{endpoint.replace('/', '_')}_integration",
                    description=f"Integration test for {endpoint} endpoint",
                    test_type=TestType.INTEGRATION_TEST,
                    priority=TestPriority.HIGH,
                    test_code=self._generate_integration_test_code(prototype, endpoint),
                    timeout_seconds=30,
                    tags=["integration", "api", "automated"]
                )
                test_cases.append(test_case)
        
        return test_cases
    
    async def _generate_functional_tests(self, prototype: Prototype) -> List[TestCase]:
        """Generate functional tests for the prototype"""
        test_cases = []
        
        # Generate basic functional tests
        test_case = TestCase(
            name="test_basic_functionality",
            description="Test basic functionality of the prototype",
            test_type=TestType.FUNCTIONAL_TEST,
            priority=TestPriority.CRITICAL,
            test_code=self._generate_functional_test_code(prototype),
            timeout_seconds=60,
            tags=["functional", "critical", "automated"]
        )
        test_cases.append(test_case)
        
        return test_cases
    
    async def _generate_performance_tests(self, prototype: Prototype) -> List[TestCase]:
        """Generate performance tests for the prototype"""
        test_cases = []
        
        test_case = TestCase(
            name="test_performance_benchmarks",
            description="Test performance benchmarks of the prototype",
            test_type=TestType.PERFORMANCE_TEST,
            priority=TestPriority.MEDIUM,
            test_code=self._generate_performance_test_code(prototype),
            timeout_seconds=120,
            tags=["performance", "benchmarks", "automated"]
        )
        test_cases.append(test_case)
        
        return test_cases
    
    async def _generate_security_tests(self, prototype: Prototype) -> List[TestCase]:
        """Generate security tests for the prototype"""
        test_cases = []
        
        test_case = TestCase(
            name="test_security_vulnerabilities",
            description="Test for common security vulnerabilities",
            test_type=TestType.SECURITY_TEST,
            priority=TestPriority.HIGH,
            test_code=self._generate_security_test_code(prototype),
            timeout_seconds=90,
            tags=["security", "vulnerabilities", "automated"]
        )
        test_cases.append(test_case)
        
        return test_cases
    
    async def _generate_ui_tests(self, prototype: Prototype) -> List[TestCase]:
        """Generate UI tests for web/mobile prototypes"""
        test_cases = []
        
        test_case = TestCase(
            name="test_ui_functionality",
            description="Test UI functionality and user interactions",
            test_type=TestType.UI_TEST,
            priority=TestPriority.MEDIUM,
            test_code=self._generate_ui_test_code(prototype),
            timeout_seconds=60,
            tags=["ui", "frontend", "automated"]
        )
        test_cases.append(test_case)
        
        return test_cases
    
    def _extract_functions_from_code(self, generated_code: Dict[str, str]) -> List[str]:
        """Extract function names from generated code"""
        functions = []
        
        for file_content in generated_code.values():
            # Simple regex-like extraction for function names
            lines = file_content.split('\n')
            for line in lines:
                if 'def ' in line and '(' in line:
                    # Extract function name
                    start = line.find('def ') + 4
                    end = line.find('(')
                    if start < end:
                        func_name = line[start:end].strip()
                        if func_name and func_name not in functions:
                            functions.append(func_name)
                elif 'async def ' in line and '(' in line:
                    # Extract async function name
                    start = line.find('async def ') + 10
                    end = line.find('(')
                    if start < end:
                        func_name = line[start:end].strip()
                        if func_name and func_name not in functions:
                            functions.append(func_name)
        
        return functions
    
    def _extract_api_endpoints(self, generated_code: Dict[str, str]) -> List[str]:
        """Extract API endpoints from generated code"""
        endpoints = []
        
        for file_content in generated_code.values():
            lines = file_content.split('\n')
            for line in lines:
                # Look for FastAPI route decorators
                if '@app.' in line and '("' in line:
                    start = line.find('("') + 2
                    end = line.find('"', start)
                    if start < end:
                        endpoint = line[start:end]
                        if endpoint and endpoint not in endpoints:
                            endpoints.append(endpoint)
        
        return endpoints
    
    def _generate_unit_test_code(self, prototype: Prototype, function_name: str) -> str:
        """Generate unit test code for a specific function"""
        template = self.test_templates.get(prototype.prototype_type.value, {}).get("unit_test", "")
        
        if not template:
            return f'''
import pytest

def test_{function_name}():
    """Test {function_name} function"""
    # TODO: Implement test for {function_name}
    assert True  # Placeholder test
'''
        
        # Replace template variables
        variables = {
            "class_name": prototype.name.replace(" ", ""),
            "function_name": function_name,
            "test_data": '{"test": "data"}',
            "expected_result": '"expected"',
            "function_call": f'{function_name}(test_data)',
            "invalid_data": '{"invalid": "data"}',
            "function_call_with_invalid_data": f'{function_name}(invalid_data)',
            "function_call_empty": f'{function_name}({{}})',
            "function_call_boundary": f'{function_name}({{"boundary": "value"}})'
        }
        
        return template.format(**variables)
    
    def _generate_integration_test_code(self, prototype: Prototype, endpoint: str) -> str:
        """Generate integration test code for an API endpoint"""
        template = self.test_templates.get(prototype.prototype_type.value, {}).get("integration_test", "")
        
        if not template:
            return f'''
import requests

def test_{endpoint.replace("/", "_")}_integration():
    """Integration test for {endpoint} endpoint"""
    response = requests.get("http://localhost:8000{endpoint}")
    assert response.status_code == 200
'''
        
        # Replace template variables
        variables = {
            "class_name": prototype.name.replace(" ", ""),
            "resource": endpoint.strip("/").replace("/", "_"),
            "endpoint": endpoint.strip("/"),
            "test_data": '{"name": "test", "description": "test data"}'
        }
        
        return template.format(**variables)
    
    def _generate_functional_test_code(self, prototype: Prototype) -> str:
        """Generate functional test code"""
        template = self.test_templates.get(prototype.prototype_type.value, {}).get("functional_test", "")
        
        if not template:
            return f'''
def test_{prototype.name.replace(" ", "_").lower()}_functionality():
    """Test basic functionality of {prototype.name}"""
    # TODO: Implement functional test
    assert True  # Placeholder test
'''
        
        variables = {
            "class_name": prototype.name.replace(" ", ""),
            "test_data": '{"test": "data"}'
        }
        
        return template.format(**variables)
    
    def _generate_performance_test_code(self, prototype: Prototype) -> str:
        """Generate performance test code"""
        template = self.test_templates.get(prototype.prototype_type.value, {}).get("performance_test", "")
        
        if not template:
            return f'''
import time

def test_{prototype.name.replace(" ", "_").lower()}_performance():
    """Test performance of {prototype.name}"""
    start_time = time.time()
    # TODO: Implement performance test
    end_time = time.time()
    
    execution_time = end_time - start_time
    assert execution_time < 5.0  # Should complete within 5 seconds
'''
        
        variables = {
            "class_name": prototype.name.replace(" ", ""),
            "endpoint": "api/test"
        }
        
        return template.format(**variables)
    
    def _generate_security_test_code(self, prototype: Prototype) -> str:
        """Generate security test code"""
        return f'''
import requests

def test_{prototype.name.replace(" ", "_").lower()}_security():
    """Test security aspects of {prototype.name}"""
    # Test for common vulnerabilities
    
    # Test SQL injection
    malicious_input = "'; DROP TABLE users; --"
    # TODO: Test with malicious input
    
    # Test XSS
    xss_payload = "<script>alert('xss')</script>"
    # TODO: Test XSS protection
    
    # Test authentication
    # TODO: Test authentication mechanisms
    
    assert True  # Placeholder - implement actual security tests
'''
    
    def _generate_ui_test_code(self, prototype: Prototype) -> str:
        """Generate UI test code"""
        template = self.test_templates.get(prototype.prototype_type.value, {}).get("ui_test", "")
        
        if not template:
            return f'''
def test_{prototype.name.replace(" ", "_").lower()}_ui():
    """Test UI functionality of {prototype.name}"""
    # TODO: Implement UI test
    assert True  # Placeholder test
'''
        
        variables = {
            "class_name": prototype.name.replace(" ", ""),
            "title": prototype.name
        }
        
        return template.format(**variables)


class TestExecutor:
    """Executes automated tests"""
    
    def __init__(self):
        self.active_executions: Dict[str, TestExecution] = {}
        self.execution_history: List[TestExecution] = []
    
    async def execute_test_suite(self, test_suite: TestSuite, 
                                prototype: Prototype) -> TestReport:
        """Execute a complete test suite"""
        try:
            logger.info(f"Starting test suite execution for {test_suite.name}")
            
            # Initialize test report
            test_report = TestReport(
                prototype_id=prototype.id,
                test_suite_id=test_suite.id,
                total_tests=len(test_suite.test_cases)
            )
            
            start_time = datetime.utcnow()
            
            # Execute tests
            if test_suite.parallel_execution:
                executions = await self._execute_tests_parallel(
                    test_suite.test_cases, test_suite, prototype
                )
            else:
                executions = await self._execute_tests_sequential(
                    test_suite.test_cases, test_suite, prototype
                )
            
            end_time = datetime.utcnow()
            
            # Update test report
            test_report.test_executions = executions
            test_report.total_duration_seconds = (end_time - start_time).total_seconds()
            
            # Calculate statistics
            self._calculate_test_statistics(test_report)
            
            # Analyze results
            test_report.failure_analysis = await self._analyze_failures(executions)
            test_report.performance_analysis = await self._analyze_performance(executions)
            test_report.recommendations = await self._generate_recommendations(test_report)
            
            # Update test suite metadata
            test_suite.last_executed = end_time
            test_suite.execution_count += 1
            
            logger.info(f"Completed test suite execution: {test_report.success_rate:.1%} success rate")
            return test_report
            
        except Exception as e:
            logger.error(f"Error executing test suite: {str(e)}")
            raise
    
    async def _execute_tests_parallel(self, test_cases: List[TestCase], 
                                    test_suite: TestSuite, 
                                    prototype: Prototype) -> List[TestExecution]:
        """Execute tests in parallel"""
        executions = []
        
        # Group tests by dependencies
        independent_tests = [tc for tc in test_cases if not tc.depends_on]
        dependent_tests = [tc for tc in test_cases if tc.depends_on]
        
        # Execute independent tests first
        if independent_tests:
            semaphore = asyncio.Semaphore(test_suite.max_parallel_tests)
            tasks = [
                self._execute_single_test_with_semaphore(tc, test_suite, prototype, semaphore)
                for tc in independent_tests
            ]
            
            independent_executions = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and add successful executions
            for execution in independent_executions:
                if isinstance(execution, TestExecution):
                    executions.append(execution)
        
        # Execute dependent tests sequentially (simplified dependency handling)
        for test_case in dependent_tests:
            execution = await self._execute_single_test(test_case, test_suite, prototype)
            executions.append(execution)
        
        return executions
    
    async def _execute_tests_sequential(self, test_cases: List[TestCase], 
                                      test_suite: TestSuite, 
                                      prototype: Prototype) -> List[TestExecution]:
        """Execute tests sequentially"""
        executions = []
        
        for test_case in test_cases:
            execution = await self._execute_single_test(test_case, test_suite, prototype)
            executions.append(execution)
            
            # Stop on failure if configured
            if test_suite.stop_on_failure and execution.status == TestStatus.FAILED:
                logger.info("Stopping test execution due to failure")
                break
        
        return executions
    
    async def _execute_single_test_with_semaphore(self, test_case: TestCase, 
                                                test_suite: TestSuite, 
                                                prototype: Prototype,
                                                semaphore: asyncio.Semaphore) -> TestExecution:
        """Execute a single test with semaphore for parallel execution"""
        async with semaphore:
            return await self._execute_single_test(test_case, test_suite, prototype)
    
    async def _execute_single_test(self, test_case: TestCase, 
                                 test_suite: TestSuite, 
                                 prototype: Prototype) -> TestExecution:
        """Execute a single test case"""
        execution = TestExecution(
            test_case_id=test_case.id,
            test_suite_id=test_suite.id,
            prototype_id=prototype.id,
            start_time=datetime.utcnow()
        )
        
        try:
            execution.status = TestStatus.RUNNING
            self.active_executions[execution.id] = execution
            
            logger.info(f"Executing test: {test_case.name}")
            
            # Setup test environment
            await self._setup_test_environment(test_case, execution)
            
            # Execute the test
            result = await self._run_test_code(test_case, execution)
            
            # Process results
            execution.actual_result = result
            execution.status = TestStatus.PASSED if result.get("success", False) else TestStatus.FAILED
            
            if not result.get("success", False):
                execution.error_message = result.get("error", "Test failed")
                execution.stack_trace = result.get("stack_trace", "")
            
            # Update metrics
            execution.assertions_passed = result.get("assertions_passed", 0)
            execution.assertions_failed = result.get("assertions_failed", 0)
            execution.performance_metrics = result.get("performance_metrics", {})
            
        except asyncio.TimeoutError:
            execution.status = TestStatus.TIMEOUT
            execution.error_message = f"Test timed out after {test_case.timeout_seconds} seconds"
            
        except Exception as e:
            execution.status = TestStatus.ERROR
            execution.error_message = str(e)
            logger.error(f"Error executing test {test_case.name}: {str(e)}")
            
        finally:
            # Cleanup
            await self._cleanup_test_environment(test_case, execution)
            
            execution.end_time = datetime.utcnow()
            execution.duration_seconds = (
                execution.end_time - execution.start_time
            ).total_seconds()
            
            # Remove from active executions
            if execution.id in self.active_executions:
                del self.active_executions[execution.id]
            
            # Add to history
            self.execution_history.append(execution)
        
        return execution
    
    async def _setup_test_environment(self, test_case: TestCase, execution: TestExecution):
        """Setup test environment"""
        try:
            # Run setup commands
            for command in test_case.setup_commands:
                await self._run_command(command, execution)
            
            # Set environment variables
            execution.execution_environment = {
                "TEST_ID": execution.id,
                "TEST_NAME": test_case.name,
                "TIMESTAMP": datetime.utcnow().isoformat(),
                **test_case.environment_requirements
            }
            
        except Exception as e:
            logger.error(f"Error setting up test environment: {str(e)}")
            raise
    
    async def _cleanup_test_environment(self, test_case: TestCase, execution: TestExecution):
        """Cleanup test environment"""
        try:
            # Run teardown commands
            for command in test_case.teardown_commands:
                await self._run_command(command, execution)
                
        except Exception as e:
            logger.error(f"Error cleaning up test environment: {str(e)}")
    
    async def _run_command(self, command: str, execution: TestExecution):
        """Run a shell command"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if stdout:
                execution.output_logs.append(f"STDOUT: {stdout.decode()}")
            if stderr:
                execution.output_logs.append(f"STDERR: {stderr.decode()}")
                
        except Exception as e:
            execution.output_logs.append(f"Command error: {str(e)}")
    
    async def _run_test_code(self, test_case: TestCase, execution: TestExecution) -> Dict[str, Any]:
        """Run the actual test code"""
        try:
            # Simulate test execution
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate test execution time
            
            # Simulate test results based on test type and content
            success_probability = self._calculate_success_probability(test_case)
            success = random.random() < success_probability
            
            result = {
                "success": success,
                "assertions_passed": random.randint(1, 5) if success else 0,
                "assertions_failed": 0 if success else random.randint(1, 3),
                "performance_metrics": {
                    "execution_time": random.uniform(0.1, 2.0),
                    "memory_usage": random.uniform(10, 100),
                    "cpu_usage": random.uniform(5, 50)
                }
            }
            
            if not success:
                result["error"] = f"Test {test_case.name} failed during execution"
                result["stack_trace"] = f"Simulated stack trace for {test_case.name}"
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stack_trace": f"Exception in test execution: {str(e)}"
            }
    
    def _calculate_success_probability(self, test_case: TestCase) -> float:
        """Calculate probability of test success based on test characteristics"""
        base_probability = 0.85  # 85% base success rate
        
        # Adjust based on test type
        type_adjustments = {
            TestType.UNIT_TEST: 0.05,
            TestType.INTEGRATION_TEST: -0.05,
            TestType.PERFORMANCE_TEST: -0.10,
            TestType.SECURITY_TEST: -0.15,
            TestType.UI_TEST: -0.10
        }
        
        adjustment = type_adjustments.get(test_case.test_type, 0)
        
        # Adjust based on priority (higher priority tests are more likely to pass)
        priority_adjustments = {
            TestPriority.CRITICAL: 0.05,
            TestPriority.HIGH: 0.02,
            TestPriority.MEDIUM: 0,
            TestPriority.LOW: -0.02
        }
        
        priority_adjustment = priority_adjustments.get(test_case.priority, 0)
        
        return max(0.1, min(0.95, base_probability + adjustment + priority_adjustment))
    
    def _calculate_test_statistics(self, test_report: TestReport):
        """Calculate test statistics for the report"""
        executions = test_report.test_executions
        
        if not executions:
            return
        
        # Count test results by status
        status_counts = {}
        for execution in executions:
            status = execution.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        test_report.passed_tests = status_counts.get("passed", 0)
        test_report.failed_tests = status_counts.get("failed", 0)
        test_report.skipped_tests = status_counts.get("skipped", 0)
        test_report.error_tests = status_counts.get("error", 0) + status_counts.get("timeout", 0)
        
        # Calculate success rate
        total_executed = len([e for e in executions if e.status != TestStatus.SKIPPED])
        test_report.success_rate = (
            test_report.passed_tests / total_executed if total_executed > 0 else 0
        )
        
        # Calculate average test duration
        durations = [e.duration_seconds for e in executions if e.duration_seconds > 0]
        test_report.average_test_duration = sum(durations) / len(durations) if durations else 0
        
        # Calculate coverage (simplified)
        coverage_scores = [
            e.coverage_percentage for e in executions if e.coverage_percentage > 0
        ]
        test_report.overall_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0
        
        # Calculate quality scores
        test_report.performance_score = self._calculate_performance_score(executions)
        test_report.security_score = self._calculate_security_score(executions)
        test_report.code_quality_score = test_report.success_rate  # Simplified
    
    def _calculate_performance_score(self, executions: List[TestExecution]) -> float:
        """Calculate performance score based on test executions"""
        performance_executions = [
            e for e in executions 
            if e.performance_metrics and e.status == TestStatus.PASSED
        ]
        
        if not performance_executions:
            return 0.8  # Default score
        
        # Calculate based on execution times
        avg_execution_time = sum(
            e.performance_metrics.get("execution_time", 1.0) 
            for e in performance_executions
        ) / len(performance_executions)
        
        # Score inversely related to execution time (faster = better)
        performance_score = max(0.1, min(1.0, 2.0 / (1.0 + avg_execution_time)))
        
        return performance_score
    
    def _calculate_security_score(self, executions: List[TestExecution]) -> float:
        """Calculate security score based on security test results"""
        security_executions = [
            e for e in executions 
            if "security" in e.test_case_id.lower()
        ]
        
        if not security_executions:
            return 0.7  # Default score when no security tests
        
        passed_security_tests = len([
            e for e in security_executions if e.status == TestStatus.PASSED
        ])
        
        return passed_security_tests / len(security_executions) if security_executions else 0.7
    
    async def _analyze_failures(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Analyze test failures to identify patterns"""
        failed_executions = [e for e in executions if e.status == TestStatus.FAILED]
        
        if not failed_executions:
            return {"total_failures": 0, "failure_patterns": []}
        
        # Analyze failure patterns
        failure_patterns = {}
        error_categories = {}
        
        for execution in failed_executions:
            # Categorize errors
            error_msg = execution.error_message or ""
            
            if "timeout" in error_msg.lower():
                error_categories["timeout"] = error_categories.get("timeout", 0) + 1
            elif "connection" in error_msg.lower():
                error_categories["connection"] = error_categories.get("connection", 0) + 1
            elif "assertion" in error_msg.lower():
                error_categories["assertion"] = error_categories.get("assertion", 0) + 1
            else:
                error_categories["other"] = error_categories.get("other", 0) + 1
        
        return {
            "total_failures": len(failed_executions),
            "failure_rate": len(failed_executions) / len(executions) if executions else 0,
            "error_categories": error_categories,
            "most_common_error": max(error_categories, key=error_categories.get) if error_categories else None,
            "failure_patterns": failure_patterns
        }
    
    async def _analyze_performance(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Analyze performance metrics from test executions"""
        performance_data = []
        
        for execution in executions:
            if execution.performance_metrics:
                performance_data.append(execution.performance_metrics)
        
        if not performance_data:
            return {"message": "No performance data available"}
        
        # Calculate aggregate metrics
        avg_execution_time = sum(
            data.get("execution_time", 0) for data in performance_data
        ) / len(performance_data)
        
        avg_memory_usage = sum(
            data.get("memory_usage", 0) for data in performance_data
        ) / len(performance_data)
        
        avg_cpu_usage = sum(
            data.get("cpu_usage", 0) for data in performance_data
        ) / len(performance_data)
        
        return {
            "average_execution_time": avg_execution_time,
            "average_memory_usage": avg_memory_usage,
            "average_cpu_usage": avg_cpu_usage,
            "performance_score": max(0.1, min(1.0, 2.0 / (1.0 + avg_execution_time))),
            "total_tests_analyzed": len(performance_data)
        }
    
    async def _generate_recommendations(self, test_report: TestReport) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Success rate recommendations
        if test_report.success_rate < 0.8:
            recommendations.append("Test success rate is below 80% - investigate failing tests")
        elif test_report.success_rate > 0.95:
            recommendations.append("Excellent test success rate - consider adding more challenging tests")
        
        # Performance recommendations
        if test_report.performance_score < 0.6:
            recommendations.append("Performance tests indicate slow execution - optimize critical paths")
        
        # Coverage recommendations
        if test_report.overall_coverage < 0.7:
            recommendations.append("Test coverage is below 70% - add more comprehensive tests")
        
        # Security recommendations
        if test_report.security_score < 0.8:
            recommendations.append("Security test results indicate vulnerabilities - review security measures")
        
        # Failure analysis recommendations
        if test_report.failure_analysis.get("total_failures", 0) > 0:
            most_common_error = test_report.failure_analysis.get("most_common_error")
            if most_common_error:
                recommendations.append(f"Most common error type is '{most_common_error}' - focus on fixing these issues")
        
        # Duration recommendations
        if test_report.average_test_duration > 30:
            recommendations.append("Average test duration is high - consider optimizing test execution")
        
        return recommendations


class TestingAutomationFramework:
    """Main testing automation framework"""
    
    def __init__(self):
        self.test_generator = TestGenerator()
        self.test_executor = TestExecutor()
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_reports: Dict[str, TestReport] = {}
    
    async def create_and_execute_tests(self, prototype: Prototype) -> TestReport:
        """Create and execute comprehensive tests for a prototype"""
        try:
            logger.info(f"Starting automated testing for prototype {prototype.id}")
            
            # Generate test suite
            test_suite = await self.test_generator.generate_test_suite(prototype)
            self.test_suites[prototype.id] = test_suite
            
            # Execute test suite
            test_report = await self.test_executor.execute_test_suite(test_suite, prototype)
            self.test_reports[prototype.id] = test_report
            
            # Update prototype with test results
            await self._update_prototype_with_results(prototype, test_report)
            
            logger.info(f"Completed automated testing for prototype {prototype.id}")
            return test_report
            
        except Exception as e:
            logger.error(f"Error in automated testing: {str(e)}")
            raise
    
    async def _update_prototype_with_results(self, prototype: Prototype, test_report: TestReport):
        """Update prototype with test results"""
        # Convert test executions to TestResult objects
        test_results = []
        
        for execution in test_report.test_executions:
            test_result = TestResult(
                test_name=f"Test {execution.test_case_id}",
                test_type=execution.test_case_id.split("_")[0] if "_" in execution.test_case_id else "general",
                status=execution.status.value,
                execution_time=execution.duration_seconds,
                error_message=execution.error_message,
                metrics=execution.performance_metrics
            )
            test_results.append(test_result)
        
        prototype.test_results = test_results
        
        # Update prototype quality metrics based on test results
        if prototype.quality_metrics:
            # Improve quality metrics based on successful tests
            if test_report.success_rate > 0.8:
                prototype.quality_metrics.reliability_score = min(
                    prototype.quality_metrics.reliability_score + 0.05, 1.0
                )
            
            if test_report.performance_score > 0.7:
                prototype.quality_metrics.performance_score = min(
                    prototype.quality_metrics.performance_score + 0.03, 1.0
                )
            
            if test_report.security_score > 0.8:
                prototype.quality_metrics.security_score = min(
                    prototype.quality_metrics.security_score + 0.04, 1.0
                )
    
    async def get_test_suite(self, prototype_id: str) -> Optional[TestSuite]:
        """Get test suite for a prototype"""
        return self.test_suites.get(prototype_id)
    
    async def get_test_report(self, prototype_id: str) -> Optional[TestReport]:
        """Get test report for a prototype"""
        return self.test_reports.get(prototype_id)
    
    async def rerun_failed_tests(self, prototype_id: str) -> Optional[TestReport]:
        """Rerun only the failed tests for a prototype"""
        try:
            test_suite = self.test_suites.get(prototype_id)
            previous_report = self.test_reports.get(prototype_id)
            
            if not test_suite or not previous_report:
                logger.error(f"No test suite or report found for prototype {prototype_id}")
                return None
            
            # Get failed test case IDs
            failed_test_ids = [
                execution.test_case_id 
                for execution in previous_report.test_executions 
                if execution.status == TestStatus.FAILED
            ]
            
            if not failed_test_ids:
                logger.info(f"No failed tests to rerun for prototype {prototype_id}")
                return previous_report
            
            # Create new test suite with only failed tests
            failed_test_cases = [
                tc for tc in test_suite.test_cases 
                if tc.id in failed_test_ids
            ]
            
            rerun_suite = TestSuite(
                name=f"{test_suite.name} - Failed Tests Rerun",
                description="Rerun of previously failed tests",
                prototype_id=prototype_id,
                test_cases=failed_test_cases
            )
            
            # Execute failed tests
            # Note: In a real implementation, we'd need the prototype object
            # For now, we'll simulate the rerun
            logger.info(f"Rerunning {len(failed_test_cases)} failed tests for prototype {prototype_id}")
            
            return previous_report
            
        except Exception as e:
            logger.error(f"Error rerunning failed tests: {str(e)}")
            return None
    
    async def generate_comprehensive_report(self, prototype_id: str) -> Dict[str, Any]:
        """Generate comprehensive testing report"""
        try:
            test_suite = self.test_suites.get(prototype_id)
            test_report = self.test_reports.get(prototype_id)
            
            if not test_suite or not test_report:
                return {"error": "No test data available for prototype"}
            
            return {
                "prototype_id": prototype_id,
                "test_suite_info": {
                    "name": test_suite.name,
                    "total_test_cases": len(test_suite.test_cases),
                    "execution_count": test_suite.execution_count,
                    "last_executed": test_suite.last_executed
                },
                "test_results_summary": {
                    "total_tests": test_report.total_tests,
                    "passed_tests": test_report.passed_tests,
                    "failed_tests": test_report.failed_tests,
                    "skipped_tests": test_report.skipped_tests,
                    "error_tests": test_report.error_tests,
                    "success_rate": test_report.success_rate,
                    "total_duration": test_report.total_duration_seconds,
                    "average_test_duration": test_report.average_test_duration
                },
                "quality_metrics": {
                    "overall_coverage": test_report.overall_coverage,
                    "performance_score": test_report.performance_score,
                    "security_score": test_report.security_score,
                    "code_quality_score": test_report.code_quality_score
                },
                "analysis": {
                    "failure_analysis": test_report.failure_analysis,
                    "performance_analysis": test_report.performance_analysis,
                    "recommendations": test_report.recommendations
                },
                "test_breakdown_by_type": self._get_test_breakdown_by_type(test_report),
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            return {"error": str(e)}
    
    def _get_test_breakdown_by_type(self, test_report: TestReport) -> Dict[str, Dict[str, int]]:
        """Get test breakdown by type"""
        breakdown = {}
        
        for execution in test_report.test_executions:
            # Extract test type from test case ID (simplified)
            test_type = "general"
            if "unit" in execution.test_case_id.lower():
                test_type = "unit"
            elif "integration" in execution.test_case_id.lower():
                test_type = "integration"
            elif "performance" in execution.test_case_id.lower():
                test_type = "performance"
            elif "security" in execution.test_case_id.lower():
                test_type = "security"
            elif "ui" in execution.test_case_id.lower():
                test_type = "ui"
            
            if test_type not in breakdown:
                breakdown[test_type] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "error": 0
                }
            
            breakdown[test_type]["total"] += 1
            
            if execution.status == TestStatus.PASSED:
                breakdown[test_type]["passed"] += 1
            elif execution.status == TestStatus.FAILED:
                breakdown[test_type]["failed"] += 1
            else:
                breakdown[test_type]["error"] += 1
        
        return breakdown
    
    async def get_testing_analytics(self) -> Dict[str, Any]:
        """Get analytics across all testing activities"""
        try:
            total_suites = len(self.test_suites)
            total_reports = len(self.test_reports)
            
            if total_reports == 0:
                return {
                    "total_test_suites": total_suites,
                    "total_test_reports": total_reports,
                    "message": "No test reports available"
                }
            
            # Aggregate statistics
            total_tests = sum(report.total_tests for report in self.test_reports.values())
            total_passed = sum(report.passed_tests for report in self.test_reports.values())
            total_failed = sum(report.failed_tests for report in self.test_reports.values())
            
            avg_success_rate = sum(
                report.success_rate for report in self.test_reports.values()
            ) / total_reports
            
            avg_performance_score = sum(
                report.performance_score for report in self.test_reports.values()
            ) / total_reports
            
            avg_security_score = sum(
                report.security_score for report in self.test_reports.values()
            ) / total_reports
            
            return {
                "total_test_suites": total_suites,
                "total_test_reports": total_reports,
                "aggregate_statistics": {
                    "total_tests_executed": total_tests,
                    "total_tests_passed": total_passed,
                    "total_tests_failed": total_failed,
                    "overall_success_rate": avg_success_rate,
                    "average_performance_score": avg_performance_score,
                    "average_security_score": avg_security_score
                },
                "testing_trends": {
                    "most_tested_prototype": max(
                        self.test_reports.keys(),
                        key=lambda k: self.test_reports[k].total_tests
                    ) if self.test_reports else None,
                    "highest_success_rate": max(
                        self.test_reports.values(),
                        key=lambda r: r.success_rate
                    ).prototype_id if self.test_reports else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting testing analytics: {str(e)}")
            return {"error": str(e)}