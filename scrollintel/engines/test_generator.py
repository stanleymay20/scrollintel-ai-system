"""
Automated Test Generation Engine

This module provides comprehensive test generation capabilities for
generated code components, including unit tests, integration tests,
end-to-end tests, and performance tests.
"""

import ast
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..models.test_generation_models import (
    TestSuite, TestCase, TestAssertion, TestType, TestFramework,
    TestGenerationRequest, TestGenerationResult, PerformanceTestConfig,
    LoadTestScenario
)
# Note: GeneratedCode and CodeComponent models would be defined elsewhere
# For now, we'll work with the test generation models directly


class TestGenerator:
    """Main test generation engine"""
    
    def __init__(self):
        self.unit_test_generator = UnitTestGenerator()
        self.integration_test_generator = IntegrationTestGenerator()
        self.e2e_test_generator = EndToEndTestGenerator()
        self.performance_test_generator = PerformanceTestGenerator()
        self.test_validator = TestValidator()
    
    def generate_comprehensive_tests(self, request: TestGenerationRequest) -> TestGenerationResult:
        """Generate comprehensive test suite for given code"""
        start_time = datetime.utcnow()
        
        # Parse the target code
        code_analysis = self._analyze_code(request.target_code, request.code_type)
        
        # Generate test suite
        test_suite = TestSuite(
            id=str(uuid.uuid4()),
            name=f"Tests for {code_analysis['name']}",
            description=f"Comprehensive test suite for {request.code_type}",
            framework=request.framework
        )
        
        # Generate different types of tests based on request
        for test_type in request.test_types:
            if test_type == TestType.UNIT:
                unit_tests = self.unit_test_generator.generate_unit_tests(
                    request.target_code, code_analysis, request
                )
                test_suite.test_cases.extend(unit_tests)
            
            elif test_type == TestType.INTEGRATION:
                integration_tests = self.integration_test_generator.generate_integration_tests(
                    request.target_code, code_analysis, request
                )
                test_suite.test_cases.extend(integration_tests)
            
            elif test_type == TestType.END_TO_END:
                e2e_tests = self.e2e_test_generator.generate_e2e_tests(
                    request.target_code, code_analysis, request
                )
                test_suite.test_cases.extend(e2e_tests)
            
            elif test_type in [TestType.PERFORMANCE, TestType.LOAD]:
                perf_tests = self.performance_test_generator.generate_performance_tests(
                    request.target_code, code_analysis, request
                )
                test_suite.test_cases.extend(perf_tests)
        
        # Update suite statistics
        test_suite.total_tests = len(test_suite.test_cases)
        
        # Validate generated tests
        validation_result = self.test_validator.validate_test_suite(test_suite)
        
        # Calculate generation time
        generation_time = (datetime.utcnow() - start_time).total_seconds()
        
        return TestGenerationResult(
            request_id=str(uuid.uuid4()),
            generated_suite=test_suite,
            generation_time=generation_time,
            estimated_coverage=self._estimate_coverage(test_suite, code_analysis),
            test_count=test_suite.total_tests,
            complexity_score=validation_result['complexity_score'],
            maintainability_score=validation_result['maintainability_score'],
            recommendations=validation_result['recommendations'],
            warnings=validation_result['warnings']
        )
    
    def _analyze_code(self, code: str, code_type: str) -> Dict[str, Any]:
        """Analyze code structure for test generation"""
        analysis = {
            'name': 'unknown',
            'type': code_type,
            'functions': [],
            'classes': [],
            'imports': [],
            'complexity': 1
        }
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['functions'].append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'returns': self._get_return_type(node),
                        'docstring': ast.get_docstring(node),
                        'line_number': node.lineno
                    })
                
                elif isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    analysis['classes'].append({
                        'name': node.name,
                        'methods': methods,
                        'docstring': ast.get_docstring(node),
                        'line_number': node.lineno
                    })
                
                elif isinstance(node, ast.Import):
                    analysis['imports'].extend([alias.name for alias in node.names])
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    analysis['imports'].extend([f"{module}.{alias.name}" for alias in node.names])
            
            # Set name based on first class or function found
            if analysis['classes']:
                analysis['name'] = analysis['classes'][0]['name']
            elif analysis['functions']:
                analysis['name'] = analysis['functions'][0]['name']
            
            # Calculate complexity
            analysis['complexity'] = self._calculate_complexity(tree)
            
        except SyntaxError:
            # Handle non-Python code or syntax errors
            analysis['name'] = self._extract_name_from_code(code, code_type)
        
        return analysis
    
    def _get_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation if available"""
        if node.returns:
            return ast.unparse(node.returns)
        return None
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    def _extract_name_from_code(self, code: str, code_type: str) -> str:
        """Extract name from non-Python code"""
        if code_type == 'api':
            # Extract API endpoint name
            match = re.search(r'@app\.route\([\'"]([^\'"]+)[\'"]', code)
            if match:
                return match.group(1).replace('/', '_').strip('_')
        
        # Default fallback
        return f"{code_type}_component"
    
    def _estimate_coverage(self, test_suite: TestSuite, code_analysis: Dict[str, Any]) -> float:
        """Estimate code coverage based on generated tests"""
        total_functions = len(code_analysis['functions'])
        total_classes = len(code_analysis['classes'])
        total_components = max(total_functions + total_classes, 1)
        
        # Count tests that cover different components
        covered_components = set()
        for test_case in test_suite.test_cases:
            # Simple heuristic: if test name contains function/class name, it covers it
            for func in code_analysis['functions']:
                if func['name'].lower() in test_case.name.lower():
                    covered_components.add(func['name'])
            
            for cls in code_analysis['classes']:
                if cls['name'].lower() in test_case.name.lower():
                    covered_components.add(cls['name'])
        
        return min(len(covered_components) / total_components, 1.0)


class UnitTestGenerator:
    """Generates unit tests for individual functions and classes"""
    
    def generate_unit_tests(self, code: str, analysis: Dict[str, Any], request: TestGenerationRequest) -> List[TestCase]:
        """Generate unit tests for functions and classes"""
        tests = []
        
        # Generate tests for functions
        for func in analysis['functions']:
            tests.extend(self._generate_function_tests(func, request))
        
        # Generate tests for classes
        for cls in analysis['classes']:
            tests.extend(self._generate_class_tests(cls, request))
        
        return tests
    
    def _generate_function_tests(self, func: Dict[str, Any], request: TestGenerationRequest) -> List[TestCase]:
        """Generate tests for a specific function"""
        tests = []
        func_name = func['name']
        
        # Happy path test
        tests.append(TestCase(
            id=str(uuid.uuid4()),
            name=f"test_{func_name}_happy_path",
            description=f"Test {func_name} with valid inputs",
            test_type=TestType.UNIT,
            framework=request.framework,
            setup_code=self._generate_setup_code(func, request.framework),
            test_code=self._generate_function_test_code(func, "happy_path", request.framework),
            assertions=[
                TestAssertion(
                    assertion_type="not_none",
                    expected_value=True,
                    actual_expression=f"result is not None",
                    description=f"Function {func_name} should return a value"
                )
            ],
            tags=["unit", "happy_path"],
            dependencies=self._get_test_dependencies(request.framework)
        ))
        
        # Edge cases if requested
        if request.include_edge_cases:
            tests.append(TestCase(
                id=str(uuid.uuid4()),
                name=f"test_{func_name}_edge_cases",
                description=f"Test {func_name} with edge case inputs",
                test_type=TestType.UNIT,
                framework=request.framework,
                setup_code=self._generate_setup_code(func, request.framework),
                test_code=self._generate_function_test_code(func, "edge_cases", request.framework),
                assertions=[
                    TestAssertion(
                        assertion_type="handles_edge_cases",
                        expected_value=True,
                        actual_expression="result is handled properly",
                        description=f"Function {func_name} should handle edge cases"
                    )
                ],
                tags=["unit", "edge_cases"]
            ))
        
        # Error cases if requested
        if request.include_error_cases:
            tests.append(TestCase(
                id=str(uuid.uuid4()),
                name=f"test_{func_name}_error_handling",
                description=f"Test {func_name} error handling",
                test_type=TestType.UNIT,
                framework=request.framework,
                setup_code=self._generate_setup_code(func, request.framework),
                test_code=self._generate_function_test_code(func, "error_cases", request.framework),
                assertions=[
                    TestAssertion(
                        assertion_type="raises_exception",
                        expected_value="ValueError",
                        actual_expression="pytest.raises(ValueError)",
                        description=f"Function {func_name} should raise appropriate exceptions"
                    )
                ],
                tags=["unit", "error_handling"]
            ))
        
        return tests
    
    def _generate_class_tests(self, cls: Dict[str, Any], request: TestGenerationRequest) -> List[TestCase]:
        """Generate tests for a class"""
        tests = []
        cls_name = cls['name']
        
        # Constructor test
        tests.append(TestCase(
            id=str(uuid.uuid4()),
            name=f"test_{cls_name.lower()}_initialization",
            description=f"Test {cls_name} initialization",
            test_type=TestType.UNIT,
            framework=request.framework,
            test_code=self._generate_class_init_test(cls, request.framework),
            assertions=[
                TestAssertion(
                    assertion_type="instance_created",
                    expected_value=True,
                    actual_expression=f"isinstance(instance, {cls_name})",
                    description=f"{cls_name} instance should be created successfully"
                )
            ],
            tags=["unit", "initialization"]
        ))
        
        # Method tests
        for method in cls['methods']:
            if not method.startswith('_'):  # Skip private methods
                tests.append(TestCase(
                    id=str(uuid.uuid4()),
                    name=f"test_{cls_name.lower()}_{method}",
                    description=f"Test {cls_name}.{method} method",
                    test_type=TestType.UNIT,
                    framework=request.framework,
                    setup_code=f"instance = {cls_name}()",
                    test_code=self._generate_method_test_code(cls_name, method, request.framework),
                    tags=["unit", "method"]
                ))
        
        return tests
    
    def _generate_setup_code(self, func: Dict[str, Any], framework: TestFramework) -> str:
        """Generate setup code for function tests"""
        if framework == TestFramework.PYTEST:
            return "# Setup test data and mocks"
        elif framework == TestFramework.JEST:
            return "// Setup test data and mocks"
        return "# Setup code"
    
    def _generate_function_test_code(self, func: Dict[str, Any], test_type: str, framework: TestFramework) -> str:
        """Generate test code for function"""
        func_name = func['name']
        args = func['args']
        
        if framework == TestFramework.PYTEST:
            if test_type == "happy_path":
                test_args = ", ".join([f"test_{arg}" for arg in args])
                return f"""
# Arrange
{chr(10).join([f"test_{arg} = 'test_value'" for arg in args])}

# Act
result = {func_name}({test_args})

# Assert
assert result is not None
"""
            elif test_type == "edge_cases":
                return f"""
# Test with empty/None values
result = {func_name}({', '.join(['None'] * len(args))})
assert result is not None or result is None  # Handle appropriately
"""
            elif test_type == "error_cases":
                return f"""
# Test with invalid inputs
with pytest.raises((ValueError, TypeError)):
    {func_name}({', '.join(['invalid_input'] * len(args))})
"""
        
        return f"# Test {func_name} - {test_type}"
    
    def _generate_class_init_test(self, cls: Dict[str, Any], framework: TestFramework) -> str:
        """Generate class initialization test"""
        cls_name = cls['name']
        
        if framework == TestFramework.PYTEST:
            return f"""
# Arrange & Act
instance = {cls_name}()

# Assert
assert instance is not None
assert isinstance(instance, {cls_name})
"""
        
        return f"# Test {cls_name} initialization"
    
    def _generate_method_test_code(self, cls_name: str, method: str, framework: TestFramework) -> str:
        """Generate method test code"""
        if framework == TestFramework.PYTEST:
            return f"""
# Act
result = instance.{method}()

# Assert
assert result is not None  # Adjust based on expected return
"""
        
        return f"# Test {cls_name}.{method}"
    
    def _get_test_dependencies(self, framework: TestFramework) -> List[str]:
        """Get required dependencies for test framework"""
        if framework == TestFramework.PYTEST:
            return ["pytest", "pytest-mock"]
        elif framework == TestFramework.JEST:
            return ["jest", "@testing-library/jest-dom"]
        return []


class IntegrationTestGenerator:
    """Generates integration tests for API endpoints and component interactions"""
    
    def generate_integration_tests(self, code: str, analysis: Dict[str, Any], request: TestGenerationRequest) -> List[TestCase]:
        """Generate integration tests"""
        tests = []
        
        # Detect if this is API code
        if self._is_api_code(code):
            tests.extend(self._generate_api_integration_tests(code, analysis, request))
        
        # Generate component integration tests
        tests.extend(self._generate_component_integration_tests(analysis, request))
        
        return tests
    
    def _is_api_code(self, code: str) -> bool:
        """Check if code contains API endpoints"""
        api_patterns = [
            r'@app\.route',
            r'@router\.',
            r'app\.get|app\.post|app\.put|app\.delete',
            r'FastAPI',
            r'Flask'
        ]
        
        for pattern in api_patterns:
            if re.search(pattern, code):
                return True
        return False
    
    def _generate_api_integration_tests(self, code: str, analysis: Dict[str, Any], request: TestGenerationRequest) -> List[TestCase]:
        """Generate API integration tests"""
        tests = []
        
        # Extract API endpoints
        endpoints = self._extract_api_endpoints(code)
        
        for endpoint in endpoints:
            # GET endpoint test
            if endpoint['method'] in ['GET', 'ALL']:
                tests.append(TestCase(
                    id=str(uuid.uuid4()),
                    name=f"test_api_{endpoint['name']}_get",
                    description=f"Test GET {endpoint['path']} endpoint",
                    test_type=TestType.INTEGRATION,
                    framework=request.framework,
                    setup_code=self._generate_api_setup_code(request.framework),
                    test_code=self._generate_api_test_code(endpoint, 'GET', request.framework),
                    assertions=[
                        TestAssertion(
                            assertion_type="status_code",
                            expected_value=200,
                            actual_expression="response.status_code",
                            description="API should return 200 OK"
                        )
                    ],
                    tags=["integration", "api", "get"]
                ))
            
            # POST endpoint test
            if endpoint['method'] in ['POST', 'ALL']:
                tests.append(TestCase(
                    id=str(uuid.uuid4()),
                    name=f"test_api_{endpoint['name']}_post",
                    description=f"Test POST {endpoint['path']} endpoint",
                    test_type=TestType.INTEGRATION,
                    framework=request.framework,
                    setup_code=self._generate_api_setup_code(request.framework),
                    test_code=self._generate_api_test_code(endpoint, 'POST', request.framework),
                    assertions=[
                        TestAssertion(
                            assertion_type="status_code",
                            expected_value=201,
                            actual_expression="response.status_code",
                            description="API should return 201 Created"
                        )
                    ],
                    tags=["integration", "api", "post"]
                ))
        
        return tests
    
    def _extract_api_endpoints(self, code: str) -> List[Dict[str, Any]]:
        """Extract API endpoints from code"""
        endpoints = []
        
        # Flask/FastAPI route patterns
        route_patterns = [
            r'@app\.route\([\'"]([^\'"]+)[\'"](?:.*methods=\[([^\]]+)\])?',
            r'@router\.(get|post|put|delete)\([\'"]([^\'"]+)[\'"]',
            r'app\.(get|post|put|delete)\([\'"]([^\'"]+)[\'"]'
        ]
        
        for pattern in route_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                if len(match.groups()) >= 2:
                    path = match.group(1) if match.group(1).startswith('/') else match.group(2)
                    method = match.group(2) if match.group(2) else 'ALL'
                    
                    endpoints.append({
                        'path': path,
                        'method': method.upper(),
                        'name': path.replace('/', '_').strip('_') or 'root'
                    })
        
        return endpoints
    
    def _generate_api_setup_code(self, framework: TestFramework) -> str:
        """Generate API test setup code"""
        if framework == TestFramework.PYTEST:
            return """
import pytest
from fastapi.testclient import TestClient
from your_app import app

client = TestClient(app)
"""
        return "# API test setup"
    
    def _generate_api_test_code(self, endpoint: Dict[str, Any], method: str, framework: TestFramework) -> str:
        """Generate API test code"""
        path = endpoint['path']
        
        if framework == TestFramework.PYTEST:
            if method == 'GET':
                return f"""
# Act
response = client.get("{path}")

# Assert
assert response.status_code == 200
assert response.json() is not None
"""
            elif method == 'POST':
                return f"""
# Arrange
test_data = {{"key": "value"}}

# Act
response = client.post("{path}", json=test_data)

# Assert
assert response.status_code in [200, 201]
"""
        
        return f"# Test {method} {path}"
    
    def _generate_component_integration_tests(self, analysis: Dict[str, Any], request: TestGenerationRequest) -> List[TestCase]:
        """Generate component integration tests"""
        tests = []
        
        # If multiple classes exist, test their interactions
        if len(analysis['classes']) > 1:
            tests.append(TestCase(
                id=str(uuid.uuid4()),
                name="test_component_integration",
                description="Test integration between components",
                test_type=TestType.INTEGRATION,
                framework=request.framework,
                test_code=self._generate_component_integration_code(analysis, request.framework),
                tags=["integration", "components"]
            ))
        
        return tests


class EndToEndTestGenerator:
    """Generates end-to-end tests for complete user workflows"""
    
    def generate_e2e_tests(self, code: str, analysis: Dict[str, Any], request: TestGenerationRequest) -> List[TestCase]:
        """Generate end-to-end tests"""
        tests = []
        
        # Generate user workflow tests
        workflows = self._identify_user_workflows(code, analysis)
        
        for workflow in workflows:
            tests.append(TestCase(
                id=str(uuid.uuid4()),
                name=f"test_e2e_{workflow['name']}",
                description=f"End-to-end test for {workflow['description']}",
                test_type=TestType.END_TO_END,
                framework=TestFramework.CYPRESS,  # Default to Cypress for E2E
                setup_code=self._generate_e2e_setup_code(),
                test_code=self._generate_e2e_test_code(workflow),
                tags=["e2e", "workflow"],
                timeout=30  # E2E tests typically need more time
            ))
        
        return tests
    
    def _identify_user_workflows(self, code: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify user workflows from code"""
        workflows = []
        
        # Basic workflow identification
        if self._is_crud_api(code):
            workflows.append({
                'name': 'crud_workflow',
                'description': 'Complete CRUD operations workflow',
                'steps': ['create', 'read', 'update', 'delete']
            })
        
        if self._has_authentication(code):
            workflows.append({
                'name': 'auth_workflow',
                'description': 'User authentication workflow',
                'steps': ['login', 'access_protected_resource', 'logout']
            })
        
        return workflows
    
    def _is_crud_api(self, code: str) -> bool:
        """Check if code implements CRUD operations"""
        crud_methods = ['get', 'post', 'put', 'delete']
        found_methods = sum(1 for method in crud_methods if method in code.lower())
        return found_methods >= 3
    
    def _has_authentication(self, code: str) -> bool:
        """Check if code has authentication"""
        auth_keywords = ['login', 'authenticate', 'token', 'session', 'auth']
        return any(keyword in code.lower() for keyword in auth_keywords)
    
    def _generate_e2e_setup_code(self) -> str:
        """Generate E2E test setup"""
        return """
// Setup test environment
beforeEach(() => {
    cy.visit('/');
    cy.clearCookies();
    cy.clearLocalStorage();
});
"""
    
    def _generate_e2e_test_code(self, workflow: Dict[str, Any]) -> str:
        """Generate E2E test code"""
        if workflow['name'] == 'crud_workflow':
            return """
// Create
cy.get('[data-testid="create-button"]').click();
cy.get('[data-testid="name-input"]').type('Test Item');
cy.get('[data-testid="submit-button"]').click();
cy.contains('Item created successfully').should('be.visible');

// Read
cy.get('[data-testid="items-list"]').should('contain', 'Test Item');

// Update
cy.get('[data-testid="edit-button"]').first().click();
cy.get('[data-testid="name-input"]').clear().type('Updated Item');
cy.get('[data-testid="submit-button"]').click();
cy.contains('Item updated successfully').should('be.visible');

// Delete
cy.get('[data-testid="delete-button"]').first().click();
cy.get('[data-testid="confirm-delete"]').click();
cy.contains('Item deleted successfully').should('be.visible');
"""
        
        return f"// E2E test for {workflow['name']}"


class PerformanceTestGenerator:
    """Generates performance and load tests"""
    
    def generate_performance_tests(self, code: str, analysis: Dict[str, Any], request: TestGenerationRequest) -> List[TestCase]:
        """Generate performance tests"""
        tests = []
        
        # Generate load tests for API endpoints
        if self._is_api_code(code):
            tests.extend(self._generate_load_tests(code, request))
        
        # Generate performance tests for functions
        for func in analysis['functions']:
            if self._should_performance_test(func):
                tests.append(self._generate_function_performance_test(func, request))
        
        return tests
    
    def _is_api_code(self, code: str) -> bool:
        """Check if code contains API endpoints"""
        return any(pattern in code for pattern in ['@app.route', 'FastAPI', 'Flask'])
    
    def _should_performance_test(self, func: Dict[str, Any]) -> bool:
        """Determine if function should have performance tests"""
        # Performance test functions that might be computationally expensive
        expensive_keywords = ['process', 'calculate', 'analyze', 'generate', 'transform']
        return any(keyword in func['name'].lower() for keyword in expensive_keywords)
    
    def _generate_load_tests(self, code: str, request: TestGenerationRequest) -> List[TestCase]:
        """Generate load tests for API endpoints"""
        tests = []
        
        endpoints = self._extract_api_endpoints(code)
        
        for endpoint in endpoints:
            tests.append(TestCase(
                id=str(uuid.uuid4()),
                name=f"test_load_{endpoint['name']}",
                description=f"Load test for {endpoint['path']} endpoint",
                test_type=TestType.LOAD,
                framework=TestFramework.PYTEST,  # Using pytest with locust
                setup_code=self._generate_load_test_setup(),
                test_code=self._generate_load_test_code(endpoint),
                tags=["performance", "load"],
                timeout=300  # 5 minutes for load tests
            ))
        
        return tests
    
    def _generate_function_performance_test(self, func: Dict[str, Any], request: TestGenerationRequest) -> TestCase:
        """Generate performance test for a function"""
        return TestCase(
            id=str(uuid.uuid4()),
            name=f"test_performance_{func['name']}",
            description=f"Performance test for {func['name']} function",
            test_type=TestType.PERFORMANCE,
            framework=request.framework,
            test_code=self._generate_function_perf_code(func, request.framework),
            assertions=[
                TestAssertion(
                    assertion_type="execution_time",
                    expected_value=1.0,
                    actual_expression="execution_time < 1.0",
                    description=f"Function {func['name']} should execute within 1 second"
                )
            ],
            tags=["performance", "timing"]
        )
    
    def _extract_api_endpoints(self, code: str) -> List[Dict[str, Any]]:
        """Extract API endpoints (reused from IntegrationTestGenerator)"""
        endpoints = []
        route_patterns = [r'@app\.route\([\'"]([^\'"]+)[\'"]']
        
        for pattern in route_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                path = match.group(1)
                endpoints.append({
                    'path': path,
                    'name': path.replace('/', '_').strip('_') or 'root'
                })
        
        return endpoints
    
    def _generate_load_test_setup(self) -> str:
        """Generate load test setup code"""
        return """
from locust import HttpUser, task, between
import time

class LoadTestUser(HttpUser):
    wait_time = between(1, 3)
"""
    
    def _generate_load_test_code(self, endpoint: Dict[str, Any]) -> str:
        """Generate load test code"""
        path = endpoint['path']
        return f"""
@task
def test_endpoint_load(self):
    start_time = time.time()
    response = self.client.get("{path}")
    end_time = time.time()
    
    assert response.status_code == 200
    assert (end_time - start_time) < 2.0  # Response time under 2 seconds
"""
    
    def _generate_function_perf_code(self, func: Dict[str, Any], framework: TestFramework) -> str:
        """Generate function performance test code"""
        func_name = func['name']
        
        if framework == TestFramework.PYTEST:
            return f"""
import time

def test_performance():
    # Arrange
    test_data = "test_input"
    
    # Act
    start_time = time.time()
    result = {func_name}(test_data)
    end_time = time.time()
    
    # Assert
    execution_time = end_time - start_time
    assert execution_time < 1.0, f"Function took {{execution_time}} seconds"
    assert result is not None
"""
        
        return f"# Performance test for {func_name}"


class TestValidator:
    """Validates generated tests for quality and completeness"""
    
    def validate_test_suite(self, test_suite: TestSuite) -> Dict[str, Any]:
        """Validate entire test suite"""
        result = {
            'complexity_score': 0.0,
            'maintainability_score': 0.0,
            'recommendations': [],
            'warnings': []
        }
        
        # Calculate complexity score
        result['complexity_score'] = self._calculate_complexity_score(test_suite)
        
        # Calculate maintainability score
        result['maintainability_score'] = self._calculate_maintainability_score(test_suite)
        
        # Generate recommendations
        result['recommendations'] = self._generate_recommendations(test_suite)
        
        # Generate warnings
        result['warnings'] = self._generate_warnings(test_suite)
        
        return result
    
    def _calculate_complexity_score(self, test_suite: TestSuite) -> float:
        """Calculate test complexity score (0-1, lower is better)"""
        if not test_suite.test_cases:
            return 0.0
        
        total_complexity = 0
        for test_case in test_suite.test_cases:
            # Simple complexity based on test code length and assertions
            code_complexity = len(test_case.test_code.split('\n')) / 50  # Normalize by 50 lines
            assertion_complexity = len(test_case.assertions) / 10  # Normalize by 10 assertions
            total_complexity += code_complexity + assertion_complexity
        
        return min(total_complexity / len(test_suite.test_cases), 1.0)
    
    def _calculate_maintainability_score(self, test_suite: TestSuite) -> float:
        """Calculate test maintainability score (0-1, higher is better)"""
        if not test_suite.test_cases:
            return 0.0
        
        maintainability_factors = []
        
        for test_case in test_suite.test_cases:
            # Good naming convention
            has_good_name = test_case.name.startswith('test_') and '_' in test_case.name
            
            # Has description
            has_description = bool(test_case.description)
            
            # Has proper tags
            has_tags = len(test_case.tags) > 0
            
            # Has assertions
            has_assertions = len(test_case.assertions) > 0
            
            # Calculate score for this test
            test_score = sum([has_good_name, has_description, has_tags, has_assertions]) / 4
            maintainability_factors.append(test_score)
        
        return sum(maintainability_factors) / len(maintainability_factors)
    
    def _generate_recommendations(self, test_suite: TestSuite) -> List[str]:
        """Generate recommendations for test improvement"""
        recommendations = []
        
        if test_suite.total_tests == 0:
            recommendations.append("Add test cases to the test suite")
        
        if test_suite.total_tests < 5:
            recommendations.append("Consider adding more comprehensive test coverage")
        
        # Check for missing test types
        test_types = {test.test_type for test in test_suite.test_cases}
        if TestType.UNIT not in test_types:
            recommendations.append("Add unit tests for better code coverage")
        
        if TestType.INTEGRATION not in test_types:
            recommendations.append("Add integration tests for component interactions")
        
        # Check for tests without assertions
        tests_without_assertions = [t for t in test_suite.test_cases if not t.assertions]
        if tests_without_assertions:
            recommendations.append(f"{len(tests_without_assertions)} tests lack proper assertions")
        
        return recommendations
    
    def _generate_warnings(self, test_suite: TestSuite) -> List[str]:
        """Generate warnings about potential issues"""
        warnings = []
        
        # Check for duplicate test names
        test_names = [test.name for test in test_suite.test_cases]
        duplicates = set([name for name in test_names if test_names.count(name) > 1])
        if duplicates:
            warnings.append(f"Duplicate test names found: {', '.join(duplicates)}")
        
        # Check for very long test methods
        long_tests = [t for t in test_suite.test_cases if len(t.test_code) > 1000]
        if long_tests:
            warnings.append(f"{len(long_tests)} tests are very long and may be hard to maintain")
        
        # Check for tests without proper setup
        tests_needing_setup = [t for t in test_suite.test_cases if t.test_type == TestType.INTEGRATION and not t.setup_code]
        if tests_needing_setup:
            warnings.append(f"{len(tests_needing_setup)} integration tests may need setup code")
        
        return warnings
    
    def _generate_component_integration_code(self, analysis: Dict[str, Any], framework: TestFramework) -> str:
        """Generate component integration test code"""
        classes = analysis['classes']
        
        if framework == TestFramework.PYTEST and len(classes) >= 2:
            class1, class2 = classes[0]['name'], classes[1]['name']
            return f"""
# Arrange
component1 = {class1}()
component2 = {class2}()

# Act
result = component1.interact_with(component2)

# Assert
assert result is not None
assert component1.state is not None
assert component2.state is not None
"""
        
        return "# Component integration test"