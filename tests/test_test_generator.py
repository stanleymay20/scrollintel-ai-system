"""
Tests for Automated Test Generation System

This module contains comprehensive tests for the test generation engine,
including unit tests, integration tests, and validation tests.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from scrollintel.models.test_generation_models import (
    TestSuite, TestCase, TestAssertion, TestType, TestFramework,
    TestGenerationRequest, TestGenerationResult
)
from scrollintel.engines.test_generator import (
    TestGenerator, UnitTestGenerator, IntegrationTestGenerator,
    EndToEndTestGenerator, PerformanceTestGenerator, TestValidator
)


class TestTestGenerator:
    """Test the main TestGenerator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.test_generator = TestGenerator()
        self.sample_python_code = '''
def calculate_sum(a, b):
    """Calculate sum of two numbers"""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("Arguments must be numbers")
    return a + b

class Calculator:
    """Simple calculator class"""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def get_history(self):
        return self.history
'''
        
        self.sample_api_code = '''
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify([{"id": 1, "name": "John"}])

@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json()
    return jsonify({"id": 2, "name": data["name"]}), 201
'''
    
    def test_generate_comprehensive_tests_python_code(self):
        """Test generating comprehensive tests for Python code"""
        request = TestGenerationRequest(
            target_code=self.sample_python_code,
            code_type="module",
            test_types=[TestType.UNIT, TestType.INTEGRATION],
            framework=TestFramework.PYTEST,
            coverage_target=0.8,
            include_edge_cases=True,
            include_error_cases=True
        )
        
        result = self.test_generator.generate_comprehensive_tests(request)
        
        assert isinstance(result, TestGenerationResult)
        assert result.test_count > 0
        assert result.estimated_coverage > 0
        assert result.generation_time > 0
        assert len(result.generated_suite.test_cases) > 0
        
        # Check that different test types were generated
        test_types = {test.test_type for test in result.generated_suite.test_cases}
        assert TestType.UNIT in test_types
    
    def test_generate_comprehensive_tests_api_code(self):
        """Test generating comprehensive tests for API code"""
        request = TestGenerationRequest(
            target_code=self.sample_api_code,
            code_type="api",
            test_types=[TestType.UNIT, TestType.INTEGRATION],
            framework=TestFramework.PYTEST,
            coverage_target=0.8
        )
        
        result = self.test_generator.generate_comprehensive_tests(request)
        
        assert isinstance(result, TestGenerationResult)
        assert result.test_count > 0
        
        # Check for API-specific tests
        test_names = [test.name for test in result.generated_suite.test_cases]
        api_tests = [name for name in test_names if 'api' in name.lower()]
        assert len(api_tests) > 0
    
    def test_analyze_code_python(self):
        """Test code analysis for Python code"""
        analysis = self.test_generator._analyze_code(self.sample_python_code, "module")
        
        assert analysis['type'] == 'module'
        assert len(analysis['functions']) == 1
        assert len(analysis['classes']) == 1
        assert analysis['functions'][0]['name'] == 'calculate_sum'
        assert analysis['classes'][0]['name'] == 'Calculator'
        assert 'add' in analysis['classes'][0]['methods']
        assert analysis['complexity'] > 1
    
    def test_analyze_code_api(self):
        """Test code analysis for API code"""
        analysis = self.test_generator._analyze_code(self.sample_api_code, "api")
        
        assert analysis['type'] == 'api'
        assert 'Flask' in analysis['imports']
    
    def test_estimate_coverage(self):
        """Test coverage estimation"""
        test_suite = TestSuite(
            id="test_suite_1",
            name="Test Suite",
            description="Test suite",
            framework=TestFramework.PYTEST,
            test_cases=[
                TestCase(
                    id="test_1",
                    name="test_calculate_sum_happy_path",
                    description="Test calculate_sum function",
                    test_type=TestType.UNIT,
                    framework=TestFramework.PYTEST,
                    test_code="assert calculate_sum(1, 2) == 3"
                )
            ]
        )
        
        analysis = {
            'functions': [{'name': 'calculate_sum'}],
            'classes': []
        }
        
        coverage = self.test_generator._estimate_coverage(test_suite, analysis)
        assert coverage == 1.0  # 100% coverage for single function


class TestUnitTestGenerator:
    """Test the UnitTestGenerator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.unit_generator = UnitTestGenerator()
        self.sample_function = {
            'name': 'add_numbers',
            'args': ['a', 'b'],
            'returns': 'int',
            'docstring': 'Add two numbers',
            'line_number': 1
        }
        self.sample_class = {
            'name': 'Calculator',
            'methods': ['add', 'subtract', 'multiply'],
            'docstring': 'Calculator class',
            'line_number': 10
        }
    
    def test_generate_unit_tests(self):
        """Test generating unit tests"""
        analysis = {
            'functions': [self.sample_function],
            'classes': [self.sample_class]
        }
        
        request = TestGenerationRequest(
            target_code="dummy_code",
            code_type="module",
            test_types=[TestType.UNIT],
            framework=TestFramework.PYTEST,
            include_edge_cases=True,
            include_error_cases=True
        )
        
        tests = self.unit_generator.generate_unit_tests("dummy_code", analysis, request)
        
        assert len(tests) > 0
        
        # Check function tests
        function_tests = [t for t in tests if 'add_numbers' in t.name]
        assert len(function_tests) >= 1
        
        # Check class tests
        class_tests = [t for t in tests if 'calculator' in t.name.lower()]
        assert len(class_tests) >= 1
    
    def test_generate_function_tests(self):
        """Test generating tests for a specific function"""
        request = TestGenerationRequest(
            target_code="dummy_code",
            code_type="function",
            test_types=[TestType.UNIT],
            framework=TestFramework.PYTEST,
            include_edge_cases=True,
            include_error_cases=True
        )
        
        tests = self.unit_generator._generate_function_tests(self.sample_function, request)
        
        assert len(tests) >= 1
        
        # Check test names
        test_names = [test.name for test in tests]
        assert any('happy_path' in name for name in test_names)
        assert any('edge_cases' in name for name in test_names)
        assert any('error_handling' in name for name in test_names)
        
        # Check test properties
        for test in tests:
            assert test.test_type == TestType.UNIT
            assert test.framework == TestFramework.PYTEST
            assert test.test_code is not None
            assert len(test.test_code.strip()) > 0
    
    def test_generate_class_tests(self):
        """Test generating tests for a class"""
        request = TestGenerationRequest(
            target_code="dummy_code",
            code_type="class",
            test_types=[TestType.UNIT],
            framework=TestFramework.PYTEST
        )
        
        tests = self.unit_generator._generate_class_tests(self.sample_class, request)
        
        assert len(tests) >= 1
        
        # Check for initialization test
        init_tests = [t for t in tests if 'initialization' in t.name]
        assert len(init_tests) == 1
        
        # Check for method tests
        method_tests = [t for t in tests if any(method in t.name for method in self.sample_class['methods'])]
        assert len(method_tests) >= 1
    
    def test_generate_test_code_pytest(self):
        """Test generating pytest-specific test code"""
        test_code = self.unit_generator._generate_function_test_code(
            self.sample_function, "happy_path", TestFramework.PYTEST
        )
        
        assert "# Arrange" in test_code
        assert "# Act" in test_code
        assert "# Assert" in test_code
        assert "assert" in test_code
        assert self.sample_function['name'] in test_code


class TestIntegrationTestGenerator:
    """Test the IntegrationTestGenerator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.integration_generator = IntegrationTestGenerator()
        self.api_code = '''
@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify([])

@app.route('/api/users', methods=['POST'])
def create_user():
    return jsonify({}), 201
'''
    
    def test_generate_integration_tests(self):
        """Test generating integration tests"""
        analysis = {'functions': [], 'classes': []}
        request = TestGenerationRequest(
            target_code=self.api_code,
            code_type="api",
            test_types=[TestType.INTEGRATION],
            framework=TestFramework.PYTEST
        )
        
        tests = self.integration_generator.generate_integration_tests(
            self.api_code, analysis, request
        )
        
        assert len(tests) > 0
        
        # Check for API tests
        api_tests = [t for t in tests if t.test_type == TestType.INTEGRATION]
        assert len(api_tests) > 0
    
    def test_is_api_code(self):
        """Test API code detection"""
        assert self.integration_generator._is_api_code(self.api_code) == True
        assert self.integration_generator._is_api_code("def regular_function(): pass") == False
    
    def test_extract_api_endpoints(self):
        """Test API endpoint extraction"""
        endpoints = self.integration_generator._extract_api_endpoints(self.api_code)
        
        assert len(endpoints) >= 1
        assert any(endpoint['path'] == '/api/users' for endpoint in endpoints)
    
    def test_generate_api_integration_tests(self):
        """Test generating API integration tests"""
        analysis = {}
        request = TestGenerationRequest(
            target_code=self.api_code,
            code_type="api",
            test_types=[TestType.INTEGRATION],
            framework=TestFramework.PYTEST
        )
        
        tests = self.integration_generator._generate_api_integration_tests(
            self.api_code, analysis, request
        )
        
        assert len(tests) > 0
        
        # Check test properties
        for test in tests:
            assert test.test_type == TestType.INTEGRATION
            assert 'api' in test.name
            assert len(test.assertions) > 0


class TestEndToEndTestGenerator:
    """Test the EndToEndTestGenerator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.e2e_generator = EndToEndTestGenerator()
        self.crud_code = '''
@app.route('/api/items', methods=['GET', 'POST'])
def items():
    pass

@app.route('/api/items/<id>', methods=['PUT', 'DELETE'])
def item_detail():
    pass
'''
    
    def test_generate_e2e_tests(self):
        """Test generating end-to-end tests"""
        analysis = {}
        request = TestGenerationRequest(
            target_code=self.crud_code,
            code_type="api",
            test_types=[TestType.END_TO_END],
            framework=TestFramework.CYPRESS
        )
        
        tests = self.e2e_generator.generate_e2e_tests(self.crud_code, analysis, request)
        
        assert len(tests) > 0
        
        # Check test properties
        for test in tests:
            assert test.test_type == TestType.END_TO_END
            assert test.framework == TestFramework.CYPRESS
            assert test.timeout is not None
    
    def test_identify_user_workflows(self):
        """Test user workflow identification"""
        analysis = {}
        workflows = self.e2e_generator._identify_user_workflows(self.crud_code, analysis)
        
        assert len(workflows) > 0
        
        # Check for CRUD workflow
        crud_workflows = [w for w in workflows if w['name'] == 'crud_workflow']
        assert len(crud_workflows) > 0
    
    def test_is_crud_api(self):
        """Test CRUD API detection"""
        assert self.e2e_generator._is_crud_api(self.crud_code) == True
        assert self.e2e_generator._is_crud_api("def simple_function(): pass") == False
    
    def test_has_authentication(self):
        """Test authentication detection"""
        auth_code = "def login(username, password): pass"
        assert self.e2e_generator._has_authentication(auth_code) == True
        assert self.e2e_generator._has_authentication("def calculate(): pass") == False


class TestPerformanceTestGenerator:
    """Test the PerformanceTestGenerator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.perf_generator = PerformanceTestGenerator()
        self.api_code = '''
@app.route('/api/process', methods=['POST'])
def process_data():
    # Expensive operation
    pass
'''
        self.function_analysis = {
            'name': 'process_large_dataset',
            'args': ['data'],
            'returns': 'dict'
        }
    
    def test_generate_performance_tests(self):
        """Test generating performance tests"""
        analysis = {'functions': [self.function_analysis]}
        request = TestGenerationRequest(
            target_code=self.api_code,
            code_type="api",
            test_types=[TestType.PERFORMANCE],
            framework=TestFramework.PYTEST,
            include_performance_tests=True
        )
        
        tests = self.perf_generator.generate_performance_tests(self.api_code, analysis, request)
        
        assert len(tests) > 0
        
        # Check test properties
        perf_tests = [t for t in tests if t.test_type in [TestType.PERFORMANCE, TestType.LOAD]]
        assert len(perf_tests) > 0
    
    def test_should_performance_test(self):
        """Test performance test necessity detection"""
        expensive_func = {'name': 'process_large_data'}
        simple_func = {'name': 'get_name'}
        
        assert self.perf_generator._should_performance_test(expensive_func) == True
        assert self.perf_generator._should_performance_test(simple_func) == False
    
    def test_generate_function_performance_test(self):
        """Test generating function performance test"""
        request = TestGenerationRequest(
            target_code="dummy",
            code_type="function",
            test_types=[TestType.PERFORMANCE],
            framework=TestFramework.PYTEST
        )
        
        test = self.perf_generator._generate_function_performance_test(self.function_analysis, request)
        
        assert test.test_type == TestType.PERFORMANCE
        assert 'performance' in test.name
        assert len(test.assertions) > 0
        assert any('execution_time' in assertion.assertion_type for assertion in test.assertions)


class TestTestValidator:
    """Test the TestValidator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = TestValidator()
        self.sample_test_suite = TestSuite(
            id="suite_1",
            name="Sample Test Suite",
            description="A sample test suite",
            framework=TestFramework.PYTEST,
            test_cases=[
                TestCase(
                    id="test_1",
                    name="test_function_happy_path",
                    description="Test function with valid input",
                    test_type=TestType.UNIT,
                    framework=TestFramework.PYTEST,
                    test_code="assert function() == expected",
                    assertions=[
                        TestAssertion(
                            assertion_type="equals",
                            expected_value="expected",
                            actual_expression="function()",
                            description="Function should return expected value"
                        )
                    ],
                    tags=["unit", "happy_path"]
                ),
                TestCase(
                    id="test_2",
                    name="test_function_edge_case",
                    description="Test function with edge case",
                    test_type=TestType.UNIT,
                    framework=TestFramework.PYTEST,
                    test_code="assert function(None) is None",
                    tags=["unit", "edge_case"]
                )
            ]
        )
    
    def test_validate_test_suite(self):
        """Test test suite validation"""
        result = self.validator.validate_test_suite(self.sample_test_suite)
        
        assert 'complexity_score' in result
        assert 'maintainability_score' in result
        assert 'recommendations' in result
        assert 'warnings' in result
        
        assert 0 <= result['complexity_score'] <= 1
        assert 0 <= result['maintainability_score'] <= 1
        assert isinstance(result['recommendations'], list)
        assert isinstance(result['warnings'], list)
    
    def test_calculate_complexity_score(self):
        """Test complexity score calculation"""
        score = self.validator._calculate_complexity_score(self.sample_test_suite)
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
    
    def test_calculate_maintainability_score(self):
        """Test maintainability score calculation"""
        score = self.validator._calculate_maintainability_score(self.sample_test_suite)
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        recommendations = self.validator._generate_recommendations(self.sample_test_suite)
        
        assert isinstance(recommendations, list)
        # Should not recommend adding tests since we have some
        assert not any('Add test cases' in rec for rec in recommendations)
    
    def test_generate_warnings(self):
        """Test warning generation"""
        warnings = self.validator._generate_warnings(self.sample_test_suite)
        
        assert isinstance(warnings, list)
    
    def test_validate_empty_test_suite(self):
        """Test validation of empty test suite"""
        empty_suite = TestSuite(
            id="empty_suite",
            name="Empty Suite",
            description="Empty test suite",
            framework=TestFramework.PYTEST,
            test_cases=[]
        )
        
        result = self.validator.validate_test_suite(empty_suite)
        
        # Should have recommendations for empty suite
        assert len(result['recommendations']) > 0
        assert any('Add test cases' in rec for rec in result['recommendations'])


class TestTestGenerationIntegration:
    """Integration tests for the complete test generation system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.test_generator = TestGenerator()
    
    def test_full_test_generation_workflow(self):
        """Test complete test generation workflow"""
        sample_code = '''
def fibonacci(n):
    """Calculate fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class MathUtils:
    @staticmethod
    def factorial(n):
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n-1)
'''
        
        request = TestGenerationRequest(
            target_code=sample_code,
            code_type="module",
            test_types=[TestType.UNIT, TestType.INTEGRATION],
            framework=TestFramework.PYTEST,
            coverage_target=0.9,
            include_edge_cases=True,
            include_error_cases=True,
            requirements=["Function should handle edge cases", "Class methods should be tested"]
        )
        
        result = self.test_generator.generate_comprehensive_tests(request)
        
        # Validate result structure
        assert isinstance(result, TestGenerationResult)
        assert result.test_count > 0
        assert result.estimated_coverage > 0
        assert result.generation_time > 0
        
        # Validate test suite
        test_suite = result.generated_suite
        assert isinstance(test_suite, TestSuite)
        assert len(test_suite.test_cases) > 0
        
        # Check test diversity
        test_types = {test.test_type for test in test_suite.test_cases}
        assert TestType.UNIT in test_types
        
        # Check test quality
        assert result.complexity_score >= 0
        assert result.maintainability_score >= 0
        
        # Validate individual tests
        for test_case in test_suite.test_cases:
            assert test_case.id is not None
            assert test_case.name is not None
            assert test_case.test_code is not None
            assert test_case.framework == TestFramework.PYTEST
    
    def test_api_test_generation_workflow(self):
        """Test API-specific test generation workflow"""
        api_code = '''
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/api/users', methods=['GET', 'POST'])
def users():
    if request.method == 'GET':
        return jsonify([])
    else:
        data = request.get_json()
        return jsonify({"id": 1, "name": data["name"]}), 201
'''
        
        request = TestGenerationRequest(
            target_code=api_code,
            code_type="api",
            test_types=[TestType.UNIT, TestType.INTEGRATION, TestType.END_TO_END],
            framework=TestFramework.PYTEST,
            coverage_target=0.8
        )
        
        result = self.test_generator.generate_comprehensive_tests(request)
        
        # Should generate API-specific tests
        assert result.test_count > 0
        
        # Check for API integration tests
        integration_tests = [t for t in result.generated_suite.test_cases 
                           if t.test_type == TestType.INTEGRATION]
        assert len(integration_tests) > 0
        
        # Check for API endpoint tests
        api_test_names = [t.name for t in result.generated_suite.test_cases]
        assert any('api' in name.lower() for name in api_test_names)


# Fixtures for pytest
@pytest.fixture
def sample_test_generator():
    """Fixture providing a TestGenerator instance"""
    return TestGenerator()


@pytest.fixture
def sample_test_request():
    """Fixture providing a sample test generation request"""
    return TestGenerationRequest(
        target_code="def sample_function(): return True",
        code_type="function",
        test_types=[TestType.UNIT],
        framework=TestFramework.PYTEST,
        coverage_target=0.8
    )


# Performance tests
class TestTestGenerationPerformance:
    """Performance tests for test generation"""
    
    def test_generation_performance_large_code(self):
        """Test generation performance with large code base"""
        # Generate a large code sample
        large_code = "\n".join([
            f"def function_{i}(x): return x * {i}" for i in range(100)
        ])
        
        request = TestGenerationRequest(
            target_code=large_code,
            code_type="module",
            test_types=[TestType.UNIT],
            framework=TestFramework.PYTEST
        )
        
        test_generator = TestGenerator()
        result = test_generator.generate_comprehensive_tests(request)
        
        # Should complete in reasonable time (less than 10 seconds)
        assert result.generation_time < 10.0
        assert result.test_count > 0
    
    def test_memory_usage_large_test_suite(self):
        """Test memory usage with large test suites"""
        # This would typically use memory profiling tools
        # For now, just ensure we can generate many tests
        request = TestGenerationRequest(
            target_code="def test_func(): pass\n" * 50,
            code_type="module",
            test_types=[TestType.UNIT],
            framework=TestFramework.PYTEST
        )
        
        test_generator = TestGenerator()
        result = test_generator.generate_comprehensive_tests(request)
        
        # Should handle large test suites
        assert result.test_count > 0
        assert len(result.generated_suite.test_cases) > 0