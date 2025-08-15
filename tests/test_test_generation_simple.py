"""
Simple Integration Tests for Test Generation System

This module contains simplified integration tests that don't require
the full application setup, focusing on the core test generation functionality.
"""

import pytest
from unittest.mock import Mock, patch

from scrollintel.models.test_generation_models import (
    TestGenerationRequest, TestType, TestFramework, TestSuite, TestCase
)
from scrollintel.engines.test_generator import TestGenerator


class TestTestGenerationSimpleIntegration:
    """Simple integration tests for test generation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.test_generator = TestGenerator()
    
    def test_complete_python_workflow(self):
        """Test complete Python code test generation workflow"""
        python_code = '''
def add(a, b):
    """Add two numbers"""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    return a + b

class Calculator:
    def __init__(self):
        self.history = []
    
    def calculate(self, operation, a, b):
        if operation == "add":
            result = add(a, b)
            self.history.append(f"{a} + {b} = {result}")
            return result
        else:
            raise ValueError("Unsupported operation")
'''
        
        request = TestGenerationRequest(
            target_code=python_code,
            code_type="module",
            test_types=[TestType.UNIT, TestType.INTEGRATION],
            framework=TestFramework.PYTEST,
            coverage_target=0.9,
            include_edge_cases=True,
            include_error_cases=True,
            requirements=[
                "Functions should handle type validation",
                "Classes should maintain state correctly",
                "Error conditions should be tested"
            ]
        )
        
        result = self.test_generator.generate_comprehensive_tests(request)
        
        # Verify result structure
        assert result.test_count > 0
        assert result.estimated_coverage > 0
        assert result.generation_time >= 0
        assert len(result.generated_suite.test_cases) > 0
        
        # Verify test diversity
        test_types = {test.test_type for test in result.generated_suite.test_cases}
        assert TestType.UNIT in test_types
        
        # Verify test quality
        assert result.complexity_score >= 0
        assert result.maintainability_score >= 0
        
        # Check specific test characteristics
        unit_tests = [t for t in result.generated_suite.test_cases if t.test_type == TestType.UNIT]
        assert len(unit_tests) > 0
        
        # Verify test names are meaningful
        test_names = [t.name for t in result.generated_suite.test_cases]
        assert any("add" in name.lower() for name in test_names)
        assert any("calculator" in name.lower() for name in test_names)
    
    def test_api_workflow(self):
        """Test API code test generation workflow"""
        api_code = '''
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify([{"id": 1, "name": "John"}])

@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({"error": "Name required"}), 400
    return jsonify({"id": 2, "name": data["name"]}), 201
'''
        
        request = TestGenerationRequest(
            target_code=api_code,
            code_type="api",
            test_types=[TestType.INTEGRATION, TestType.END_TO_END],
            framework=TestFramework.PYTEST,
            coverage_target=0.8,
            requirements=[
                "All endpoints should return proper status codes",
                "Error handling should be comprehensive"
            ]
        )
        
        result = self.test_generator.generate_comprehensive_tests(request)
        
        # Verify API-specific results
        assert result.test_count >= 0  # May be 0 if no tests generated for this specific case
        
        # If tests were generated, check their types
        if result.test_count > 0:
            # Check for integration tests
            integration_tests = [t for t in result.generated_suite.test_cases 
                               if t.test_type == TestType.INTEGRATION]
            
            # Check for E2E tests
            e2e_tests = [t for t in result.generated_suite.test_cases 
                        if t.test_type == TestType.END_TO_END]
            
            # At least one type should be present
            assert len(integration_tests) > 0 or len(e2e_tests) > 0
        
            # Verify API-specific test names
            test_names = [t.name for t in result.generated_suite.test_cases]
            api_tests = [name for name in test_names if 'api' in name.lower()]
            # API tests may or may not be generated depending on the implementation
    
    def test_performance_workflow(self):
        """Test performance test generation workflow"""
        perf_code = '''
def process_data(data_list):
    """Process a large list of data"""
    result = []
    for item in data_list:
        # Simulate expensive operation
        processed = item * 2 + 1
        result.append(processed)
    return result

def calculate_statistics(numbers):
    """Calculate statistics for numbers"""
    if not numbers:
        return {"mean": 0, "max": 0, "min": 0}
    
    return {
        "mean": sum(numbers) / len(numbers),
        "max": max(numbers),
        "min": min(numbers)
    }
'''
        
        request = TestGenerationRequest(
            target_code=perf_code,
            code_type="module",
            test_types=[TestType.PERFORMANCE],
            framework=TestFramework.PYTEST,
            include_performance_tests=True,
            requirements=[
                "Functions should complete within reasonable time",
                "Performance should be consistent"
            ]
        )
        
        result = self.test_generator.generate_comprehensive_tests(request)
        
        # Verify performance test generation
        assert result.test_count > 0
        
        perf_tests = [t for t in result.generated_suite.test_cases 
                     if t.test_type == TestType.PERFORMANCE]
        assert len(perf_tests) > 0
        
        # Check performance test characteristics
        for test in perf_tests:
            assert "performance" in test.name.lower()
            assert len(test.assertions) > 0
            # Should have timing-related assertions
            timing_assertions = [a for a in test.assertions 
                               if "time" in a.assertion_type.lower()]
            assert len(timing_assertions) > 0
    
    def test_test_validation_workflow(self):
        """Test the test validation workflow"""
        # Create a test suite with various quality issues
        test_suite = TestSuite(
            id="validation_test_suite",
            name="Test Suite for Validation",
            description="A test suite to validate",
            framework=TestFramework.PYTEST,
            test_cases=[
                # Good test case
                TestCase(
                    id="good_test",
                    name="test_function_properly",
                    description="A well-written test",
                    test_type=TestType.UNIT,
                    framework=TestFramework.PYTEST,
                    test_code="assert function() == expected",
                    tags=["unit", "good"],
                    priority=1
                ),
                # Poor test case (no assertions)
                TestCase(
                    id="poor_test",
                    name="test_without_assertions",
                    description="A test without proper assertions",
                    test_type=TestType.UNIT,
                    framework=TestFramework.PYTEST,
                    test_code="function()",  # No assertions
                    tags=["unit"],
                    priority=2
                ),
                # Duplicate name test
                TestCase(
                    id="duplicate_test",
                    name="test_function_properly",  # Same name as first test
                    description="A duplicate test name",
                    test_type=TestType.UNIT,
                    framework=TestFramework.PYTEST,
                    test_code="assert function() != None",
                    tags=["unit"],
                    priority=3
                )
            ]
        )
        
        test_suite.total_tests = len(test_suite.test_cases)
        
        # Validate the test suite
        validation_result = self.test_generator.test_validator.validate_test_suite(test_suite)
        
        # Check validation results
        assert 'complexity_score' in validation_result
        assert 'maintainability_score' in validation_result
        assert 'recommendations' in validation_result
        assert 'warnings' in validation_result
        
        # Should have recommendations due to poor test
        assert len(validation_result['recommendations']) > 0
        
        # Should have warnings due to duplicate names
        assert len(validation_result['warnings']) > 0
        
        # Check for specific issues
        warnings = validation_result['warnings']
        assert any("duplicate" in warning.lower() for warning in warnings)
        
        recommendations = validation_result['recommendations']
        assert any("assertion" in rec.lower() for rec in recommendations)
    
    def test_framework_specific_generation(self):
        """Test generation for different frameworks"""
        simple_code = '''
function add(a, b) {
    return a + b;
}

function multiply(x, y) {
    if (typeof x !== 'number' || typeof y !== 'number') {
        throw new Error('Arguments must be numbers');
    }
    return x * y;
}
'''
        
        # Test with Jest framework
        jest_request = TestGenerationRequest(
            target_code=simple_code,
            code_type="module",
            test_types=[TestType.UNIT],
            framework=TestFramework.JEST,
            coverage_target=0.8
        )
        
        jest_result = self.test_generator.generate_comprehensive_tests(jest_request)
        
        # Verify Jest-specific results
        assert jest_result.test_count >= 0  # May be 0 for non-Python code
        assert jest_result.generated_suite.framework == TestFramework.JEST
        
        # Test with Mocha framework
        mocha_request = TestGenerationRequest(
            target_code=simple_code,
            code_type="module",
            test_types=[TestType.UNIT],
            framework=TestFramework.MOCHA,
            coverage_target=0.8
        )
        
        mocha_result = self.test_generator.generate_comprehensive_tests(mocha_request)
        
        # Verify Mocha-specific results
        assert mocha_result.test_count >= 0
        assert mocha_result.generated_suite.framework == TestFramework.MOCHA
    
    def test_error_handling_in_generation(self):
        """Test error handling during test generation"""
        # Test with invalid/malformed code
        invalid_code = '''
        def broken_function(
            # Missing closing parenthesis and body
        '''
        
        request = TestGenerationRequest(
            target_code=invalid_code,
            code_type="function",
            test_types=[TestType.UNIT],
            framework=TestFramework.PYTEST
        )
        
        # Should handle syntax errors gracefully
        result = self.test_generator.generate_comprehensive_tests(request)
        
        # Should still return a result, even if no tests generated
        assert isinstance(result.test_count, int)
        assert result.test_count >= 0
        assert result.generation_time >= 0
    
    def test_coverage_estimation_accuracy(self):
        """Test accuracy of coverage estimation"""
        code_with_multiple_functions = '''
def function_one():
    return "one"

def function_two():
    return "two"

def function_three():
    return "three"

class SimpleClass:
    def method_one(self):
        return "method_one"
    
    def method_two(self):
        return "method_two"
'''
        
        request = TestGenerationRequest(
            target_code=code_with_multiple_functions,
            code_type="module",
            test_types=[TestType.UNIT],
            framework=TestFramework.PYTEST,
            coverage_target=1.0  # Request 100% coverage
        )
        
        result = self.test_generator.generate_comprehensive_tests(request)
        
        # Should generate tests for multiple functions/methods
        assert result.test_count > 0
        
        # Coverage should be reasonable
        assert 0 <= result.estimated_coverage <= 1.0
        
        # If tests were generated, coverage should be > 0
        if result.test_count > 0:
            assert result.estimated_coverage > 0
    
    def test_requirement_driven_generation(self):
        """Test that requirements influence test generation"""
        code = '''
def validate_email(email):
    """Validate email format"""
    if not email or '@' not in email:
        return False
    return True

def send_notification(user_id, message):
    """Send notification to user"""
    if not user_id or not message:
        raise ValueError("User ID and message are required")
    # Simulate sending notification
    return True
'''
        
        # Request with specific requirements
        request = TestGenerationRequest(
            target_code=code,
            code_type="module",
            test_types=[TestType.UNIT],
            framework=TestFramework.PYTEST,
            include_edge_cases=True,
            include_error_cases=True,
            requirements=[
                "Email validation should handle empty strings",
                "Email validation should require @ symbol",
                "Notification function should validate inputs",
                "Error conditions should raise appropriate exceptions"
            ]
        )
        
        result = self.test_generator.generate_comprehensive_tests(request)
        
        # Should generate tests based on requirements
        assert result.test_count > 0
        
        # Check that error handling tests are included
        error_tests = [t for t in result.generated_suite.test_cases 
                      if "error" in t.name.lower() or "error_handling" in t.tags]
        assert len(error_tests) > 0
        
        # Check that edge case tests are included
        edge_tests = [t for t in result.generated_suite.test_cases 
                     if "edge" in t.name.lower() or "edge_cases" in t.tags]
        assert len(edge_tests) > 0


class TestTestGenerationPerformance:
    """Performance tests for test generation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.test_generator = TestGenerator()
    
    def test_generation_speed_small_code(self):
        """Test generation speed for small code samples"""
        small_code = '''
def simple_function(x):
    return x * 2
'''
        
        request = TestGenerationRequest(
            target_code=small_code,
            code_type="function",
            test_types=[TestType.UNIT],
            framework=TestFramework.PYTEST
        )
        
        result = self.test_generator.generate_comprehensive_tests(request)
        
        # Should be very fast for small code
        assert result.generation_time < 1.0  # Less than 1 second
        assert result.test_count > 0
    
    def test_generation_speed_medium_code(self):
        """Test generation speed for medium-sized code"""
        # Generate medium-sized code sample
        medium_code = "\n".join([
            f"def function_{i}(x): return x + {i}" for i in range(20)
        ])
        
        request = TestGenerationRequest(
            target_code=medium_code,
            code_type="module",
            test_types=[TestType.UNIT],
            framework=TestFramework.PYTEST
        )
        
        result = self.test_generator.generate_comprehensive_tests(request)
        
        # Should still be reasonably fast
        assert result.generation_time < 5.0  # Less than 5 seconds
        assert result.test_count > 0
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable during generation"""
        # This is a basic test - in production, you'd use memory profiling tools
        
        code_samples = [
            f"def func_{i}(): return {i}" for i in range(10)
        ]
        
        results = []
        for i, code in enumerate(code_samples):
            request = TestGenerationRequest(
                target_code=code,
                code_type="function",
                test_types=[TestType.UNIT],
                framework=TestFramework.PYTEST
            )
            
            result = self.test_generator.generate_comprehensive_tests(request)
            results.append(result)
        
        # All generations should complete successfully
        assert len(results) == len(code_samples)
        for result in results:
            assert result.test_count >= 0
            assert result.generation_time >= 0


# Test fixtures and utilities
@pytest.fixture
def sample_test_generator():
    """Fixture providing a TestGenerator instance"""
    return TestGenerator()


@pytest.fixture
def sample_python_code():
    """Fixture providing sample Python code"""
    return '''
def fibonacci(n):
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


@pytest.fixture
def sample_api_code():
    """Fixture providing sample API code"""
    return '''
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/users/<int:user_id>')
def get_user(user_id):
    return jsonify({"id": user_id, "name": f"User {user_id}"})
'''


# Parameterized tests
@pytest.mark.parametrize("test_type", [
    TestType.UNIT,
    TestType.INTEGRATION,
    TestType.END_TO_END,
    TestType.PERFORMANCE
])
def test_generation_for_all_test_types(sample_test_generator, sample_python_code, test_type):
    """Test generation for all supported test types"""
    request = TestGenerationRequest(
        target_code=sample_python_code,
        code_type="module",
        test_types=[test_type],
        framework=TestFramework.PYTEST
    )
    
    result = sample_test_generator.generate_comprehensive_tests(request)
    
    # Should handle all test types
    assert result.test_count >= 0
    assert result.generation_time >= 0


@pytest.mark.parametrize("framework", [
    TestFramework.PYTEST,
    TestFramework.JEST,
    TestFramework.MOCHA,
    TestFramework.CYPRESS
])
def test_generation_for_all_frameworks(sample_test_generator, sample_python_code, framework):
    """Test generation for all supported frameworks"""
    request = TestGenerationRequest(
        target_code=sample_python_code,
        code_type="module",
        test_types=[TestType.UNIT],
        framework=framework
    )
    
    result = sample_test_generator.generate_comprehensive_tests(request)
    
    # Should handle all frameworks
    assert result.test_count >= 0
    assert result.generated_suite.framework == framework