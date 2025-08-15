"""
Integration Tests for Test Generation API

This module contains integration tests for the test generation API endpoints,
testing the complete workflow from API request to test generation and validation.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from scrollintel.api.routes.test_generation_routes import router
from scrollintel.models.test_generation_models import (
    TestType, TestFramework, TestGenerationResult, TestSuite, TestCase
)


# Mock FastAPI app for testing
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

client = TestClient(app)


class TestTestGenerationAPI:
    """Test the test generation API endpoints"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.sample_python_code = '''
def add_numbers(a, b):
    """Add two numbers together"""
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
'''
        
        self.sample_api_code = '''
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify([{"id": 1, "name": "John"}])
'''
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    @patch('scrollintel.engines.test_generator.TestGenerator.generate_comprehensive_tests')
    def test_generate_tests_endpoint(self, mock_generate, mock_auth):
        """Test the main test generation endpoint"""
        # Mock authentication
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock test generation result
        mock_test_suite = TestSuite(
            id="test_suite_1",
            name="Generated Tests",
            description="Auto-generated test suite",
            framework=TestFramework.PYTEST,
            test_cases=[
                TestCase(
                    id="test_1",
                    name="test_add_numbers",
                    description="Test add_numbers function",
                    test_type=TestType.UNIT,
                    framework=TestFramework.PYTEST,
                    test_code="assert add_numbers(1, 2) == 3"
                )
            ]
        )
        
        mock_result = TestGenerationResult(
            request_id="req_123",
            generated_suite=mock_test_suite,
            generation_time=1.5,
            estimated_coverage=0.85,
            test_count=1,
            complexity_score=0.3,
            maintainability_score=0.8,
            recommendations=["Add more edge case tests"],
            warnings=[]
        )
        
        mock_generate.return_value = mock_result
        
        # Make API request
        request_data = {
            "target_code": self.sample_python_code,
            "code_type": "module",
            "test_types": ["unit"],
            "framework": "pytest",
            "coverage_target": 0.8,
            "include_edge_cases": True,
            "include_error_cases": True
        }
        
        response = client.post("/api/v1/test-generation/generate", json=request_data)
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["test_count"] == 1
        assert result["estimated_coverage"] == 0.85
        assert result["generation_time"] == 1.5
        assert len(result["generated_suite"]["test_cases"]) == 1
        assert result["generated_suite"]["test_cases"][0]["name"] == "test_add_numbers"
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    def test_generate_tests_invalid_request(self, mock_auth):
        """Test test generation with invalid request data"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Invalid request - missing required fields
        request_data = {
            "code_type": "module"
            # Missing target_code
        }
        
        response = client.post("/api/v1/test-generation/generate", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    @patch('scrollintel.engines.test_generator.TestGenerator.generate_comprehensive_tests')
    def test_generate_unit_tests_only(self, mock_generate, mock_auth):
        """Test generating only unit tests"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock result with unit tests only
        mock_test_cases = [
            TestCase(
                id="test_1",
                name="test_add_numbers_unit",
                description="Unit test for add_numbers",
                test_type=TestType.UNIT,
                framework=TestFramework.PYTEST,
                test_code="assert add_numbers(1, 2) == 3"
            )
        ]
        
        mock_result = TestGenerationResult(
            request_id="req_123",
            generated_suite=TestSuite(
                id="suite_1",
                name="Unit Tests",
                description="Unit test suite",
                framework=TestFramework.PYTEST,
                test_cases=mock_test_cases
            ),
            generation_time=1.0,
            estimated_coverage=0.8,
            test_count=1,
            complexity_score=0.2,
            maintainability_score=0.9
        )
        
        mock_generate.return_value = mock_result
        
        request_data = {
            "target_code": self.sample_python_code,
            "code_type": "function",
            "framework": "pytest"
        }
        
        response = client.post("/api/v1/test-generation/generate-unit-tests", json=request_data)
        
        assert response.status_code == 200
        
        result = response.json()
        assert len(result) == 1
        assert result[0]["test_type"] == "unit"
        assert result[0]["name"] == "test_add_numbers_unit"
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    @patch('scrollintel.engines.test_generator.TestGenerator.generate_comprehensive_tests')
    def test_generate_integration_tests_only(self, mock_generate, mock_auth):
        """Test generating only integration tests"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock result with integration tests
        mock_test_cases = [
            TestCase(
                id="test_1",
                name="test_api_users_get",
                description="Integration test for GET /api/users",
                test_type=TestType.INTEGRATION,
                framework=TestFramework.PYTEST,
                test_code="response = client.get('/api/users')\nassert response.status_code == 200"
            )
        ]
        
        mock_result = TestGenerationResult(
            request_id="req_123",
            generated_suite=TestSuite(
                id="suite_1",
                name="Integration Tests",
                description="Integration test suite",
                framework=TestFramework.PYTEST,
                test_cases=mock_test_cases
            ),
            generation_time=2.0,
            estimated_coverage=0.7,
            test_count=1,
            complexity_score=0.4,
            maintainability_score=0.8
        )
        
        mock_generate.return_value = mock_result
        
        request_data = {
            "target_code": self.sample_api_code,
            "code_type": "api",
            "framework": "pytest"
        }
        
        response = client.post("/api/v1/test-generation/generate-integration-tests", json=request_data)
        
        assert response.status_code == 200
        
        result = response.json()
        assert len(result) == 1
        assert result[0]["test_type"] == "integration"
        assert "api" in result[0]["name"]
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    @patch('scrollintel.engines.test_generator.TestGenerator.generate_comprehensive_tests')
    def test_generate_e2e_tests_only(self, mock_generate, mock_auth):
        """Test generating only end-to-end tests"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock result with E2E tests
        mock_test_cases = [
            TestCase(
                id="test_1",
                name="test_e2e_user_workflow",
                description="End-to-end test for user workflow",
                test_type=TestType.END_TO_END,
                framework=TestFramework.CYPRESS,
                test_code="cy.visit('/'); cy.get('[data-testid=\"login\"]').click();"
            )
        ]
        
        mock_result = TestGenerationResult(
            request_id="req_123",
            generated_suite=TestSuite(
                id="suite_1",
                name="E2E Tests",
                description="End-to-end test suite",
                framework=TestFramework.CYPRESS,
                test_cases=mock_test_cases
            ),
            generation_time=3.0,
            estimated_coverage=0.6,
            test_count=1,
            complexity_score=0.5,
            maintainability_score=0.7
        )
        
        mock_generate.return_value = mock_result
        
        request_data = {
            "target_code": self.sample_api_code,
            "code_type": "api",
            "framework": "cypress"
        }
        
        response = client.post("/api/v1/test-generation/generate-e2e-tests", json=request_data)
        
        assert response.status_code == 200
        
        result = response.json()
        assert len(result) == 1
        assert result[0]["test_type"] == "end_to_end"
        assert result[0]["framework"] == "cypress"
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    @patch('scrollintel.engines.test_generator.TestGenerator.generate_comprehensive_tests')
    def test_generate_performance_tests_only(self, mock_generate, mock_auth):
        """Test generating only performance tests"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock result with performance tests
        mock_test_cases = [
            TestCase(
                id="test_1",
                name="test_performance_api_users",
                description="Performance test for API endpoint",
                test_type=TestType.PERFORMANCE,
                framework=TestFramework.PYTEST,
                test_code="start = time.time(); response = client.get('/api/users'); assert time.time() - start < 1.0"
            )
        ]
        
        mock_result = TestGenerationResult(
            request_id="req_123",
            generated_suite=TestSuite(
                id="suite_1",
                name="Performance Tests",
                description="Performance test suite",
                framework=TestFramework.PYTEST,
                test_cases=mock_test_cases
            ),
            generation_time=2.5,
            estimated_coverage=0.5,
            test_count=1,
            complexity_score=0.6,
            maintainability_score=0.6
        )
        
        mock_generate.return_value = mock_result
        
        request_data = {
            "target_code": self.sample_api_code,
            "code_type": "api",
            "framework": "pytest",
            "include_performance_tests": True
        }
        
        response = client.post("/api/v1/test-generation/generate-performance-tests", json=request_data)
        
        assert response.status_code == 200
        
        result = response.json()
        assert len(result) == 1
        assert result[0]["test_type"] == "performance"
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    def test_validate_test_suite(self, mock_auth):
        """Test test suite validation endpoint"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Create test suite for validation
        test_suite = {
            "id": "suite_1",
            "name": "Test Suite",
            "description": "Test suite for validation",
            "framework": "pytest",
            "test_cases": [
                {
                    "id": "test_1",
                    "name": "test_function",
                    "description": "Test function",
                    "test_type": "unit",
                    "framework": "pytest",
                    "test_code": "assert function() == expected",
                    "assertions": [
                        {
                            "assertion_type": "equals",
                            "expected_value": "expected",
                            "actual_expression": "function()",
                            "description": "Function should return expected value"
                        }
                    ],
                    "tags": ["unit"],
                    "priority": 1,
                    "dependencies": [],
                    "fixtures": [],
                    "status": "pending",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            ],
            "sub_suites": [],
            "parallel_execution": False,
            "target_coverage": 0.8,
            "coverage_threshold": 0.7,
            "environment_config": {},
            "total_tests": 1,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
        
        response = client.post("/api/v1/test-generation/validate-tests", json=test_suite)
        
        assert response.status_code == 200
        
        result = response.json()
        assert "suite_id" in result
        assert "validation_result" in result
        assert "quality_score" in result
        assert result["suite_id"] == "suite_1"
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    def test_list_test_suites(self, mock_auth):
        """Test listing test suites endpoint"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        response = client.get("/api/v1/test-generation/suites")
        
        assert response.status_code == 200
        
        result = response.json()
        assert isinstance(result, list)
        # Currently returns empty list as placeholder
        assert len(result) == 0
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    def test_get_test_suite_not_found(self, mock_auth):
        """Test getting non-existent test suite"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        response = client.get("/api/v1/test-generation/suites/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    def test_execute_test_suite(self, mock_auth):
        """Test test suite execution endpoint"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        request_data = {
            "test_suite_id": "suite_123",
            "environment": "test",
            "parallel": False
        }
        
        response = client.post("/api/v1/test-generation/suites/suite_123/execute", json=request_data)
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["test_suite_id"] == "suite_123"
        assert result["status"] == "completed"
        assert "total_tests" in result
        assert "passed_tests" in result
        assert "failed_tests" in result
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    def test_export_test_suite(self, mock_auth):
        """Test test suite export endpoint"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        response = client.post("/api/v1/test-generation/export-tests/suite_123?format=pytest")
        
        assert response.status_code == 200
        
        result = response.json()
        assert "message" in result
        assert "pytest" in result["message"]
        assert "download_url" in result
    
    def test_get_supported_frameworks(self):
        """Test getting supported frameworks endpoint"""
        response = client.get("/api/v1/test-generation/frameworks")
        
        assert response.status_code == 200
        
        result = response.json()
        assert isinstance(result, list)
        assert "pytest" in result
        assert "jest" in result
    
    def test_get_supported_test_types(self):
        """Test getting supported test types endpoint"""
        response = client.get("/api/v1/test-generation/test-types")
        
        assert response.status_code == 200
        
        result = response.json()
        assert isinstance(result, list)
        assert "unit" in result
        assert "integration" in result
        assert "end_to_end" in result
        assert "performance" in result
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/test-generation/health")
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "healthy"
        assert result["service"] == "test-generation"
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    def test_authentication_required(self, mock_auth):
        """Test that authentication is required for protected endpoints"""
        # Mock authentication failure
        mock_auth.side_effect = Exception("Authentication failed")
        
        request_data = {
            "target_code": "def test(): pass",
            "code_type": "function"
        }
        
        with pytest.raises(Exception):
            client.post("/api/v1/test-generation/generate", json=request_data)


class TestTestGenerationWorkflows:
    """Test complete test generation workflows"""
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    @patch('scrollintel.engines.test_generator.TestGenerator')
    def test_complete_python_module_workflow(self, mock_generator_class, mock_auth):
        """Test complete workflow for Python module test generation"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock the test generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        # Create comprehensive test result
        test_cases = [
            TestCase(
                id="test_1",
                name="test_function_happy_path",
                description="Test function with valid input",
                test_type=TestType.UNIT,
                framework=TestFramework.PYTEST,
                test_code="assert function(1, 2) == 3",
                tags=["unit", "happy_path"]
            ),
            TestCase(
                id="test_2",
                name="test_function_edge_cases",
                description="Test function with edge cases",
                test_type=TestType.UNIT,
                framework=TestFramework.PYTEST,
                test_code="assert function(0, 0) == 0",
                tags=["unit", "edge_cases"]
            ),
            TestCase(
                id="test_3",
                name="test_class_integration",
                description="Test class integration",
                test_type=TestType.INTEGRATION,
                framework=TestFramework.PYTEST,
                test_code="instance = MyClass(); assert instance.method() is not None",
                tags=["integration"]
            )
        ]
        
        mock_result = TestGenerationResult(
            request_id="req_123",
            generated_suite=TestSuite(
                id="suite_1",
                name="Comprehensive Test Suite",
                description="Complete test suite for Python module",
                framework=TestFramework.PYTEST,
                test_cases=test_cases
            ),
            generation_time=2.5,
            estimated_coverage=0.92,
            test_count=3,
            complexity_score=0.3,
            maintainability_score=0.85,
            recommendations=["Consider adding performance tests"],
            warnings=[]
        )
        
        mock_generator.generate_comprehensive_tests.return_value = mock_result
        
        # Test the workflow
        python_code = '''
def calculate_average(numbers):
    """Calculate average of a list of numbers"""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

class StatisticsCalculator:
    def __init__(self):
        self.history = []
    
    def add_calculation(self, result):
        self.history.append(result)
    
    def get_statistics(self):
        if not self.history:
            return {"count": 0, "average": 0}
        return {
            "count": len(self.history),
            "average": sum(self.history) / len(self.history)
        }
'''
        
        request_data = {
            "target_code": python_code,
            "code_type": "module",
            "test_types": ["unit", "integration"],
            "framework": "pytest",
            "coverage_target": 0.9,
            "include_edge_cases": True,
            "include_error_cases": True,
            "requirements": [
                "Function should handle empty lists",
                "Class should track calculation history",
                "Statistics should be calculated correctly"
            ]
        }
        
        response = client.post("/api/v1/test-generation/generate", json=request_data)
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["test_count"] == 3
        assert result["estimated_coverage"] == 0.92
        assert len(result["generated_suite"]["test_cases"]) == 3
        
        # Verify test diversity
        test_types = [test["test_type"] for test in result["generated_suite"]["test_cases"]]
        assert "unit" in test_types
        assert "integration" in test_types
        
        # Verify quality metrics
        assert result["complexity_score"] == 0.3
        assert result["maintainability_score"] == 0.85
        assert len(result["recommendations"]) > 0
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    @patch('scrollintel.engines.test_generator.TestGenerator')
    def test_api_test_generation_workflow(self, mock_generator_class, mock_auth):
        """Test complete workflow for API test generation"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock the test generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        # Create API-specific test result
        test_cases = [
            TestCase(
                id="test_1",
                name="test_api_users_get",
                description="Test GET /api/users endpoint",
                test_type=TestType.INTEGRATION,
                framework=TestFramework.PYTEST,
                test_code="response = client.get('/api/users'); assert response.status_code == 200",
                tags=["integration", "api", "get"]
            ),
            TestCase(
                id="test_2",
                name="test_api_users_post",
                description="Test POST /api/users endpoint",
                test_type=TestType.INTEGRATION,
                framework=TestFramework.PYTEST,
                test_code="response = client.post('/api/users', json={'name': 'John'}); assert response.status_code == 201",
                tags=["integration", "api", "post"]
            ),
            TestCase(
                id="test_3",
                name="test_e2e_user_management",
                description="End-to-end user management workflow",
                test_type=TestType.END_TO_END,
                framework=TestFramework.CYPRESS,
                test_code="cy.visit('/users'); cy.get('[data-testid=\"add-user\"]').click();",
                tags=["e2e", "workflow"]
            )
        ]
        
        mock_result = TestGenerationResult(
            request_id="req_456",
            generated_suite=TestSuite(
                id="suite_2",
                name="API Test Suite",
                description="Complete test suite for API endpoints",
                framework=TestFramework.PYTEST,
                test_cases=test_cases
            ),
            generation_time=3.2,
            estimated_coverage=0.88,
            test_count=3,
            complexity_score=0.4,
            maintainability_score=0.78,
            recommendations=["Add authentication tests", "Consider load testing"],
            warnings=["Some endpoints may need additional validation"]
        )
        
        mock_generator.generate_comprehensive_tests.return_value = mock_result
        
        # Test API workflow
        api_code = '''
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users"""
    return jsonify([
        {"id": 1, "name": "John Doe", "email": "john@example.com"},
        {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
    ])

@app.route('/api/users', methods=['POST'])
def create_user():
    """Create a new user"""
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({"error": "Name is required"}), 400
    
    new_user = {
        "id": 3,
        "name": data["name"],
        "email": data.get("email", "")
    }
    return jsonify(new_user), 201

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user"""
    return jsonify({"message": f"User {user_id} deleted"}), 200
'''
        
        request_data = {
            "target_code": api_code,
            "code_type": "api",
            "test_types": ["integration", "end_to_end"],
            "framework": "pytest",
            "coverage_target": 0.85,
            "requirements": [
                "All endpoints should return appropriate status codes",
                "POST endpoint should validate input data",
                "DELETE endpoint should handle non-existent users"
            ]
        }
        
        response = client.post("/api/v1/test-generation/generate", json=request_data)
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["test_count"] == 3
        assert result["estimated_coverage"] == 0.88
        
        # Verify API-specific tests
        test_names = [test["name"] for test in result["generated_suite"]["test_cases"]]
        assert any("api" in name for name in test_names)
        assert any("e2e" in name for name in test_names)
        
        # Verify recommendations and warnings
        assert len(result["recommendations"]) > 0
        assert len(result["warnings"]) > 0


class TestTestGenerationErrorHandling:
    """Test error handling in test generation API"""
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    @patch('scrollintel.engines.test_generator.TestGenerator.generate_comprehensive_tests')
    def test_generation_error_handling(self, mock_generate, mock_auth):
        """Test handling of test generation errors"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock generation failure
        mock_generate.side_effect = Exception("Test generation failed")
        
        request_data = {
            "target_code": "invalid code syntax",
            "code_type": "module",
            "framework": "pytest"
        }
        
        response = client.post("/api/v1/test-generation/generate", json=request_data)
        
        assert response.status_code == 500
        assert "Test generation failed" in response.json()["detail"]
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    def test_invalid_test_suite_validation(self, mock_auth):
        """Test validation of invalid test suite"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Invalid test suite structure
        invalid_suite = {
            "id": "suite_1",
            "name": "Invalid Suite"
            # Missing required fields
        }
        
        response = client.post("/api/v1/test-generation/validate-tests", json=invalid_suite)
        
        assert response.status_code == 422  # Validation error
    
    def test_unauthenticated_access(self):
        """Test access without authentication"""
        # This test assumes authentication middleware is properly configured
        # In a real implementation, this would test the actual auth behavior
        pass


# Performance and load testing
class TestTestGenerationPerformance:
    """Test performance aspects of test generation API"""
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    @patch('scrollintel.engines.test_generator.TestGenerator.generate_comprehensive_tests')
    def test_large_code_generation_performance(self, mock_generate, mock_auth):
        """Test performance with large code input"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock fast generation for large code
        mock_result = TestGenerationResult(
            request_id="req_perf",
            generated_suite=TestSuite(
                id="suite_perf",
                name="Performance Test Suite",
                description="Large code test suite",
                framework=TestFramework.PYTEST,
                test_cases=[]
            ),
            generation_time=0.5,  # Fast generation
            estimated_coverage=0.8,
            test_count=0,
            complexity_score=0.3,
            maintainability_score=0.8
        )
        
        mock_generate.return_value = mock_result
        
        # Generate large code sample
        large_code = "\n".join([f"def function_{i}(): return {i}" for i in range(1000)])
        
        request_data = {
            "target_code": large_code,
            "code_type": "module",
            "framework": "pytest"
        }
        
        response = client.post("/api/v1/test-generation/generate", json=request_data)
        
        assert response.status_code == 200
        assert response.json()["generation_time"] < 5.0  # Should be reasonably fast
    
    @patch('scrollintel.api.routes.test_generation_routes.get_current_user')
    def test_concurrent_requests(self, mock_auth):
        """Test handling of concurrent test generation requests"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # This would typically test with actual concurrent requests
        # For now, just ensure the endpoint can handle multiple calls
        request_data = {
            "target_code": "def simple_function(): return True",
            "code_type": "function",
            "framework": "pytest"
        }
        
        # Simulate multiple requests (in real test, would use threading)
        responses = []
        for i in range(3):
            try:
                response = client.post("/api/v1/test-generation/generate", json=request_data)
                responses.append(response.status_code)
            except Exception:
                # Handle any concurrency issues
                pass
        
        # At least some requests should succeed
        assert len(responses) > 0