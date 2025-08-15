"""
Demo Script for Automated Test Generation System

This script demonstrates the capabilities of the automated test generation system,
showing how it can generate comprehensive test suites for different types of code.
"""

import asyncio
import json
from datetime import datetime
from typing import List

from scrollintel.models.test_generation_models import (
    TestGenerationRequest, TestType, TestFramework
)
from scrollintel.engines.test_generator import TestGenerator


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_test_case(test_case, index: int):
    """Print a formatted test case"""
    print(f"\n{index}. {test_case.name}")
    print(f"   Type: {test_case.test_type.value}")
    print(f"   Framework: {test_case.framework.value}")
    print(f"   Description: {test_case.description}")
    print(f"   Tags: {', '.join(test_case.tags)}")
    if test_case.assertions:
        print(f"   Assertions: {len(test_case.assertions)}")
    print(f"   Code Preview:")
    code_lines = test_case.test_code.strip().split('\n')[:3]
    for line in code_lines:
        print(f"     {line}")
    if len(test_case.test_code.strip().split('\n')) > 3:
        print("     ...")


def demo_python_function_testing():
    """Demo test generation for Python functions"""
    print_header("Python Function Test Generation Demo")
    
    sample_code = '''
def calculate_fibonacci(n):
    """
    Calculate the nth Fibonacci number using iterative approach.
    
    Args:
        n (int): The position in the Fibonacci sequence
        
    Returns:
        int: The nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Fibonacci sequence is not defined for negative numbers")
    elif n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

def factorial(n):
    """Calculate factorial of a number"""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n <= 1:
        return 1
    else:
        return n * factorial(n - 1)
'''
    
    print("Sample Code:")
    print(sample_code)
    
    # Generate tests
    test_generator = TestGenerator()
    
    request = TestGenerationRequest(
        target_code=sample_code,
        code_type="module",
        test_types=[TestType.UNIT],
        framework=TestFramework.PYTEST,
        coverage_target=0.9,
        include_edge_cases=True,
        include_error_cases=True,
        requirements=[
            "Functions should handle edge cases properly",
            "Error conditions should raise appropriate exceptions",
            "Mathematical correctness should be verified"
        ]
    )
    
    print("\nGenerating tests...")
    result = test_generator.generate_comprehensive_tests(request)
    
    print(f"\nGeneration Results:")
    print(f"  Tests Generated: {result.test_count}")
    print(f"  Estimated Coverage: {result.estimated_coverage:.1%}")
    print(f"  Generation Time: {result.generation_time:.2f}s")
    print(f"  Complexity Score: {result.complexity_score:.2f}")
    print(f"  Maintainability Score: {result.maintainability_score:.2f}")
    
    print(f"\nGenerated Test Cases:")
    for i, test_case in enumerate(result.generated_suite.test_cases, 1):
        print_test_case(test_case, i)
    
    if result.recommendations:
        print(f"\nRecommendations:")
        for rec in result.recommendations:
            print(f"  • {rec}")
    
    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")


def demo_class_testing():
    """Demo test generation for Python classes"""
    print_header("Python Class Test Generation Demo")
    
    sample_code = '''
class BankAccount:
    """A simple bank account class"""
    
    def __init__(self, account_number: str, initial_balance: float = 0.0):
        self.account_number = account_number
        self.balance = initial_balance
        self.transaction_history = []
    
    def deposit(self, amount: float) -> bool:
        """Deposit money into the account"""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        self.balance += amount
        self.transaction_history.append(f"Deposit: +${amount:.2f}")
        return True
    
    def withdraw(self, amount: float) -> bool:
        """Withdraw money from the account"""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        
        self.balance -= amount
        self.transaction_history.append(f"Withdrawal: -${amount:.2f}")
        return True
    
    def get_balance(self) -> float:
        """Get current account balance"""
        return self.balance
    
    def get_transaction_history(self) -> List[str]:
        """Get transaction history"""
        return self.transaction_history.copy()
'''
    
    print("Sample Code:")
    print(sample_code)
    
    # Generate tests
    test_generator = TestGenerator()
    
    request = TestGenerationRequest(
        target_code=sample_code,
        code_type="class",
        test_types=[TestType.UNIT, TestType.INTEGRATION],
        framework=TestFramework.PYTEST,
        coverage_target=0.95,
        include_edge_cases=True,
        include_error_cases=True,
        requirements=[
            "All methods should be tested",
            "State changes should be verified",
            "Error conditions should be handled",
            "Transaction history should be maintained"
        ]
    )
    
    print("\nGenerating tests...")
    result = test_generator.generate_comprehensive_tests(request)
    
    print(f"\nGeneration Results:")
    print(f"  Tests Generated: {result.test_count}")
    print(f"  Estimated Coverage: {result.estimated_coverage:.1%}")
    print(f"  Generation Time: {result.generation_time:.2f}s")
    
    print(f"\nGenerated Test Cases:")
    for i, test_case in enumerate(result.generated_suite.test_cases, 1):
        print_test_case(test_case, i)


def demo_api_testing():
    """Demo test generation for API endpoints"""
    print_header("API Test Generation Demo")
    
    sample_code = '''
from flask import Flask, request, jsonify
from typing import Dict, List

app = Flask(__name__)

# In-memory storage for demo
users_db = {}
next_user_id = 1

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"})

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users"""
    return jsonify(list(users_db.values()))

@app.route('/api/users', methods=['POST'])
def create_user():
    """Create a new user"""
    global next_user_id
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    if 'name' not in data:
        return jsonify({"error": "Name is required"}), 400
    
    if 'email' not in data:
        return jsonify({"error": "Email is required"}), 400
    
    user = {
        "id": next_user_id,
        "name": data["name"],
        "email": data["email"],
        "created_at": "2024-01-01T00:00:00Z"
    }
    
    users_db[next_user_id] = user
    next_user_id += 1
    
    return jsonify(user), 201

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id: int):
    """Get a specific user"""
    if user_id not in users_db:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify(users_db[user_id])

@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id: int):
    """Update a user"""
    if user_id not in users_db:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    user = users_db[user_id]
    user.update(data)
    
    return jsonify(user)

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id: int):
    """Delete a user"""
    if user_id not in users_db:
        return jsonify({"error": "User not found"}), 404
    
    del users_db[user_id]
    return jsonify({"message": "User deleted successfully"})
'''
    
    print("Sample API Code:")
    print(sample_code[:500] + "..." if len(sample_code) > 500 else sample_code)
    
    # Generate tests
    test_generator = TestGenerator()
    
    request = TestGenerationRequest(
        target_code=sample_code,
        code_type="api",
        test_types=[TestType.UNIT, TestType.INTEGRATION, TestType.END_TO_END],
        framework=TestFramework.PYTEST,
        coverage_target=0.85,
        include_edge_cases=True,
        include_error_cases=True,
        requirements=[
            "All endpoints should return appropriate status codes",
            "Error handling should be comprehensive",
            "CRUD operations should work correctly",
            "Data validation should be enforced"
        ]
    )
    
    print("\nGenerating tests...")
    result = test_generator.generate_comprehensive_tests(request)
    
    print(f"\nGeneration Results:")
    print(f"  Tests Generated: {result.test_count}")
    print(f"  Estimated Coverage: {result.estimated_coverage:.1%}")
    print(f"  Generation Time: {result.generation_time:.2f}s")
    
    # Group tests by type
    test_types = {}
    for test_case in result.generated_suite.test_cases:
        test_type = test_case.test_type.value
        if test_type not in test_types:
            test_types[test_type] = []
        test_types[test_type].append(test_case)
    
    for test_type, tests in test_types.items():
        print(f"\n{test_type.upper()} Tests ({len(tests)}):")
        for i, test_case in enumerate(tests, 1):
            print_test_case(test_case, i)


def demo_performance_testing():
    """Demo performance test generation"""
    print_header("Performance Test Generation Demo")
    
    sample_code = '''
import time
import requests
from typing import List, Dict

def process_large_dataset(data: List[Dict]) -> Dict:
    """Process a large dataset with complex operations"""
    result = {
        "total_records": len(data),
        "processed_records": 0,
        "categories": {},
        "statistics": {}
    }
    
    for record in data:
        # Simulate complex processing
        time.sleep(0.001)  # Simulate processing time
        
        category = record.get("category", "unknown")
        if category not in result["categories"]:
            result["categories"][category] = 0
        result["categories"][category] += 1
        
        result["processed_records"] += 1
    
    # Calculate statistics
    result["statistics"] = {
        "avg_per_category": result["processed_records"] / len(result["categories"]) if result["categories"] else 0,
        "processing_efficiency": result["processed_records"] / result["total_records"] if result["total_records"] else 0
    }
    
    return result

def fetch_external_data(url: str, timeout: int = 30) -> Dict:
    """Fetch data from external API"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch data: {str(e)}")

class DataProcessor:
    """High-performance data processor"""
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.processed_count = 0
    
    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of data"""
        processed = []
        for item in batch:
            # Simulate processing
            processed_item = {
                **item,
                "processed_at": time.time(),
                "batch_id": len(processed) // self.batch_size
            }
            processed.append(processed_item)
        
        self.processed_count += len(processed)
        return processed
'''
    
    print("Sample Code:")
    print(sample_code[:500] + "..." if len(sample_code) > 500 else sample_code)
    
    # Generate performance tests
    test_generator = TestGenerator()
    
    request = TestGenerationRequest(
        target_code=sample_code,
        code_type="module",
        test_types=[TestType.PERFORMANCE, TestType.LOAD],
        framework=TestFramework.PYTEST,
        include_performance_tests=True,
        requirements=[
            "Functions should complete within acceptable time limits",
            "Memory usage should be reasonable",
            "System should handle concurrent requests",
            "Performance should degrade gracefully under load"
        ]
    )
    
    print("\nGenerating performance tests...")
    result = test_generator.generate_comprehensive_tests(request)
    
    print(f"\nGeneration Results:")
    print(f"  Tests Generated: {result.test_count}")
    print(f"  Generation Time: {result.generation_time:.2f}s")
    
    print(f"\nGenerated Performance Tests:")
    for i, test_case in enumerate(result.generated_suite.test_cases, 1):
        print_test_case(test_case, i)


def demo_test_validation():
    """Demo test suite validation"""
    print_header("Test Suite Validation Demo")
    
    # Create a sample test suite for validation
    from scrollintel.models.test_generation_models import TestSuite, TestCase, TestAssertion
    
    test_suite = TestSuite(
        id="demo_suite",
        name="Demo Test Suite",
        description="A sample test suite for validation demo",
        framework=TestFramework.PYTEST,
        test_cases=[
            TestCase(
                id="test_1",
                name="test_function_happy_path",
                description="Test function with valid input",
                test_type=TestType.UNIT,
                framework=TestFramework.PYTEST,
                test_code="""
def test_function_happy_path():
    # Arrange
    input_value = 5
    expected_result = 25
    
    # Act
    result = square_function(input_value)
    
    # Assert
    assert result == expected_result
    assert isinstance(result, int)
""",
                assertions=[
                    TestAssertion(
                        assertion_type="equals",
                        expected_value=25,
                        actual_expression="square_function(5)",
                        description="Function should return square of input"
                    )
                ],
                tags=["unit", "happy_path", "math"],
                priority=1
            ),
            TestCase(
                id="test_2",
                name="test_function_edge_cases",
                description="Test function with edge cases",
                test_type=TestType.UNIT,
                framework=TestFramework.PYTEST,
                test_code="""
def test_function_edge_cases():
    # Test with zero
    assert square_function(0) == 0
    
    # Test with negative numbers
    assert square_function(-3) == 9
    
    # Test with large numbers
    assert square_function(1000) == 1000000
""",
                tags=["unit", "edge_cases"],
                priority=2
            ),
            TestCase(
                id="test_3",
                name="test_function_error_handling",
                description="Test function error handling",
                test_type=TestType.UNIT,
                framework=TestFramework.PYTEST,
                test_code="""
def test_function_error_handling():
    with pytest.raises(TypeError):
        square_function("invalid")
    
    with pytest.raises(TypeError):
        square_function(None)
""",
                tags=["unit", "error_handling"],
                priority=1
            )
        ]
    )
    
    # Update suite statistics
    test_suite.total_tests = len(test_suite.test_cases)
    
    print("Sample Test Suite:")
    print(f"  Name: {test_suite.name}")
    print(f"  Framework: {test_suite.framework.value}")
    print(f"  Total Tests: {test_suite.total_tests}")
    
    # Validate the test suite
    test_generator = TestGenerator()
    validation_result = test_generator.test_validator.validate_test_suite(test_suite)
    
    print(f"\nValidation Results:")
    print(f"  Complexity Score: {validation_result['complexity_score']:.2f} (lower is better)")
    print(f"  Maintainability Score: {validation_result['maintainability_score']:.2f} (higher is better)")
    
    quality_score = (validation_result['maintainability_score'] + 
                    (1 - validation_result['complexity_score'])) / 2
    print(f"  Overall Quality Score: {quality_score:.2f}")
    
    if validation_result['recommendations']:
        print(f"\nRecommendations:")
        for rec in validation_result['recommendations']:
            print(f"  • {rec}")
    
    if validation_result['warnings']:
        print(f"\nWarnings:")
        for warning in validation_result['warnings']:
            print(f"  ⚠ {warning}")


def demo_framework_comparison():
    """Demo test generation for different frameworks"""
    print_header("Framework Comparison Demo")
    
    sample_code = '''
function calculateArea(length, width) {
    if (typeof length !== 'number' || typeof width !== 'number') {
        throw new Error('Both parameters must be numbers');
    }
    
    if (length < 0 || width < 0) {
        throw new Error('Dimensions cannot be negative');
    }
    
    return length * width;
}

class Rectangle {
    constructor(length, width) {
        this.length = length;
        this.width = width;
    }
    
    getArea() {
        return calculateArea(this.length, this.width);
    }
    
    getPerimeter() {
        return 2 * (this.length + this.width);
    }
}
'''
    
    print("Sample JavaScript Code:")
    print(sample_code)
    
    # Generate tests for different frameworks
    frameworks = [TestFramework.JEST, TestFramework.MOCHA]
    
    test_generator = TestGenerator()
    
    for framework in frameworks:
        print(f"\n--- {framework.value.upper()} Framework ---")
        
        request = TestGenerationRequest(
            target_code=sample_code,
            code_type="module",
            test_types=[TestType.UNIT],
            framework=framework,
            coverage_target=0.8,
            include_edge_cases=True,
            include_error_cases=True
        )
        
        result = test_generator.generate_comprehensive_tests(request)
        
        print(f"Tests Generated: {result.test_count}")
        print(f"Estimated Coverage: {result.estimated_coverage:.1%}")
        
        if result.generated_suite.test_cases:
            print("Sample Test Case:")
            sample_test = result.generated_suite.test_cases[0]
            print(f"  Name: {sample_test.name}")
            print(f"  Framework: {sample_test.framework.value}")
            code_preview = sample_test.test_code.strip().split('\n')[:5]
            for line in code_preview:
                print(f"    {line}")


def main():
    """Run all demo scenarios"""
    print("🚀 Automated Test Generation System Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive test generation capabilities")
    print("of the ScrollIntel automated code generation system.")
    
    try:
        # Run all demos
        demo_python_function_testing()
        demo_class_testing()
        demo_api_testing()
        demo_performance_testing()
        demo_test_validation()
        demo_framework_comparison()
        
        print_header("Demo Complete")
        print("✅ All test generation demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Unit test generation for functions and classes")
        print("  • Integration test generation for APIs")
        print("  • End-to-end test generation for workflows")
        print("  • Performance and load test generation")
        print("  • Test suite validation and quality assessment")
        print("  • Multi-framework support (pytest, jest, cypress, etc.)")
        print("  • Comprehensive error handling and edge case testing")
        print("  • Intelligent code analysis and test recommendation")
        
        print(f"\nNext Steps:")
        print("  1. Integrate with your development workflow")
        print("  2. Customize test generation templates")
        print("  3. Set up automated test execution")
        print("  4. Configure quality gates and coverage thresholds")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()