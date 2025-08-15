"""
Integration Tests for Quality Assurance System

Tests the complete quality assurance workflow including all components
working together for the automated code generation system.
"""

import pytest
from scrollintel.engines.code_validator import CodeValidator
from scrollintel.engines.security_scanner import SecurityScanner
from scrollintel.engines.performance_analyzer import PerformanceAnalyzer
from scrollintel.engines.compliance_checker import ComplianceChecker
from scrollintel.engines.code_review_engine import CodeReviewEngine

class TestQualityAssuranceIntegration:
    
    def setup_method(self):
        self.validator = CodeValidator()
        self.security_scanner = SecurityScanner()
        self.performance_analyzer = PerformanceAnalyzer()
        self.compliance_checker = ComplianceChecker()
        self.review_engine = CodeReviewEngine()
    
    def test_complete_quality_assurance_workflow(self):
        """Test complete QA workflow with problematic code"""
        problematic_code = '''
import os
import pickle

# Hardcoded credentials - security issue
API_KEY = "sk-1234567890abcdef1234567890abcdef"
DATABASE_PASSWORD = "super_secret_password123"

def ProcessUserData(user_input, data_file):  # Bad naming - compliance issue
    # SQL injection vulnerability - security issue
    query = "SELECT * FROM users WHERE id = " + user_input
    cursor.execute(query)
    
    # Command injection vulnerability - security issue
    os.system(user_input)
    
    # Insecure deserialization - security issue
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Performance issue - string concatenation in loop
    result = ""
    for i in range(10000):
        for j in range(100):  # Nested loops - performance issue
            result += str(i) + str(j) + ", "
    
    # Line too long - compliance issue
    very_long_variable_name_that_exceeds_recommended_line_length = "This is a very long line that definitely exceeds the PEP8 recommended maximum of 79 characters"
    
    # Missing docstring - compliance issue
    # Bare except clause - compliance issue
    try:
        risky_operation()
    except:
        pass
    
    return result

def risky_operation():
    pass
'''
        
        # Run individual analysis tools
        validation_result = self.validator.validate_code(problematic_code, 'python')
        security_report = self.security_scanner.scan_code(problematic_code, 'python')
        performance_report = self.performance_analyzer.analyze_performance(problematic_code, 'python')
        compliance_report = self.compliance_checker.check_compliance(problematic_code, 'python')
        
        # Verify each tool found issues
        assert len(validation_result.issues) > 0
        assert len(security_report.vulnerabilities) > 0
        assert len(performance_report.issues) > 0
        assert len(compliance_report.violations) > 0
        
        # Run comprehensive review
        review_report = self.review_engine.review_code(problematic_code, 'python')
        
        # Verify comprehensive review consolidates all issues
        assert len(review_report.issues) > 0
        
        # Check that all analysis sources are represented
        sources = {issue.source for issue in review_report.issues}
        assert 'security' in sources
        assert 'performance' in sources
        assert 'compliance' in sources
        
        # Verify quality scores reflect the issues
        assert review_report.metrics.overall_score < 80  # Should be lower due to many issues
        assert review_report.metrics.security_score < 30  # Many security issues (inverted risk)
        # Performance and compliance scores may vary based on specific issues found
        
        # Verify deployment readiness
        assert "NOT READY" in review_report.summary.deployment_readiness
        
        # Verify recommendations are generated
        assert len(review_report.recommendations) > 0
        assert len(review_report.improvement_plan) > 0
    
    def test_high_quality_code_workflow(self):
        """Test QA workflow with high-quality code"""
        high_quality_code = '''
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
        raise ValueError("n must be non-negative")
    
    if n <= 1:
        return n
    
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr

def validate_input(value):
    """
    Validate input value is a non-negative integer.
    
    Args:
        value: The value to validate
        
    Returns:
        int: The validated integer value
        
    Raises:
        ValueError: If value is not a valid non-negative integer
    """
    try:
        num = int(value)
        if num < 0:
            raise ValueError("Value must be non-negative")
        return num
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid input: {e}")

if __name__ == "__main__":
    try:
        n = validate_input(input("Enter a number: "))
        result = calculate_fibonacci(n)
        print(f"Fibonacci({n}) = {result}")
    except ValueError as e:
        print(f"Error: {e}")
'''
        
        # Run comprehensive review
        review_report = self.review_engine.review_code(high_quality_code, 'python')
        
        # Verify high quality scores
        assert review_report.metrics.overall_score > 80
        assert review_report.metrics.security_score > 90  # No security issues
        assert review_report.metrics.compliance_score > 80  # Good compliance
        assert review_report.metrics.maintainability_score > 80  # Well structured
        assert review_report.metrics.readability_score > 80  # Well documented
        
        # Verify deployment readiness
        assert "READY" in review_report.summary.deployment_readiness
        
        # Should have minimal issues
        critical_issues = [issue for issue in review_report.issues 
                          if issue.priority.value == 'critical']
        assert len(critical_issues) == 0
    
    def test_javascript_quality_assurance(self):
        """Test QA workflow for JavaScript code"""
        js_code = '''
function processUserData(userInput) {
    // XSS vulnerability
    document.getElementById('output').innerHTML = userInput;
    
    // Use of var instead of let/const
    var result = '';
    
    // Inefficient DOM queries in loop
    for (var i = 0; i < 100; i++) {
        document.getElementById('item-' + i).style.display = 'block';
        result += i + ', ';
    }
    
    // Console statement (should not be in production)
    console.log('Processing complete');
    
    // Eval usage - security risk
    return eval('(' + userInput + ')');
}
'''
        
        review_report = self.review_engine.review_code(js_code, 'javascript')
        
        # Should detect multiple types of issues
        assert len(review_report.issues) > 0
        
        # Check for specific issue categories
        categories = {issue.category for issue in review_report.issues}
        assert 'security' in categories  # XSS and eval
        assert 'performance' in categories  # DOM queries in loop
        
        # Should have security vulnerabilities
        security_issues = [issue for issue in review_report.issues 
                          if issue.category == 'security']
        assert len(security_issues) > 0
        
        # Should not be ready for deployment
        assert "NOT READY" in review_report.summary.deployment_readiness
    
    def test_comprehensive_review_integration(self):
        """Test that comprehensive review properly integrates all tools"""
        test_code = '''
import os
password = "secret123"

def test_function(user_input):
    os.system(user_input)  # Security issue
    result = ""
    for i in range(1000):
        result += str(i)  # Performance issue
    x = "This is a very long line that exceeds recommended length"  # Compliance
    return result
'''
        
        review_report = self.review_engine.review_code(test_code, 'python')
        
        # Should have issues from multiple sources
        sources = {issue.source for issue in review_report.issues}
        assert len(sources) >= 2
        
        # Should have proper quality metrics
        assert 0 <= review_report.metrics.overall_score <= 100
        assert 0 <= review_report.metrics.security_score <= 100
        assert 0 <= review_report.metrics.performance_score <= 100
        assert 0 <= review_report.metrics.compliance_score <= 100
        
        # Should have recommendations and improvement plan
        assert len(review_report.recommendations) > 0
        assert len(review_report.improvement_plan) > 0