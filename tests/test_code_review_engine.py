"""
Tests for Code Review Engine
"""

import pytest
from scrollintel.engines.code_review_engine import (
    CodeReviewEngine, CodeReviewReport, ReviewIssue, ReviewPriority
)

class TestCodeReviewEngine:
    
    def setup_method(self):
        self.engine = CodeReviewEngine()
    
    def test_comprehensive_code_review(self):
        """Test comprehensive code review with multiple issue types"""
        code = """
import os
password = "hardcoded_secret123"

def process_data(user_input):
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE id = " + user_input
    cursor.execute(query)
    
    # Performance issue - string concatenation in loop
    result = ""
    for i in range(1000):
        result += str(i)
    
    # Command injection
    os.system(user_input)
    
    # Long line violation
    x = "This is a very long line that exceeds the recommended maximum length and should trigger a compliance violation for line length"
    
    return result
"""
        report = self.engine.review_code(code, 'python')
        
        # Should have issues from all analysis tools
        assert len(report.issues) > 0
        
        # Check for different issue sources
        sources = {issue.source for issue in report.issues}
        assert 'security' in sources
        assert 'performance' in sources
        assert 'compliance' in sources
        
        # Check for different categories
        categories = {issue.category for issue in report.issues}
        assert 'security' in categories
        assert 'performance' in categories
    
    def test_issue_prioritization(self):
        """Test that issues are properly prioritized"""
        code = """
import os
def dangerous_function(user_input):
    os.system(user_input)  # Critical security issue
    x = "long line"  # Minor compliance issue
"""
        report = self.engine.review_code(code, 'python')
        
        # Critical issues should come first
        critical_issues = [issue for issue in report.issues 
                          if issue.priority == ReviewPriority.CRITICAL]
        assert len(critical_issues) > 0
        
        # Issues should be sorted by priority
        priorities = [issue.priority for issue in report.issues]
        priority_values = [p.value for p in priorities]
        # Check that critical comes before low priority issues
        if 'critical' in priority_values and 'low' in priority_values:
            critical_index = priority_values.index('critical')
            low_index = priority_values.index('low')
            assert critical_index < low_index
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation"""
        # High-quality code
        good_code = """
def calculate_sum(numbers):
    \"\"\"Calculate the sum of a list of numbers.\"\"\"
    return sum(numbers)
"""
        good_report = self.engine.review_code(good_code, 'python')
        
        # Low-quality code
        bad_code = """
import os
password = "secret123"
def BadFunction(user_input):
    os.system(user_input)
    query = "SELECT * FROM users WHERE id = " + user_input
    result = ""
    for i in range(1000):
        result += str(i)
    return result
"""
        bad_report = self.engine.review_code(bad_code, 'python')
        
        # Good code should have higher scores
        assert good_report.metrics.overall_score > bad_report.metrics.overall_score
        assert good_report.metrics.security_score > bad_report.metrics.security_score
        assert good_report.metrics.compliance_score > bad_report.metrics.compliance_score
    
    def test_deployment_readiness_assessment(self):
        """Test deployment readiness assessment"""
        # Code with critical issues
        critical_code = """
import os
def execute_command(cmd):
    os.system(cmd)  # Critical security issue
"""
        critical_report = self.engine.review_code(critical_code, 'python')
        assert "NOT READY" in critical_report.summary.deployment_readiness
        
        # Clean code
        clean_code = """
def add_numbers(a, b):
    \"\"\"Add two numbers together.\"\"\"
    return a + b
"""
        clean_report = self.engine.review_code(clean_code, 'python')
        assert "READY" in clean_report.summary.deployment_readiness or "CAUTION" in clean_report.summary.deployment_readiness
    
    def test_fix_time_estimation(self):
        """Test fix time estimation"""
        code = """
import os
def function_with_issues(user_input):
    os.system(user_input)  # Critical - should take longer to fix
    x = "short line"  # Minor issue - quick fix
"""
        report = self.engine.review_code(code, 'python')
        
        # Should provide time estimate
        assert report.summary.estimated_fix_time
        assert any(unit in report.summary.estimated_fix_time.lower() 
                  for unit in ['minute', 'hour', 'day'])
    
    def test_recommendation_generation(self):
        """Test recommendation generation"""
        code = """
import os
password = "hardcoded_password"
def vulnerable_function(user_input):
    os.system(user_input)
    query = "SELECT * FROM users WHERE id = " + user_input
    cursor.execute(query)
"""
        report = self.engine.review_code(code, 'python')
        
        assert len(report.recommendations) > 0
        
        # Should have security-related recommendations
        security_recs = [rec for rec in report.recommendations 
                        if 'security' in rec.lower()]
        assert len(security_recs) > 0
    
    def test_improvement_plan_generation(self):
        """Test improvement plan generation"""
        code = """
import os
password = "secret"
def bad_function(user_input):
    os.system(user_input)
    result = ""
    for i in range(1000):
        result += str(i)
"""
        report = self.engine.review_code(code, 'python')
        
        assert len(report.improvement_plan) > 0
        
        # Should have phases
        phases = [step for step in report.improvement_plan 
                 if 'phase' in step.lower()]
        assert len(phases) > 0
    
    def test_maintainability_score_calculation(self):
        """Test maintainability score calculation"""
        # Simple, maintainable code
        simple_code = """
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b
"""
        simple_report = self.engine.review_code(simple_code, 'python')
        
        # Complex, hard to maintain code
        complex_code = "\n".join([
            "def complex_function():",
            "    # Very long function with many lines"
        ] + [f"    x{i} = {i}" for i in range(100)] + [
            "    return sum([" + ", ".join(f"x{i}" for i in range(100)) + "])"
        ])
        
        complex_report = self.engine.review_code(complex_code, 'python')
        
        assert simple_report.metrics.maintainability_score > complex_report.metrics.maintainability_score
    
    def test_readability_score_calculation(self):
        """Test readability score calculation"""
        # Well-formatted code with comments
        readable_code = """
def calculate_average(numbers):
    \"\"\"Calculate the average of a list of numbers.\"\"\"
    # Check for empty list
    if not numbers:
        return 0
    
    # Calculate sum and divide by count
    total = sum(numbers)
    count = len(numbers)
    return total / count
"""
        readable_report = self.engine.review_code(readable_code, 'python')
        
        # Poorly formatted code without comments
        unreadable_code = """
def BadFunctionName(x):
  y=x+1
  z=y*2
  return z
"""
        unreadable_report = self.engine.review_code(unreadable_code, 'python')
        
        assert readable_report.metrics.readability_score > unreadable_report.metrics.readability_score
    
    def test_issue_consolidation(self):
        """Test that issues from different tools are properly consolidated"""
        code = """
import os
def test_function(user_input):
    os.system(user_input)  # Security issue
    result = ""
    for i in range(1000):
        result += str(i)  # Performance issue
    x = "This is a very long line that exceeds recommended length"  # Compliance issue
"""
        report = self.engine.review_code(code, 'python')
        
        # Should have issues from multiple sources
        sources = {issue.source for issue in report.issues}
        assert len(sources) >= 2
        
        # Each issue should have proper attributes
        for issue in report.issues:
            assert issue.priority is not None
            assert issue.category is not None
            assert issue.source is not None
            assert issue.title is not None
            assert issue.description is not None
    
    def test_summary_statistics(self):
        """Test summary statistics generation"""
        code = """
import os
password = "secret"
def function(user_input):
    os.system(user_input)
    x = "long line that exceeds maximum length recommendations"
"""
        report = self.engine.review_code(code, 'python')
        
        # Check summary statistics
        assert report.summary.total_issues > 0
        assert isinstance(report.summary.issues_by_priority, dict)
        assert isinstance(report.summary.issues_by_category, dict)
        assert isinstance(report.summary.issues_by_source, dict)
        
        # Counts should add up
        priority_total = sum(report.summary.issues_by_priority.values())
        assert priority_total == report.summary.total_issues
    
    def test_json_report_generation(self):
        """Test JSON report generation"""
        code = """
def simple_function():
    return "test"
"""
        report = self.engine.review_code(code, 'python')
        json_report = self.engine.generate_review_report_json(report)
        
        assert isinstance(json_report, str)
        assert 'issues' in json_report
        assert 'metrics' in json_report
        assert 'recommendations' in json_report
    
    def test_html_report_generation(self):
        """Test HTML report generation"""
        code = """
def simple_function():
    return "test"
"""
        report = self.engine.review_code(code, 'python')
        html_report = self.engine.generate_review_report_html(report)
        
        assert isinstance(html_report, str)
        assert '<html>' in html_report
        assert 'Code Review Report' in html_report
        assert str(report.metrics.overall_score) in html_report
    
    def test_error_handling(self):
        """Test error handling in code review"""
        # This should not crash the engine
        report = self.engine.review_code(None, 'python')
        
        assert len(report.issues) > 0
        assert report.metrics.overall_score == 0
        assert "NOT READY" in report.summary.deployment_readiness
        assert 'error' in report.metadata
    
    def test_context_usage(self):
        """Test usage of context in code review"""
        code = """
def simple_function():
    return "test"
"""
        context = {
            'project_type': 'web_application',
            'expected_load': 'high',
            'security_requirements': 'strict'
        }
        
        report = self.engine.review_code(code, 'python', context=context)
        
        # Should complete successfully with context
        assert isinstance(report, CodeReviewReport)
        assert report.metadata['code_length'] > 0
    
    def test_standards_application(self):
        """Test application of specific coding standards"""
        code = """
def function_with_long_line():
    x = "This is a very long line that exceeds the recommended maximum length and should trigger violations"
    return x
"""
        report = self.engine.review_code(code, 'python', standards=['PEP8'])
        
        # Should apply PEP8 standards
        assert 'PEP8' in str(report.metadata.get('standards_applied', []))
    
    def test_different_languages(self):
        """Test code review for different programming languages"""
        # Python code
        python_code = """
def test():
    return "python"
"""
        python_report = self.engine.review_code(python_code, 'python')
        assert python_report.metadata['language'] == 'python'
        
        # JavaScript code
        js_code = """
function test() {
    var x = 5;
    return x;
}
"""
        js_report = self.engine.review_code(js_code, 'javascript')
        assert js_report.metadata['language'] == 'javascript'
        
        # Both should complete successfully
        assert isinstance(python_report, CodeReviewReport)
        assert isinstance(js_report, CodeReviewReport)
    
    def test_issue_line_number_tracking(self):
        """Test that line numbers are properly tracked for issues"""
        code = """
def function():
    import os
    os.system("command")  # This should be on line 3
"""
        report = self.engine.review_code(code, 'python')
        
        # Find security issues that should have line numbers
        security_issues = [issue for issue in report.issues 
                          if issue.category == 'security']
        
        if security_issues:
            # At least some issues should have line numbers
            line_numbers = [issue.line_number for issue in security_issues 
                           if issue.line_number is not None]
            assert len(line_numbers) > 0