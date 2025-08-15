"""
Tests for Code Validator Engine
"""

import pytest
from scrollintel.engines.code_validator import (
    CodeValidator, ValidationResult, ValidationIssue, ValidationSeverity
)

class TestCodeValidator:
    
    def setup_method(self):
        self.validator = CodeValidator()
    
    def test_validate_python_syntax_error(self):
        """Test Python syntax error detection"""
        code = """
def broken_function(
    print("Missing closing parenthesis")
"""
        result = self.validator.validate_code(code, 'python')
        
        assert not result.is_valid
        assert len(result.issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in result.issues)
        assert result.score < 100
    
    def test_validate_python_valid_code(self):
        """Test validation of valid Python code"""
        code = """
def hello_world():
    \"\"\"A simple hello world function\"\"\"
    return "Hello, World!"

if __name__ == "__main__":
    print(hello_world())
"""
        result = self.validator.validate_code(code, 'python')
        
        assert result.is_valid
        assert result.score > 80  # Should have high score for clean code
    
    def test_validate_python_complexity_warning(self):
        """Test detection of high complexity functions"""
        code = """
def complex_function(x):
    if x > 10:
        if x > 20:
            if x > 30:
                if x > 40:
                    if x > 50:
                        if x > 60:
                            if x > 70:
                                if x > 80:
                                    if x > 90:
                                        if x > 100:
                                            return "very high"
                                        return "high"
                                    return "medium-high"
                                return "medium"
                            return "low-medium"
                        return "low"
                    return "very low"
                return "minimal"
            return "tiny"
        return "small"
    return "zero"
"""
        result = self.validator.validate_code(code, 'python')
        
        # Should detect high complexity
        complexity_issues = [issue for issue in result.issues 
                           if 'complexity' in issue.message.lower()]
        assert len(complexity_issues) > 0
    
    def test_validate_javascript_syntax(self):
        """Test JavaScript syntax validation"""
        code = """
function testFunction() {
    let x = 5;
    if (x > 3 {  // Missing closing parenthesis
        console.log("Greater than 3");
    }
}
"""
        result = self.validator.validate_code(code, 'javascript')
        
        assert not result.is_valid
        assert len(result.issues) > 0
    
    def test_validate_javascript_var_usage(self):
        """Test detection of var usage in JavaScript"""
        code = """
function oldStyle() {
    var x = 5;  // Should trigger warning
    var y = 10; // Should trigger warning
    return x + y;
}
"""
        result = self.validator.validate_code(code, 'javascript')
        
        var_issues = [issue for issue in result.issues 
                     if 'var' in issue.message.lower()]
        assert len(var_issues) > 0
    
    def test_validate_typescript_any_usage(self):
        """Test detection of 'any' type usage in TypeScript"""
        code = """
function processData(data: any): any {
    return data.someProperty;
}
"""
        result = self.validator.validate_code(code, 'typescript')
        
        any_issues = [issue for issue in result.issues 
                     if 'any' in issue.message.lower()]
        assert len(any_issues) > 0
    
    def test_validate_sql_select_star(self):
        """Test detection of SELECT * in SQL"""
        code = """
SELECT * FROM users WHERE id = 1;
SELECT name, email FROM users WHERE active = 1;
"""
        result = self.validator.validate_code(code, 'sql')
        
        select_star_issues = [issue for issue in result.issues 
                            if 'select' in issue.message.lower() and '*' in issue.message]
        assert len(select_star_issues) > 0
    
    def test_validate_long_lines(self):
        """Test detection of long lines"""
        code = "x = 'This is a very long line that exceeds the recommended maximum length and should trigger a warning about line length being too long for readability'"
        
        result = self.validator.validate_code(code, 'python')
        
        long_line_issues = [issue for issue in result.issues 
                          if 'long' in issue.message.lower()]
        assert len(long_line_issues) > 0
    
    def test_validate_todo_comments(self):
        """Test detection of TODO comments"""
        code = """
def process_data():
    # TODO: Implement proper error handling
    # FIXME: This is a temporary hack
    return "processed"
"""
        result = self.validator.validate_code(code, 'python')
        
        todo_issues = [issue for issue in result.issues 
                      if 'todo' in issue.message.lower() or 'fixme' in issue.message.lower()]
        assert len(todo_issues) > 0
    
    def test_validate_unsupported_language(self):
        """Test handling of unsupported language"""
        code = "some code in unknown language"
        result = self.validator.validate_code(code, 'unknown_language')
        
        assert result.is_valid  # Should not fail, just warn
        unsupported_issues = [issue for issue in result.issues 
                            if 'unsupported' in issue.message.lower()]
        assert len(unsupported_issues) > 0
    
    def test_get_improvement_suggestions(self):
        """Test generation of improvement suggestions"""
        code = """
def bad_function(
    print("Syntax error")
    x = 'This is a very long line that exceeds the recommended maximum length and should trigger a warning'
"""
        result = self.validator.validate_code(code, 'python')
        suggestions = self.validator.get_improvement_suggestions(result)
        
        assert len(suggestions) > 0
        assert any('error' in suggestion.lower() for suggestion in suggestions)
    
    def test_validation_metrics(self):
        """Test validation metrics calculation"""
        code = """
def simple_function():
    return "Hello"
"""
        result = self.validator.validate_code(code, 'python')
        
        assert 'total_lines' in result.metrics
        assert 'non_empty_lines' in result.metrics
        assert result.metrics['total_lines'] > 0
    
    def test_python_import_validation(self):
        """Test Python import validation"""
        code = """
from .relative_module import something
import os
import sys
"""
        result = self.validator.validate_code(code, 'python')
        
        # Should detect relative import warning
        import_issues = [issue for issue in result.issues 
                        if 'import' in issue.message.lower()]
        assert len(import_issues) > 0
    
    def test_java_structure_validation(self):
        """Test Java code structure validation"""
        code = """
class TestClass {
    private void method() {
        System.out.println("Hello");
    }
}
"""
        result = self.validator.validate_code(code, 'java')
        
        # Should warn about missing public class
        structure_issues = [issue for issue in result.issues 
                          if 'public class' in issue.message.lower()]
        assert len(structure_issues) > 0
    
    def test_validation_error_handling(self):
        """Test error handling in validation"""
        # This should not crash the validator
        result = self.validator.validate_code(None, 'python')
        
        assert not result.is_valid
        assert len(result.issues) > 0
        assert result.score == 0.0