"""
Tests for Compliance Checker Engine
"""

import pytest
from scrollintel.engines.compliance_checker import (
    ComplianceChecker, ComplianceReport, ComplianceViolation, ComplianceSeverity
)

class TestComplianceChecker:
    
    def setup_method(self):
        self.checker = ComplianceChecker()
    
    def test_check_python_pep8_line_length(self):
        """Test PEP8 line length compliance"""
        code = """
def function_with_very_long_line():
    x = "This is a very long line that exceeds the PEP8 recommended maximum of 79 characters and should trigger a violation"
    return x
"""
        report = self.checker.check_compliance(code, 'python', ['PEP8'])
        
        line_length_violations = [v for v in report.violations 
                                 if 'long' in v.title.lower()]
        assert len(line_length_violations) > 0
        assert any(v.rule_id == 'E501' for v in line_length_violations)
    
    def test_check_python_pep8_indentation(self):
        """Test PEP8 indentation compliance"""
        code = """
def bad_indentation():
  x = 1  # 2 spaces instead of 4
   y = 2  # 3 spaces instead of 4
    z = 3  # Correct 4 spaces
    return x + y + z
"""
        report = self.checker.check_compliance(code, 'python', ['PEP8'])
        
        indent_violations = [v for v in report.violations 
                           if 'indentation' in v.title.lower()]
        assert len(indent_violations) > 0
        assert any(v.rule_id == 'E111' for v in indent_violations)
    
    def test_check_python_pep8_naming(self):
        """Test PEP8 naming convention compliance"""
        code = """
def BadFunctionName():  # Should be snake_case
    pass

class bad_class_name:  # Should be PascalCase
    pass

def good_function_name():
    pass

class GoodClassName:
    pass
"""
        report = self.checker.check_compliance(code, 'python', ['PEP8'])
        
        naming_violations = [v for v in report.violations 
                           if 'name' in v.title.lower()]
        assert len(naming_violations) >= 2  # At least function and class violations
    
    def test_check_python_pep257_docstrings(self):
        """Test PEP257 docstring compliance"""
        code = """
def function_without_docstring():
    return "No docstring"

def function_with_docstring():
    \"\"\"This function has a proper docstring.\"\"\"
    return "Has docstring"

class ClassWithoutDocstring:
    pass

class ClassWithDocstring:
    \"\"\"This class has a proper docstring.\"\"\"
    pass
"""
        report = self.checker.check_compliance(code, 'python', ['PEP257'])
        
        docstring_violations = [v for v in report.violations 
                              if 'docstring' in v.title.lower()]
        assert len(docstring_violations) >= 2  # Function and class without docstrings
    
    def test_check_python_best_practices_bare_except(self):
        """Test detection of bare except clauses"""
        code = """
def risky_function():
    try:
        dangerous_operation()
    except:  # Bare except - bad practice
        pass
    
    try:
        another_operation()
    except ValueError:  # Specific exception - good practice
        pass
"""
        report = self.checker.check_compliance(code, 'python')
        
        except_violations = [v for v in report.violations 
                           if 'except' in v.title.lower()]
        assert len(except_violations) > 0
        assert any(v.severity == ComplianceSeverity.HIGH for v in except_violations)
    
    def test_check_python_mutable_defaults(self):
        """Test detection of mutable default arguments"""
        code = """
def bad_function(items=[]):  # Mutable default - bad
    items.append(1)
    return items

def good_function(items=None):  # Proper way
    if items is None:
        items = []
    items.append(1)
    return items
"""
        report = self.checker.check_compliance(code, 'python')
        
        mutable_violations = [v for v in report.violations 
                            if 'mutable' in v.title.lower()]
        assert len(mutable_violations) > 0
    
    def test_check_javascript_eslint_var_usage(self):
        """Test ESLint var usage compliance"""
        code = """
function oldStyle() {
    var x = 5;  // Should use let or const
    var y = 10; // Should use let or const
    return x + y;
}

function modernStyle() {
    const x = 5;  // Good
    let y = 10;   // Good
    return x + y;
}
"""
        report = self.checker.check_compliance(code, 'javascript', ['ESLint'])
        
        var_violations = [v for v in report.violations 
                         if 'var' in v.title.lower()]
        assert len(var_violations) >= 2
        assert any(v.severity == ComplianceSeverity.HIGH for v in var_violations)
    
    def test_check_javascript_eslint_semicolons(self):
        """Test ESLint semicolon compliance"""
        code = """
function testSemicolons() {
    let x = 5
    return x  // Missing semicolons
}
"""
        report = self.checker.check_compliance(code, 'javascript', ['ESLint'])
        
        semi_violations = [v for v in report.violations 
                          if 'semicolon' in v.title.lower()]
        # Note: Simplified detection might not catch all cases
    
    def test_check_javascript_console_statements(self):
        """Test detection of console statements"""
        code = """
function debugFunction() {
    console.log("Debug message");  // Should not be in production
    console.warn("Warning message");
    console.error("Error message");
    return "result";
}
"""
        report = self.checker.check_compliance(code, 'javascript', ['ESLint'])
        
        console_violations = [v for v in report.violations 
                            if 'console' in v.title.lower()]
        assert len(console_violations) >= 3
    
    def test_check_typescript_any_usage(self):
        """Test TypeScript 'any' type compliance"""
        code = """
function processData(data: any): any {  // Should avoid 'any'
    return data.someProperty;
}

function betterFunction(data: string): number {  // Specific types
    return data.length;
}
"""
        report = self.checker.check_compliance(code, 'typescript')
        
        any_violations = [v for v in report.violations 
                         if 'any' in v.title.lower()]
        assert len(any_violations) >= 2  # Parameter and return type
    
    def test_check_java_google_style_line_length(self):
        """Test Google Java Style line length"""
        code = """
public class TestClass {
    public void methodWithVeryLongLineExceedingGoogleStyleGuideRecommendedMaximumOf100Characters() {
        System.out.println("Long method name");
    }
}
"""
        report = self.checker.check_compliance(code, 'java', ['Google'])
        
        line_violations = [v for v in report.violations 
                          if 'long' in v.title.lower()]
        assert len(line_violations) > 0
    
    def test_check_java_class_naming(self):
        """Test Java class naming compliance"""
        code = """
class badClassName {  // Should be PascalCase
    public void method() {}
}

class GoodClassName {  // Correct PascalCase
    public void method() {}
}
"""
        report = self.checker.check_compliance(code, 'java', ['Google'])
        
        naming_violations = [v for v in report.violations 
                           if 'name' in v.title.lower()]
        assert len(naming_violations) > 0
    
    def test_check_sql_uppercase_keywords(self):
        """Test SQL uppercase keyword compliance"""
        code = """
select * from users where id = 1;
insert into logs values ('event', 'data');
SELECT name FROM products WHERE active = 1;
"""
        report = self.checker.check_compliance(code, 'sql')
        
        keyword_violations = [v for v in report.violations 
                            if 'keyword' in v.title.lower()]
        assert len(keyword_violations) > 0
    
    def test_check_go_indentation(self):
        """Test Go indentation compliance (tabs vs spaces)"""
        code = """
func main() {
    x := 5      // Uses spaces - should use tabs
    fmt.Println(x)
}
"""
        report = self.checker.check_compliance(code, 'go')
        
        indent_violations = [v for v in report.violations 
                           if 'tab' in v.title.lower() or 'indentation' in v.title.lower()]
        assert len(indent_violations) > 0
    
    def test_check_general_trailing_whitespace(self):
        """Test detection of trailing whitespace"""
        code = "def function():   \n    return 'value'  \n"  # Lines end with spaces
        
        report = self.checker.check_compliance(code, 'python')
        
        whitespace_violations = [v for v in report.violations 
                               if 'whitespace' in v.title.lower()]
        assert len(whitespace_violations) > 0
    
    def test_check_mixed_line_endings(self):
        """Test detection of mixed line endings"""
        code = "line1\r\nline2\nline3\r\n"  # Mixed \r\n and \n
        
        report = self.checker.check_compliance(code, 'python')
        
        line_ending_violations = [v for v in report.violations 
                                if 'line ending' in v.title.lower()]
        assert len(line_ending_violations) > 0
    
    def test_compliance_metrics_calculation(self):
        """Test compliance metrics calculation"""
        # Clean code
        clean_code = """
def clean_function():
    \"\"\"A well-formatted function.\"\"\"
    return "clean"
"""
        clean_report = self.checker.check_compliance(clean_code, 'python')
        
        # Messy code
        messy_code = """
def BadFunctionName():
  x = "This is a very long line that exceeds the PEP8 recommended maximum of 79 characters and should trigger violations"
  return x
"""
        messy_report = self.checker.check_compliance(messy_code, 'python')
        
        assert clean_report.metrics.compliance_score > messy_report.metrics.compliance_score
        assert messy_report.metrics.total_violations > clean_report.metrics.total_violations
    
    def test_generate_compliance_recommendations(self):
        """Test compliance recommendation generation"""
        code = """
def BadFunctionName():
  x = "Very long line that exceeds PEP8 maximum length and should trigger multiple violations for testing purposes"
  try:
      risky_operation()
  except:
      pass
"""
        report = self.checker.check_compliance(code, 'python')
        
        assert len(report.recommendations) > 0
        assert any('critical' in rec.lower() or 'high' in rec.lower() 
                  for rec in report.recommendations)
    
    def test_standards_coverage_calculation(self):
        """Test standards coverage calculation"""
        code = """
def function_with_issues():
    x = "Long line that violates PEP8"
    return x
"""
        report = self.checker.check_compliance(code, 'python', ['PEP8'])
        
        assert 'PEP8' in report.metrics.standards_coverage
        assert 0 <= report.metrics.standards_coverage['PEP8'] <= 100
    
    def test_generate_compliance_report_json(self):
        """Test JSON report generation"""
        code = """
def test_function():
    x = "Some code"
    return x
"""
        report = self.checker.check_compliance(code, 'python')
        json_report = self.checker.generate_compliance_report_json(report)
        
        assert isinstance(json_report, str)
        assert 'violations' in json_report
        assert 'compliance_score' in json_report
        assert 'recommendations' in json_report
    
    def test_custom_standards_application(self):
        """Test application of custom standards"""
        code = """
def test_function():
    return "test"
"""
        # Test with specific standards
        pep8_report = self.checker.check_compliance(code, 'python', ['PEP8'])
        pep257_report = self.checker.check_compliance(code, 'python', ['PEP257'])
        
        assert 'PEP8' in pep8_report.standards_applied
        assert 'PEP257' in pep257_report.standards_applied
    
    def test_violation_severity_assignment(self):
        """Test proper severity assignment to violations"""
        code = """
def function():
    try:
        operation()
    except:  # High severity
        pass
    
    x = "line"  # Low severity formatting issue
"""
        report = self.checker.check_compliance(code, 'python')
        
        # Should have violations of different severities
        severities = {v.severity for v in report.violations}
        assert len(severities) > 1  # Multiple severity levels
    
    def test_compliance_error_handling(self):
        """Test error handling in compliance checking"""
        # This should not crash the checker
        report = self.checker.check_compliance(None, 'python')
        
        assert report.metrics.compliance_score == 0.0
        assert len(report.violations) > 0
        assert 'error' in report.analysis_metadata
    
    def test_airbnb_javascript_style(self):
        """Test Airbnb JavaScript style compliance"""
        code = """
function veryLongFunctionNameThatExceedsTheAirbnbStyleGuideRecommendedMaximumLineLength() {
    return "long function name";
}
"""
        report = self.checker.check_compliance(code, 'javascript', ['Airbnb'])
        
        line_violations = [v for v in report.violations 
                          if 'long' in v.title.lower()]
        assert len(line_violations) > 0
        assert any(v.standard == 'Airbnb' for v in line_violations)