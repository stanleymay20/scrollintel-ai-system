"""
Code Validator Engine for Automated Code Generation System

This module provides comprehensive code validation including syntax checking,
logic validation, and structural analysis for generated code.
"""

import ast
import re
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationIssue:
    severity: ValidationSeverity
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    rule: Optional[str] = None
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue]
    score: float  # 0-100 quality score
    metrics: Dict[str, Any]

class CodeValidator:
    """
    Comprehensive code validator with syntax and logic validation capabilities
    """
    
    def __init__(self):
        self.language_validators = {
            'python': self._validate_python,
            'javascript': self._validate_javascript,
            'typescript': self._validate_typescript,
            'java': self._validate_java,
            'sql': self._validate_sql
        }
        
    def validate_code(self, code: str, language: str, context: Optional[Dict] = None) -> ValidationResult:
        """
        Validate code for syntax, logic, and structural issues
        
        Args:
            code: Source code to validate
            language: Programming language
            context: Additional context for validation
            
        Returns:
            ValidationResult with issues and quality metrics
        """
        try:
            issues = []
            metrics = {}
            
            # Language-specific validation
            if language.lower() in self.language_validators:
                lang_issues, lang_metrics = self.language_validators[language.lower()](code, context)
                issues.extend(lang_issues)
                metrics.update(lang_metrics)
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"No specific validator for unsupported language: {language}",
                    rule="unsupported_language"
                ))
            
            # General code quality checks
            general_issues, general_metrics = self._validate_general_quality(code, language)
            issues.extend(general_issues)
            metrics.update(general_metrics)
            
            # Calculate quality score
            score = self._calculate_quality_score(issues, metrics)
            
            # Determine if code is valid (no errors)
            is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
            
            return ValidationResult(
                is_valid=is_valid,
                issues=issues,
                score=score,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Code validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation error: {str(e)}",
                    rule="validation_exception"
                )],
                score=0.0,
                metrics={}
            )
    
    def _validate_python(self, code: str, context: Optional[Dict] = None) -> Tuple[List[ValidationIssue], Dict]:
        """Validate Python code"""
        issues = []
        metrics = {}
        
        try:
            # Parse AST for syntax validation
            tree = ast.parse(code)
            metrics['ast_nodes'] = len(list(ast.walk(tree)))
            
            # Check for common Python issues
            issues.extend(self._check_python_best_practices(tree, code))
            
        except SyntaxError as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno,
                column=e.offset,
                rule="syntax_error"
            ))
        
        # Check imports and dependencies
        import_issues = self._validate_python_imports(code)
        issues.extend(import_issues)
        
        return issues, metrics
    
    def _validate_javascript(self, code: str, context: Optional[Dict] = None) -> Tuple[List[ValidationIssue], Dict]:
        """Validate JavaScript code"""
        issues = []
        metrics = {}
        
        # Basic syntax checks
        js_issues = self._check_javascript_syntax(code)
        issues.extend(js_issues)
        
        # Check for common JS patterns
        pattern_issues = self._check_javascript_patterns(code)
        issues.extend(pattern_issues)
        
        metrics['lines_of_code'] = len(code.split('\n'))
        
        return issues, metrics
    
    def _validate_typescript(self, code: str, context: Optional[Dict] = None) -> Tuple[List[ValidationIssue], Dict]:
        """Validate TypeScript code"""
        issues = []
        metrics = {}
        
        # TypeScript-specific validation
        type_issues = self._check_typescript_types(code)
        issues.extend(type_issues)
        
        # Also run JavaScript validation
        js_issues, js_metrics = self._validate_javascript(code, context)
        issues.extend(js_issues)
        metrics.update(js_metrics)
        
        return issues, metrics
    
    def _validate_java(self, code: str, context: Optional[Dict] = None) -> Tuple[List[ValidationIssue], Dict]:
        """Validate Java code"""
        issues = []
        metrics = {}
        
        # Basic Java syntax and structure checks
        java_issues = self._check_java_structure(code)
        issues.extend(java_issues)
        
        metrics['classes'] = len(re.findall(r'class\s+\w+', code))
        metrics['methods'] = len(re.findall(r'(public|private|protected).*?\w+\s*\(', code))
        
        return issues, metrics
    
    def _validate_sql(self, code: str, context: Optional[Dict] = None) -> Tuple[List[ValidationIssue], Dict]:
        """Validate SQL code"""
        issues = []
        metrics = {}
        
        # SQL syntax and best practices
        sql_issues = self._check_sql_patterns(code)
        issues.extend(sql_issues)
        
        metrics['statements'] = len(re.findall(r';', code))
        
        return issues, metrics
    
    def _validate_general_quality(self, code: str, language: str) -> Tuple[List[ValidationIssue], Dict]:
        """General code quality validation"""
        issues = []
        metrics = {}
        
        lines = code.split('\n')
        metrics['total_lines'] = len(lines)
        metrics['non_empty_lines'] = len([line for line in lines if line.strip()])
        
        # Check line length
        long_lines = [(i+1, line) for i, line in enumerate(lines) if len(line) > 120]
        for line_num, line in long_lines:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Line too long ({len(line)} characters)",
                line_number=line_num,
                rule="line_length",
                suggestion="Consider breaking long lines for better readability"
            ))
        
        # Check for TODO/FIXME comments
        todo_pattern = re.compile(r'(TODO|FIXME|HACK)', re.IGNORECASE)
        for i, line in enumerate(lines):
            if todo_pattern.search(line):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Found TODO/FIXME comment",
                    line_number=i+1,
                    rule="todo_comment"
                ))
        
        return issues, metrics
    
    def _check_python_best_practices(self, tree: ast.AST, code: str) -> List[ValidationIssue]:
        """Check Python-specific best practices"""
        issues = []
        
        # Check for unused imports (simplified)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append(alias.name)
        
        # Check function complexity
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_cyclomatic_complexity(node)
                if complexity > 10:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Function '{node.name}' has high complexity ({complexity})",
                        line_number=node.lineno,
                        rule="high_complexity",
                        suggestion="Consider breaking down into smaller functions"
                    ))
        
        return issues
    
    def _validate_python_imports(self, code: str) -> List[ValidationIssue]:
        """Validate Python imports"""
        issues = []
        
        # Check for relative imports in generated code
        if re.search(r'from\s+\.', code):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Relative imports found - ensure proper package structure",
                rule="relative_imports"
            ))
        
        return issues
    
    def _check_javascript_syntax(self, code: str) -> List[ValidationIssue]:
        """Basic JavaScript syntax validation"""
        issues = []
        
        # Check for common syntax issues
        if code.count('{') != code.count('}'):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Mismatched braces",
                rule="syntax_error"
            ))
        
        if code.count('(') != code.count(')'):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Mismatched parentheses",
                rule="syntax_error"
            ))
        
        return issues
    
    def _check_javascript_patterns(self, code: str) -> List[ValidationIssue]:
        """Check JavaScript patterns and best practices"""
        issues = []
        
        # Check for var usage (prefer let/const)
        if re.search(r'\bvar\s+', code):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Use 'let' or 'const' instead of 'var'",
                rule="var_usage",
                suggestion="Replace 'var' with 'let' or 'const'"
            ))
        
        return issues
    
    def _check_typescript_types(self, code: str) -> List[ValidationIssue]:
        """Check TypeScript type annotations"""
        issues = []
        
        # Check for 'any' type usage
        if re.search(r':\s*any\b', code):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Avoid using 'any' type",
                rule="any_type",
                suggestion="Use specific types instead of 'any'"
            ))
        
        return issues
    
    def _check_java_structure(self, code: str) -> List[ValidationIssue]:
        """Check Java code structure"""
        issues = []
        
        # Check for public class
        if not re.search(r'public\s+class\s+\w+', code):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No public class found",
                rule="java_structure"
            ))
        
        return issues
    
    def _check_sql_patterns(self, code: str) -> List[ValidationIssue]:
        """Check SQL patterns and best practices"""
        issues = []
        
        # Check for SELECT *
        if re.search(r'SELECT\s+\*', code, re.IGNORECASE):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Avoid SELECT * in production code",
                rule="select_star",
                suggestion="Specify column names explicitly"
            ))
        
        return issues
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_quality_score(self, issues: List[ValidationIssue], metrics: Dict) -> float:
        """Calculate overall quality score (0-100)"""
        base_score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.ERROR:
                base_score -= 20
            elif issue.severity == ValidationSeverity.WARNING:
                base_score -= 5
            elif issue.severity == ValidationSeverity.INFO:
                base_score -= 1
        
        return max(0.0, base_score)
    
    def get_improvement_suggestions(self, validation_result: ValidationResult) -> List[str]:
        """Generate improvement suggestions based on validation results"""
        suggestions = []
        
        error_count = sum(1 for issue in validation_result.issues 
                         if issue.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for issue in validation_result.issues 
                           if issue.severity == ValidationSeverity.WARNING)
        
        if error_count > 0:
            suggestions.append(f"Fix {error_count} syntax/logic errors before deployment")
        
        if warning_count > 5:
            suggestions.append("Consider addressing warnings to improve code quality")
        
        if validation_result.score < 70:
            suggestions.append("Code quality is below recommended threshold - review and refactor")
        
        # Add specific suggestions from issues
        for issue in validation_result.issues:
            if issue.suggestion:
                suggestions.append(issue.suggestion)
        
        return list(set(suggestions))  # Remove duplicates