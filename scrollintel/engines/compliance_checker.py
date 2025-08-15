"""
Compliance Checker Engine for Automated Code Generation System

This module provides comprehensive compliance checking for coding standards,
best practices, and organizational guidelines across multiple programming languages.
"""

import re
import ast
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ComplianceSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ComplianceViolation:
    severity: ComplianceSeverity
    category: str
    rule_id: str
    title: str
    description: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    standard: Optional[str] = None  # e.g., "PEP8", "ESLint", "Google Style"

@dataclass
class ComplianceMetrics:
    total_violations: int
    violations_by_severity: Dict[str, int]
    violations_by_category: Dict[str, int]
    compliance_score: float  # 0-100 (higher = more compliant)
    standards_coverage: Dict[str, float]  # Coverage per standard

@dataclass
class ComplianceReport:
    violations: List[ComplianceViolation]
    metrics: ComplianceMetrics
    recommendations: List[str]
    standards_applied: List[str]
    analysis_metadata: Dict[str, Any]

class ComplianceChecker:
    """
    Comprehensive compliance checker for coding standards and best practices
    """
    
    def __init__(self):
        self.standards = self._load_coding_standards()
        self.language_checkers = {
            'python': self._check_python_compliance,
            'javascript': self._check_javascript_compliance,
            'typescript': self._check_typescript_compliance,
            'java': self._check_java_compliance,
            'sql': self._check_sql_compliance,
            'go': self._check_go_compliance
        }
        
    def check_compliance(self, code: str, language: str, 
                        standards: Optional[List[str]] = None,
                        context: Optional[Dict] = None) -> ComplianceReport:
        """
        Check code compliance against coding standards and best practices
        
        Args:
            code: Source code to check
            language: Programming language
            standards: Specific standards to apply (if None, uses defaults)
            context: Additional context (project type, team preferences, etc.)
            
        Returns:
            ComplianceReport with violations and recommendations
        """
        try:
            violations = []
            applied_standards = standards or self._get_default_standards(language)
            
            # Language-specific compliance checking
            if language.lower() in self.language_checkers:
                lang_violations = self.language_checkers[language.lower()](
                    code, applied_standards, context
                )
                violations.extend(lang_violations)
            
            # General compliance patterns
            general_violations = self._check_general_compliance(code, language, applied_standards)
            violations.extend(general_violations)
            
            # Calculate metrics
            metrics = self._calculate_compliance_metrics(violations, applied_standards)
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(violations, metrics)
            
            return ComplianceReport(
                violations=violations,
                metrics=metrics,
                recommendations=recommendations,
                standards_applied=applied_standards,
                analysis_metadata={
                    'language': language,
                    'lines_checked': len(code.split('\n')),
                    'standards_count': len(applied_standards)
                }
            )
            
        except Exception as e:
            logger.error(f"Compliance check failed: {str(e)}")
            return ComplianceReport(
                violations=[ComplianceViolation(
                    severity=ComplianceSeverity.HIGH,
                    category="check_error",
                    rule_id="COMPLIANCE_ERROR",
                    title="Compliance Check Failed",
                    description=f"Unable to complete compliance check: {str(e)}"
                )],
                metrics=ComplianceMetrics(1, {'high': 1}, {'check_error': 1}, 0.0, {}),
                recommendations=["Manual compliance review required"],
                standards_applied=[],
                analysis_metadata={'error': str(e)}
            )
    
    def _load_coding_standards(self) -> Dict[str, Dict[str, Any]]:
        """Load coding standards and rules for different languages"""
        return {
            'python': {
                'PEP8': {
                    'line_length': {'max': 79, 'severity': ComplianceSeverity.MEDIUM},
                    'indentation': {'spaces': 4, 'severity': ComplianceSeverity.HIGH},
                    'naming_conventions': {
                        'function': r'^[a-z_][a-z0-9_]*$',
                        'class': r'^[A-Z][a-zA-Z0-9]*$',
                        'constant': r'^[A-Z_][A-Z0-9_]*$',
                        'severity': ComplianceSeverity.MEDIUM
                    },
                    'imports': {
                        'order': ['standard', 'third_party', 'local'],
                        'severity': ComplianceSeverity.LOW
                    }
                },
                'PEP257': {
                    'docstrings': {
                        'required_for': ['class', 'function', 'module'],
                        'severity': ComplianceSeverity.MEDIUM
                    }
                }
            },
            'javascript': {
                'ESLint': {
                    'semicolons': {'required': True, 'severity': ComplianceSeverity.MEDIUM},
                    'quotes': {'style': 'single', 'severity': ComplianceSeverity.LOW},
                    'var_declaration': {'prefer': 'const_let', 'severity': ComplianceSeverity.HIGH},
                    'function_style': {'prefer': 'arrow', 'severity': ComplianceSeverity.LOW}
                },
                'Airbnb': {
                    'line_length': {'max': 100, 'severity': ComplianceSeverity.MEDIUM},
                    'trailing_comma': {'required': True, 'severity': ComplianceSeverity.LOW}
                }
            },
            'java': {
                'Google': {
                    'line_length': {'max': 100, 'severity': ComplianceSeverity.MEDIUM},
                    'indentation': {'spaces': 2, 'severity': ComplianceSeverity.HIGH},
                    'naming_conventions': {
                        'class': r'^[A-Z][a-zA-Z0-9]*$',
                        'method': r'^[a-z][a-zA-Z0-9]*$',
                        'constant': r'^[A-Z_][A-Z0-9_]*$',
                        'severity': ComplianceSeverity.MEDIUM
                    }
                },
                'Oracle': {
                    'braces': {'style': 'egyptian', 'severity': ComplianceSeverity.LOW},
                    'javadoc': {'required_for': ['public'], 'severity': ComplianceSeverity.MEDIUM}
                }
            }
        }
    
    def _get_default_standards(self, language: str) -> List[str]:
        """Get default standards for a language"""
        defaults = {
            'python': ['PEP8', 'PEP257'],
            'javascript': ['ESLint'],
            'typescript': ['ESLint'],
            'java': ['Google'],
            'sql': ['ANSI'],
            'go': ['Go_Standard']
        }
        return defaults.get(language.lower(), [])
    
    def _check_python_compliance(self, code: str, standards: List[str], 
                                context: Optional[Dict] = None) -> List[ComplianceViolation]:
        """Check Python code compliance"""
        violations = []
        
        try:
            tree = ast.parse(code)
            lines = code.split('\n')
            
            # PEP8 checks
            if 'PEP8' in standards:
                violations.extend(self._check_pep8_compliance(code, lines, tree))
            
            # PEP257 docstring checks
            if 'PEP257' in standards:
                violations.extend(self._check_pep257_compliance(tree))
            
            # General Python best practices
            violations.extend(self._check_python_best_practices(tree, lines))
            
        except SyntaxError as e:
            violations.append(ComplianceViolation(
                severity=ComplianceSeverity.CRITICAL,
                category="syntax",
                rule_id="SYNTAX_ERROR",
                title="Syntax Error",
                description=f"Code contains syntax error: {e.msg}",
                line_number=e.lineno,
                standard="Python"
            ))
        
        return violations
    
    def _check_pep8_compliance(self, code: str, lines: List[str], tree: ast.AST) -> List[ComplianceViolation]:
        """Check PEP8 compliance"""
        violations = []
        
        # Line length check
        for i, line in enumerate(lines, 1):
            if len(line) > 79:
                violations.append(ComplianceViolation(
                    severity=ComplianceSeverity.MEDIUM,
                    category="formatting",
                    rule_id="E501",
                    title="Line Too Long",
                    description=f"Line {i} has {len(line)} characters (max 79)",
                    line_number=i,
                    code_snippet=line[:50] + "..." if len(line) > 50 else line,
                    recommendation="Break long lines or use parentheses for continuation",
                    standard="PEP8"
                ))
        
        # Indentation check (simplified)
        for i, line in enumerate(lines, 1):
            if line.strip() and not line.startswith('#'):
                leading_spaces = len(line) - len(line.lstrip(' '))
                if leading_spaces % 4 != 0 and leading_spaces > 0:
                    violations.append(ComplianceViolation(
                        severity=ComplianceSeverity.HIGH,
                        category="formatting",
                        rule_id="E111",
                        title="Indentation Not Multiple of Four",
                        description=f"Line {i} has {leading_spaces} leading spaces",
                        line_number=i,
                        recommendation="Use 4 spaces per indentation level",
                        standard="PEP8"
                    ))
        
        # Naming convention checks
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    violations.append(ComplianceViolation(
                        severity=ComplianceSeverity.MEDIUM,
                        category="naming",
                        rule_id="N802",
                        title="Function Name Not Snake Case",
                        description=f"Function '{node.name}' should use snake_case",
                        line_number=node.lineno,
                        recommendation="Use lowercase with underscores for function names",
                        standard="PEP8"
                    ))
            
            elif isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    violations.append(ComplianceViolation(
                        severity=ComplianceSeverity.MEDIUM,
                        category="naming",
                        rule_id="N801",
                        title="Class Name Not PascalCase",
                        description=f"Class '{node.name}' should use PascalCase",
                        line_number=node.lineno,
                        recommendation="Use PascalCase for class names",
                        standard="PEP8"
                    ))
        
        # Import organization check
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append((node.lineno, node))
        
        if len(imports) > 1:
            # Check if imports are grouped properly
            prev_line = 0
            for line_no, import_node in imports:
                if line_no - prev_line > 2:  # Gap indicates grouping
                    continue
                prev_line = line_no
        
        return violations
    
    def _check_pep257_compliance(self, tree: ast.AST) -> List[ComplianceViolation]:
        """Check PEP257 docstring compliance"""
        violations = []
        
        # Check module docstring
        if (isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, ast.Str)):
            # Module has docstring
            pass
        else:
            violations.append(ComplianceViolation(
                severity=ComplianceSeverity.MEDIUM,
                category="documentation",
                rule_id="D100",
                title="Missing Module Docstring",
                description="Module should have a docstring",
                line_number=1,
                recommendation="Add a module-level docstring",
                standard="PEP257"
            ))
        
        # Check class and function docstrings
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not self._has_docstring(node):
                    violations.append(ComplianceViolation(
                        severity=ComplianceSeverity.MEDIUM,
                        category="documentation",
                        rule_id="D101",
                        title="Missing Class Docstring",
                        description=f"Class '{node.name}' should have a docstring",
                        line_number=node.lineno,
                        recommendation="Add a class docstring describing its purpose",
                        standard="PEP257"
                    ))
            
            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_') and not self._has_docstring(node):
                    violations.append(ComplianceViolation(
                        severity=ComplianceSeverity.MEDIUM,
                        category="documentation",
                        rule_id="D102",
                        title="Missing Function Docstring",
                        description=f"Function '{node.name}' should have a docstring",
                        line_number=node.lineno,
                        recommendation="Add a function docstring describing parameters and return value",
                        standard="PEP257"
                    ))
        
        return violations
    
    def _check_python_best_practices(self, tree: ast.AST, lines: List[str]) -> List[ComplianceViolation]:
        """Check Python best practices"""
        violations = []
        
        # Check for bare except clauses
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    violations.append(ComplianceViolation(
                        severity=ComplianceSeverity.HIGH,
                        category="error_handling",
                        rule_id="E722",
                        title="Bare Except Clause",
                        description="Bare except clauses catch all exceptions",
                        line_number=node.lineno,
                        recommendation="Specify exception types to catch",
                        standard="Python Best Practices"
                    ))
        
        # Check for mutable default arguments
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        violations.append(ComplianceViolation(
                            severity=ComplianceSeverity.HIGH,
                            category="logic",
                            rule_id="B006",
                            title="Mutable Default Argument",
                            description=f"Function '{node.name}' has mutable default argument",
                            line_number=node.lineno,
                            recommendation="Use None as default and create mutable object inside function",
                            standard="Python Best Practices"
                        ))
        
        return violations
    
    def _check_javascript_compliance(self, code: str, standards: List[str], 
                                   context: Optional[Dict] = None) -> List[ComplianceViolation]:
        """Check JavaScript/TypeScript code compliance"""
        violations = []
        lines = code.split('\n')
        
        # ESLint-style checks
        if 'ESLint' in standards:
            violations.extend(self._check_eslint_compliance(code, lines))
        
        # Airbnb style checks
        if 'Airbnb' in standards:
            violations.extend(self._check_airbnb_compliance(code, lines))
        
        return violations
    
    def _check_eslint_compliance(self, code: str, lines: List[str]) -> List[ComplianceViolation]:
        """Check ESLint compliance"""
        violations = []
        
        # Check for var usage
        for i, line in enumerate(lines, 1):
            if re.search(r'\bvar\s+', line):
                violations.append(ComplianceViolation(
                    severity=ComplianceSeverity.HIGH,
                    category="variable_declaration",
                    rule_id="no-var",
                    title="Use of 'var'",
                    description="Use 'let' or 'const' instead of 'var'",
                    line_number=i,
                    code_snippet=line.strip(),
                    recommendation="Replace 'var' with 'let' or 'const'",
                    standard="ESLint"
                ))
        
        # Check for missing semicolons
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if (stripped and 
                not stripped.startswith('//') and 
                not stripped.startswith('/*') and
                not stripped.endswith(';') and
                not stripped.endswith('{') and
                not stripped.endswith('}') and
                re.search(r'(return|break|continue|throw)\s+', stripped)):
                violations.append(ComplianceViolation(
                    severity=ComplianceSeverity.MEDIUM,
                    category="syntax",
                    rule_id="semi",
                    title="Missing Semicolon",
                    description="Statement should end with semicolon",
                    line_number=i,
                    code_snippet=stripped,
                    recommendation="Add semicolon at end of statement",
                    standard="ESLint"
                ))
        
        # Check for console.log (should not be in production)
        for i, line in enumerate(lines, 1):
            if re.search(r'console\.(log|warn|error)', line):
                violations.append(ComplianceViolation(
                    severity=ComplianceSeverity.LOW,
                    category="debugging",
                    rule_id="no-console",
                    title="Console Statement",
                    description="Console statements should not be in production code",
                    line_number=i,
                    code_snippet=line.strip(),
                    recommendation="Remove console statements or use proper logging",
                    standard="ESLint"
                ))
        
        return violations
    
    def _check_airbnb_compliance(self, code: str, lines: List[str]) -> List[ComplianceViolation]:
        """Check Airbnb style guide compliance"""
        violations = []
        
        # Line length check
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                violations.append(ComplianceViolation(
                    severity=ComplianceSeverity.MEDIUM,
                    category="formatting",
                    rule_id="max-len",
                    title="Line Too Long",
                    description=f"Line {i} exceeds 100 characters",
                    line_number=i,
                    recommendation="Break long lines for better readability",
                    standard="Airbnb"
                ))
        
        return violations
    
    def _check_typescript_compliance(self, code: str, standards: List[str], 
                                   context: Optional[Dict] = None) -> List[ComplianceViolation]:
        """Check TypeScript-specific compliance"""
        violations = []
        
        # Run JavaScript checks first
        violations.extend(self._check_javascript_compliance(code, standards, context))
        
        # TypeScript-specific checks
        lines = code.split('\n')
        
        # Check for 'any' type usage
        for i, line in enumerate(lines, 1):
            if re.search(r':\s*any\b', line):
                violations.append(ComplianceViolation(
                    severity=ComplianceSeverity.MEDIUM,
                    category="typing",
                    rule_id="no-any",
                    title="Use of 'any' Type",
                    description="Avoid using 'any' type",
                    line_number=i,
                    code_snippet=line.strip(),
                    recommendation="Use specific types instead of 'any'",
                    standard="TypeScript"
                ))
        
        return violations
    
    def _check_java_compliance(self, code: str, standards: List[str], 
                              context: Optional[Dict] = None) -> List[ComplianceViolation]:
        """Check Java code compliance"""
        violations = []
        lines = code.split('\n')
        
        # Google Java Style checks
        if 'Google' in standards:
            violations.extend(self._check_google_java_style(code, lines))
        
        return violations
    
    def _check_google_java_style(self, code: str, lines: List[str]) -> List[ComplianceViolation]:
        """Check Google Java Style Guide compliance"""
        violations = []
        
        # Line length check
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                violations.append(ComplianceViolation(
                    severity=ComplianceSeverity.MEDIUM,
                    category="formatting",
                    rule_id="LineLength",
                    title="Line Too Long",
                    description=f"Line {i} exceeds 100 characters",
                    line_number=i,
                    recommendation="Break long lines",
                    standard="Google Java Style"
                ))
        
        # Check class naming
        class_pattern = r'class\s+([A-Z][a-zA-Z0-9]*)'
        for i, line in enumerate(lines, 1):
            match = re.search(class_pattern, line)
            if match:
                class_name = match.group(1)
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                    violations.append(ComplianceViolation(
                        severity=ComplianceSeverity.MEDIUM,
                        category="naming",
                        rule_id="ClassNames",
                        title="Invalid Class Name",
                        description=f"Class name '{class_name}' should be PascalCase",
                        line_number=i,
                        recommendation="Use PascalCase for class names",
                        standard="Google Java Style"
                    ))
        
        return violations
    
    def _check_sql_compliance(self, code: str, standards: List[str], 
                             context: Optional[Dict] = None) -> List[ComplianceViolation]:
        """Check SQL code compliance"""
        violations = []
        
        # Check for uppercase keywords
        keywords = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'JOIN']
        for keyword in keywords:
            if re.search(rf'\b{keyword.lower()}\b', code):
                violations.append(ComplianceViolation(
                    severity=ComplianceSeverity.LOW,
                    category="formatting",
                    rule_id="UppercaseKeywords",
                    title="Lowercase SQL Keywords",
                    description=f"SQL keyword '{keyword.lower()}' should be uppercase",
                    recommendation="Use uppercase for SQL keywords",
                    standard="SQL Style"
                ))
        
        return violations
    
    def _check_go_compliance(self, code: str, standards: List[str], 
                            context: Optional[Dict] = None) -> List[ComplianceViolation]:
        """Check Go code compliance"""
        violations = []
        lines = code.split('\n')
        
        # Check for gofmt compliance (simplified)
        for i, line in enumerate(lines, 1):
            # Check for tabs vs spaces (Go prefers tabs)
            if line.startswith('    ') and not line.startswith('\t'):
                violations.append(ComplianceViolation(
                    severity=ComplianceSeverity.MEDIUM,
                    category="formatting",
                    rule_id="gofmt",
                    title="Use Tabs for Indentation",
                    description="Go code should use tabs for indentation",
                    line_number=i,
                    recommendation="Use tabs instead of spaces for indentation",
                    standard="Go Standard"
                ))
        
        return violations
    
    def _check_general_compliance(self, code: str, language: str, 
                                 standards: List[str]) -> List[ComplianceViolation]:
        """Check general compliance patterns across languages"""
        violations = []
        lines = code.split('\n')
        
        # Check for trailing whitespace
        for i, line in enumerate(lines, 1):
            if line.endswith(' ') or line.endswith('\t'):
                violations.append(ComplianceViolation(
                    severity=ComplianceSeverity.LOW,
                    category="formatting",
                    rule_id="trailing-whitespace",
                    title="Trailing Whitespace",
                    description=f"Line {i} has trailing whitespace",
                    line_number=i,
                    recommendation="Remove trailing whitespace",
                    standard="General"
                ))
        
        # Check for mixed line endings (simplified check)
        if '\r\n' in code and '\n' in code.replace('\r\n', ''):
            violations.append(ComplianceViolation(
                severity=ComplianceSeverity.MEDIUM,
                category="formatting",
                rule_id="mixed-line-endings",
                title="Mixed Line Endings",
                description="File contains mixed line endings",
                recommendation="Use consistent line endings throughout the file",
                standard="General"
            ))
        
        return violations
    
    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if an AST node has a docstring"""
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            return True
        return False
    
    def _calculate_compliance_metrics(self, violations: List[ComplianceViolation], 
                                    standards: List[str]) -> ComplianceMetrics:
        """Calculate compliance metrics"""
        total_violations = len(violations)
        
        # Count by severity
        severity_counts = {}
        for violation in violations:
            severity = violation.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by category
        category_counts = {}
        for violation in violations:
            category = violation.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Calculate compliance score (0-100)
        base_score = 100.0
        for violation in violations:
            if violation.severity == ComplianceSeverity.CRITICAL:
                base_score -= 10
            elif violation.severity == ComplianceSeverity.HIGH:
                base_score -= 5
            elif violation.severity == ComplianceSeverity.MEDIUM:
                base_score -= 2
            elif violation.severity == ComplianceSeverity.LOW:
                base_score -= 1
        
        compliance_score = max(0.0, base_score)
        
        # Standards coverage (simplified)
        standards_coverage = {}
        for standard in standards:
            standard_violations = [v for v in violations if v.standard == standard]
            # Higher coverage = fewer violations for that standard
            coverage = max(0, 100 - len(standard_violations) * 5)
            standards_coverage[standard] = coverage
        
        return ComplianceMetrics(
            total_violations=total_violations,
            violations_by_severity=severity_counts,
            violations_by_category=category_counts,
            compliance_score=compliance_score,
            standards_coverage=standards_coverage
        )
    
    def _generate_compliance_recommendations(self, violations: List[ComplianceViolation], 
                                           metrics: ComplianceMetrics) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Priority recommendations based on severity
        critical_count = metrics.violations_by_severity.get('critical', 0)
        high_count = metrics.violations_by_severity.get('high', 0)
        
        if critical_count > 0:
            recommendations.append(f"Fix {critical_count} critical compliance violations immediately")
        
        if high_count > 0:
            recommendations.append(f"Address {high_count} high-priority compliance issues")
        
        # Category-specific recommendations
        top_categories = sorted(metrics.violations_by_category.items(), 
                               key=lambda x: x[1], reverse=True)[:3]
        
        for category, count in top_categories:
            if count > 2:
                if category == "formatting":
                    recommendations.append("Use automated code formatting tools")
                elif category == "naming":
                    recommendations.append("Review and standardize naming conventions")
                elif category == "documentation":
                    recommendations.append("Add missing documentation and docstrings")
                elif category == "error_handling":
                    recommendations.append("Improve error handling practices")
        
        # Overall score recommendations
        if metrics.compliance_score < 70:
            recommendations.append("Consider comprehensive code review and refactoring")
        elif metrics.compliance_score < 85:
            recommendations.append("Address remaining compliance issues for better code quality")
        
        # Standards-specific recommendations
        for standard, coverage in metrics.standards_coverage.items():
            if coverage < 80:
                recommendations.append(f"Improve compliance with {standard} standard")
        
        return recommendations
    
    def generate_compliance_report_json(self, report: ComplianceReport) -> str:
        """Generate JSON report for compliance check results"""
        report_data = {
            'summary': {
                'total_violations': report.metrics.total_violations,
                'compliance_score': report.metrics.compliance_score,
                'violations_by_severity': report.metrics.violations_by_severity,
                'violations_by_category': report.metrics.violations_by_category
            },
            'violations': [
                {
                    'severity': violation.severity.value,
                    'category': violation.category,
                    'rule_id': violation.rule_id,
                    'title': violation.title,
                    'description': violation.description,
                    'line_number': violation.line_number,
                    'recommendation': violation.recommendation,
                    'standard': violation.standard
                }
                for violation in report.violations
            ],
            'recommendations': report.recommendations,
            'standards_applied': report.standards_applied,
            'standards_coverage': report.metrics.standards_coverage,
            'metadata': report.analysis_metadata
        }
        
        return json.dumps(report_data, indent=2)