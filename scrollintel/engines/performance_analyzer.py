"""
Performance Analyzer Engine for Automated Code Generation System

This module provides comprehensive performance analysis and optimization
recommendations for generated code across multiple programming languages.
"""

import re
import ast
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PerformanceImpact(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class PerformanceIssue:
    impact: PerformanceImpact
    category: str
    title: str
    description: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    estimated_impact: Optional[str] = None  # e.g., "10x slower", "O(n²)"
    recommendation: Optional[str] = None
    confidence: float = 1.0

@dataclass
class PerformanceMetrics:
    complexity_score: float  # 0-100 (higher = more complex)
    memory_efficiency: float  # 0-100 (higher = more efficient)
    cpu_efficiency: float  # 0-100 (higher = more efficient)
    scalability_score: float  # 0-100 (higher = more scalable)
    overall_score: float  # 0-100 overall performance score

@dataclass
class PerformanceReport:
    issues: List[PerformanceIssue]
    metrics: PerformanceMetrics
    optimizations: List[str]
    bottlenecks: List[str]
    analysis_metadata: Dict[str, Any]

class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for code optimization recommendations
    """
    
    def __init__(self):
        self.language_analyzers = {
            'python': self._analyze_python_performance,
            'javascript': self._analyze_javascript_performance,
            'typescript': self._analyze_javascript_performance,
            'java': self._analyze_java_performance,
            'sql': self._analyze_sql_performance,
            'go': self._analyze_go_performance
        }
        
        self.complexity_patterns = self._load_complexity_patterns()
        
    def analyze_performance(self, code: str, language: str, context: Optional[Dict] = None) -> PerformanceReport:
        """
        Analyze code performance and provide optimization recommendations
        
        Args:
            code: Source code to analyze
            language: Programming language
            context: Additional context (expected load, data size, etc.)
            
        Returns:
            PerformanceReport with issues, metrics, and recommendations
        """
        try:
            issues = []
            
            # Language-specific performance analysis
            if language.lower() in self.language_analyzers:
                lang_issues = self.language_analyzers[language.lower()](code, context)
                issues.extend(lang_issues)
            
            # General performance patterns
            general_issues = self._analyze_general_performance(code, language)
            issues.extend(general_issues)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(code, language, issues)
            
            # Generate optimization recommendations
            optimizations = self._generate_optimizations(issues, metrics)
            
            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks(issues, code)
            
            return PerformanceReport(
                issues=issues,
                metrics=metrics,
                optimizations=optimizations,
                bottlenecks=bottlenecks,
                analysis_metadata={
                    'language': language,
                    'lines_analyzed': len(code.split('\n')),
                    'analysis_time': time.time()
                }
            )
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {str(e)}")
            return PerformanceReport(
                issues=[PerformanceIssue(
                    impact=PerformanceImpact.HIGH,
                    category="analysis_error",
                    title="Performance Analysis Failed",
                    description=f"Unable to complete performance analysis: {str(e)}"
                )],
                metrics=PerformanceMetrics(0, 0, 0, 0, 0),
                optimizations=["Manual performance review required"],
                bottlenecks=["Analysis failure - manual review needed"],
                analysis_metadata={'error': str(e)}
            )
    
    def _analyze_python_performance(self, code: str, context: Optional[Dict] = None) -> List[PerformanceIssue]:
        """Analyze Python code for performance issues"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            # Analyze loops and nested structures
            for node in ast.walk(tree):
                # Check for nested loops
                if isinstance(node, (ast.For, ast.While)):
                    nested_loops = self._count_nested_loops(node)
                    if nested_loops > 2:
                        issues.append(PerformanceIssue(
                            impact=PerformanceImpact.HIGH,
                            category="algorithmic_complexity",
                            title="Deeply Nested Loops",
                            description=f"Found {nested_loops} levels of nested loops",
                            line_number=node.lineno,
                            estimated_impact=f"O(n^{nested_loops})",
                            recommendation="Consider algorithmic optimization or data structure changes"
                        ))
                
                # Check for inefficient list operations
                if isinstance(node, ast.ListComp):
                    # Check for nested list comprehensions
                    nested_comps = len([n for n in ast.walk(node) if isinstance(n, ast.ListComp)])
                    if nested_comps > 1:
                        issues.append(PerformanceIssue(
                            impact=PerformanceImpact.MEDIUM,
                            category="data_structure_usage",
                            title="Nested List Comprehensions",
                            description="Nested list comprehensions can be memory intensive",
                            line_number=node.lineno,
                            recommendation="Consider using generators or breaking into separate operations"
                        ))
                
                # Check for string concatenation in loops
                if isinstance(node, ast.For):
                    for child in ast.walk(node):
                        if (isinstance(child, ast.AugAssign) and 
                            isinstance(child.op, ast.Add) and
                            self._is_string_operation(child)):
                            issues.append(PerformanceIssue(
                                impact=PerformanceImpact.MEDIUM,
                                category="string_operations",
                                title="String Concatenation in Loop",
                                description="String concatenation in loops is inefficient",
                                line_number=child.lineno,
                                estimated_impact="O(n²) memory usage",
                                recommendation="Use list.join() or io.StringIO instead"
                            ))
            
            # Check for global variable usage
            global_vars = self._find_global_variables(tree)
            if len(global_vars) > 5:
                issues.append(PerformanceIssue(
                    impact=PerformanceImpact.LOW,
                    category="variable_scope",
                    title="Excessive Global Variables",
                    description=f"Found {len(global_vars)} global variables",
                    recommendation="Consider using local variables or class attributes"
                ))
            
        except SyntaxError:
            # Fall back to pattern matching if AST parsing fails
            pass
        
        # Pattern-based analysis
        pattern_issues = self._analyze_python_patterns(code)
        issues.extend(pattern_issues)
        
        return issues
    
    def _analyze_javascript_performance(self, code: str, context: Optional[Dict] = None) -> List[PerformanceIssue]:
        """Analyze JavaScript/TypeScript code for performance issues"""
        issues = []
        
        # Check for DOM manipulation in loops
        if re.search(r'for\s*\([^}]*\{[^}]*document\.(getElementById|querySelector)', code, re.DOTALL):
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.HIGH,
                category="dom_manipulation",
                title="DOM Queries in Loop",
                description="DOM queries inside loops cause performance issues",
                recommendation="Cache DOM elements outside the loop"
            ))
        
        # Check for inefficient array operations
        if re.search(r'\.forEach\s*\([^}]*\.push\s*\(', code):
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.MEDIUM,
                category="array_operations",
                title="Inefficient Array Operations",
                description="Using forEach with push is less efficient than map",
                recommendation="Use map() instead of forEach() with push()"
            ))
        
        # Check for memory leaks patterns
        if re.search(r'setInterval\s*\(', code) and not re.search(r'clearInterval', code):
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.MEDIUM,
                category="memory_management",
                title="Potential Memory Leak",
                description="setInterval without clearInterval can cause memory leaks",
                recommendation="Always clear intervals when no longer needed"
            ))
        
        # Check for synchronous operations that should be async
        sync_patterns = [
            (r'XMLHttpRequest\(\)', "Use fetch() API instead"),
            (r'\.readFileSync\s*\(', "Use async file operations"),
            (r'JSON\.parse\s*\([^)]*\)', "Consider streaming JSON parsing for large data")
        ]
        
        for pattern, recommendation in sync_patterns:
            if re.search(pattern, code):
                issues.append(PerformanceIssue(
                    impact=PerformanceImpact.MEDIUM,
                    category="async_operations",
                    title="Synchronous Operation",
                    description="Synchronous operations can block the event loop",
                    recommendation=recommendation
                ))
        
        return issues
    
    def _analyze_java_performance(self, code: str, context: Optional[Dict] = None) -> List[PerformanceIssue]:
        """Analyze Java code for performance issues"""
        issues = []
        
        # Check for string concatenation in loops
        if re.search(r'for\s*\([^}]*\{[^}]*\+\s*=.*String', code, re.DOTALL):
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.HIGH,
                category="string_operations",
                title="String Concatenation in Loop",
                description="String concatenation in loops creates many objects",
                estimated_impact="O(n²) time complexity",
                recommendation="Use StringBuilder instead"
            ))
        
        # Check for inefficient collections usage
        if re.search(r'Vector\s*<', code):
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.MEDIUM,
                category="data_structure_usage",
                title="Legacy Collection Usage",
                description="Vector is synchronized and slower than ArrayList",
                recommendation="Use ArrayList instead of Vector"
            ))
        
        # Check for autoboxing in loops
        if re.search(r'for\s*\([^}]*Integer.*\+\+', code):
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.MEDIUM,
                category="object_creation",
                title="Autoboxing in Loop",
                description="Autoboxing creates unnecessary objects",
                recommendation="Use primitive types in performance-critical loops"
            ))
        
        return issues
    
    def _analyze_sql_performance(self, code: str, context: Optional[Dict] = None) -> List[PerformanceIssue]:
        """Analyze SQL code for performance issues"""
        issues = []
        
        # Check for SELECT * usage
        if re.search(r'SELECT\s+\*', code, re.IGNORECASE):
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.MEDIUM,
                category="query_optimization",
                title="SELECT * Usage",
                description="SELECT * retrieves unnecessary columns",
                recommendation="Specify only needed columns"
            ))
        
        # Check for missing WHERE clauses
        select_count = len(re.findall(r'SELECT', code, re.IGNORECASE))
        where_count = len(re.findall(r'WHERE', code, re.IGNORECASE))
        
        if select_count > where_count and select_count > 1:
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.HIGH,
                category="query_optimization",
                title="Missing WHERE Clauses",
                description="Queries without WHERE clauses scan entire tables",
                recommendation="Add appropriate WHERE clauses to filter data"
            ))
        
        # Check for LIKE with leading wildcards
        if re.search(r"LIKE\s+['\"]%", code, re.IGNORECASE):
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.HIGH,
                category="indexing",
                title="Leading Wildcard in LIKE",
                description="LIKE with leading % prevents index usage",
                recommendation="Avoid leading wildcards or use full-text search"
            ))
        
        # Check for functions in WHERE clauses
        if re.search(r'WHERE\s+\w+\s*\([^)]*\)\s*=', code, re.IGNORECASE):
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.MEDIUM,
                category="indexing",
                title="Function in WHERE Clause",
                description="Functions in WHERE clauses prevent index usage",
                recommendation="Restructure query to avoid functions on indexed columns"
            ))
        
        return issues
    
    def _analyze_go_performance(self, code: str, context: Optional[Dict] = None) -> List[PerformanceIssue]:
        """Analyze Go code for performance issues"""
        issues = []
        
        # Check for string concatenation in loops
        if re.search(r'for\s+[^{]*\{[^}]*\+=.*string', code, re.DOTALL):
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.HIGH,
                category="string_operations",
                title="String Concatenation in Loop",
                description="String concatenation in Go loops is inefficient",
                recommendation="Use strings.Builder or bytes.Buffer"
            ))
        
        # Check for defer in loops
        if re.search(r'for\s+[^{]*\{[^}]*defer\s+', code, re.DOTALL):
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.MEDIUM,
                category="resource_management",
                title="Defer in Loop",
                description="defer in loops can cause memory buildup",
                recommendation="Move defer outside loop or use explicit cleanup"
            ))
        
        return issues
    
    def _analyze_general_performance(self, code: str, language: str) -> List[PerformanceIssue]:
        """Analyze general performance patterns across languages"""
        issues = []
        
        # Check for deeply nested code
        max_nesting = self._calculate_max_nesting_level(code)
        if max_nesting > 5:
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.MEDIUM,
                category="code_structure",
                title="Deep Nesting",
                description=f"Maximum nesting level: {max_nesting}",
                recommendation="Refactor to reduce nesting complexity"
            ))
        
        # Check for large functions
        functions = self._extract_functions(code, language)
        for func_name, func_lines in functions.items():
            if func_lines > 100:
                issues.append(PerformanceIssue(
                    impact=PerformanceImpact.LOW,
                    category="code_structure",
                    title="Large Function",
                    description=f"Function '{func_name}' has {func_lines} lines",
                    recommendation="Consider breaking large functions into smaller ones"
                ))
        
        return issues
    
    def _analyze_python_patterns(self, code: str) -> List[PerformanceIssue]:
        """Analyze Python-specific performance patterns"""
        issues = []
        
        # Check for inefficient dictionary operations
        if re.search(r'\.keys\(\)\s*in\s+', code):
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.LOW,
                category="data_structure_usage",
                title="Inefficient Dictionary Check",
                description="Using .keys() for membership testing is unnecessary",
                recommendation="Use 'key in dict' instead of 'key in dict.keys()'"
            ))
        
        # Check for list.append() in loops that could use list comprehension
        if re.search(r'for\s+\w+\s+in\s+[^:]+:\s*\w+\.append\(', code):
            issues.append(PerformanceIssue(
                impact=PerformanceImpact.LOW,
                category="pythonic_code",
                title="Loop with Append",
                description="Loop with append could be optimized",
                recommendation="Consider using list comprehension"
            ))
        
        return issues
    
    def _load_complexity_patterns(self) -> Dict[str, List[str]]:
        """Load patterns that indicate algorithmic complexity"""
        return {
            'nested_loops': [
                r'for\s+[^{]*\{[^}]*for\s+[^{]*\{[^}]*for',  # Triple nested
                r'while\s*\([^)]*\)[^{]*\{[^}]*while\s*\([^)]*\)[^{]*\{[^}]*while'
            ],
            'recursive_calls': [
                r'def\s+(\w+)[^{]*\{[^}]*\1\s*\(',
                r'function\s+(\w+)[^{]*\{[^}]*\1\s*\('
            ]
        }
    
    def _count_nested_loops(self, node: ast.AST) -> int:
        """Count the depth of nested loops in an AST node"""
        max_depth = 0
        current_depth = 0
        
        def count_depth(n, depth):
            nonlocal max_depth
            if isinstance(n, (ast.For, ast.While)):
                depth += 1
                max_depth = max(max_depth, depth)
            
            for child in ast.iter_child_nodes(n):
                count_depth(child, depth)
        
        count_depth(node, 0)
        return max_depth
    
    def _is_string_operation(self, node: ast.AST) -> bool:
        """Check if an AST node involves string operations"""
        # Simplified check - in practice, would need more sophisticated analysis
        return True  # Placeholder implementation
    
    def _find_global_variables(self, tree: ast.AST) -> List[str]:
        """Find global variables in Python AST"""
        globals_found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                globals_found.extend(node.names)
        return globals_found
    
    def _calculate_max_nesting_level(self, code: str) -> int:
        """Calculate maximum nesting level in code"""
        max_level = 0
        current_level = 0
        
        for char in code:
            if char == '{':
                current_level += 1
                max_level = max(max_level, current_level)
            elif char == '}':
                current_level = max(0, current_level - 1)
        
        return max_level
    
    def _extract_functions(self, code: str, language: str) -> Dict[str, int]:
        """Extract functions and their line counts"""
        functions = {}
        
        if language.lower() == 'python':
            # Simple regex for Python functions
            pattern = r'def\s+(\w+)\s*\([^)]*\):'
            matches = re.finditer(pattern, code)
            
            for match in matches:
                func_name = match.group(1)
                # Count lines until next function or end
                start_pos = match.end()
                remaining_code = code[start_pos:]
                next_func = re.search(r'\ndef\s+\w+', remaining_code)
                
                if next_func:
                    func_code = remaining_code[:next_func.start()]
                else:
                    func_code = remaining_code
                
                functions[func_name] = len(func_code.split('\n'))
        
        return functions
    
    def _calculate_performance_metrics(self, code: str, language: str, issues: List[PerformanceIssue]) -> PerformanceMetrics:
        """Calculate performance metrics based on code analysis"""
        
        # Calculate complexity score (higher = more complex)
        complexity_factors = 0
        for issue in issues:
            if issue.category == "algorithmic_complexity":
                complexity_factors += 20
            elif issue.category == "code_structure":
                complexity_factors += 10
        
        complexity_score = min(100, complexity_factors)
        
        # Calculate efficiency scores
        efficiency_deductions = 0
        for issue in issues:
            if issue.impact == PerformanceImpact.CRITICAL:
                efficiency_deductions += 25
            elif issue.impact == PerformanceImpact.HIGH:
                efficiency_deductions += 15
            elif issue.impact == PerformanceImpact.MEDIUM:
                efficiency_deductions += 8
        
        cpu_efficiency = max(0, 100 - efficiency_deductions)
        memory_efficiency = max(0, 100 - efficiency_deductions * 0.8)
        
        # Calculate scalability score
        scalability_issues = [i for i in issues if 'loop' in i.category or 'complexity' in i.category]
        scalability_score = max(0, 100 - len(scalability_issues) * 15)
        
        # Overall score
        overall_score = (cpu_efficiency + memory_efficiency + scalability_score) / 3
        
        return PerformanceMetrics(
            complexity_score=complexity_score,
            memory_efficiency=memory_efficiency,
            cpu_efficiency=cpu_efficiency,
            scalability_score=scalability_score,
            overall_score=overall_score
        )
    
    def _generate_optimizations(self, issues: List[PerformanceIssue], metrics: PerformanceMetrics) -> List[str]:
        """Generate optimization recommendations"""
        optimizations = []
        
        # Category-based optimizations
        categories = {}
        for issue in issues:
            categories[issue.category] = categories.get(issue.category, 0) + 1
        
        if 'algorithmic_complexity' in categories:
            optimizations.append("Review algorithms for better time complexity")
        
        if 'string_operations' in categories:
            optimizations.append("Optimize string operations using appropriate data structures")
        
        if 'data_structure_usage' in categories:
            optimizations.append("Choose more efficient data structures")
        
        if 'dom_manipulation' in categories:
            optimizations.append("Minimize DOM operations and batch updates")
        
        # Metric-based optimizations
        if metrics.cpu_efficiency < 70:
            optimizations.append("Focus on CPU-intensive operations optimization")
        
        if metrics.memory_efficiency < 70:
            optimizations.append("Implement memory usage optimizations")
        
        if metrics.scalability_score < 60:
            optimizations.append("Address scalability concerns for larger datasets")
        
        return optimizations
    
    def _identify_bottlenecks(self, issues: List[PerformanceIssue], code: str) -> List[str]:
        """Identify potential performance bottlenecks"""
        bottlenecks = []
        
        # High-impact issues are likely bottlenecks
        critical_issues = [i for i in issues if i.impact == PerformanceImpact.CRITICAL]
        high_issues = [i for i in issues if i.impact == PerformanceImpact.HIGH]
        
        for issue in critical_issues:
            bottlenecks.append(f"Critical: {issue.title} (Line {issue.line_number or 'unknown'})")
        
        for issue in high_issues:
            bottlenecks.append(f"High Impact: {issue.title} (Line {issue.line_number or 'unknown'})")
        
        # Code size bottlenecks
        lines = len(code.split('\n'))
        if lines > 1000:
            bottlenecks.append(f"Large codebase ({lines} lines) may impact compilation/loading time")
        
        return bottlenecks