"""
Component Assessment Engine

Analyzes components for code quality, security, performance, and production readiness.
"""

import ast
import asyncio
import os
import re
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
import logging

from .production_upgrade_framework import (
    ComponentAssessment, Issue, Recommendation, UpgradeCategory, 
    Priority, Severity, RiskLevel
)

logger = logging.getLogger(__name__)


class CodeQualityAnalyzer:
    """Analyzes code quality metrics"""
    
    def __init__(self):
        self.complexity_threshold = 10
        self.line_length_threshold = 120
        
    async def analyze(self, component_path: str) -> Dict[str, float]:
        """Analyze code quality metrics for a component"""
        try:
            metrics = {
                'complexity_score': await self._analyze_complexity(component_path),
                'style_score': await self._analyze_style(component_path),
                'structure_score': await self._analyze_structure(component_path),
                'maintainability_score': await self._analyze_maintainability(component_path)
            }
            
            # Calculate overall score
            overall_score = sum(metrics.values()) / len(metrics)
            metrics['overall_score'] = overall_score
            
            return metrics
            
        except Exception as e:
            logger.error(f"Code quality analysis failed for {component_path}: {str(e)}")
            return {'overall_score': 0.0}
    
    async def _analyze_complexity(self, component_path: str) -> float:
        """Analyze cyclomatic complexity"""
        if not os.path.exists(component_path):
            return 0.0
            
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            complexity_scores = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = self._calculate_cyclomatic_complexity(node)
                    complexity_scores.append(min(complexity / self.complexity_threshold, 1.0))
            
            if not complexity_scores:
                return 8.0  # Default score for files without functions
                
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            return max(0.0, 10.0 - (avg_complexity * 10.0))
            
        except Exception as e:
            logger.warning(f"Complexity analysis failed for {component_path}: {str(e)}")
            return 5.0
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
                
        return complexity
    
    async def _analyze_style(self, component_path: str) -> float:
        """Analyze code style compliance"""
        if not os.path.exists(component_path):
            return 0.0
            
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            style_violations = 0
            total_checks = 0
            
            for line_num, line in enumerate(lines, 1):
                total_checks += 1
                
                # Check line length
                if len(line.rstrip()) > self.line_length_threshold:
                    style_violations += 1
                
                # Check for proper indentation (4 spaces)
                if line.startswith(' ') and not line.startswith('    '):
                    if len(line) - len(line.lstrip()) % 4 != 0:
                        style_violations += 1
            
            if total_checks == 0:
                return 8.0
                
            compliance_rate = 1.0 - (style_violations / total_checks)
            return compliance_rate * 10.0
            
        except Exception as e:
            logger.warning(f"Style analysis failed for {component_path}: {str(e)}")
            return 5.0
    
    async def _analyze_structure(self, component_path: str) -> float:
        """Analyze code structure and organization"""
        if not os.path.exists(component_path):
            return 0.0
            
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Check for proper imports organization
            imports_score = self._check_imports_organization(tree)
            
            # Check for proper class/function organization
            organization_score = self._check_code_organization(tree)
            
            # Check for proper docstrings
            documentation_score = self._check_documentation(tree)
            
            return (imports_score + organization_score + documentation_score) / 3.0
            
        except Exception as e:
            logger.warning(f"Structure analysis failed for {component_path}: {str(e)}")
            return 5.0
    
    def _check_imports_organization(self, tree: ast.AST) -> float:
        """Check if imports are properly organized"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node.lineno)
        
        if not imports:
            return 8.0
            
        # Check if imports are at the top
        first_non_import = None
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Module)):
                if hasattr(node, 'lineno'):
                    first_non_import = node.lineno
                    break
        
        if first_non_import and any(imp > first_non_import for imp in imports):
            return 6.0  # Imports mixed with code
            
        return 9.0
    
    def _check_code_organization(self, tree: ast.AST) -> float:
        """Check code organization patterns"""
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.lineno)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.lineno)
        
        # Basic organization check - classes before standalone functions
        if classes and functions:
            if any(func < max(classes) for func in functions):
                return 7.0  # Mixed organization
                
        return 8.5
    
    def _check_documentation(self, tree: ast.AST) -> float:
        """Check for proper documentation"""
        documented_items = 0
        total_items = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                total_items += 1
                if ast.get_docstring(node):
                    documented_items += 1
        
        if total_items == 0:
            return 8.0
            
        documentation_rate = documented_items / total_items
        return documentation_rate * 10.0
    
    async def _analyze_maintainability(self, component_path: str) -> float:
        """Analyze code maintainability factors"""
        if not os.path.exists(component_path):
            return 0.0
            
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for code smells
            code_smells = 0
            
            # Long parameter lists
            if re.search(r'def \w+\([^)]{100,}\)', content):
                code_smells += 1
            
            # Deep nesting (more than 4 levels)
            lines = content.split('\n')
            for line in lines:
                indent_level = (len(line) - len(line.lstrip())) // 4
                if indent_level > 4:
                    code_smells += 1
                    break
            
            # Duplicate code patterns
            if len(re.findall(r'(\w+.*\n.*\w+.*\n.*\w+.*)', content)) > 5:
                code_smells += 1
            
            # Calculate maintainability score
            maintainability_score = max(0.0, 10.0 - code_smells)
            return maintainability_score
            
        except Exception as e:
            logger.warning(f"Maintainability analysis failed for {component_path}: {str(e)}")
            return 5.0


class SecurityScanner:
    """Scans for security vulnerabilities and issues"""
    
    def __init__(self):
        self.security_patterns = [
            (r'eval\s*\(', 'Use of eval() function', Severity.CRITICAL),
            (r'exec\s*\(', 'Use of exec() function', Severity.CRITICAL),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', 'Shell injection risk', Severity.CRITICAL),
            (r'pickle\.loads?\s*\(', 'Unsafe pickle usage', Severity.ERROR),
            (r'input\s*\([^)]*\)', 'Use of input() function', Severity.WARNING),
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password', Severity.CRITICAL),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key', Severity.CRITICAL),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret', Severity.CRITICAL),
        ]
    
    async def scan(self, component_path: str) -> Dict[str, Any]:
        """Perform security scan on component"""
        try:
            vulnerabilities = await self._scan_vulnerabilities(component_path)
            security_score = await self._calculate_security_score(vulnerabilities)
            
            return {
                'vulnerabilities': vulnerabilities,
                'security_score': security_score,
                'critical_count': len([v for v in vulnerabilities if v['severity'] == 'critical']),
                'high_count': len([v for v in vulnerabilities if v['severity'] == 'error']),
                'medium_count': len([v for v in vulnerabilities if v['severity'] == 'warning'])
            }
            
        except Exception as e:
            logger.error(f"Security scan failed for {component_path}: {str(e)}")
            return {'security_score': 0.0, 'vulnerabilities': []}
    
    async def _scan_vulnerabilities(self, component_path: str) -> List[Dict[str, Any]]:
        """Scan for security vulnerabilities"""
        if not os.path.exists(component_path):
            return []
            
        vulnerabilities = []
        
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for pattern, description, severity in self.security_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        vulnerabilities.append({
                            'line': line_num,
                            'description': description,
                            'severity': severity.value,
                            'code': line.strip(),
                            'pattern': pattern
                        })
            
            return vulnerabilities
            
        except Exception as e:
            logger.warning(f"Vulnerability scan failed for {component_path}: {str(e)}")
            return []
    
    async def _calculate_security_score(self, vulnerabilities: List[Dict[str, Any]]) -> float:
        """Calculate overall security score"""
        if not vulnerabilities:
            return 9.5
        
        # Weight vulnerabilities by severity
        severity_weights = {
            'critical': 10.0,
            'error': 5.0,
            'warning': 2.0,
            'info': 1.0
        }
        
        total_weight = sum(severity_weights.get(v['severity'], 1.0) for v in vulnerabilities)
        
        # Calculate score (10 - penalty)
        penalty = min(total_weight * 0.5, 10.0)
        return max(0.0, 10.0 - penalty)


class PerformanceProfiler:
    """Profiles component performance characteristics"""
    
    async def profile(self, component_path: str) -> Dict[str, float]:
        """Profile component performance"""
        try:
            metrics = {
                'import_time': await self._measure_import_time(component_path),
                'memory_efficiency': await self._analyze_memory_patterns(component_path),
                'algorithmic_efficiency': await self._analyze_algorithmic_complexity(component_path),
                'io_efficiency': await self._analyze_io_patterns(component_path)
            }
            
            # Calculate overall performance score
            overall_score = sum(metrics.values()) / len(metrics)
            metrics['overall_score'] = overall_score
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance profiling failed for {component_path}: {str(e)}")
            return {'overall_score': 5.0}
    
    async def _measure_import_time(self, component_path: str) -> float:
        """Measure import time performance"""
        # For now, return a default score
        # In a real implementation, this would measure actual import time
        return 8.0
    
    async def _analyze_memory_patterns(self, component_path: str) -> float:
        """Analyze memory usage patterns"""
        if not os.path.exists(component_path):
            return 0.0
            
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for potential memory issues
            memory_issues = 0
            
            # Large data structures
            if re.search(r'\[\s*.*\s*for\s+.*\s+in\s+.*\]', content):
                if len(re.findall(r'\[.*for.*in.*\]', content)) > 5:
                    memory_issues += 1
            
            # Potential memory leaks
            if 'global ' in content:
                memory_issues += 1
            
            # Calculate score
            return max(0.0, 10.0 - memory_issues * 2.0)
            
        except Exception as e:
            logger.warning(f"Memory analysis failed for {component_path}: {str(e)}")
            return 5.0
    
    async def _analyze_algorithmic_complexity(self, component_path: str) -> float:
        """Analyze algorithmic complexity"""
        if not os.path.exists(component_path):
            return 0.0
            
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for nested loops (potential O(n²) or worse)
            nested_loops = len(re.findall(r'for\s+.*:\s*\n\s*.*for\s+.*:', content, re.MULTILINE))
            
            # Look for recursive patterns
            recursive_calls = len(re.findall(r'def\s+(\w+).*:\s*.*\1\s*\(', content, re.MULTILINE))
            
            complexity_penalty = nested_loops * 2 + recursive_calls * 1
            return max(0.0, 10.0 - complexity_penalty)
            
        except Exception as e:
            logger.warning(f"Algorithmic analysis failed for {component_path}: {str(e)}")
            return 5.0
    
    async def _analyze_io_patterns(self, component_path: str) -> float:
        """Analyze I/O efficiency patterns"""
        if not os.path.exists(component_path):
            return 0.0
            
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for synchronous I/O in async contexts
            io_issues = 0
            
            if 'async def' in content and 'open(' in content:
                if 'aiofiles' not in content:
                    io_issues += 1
            
            # Look for database queries in loops
            if re.search(r'for\s+.*:\s*.*\.(query|execute|fetch)', content, re.MULTILINE):
                io_issues += 1
            
            return max(0.0, 10.0 - io_issues * 3.0)
            
        except Exception as e:
            logger.warning(f"I/O analysis failed for {component_path}: {str(e)}")
            return 5.0


class ComponentAssessmentEngine:
    """Main engine for assessing component production readiness"""
    
    def __init__(self):
        self.code_analyzer = CodeQualityAnalyzer()
        self.security_scanner = SecurityScanner()
        self.performance_profiler = PerformanceProfiler()
    
    async def assess(self, component_path: str) -> ComponentAssessment:
        """
        Perform comprehensive assessment of a component
        
        Args:
            component_path: Path to the component to assess
            
        Returns:
            ComponentAssessment with detailed analysis results
        """
        try:
            logger.info(f"Starting assessment of component: {component_path}")
            
            # Perform parallel analysis
            code_quality_task = self.code_analyzer.analyze(component_path)
            security_task = self.security_scanner.scan(component_path)
            performance_task = self.performance_profiler.profile(component_path)
            
            code_quality, security_results, performance_results = await asyncio.gather(
                code_quality_task, security_task, performance_task
            )
            
            # Calculate scores
            code_quality_score = code_quality.get('overall_score', 0.0)
            security_score = security_results.get('security_score', 0.0)
            performance_score = performance_results.get('overall_score', 0.0)
            
            # Calculate test coverage (placeholder)
            test_coverage = await self._calculate_test_coverage(component_path)
            
            # Calculate documentation score
            documentation_score = await self._calculate_documentation_score(component_path)
            
            # Calculate overall production readiness score
            production_readiness_score = (
                code_quality_score * 0.25 +
                security_score * 0.25 +
                performance_score * 0.20 +
                test_coverage * 0.20 +
                documentation_score * 0.10
            )
            
            # Identify issues and recommendations
            issues = await self._identify_issues(component_path, code_quality, security_results, performance_results)
            recommendations = await self._generate_recommendations(issues, production_readiness_score)
            
            # Get current version (placeholder)
            current_version = "1.0.0"
            
            assessment = ComponentAssessment(
                component_path=component_path,
                current_version=current_version,
                code_quality_score=code_quality_score,
                security_score=security_score,
                performance_score=performance_score,
                test_coverage=test_coverage,
                documentation_score=documentation_score,
                production_readiness_score=production_readiness_score,
                identified_issues=issues,
                upgrade_recommendations=recommendations
            )
            
            logger.info(f"Assessment completed. Production readiness score: {production_readiness_score:.2f}")
            return assessment
            
        except Exception as e:
            logger.error(f"Assessment failed for {component_path}: {str(e)}")
            raise
    
    async def _calculate_test_coverage(self, component_path: str) -> float:
        """Calculate test coverage for the component"""
        # Placeholder implementation
        # In a real implementation, this would run coverage tools
        return 75.0
    
    async def _calculate_documentation_score(self, component_path: str) -> float:
        """Calculate documentation completeness score"""
        if not os.path.exists(component_path):
            return 0.0
            
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Count documented vs undocumented items
            total_items = 0
            documented_items = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_items += 1
                    if ast.get_docstring(node):
                        documented_items += 1
            
            if total_items == 0:
                return 8.0
                
            documentation_rate = documented_items / total_items
            return documentation_rate * 10.0
            
        except Exception as e:
            logger.warning(f"Documentation analysis failed for {component_path}: {str(e)}")
            return 5.0
    
    async def _identify_issues(self, component_path: str, code_quality: Dict, 
                             security_results: Dict, performance_results: Dict) -> List[Issue]:
        """Identify specific issues in the component"""
        issues = []
        
        # Code quality issues
        if code_quality.get('overall_score', 0) < 7.0:
            issues.append(Issue(
                id=f"cq_{len(issues)}",
                category=UpgradeCategory.CODE_QUALITY,
                severity=Severity.WARNING,
                description="Code quality score below acceptable threshold",
                file_path=component_path,
                suggested_fix="Refactor code to improve readability and maintainability"
            ))
        
        # Security issues
        for vuln in security_results.get('vulnerabilities', []):
            severity_map = {
                'critical': Severity.CRITICAL,
                'error': Severity.ERROR,
                'warning': Severity.WARNING,
                'info': Severity.INFO
            }
            
            issues.append(Issue(
                id=f"sec_{len(issues)}",
                category=UpgradeCategory.SECURITY,
                severity=severity_map.get(vuln['severity'], Severity.WARNING),
                description=vuln['description'],
                file_path=component_path,
                line_number=vuln.get('line'),
                suggested_fix="Review and remediate security vulnerability"
            ))
        
        # Performance issues
        if performance_results.get('overall_score', 0) < 7.0:
            issues.append(Issue(
                id=f"perf_{len(issues)}",
                category=UpgradeCategory.PERFORMANCE,
                severity=Severity.WARNING,
                description="Performance score below acceptable threshold",
                file_path=component_path,
                suggested_fix="Optimize algorithms and resource usage"
            ))
        
        return issues
    
    async def _generate_recommendations(self, issues: List[Issue], 
                                      production_readiness_score: float) -> List[Recommendation]:
        """Generate upgrade recommendations based on assessment"""
        recommendations = []
        
        # Critical issues first
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        if critical_issues:
            recommendations.append(Recommendation(
                id="rec_critical",
                category=UpgradeCategory.SECURITY,
                priority=Priority.CRITICAL,
                title="Address Critical Security Issues",
                description="Resolve all critical security vulnerabilities immediately",
                estimated_effort=timedelta(hours=8),
                benefits=["Eliminate security risks", "Meet compliance requirements"]
            ))
        
        # Code quality improvements
        code_quality_issues = [i for i in issues if i.category == UpgradeCategory.CODE_QUALITY]
        if code_quality_issues:
            recommendations.append(Recommendation(
                id="rec_code_quality",
                category=UpgradeCategory.CODE_QUALITY,
                priority=Priority.HIGH,
                title="Improve Code Quality",
                description="Refactor code to meet production quality standards",
                estimated_effort=timedelta(hours=16),
                benefits=["Improved maintainability", "Reduced technical debt"]
            ))
        
        # Add error handling if missing
        recommendations.append(Recommendation(
            id="rec_error_handling",
            category=UpgradeCategory.ERROR_HANDLING,
            priority=Priority.HIGH,
            title="Implement Comprehensive Error Handling",
            description="Add proper error handling and recovery mechanisms",
            estimated_effort=timedelta(hours=12),
            benefits=["Improved reliability", "Better user experience"]
        ))
        
        # Add monitoring if missing
        recommendations.append(Recommendation(
            id="rec_monitoring",
            category=UpgradeCategory.MONITORING,
            priority=Priority.MEDIUM,
            title="Add Production Monitoring",
            description="Implement logging, metrics, and health checks",
            estimated_effort=timedelta(hours=8),
            benefits=["Better observability", "Faster issue detection"]
        ))
        
        return recommendations