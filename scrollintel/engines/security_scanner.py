"""
Security Scanner Engine for Automated Code Generation System

This module provides comprehensive security vulnerability detection
for generated code across multiple programming languages.
"""

import re
import ast
import hashlib
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SecuritySeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class SecurityVulnerability:
    severity: SecuritySeverity
    category: str
    title: str
    description: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID
    recommendation: Optional[str] = None
    confidence: float = 1.0  # 0-1 confidence score

@dataclass
class SecurityReport:
    vulnerabilities: List[SecurityVulnerability]
    risk_score: float  # 0-100 risk score
    summary: Dict[str, int]  # Count by severity
    recommendations: List[str]
    scan_metadata: Dict[str, Any]

class SecurityScanner:
    """
    Comprehensive security scanner for vulnerability detection in generated code
    """
    
    def __init__(self):
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        self.language_scanners = {
            'python': self._scan_python_security,
            'javascript': self._scan_javascript_security,
            'typescript': self._scan_javascript_security,  # Same patterns as JS
            'java': self._scan_java_security,
            'sql': self._scan_sql_security,
            'php': self._scan_php_security
        }
        
    def scan_code(self, code: str, language: str, context: Optional[Dict] = None) -> SecurityReport:
        """
        Scan code for security vulnerabilities
        
        Args:
            code: Source code to scan
            language: Programming language
            context: Additional context (framework, dependencies, etc.)
            
        Returns:
            SecurityReport with vulnerabilities and recommendations
        """
        try:
            vulnerabilities = []
            
            # Language-specific security scanning
            if language.lower() in self.language_scanners:
                lang_vulns = self.language_scanners[language.lower()](code, context)
                vulnerabilities.extend(lang_vulns)
            
            # General security patterns
            general_vulns = self._scan_general_security_patterns(code, language)
            vulnerabilities.extend(general_vulns)
            
            # Calculate risk score and summary
            risk_score = self._calculate_risk_score(vulnerabilities)
            summary = self._generate_summary(vulnerabilities)
            recommendations = self._generate_recommendations(vulnerabilities)
            
            return SecurityReport(
                vulnerabilities=vulnerabilities,
                risk_score=risk_score,
                summary=summary,
                recommendations=recommendations,
                scan_metadata={
                    'language': language,
                    'lines_scanned': len(code.split('\n')),
                    'patterns_checked': len(self.vulnerability_patterns.get(language.lower(), {}))
                }
            )
            
        except Exception as e:
            logger.error(f"Security scan failed: {str(e)}")
            return SecurityReport(
                vulnerabilities=[SecurityVulnerability(
                    severity=SecuritySeverity.HIGH,
                    category="scan_error",
                    title="Security Scan Failed",
                    description=f"Unable to complete security scan: {str(e)}"
                )],
                risk_score=100.0,  # Assume high risk if scan fails
                summary={'critical': 0, 'high': 1, 'medium': 0, 'low': 0, 'info': 0},
                recommendations=["Manual security review required due to scan failure"],
                scan_metadata={'error': str(e)}
            )
    
    def _load_vulnerability_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load vulnerability detection patterns for different languages"""
        return {
            'python': {
                'sql_injection': {
                    'patterns': [
                        r'execute\s*\(\s*["\'].*%.*["\']',
                        r'cursor\.execute\s*\(\s*["\'].*\+.*["\']',
                        r'query\s*=.*\+.*',
                        r'\.format\s*\(.*\).*execute'
                    ],
                    'severity': SecuritySeverity.CRITICAL,
                    'cwe': 'CWE-89'
                },
                'command_injection': {
                    'patterns': [
                        r'os\.system\s*\(',
                        r'subprocess\.call\s*\(',
                        r'eval\s*\(',
                        r'exec\s*\('
                    ],
                    'severity': SecuritySeverity.HIGH,
                    'cwe': 'CWE-78'
                },
                'hardcoded_secrets': {
                    'patterns': [
                        r'password\s*=\s*["\'][^"\']+["\']',
                        r'api_key\s*=\s*["\'][^"\']+["\']',
                        r'secret\s*=\s*["\'][^"\']+["\']',
                        r'token\s*=\s*["\'][^"\']+["\']'
                    ],
                    'severity': SecuritySeverity.HIGH,
                    'cwe': 'CWE-798'
                }
            },
            'javascript': {
                'xss_vulnerability': {
                    'patterns': [
                        r'innerHTML\s*=.*\+',
                        r'document\.write\s*\(',
                        r'eval\s*\(',
                        r'setTimeout\s*\(\s*["\'].*\+.*["\']'
                    ],
                    'severity': SecuritySeverity.HIGH,
                    'cwe': 'CWE-79'
                },
                'prototype_pollution': {
                    'patterns': [
                        r'__proto__',
                        r'constructor\.prototype',
                        r'Object\.prototype'
                    ],
                    'severity': SecuritySeverity.MEDIUM,
                    'cwe': 'CWE-1321'
                }
            },
            'sql': {
                'sql_injection': {
                    'patterns': [
                        r'WHERE.*=.*\+',
                        r'SELECT.*\+.*FROM',
                        r'INSERT.*VALUES.*\+'
                    ],
                    'severity': SecuritySeverity.CRITICAL,
                    'cwe': 'CWE-89'
                }
            }
        }
    
    def _scan_python_security(self, code: str, context: Optional[Dict] = None) -> List[SecurityVulnerability]:
        """Scan Python code for security vulnerabilities"""
        vulnerabilities = []
        
        try:
            # Parse AST for deeper analysis
            tree = ast.parse(code)
            
            # Check for dangerous function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        
                        # Check for eval/exec usage
                        if func_name in ['eval', 'exec']:
                            vulnerabilities.append(SecurityVulnerability(
                                severity=SecuritySeverity.HIGH,
                                category="code_injection",
                                title="Dangerous Function Usage",
                                description=f"Use of {func_name}() can lead to code injection",
                                line_number=node.lineno,
                                cwe_id="CWE-94",
                                recommendation=f"Avoid using {func_name}() with user input"
                            ))
                    
                    elif isinstance(node.func, ast.Attribute):
                        # Check for os.system calls
                        if (isinstance(node.func.value, ast.Name) and 
                            node.func.value.id == 'os' and 
                            node.func.attr == 'system'):
                            vulnerabilities.append(SecurityVulnerability(
                                severity=SecuritySeverity.HIGH,
                                category="command_injection",
                                title="Command Injection Risk",
                                description="os.system() usage can lead to command injection",
                                line_number=node.lineno,
                                cwe_id="CWE-78",
                                recommendation="Use subprocess with shell=False instead"
                            ))
            
            # Check imports for potentially dangerous modules
            dangerous_imports = ['pickle', 'marshal', 'shelve']
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_imports:
                            vulnerabilities.append(SecurityVulnerability(
                                severity=SecuritySeverity.MEDIUM,
                                category="insecure_deserialization",
                                title="Potentially Unsafe Import",
                                description=f"Import of {alias.name} can be unsafe with untrusted data",
                                line_number=node.lineno,
                                cwe_id="CWE-502",
                                recommendation="Validate data before deserialization"
                            ))
        
        except SyntaxError:
            # If AST parsing fails, fall back to pattern matching
            pass
        
        # Pattern-based scanning
        pattern_vulns = self._scan_with_patterns(code, 'python')
        vulnerabilities.extend(pattern_vulns)
        
        return vulnerabilities
    
    def _scan_javascript_security(self, code: str, context: Optional[Dict] = None) -> List[SecurityVulnerability]:
        """Scan JavaScript/TypeScript code for security vulnerabilities"""
        vulnerabilities = []
        
        # Check for DOM manipulation vulnerabilities
        if re.search(r'innerHTML\s*=.*\+', code):
            vulnerabilities.append(SecurityVulnerability(
                severity=SecuritySeverity.HIGH,
                category="xss",
                title="Potential XSS Vulnerability",
                description="Dynamic innerHTML assignment can lead to XSS",
                cwe_id="CWE-79",
                recommendation="Use textContent or sanitize input"
            ))
        
        # Check for eval usage
        if re.search(r'\beval\s*\(', code):
            vulnerabilities.append(SecurityVulnerability(
                severity=SecuritySeverity.HIGH,
                category="code_injection",
                title="Code Injection Risk",
                description="eval() usage can lead to code injection",
                cwe_id="CWE-94",
                recommendation="Avoid eval() or use safer alternatives"
            ))
        
        # Check for prototype pollution
        if re.search(r'__proto__', code):
            vulnerabilities.append(SecurityVulnerability(
                severity=SecuritySeverity.MEDIUM,
                category="prototype_pollution",
                title="Prototype Pollution Risk",
                description="Direct __proto__ manipulation can be dangerous",
                cwe_id="CWE-1321",
                recommendation="Use Object.create() or Object.setPrototypeOf()"
            ))
        
        # Pattern-based scanning
        pattern_vulns = self._scan_with_patterns(code, 'javascript')
        vulnerabilities.extend(pattern_vulns)
        
        return vulnerabilities
    
    def _scan_java_security(self, code: str, context: Optional[Dict] = None) -> List[SecurityVulnerability]:
        """Scan Java code for security vulnerabilities"""
        vulnerabilities = []
        
        # Check for SQL injection patterns
        if re.search(r'Statement.*executeQuery.*\+', code):
            vulnerabilities.append(SecurityVulnerability(
                severity=SecuritySeverity.CRITICAL,
                category="sql_injection",
                title="SQL Injection Vulnerability",
                description="String concatenation in SQL queries",
                cwe_id="CWE-89",
                recommendation="Use PreparedStatement with parameterized queries"
            ))
        
        # Check for deserialization vulnerabilities
        if re.search(r'ObjectInputStream.*readObject', code):
            vulnerabilities.append(SecurityVulnerability(
                severity=SecuritySeverity.HIGH,
                category="insecure_deserialization",
                title="Insecure Deserialization",
                description="Deserializing untrusted data can be dangerous",
                cwe_id="CWE-502",
                recommendation="Validate and sanitize serialized data"
            ))
        
        return vulnerabilities
    
    def _scan_sql_security(self, code: str, context: Optional[Dict] = None) -> List[SecurityVulnerability]:
        """Scan SQL code for security vulnerabilities"""
        vulnerabilities = []
        
        # Check for dynamic SQL construction
        if re.search(r'WHERE.*=.*\+', code, re.IGNORECASE):
            vulnerabilities.append(SecurityVulnerability(
                severity=SecuritySeverity.CRITICAL,
                category="sql_injection",
                title="SQL Injection Vulnerability",
                description="Dynamic SQL construction detected",
                cwe_id="CWE-89",
                recommendation="Use parameterized queries"
            ))
        
        return vulnerabilities
    
    def _scan_php_security(self, code: str, context: Optional[Dict] = None) -> List[SecurityVulnerability]:
        """Scan PHP code for security vulnerabilities"""
        vulnerabilities = []
        
        # Check for eval usage
        if re.search(r'\beval\s*\(', code):
            vulnerabilities.append(SecurityVulnerability(
                severity=SecuritySeverity.HIGH,
                category="code_injection",
                title="Code Injection Risk",
                description="eval() usage in PHP is dangerous",
                cwe_id="CWE-94",
                recommendation="Avoid eval() or validate input thoroughly"
            ))
        
        return vulnerabilities
    
    def _scan_general_security_patterns(self, code: str, language: str) -> List[SecurityVulnerability]:
        """Scan for general security patterns across languages"""
        vulnerabilities = []
        
        # Check for hardcoded credentials
        credential_patterns = [
            (r'password\s*[:=]\s*["\'][^"\']{8,}["\']', 'password'),
            (r'api_key\s*[:=]\s*["\'][^"\']{16,}["\']', 'API key'),
            (r'secret\s*[:=]\s*["\'][^"\']{16,}["\']', 'secret'),
            (r'token\s*[:=]\s*["\'][^"\']{20,}["\']', 'token')
        ]
        
        for pattern, cred_type in credential_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                vulnerabilities.append(SecurityVulnerability(
                    severity=SecuritySeverity.HIGH,
                    category="hardcoded_credentials",
                    title=f"Hardcoded {cred_type.title()}",
                    description=f"Hardcoded {cred_type} found in source code",
                    line_number=line_num,
                    code_snippet=match.group(),
                    cwe_id="CWE-798",
                    recommendation=f"Move {cred_type} to environment variables or secure configuration"
                ))
        
        # Check for weak cryptographic practices
        weak_crypto_patterns = [
            (r'\bmd5\s*\(', 'MD5 hash function'),
            (r'\bsha1\s*\(', 'SHA1 hash function'),
            (r'DES\s*\(', 'DES encryption'),
            (r'RC4', 'RC4 cipher')
        ]
        
        for pattern, crypto_type in weak_crypto_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                vulnerabilities.append(SecurityVulnerability(
                    severity=SecuritySeverity.MEDIUM,
                    category="weak_cryptography",
                    title=f"Weak Cryptographic Function",
                    description=f"Use of {crypto_type} which is cryptographically weak",
                    cwe_id="CWE-327",
                    recommendation="Use stronger cryptographic functions (SHA-256, AES, etc.)"
                ))
        
        return vulnerabilities
    
    def _scan_with_patterns(self, code: str, language: str) -> List[SecurityVulnerability]:
        """Scan code using predefined vulnerability patterns"""
        vulnerabilities = []
        
        patterns = self.vulnerability_patterns.get(language, {})
        
        for vuln_type, config in patterns.items():
            for pattern in config['patterns']:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    vulnerabilities.append(SecurityVulnerability(
                        severity=config['severity'],
                        category=vuln_type,
                        title=f"{vuln_type.replace('_', ' ').title()} Detected",
                        description=f"Pattern matching {vuln_type} vulnerability",
                        line_number=line_num,
                        code_snippet=match.group(),
                        cwe_id=config.get('cwe'),
                        recommendation=f"Review and fix {vuln_type} vulnerability"
                    ))
        
        return vulnerabilities
    
    def _calculate_risk_score(self, vulnerabilities: List[SecurityVulnerability]) -> float:
        """Calculate overall risk score (0-100)"""
        if not vulnerabilities:
            return 0.0
        
        severity_weights = {
            SecuritySeverity.CRITICAL: 25,
            SecuritySeverity.HIGH: 15,
            SecuritySeverity.MEDIUM: 8,
            SecuritySeverity.LOW: 3,
            SecuritySeverity.INFO: 1
        }
        
        total_score = sum(severity_weights.get(vuln.severity, 0) * vuln.confidence 
                         for vuln in vulnerabilities)
        
        # Cap at 100
        return min(100.0, total_score)
    
    def _generate_summary(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, int]:
        """Generate summary of vulnerabilities by severity"""
        summary = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'info': 0
        }
        
        for vuln in vulnerabilities:
            summary[vuln.severity.value] += 1
        
        return summary
    
    def _generate_recommendations(self, vulnerabilities: List[SecurityVulnerability]) -> List[str]:
        """Generate security recommendations based on found vulnerabilities"""
        recommendations = []
        
        # Count vulnerabilities by category
        categories = {}
        for vuln in vulnerabilities:
            categories[vuln.category] = categories.get(vuln.category, 0) + 1
        
        # Generate category-specific recommendations
        if 'sql_injection' in categories:
            recommendations.append("Implement parameterized queries to prevent SQL injection")
        
        if 'xss' in categories:
            recommendations.append("Sanitize user input and use safe DOM manipulation methods")
        
        if 'hardcoded_credentials' in categories:
            recommendations.append("Move sensitive credentials to environment variables")
        
        if 'code_injection' in categories:
            recommendations.append("Avoid dynamic code execution with user input")
        
        if 'weak_cryptography' in categories:
            recommendations.append("Upgrade to stronger cryptographic algorithms")
        
        # General recommendations based on severity
        critical_count = sum(1 for v in vulnerabilities if v.severity == SecuritySeverity.CRITICAL)
        high_count = sum(1 for v in vulnerabilities if v.severity == SecuritySeverity.HIGH)
        
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical security vulnerabilities immediately")
        
        if high_count > 0:
            recommendations.append(f"Review and fix {high_count} high-severity security issues")
        
        if len(vulnerabilities) > 10:
            recommendations.append("Consider comprehensive security code review")
        
        return recommendations
    
    def generate_security_report_json(self, report: SecurityReport) -> str:
        """Generate JSON report for security scan results"""
        report_data = {
            'summary': report.summary,
            'risk_score': report.risk_score,
            'vulnerabilities': [
                {
                    'severity': vuln.severity.value,
                    'category': vuln.category,
                    'title': vuln.title,
                    'description': vuln.description,
                    'line_number': vuln.line_number,
                    'cwe_id': vuln.cwe_id,
                    'recommendation': vuln.recommendation,
                    'confidence': vuln.confidence
                }
                for vuln in report.vulnerabilities
            ],
            'recommendations': report.recommendations,
            'metadata': report.scan_metadata
        }
        
        return json.dumps(report_data, indent=2)