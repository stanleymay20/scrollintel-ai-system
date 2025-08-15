"""
Tests for Security Scanner Engine
"""

import pytest
from scrollintel.engines.security_scanner import (
    SecurityScanner, SecurityReport, SecurityVulnerability, SecuritySeverity
)

class TestSecurityScanner:
    
    def setup_method(self):
        self.scanner = SecurityScanner()
    
    def test_scan_python_sql_injection(self):
        """Test detection of SQL injection vulnerabilities in Python"""
        code = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
    return cursor.fetchone()
"""
        report = self.scanner.scan_code(code, 'python')
        
        sql_injection_vulns = [v for v in report.vulnerabilities 
                              if v.category == 'sql_injection']
        assert len(sql_injection_vulns) > 0
        assert report.risk_score > 0
    
    def test_scan_python_command_injection(self):
        """Test detection of command injection vulnerabilities"""
        code = """
import os
import subprocess

def execute_command(user_input):
    os.system(user_input)  # Dangerous
    subprocess.call(user_input, shell=True)  # Also dangerous
"""
        report = self.scanner.scan_code(code, 'python')
        
        command_injection_vulns = [v for v in report.vulnerabilities 
                                  if v.category == 'command_injection']
        assert len(command_injection_vulns) > 0
        assert any(v.severity == SecuritySeverity.HIGH for v in command_injection_vulns)
    
    def test_scan_python_eval_usage(self):
        """Test detection of eval/exec usage"""
        code = """
def process_input(user_code):
    result = eval(user_code)  # Very dangerous
    exec(user_code)  # Also very dangerous
    return result
"""
        report = self.scanner.scan_code(code, 'python')
        
        code_injection_vulns = [v for v in report.vulnerabilities 
                               if v.category == 'code_injection']
        assert len(code_injection_vulns) > 0
        assert any(v.severity == SecuritySeverity.HIGH for v in code_injection_vulns)
    
    def test_scan_python_dangerous_imports(self):
        """Test detection of dangerous imports"""
        code = """
import pickle
import marshal
import shelve

def load_data(data):
    return pickle.loads(data)  # Dangerous with untrusted data
"""
        report = self.scanner.scan_code(code, 'python')
        
        deserialization_vulns = [v for v in report.vulnerabilities 
                                if 'deserialization' in v.category]
        assert len(deserialization_vulns) > 0
    
    def test_scan_javascript_xss(self):
        """Test detection of XSS vulnerabilities in JavaScript"""
        code = """
function updateContent(userInput) {
    document.getElementById('content').innerHTML = userInput;  // XSS risk
    document.write(userInput);  // Also XSS risk
}
"""
        report = self.scanner.scan_code(code, 'javascript')
        
        xss_vulns = [v for v in report.vulnerabilities 
                    if v.category == 'xss']
        assert len(xss_vulns) > 0
        assert any(v.severity == SecuritySeverity.HIGH for v in xss_vulns)
    
    def test_scan_javascript_eval(self):
        """Test detection of eval usage in JavaScript"""
        code = """
function processCode(userCode) {
    return eval(userCode);  // Code injection risk
}
"""
        report = self.scanner.scan_code(code, 'javascript')
        
        code_injection_vulns = [v for v in report.vulnerabilities 
                               if v.category == 'code_injection']
        assert len(code_injection_vulns) > 0
    
    def test_scan_javascript_prototype_pollution(self):
        """Test detection of prototype pollution"""
        code = """
function merge(target, source) {
    for (let key in source) {
        if (key === '__proto__') {  // Dangerous
            target[key] = source[key];
        }
    }
}
"""
        report = self.scanner.scan_code(code, 'javascript')
        
        prototype_vulns = [v for v in report.vulnerabilities 
                          if 'prototype' in v.category]
        assert len(prototype_vulns) > 0
    
    def test_scan_java_sql_injection(self):
        """Test detection of SQL injection in Java"""
        code = """
public void getUser(String userId) {
    String query = "SELECT * FROM users WHERE id = " + userId;
    Statement stmt = connection.createStatement();
    ResultSet rs = stmt.executeQuery(query);
}
"""
        report = self.scanner.scan_code(code, 'java')
        
        sql_injection_vulns = [v for v in report.vulnerabilities 
                              if v.category == 'sql_injection']
        assert len(sql_injection_vulns) > 0
        assert any(v.severity == SecuritySeverity.CRITICAL for v in sql_injection_vulns)
    
    def test_scan_java_deserialization(self):
        """Test detection of insecure deserialization in Java"""
        code = """
public Object deserialize(InputStream input) {
    ObjectInputStream ois = new ObjectInputStream(input);
    return ois.readObject();  // Dangerous with untrusted data
}
"""
        report = self.scanner.scan_code(code, 'java')
        
        deserialization_vulns = [v for v in report.vulnerabilities 
                                if 'deserialization' in v.category]
        assert len(deserialization_vulns) > 0
    
    def test_scan_sql_injection_patterns(self):
        """Test detection of SQL injection patterns in SQL"""
        code = """
SELECT * FROM users WHERE name = 'admin' + @user_input;
INSERT INTO logs VALUES ('event', @data + 'extra');
"""
        report = self.scanner.scan_code(code, 'sql')
        
        sql_injection_vulns = [v for v in report.vulnerabilities 
                              if v.category == 'sql_injection']
        assert len(sql_injection_vulns) > 0
    
    def test_scan_hardcoded_credentials(self):
        """Test detection of hardcoded credentials"""
        code = """
API_KEY = "sk-1234567890abcdef1234567890abcdef"
password = "super_secret_password123"
database_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
"""
        report = self.scanner.scan_code(code, 'python')
        
        credential_vulns = [v for v in report.vulnerabilities 
                           if 'credential' in v.category]
        assert len(credential_vulns) > 0
        assert any(v.severity == SecuritySeverity.HIGH for v in credential_vulns)
    
    def test_scan_weak_cryptography(self):
        """Test detection of weak cryptographic functions"""
        code = """
import hashlib
import md5

def hash_password(password):
    return md5.md5(password).hexdigest()  # Weak
    
def hash_data(data):
    return hashlib.sha1(data).hexdigest()  # Also weak
"""
        report = self.scanner.scan_code(code, 'python')
        
        crypto_vulns = [v for v in report.vulnerabilities 
                       if 'cryptography' in v.category]
        assert len(crypto_vulns) > 0
    
    def test_scan_php_eval(self):
        """Test detection of eval usage in PHP"""
        code = """
<?php
function process_code($user_code) {
    return eval($user_code);  // Very dangerous
}
?>
"""
        report = self.scanner.scan_code(code, 'php')
        
        code_injection_vulns = [v for v in report.vulnerabilities 
                               if v.category == 'code_injection']
        assert len(code_injection_vulns) > 0
    
    def test_risk_score_calculation(self):
        """Test risk score calculation"""
        # High-risk code
        high_risk_code = """
import os
def execute(cmd):
    os.system(cmd)
    eval(cmd)
"""
        high_risk_report = self.scanner.scan_code(high_risk_code, 'python')
        
        # Low-risk code
        low_risk_code = """
def add_numbers(a, b):
    return a + b
"""
        low_risk_report = self.scanner.scan_code(low_risk_code, 'python')
        
        assert high_risk_report.risk_score > low_risk_report.risk_score
    
    def test_generate_summary(self):
        """Test summary generation"""
        code = """
import os
password = "hardcoded_password"
def execute(cmd):
    os.system(cmd)
"""
        report = self.scanner.scan_code(code, 'python')
        
        assert 'critical' in report.summary
        assert 'high' in report.summary
        assert 'medium' in report.summary
        assert 'low' in report.summary
        assert 'info' in report.summary
        assert sum(report.summary.values()) == len(report.vulnerabilities)
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        code = """
import os
password = "secret123"
def execute(cmd):
    os.system(cmd)
    eval(cmd)
"""
        report = self.scanner.scan_code(code, 'python')
        
        assert len(report.recommendations) > 0
        assert any('credential' in rec.lower() for rec in report.recommendations)
        assert any('injection' in rec.lower() for rec in report.recommendations)
    
    def test_generate_security_report_json(self):
        """Test JSON report generation"""
        code = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
"""
        report = self.scanner.scan_code(code, 'python')
        json_report = self.scanner.generate_security_report_json(report)
        
        assert isinstance(json_report, str)
        assert 'vulnerabilities' in json_report
        assert 'risk_score' in json_report
        assert 'recommendations' in json_report
    
    def test_scan_error_handling(self):
        """Test error handling in security scan"""
        # This should not crash the scanner
        report = self.scanner.scan_code(None, 'python')
        
        assert report.risk_score == 100.0  # Assume high risk on error
        assert len(report.vulnerabilities) > 0
        assert 'error' in report.scan_metadata
    
    def test_vulnerability_confidence_scoring(self):
        """Test vulnerability confidence scoring"""
        code = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
"""
        report = self.scanner.scan_code(code, 'python')
        
        for vuln in report.vulnerabilities:
            assert 0 <= vuln.confidence <= 1
    
    def test_cwe_id_assignment(self):
        """Test CWE ID assignment to vulnerabilities"""
        code = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
"""
        report = self.scanner.scan_code(code, 'python')
        
        sql_vulns = [v for v in report.vulnerabilities if v.category == 'sql_injection']
        if sql_vulns:
            assert sql_vulns[0].cwe_id is not None
            assert 'CWE-' in sql_vulns[0].cwe_id