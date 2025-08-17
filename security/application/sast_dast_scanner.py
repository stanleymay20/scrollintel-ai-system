"""
SAST/DAST Security Scanner Integration
Implements automated security scanning in CI/CD pipeline with security gates
"""

import asyncio
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ScanType(Enum):
    SAST = "static"
    DAST = "dynamic"
    DEPENDENCY = "dependency"
    CONTAINER = "container"

class SeverityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class SecurityFinding:
    id: str
    title: str
    description: str
    severity: SeverityLevel
    scan_type: ScanType
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    remediation: Optional[str] = None
    confidence: Optional[str] = None

@dataclass
class ScanResult:
    scan_id: str
    scan_type: ScanType
    timestamp: datetime
    findings: List[SecurityFinding]
    passed_security_gate: bool
    total_vulnerabilities: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int

class SecurityScanner:
    """Comprehensive security scanner with SAST/DAST capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.security_gates = config.get('security_gates', {})
        self.scan_tools = config.get('scan_tools', {})
        self.results_storage = config.get('results_storage', '/tmp/security_scans')
        Path(self.results_storage).mkdir(parents=True, exist_ok=True)
    
    async def run_comprehensive_scan(self, target_path: str, scan_types: List[ScanType]) -> Dict[ScanType, ScanResult]:
        """Run comprehensive security scan with multiple tools"""
        results = {}
        
        for scan_type in scan_types:
            try:
                if scan_type == ScanType.SAST:
                    results[scan_type] = await self._run_sast_scan(target_path)
                elif scan_type == ScanType.DAST:
                    results[scan_type] = await self._run_dast_scan(target_path)
                elif scan_type == ScanType.DEPENDENCY:
                    results[scan_type] = await self._run_dependency_scan(target_path)
                elif scan_type == ScanType.CONTAINER:
                    results[scan_type] = await self._run_container_scan(target_path)
            except Exception as e:
                print(f"Error running {scan_type.value} scan: {e}")
                # Create failed scan result
                results[scan_type] = ScanResult(
                    scan_id=f"failed_{scan_type.value}_{datetime.now().isoformat()}",
                    scan_type=scan_type,
                    timestamp=datetime.now(),
                    findings=[],
                    passed_security_gate=False,
                    total_vulnerabilities=0,
                    critical_count=0,
                    high_count=0,
                    medium_count=0,
                    low_count=0
                )
        
        return results
    
    async def _run_sast_scan(self, target_path: str) -> ScanResult:
        """Run Static Application Security Testing"""
        scan_id = f"sast_{datetime.now().isoformat()}"
        findings = []
        
        # Semgrep SAST scanning
        if self.scan_tools.get('semgrep', {}).get('enabled', True):
            semgrep_findings = await self._run_semgrep(target_path)
            findings.extend(semgrep_findings)
        
        # Bandit for Python security issues
        if self.scan_tools.get('bandit', {}).get('enabled', True):
            bandit_findings = await self._run_bandit(target_path)
            findings.extend(bandit_findings)
        
        # CodeQL analysis
        if self.scan_tools.get('codeql', {}).get('enabled', False):
            codeql_findings = await self._run_codeql(target_path)
            findings.extend(codeql_findings)
        
        return self._create_scan_result(scan_id, ScanType.SAST, findings)
    
    async def _run_dast_scan(self, target_url: str) -> ScanResult:
        """Run Dynamic Application Security Testing"""
        scan_id = f"dast_{datetime.now().isoformat()}"
        findings = []
        
        # OWASP ZAP scanning
        if self.scan_tools.get('zap', {}).get('enabled', True):
            zap_findings = await self._run_zap_scan(target_url)
            findings.extend(zap_findings)
        
        # Nuclei vulnerability scanner
        if self.scan_tools.get('nuclei', {}).get('enabled', True):
            nuclei_findings = await self._run_nuclei(target_url)
            findings.extend(nuclei_findings)
        
        return self._create_scan_result(scan_id, ScanType.DAST, findings)
    
    async def _run_dependency_scan(self, target_path: str) -> ScanResult:
        """Run dependency vulnerability scanning"""
        scan_id = f"dependency_{datetime.now().isoformat()}"
        findings = []
        
        # Safety for Python dependencies
        if self.scan_tools.get('safety', {}).get('enabled', True):
            safety_findings = await self._run_safety_scan(target_path)
            findings.extend(safety_findings)
        
        # npm audit for Node.js dependencies
        if self.scan_tools.get('npm_audit', {}).get('enabled', True):
            npm_findings = await self._run_npm_audit(target_path)
            findings.extend(npm_findings)
        
        # Snyk vulnerability scanning
        if self.scan_tools.get('snyk', {}).get('enabled', False):
            snyk_findings = await self._run_snyk_scan(target_path)
            findings.extend(snyk_findings)
        
        return self._create_scan_result(scan_id, ScanType.DEPENDENCY, findings)
    
    async def _run_container_scan(self, image_name: str) -> ScanResult:
        """Run container security scanning"""
        scan_id = f"container_{datetime.now().isoformat()}"
        findings = []
        
        # Trivy container scanning
        if self.scan_tools.get('trivy', {}).get('enabled', True):
            trivy_findings = await self._run_trivy_scan(image_name)
            findings.extend(trivy_findings)
        
        # Docker bench security
        if self.scan_tools.get('docker_bench', {}).get('enabled', True):
            bench_findings = await self._run_docker_bench()
            findings.extend(bench_findings)
        
        return self._create_scan_result(scan_id, ScanType.CONTAINER, findings)
    
    async def _run_semgrep(self, target_path: str) -> List[SecurityFinding]:
        """Run Semgrep SAST analysis"""
        findings = []
        try:
            cmd = [
                'semgrep',
                '--config=auto',
                '--json',
                '--quiet',
                target_path
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                data = json.loads(stdout.decode())
                for finding in data.get('results', []):
                    findings.append(SecurityFinding(
                        id=finding.get('check_id', 'unknown'),
                        title=finding.get('message', 'Security issue detected'),
                        description=finding.get('extra', {}).get('message', ''),
                        severity=self._map_severity(finding.get('extra', {}).get('severity', 'INFO')),
                        scan_type=ScanType.SAST,
                        file_path=finding.get('path'),
                        line_number=finding.get('start', {}).get('line'),
                        remediation=finding.get('extra', {}).get('fix', '')
                    ))
        except Exception as e:
            print(f"Semgrep scan failed: {e}")
        
        return findings
    
    async def _run_bandit(self, target_path: str) -> List[SecurityFinding]:
        """Run Bandit Python security analysis"""
        findings = []
        try:
            cmd = [
                'bandit',
                '-r',
                '-f', 'json',
                target_path
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if stdout:
                data = json.loads(stdout.decode())
                for finding in data.get('results', []):
                    findings.append(SecurityFinding(
                        id=finding.get('test_id', 'unknown'),
                        title=finding.get('test_name', 'Security issue'),
                        description=finding.get('issue_text', ''),
                        severity=self._map_severity(finding.get('issue_severity', 'LOW')),
                        scan_type=ScanType.SAST,
                        file_path=finding.get('filename'),
                        line_number=finding.get('line_number'),
                        confidence=finding.get('issue_confidence')
                    ))
        except Exception as e:
            print(f"Bandit scan failed: {e}")
        
        return findings
    
    async def _run_zap_scan(self, target_url: str) -> List[SecurityFinding]:
        """Run OWASP ZAP dynamic scan"""
        findings = []
        try:
            # ZAP baseline scan
            cmd = [
                'zap-baseline.py',
                '-t', target_url,
                '-J', '/tmp/zap-report.json'
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await result.communicate()
            
            # Parse ZAP results
            if Path('/tmp/zap-report.json').exists():
                with open('/tmp/zap-report.json', 'r') as f:
                    data = json.load(f)
                    
                for site in data.get('site', []):
                    for alert in site.get('alerts', []):
                        findings.append(SecurityFinding(
                            id=alert.get('pluginid', 'unknown'),
                            title=alert.get('name', 'Security vulnerability'),
                            description=alert.get('desc', ''),
                            severity=self._map_zap_severity(alert.get('riskdesc', 'Low')),
                            scan_type=ScanType.DAST,
                            remediation=alert.get('solution', '')
                        ))
        except Exception as e:
            print(f"ZAP scan failed: {e}")
        
        return findings
    
    async def _run_safety_scan(self, target_path: str) -> List[SecurityFinding]:
        """Run Safety dependency vulnerability scan"""
        findings = []
        try:
            cmd = [
                'safety',
                'check',
                '--json',
                '--file', f"{target_path}/requirements.txt"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if stdout:
                data = json.loads(stdout.decode())
                for vuln in data:
                    findings.append(SecurityFinding(
                        id=vuln.get('id', 'unknown'),
                        title=f"Vulnerable dependency: {vuln.get('package_name')}",
                        description=vuln.get('advisory', ''),
                        severity=SeverityLevel.HIGH,  # Safety vulnerabilities are typically high
                        scan_type=ScanType.DEPENDENCY,
                        remediation=f"Update to version {vuln.get('analyzed_version', 'latest')}"
                    ))
        except Exception as e:
            print(f"Safety scan failed: {e}")
        
        return findings
    
    def _create_scan_result(self, scan_id: str, scan_type: ScanType, findings: List[SecurityFinding]) -> ScanResult:
        """Create scan result with security gate evaluation"""
        critical_count = sum(1 for f in findings if f.severity == SeverityLevel.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SeverityLevel.HIGH)
        medium_count = sum(1 for f in findings if f.severity == SeverityLevel.MEDIUM)
        low_count = sum(1 for f in findings if f.severity == SeverityLevel.LOW)
        
        # Evaluate security gate
        passed_gate = self._evaluate_security_gate(critical_count, high_count, medium_count, low_count)
        
        return ScanResult(
            scan_id=scan_id,
            scan_type=scan_type,
            timestamp=datetime.now(),
            findings=findings,
            passed_security_gate=passed_gate,
            total_vulnerabilities=len(findings),
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count
        )
    
    def _evaluate_security_gate(self, critical: int, high: int, medium: int, low: int) -> bool:
        """Evaluate if scan passes security gate thresholds"""
        gates = self.security_gates
        
        if critical > gates.get('max_critical', 0):
            return False
        if high > gates.get('max_high', 5):
            return False
        if medium > gates.get('max_medium', 20):
            return False
        
        return True
    
    def _map_severity(self, severity_str: str) -> SeverityLevel:
        """Map string severity to enum"""
        severity_map = {
            'CRITICAL': SeverityLevel.CRITICAL,
            'HIGH': SeverityLevel.HIGH,
            'MEDIUM': SeverityLevel.MEDIUM,
            'LOW': SeverityLevel.LOW,
            'INFO': SeverityLevel.INFO
        }
        return severity_map.get(severity_str.upper(), SeverityLevel.INFO)
    
    def _map_zap_severity(self, risk_desc: str) -> SeverityLevel:
        """Map ZAP risk description to severity level"""
        if 'High' in risk_desc:
            return SeverityLevel.HIGH
        elif 'Medium' in risk_desc:
            return SeverityLevel.MEDIUM
        elif 'Low' in risk_desc:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.INFO
    
    async def generate_security_report(self, scan_results: Dict[ScanType, ScanResult]) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        total_findings = sum(len(result.findings) for result in scan_results.values())
        total_critical = sum(result.critical_count for result in scan_results.values())
        total_high = sum(result.high_count for result in scan_results.values())
        
        overall_passed = all(result.passed_security_gate for result in scan_results.values())
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_security_gate_passed': overall_passed,
            'summary': {
                'total_findings': total_findings,
                'critical_vulnerabilities': total_critical,
                'high_vulnerabilities': total_high,
                'scan_types_completed': len(scan_results)
            },
            'scan_results': {
                scan_type.value: {
                    'scan_id': result.scan_id,
                    'passed_gate': result.passed_security_gate,
                    'findings_count': result.total_vulnerabilities,
                    'severity_breakdown': {
                        'critical': result.critical_count,
                        'high': result.high_count,
                        'medium': result.medium_count,
                        'low': result.low_count
                    }
                }
                for scan_type, result in scan_results.items()
            },
            'recommendations': self._generate_recommendations(scan_results)
        }
        
        return report
    
    def _generate_recommendations(self, scan_results: Dict[ScanType, ScanResult]) -> List[str]:
        """Generate security recommendations based on scan results"""
        recommendations = []
        
        for scan_type, result in scan_results.items():
            if result.critical_count > 0:
                recommendations.append(f"URGENT: Address {result.critical_count} critical vulnerabilities from {scan_type.value} scan")
            
            if result.high_count > 5:
                recommendations.append(f"High priority: Reduce {result.high_count} high-severity issues from {scan_type.value} scan")
        
        if not any(result.passed_security_gate for result in scan_results.values()):
            recommendations.append("Security gate failed - deployment should be blocked until issues are resolved")
        
        return recommendations

class CICDSecurityIntegration:
    """Integration with CI/CD pipeline for automated security scanning"""
    
    def __init__(self, scanner: SecurityScanner):
        self.scanner = scanner
    
    async def run_pipeline_security_check(self, project_path: str, target_url: Optional[str] = None) -> bool:
        """Run security checks as part of CI/CD pipeline"""
        scan_types = [ScanType.SAST, ScanType.DEPENDENCY]
        
        if target_url:
            scan_types.append(ScanType.DAST)
        
        # Check for Dockerfile
        if Path(f"{project_path}/Dockerfile").exists():
            scan_types.append(ScanType.CONTAINER)
        
        results = await self.scanner.run_comprehensive_scan(project_path, scan_types)
        report = await self.scanner.generate_security_report(results)
        
        # Save report for CI/CD system
        report_path = f"{project_path}/security-report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report['overall_security_gate_passed']
    
    def create_security_gate_config(self) -> Dict[str, Any]:
        """Create security gate configuration for CI/CD"""
        return {
            'security_gates': {
                'max_critical': 0,
                'max_high': 5,
                'max_medium': 20,
                'block_deployment_on_failure': True
            },
            'scan_tools': {
                'semgrep': {'enabled': True},
                'bandit': {'enabled': True},
                'safety': {'enabled': True},
                'zap': {'enabled': True},
                'trivy': {'enabled': True}
            },
            'notifications': {
                'slack_webhook': None,
                'email_recipients': [],
                'teams_webhook': None
            }
        }