"""
Infrastructure-as-Code Security Scanner
Implements security scanning for Terraform and Helm configurations
"""

import os
import json
import yaml
import subprocess
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
import hashlib

logger = logging.getLogger(__name__)

class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityFinding:
    rule_id: str
    severity: SeverityLevel
    title: str
    description: str
    file_path: str
    line_number: int
    resource: str
    remediation: str

@dataclass
class ScanResult:
    scan_id: str
    timestamp: str
    total_files: int
    findings: List[SecurityFinding]
    passed_checks: int
    failed_checks: int
    skipped_checks: int

class TerraformSecurityScanner:
    """Security scanner for Terraform configurations"""
    
    def __init__(self):
        self.security_rules = self._load_terraform_rules()
        
    def scan_directory(self, directory: str) -> ScanResult:
        """Scan Terraform files in directory"""
        findings = []
        total_files = 0
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.tf', '.tfvars')):
                    file_path = os.path.join(root, file)
                    total_files += 1
                    file_findings = self._scan_terraform_file(file_path)
                    findings.extend(file_findings)
        
        return ScanResult(
            scan_id=hashlib.md5(directory.encode()).hexdigest()[:8],
            timestamp=str(int(time.time())),
            total_files=total_files,
            findings=findings,
            passed_checks=len([f for f in findings if f.severity == SeverityLevel.LOW]),
            failed_checks=len([f for f in findings if f.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]),
            skipped_checks=0
        )
    
    def _scan_terraform_file(self, file_path: str) -> List[SecurityFinding]:
        """Scan individual Terraform file"""
        findings = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for rule in self.security_rules:
                    if self._check_rule(line, rule):
                        finding = SecurityFinding(
                            rule_id=rule['id'],
                            severity=SeverityLevel(rule['severity']),
                            title=rule['title'],
                            description=rule['description'],
                            file_path=file_path,
                            line_number=line_num,
                            resource=self._extract_resource_name(line),
                            remediation=rule['remediation']
                        )
                        findings.append(finding)
                        
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")
            
        return findings
    
    def _load_terraform_rules(self) -> List[Dict[str, Any]]:
        """Load Terraform security rules"""
        return [
            {
                'id': 'TF001',
                'severity': 'critical',
                'title': 'Hardcoded secrets detected',
                'description': 'Secrets should not be hardcoded in Terraform files',
                'pattern': r'(password|secret|key)\s*=\s*["\'][^"\']+["\']',
                'remediation': 'Use variables or secret management systems'
            },
            {
                'id': 'TF002',
                'severity': 'high',
                'title': 'Public S3 bucket',
                'description': 'S3 bucket allows public access',
                'pattern': r'acl\s*=\s*["\']public-read["\']',
                'remediation': 'Restrict S3 bucket access to specific users/roles'
            },
            {
                'id': 'TF003',
                'severity': 'high',
                'title': 'Security group allows all traffic',
                'description': 'Security group rule allows traffic from 0.0.0.0/0',
                'pattern': r'cidr_blocks\s*=\s*\[["\']0\.0\.0\.0/0["\']',
                'remediation': 'Restrict CIDR blocks to specific IP ranges'
            },
            {
                'id': 'TF004',
                'severity': 'medium',
                'title': 'Unencrypted storage',
                'description': 'Storage resource is not encrypted',
                'pattern': r'encrypted\s*=\s*false',
                'remediation': 'Enable encryption for storage resources'
            },
            {
                'id': 'TF005',
                'severity': 'medium',
                'title': 'Missing backup configuration',
                'description': 'Database resource missing backup configuration',
                'pattern': r'resource\s+"aws_db_instance".*(?!.*backup_retention_period)',
                'remediation': 'Configure backup retention for databases'
            }
        ]
    
    def _check_rule(self, line: str, rule: Dict[str, Any]) -> bool:
        """Check if line matches security rule"""
        pattern = rule['pattern']
        return bool(re.search(pattern, line, re.IGNORECASE))
    
    def _extract_resource_name(self, line: str) -> str:
        """Extract resource name from Terraform line"""
        match = re.search(r'resource\s+"([^"]+)"\s+"([^"]+)"', line)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
        return "unknown"

class HelmSecurityScanner:
    """Security scanner for Helm charts"""
    
    def __init__(self):
        self.security_rules = self._load_helm_rules()
        
    def scan_chart(self, chart_path: str) -> ScanResult:
        """Scan Helm chart for security issues"""
        findings = []
        total_files = 0
        
        # Scan values.yaml
        values_path = os.path.join(chart_path, 'values.yaml')
        if os.path.exists(values_path):
            total_files += 1
            findings.extend(self._scan_values_file(values_path))
        
        # Scan templates
        templates_dir = os.path.join(chart_path, 'templates')
        if os.path.exists(templates_dir):
            for file in os.listdir(templates_dir):
                if file.endswith('.yaml'):
                    file_path = os.path.join(templates_dir, file)
                    total_files += 1
                    findings.extend(self._scan_template_file(file_path))
        
        return ScanResult(
            scan_id=hashlib.md5(chart_path.encode()).hexdigest()[:8],
            timestamp=str(int(time.time())),
            total_files=total_files,
            findings=findings,
            passed_checks=len([f for f in findings if f.severity == SeverityLevel.LOW]),
            failed_checks=len([f for f in findings if f.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]),
            skipped_checks=0
        )
    
    def _scan_values_file(self, file_path: str) -> List[SecurityFinding]:
        """Scan Helm values.yaml file"""
        findings = []
        
        try:
            with open(file_path, 'r') as f:
                content = yaml.safe_load(f)
            
            # Check for security misconfigurations
            findings.extend(self._check_security_context(content, file_path))
            findings.extend(self._check_resource_limits(content, file_path))
            findings.extend(self._check_network_policies(content, file_path))
            
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")
            
        return findings
    
    def _scan_template_file(self, file_path: str) -> List[SecurityFinding]:
        """Scan Helm template file"""
        findings = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for rule in self.security_rules:
                    if self._check_template_rule(line, rule):
                        finding = SecurityFinding(
                            rule_id=rule['id'],
                            severity=SeverityLevel(rule['severity']),
                            title=rule['title'],
                            description=rule['description'],
                            file_path=file_path,
                            line_number=line_num,
                            resource=self._extract_k8s_resource(line),
                            remediation=rule['remediation']
                        )
                        findings.append(finding)
                        
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")
            
        return findings
    
    def _load_helm_rules(self) -> List[Dict[str, Any]]:
        """Load Helm security rules"""
        return [
            {
                'id': 'HELM001',
                'severity': 'critical',
                'title': 'Privileged container',
                'description': 'Container runs with privileged access',
                'pattern': r'privileged:\s*true',
                'remediation': 'Remove privileged access unless absolutely necessary'
            },
            {
                'id': 'HELM002',
                'severity': 'high',
                'title': 'Root user',
                'description': 'Container runs as root user',
                'pattern': r'runAsUser:\s*0',
                'remediation': 'Run container as non-root user'
            },
            {
                'id': 'HELM003',
                'severity': 'medium',
                'title': 'Missing resource limits',
                'description': 'Container missing resource limits',
                'pattern': r'resources:\s*{}',
                'remediation': 'Set CPU and memory limits for containers'
            },
            {
                'id': 'HELM004',
                'severity': 'medium',
                'title': 'Writable root filesystem',
                'description': 'Container has writable root filesystem',
                'pattern': r'readOnlyRootFilesystem:\s*false',
                'remediation': 'Set readOnlyRootFilesystem to true'
            }
        ]
    
    def _check_security_context(self, content: Dict[str, Any], file_path: str) -> List[SecurityFinding]:
        """Check security context configuration"""
        findings = []
        
        security_context = content.get('securityContext', {})
        
        if security_context.get('runAsUser') == 0:
            findings.append(SecurityFinding(
                rule_id='HELM_SC001',
                severity=SeverityLevel.HIGH,
                title='Container runs as root',
                description='Security context allows running as root user',
                file_path=file_path,
                line_number=0,
                resource='securityContext',
                remediation='Set runAsUser to non-zero value'
            ))
        
        return findings
    
    def _check_resource_limits(self, content: Dict[str, Any], file_path: str) -> List[SecurityFinding]:
        """Check resource limits configuration"""
        findings = []
        
        resources = content.get('resources', {})
        
        if not resources.get('limits'):
            findings.append(SecurityFinding(
                rule_id='HELM_RL001',
                severity=SeverityLevel.MEDIUM,
                title='Missing resource limits',
                description='Container missing resource limits',
                file_path=file_path,
                line_number=0,
                resource='resources',
                remediation='Set CPU and memory limits'
            ))
        
        return findings
    
    def _check_network_policies(self, content: Dict[str, Any], file_path: str) -> List[SecurityFinding]:
        """Check network policies configuration"""
        findings = []
        
        network_policy = content.get('networkPolicy', {})
        
        if not network_policy.get('enabled', False):
            findings.append(SecurityFinding(
                rule_id='HELM_NP001',
                severity=SeverityLevel.MEDIUM,
                title='Network policy disabled',
                description='Network policy is not enabled',
                file_path=file_path,
                line_number=0,
                resource='networkPolicy',
                remediation='Enable network policy for micro-segmentation'
            ))
        
        return findings
    
    def _check_template_rule(self, line: str, rule: Dict[str, Any]) -> bool:
        """Check if template line matches security rule"""
        pattern = rule['pattern']
        return bool(re.search(pattern, line, re.IGNORECASE))
    
    def _extract_k8s_resource(self, line: str) -> str:
        """Extract Kubernetes resource name from template line"""
        match = re.search(r'kind:\s*(\w+)', line)
        if match:
            return match.group(1)
        return "unknown"

class InfrastructureSecurityScanner:
    """Main infrastructure security scanner"""
    
    def __init__(self):
        self.terraform_scanner = TerraformSecurityScanner()
        self.helm_scanner = HelmSecurityScanner()
        
    def scan_infrastructure(self, config_path: str) -> Dict[str, ScanResult]:
        """Scan infrastructure configurations"""
        results = {}
        
        # Scan Terraform configurations
        terraform_path = os.path.join(config_path, 'terraform')
        if os.path.exists(terraform_path):
            results['terraform'] = self.terraform_scanner.scan_directory(terraform_path)
        
        # Scan Helm charts
        helm_path = os.path.join(config_path, 'helm')
        if os.path.exists(helm_path):
            for chart in os.listdir(helm_path):
                chart_path = os.path.join(helm_path, chart)
                if os.path.isdir(chart_path):
                    results[f'helm_{chart}'] = self.helm_scanner.scan_chart(chart_path)
        
        return results
    
    def generate_report(self, results: Dict[str, ScanResult]) -> str:
        """Generate security scan report"""
        report = "# Infrastructure Security Scan Report\n\n"
        
        total_findings = 0
        critical_findings = 0
        high_findings = 0
        
        for scan_type, result in results.items():
            report += f"## {scan_type.upper()} Scan Results\n\n"
            report += f"- Total files scanned: {result.total_files}\n"
            report += f"- Total findings: {len(result.findings)}\n"
            report += f"- Passed checks: {result.passed_checks}\n"
            report += f"- Failed checks: {result.failed_checks}\n\n"
            
            if result.findings:
                report += "### Findings\n\n"
                for finding in result.findings:
                    report += f"**{finding.severity.value.upper()}**: {finding.title}\n"
                    report += f"- File: {finding.file_path}:{finding.line_number}\n"
                    report += f"- Resource: {finding.resource}\n"
                    report += f"- Description: {finding.description}\n"
                    report += f"- Remediation: {finding.remediation}\n\n"
                    
                    total_findings += 1
                    if finding.severity == SeverityLevel.CRITICAL:
                        critical_findings += 1
                    elif finding.severity == SeverityLevel.HIGH:
                        high_findings += 1
        
        # Summary
        report = f"# Infrastructure Security Scan Summary\n\n" \
                f"- **Total Findings**: {total_findings}\n" \
                f"- **Critical**: {critical_findings}\n" \
                f"- **High**: {high_findings}\n\n" + report
        
        return report
    
    def run_external_scanners(self, config_path: str) -> Dict[str, Any]:
        """Run external security scanners"""
        results = {}
        
        # Run Checkov for Terraform
        try:
            checkov_result = subprocess.run([
                'checkov', '-d', config_path, '--framework', 'terraform', '--output', 'json'
            ], capture_output=True, text=True, timeout=300)
            
            if checkov_result.returncode == 0:
                results['checkov'] = json.loads(checkov_result.stdout)
        except Exception as e:
            logger.error(f"Checkov scan failed: {e}")
        
        # Run Trivy for container images
        try:
            trivy_result = subprocess.run([
                'trivy', 'config', config_path, '--format', 'json'
            ], capture_output=True, text=True, timeout=300)
            
            if trivy_result.returncode == 0:
                results['trivy'] = json.loads(trivy_result.stdout)
        except Exception as e:
            logger.error(f"Trivy scan failed: {e}")
        
        return results

# Example usage
if __name__ == "__main__":
    import time
    
    scanner = InfrastructureSecurityScanner()
    results = scanner.scan_infrastructure('./infrastructure')
    report = scanner.generate_report(results)
    
    print(report)
    
    # Save report
    with open(f'security_scan_report_{int(time.time())}.md', 'w') as f:
        f.write(report)