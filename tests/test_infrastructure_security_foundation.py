"""
Tests for Infrastructure Security Foundation
Tests zero-trust gateway, container security policies, mTLS, and security scanning
"""

import pytest
import os
import tempfile
import yaml
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import the security modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'security'))

from zero_trust_gateway import (
    ZeroTrustGateway, IdentityVerifier, RiskAssessor, PolicyEngine,
    Identity, SecurityContext, AccessRequest, AccessDecision
)
from container.security_policies import ContainerSecurityPolicyManager, SecurityLevel
from mtls.certificate_manager import CertificateManager, CertificateAuthority, ServiceIdentity
from infrastructure.terraform_security_scanner import InfrastructureSecurityScanner

class TestZeroTrustGateway:
    """Test zero-trust network architecture implementation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.jwt_secret = "test-secret-key"
        self.gateway = ZeroTrustGateway(self.jwt_secret)
        
    def test_identity_verification_success(self):
        """Test successful identity verification"""
        import jwt
        
        # Create valid JWT token
        payload = {
            'user_id': 'test-user',
            'roles': ['user', 'api_user'],
            'attributes': {'department': 'engineering'},
            'trust_score': 0.8,
            'last_verified': datetime.utcnow().isoformat()
        }
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        
        credentials = {'token': token}
        identity = self.gateway.identity_verifier.verify(credentials)
        
        assert identity is not None
        assert identity.user_id == 'test-user'
        assert 'user' in identity.roles
        assert identity.trust_score == 0.8
        
    def test_identity_verification_failure(self):
        """Test identity verification with invalid token"""
        credentials = {'token': 'invalid-token'}
        identity = self.gateway.identity_verifier.verify(credentials)
        
        assert identity is None
        
    def test_risk_assessment_low_risk(self):
        """Test risk assessment for low-risk scenario"""
        identity = Identity(
            user_id='test-user',
            roles=['user'],
            attributes={},
            trust_score=0.9,
            last_verified=datetime.utcnow()
        )
        
        context = SecurityContext(
            source_ip='192.168.1.100',
            user_agent='Mozilla/5.0',
            device_fingerprint='trusted-device',
            location='San Francisco',
            time_of_access=datetime.utcnow().replace(hour=10),  # Business hours
            network_segment='internal'
        )
        
        risk_score = self.gateway.risk_assessor.calculate_risk(identity, '/api/data', context)
        
        assert risk_score < 0.5  # Should be low risk
        
    def test_risk_assessment_high_risk(self):
        """Test risk assessment for high-risk scenario"""
        identity = Identity(
            user_id='test-user',
            roles=['user'],
            attributes={},
            trust_score=0.3,  # Low trust
            last_verified=datetime.utcnow()
        )
        
        context = SecurityContext(
            source_ip='10.0.0.1',  # Suspicious IP
            user_agent='curl/7.68.0',
            device_fingerprint='unknown-device',
            location=None,
            time_of_access=datetime.utcnow().replace(hour=2),  # Off hours
            network_segment='external'
        )
        
        risk_score = self.gateway.risk_assessor.calculate_risk(identity, '/admin/users', context)
        
        assert risk_score > 0.6  # Should be high risk
        
    def test_policy_evaluation_allow(self):
        """Test policy evaluation that allows access"""
        identity = Identity(
            user_id='test-user',
            roles=['user'],
            attributes={},
            trust_score=0.8,
            last_verified=datetime.utcnow()
        )
        
        decision = self.gateway.policy_engine.evaluate(identity, '/api/data', 0.3)
        
        assert decision.decision == AccessDecision.ALLOW
        assert decision.risk_score == 0.3
        
    def test_policy_evaluation_deny(self):
        """Test policy evaluation that denies access"""
        identity = Identity(
            user_id='test-user',
            roles=['user'],  # No admin role
            attributes={},
            trust_score=0.8,
            last_verified=datetime.utcnow()
        )
        
        decision = self.gateway.policy_engine.evaluate(identity, '/admin/config', 0.9)
        
        assert decision.decision == AccessDecision.DENY
        
    def test_full_authorization_flow(self):
        """Test complete authorization flow"""
        import jwt
        
        # Create test identity
        payload = {
            'user_id': 'test-user',
            'roles': ['user'],
            'attributes': {},
            'trust_score': 0.8,
            'last_verified': datetime.utcnow().isoformat()
        }
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        
        identity = Identity(
            user_id='test-user',
            roles=['user'],
            attributes={},
            trust_score=0.8,
            last_verified=datetime.utcnow()
        )
        
        context = SecurityContext(
            source_ip='192.168.1.100',
            user_agent='Mozilla/5.0',
            device_fingerprint='trusted-device',
            location='San Francisco',
            time_of_access=datetime.utcnow().replace(hour=10),
            network_segment='internal'
        )
        
        request = AccessRequest(
            identity=identity,
            resource='/api/data',
            action='GET',
            context=context,
            credentials={'token': token}
        )
        
        decision = self.gateway.authorize_request(request)
        
        assert decision.decision in [AccessDecision.ALLOW, AccessDecision.CHALLENGE]
        assert len(self.gateway.audit_log) > 0

class TestContainerSecurityPolicies:
    """Test container security policies implementation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.policy_manager = ContainerSecurityPolicyManager()
        
    def test_restricted_policy_generation(self):
        """Test generation of restricted security policy"""
        psp = self.policy_manager.generate_pod_security_policy("restricted")
        
        assert psp["kind"] == "PodSecurityPolicy"
        assert psp["spec"]["privileged"] == False
        assert psp["spec"]["allowPrivilegeEscalation"] == False
        assert "ALL" in psp["spec"]["requiredDropCapabilities"]
        assert psp["spec"]["runAsUser"]["rule"] == "MustRunAsNonRoot"
        
    def test_security_context_constraints_generation(self):
        """Test generation of OpenShift SecurityContextConstraints"""
        scc = self.policy_manager.generate_security_context_constraints("restricted")
        
        assert scc["kind"] == "SecurityContextConstraints"
        assert scc["allowPrivilegedContainer"] == False
        assert scc["runAsUser"]["type"] == "MustRunAsNonRoot"
        assert "ALL" in scc["requiredDropCapabilities"]
        
    def test_secure_deployment_generation(self):
        """Test generation of secure deployment manifest"""
        deployment = self.policy_manager.generate_deployment_with_security(
            "test-app", "test:latest", "api-service"
        )
        
        assert deployment["kind"] == "Deployment"
        assert deployment["spec"]["template"]["spec"]["securityContext"]["runAsNonRoot"] == True
        
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert container["securityContext"]["allowPrivilegeEscalation"] == False
        assert container["securityContext"]["readOnlyRootFilesystem"] == True
        assert "ALL" in container["securityContext"]["capabilities"]["drop"]
        
    def test_pod_security_validation_success(self):
        """Test pod security validation with compliant spec"""
        pod_spec = {
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 1000
            },
            "containers": [
                {
                    "name": "app",
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "readOnlyRootFilesystem": True,
                        "capabilities": {
                            "drop": ["ALL"]
                        }
                    }
                }
            ],
            "volumes": [
                {"name": "config", "configMap": {"name": "app-config"}}
            ]
        }
        
        violations = self.policy_manager.validate_pod_security(pod_spec)
        assert len(violations) == 0
        
    def test_pod_security_validation_violations(self):
        """Test pod security validation with violations"""
        pod_spec = {
            "securityContext": {
                "runAsUser": 0  # Root user - violation
            },
            "containers": [
                {
                    "name": "app",
                    "securityContext": {
                        "privileged": True,  # Privileged - violation
                        "allowPrivilegeEscalation": True  # Privilege escalation - violation
                    }
                }
            ],
            "volumes": [
                {"name": "host", "hostPath": {"path": "/etc"}}  # May be violation depending on policy
            ]
        }
        
        violations = self.policy_manager.validate_pod_security(pod_spec)
        assert len(violations) > 0
        assert any("root" in v.lower() for v in violations)
        assert any("privileged" in v.lower() for v in violations)

class TestMTLSCertificateManager:
    """Test mTLS certificate management"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock CA certificate and key files
        self.ca_cert_path = os.path.join(self.temp_dir, "ca.crt")
        self.ca_key_path = os.path.join(self.temp_dir, "ca.key")
        
        # Create minimal CA cert and key for testing
        self._create_test_ca_files()
        
    def _create_test_ca_files(self):
        """Create test CA certificate and key files"""
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend
        
        # Generate CA private key
        ca_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Create CA certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Test"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test CA"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Test CA"),
        ])
        
        ca_cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            ca_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        ).sign(ca_key, hashes.SHA256(), default_backend())
        
        # Write CA certificate
        with open(self.ca_cert_path, "wb") as f:
            f.write(ca_cert.public_bytes(serialization.Encoding.PEM))
        
        # Write CA private key
        with open(self.ca_key_path, "wb") as f:
            f.write(ca_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
    
    def test_certificate_authority_initialization(self):
        """Test CA initialization"""
        ca = CertificateAuthority(self.ca_cert_path, self.ca_key_path)
        
        assert ca.ca_cert is not None
        assert ca.ca_key is not None
        
    def test_service_certificate_issuance(self):
        """Test service certificate issuance"""
        ca = CertificateAuthority(self.ca_cert_path, self.ca_key_path)
        
        service_identity = ServiceIdentity(
            service_name="test-service",
            namespace="test-namespace",
            cluster="test-cluster",
            environment="test"
        )
        
        cert_pem, key_pem = ca.issue_certificate(service_identity)
        
        assert cert_pem is not None
        assert key_pem is not None
        assert b"BEGIN CERTIFICATE" in cert_pem
        assert b"BEGIN PRIVATE KEY" in key_pem
        
    def test_certificate_manager_operations(self):
        """Test certificate manager operations"""
        ca = CertificateAuthority(self.ca_cert_path, self.ca_key_path)
        cert_store = os.path.join(self.temp_dir, "certs")
        cert_manager = CertificateManager(ca, cert_store)
        
        service_identity = ServiceIdentity(
            service_name="api",
            namespace="scrollintel",
            cluster="test",
            environment="test"
        )
        
        # Get or create certificate
        cert_path, key_path = cert_manager.get_or_create_certificate(service_identity)
        
        assert os.path.exists(cert_path)
        assert os.path.exists(key_path)
        
        # Test certificate info extraction
        cert_info = cert_manager.get_certificate_info(cert_path)
        assert cert_info is not None
        assert "api" in cert_info.subject_alt_names[0]
        
        # Test certificate listing
        certificates = cert_manager.list_certificates()
        assert len(certificates) == 1
        assert certificates[0]['common_name'].startswith('spiffe://')
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

class TestInfrastructureSecurityScanner:
    """Test infrastructure security scanning"""
    
    def setup_method(self):
        """Setup test environment"""
        self.scanner = InfrastructureSecurityScanner()
        self.temp_dir = tempfile.mkdtemp()
        
    def test_terraform_security_scanning(self):
        """Test Terraform security scanning"""
        # Create test Terraform file with security issues
        terraform_dir = os.path.join(self.temp_dir, "terraform")
        os.makedirs(terraform_dir)
        
        terraform_content = """
resource "aws_s3_bucket" "example" {
  bucket = "my-bucket"
  acl    = "public-read"  # Security issue
}

resource "aws_security_group" "example" {
  ingress {
    cidr_blocks = ["0.0.0.0/0"]  # Security issue
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
  }
}

variable "password" {
  default = "hardcoded-secret"  # Security issue
}
"""
        
        with open(os.path.join(terraform_dir, "main.tf"), "w") as f:
            f.write(terraform_content)
        
        # Scan the directory
        result = self.scanner.terraform_scanner.scan_directory(terraform_dir)
        
        assert result.total_files == 1
        assert len(result.findings) > 0
        
        # Check for specific security findings
        finding_titles = [f.title for f in result.findings]
        assert any("bucket" in title.lower() for title in finding_titles)
        assert any("security group" in title.lower() for title in finding_titles)
        
    def test_helm_security_scanning(self):
        """Test Helm chart security scanning"""
        # Create test Helm chart with security issues
        chart_dir = os.path.join(self.temp_dir, "helm", "test-chart")
        os.makedirs(chart_dir)
        os.makedirs(os.path.join(chart_dir, "templates"))
        
        # Create values.yaml with security issues
        values_content = {
            "securityContext": {
                "runAsUser": 0,  # Root user - security issue
                "runAsNonRoot": False
            },
            "resources": {},  # Missing resource limits - security issue
            "networkPolicy": {
                "enabled": False  # Network policy disabled - security issue
            }
        }
        
        with open(os.path.join(chart_dir, "values.yaml"), "w") as f:
            yaml.dump(values_content, f)
        
        # Create template with security issues
        template_content = """
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: app
        securityContext:
          privileged: true  # Security issue
          runAsUser: 0      # Security issue
"""
        
        with open(os.path.join(chart_dir, "templates", "deployment.yaml"), "w") as f:
            f.write(template_content)
        
        # Scan the chart
        result = self.scanner.helm_scanner.scan_chart(chart_dir)
        
        assert result.total_files == 2
        assert len(result.findings) > 0
        
    def test_security_report_generation(self):
        """Test security report generation"""
        # Create mock scan results
        from infrastructure.terraform_security_scanner import ScanResult, SecurityFinding, SeverityLevel
        
        findings = [
            SecurityFinding(
                rule_id="TEST001",
                severity=SeverityLevel.CRITICAL,
                title="Test Critical Finding",
                description="This is a test critical finding",
                file_path="/test/file.tf",
                line_number=10,
                resource="test_resource",
                remediation="Fix this issue"
            )
        ]
        
        results = {
            "terraform": ScanResult(
                scan_id="test123",
                timestamp="1234567890",
                total_files=1,
                findings=findings,
                passed_checks=0,
                failed_checks=1,
                skipped_checks=0
            )
        }
        
        report = self.scanner.generate_report(results)
        
        assert "Infrastructure Security Scan" in report
        assert "CRITICAL" in report
        assert "Test Critical Finding" in report
        
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

class TestSecurityIntegration:
    """Integration tests for security components"""
    
    def test_end_to_end_security_flow(self):
        """Test end-to-end security flow"""
        # This test would verify that all security components work together
        # In a real implementation, this would test:
        # 1. Zero-trust gateway authorizes request
        # 2. mTLS certificates are used for service communication
        # 3. Container security policies are enforced
        # 4. Infrastructure scanning detects issues
        
        # For now, we'll just verify components can be initialized together
        jwt_secret = "test-secret"
        gateway = ZeroTrustGateway(jwt_secret)
        policy_manager = ContainerSecurityPolicyManager()
        scanner = InfrastructureSecurityScanner()
        
        assert gateway is not None
        assert policy_manager is not None
        assert scanner is not None
        
        # Verify they have expected functionality
        assert hasattr(gateway, 'authorize_request')
        assert hasattr(policy_manager, 'validate_pod_security')
        assert hasattr(scanner, 'scan_infrastructure')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])