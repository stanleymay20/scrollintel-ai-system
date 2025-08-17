"""
Test suite for Application Security Framework
Tests all components: SAST/DAST, RASP, API Gateway, Secrets Manager, Input Validation
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Import security components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "security"))

from application.sast_dast_scanner import (
    SecurityScanner, ScanType, SeverityLevel, SecurityFinding, ScanResult
)
from application.rasp_protection import (
    RASPEngine, ThreatType, ThreatSeverity, ResponseAction, ThreatEvent
)
from application.api_gateway import (
    SecureAPIGateway, APIEndpoint, RateLimitRule, RateLimitType, AuthenticationMethod
)
from application.secrets_manager import (
    SecretsManager, SecretType, SecretProvider, SecretMetadata
)
from application.input_validation import (
    InputValidator, ValidationRule, InputType, ValidationResult, CommonValidationRules
)

class TestSASTDASTScanner:
    """Test SAST/DAST security scanner"""
    
    @pytest.fixture
    def scanner_config(self):
        return {
            'security_gates': {
                'max_critical': 0,
                'max_high': 5,
                'max_medium': 20
            },
            'scan_tools': {
                'semgrep': {'enabled': True},
                'bandit': {'enabled': True},
                'safety': {'enabled': True}
            },
            'results_storage': '/tmp/test_security_scans'
        }
    
    @pytest.fixture
    def security_scanner(self, scanner_config):
        return SecurityScanner(scanner_config)
    
    def test_scanner_initialization(self, security_scanner):
        """Test scanner initialization"""
        assert security_scanner.config is not None
        assert security_scanner.security_gates is not None
        assert security_scanner.scan_tools is not None
    
    def test_security_finding_creation(self):
        """Test security finding creation"""
        finding = SecurityFinding(
            id="test_001",
            title="SQL Injection",
            description="Potential SQL injection vulnerability",
            severity=SeverityLevel.HIGH,
            scan_type=ScanType.SAST,
            file_path="test.py",
            line_number=42
        )
        
        assert finding.id == "test_001"
        assert finding.severity == SeverityLevel.HIGH
        assert finding.scan_type == ScanType.SAST
    
    def test_scan_result_creation(self, security_scanner):
        """Test scan result creation"""
        findings = [
            SecurityFinding(
                id="test_001",
                title="Test Finding",
                description="Test description",
                severity=SeverityLevel.HIGH,
                scan_type=ScanType.SAST
            )
        ]
        
        result = security_scanner._create_scan_result("test_scan", ScanType.SAST, findings)
        
        assert result.scan_id == "test_scan"
        assert result.scan_type == ScanType.SAST
        assert result.total_vulnerabilities == 1
        assert result.high_count == 1
    
    def test_security_gate_evaluation(self, security_scanner):
        """Test security gate evaluation"""
        # Should pass with low numbers
        assert security_scanner._evaluate_security_gate(0, 2, 10, 5) == True
        
        # Should fail with too many critical
        assert security_scanner._evaluate_security_gate(1, 2, 10, 5) == False
        
        # Should fail with too many high
        assert security_scanner._evaluate_security_gate(0, 10, 10, 5) == False
    
    @pytest.mark.asyncio
    async def test_comprehensive_scan(self, security_scanner):
        """Test comprehensive security scan"""
        # Create test directory with sample code
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.py")
            with open(test_file, "w") as f:
                f.write("""
import subprocess
def unsafe_function(user_input):
    subprocess.run(f"ls {user_input}", shell=True)
""")
            
            # Mock the actual scanner tools since they may not be installed
            with patch.object(security_scanner, '_run_semgrep', return_value=[]):
                with patch.object(security_scanner, '_run_bandit', return_value=[]):
                    results = await security_scanner.run_comprehensive_scan(
                        temp_dir, [ScanType.SAST]
                    )
                    
                    assert ScanType.SAST in results
                    assert isinstance(results[ScanType.SAST], ScanResult)

class TestRASPProtection:
    """Test RASP protection system"""
    
    @pytest.fixture
    def rasp_config(self):
        return {
            'enabled': True,
            'learning_mode': False,
            'anomaly_threshold': 0.8,
            'max_requests_per_5min': 100
        }
    
    @pytest.fixture
    def rasp_engine(self, rasp_config):
        return RASPEngine(rasp_config)
    
    def test_rasp_initialization(self, rasp_engine):
        """Test RASP engine initialization"""
        assert rasp_engine.enabled == True
        assert rasp_engine.learning_mode == False
        assert len(rasp_engine.attack_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_sql_injection_detection(self, rasp_engine):
        """Test SQL injection detection"""
        request_data = {
            'source_ip': '192.168.1.100',
            'user_id': 'test_user',
            'path': '/api/query',
            'method': 'POST',
            'headers': {'User-Agent': 'TestAgent/1.0'},
            'body': "'; DROP TABLE users; --",
            'query_params': {}
        }
        
        threat_event = await rasp_engine.analyze_request(request_data)
        
        assert threat_event is not None
        assert threat_event.threat_type == ThreatType.SQL_INJECTION
        assert threat_event.severity == ThreatSeverity.CRITICAL
        assert threat_event.confidence_score >= 0.9
    
    @pytest.mark.asyncio
    async def test_xss_detection(self, rasp_engine):
        """Test XSS detection"""
        request_data = {
            'source_ip': '192.168.1.100',
            'user_id': 'test_user',
            'path': '/api/comment',
            'method': 'POST',
            'headers': {'User-Agent': 'TestAgent/1.0'},
            'body': '<script>alert("xss")</script>',
            'query_params': {}
        }
        
        threat_event = await rasp_engine.analyze_request(request_data)
        
        assert threat_event is not None
        assert threat_event.threat_type == ThreatType.XSS
        assert threat_event.severity == ThreatSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_brute_force_detection(self, rasp_engine):
        """Test brute force detection"""
        # Simulate multiple requests from same IP
        for i in range(150):  # Exceed the limit of 100
            request_data = {
                'source_ip': '192.168.1.100',
                'user_id': f'user_{i}',
                'path': '/api/login',
                'method': 'POST',
                'headers': {'User-Agent': 'TestAgent/1.0'},
                'body': f'username=user{i}&password=pass{i}',
                'query_params': {}
            }
            
            threat_event = await rasp_engine.analyze_request(request_data)
            
            if i >= 100:  # Should detect brute force after 100 requests
                assert threat_event is not None
                assert threat_event.threat_type == ThreatType.BRUTE_FORCE
                break
    
    def test_threat_summary(self, rasp_engine):
        """Test threat summary generation"""
        summary = rasp_engine.get_threat_summary()
        
        assert 'total_threats_24h' in summary
        assert 'active_blocks' in summary
        assert 'threat_types' in summary
        assert 'severity_breakdown' in summary

class TestAPIGateway:
    """Test secure API gateway"""
    
    @pytest.fixture
    def gateway_config(self):
        return {
            'jwt_secret': 'test-secret-key',
            'jwt_algorithm': 'HS256',
            'security_headers': {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY'
            }
        }
    
    @pytest.fixture
    def api_gateway(self, gateway_config):
        return SecureAPIGateway(gateway_config)
    
    def test_gateway_initialization(self, api_gateway):
        """Test API gateway initialization"""
        assert api_gateway.config is not None
        assert api_gateway.jwt_secret == 'test-secret-key'
        assert len(api_gateway.endpoints) > 0
    
    def test_api_key_management(self, api_gateway):
        """Test API key creation and management"""
        # Create API key
        api_key = api_gateway.add_api_key(
            name="test_key",
            permissions=["data.read", "data.write"]
        )
        
        assert api_key is not None
        assert len(api_key) > 10  # Should be a reasonable length
        
        # Verify key was stored
        assert len(api_gateway.api_keys) > 0
        
        # Test key revocation
        key_id = list(api_gateway.api_keys.keys())[0]
        api_gateway.revoke_api_key(key_id)
        assert api_gateway.api_keys[key_id].is_active == False
    
    def test_endpoint_configuration(self, api_gateway):
        """Test endpoint configuration"""
        endpoint_config = api_gateway._find_endpoint_config('/api/v1/auth/login', 'POST')
        
        assert endpoint_config is not None
        assert endpoint_config.auth_required == False
        assert len(endpoint_config.rate_limits) > 0
    
    def test_ip_filtering(self, api_gateway):
        """Test IP filtering functionality"""
        # Test IP in list
        assert api_gateway._ip_in_list('192.168.1.100', ['192.168.1.100']) == True
        assert api_gateway._ip_in_list('192.168.1.100', ['192.168.1.0/24']) == True
        assert api_gateway._ip_in_list('10.0.0.1', ['192.168.1.0/24']) == False
    
    def test_security_headers(self, api_gateway):
        """Test security headers"""
        headers = api_gateway.get_security_headers()
        
        assert 'X-Content-Type-Options' in headers
        assert 'X-Frame-Options' in headers
        assert headers['X-Content-Type-Options'] == 'nosniff'
    
    def test_metrics_collection(self, api_gateway):
        """Test metrics collection"""
        metrics = api_gateway.get_metrics()
        
        assert 'request_metrics' in metrics
        assert 'error_metrics' in metrics
        assert 'active_api_keys' in metrics

class TestSecretsManager:
    """Test secrets management system"""
    
    @pytest.fixture
    def secrets_config(self):
        return {
            'cache_ttl': 300,
            'master_password': 'test-password',
            'salt': b'test-salt-16byte',
            'local_storage': {
                'storage_path': '/tmp/test_secrets.json',
                'encryption_key': 'test-encryption-key'
            }
        }
    
    @pytest.fixture
    def secrets_manager(self, secrets_config):
        return SecretsManager(secrets_config)
    
    @pytest.mark.asyncio
    async def test_secret_storage_and_retrieval(self, secrets_manager):
        """Test storing and retrieving secrets"""
        # Store a secret
        secret_id = await secrets_manager.store_secret(
            name="test_api_key",
            value="test-secret-value-12345",
            secret_type=SecretType.API_KEYS,
            tags={'environment': 'test'}
        )
        
        assert secret_id is not None
        
        # Retrieve the secret
        secret = await secrets_manager.get_secret(secret_id)
        
        assert secret is not None
        assert secret.value == "test-secret-value-12345"
        assert secret.metadata.secret_type == SecretType.API_KEYS
        assert secret.metadata.tags['environment'] == 'test'
    
    @pytest.mark.asyncio
    async def test_secret_update(self, secrets_manager):
        """Test updating secrets"""
        # Store initial secret
        secret_id = await secrets_manager.store_secret(
            name="test_update",
            value="initial_value",
            secret_type=SecretType.CONFIGURATION
        )
        
        # Update the secret
        success = await secrets_manager.update_secret(secret_id, "updated_value")
        assert success == True
        
        # Verify update
        secret = await secrets_manager.get_secret(secret_id)
        assert secret.value == "updated_value"
    
    @pytest.mark.asyncio
    async def test_secret_deletion(self, secrets_manager):
        """Test deleting secrets"""
        # Store a secret
        secret_id = await secrets_manager.store_secret(
            name="test_delete",
            value="delete_me",
            secret_type=SecretType.API_KEYS
        )
        
        # Delete the secret
        success = await secrets_manager.delete_secret(secret_id)
        assert success == True
        
        # Verify deletion
        secret = await secrets_manager.get_secret(secret_id)
        assert secret is None
    
    @pytest.mark.asyncio
    async def test_secret_listing(self, secrets_manager):
        """Test listing secrets"""
        # Store multiple secrets
        await secrets_manager.store_secret(
            name="test_list_1",
            value="value1",
            secret_type=SecretType.API_KEYS,
            tags={'env': 'test'}
        )
        
        await secrets_manager.store_secret(
            name="test_list_2",
            value="value2",
            secret_type=SecretType.DATABASE_CREDENTIALS,
            tags={'env': 'prod'}
        )
        
        # List all secrets
        all_secrets = await secrets_manager.list_secrets()
        assert len(all_secrets) >= 2
        
        # List by type
        api_key_secrets = await secrets_manager.list_secrets(secret_type=SecretType.API_KEYS)
        assert len(api_key_secrets) >= 1
        
        # List by tags
        test_secrets = await secrets_manager.list_secrets(tags={'env': 'test'})
        assert len(test_secrets) >= 1
    
    @pytest.mark.asyncio
    async def test_secret_rotation(self, secrets_manager):
        """Test secret rotation"""
        # Store a secret
        secret_id = await secrets_manager.store_secret(
            name="test_rotation",
            value="original_value",
            secret_type=SecretType.API_KEYS
        )
        
        # Rotate the secret
        success = await secrets_manager.rotate_secret(secret_id)
        assert success == True
        
        # Verify rotation
        secret = await secrets_manager.get_secret(secret_id)
        assert secret.value != "original_value"

class TestInputValidation:
    """Test input validation framework"""
    
    @pytest.fixture
    def validator_config(self):
        return {
            'strict_mode': True,
            'auto_sanitize': True
        }
    
    @pytest.fixture
    def input_validator(self, validator_config):
        return InputValidator(validator_config)
    
    def test_validator_initialization(self, input_validator):
        """Test validator initialization"""
        assert input_validator.strict_mode == True
        assert input_validator.auto_sanitize == True
        assert len(input_validator.security_patterns) > 0
    
    def test_email_validation(self, input_validator):
        """Test email validation"""
        rule = ValidationRule(
            name='email',
            input_type=InputType.EMAIL,
            required=True
        )
        
        # Valid email
        result = input_validator._validate_field('email', 'test@example.com', rule)
        assert result['is_valid'] == True
        
        # Invalid email
        result = input_validator._validate_field('email', 'invalid-email', rule)
        assert result['is_valid'] == False
    
    def test_sql_injection_detection(self, input_validator):
        """Test SQL injection detection"""
        malicious_input = "'; DROP TABLE users; --"
        
        security_check = input_validator._check_security_threats(malicious_input)
        
        assert security_check['is_threat'] == True
        assert security_check['threat_type'] == 'sql_injection'
    
    def test_xss_detection(self, input_validator):
        """Test XSS detection"""
        malicious_input = '<script>alert("xss")</script>'
        
        security_check = input_validator._check_security_threats(malicious_input)
        
        assert security_check['is_threat'] == True
        assert security_check['threat_type'] == 'xss'
    
    def test_html_sanitization(self, input_validator):
        """Test HTML sanitization"""
        html_input = '<p>Safe content</p><script>alert("xss")</script>'
        
        sanitized = input_validator._sanitize_html(html_input)
        
        assert '<p>Safe content</p>' in sanitized
        assert '<script>' not in sanitized
    
    def test_filename_sanitization(self, input_validator):
        """Test filename sanitization"""
        dangerous_filename = '../../../etc/passwd'
        
        sanitized = input_validator._sanitize_filename(dangerous_filename)
        
        assert '../' not in sanitized
        assert sanitized != dangerous_filename
    
    def test_credit_card_validation(self, input_validator):
        """Test credit card validation"""
        # Valid credit card (test number)
        valid_cc = '4111111111111111'
        assert input_validator._is_valid_credit_card(valid_cc) == True
        
        # Invalid credit card
        invalid_cc = '1234567890123456'
        assert input_validator._is_valid_credit_card(invalid_cc) == False
    
    def test_comprehensive_validation(self, input_validator):
        """Test comprehensive input validation"""
        data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'age': '25',
            'bio': '<p>Hello world</p><script>alert("xss")</script>'
        }
        
        rules = {
            'username': ValidationRule(
                name='username',
                input_type=InputType.TEXT,
                required=True,
                min_length=3,
                max_length=50
            ),
            'email': ValidationRule(
                name='email',
                input_type=InputType.EMAIL,
                required=True
            ),
            'age': ValidationRule(
                name='age',
                input_type=InputType.NUMBER,
                required=True
            ),
            'bio': ValidationRule(
                name='bio',
                input_type=InputType.HTML,
                required=False
            )
        }
        
        result = input_validator.validate_input(data, rules)
        
        assert result.is_valid == True
        assert result.result == ValidationResult.SANITIZED  # Due to HTML sanitization
        assert '<script>' not in result.sanitized_data['bio']
        assert result.sanitized_data['age'] == 25  # Should be converted to int
    
    def test_common_validation_rules(self):
        """Test common validation rules"""
        user_rules = CommonValidationRules.user_registration()
        
        assert 'username' in user_rules
        assert 'email' in user_rules
        assert 'password' in user_rules
        
        api_rules = CommonValidationRules.api_request()
        
        assert 'data' in api_rules
        assert 'format' in api_rules

class TestIntegration:
    """Integration tests for all security components"""
    
    @pytest.fixture
    def security_components(self):
        """Create all security components for integration testing"""
        # Input validator
        validator = InputValidator({'strict_mode': True, 'auto_sanitize': True})
        
        # RASP engine
        rasp = RASPEngine({
            'enabled': True,
            'learning_mode': False,
            'max_requests_per_5min': 100
        })
        
        # API gateway
        gateway = SecureAPIGateway({
            'jwt_secret': 'test-secret',
            'security_headers': {'X-Test': 'test'}
        })
        
        # Secrets manager
        secrets = SecretsManager({
            'cache_ttl': 300,
            'master_password': 'test-password',
            'salt': b'test-salt-16byte',
            'local_storage': {
                'storage_path': '/tmp/test_integration_secrets.json',
                'encryption_key': 'test-key'
            }
        })
        
        return {
            'validator': validator,
            'rasp': rasp,
            'gateway': gateway,
            'secrets': secrets
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_request_processing(self, security_components):
        """Test end-to-end request processing through all security layers"""
        validator = security_components['validator']
        rasp = security_components['rasp']
        gateway = security_components['gateway']
        
        # 1. Input validation
        input_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'message': 'Hello world!'
        }
        
        rules = {
            'username': ValidationRule(
                name='username',
                input_type=InputType.TEXT,
                required=True,
                min_length=3
            ),
            'email': ValidationRule(
                name='email',
                input_type=InputType.EMAIL,
                required=True
            ),
            'message': ValidationRule(
                name='message',
                input_type=InputType.TEXT,
                required=True,
                max_length=1000
            )
        }
        
        validation_result = validator.validate_input(input_data, rules)
        assert validation_result.is_valid == True
        
        # 2. RASP analysis
        request_data = {
            'source_ip': '127.0.0.1',
            'user_id': 'test_user',
            'path': '/api/message',
            'method': 'POST',
            'headers': {'User-Agent': 'TestAgent/1.0'},
            'body': json.dumps(validation_result.sanitized_data),
            'query_params': {}
        }
        
        threat_event = await rasp.analyze_request(request_data)
        
        # Should not detect threat for legitimate request
        assert threat_event is None or not threat_event.blocked
        
        # 3. API Gateway processing would happen here
        # (This would typically be done in middleware)
        
        print("✅ End-to-end request processing test passed")
    
    @pytest.mark.asyncio
    async def test_malicious_request_blocking(self, security_components):
        """Test that malicious requests are properly blocked"""
        validator = security_components['validator']
        rasp = security_components['rasp']
        
        # Malicious input data
        malicious_data = {
            'username': '<script>alert("xss")</script>',
            'email': 'test@example.com',
            'query': "'; DROP TABLE users; --"
        }
        
        rules = {
            'username': ValidationRule(
                name='username',
                input_type=InputType.TEXT,
                required=True
            ),
            'email': ValidationRule(
                name='email',
                input_type=InputType.EMAIL,
                required=True
            ),
            'query': ValidationRule(
                name='query',
                input_type=InputType.TEXT,
                required=True
            )
        }
        
        # 1. Input validation should catch XSS
        validation_result = validator.validate_input(malicious_data, rules)
        
        # Should either be invalid or sanitized
        if validation_result.is_valid:
            assert validation_result.result == ValidationResult.SANITIZED
            assert '<script>' not in validation_result.sanitized_data['username']
        
        # 2. RASP should catch SQL injection
        request_data = {
            'source_ip': '192.168.1.100',
            'user_id': 'attacker',
            'path': '/api/query',
            'method': 'POST',
            'headers': {'User-Agent': 'AttackBot/1.0'},
            'body': malicious_data['query'],
            'query_params': {}
        }
        
        threat_event = await rasp.analyze_request(request_data)
        
        assert threat_event is not None
        assert threat_event.threat_type == ThreatType.SQL_INJECTION
        assert threat_event.blocked == True or threat_event.response_action == ResponseAction.BLOCK
        
        print("✅ Malicious request blocking test passed")
    
    @pytest.mark.asyncio
    async def test_secrets_integration(self, security_components):
        """Test secrets management integration"""
        secrets = security_components['secrets']
        
        # Store API key for gateway
        api_key_secret_id = await secrets.store_secret(
            name="gateway_api_key",
            value="super-secret-api-key-12345",
            secret_type=SecretType.API_KEYS,
            tags={'component': 'api_gateway', 'environment': 'test'}
        )
        
        # Store database credentials
        db_secret_id = await secrets.store_secret(
            name="database_credentials",
            value={
                "username": "dbuser",
                "password": "dbpass123",
                "host": "localhost",
                "port": 5432,
                "database": "testdb"
            },
            secret_type=SecretType.DATABASE_CREDENTIALS,
            tags={'component': 'database', 'environment': 'test'}
        )
        
        # Retrieve secrets
        api_key_secret = await secrets.get_secret(api_key_secret_id)
        db_secret = await secrets.get_secret(db_secret_id)
        
        assert api_key_secret is not None
        assert api_key_secret.value == "super-secret-api-key-12345"
        
        assert db_secret is not None
        assert db_secret.value['username'] == "dbuser"
        assert db_secret.value['password'] == "dbpass123"
        
        # Test secret rotation
        rotation_success = await secrets.rotate_secret(api_key_secret_id)
        assert rotation_success == True
        
        # Verify rotation
        rotated_secret = await secrets.get_secret(api_key_secret_id)
        assert rotated_secret.value != "super-secret-api-key-12345"
        
        print("✅ Secrets integration test passed")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])