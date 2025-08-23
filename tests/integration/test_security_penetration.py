"""
Security Penetration Testing Suite
Comprehensive security testing for all system components
"""
import pytest
import asyncio
import hashlib
import jwt
import time
import base64
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np

from scrollintel.security.enterprise_security_framework import EnterpriseSecurityFramework
from scrollintel.security.auth import AuthenticationManager
from scrollintel.core.realtime_orchestration_engine import RealtimeOrchestrationEngine
from scrollintel.api.middleware.security_middleware import SecurityMiddleware
from scrollintel.models.security_models import SecurityEvent, ThreatLevel


class TestAuthenticationSecurity:
    """Test authentication and authorization security"""
    
    @pytest.fixture
    def auth_manager(self):
        """Create authentication manager for testing"""
        return AuthenticationManager()
    
    @pytest.fixture
    def security_framework(self):
        """Create security framework for testing"""
        return EnterpriseSecurityFramework()
    
    @pytest.mark.asyncio
    async def test_brute_force_protection(self, auth_manager):
        """Test protection against brute force attacks"""
        
        # Simulate brute force attack
        username = "test_user"
        failed_attempts = []
        
        for attempt in range(10):
            try:
                result = await auth_manager.authenticate(
                    username=username,
                    password=f"wrong_password_{attempt}",
                    ip_address="192.168.1.100"
                )
                failed_attempts.append(result)
            except Exception as e:
                failed_attempts.append({'error': str(e), 'attempt': attempt})
        
        # Validate brute force protection
        assert len(failed_attempts) >= 5, "Should have multiple failed attempts"
        
        # Check if account gets locked after threshold
        lockout_detected = any('locked' in str(attempt).lower() or 'blocked' in str(attempt).lower() 
                              for attempt in failed_attempts[-3:])
        
        assert lockout_detected, "Account should be locked after multiple failed attempts"
        
        # Test rate limiting
        rapid_attempts = []
        start_time = time.time()
        
        for i in range(5):
            try:
                result = await auth_manager.authenticate(
                    username=username,
                    password="wrong_password",
                    ip_address="192.168.1.100"
                )
                rapid_attempts.append(result)
            except Exception as e:
                rapid_attempts.append({'error': str(e), 'timestamp': time.time()})
            
            if i < 4:  # Don't sleep after last attempt
                await asyncio.sleep(0.1)  # Rapid attempts
        
        total_time = time.time() - start_time
        
        # Validate rate limiting (should slow down or block rapid attempts)
        rate_limiting_active = total_time > 2.0 or any('rate' in str(attempt).lower() 
                                                      for attempt in rapid_attempts)
        
        assert rate_limiting_active, "Rate limiting should be active for rapid attempts"
    
    @pytest.mark.asyncio
    async def test_session_security(self, auth_manager):
        """Test session management security"""
        
        # Test valid authentication
        auth_result = await auth_manager.authenticate(
            username="valid_user",
            password="correct_password",
            ip_address="192.168.1.50"
        )
        
        assert auth_result['authenticated'], "Valid authentication should succeed"
        session_token = auth_result['session_token']
        
        # Test session validation
        session_valid = await auth_manager.validate_session(session_token)
        assert session_valid['valid'], "Valid session should be accepted"
        
        # Test session hijacking protection
        # Attempt to use session from different IP
        hijack_attempt = await auth_manager.validate_session(
            session_token,
            ip_address="10.0.0.1"  # Different IP
        )
        
        assert not hijack_attempt['valid'], "Session should be invalid from different IP"
        
        # Test session timeout
        # Mock expired session
        with patch('time.time', return_value=time.time() + 3600):  # 1 hour later
            expired_session = await auth_manager.validate_session(session_token)
            assert not expired_session['valid'], "Expired session should be invalid"
        
        # Test concurrent session limits
        concurrent_sessions = []
        for i in range(5):
            session_result = await auth_manager.authenticate(
                username="valid_user",
                password="correct_password",
                ip_address=f"192.168.1.{50 + i}"
            )
            concurrent_sessions.append(session_result)
        
        # Validate concurrent session limits
        valid_sessions = sum(1 for session in concurrent_sessions if session.get('authenticated'))
        assert valid_sessions <= 3, "Should limit concurrent sessions per user"
    
    @pytest.mark.asyncio
    async def test_jwt_token_security(self, auth_manager):
        """Test JWT token security and validation"""
        
        # Generate JWT token
        payload = {
            'user_id': 'test_user_123',
            'role': 'user',
            'exp': datetime.utcnow() + timedelta(hours=1),
            'iat': datetime.utcnow(),
            'iss': 'scrollintel'
        }
        
        secret_key = "test_secret_key_for_jwt"
        token = jwt.encode(payload, secret_key, algorithm='HS256')
        
        # Test valid token
        decoded = await auth_manager.validate_jwt_token(token, secret_key)
        assert decoded['valid'], "Valid JWT should be accepted"
        assert decoded['payload']['user_id'] == 'test_user_123'
        
        # Test token tampering
        tampered_token = token[:-5] + "XXXXX"  # Tamper with signature
        tampered_result = await auth_manager.validate_jwt_token(tampered_token, secret_key)
        assert not tampered_result['valid'], "Tampered JWT should be rejected"
        
        # Test expired token
        expired_payload = payload.copy()
        expired_payload['exp'] = datetime.utcnow() - timedelta(hours=1)  # Expired
        expired_token = jwt.encode(expired_payload, secret_key, algorithm='HS256')
        
        expired_result = await auth_manager.validate_jwt_token(expired_token, secret_key)
        assert not expired_result['valid'], "Expired JWT should be rejected"
        
        # Test algorithm confusion attack
        # Try to use 'none' algorithm
        none_token = jwt.encode(payload, '', algorithm='none')
        none_result = await auth_manager.validate_jwt_token(none_token, secret_key)
        assert not none_result['valid'], "JWT with 'none' algorithm should be rejected"
    
    @pytest.mark.asyncio
    async def test_privilege_escalation_protection(self, auth_manager, security_framework):
        """Test protection against privilege escalation"""
        
        # Create user with limited privileges
        limited_user = {
            'user_id': 'limited_user',
            'role': 'viewer',
            'permissions': ['read_data', 'view_reports']
        }
        
        # Test normal authorized operation
        read_result = await security_framework.authorize_action(
            user=limited_user,
            action='read_data',
            resource='customer_data'
        )
        assert read_result['authorized'], "User should be able to read data"
        
        # Test unauthorized operation (privilege escalation attempt)
        admin_result = await security_framework.authorize_action(
            user=limited_user,
            action='delete_data',
            resource='customer_data'
        )
        assert not admin_result['authorized'], "User should not be able to delete data"
        
        # Test role manipulation attempt
        escalation_attempts = [
            {'user_id': 'limited_user', 'role': 'admin'},  # Role change
            {'user_id': 'limited_user', 'permissions': ['admin_access']},  # Permission addition
            {'user_id': 'admin', 'role': 'viewer'},  # User ID spoofing
        ]
        
        for attempt in escalation_attempts:
            escalation_result = await security_framework.detect_privilege_escalation(
                original_user=limited_user,
                modified_user=attempt
            )
            assert escalation_result['escalation_detected'], f"Should detect escalation: {attempt}"
        
        # Test horizontal privilege escalation
        user_a = {'user_id': 'user_a', 'role': 'user', 'customer_id': '123'}
        user_b_data_access = await security_framework.authorize_action(
            user=user_a,
            action='read_data',
            resource='customer_data',
            resource_owner='456'  # Different customer
        )
        assert not user_b_data_access['authorized'], "User should not access other user's data"


class TestDataSecurity:
    """Test data protection and encryption security"""
    
    @pytest.fixture
    def security_framework(self):
        return EnterpriseSecurityFramework()
    
    @pytest.mark.asyncio
    async def test_data_encryption_security(self, security_framework):
        """Test data encryption and decryption security"""
        
        # Test sensitive data encryption
        sensitive_data = {
            'ssn': '123-45-6789',
            'credit_card': '4111-1111-1111-1111',
            'email': 'user@example.com',
            'phone': '+1-555-123-4567'
        }
        
        # Encrypt data
        encrypted_data = await security_framework.encrypt_sensitive_data(sensitive_data)
        
        # Validate encryption
        assert encrypted_data['ssn'] != sensitive_data['ssn'], "SSN should be encrypted"
        assert encrypted_data['credit_card'] != sensitive_data['credit_card'], "Credit card should be encrypted"
        
        # Test decryption with valid key
        decrypted_data = await security_framework.decrypt_sensitive_data(
            encrypted_data,
            encryption_key="valid_key"
        )
        assert decrypted_data['ssn'] == sensitive_data['ssn'], "Decryption should restore original data"
        
        # Test decryption with invalid key
        try:
            invalid_decrypt = await security_framework.decrypt_sensitive_data(
                encrypted_data,
                encryption_key="invalid_key"
            )
            assert False, "Decryption with invalid key should fail"
        except Exception:
            pass  # Expected to fail
        
        # Test encryption strength
        # Same data should produce different encrypted values (due to salt/IV)
        encrypted_again = await security_framework.encrypt_sensitive_data(sensitive_data)
        assert encrypted_data['ssn'] != encrypted_again['ssn'], "Encryption should use random salt/IV"
    
    @pytest.mark.asyncio
    async def test_data_masking_security(self, security_framework):
        """Test data masking and anonymization"""
        
        # Test PII data masking
        pii_data = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
            'ssn': ['123-45-6789', '987-65-4321', '555-12-3456', '111-22-3333', '999-88-7777'],
            'email': ['john@example.com', 'jane@test.org', 'bob@company.com', 'alice@email.net', 'charlie@domain.co'],
            'salary': [50000, 75000, 60000, 80000, 55000]
        })
        
        # Apply data masking
        masked_data = await security_framework.apply_data_masking(
            pii_data,
            masking_rules={
                'ssn': 'full_mask',
                'email': 'partial_mask',
                'salary': 'range_mask',
                'name': 'pseudonymize'
            }
        )
        
        # Validate masking
        assert all(masked_data['ssn'] == '***-**-****'), "SSN should be fully masked"
        assert all('@' in email for email in masked_data['email']), "Email should preserve domain"
        assert all(name != original for name, original in zip(masked_data['name'], pii_data['name'])), "Names should be pseudonymized"
        
        # Test data anonymization
        anonymized_data = await security_framework.anonymize_dataset(
            pii_data,
            k_anonymity=3,
            quasi_identifiers=['salary', 'name']
        )
        
        # Validate k-anonymity
        group_sizes = anonymized_data.groupby(['salary', 'name']).size()
        assert all(size >= 3 for size in group_sizes), "K-anonymity should be maintained"
    
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, security_framework):
        """Test protection against SQL injection attacks"""
        
        # Common SQL injection payloads
        injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM passwords --",
            "'; INSERT INTO admin VALUES ('hacker', 'password'); --",
            "' OR 1=1 --",
            "admin'--",
            "' OR 'x'='x",
            "1' AND (SELECT COUNT(*) FROM users) > 0 --"
        ]
        
        for payload in injection_payloads:
            # Test input sanitization
            sanitized = await security_framework.sanitize_sql_input(payload)
            
            # Validate that dangerous SQL is neutralized
            dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'UNION', 'SELECT']
            payload_upper = sanitized.upper()
            
            # Should not contain dangerous SQL keywords in executable form
            for keyword in dangerous_keywords:
                if keyword in payload_upper:
                    # Should be escaped or quoted
                    assert f"'{keyword}'" in sanitized or f'"{keyword}"' in sanitized, f"SQL keyword {keyword} not properly escaped"
            
            # Test parameterized query protection
            safe_query = await security_framework.create_parameterized_query(
                "SELECT * FROM users WHERE username = ?",
                [payload]
            )
            
            # Validate that payload is treated as parameter, not SQL
            assert payload not in safe_query['query'], "Payload should not be directly in query"
            assert payload in safe_query['parameters'], "Payload should be in parameters"
    
    @pytest.mark.asyncio
    async def test_data_leakage_prevention(self, security_framework):
        """Test prevention of data leakage"""
        
        # Test error message sanitization
        sensitive_errors = [
            "Database connection failed: password='secret123' host='internal.db.com'",
            "User 'admin' authentication failed with password 'admin123'",
            "File not found: /etc/passwd",
            "SQL Error: Table 'secret_data' doesn't exist"
        ]
        
        for error in sensitive_errors:
            sanitized_error = await security_framework.sanitize_error_message(error)
            
            # Should not contain sensitive information
            sensitive_patterns = ['password=', 'passwd', 'secret', 'admin123', 'internal.db.com']
            for pattern in sensitive_patterns:
                assert pattern.lower() not in sanitized_error.lower(), f"Sensitive pattern '{pattern}' found in error message"
        
        # Test log sanitization
        log_entries = [
            "User login: username=john, password=secret123, ip=192.168.1.1",
            "API call: /api/users?ssn=123-45-6789&token=abc123xyz",
            "Database query: SELECT * FROM users WHERE credit_card='4111-1111-1111-1111'"
        ]
        
        for log_entry in log_entries:
            sanitized_log = await security_framework.sanitize_log_entry(log_entry)
            
            # Should mask sensitive data in logs
            sensitive_data = ['secret123', '123-45-6789', '4111-1111-1111-1111', 'abc123xyz']
            for data in sensitive_data:
                assert data not in sanitized_log, f"Sensitive data '{data}' found in log"
        
        # Test response data filtering
        api_response = {
            'users': [
                {'id': 1, 'name': 'John', 'ssn': '123-45-6789', 'password_hash': 'abc123'},
                {'id': 2, 'name': 'Jane', 'ssn': '987-65-4321', 'password_hash': 'def456'}
            ],
            'internal_config': {'db_password': 'secret', 'api_key': 'xyz789'}
        }
        
        filtered_response = await security_framework.filter_response_data(
            api_response,
            user_role='user'  # Non-admin user
        )
        
        # Should remove sensitive fields for non-admin users
        assert 'ssn' not in str(filtered_response), "SSN should be filtered from response"
        assert 'password_hash' not in str(filtered_response), "Password hash should be filtered"
        assert 'internal_config' not in filtered_response, "Internal config should be filtered"


class TestNetworkSecurity:
    """Test network and communication security"""
    
    @pytest.fixture
    def security_middleware(self):
        return SecurityMiddleware()
    
    @pytest.mark.asyncio
    async def test_ddos_protection(self, security_middleware):
        """Test DDoS attack protection"""
        
        # Simulate DDoS attack from single IP
        attacker_ip = "192.168.1.100"
        requests_per_second = 100
        
        blocked_requests = 0
        allowed_requests = 0
        
        # Send rapid requests
        for i in range(requests_per_second):
            request = {
                'ip_address': attacker_ip,
                'timestamp': time.time(),
                'endpoint': '/api/data',
                'user_agent': 'AttackBot/1.0'
            }
            
            result = await security_middleware.check_rate_limit(request)
            
            if result['blocked']:
                blocked_requests += 1
            else:
                allowed_requests += 1
            
            await asyncio.sleep(0.01)  # 10ms between requests
        
        # Validate DDoS protection
        block_rate = blocked_requests / requests_per_second
        assert block_rate >= 0.8, f"Should block most DDoS requests, blocked: {block_rate:.2%}"
        
        # Test distributed DDoS (multiple IPs)
        distributed_ips = [f"10.0.0.{i}" for i in range(1, 21)]  # 20 different IPs
        distributed_blocked = 0
        
        for ip in distributed_ips:
            for j in range(10):  # 10 requests per IP
                request = {
                    'ip_address': ip,
                    'timestamp': time.time(),
                    'endpoint': '/api/data'
                }
                
                result = await security_middleware.check_distributed_ddos(request)
                if result['blocked']:
                    distributed_blocked += 1
        
        # Should detect distributed attack pattern
        distributed_block_rate = distributed_blocked / (len(distributed_ips) * 10)
        assert distributed_block_rate >= 0.3, "Should detect distributed DDoS pattern"
    
    @pytest.mark.asyncio
    async def test_ssl_tls_security(self, security_middleware):
        """Test SSL/TLS security configuration"""
        
        # Test SSL certificate validation
        ssl_configs = [
            {
                'certificate': 'valid_cert.pem',
                'private_key': 'valid_key.pem',
                'protocol': 'TLSv1.3',
                'cipher_suites': ['TLS_AES_256_GCM_SHA384', 'TLS_CHACHA20_POLY1305_SHA256']
            },
            {
                'certificate': 'expired_cert.pem',
                'private_key': 'valid_key.pem',
                'protocol': 'TLSv1.2',
                'cipher_suites': ['TLS_RSA_WITH_AES_128_CBC_SHA']  # Weak cipher
            },
            {
                'certificate': 'self_signed_cert.pem',
                'private_key': 'valid_key.pem',
                'protocol': 'SSLv3',  # Deprecated protocol
                'cipher_suites': ['TLS_AES_256_GCM_SHA384']
            }
        ]
        
        for i, config in enumerate(ssl_configs):
            validation_result = await security_middleware.validate_ssl_config(config)
            
            if i == 0:  # Valid config
                assert validation_result['valid'], "Valid SSL config should pass validation"
                assert validation_result['security_score'] >= 0.8, "Should have high security score"
            else:  # Invalid configs
                assert not validation_result['valid'] or validation_result['security_score'] < 0.6, f"Invalid SSL config {i} should fail validation"
        
        # Test cipher suite security
        weak_ciphers = [
            'TLS_RSA_WITH_RC4_128_SHA',
            'TLS_RSA_WITH_DES_CBC_SHA',
            'TLS_RSA_WITH_NULL_SHA'
        ]
        
        for cipher in weak_ciphers:
            cipher_result = await security_middleware.validate_cipher_suite(cipher)
            assert not cipher_result['secure'], f"Weak cipher {cipher} should be flagged as insecure"
    
    @pytest.mark.asyncio
    async def test_api_security(self, security_middleware):
        """Test API security measures"""
        
        # Test API key validation
        api_keys = [
            {'key': 'valid_api_key_123', 'permissions': ['read', 'write'], 'expires': time.time() + 3600},
            {'key': 'expired_key_456', 'permissions': ['read'], 'expires': time.time() - 3600},
            {'key': 'revoked_key_789', 'permissions': ['read'], 'revoked': True},
            {'key': 'weak_key', 'permissions': ['admin'], 'expires': time.time() + 3600}  # Weak key
        ]
        
        for api_key in api_keys:
            validation_result = await security_middleware.validate_api_key(api_key['key'])
            
            if api_key['key'] == 'valid_api_key_123':
                assert validation_result['valid'], "Valid API key should be accepted"
            else:
                assert not validation_result['valid'], f"Invalid API key {api_key['key']} should be rejected"
        
        # Test request signature validation
        request_data = {
            'method': 'POST',
            'url': '/api/users',
            'body': '{"name": "John", "email": "john@example.com"}',
            'timestamp': str(int(time.time()))
        }
        
        # Generate valid signature
        secret_key = "api_secret_key"
        signature_string = f"{request_data['method']}{request_data['url']}{request_data['body']}{request_data['timestamp']}"
        valid_signature = hashlib.hmac.new(
            secret_key.encode(),
            signature_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Test valid signature
        signature_result = await security_middleware.validate_request_signature(
            request_data,
            valid_signature,
            secret_key
        )
        assert signature_result['valid'], "Valid signature should be accepted"
        
        # Test invalid signature
        invalid_signature = "invalid_signature_123"
        invalid_result = await security_middleware.validate_request_signature(
            request_data,
            invalid_signature,
            secret_key
        )
        assert not invalid_result['valid'], "Invalid signature should be rejected"
        
        # Test replay attack protection
        old_request = request_data.copy()
        old_request['timestamp'] = str(int(time.time()) - 3600)  # 1 hour old
        
        old_signature = hashlib.hmac.new(
            secret_key.encode(),
            f"{old_request['method']}{old_request['url']}{old_request['body']}{old_request['timestamp']}".encode(),
            hashlib.sha256
        ).hexdigest()
        
        replay_result = await security_middleware.validate_request_signature(
            old_request,
            old_signature,
            secret_key
        )
        assert not replay_result['valid'], "Old timestamp should be rejected (replay protection)"


class TestThreatDetection:
    """Test threat detection and response"""
    
    @pytest.fixture
    def security_framework(self):
        return EnterpriseSecurityFramework()
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, security_framework):
        """Test detection of anomalous behavior"""
        
        # Normal user behavior baseline
        normal_behavior = [
            {'user_id': 'user123', 'action': 'login', 'timestamp': time.time(), 'ip': '192.168.1.10'},
            {'user_id': 'user123', 'action': 'view_dashboard', 'timestamp': time.time() + 60, 'ip': '192.168.1.10'},
            {'user_id': 'user123', 'action': 'generate_report', 'timestamp': time.time() + 120, 'ip': '192.168.1.10'},
            {'user_id': 'user123', 'action': 'logout', 'timestamp': time.time() + 300, 'ip': '192.168.1.10'}
        ]
        
        # Establish baseline
        await security_framework.establish_behavior_baseline(normal_behavior)
        
        # Anomalous behaviors to test
        anomalous_behaviors = [
            # Geographic anomaly
            {'user_id': 'user123', 'action': 'login', 'timestamp': time.time(), 'ip': '203.0.113.1', 'country': 'Unknown'},
            
            # Time-based anomaly
            {'user_id': 'user123', 'action': 'bulk_download', 'timestamp': time.time(), 'hour': 3},  # 3 AM
            
            # Volume anomaly
            {'user_id': 'user123', 'action': 'api_call', 'timestamp': time.time(), 'count': 1000},  # 1000 calls in short time
            
            # Permission anomaly
            {'user_id': 'user123', 'action': 'admin_access', 'timestamp': time.time(), 'ip': '192.168.1.10'},
            
            # Data access anomaly
            {'user_id': 'user123', 'action': 'access_sensitive_data', 'timestamp': time.time(), 'data_volume': '10GB'}
        ]
        
        for behavior in anomalous_behaviors:
            anomaly_result = await security_framework.detect_anomaly(behavior)
            
            assert anomaly_result['is_anomaly'], f"Should detect anomaly in behavior: {behavior['action']}"
            assert anomaly_result['risk_score'] >= 0.7, f"Anomaly should have high risk score: {behavior['action']}"
            assert 'recommended_action' in anomaly_result, "Should provide recommended action"
    
    @pytest.mark.asyncio
    async def test_malware_detection(self, security_framework):
        """Test malware and malicious content detection"""
        
        # Simulate file uploads with various content
        test_files = [
            {
                'filename': 'document.pdf',
                'content': b'%PDF-1.4 normal document content',
                'size': 1024,
                'mime_type': 'application/pdf'
            },
            {
                'filename': 'malware.exe',
                'content': b'MZ\x90\x00suspicious_executable_content',
                'size': 2048,
                'mime_type': 'application/x-executable'
            },
            {
                'filename': 'script.js',
                'content': b'eval(atob("malicious_base64_code"))',
                'size': 512,
                'mime_type': 'application/javascript'
            },
            {
                'filename': 'image.jpg',
                'content': b'\xff\xd8\xff\xe0normal_image_data',
                'size': 4096,
                'mime_type': 'image/jpeg'
            }
        ]
        
        for file_data in test_files:
            scan_result = await security_framework.scan_for_malware(file_data)
            
            if 'malware' in file_data['filename'] or 'eval(' in file_data['content'].decode('utf-8', errors='ignore'):
                assert scan_result['threat_detected'], f"Should detect threat in {file_data['filename']}"
                assert scan_result['threat_level'] >= ThreatLevel.HIGH, "Malware should be high threat"
            else:
                assert not scan_result['threat_detected'], f"Should not flag clean file {file_data['filename']}"
        
        # Test signature-based detection
        malicious_signatures = [
            b'\x4d\x5a\x90\x00',  # PE executable header
            b'eval(',  # JavaScript eval
            b'<script>alert(',  # XSS attempt
            b'SELECT * FROM',  # SQL injection attempt
        ]
        
        for signature in malicious_signatures:
            signature_result = await security_framework.check_malicious_signature(signature)
            assert signature_result['malicious'], f"Should detect malicious signature: {signature}"
    
    @pytest.mark.asyncio
    async def test_intrusion_detection(self, security_framework):
        """Test intrusion detection system"""
        
        # Simulate network traffic patterns
        network_events = [
            # Normal traffic
            {'src_ip': '192.168.1.10', 'dst_ip': '10.0.0.5', 'port': 443, 'protocol': 'HTTPS', 'size': 1024},
            {'src_ip': '192.168.1.11', 'dst_ip': '10.0.0.5', 'port': 80, 'protocol': 'HTTP', 'size': 512},
            
            # Port scanning
            {'src_ip': '203.0.113.100', 'dst_ip': '10.0.0.5', 'port': 22, 'protocol': 'SSH', 'size': 64},
            {'src_ip': '203.0.113.100', 'dst_ip': '10.0.0.5', 'port': 23, 'protocol': 'Telnet', 'size': 64},
            {'src_ip': '203.0.113.100', 'dst_ip': '10.0.0.5', 'port': 21, 'protocol': 'FTP', 'size': 64},
            
            # Data exfiltration
            {'src_ip': '192.168.1.50', 'dst_ip': '198.51.100.1', 'port': 443, 'protocol': 'HTTPS', 'size': 1048576},  # 1MB
            
            # Brute force
            {'src_ip': '203.0.113.200', 'dst_ip': '10.0.0.5', 'port': 22, 'protocol': 'SSH', 'attempts': 100}
        ]
        
        intrusion_alerts = []
        
        for event in network_events:
            ids_result = await security_framework.analyze_network_event(event)
            
            if ids_result['suspicious']:
                intrusion_alerts.append({
                    'event': event,
                    'alert_type': ids_result['alert_type'],
                    'severity': ids_result['severity']
                })
        
        # Validate intrusion detection
        assert len(intrusion_alerts) >= 3, "Should detect multiple intrusion attempts"
        
        # Check for specific attack types
        alert_types = [alert['alert_type'] for alert in intrusion_alerts]
        assert 'port_scan' in alert_types, "Should detect port scanning"
        assert 'data_exfiltration' in alert_types or 'large_transfer' in alert_types, "Should detect data exfiltration"
        assert 'brute_force' in alert_types, "Should detect brute force attempts"
    
    @pytest.mark.asyncio
    async def test_incident_response(self, security_framework):
        """Test automated incident response"""
        
        # Simulate security incidents
        security_incidents = [
            {
                'type': 'brute_force_attack',
                'severity': 'high',
                'source_ip': '203.0.113.100',
                'target': 'login_system',
                'details': {'failed_attempts': 50, 'duration': 300}
            },
            {
                'type': 'data_breach_attempt',
                'severity': 'critical',
                'source_ip': '198.51.100.50',
                'target': 'customer_database',
                'details': {'data_accessed': 'PII', 'volume': '10000_records'}
            },
            {
                'type': 'malware_detected',
                'severity': 'medium',
                'source': 'file_upload',
                'target': 'application_server',
                'details': {'malware_type': 'trojan', 'file': 'document.pdf'}
            }
        ]
        
        for incident in security_incidents:
            response_result = await security_framework.handle_security_incident(incident)
            
            # Validate incident response
            assert response_result['response_initiated'], f"Should initiate response for {incident['type']}"
            assert 'actions_taken' in response_result, "Should specify actions taken"
            assert 'containment_measures' in response_result, "Should implement containment"
            
            # Validate response appropriateness
            if incident['severity'] == 'critical':
                assert 'isolate' in str(response_result['actions_taken']).lower(), "Critical incidents should trigger isolation"
                assert 'notify_admin' in str(response_result['actions_taken']).lower(), "Critical incidents should notify admin"
            
            if incident['type'] == 'brute_force_attack':
                assert 'block_ip' in str(response_result['actions_taken']).lower(), "Brute force should trigger IP blocking"
            
            if incident['type'] == 'malware_detected':
                assert 'quarantine' in str(response_result['actions_taken']).lower(), "Malware should be quarantined"
        
        # Test incident escalation
        critical_incident = {
            'type': 'advanced_persistent_threat',
            'severity': 'critical',
            'persistence': True,
            'lateral_movement': True,
            'data_exfiltration': True
        }
        
        escalation_result = await security_framework.escalate_incident(critical_incident)
        
        assert escalation_result['escalated'], "Critical APT should be escalated"
        assert 'security_team_notified' in escalation_result, "Security team should be notified"
        assert 'external_help_requested' in escalation_result, "Should request external help for APT"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])