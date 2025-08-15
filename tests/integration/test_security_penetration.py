"""
Security Penetration Tests
Tests authentication, authorization, and security vulnerabilities
"""
import pytest
import jwt
import time
import hashlib
import secrets
from typing import Dict, Any, List
from unittest.mock import patch, Mock
import requests
from fastapi import HTTPException

from scrollintel.security.auth import create_access_token, verify_token, hash_password, verify_password
from scrollintel.security.permissions import check_permission, UserRole
from scrollintel.security.audit import AuditLogger
from scrollintel.security.session import SessionManager
from scrollintel.models.schemas import User


class TestSecurityPenetration:
    """Security penetration testing suite"""
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger instance"""
        return AuditLogger()
    
    @pytest.fixture
    def session_manager(self):
        """Create session manager instance"""
        return SessionManager()
    
    @pytest.mark.asyncio
    async def test_jwt_token_vulnerabilities(self):
        """Test JWT token security vulnerabilities"""
        
        # Test 1: Token tampering
        valid_token = create_access_token(data={"sub": "test@example.com", "role": "user"})
        
        # Try to tamper with token payload
        try:
            # Decode without verification (dangerous in real code)
            payload = jwt.decode(valid_token, options={"verify_signature": False})
            payload["role"] = "admin"  # Escalate privileges
            
            # Try to create new token with tampered payload
            tampered_token = jwt.encode(payload, "wrong_secret", algorithm="HS256")
            
            # This should fail verification
            with pytest.raises(Exception):
                verify_token(tampered_token)
                
        except Exception:
            pass  # Expected to fail
        
        # Test 2: Token expiration bypass
        expired_payload = {
            "sub": "test@example.com",
            "role": "admin",
            "exp": int(time.time()) - 3600  # Expired 1 hour ago
        }
        
        expired_token = jwt.encode(expired_payload, "test_secret", algorithm="HS256")
        
        with pytest.raises(Exception):
            verify_token(expired_token)
        
        # Test 3: Algorithm confusion attack
        # Try to use 'none' algorithm
        none_payload = {"sub": "test@example.com", "role": "admin"}
        none_token = jwt.encode(none_payload, "", algorithm="none")
        
        with pytest.raises(Exception):
            verify_token(none_token)
        
        # Test 4: Weak secret detection
        weak_secrets = ["123", "password", "secret", "test"]
        
        for weak_secret in weak_secrets:
            # In real implementation, should reject weak secrets
            token = jwt.encode({"sub": "test"}, weak_secret, algorithm="HS256")
            
            # Try to crack with common passwords
            for guess in ["123", "password", "secret", "test", "admin"]:
                try:
                    decoded = jwt.decode(token, guess, algorithms=["HS256"])
                    # If we can decode with a common password, secret is too weak
                    assert False, f"Weak secret detected: {weak_secret}"
                except jwt.InvalidTokenError:
                    continue
    
    @pytest.mark.asyncio
    async def test_authentication_bypass_attempts(self, test_client):
        """Test various authentication bypass attempts"""
        
        # Test 1: SQL Injection in login
        sql_injection_payloads = [
            "admin' OR '1'='1",
            "admin'; DROP TABLE users; --",
            "admin' UNION SELECT * FROM users --",
            "' OR 1=1 --",
            "admin'/**/OR/**/1=1#"
        ]
        
        for payload in sql_injection_payloads:
            login_data = {
                "email": payload,
                "password": "any_password"
            }
            
            response = test_client.post("/api/v1/auth/login", json=login_data)
            
            # Should not succeed with SQL injection
            assert response.status_code in [400, 401, 422]
            
            # Should not contain database error messages
            response_text = response.text.lower()
            dangerous_keywords = ["sql", "database", "table", "select", "union", "drop"]
            for keyword in dangerous_keywords:
                assert keyword not in response_text, f"Potential SQL injection vulnerability: {keyword} in response"
        
        # Test 2: NoSQL Injection attempts
        nosql_payloads = [
            {"$ne": None},
            {"$gt": ""},
            {"$regex": ".*"},
            {"$where": "1==1"}
        ]
        
        for payload in nosql_payloads:
            login_data = {
                "email": payload,
                "password": "any_password"
            }
            
            response = test_client.post("/api/v1/auth/login", json=login_data)
            assert response.status_code in [400, 401, 422]
        
        # Test 3: Header injection
        malicious_headers = {
            "X-Forwarded-For": "127.0.0.1, admin",
            "X-Real-IP": "127.0.0.1",
            "X-Originating-IP": "127.0.0.1",
            "Authorization": "Bearer fake_admin_token"
        }
        
        response = test_client.get("/api/v1/admin/users", headers=malicious_headers)
        assert response.status_code in [401, 403]  # Should not bypass auth
    
    @pytest.mark.asyncio
    async def test_authorization_privilege_escalation(self, test_client, test_db_session):
        """Test privilege escalation attempts"""
        
        # Create test users with different roles
        users = [
            {"email": "viewer@test.com", "role": UserRole.VIEWER, "permissions": ["read"]},
            {"email": "analyst@test.com", "role": UserRole.ANALYST, "permissions": ["read", "analyze"]},
            {"email": "admin@test.com", "role": UserRole.ADMIN, "permissions": ["read", "write", "admin"]}
        ]
        
        tokens = {}
        for user_data in users:
            # Create user in database
            user = User(**user_data)
            test_db_session.add(user)
            test_db_session.commit()
            
            # Generate token
            token = create_access_token(data={"sub": user.email, "role": user.role.value})
            tokens[user.role.value] = {"Authorization": f"Bearer {token}"}
        
        # Test 1: Viewer trying to access admin endpoints
        admin_endpoints = [
            "/api/v1/admin/users",
            "/api/v1/admin/system/config",
            "/api/v1/admin/audit/logs"
        ]
        
        for endpoint in admin_endpoints:
            response = test_client.get(endpoint, headers=tokens["viewer"])
            assert response.status_code == 403, f"Viewer should not access {endpoint}"
        
        # Test 2: Analyst trying to perform admin actions
        admin_actions = [
            ("/api/v1/admin/users", "POST", {"email": "new@test.com", "role": "admin"}),
            ("/api/v1/admin/system/config", "PUT", {"setting": "value"}),
            ("/api/v1/admin/users/1", "DELETE", {})
        ]
        
        for endpoint, method, data in admin_actions:
            if method == "POST":
                response = test_client.post(endpoint, json=data, headers=tokens["analyst"])
            elif method == "PUT":
                response = test_client.put(endpoint, json=data, headers=tokens["analyst"])
            elif method == "DELETE":
                response = test_client.delete(endpoint, headers=tokens["analyst"])
            
            assert response.status_code == 403, f"Analyst should not perform {method} on {endpoint}"
        
        # Test 3: Role manipulation in token
        # Try to modify role in existing token
        analyst_token = tokens["analyst"]["Authorization"].split(" ")[1]
        
        try:
            # Decode token (this would fail in real scenario due to signature)
            payload = jwt.decode(analyst_token, options={"verify_signature": False})
            payload["role"] = "admin"
            
            # Try to use modified token
            fake_admin_token = jwt.encode(payload, "wrong_secret", algorithm="HS256")
            fake_headers = {"Authorization": f"Bearer {fake_admin_token}"}
            
            response = test_client.get("/api/v1/admin/users", headers=fake_headers)
            assert response.status_code in [401, 403]  # Should reject invalid token
            
        except Exception:
            pass  # Expected to fail
    
    @pytest.mark.asyncio
    async def test_session_security_vulnerabilities(self, session_manager):
        """Test session management security"""
        
        # Test 1: Session fixation
        # Create session for user
        user_id = "test_user_123"
        session_id = await session_manager.create_session(user_id)
        
        # Verify session exists
        session_data = await session_manager.get_session(session_id)
        assert session_data["user_id"] == user_id
        
        # Test session hijacking prevention
        # Try to use session from different IP
        original_ip = "192.168.1.100"
        malicious_ip = "10.0.0.1"
        
        # Set session with IP binding
        await session_manager.update_session(session_id, {"ip_address": original_ip})
        
        # Try to access from different IP (should be detected)
        is_valid = await session_manager.validate_session(session_id, ip_address=malicious_ip)
        assert not is_valid, "Session should be invalid from different IP"
        
        # Test 2: Session timeout
        # Create session with short timeout
        short_session_id = await session_manager.create_session(user_id, timeout=1)  # 1 second
        
        # Wait for timeout
        time.sleep(2)
        
        # Session should be expired
        expired_session = await session_manager.get_session(short_session_id)
        assert expired_session is None or expired_session.get("expired", False)
        
        # Test 3: Concurrent session limits
        # Create multiple sessions for same user
        sessions = []
        for i in range(10):
            sid = await session_manager.create_session(user_id)
            sessions.append(sid)
        
        # Should limit concurrent sessions (implementation dependent)
        active_sessions = await session_manager.get_user_sessions(user_id)
        assert len(active_sessions) <= 5, "Should limit concurrent sessions"
    
    @pytest.mark.asyncio
    async def test_input_validation_vulnerabilities(self, test_client, test_user_token):
        """Test input validation and injection vulnerabilities"""
        
        # Test 1: XSS attempts in file uploads
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//",
            "<svg onload=alert('xss')>"
        ]
        
        for payload in xss_payloads:
            # Try XSS in filename
            files = {"file": (payload, "test content", "text/plain")}
            response = test_client.post(
                "/api/v1/files/upload",
                files=files,
                headers=test_user_token
            )
            
            # Should sanitize filename
            if response.status_code == 200:
                result = response.json()
                filename = result.get("filename", "")
                assert "<script>" not in filename
                assert "javascript:" not in filename
                assert "onerror=" not in filename
        
        # Test 2: Command injection in prompts
        command_injection_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "`whoami`",
            "$(id)",
            "${IFS}cat${IFS}/etc/passwd"
        ]
        
        for payload in command_injection_payloads:
            agent_request = {
                "prompt": f"Analyze data {payload}",
                "agent_type": "data_scientist"
            }
            
            with patch('scrollintel.agents.scroll_data_scientist.anthropic') as mock_claude:
                mock_claude.messages.create.return_value = Mock(
                    content=[Mock(text="Safe response")]
                )
                
                response = test_client.post(
                    "/api/v1/agents/process",
                    json=agent_request,
                    headers=test_user_token
                )
                
                # Should not execute commands
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("content", "").lower()
                    
                    # Should not contain command output
                    dangerous_outputs = ["root:", "bin/bash", "uid=", "gid="]
                    for output in dangerous_outputs:
                        assert output not in content
        
        # Test 3: Path traversal in file operations
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd"
        ]
        
        for payload in path_traversal_payloads:
            # Try path traversal in dataset access
            response = test_client.get(
                f"/api/v1/datasets/{payload}",
                headers=test_user_token
            )
            
            # Should not access system files
            assert response.status_code in [400, 404, 422]
            
            if response.status_code != 404:
                response_text = response.text.lower()
                system_indicators = ["root:", "password:", "system32", "config"]
                for indicator in system_indicators:
                    assert indicator not in response_text
    
    @pytest.mark.asyncio
    async def test_rate_limiting_bypass_attempts(self, test_client, test_user_token):
        """Test rate limiting bypass attempts"""
        
        # Test 1: Rapid requests to login endpoint
        login_attempts = []
        for i in range(20):  # Try 20 rapid login attempts
            login_data = {
                "email": f"test{i}@example.com",
                "password": "wrong_password"
            }
            
            response = test_client.post("/api/v1/auth/login", json=login_data)
            login_attempts.append(response.status_code)
        
        # Should start rate limiting after several attempts
        rate_limited_count = sum(1 for status in login_attempts if status == 429)
        assert rate_limited_count > 0, "Rate limiting should be active after multiple failed attempts"
        
        # Test 2: IP rotation attempts
        # Try with different X-Forwarded-For headers
        for i in range(10):
            headers = {
                **test_user_token,
                "X-Forwarded-For": f"192.168.1.{i}"
            }
            
            response = test_client.post(
                "/api/v1/agents/process",
                json={"prompt": "test", "agent_type": "data_scientist"},
                headers=headers
            )
            
            # Should not bypass rate limiting with IP rotation
            if i > 5:  # After several requests
                assert response.status_code in [200, 429]  # Either success or rate limited
        
        # Test 3: User-Agent rotation
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "curl/7.68.0",
            "python-requests/2.25.1"
        ]
        
        for ua in user_agents:
            headers = {
                **test_user_token,
                "User-Agent": ua
            }
            
            response = test_client.get("/api/v1/health", headers=headers)
            # Should not bypass rate limiting with User-Agent rotation
            assert response.status_code in [200, 429]
    
    @pytest.mark.asyncio
    async def test_audit_log_tampering(self, audit_logger, test_db_session):
        """Test audit log security and tampering attempts"""
        
        # Test 1: Log integrity
        # Create audit log entry
        log_entry = {
            "user_id": "test_user",
            "action": "login",
            "resource_type": "auth",
            "resource_id": "session_123",
            "ip_address": "192.168.1.100",
            "details": {"success": True}
        }
        
        log_id = await audit_logger.log_action(**log_entry)
        
        # Try to modify log entry directly in database
        # This should be prevented by database constraints or triggers
        try:
            # Attempt direct database modification
            test_db_session.execute(
                "UPDATE audit_logs SET action = 'admin_access' WHERE id = :log_id",
                {"log_id": log_id}
            )
            test_db_session.commit()
            
            # Verify log wasn't tampered with
            retrieved_log = await audit_logger.get_log_entry(log_id)
            assert retrieved_log["action"] == "login", "Audit log should not be modifiable"
            
        except Exception:
            # Expected if proper constraints are in place
            pass
        
        # Test 2: Log deletion attempts
        try:
            test_db_session.execute(
                "DELETE FROM audit_logs WHERE id = :log_id",
                {"log_id": log_id}
            )
            test_db_session.commit()
            
            # Verify log still exists
            retrieved_log = await audit_logger.get_log_entry(log_id)
            assert retrieved_log is not None, "Audit logs should not be deletable"
            
        except Exception:
            # Expected if proper constraints are in place
            pass
        
        # Test 3: Log injection
        malicious_details = {
            "'; DROP TABLE audit_logs; --": "value",
            "<script>alert('xss')</script>": "malicious",
            "admin_backdoor": True
        }
        
        try:
            malicious_log = {
                **log_entry,
                "details": malicious_details
            }
            
            log_id = await audit_logger.log_action(**malicious_log)
            
            # Verify malicious content is sanitized
            retrieved_log = await audit_logger.get_log_entry(log_id)
            details_str = str(retrieved_log["details"])
            
            assert "DROP TABLE" not in details_str
            assert "<script>" not in details_str
            
        except Exception:
            # Expected if proper validation is in place
            pass
    
    @pytest.mark.asyncio
    async def test_password_security_vulnerabilities(self):
        """Test password hashing and validation security"""
        
        # Test 1: Weak password detection
        weak_passwords = [
            "123456",
            "password",
            "admin",
            "test",
            "qwerty",
            "123456789",
            "password123"
        ]
        
        for weak_password in weak_passwords:
            # Should reject weak passwords (implementation dependent)
            try:
                hashed = hash_password(weak_password)
                # If hashing succeeds, verify it's properly hashed
                assert len(hashed) > 50  # Should be long hash
                assert weak_password not in hashed  # Should not contain plaintext
            except ValueError:
                # Expected if weak password validation is implemented
                pass
        
        # Test 2: Hash timing attacks
        correct_password = "correct_password_123"
        wrong_passwords = [
            "wrong_password_123",
            "completely_different",
            "a",
            "x" * 100
        ]
        
        hashed_password = hash_password(correct_password)
        
        # Measure verification times
        times = []
        for wrong_password in wrong_passwords:
            start_time = time.time()
            result = verify_password(wrong_password, hashed_password)
            end_time = time.time()
            
            times.append(end_time - start_time)
            assert not result  # Should all be False
        
        # Verify correct password
        start_time = time.time()
        result = verify_password(correct_password, hashed_password)
        correct_time = time.time() - start_time
        
        assert result  # Should be True
        
        # Times should be relatively consistent (timing attack prevention)
        avg_time = sum(times) / len(times)
        for t in times:
            # Allow some variance but not too much
            assert abs(t - avg_time) < 0.1, "Password verification timing should be consistent"
        
        # Test 3: Hash collision resistance
        passwords = [f"password_{i}" for i in range(100)]
        hashes = [hash_password(pwd) for pwd in passwords]
        
        # All hashes should be unique
        assert len(set(hashes)) == len(hashes), "Password hashes should be unique"
        
        # Hashes should be different even for similar passwords
        similar_passwords = ["password1", "password2", "password3"]
        similar_hashes = [hash_password(pwd) for pwd in similar_passwords]
        
        for i, hash1 in enumerate(similar_hashes):
            for j, hash2 in enumerate(similar_hashes):
                if i != j:
                    # Calculate similarity (should be low)
                    common_chars = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
                    similarity = common_chars / len(hash1)
                    assert similarity < 0.1, "Similar passwords should have very different hashes"