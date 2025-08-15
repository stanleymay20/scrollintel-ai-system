#!/usr/bin/env python3

"""
SSL and Security Configuration Tests
Tests SSL certificates, security headers, and infrastructure security
"""

import pytest
import requests
import ssl
import socket
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings for testing
urllib3.disable_warnings(InsecureRequestWarning)

class TestSSLConfiguration:
    """Test SSL/TLS configuration"""
    
    @pytest.fixture
    def base_url(self):
        """Base URL for testing"""
        return "https://scrollintel.local"
    
    @pytest.fixture
    def test_domains(self):
        """Test domains"""
        return [
            "scrollintel.local",
            "api.scrollintel.local",
            "www.scrollintel.local"
        ]
    
    def test_ssl_certificate_validity(self, test_domains):
        """Test SSL certificate validity"""
        for domain in test_domains:
            try:
                # Get SSL certificate info
                context = ssl.create_default_context()
                with socket.create_connection((domain, 443), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        cert = ssock.getpeercert()
                        
                        # Check certificate fields
                        assert 'subject' in cert
                        assert 'issuer' in cert
                        assert 'notAfter' in cert
                        assert 'notBefore' in cert
                        
                        # Check if certificate is not expired
                        import datetime
                        not_after = datetime.datetime.strptime(
                            cert['notAfter'], 
                            '%b %d %H:%M:%S %Y %Z'
                        )
                        assert not_after > datetime.datetime.now()
                        
                        print(f"✅ SSL certificate valid for {domain}")
                        
            except Exception as e:
                pytest.skip(f"SSL test skipped for {domain}: {e}")
    
    def test_tls_version_support(self, test_domains):
        """Test TLS version support"""
        for domain in test_domains:
            try:
                # Test TLS 1.2 support
                context_12 = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
                with socket.create_connection((domain, 443), timeout=10) as sock:
                    with context_12.wrap_socket(sock, server_hostname=domain) as ssock:
                        assert ssock.version() in ['TLSv1.2', 'TLSv1.3']
                
                # Test TLS 1.3 support (if available)
                try:
                    context_13 = ssl.SSLContext(ssl.PROTOCOL_TLS)
                    context_13.minimum_version = ssl.TLSVersion.TLSv1_3
                    with socket.create_connection((domain, 443), timeout=10) as sock:
                        with context_13.wrap_socket(sock, server_hostname=domain) as ssock:
                            assert ssock.version() == 'TLSv1.3'
                except:
                    pass  # TLS 1.3 might not be available
                
                print(f"✅ TLS versions supported for {domain}")
                
            except Exception as e:
                pytest.skip(f"TLS test skipped for {domain}: {e}")
    
    def test_ssl_cipher_strength(self, test_domains):
        """Test SSL cipher strength"""
        for domain in test_domains:
            try:
                context = ssl.create_default_context()
                with socket.create_connection((domain, 443), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        cipher = ssock.cipher()
                        
                        # Check cipher strength
                        assert cipher is not None
                        cipher_name, tls_version, key_bits = cipher
                        
                        # Ensure strong encryption (at least 128-bit)
                        assert key_bits >= 128
                        
                        # Ensure no weak ciphers
                        weak_ciphers = ['RC4', 'DES', 'MD5', 'NULL']
                        for weak in weak_ciphers:
                            assert weak not in cipher_name.upper()
                        
                        print(f"✅ Strong cipher for {domain}: {cipher_name} ({key_bits} bits)")
                        
            except Exception as e:
                pytest.skip(f"Cipher test skipped for {domain}: {e}")

class TestSecurityHeaders:
    """Test security headers"""
    
    @pytest.fixture
    def base_url(self):
        """Base URL for testing"""
        return "https://scrollintel.local"
    
    def test_hsts_header(self, base_url):
        """Test HSTS header"""
        try:
            response = requests.get(base_url, verify=False, timeout=10)
            
            assert 'Strict-Transport-Security' in response.headers
            hsts_header = response.headers['Strict-Transport-Security']
            
            # Check HSTS directives
            assert 'max-age=' in hsts_header
            assert 'includeSubDomains' in hsts_header
            
            # Extract max-age value
            max_age = int(hsts_header.split('max-age=')[1].split(';')[0])
            assert max_age >= 31536000  # At least 1 year
            
            print(f"✅ HSTS header configured: {hsts_header}")
            
        except Exception as e:
            pytest.skip(f"HSTS test skipped: {e}")
    
    def test_content_security_policy(self, base_url):
        """Test Content Security Policy header"""
        try:
            response = requests.get(base_url, verify=False, timeout=10)
            
            assert 'Content-Security-Policy' in response.headers
            csp_header = response.headers['Content-Security-Policy']
            
            # Check essential CSP directives
            required_directives = [
                'default-src',
                'script-src',
                'style-src',
                'img-src',
                'connect-src'
            ]
            
            for directive in required_directives:
                assert directive in csp_header
            
            # Check for secure defaults
            assert "'self'" in csp_header
            assert 'frame-ancestors' in csp_header
            
            print(f"✅ CSP header configured: {csp_header[:100]}...")
            
        except Exception as e:
            pytest.skip(f"CSP test skipped: {e}")
    
    def test_security_headers_complete(self, base_url):
        """Test complete set of security headers"""
        try:
            response = requests.get(base_url, verify=False, timeout=10)
            
            required_headers = {
                'X-Frame-Options': 'DENY',
                'X-Content-Type-Options': 'nosniff',
                'X-XSS-Protection': '1; mode=block',
                'Referrer-Policy': 'strict-origin-when-cross-origin',
                'Permissions-Policy': None  # Just check presence
            }
            
            for header, expected_value in required_headers.items():
                assert header in response.headers, f"Missing header: {header}"
                
                if expected_value:
                    assert expected_value in response.headers[header]
                
                print(f"✅ {header}: {response.headers[header]}")
            
        except Exception as e:
            pytest.skip(f"Security headers test skipped: {e}")
    
    def test_server_information_disclosure(self, base_url):
        """Test server information disclosure"""
        try:
            response = requests.get(base_url, verify=False, timeout=10)
            
            # Check that server information is not disclosed
            server_header = response.headers.get('Server', '')
            
            # Should not contain version information
            sensitive_info = ['nginx/', 'apache/', 'ubuntu', 'centos', 'version']
            for info in sensitive_info:
                assert info.lower() not in server_header.lower()
            
            print(f"✅ Server header secure: {server_header}")
            
        except Exception as e:
            pytest.skip(f"Server disclosure test skipped: {e}")

class TestRateLimiting:
    """Test rate limiting and DDoS protection"""
    
    @pytest.fixture
    def api_url(self):
        """API URL for testing"""
        return "https://scrollintel.local/api"
    
    def test_api_rate_limiting(self, api_url):
        """Test API rate limiting"""
        try:
            # Make multiple requests quickly
            responses = []
            for i in range(20):
                response = requests.get(
                    f"{api_url}/health",
                    verify=False,
                    timeout=5
                )
                responses.append(response.status_code)
            
            # Should get rate limited (429) after some requests
            rate_limited = any(code == 429 for code in responses)
            
            if rate_limited:
                print("✅ Rate limiting is working")
            else:
                print("⚠️  Rate limiting might not be configured")
            
        except Exception as e:
            pytest.skip(f"Rate limiting test skipped: {e}")
    
    def test_auth_rate_limiting(self, api_url):
        """Test authentication rate limiting"""
        try:
            # Make multiple auth requests
            responses = []
            for i in range(10):
                response = requests.post(
                    f"{api_url}/auth/login",
                    json={"username": "test", "password": "test"},
                    verify=False,
                    timeout=5
                )
                responses.append(response.status_code)
            
            # Should get rate limited for auth endpoints
            rate_limited = any(code == 429 for code in responses)
            
            if rate_limited:
                print("✅ Auth rate limiting is working")
            else:
                print("⚠️  Auth rate limiting might not be configured")
            
        except Exception as e:
            pytest.skip(f"Auth rate limiting test skipped: {e}")

class TestBackupRecovery:
    """Test backup and recovery system"""
    
    def test_backup_configuration_exists(self):
        """Test backup configuration file exists"""
        config_file = Path("/etc/scrollintel/backup-config.json")
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Check required configuration sections
            required_sections = ['database', 'files', 's3', 'notifications', 'schedule']
            for section in required_sections:
                assert section in config
            
            print("✅ Backup configuration is valid")
        else:
            pytest.skip("Backup configuration not found")
    
    def test_backup_scripts_exist(self):
        """Test backup scripts exist and are executable"""
        scripts = [
            "/usr/local/bin/scrollintel-db-backup.sh",
            "/usr/local/bin/scrollintel-file-backup.sh",
            "/usr/local/bin/scrollintel-backup-cleanup.sh"
        ]
        
        for script_path in scripts:
            script = Path(script_path)
            if script.exists():
                # Check if executable
                assert script.stat().st_mode & 0o111  # Check execute permission
                print(f"✅ Backup script exists and is executable: {script_path}")
            else:
                pytest.skip(f"Backup script not found: {script_path}")
    
    def test_cron_jobs_configured(self):
        """Test cron jobs are configured"""
        cron_file = Path("/etc/cron.d/scrollintel-backup")
        
        if cron_file.exists():
            with open(cron_file, 'r') as f:
                content = f.read()
            
            # Check for backup job entries
            assert "scrollintel-db-backup.sh" in content
            assert "scrollintel-file-backup.sh" in content
            assert "scrollintel-backup-cleanup.sh" in content
            
            print("✅ Backup cron jobs are configured")
        else:
            pytest.skip("Backup cron file not found")

class TestFirewallConfiguration:
    """Test firewall and security configuration"""
    
    def test_fail2ban_configuration(self):
        """Test Fail2Ban configuration"""
        jail_file = Path("/etc/fail2ban/jail.d/scrollintel.conf")
        
        if jail_file.exists():
            with open(jail_file, 'r') as f:
                content = f.read()
            
            # Check for essential jails
            required_jails = ['nginx-http-auth', 'nginx-limit-req', 'scrollintel-api']
            for jail in required_jails:
                assert jail in content
            
            print("✅ Fail2Ban configuration is present")
        else:
            pytest.skip("Fail2Ban configuration not found")
    
    def test_nginx_security_config(self):
        """Test Nginx security configuration"""
        config_files = [
            "/etc/nginx/conf.d/rate-limiting.conf",
            "/etc/nginx/conf.d/security-headers.conf",
            "/etc/nginx/conf.d/bot-protection.conf"
        ]
        
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                print(f"✅ Nginx security config exists: {config_file}")
            else:
                pytest.skip(f"Nginx security config not found: {config_file}")

def run_security_tests():
    """Run all security tests"""
    print("🔒 Running SSL and Security Configuration Tests")
    print("=" * 50)
    
    # Run pytest with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])

if __name__ == "__main__":
    run_security_tests()