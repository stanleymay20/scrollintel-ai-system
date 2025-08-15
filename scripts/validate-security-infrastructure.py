#!/usr/bin/env python3

"""
Security Infrastructure Validation Script
Validates SSL, CDN, and security infrastructure configuration
"""

import os
import sys
import json
import subprocess
import requests
import ssl
import socket
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings for testing
urllib3.disable_warnings(InsecureRequestWarning)

class SecurityValidator:
    """Validates security infrastructure configuration"""
    
    def __init__(self):
        self.results = {
            "ssl": {},
            "security_headers": {},
            "rate_limiting": {},
            "backup_system": {},
            "firewall": {},
            "cdn": {},
            "overall_score": 0
        }
        self.total_checks = 0
        self.passed_checks = 0
    
    def log_result(self, category: str, check: str, passed: bool, message: str = "") -> None:
        """Log validation result"""
        self.total_checks += 1
        if passed:
            self.passed_checks += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        print(f"{status} {category}: {check}")
        if message:
            print(f"    {message}")
        
        if category not in self.results:
            self.results[category] = {}
        
        self.results[category][check] = {
            "passed": passed,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
    
    def validate_ssl_configuration(self, domains: List[str]) -> None:
        """Validate SSL/TLS configuration"""
        print("\n🔒 Validating SSL Configuration...")
        
        for domain in domains:
            try:
                # Test SSL certificate
                context = ssl.create_default_context()
                with socket.create_connection((domain, 443), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        cert = ssock.getpeercert()
                        
                        # Check certificate validity
                        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        days_until_expiry = (not_after - datetime.now()).days
                        
                        if days_until_expiry > 30:
                            self.log_result(
                                "ssl", 
                                f"Certificate validity ({domain})", 
                                True,
                                f"Valid for {days_until_expiry} days"
                            )
                        else:
                            self.log_result(
                                "ssl", 
                                f"Certificate validity ({domain})", 
                                False,
                                f"Expires in {days_until_expiry} days"
                            )
                        
                        # Check TLS version
                        tls_version = ssock.version()
                        if tls_version in ['TLSv1.2', 'TLSv1.3']:
                            self.log_result(
                                "ssl", 
                                f"TLS version ({domain})", 
                                True,
                                f"Using {tls_version}"
                            )
                        else:
                            self.log_result(
                                "ssl", 
                                f"TLS version ({domain})", 
                                False,
                                f"Using insecure {tls_version}"
                            )
                        
                        # Check cipher strength
                        cipher = ssock.cipher()
                        if cipher and cipher[2] >= 128:
                            self.log_result(
                                "ssl", 
                                f"Cipher strength ({domain})", 
                                True,
                                f"{cipher[0]} ({cipher[2]} bits)"
                            )
                        else:
                            self.log_result(
                                "ssl", 
                                f"Cipher strength ({domain})", 
                                False,
                                f"Weak cipher: {cipher}"
                            )
            
            except Exception as e:
                self.log_result(
                    "ssl", 
                    f"SSL connection ({domain})", 
                    False,
                    f"Connection failed: {e}"
                )
    
    def validate_security_headers(self, base_url: str) -> None:
        """Validate security headers"""
        print("\n🛡️  Validating Security Headers...")
        
        try:
            response = requests.get(base_url, verify=False, timeout=10)
            
            # Required security headers
            required_headers = {
                'Strict-Transport-Security': {
                    'required': True,
                    'check': lambda x: 'max-age=' in x and int(x.split('max-age=')[1].split(';')[0]) >= 31536000
                },
                'X-Frame-Options': {
                    'required': True,
                    'check': lambda x: x.upper() in ['DENY', 'SAMEORIGIN']
                },
                'X-Content-Type-Options': {
                    'required': True,
                    'check': lambda x: x.lower() == 'nosniff'
                },
                'X-XSS-Protection': {
                    'required': True,
                    'check': lambda x: '1' in x and 'mode=block' in x
                },
                'Content-Security-Policy': {
                    'required': True,
                    'check': lambda x: 'default-src' in x and "'self'" in x
                },
                'Referrer-Policy': {
                    'required': True,
                    'check': lambda x: len(x) > 0
                },
                'Permissions-Policy': {
                    'required': False,
                    'check': lambda x: len(x) > 0
                }
            }
            
            for header, config in required_headers.items():
                if header in response.headers:
                    header_value = response.headers[header]
                    if config['check'](header_value):
                        self.log_result(
                            "security_headers", 
                            header, 
                            True,
                            f"Configured: {header_value[:50]}..."
                        )
                    else:
                        self.log_result(
                            "security_headers", 
                            header, 
                            False,
                            f"Misconfigured: {header_value[:50]}..."
                        )
                else:
                    self.log_result(
                        "security_headers", 
                        header, 
                        not config['required'],
                        "Header missing"
                    )
            
            # Check server information disclosure
            server_header = response.headers.get('Server', '')
            sensitive_info = ['nginx/', 'apache/', 'version', 'ubuntu', 'centos']
            has_sensitive = any(info.lower() in server_header.lower() for info in sensitive_info)
            
            self.log_result(
                "security_headers", 
                "Server information disclosure", 
                not has_sensitive,
                f"Server header: {server_header}"
            )
        
        except Exception as e:
            self.log_result(
                "security_headers", 
                "HTTP connection", 
                False,
                f"Connection failed: {e}"
            )
    
    def validate_rate_limiting(self, api_url: str) -> None:
        """Validate rate limiting configuration"""
        print("\n⏱️  Validating Rate Limiting...")
        
        try:
            # Test general API rate limiting
            responses = []
            for i in range(15):
                try:
                    response = requests.get(
                        f"{api_url}/health",
                        verify=False,
                        timeout=2
                    )
                    responses.append(response.status_code)
                except:
                    responses.append(0)
            
            rate_limited = any(code == 429 for code in responses)
            self.log_result(
                "rate_limiting", 
                "API rate limiting", 
                rate_limited,
                f"Got {responses.count(429)} rate limit responses out of 15 requests"
            )
            
            # Test auth rate limiting
            auth_responses = []
            for i in range(8):
                try:
                    response = requests.post(
                        f"{api_url}/auth/login",
                        json={"username": "test", "password": "test"},
                        verify=False,
                        timeout=2
                    )
                    auth_responses.append(response.status_code)
                except:
                    auth_responses.append(0)
            
            auth_rate_limited = any(code == 429 for code in auth_responses)
            self.log_result(
                "rate_limiting", 
                "Auth rate limiting", 
                auth_rate_limited,
                f"Got {auth_responses.count(429)} rate limit responses out of 8 auth requests"
            )
        
        except Exception as e:
            self.log_result(
                "rate_limiting", 
                "Rate limiting test", 
                False,
                f"Test failed: {e}"
            )
    
    def validate_backup_system(self) -> None:
        """Validate backup and recovery system"""
        print("\n💾 Validating Backup System...")
        
        # Check backup configuration
        config_file = Path("/etc/scrollintel/backup-config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                required_sections = ['database', 'files', 's3', 'notifications', 'schedule']
                all_sections_present = all(section in config for section in required_sections)
                
                self.log_result(
                    "backup_system", 
                    "Configuration file", 
                    all_sections_present,
                    f"Sections present: {list(config.keys())}"
                )
            except Exception as e:
                self.log_result(
                    "backup_system", 
                    "Configuration file", 
                    False,
                    f"Invalid configuration: {e}"
                )
        else:
            self.log_result(
                "backup_system", 
                "Configuration file", 
                False,
                "Configuration file not found"
            )
        
        # Check backup scripts
        scripts = [
            "/usr/local/bin/scrollintel-db-backup.sh",
            "/usr/local/bin/scrollintel-file-backup.sh",
            "/usr/local/bin/scrollintel-backup-cleanup.sh"
        ]
        
        for script_path in scripts:
            script = Path(script_path)
            if script.exists() and script.stat().st_mode & 0o111:
                self.log_result(
                    "backup_system", 
                    f"Script {script.name}", 
                    True,
                    "Exists and executable"
                )
            else:
                self.log_result(
                    "backup_system", 
                    f"Script {script.name}", 
                    False,
                    "Missing or not executable"
                )
        
        # Check cron configuration
        cron_file = Path("/etc/cron.d/scrollintel-backup")
        if cron_file.exists():
            with open(cron_file, 'r') as f:
                content = f.read()
            
            has_db_backup = "scrollintel-db-backup.sh" in content
            has_file_backup = "scrollintel-file-backup.sh" in content
            has_cleanup = "scrollintel-backup-cleanup.sh" in content
            
            self.log_result(
                "backup_system", 
                "Cron jobs", 
                has_db_backup and has_file_backup and has_cleanup,
                f"DB: {has_db_backup}, Files: {has_file_backup}, Cleanup: {has_cleanup}"
            )
        else:
            self.log_result(
                "backup_system", 
                "Cron jobs", 
                False,
                "Cron file not found"
            )
    
    def validate_firewall_configuration(self) -> None:
        """Validate firewall and security configuration"""
        print("\n🔥 Validating Firewall Configuration...")
        
        # Check Fail2Ban configuration
        jail_file = Path("/etc/fail2ban/jail.d/scrollintel.conf")
        if jail_file.exists():
            with open(jail_file, 'r') as f:
                content = f.read()
            
            required_jails = ['nginx-http-auth', 'nginx-limit-req', 'scrollintel-api']
            jails_present = [jail for jail in required_jails if jail in content]
            
            self.log_result(
                "firewall", 
                "Fail2Ban jails", 
                len(jails_present) == len(required_jails),
                f"Configured jails: {jails_present}"
            )
        else:
            self.log_result(
                "firewall", 
                "Fail2Ban configuration", 
                False,
                "Configuration file not found"
            )
        
        # Check Nginx security configurations
        nginx_configs = [
            ("/etc/nginx/conf.d/rate-limiting.conf", "Rate limiting"),
            ("/etc/nginx/conf.d/security-headers.conf", "Security headers"),
            ("/etc/nginx/conf.d/bot-protection.conf", "Bot protection"),
            ("/etc/nginx/conf.d/geo-blocking.conf", "Geo blocking")
        ]
        
        for config_path, config_name in nginx_configs:
            config_file = Path(config_path)
            self.log_result(
                "firewall", 
                config_name, 
                config_file.exists(),
                f"Config file: {config_path}"
            )
        
        # Check DDoS monitoring service
        service_file = Path("/etc/systemd/system/ddos-monitor.service")
        if service_file.exists():
            try:
                result = subprocess.run(
                    ["systemctl", "is-enabled", "ddos-monitor"],
                    capture_output=True,
                    text=True
                )
                is_enabled = result.returncode == 0
                
                self.log_result(
                    "firewall", 
                    "DDoS monitoring service", 
                    is_enabled,
                    f"Service enabled: {is_enabled}"
                )
            except Exception as e:
                self.log_result(
                    "firewall", 
                    "DDoS monitoring service", 
                    False,
                    f"Check failed: {e}"
                )
        else:
            self.log_result(
                "firewall", 
                "DDoS monitoring service", 
                False,
                "Service file not found"
            )
    
    def validate_cdn_configuration(self, base_url: str) -> None:
        """Validate CDN configuration"""
        print("\n🌐 Validating CDN Configuration...")
        
        try:
            # Test static asset caching
            static_url = f"{base_url}/static/test.js"
            response = requests.get(static_url, verify=False, timeout=10)
            
            # Check cache headers
            cache_control = response.headers.get('Cache-Control', '')
            expires = response.headers.get('Expires', '')
            
            has_caching = 'public' in cache_control or 'max-age' in cache_control or expires
            
            self.log_result(
                "cdn", 
                "Static asset caching", 
                has_caching,
                f"Cache-Control: {cache_control}, Expires: {expires}"
            )
            
            # Check compression
            content_encoding = response.headers.get('Content-Encoding', '')
            has_compression = 'gzip' in content_encoding or 'br' in content_encoding
            
            self.log_result(
                "cdn", 
                "Content compression", 
                has_compression,
                f"Content-Encoding: {content_encoding}"
            )
            
            # Check CDN headers (Cloudflare)
            cf_ray = response.headers.get('CF-Ray', '')
            cf_cache_status = response.headers.get('CF-Cache-Status', '')
            
            has_cdn = bool(cf_ray or cf_cache_status)
            
            self.log_result(
                "cdn", 
                "CDN integration", 
                has_cdn,
                f"CF-Ray: {cf_ray}, CF-Cache-Status: {cf_cache_status}"
            )
        
        except Exception as e:
            self.log_result(
                "cdn", 
                "CDN validation", 
                False,
                f"Test failed: {e}"
            )
    
    def generate_report(self) -> Dict:
        """Generate validation report"""
        if self.total_checks > 0:
            self.results["overall_score"] = (self.passed_checks / self.total_checks) * 100
        
        print(f"\n📊 Security Infrastructure Validation Report")
        print("=" * 50)
        print(f"Overall Score: {self.results['overall_score']:.1f}%")
        print(f"Passed Checks: {self.passed_checks}/{self.total_checks}")
        
        # Category breakdown
        for category, checks in self.results.items():
            if category == "overall_score":
                continue
            
            if isinstance(checks, dict) and checks:
                passed = sum(1 for check in checks.values() if check.get('passed', False))
                total = len(checks)
                percentage = (passed / total) * 100 if total > 0 else 0
                
                print(f"\n{category.replace('_', ' ').title()}: {percentage:.1f}% ({passed}/{total})")
                
                for check_name, check_result in checks.items():
                    status = "✅" if check_result.get('passed', False) else "❌"
                    print(f"  {status} {check_name}")
        
        return self.results
    
    def save_report(self, filename: str = None) -> None:
        """Save validation report to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"security_validation_report_{timestamp}.json"
        
        report_path = Path(filename)
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_path}")

def main():
    """Main validation function"""
    print("🔒 ScrollIntel Security Infrastructure Validation")
    print("=" * 50)
    
    # Configuration
    domains = ["scrollintel.local", "api.scrollintel.local"]
    base_url = "https://scrollintel.local"
    api_url = "https://scrollintel.local/api"
    
    # Initialize validator
    validator = SecurityValidator()
    
    try:
        # Run all validations
        validator.validate_ssl_configuration(domains)
        validator.validate_security_headers(base_url)
        validator.validate_rate_limiting(api_url)
        validator.validate_backup_system()
        validator.validate_firewall_configuration()
        validator.validate_cdn_configuration(base_url)
        
        # Generate and save report
        report = validator.generate_report()
        validator.save_report()
        
        # Exit with appropriate code
        if validator.results["overall_score"] >= 80:
            print("\n✅ Security infrastructure validation PASSED")
            sys.exit(0)
        else:
            print("\n❌ Security infrastructure validation FAILED")
            print("Please address the failed checks before proceeding to production")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()