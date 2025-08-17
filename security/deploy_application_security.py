"""
Application Security Framework Deployment Script
Deploys and configures all application security components
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import yaml

# Add the security modules to path
sys.path.append(str(Path(__file__).parent))

from application.sast_dast_scanner import SecurityScanner, CICDSecurityIntegration, ScanType
from application.rasp_protection import RASPEngine, RASPMiddleware
from application.api_gateway import SecureAPIGateway, APIGatewayMiddleware, create_secure_api_gateway
from application.secrets_manager import SecretsManager, SecretType, SecretProvider
from application.input_validation import InputValidator, CommonValidationRules, create_input_validator

class ApplicationSecurityDeployment:
    """Deploy and configure application security framework"""
    
    def __init__(self, config_path: str = "security/config/application_security_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.security_scanner = None
        self.rasp_engine = None
        self.api_gateway = None
        self.secrets_manager = None
        self.input_validator = None
        
        # Deployment status
        self.deployment_status = {
            'sast_dast_scanner': False,
            'rasp_protection': False,
            'api_gateway': False,
            'secrets_manager': False,
            'input_validation': False
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return self._create_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        default_config = {
            'security_scanner': {
                'enabled': True,
                'security_gates': {
                    'max_critical': 0,
                    'max_high': 5,
                    'max_medium': 20
                },
                'scan_tools': {
                    'semgrep': {'enabled': True},
                    'bandit': {'enabled': True},
                    'safety': {'enabled': True},
                    'zap': {'enabled': True},
                    'trivy': {'enabled': True}
                },
                'results_storage': '/tmp/security_scans'
            },
            
            'rasp_protection': {
                'enabled': True,
                'learning_mode': False,
                'anomaly_threshold': 0.8,
                'max_requests_per_5min': 100,
                'alert_webhook': None
            },
            
            'api_gateway': {
                'enabled': True,
                'jwt_secret': 'your-jwt-secret-key-change-this',
                'jwt_algorithm': 'HS256',
                'redis_url': 'redis://localhost:6379',
                'security_headers': {
                    'X-Content-Type-Options': 'nosniff',
                    'X-Frame-Options': 'DENY',
                    'X-XSS-Protection': '1; mode=block',
                    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
                    'Content-Security-Policy': "default-src 'self'",
                    'Referrer-Policy': 'strict-origin-when-cross-origin'
                },
                'admin_ip_whitelist': ['127.0.0.1', '::1']
            },
            
            'secrets_manager': {
                'enabled': True,
                'cache_ttl': 300,
                'master_password': 'change-this-master-password',
                'salt': b'change-this-salt-16b',
                'vault': {
                    'enabled': False,
                    'url': 'http://localhost:8200',
                    'token': None,
                    'mount_point': 'secret'
                },
                'aws_secrets_manager': {
                    'enabled': False,
                    'region': 'us-east-1',
                    'access_key_id': None,
                    'secret_access_key': None
                },
                'local_storage': {
                    'storage_path': '/tmp/encrypted_secrets.json',
                    'encryption_key': 'local-encryption-key'
                }
            },
            
            'input_validation': {
                'enabled': True,
                'strict_mode': True,
                'auto_sanitize': True,
                'allowed_html_tags': [
                    'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
                ],
                'allowed_html_attributes': {
                    'a': ['href', 'title'],
                    '*': ['class']
                }
            }
        }
        
        # Save default config
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    async def deploy_all(self) -> bool:
        """Deploy all application security components"""
        print("🔒 Starting Application Security Framework Deployment...")
        
        success = True
        
        # Deploy components in order
        components = [
            ('Input Validation', self.deploy_input_validation),
            ('Secrets Manager', self.deploy_secrets_manager),
            ('SAST/DAST Scanner', self.deploy_security_scanner),
            ('RASP Protection', self.deploy_rasp_protection),
            ('API Gateway', self.deploy_api_gateway)
        ]
        
        for component_name, deploy_func in components:
            try:
                print(f"\n📦 Deploying {component_name}...")
                result = await deploy_func()
                if result:
                    print(f"✅ {component_name} deployed successfully")
                else:
                    print(f"❌ {component_name} deployment failed")
                    success = False
            except Exception as e:
                print(f"❌ {component_name} deployment error: {e}")
                success = False
        
        # Run integration tests
        if success:
            print("\n🧪 Running integration tests...")
            test_result = await self.run_integration_tests()
            if test_result:
                print("✅ Integration tests passed")
            else:
                print("❌ Integration tests failed")
                success = False
        
        # Generate deployment report
        await self.generate_deployment_report()
        
        if success:
            print("\n🎉 Application Security Framework deployed successfully!")
            print("📋 Check deployment report: security/deployment_report.json")
        else:
            print("\n⚠️  Application Security Framework deployment completed with errors")
        
        return success
    
    async def deploy_input_validation(self) -> bool:
        """Deploy input validation framework"""
        try:
            config = self.config.get('input_validation', {})
            if not config.get('enabled', True):
                print("Input validation disabled in config")
                return True
            
            self.input_validator = create_input_validator(config)
            
            # Test validation with sample data
            test_data = {
                'email': 'test@example.com',
                'username': 'testuser',
                'age': '25'
            }
            
            rules = {
                'email': CommonValidationRules.user_registration()['email'],
                'username': CommonValidationRules.user_registration()['username'],
                'age': CommonValidationRules.api_request()['format']  # Will fail, testing error handling
            }
            
            result = self.input_validator.validate_input(test_data, rules)
            print(f"Input validation test result: {result.result.value}")
            
            self.deployment_status['input_validation'] = True
            return True
            
        except Exception as e:
            print(f"Input validation deployment error: {e}")
            return False
    
    async def deploy_secrets_manager(self) -> bool:
        """Deploy secrets management system"""
        try:
            config = self.config.get('secrets_manager', {})
            if not config.get('enabled', True):
                print("Secrets manager disabled in config")
                return True
            
            self.secrets_manager = SecretsManager(config)
            
            # Test storing and retrieving a secret
            test_secret_id = await self.secrets_manager.store_secret(
                name="test_api_key",
                value="test-api-key-value-12345",
                secret_type=SecretType.API_KEYS,
                tags={'environment': 'test', 'component': 'deployment'}
            )
            
            retrieved_secret = await self.secrets_manager.get_secret(test_secret_id)
            if retrieved_secret and retrieved_secret.value == "test-api-key-value-12345":
                print("Secrets manager test successful")
                
                # Clean up test secret
                await self.secrets_manager.delete_secret(test_secret_id)
                
                self.deployment_status['secrets_manager'] = True
                return True
            else:
                print("Secrets manager test failed")
                return False
                
        except Exception as e:
            print(f"Secrets manager deployment error: {e}")
            return False
    
    async def deploy_security_scanner(self) -> bool:
        """Deploy SAST/DAST security scanner"""
        try:
            config = self.config.get('security_scanner', {})
            if not config.get('enabled', True):
                print("Security scanner disabled in config")
                return True
            
            self.security_scanner = SecurityScanner(config)
            
            # Test scanner with sample code
            test_code_path = "/tmp/test_security_scan"
            os.makedirs(test_code_path, exist_ok=True)
            
            # Create test file with potential security issue
            with open(f"{test_code_path}/test.py", "w") as f:
                f.write("""
import os
import subprocess

def unsafe_function(user_input):
    # This should be detected by security scanners
    command = f"ls {user_input}"
    subprocess.run(command, shell=True)
    
def sql_query(user_id):
    # This should also be detected
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    return query
""")
            
            # Run SAST scan
            scan_results = await self.security_scanner.run_comprehensive_scan(
                test_code_path, [ScanType.SAST]
            )
            
            if ScanType.SAST in scan_results:
                sast_result = scan_results[ScanType.SAST]
                print(f"SAST scan completed: {sast_result.total_vulnerabilities} findings")
                
                # Generate report
                report = await self.security_scanner.generate_security_report(scan_results)
                print(f"Security gate passed: {report['overall_security_gate_passed']}")
                
                self.deployment_status['sast_dast_scanner'] = True
                return True
            else:
                print("SAST scan failed")
                return False
                
        except Exception as e:
            print(f"Security scanner deployment error: {e}")
            return False
    
    async def deploy_rasp_protection(self) -> bool:
        """Deploy RASP protection"""
        try:
            config = self.config.get('rasp_protection', {})
            if not config.get('enabled', True):
                print("RASP protection disabled in config")
                return True
            
            self.rasp_engine = RASPEngine(config)
            
            # Test RASP with sample malicious request
            test_request = {
                'source_ip': '192.168.1.100',
                'user_id': 'test_user',
                'path': '/api/test',
                'method': 'POST',
                'headers': {'User-Agent': 'TestAgent/1.0'},
                'body': "'; DROP TABLE users; --",  # SQL injection attempt
                'query_params': {}
            }
            
            threat_event = await self.rasp_engine.analyze_request(test_request)
            
            if threat_event:
                print(f"RASP detected threat: {threat_event.threat_type.value}")
                print(f"Confidence: {threat_event.confidence_score}")
                print(f"Action: {threat_event.response_action.value}")
                
                self.deployment_status['rasp_protection'] = True
                return True
            else:
                print("RASP test failed - no threat detected")
                return False
                
        except Exception as e:
            print(f"RASP protection deployment error: {e}")
            return False
    
    async def deploy_api_gateway(self) -> bool:
        """Deploy secure API gateway"""
        try:
            config = self.config.get('api_gateway', {})
            if not config.get('enabled', True):
                print("API gateway disabled in config")
                return True
            
            self.api_gateway = create_secure_api_gateway(config)
            
            # Add test API key
            test_api_key = self.api_gateway.add_api_key(
                name="test_key",
                permissions=["data.read", "data.write"]
            )
            
            print(f"Test API key created: {test_api_key[:8]}...")
            
            # Test rate limiting configuration
            metrics = self.api_gateway.get_metrics()
            print(f"API Gateway metrics: {metrics}")
            
            self.deployment_status['api_gateway'] = True
            return True
            
        except Exception as e:
            print(f"API gateway deployment error: {e}")
            return False
    
    async def run_integration_tests(self) -> bool:
        """Run integration tests for all components"""
        try:
            print("Running comprehensive integration tests...")
            
            # Test 1: End-to-end request processing
            if self.input_validator and self.rasp_engine and self.api_gateway:
                # Simulate a complete request flow
                test_data = {
                    'username': 'testuser',
                    'email': 'test@example.com',
                    'data': '{"key": "value"}'
                }
                
                # Input validation
                rules = CommonValidationRules.user_registration()
                validation_result = self.input_validator.validate_input(test_data, rules)
                
                if not validation_result.is_valid:
                    print("❌ Integration test failed: Input validation")
                    return False
                
                # RASP analysis
                request_data = {
                    'source_ip': '127.0.0.1',
                    'user_id': 'test_user',
                    'path': '/api/test',
                    'method': 'POST',
                    'headers': {'User-Agent': 'TestAgent/1.0'},
                    'body': json.dumps(validation_result.sanitized_data),
                    'query_params': {}
                }
                
                threat_event = await self.rasp_engine.analyze_request(request_data)
                
                if threat_event and threat_event.blocked:
                    print("❌ Integration test failed: RASP blocked legitimate request")
                    return False
                
                print("✅ End-to-end request processing test passed")
            
            # Test 2: Security threat detection
            if self.rasp_engine and self.input_validator:
                malicious_data = {
                    'username': '<script>alert("xss")</script>',
                    'email': 'test@example.com',
                    'query': "'; DROP TABLE users; --"
                }
                
                # Input validation should catch XSS
                rules = CommonValidationRules.user_registration()
                validation_result = self.input_validator.validate_input(malicious_data, rules)
                
                if validation_result.is_valid:
                    print("❌ Integration test failed: Input validation missed XSS")
                    return False
                
                # RASP should catch SQL injection
                request_data = {
                    'source_ip': '192.168.1.100',
                    'user_id': 'test_user',
                    'path': '/api/query',
                    'method': 'POST',
                    'headers': {'User-Agent': 'TestAgent/1.0'},
                    'body': malicious_data['query'],
                    'query_params': {}
                }
                
                threat_event = await self.rasp_engine.analyze_request(request_data)
                
                if not threat_event or not threat_event.blocked:
                    print("❌ Integration test failed: RASP missed SQL injection")
                    return False
                
                print("✅ Security threat detection test passed")
            
            # Test 3: Secrets management integration
            if self.secrets_manager:
                # Store a secret that would be used by other components
                secret_id = await self.secrets_manager.store_secret(
                    name="integration_test_secret",
                    value={"api_key": "test-key", "database_url": "postgresql://test"},
                    secret_type=SecretType.CONFIGURATION
                )
                
                # Retrieve and verify
                secret = await self.secrets_manager.get_secret(secret_id)
                if not secret or secret.value.get('api_key') != 'test-key':
                    print("❌ Integration test failed: Secrets management")
                    return False
                
                # Clean up
                await self.secrets_manager.delete_secret(secret_id)
                print("✅ Secrets management integration test passed")
            
            print("✅ All integration tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Integration tests failed: {e}")
            return False
    
    async def generate_deployment_report(self):
        """Generate deployment report"""
        report = {
            'deployment_timestamp': str(asyncio.get_event_loop().time()),
            'deployment_status': self.deployment_status,
            'configuration': self.config,
            'components': {
                'input_validation': {
                    'status': self.deployment_status['input_validation'],
                    'description': 'Input validation and sanitization framework'
                },
                'secrets_manager': {
                    'status': self.deployment_status['secrets_manager'],
                    'description': 'Secure secrets management system'
                },
                'sast_dast_scanner': {
                    'status': self.deployment_status['sast_dast_scanner'],
                    'description': 'Static and dynamic application security testing'
                },
                'rasp_protection': {
                    'status': self.deployment_status['rasp_protection'],
                    'description': 'Runtime application self-protection'
                },
                'api_gateway': {
                    'status': self.deployment_status['api_gateway'],
                    'description': 'Secure API gateway with rate limiting and authentication'
                }
            },
            'security_metrics': {
                'total_components': len(self.deployment_status),
                'successful_deployments': sum(self.deployment_status.values()),
                'failed_deployments': len(self.deployment_status) - sum(self.deployment_status.values())
            },
            'next_steps': [
                "Configure CI/CD pipeline integration for security scanning",
                "Set up monitoring and alerting for security events",
                "Configure external secret providers (Vault, AWS Secrets Manager)",
                "Implement custom validation rules for specific use cases",
                "Set up security dashboards and reporting"
            ]
        }
        
        # Save report
        report_path = "security/deployment_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Deployment report saved to: {report_path}")

async def main():
    """Main deployment function"""
    deployment = ApplicationSecurityDeployment()
    success = await deployment.deploy_all()
    
    if success:
        print("\n🔒 Application Security Framework is ready!")
        print("\n📚 Usage Examples:")
        print("1. Input Validation:")
        print("   from security.application.input_validation import create_input_validator")
        print("   validator = create_input_validator()")
        print("   result = validator.validate_input(data, rules)")
        
        print("\n2. RASP Protection:")
        print("   from security.application.rasp_protection import RASPEngine")
        print("   rasp = RASPEngine(config)")
        print("   threat = await rasp.analyze_request(request_data)")
        
        print("\n3. API Gateway:")
        print("   from security.application.api_gateway import create_secure_api_gateway")
        print("   gateway = create_secure_api_gateway(config)")
        print("   context = await gateway.process_request(request)")
        
        print("\n4. Secrets Manager:")
        print("   from security.application.secrets_manager import SecretsManager")
        print("   secrets = SecretsManager(config)")
        print("   secret_id = await secrets.store_secret(name, value, secret_type)")
        
        return 0
    else:
        print("\n❌ Application Security Framework deployment failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)