"""
Application Security Framework Demo
Demonstrates all security components without external dependencies
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Add the security modules to path
sys.path.append(str(Path(__file__).parent))

print("🔒 Application Security Framework Demo")
print("=" * 50)

async def demo_input_validation():
    """Demo input validation framework"""
    print("\n📝 Input Validation Framework Demo")
    print("-" * 40)
    
    try:
        from application.input_validation_simple import InputValidator, ValidationRule, InputType, CommonValidationRules
        
        # Create validator
        validator = InputValidator({
            'strict_mode': True,
            'auto_sanitize': True
        })
        
        # Test data
        test_data = {
            'username': 'testuser123',
            'email': 'test@example.com',
            'age': '25',
            'bio': '<p>Hello world!</p><script>alert("xss")</script>',
            'malicious_query': "'; DROP TABLE users; --"
        }
        
        # Validation rules
        rules = {
            'username': ValidationRule(
                name='username',
                input_type=InputType.TEXT,
                required=True,
                min_length=3,
                max_length=50,
                pattern=r'^[a-zA-Z0-9_-]+$'
            ),
            'email': ValidationRule(
                name='email',
                input_type=InputType.TEXT,  # Simplified for demo
                required=True,
                pattern=r'^[^@]+@[^@]+\.[^@]+$'
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
            ),
            'malicious_query': ValidationRule(
                name='malicious_query',
                input_type=InputType.TEXT,
                required=False
            )
        }
        
        # Validate input
        result = validator.validate_input(test_data, rules)
        
        print(f"✅ Validation Result: {result.result.value}")
        print(f"   Valid: {result.is_valid}")
        print(f"   Errors: {len(result.errors)}")
        print(f"   Warnings: {len(result.warnings)}")
        print(f"   Blocked fields: {result.blocked_fields}")
        
        if result.errors:
            print("   Error details:")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"     - {error.field}: {error.message}")
        
        if result.sanitized_data:
            print("   Sanitized data sample:")
            for key, value in list(result.sanitized_data.items())[:3]:
                print(f"     - {key}: {str(value)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Input validation demo failed: {e}")
        return False

async def demo_rasp_protection():
    """Demo RASP protection system"""
    print("\n🛡️  RASP Protection Demo")
    print("-" * 40)
    
    try:
        from application.rasp_protection import RASPEngine, ThreatType
        
        # Create RASP engine
        rasp = RASPEngine({
            'enabled': True,
            'learning_mode': False,
            'max_requests_per_5min': 100
        })
        
        # Test legitimate request
        legitimate_request = {
            'source_ip': '127.0.0.1',
            'user_id': 'test_user',
            'path': '/api/data',
            'method': 'GET',
            'headers': {'User-Agent': 'Mozilla/5.0'},
            'body': '{"action": "get_data"}',
            'query_params': {'limit': '10'}
        }
        
        threat = await rasp.analyze_request(legitimate_request)
        print(f"✅ Legitimate request: {'Blocked' if threat and threat.blocked else 'Allowed'}")
        
        # Test malicious requests
        malicious_requests = [
            {
                'name': 'SQL Injection',
                'data': {
                    'source_ip': '192.168.1.100',
                    'user_id': 'attacker',
                    'path': '/api/query',
                    'method': 'POST',
                    'headers': {'User-Agent': 'AttackBot/1.0'},
                    'body': "'; DROP TABLE users; --",
                    'query_params': {}
                }
            },
            {
                'name': 'XSS Attack',
                'data': {
                    'source_ip': '192.168.1.101',
                    'user_id': 'attacker2',
                    'path': '/api/comment',
                    'method': 'POST',
                    'headers': {'User-Agent': 'AttackBot/2.0'},
                    'body': '<script>alert("xss")</script>',
                    'query_params': {}
                }
            },
            {
                'name': 'Command Injection',
                'data': {
                    'source_ip': '192.168.1.102',
                    'user_id': 'attacker3',
                    'path': '/api/system',
                    'method': 'POST',
                    'headers': {'User-Agent': 'AttackBot/3.0'},
                    'body': '; cat /etc/passwd',
                    'query_params': {}
                }
            }
        ]
        
        for attack in malicious_requests:
            threat = await rasp.analyze_request(attack['data'])
            if threat:
                print(f"🚨 {attack['name']}: BLOCKED")
                print(f"   Threat Type: {threat.threat_type.value}")
                print(f"   Severity: {threat.severity.value}")
                print(f"   Confidence: {threat.confidence_score:.2f}")
                print(f"   Action: {threat.response_action.value}")
            else:
                print(f"⚠️  {attack['name']}: Not detected")
        
        # Show threat summary
        summary = rasp.get_threat_summary()
        print(f"\n📊 Threat Summary:")
        print(f"   Total threats (24h): {summary['total_threats_24h']}")
        print(f"   Active blocks: {summary['active_blocks']}")
        
        return True
        
    except Exception as e:
        print(f"❌ RASP protection demo failed: {e}")
        return False

async def demo_api_gateway():
    """Demo API gateway functionality"""
    print("\n🚪 API Gateway Demo")
    print("-" * 40)
    
    try:
        from application.api_gateway import SecureAPIGateway
        
        # Create API gateway
        gateway = SecureAPIGateway({
            'jwt_secret': 'demo-secret-key',
            'security_headers': {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY'
            }
        })
        
        # Create API key
        api_key = gateway.add_api_key(
            name="demo_key",
            permissions=["data.read", "data.write"]
        )
        
        print(f"✅ API Key created: {api_key[:8]}...")
        
        # Test endpoint configuration
        endpoint = gateway._find_endpoint_config('/api/v1/data/test', 'GET')
        if endpoint:
            print(f"✅ Endpoint found: {endpoint.path}")
            print(f"   Auth required: {endpoint.auth_required}")
            print(f"   Rate limits: {len(endpoint.rate_limits)}")
        
        # Test security headers
        headers = gateway.get_security_headers()
        print(f"✅ Security headers: {len(headers)} configured")
        for header, value in list(headers.items())[:3]:
            print(f"   {header}: {value}")
        
        # Test metrics
        metrics = gateway.get_metrics()
        print(f"✅ Metrics available: {len(metrics)} categories")
        
        return True
        
    except Exception as e:
        print(f"❌ API gateway demo failed: {e}")
        return False

async def demo_secrets_manager():
    """Demo secrets management system"""
    print("\n🔐 Secrets Manager Demo")
    print("-" * 40)
    
    try:
        from application.secrets_manager import SecretsManager, SecretType
        
        # Create secrets manager
        secrets = SecretsManager({
            'cache_ttl': 300,
            'master_password': 'demo-password',
            'salt': b'demo-salt-16byte',
            'local_storage': {
                'storage_path': '/tmp/demo_secrets.json',
                'encryption_key': 'demo-encryption-key'
            }
        })
        
        # Store different types of secrets
        secrets_to_store = [
            {
                'name': 'api_key_demo',
                'value': 'demo-api-key-12345',
                'type': SecretType.API_KEYS,
                'tags': {'environment': 'demo', 'service': 'api'}
            },
            {
                'name': 'db_credentials_demo',
                'value': {
                    'username': 'dbuser',
                    'password': 'dbpass123',
                    'host': 'localhost',
                    'port': 5432
                },
                'type': SecretType.DATABASE_CREDENTIALS,
                'tags': {'environment': 'demo', 'service': 'database'}
            },
            {
                'name': 'encryption_key_demo',
                'value': 'demo-encryption-key-abcdef',
                'type': SecretType.ENCRYPTION_KEYS,
                'tags': {'environment': 'demo', 'service': 'encryption'}
            }
        ]
        
        stored_ids = []
        for secret_info in secrets_to_store:
            secret_id = await secrets.store_secret(
                name=secret_info['name'],
                value=secret_info['value'],
                secret_type=secret_info['type'],
                tags=secret_info['tags']
            )
            stored_ids.append(secret_id)
            print(f"✅ Stored {secret_info['type'].value}: {secret_info['name']}")
        
        # Retrieve secrets
        for i, secret_id in enumerate(stored_ids):
            secret = await secrets.get_secret(secret_id)
            if secret:
                print(f"✅ Retrieved: {secret.metadata.name}")
                print(f"   Type: {secret.metadata.secret_type.value}")
                print(f"   Tags: {secret.metadata.tags}")
                print(f"   Access count: {secret.metadata.access_count}")
        
        # List secrets
        all_secrets = await secrets.list_secrets()
        print(f"✅ Total secrets stored: {len(all_secrets)}")
        
        # Test secret rotation
        if stored_ids:
            rotation_success = await secrets.rotate_secret(stored_ids[0])
            print(f"✅ Secret rotation: {'Success' if rotation_success else 'Failed'}")
        
        # Clean up demo secrets
        for secret_id in stored_ids:
            await secrets.delete_secret(secret_id)
        
        return True
        
    except Exception as e:
        print(f"❌ Secrets manager demo failed: {e}")
        return False

async def demo_security_scanner():
    """Demo security scanner functionality"""
    print("\n🔍 Security Scanner Demo")
    print("-" * 40)
    
    try:
        from application.sast_dast_scanner import SecurityScanner, ScanType
        
        # Create security scanner
        scanner = SecurityScanner({
            'security_gates': {
                'max_critical': 0,
                'max_high': 5,
                'max_medium': 20
            },
            'scan_tools': {
                'semgrep': {'enabled': True},
                'bandit': {'enabled': True}
            }
        })
        
        # Create test code with security issues
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "vulnerable_code.py")
            with open(test_file, "w") as f:
                f.write("""
import subprocess
import os

def unsafe_function(user_input):
    # Command injection vulnerability
    command = f"ls {user_input}"
    subprocess.run(command, shell=True)

def sql_query(user_id):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    return query

def file_access(filename):
    # Path traversal vulnerability
    with open(f"/data/{filename}", "r") as f:
        return f.read()

# Hardcoded secret
API_KEY = "secret-key-12345"
""")
            
            print(f"✅ Created test file with vulnerabilities")
            
            # Mock the scanner tools since they may not be installed
            async def mock_semgrep(target_path):
                return [
                    {
                        'id': 'command-injection',
                        'title': 'Command Injection',
                        'severity': 'HIGH',
                        'file_path': test_file,
                        'line_number': 7
                    }
                ]
            
            async def mock_bandit(target_path):
                return [
                    {
                        'id': 'hardcoded-password',
                        'title': 'Hardcoded Secret',
                        'severity': 'MEDIUM',
                        'file_path': test_file,
                        'line_number': 19
                    }
                ]
            
            # Replace scanner methods with mocks
            scanner._run_semgrep = mock_semgrep
            scanner._run_bandit = mock_bandit
            
            # Run scan
            results = await scanner.run_comprehensive_scan(temp_dir, [ScanType.SAST])
            
            if ScanType.SAST in results:
                result = results[ScanType.SAST]
                print(f"✅ SAST scan completed")
                print(f"   Total findings: {result.total_vulnerabilities}")
                print(f"   Critical: {result.critical_count}")
                print(f"   High: {result.high_count}")
                print(f"   Medium: {result.medium_count}")
                print(f"   Security gate passed: {result.passed_security_gate}")
                
                # Generate report
                report = await scanner.generate_security_report(results)
                print(f"✅ Security report generated")
                print(f"   Overall gate passed: {report['overall_security_gate_passed']}")
                print(f"   Recommendations: {len(report['recommendations'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Security scanner demo failed: {e}")
        return False

async def demo_integration():
    """Demo integration between all components"""
    print("\n🔗 Integration Demo")
    print("-" * 40)
    
    try:
        from application.input_validation_simple import InputValidator, ValidationRule, InputType
        from application.rasp_protection import RASPEngine
        from application.api_gateway import SecureAPIGateway
        from application.secrets_manager import SecretsManager, SecretType
        
        # Initialize all components
        validator = InputValidator({'strict_mode': True, 'auto_sanitize': True})
        rasp = RASPEngine({'enabled': True, 'learning_mode': False})
        gateway = SecureAPIGateway({'jwt_secret': 'integration-test'})
        secrets = SecretsManager({
            'master_password': 'integration-password',
            'salt': b'integration-salt',
            'local_storage': {'storage_path': '/tmp/integration_secrets.json'}
        })
        
        print("✅ All components initialized")
        
        # Simulate end-to-end request processing
        print("\n🔄 Processing legitimate request...")
        
        # 1. Input validation
        input_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'message': 'Hello, world!'
        }
        
        rules = {
            'username': ValidationRule('username', InputType.TEXT, required=True, min_length=3),
            'email': ValidationRule('email', InputType.TEXT, required=True),
            'message': ValidationRule('message', InputType.TEXT, required=True, max_length=1000)
        }
        
        validation_result = validator.validate_input(input_data, rules)
        print(f"   Input validation: {'✅ Passed' if validation_result.is_valid else '❌ Failed'}")
        
        # 2. RASP analysis
        request_data = {
            'source_ip': '127.0.0.1',
            'user_id': 'test_user',
            'path': '/api/message',
            'method': 'POST',
            'headers': {'User-Agent': 'TestClient/1.0'},
            'body': json.dumps(validation_result.sanitized_data if validation_result.is_valid else input_data),
            'query_params': {}
        }
        
        threat = await rasp.analyze_request(request_data)
        print(f"   RASP analysis: {'✅ Clean' if not threat or not threat.blocked else '🚨 Threat detected'}")
        
        # 3. Store API configuration in secrets
        config_secret_id = await secrets.store_secret(
            name="api_config",
            value={"rate_limit": 1000, "timeout": 30},
            secret_type=SecretType.CONFIGURATION,
            tags={'component': 'api_gateway'}
        )
        
        config_secret = await secrets.get_secret(config_secret_id)
        print(f"   Secrets management: {'✅ Working' if config_secret else '❌ Failed'}")
        
        # 4. API Gateway processing
        api_key = gateway.add_api_key("integration_test", ["data.read"])
        print(f"   API Gateway: ✅ Key created ({api_key[:8]}...)")
        
        print("\n🔄 Processing malicious request...")
        
        # Test with malicious input
        malicious_data = {
            'username': '<script>alert("xss")</script>',
            'email': 'attacker@evil.com',
            'message': "'; DROP TABLE messages; --"
        }
        
        # Input validation should catch this
        malicious_validation = validator.validate_input(malicious_data, rules)
        print(f"   Input validation: {'🚨 Blocked' if not malicious_validation.is_valid or malicious_validation.blocked_fields else '⚠️ Passed'}")
        
        # RASP should also catch SQL injection
        malicious_request = {
            'source_ip': '192.168.1.100',
            'user_id': 'attacker',
            'path': '/api/message',
            'method': 'POST',
            'headers': {'User-Agent': 'AttackBot/1.0'},
            'body': malicious_data['message'],
            'query_params': {}
        }
        
        malicious_threat = await rasp.analyze_request(malicious_request)
        print(f"   RASP analysis: {'🚨 Threat blocked' if malicious_threat and malicious_threat.blocked else '⚠️ Not detected'}")
        
        # Clean up
        await secrets.delete_secret(config_secret_id)
        
        print("\n✅ Integration demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration demo failed: {e}")
        return False

async def main():
    """Main demo function"""
    print("Starting Application Security Framework Demo...\n")
    
    demos = [
        ("Input Validation", demo_input_validation),
        ("RASP Protection", demo_rasp_protection),
        ("API Gateway", demo_api_gateway),
        ("Secrets Manager", demo_secrets_manager),
        ("Security Scanner", demo_security_scanner),
        ("Integration", demo_integration)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            result = await demo_func()
            results[demo_name] = result
        except Exception as e:
            print(f"❌ {demo_name} demo failed with exception: {e}")
            results[demo_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 DEMO SUMMARY")
    print("=" * 50)
    
    successful = sum(results.values())
    total = len(results)
    
    for demo_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{demo_name:20} {status}")
    
    print(f"\nOverall: {successful}/{total} demos passed")
    
    if successful == total:
        print("\n🎉 All Application Security Framework components working!")
        print("\n📚 Next Steps:")
        print("1. Install external dependencies (Redis, Vault, etc.) for full functionality")
        print("2. Configure CI/CD pipeline integration")
        print("3. Set up monitoring and alerting")
        print("4. Customize validation rules for your use case")
        print("5. Configure external secret providers")
    else:
        print(f"\n⚠️  {total - successful} demo(s) failed - check error messages above")
    
    return successful == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)