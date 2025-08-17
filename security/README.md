# Application Security Framework

Enterprise-grade security hardening implementation for ScrollIntel, providing comprehensive protection through multiple security layers.

## Overview

This security framework implements:

- **SAST/DAST Security Scanning** - Automated security scanning in CI/CD pipeline
- **Runtime Application Self-Protection (RASP)** - Real-time threat detection and blocking
- **Secure API Gateway** - Authentication, authorization, and rate limiting
- **Secrets Management** - Secure storage and rotation of sensitive data
- **Input Validation & Sanitization** - Comprehensive input security framework

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Framework                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ SAST/DAST   │  │ Input       │  │ RASP Protection     │  │
│  │ Scanner     │  │ Validation  │  │ - SQL Injection     │  │
│  │ - Bandit    │  │ - XSS       │  │ - XSS Detection     │  │
│  │ - Semgrep   │  │ - Sanitize  │  │ - Command Injection │  │
│  │ - Nuclei    │  │ - Validate  │  │ - Path Traversal    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────────────────────────────┐  │
│  │ API Gateway │  │ Secrets Manager                         │  │
│  │ - Auth      │  │ - HashiCorp Vault                       │  │
│  │ - Rate Limit│  │ - AWS Secrets Manager                   │  │
│  │ - CORS      │  │ - Local Encrypted Storage               │  │
│  │ - Headers   │  │ - Automatic Rotation                    │  │
│  └─────────────┘  └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Clone and navigate to security directory
cd security

# Install dependencies
pip install -r requirements.txt

# Deploy security framework
python deploy_application_security.py
```

### 2. Configuration

Set required environment variables:

```bash
export JWT_SECRET="your-jwt-secret-key"
export SECRETS_MASTER_KEY="your-master-encryption-key"
export REDIS_URL="redis://localhost:6379"  # Optional for rate limiting
```

### 3. Basic Usage

```python
from security.application.rasp_protection import RASPMiddleware
from security.application.api_gateway import SecureAPIGateway
from security.application.input_validation import InputValidator

# Initialize components
rasp = RASPMiddleware(config)
gateway = SecureAPIGateway(config)
validator = InputValidator()

# Process request through security pipeline
async def secure_request_handler(request_data):
    # 1. Input validation
    validation_result = validator.validate_data(request_data, rules)
    
    # 2. RASP protection
    rasp_result = await rasp.process_request(request_data)
    
    # 3. API gateway
    gateway_result = await gateway.process_request(request_data)
    
    return gateway_result['allowed']
```

## Components

### SAST/DAST Scanner

Automated security scanning integrated into CI/CD pipeline.

**Features:**
- Static Application Security Testing (SAST)
- Dynamic Application Security Testing (DAST)
- Multiple tool integration (Bandit, Semgrep, Nuclei)
- Automated security gates
- Configurable severity thresholds

**Configuration:** `security/config/scanner_config.yaml`

```yaml
sast:
  enabled: true
  tools: [bandit, semgrep]
  severity_threshold: "HIGH"

dast:
  enabled: true
  tools: [nuclei]
  target_url: "http://localhost:8000"
```

### RASP Protection

Runtime Application Self-Protection with real-time threat detection.

**Features:**
- SQL injection detection
- Cross-site scripting (XSS) prevention
- Command injection blocking
- Path traversal protection
- Behavioral analysis
- Automatic threat blocking

**Usage:**
```python
from security.application.rasp_protection import RASPMiddleware

rasp = RASPMiddleware({
    'enabled': True,
    'block_threshold': 'HIGH',
    'auto_block_enabled': True
})

result = await rasp.process_request(request_data)
if not result['allowed']:
    return 403, "Request blocked by security policy"
```

### API Gateway

Secure API gateway with authentication, authorization, and rate limiting.

**Features:**
- Multiple authentication methods (JWT, API Key, Basic Auth)
- Role-based access control (RBAC)
- Advanced rate limiting
- CORS protection
- Security headers
- Request/response validation

**Configuration:** `security/config/gateway_config.json`

```json
{
  "authentication": {
    "auth_methods": ["API_KEY", "JWT_TOKEN"],
    "jwt_secret": "${JWT_SECRET}"
  },
  "authorization": {
    "permissions_map": {
      "GET:/api/users": ["users:read"],
      "POST:/api/users": ["users:write"]
    }
  },
  "rate_limiting": {
    "rules": [
      {
        "type": "REQUESTS_PER_MINUTE",
        "limit": 60,
        "applies_to": "authenticated"
      }
    ]
  }
}
```

### Secrets Manager

Secure secrets management with multiple backend support.

**Features:**
- HashiCorp Vault integration
- AWS Secrets Manager support
- Local encrypted storage
- Automatic secret rotation
- Audit logging
- Multiple secret types

**Usage:**
```python
from security.application.secrets_manager import SecretsManager, SecretType

manager = SecretsManager(config)

# Store secret
await manager.store_secret(
    'database_password',
    'super_secret_password',
    SecretType.DATABASE_PASSWORD
)

# Retrieve secret
secret = await manager.retrieve_secret('database_password')
print(secret.value)
```

### Input Validation

Comprehensive input validation and sanitization framework.

**Features:**
- Multiple input types (string, email, URL, JSON, HTML, etc.)
- Security pattern detection
- Automatic sanitization
- Custom validation rules
- XSS prevention
- SQL injection detection

**Usage:**
```python
from security.application.input_validation import InputValidator, ValidationRule, InputType

validator = InputValidator(strict_mode=True)

rules = {
    'email': ValidationRule(InputType.EMAIL, required=True),
    'age': ValidationRule(InputType.INTEGER, min_value=0, max_value=150),
    'bio': ValidationRule(InputType.HTML, sanitize=True)
}

results = validator.validate_data(user_data, rules)
if validator.is_valid(results):
    clean_data = validator.get_sanitized_data(results)
```

## Security Policies

### Default Security Rules

1. **Authentication Required**: All API endpoints require authentication except health checks
2. **Rate Limiting**: 60 requests/minute for authenticated users, 10/minute for anonymous
3. **Input Validation**: All user inputs are validated and sanitized
4. **SQL Injection Protection**: Automatic detection and blocking
5. **XSS Prevention**: HTML sanitization and script blocking
6. **Secrets Encryption**: All secrets encrypted at rest with AES-256
7. **Audit Logging**: All security events logged and monitored

### Threat Detection

The RASP system detects and blocks:

- SQL injection attempts
- Cross-site scripting (XSS)
- Command injection
- Path traversal attacks
- LDAP injection
- Suspicious behavioral patterns
- Rate limit violations

### Security Headers

Automatically applied security headers:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
Referrer-Policy: strict-origin-when-cross-origin
```

## Monitoring and Alerting

### Security Metrics

- Request processing times
- Threat detection rates
- Authentication success/failure rates
- Rate limiting violations
- Secret access patterns

### Logging

All security events are logged with:

- Timestamp
- Source IP
- User ID (if authenticated)
- Event type
- Threat level
- Action taken

### Alerts

Automatic alerts for:

- Critical security threats
- Authentication failures
- Rate limit violations
- Secret access anomalies
- System errors

## Testing

Run the complete test suite:

```bash
# Run all security tests
python -m pytest tests/test_application_security_framework.py -v

# Run specific component tests
python -m pytest tests/test_application_security_framework.py::TestRASPProtection -v

# Run with coverage
python -m pytest tests/test_application_security_framework.py --cov=security --cov-report=html
```

## Deployment

### Development Environment

```bash
# Deploy with default settings
python security/deploy_application_security.py

# Check deployment status
cat security/reports/deployment_report.json
```

### Production Environment

1. **Set Environment Variables:**
```bash
export JWT_SECRET="$(openssl rand -base64 32)"
export SECRETS_MASTER_KEY="$(openssl rand -base64 32)"
export REDIS_URL="redis://your-redis-server:6379"
```

2. **Configure External Services:**
```bash
# HashiCorp Vault
export VAULT_URL="https://your-vault-server:8200"
export VAULT_TOKEN="your-vault-token"

# AWS Secrets Manager
export AWS_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

3. **Deploy:**
```bash
python security/deploy_application_security.py --environment production
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install security tools
RUN pip install bandit semgrep

# Copy security framework
COPY security/ /app/security/
WORKDIR /app

# Install dependencies
RUN pip install -r security/requirements.txt

# Deploy security framework
RUN python security/deploy_application_security.py

EXPOSE 8000
CMD ["python", "app.py"]
```

## Configuration Reference

### Scanner Configuration (`scanner_config.yaml`)

```yaml
sast:
  enabled: true
  tools: [bandit, semgrep, sonarqube]
  severity_threshold: "HIGH"
  
dast:
  enabled: true
  tools: [nuclei, zap]
  target_url: "http://localhost:8000"

security_gate:
  fail_on_critical: true
  fail_on_high: true
  max_findings:
    critical: 0
    high: 0
```

### Gateway Configuration (`gateway_config.json`)

```json
{
  "authentication": {
    "jwt_secret": "${JWT_SECRET}",
    "auth_methods": ["API_KEY", "JWT_TOKEN"],
    "jwt_expiration": 3600
  },
  "authorization": {
    "permissions_map": {
      "GET:/api/users": ["users:read"]
    },
    "role_permissions": {
      "admin": ["*"],
      "user": ["users:read"]
    }
  },
  "rate_limiting": {
    "enabled": true,
    "rules": [
      {
        "type": "REQUESTS_PER_MINUTE",
        "limit": 60,
        "applies_to": "authenticated"
      }
    ]
  }
}
```

### Secrets Configuration (`secrets_config.json`)

```json
{
  "default_provider": "local_encrypted",
  "providers": {
    "local_encrypted": {
      "enabled": true,
      "master_key": "${SECRETS_MASTER_KEY}"
    },
    "vault": {
      "enabled": false,
      "url": "${VAULT_URL}",
      "token": "${VAULT_TOKEN}"
    }
  },
  "rotation": {
    "enabled": true,
    "schedule": "0 2 * * *"
  }
}
```

## Troubleshooting

### Common Issues

1. **JWT Secret Not Set**
```bash
export JWT_SECRET="$(openssl rand -base64 32)"
```

2. **Secrets Master Key Missing**
```bash
export SECRETS_MASTER_KEY="$(openssl rand -base64 32)"
```

3. **Redis Connection Failed**
```bash
# Install and start Redis
sudo apt-get install redis-server
sudo systemctl start redis-server
```

4. **Security Tools Not Found**
```bash
pip install bandit semgrep
# For Nuclei, see installation guide
```

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('security').setLevel(logging.DEBUG)
```

### Performance Issues

1. **Disable strict mode for development:**
```python
validator = InputValidator(strict_mode=False)
```

2. **Reduce rate limiting:**
```json
{
  "rate_limiting": {
    "rules": [
      {
        "type": "REQUESTS_PER_MINUTE",
        "limit": 1000
      }
    ]
  }
}
```

## Security Best Practices

1. **Regular Updates**: Keep security tools and dependencies updated
2. **Secret Rotation**: Implement automatic secret rotation
3. **Monitoring**: Set up comprehensive security monitoring
4. **Testing**: Run security tests in CI/CD pipeline
5. **Incident Response**: Have incident response procedures ready
6. **Backup**: Regular backup of security configurations and logs
7. **Access Control**: Implement principle of least privilege
8. **Audit**: Regular security audits and penetration testing

## Contributing

1. Follow security coding standards
2. Add tests for new security features
3. Update documentation
4. Security review required for all changes
5. Test against OWASP Top 10 vulnerabilities

## License

This security framework is proprietary to ScrollIntel and contains enterprise security implementations.

## Support

For security issues or questions:
- Email: security@scrollintel.com
- Internal: #security-team channel
- Emergency: security-oncall@scrollintel.com