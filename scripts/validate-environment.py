#!/usr/bin/env python3
"""
Environment Configuration Validation Script
Validates that all required environment variables are properly set
"""

import os
import sys
from typing import Dict, List, Tuple

def validate_environment() -> Tuple[bool, List[str]]:
    """Validate environment configuration"""
    
    issues = []
    
    # Required variables for production
    required_vars = {
        'ENVIRONMENT': 'Environment type (production/staging/development)',
        'POSTGRES_PASSWORD': 'Database password',
        'JWT_SECRET_KEY': 'JWT secret key for authentication',
        'OPENAI_API_KEY': 'OpenAI API key for AI features'
    }
    
    # Optional but recommended variables
    recommended_vars = {
        'ANTHROPIC_API_KEY': 'Anthropic API key for Claude AI',
        'SMTP_SERVER': 'SMTP server for email notifications',
        'EMAIL_PASSWORD': 'Email password for SMTP authentication',
        'REDIS_HOST': 'Redis host for caching'
    }
    
    # Check required variables
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            issues.append(f"❌ MISSING: {var} - {description}")
        elif value in ['your-openai-api-key-here', 'your-secret-key-change-in-production']:
            issues.append(f"⚠️  PLACEHOLDER: {var} - {description}")
        else:
            print(f"✅ {var}: Configured")
    
    # Check recommended variables
    for var, description in recommended_vars.items():
        value = os.getenv(var)
        if not value:
            print(f"ℹ️  OPTIONAL: {var} - {description}")
        else:
            print(f"✅ {var}: Configured")
    
    # Validate specific formats
    openai_key = os.getenv('OPENAI_API_KEY', '')
    if openai_key and not openai_key.startswith('sk-'):
        issues.append(f"⚠️  INVALID FORMAT: OPENAI_API_KEY should start with 'sk-'")
    
    jwt_secret = os.getenv('JWT_SECRET_KEY', '')
    if jwt_secret and len(jwt_secret) < 32:
        issues.append(f"⚠️  WEAK: JWT_SECRET_KEY should be at least 32 characters long")
    
    return len(issues) == 0, issues

def main():
    """Main validation function"""
    
    print("🔍 ScrollIntel Environment Validation")
    print("=" * 50)
    
    # Load .env file if it exists
    env_file = '.env'
    if os.path.exists(env_file):
        print(f"📁 Loading environment from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    try:
                        key, value = line.strip().split('=', 1)
                        if key not in os.environ:  # Don't override existing env vars
                            os.environ[key] = value
                    except ValueError:
                        continue
    
    # Validate configuration
    is_valid, issues = validate_environment()
    
    if issues:
        print("\n🚨 Configuration Issues:")
        for issue in issues:
            print(f"   {issue}")
    
    if is_valid:
        print("\n🎉 Environment configuration is valid!")
        print("✅ Ready for production deployment")
        return True
    else:
        print("\n❌ Environment configuration has issues")
        print("💡 Fix the issues above before deploying to production")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
