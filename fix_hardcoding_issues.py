#!/usr/bin/env python3
"""
Comprehensive Hardcoding Fix for ScrollIntel
Addresses all hardcoding issues in both backend and frontend
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any

class HardcodingFixer:
    """Comprehensive hardcoding fix utility"""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
        
    def fix_backend_hardcoding(self):
        """Fix hardcoding issues in backend Python files"""
        print("🔧 Fixing Backend Hardcoding Issues...")
        
        # Fix configuration files
        self._fix_config_files()
        
        # Fix deployment scripts
        self._fix_deployment_scripts()
        
        # Fix test files
        self._fix_test_files()
        
        # Fix API endpoints
        self._fix_api_endpoints()
        
        # Fix database configurations
        self._fix_database_configs()
        
    def fix_frontend_hardcoding(self):
        """Fix hardcoding issues in frontend TypeScript/JavaScript files"""
        print("🎨 Fixing Frontend Hardcoding Issues...")
        
        # Fix Next.js configuration
        self._fix_nextjs_config()
        
        # Fix API client configurations
        self._fix_api_client()
        
        # Fix environment handling
        self._fix_frontend_env()
        
        # Fix component hardcoding
        self._fix_component_hardcoding()
        
    def _fix_config_files(self):
        """Fix hardcoded values in configuration files"""
        config_files = [
            "scrollintel/core/config.py",
            "scrollintel/core/configuration_manager.py",
            "scrollintel/models/database.py"
        ]
        
        for file_path in config_files:
            if os.path.exists(file_path):
                self._fix_python_file_hardcoding(file_path)
                
    def _fix_deployment_scripts(self):
        """Fix hardcoded URLs in deployment scripts"""
        deployment_files = [
            "deploy_simple.py",
            "verify_deployment.py", 
            "upgrade_to_heavy_volume.py",
            "deploy_now.py",
            "deploy_cloud_now.py",
            "deploy_railway_now.py",
            "deploy_render_now.py"
        ]
        
        for file_path in deployment_files:
            if os.path.exists(file_path):
                print(f"  Fixing {file_path}...")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Replace hardcoded localhost URLs
                content = re.sub(
                    r'"http://localhost:(\d+)"',
                    r'f"http://{os.getenv(\'API_HOST\', \'localhost\')}:\1"',
                    content
                )
                
                content = re.sub(
                    r"'http://localhost:(\d+)'",
                    r'f"http://{os.getenv(\'API_HOST\', \'localhost\')}:\1"',
                    content
                )
                
                # Replace hardcoded database URLs
                content = re.sub(
                    r'"postgresql://[^"]*"',
                    r'os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/scrollintel")',
                    content
                )
                
                # Add environment variable import if needed
                if "os.getenv" in content and "import os" not in content:
                    content = "import os\n" + content
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixes_applied.append(f"Fixed hardcoded URLs in {file_path}")
                    print(f"    ✅ Fixed {file_path}")
                else:
                    print(f"    ℹ️  No hardcoding found in {file_path}")
                    
    def _fix_test_files(self):
        """Fix hardcoded values in test files"""
        test_patterns = [
            "test_*.py",
            "tests/test_*.py",
            "**/test_*.py"
        ]
        
        test_files = []
        for pattern in test_patterns:
            test_files.extend(Path(".").glob(pattern))
            
        for file_path in test_files:
            if file_path.exists():
                self._fix_python_file_hardcoding(str(file_path))
                
    def _fix_api_endpoints(self):
        """Fix hardcoded API endpoints"""
        api_files = [
            "scrollintel/api/main.py",
            "scrollintel/api/gateway.py",
            "scrollintel/api/routes/*.py"
        ]
        
        for pattern in api_files:
            for file_path in Path(".").glob(pattern):
                if file_path.exists():
                    self._fix_python_file_hardcoding(str(file_path))
                    
    def _fix_database_configs(self):
        """Fix hardcoded database configurations"""
        db_files = [
            "scrollintel/models/database.py",
            "scrollintel/core/database_connection_manager.py",
            "alembic/env.py"
        ]
        
        for file_path in db_files:
            if os.path.exists(file_path):
                self._fix_python_file_hardcoding(file_path)
                
    def _fix_python_file_hardcoding(self, file_path: str):
        """Fix hardcoding in a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix common hardcoded patterns
            replacements = [
                # Database URLs
                (r'"postgresql://postgres:password@localhost:5432/[^"]*"', 
                 'os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/scrollintel")'),
                
                # Redis URLs
                (r'"redis://localhost:6379"', 
                 'os.getenv("REDIS_URL", "redis://localhost:6379")'),
                
                # API URLs
                (r'"http://localhost:8000"', 
                 'os.getenv("API_URL", "http://localhost:8000")'),
                
                # Secret keys
                (r'"your-secret-key-change-in-production"', 
                 'os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")'),
                
                # API keys placeholders
                (r'"your-openai-api-key-here"', 
                 'os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")'),
                
                # SMTP settings
                (r'"localhost".*# SMTP', 
                 'os.getenv("SMTP_SERVER", "localhost")  # SMTP'),
            ]
            
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)
            
            # Add import os if needed
            if "os.getenv" in content and "import os" not in content:
                # Find the right place to add import
                lines = content.split('\n')
                import_added = False
                
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        continue
                    else:
                        lines.insert(i, 'import os')
                        import_added = True
                        break
                
                if not import_added:
                    lines.insert(0, 'import os')
                
                content = '\n'.join(lines)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixes_applied.append(f"Fixed hardcoding in {file_path}")
                
        except Exception as e:
            self.issues_found.append(f"Error fixing {file_path}: {e}")
            
    def _fix_nextjs_config(self):
        """Fix Next.js configuration hardcoding"""
        config_file = "frontend/next.config.js"
        
        if os.path.exists(config_file):
            print(f"  Fixing {config_file}...")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix hardcoded localhost in image domains
            content = re.sub(
                r"domains: \['localhost'\]",
                r"domains: [process.env.NEXT_PUBLIC_API_HOST || 'localhost']",
                content
            )
            
            # Ensure environment variables are properly configured
            if "NEXT_PUBLIC_API_URL" not in content:
                content = re.sub(
                    r"env: {",
                    r"""env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',""",
                    content
                )
            
            if content != original_content:
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixes_applied.append(f"Fixed Next.js configuration")
                print(f"    ✅ Fixed {config_file}")
            else:
                print(f"    ℹ️  No hardcoding found in {config_file}")
                
    def _fix_api_client(self):
        """Fix API client hardcoding"""
        api_file = "frontend/src/lib/api.ts"
        
        if os.path.exists(api_file):
            print(f"  Fixing {api_file}...")
            
            with open(api_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Ensure proper environment variable usage
            content = re.sub(
                r"const API_BASE_URL = process\.env\.NEXT_PUBLIC_API_URL \|\| 'http://localhost:8000'",
                r"""const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 
  (typeof window !== 'undefined' && window.location.hostname !== 'localhost' 
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : 'http://localhost:8000')""",
                content
            )
            
            if content != original_content:
                with open(api_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixes_applied.append(f"Fixed API client configuration")
                print(f"    ✅ Fixed {api_file}")
            else:
                print(f"    ℹ️  No hardcoding found in {api_file}")
                
    def _fix_frontend_env(self):
        """Fix frontend environment handling"""
        # Create frontend environment template
        frontend_env_template = """# Frontend Environment Variables
# Copy this to frontend/.env.local for development

# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Application Configuration
NEXT_PUBLIC_APP_NAME=ScrollIntel
NEXT_PUBLIC_APP_VERSION=4.0.0

# Analytics (Optional)
NEXT_PUBLIC_GA_ID=
NEXT_PUBLIC_POSTHOG_KEY=

# Feature Flags
NEXT_PUBLIC_ENABLE_ANALYTICS=false
NEXT_PUBLIC_ENABLE_MONITORING=false
"""
        
        frontend_env_path = "frontend/.env.example"
        with open(frontend_env_path, 'w', encoding='utf-8') as f:
            f.write(frontend_env_template)
        
        self.fixes_applied.append("Created frontend environment template")
        print(f"    ✅ Created {frontend_env_path}")
        
    def _fix_component_hardcoding(self):
        """Fix hardcoding in React components"""
        component_files = list(Path("frontend/src").glob("**/*.tsx")) + list(Path("frontend/src").glob("**/*.ts"))
        
        for file_path in component_files:
            if file_path.exists() and file_path.name != "api.ts":  # Skip api.ts as it's already fixed
                self._fix_typescript_file_hardcoding(str(file_path))
                
    def _fix_typescript_file_hardcoding(self, file_path: str):
        """Fix hardcoding in TypeScript/JavaScript files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix common hardcoded patterns in frontend
            replacements = [
                # API URLs
                (r"'http://localhost:8000'", 
                 "process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'"),
                
                (r'"http://localhost:8000"', 
                 'process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"'),
                
                # WebSocket URLs
                (r"'ws://localhost:8000'", 
                 "process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'"),
                
                (r'"ws://localhost:8000"', 
                 'process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000"'),
                
                # App names and versions
                (r'"ScrollIntel"(?=.*title|.*name)', 
                 'process.env.NEXT_PUBLIC_APP_NAME || "ScrollIntel"'),
            ]
            
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                # Only log significant fixes to avoid spam
                if "localhost" in original_content:
                    self.fixes_applied.append(f"Fixed hardcoding in {file_path}")
                
        except Exception as e:
            self.issues_found.append(f"Error fixing {file_path}: {e}")
            
    def create_comprehensive_env_templates(self):
        """Create comprehensive environment templates"""
        print("📝 Creating Environment Templates...")
        
        # Backend production template
        backend_prod_template = """# ScrollIntel Production Configuration
# Copy this to .env.production and fill in real values

# === CRITICAL: Set these for production ===
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
NODE_ENV=production

# === Application Configuration ===
API_HOST=0.0.0.0
API_PORT=8000
BASE_URL=https://yourdomain.com
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# === Database (REQUIRED) ===
DATABASE_URL=postgresql://username:password@host:5432/scrollintel
POSTGRES_HOST=your-production-db-host
POSTGRES_PORT=5432
POSTGRES_DB=scrollintel
POSTGRES_USER=your-db-user
POSTGRES_PASSWORD=your-secure-db-password

# === AI Services (REQUIRED for AI features) ===
OPENAI_API_KEY=sk-your-actual-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
OPENAI_MODEL=gpt-4
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# === Security (REQUIRED) ===
JWT_SECRET_KEY=your-very-secure-jwt-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# === Email Service (REQUIRED for user onboarding) ===
SMTP_SERVER=your-smtp-server.com
SMTP_PORT=587
EMAIL_USERNAME=your-email@yourdomain.com
EMAIL_PASSWORD=your-email-password
FROM_EMAIL=noreply@yourdomain.com

# === Redis (Optional - will use in-memory if not available) ===
REDIS_URL=redis://username:password@host:6379
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password
REDIS_DB=0

# === File Storage ===
MAX_FILE_SIZE=100MB
UPLOAD_DIR=./uploads

# === Monitoring & Analytics (Optional) ===
SENTRY_DSN=your-sentry-dsn
POSTHOG_API_KEY=your-posthog-key

# === External Services (Optional) ===
STRIPE_API_KEY=your-stripe-api-key
STRIPE_WEBHOOK_SECRET=your-stripe-webhook-secret

# === Vector Database (Optional) ===
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1

# === Supabase (Optional) ===
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_SERVICE_ROLE=your-supabase-service-role-key
SUPABASE_JWT_SECRET=your-supabase-jwt-secret

# === Rate Limiting ===
RATE_LIMIT_PER_SECOND=10
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000
RATE_LIMIT_PER_DAY=10000

# === Scaling Configuration ===
MIN_INSTANCES=2
MAX_INSTANCES=10
TARGET_CPU=70.0
TARGET_MEMORY=80.0
"""
        
        # Frontend production template
        frontend_prod_template = """# Frontend Production Environment Variables
# Copy this to frontend/.env.production and fill in real values

# === API Configuration ===
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_WS_URL=wss://api.yourdomain.com

# === Application Configuration ===
NEXT_PUBLIC_APP_NAME=ScrollIntel
NEXT_PUBLIC_APP_VERSION=4.0.0
NEXT_PUBLIC_APP_DESCRIPTION=Advanced AI-Powered Business Intelligence Platform

# === Analytics & Monitoring ===
NEXT_PUBLIC_GA_ID=G-XXXXXXXXXX
NEXT_PUBLIC_POSTHOG_KEY=your-posthog-key
NEXT_PUBLIC_SENTRY_DSN=your-frontend-sentry-dsn

# === Feature Flags ===
NEXT_PUBLIC_ENABLE_ANALYTICS=true
NEXT_PUBLIC_ENABLE_MONITORING=true
NEXT_PUBLIC_ENABLE_DEBUG=false

# === External Services ===
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_live_your-stripe-key
"""
        
        # Development template
        dev_template = """# ScrollIntel Development Configuration
# Copy this to .env.development for local development

# === Environment ===
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
NODE_ENV=development

# === Application ===
API_HOST=localhost
API_PORT=8000
BASE_URL=http://localhost:3000

# === Database ===
DATABASE_URL=postgresql://postgres:password@localhost:5432/scrollintel
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=scrollintel
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# === AI Services (Use test keys for development) ===
OPENAI_API_KEY=your-dev-openai-key
ANTHROPIC_API_KEY=your-dev-anthropic-key

# === Security ===
JWT_SECRET_KEY=dev-secret-key-not-for-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# === Email (Use test SMTP for development) ===
SMTP_SERVER=localhost
SMTP_PORT=1025
EMAIL_USERNAME=
EMAIL_PASSWORD=
FROM_EMAIL=dev@scrollintel.local

# === Redis (Optional for development) ===
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
SKIP_REDIS=true

# === File Storage ===
MAX_FILE_SIZE=50MB
UPLOAD_DIR=./uploads

# === Development Tools ===
ENABLE_PROFILING=true
ENABLE_DEBUG_TOOLBAR=true
"""
        
        # Write templates
        templates = [
            (".env.production.template", backend_prod_template),
            ("frontend/.env.production.template", frontend_prod_template),
            (".env.development.template", dev_template)
        ]
        
        for file_path, content in templates:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.fixes_applied.append(f"Created {file_path}")
            print(f"    ✅ Created {file_path}")
            
    def validate_current_config(self):
        """Validate current configuration for production readiness"""
        print("\n🔍 Validating Current Configuration...")
        
        # Load current .env
        env_vars = {}
        if os.path.exists(".env"):
            with open(".env", "r", encoding='utf-8') as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        try:
                            key, value = line.strip().split("=", 1)
                            env_vars[key] = value
                        except ValueError:
                            continue
        
        # Check critical variables
        critical_vars = {
            "POSTGRES_PASSWORD": "Database password",
            "JWT_SECRET_KEY": "JWT secret key", 
            "OPENAI_API_KEY": "OpenAI API key"
        }
        
        placeholder_values = [
            "", 
            "your-openai-api-key-here", 
            "your-secret-key-change-in-production",
            "your-anthropic-key-here",
            "your-secure-db-password"
        ]
        
        issues = []
        for var, description in critical_vars.items():
            if var not in env_vars:
                issues.append(f"❌ Missing: {var} ({description})")
            elif env_vars[var] in placeholder_values:
                issues.append(f"⚠️  Placeholder value: {var} ({description})")
            else:
                print(f"✅ Configured: {var}")
        
        # Check frontend configuration
        frontend_env_path = "frontend/.env.local"
        if not os.path.exists(frontend_env_path):
            issues.append(f"⚠️  Missing frontend environment file: {frontend_env_path}")
        
        if issues:
            print("\n🚨 Configuration Issues Found:")
            for issue in issues:
                print(f"   {issue}")
            print("\n💡 Fix these before production deployment!")
            return False
        else:
            print("\n🎉 Configuration looks good for production!")
            return True
            
    def generate_report(self):
        """Generate a comprehensive report of fixes applied"""
        print("\n" + "=" * 60)
        print("📊 HARDCODING FIX REPORT")
        print("=" * 60)
        
        print(f"\n✅ Fixes Applied ({len(self.fixes_applied)}):")
        for fix in self.fixes_applied:
            print(f"   • {fix}")
            
        if self.issues_found:
            print(f"\n⚠️  Issues Found ({len(self.issues_found)}):")
            for issue in self.issues_found:
                print(f"   • {issue}")
        
        print(f"\n📋 Next Steps:")
        print("   1. Review the generated .env templates")
        print("   2. Copy appropriate templates to .env files")
        print("   3. Fill in real API keys and credentials")
        print("   4. Test in development environment")
        print("   5. Deploy to staging for validation")
        print("   6. Deploy to production")
        
        print(f"\n💡 Important Notes:")
        print("   • Never commit real API keys to version control")
        print("   • Use different keys for development/staging/production")
        print("   • Regularly rotate secrets and API keys")
        print("   • Monitor usage and set up alerts")
        
        return len(self.issues_found) == 0

def main():
    """Main function to fix all hardcoding issues"""
    print("🔧 ScrollIntel Comprehensive Hardcoding Fix")
    print("=" * 60)
    
    fixer = HardcodingFixer()
    
    try:
        # 1. Fix backend hardcoding
        fixer.fix_backend_hardcoding()
        
        # 2. Fix frontend hardcoding  
        fixer.fix_frontend_hardcoding()
        
        # 3. Create comprehensive environment templates
        fixer.create_comprehensive_env_templates()
        
        # 4. Validate current configuration
        config_ok = fixer.validate_current_config()
        
        # 5. Generate report
        success = fixer.generate_report()
        
        return success and config_ok
        
    except Exception as e:
        print(f"\n❌ Error during hardcoding fix: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)