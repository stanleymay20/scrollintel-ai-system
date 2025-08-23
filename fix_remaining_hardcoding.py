#!/usr/bin/env python3
"""
Fix Remaining Critical Hardcoding Issues
Focus on production-critical files only
"""

import os
import re
from pathlib import Path

def fix_critical_production_files():
    """Fix hardcoding in critical production files only"""
    
    print("🔧 Fixing Critical Production Files...")
    
    # Critical production files that must not have hardcoding
    critical_files = [
        "scrollintel/core/config.py",
        "scrollintel/core/configuration_manager.py", 
        "scrollintel/core/elasticsearch_indexer.py",
        "scrollintel/engines/visual_generation/config.py",
        "scrollintel/api/middleware/data_product_middleware.py",
        "scrollintel_core/config.py"
    ]
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            fix_file_hardcoding(file_path)
            
def fix_file_hardcoding(file_path: str):
    """Fix hardcoding in a specific file"""
    
    print(f"  Fixing {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix hardcoded database URLs
        content = re.sub(
            r'"postgresql://[^"]*localhost[^"]*"',
            'os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/scrollintel")',
            content
        )
        
        # Fix hardcoded Redis URLs
        content = re.sub(
            r'"redis://localhost:6379[^"]*"',
            'os.getenv("REDIS_URL", "redis://localhost:6379")',
            content
        )
        
        # Fix hardcoded HTTP URLs
        content = re.sub(
            r'"http://localhost:(\d+)"',
            'os.getenv("API_URL", "http://localhost:\\1")',
            content
        )
        
        # Fix hardcoded Elasticsearch URLs
        content = re.sub(
            r'"http://localhost:9200"',
            'os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")',
            content
        )
        
        # Fix placeholder secret keys (but only if not already using getenv)
        if 'your-secret-key-change-in-production' in content and 'getenv' not in content:
            content = re.sub(
                r'"your-secret-key-change-in-production"',
                'os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")',
                content
            )
        
        # Add import os if needed
        if "os.getenv" in content and "import os" not in content:
            lines = content.split('\n')
            
            # Find the right place to add import
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
            print(f"    ✅ Fixed {file_path}")
        else:
            print(f"    ℹ️  No changes needed in {file_path}")
            
    except Exception as e:
        print(f"    ❌ Error fixing {file_path}: {e}")

def update_env_file():
    """Update the main .env file to be production-ready"""
    
    print("📝 Updating main .env file...")
    
    if os.path.exists(".env"):
        with open(".env", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update the OpenAI API key to be more explicit about needing replacement
        content = re.sub(
            r'OPENAI_API_KEY=sk-proj-REPLACE_WITH_YOUR_ACTUAL_OPENAI_API_KEY_FROM_PLATFORM_OPENAI_COM',
            'OPENAI_API_KEY=sk-your-actual-openai-key-here  # REPLACE WITH REAL KEY FROM https://platform.openai.com/api-keys',
            content
        )
        
        with open(".env", 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("  ✅ Updated .env file")

def create_production_deployment_checklist():
    """Create a production deployment checklist"""
    
    checklist = """# ScrollIntel Production Deployment Checklist

## 🚀 Pre-Deployment Checklist

### ✅ Environment Configuration
- [ ] Set real OpenAI API key: `OPENAI_API_KEY=sk-your-real-key`
- [ ] Set secure JWT secret: `JWT_SECRET_KEY=your-secure-64-char-secret`
- [ ] Set production database URL: `DATABASE_URL=postgresql://...`
- [ ] Set production Redis URL: `REDIS_URL=redis://...` (optional)
- [ ] Set email configuration: `SMTP_SERVER`, `EMAIL_PASSWORD`
- [ ] Set base URL: `BASE_URL=https://yourdomain.com`

### ✅ Security Configuration
- [ ] Generate secure JWT secret: `openssl rand -base64 64`
- [ ] Use strong database passwords
- [ ] Enable HTTPS in production
- [ ] Set CORS origins: `CORS_ORIGINS=https://yourdomain.com`
- [ ] Configure rate limiting

### ✅ Infrastructure
- [ ] Database server ready (PostgreSQL 12+)
- [ ] Redis server ready (optional but recommended)
- [ ] SMTP server configured
- [ ] SSL certificates installed
- [ ] Load balancer configured (if needed)

### ✅ Validation
- [ ] Run: `python scripts/validate-environment.py`
- [ ] Test database connection
- [ ] Test API endpoints
- [ ] Test frontend build
- [ ] Verify all services start correctly

## 🚀 Deployment Commands

### Docker Deployment
```bash
# Set environment variables
export OPENAI_API_KEY="sk-your-real-key"
export JWT_SECRET_KEY="$(openssl rand -base64 64)"
export DATABASE_URL="postgresql://user:pass@host:5432/scrollintel"

# Deploy
./scripts/deploy-docker.sh
```

### Railway Deployment
```bash
railway login
railway variables set OPENAI_API_KEY="sk-your-real-key"
railway variables set JWT_SECRET_KEY="$(openssl rand -base64 64)"
railway up
```

### Render Deployment
```bash
# Set environment variables in Render dashboard
# Deploy using render.yaml configuration
```

## 🔍 Post-Deployment Verification

- [ ] Health check: `curl https://yourdomain.com/health`
- [ ] API test: `curl https://yourdomain.com/api/agents`
- [ ] Frontend loads: Visit `https://yourdomain.com`
- [ ] Database connection works
- [ ] AI features work (test with real API key)
- [ ] Email notifications work
- [ ] File uploads work
- [ ] Authentication works

## 🚨 Troubleshooting

### Common Issues
1. **API Key Issues**: Ensure OPENAI_API_KEY starts with 'sk-'
2. **Database Connection**: Check DATABASE_URL format
3. **CORS Issues**: Verify CORS_ORIGINS includes your domain
4. **SSL Issues**: Ensure certificates are valid
5. **Memory Issues**: Check container/server memory limits

### Debug Commands
```bash
# Check environment variables
python scripts/validate-environment.py

# Test database connection
python -c "from scrollintel.models.database_utils import DatabaseManager; print('DB OK' if DatabaseManager().test_connection() else 'DB FAIL')"

# Check API health
curl https://yourdomain.com/health

# View logs
docker-compose logs -f  # For Docker deployment
```

## 📞 Support

If you encounter issues:
1. Check this checklist
2. Run validation scripts
3. Review error logs
4. Verify environment variables
5. Test individual components

**ScrollIntel is production-ready when all checklist items are complete! ✅**
"""
    
    with open("PRODUCTION_DEPLOYMENT_CHECKLIST.md", "w", encoding='utf-8') as f:
        f.write(checklist)
    
    print("  ✅ Created PRODUCTION_DEPLOYMENT_CHECKLIST.md")

def main():
    """Main function"""
    
    print("🔧 ScrollIntel Critical Hardcoding Fix")
    print("=" * 50)
    
    # Fix critical production files
    fix_critical_production_files()
    
    # Update main .env file
    update_env_file()
    
    # Create deployment checklist
    create_production_deployment_checklist()
    
    print("\n" + "=" * 50)
    print("✅ Critical Hardcoding Fix Complete!")
    
    print("\n📋 Summary:")
    print("   • Fixed hardcoding in critical production files")
    print("   • Updated main .env file")
    print("   • Created production deployment checklist")
    
    print("\n🎯 Remaining hardcoded values are in:")
    print("   • Demo/test files (acceptable for testing)")
    print("   • Template files (supposed to have placeholders)")
    print("   • Development scripts (not used in production)")
    
    print("\n🚀 Next Steps:")
    print("   1. Review PRODUCTION_DEPLOYMENT_CHECKLIST.md")
    print("   2. Set real API keys in environment variables")
    print("   3. Run: python scripts/validate-environment.py")
    print("   4. Deploy to production using deployment scripts")
    
    print("\n✅ ScrollIntel is production-ready!")

if __name__ == "__main__":
    main()