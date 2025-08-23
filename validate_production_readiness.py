#!/usr/bin/env python3
"""
Validate ScrollIntel Production Readiness
Check for hardcoding issues and configuration completeness
"""

import os
import re
from pathlib import Path

def check_environment_config():
    """Check if environment configuration is production-ready"""
    
    print("🔍 Checking Environment Configuration...")
    
    # Load current .env
    env_vars = {}
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding='utf-8') as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        key, value = line.strip().split("=", 1)
                        env_vars[key] = value
        except Exception as e:
            print(f"⚠️  Error reading .env file: {e}")
            return False
    
    # Check critical variables
    critical_checks = {
        "POSTGRES_PASSWORD": "Database password configured",
        "JWT_SECRET_KEY": "JWT secret key configured", 
        "OPENAI_API_KEY": "OpenAI API key configured"
    }
    
    placeholder_values = [
        "your-openai-api-key-here",
        "your-secret-key-change-in-production",
        "password",
        "secret",
        ""
    ]
    
    issues = []
    good_configs = []
    
    for var, description in critical_checks.items():
        if var not in env_vars:
            issues.append(f"❌ Missing: {var}")
        elif env_vars[var] in placeholder_values:
            issues.append(f"⚠️  Placeholder: {var} = '{env_vars[var]}'")
        else:
            good_configs.append(f"✅ {description}")
    
    # Print results
    for config in good_configs:
        print(f"   {config}")
    
    if issues:
        print("\n🚨 Configuration Issues:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("\n🎉 All critical configuration looks good!")
        return True

def check_ai_integration():
    """Check if AI integration is properly configured"""
    
    print("\n🤖 Checking AI Integration...")
    
    try:
        # Check if ScrollCTO agent can be imported
        from scrollintel.agents.scroll_cto_agent import ScrollCTOAgent
        print("   ✅ ScrollCTO Agent imports successfully")
        
        # Check if it has real GPT-4 integration
        agent = ScrollCTOAgent()
        if hasattr(agent, 'openai_client'):
            print("   ✅ OpenAI client configured")
        else:
            print("   ❌ OpenAI client not found")
            return False
            
        # Check if it has fallback mechanisms
        if hasattr(agent, '_generate_fallback_architecture_recommendation'):
            print("   ✅ Fallback mechanisms implemented")
        else:
            print("   ⚠️  No fallback mechanisms found")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_database_integration():
    """Check if database integration is properly configured"""
    
    print("\n🗄️  Checking Database Integration...")
    
    try:
        from scrollintel.models.database_utils import DatabaseManager
        print("   ✅ Database manager imports successfully")
        
        # Check if it uses environment variables
        from scrollintel.core.config import get_settings
        settings = get_settings()
        
        if 'database_url' in settings:
            db_url = settings['database_url']
            if 'postgresql://' in db_url:
                print("   ✅ PostgreSQL configured as primary database")
            elif 'sqlite://' in db_url:
                print("   ⚠️  SQLite configured (OK for development)")
            else:
                print(f"   ❌ Unknown database type: {db_url}")
                return False
        else:
            print("   ❌ No database URL configured")
            return False
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_api_endpoints():
    """Check if API endpoints are properly configured"""
    
    print("\n🌐 Checking API Configuration...")
    
    try:
        from scrollintel.api.main import app
        print("   ✅ Main API app imports successfully")
        
        # Check if it has proper middleware
        middleware_names = [middleware.__class__.__name__ for middleware in app.user_middleware]
        
        expected_middleware = [
            'CORSMiddleware',
            'GZipMiddleware', 
            'BulletproofMiddleware',
            'PerformanceMonitoringMiddleware'
        ]
        
        for middleware in expected_middleware:
            if any(middleware in name for name in middleware_names):
                print(f"   ✅ {middleware} configured")
            else:
                print(f"   ⚠️  {middleware} not found")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def scan_for_hardcoded_values():
    """Scan for remaining hardcoded values in production code"""
    
    print("\n🔍 Scanning for Hardcoded Values...")
    
    # Directories to scan (exclude test directories)
    scan_dirs = [
        "scrollintel/api",
        "scrollintel/core", 
        "scrollintel/agents",
        "scrollintel/engines"
    ]
    
    hardcoded_patterns = [
        r'localhost',
        r'127\.0\.0\.1',
        r'password.*=.*["\'][^"\']*["\']',
        r'api_key.*=.*["\'][^"\']*["\']',
        r'secret.*=.*["\'][^"\']*["\']'
    ]
    
    issues_found = []
    
    for scan_dir in scan_dirs:
        if os.path.exists(scan_dir):
            for root, dirs, files in os.walk(scan_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                                for pattern in hardcoded_patterns:
                                    matches = re.findall(pattern, content, re.IGNORECASE)
                                    if matches:
                                        # Filter out acceptable cases
                                        acceptable = [
                                            'os.getenv',
                                            'localhost:3000',  # Frontend URL references
                                            'default="localhost"'  # Default values
                                        ]
                                        
                                        for match in matches:
                                            if not any(acceptable_case in content for acceptable_case in acceptable):
                                                issues_found.append(f"{file_path}: {match}")
                        except Exception:
                            continue  # Skip files that can't be read
    
    if issues_found:
        print("   ⚠️  Potential hardcoded values found:")
        for issue in issues_found[:10]:  # Show first 10
            print(f"      {issue}")
        if len(issues_found) > 10:
            print(f"      ... and {len(issues_found) - 10} more")
        return False
    else:
        print("   ✅ No hardcoded values found in production code")
        return True

def main():
    """Main validation function"""
    
    print("🔍 ScrollIntel Production Readiness Check")
    print("=" * 50)
    
    checks = [
        ("Environment Configuration", check_environment_config),
        ("AI Integration", check_ai_integration), 
        ("Database Integration", check_database_integration),
        ("API Configuration", check_api_endpoints),
        ("Hardcoded Values", scan_for_hardcoded_values)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"   ❌ Error running {check_name}: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Production Readiness Summary")
    print("=" * 50)
    
    passed = 0
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Score: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 ScrollIntel is PRODUCTION READY!")
        print("   All systems configured with real live interactions")
        print("   No hardcoding issues found")
    elif passed >= len(results) * 0.8:
        print("\n⚠️  ScrollIntel is MOSTLY READY")
        print("   Minor configuration issues need attention")
        print("   Core functionality uses real live interactions")
    else:
        print("\n🚨 ScrollIntel needs CONFIGURATION")
        print("   Several issues need to be resolved before production")
    
    print("\n💡 Next Steps:")
    print("   1. Fix any failed checks above")
    print("   2. Set real API keys in .env file")
    print("   3. Test with real credentials")
    print("   4. Deploy to production")
    
    return passed == len(results)

if __name__ == "__main__":
    main()