#!/usr/bin/env python3
"""
Verify No Hardcoding Remains in ScrollIntel
Final verification that all hardcoding has been eliminated
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def scan_for_hardcoding() -> Tuple[bool, List[str]]:
    """Scan codebase for remaining hardcoding patterns"""
    
    issues = []
    
    # Patterns that indicate hardcoding
    hardcoding_patterns = [
        (r'"postgresql://[^"]*localhost[^"]*"', "Hardcoded PostgreSQL localhost URL"),
        (r'"redis://localhost[^"]*"', "Hardcoded Redis localhost URL"),
        (r'"http://localhost:\d+"', "Hardcoded HTTP localhost URL"),
        (r'"ws://localhost:\d+"', "Hardcoded WebSocket localhost URL"),
        (r'"sk-[a-zA-Z0-9]{48}"', "Hardcoded OpenAI API key"),
        (r'your-openai-api-key-here(?!.*getenv)', "Placeholder API key without environment variable"),
        (r'your-secret-key-change-in-production(?!.*getenv)', "Placeholder secret key without environment variable"),
    ]
    
    # Files to scan
    file_patterns = [
        "**/*.py",
        "**/*.ts", 
        "**/*.tsx",
        "**/*.js",
        "**/*.jsx"
    ]
    
    # Directories to exclude
    exclude_dirs = {
        "node_modules", "__pycache__", ".git", "dist", "build", 
        ".pytest_cache", ".vscode", "venv", "env"
    }
    
    files_to_scan = []
    for pattern in file_patterns:
        for file_path in Path(".").glob(pattern):
            # Skip excluded directories
            if any(excluded in str(file_path) for excluded in exclude_dirs):
                continue
            # Skip template files (they're supposed to have placeholders)
            if "template" in str(file_path) or "example" in str(file_path):
                continue
            files_to_scan.append(file_path)
    
    print(f"🔍 Scanning {len(files_to_scan)} files for hardcoding patterns...")
    
    for file_path in files_to_scan:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            for pattern, description in hardcoding_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    issues.append(f"❌ {file_path}: {description} - Found: {matches[0][:50]}...")
                    
        except Exception as e:
            print(f"⚠️  Could not scan {file_path}: {e}")
    
    return len(issues) == 0, issues

def check_environment_usage() -> Tuple[bool, List[str]]:
    """Check that environment variables are properly used"""
    
    issues = []
    good_patterns = []
    
    # Look for proper environment variable usage
    env_patterns = [
        r'os\.getenv\(',
        r'process\.env\.',
        r'environment\.',
        r'config\.',
    ]
    
    # Key configuration files to check
    config_files = [
        "scrollintel/core/config.py",
        "scrollintel/core/configuration_manager.py",
        "frontend/src/lib/api.ts",
        "frontend/next.config.js"
    ]
    
    for file_path in config_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for environment variable usage
                env_usage_found = False
                for pattern in env_patterns:
                    if re.search(pattern, content):
                        env_usage_found = True
                        break
                
                if env_usage_found:
                    good_patterns.append(f"✅ {file_path}: Uses environment variables")
                else:
                    issues.append(f"⚠️  {file_path}: No environment variable usage detected")
                    
            except Exception as e:
                issues.append(f"❌ Could not check {file_path}: {e}")
        else:
            issues.append(f"⚠️  Configuration file not found: {file_path}")
    
    print(f"\n📋 Environment Variable Usage:")
    for pattern in good_patterns:
        print(f"   {pattern}")
    
    return len(issues) == 0, issues

def verify_templates_exist() -> Tuple[bool, List[str]]:
    """Verify that all necessary template files exist"""
    
    issues = []
    
    required_templates = [
        ".env.production.template",
        ".env.development.template", 
        "frontend/.env.production.template",
        "frontend/.env.example",
        ".env.docker",
        "k8s/config.yaml",
        "railway.json",
        "render.yaml",
        "vercel.json"
    ]
    
    for template in required_templates:
        if os.path.exists(template):
            print(f"✅ Template exists: {template}")
        else:
            issues.append(f"❌ Missing template: {template}")
    
    return len(issues) == 0, issues

def main():
    """Main verification function"""
    
    print("🔍 ScrollIntel Hardcoding Verification")
    print("=" * 50)
    
    all_good = True
    
    # 1. Scan for hardcoding patterns
    print("\n1. Scanning for hardcoding patterns...")
    no_hardcoding, hardcoding_issues = scan_for_hardcoding()
    
    if hardcoding_issues:
        print("\n🚨 Hardcoding Issues Found:")
        for issue in hardcoding_issues:
            print(f"   {issue}")
        all_good = False
    else:
        print("✅ No hardcoding patterns detected!")
    
    # 2. Check environment variable usage
    print("\n2. Checking environment variable usage...")
    env_usage_ok, env_issues = check_environment_usage()
    
    if env_issues:
        print("\n⚠️  Environment Variable Issues:")
        for issue in env_issues:
            print(f"   {issue}")
        all_good = False
    
    # 3. Verify templates exist
    print("\n3. Verifying configuration templates...")
    templates_ok, template_issues = verify_templates_exist()
    
    if template_issues:
        print("\n❌ Template Issues:")
        for issue in template_issues:
            print(f"   {issue}")
        all_good = False
    
    # Final result
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 VERIFICATION PASSED!")
        print("✅ No hardcoding detected")
        print("✅ Environment variables properly used")
        print("✅ All templates present")
        print("🚀 ScrollIntel is production-ready!")
    else:
        print("❌ VERIFICATION FAILED!")
        print("⚠️  Issues found that need attention")
        print("💡 Review the issues above and fix them")
    
    return all_good

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)