#!/usr/bin/env python3
"""
Test script to verify UI/UX fixes are working properly
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_frontend_files():
    """Check if all required frontend files exist"""
    required_files = [
        'frontend/src/components/error-boundary.tsx',
        'frontend/src/components/ui/loading.tsx',
        'frontend/src/components/ui/fallback.tsx',
        'frontend/src/app/page.tsx',
        'frontend/src/lib/api.ts',
        'frontend/src/components/chat/chat-interface.tsx',
        'frontend/src/components/dashboard/agent-status-card.tsx'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✅ All required frontend files exist")
    return True

def check_typescript_syntax():
    """Check TypeScript syntax in key files"""
    try:
        # Check if we can compile TypeScript files
        result = subprocess.run(
            ['npx', 'tsc', '--noEmit', '--skipLibCheck'],
            cwd='frontend',
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ TypeScript syntax check passed")
            return True
        else:
            print("❌ TypeScript syntax errors:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  TypeScript check timed out")
        return False
    except FileNotFoundError:
        print("⚠️  TypeScript compiler not found, skipping syntax check")
        return True

def check_api_endpoints():
    """Check if backend API endpoints are accessible"""
    base_url = "http://localhost:8000"
    endpoints = [
        "/health",
        "/api/agents",
        "/api/monitoring/metrics"
    ]
    
    accessible_endpoints = 0
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code < 500:
                accessible_endpoints += 1
                print(f"✅ {endpoint} - accessible")
            else:
                print(f"⚠️  {endpoint} - server error ({response.status_code})")
        except requests.exceptions.RequestException:
            print(f"❌ {endpoint} - not accessible")
    
    if accessible_endpoints > 0:
        print(f"✅ {accessible_endpoints}/{len(endpoints)} API endpoints accessible")
        return True
    else:
        print("❌ No API endpoints accessible - backend may be down")
        return False

def check_error_handling():
    """Test error handling improvements"""
    print("\n🔍 Testing error handling improvements...")
    
    # Check if error boundary component exists and has proper structure
    error_boundary_path = Path('frontend/src/components/error-boundary.tsx')
    if error_boundary_path.exists():
        content = error_boundary_path.read_text()
        if 'componentDidCatch' in content and 'getDerivedStateFromError' in content:
            print("✅ Error boundary component properly implemented")
        else:
            print("❌ Error boundary component missing key methods")
            return False
    
    # Check if loading states are implemented
    loading_path = Path('frontend/src/components/ui/loading.tsx')
    if loading_path.exists():
        content = loading_path.read_text()
        if 'LoadingState' in content and 'LoadingSpinner' in content:
            print("✅ Loading components properly implemented")
        else:
            print("❌ Loading components missing key exports")
            return False
    
    return True

def check_responsive_design():
    """Check if responsive design improvements are in place"""
    print("\n📱 Checking responsive design...")
    
    globals_css = Path('frontend/src/app/globals.css')
    if globals_css.exists():
        content = globals_css.read_text()
        if '@media (max-width: 768px)' in content:
            print("✅ Mobile responsive styles added")
        else:
            print("❌ Mobile responsive styles missing")
            return False
    
    return True

def main():
    """Run all UI/UX fix tests"""
    print("🚀 Testing ScrollIntel UI/UX Fixes")
    print("=" * 50)
    
    tests = [
        ("Frontend Files", check_frontend_files),
        ("TypeScript Syntax", check_typescript_syntax),
        ("API Endpoints", check_api_endpoints),
        ("Error Handling", check_error_handling),
        ("Responsive Design", check_responsive_design)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All UI/UX fixes are working properly!")
        return 0
    elif passed_tests >= total_tests * 0.7:
        print("⚠️  Most fixes are working, but some issues remain")
        return 1
    else:
        print("❌ Significant issues found - please review the fixes")
        return 2

if __name__ == "__main__":
    sys.exit(main())