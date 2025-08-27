#!/usr/bin/env python3
"""
Test ScrollIntel before deployment to scrollintel.com
"""

import requests
import subprocess
import sys
import time
import os
from pathlib import Path

def test_local_backend():
    """Test if local backend is working"""
    print("🧪 Testing local backend...")
    
    try:
        # Try to start the backend
        print("Starting backend...")
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "scrollintel.api.simple_main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Test health endpoint
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend health check passed")
                backend_working = True
            else:
                print(f"⚠️  Backend health check failed: {response.status_code}")
                backend_working = False
        except requests.exceptions.RequestException as e:
            print(f"❌ Backend not responding: {e}")
            backend_working = False
        
        # Test API docs
        try:
            response = requests.get("http://localhost:8000/docs", timeout=5)
            if response.status_code == 200:
                print("✅ API documentation accessible")
            else:
                print("⚠️  API documentation not accessible")
        except:
            print("⚠️  API documentation not accessible")
        
        # Cleanup
        backend_process.terminate()
        backend_process.wait()
        
        return backend_working
        
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False

def test_frontend_files():
    """Test if frontend files exist"""
    print("🧪 Testing frontend files...")
    
    frontend_path = Path("frontend")
    if not frontend_path.exists():
        print("❌ Frontend directory not found")
        return False
    
    required_files = [
        "frontend/package.json",
        "frontend/next.config.js",
        "frontend/src/app/page.tsx"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing frontend files: {missing_files}")
        return False
    else:
        print("✅ Frontend files present")
        return True

def test_environment_variables():
    """Test environment variables"""
    print("🧪 Testing environment variables...")
    
    # Check for .env file
    if Path(".env").exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found (optional)")
    
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✅ OpenAI API key configured")
        return True
    else:
        print("⚠️  OpenAI API key not found in environment")
        print("   Set OPENAI_API_KEY in your deployment platform")
        return False

def test_git_repository():
    """Test Git repository status"""
    print("🧪 Testing Git repository...")
    
    try:
        # Check if git repo exists
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git repository initialized")
        else:
            print("❌ Git repository not initialized")
            return False
        
        # Check remote
        result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True)
        if 'github.com' in result.stdout:
            print("✅ GitHub remote configured")
            return True
        else:
            print("❌ GitHub remote not configured")
            return False
            
    except FileNotFoundError:
        print("❌ Git not installed")
        return False

def main():
    """Run all pre-deployment tests"""
    print("🚀 ScrollIntel Pre-Deployment Test")
    print("==================================")
    
    tests = [
        ("Git Repository", test_git_repository),
        ("Environment Variables", test_environment_variables),
        ("Frontend Files", test_frontend_files),
        ("Local Backend", test_local_backend)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        results[test_name] = test_func()
    
    print("\n📊 Test Results Summary:")
    print("========================")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n🎯 Deployment Readiness:")
    if all_passed:
        print("✅ ScrollIntel is ready for deployment!")
        print("📖 Follow DEPLOY_SCROLLINTEL_COM_NOW.md to deploy")
        print("🚀 Recommended: Use Railway for easiest deployment")
    else:
        print("⚠️  Some tests failed, but you can still deploy")
        print("📖 Follow DEPLOY_SCROLLINTEL_COM_NOW.md for instructions")
        print("💡 Fix any issues after deployment if needed")
    
    print(f"\n🌐 Your GitHub repo: https://github.com/stanleymay20/scrollintel-ai-system")
    print("🎉 Ready to make ScrollIntel live at scrollintel.com!")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)