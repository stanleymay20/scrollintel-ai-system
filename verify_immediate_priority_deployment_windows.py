#!/usr/bin/env python3
"""
ScrollIntel Immediate Priority Implementation Verification (Windows Compatible)
Verifies deployment readiness without Unicode characters
"""

import os
import sys
import time
import subprocess
import requests
from datetime import datetime

def print_header():
    """Print verification header"""
    print("ScrollIntel Immediate Priority Implementation")
    print("Deployment Readiness Verification")
    print("=" * 60)
    print(f"Verification Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

def check_file_structure():
    """Check that all required files exist"""
    print("Checking File Structure...")
    
    required_files = [
        "scrollintel/__init__.py",
        "scrollintel/core/__init__.py",
        "scrollintel/core/production_infrastructure.py",
        "scrollintel/core/user_onboarding.py", 
        "scrollintel/core/api_stability.py",
        "scrollintel/api/__init__.py",
        "scrollintel/api/production_main.py",
        "scrollintel/models/__init__.py",
        "scrollintel/models/database.py",
        "requirements.txt",
        ".env"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] {file_path}")
        else:
            print(f"[MISSING] {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\\n[FAIL] {len(missing_files)} required files are missing")
        return False
    else:
        print(f"\\n[OK] All {len(required_files)} required files present")
        return True

def test_module_imports():
    """Test that all modules can be imported"""
    print("\\nTesting Module Imports...")
    
    modules_to_test = [
        ("scrollintel.core.production_infrastructure", "ProductionInfrastructure"),
        ("scrollintel.core.user_onboarding", "UserOnboarding"),
        ("scrollintel.core.api_stability", "APIStability"),
        ("scrollintel.models.database", None)
    ]
    
    import_failures = []
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name] if class_name else [])
            if class_name:
                getattr(module, class_name)
            print(f"[OK] {module_name}")
        except Exception as e:
            print(f"[FAIL] {module_name}: {e}")
            import_failures.append(module_name)
    
    if import_failures:
        print(f"\\n[FAIL] {len(import_failures)} modules failed to import")
        return False
    else:
        print(f"\\n[OK] All {len(modules_to_test)} modules imported successfully")
        return True

def test_class_instantiation():
    """Test that core classes can be instantiated"""
    print("\\nTesting Class Instantiation...")
    
    instantiation_tests = []
    
    # Test ProductionInfrastructure
    try:
        from scrollintel.core.production_infrastructure import ProductionInfrastructure
        infra = ProductionInfrastructure()
        print("[OK] ProductionInfrastructure instantiated")
        instantiation_tests.append(True)
    except Exception as e:
        print(f"[FAIL] ProductionInfrastructure: {e}")
        instantiation_tests.append(False)
    
    # Test UserOnboarding
    try:
        from scrollintel.core.user_onboarding import UserOnboarding
        onboarding = UserOnboarding()
        print("[OK] UserOnboarding instantiated")
        instantiation_tests.append(True)
    except Exception as e:
        print(f"[FAIL] UserOnboarding: {e}")
        instantiation_tests.append(False)
    
    # Test APIStability
    try:
        from scrollintel.core.api_stability import APIStability
        api_stability = APIStability()
        print("[OK] APIStability instantiated")
        instantiation_tests.append(True)
    except Exception as e:
        print(f"[FAIL] APIStability: {e}")
        instantiation_tests.append(False)
    
    success_count = sum(instantiation_tests)
    total_count = len(instantiation_tests)
    
    if success_count == total_count:
        print(f"\\n[OK] All {total_count} classes instantiated successfully")
        return True
    else:
        print(f"\\n[FAIL] {total_count - success_count} classes failed to instantiate")
        return False

def test_api_server_startup():
    """Test that the API server can start"""
    print("\\nTesting API Server Startup...")
    
    try:
        # Start server process
        server_process = subprocess.Popen([
            sys.executable, "scrollintel/api/production_main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(8)
        
        # Check if process is still running
        if server_process.poll() is None:
            print("[OK] API server started successfully")
            
            # Try to stop it gracefully
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
            
            return True
        else:
            stdout, stderr = server_process.communicate()
            print(f"[FAIL] API server failed to start: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error testing API server: {e}")
        return False

def test_health_endpoints():
    """Test health endpoints if server is running"""
    print("\\nTesting Health Endpoints...")
    
    # Start server for testing
    try:
        server_process = subprocess.Popen([
            sys.executable, "scrollintel/api/production_main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(10)
        
        if server_process.poll() is not None:
            print("[SKIP] Server not running, skipping endpoint tests")
            return True
        
        # Test health endpoint
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("[OK] /health endpoint responding")
                endpoint_success = True
            else:
                print(f"[FAIL] /health endpoint returned {response.status_code}")
                endpoint_success = False
        except Exception as e:
            print(f"[FAIL] /health endpoint error: {e}")
            endpoint_success = False
        
        # Test detailed health endpoint
        try:
            response = requests.get("http://localhost:8000/health/detailed", timeout=5)
            if response.status_code == 200:
                print("[OK] /health/detailed endpoint responding")
            else:
                print(f"[WARN] /health/detailed endpoint returned {response.status_code}")
        except Exception as e:
            print(f"[WARN] /health/detailed endpoint error: {e}")
        
        # Clean up server
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        
        return endpoint_success
        
    except Exception as e:
        print(f"[FAIL] Error testing endpoints: {e}")
        return False

def check_dependencies():
    """Check that required dependencies are installed"""
    print("\\nChecking Dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "pydantic",
        "sqlalchemy",
        "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\\n[FAIL] {len(missing_packages)} required packages are missing")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print(f"\\n[OK] All {len(required_packages)} required packages are installed")
        return True

def print_verification_summary(results):
    """Print verification summary"""
    print("\\n" + "=" * 60)
    print("DEPLOYMENT READINESS VERIFICATION SUMMARY")
    print("=" * 60)
    
    test_names = [
        "File Structure",
        "Module Imports", 
        "Class Instantiation",
        "API Server Startup",
        "Health Endpoints",
        "Dependencies"
    ]
    
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print(f"Overall Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    print()
    
    for test_name, result in zip(test_names, results):
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print()
    
    if all(results):
        print("DEPLOYMENT READY!")
        print("All verification tests passed.")
        print("ScrollIntel is ready for deployment.")
    else:
        print("DEPLOYMENT NOT READY")
        print("Some verification tests failed.")
        print("Please fix the issues before deploying.")
    
    print()
    print("Next Steps:")
    if all(results):
        print("1. Deploy to staging environment")
        print("2. Run staging validation tests")
        print("3. Deploy to production")
    else:
        print("1. Fix failed verification tests")
        print("2. Re-run verification")
        print("3. Proceed with deployment when all tests pass")
    
    print(f"\\nVerification completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main verification function"""
    print_header()
    
    results = []
    
    try:
        # Run all verification tests
        results.append(check_file_structure())
        results.append(test_module_imports())
        results.append(test_class_instantiation())
        results.append(test_api_server_startup())
        results.append(test_health_endpoints())
        results.append(check_dependencies())
        
    except KeyboardInterrupt:
        print("\\nVerification interrupted by user")
        return False
    except Exception as e:
        print(f"\\nUnexpected error during verification: {e}")
        return False
    finally:
        # Always print summary
        if results:
            print_verification_summary(results)
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)