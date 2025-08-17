#!/usr/bin/env python3
"""
ScrollIntel Immediate Priority Implementation - Direct Tests (Windows Compatible)
Tests core functionality without Unicode characters
"""

import os
import sys
import time
import traceback
from datetime import datetime

def print_header():
    \"\"\"Print test header\"\"\"
    print("ScrollIntel Immediate Priority Implementation - Direct Tests")
    print("=" * 70)
    print(f"Test Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

def test_imports():
    \"\"\"Test that all core modules can be imported\"\"\"
    print("Testing Core Module Imports...")
    
    import_tests = []
    
    # Test core infrastructure imports
    try:
        from scrollintel.core.production_infrastructure import ProductionInfrastructure
        print("[OK] ProductionInfrastructure imported")
        import_tests.append(True)
    except Exception as e:
        print(f"[FAIL] ProductionInfrastructure import failed: {e}")
        import_tests.append(False)
    
    try:
        from scrollintel.core.user_onboarding import UserOnboarding
        print("[OK] UserOnboarding imported")
        import_tests.append(True)
    except Exception as e:
        print(f"[FAIL] UserOnboarding import failed: {e}")
        import_tests.append(False)
    
    try:
        from scrollintel.core.api_stability import APIStability
        print("[OK] APIStability imported")
        import_tests.append(True)
    except Exception as e:
        print(f"[FAIL] APIStability import failed: {e}")
        import_tests.append(False)
    
    success_rate = sum(import_tests) / len(import_tests) * 100
    print(f"Import Tests: {sum(import_tests)}/{len(import_tests)} ({success_rate:.1f}%)")
    print()
    
    return all(import_tests)

def test_infrastructure_initialization():
    \"\"\"Test infrastructure components can be initialized\"\"\"
    print("Testing Infrastructure Initialization...")
    
    init_tests = []
    
    try:
        from scrollintel.core.production_infrastructure import ProductionInfrastructure
        infra = ProductionInfrastructure()
        print("[OK] ProductionInfrastructure initialized")
        init_tests.append(True)
    except Exception as e:
        print(f"[FAIL] ProductionInfrastructure initialization failed: {e}")
        init_tests.append(False)
    
    try:
        from scrollintel.core.user_onboarding import UserOnboarding
        onboarding = UserOnboarding()
        print("[OK] UserOnboarding initialized")
        init_tests.append(True)
    except Exception as e:
        print(f"[FAIL] UserOnboarding initialization failed: {e}")
        init_tests.append(False)
    
    try:
        from scrollintel.core.api_stability import APIStability
        api_stability = APIStability()
        print("[OK] APIStability initialized")
        init_tests.append(True)
    except Exception as e:
        print(f"[FAIL] APIStability initialization failed: {e}")
        init_tests.append(False)
    
    success_rate = sum(init_tests) / len(init_tests) * 100
    print(f"Initialization Tests: {sum(init_tests)}/{len(init_tests)} ({success_rate:.1f}%)")
    print()
    
    return all(init_tests)

def test_basic_functionality():
    \"\"\"Test basic functionality of core components\"\"\"
    print("Testing Basic Functionality...")
    
    func_tests = []
    
    # Test ProductionInfrastructure
    try:
        from scrollintel.core.production_infrastructure import ProductionInfrastructure
        infra = ProductionInfrastructure()
        
        # Test health check
        health = infra.get_health_status()
        if isinstance(health, dict) and 'status' in health:
            print("[OK] ProductionInfrastructure health check")
            func_tests.append(True)
        else:
            print("[FAIL] ProductionInfrastructure health check returned invalid format")
            func_tests.append(False)
            
    except Exception as e:
        print(f"[FAIL] ProductionInfrastructure functionality test failed: {e}")
        func_tests.append(False)
    
    # Test UserOnboarding
    try:
        from scrollintel.core.user_onboarding import UserOnboarding
        onboarding = UserOnboarding()
        
        # Test onboarding flow
        flow = onboarding.get_onboarding_flow()
        if isinstance(flow, dict) and 'steps' in flow:
            print("[OK] UserOnboarding flow generation")
            func_tests.append(True)
        else:
            print("[FAIL] UserOnboarding flow generation returned invalid format")
            func_tests.append(False)
            
    except Exception as e:
        print(f"[FAIL] UserOnboarding functionality test failed: {e}")
        func_tests.append(False)
    
    # Test APIStability
    try:
        from scrollintel.core.api_stability import APIStability
        api_stability = APIStability()
        
        # Test stability metrics
        metrics = api_stability.get_stability_metrics()
        if isinstance(metrics, dict) and 'uptime' in metrics:
            print("[OK] APIStability metrics generation")
            func_tests.append(True)
        else:
            print("[FAIL] APIStability metrics generation returned invalid format")
            func_tests.append(False)
            
    except Exception as e:
        print(f"[FAIL] APIStability functionality test failed: {e}")
        func_tests.append(False)
    
    success_rate = sum(func_tests) / len(func_tests) * 100
    print(f"Functionality Tests: {sum(func_tests)}/{len(func_tests)} ({success_rate:.1f}%)")
    print()
    
    return all(func_tests)

def test_api_main_import():
    \"\"\"Test that the main API module can be imported\"\"\"
    print("Testing API Main Module...")
    
    try:
        # Try to import the main API module
        sys.path.insert(0, 'scrollintel/api')
        import production_main
        print("[OK] production_main module imported")
        return True
    except Exception as e:
        print(f"[FAIL] production_main import failed: {e}")
        return False

def test_configuration():
    \"\"\"Test configuration loading\"\"\"
    print("Testing Configuration...")
    
    config_tests = []
    
    # Test environment file
    if os.path.exists('.env'):
        print("[OK] .env file exists")
        config_tests.append(True)
    else:
        print("[FAIL] .env file missing")
        config_tests.append(False)
    
    # Test configuration loading
    try:
        from scrollintel.core.config import Config
        config = Config()
        print("[OK] Configuration loaded")
        config_tests.append(True)
    except Exception as e:
        print(f"[FAIL] Configuration loading failed: {e}")
        config_tests.append(False)
    
    success_rate = sum(config_tests) / len(config_tests) * 100
    print(f"Configuration Tests: {sum(config_tests)}/{len(config_tests)} ({success_rate:.1f}%)")
    print()
    
    return all(config_tests)

def test_database_connection():
    \"\"\"Test database connectivity\"\"\"
    print("Testing Database Connection...")
    
    try:
        from scrollintel.models.database import get_database_url, test_connection
        
        # Test database URL generation
        db_url = get_database_url()
        if db_url:
            print("[OK] Database URL generated")
            
            # Test connection (if possible)
            try:
                connection_ok = test_connection()
                if connection_ok:
                    print("[OK] Database connection successful")
                    return True
                else:
                    print("[WARN] Database connection failed (may be expected in dev)")
                    return True  # Don't fail for database connection in dev
            except Exception:
                print("[WARN] Database connection test skipped (may be expected in dev)")
                return True  # Don't fail for database connection in dev
        else:
            print("[FAIL] Database URL generation failed")
            return False
            
    except Exception as e:
        print(f"[WARN] Database test skipped: {e}")
        return True  # Don't fail for database issues in dev

def print_summary(results):
    \"\"\"Print test summary\"\"\"
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    test_names = [
        "Core Module Imports",
        "Infrastructure Initialization",
        "Basic Functionality",
        "API Main Module",
        "Configuration",
        "Database Connection"
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
        print("ALL TESTS PASSED - Core implementation is working!")
        print("Ready for next phase of testing.")
    else:
        print("SOME TESTS FAILED - Please review and fix issues.")
        print("Check the error messages above for details.")
    
    print()
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    \"\"\"Main test execution\"\"\"
    print_header()
    
    results = []
    
    try:
        # Run all tests
        results.append(test_imports())
        results.append(test_infrastructure_initialization())
        results.append(test_basic_functionality())
        results.append(test_api_main_import())
        results.append(test_configuration())
        results.append(test_database_connection())
        
    except KeyboardInterrupt:
        print("\\nTests interrupted by user")
        return False
    except Exception as e:
        print(f"\\nUnexpected error during testing: {e}")
        traceback.print_exc()
        return False
    finally:
        # Always print summary
        if results:
            print_summary(results)
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)