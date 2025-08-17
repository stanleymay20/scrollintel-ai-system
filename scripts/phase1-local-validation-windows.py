#!/usr/bin/env python3
"""
Phase 1: Local Development Validation Script (Windows Compatible)
Validates ScrollIntel is ready for local development testing
"""

import os
import sys
import subprocess
import time
import requests
import json
from datetime import datetime

def print_phase_header():
    """Print phase 1 header"""
    print("ScrollIntel Launch - Phase 1: Local Development Validation")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

def check_prerequisites():
    """Check if prerequisites are met"""
    print("Checking Prerequisites...")
    
    checks = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"[OK] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        checks.append(True)
    else:
        print(f"[FAIL] Python version {python_version.major}.{python_version.minor} < 3.8")
        checks.append(False)
    
    # Check required files
    required_files = [
        "requirements.txt",
        "scrollintel/core/production_infrastructure.py",
        "scrollintel/core/user_onboarding.py",
        "scrollintel/core/api_stability.py",
        "scrollintel/api/production_main.py",
        "test_immediate_priority_direct.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] {file_path}")
            checks.append(True)
        else:
            print(f"[FAIL] {file_path}")
            checks.append(False)
    
    # Check .env file
    if os.path.exists(".env"):
        print("[OK] .env file exists")
        checks.append(True)
    else:
        print("[FAIL] .env file missing")
        checks.append(False)
    
    # Calculate success rate
    passed = sum(checks)
    total = len(checks)
    success_rate = (passed / total) * 100
    
    print(f"Prerequisites Check: {passed}/{total} ({success_rate:.1f}%)")
    print()
    
    return all(checks)

def install_dependencies():
    """Install required dependencies"""
    print("Installing Dependencies...")
    
    try:
        # Install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("[OK] Dependencies installed successfully")
            return True
        else:
            print(f"[FAIL] Failed to install dependencies: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[FAIL] Dependency installation timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Error installing dependencies: {e}")
        return False

def run_core_tests():
    """Run core validation tests"""
    print("Running Core Tests...")
    
    test_results = []
    
    # Test 1: Direct priority test
    print("Running test_immediate_priority_direct_windows.py...")
    try:
        result = subprocess.run([
            sys.executable, "test_immediate_priority_direct_windows.py"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("[OK] test_immediate_priority_direct_windows.py - PASSED")
            test_results.append(True)
        else:
            print(f"[FAIL] test_immediate_priority_direct_windows.py - FAILED: {result.stderr}")
            test_results.append(False)
            
    except subprocess.TimeoutExpired:
        print("[FAIL] test_immediate_priority_direct_windows.py - TIMEOUT")
        test_results.append(False)
    except Exception as e:
        print(f"[FAIL] test_immediate_priority_direct_windows.py - ERROR: {e}")
        test_results.append(False)
    
    # Test 2: Deployment verification
    if os.path.exists("verify_immediate_priority_deployment_windows.py"):
        print("Running verify_immediate_priority_deployment_windows.py...")
        try:
            result = subprocess.run([
                sys.executable, "verify_immediate_priority_deployment_windows.py"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("[OK] verify_immediate_priority_deployment_windows.py - PASSED")
                test_results.append(True)
            else:
                print(f"[FAIL] verify_immediate_priority_deployment_windows.py - FAILED: {result.stderr}")
                test_results.append(False)
                
        except subprocess.TimeoutExpired:
            print("[FAIL] verify_immediate_priority_deployment_windows.py - TIMEOUT")
            test_results.append(False)
        except Exception as e:
            print(f"[FAIL] verify_immediate_priority_deployment_windows.py - ERROR: {e}")
            test_results.append(False)
    
    # Calculate test results
    passed = sum(test_results)
    total = len(test_results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"Test Results: {passed}/{total} ({success_rate:.1f}%)")
    print()
    
    return all(test_results)

def start_development_server():
    """Start development server"""
    print("Starting Development Server...")
    
    try:
        # Start server in background
        server_process = subprocess.Popen([
            sys.executable, "scrollintel/api/production_main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("Waiting for server to start...")
        time.sleep(10)
        
        # Check if server is still running
        if server_process.poll() is None:
            print("[OK] Development server started successfully")
            return server_process
        else:
            stdout, stderr = server_process.communicate()
            print(f"[FAIL] Development server failed to start: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"[FAIL] Error starting development server: {e}")
        return None

def validate_endpoints(base_url="http://localhost:8000"):
    """Validate server endpoints"""
    print(f"Validating Endpoints ({base_url})...")
    
    endpoints = [
        ("/health", "Basic health check"),
        ("/health/detailed", "Detailed system status")
    ]
    
    results = []
    
    for endpoint, description in endpoints:
        url = f"{base_url}{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"[OK] {endpoint} - {description}")
                results.append(True)
            else:
                print(f"[FAIL] {endpoint} - HTTP {response.status_code}")
                results.append(False)
                
        except requests.exceptions.ConnectionError:
            print(f"[FAIL] {endpoint} - Connection failed")
            results.append(False)
        except requests.exceptions.Timeout:
            print(f"[FAIL] {endpoint} - Timeout")
            results.append(False)
        except Exception as e:
            print(f"[FAIL] {endpoint} - Error: {e}")
            results.append(False)
    
    success_rate = sum(results) / len(results) * 100
    print(f"Endpoint Validation: {sum(results)}/{len(results)} ({success_rate:.1f}%)")
    print()
    
    return all(results)

def test_basic_functionality(base_url="http://localhost:8000"):
    """Test basic functionality"""
    print(f"Testing Basic Functionality ({base_url})...")
    
    tests = []
    
    # Test 1: Health endpoint returns JSON
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200 and response.headers.get('content-type', '').startswith('application/json'):
            print("[OK] Health endpoint returns proper JSON")
            tests.append(True)
        else:
            print("[FAIL] Health endpoint does not return proper JSON")
            tests.append(False)
    except Exception as e:
        print(f"[FAIL] Health endpoint test failed: {e}")
        tests.append(False)
    
    # Test 2: Detailed health endpoint returns system info
    try:
        response = requests.get(f"{base_url}/health/detailed", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "infrastructure" in data or "api_stability" in data:
                print("[OK] Detailed health endpoint returns system info")
                tests.append(True)
            else:
                print("[FAIL] Detailed health endpoint missing system info")
                tests.append(False)
        else:
            print("[FAIL] Detailed health endpoint failed")
            tests.append(False)
    except Exception as e:
        print(f"[FAIL] Detailed health endpoint test failed: {e}")
        tests.append(False)
    
    success_rate = sum(tests) / len(tests) * 100
    print(f"Functionality Tests: {sum(tests)}/{len(tests)} ({success_rate:.1f}%)")
    print()
    
    return all(tests)

def cleanup_server(server_process):
    """Clean up server process"""
    print("Cleaning up server process...")
    
    if server_process and server_process.poll() is None:
        try:
            server_process.terminate()
            server_process.wait(timeout=10)
            print("[OK] Server stopped gracefully")
        except subprocess.TimeoutExpired:
            server_process.kill()
            print("[OK] Server force stopped")
        except Exception as e:
            print(f"[WARN] Error stopping server: {e}")
    else:
        print("[OK] Server already stopped")

def print_completion_report(results):
    """Print phase completion report"""
    print("=" * 70)
    print("PHASE 1 COMPLETION REPORT")
    print("=" * 70)
    
    phase_names = [
        "Prerequisites Check",
        "Dependency Installation", 
        "Core Tests",
        "Server Startup",
        "Endpoint Validation",
        "Functionality Tests"
    ]
    
    passed_count = sum(results)
    total_count = len(results)
    overall_success = (passed_count / total_count) * 100
    
    print(f"Overall Success Rate: {passed_count}/{total_count} ({overall_success:.1f}%)")
    
    for i, (phase_name, result) in enumerate(zip(phase_names, results)):
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} - {phase_name}")
    
    if all(results):
        print("PHASE 1 COMPLETE - Ready for Phase 2 (Staging)")
        print("Next Steps:")
        print("1. Set up staging environment")
        print("2. Configure staging deployment")
        print("3. Run Phase 2 validation script")
    else:
        print("PHASE 1 INCOMPLETE - Issues need to be resolved")
        print("Please fix the failed components before proceeding")
    
    print()
    print("What's Next?")
    print("=" * 30)
    if all(results):
        print("If Phase 1 was successful:")
        print("• Use the Launch Coordinator for Phase 2+")
        print("• Command: python scripts/launch-coordinator.py")
        print("Launch Coordinator Options:")
        print("• Interactive menu for phase management")
        print("• Run individual phases: python scripts/launch-coordinator.py phase2")
        print("• Check status: python scripts/launch-coordinator.py status")
        print("• Run all phases: python scripts/launch-coordinator.py all")
    else:
        print("If Phase 1 failed:")
        print("• Review error messages and fix issues")
        print("• Re-run this script: python scripts/phase1-local-validation-windows.py")
        print("• Check the troubleshooting guide in the launch plan")

def main():
    """Main execution function"""
    print_phase_header()
    
    results = []
    server_process = None
    
    try:
        # Step 1: Check prerequisites
        results.append(check_prerequisites())
        
        # Step 2: Install dependencies
        results.append(install_dependencies())
        
        # Step 3: Run core tests
        results.append(run_core_tests())
        
        # Step 4: Start development server
        server_process = start_development_server()
        results.append(server_process is not None)
        
        # Step 5: Validate endpoints (only if server started)
        if server_process:
            results.append(validate_endpoints())
            
            # Step 6: Test basic functionality
            results.append(test_basic_functionality())
        else:
            results.extend([False, False])  # Skip endpoint and functionality tests
        
    except KeyboardInterrupt:
        print("\\nOperation cancelled by user")
        results.extend([False] * (6 - len(results)))  # Fill remaining with False
    except Exception as e:
        print(f"\\nUnexpected error: {e}")
        results.extend([False] * (6 - len(results)))  # Fill remaining with False
    finally:
        # Always cleanup server
        if server_process:
            cleanup_server(server_process)
    
    # Print completion report
    print_completion_report(results)
    
    # Return overall success
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)