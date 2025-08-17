#!/usr/bin/env python3
"""
Phase 1: Local Development Validation Script
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
    try:
        print("🚀 ScrollIntel Launch - Phase 1: Local Development Validation")
    except UnicodeEncodeError:
        print("ScrollIntel Launch - Phase 1: Local Development Validation")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

def check_prerequisites():
    """Check if prerequisites are met"""
    print("📋 Checking Prerequisites...")
    
    checks = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        checks.append(True)
    else:
        print(f"❌ Python version {python_version.major}.{python_version.minor} < 3.8")
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
            print(f"✅ {file_path}")
            checks.append(True)
        else:
            print(f"❌ {file_path} - Missing")
            checks.append(False)
    
    # Check environment file
    if os.path.exists(".env"):
        print("✅ .env file exists")
        checks.append(True)
    elif os.path.exists(".env.example"):
        print("⚠️  .env.example exists but .env missing - will create")
        try:
            subprocess.run(["cp", ".env.example", ".env"], check=True)
            print("✅ Created .env from .env.example")
            checks.append(True)
        except:
            print("❌ Failed to create .env file")
            checks.append(False)
    else:
        print("❌ No .env or .env.example file found")
        checks.append(False)
    
    success_rate = sum(checks) / len(checks) * 100
    print(f"\n📊 Prerequisites Check: {sum(checks)}/{len(checks)} ({success_rate:.1f}%)")
    
    return all(checks)

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Dependencies...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Dependency installation timed out")
        return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def run_core_tests():
    """Run core functionality tests"""
    print("\n🧪 Running Core Tests...")
    
    test_files = [
        "test_immediate_priority_direct.py",
        "verify_immediate_priority_deployment.py"
    ]
    
    test_results = []
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n🔍 Running {test_file}...")
            try:
                result = subprocess.run([
                    sys.executable, test_file
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    print(f"✅ {test_file} - PASSED")
                    test_results.append(True)
                else:
                    print(f"❌ {test_file} - FAILED")
                    print(f"Error: {result.stderr}")
                    test_results.append(False)
                    
            except subprocess.TimeoutExpired:
                print(f"❌ {test_file} - TIMEOUT")
                test_results.append(False)
            except Exception as e:
                print(f"❌ {test_file} - ERROR: {e}")
                test_results.append(False)
        else:
            print(f"❌ {test_file} - NOT FOUND")
            test_results.append(False)
    
    success_rate = sum(test_results) / len(test_results) * 100
    print(f"\n📊 Test Results: {sum(test_results)}/{len(test_results)} ({success_rate:.1f}%)")
    
    return all(test_results)

def start_development_server():
    """Start the development server"""
    print("\n🖥️  Starting Development Server...")
    
    try:
        # Start server in background
        server_process = subprocess.Popen([
            sys.executable, "scrollintel/api/production_main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(10)
        
        # Check if server is still running
        if server_process.poll() is None:
            print("✅ Development server started successfully")
            return server_process
        else:
            stdout, stderr = server_process.communicate()
            print(f"❌ Server failed to start: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def validate_endpoints(base_url="http://localhost:8000"):
    """Validate core endpoints"""
    print(f"\n🔍 Validating Endpoints ({base_url})...")
    
    endpoints = [
        ("/health", "Basic health check"),
        ("/health/detailed", "Detailed system status"),
        ("/docs", "API documentation (if available)")
    ]
    
    results = []
    
    for endpoint, description in endpoints:
        url = f"{base_url}{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ {endpoint} - {description}")
                results.append(True)
            elif response.status_code == 404 and endpoint == "/docs":
                print(f"⚠️  {endpoint} - Not available (expected in production mode)")
                results.append(True)  # This is acceptable
            else:
                print(f"❌ {endpoint} - HTTP {response.status_code}")
                results.append(False)
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {endpoint} - Connection failed")
            results.append(False)
        except requests.exceptions.Timeout:
            print(f"❌ {endpoint} - Timeout")
            results.append(False)
        except Exception as e:
            print(f"❌ {endpoint} - Error: {e}")
            results.append(False)
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Endpoint Validation: {sum(results)}/{len(results)} ({success_rate:.1f}%)")
    
    return all(results)

def test_basic_functionality(base_url="http://localhost:8000"):
    """Test basic API functionality"""
    print(f"\n⚙️  Testing Basic Functionality ({base_url})...")
    
    tests = []
    
    # Test health endpoint response format
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "status" in data:
                print("✅ Health endpoint returns proper JSON")
                tests.append(True)
            else:
                print("❌ Health endpoint missing 'status' field")
                tests.append(False)
        else:
            print(f"❌ Health endpoint returned {response.status_code}")
            tests.append(False)
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        tests.append(False)
    
    # Test detailed health endpoint
    try:
        response = requests.get(f"{base_url}/health/detailed", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "infrastructure" in data or "api_stability" in data:
                print("✅ Detailed health endpoint returns system info")
                tests.append(True)
            else:
                print("❌ Detailed health endpoint missing system info")
                tests.append(False)
        else:
            print(f"❌ Detailed health endpoint returned {response.status_code}")
            tests.append(False)
    except Exception as e:
        print(f"❌ Detailed health endpoint test failed: {e}")
        tests.append(False)
    
    success_rate = sum(tests) / len(tests) * 100
    print(f"\n📊 Functionality Tests: {sum(tests)}/{len(tests)} ({success_rate:.1f}%)")
    
    return all(tests)

def cleanup_server(server_process):
    """Clean up server process"""
    if server_process and server_process.poll() is None:
        print("\n🧹 Cleaning up server process...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
            print("✅ Server stopped gracefully")
        except subprocess.TimeoutExpired:
            server_process.kill()
            print("⚠️  Server force-killed")

def generate_phase1_report(results):
    """Generate Phase 1 completion report"""
    print("\n" + "=" * 70)
    print("📊 PHASE 1 COMPLETION REPORT")
    print("=" * 70)
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    success_rate = (passed_checks / total_checks) * 100
    
    print(f"Overall Success Rate: {passed_checks}/{total_checks} ({success_rate:.1f}%)")
    print()
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check_name}")
    
    print()
    
    if success_rate >= 90:
        print("🎉 PHASE 1 COMPLETE - Ready for Phase 2 (Staging)")
        print("Next Steps:")
        print("1. Set up staging environment")
        print("2. Configure staging deployment")
        print("3. Run Phase 2 validation script")
        return True
    elif success_rate >= 70:
        print("⚠️  PHASE 1 PARTIAL - Some issues need attention")
        print("Recommendations:")
        print("1. Fix failing checks")
        print("2. Re-run Phase 1 validation")
        print("3. Proceed to Phase 2 when all checks pass")
        return False
    else:
        print("❌ PHASE 1 FAILED - Critical issues must be resolved")
        print("Required Actions:")
        print("1. Address all failing checks")
        print("2. Review system configuration")
        print("3. Re-run Phase 1 validation")
        return False

def main():
    """Main Phase 1 validation function"""
    print_phase_header()
    
    results = {}
    server_process = None
    
    try:
        # Step 1: Prerequisites
        results["Prerequisites Check"] = check_prerequisites()
        
        if not results["Prerequisites Check"]:
            print("\n❌ Prerequisites not met. Please fix issues and try again.")
            return False
        
        # Step 2: Dependencies
        results["Dependency Installation"] = install_dependencies()
        
        # Step 3: Core Tests
        results["Core Tests"] = run_core_tests()
        
        # Step 4: Server Startup
        server_process = start_development_server()
        results["Server Startup"] = server_process is not None
        
        if server_process:
            # Step 5: Endpoint Validation
            results["Endpoint Validation"] = validate_endpoints()
            
            # Step 6: Functionality Tests
            results["Functionality Tests"] = test_basic_functionality()
        else:
            results["Endpoint Validation"] = False
            results["Functionality Tests"] = False
        
        # Generate report
        phase1_success = generate_phase1_report(results)
        
        return phase1_success
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Phase 1 validation interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n❌ Phase 1 validation failed with error: {e}")
        return False
    finally:
        # Always cleanup
        cleanup_server(server_process)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)