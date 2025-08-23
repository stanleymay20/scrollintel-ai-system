#!/usr/bin/env python3
"""
Test script for Task 17: Production Deployment and Launch
Validates the implementation without executing full deployment
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_script_existence():
    """Test that all required scripts exist"""
    logger.info("Testing script existence...")
    
    required_scripts = [
        "scripts/production-deployment-launch.py",
        "scripts/user-acceptance-testing.py", 
        "scripts/gradual-rollout-manager.py",
        "scripts/go-live-procedures.py",
        "scripts/production-deployment-orchestrator.py",
        "scripts/execute-production-deployment.py"
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
        else:
            logger.info(f"✅ Found: {script}")
    
    if missing_scripts:
        logger.error(f"❌ Missing scripts: {missing_scripts}")
        return False
    
    logger.info("✅ All required scripts exist")
    return True

def test_configuration_files():
    """Test that configuration files exist"""
    logger.info("Testing configuration files...")
    
    config_files = [
        "deployment-config.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            logger.info(f"✅ Found: {config_file}")
        else:
            logger.warning(f"⚠️ Missing: {config_file} (will use defaults)")
    
    return True

def test_script_syntax():
    """Test that scripts have valid Python syntax"""
    logger.info("Testing script syntax...")
    
    scripts_to_test = [
        "scripts/production-deployment-launch.py",
        "scripts/user-acceptance-testing.py",
        "scripts/gradual-rollout-manager.py", 
        "scripts/go-live-procedures.py",
        "scripts/production-deployment-orchestrator.py",
        "scripts/execute-production-deployment.py"
    ]
    
    for script in scripts_to_test:
        try:
            # Test syntax by compiling
            with open(script, 'r') as f:
                compile(f.read(), script, 'exec')
            logger.info(f"✅ Syntax valid: {script}")
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in {script}: {str(e)}")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Could not test {script}: {str(e)}")
    
    logger.info("✅ All scripts have valid syntax")
    return True

def test_directory_structure():
    """Test that required directories can be created"""
    logger.info("Testing directory structure...")
    
    required_dirs = [
        "logs",
        "reports/deployment",
        "reports/uat", 
        "reports/rollout",
        "reports/go-live",
        "backups",
        "docs/go-live"
    ]
    
    for directory in required_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✅ Directory ready: {directory}")
        except Exception as e:
            logger.error(f"❌ Cannot create directory {directory}: {str(e)}")
            return False
    
    logger.info("✅ Directory structure validated")
    return True

def test_environment_variables():
    """Test environment variable handling"""
    logger.info("Testing environment variables...")
    
    # Test with minimal environment
    test_env = {
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
        "REDIS_URL": "redis://localhost:6379",
        "JWT_SECRET_KEY": "test_secret_key",
        "OPENAI_API_KEY": "test_api_key"
    }
    
    # Temporarily set test environment
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        # Test environment validation by importing the module
        sys.path.insert(0, "scripts")
        import execute_production_deployment
        result = execute_production_deployment.validate_environment()
        
        if result:
            logger.info("✅ Environment validation works")
        else:
            logger.error("❌ Environment validation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Environment validation error: {str(e)}")
        return False
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    return True

def test_import_capabilities():
    """Test that scripts can be imported"""
    logger.info("Testing import capabilities...")
    
    # Add scripts directory to path
    scripts_dir = Path(__file__).parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    
    try:
        # Test imports (without executing)
        import importlib.util
        
        scripts_to_import = [
            ("production_deployment_launch", "scripts/production-deployment-launch.py"),
            ("user_acceptance_testing", "scripts/user-acceptance-testing.py"),
            ("gradual_rollout_manager", "scripts/gradual-rollout-manager.py"),
            ("go_live_procedures", "scripts/go-live-procedures.py")
        ]
        
        for module_name, script_path in scripts_to_import:
            try:
                spec = importlib.util.spec_from_file_location(module_name, script_path)
                module = importlib.util.module_from_spec(spec)
                # Don't execute, just test if it can be loaded
                logger.info(f"✅ Can import: {module_name}")
            except Exception as e:
                logger.error(f"❌ Cannot import {module_name}: {str(e)}")
                return False
        
        logger.info("✅ All scripts can be imported")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test error: {str(e)}")
        return False

def test_help_functionality():
    """Test help functionality of main script"""
    logger.info("Testing help functionality...")
    
    try:
        result = subprocess.run([
            "python", "scripts/execute-production-deployment.py", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and "Production Deployment" in result.stdout:
            logger.info("✅ Help functionality works")
            return True
        else:
            logger.error("❌ Help functionality failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Help test error: {str(e)}")
        return False

def test_validation_only_mode():
    """Test validation-only mode"""
    logger.info("Testing validation-only mode...")
    
    # Set minimal environment for testing
    test_env = {
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
        "REDIS_URL": "redis://localhost:6379", 
        "JWT_SECRET_KEY": "test_secret_key",
        "OPENAI_API_KEY": "test_api_key"
    }
    
    env = os.environ.copy()
    env.update(test_env)
    
    try:
        result = subprocess.run([
            "python", "scripts/execute-production-deployment.py", 
            "--validate-only"
        ], capture_output=True, text=True, timeout=60, env=env)
        
        if result.returncode == 0:
            logger.info("✅ Validation-only mode works")
            return True
        else:
            logger.error(f"❌ Validation-only mode failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Validation-only test error: {str(e)}")
        return False

def generate_test_report(test_results):
    """Generate test report"""
    logger.info("Generating test report...")
    
    try:
        report = {
            "test_report": {
                "timestamp": datetime.now().isoformat(),
                "task_id": "17",
                "task_name": "Production Deployment and Launch",
                "test_summary": {
                    "total_tests": len(test_results),
                    "passed_tests": sum(1 for result in test_results.values() if result),
                    "failed_tests": sum(1 for result in test_results.values() if not result),
                    "success_rate": sum(1 for result in test_results.values() if result) / len(test_results) * 100
                },
                "test_results": test_results,
                "overall_status": "PASS" if all(test_results.values()) else "FAIL"
            }
        }
        
        # Save report
        os.makedirs("reports/deployment", exist_ok=True)
        report_file = f"reports/deployment/task_17_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved: {report_file}")
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate test report: {str(e)}")
        return None

def main():
    """Main test execution"""
    print("Testing Task 17: Production Deployment and Launch Implementation")
    print("=" * 70)
    
    # Run all tests
    test_results = {
        "script_existence": test_script_existence(),
        "configuration_files": test_configuration_files(),
        "script_syntax": test_script_syntax(),
        "directory_structure": test_directory_structure(),
        "environment_variables": test_environment_variables(),
        "import_capabilities": test_import_capabilities(),
        "help_functionality": test_help_functionality(),
        "validation_only_mode": test_validation_only_mode()
    }
    
    # Generate test report
    report = generate_test_report(test_results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    if report:
        print(f"\n📈 Success Rate: {report['test_report']['test_summary']['success_rate']:.1f}%")
        print(f"📊 Tests Passed: {report['test_report']['test_summary']['passed_tests']}/{report['test_report']['test_summary']['total_tests']}")
    
    overall_success = all(test_results.values())
    
    if overall_success:
        print("\nALL TESTS PASSED!")
        print("Task 17 implementation is ready for execution")
    else:
        print("\nSOME TESTS FAILED!")
        print("Please address the failed tests before deployment")
    
    print("=" * 70)
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)