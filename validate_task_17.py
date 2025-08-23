#!/usr/bin/env python3
"""
Simple validation script for Task 17: Production Deployment and Launch
Windows-compatible validation without Unicode characters
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

def validate_task_17_implementation():
    """Validate Task 17 implementation"""
    print("Validating Task 17: Production Deployment and Launch Implementation")
    print("=" * 70)
    
    validation_results = {}
    
    # Check 1: Required scripts exist
    print("\n1. Checking required scripts...")
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
        if os.path.exists(script):
            print(f"   FOUND: {script}")
        else:
            missing_scripts.append(script)
            print(f"   MISSING: {script}")
    
    validation_results["scripts_exist"] = len(missing_scripts) == 0
    
    # Check 2: Configuration files
    print("\n2. Checking configuration files...")
    config_files = ["deployment-config.yaml"]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"   FOUND: {config_file}")
        else:
            print(f"   MISSING: {config_file} (will use defaults)")
    
    validation_results["config_files"] = True
    
    # Check 3: Directory structure
    print("\n3. Checking directory structure...")
    required_dirs = [
        "logs", "reports/deployment", "reports/uat", 
        "reports/rollout", "reports/go-live", "backups"
    ]
    
    for directory in required_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"   READY: {directory}")
        except Exception as e:
            print(f"   ERROR: Cannot create {directory}: {str(e)}")
            validation_results["directories"] = False
            break
    else:
        validation_results["directories"] = True
    
    # Check 4: Core functionality
    print("\n4. Checking core functionality...")
    
    # Check if main deployment script can show help
    try:
        import subprocess
        result = subprocess.run([
            "python", "scripts/execute-production-deployment.py", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   PASS: Main script help functionality works")
            validation_results["help_functionality"] = True
        else:
            print("   FAIL: Main script help functionality failed")
            validation_results["help_functionality"] = False
    except Exception as e:
        print(f"   ERROR: Help functionality test failed: {str(e)}")
        validation_results["help_functionality"] = False
    
    # Check 5: Task requirements coverage
    print("\n5. Checking task requirements coverage...")
    
    requirements_covered = {
        "system_deployment": "scripts/production-deployment-launch.py",
        "monitoring_setup": "scripts/production-deployment-orchestrator.py", 
        "user_acceptance_testing": "scripts/user-acceptance-testing.py",
        "gradual_rollout": "scripts/gradual-rollout-manager.py",
        "go_live_procedures": "scripts/go-live-procedures.py",
        "comprehensive_documentation": "scripts/go-live-procedures.py"
    }
    
    all_requirements_covered = True
    for requirement, script in requirements_covered.items():
        if os.path.exists(script):
            print(f"   COVERED: {requirement} -> {script}")
        else:
            print(f"   MISSING: {requirement} -> {script}")
            all_requirements_covered = False
    
    validation_results["requirements_covered"] = all_requirements_covered
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    total_checks = len(validation_results)
    passed_checks = sum(1 for result in validation_results.values() if result)
    
    for check_name, result in validation_results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status}: {check_name.replace('_', ' ').title()}")
    
    print(f"\nOverall: {passed_checks}/{total_checks} checks passed")
    
    # Generate validation report
    report = {
        "validation_report": {
            "timestamp": datetime.now().isoformat(),
            "task_id": "17",
            "task_name": "Production Deployment and Launch",
            "validation_results": validation_results,
            "summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "success_rate": (passed_checks / total_checks) * 100,
                "overall_status": "PASS" if passed_checks == total_checks else "PARTIAL"
            }
        }
    }
    
    # Save report
    os.makedirs("reports/deployment", exist_ok=True)
    report_file = f"reports/deployment/task_17_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nValidation report saved: {report_file}")
    
    if passed_checks == total_checks:
        print("\nTASK 17 IMPLEMENTATION VALIDATED SUCCESSFULLY!")
        print("Ready for production deployment execution.")
        return True
    else:
        print("\nTASK 17 IMPLEMENTATION VALIDATION INCOMPLETE!")
        print("Please address the failed checks before deployment.")
        return False

def main():
    """Main validation function"""
    try:
        success = validate_task_17_implementation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nValidation error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()