#!/usr/bin/env python3
"""
Test script for the enhanced deployment status checker
"""

import sys
import os
import subprocess
import time

def test_deployment_checker():
    """Test the deployment checker functionality"""
    print("🧪 Testing Enhanced Deployment Status Checker")
    print("=" * 50)
    
    # Test basic functionality
    print("\n1. Testing basic check...")
    try:
        result = subprocess.run([sys.executable, "check_deployment_status.py"], 
                              capture_output=True, text=True, timeout=30)
        print(f"✅ Basic check completed (exit code: {result.returncode})")
        if result.stdout:
            print("📋 Sample output:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
    except subprocess.TimeoutExpired:
        print("⏰ Basic check timed out")
    except Exception as e:
        print(f"❌ Basic check failed: {e}")
    
    # Test verbose mode
    print("\n2. Testing verbose mode...")
    try:
        result = subprocess.run([sys.executable, "check_deployment_status.py", "--verbose"], 
                              capture_output=True, text=True, timeout=30)
        print(f"✅ Verbose check completed (exit code: {result.returncode})")
    except Exception as e:
        print(f"❌ Verbose check failed: {e}")
    
    # Test JSON output
    print("\n3. Testing JSON output...")
    try:
        result = subprocess.run([sys.executable, "check_deployment_status.py", "--json"], 
                              capture_output=True, text=True, timeout=30)
        print(f"✅ JSON output check completed (exit code: {result.returncode})")
        if result.stdout:
            import json
            try:
                data = json.loads(result.stdout)
                print(f"📊 JSON contains {len(data.get('results', []))} service checks")
            except json.JSONDecodeError:
                print("⚠️  JSON output may be malformed")
    except Exception as e:
        print(f"❌ JSON output check failed: {e}")
    
    # Test save report
    print("\n4. Testing save report...")
    try:
        result = subprocess.run([sys.executable, "check_deployment_status.py", "--save-report"], 
                              capture_output=True, text=True, timeout=30)
        print(f"✅ Save report check completed (exit code: {result.returncode})")
        
        # Check if report file was created
        import glob
        report_files = glob.glob("deployment_status_*.json")
        if report_files:
            print(f"📄 Report file created: {report_files[-1]}")
            # Clean up
            for file in report_files:
                try:
                    os.remove(file)
                    print(f"🧹 Cleaned up: {file}")
                except:
                    pass
    except Exception as e:
        print(f"❌ Save report check failed: {e}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_deployment_checker()