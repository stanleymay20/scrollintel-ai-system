#!/usr/bin/env python3
"""
ScrollIntel Launch Starter
Quick start script to begin the step-by-step launch process
"""

import os
import sys
import subprocess
from datetime import datetime

def print_welcome():
    """Print welcome message"""
    print("🚀 Welcome to ScrollIntel Step-by-Step Launch!")
    print("=" * 60)
    print("This script will guide you through launching ScrollIntel")
    print("in a controlled, phase-by-phase approach.")
    print("=" * 60)
    print()

def check_launch_readiness():
    """Check if we're ready to start the launch"""
    print("🔍 Checking Launch Readiness...")
    
    # Check if we have the immediate priority implementation
    required_files = [
        "scrollintel/core/production_infrastructure.py",
        "scrollintel/core/user_onboarding.py", 
        "scrollintel/core/api_stability.py",
        "scrollintel/api/production_main.py",
        "scripts/phase1-local-validation.py",
        "scripts/launch-coordinator.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure the immediate priority implementation is complete.")
        return False
    
    print("✅ All required files present")
    
    # Check if Python version is adequate
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} < 3.8 required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    return True

def show_launch_phases():
    """Show the launch phases"""
    print("\n📋 Launch Phases Overview:")
    print()
    
    phases = [
        ("Phase 1", "Local Development Validation", "1 day", "Validate all systems work locally"),
        ("Phase 2", "Staging Environment Setup", "2 days", "Deploy and test in staging environment"),
        ("Phase 3", "Limited Production Beta", "4 days", "Production deployment with limited access"),
        ("Phase 4", "Soft Launch", "7 days", "Open to broader user base"),
        ("Phase 5", "Full Production Launch", "Ongoing", "Full public availability")
    ]
    
    for phase, name, duration, description in phases:
        print(f"🎯 {phase}: {name}")
        print(f"   Duration: {duration}")
        print(f"   Goal: {description}")
        print()

def get_user_confirmation():
    """Get user confirmation to proceed"""
    print("🤔 Ready to start the launch process?")
    print()
    print("This will:")
    print("• Run comprehensive validation tests")
    print("• Start the development server")
    print("• Validate all core functionality")
    print("• Generate a Phase 1 completion report")
    print()
    
    while True:
        response = input("Start Phase 1 now? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")

def start_phase1():
    """Start Phase 1 validation"""
    print("\n🚀 Starting Phase 1: Local Development Validation")
    print("=" * 60)
    
    try:
        # Run Phase 1 validation script
        result = subprocess.run([
            sys.executable, "scripts/phase1-local-validation.py"
        ], timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            print("\n🎉 Phase 1 completed successfully!")
            print("\nNext Steps:")
            print("1. Review the Phase 1 report above")
            print("2. If all checks passed, proceed to Phase 2")
            print("3. Run: python scripts/launch-coordinator.py")
            return True
        else:
            print("\n❌ Phase 1 failed!")
            print("\nTroubleshooting:")
            print("1. Review the error messages above")
            print("2. Fix any identified issues")
            print("3. Re-run: python start-launch.py")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n⏰ Phase 1 validation timed out!")
        print("This might indicate a system issue. Please check:")
        print("1. System resources (CPU, memory)")
        print("2. Network connectivity")
        print("3. Database/Redis availability")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Phase 1 interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Phase 1 failed with error: {e}")
        return False

def show_next_steps():
    """Show next steps after Phase 1"""
    print("\n📋 What's Next?")
    print("=" * 30)
    print()
    print("If Phase 1 was successful:")
    print("• Use the Launch Coordinator for Phase 2+")
    print("• Command: python scripts/launch-coordinator.py")
    print()
    print("Launch Coordinator Options:")
    print("• Interactive menu for phase management")
    print("• Run individual phases: python scripts/launch-coordinator.py phase2")
    print("• Check status: python scripts/launch-coordinator.py status")
    print("• Run all phases: python scripts/launch-coordinator.py all")
    print()
    print("If Phase 1 failed:")
    print("• Review error messages and fix issues")
    print("• Re-run this script: python start-launch.py")
    print("• Check the troubleshooting guide in the launch plan")

def main():
    """Main function"""
    print_welcome()
    
    # Check readiness
    if not check_launch_readiness():
        print("\n❌ Launch readiness check failed!")
        print("Please address the issues above and try again.")
        return False
    
    # Show phases
    show_launch_phases()
    
    # Get confirmation
    if not get_user_confirmation():
        print("\nLaunch cancelled. You can run this script again when ready.")
        print("Command: python start-launch.py")
        return False
    
    # Start Phase 1
    success = start_phase1()
    
    # Show next steps
    show_next_steps()
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nLaunch process interrupted. Goodbye! 👋")
        sys.exit(1)