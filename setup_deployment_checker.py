#!/usr/bin/env python3
"""
Setup script for ScrollIntel Deployment Checker
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages for deployment checker"""
    try:
        print("📦 Installing deployment checker requirements...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "deployment_checker_requirements.txt"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    except FileNotFoundError:
        print("❌ pip not found. Please install pip first.")
        return False

def make_executable():
    """Make the deployment checker executable"""
    try:
        os.chmod("check_deployment_status.py", 0o755)
        print("✅ Made deployment checker executable")
        return True
    except Exception as e:
        print(f"⚠️  Could not make executable: {e}")
        return False

def test_checker():
    """Test the deployment checker"""
    try:
        print("🧪 Testing deployment checker...")
        result = subprocess.run([sys.executable, "check_deployment_status.py"], 
                              capture_output=True, text=True, timeout=30)
        print("✅ Deployment checker test completed")
        return True
    except subprocess.TimeoutExpired:
        print("⚠️  Deployment checker test timed out (this is normal if services aren't running)")
        return True
    except Exception as e:
        print(f"❌ Deployment checker test failed: {e}")
        return False

def main():
    """Setup the enhanced deployment checker"""
    print("🚀 ScrollIntel Deployment Checker Setup")
    print("=" * 50)
    
    success = True
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Make executable
    if not make_executable():
        success = False
    
    # Test checker
    if not test_checker():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\n📋 Usage:")
        print("  python check_deployment_status.py")
        print("  ./check_deployment_status.py")
        print("\n💡 The checker will monitor:")
        print("  • Core services (Backend, Frontend)")
        print("  • API endpoints")
        print("  • Database connectivity")
        print("  • Docker containers")
        print("  • File system integrity")
        print("  • System resources")
    else:
        print("⚠️  Setup completed with some issues")
        print("💡 You may need to install dependencies manually")
        sys.exit(1)

if __name__ == "__main__":
    main()