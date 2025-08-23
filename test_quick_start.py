#!/usr/bin/env python3
"""
Quick test to verify ScrollIntel can start
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Test core imports
        from scrollintel.core.config import get_config
        print("✅ Core config imported")
        
        from scrollintel.api.gateway import app
        print("✅ API gateway imported")
        
        from scrollintel.models.database import Base
        print("✅ Database models imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("🔧 Testing configuration...")
    
    try:
        from scrollintel.core.config import get_config
        config = get_config()
        print("✅ Configuration loaded successfully")
        print(f"   Environment: {config.get('environment', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False

def test_api_creation():
    """Test API app creation"""
    print("🌐 Testing API creation...")
    
    try:
        from scrollintel.api.gateway import app
        print("✅ API app created successfully")
        print(f"   App type: {type(app)}")
        return True
    except Exception as e:
        print(f"❌ API creation failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 ScrollIntel Quick Start Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("API Creation Test", test_api_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️ {test_name} failed")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! ScrollIntel is ready to deploy.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)