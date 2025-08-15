#!/usr/bin/env python3
"""Simple test to verify board testing framework import"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    print("Testing import...")
    from scrollintel.core.board_testing_framework import BoardTestingFramework
    print("✅ Import successful!")
    
    # Test instantiation
    framework = BoardTestingFramework()
    print("✅ Framework instantiation successful!")
    
    print(f"Framework has {len(framework.benchmark_thresholds)} benchmark thresholds")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")