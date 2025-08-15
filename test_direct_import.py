#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nTrying direct import...")

try:
    # Try importing the module directly
    import scrollintel.engines.visual_generation.models.ultra_performance_pipeline as up_module
    print("✅ Module imported successfully!")
    
    print("Module attributes:")
    for attr in dir(up_module):
        if not attr.startswith('_'):
            print(f"  - {attr}: {type(getattr(up_module, attr))}")
    
    # Try to get the class
    if hasattr(up_module, 'UltraRealisticVideoGenerationPipeline'):
        cls = getattr(up_module, 'UltraRealisticVideoGenerationPipeline')
        print(f"✅ Found class: {cls}")
        
        # Try to instantiate
        instance = cls()
        print("✅ Instance created successfully!")
    else:
        print("❌ UltraRealisticVideoGenerationPipeline not found in module")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()