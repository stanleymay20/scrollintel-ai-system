#!/usr/bin/env python3

import sys
import traceback

print("Testing orchestration engine imports...")

try:
    print("1. Testing basic import...")
    import scrollintel.core.realtime_orchestration_engine as roe
    print(f"   Module imported. Dir: {[x for x in dir(roe) if not x.startswith('_')]}")
    
    print("2. Testing specific class imports...")
    
    try:
        from scrollintel.core.realtime_orchestration_engine import TaskPriority
        print("   TaskPriority imported successfully")
    except ImportError as e:
        print(f"   TaskPriority import failed: {e}")
    
    try:
        from scrollintel.core.realtime_orchestration_engine import OrchestrationTask
        print("   OrchestrationTask imported successfully")
    except ImportError as e:
        print(f"   OrchestrationTask import failed: {e}")
    
    try:
        from scrollintel.core.realtime_orchestration_engine import RealTimeOrchestrationEngine
        print("   RealTimeOrchestrationEngine imported successfully")
    except ImportError as e:
        print(f"   RealTimeOrchestrationEngine import failed: {e}")
        
    print("3. Checking file content...")
    with open('scrollintel/core/realtime_orchestration_engine.py', 'r') as f:
        content = f.read()
        print(f"   File size: {len(content)} characters")
        print(f"   Contains 'class RealTimeOrchestrationEngine': {'class RealTimeOrchestrationEngine' in content}")
        print(f"   Contains 'class TaskPriority': {'class TaskPriority' in content}")
        
except Exception as e:
    print(f"Error during import test: {e}")
    traceback.print_exc()