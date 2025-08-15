#!/usr/bin/env python3

print("Testing engine file step by step...")

try:
    print("1. Testing basic imports...")
    import asyncio
    from datetime import datetime, timedelta
    from typing import List, Dict, Optional, Any
    import logging
    print("   ✅ Basic imports successful")
except Exception as e:
    print(f"   ❌ Basic imports failed: {e}")
    exit(1)

try:
    print("2. Testing models import...")
    from scrollintel.models.stakeholder_confidence_models import (
        StakeholderProfile, ConfidenceMetrics, ConfidenceBuildingStrategy,
        TrustMaintenanceAction, CommunicationPlan, ConfidenceAssessment,
        StakeholderFeedback, ConfidenceAlert, StakeholderType, ConfidenceLevel,
        TrustIndicator
    )
    print("   ✅ Models import successful")
except Exception as e:
    print(f"   ❌ Models import failed: {e}")
    exit(1)

try:
    print("3. Testing engine file execution...")
    exec(open('scrollintel/engines/stakeholder_confidence_engine.py').read())
    print("   ✅ Engine file executed successfully")
    
    # Check if class is defined in local scope
    if 'StakeholderConfidenceEngine' in locals():
        print("   ✅ StakeholderConfidenceEngine class found in locals")
        engine = locals()['StakeholderConfidenceEngine']()
        print(f"   ✅ Engine instance created: {type(engine)}")
    else:
        print("   ❌ StakeholderConfidenceEngine class not found in locals")
        print(f"   Available names: {[name for name in locals().keys() if not name.startswith('_')]}")
        
except Exception as e:
    print(f"   ❌ Engine file execution failed: {e}")
    import traceback
    traceback.print_exc()

print("Debug complete.")