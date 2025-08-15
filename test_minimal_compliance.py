#!/usr/bin/env python3
"""Minimal test for compliance analyzer"""

import sys
import traceback

try:
    # Test step by step
    print("Step 1: Import pandas")
    import pandas as pd
    print("✓ Pandas imported")
    
    print("Step 2: Import models")
    from ai_data_readiness.models.compliance_models import SensitiveDataType, RegulationType
    print("✓ Models imported")
    
    print("Step 3: Import exceptions")
    from ai_data_readiness.core.exceptions import ComplianceAnalysisError
    print("✓ Exceptions imported")
    
    print("Step 4: Try to import the module")
    import ai_data_readiness.engines.compliance_analyzer
    print("✓ Module imported")
    
    print("Step 5: Check module contents")
    module = ai_data_readiness.engines.compliance_analyzer
    print(f"Module file: {module.__file__}")
    print(f"Module attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}")
    
    print("Step 6: Try to get ComplianceAnalyzer class")
    if hasattr(module, 'ComplianceAnalyzer'):
        ComplianceAnalyzer = module.ComplianceAnalyzer
        print("✓ ComplianceAnalyzer found")
        
        print("Step 7: Create instance")
        analyzer = ComplianceAnalyzer()
        print("✓ Instance created")
    else:
        print("❌ ComplianceAnalyzer not found in module")
        
except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc()