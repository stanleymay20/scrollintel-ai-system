#!/usr/bin/env python3

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from scrollintel.engines.crisis_communication_integration import CrisisCommunicationIntegration
    print("✅ Import successful!")
    
    # Test basic functionality
    integration = CrisisCommunicationIntegration()
    print(f"✅ Instance created: {type(integration)}")
    
    # Test basic methods
    print(f"✅ Active crises: {len(integration.get_active_crises())}")
    print(f"✅ Message templates: {len(integration.message_templates)}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    
    # Debug information
    print("\nDebugging information:")
    try:
        import scrollintel.engines.crisis_communication_integration as module
        print(f"Module loaded: {module}")
        print(f"Module file: {module.__file__}")
        print(f"Module attributes: {[x for x in dir(module) if not x.startswith('_')]}")
        
        # Try to read the file directly
        with open(module.__file__, 'r') as f:
            content = f.read()
            print(f"File size: {len(content)} characters")
            print(f"Contains 'class CrisisCommunicationIntegration': {'class CrisisCommunicationIntegration' in content}")
            
    except Exception as debug_e:
        print(f"Debug failed: {debug_e}")

except Exception as e:
    print(f"❌ Other error: {e}")