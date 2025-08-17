#!/usr/bin/env python3
"""
Debug configuration loading issue
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from scrollintel.core.config import get_settings
    print("✅ Successfully imported get_settings")
    
    settings = get_settings()
    print(f"✅ Settings loaded: {type(settings)}")
    
    print("Settings keys:", list(settings.keys()) if isinstance(settings, dict) else "Not a dict")
    
    if isinstance(settings, dict):
        if 'session_timeout_minutes' in settings:
            print(f"✅ session_timeout_minutes: {settings['session_timeout_minutes']}")
        else:
            print("❌ session_timeout_minutes not found in settings")
            print("Available keys:", list(settings.keys()))
    
    # Try to access the problematic attribute
    try:
        timeout = settings.session_timeout_minutes
        print(f"✅ Accessed via attribute: {timeout}")
    except AttributeError as e:
        print(f"❌ AttributeError: {e}")
        print(f"Settings type: {type(settings)}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()