#!/usr/bin/env python3
"""
Simple configuration test
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ['DATABASE_URL'] = 'postgresql://postgres:boatemaa1612@localhost:5432/scrollintel'
os.environ['SESSION_TIMEOUT_MINUTES'] = '60'

try:
    # Import just the configuration manager
    from scrollintel.core.configuration_manager import get_config
    print("✅ Successfully imported configuration manager")
    
    config = get_config()
    print(f"✅ Config loaded: {type(config)}")
    
    # Check session configuration
    print(f"Session config: {config.session}")
    print(f"Session timeout: {config.session.timeout_minutes}")
    
    # Now try the legacy config
    from scrollintel.core.config import get_config as get_legacy_config
    print("✅ Successfully imported legacy config")
    
    legacy_config = get_legacy_config()
    print(f"✅ Legacy config loaded: {type(legacy_config)}")
    
    if isinstance(legacy_config, dict):
        print(f"session_timeout_minutes in dict: {'session_timeout_minutes' in legacy_config}")
        if 'session_timeout_minutes' in legacy_config:
            print(f"Value: {legacy_config['session_timeout_minutes']}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()