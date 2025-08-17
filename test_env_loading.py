#!/usr/bin/env python3
"""
Test environment variable loading
"""

import os
from pathlib import Path

# Load .env file manually
from dotenv import load_dotenv

# Load .env file
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env file from {env_path}")
else:
    print("❌ .env file not found")

# Check key environment variables
print(f"DATABASE_URL: {os.getenv('DATABASE_URL', 'Not set')}")
print(f"POSTGRES_PASSWORD: {os.getenv('POSTGRES_PASSWORD', 'Not set')}")
print(f"SESSION_TIMEOUT_MINUTES: {os.getenv('SESSION_TIMEOUT_MINUTES', 'Not set')}")
print(f"MAX_FILE_SIZE: {os.getenv('MAX_FILE_SIZE', 'Not set')}")

# Test configuration loading
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from scrollintel.core.config import get_settings
    settings = get_settings()
    print(f"✅ Settings loaded: {type(settings)}")
    print(f"Database URL from settings: {settings.get('database_url', 'Not found')}")
    
except Exception as e:
    print(f"❌ Error loading settings: {e}")
    import traceback
    traceback.print_exc()