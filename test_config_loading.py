#!/usr/bin/env python3
"""
Test configuration loading
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scrollintel.core.configuration_manager import get_config

def main():
    """Test configuration loading"""
    print("=== Configuration Loading Test ===")
    
    # Check environment variables before loading config
    print("Environment variables before config loading:")
    print(f"  DATABASE_URL: {os.getenv('DATABASE_URL', 'Not set')}")
    print(f"  POSTGRES_PASSWORD: {os.getenv('POSTGRES_PASSWORD', 'Not set')}")
    
    # Load configuration
    config = get_config()
    
    print(f"\nLoaded configuration:")
    print(f"  Primary URL: {config.database.primary_url}")
    print(f"  Fallback URL: {config.database.fallback_url}")
    
    # Check environment variables after loading config
    print("\nEnvironment variables after config loading:")
    print(f"  DATABASE_URL: {os.getenv('DATABASE_URL', 'Not set')}")
    print(f"  POSTGRES_PASSWORD: {os.getenv('POSTGRES_PASSWORD', 'Not set')}")

if __name__ == "__main__":
    main()