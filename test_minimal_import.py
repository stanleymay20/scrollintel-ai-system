#!/usr/bin/env python3
"""Test minimal import."""

print("Testing minimal import...")

# Test basic imports
try:
    import asyncio
    print("✓ asyncio imported")
except Exception as e:
    print(f"✗ asyncio failed: {e}")

try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy failed: {e}")

try:
    from pathlib import Path
    print("✓ pathlib imported")
except Exception as e:
    print(f"✗ pathlib failed: {e}")

try:
    from enum import Enum
    print("✓ enum imported")
except Exception as e:
    print(f"✗ enum failed: {e}")

# Test ScrollIntel imports
try:
    from scrollintel.engines.base_engine import BaseEngine
    print("✓ BaseEngine imported")
except Exception as e:
    print(f"✗ BaseEngine failed: {e}")

try:
    from scrollintel.models.style_transfer_models import StyleTransferRequest
    print("✓ StyleTransferRequest imported")
except Exception as e:
    print(f"✗ StyleTransferRequest failed: {e}")

# Test creating a minimal class
try:
    class TestEngine(BaseEngine):
        def __init__(self):
            super().__init__("test", "Test Engine", [])
    
    print("✓ TestEngine created successfully")
except Exception as e:
    print(f"✗ TestEngine failed: {e}")

print("All tests completed")