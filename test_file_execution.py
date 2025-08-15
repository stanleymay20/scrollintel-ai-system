#!/usr/bin/env python3

import sys
import traceback

print("Testing file execution...")

try:
    # Read the file content
    with open('scrollintel/engines/emotion_simulator.py', 'r') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    
    # Try to execute it
    exec(content)
    print("✓ File executed successfully")
    print("Available names:", [name for name in locals() if not name.startswith('_') and name not in ['sys', 'traceback', 'content', 'f']])
    
except Exception as e:
    print(f"❌ Error executing file: {e}")
    traceback.print_exc()
    
    # Try to find where the error occurs
    lines = content.split('\n')
    print(f"\nFile has {len(lines)} lines")
    
    # Try executing line by line to find the issue
    for i, line in enumerate(lines[:50], 1):  # Check first 50 lines
        try:
            if line.strip() and not line.strip().startswith('#'):
                exec(line)
        except Exception as line_error:
            print(f"Error on line {i}: {line}")
            print(f"Error: {line_error}")
            break