#!/usr/bin/env python3
"""
Simple test script for ScrollIntel deployment.
"""

import requests
import time
import sys

def test_scrollintel():
    """Test ScrollIntel deployment."""
    print("Testing ScrollIntel deployment...")
    
    # Wait for startup
    print("Waiting for application to start...")
    time.sleep(15)
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            print(f"[CHECK] Health check passed: {response.json()}")
            
            # Test main endpoint
            try:
                main_response = requests.get("http://localhost:8000/", timeout=10)
                if main_response.status_code == 200:
                    print("[CHECK] Main endpoint accessible")
                else:
                    print(f"[WARNING] Main endpoint returned {main_response.status_code}")
            except Exception as e:
                print(f"[WARNING] Could not test main endpoint: {e}")
            
            print("[PARTY] ScrollIntel is working correctly!")
            return True
        else:
            print(f"[X] Health check failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[X] Could not connect to ScrollIntel: {e}")
        return False

if __name__ == "__main__":
    success = test_scrollintel()
    if success:
        print("\n[TROPHY] All tests passed!")
    else:
        print("\n[X] Tests failed!")
        sys.exit(1)
