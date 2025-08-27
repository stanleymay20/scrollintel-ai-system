#!/usr/bin/env python3
"""
Test Railway setup locally
"""

import requests
import time
import subprocess
import sys
import os

def test_railway_app():
    """Test the Railway app locally"""
    
    print("Testing Railway setup...")
    
    # Set environment variables
    os.environ["PORT"] = "8000"
    os.environ["RAILWAY_ENVIRONMENT"] = "test"
    
    # Start the app in background
    print("Starting app...")
    process = subprocess.Popen([
        sys.executable, "start_railway.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for startup
    time.sleep(3)
    
    try:
        # Test health endpoint
        print("Testing /health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=10)
        print(f"Health check status: {response.status_code}")
        print(f"Health response: {response.json()}")
        
        # Test root endpoint
        print("Testing / endpoint...")
        response = requests.get("http://localhost:8000/", timeout=10)
        print(f"Root status: {response.status_code}")
        print(f"Root response: {response.json()}")
        
        # Test ping endpoint
        print("Testing /ping endpoint...")
        response = requests.get("http://localhost:8000/ping", timeout=10)
        print(f"Ping status: {response.status_code}")
        print(f"Ping response: {response.json()}")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        # Stop the process
        process.terminate()
        process.wait()

if __name__ == "__main__":
    test_railway_app()