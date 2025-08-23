#!/usr/bin/env python3
"""
Test if we can start a simple server
"""

import sys
import socket

def test_port(port):
    """Test if we can bind to a port"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            print(f"✅ Port {port} is available")
            return True
    except OSError as e:
        print(f"❌ Port {port} is not available: {e}")
        return False

def main():
    print("🧪 Testing ScrollIntel Server Setup")
    print("=" * 40)
    
    # Test Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Test port availability
    ports = [8000, 8001, 8002, 8003, 8080]
    available_port = None
    
    for port in ports:
        if test_port(port):
            available_port = port
            break
    
    if not available_port:
        print("\n❌ No ports available. Try:")
        print("   1. Close other development servers")
        print("   2. Run as administrator")
        print("   3. Restart your computer")
        return 1
    
    # Test FastAPI import
    try:
        from fastapi import FastAPI
        print("✅ FastAPI is available")
    except ImportError:
        print("❌ FastAPI not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi"])
        print("✅ FastAPI installed")
    
    # Test uvicorn import
    try:
        import uvicorn
        print("✅ Uvicorn is available")
    except ImportError:
        print("❌ Uvicorn not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn[standard]"])
        print("✅ Uvicorn installed")
    
    # Create and test a minimal server
    print(f"\n🚀 Testing minimal server on port {available_port}...")
    
    try:
        from fastapi import FastAPI
        import uvicorn
        import threading
        import time
        import requests
        
        app = FastAPI()
        
        @app.get("/")
        def read_root():
            return {"message": "ScrollIntel test server is working!"}
        
        @app.get("/health")
        def health():
            return {"status": "healthy"}
        
        # Start server in a thread
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=available_port, log_level="error")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Test the server
        try:
            response = requests.get(f"http://127.0.0.1:{available_port}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server test successful!")
                print(f"🌐 Server is working at: http://127.0.0.1:{available_port}")
                return 0
            else:
                print(f"❌ Server responded with status {response.status_code}")
                return 1
        except Exception as e:
            print(f"❌ Could not connect to server: {e}")
            return 1
            
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return 1

if __name__ == "__main__":
    result = main()
    input("\nPress Enter to continue...")
    sys.exit(result)