#!/usr/bin/env python3
"""
ScrollIntel.com Local Test Script
Test your ScrollIntel platform before deploying to production
"""

import subprocess
import time
import requests
import os
import sys

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                ScrollIntel.com Local Test                    ║
║            Test Your AI Platform Before Launch               ║
╚══════════════════════════════════════════════════════════════╝
    """)

def check_requirements():
    """Check if required files exist"""
    required_files = [
        'run_simple.py',
        'scrollintel/api/main.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def test_api_health():
    """Test if the API is responding"""
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            print("✅ API health check passed")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API not responding: {e}")
        return False

def test_api_docs():
    """Test if API documentation is accessible"""
    try:
        response = requests.get('http://localhost:8000/docs', timeout=5)
        if response.status_code == 200:
            print("✅ API documentation accessible")
            return True
        else:
            print(f"❌ API documentation failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API documentation not accessible: {e}")
        return False

def main():
    print_banner()
    
    print("🧪 Testing ScrollIntel platform locally...")
    print("This will help you verify everything works before deploying to scrollintel.com")
    print()
    
    # Check requirements
    print("1. 📋 Checking requirements...")
    if not check_requirements():
        print("❌ Requirements check failed. Please ensure all files are present.")
        return False
    
    print()
    print("2. 🚀 Starting ScrollIntel in simple mode...")
    print("   This may take 30-60 seconds...")
    
    # Start the simple server
    try:
        # Start the server in the background
        process = subprocess.Popen([
            sys.executable, 'run_simple.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("   ⏳ Waiting for server to start...")
        time.sleep(15)
        
        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start:")
            print(f"   stdout: {stdout.decode()}")
            print(f"   stderr: {stderr.decode()}")
            return False
        
        print("   ✅ Server started successfully")
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False
    
    print()
    print("3. 🏥 Running health checks...")
    
    # Wait a bit more for full startup
    time.sleep(10)
    
    # Test API health
    api_healthy = test_api_health()
    
    # Test API docs
    docs_accessible = test_api_docs()
    
    print()
    print("4. 📊 Test Results:")
    print("=" * 40)
    
    if api_healthy and docs_accessible:
        print("✅ All tests passed!")
        print()
        print("🎉 Your ScrollIntel platform is working perfectly!")
        print()
        print("🌐 Access your platform:")
        print("   Main App: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        print("   Health Check: http://localhost:8000/health")
        print()
        print("💡 Try these features:")
        print("   1. Upload a CSV file")
        print("   2. Chat with AI agents")
        print("   3. Generate insights")
        print("   4. Build ML models")
        print()
        print("🚀 Ready to deploy to scrollintel.com!")
        print("   Run: ./deploy_scrollintel_production.sh")
        
    else:
        print("❌ Some tests failed. Check the logs above.")
        print("   Try running: python run_simple.py")
        print("   And check for any error messages.")
    
    print()
    print("🛑 To stop the server:")
    print("   Press Ctrl+C in the terminal running the server")
    print("   Or run: pkill -f run_simple.py")
    
    # Keep the process running
    try:
        print()
        print("⏳ Server is running. Press Ctrl+C to stop...")
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        process.terminate()
        process.wait()
        print("✅ Server stopped")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)