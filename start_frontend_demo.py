#!/usr/bin/env python3
"""
Start ScrollIntel Frontend Demo
"""

import subprocess
import os
import sys
import time

def start_frontend():
    """Start the Next.js frontend"""
    try:
        # Change to frontend directory
        frontend_dir = os.path.join(os.getcwd(), 'frontend')
        
        if not os.path.exists(frontend_dir):
            print("❌ Frontend directory not found")
            return False
            
        print("🚀 Starting ScrollIntel Frontend...")
        print(f"📁 Frontend directory: {frontend_dir}")
        
        # Start the development server
        process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("✅ Frontend starting...")
        print("🌐 Frontend will be available at: http://localhost:3000")
        print("🔗 Backend API is running at: http://localhost:8000")
        print("\n📋 Next Steps:")
        print("   1. Wait for frontend to start (usually 10-30 seconds)")
        print("   2. Open http://localhost:3000 in your browser")
        print("   3. Test the full ScrollIntel experience!")
        
        # Wait a bit and check if it's starting
        time.sleep(5)
        
        if process.poll() is None:
            print("\n✅ Frontend process is running!")
            print("💡 Press Ctrl+C to stop the frontend when done")
            
            # Keep the process running
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping frontend...")
                process.terminate()
                process.wait()
                print("✅ Frontend stopped")
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Frontend failed to start:")
            print(f"   stdout: {stdout}")
            print(f"   stderr: {stderr}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return False

def main():
    """Main function"""
    print("🎯 ScrollIntel Frontend Launcher")
    print("=" * 50)
    
    # Check if backend is running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print("⚠️  Backend may not be fully ready")
    except:
        print("❌ Backend is not running!")
        print("   Please start it first: python run_simple.py")
        return
    
    print("\n" + "=" * 50)
    
    # Start frontend
    start_frontend()

if __name__ == "__main__":
    main()