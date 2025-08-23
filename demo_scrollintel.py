#!/usr/bin/env python3
"""
ScrollIntel Ultimate Demo Launcher
The easiest way to demo ScrollIntel - automatically chooses the best option
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def print_banner():
    print("\n" + "="*60)
    print("🚀 ScrollIntel™ Ultimate Demo Launcher")
    print("="*60)
    print("🤖 AI-Powered CTO Platform")
    print("🎯 Automatically choosing the best demo option...")
    print("="*60 + "\n")

def check_node_js():
    """Check if Node.js is available"""
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Node.js found: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("ℹ️  Node.js not found - will use backend-only demo")
    return False

def check_frontend_ready():
    """Check if frontend is ready"""
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("ℹ️  Frontend directory not found")
        return False
    
    if not (frontend_dir / "package.json").exists():
        print("ℹ️  Frontend package.json not found")
        return False
    
    print("✅ Frontend directory ready")
    return True

def main():
    print_banner()
    
    # Check if we're in the right directory
    if not Path("scrollintel").exists():
        print("❌ Please run this script from the ScrollIntel root directory")
        print("💡 Make sure you're in the folder containing the 'scrollintel' directory")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Determine the best demo option
    has_node = check_node_js()
    has_frontend = check_frontend_ready()
    
    print("\n🎯 Choosing demo option...")
    
    if has_node and has_frontend:
        print("🌟 Full Stack Demo - Starting both backend and frontend!")
        print("📱 Frontend: http://localhost:3000")
        print("🔧 Backend: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        
        print("\n⏳ Starting full stack demo...")
        try:
            subprocess.run([sys.executable, "start_full_demo.py"])
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("💡 Falling back to backend-only demo...")
            subprocess.run([sys.executable, "start_backend_demo.py"])
    
    else:
        print("🔧 Backend-Only Demo - Perfect for quick testing!")
        print("🔧 Backend: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("🌐 Simple UI: Open simple_frontend.html in your browser")
        
        print("\n⏳ Starting backend demo...")
        
        # Start backend in background
        backend_process = subprocess.Popen([
            sys.executable, "start_backend_demo.py"
        ])
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        # Open simple frontend
        simple_frontend = Path("simple_frontend.html")
        if simple_frontend.exists():
            print("🌐 Opening simple web interface...")
            try:
                webbrowser.open(f"file://{simple_frontend.absolute()}")
            except Exception as e:
                print(f"❌ Could not open browser: {e}")
                print(f"💡 Please open {simple_frontend.absolute()} manually")
        
        try:
            backend_process.wait()
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
            backend_process.terminate()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please check the setup and try again")
        input("Press Enter to exit...")