#!/usr/bin/env python3
"""
ScrollIntel Backend Demo Launcher
Starts just the backend API server for testing
"""

import os
import sys
import subprocess
import time
import signal
import webbrowser
from pathlib import Path
import socket

def print_banner():
    print("\n" + "="*60)
    print("🚀 ScrollIntel Backend Demo Launcher")
    print("="*60)
    print("Starting backend API server...")
    print("API will be available at: http://localhost:8000")
    print("="*60 + "\n")

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def start_backend():
    """Start the FastAPI backend server"""
    print("🔧 Starting Backend API...")
    
    # Find available port for backend
    backend_port = find_available_port(8000)
    if not backend_port:
        print("❌ Could not find available port for backend")
        return None, None
    
    try:
        # Use the simple main for better compatibility
        backend_process = subprocess.Popen([
            sys.executable, "-c", f"""
import sys
sys.path.insert(0, '.')
import uvicorn
uvicorn.run(
    'scrollintel.api.simple_main:app',
    host='127.0.0.1',
    port={backend_port},
    reload=False,
    log_level='info'
)
"""
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for backend to start
        print(f"⏳ Waiting for backend to start on port {backend_port}...")
        time.sleep(3)
        
        # Check if backend is running
        if backend_process.poll() is None:
            print(f"✅ Backend API running at http://localhost:{backend_port}")
            return backend_process, backend_port
        else:
            print("❌ Backend failed to start")
            return None, None
            
    except Exception as e:
        print(f"❌ Backend failed to start: {e}")
        return None, None

def open_browser(backend_port):
    """Open the API docs in browser"""
    url = f"http://localhost:{backend_port}/docs"
    print(f"🌐 Opening API documentation: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"❌ Could not open browser automatically: {e}")
        print(f"💡 Please open {url} manually in your browser")

def show_status(backend_port):
    """Show running status"""
    print("\n" + "="*60)
    print("🎉 ScrollIntel Backend Demo is Running!")
    print("="*60)
    print(f"🔧 Backend API:     http://localhost:{backend_port}")
    print(f"📚 API Docs:        http://localhost:{backend_port}/docs")
    print(f"❤️  Health Check:   http://localhost:{backend_port}/health")
    print(f"🤖 Agents List:     http://localhost:{backend_port}/api/agents")
    print(f"📊 Metrics:         http://localhost:{backend_port}/api/monitoring/metrics")
    print("="*60)
    print("\n💡 Quick Start:")
    print("1. Visit the API docs to explore available endpoints")
    print("2. Try the health check endpoint")
    print("3. Test the agent chat functionality")
    print("4. Use Ctrl+C to stop the server")
    print("\n🌟 ScrollIntel™ - AI-Powered CTO Platform")
    print("="*60 + "\n")

def cleanup_processes(backend_process):
    """Clean up processes on exit"""
    print("\n🛑 Shutting down...")
    
    if backend_process:
        print("🔄 Stopping backend...")
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
        print("✅ Backend stopped")
    
    print("👋 ScrollIntel backend demo stopped successfully")

def main():
    print_banner()
    
    # Check if we're in the right directory
    if not Path("scrollintel").exists():
        print("❌ Please run this script from the ScrollIntel root directory")
        sys.exit(1)
    
    backend_process = None
    
    try:
        # Start backend
        backend_process, backend_port = start_backend()
        if not backend_process:
            print("❌ Failed to start backend. Exiting.")
            sys.exit(1)
        
        # Show status and open browser
        show_status(backend_port)
        open_browser(backend_port)
        
        # Keep running until interrupted
        print("🔄 Backend service is running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
            
            # Check if process is still running
            if backend_process and backend_process.poll() is not None:
                print("⚠️  Backend process stopped unexpectedly")
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        cleanup_processes(backend_process)

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Interrupt received, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    main()