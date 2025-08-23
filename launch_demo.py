#!/usr/bin/env python3
"""
ScrollIntel Demo Launcher
Quick launch script for demonstration
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("🚀 ScrollIntel Demo Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("scrollintel").exists():
        print("❌ Error: Please run from the ScrollIntel root directory")
        return
    
    print("✅ Starting ScrollIntel API server...")
    
    try:
        # Start the server in the background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "scrollintel.api.simple_main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
        
        print("⏳ Waiting for server to start...")
        time.sleep(3)
        
        print("✅ ScrollIntel is now running!")
        print()
        print("🌐 Access URLs:")
        print("   Main API: http://localhost:8000")
        print("   Health Check: http://localhost:8000/health")
        print("   API Docs: http://localhost:8000/docs")
        print("   Dashboard: http://localhost:8000/api/dashboard")
        print()
        print("🤖 Available AI Agents:")
        print("   • CTO Agent - Strategic technical decisions")
        print("   • Data Scientist - Data analysis and insights")
        print("   • ML Engineer - Model building and deployment")
        print("   • AI Engineer - AI system architecture")
        print("   • Business Analyst - Business intelligence")
        print("   • QA Engineer - Quality assurance and testing")
        print()
        print("📁 Demo Features:")
        print("   • Upload files (CSV, Excel, JSON)")
        print("   • Chat with AI agents")
        print("   • Build ML models")
        print("   • View analytics dashboard")
        print("   • Real-time health monitoring")
        print()
        print("🔥 Try these demo commands:")
        print("   curl http://localhost:8000/health")
        print("   curl http://localhost:8000/api/agents")
        print("   curl http://localhost:8000/api/dashboard")
        print()
        print("Press Ctrl+C to stop the server")
        
        # Open browser
        try:
            webbrowser.open("http://localhost:8000")
        except:
            pass
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping ScrollIntel...")
        process.terminate()
        print("✅ ScrollIntel stopped successfully")
    except Exception as e:
        print(f"❌ Error starting ScrollIntel: {e}")

if __name__ == "__main__":
    main()