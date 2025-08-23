#!/usr/bin/env python3
"""
ScrollIntel™ Full Stack Application Launcher
Starts the complete ScrollIntel application with backend API and Next.js frontend
"""

import os
import sys
import subprocess
import time
import threading
import signal
import webbrowser
from pathlib import Path
import socket
import json

def print_banner():
    print("\n" + "="*70)
    print("🚀 SCROLLINTEL™ FULL STACK APPLICATION")
    print("="*70)
    print("🤖 AI-Powered CTO Platform")
    print("📊 Advanced Analytics & Machine Learning")
    print("🎨 Modern React Frontend with Next.js 14")
    print("🔧 FastAPI Backend with Real-time Features")
    print("="*70)
    print("Starting complete application stack...")
    print("="*70 + "\n")

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

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking system dependencies...")
    
    # Check Python packages
    required_packages = ['fastapi', 'uvicorn', 'sqlalchemy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing Python packages...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-r", "requirements.txt"
            ])
            print("✅ Python dependencies installed")
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    # Check Node.js
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js {result.stdout.strip()}")
        else:
            print("❌ Node.js not found")
            return False
    except FileNotFoundError:
        print("❌ Node.js not installed")
        print("💡 Please install Node.js from https://nodejs.org/")
        return False
    
    # Check npm
    try:
        result = subprocess.run(['npm', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ npm {result.stdout.strip()}")
        else:
            print("❌ npm not found")
            return False
    except FileNotFoundError:
        print("❌ npm not found")
        return False
    
    print("✅ All dependencies available\n")
    return True

def setup_environment():
    """Setup environment variables and configuration"""
    print("🔧 Setting up environment...")
    
    # Create .env if it doesn't exist
    env_file = Path('.env')
    if not env_file.exists():
        if Path('.env.example').exists():
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ Created .env from .env.example")
        else:
            # Create basic .env
            with open('.env', 'w') as f:
                f.write("""# ScrollIntel Environment Configuration
DATABASE_URL=sqlite:///./scrollintel.db
JWT_SECRET_KEY=scrollintel-dev-secret-key-change-in-production
DEBUG=true
ENVIRONMENT=development
API_PORT=8000
FRONTEND_PORT=3000

# Optional API Keys (for AI features)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HUGGINGFACE_API_KEY=

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=ScrollIntel
NEXT_PUBLIC_APP_VERSION=4.0.0
""")
            print("✅ Created .env file")
    
    print("✅ Environment configured\n")

def start_backend():
    """Start the FastAPI backend server with full features"""
    print("🔧 Starting Backend API Server...")
    
    # Find available port for backend
    backend_port = find_available_port(8000)
    if not backend_port:
        print("❌ Could not find available port for backend")
        return None, None
    
    try:
        # Initialize database first
        print("📊 Initializing database...")
        try:
            subprocess.run([sys.executable, "init_database.py"], 
                         check=True, capture_output=True, timeout=30)
            print("✅ Database initialized")
        except subprocess.TimeoutExpired:
            print("⚠️  Database initialization taking longer than expected, continuing...")
        except Exception as e:
            print(f"⚠️  Database initialization warning: {e}")
        
        # Start the full API server (not just simple)
        print(f"🚀 Starting full ScrollIntel API on port {backend_port}...")
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "scrollintel.api.main:app",
            "--host", "127.0.0.1",
            "--port", str(backend_port),
            "--reload",
            "--log-level", "info"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for backend to start
        print("⏳ Waiting for backend to initialize...")
        time.sleep(5)
        
        # Check if backend is running
        if backend_process.poll() is None:
            print(f"✅ Backend API running at http://localhost:{backend_port}")
            print(f"📚 API Documentation: http://localhost:{backend_port}/docs")
            print(f"🔍 Interactive API: http://localhost:{backend_port}/redoc")
            return backend_process, backend_port
        else:
            print("❌ Backend failed to start, trying simple fallback...")
            # Fallback to simple API
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
            
            time.sleep(3)
            if backend_process.poll() is None:
                print(f"✅ Backend API (simple) running at http://localhost:{backend_port}")
                return backend_process, backend_port
            else:
                print("❌ Both full and simple backend failed to start")
                return None, None
            
    except Exception as e:
        print(f"❌ Backend failed to start: {e}")
        return None, None

def start_frontend(backend_port):
    """Start the Next.js frontend development server with full features"""
    print("🎨 Starting Next.js Frontend...")
    frontend_dir = Path("frontend")
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return None, None
    
    # Find available port for frontend
    frontend_port = find_available_port(3000)
    if not frontend_port:
        print("❌ Could not find available port for frontend")
        return None, None
    
    try:
        # Check and install dependencies
        node_modules = frontend_dir / "node_modules"
        package_json = frontend_dir / "package.json"
        
        if package_json.exists():
            print("📋 Found package.json with modern dependencies")
            
            if not node_modules.exists():
                print("📦 Installing frontend dependencies (this may take a few minutes)...")
                result = subprocess.run(["npm", "install"], 
                                      cwd=frontend_dir, 
                                      capture_output=True, 
                                      text=True,
                                      timeout=300)  # 5 minute timeout
                if result.returncode == 0:
                    print("✅ Frontend dependencies installed successfully")
                else:
                    print(f"⚠️  npm install had warnings: {result.stderr}")
                    print("Continuing anyway...")
            else:
                print("✅ Frontend dependencies already installed")
        
        # Set comprehensive environment variables
        env = os.environ.copy()
        env.update({
            'NEXT_PUBLIC_API_URL': f'http://localhost:{backend_port}',
            'PORT': str(frontend_port),
            'NEXT_PUBLIC_APP_NAME': 'ScrollIntel',
            'NEXT_PUBLIC_APP_VERSION': '4.0.0',
            'NEXT_PUBLIC_ENVIRONMENT': 'development',
            'NODE_ENV': 'development'
        })
        
        print(f"🚀 Starting Next.js development server...")
        print(f"   📱 Frontend will be available at: http://localhost:{frontend_port}")
        print(f"   🔗 Connected to API at: http://localhost:{backend_port}")
        
        frontend_process = subprocess.Popen([
            "npm", "run", "dev"
        ], cwd=frontend_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for frontend to start (Next.js can take longer)
        print("⏳ Waiting for Next.js to compile and start...")
        print("   (This may take 30-60 seconds for the first run)")
        
        # Check periodically if it's ready
        for i in range(12):  # Check for up to 60 seconds
            time.sleep(5)
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                return None, None
            
            # Try to connect to see if it's ready
            try:
                import urllib.request
                urllib.request.urlopen(f'http://localhost:{frontend_port}', timeout=1)
                break
            except:
                if i < 11:  # Don't print on last iteration
                    print(f"   ⏳ Still starting... ({(i+1)*5}s)")
                continue
        
        if frontend_process.poll() is None:
            print(f"✅ Next.js Frontend running at http://localhost:{frontend_port}")
            print(f"🎨 Modern React UI with Tailwind CSS")
            print(f"📊 Advanced data visualization components")
            print(f"🤖 AI agent chat interfaces")
            return frontend_process, frontend_port
        else:
            print("❌ Frontend failed to start")
            return None, None
            
    except subprocess.TimeoutExpired:
        print("❌ Frontend dependency installation timed out")
        print("💡 Try running 'npm install' manually in the frontend directory")
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend failed to start: {e}")
        return None, None
    except FileNotFoundError:
        print("❌ Node.js/npm not found. Please install Node.js from https://nodejs.org/")
        return None, None

def open_browser(frontend_port):
    """Open the application in browser"""
    url = f"http://localhost:{frontend_port}"
    print(f"🌐 Opening ScrollIntel in your browser: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"❌ Could not open browser automatically: {e}")
        print(f"💡 Please open {url} manually in your browser")

def show_status(backend_port, frontend_port):
    """Show comprehensive running status"""
    print("\n" + "="*70)
    print("🎉 SCROLLINTEL™ FULL STACK APPLICATION IS RUNNING!")
    print("="*70)
    print(f"🌐 Frontend Application:    http://localhost:{frontend_port}")
    print(f"🔧 Backend API:             http://localhost:{backend_port}")
    print(f"📚 API Documentation:       http://localhost:{backend_port}/docs")
    print(f"🔍 Alternative API Docs:    http://localhost:{backend_port}/redoc")
    print(f"❤️  Health Check:           http://localhost:{backend_port}/health")
    print("="*70)
    print("\n🚀 FEATURES AVAILABLE:")
    print("┌─ 🤖 AI AGENTS")
    print("│  ├─ CTO Agent - Strategic technology leadership")
    print("│  ├─ Data Scientist - Advanced analytics & ML")
    print("│  ├─ ML Engineer - Model development & deployment")
    print("│  ├─ AI Engineer - AI system architecture")
    print("│  ├─ BI Agent - Business intelligence & reporting")
    print("│  └─ QA Agent - Quality assurance & testing")
    print("│")
    print("├─ 📊 DASHBOARDS")
    print("│  ├─ Executive Dashboard - High-level metrics")
    print("│  ├─ System Monitoring - Real-time performance")
    print("│  ├─ Analytics Dashboard - Data insights")
    print("│  └─ Agent Status - AI agent monitoring")
    print("│")
    print("├─ 🎨 ADVANCED UI")
    print("│  ├─ Modern React Components with Radix UI")
    print("│  ├─ Tailwind CSS styling")
    print("│  ├─ Framer Motion animations")
    print("│  ├─ Real-time data visualization")
    print("│  └─ Responsive design")
    print("│")
    print("└─ 🔧 DEVELOPMENT TOOLS")
    print("   ├─ Hot reload for both frontend and backend")
    print("   ├─ TypeScript support")
    print("   ├─ Interactive API documentation")
    print("   └─ Real-time WebSocket connections")
    print("\n💡 QUICK START GUIDE:")
    print("1. 🌐 Open the frontend in your browser (should open automatically)")
    print("2. 💬 Try the chat interface - ask any AI agent a question")
    print("3. 📊 Explore the dashboard - view system metrics and analytics")
    print("4. 📁 Upload data files - CSV, Excel, JSON for analysis")
    print("5. 🔧 Check API docs - explore all available endpoints")
    print("6. ⚙️  Monitor agents - see real-time agent status and performance")
    print("\n🛑 TO STOP: Press Ctrl+C in this terminal")
    print("\n🌟 ScrollIntel™ - Where AI meets unlimited potential!")
    print("="*70 + "\n")

def cleanup_processes(backend_process, frontend_process):
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
    
    if frontend_process:
        print("🔄 Stopping frontend...")
        frontend_process.terminate()
        try:
            frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            frontend_process.kill()
        print("✅ Frontend stopped")
    
    print("👋 ScrollIntel demo stopped successfully")

def main():
    print_banner()
    
    # Check if we're in the right directory
    if not Path("scrollintel").exists():
        print("❌ Please run this script from the ScrollIntel root directory")
        sys.exit(1)
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend
        backend_process, backend_port = start_backend()
        if not backend_process:
            print("❌ Failed to start backend. Exiting.")
            sys.exit(1)
        
        # Start frontend
        frontend_process, frontend_port = start_frontend(backend_port)
        if not frontend_process:
            print("❌ Failed to start frontend. Backend will continue running.")
            print(f"💡 You can still access the API at http://localhost:{backend_port}")
        else:
            # Show status and open browser
            show_status(backend_port, frontend_port)
            open_browser(frontend_port)
        
        # Keep running until interrupted
        print("🔄 Both services are running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process and backend_process.poll() is not None:
                print("⚠️  Backend process stopped unexpectedly")
                break
            
            if frontend_process and frontend_process.poll() is not None:
                print("⚠️  Frontend process stopped unexpectedly")
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        cleanup_processes(backend_process, frontend_process)

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Interrupt received, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    main()