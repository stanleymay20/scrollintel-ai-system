#!/usr/bin/env python3
"""
ScrollIntel Full Stack Launcher
Starts both frontend and backend with optimized UI/UX
"""

import asyncio
import subprocess
import sys
import os
import time
import threading
import webbrowser
from pathlib import Path
import signal
import psutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class ScrollIntelLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        
    def print_banner(self):
        """Print ScrollIntel banner"""
        print("\n" + "="*60)
        print("🚀 SCROLLINTEL™ FULL STACK LAUNCHER")
        print("="*60)
        print("🤖 AI-Powered CTO Platform")
        print("📊 Advanced Analytics & ML Models")
        print("🎨 Optimized UI/UX Experience")
        print("="*60 + "\n")
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        # Check Python packages
        required_packages = ['fastapi', 'uvicorn', 'sqlalchemy', 'pandas']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} - OK")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package} - Missing")
        
        if missing_packages:
            print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-r", "requirements.txt"
            ])
            print("✅ Dependencies installed successfully")
        
        # Check Node.js for frontend
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Node.js {result.stdout.strip()} - OK")
            else:
                print("❌ Node.js - Not found")
                return False
        except FileNotFoundError:
            print("❌ Node.js - Not installed")
            print("💡 Please install Node.js from https://nodejs.org/")
            return False
        
        return True
    
    def setup_environment(self):
        """Setup environment variables"""
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
JWT_SECRET_KEY=your-secret-key-here
DEBUG=true
ENVIRONMENT=development
API_PORT=8000
FRONTEND_PORT=3000

# Optional API Keys (for AI features)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
""")
                print("✅ Created basic .env file")
        
        # Load environment variables
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        
        print("✅ Environment configured")
    
    def find_available_port(self, start_port, max_attempts=10):
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            try:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        return None
    
    def start_backend(self):
        """Start the backend API server"""
        print("🔄 Starting backend API server...")
        
        # Find available port
        backend_port = self.find_available_port(8000)
        if not backend_port:
            print("❌ Could not find available port for backend")
            return False
        
        try:
            # Initialize database first
            print("📊 Initializing database...")
            subprocess.run([sys.executable, "init_database.py"], 
                         check=True, capture_output=True)
            print("✅ Database initialized")
            
            # Start backend server
            self.backend_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "scrollintel.api.main:app",
                "--host", "127.0.0.1",
                "--port", str(backend_port),
                "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for backend to start
            print(f"⏳ Waiting for backend to start on port {backend_port}...")
            time.sleep(5)
            
            # Check if backend is running
            if self.backend_process.poll() is None:
                print(f"✅ Backend API running at http://localhost:{backend_port}")
                print(f"📚 API Documentation: http://localhost:{backend_port}/docs")
                os.environ['BACKEND_PORT'] = str(backend_port)
                return True
            else:
                print("❌ Backend failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the frontend development server"""
        print("🔄 Starting frontend development server...")
        
        frontend_dir = Path("frontend")
        if not frontend_dir.exists():
            print("❌ Frontend directory not found")
            return False
        
        # Find available port
        frontend_port = self.find_available_port(3000)
        if not frontend_port:
            print("❌ Could not find available port for frontend")
            return False
        
        try:
            # Install dependencies if needed
            package_json = frontend_dir / "package.json"
            node_modules = frontend_dir / "node_modules"
            
            if package_json.exists() and not node_modules.exists():
                print("📦 Installing frontend dependencies...")
                subprocess.run(["npm", "install"], 
                             cwd=frontend_dir, check=True)
                print("✅ Frontend dependencies installed")
            
            # Set environment variables for frontend
            env = os.environ.copy()
            backend_port = os.environ.get('BACKEND_PORT', '8000')
            env['NEXT_PUBLIC_API_URL'] = f'http://localhost:{backend_port}'
            env['PORT'] = str(frontend_port)
            
            # Start frontend server
            self.frontend_process = subprocess.Popen([
                "npm", "run", "dev"
            ], cwd=frontend_dir, env=env,
               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for frontend to start
            print(f"⏳ Waiting for frontend to start on port {frontend_port}...")
            time.sleep(10)
            
            # Check if frontend is running
            if self.frontend_process.poll() is None:
                print(f"✅ Frontend running at http://localhost:{frontend_port}")
                os.environ['FRONTEND_PORT'] = str(frontend_port)
                return True
            else:
                print("❌ Frontend failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting frontend: {e}")
            return False
    
    def open_browser(self):
        """Open the application in the default browser"""
        frontend_port = os.environ.get('FRONTEND_PORT', '3000')
        url = f"http://localhost:{frontend_port}"
        
        print(f"🌐 Opening ScrollIntel in your browser: {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"❌ Could not open browser automatically: {e}")
            print(f"💡 Please open {url} manually in your browser")
    
    def show_status(self):
        """Show the current status and access points"""
        backend_port = os.environ.get('BACKEND_PORT', '8000')
        frontend_port = os.environ.get('FRONTEND_PORT', '3000')
        
        print("\n" + "="*60)
        print("🎉 SCROLLINTEL™ IS NOW RUNNING!")
        print("="*60)
        print(f"🌐 Frontend:        http://localhost:{frontend_port}")
        print(f"🔧 Backend API:     http://localhost:{backend_port}")
        print(f"📚 API Docs:        http://localhost:{backend_port}/docs")
        print(f"❤️  Health Check:   http://localhost:{backend_port}/health")
        print("="*60)
        print("\n🚀 Quick Start Guide:")
        print("1. Upload your data files (CSV, Excel, JSON)")
        print("2. Chat with AI agents for insights")
        print("3. Build ML models with AutoML")
        print("4. Create interactive dashboards")
        print("5. Export results and reports")
        print("\n💡 Tips:")
        print("- Use Ctrl+C to stop all services")
        print("- Check logs in the terminal for debugging")
        print("- Visit /docs for complete API documentation")
        print("\n🌟 ScrollIntel™ - Where AI meets unlimited potential!")
        print("="*60 + "\n")
    
    def monitor_processes(self):
        """Monitor backend and frontend processes"""
        while self.running:
            try:
                # Check backend
                if self.backend_process and self.backend_process.poll() is not None:
                    print("⚠️  Backend process stopped unexpectedly")
                    break
                
                # Check frontend
                if self.frontend_process and self.frontend_process.poll() is not None:
                    print("⚠️  Frontend process stopped unexpectedly")
                    break
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                break
    
    def cleanup(self):
        """Clean up processes"""
        print("\n🛑 Shutting down ScrollIntel...")
        self.running = False
        
        if self.backend_process:
            print("🔄 Stopping backend...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
            print("✅ Backend stopped")
        
        if self.frontend_process:
            print("🔄 Stopping frontend...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
            print("✅ Frontend stopped")
        
        print("👋 ScrollIntel stopped successfully")
    
    def run(self):
        """Main run method"""
        try:
            self.print_banner()
            
            if not self.check_dependencies():
                return 1
            
            self.setup_environment()
            
            if not self.start_backend():
                return 1
            
            if not self.start_frontend():
                self.cleanup()
                return 1
            
            self.show_status()
            self.open_browser()
            
            # Monitor processes
            self.monitor_processes()
            
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return 1
        finally:
            self.cleanup()
        
        return 0

def main():
    """Main entry point"""
    launcher = ScrollIntelLauncher()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        launcher.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    return launcher.run()

if __name__ == "__main__":
    sys.exit(main())