#!/usr/bin/env python3
"""
ScrollIntel Production Stack Launcher
Full production-ready frontend and backend with optimized UI/UX
"""

import asyncio
import subprocess
import sys
import os
import time
import threading
import webbrowser
import json
import shutil
from pathlib import Path
import signal
import psutil
import socket

class ProductionScrollIntelLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.database_process = None
        self.redis_process = None
        self.running = True
        self.backend_port = 8000
        self.frontend_port = 3000
        
    def print_banner(self):
        """Print ScrollIntel production banner"""
        print("\n" + "="*70)
        print("SCROLLINTEL PRODUCTION STACK LAUNCHER")
        print("="*70)
        print("AI-Powered CTO Platform - Production Ready")
        print("Full Stack: Backend API + Next.js Frontend")
        print("Enterprise Features: Security, Monitoring, Analytics")
        print("="*70 + "\n")
    
    def check_system_requirements(self):
        """Check system requirements and dependencies"""
        print("Checking system requirements...")
        
        requirements_met = True
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print("ERROR: Python 3.8+ required")
            requirements_met = False
        else:
            print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check Node.js
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                node_version = result.stdout.strip()
                print(f"✓ Node.js {node_version}")
            else:
                print("WARNING: Node.js not found, will run backend only")
        except FileNotFoundError:
            print("WARNING: Node.js not installed, will run backend only")
        
        # Check npm
        try:
            result = subprocess.run(['npm', '--version'], capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                npm_version = result.stdout.strip()
                print(f"✓ npm {npm_version}")
            else:
                print("WARNING: npm not found, will try to continue")
        except FileNotFoundError:
            print("WARNING: npm not found, will try to continue")
        
        # Check available ports
        if not self.check_port_available(self.backend_port):
            print(f"WARNING: Port {self.backend_port} is busy, will find alternative")
            self.backend_port = self.find_available_port(8000)
        
        if not self.check_port_available(self.frontend_port):
            print(f"WARNING: Port {self.frontend_port} is busy, will find alternative")
            self.frontend_port = self.find_available_port(3000)
        
        return requirements_met
    
    def check_port_available(self, port):
        """Check if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def find_available_port(self, start_port, max_attempts=10):
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            if self.check_port_available(port):
                return port
        return None
    
    def setup_production_environment(self):
        """Setup production environment configuration"""
        print("Setting up production environment...")
        
        # Create production .env if it doesn't exist
        env_file = Path('.env')
        if not env_file.exists():
            self.create_production_env()
        
        # Set environment variables
        os.environ['NODE_ENV'] = 'production'
        os.environ['ENVIRONMENT'] = 'production'
        os.environ['DEBUG'] = 'false'
        os.environ['DATABASE_URL'] = 'sqlite:///./scrollintel_production.db'
        os.environ['JWT_SECRET_KEY'] = self.generate_secure_key()
        os.environ['API_PORT'] = str(self.backend_port)
        os.environ['FRONTEND_PORT'] = str(self.frontend_port)
        os.environ['NEXT_PUBLIC_API_URL'] = f'http://localhost:{self.backend_port}'
        
        print("✓ Production environment configured")
    
    def create_production_env(self):
        """Create production environment file"""
        env_content = f"""# ScrollIntel Production Environment
ENVIRONMENT=production
DEBUG=false
NODE_ENV=production

# Database
DATABASE_URL=sqlite:///./scrollintel_production.db

# Security
JWT_SECRET_KEY={self.generate_secure_key()}
ENCRYPTION_KEY={self.generate_secure_key()}

# API Configuration
API_HOST=0.0.0.0
API_PORT={self.backend_port}

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:{self.backend_port}
NEXT_PUBLIC_APP_NAME=ScrollIntel
NEXT_PUBLIC_APP_VERSION=1.0.0

# Features
ENABLE_MONITORING=true
ENABLE_ANALYTICS=true
ENABLE_SECURITY=true
ENABLE_CACHING=true

# Performance
MAX_WORKERS=4
ASYNC_POOL_SIZE=8
CACHE_SIZE=1000

# AI Services (Optional - Add your keys)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✓ Created production .env file")
    
    def generate_secure_key(self):
        """Generate a secure random key"""
        import secrets
        return secrets.token_urlsafe(32)
    
    def install_dependencies(self):
        """Install Python and Node.js dependencies"""
        print("Installing dependencies...")
        
        # Install Python dependencies
        print("Installing Python dependencies...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, capture_output=True)
            print("✓ Python dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install Python dependencies: {e}")
            return False
        
        # Install Node.js dependencies
        frontend_dir = Path("frontend")
        if frontend_dir.exists():
            print("Installing Node.js dependencies...")
            try:
                subprocess.run([
                    "npm", "install"
                ], cwd=frontend_dir, check=True, capture_output=True)
                print("✓ Node.js dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Failed to install Node.js dependencies: {e}")
                return False
        
        return True
    
    def initialize_database(self):
        """Initialize the production database"""
        print("Initializing production database...")
        
        try:
            # Create a simple database initialization
            subprocess.run([
                sys.executable, "-c", """
import sqlite3
import os

# Create database
db_path = 'scrollintel_production.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create basic tables
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        user_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_size INTEGER,
        mime_type TEXT,
        user_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS analytics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        event_data TEXT,
        user_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
''')

conn.commit()
conn.close()
print("Database initialized successfully")
"""
            ], check=True, capture_output=True)
            print("✓ Production database initialized")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Database initialization failed: {e}")
            return False
    
    def build_frontend(self):
        """Build the frontend for production"""
        print("Building frontend for production...")
        
        frontend_dir = Path("frontend")
        if not frontend_dir.exists():
            print("ERROR: Frontend directory not found")
            return False
        
        try:
            # Set environment variables for build
            env = os.environ.copy()
            env['NEXT_PUBLIC_API_URL'] = f'http://localhost:{self.backend_port}'
            env['NODE_ENV'] = 'production'
            
            # Build the frontend
            subprocess.run([
                "npm", "run", "build"
            ], cwd=frontend_dir, env=env, check=True, capture_output=True)
            
            print("✓ Frontend built successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Frontend build failed: {e}")
            return False
    
    def start_backend(self):
        """Start the production backend server"""
        print("Starting production backend server...")
        
        try:
            # Start the backend with production settings
            self.backend_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "scrollintel.api.production_ready_main:app",
                "--host", "0.0.0.0",
                "--port", str(self.backend_port),
                "--workers", "1",
                "--access-log",
                "--log-level", "info"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Wait for backend to start
            print(f"Waiting for backend to start on port {self.backend_port}...")
            time.sleep(10)
            
            # Check if backend is running
            if self.backend_process.poll() is None:
                print(f"✓ Backend API running at http://localhost:{self.backend_port}")
                return True
            else:
                stdout, stderr = self.backend_process.communicate()
                print(f"ERROR: Backend failed to start")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            print(f"ERROR: Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the production frontend server"""
        print("Starting production frontend server...")
        
        frontend_dir = Path("frontend")
        if not frontend_dir.exists():
            print("ERROR: Frontend directory not found")
            return False
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env['NODE_ENV'] = 'production'
            env['PORT'] = str(self.frontend_port)
            env['NEXT_PUBLIC_API_URL'] = f'http://localhost:{self.backend_port}'
            
            # Start the frontend in production mode
            self.frontend_process = subprocess.Popen([
                "npm", "start"
            ], cwd=frontend_dir, env=env,
               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Wait for frontend to start
            print(f"Waiting for frontend to start on port {self.frontend_port}...")
            time.sleep(15)
            
            # Check if frontend is running
            if self.frontend_process.poll() is None:
                print(f"✓ Frontend running at http://localhost:{self.frontend_port}")
                return True
            else:
                stdout, stderr = self.frontend_process.communicate()
                print(f"ERROR: Frontend failed to start")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            print(f"ERROR: Failed to start frontend: {e}")
            return False
    
    def run_health_checks(self):
        """Run comprehensive health checks"""
        print("Running health checks...")
        
        # Check backend health
        try:
            import requests
            response = requests.get(f'http://localhost:{self.backend_port}/health', timeout=5)
            if response.status_code == 200:
                print("✓ Backend health check passed")
            else:
                print(f"WARNING: Backend health check failed with status {response.status_code}")
        except Exception as e:
            print(f"WARNING: Backend health check failed: {e}")
        
        # Check frontend accessibility
        try:
            import requests
            response = requests.get(f'http://localhost:{self.frontend_port}', timeout=5)
            if response.status_code == 200:
                print("✓ Frontend accessibility check passed")
            else:
                print(f"WARNING: Frontend check failed with status {response.status_code}")
        except Exception as e:
            print(f"WARNING: Frontend check failed: {e}")
    
    def show_production_status(self):
        """Show production status and access points"""
        print("\n" + "="*70)
        print("SCROLLINTEL PRODUCTION STACK IS RUNNING!")
        print("="*70)
        print(f"🌐 Frontend Application:  http://localhost:{self.frontend_port}")
        print(f"🔧 Backend API:           http://localhost:{self.backend_port}")
        print(f"📚 API Documentation:     http://localhost:{self.backend_port}/docs")
        print(f"❤️  Health Check:         http://localhost:{self.backend_port}/health")
        print(f"📊 System Metrics:       http://localhost:{self.backend_port}/metrics")
        print("="*70)
        print("\n🚀 Production Features:")
        print("✓ Optimized Next.js frontend with SSR")
        print("✓ High-performance FastAPI backend")
        print("✓ Production database with SQLite")
        print("✓ Security middleware and authentication")
        print("✓ Performance monitoring and analytics")
        print("✓ Error handling and logging")
        print("✓ Responsive UI with Tailwind CSS")
        print("✓ Real-time WebSocket connections")
        print("\n💼 Enterprise Capabilities:")
        print("• AI-powered CTO agents and analysis")
        print("• Advanced data processing and ML models")
        print("• Interactive dashboards and visualizations")
        print("• File upload and processing system")
        print("• User management and role-based access")
        print("• Comprehensive API with OpenAPI docs")
        print("\n💡 Usage Tips:")
        print("- Access the main application at the frontend URL")
        print("- Use the API documentation for integration")
        print("- Monitor system health via health endpoints")
        print("- Use Ctrl+C to gracefully shutdown all services")
        print("\n🌟 ScrollIntel - Production AI Platform Ready!")
        print("="*70 + "\n")
    
    def open_browser(self):
        """Open the application in the default browser"""
        url = f"http://localhost:{self.frontend_port}"
        print(f"Opening ScrollIntel application: {url}")
        try:
            webbrowser.open(url)
            time.sleep(2)
            # Also open API docs in a new tab
            api_docs_url = f"http://localhost:{self.backend_port}/docs"
            webbrowser.open(api_docs_url)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open {url} manually in your browser")
    
    def monitor_processes(self):
        """Monitor all processes and restart if needed"""
        print("Monitoring services... (Press Ctrl+C to stop)")
        
        while self.running:
            try:
                # Check backend
                if self.backend_process and self.backend_process.poll() is not None:
                    print("WARNING: Backend process stopped unexpectedly")
                    print("Attempting to restart backend...")
                    if self.start_backend():
                        print("✓ Backend restarted successfully")
                    else:
                        print("ERROR: Failed to restart backend")
                        break
                
                # Check frontend
                if self.frontend_process and self.frontend_process.poll() is not None:
                    print("WARNING: Frontend process stopped unexpectedly")
                    print("Attempting to restart frontend...")
                    if self.start_frontend():
                        print("✓ Frontend restarted successfully")
                    else:
                        print("ERROR: Failed to restart frontend")
                        break
                
                # System resource monitoring
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                if cpu_percent > 80:
                    print(f"WARNING: High CPU usage: {cpu_percent}%")
                
                if memory_percent > 85:
                    print(f"WARNING: High memory usage: {memory_percent}%")
                
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)
    
    def cleanup(self):
        """Clean up all processes and resources"""
        print("\nShutting down ScrollIntel production stack...")
        self.running = False
        
        processes = [
            ("Frontend", self.frontend_process),
            ("Backend", self.backend_process),
        ]
        
        for name, process in processes:
            if process:
                print(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=15)
                    print(f"✓ {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}...")
                    process.kill()
                    print(f"✓ {name} force stopped")
        
        print("✓ All services stopped")
        print("ScrollIntel production stack shutdown complete")
    
    def run(self):
        """Main run method for production stack"""
        try:
            self.print_banner()
            
            # System checks
            if not self.check_system_requirements():
                print("System requirements not met. Please install missing dependencies.")
                return 1
            
            # Setup
            self.setup_production_environment()
            
            if not self.install_dependencies():
                print("Failed to install dependencies")
                return 1
            
            if not self.initialize_database():
                print("Failed to initialize database")
                return 1
            
            if not self.build_frontend():
                print("Failed to build frontend")
                return 1
            
            # Start services
            if not self.start_backend():
                print("Failed to start backend")
                return 1
            
            if not self.start_frontend():
                self.cleanup()
                return 1
            
            # Health checks
            time.sleep(5)
            self.run_health_checks()
            
            # Show status and open browser
            self.show_production_status()
            self.open_browser()
            
            # Monitor services
            self.monitor_processes()
            
        except KeyboardInterrupt:
            print("\nReceived shutdown signal")
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            self.cleanup()
        
        return 0

def main():
    """Main entry point"""
    launcher = ProductionScrollIntelLauncher()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        launcher.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    return launcher.run()

if __name__ == "__main__":
    sys.exit(main())