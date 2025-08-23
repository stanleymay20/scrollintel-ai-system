#!/usr/bin/env python3
"""
ScrollIntel Backend Production Launcher
Production-ready backend with comprehensive features
"""

import subprocess
import sys
import os
import time
import webbrowser
import socket
from pathlib import Path
import signal

class BackendProductionLauncher:
    def __init__(self):
        self.backend_process = None
        self.running = True
        self.backend_port = 8000
        
    def print_banner(self):
        """Print ScrollIntel production banner"""
        print("\n" + "="*70)
        print("SCROLLINTEL BACKEND PRODUCTION LAUNCHER")
        print("="*70)
        print("AI-Powered CTO Platform - Backend API")
        print("Production Ready FastAPI with Full Features")
        print("="*70 + "\n")
    
    def find_available_port(self, start_port, max_attempts=10):
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        return None
    
    def setup_production_environment(self):
        """Setup production environment configuration"""
        print("Setting up production environment...")
        
        # Find available port
        self.backend_port = self.find_available_port(8000)
        if not self.backend_port:
            print("ERROR: Could not find available port")
            return False
        
        # Set environment variables
        os.environ['NODE_ENV'] = 'production'
        os.environ['ENVIRONMENT'] = 'production'
        os.environ['DEBUG'] = 'false'
        os.environ['DATABASE_URL'] = 'sqlite:///./scrollintel_production.db'
        os.environ['JWT_SECRET_KEY'] = 'production-secret-key-change-in-real-deployment'
        os.environ['API_PORT'] = str(self.backend_port)
        
        print(f"✓ Production environment configured on port {self.backend_port}")
        return True
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("Installing Python dependencies...")
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "fastapi", "uvicorn[standard]", "python-multipart", 
                "psutil", "requests"
            ], check=True, capture_output=True)
            print("✓ Python dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install Python dependencies: {e}")
            return False
    
    def initialize_database(self):
        """Initialize the production database"""
        print("Initializing production database...")
        
        try:
            import sqlite3
            
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
            
            print("✓ Production database initialized")
            return True
        except Exception as e:
            print(f"ERROR: Database initialization failed: {e}")
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
            time.sleep(8)
            
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
    
    def run_health_checks(self):
        """Run comprehensive health checks"""
        print("Running health checks...")
        
        try:
            import requests
            response = requests.get(f'http://localhost:{self.backend_port}/health', timeout=5)
            if response.status_code == 200:
                print("✓ Backend health check passed")
                health_data = response.json()
                print(f"  Database: {health_data.get('checks', {}).get('database', 'unknown')}")
            else:
                print(f"WARNING: Backend health check failed with status {response.status_code}")
        except Exception as e:
            print(f"WARNING: Backend health check failed: {e}")
    
    def show_production_status(self):
        """Show production status and access points"""
        print("\n" + "="*70)
        print("SCROLLINTEL BACKEND PRODUCTION IS RUNNING!")
        print("="*70)
        print(f"🔧 Backend API:           http://localhost:{self.backend_port}")
        print(f"📚 API Documentation:     http://localhost:{self.backend_port}/docs")
        print(f"📖 ReDoc Documentation:   http://localhost:{self.backend_port}/redoc")
        print(f"❤️  Health Check:         http://localhost:{self.backend_port}/health")
        print(f"📊 System Metrics:       http://localhost:{self.backend_port}/metrics")
        print("="*70)
        print("\n🚀 Production Features:")
        print("✓ High-performance FastAPI backend")
        print("✓ Production SQLite database")
        print("✓ Comprehensive API endpoints")
        print("✓ Health monitoring and metrics")
        print("✓ Interactive API documentation")
        print("✓ Error handling and logging")
        print("✓ CORS and security middleware")
        print("\n💼 Available Endpoints:")
        print("• GET  /              - API information")
        print("• GET  /health        - Health check")
        print("• GET  /metrics       - System metrics")
        print("• GET  /api/agents    - Available AI agents")
        print("• POST /api/analyze   - Data analysis")
        print("• POST /api/files/upload - File upload")
        print("• GET  /api/dashboard - Dashboard data")
        print("• POST /api/auth/login - User authentication")
        print("\n💡 Usage Tips:")
        print("- Visit /docs for interactive API documentation")
        print("- Use /health to monitor system status")
        print("- Check /metrics for performance data")
        print("- Use Ctrl+C to gracefully shutdown")
        print("\n🌟 ScrollIntel Backend - Production Ready!")
        print("="*70 + "\n")
    
    def open_browser(self):
        """Open the API documentation in browser"""
        api_url = f"http://localhost:{self.backend_port}"
        docs_url = f"http://localhost:{self.backend_port}/docs"
        
        print(f"Opening API documentation: {docs_url}")
        try:
            webbrowser.open(docs_url)
            time.sleep(2)
            webbrowser.open(api_url)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open {docs_url} manually in your browser")
    
    def monitor_process(self):
        """Monitor backend process"""
        print("Monitoring backend service... (Press Ctrl+C to stop)")
        
        while self.running:
            try:
                if self.backend_process and self.backend_process.poll() is not None:
                    print("WARNING: Backend process stopped unexpectedly")
                    break
                
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)
    
    def cleanup(self):
        """Clean up backend process"""
        print("\nShutting down ScrollIntel backend...")
        self.running = False
        
        if self.backend_process:
            print("Stopping backend...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=15)
                print("✓ Backend stopped gracefully")
            except subprocess.TimeoutExpired:
                print("Force killing backend...")
                self.backend_process.kill()
                print("✓ Backend force stopped")
        
        print("✓ ScrollIntel backend shutdown complete")
    
    def run(self):
        """Main run method"""
        try:
            self.print_banner()
            
            if not self.setup_production_environment():
                return 1
            
            if not self.install_dependencies():
                print("Failed to install dependencies")
                return 1
            
            if not self.initialize_database():
                print("Failed to initialize database")
                return 1
            
            if not self.start_backend():
                print("Failed to start backend")
                return 1
            
            # Health checks
            time.sleep(3)
            self.run_health_checks()
            
            # Show status and open browser
            self.show_production_status()
            self.open_browser()
            
            # Monitor service
            self.monitor_process()
            
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
    launcher = BackendProductionLauncher()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        launcher.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    return launcher.run()

if __name__ == "__main__":
    sys.exit(main())