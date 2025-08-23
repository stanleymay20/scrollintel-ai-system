#!/usr/bin/env python3
"""
Simple ScrollIntel Stack Launcher
Starts both frontend and backend with minimal dependencies
"""

import subprocess
import sys
import os
import time
import threading
import webbrowser
from pathlib import Path
import signal

class SimpleScrollIntelLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        
    def print_banner(self):
        """Print ScrollIntel banner"""
        print("\n" + "="*60)
        print("SCROLLINTEL SIMPLE STACK LAUNCHER")
        print("="*60)
        print("AI-Powered CTO Platform")
        print("Advanced Analytics & ML Models")
        print("Optimized UI/UX Experience")
        print("="*60 + "\n")
    
    def setup_simple_env(self):
        """Setup simple environment for local development"""
        print("Setting up simple environment...")
        
        # Set environment variables for simple local setup
        os.environ['DATABASE_URL'] = 'sqlite:///./scrollintel.db'
        os.environ['DEBUG'] = 'true'
        os.environ['ENVIRONMENT'] = 'development'
        os.environ['JWT_SECRET_KEY'] = 'simple-dev-secret-key-not-for-production'
        
        print("Simple environment configured")
    
    def start_simple_backend(self):
        """Start the simple backend server"""
        print("Starting simple backend server...")
        
        try:
            # Start the simple backend
            self.backend_process = subprocess.Popen([
                sys.executable, "run_simple.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Wait a bit for backend to start
            time.sleep(8)
            
            # Check if backend is running
            if self.backend_process.poll() is None:
                print("Backend API running at http://localhost:8000")
                print("API Documentation: http://localhost:8000/docs")
                return True
            else:
                stdout, stderr = self.backend_process.communicate()
                print(f"Backend failed to start")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            print(f"Error starting backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the frontend development server"""
        print("Starting frontend development server...")
        
        frontend_dir = Path("frontend")
        if not frontend_dir.exists():
            print("Frontend directory not found")
            return False
        
        try:
            # Check if node_modules exists
            node_modules = frontend_dir / "node_modules"
            if not node_modules.exists():
                print("Installing frontend dependencies...")
                subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
                print("Frontend dependencies installed")
            
            # Set environment variables for frontend
            env = os.environ.copy()
            env['NEXT_PUBLIC_API_URL'] = 'http://localhost:8000'
            env['PORT'] = '3000'
            
            # Start frontend server
            self.frontend_process = subprocess.Popen([
                "npm", "run", "dev"
            ], cwd=frontend_dir, env=env,
               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Wait for frontend to start
            print("Waiting for frontend to start...")
            time.sleep(15)
            
            # Check if frontend is running
            if self.frontend_process.poll() is None:
                print("Frontend running at http://localhost:3000")
                return True
            else:
                stdout, stderr = self.frontend_process.communicate()
                print(f"Frontend failed to start")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            print(f"Error starting frontend: {e}")
            return False
    
    def show_status(self):
        """Show the current status and access points"""
        print("\n" + "="*60)
        print("SCROLLINTEL IS NOW RUNNING!")
        print("="*60)
        print("Frontend:        http://localhost:3000")
        print("Backend API:     http://localhost:8000")
        print("API Docs:        http://localhost:8000/docs")
        print("Health Check:    http://localhost:8000/health")
        print("="*60)
        print("\nQuick Start Guide:")
        print("1. Upload your data files (CSV, Excel, JSON)")
        print("2. Chat with AI agents for insights")
        print("3. Build ML models with AutoML")
        print("4. Create interactive dashboards")
        print("5. Export results and reports")
        print("\nTips:")
        print("- Use Ctrl+C to stop all services")
        print("- Check logs in the terminal for debugging")
        print("- Visit /docs for complete API documentation")
        print("\nScrollIntel - Where AI meets unlimited potential!")
        print("="*60 + "\n")
    
    def open_browser(self):
        """Open the application in the default browser"""
        url = "http://localhost:3000"
        print(f"Opening ScrollIntel in your browser: {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open {url} manually in your browser")
    
    def monitor_processes(self):
        """Monitor backend and frontend processes"""
        while self.running:
            try:
                # Check backend
                if self.backend_process and self.backend_process.poll() is not None:
                    print("Backend process stopped unexpectedly")
                    break
                
                # Check frontend
                if self.frontend_process and self.frontend_process.poll() is not None:
                    print("Frontend process stopped unexpectedly")
                    break
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                break
    
    def cleanup(self):
        """Clean up processes"""
        print("\nShutting down ScrollIntel...")
        self.running = False
        
        if self.backend_process:
            print("Stopping backend...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
            print("Backend stopped")
        
        if self.frontend_process:
            print("Stopping frontend...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
            print("Frontend stopped")
        
        print("ScrollIntel stopped successfully")
    
    def run(self):
        """Main run method"""
        try:
            self.print_banner()
            self.setup_simple_env()
            
            if not self.start_simple_backend():
                return 1
            
            if not self.start_frontend():
                self.cleanup()
                return 1
            
            self.show_status()
            self.open_browser()
            
            # Monitor processes
            self.monitor_processes()
            
        except KeyboardInterrupt:
            print("\nReceived interrupt signal")
        except Exception as e:
            print(f"Unexpected error: {e}")
            return 1
        finally:
            self.cleanup()
        
        return 0

def main():
    """Main entry point"""
    launcher = SimpleScrollIntelLauncher()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        launcher.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    return launcher.run()

if __name__ == "__main__":
    sys.exit(main())