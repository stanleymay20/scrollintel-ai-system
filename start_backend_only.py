#!/usr/bin/env python3
"""
ScrollIntel Backend Only Launcher
Starts just the backend with a simple HTML frontend
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path
import signal

class BackendLauncher:
    def __init__(self):
        self.backend_process = None
        self.running = True
        
    def print_banner(self):
        """Print ScrollIntel banner"""
        print("\n" + "="*60)
        print("SCROLLINTEL BACKEND LAUNCHER")
        print("="*60)
        print("AI-Powered CTO Platform")
        print("Backend API with Simple Frontend")
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
    
    def create_simple_frontend(self):
        """Create a simple HTML frontend"""
        print("Creating simple HTML frontend...")
        
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ScrollIntel - AI-Powered CTO Platform</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        
        .container {
            text-align: center;
            max-width: 800px;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        h1 {
            font-size: 3rem;
            margin-bottom: 1rem;
            background: linear-gradient(45deg, #fff, #f0f0f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .subtitle {
            font-size: 1.2rem;
            margin-bottom: 2rem;
            opacity: 0.9;
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .feature {
            background: rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .feature h3 {
            margin-bottom: 0.5rem;
            color: #fff;
        }
        
        .feature p {
            opacity: 0.8;
            font-size: 0.9rem;
        }
        
        .links {
            margin-top: 2rem;
        }
        
        .btn {
            display: inline-block;
            padding: 12px 24px;
            margin: 0.5rem;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }
        
        .status {
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(0, 255, 0, 0.1);
            border: 1px solid rgba(0, 255, 0, 0.3);
            border-radius: 8px;
        }
        
        .api-test {
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        
        #apiResult {
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
            font-family: monospace;
            text-align: left;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>ScrollIntel</h1>
        <p class="subtitle">AI-Powered CTO Platform</p>
        
        <div class="features">
            <div class="feature">
                <h3>AI Agents</h3>
                <p>CTO, Data Scientist, ML Engineer, and more</p>
            </div>
            <div class="feature">
                <h3>Data Analysis</h3>
                <p>Upload and analyze your data with AI</p>
            </div>
            <div class="feature">
                <h3>ML Models</h3>
                <p>Build and deploy machine learning models</p>
            </div>
            <div class="feature">
                <h3>Dashboards</h3>
                <p>Interactive visualizations and reports</p>
            </div>
        </div>
        
        <div class="status">
            <strong>Backend Status:</strong> <span id="backendStatus">Checking...</span>
        </div>
        
        <div class="links">
            <a href="http://localhost:8000/docs" class="btn" target="_blank">API Documentation</a>
            <a href="http://localhost:8000/health" class="btn" target="_blank">Health Check</a>
            <a href="http://localhost:8000" class="btn" target="_blank">API Root</a>
        </div>
        
        <div class="api-test">
            <h3>Test API Connection</h3>
            <button onclick="testAPI()" class="btn">Test Backend API</button>
            <div id="apiResult"></div>
        </div>
    </div>
    
    <script>
        // Check backend status
        async function checkBackendStatus() {
            try {
                const response = await fetch('http://localhost:8000/health');
                if (response.ok) {
                    document.getElementById('backendStatus').textContent = 'Online';
                    document.getElementById('backendStatus').style.color = '#00ff00';
                } else {
                    document.getElementById('backendStatus').textContent = 'Error';
                    document.getElementById('backendStatus').style.color = '#ff0000';
                }
            } catch (error) {
                document.getElementById('backendStatus').textContent = 'Offline';
                document.getElementById('backendStatus').style.color = '#ff0000';
            }
        }
        
        // Test API connection
        async function testAPI() {
            const resultDiv = document.getElementById('apiResult');
            resultDiv.textContent = 'Testing API connection...';
            
            try {
                const response = await fetch('http://localhost:8000/');
                const data = await response.json();
                resultDiv.textContent = JSON.stringify(data, null, 2);
                resultDiv.style.color = '#00ff00';
            } catch (error) {
                resultDiv.textContent = 'Error: ' + error.message;
                resultDiv.style.color = '#ff0000';
            }
        }
        
        // Check status on load
        checkBackendStatus();
        
        // Check status every 30 seconds
        setInterval(checkBackendStatus, 30000);
    </script>
</body>
</html>"""
        
        with open('simple_frontend.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("Simple HTML frontend created: simple_frontend.html")
    
    def start_backend(self):
        """Start the backend server"""
        print("Starting backend server...")
        
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
    
    def show_status(self):
        """Show the current status and access points"""
        print("\n" + "="*60)
        print("SCROLLINTEL IS NOW RUNNING!")
        print("="*60)
        print("Frontend:        file:///" + os.path.abspath('simple_frontend.html'))
        print("Backend API:     http://localhost:8000")
        print("API Docs:        http://localhost:8000/docs")
        print("Health Check:    http://localhost:8000/health")
        print("="*60)
        print("\nQuick Start Guide:")
        print("1. Open the frontend HTML file in your browser")
        print("2. Test the API connection")
        print("3. Visit /docs for complete API documentation")
        print("4. Use the API endpoints to interact with ScrollIntel")
        print("\nTips:")
        print("- Use Ctrl+C to stop the backend")
        print("- Check logs in the terminal for debugging")
        print("- The frontend will auto-refresh backend status")
        print("\nScrollIntel - Where AI meets unlimited potential!")
        print("="*60 + "\n")
    
    def open_browser(self):
        """Open the application in the default browser"""
        frontend_path = os.path.abspath('simple_frontend.html')
        url = f"file:///{frontend_path}"
        print(f"Opening ScrollIntel frontend: {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open {frontend_path} manually in your browser")
    
    def monitor_backend(self):
        """Monitor backend process"""
        while self.running:
            try:
                # Check backend
                if self.backend_process and self.backend_process.poll() is not None:
                    print("Backend process stopped unexpectedly")
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
        
        print("ScrollIntel stopped successfully")
    
    def run(self):
        """Main run method"""
        try:
            self.print_banner()
            self.setup_simple_env()
            self.create_simple_frontend()
            
            if not self.start_backend():
                return 1
            
            self.show_status()
            self.open_browser()
            
            # Monitor backend
            self.monitor_backend()
            
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
    launcher = BackendLauncher()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        launcher.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    return launcher.run()

if __name__ == "__main__":
    sys.exit(main())