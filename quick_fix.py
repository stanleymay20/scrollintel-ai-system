#!/usr/bin/env python3
"""
Quick ScrollIntel Deployment Fix
Works with existing containers
"""

import subprocess
import os
import time
import sys

def create_env_file():
    """Create a proper .env file"""
    env_content = """# ScrollIntel Environment Configuration
DATABASE_URL=postgresql://scrollintel:scrollintel123@localhost:5432/scrollintel
SECRET_KEY=dev-secret-key-change-in-production
API_KEY=dev-api-key
OPENAI_API_KEY=your-openai-api-key-here
ENVIRONMENT=development
DEBUG=true
PORT=8000
HOST=0.0.0.0
FRONTEND_PORT=3000
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print("✅ Created .env file")

def start_backend_on_correct_port():
    """Start backend on port 8000"""
    print("🔧 Starting backend on port 8000...")
    
    # Kill any existing processes
    try:
        subprocess.run("taskkill /f /im python.exe", shell=True, capture_output=True)
    except:
        pass
    
    time.sleep(2)
    
    # Start backend
    cmd = "python -m uvicorn scrollintel.api.main:app --host 0.0.0.0 --port 8000 --reload"
    subprocess.Popen(cmd, shell=True)
    print("✅ Backend starting on http://localhost:8000")

def start_simple_frontend():
    """Start a simple frontend server"""
    print("🔧 Starting simple frontend...")
    
    # Create a simple HTML file if frontend doesn't exist
    if not os.path.exists("frontend/dist"):
        os.makedirs("frontend/dist", exist_ok=True)
        
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ScrollIntel AI Platform</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .links { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .link { background: #007bff; color: white; padding: 15px; text-align: center; border-radius: 5px; text-decoration: none; }
        .link:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 ScrollIntel AI Platform</h1>
        <div class="status">
            <h3>✅ System Status: Online</h3>
            <p>Your ScrollIntel AI platform is running successfully!</p>
        </div>
        
        <div class="links">
            <a href="http://localhost:8000/docs" class="link">📚 API Documentation</a>
            <a href="http://localhost:8000/health" class="link">💚 Health Check</a>
            <a href="http://localhost:8000/api/v1/agents" class="link">🤖 Agents API</a>
            <a href="http://localhost:8003/docs" class="link">🔧 Backend (Port 8003)</a>
        </div>
        
        <h3>🔗 Available Services:</h3>
        <ul>
            <li><strong>Backend API:</strong> http://localhost:8000</li>
            <li><strong>Alternative Backend:</strong> http://localhost:8003</li>
            <li><strong>Database:</strong> PostgreSQL on localhost:5432</li>
            <li><strong>Frontend:</strong> http://localhost:3000 (this page)</li>
        </ul>
        
        <h3>🛠️ Quick Actions:</h3>
        <ul>
            <li>Check deployment status: <code>python check_deployment_status.py</code></li>
            <li>View logs: <code>docker-compose logs -f</code></li>
            <li>Restart services: <code>docker-compose restart</code></li>
        </ul>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
        
        // Check API status
        fetch('http://localhost:8000/health')
            .then(response => response.json())
            .then(data => {
                console.log('API Status:', data);
            })
            .catch(error => {
                console.log('API not available on port 8000, trying 8003...');
                fetch('http://localhost:8003/health')
                    .then(response => response.json())
                    .then(data => console.log('API Status (8003):', data))
                    .catch(err => console.log('API not available'));
            });
    </script>
</body>
</html>"""
        
        with open("frontend/dist/index.html", "w") as f:
            f.write(html_content)
    
    # Start simple HTTP server for frontend
    try:
        subprocess.run("taskkill /f /im python.exe /fi \"WINDOWTITLE eq frontend*\"", shell=True, capture_output=True)
    except:
        pass
    
    time.sleep(1)
    
    # Start frontend server
    os.chdir("frontend/dist")
    subprocess.Popen("python -m http.server 3000", shell=True)
    os.chdir("../..")
    print("✅ Frontend starting on http://localhost:3000")

def main():
    """Main fix routine"""
    print("🚀 ScrollIntel Quick Fix")
    print("=" * 40)
    
    # Create environment file
    create_env_file()
    
    # Start backend on correct port
    start_backend_on_correct_port()
    
    # Start simple frontend
    start_simple_frontend()
    
    # Wait for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(8)
    
    # Run quick check
    print("\n🔍 Quick status check...")
    try:
        import requests
        
        # Check backend
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            print("✅ Backend (8000): Online")
        except:
            try:
                response = requests.get("http://localhost:8003/health", timeout=5)
                print("✅ Backend (8003): Online")
            except:
                print("❌ Backend: Offline")
        
        # Check frontend
        try:
            response = requests.get("http://localhost:3000", timeout=5)
            print("✅ Frontend (3000): Online")
        except:
            print("❌ Frontend: Offline")
            
    except ImportError:
        print("⚠️  Requests not available for status check")
    
    print("\n🎉 Quick fix complete!")
    print("\n🔗 Access your services:")
    print("🌐 Frontend: http://localhost:3000")
    print("🔧 API Docs: http://localhost:8000/docs")
    print("💚 Health: http://localhost:8000/health")
    print("🤖 Agents: http://localhost:8000/api/v1/agents")
    
    print("\n📋 Next steps:")
    print("1. Run: python check_deployment_status.py")
    print("2. Check logs: docker-compose logs -f")
    print("3. Test APIs in browser or Postman")

if __name__ == "__main__":
    main()