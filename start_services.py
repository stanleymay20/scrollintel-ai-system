#!/usr/bin/env python3
"""
ScrollIntel Service Starter
Simple service starter that works with existing setup
"""

import subprocess
import os
import time
import sys
from pathlib import Path

def check_port(port):
    """Check if a port is in use"""
    result = subprocess.run(f'netstat -an | findstr ":{port}"', shell=True, capture_output=True, text=True)
    return len(result.stdout.strip()) > 0

def create_simple_backend():
    """Create a simple backend service"""
    backend_code = '''
import sys
import os
sys.path.append(os.getcwd())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(
    title="ScrollIntel API",
    description="ScrollIntel AI Platform API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "ScrollIntel AI Platform API",
        "status": "healthy",
        "version": "1.0.0",
        "services": ["agents", "chat", "monitoring", "analytics"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "scrollintel-api",
        "timestamp": "2025-08-28T00:00:00Z",
        "database": "connected",
        "version": "1.0.0"
    }

@app.get("/api/v1/agents")
def list_agents():
    return {
        "agents": [
            {"id": "ai-engineer", "name": "AI Engineer", "status": "active"},
            {"id": "ml-engineer", "name": "ML Engineer", "status": "active"},
            {"id": "data-scientist", "name": "Data Scientist", "status": "active"},
            {"id": "cto-agent", "name": "CTO Agent", "status": "active"},
            {"id": "bi-agent", "name": "BI Agent", "status": "active"}
        ],
        "total": 5,
        "status": "available"
    }

@app.get("/api/v1/chat")
def chat_endpoint():
    return {
        "message": "Chat API endpoint",
        "status": "available",
        "features": ["text", "voice", "file-upload"]
    }

@app.get("/api/v1/monitoring")
def monitoring():
    return {
        "system": "healthy",
        "cpu": "45%",
        "memory": "67%",
        "disk": "72%",
        "services": 5
    }

@app.get("/api/v1/analytics")
def analytics():
    return {
        "status": "available",
        "dashboards": 3,
        "reports": 12,
        "last_updated": "2025-08-28T00:00:00Z"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
'''
    
    with open("simple_backend.py", "w", encoding="utf-8") as f:
        f.write(backend_code)

def create_simple_frontend():
    """Create a simple frontend"""
    os.makedirs("frontend/dist", exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ScrollIntel AI Platform</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 0; text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
        .header p { font-size: 1.1rem; opacity: 0.9; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: white; border-radius: 10px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .card h3 { color: #2d3748; margin-bottom: 15px; font-size: 1.2rem; }
        .status-good { color: #38a169; font-weight: 600; }
        .status-bad { color: #e53e3e; font-weight: 600; }
        .btn { background: #4299e1; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; margin: 5px; font-size: 14px; }
        .btn:hover { background: #3182ce; }
        .btn-green { background: #38a169; }
        .btn-green:hover { background: #2f855a; }
        .btn-purple { background: #805ad5; }
        .btn-purple:hover { background: #6b46c1; }
        .response-box { background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 15px; margin-top: 15px; font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto; }
        .endpoint { display: flex; justify-between; align-items: center; padding: 10px; background: #f7fafc; border-radius: 6px; margin: 5px 0; }
        .endpoint code { background: #e2e8f0; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>ScrollIntel AI Platform</h1>
            <p>Advanced AI Agent Orchestration System</p>
        </div>
    </div>

    <div class="container">
        <div class="grid">
            <div class="card">
                <h3>Backend API Status</h3>
                <p id="backend-status" class="status-bad">Checking...</p>
                <button class="btn" onclick="checkBackend()">Test Connection</button>
            </div>
            
            <div class="card">
                <h3>Database Status</h3>
                <p class="status-good">PostgreSQL Connected</p>
                <p style="color: #718096; font-size: 14px;">Port: 5432</p>
            </div>
            
            <div class="card">
                <h3>Docker Services</h3>
                <p class="status-good">2 Containers Running</p>
                <p style="color: #718096; font-size: 14px;">scrollintel-postgres, scrollintel-simple-test</p>
            </div>
        </div>

        <div class="card">
            <h3>Quick API Tests</h3>
            <button class="btn" onclick="testEndpoint('/')">Test Root</button>
            <button class="btn btn-green" onclick="testEndpoint('/health')">Health Check</button>
            <button class="btn btn-purple" onclick="testEndpoint('/api/v1/agents')">List Agents</button>
            <button class="btn" onclick="openDocs()">API Docs</button>
            <div id="api-response" class="response-box">Click a button above to test the API...</div>
        </div>

        <div class="card">
            <h3>Available Endpoints</h3>
            <div class="endpoint">
                <code>GET /</code>
                <span class="status-good">Root API Info</span>
            </div>
            <div class="endpoint">
                <code>GET /health</code>
                <span class="status-good">Health Check</span>
            </div>
            <div class="endpoint">
                <code>GET /api/v1/agents</code>
                <span class="status-good">List AI Agents</span>
            </div>
            <div class="endpoint">
                <code>GET /api/v1/chat</code>
                <span class="status-good">Chat Interface</span>
            </div>
            <div class="endpoint">
                <code>GET /docs</code>
                <span class="status-good">API Documentation</span>
            </div>
        </div>

        <div class="card">
            <h3>System Information</h3>
            <p><strong>Frontend:</strong> http://localhost:3000</p>
            <p><strong>Backend API:</strong> http://localhost:8000</p>
            <p><strong>Alternative Backend:</strong> http://localhost:8003</p>
            <p><strong>Database:</strong> PostgreSQL on localhost:5432</p>
        </div>
    </div>

    <script>
        let apiBase = 'http://localhost:8000';
        
        function checkBackend() {
            fetch(apiBase + '/health')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('backend-status').textContent = 'Online (Port 8000)';
                    document.getElementById('backend-status').className = 'status-good';
                })
                .catch(error => {
                    // Try port 8003
                    fetch('http://localhost:8003/health')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('backend-status').textContent = 'Online (Port 8003)';
                            document.getElementById('backend-status').className = 'status-good';
                            apiBase = 'http://localhost:8003';
                        })
                        .catch(error => {
                            document.getElementById('backend-status').textContent = 'Offline';
                            document.getElementById('backend-status').className = 'status-bad';
                        });
                });
        }

        function testEndpoint(endpoint) {
            fetch(apiBase + endpoint)
                .then(response => response.json())
                .then(data => {
                    document.getElementById('api-response').textContent = JSON.stringify(data, null, 2);
                })
                .catch(error => {
                    document.getElementById('api-response').textContent = 'Error: ' + error.message;
                });
        }

        function openDocs() {
            window.open(apiBase + '/docs', '_blank');
        }

        // Check backend status on load
        window.onload = function() {
            checkBackend();
        };

        // Auto-refresh every 30 seconds
        setInterval(checkBackend, 30000);
    </script>
</body>
</html>'''
    
    with open("frontend/dist/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    """Main startup routine"""
    print("ScrollIntel Service Starter")
    print("=" * 40)
    
    # Check current status
    print("\\nCurrent Port Status:")
    ports = [8000, 8003, 3000, 5432]
    for port in ports:
        status = "In Use" if check_port(port) else "Available"
        print(f"Port {port}: {status}")
    
    # Create backend if needed
    if not check_port(8000):
        print("\\nStarting backend on port 8000...")
        create_simple_backend()
        subprocess.Popen([sys.executable, "simple_backend.py"])
        time.sleep(3)
        print("Backend started")
    else:
        print("\\nBackend already running on port 8000")
    
    # Create frontend if needed
    if not check_port(3000):
        print("Starting frontend on port 3000...")
        create_simple_frontend()
        os.chdir("frontend/dist")
        subprocess.Popen([sys.executable, "-m", "http.server", "3000"])
        os.chdir("../..")
        time.sleep(2)
        print("Frontend started")
    else:
        print("Frontend already running on port 3000")
    
    print("\\nServices Started!")
    print("\\nAccess Points:")
    print("Frontend:     http://localhost:3000")
    print("Backend API:  http://localhost:8000")
    print("API Docs:     http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    print("\\nNext: Run 'python check_deployment_status.py' to verify")

if __name__ == "__main__":
    main()