#!/usr/bin/env python3
"""
Simple HTTP server for ScrollIntel development
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from datetime import datetime
import random

class ScrollIntelHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        
        # Add CORS headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        # Route handling
        if self.path == '/':
            response = {
                "message": "ScrollIntel API is running",
                "version": "1.0.0",
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat()
            }
        elif self.path == '/health':
            response = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "ScrollIntel API"
            }
        elif self.path == '/api/monitoring/metrics':
            response = {
                "cpu_usage": round(random.uniform(20, 80), 1),
                "memory_usage": round(random.uniform(40, 85), 1),
                "disk_usage": round(random.uniform(30, 70), 1),
                "active_connections": random.randint(50, 200),
                "response_time": round(random.uniform(150, 300), 0),
                "uptime": 99.9
            }
        elif self.path == '/api/agents':
            response = [
                {
                    "id": "cto-agent",
                    "name": "CTO Agent",
                    "description": "Strategic technology leadership and decision making",
                    "status": "active",
                    "capabilities": ["Strategic Planning", "Technology Assessment", "Team Leadership"],
                    "last_active": datetime.utcnow().isoformat()
                },
                {
                    "id": "data-scientist",
                    "name": "Data Scientist Agent",
                    "description": "Advanced data analysis and machine learning",
                    "status": "active",
                    "capabilities": ["Data Analysis", "Machine Learning", "Statistical Modeling"],
                    "last_active": datetime.utcnow().isoformat()
                },
                {
                    "id": "ml-engineer",
                    "name": "ML Engineer Agent",
                    "description": "Machine learning model development and deployment",
                    "status": "active",
                    "capabilities": ["Model Development", "MLOps", "Performance Optimization"],
                    "last_active": datetime.utcnow().isoformat()
                }
            ]
        elif self.path == '/api/dashboard':
            response = {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "operational",
                "active_users": random.randint(10, 50),
                "total_requests_today": random.randint(1000, 5000),
                "success_rate": round(random.uniform(98.5, 99.9), 1)
            }
        else:
            response = {"error": "Not found", "path": self.path}
        
        # Send response
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def do_POST(self):
        """Handle POST requests"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # Add CORS headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        if self.path == '/api/agents/chat':
            try:
                request_data = json.loads(post_data.decode())
                message = request_data.get("message", "")
                agent_id = request_data.get("agent_id", "default")
                
                # Simple response logic
                if "data" in message.lower():
                    response_content = "I can help you analyze your data. Please upload your dataset and I'll provide insights."
                elif "help" in message.lower():
                    response_content = "I'm here to help! You can ask me about data analysis, strategic planning, or upload files for processing."
                else:
                    response_content = f"Hello! I'm the {agent_id.replace('-', ' ').title()}. How can I assist you today?"
                
                response = {
                    "id": f"msg_{int(datetime.now().timestamp())}",
                    "content": response_content,
                    "agent_id": agent_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {
                        "processing_time": round(random.uniform(0.5, 2.0), 2),
                        "template_used": "simple_response"
                    }
                }
            except Exception as e:
                response = {"error": f"Chat failed: {str(e)}"}
        else:
            response = {"error": "Not found", "path": self.path}
        
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def main():
    """Start the simple server"""
    port = 8000
    
    # Try to find an available port
    import socket
    for test_port in [8000, 8001, 8002, 8003, 8080]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', test_port))
                port = test_port
                break
        except OSError:
            continue
    
    server = HTTPServer(('127.0.0.1', port), ScrollIntelHandler)
    
    print("🚀 ScrollIntel Simple Backend Server")
    print("=" * 40)
    print(f"🌐 Server running on: http://127.0.0.1:{port}")
    print(f"📚 API endpoints available:")
    print(f"   - GET  /health")
    print(f"   - GET  /api/agents")
    print(f"   - GET  /api/monitoring/metrics")
    print(f"   - POST /api/agents/chat")
    print(f"   - GET  /api/dashboard")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        server.shutdown()

if __name__ == "__main__":
    main()