#!/usr/bin/env python3
"""
ScrollIntel Demo - Start Now
"""

print('🌟 ScrollIntel Demo Starting...')

# Simple HTTP server
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime
import random
import threading
import time
import webbrowser
import os

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        if self.path == '/health':
            response = {
                'status': 'healthy', 
                'timestamp': datetime.utcnow().isoformat(),
                'service': 'ScrollIntel API'
            }
        elif self.path == '/api/agents':
            response = [
                {
                    'id': 'cto-agent', 
                    'name': 'CTO Agent', 
                    'description': 'Strategic technology leadership and decision making',
                    'status': 'active',
                    'capabilities': ['Strategic Planning', 'Technology Assessment', 'Team Leadership']
                },
                {
                    'id': 'data-scientist', 
                    'name': 'Data Scientist Agent', 
                    'description': 'Advanced data analysis and machine learning',
                    'status': 'active',
                    'capabilities': ['Data Analysis', 'Machine Learning', 'Statistical Modeling']
                },
                {
                    'id': 'ml-engineer', 
                    'name': 'ML Engineer Agent', 
                    'description': 'Machine learning model development and deployment',
                    'status': 'active',
                    'capabilities': ['Model Development', 'MLOps', 'Performance Optimization']
                }
            ]
        elif self.path == '/api/monitoring/metrics':
            response = {
                'cpu_usage': round(random.uniform(20, 80), 1),
                'memory_usage': round(random.uniform(40, 85), 1),
                'disk_usage': round(random.uniform(30, 70), 1),
                'active_connections': random.randint(50, 200),
                'response_time': round(random.uniform(150, 300), 0),
                'uptime': 99.9
            }
        elif self.path == '/api/dashboard':
            response = {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'operational',
                'active_users': random.randint(10, 50),
                'total_requests_today': random.randint(1000, 5000),
                'success_rate': round(random.uniform(98.5, 99.9), 1)
            }
        else:
            response = {
                'message': 'ScrollIntel API is running', 
                'version': '1.0.0',
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat()
            }
        
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            post_data = self.rfile.read(content_length)
        else:
            post_data = b'{}'
            
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        if self.path == '/api/agents/chat':
            try:
                request_data = json.loads(post_data.decode())
                message = request_data.get('message', '')
                agent_id = request_data.get('agent_id', 'default')
                
                # Simple response logic
                if 'data' in message.lower():
                    response_content = 'I can help you analyze your data. Please upload your dataset and I will provide insights.'
                elif 'help' in message.lower():
                    response_content = 'I am here to help! You can ask me about data analysis, strategic planning, or upload files for processing.'
                elif 'hello' in message.lower() or 'hi' in message.lower():
                    response_content = f'Hello! I am the {agent_id.replace("-", " ").title()}. How can I assist you today?'
                else:
                    response_content = f'Thank you for your message: "{message}". I am ready to help you with any questions or tasks.'
                
                response = {
                    'id': f'msg_{int(datetime.now().timestamp())}',
                    'content': response_content,
                    'agent_id': agent_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'metadata': {
                        'processing_time': round(random.uniform(0.5, 2.0), 2),
                        'template_used': 'simple_response'
                    }
                }
            except Exception as e:
                response = {'error': f'Chat failed: {str(e)}'}
        else:
            response = {'content': 'Hello! I am your AI assistant. How can I help you today?'}
        
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def start_server():
    try:
        server = HTTPServer(('127.0.0.1', 8000), Handler)
        print('🚀 Backend running on http://127.0.0.1:8000')
        server.serve_forever()
    except Exception as e:
        print(f'❌ Server error: {e}')

def main():
    print('📁 Working directory:', os.getcwd())
    
    # Start server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    print('✅ ScrollIntel Backend is running!')
    print('🌐 API: http://127.0.0.1:8000')
    print('📊 Health: http://127.0.0.1:8000/health')
    print('🤖 Agents: http://127.0.0.1:8000/api/agents')
    print('📈 Metrics: http://127.0.0.1:8000/api/monitoring/metrics')
    print()
    
    # Try to open frontend
    frontend_file = 'simple_frontend.html'
    if os.path.exists(frontend_file):
        frontend_path = os.path.abspath(frontend_file)
        print(f'🎨 Opening frontend: {frontend_path}')
        try:
            webbrowser.open(f'file://{frontend_path}')
        except Exception as e:
            print(f'⚠️  Could not open browser: {e}')
            print(f'   Please manually open: {frontend_path}')
    else:
        print('⚠️  Frontend file not found, but backend is running')
    
    print()
    print('Press Ctrl+C to stop...')
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n🛑 Demo stopped by user')
        print('Thank you for using ScrollIntel!')

if __name__ == '__main__':
    main()