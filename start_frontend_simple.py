#!/usr/bin/env python3
"""
Simple frontend server for ScrollIntel
"""

import http.server
import socketserver
import webbrowser
import os
import threading
import time

def start_frontend_server():
    """Start a simple HTTP server for the frontend"""
    
    PORT = 3000
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=os.getcwd(), **kwargs)
        
        def end_headers(self):
            # Add CORS headers
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"🌐 Frontend server running at http://localhost:{PORT}")
            print(f"📁 Serving: {os.getcwd()}")
            print("🚀 Opening browser...")
            
            # Open browser after a short delay
            def open_browser():
                time.sleep(2)
                webbrowser.open(f'http://localhost:{PORT}/simple_frontend.html')
            
            threading.Thread(target=open_browser, daemon=True).start()
            
            print("\n💡 Press Ctrl+C to stop the server")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {PORT} is already in use")
            print("💡 Try stopping other services or use a different port")
        else:
            print(f"❌ Server error: {e}")

if __name__ == "__main__":
    start_frontend_server()