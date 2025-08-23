#!/usr/bin/env python3
"""
Fix ScrollIntel Frontend Runtime Error
This script identifies and fixes common runtime errors in the frontend.
"""

import os
import sys
import subprocess
import json
import time
import requests
from pathlib import Path

def check_backend_health():
    """Check if the backend API is running"""
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def start_simple_backend():
    """Start a simple backend server for development"""
    print("🚀 Starting simple backend server...")
    
    # Create a simple FastAPI server
    simple_backend_code = '''
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import random

app = FastAPI(title="ScrollIntel Simple Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/agents")
async def get_agents():
    return {
        "success": True,
        "data": [
            {
                "id": "cto-agent",
                "name": "CTO Agent",
                "description": "Strategic technology leadership and decision making",
                "status": "active",
                "capabilities": ["Strategic Planning", "Technology Assessment", "Team Leadership"]
            },
            {
                "id": "data-scientist",
                "name": "Data Scientist",
                "description": "Advanced data analysis and machine learning",
                "status": "active",
                "capabilities": ["Data Analysis", "Machine Learning", "Statistical Modeling"]
            }
        ]
    }

@app.get("/api/monitoring/metrics")
async def get_system_metrics():
    return {
        "success": True,
        "data": {
            "cpu_usage": random.randint(30, 70),
            "memory_usage": random.randint(40, 80),
            "disk_usage": random.randint(20, 60),
            "active_connections": random.randint(100, 200),
            "response_time": random.randint(200, 400),
            "uptime": 99.9
        }
    }

@app.post("/api/agents/chat")
async def chat_with_agent(request: dict):
    message = request.get("message", "")
    agent_id = request.get("agent_id", "default")
    
    # Simple response based on message content
    if "hello" in message.lower():
        response = f"Hello! I'm the {agent_id} agent. How can I help you today?"
    elif "data" in message.lower():
        response = "I can help you analyze data, create visualizations, and provide insights. What data would you like to work with?"
    elif "help" in message.lower():
        response = "I'm here to help! You can ask me about data analysis, strategic planning, or upload files for analysis."
    else:
        response = f"I understand you're asking about: {message}. Let me help you with that!"
    
    return {
        "success": True,
        "data": {
            "id": f"msg_{int(time.time())}",
            "content": response,
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    # Write the simple backend
    with open('simple_backend_fix.py', 'w') as f:
        f.write(simple_backend_code)
    
    return 'simple_backend_fix.py'

def fix_frontend_config():
    """Fix frontend configuration issues"""
    print("🔧 Fixing frontend configuration...")
    
    # Check if .env.local exists in frontend
    frontend_env = Path('frontend/.env.local')
    if not frontend_env.exists():
        env_content = '''NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=ScrollIntel
NEXT_PUBLIC_APP_VERSION=4.0.0
'''
        frontend_env.write_text(env_content)
        print("✅ Created frontend/.env.local")
    
    # Update next.config.js to handle API errors gracefully
    next_config_path = Path('frontend/next.config.js')
    if next_config_path.exists():
        next_config_content = '''/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ]
  },
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET, POST, PUT, DELETE, OPTIONS' },
          { key: 'Access-Control-Allow-Headers', value: 'Content-Type, Authorization' },
        ],
      },
    ]
  },
}

module.exports = nextConfig
'''
        next_config_path.write_text(next_config_content)
        print("✅ Updated next.config.js")

def main():
    print("🔍 ScrollIntel Runtime Error Fix\n")
    
    # Check if backend is running
    if check_backend_health():
        print("✅ Backend is running and healthy")
    else:
        print("❌ Backend is not running")
        print("🚀 Starting simple backend for development...")
        
        # Start simple backend
        backend_file = start_simple_backend()
        print(f"✅ Created {backend_file}")
        print("📝 To start the backend, run:")
        print(f"   python {backend_file}")
        print()
    
    # Fix frontend configuration
    fix_frontend_config()
    
    print("\n🎯 Next Steps:")
    print("1. Start the backend: python simple_backend_fix.py")
    print("2. In another terminal, go to frontend directory: cd frontend")
    print("3. Install dependencies: npm install")
    print("4. Start frontend: npm run dev")
    print("5. Open http://localhost:3000 in your browser")
    print("\n✨ This should resolve the runtime error!")

if __name__ == "__main__":
    main()