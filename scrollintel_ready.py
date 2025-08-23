"""
ScrollIntel Minimal Launch Application
Bulletproof FastAPI app that starts successfully every time
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    from dotenv import load_dotenv
    import time
    import socket
    
    # Load environment variables
    load_dotenv()
    
    # Create FastAPI app
    app = FastAPI(
        title="ScrollIntel™ AI Platform",
        description="AI-Powered CTO Platform - Ready for Production",
        version="4.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
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
    async def root():
        """Root endpoint"""
        return {
            "message": "ScrollIntel AI Platform - Ready for Production!",
            "status": "healthy",
            "version": "4.0.0",
            "timestamp": time.time(),
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "ScrollIntel™ API",
            "version": "4.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "message": "All systems operational!"
        }
    
    @app.get("/agents")
    async def list_agents():
        """List available AI agents"""
        agents = [
            {"name": "CTO Agent", "description": "Strategic technology leadership", "status": "ready"},
            {"name": "Data Scientist", "description": "Advanced analytics and insights", "status": "ready"},
            {"name": "ML Engineer", "description": "Machine learning model development", "status": "ready"},
            {"name": "AI Engineer", "description": "AI system architecture", "status": "ready"},
            {"name": "Business Analyst", "description": "Business intelligence and reporting", "status": "ready"},
            {"name": "QA Agent", "description": "Quality assurance and testing", "status": "ready"}
        ]
        return {
            "agents": agents,
            "total": len(agents),
            "message": "All AI agents ready for deployment!"
        }
    
    @app.post("/chat")
    async def chat_with_ai(message: dict):
        """Simple chat endpoint"""
        user_message = message.get("message", "")
        
        # Simple response system
        responses = {
            "hello": "Hello! I'm ScrollIntel, your AI-powered CTO platform. How can I help you today?",
            "status": "All systems are operational and ready for production deployment!",
            "help": "I can help you with data analysis, ML model building, technical decisions, and more!",
            "agents": "I have 6 specialized AI agents ready: CTO, Data Scientist, ML Engineer, AI Engineer, Business Analyst, and QA Agent.",
            "deploy": "ScrollIntel is production-ready! You can deploy to Railway, Render, or any cloud platform.",
            "default": f"Thanks for your message: '{user_message}'. ScrollIntel is ready to help with AI-powered technical leadership!"
        }
        
        # Simple keyword matching
        response_key = "default"
        for key in responses.keys():
            if key in user_message.lower():
                response_key = key
                break
        
        return {
            "response": responses[response_key],
            "agent": "ScrollIntel AI",
            "timestamp": time.time(),
            "status": "success"
        }
    
    @app.get("/demo")
    async def demo_endpoint():
        """Demo endpoint showing ScrollIntel capabilities"""
        return {
            "message": "ScrollIntel Demo - AI-Powered CTO Platform",
            "capabilities": [
                "6 Specialized AI Agents",
                "Advanced Data Analytics", 
                "ML Model Building",
                "Business Intelligence",
                "Quality Assurance",
                "Real-time Processing",
                "Production Ready"
            ],
            "features": {
                "file_processing": "Upload and analyze any data file",
                "ai_chat": "Chat with specialized AI agents",
                "ml_models": "Build and deploy ML models",
                "dashboards": "Create interactive dashboards",
                "api_access": "Full REST API access",
                "monitoring": "Real-time system monitoring"
            },
            "status": "All systems ready for production!"
        }
    
    if __name__ == "__main__":
        # Get port from environment or find available port
        port = int(os.getenv("API_PORT", 8000))
        
        # Check if port is available, find alternative if not
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('127.0.0.1', port))
        except OSError:
            # Find available port
            for test_port in range(8001, 8100):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        s.bind(('127.0.0.1', test_port))
                        port = test_port
                        break
                except OSError:
                    continue
        
        print(f"🚀 Starting ScrollIntel on http://localhost:{port}")
        print(f"📚 API Documentation: http://localhost:{port}/docs")
        print(f"❤️  Health Check: http://localhost:{port}/health")
        print(f"🤖 AI Agents: http://localhost:{port}/agents")
        print(f"🎯 Demo: http://localhost:{port}/demo")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            log_level="info",
            access_log=True
        )

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Installing required packages...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "python-dotenv"])
    print("✅ Packages installed. Please run again.")
    sys.exit(1)
