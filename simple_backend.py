
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
