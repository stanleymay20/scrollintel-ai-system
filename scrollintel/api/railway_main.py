"""
Simplified ScrollIntel FastAPI Application for Railway Deployment
"""

import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Create FastAPI application
app = FastAPI(
    title="ScrollIntel API",
    description="AI-powered CTO replacement platform",
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

# Simple health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check for Railway"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": time.time(),
            "service": "ScrollIntel API",
            "environment": os.getenv("RAILWAY_ENVIRONMENT", "production"),
            "port": os.getenv("PORT", "8000")
        }
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "message": "ScrollIntel API is running on Railway",
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0"
        }
    )

# Ping endpoint
@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return JSONResponse(
        status_code=200,
        content={"pong": True, "timestamp": time.time()}
    )

# API info endpoint
@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": "ScrollIntel API",
        "version": "1.0.0",
        "description": "AI-powered CTO replacement platform",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "ping": "/ping",
            "root": "/",
            "info": "/api/info"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "scrollintel.api.railway_main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )