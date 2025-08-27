"""
Simple health check endpoint for Railway deployment
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import time
import os

def add_simple_health_routes(app: FastAPI):
    """Add simple health routes that don't depend on complex dependencies"""
    
    @app.get("/health")
    async def simple_health():
        """Simple health check for Railway"""
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "timestamp": time.time(),
                "service": "ScrollIntel API",
                "environment": os.getenv("ENVIRONMENT", "production"),
                "port": os.getenv("PORT", "8000")
            }
        )
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return JSONResponse(
            status_code=200,
            content={
                "message": "ScrollIntel API is running",
                "status": "healthy",
                "timestamp": time.time()
            }
        )
    
    @app.get("/ping")
    async def ping():
        """Simple ping endpoint"""
        return JSONResponse(
            status_code=200,
            content={"pong": True, "timestamp": time.time()}
        )