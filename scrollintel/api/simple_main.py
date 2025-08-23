"""
Simple ScrollIntel FastAPI Application for Development
Minimal setup without complex middleware for easier debugging
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime

from .routes.simple_routes import router as simple_router

# Create FastAPI application
app = FastAPI(
    title="ScrollIntel API (Simple)",
    description="Simplified ScrollIntel API for development",
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

# Include simple routes
app.include_router(simple_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ScrollIntel Simple API is running",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Simple global exception handler"""
    print(f"Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import socket
    
    # Find an available port
    ports_to_try = [8000, 8001, 8002, 8003, 8080]
    port_to_use = 8000
    
    for port in ports_to_try:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                port_to_use = port
                break
        except OSError:
            continue
    
    print(f"🚀 Starting ScrollIntel API on port {port_to_use}")
    
    uvicorn.run(
        "scrollintel.api.simple_main:app",
        host="127.0.0.1",  # Use localhost for Windows compatibility
        port=port_to_use,
        reload=True,
        log_level="info"
    )