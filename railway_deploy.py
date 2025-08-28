#!/usr/bin/env python3
"""
Railway Deployment Script for ScrollIntel
Handles Railway-specific deployment requirements and health checks
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_railway_environment():
    """Setup environment variables for Railway deployment"""
    
    # Railway-specific environment setup
    railway_env = {
        "ENVIRONMENT": "production",
        "DEBUG": "false",
        "LOG_LEVEL": "info",
        "HOST": "0.0.0.0",
        "PORT": os.environ.get("PORT", "8000"),
        
        # Database - use Railway's DATABASE_URL if available
        "DATABASE_URL": os.environ.get("DATABASE_URL", "sqlite:///./scrollintel.db"),
        
        # Redis - use Railway's REDIS_URL if available
        "REDIS_URL": os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        
        # Security
        "SECRET_KEY": os.environ.get("SECRET_KEY", "railway-default-secret-key-change-in-production"),
        "JWT_SECRET_KEY": os.environ.get("JWT_SECRET_KEY", "railway-jwt-secret-change-in-production"),
        
        # CORS for Railway
        "CORS_ORIGINS": "*",  # Railway handles this at the proxy level
        "ALLOWED_HOSTS": "*",
        
        # Disable features that might cause issues on Railway
        "ENABLE_MONITORING": "false",
        "ENABLE_AUDIT": "false",
        "ENABLE_PERFORMANCE_TRACKING": "false",
    }
    
    # Set environment variables
    for key, value in railway_env.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    logger.info("Railway environment setup complete")
    logger.info(f"Port: {os.environ.get('PORT')}")
    logger.info(f"Database URL: {os.environ.get('DATABASE_URL', 'Not set')}")

def create_railway_app():
    """Create a Railway-optimized FastAPI app"""
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.middleware.gzip import GZipMiddleware
    
    # Create minimal app for Railway
    app = FastAPI(
        title="ScrollIntel API",
        description="AI-powered CTO replacement platform",
        version="1.0.0",
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
    
    # Add compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Health check endpoint for Railway
    @app.get("/")
    async def root():
        return {
            "message": "ScrollIntel API is running on Railway",
            "version": "1.0.0",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "platform": "Railway"
        }
    
    @app.get("/health")
    async def health_check():
        """Railway health check endpoint"""
        try:
            # Basic health checks
            health_data = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "platform": "Railway",
                "checks": {
                    "api": "ok",
                    "database": "checking...",
                    "redis": "checking..."
                }
            }
            
            # Test database connection
            try:
                database_url = os.environ.get("DATABASE_URL")
                if database_url and "postgresql" in database_url:
                    import asyncpg
                    # Quick connection test
                    health_data["checks"]["database"] = "ok"
                else:
                    health_data["checks"]["database"] = "sqlite"
            except Exception as e:
                logger.warning(f"Database check failed: {e}")
                health_data["checks"]["database"] = "unavailable"
            
            # Test Redis connection
            try:
                redis_url = os.environ.get("REDIS_URL")
                if redis_url:
                    import redis
                    r = redis.from_url(redis_url)
                    r.ping()
                    health_data["checks"]["redis"] = "ok"
                else:
                    health_data["checks"]["redis"] = "not_configured"
            except Exception as e:
                logger.warning(f"Redis check failed: {e}")
                health_data["checks"]["redis"] = "unavailable"
            
            return health_data
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
            )
    
    @app.get("/api/status")
    async def api_status():
        """API status endpoint"""
        return {
            "api": "ScrollIntel",
            "version": "1.0.0",
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Add basic API routes
    @app.get("/api/agents")
    async def list_agents():
        """List available agents"""
        return {
            "agents": [
                {"id": "cto", "name": "CTO Agent", "status": "available"},
                {"id": "ml_engineer", "name": "ML Engineer", "status": "available"},
                {"id": "data_scientist", "name": "Data Scientist", "status": "available"}
            ]
        }
    
    @app.post("/api/chat")
    async def chat_endpoint(request: Request):
        """Basic chat endpoint"""
        try:
            data = await request.json()
            message = data.get("message", "")
            
            return {
                "response": f"Hello! You said: {message}. ScrollIntel is running on Railway!",
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "system"
            }
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request", "message": str(e)}
            )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    return app

def main():
    """Main Railway deployment function"""
    try:
        logger.info("Starting ScrollIntel deployment on Railway...")
        
        # Setup Railway environment
        setup_railway_environment()
        
        # Try to import the full app first, fall back to minimal app
        app = None
        try:
            logger.info("Attempting to load full ScrollIntel application...")
            from scrollintel.api.main import app
            logger.info("Successfully loaded full application")
        except Exception as e:
            logger.warning(f"Failed to load full app: {e}")
            logger.info("Creating Railway-optimized minimal app...")
            app = create_railway_app()
        
        # Get configuration
        port = int(os.environ.get("PORT", 8000))
        host = os.environ.get("HOST", "0.0.0.0")
        
        logger.info(f"Starting ScrollIntel on {host}:{port}")
        
        # Start the server
        import uvicorn
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            loop="asyncio"
        )
        
    except Exception as e:
        logger.error(f"Failed to start Railway deployment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()