"""Main FastAPI application for AI Data Readiness Platform."""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager
from typing import Optional

from .middleware.auth import AuthMiddleware, get_current_user
from .middleware.validation import ValidationMiddleware
from .middleware.rate_limiting import RateLimitMiddleware
from .graphql.app import setup_graphql
from .routes import (
    auth_routes,
    datasets_routes,
    quality_routes,
    bias_routes,
    feature_routes,
    compliance_routes,
    usage_tracking_routes,
    lineage_routes,
    drift_routes,
    processing_routes,
    health_routes
)
from ..core.config import Config
from ..models.database import init_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize configuration
config = Config()

# Security scheme
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting AI Data Readiness Platform API")
    
    # Initialize database
    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Validate configuration
    config.validate()
    logger.info("Configuration validated")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Data Readiness Platform API")


# Create FastAPI application
app = FastAPI(
    title="AI Data Readiness Platform API",
    description="Comprehensive API for AI data preparation, quality assessment, and optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add custom middleware
app.add_middleware(AuthMiddleware)
app.add_middleware(ValidationMiddleware)
app.add_middleware(RateLimitMiddleware)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "HTTPException"
            }
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors."""
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": 400,
                "message": str(exc),
                "type": "ValueError"
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "type": "InternalServerError"
            }
        }
    )


# Include routers
app.include_router(health_routes.router, prefix="/api/v1", tags=["health"])
app.include_router(auth_routes.router, prefix="/api/v1", tags=["authentication"])
app.include_router(datasets_routes.router, prefix="/api/v1", tags=["datasets"])
app.include_router(quality_routes.router, prefix="/api/v1", tags=["quality"])
app.include_router(bias_routes.router, prefix="/api/v1", tags=["bias"])
app.include_router(feature_routes.router, prefix="/api/v1", tags=["features"])
app.include_router(compliance_routes.router, prefix="/api/v1", tags=["compliance"])
app.include_router(usage_tracking_routes.router, prefix="/api/v1", tags=["usage-tracking"])
app.include_router(lineage_routes.router, prefix="/api/v1", tags=["lineage"])
app.include_router(drift_routes.router, prefix="/api/v1", tags=["drift"])
app.include_router(processing_routes.router, prefix="/api/v1", tags=["processing"])

# Setup GraphQL
setup_graphql(app)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Data Readiness Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "service": "ai-data-readiness-platform"}


if __name__ == "__main__":
    uvicorn.run(
        "ai_data_readiness.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )