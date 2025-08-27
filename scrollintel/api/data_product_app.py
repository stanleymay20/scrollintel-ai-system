"""
Data Product API Application

FastAPI application that integrates REST APIs, GraphQL, WebSocket support,
and comprehensive middleware for the Data Product Registry.
"""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# from graphene import Schema  # Temporarily disabled for Docker build
from starlette.graphql import GraphQLApp
from starlette.middleware.sessions import SessionMiddleware
import logging
from contextlib import asynccontextmanager

from scrollintel.api.routes.data_product_routes import router as data_product_router
from scrollintel.api.graphql.data_product_schema import schema as graphql_schema
from scrollintel.api.websocket.data_product_websocket import (
    websocket_endpoint,
    start_background_tasks,
    manager as websocket_manager
)
from scrollintel.api.middleware.data_product_middleware import (
    RateLimitMiddleware,
    AuthenticationMiddleware,
    AccessControlMiddleware,
    AuditLoggingMiddleware,
    SecurityHeadersMiddleware
)
from scrollintel.models.database import get_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audit logger
audit_logger = logging.getLogger("audit")
audit_handler = logging.FileHandler("logs/data_product_audit.log")
audit_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Data Product API application")
    start_background_tasks()
    yield
    # Shutdown
    logger.info("Shutting down Data Product API application")


# Create FastAPI application
app = FastAPI(
    title="ScrollIntel G6 Data Product Registry API",
    description="""
    Enhanced Data Product Registry API with comprehensive features:
    
    ## Features
    - **REST API**: Full CRUD operations for data products
    - **GraphQL**: Complex queries with relationships and filtering
    - **WebSocket**: Real-time updates and notifications
    - **Authentication**: JWT-based authentication with RBAC
    - **Rate Limiting**: Configurable rate limits per endpoint
    - **Audit Logging**: Comprehensive audit trail
    - **Search**: Full-text and semantic search capabilities
    - **Governance**: Data lineage, quality metrics, and compliance
    
    ## Authentication
    All endpoints (except health checks) require a valid JWT token in the Authorization header:
    ```
    Authorization: Bearer <your-jwt-token>
    ```
    
    ## Rate Limits
    - Create operations: 50 requests/hour
    - Update operations: 30 requests/hour
    - Delete operations: 10 requests/hour
    - Read operations: 200 requests/hour
    - Search operations: 100 requests/hour
    
    ## WebSocket
    Connect to `/ws/data-products` for real-time updates on:
    - Data product changes
    - Quality alerts
    - Verification status updates
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware (order matters - last added is executed first)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(AccessControlMiddleware)
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(RateLimitMiddleware, default_rate_limit=1000, window_seconds=3600)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session middleware for GraphQL
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

# Include REST API routes
app.include_router(data_product_router)

# GraphQL endpoint
app.add_route("/graphql", GraphQLApp(schema=graphql_schema))

# WebSocket endpoint
@app.websocket("/ws/data-products")
async def websocket_data_products(websocket: WebSocket, db=Depends(get_db)):
    """WebSocket endpoint for real-time data product updates"""
    await websocket_endpoint(websocket, db)


# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "data-product-registry-api",
        "version": "1.0.0"
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status"""
    try:
        # Check database connection
        db = next(get_db())
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check WebSocket connections
    ws_stats = websocket_manager.get_connection_stats()
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "service": "data-product-registry-api",
        "version": "1.0.0",
        "components": {
            "database": db_status,
            "websocket": {
                "status": "healthy",
                "active_connections": ws_stats["total_connections"],
                "subscriptions": {
                    "product_subscriptions": len(ws_stats["product_subscriptions"]),
                    "quality_subscribers": ws_stats["quality_subscribers"],
                    "verification_subscribers": ws_stats["verification_subscribers"],
                    "global_subscribers": ws_stats["global_subscribers"]
                }
            }
        }
    }


@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    ws_stats = websocket_manager.get_connection_stats()
    
    return {
        "api_metrics": {
            "total_endpoints": len(app.routes),
            "websocket_connections": ws_stats["total_connections"],
            "active_subscriptions": {
                "product_specific": len(ws_stats["product_subscriptions"]),
                "quality_alerts": ws_stats["quality_subscribers"],
                "verification_updates": ws_stats["verification_subscribers"],
                "global_updates": ws_stats["global_subscribers"]
            }
        },
        "timestamp": "2024-01-01T00:00:00Z"
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested resource {request.url.path} was not found",
            "path": str(request.url.path)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "path": str(request.url.path)
        }
    )


# API documentation endpoints
@app.get("/api/v1/schema")
async def get_api_schema():
    """Get OpenAPI schema"""
    return app.openapi()


@app.get("/api/v1/endpoints")
async def list_endpoints():
    """List all available endpoints"""
    endpoints = []
    
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            for method in route.methods:
                if method != 'HEAD':  # Exclude HEAD methods
                    endpoints.append({
                        "method": method,
                        "path": route.path,
                        "name": getattr(route, 'name', None),
                        "summary": getattr(route, 'summary', None)
                    })
    
    return {
        "endpoints": sorted(endpoints, key=lambda x: (x['path'], x['method'])),
        "total_count": len(endpoints)
    }


# Development and testing endpoints (remove in production)
@app.get("/dev/websocket-stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics (development only)"""
    return websocket_manager.get_connection_stats()


@app.post("/dev/generate-token")
async def generate_dev_token(user_id: str, permissions: list = None):
    """Generate development JWT token (remove in production)"""
    from scrollintel.api.middleware.data_product_middleware import generate_access_token
    
    if permissions is None:
        permissions = [
            "data_product:create",
            "data_product:read",
            "data_product:update",
            "data_product:delete",
            "data_product:search",
            "data_product:manage_provenance",
            "data_product:manage_quality",
            "data_product:manage_bias",
            "data_product:verify"
        ]
    
    token = generate_access_token(user_id, permissions, expires_in=86400)  # 24 hours
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 86400,
        "user_id": user_id,
        "permissions": permissions
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "scrollintel.api.data_product_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )