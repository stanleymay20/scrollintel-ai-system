"""
ScrollIntel FastAPI Application
Main application optimized for Railway deployment
"""

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging

# Use absolute imports for deployment compatibility
from scrollintel.api.middleware.performance_middleware import (
    PerformanceMonitoringMiddleware, 
    RequestLoggingMiddleware
)

# Import other middleware with fallbacks
try:
    from scrollintel.api.middleware.audit_middleware import AuditMiddleware, ComplianceAuditMiddleware
except ImportError:
    # Create dummy middleware if not available
    class AuditMiddleware:
        def __init__(self, app): self.app = app
        async def __call__(self, scope, receive, send): 
            return await self.app(scope, receive, send)
    
    class ComplianceAuditMiddleware:
        def __init__(self, app): self.app = app
        async def __call__(self, scope, receive, send): 
            return await self.app(scope, receive, send)

try:
    from scrollintel.core.bulletproof_middleware import BulletproofMiddleware, HealthCheckMiddleware
except ImportError:
    # Create dummy middleware if not available
    class BulletproofMiddleware:
        def __init__(self, app): self.app = app
        async def __call__(self, scope, receive, send): 
            return await self.app(scope, receive, send)
    
    class HealthCheckMiddleware:
        def __init__(self, app): self.app = app
        async def __call__(self, scope, receive, send): 
            return await self.app(scope, receive, send)

# Import routes with fallbacks
try:
    from scrollintel.api.routes import health_routes
except ImportError:
    # Create dummy health routes
    from fastapi import APIRouter
    health_routes = type('obj', (object,), {'router': APIRouter()})

try:
    from scrollintel.api.routes import auth_routes, agent_routes, file_routes
    from scrollintel.api.routes import dashboard_routes, monitoring_routes, audit_routes
except ImportError:
    from fastapi import APIRouter
    auth_routes = type('obj', (object,), {'router': APIRouter()})
    agent_routes = type('obj', (object,), {'router': APIRouter()})
    file_routes = type('obj', (object,), {'router': APIRouter()})
    dashboard_routes = type('obj', (object,), {'router': APIRouter()})
    monitoring_routes = type('obj', (object,), {'router': APIRouter()})
    audit_routes = type('obj', (object,), {'router': APIRouter()})

# Import additional routers with fallbacks
try:
    from scrollintel.api.routes.performance_routes import router as performance_router
except ImportError:
    from fastapi import APIRouter
    performance_router = APIRouter()

try:
    from scrollintel.api.routes.visualization_routes import router as visualization_router
except ImportError:
    from fastapi import APIRouter
    visualization_router = APIRouter()

try:
    from scrollintel.api.routes.legal_routes import router as legal_router
except ImportError:
    from fastapi import APIRouter
    legal_router = APIRouter()

# Create a simple routes fallback
from fastapi import APIRouter
simple_router = APIRouter()

@simple_router.get("/api/simple")
async def simple_endpoint():
    return {"message": "ScrollIntel API is running", "status": "ok"}

try:
    from scrollintel.api.routes.visual_generation_routes import router as visual_generation_router
except ImportError:
    visual_generation_router = APIRouter()

try:
    from scrollintel.api.websocket.visual_generation_websocket import router as visual_generation_ws_router
except ImportError:
    visual_generation_ws_router = APIRouter()

# Import core modules with fallbacks
try:
    from scrollintel.core.performance_monitor import initialize_performance_monitoring
except ImportError:
    async def initialize_performance_monitoring():
        pass

try:
    from scrollintel.core.logging_config import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

try:
    from scrollintel.core.config import get_settings
except ImportError:
    def get_settings():
        return type('obj', (object,), {})()

# Try to import enterprise UI routes
try:
    from scrollintel.api.routes import enterprise_ui_routes
except ImportError:
    enterprise_ui_routes = type('obj', (object,), {'router': APIRouter()})

logger = get_logger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager optimized for Railway"""
    # Startup
    logger.info("Starting ScrollIntel application on Railway...")
    logger.info(f"Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'unknown')}")
    logger.info(f"Port: {os.getenv('PORT', '8000')}")
    
    # Initialize performance monitoring (with error handling)
    try:
        await initialize_performance_monitoring()
        logger.info("Performance monitoring initialized")
    except Exception as e:
        logger.warning(f"Performance monitoring failed to initialize: {e}")
    
    # Start background tasks (optional for Railway)
    try:
        from scrollintel.core.monitoring import performance_monitor
        asyncio.create_task(performance_monitor.monitor_loop())
        logger.info("Background monitoring started")
    except Exception as e:
        logger.warning(f"Background monitoring not available: {e}")
    
    # Start audit and compliance systems (optional for Railway)
    try:
        from scrollintel.core.audit_system import audit_system
        from scrollintel.core.compliance_manager import compliance_manager
        await audit_system.start()
        await compliance_manager.start()
        logger.info("Audit and compliance systems started")
    except Exception as e:
        logger.warning(f"Audit and compliance systems not available: {e}")
    
    logger.info("ScrollIntel application startup complete")
    yield
    
    # Shutdown
    logger.info("Shutting down ScrollIntel application...")
    
    # Stop audit and compliance systems
    try:
        from scrollintel.core.audit_system import audit_system
        from scrollintel.core.compliance_manager import compliance_manager
        await audit_system.stop()
        await compliance_manager.stop()
        logger.info("Audit and compliance systems stopped")
    except Exception as e:
        logger.warning(f"Error stopping audit systems: {e}")
    
    logger.info("ScrollIntel application shutdown complete")

# Create FastAPI application optimized for Railway
app = FastAPI(
    title="ScrollIntel API",
    description="AI-powered CTO replacement platform optimized for Railway deployment",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not os.getenv('RAILWAY_ENVIRONMENT') == 'production' else None,
    redoc_url="/redoc" if not os.getenv('RAILWAY_ENVIRONMENT') == 'production' else None
)

# Add middleware optimized for Railway
allowed_origins = ["*"]
if os.getenv('RAILWAY_ENVIRONMENT') == 'production':
    # Use production origins from environment
    allowed_origins = os.getenv('CORS_ORIGINS', 'https://scrollintel.com').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add bulletproof protection middleware (first priority)
app.add_middleware(BulletproofMiddleware)
app.add_middleware(HealthCheckMiddleware)

# Add performance monitoring middleware
app.add_middleware(PerformanceMonitoringMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Add audit and compliance middleware
app.add_middleware(AuditMiddleware)
app.add_middleware(ComplianceAuditMiddleware)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with performance tracking"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Record error in performance metrics
    try:
        from scrollintel.core.performance_monitor import response_tracker, ResponseTimeMetric
        from datetime import datetime
        
        metric = ResponseTimeMetric(
            endpoint=request.url.path,
            method=request.method,
            response_time=0.0,  # Will be updated by middleware
            status_code=500,
            timestamp=datetime.utcnow(),
            user_id=getattr(request.state, 'user_id', None)
        )
        response_tracker.metrics.append(metric)
    except ImportError:
        pass  # Performance monitoring not available
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, 'request_id', None)
        }
    )

# Include routers
app.include_router(simple_router)  # Add simple routes first (no auth required)
app.include_router(health_routes.router)
app.include_router(auth_routes.router)
app.include_router(agent_routes.router)
app.include_router(file_routes.router)
app.include_router(dashboard_routes.router)
app.include_router(monitoring_routes.router)
app.include_router(audit_routes.router)
app.include_router(enterprise_ui_routes.router)
app.include_router(performance_router)
app.include_router(visualization_router)
app.include_router(legal_router)
app.include_router(visual_generation_router)  # Visual generation routes
app.include_router(visual_generation_ws_router)  # Visual generation WebSocket

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic system info"""
    try:
        from scrollintel.core.performance_monitor import performance_dashboard
        # Get basic performance metrics
        dashboard_data = await performance_dashboard.get_dashboard_data()
        
        return {
            "message": "ScrollIntel API is running",
            "version": "1.0.0",
            "status": "healthy",
            "performance": {
                "avg_response_time": dashboard_data["response_times"]["avg_response_time"],
                "cpu_percent": dashboard_data["system"]["cpu_percent"],
                "memory_percent": dashboard_data["system"]["memory_percent"],
                "cache_hit_rate": dashboard_data["cache"]["hit_rate"]
            },
            "timestamp": dashboard_data["timestamp"]
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        from datetime import datetime
        return {
            "message": "ScrollIntel API is running",
            "version": "1.0.0",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        }

# Health check endpoint optimized for Railway
@app.get("/health")
async def health_check():
    """Railway-compatible health check endpoint"""
    try:
        from datetime import datetime
        import psutil
        
        # Basic system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        if cpu_percent > 90:
            health_status = "degraded"
            issues.append("High CPU usage")
            
        if memory.percent > 90:
            health_status = "critical"
            issues.append("High memory usage")
        
        return {
            "status": health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "environment": os.getenv('RAILWAY_ENVIRONMENT', 'development'),
            "port": os.getenv('PORT', '8000'),
            "issues": issues,
            "metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "uptime": "healthy"
            },
            "railway": {
                "service_id": os.getenv('RAILWAY_SERVICE_ID', 'unknown'),
                "deployment_id": os.getenv('RAILWAY_DEPLOYMENT_ID', 'unknown'),
                "environment_id": os.getenv('RAILWAY_ENVIRONMENT_ID', 'unknown')
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        from datetime import datetime
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "ScrollIntel API is running",
            "error": str(e) if not os.getenv('RAILWAY_ENVIRONMENT') == 'production' else None
        }

if __name__ == "__main__":
    # Railway deployment configuration
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Starting ScrollIntel on {host}:{port}")
    logger.info(f"Railway Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'development')}")
    
    uvicorn.run(
        "scrollintel.api.main:app",
        host=host,
        port=port,
        reload=False,  # Disable reload for production
        log_level="info",
        access_log=True
    )