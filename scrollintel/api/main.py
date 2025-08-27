"""
ScrollIntel FastAPI Application
Main application with performance monitoring integration
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .middleware.performance_middleware import (
    PerformanceMonitoringMiddleware, 
    RequestLoggingMiddleware
)
from .middleware.audit_middleware import AuditMiddleware, ComplianceAuditMiddleware
from ..core.bulletproof_middleware import BulletproofMiddleware, HealthCheckMiddleware
from .routes import (
    health_routes, auth_routes, agent_routes, file_routes,
    dashboard_routes, monitoring_routes, audit_routes, enterprise_ui_routes
)
from .routes.performance_routes import router as performance_router
from .routes.visualization_routes import router as visualization_router
from .routes.legal_routes import router as legal_router
from .routes.simple_routes import router as simple_router
from .routes.visual_generation_routes import router as visual_generation_router
from .websocket.visual_generation_websocket import router as visual_generation_ws_router
from ..core.performance_monitor import initialize_performance_monitoring
from ..core.logging_config import get_logger
from ..core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting ScrollIntel application...")
    
    # Initialize performance monitoring
    await initialize_performance_monitoring()
    logger.info("Performance monitoring initialized")
    
    # Start background tasks
    from ..core.monitoring import performance_monitor
    asyncio.create_task(performance_monitor.monitor_loop())
    logger.info("Background monitoring started")
    
    # Start audit and compliance systems
    from ..core.audit_system import audit_system
    from ..core.compliance_manager import compliance_manager
    await audit_system.start()
    await compliance_manager.start()
    logger.info("Audit and compliance systems started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ScrollIntel application...")
    
    # Stop audit and compliance systems
    from ..core.audit_system import audit_system
    from ..core.compliance_manager import compliance_manager
    await audit_system.stop()
    await compliance_manager.stop()
    logger.info("Audit and compliance systems stopped")

# Create FastAPI application
app = FastAPI(
    title="ScrollIntel API",
    description="AI-powered CTO replacement platform with comprehensive performance monitoring",
    version="1.0.0",
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
    from ..core.performance_monitor import response_tracker
    from ..core.performance_monitor import ResponseTimeMetric
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
    from ..core.performance_monitor import performance_dashboard
    
    try:
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
        return {
            "message": "ScrollIntel API is running",
            "version": "1.0.0",
            "status": "healthy",
            "timestamp": "unknown"
        }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Detailed health check with performance metrics"""
    from ..core.performance_monitor import performance_dashboard
    
    try:
        dashboard_data = await performance_dashboard.get_dashboard_data()
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        if dashboard_data["response_times"]["avg_response_time"] > 2.0:
            health_status = "degraded"
            issues.append("High response times")
            
        if dashboard_data["system"]["cpu_percent"] > 80:
            health_status = "degraded"
            issues.append("High CPU usage")
            
        if dashboard_data["system"]["memory_percent"] > 85:
            health_status = "critical"
            issues.append("High memory usage")
            
        return {
            "status": health_status,
            "timestamp": dashboard_data["timestamp"],
            "issues": issues,
            "metrics": {
                "response_time": dashboard_data["response_times"]["avg_response_time"],
                "cpu_percent": dashboard_data["system"]["cpu_percent"],
                "memory_percent": dashboard_data["system"]["memory_percent"],
                "cache_hit_rate": dashboard_data["cache"]["hit_rate"],
                "database_queries": dashboard_data["database"]["total_queries"]
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unknown",
            "timestamp": "unknown",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "scrollintel.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )