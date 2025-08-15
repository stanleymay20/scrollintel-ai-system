"""
FastAPI Gateway and Routing System for ScrollIntel.
Main entry point for all API requests with middleware and routing.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..core.config import get_config
from ..core.registry import AgentRegistry, TaskOrchestrator
from ..core.interfaces import AgentError, SecurityError, EngineError
from ..core.error_middleware import ErrorHandlingMiddleware
from ..core.error_monitoring import error_monitor, start_error_monitoring, stop_error_monitoring
from ..security.middleware import SecurityMiddleware, RateLimitMiddleware
from ..security.auth import authenticator
from .routes import agent_routes, auth_routes, health_routes, admin_routes, file_routes, automodel_routes, scroll_qa_routes, scroll_viz_routes


class ScrollIntelGateway:
    """Main FastAPI gateway for ScrollIntel system."""
    
    def __init__(self):
        self.config = get_config()
        self.agent_registry = AgentRegistry()
        self.task_orchestrator = TaskOrchestrator(self.agent_registry)
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Application lifespan manager."""
            # Startup
            await self._startup()
            yield
            # Shutdown
            await self._shutdown()
        
        app = FastAPI(
            title="ScrollIntel™ API",
            description="Sovereign AI Intelligence System API",
            version="1.0.0",
            docs_url="/docs" if not self.config.is_production else None,
            redoc_url="/redoc" if not self.config.is_production else None,
            lifespan=lifespan
        )
        
        # Add middleware
        self._add_middleware(app)
        
        # Add exception handlers
        self._add_exception_handlers(app)
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_middleware(self, app: FastAPI) -> None:
        """Add middleware to the FastAPI application."""
        
        # Error handling middleware (should be first to catch all errors)
        app.add_middleware(
            ErrorHandlingMiddleware,
            enable_detailed_errors=not self.config.is_production
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if not self.config.is_production else ["https://scrollintel.com"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Rate limiting
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=self.config.rate_limit_requests
        )
        
        # Security middleware (authentication, authorization, audit)
        app.add_middleware(
            SecurityMiddleware,
            excluded_paths=[
                "/docs",
                "/redoc",
                "/openapi.json",
                "/health",
                "/auth/login",
                "/auth/register",
                "/auth/refresh",
                "/auth/password-requirements",
                "/",
                "/status"
            ]
        )
    
    def _add_exception_handlers(self, app: FastAPI) -> None:
        """Add custom exception handlers."""
        
        @app.exception_handler(AgentError)
        async def agent_error_handler(request: Request, exc: AgentError):
            """Handle agent-related errors."""
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": "Agent Error",
                    "message": str(exc),
                    "type": "agent_error",
                    "timestamp": time.time()
                }
            )
        
        @app.exception_handler(SecurityError)
        async def security_error_handler(request: Request, exc: SecurityError):
            """Handle security-related errors."""
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Security Error",
                    "message": str(exc),
                    "type": "security_error",
                    "timestamp": time.time()
                }
            )
        
        @app.exception_handler(EngineError)
        async def engine_error_handler(request: Request, exc: EngineError):
            """Handle engine-related errors."""
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Engine Error",
                    "message": str(exc),
                    "type": "engine_error",
                    "timestamp": time.time()
                }
            )
        
        @app.exception_handler(ValueError)
        async def value_error_handler(request: Request, exc: ValueError):
            """Handle validation errors."""
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Validation Error",
                    "message": str(exc),
                    "type": "validation_error",
                    "timestamp": time.time()
                }
            )
        
        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle unexpected errors."""
            # Log the error for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred" if self.config.is_production else str(exc),
                    "type": "internal_error",
                    "timestamp": time.time()
                }
            )
        
        @app.exception_handler(StarletteHTTPException)
        async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
            """Handle HTTP exceptions with custom format."""
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": "HTTP Error",
                    "message": exc.detail,
                    "status_code": exc.status_code,
                    "timestamp": time.time()
                }
            )
    
    def _add_routes(self, app: FastAPI) -> None:
        """Add route handlers to the application."""
        
        # Health check routes (no authentication required)
        app.include_router(
            health_routes.create_health_router(self.agent_registry),
            prefix="/health",
            tags=["Health"]
        )
        
        # Authentication routes (no authentication required)
        app.include_router(
            auth_routes.create_auth_router(),
            prefix="/auth",
            tags=["Authentication"]
        )
        
        # Agent routes (authentication required)
        app.include_router(
            agent_routes.create_agent_router(self.agent_registry, self.task_orchestrator),
            prefix="/agents",
            tags=["Agents"]
        )
        
        # Admin routes (admin authentication required)
        app.include_router(
            admin_routes.create_admin_router(self.agent_registry),
            prefix="/admin",
            tags=["Administration"]
        )
        
        # File upload routes (authentication required)
        app.include_router(
            file_routes.create_file_router(),
            prefix="/files",
            tags=["File Management"]
        )
        
        # AutoModel routes (authentication required)
        app.include_router(
            automodel_routes.router,
            tags=["AutoModel"]
        )
        
        # ScrollQA routes (authentication required)
        app.include_router(
            scroll_qa_routes.router,
            tags=["ScrollQA"]
        )
        
        # ScrollViz routes (authentication required)
        app.include_router(
            scroll_viz_routes.router,
            tags=["ScrollViz"]
        )
        
        # Vault routes (authentication required)
        from .routes import vault_routes
        app.include_router(
            vault_routes.router,
            tags=["Vault"]
        )
        
        # ScrollInsightRadar routes (authentication required)
        from .routes import scroll_insight_radar_routes
        app.include_router(
            scroll_insight_radar_routes.router,
            tags=["ScrollInsightRadar"]
        )
        
        # Visual Generation routes (authentication required)
        from .routes import visual_generation_routes
        app.include_router(
            visual_generation_routes.router,
            tags=["Visual Generation"]
        )
        
        # Continuous Innovation routes (authentication required)
        from .routes import continuous_innovation_routes
        app.include_router(
            continuous_innovation_routes.router,
            tags=["Continuous Innovation"]
        )
        
        # Root endpoint
        @app.get("/", tags=["Root"])
        async def root():
            """Root endpoint with system information."""
            return {
                "name": "ScrollIntel™ API",
                "version": "1.0.0",
                "description": "Sovereign AI Intelligence System",
                "status": "operational",
                "timestamp": time.time(),
                "environment": self.config.environment
            }
        
        # System status endpoint
        @app.get("/status", tags=["System"])
        async def system_status():
            """Get system status information."""
            registry_status = self.agent_registry.get_registry_status()
            error_metrics = error_monitor.get_metrics()
            
            return {
                "system": "ScrollIntel™",
                "status": "operational",
                "timestamp": time.time(),
                "environment": self.config.environment,
                "agents": registry_status,
                "uptime": time.time() - getattr(self, '_start_time', time.time()),
                "error_metrics": {
                    "total_components": error_metrics.get("total_components", 0),
                    "healthy_components": error_metrics.get("healthy_components", 0),
                    "degraded_components": error_metrics.get("degraded_components", 0),
                    "unhealthy_components": error_metrics.get("unhealthy_components", 0),
                    "overall_success_rate": error_metrics.get("overall_success_rate", 1.0)
                },
                "active_alerts": len(error_monitor.get_active_alerts())
            }
        
        # Error monitoring endpoints
        @app.get("/monitoring/metrics", tags=["Monitoring"])
        async def get_error_metrics():
            """Get detailed error metrics."""
            return error_monitor.get_metrics()
        
        @app.get("/monitoring/alerts", tags=["Monitoring"])
        async def get_active_alerts():
            """Get active alerts."""
            return {
                "active_alerts": error_monitor.get_active_alerts(),
                "timestamp": time.time()
            }
        
        @app.get("/monitoring/component/{component_name}", tags=["Monitoring"])
        async def get_component_metrics(component_name: str):
            """Get metrics for a specific component."""
            return error_monitor.get_component_metrics(component_name)
        
        @app.post("/monitoring/alerts/{alert_id}/resolve", tags=["Monitoring"])
        async def resolve_alert(alert_id: str):
            """Resolve an active alert."""
            error_monitor.resolve_alert(alert_id)
            return {"message": f"Alert {alert_id} resolved", "timestamp": time.time()}
    
    async def _startup(self) -> None:
        """Application startup tasks."""
        self._start_time = time.time()
        
        # Initialize logging
        import logging
        logging.basicConfig(
            level=getattr(logging, self.config.system.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger(__name__)
        logger.info("Starting ScrollIntel™ API Gateway")
        logger.info(f"Environment: {self.config.environment}")
        logger.info(f"Debug mode: {self.config.debug}")
        
        # Start error monitoring system
        logger.info("Starting error monitoring system...")
        await start_error_monitoring()
        logger.info("Error monitoring system started")
        
        # Initialize agent registry
        logger.info("Initializing agent registry...")
        
        # TODO: Register default agents here when they are implemented
        # For now, we'll just log that the registry is ready
        logger.info("Agent registry initialized")
        
        logger.info("ScrollIntel™ API Gateway started successfully")
    
    async def _shutdown(self) -> None:
        """Application shutdown tasks."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Shutting down ScrollIntel™ API Gateway")
        
        # Stop error monitoring system
        logger.info("Stopping error monitoring system...")
        await stop_error_monitoring()
        logger.info("Error monitoring system stopped")
        
        # Perform health check on all agents before shutdown
        try:
            health_results = await self.agent_registry.health_check_all()
            unhealthy_agents = [agent_id for agent_id, healthy in health_results.items() if not healthy]
            if unhealthy_agents:
                logger.warning(f"Unhealthy agents during shutdown: {unhealthy_agents}")
        except Exception as e:
            logger.error(f"Error during agent health check: {str(e)}")
        
        # Clean up resources
        # TODO: Add cleanup for database connections, Redis, etc.
        
        logger.info("ScrollIntel™ API Gateway shutdown complete")
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app


# Global gateway instance
gateway = ScrollIntelGateway()
app = gateway.get_app()


# Convenience function for getting the application
def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    return gateway.get_app()


# For uvicorn and other ASGI servers
def get_application() -> FastAPI:
    """Get the application instance for ASGI servers."""
    return app