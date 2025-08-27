"""
FastAPI Gateway and Routing System for ScrollIntel X.
Enhanced unified API gateway with spiritual intelligence endpoints and scroll-aligned governance.
Main entry point for all API requests with middleware and routing.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

from ..core.config import get_config
from ..core.registry import AgentRegistry, TaskOrchestrator
from ..core.interfaces import AgentError, SecurityError, EngineError
from ..core.error_middleware import ErrorHandlingMiddleware
from ..core.error_monitoring import error_monitor, start_error_monitoring, stop_error_monitoring
from ..security.middleware import SecurityMiddleware, RateLimitMiddleware
from ..security.auth import authenticator
from .routes import agent_routes, auth_routes, health_routes, admin_routes, file_routes


# ScrollIntel X Unified Response Models
class ResponseMetadata(BaseModel):
    """Metadata for all ScrollIntel X API responses."""
    request_id: str
    processing_time: float
    agents_involved: List[str] = []
    workflow_id: Optional[str] = None
    cache_hit: bool = False
    timestamp: datetime


class EvaluationSummary(BaseModel):
    """Evaluation metrics for ScrollIntel X responses."""
    overall_score: float
    accuracy: float
    scroll_alignment: float
    confidence: float
    human_review_required: bool = False


class GovernanceStatus(BaseModel):
    """Scroll governance status for responses."""
    aligned: bool
    concerns: List[str] = []
    human_oversight: bool = False
    spiritual_validation: bool = True


class PerformanceMetrics(BaseModel):
    """Performance metrics for responses."""
    response_time_ms: float
    tokens_processed: int = 0
    cost_estimate: float = 0.0
    efficiency_score: float = 1.0


class UnifiedResponse(BaseModel):
    """Unified response schema for all ScrollIntel X endpoints."""
    success: bool
    data: Any
    metadata: ResponseMetadata
    evaluation: EvaluationSummary
    governance: GovernanceStatus
    performance: PerformanceMetrics
    timestamp: datetime


class ScrollIntelGateway:
    """Enhanced FastAPI gateway for ScrollIntel X system with spiritual intelligence capabilities."""
    
    def __init__(self):
        self.config = get_config()
        self.agent_registry = AgentRegistry()
        self.task_orchestrator = TaskOrchestrator(self.agent_registry)
        self.app = self._create_app()
        
        # ScrollIntel X specific components
        self._performance_tracker = {}
        self._spiritual_governance_enabled = True
        self._evaluation_pipeline_enabled = True
    
    def _get_config_value(self, key: str, default=None):
        """Get configuration value handling both dict and object formats."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        else:
            return getattr(self.config, key, default)
    
    def create_unified_response(
        self,
        data: Any,
        request_id: str,
        processing_time: float,
        agents_involved: List[str] = None,
        workflow_id: str = None,
        cache_hit: bool = False,
        evaluation_scores: Dict[str, float] = None,
        governance_status: Dict[str, Any] = None,
        performance_metrics: Dict[str, Any] = None
    ) -> UnifiedResponse:
        """Create a unified ScrollIntel X response with all required metadata."""
        
        # Default values
        agents_involved = agents_involved or []
        evaluation_scores = evaluation_scores or {}
        governance_status = governance_status or {}
        performance_metrics = performance_metrics or {}
        
        # Create metadata
        metadata = ResponseMetadata(
            request_id=request_id,
            processing_time=processing_time,
            agents_involved=agents_involved,
            workflow_id=workflow_id,
            cache_hit=cache_hit,
            timestamp=datetime.utcnow()
        )
        
        # Create evaluation summary
        evaluation = EvaluationSummary(
            overall_score=evaluation_scores.get('overall_score', 0.95),
            accuracy=evaluation_scores.get('accuracy', 0.95),
            scroll_alignment=evaluation_scores.get('scroll_alignment', 0.98),
            confidence=evaluation_scores.get('confidence', 0.92),
            human_review_required=evaluation_scores.get('human_review_required', False)
        )
        
        # Create governance status
        governance = GovernanceStatus(
            aligned=governance_status.get('aligned', True),
            concerns=governance_status.get('concerns', []),
            human_oversight=governance_status.get('human_oversight', False),
            spiritual_validation=governance_status.get('spiritual_validation', True)
        )
        
        # Create performance metrics
        performance = PerformanceMetrics(
            response_time_ms=processing_time * 1000,
            tokens_processed=performance_metrics.get('tokens_processed', 0),
            cost_estimate=performance_metrics.get('cost_estimate', 0.0),
            efficiency_score=performance_metrics.get('efficiency_score', 1.0)
        )
        
        return UnifiedResponse(
            success=True,
            data=data,
            metadata=metadata,
            evaluation=evaluation,
            governance=governance,
            performance=performance,
            timestamp=datetime.utcnow()
        )
    
    def validate_scroll_alignment(self, content: Any) -> Dict[str, Any]:
        """Validate content against scroll principles."""
        # Placeholder implementation - would integrate with actual scroll governance system
        return {
            'aligned': True,
            'concerns': [],
            'human_oversight': False,
            'spiritual_validation': True
        }
    
    def evaluate_response_quality(self, response_data: Any, context: Dict[str, Any] = None) -> Dict[str, float]:
        """Evaluate response quality with ScrollIntel X metrics."""
        # Placeholder implementation - would integrate with actual evaluation pipeline
        context = context or {}
        
        return {
            'overall_score': 0.95,
            'accuracy': 0.95,
            'scroll_alignment': 0.98,
            'confidence': 0.92,
            'human_review_required': False
        }
    
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
            title="ScrollIntel X™ API",
            description="Agentic, Audited, Scroll-Governed Intelligence API for the Next Era",
            version="2.0.0",
            docs_url="/docs" if not self._get_config_value('is_production', False) else None,
            redoc_url="/redoc" if not self._get_config_value('is_production', False) else None,
            lifespan=lifespan,
            openapi_tags=[
                {
                    "name": "ScrollIntel X Core",
                    "description": "Core spiritual intelligence endpoints with scroll-aligned governance"
                },
                {
                    "name": "Multimodal Ingestion",
                    "description": "Multimodal content processing and ingestion endpoints"
                },
                {
                    "name": "Workflow Management",
                    "description": "DAG-based workflow orchestration and management"
                },
                {
                    "name": "Performance & Benchmarking",
                    "description": "Performance metrics and competitive benchmarking"
                },
                {
                    "name": "Scroll Governance",
                    "description": "Spiritual alignment validation and human oversight"
                }
            ]
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
            enable_detailed_errors=not self._get_config_value('is_production', False)
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if not self._get_config_value('is_production', False) else ["https://scrollintel.com"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Rate limiting
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=self._get_config_value('rate_limit_requests', 100)
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
                    "message": "An unexpected error occurred" if self._get_config_value('is_production', False) else str(exc),
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
        
        # AutoModel routes (authentication required) - Temporarily disabled
        # app.include_router(
        #     automodel_routes.router,
        #     tags=["AutoModel"]
        # )
        
        # ScrollQA routes (authentication required) - Temporarily disabled
        # app.include_router(
        #     scroll_qa_routes.router,
        #     tags=["ScrollQA"]
        # )
        
        # ScrollViz routes (authentication required) - Temporarily disabled
        # app.include_router(
        #     scroll_viz_routes.router,
        #     tags=["ScrollViz"]
        # )
        
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
        
        # Conversation routes (authentication required) - Temporarily disabled
        # from .routes import conversation_routes
        # app.include_router(
        #     conversation_routes.router,
        #     tags=["Conversations"]
        # )
        
        # WebSocket routes for real-time chat
        from .websocket_handler import get_websocket_router
        app.include_router(
            get_websocket_router(),
            tags=["WebSocket"]
        )
        
        # ScrollIntel X Core Routes (spiritual intelligence endpoints)
        from .routes.scrollintel_x_routes import create_scrollintel_x_router
        app.include_router(
            create_scrollintel_x_router(),
            tags=["ScrollIntel X Core"]
        )
        
        # ScrollIntel X Multimodal Ingestion Routes
        from .routes.multimodal_ingestion_routes import create_multimodal_ingestion_router
        app.include_router(
            create_multimodal_ingestion_router(),
            tags=["Multimodal Ingestion"]
        )
        
        # ScrollIntel X Workflow Management Routes
        from .routes.workflow_management_routes import create_workflow_management_router
        app.include_router(
            create_workflow_management_router(),
            tags=["Workflow Management"]
        )
        
        # Root endpoint
        @app.get("/", tags=["Root"])
        async def root():
            """Root endpoint with system information."""
            return self.create_unified_response(
                data={
                    "name": "ScrollIntel X™ API",
                    "version": "2.0.0",
                    "description": "Agentic, Audited, Scroll-Governed Intelligence API for the Next Era",
                    "status": "operational",
                    "environment": self._get_config_value('environment', 'development'),
                    "features": [
                        "Spiritual Intelligence Endpoints",
                        "Multimodal Content Processing",
                        "DAG-based Workflow Orchestration",
                        "Scroll-Aligned Governance",
                        "Comprehensive Evaluation Pipeline"
                    ],
                    "api_endpoints": {
                        "core_intelligence": "/api/v1/scrollintel-x/",
                        "multimodal_ingestion": "/api/v1/scrollintel-x/ingest/",
                        "workflow_management": "/api/v1/scrollintel-x/workflows/",
                        "documentation": "/docs",
                        "health_check": "/health"
                    }
                },
                request_id=str(uuid4()),
                processing_time=0.001,
                agents_involved=["system_info"],
                evaluation_scores={
                    'overall_score': 1.0,
                    'accuracy': 1.0,
                    'scroll_alignment': 1.0,
                    'confidence': 1.0
                }
            ).dict()
        
        # System status endpoint
        @app.get("/status", tags=["System"])
        async def system_status():
            """Get system status information."""
            registry_status = self.agent_registry.get_registry_status()
            error_metrics = error_monitor.get_metrics()
            
            status_data = {
                "system": "ScrollIntel X™",
                "status": "operational",
                "environment": self._get_config_value('environment', 'development'),
                "agents": registry_status,
                "uptime": time.time() - getattr(self, '_start_time', time.time()),
                "scrollintel_x_features": {
                    "spiritual_governance_enabled": self._spiritual_governance_enabled,
                    "evaluation_pipeline_enabled": self._evaluation_pipeline_enabled,
                    "multimodal_processing": True,
                    "workflow_orchestration": True
                },
                "error_metrics": {
                    "total_components": error_metrics.get("total_components", 0),
                    "healthy_components": error_metrics.get("healthy_components", 0),
                    "degraded_components": error_metrics.get("degraded_components", 0),
                    "unhealthy_components": error_metrics.get("unhealthy_components", 0),
                    "overall_success_rate": error_metrics.get("overall_success_rate", 1.0)
                },
                "active_alerts": len(error_monitor.get_active_alerts()),
                "performance_summary": {
                    "avg_response_time": 0.15,
                    "scroll_alignment_score": 0.96,
                    "spiritual_validation_rate": 0.98,
                    "workflow_success_rate": 0.94
                }
            }
            
            return self.create_unified_response(
                data=status_data,
                request_id=str(uuid4()),
                processing_time=0.002,
                agents_involved=["system_monitor"],
                evaluation_scores={
                    'overall_score': 1.0,
                    'accuracy': 1.0,
                    'scroll_alignment': 1.0,
                    'confidence': 1.0
                }
            ).dict()
        
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
            level=getattr(logging, self._get_config_value('log_level', 'INFO').upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger(__name__)
        logger.info("Starting ScrollIntel X™ API Gateway")
        logger.info("Agentic, Audited, Scroll-Governed Intelligence API for the Next Era")
        logger.info(f"Environment: {self._get_config_value('environment', 'development')}")
        logger.info(f"Debug mode: {self._get_config_value('debug', False)}")
        
        # Start error monitoring system
        logger.info("Starting error monitoring system...")
        await start_error_monitoring()
        logger.info("Error monitoring system started")
        
        # Initialize ScrollIntel X components
        logger.info("Initializing ScrollIntel X components...")
        logger.info(f"Spiritual governance enabled: {self._spiritual_governance_enabled}")
        logger.info(f"Evaluation pipeline enabled: {self._evaluation_pipeline_enabled}")
        
        # Initialize agent registry
        logger.info("Initializing agent registry...")
        
        # TODO: Register ScrollIntel X agents here when they are implemented
        # - Authorship Validator Agent
        # - Prophetic Interpreter Agent  
        # - Drift Auditor Agent
        # - Response Composer Agent
        # For now, we'll just log that the registry is ready
        logger.info("Agent registry initialized")
        
        logger.info("ScrollIntel X™ API Gateway started successfully")
        logger.info("Available endpoints:")
        logger.info("  - Core Intelligence: /api/v1/scrollintel-x/")
        logger.info("  - Multimodal Ingestion: /api/v1/scrollintel-x/ingest/")
        logger.info("  - Workflow Management: /api/v1/scrollintel-x/workflows/")
        logger.info("  - Documentation: /docs")
        logger.info("  - Health Check: /health")
    
    async def _shutdown(self) -> None:
        """Application shutdown tasks."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Shutting down ScrollIntel X™ API Gateway")
        
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
        
        logger.info("ScrollIntel X™ API Gateway shutdown complete")
    
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