"""
Advanced Analytics Dashboard API Gateway

This module provides the main API gateway for the Advanced Analytics Dashboard System,
including REST endpoints, GraphQL API, webhook system, authentication, and rate limiting.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import json
import asyncio
from contextlib import asynccontextmanager

# Import route modules
from .routes.dashboard_routes import router as dashboard_router
from .routes.roi_routes import router as roi_router
from .routes.insight_routes import router as insight_router
from .routes.predictive_routes import router as predictive_router
from .routes.advanced_analytics_routes import router as advanced_analytics_router

# Import core components
from ..core.dashboard_manager import DashboardManager
from ..engines.roi_calculator import ROICalculator
from ..engines.insight_generator import InsightGenerator
from ..engines.predictive_engine import PredictiveEngine
from ..core.websocket_manager import WebSocketManager
from ..core.api_key_manager import APIKeyManager
from ..core.audit_system import AuditSystem

# Import GraphQL components
from .graphql.analytics_schema import analytics_schema
from .graphql.analytics_resolvers import AnalyticsResolvers

# Import webhook system
from .webhooks.webhook_manager import WebhookManager
from .webhooks.webhook_handlers import WebhookHandlers

# Import middleware
from .middleware.rate_limiter import RateLimitMiddleware
from .middleware.auth_middleware import AuthMiddleware
from .middleware.audit_middleware import AuditMiddleware
from .middleware.error_handler import ErrorHandlerMiddleware

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

class AnalyticsDashboardAPI:
    """Main API gateway for Advanced Analytics Dashboard System"""
    
    def __init__(self):
        self.app = None
        self.dashboard_manager = DashboardManager()
        self.roi_calculator = ROICalculator()
        self.insight_generator = InsightGenerator()
        self.predictive_engine = PredictiveEngine()
        self.websocket_manager = WebSocketManager()
        self.api_key_manager = APIKeyManager()
        self.audit_system = AuditSystem()
        self.webhook_manager = WebhookManager()
        self.webhook_handlers = WebhookHandlers()
        self.analytics_resolvers = AnalyticsResolvers()
        
        # Rate limiting configuration
        self.rate_limits = {
            "default": {"requests": 100, "window": 60},  # 100 requests per minute
            "premium": {"requests": 1000, "window": 60},  # 1000 requests per minute
            "enterprise": {"requests": 10000, "window": 60}  # 10000 requests per minute
        }
        
    async def create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self._startup()
            yield
            # Shutdown
            await self._shutdown()
        
        self.app = FastAPI(
            title="Advanced Analytics Dashboard API",
            description="Comprehensive API for executive analytics, ROI tracking, and business intelligence",
            version="1.0.0",
            docs_url=None,  # We'll create custom docs
            redoc_url=None,
            lifespan=lifespan
        )
        
        # Add middleware
        await self._setup_middleware()
        
        # Add routes
        await self._setup_routes()
        
        # Setup GraphQL
        await self._setup_graphql()
        
        # Setup webhooks
        await self._setup_webhooks()
        
        # Setup WebSocket
        await self._setup_websockets()
        
        # Setup custom documentation
        await self._setup_documentation()
        
        return self.app
    
    async def _startup(self):
        """Application startup tasks"""
        logger.info("Starting Advanced Analytics Dashboard API...")
        
        # Initialize components
        await self.dashboard_manager.initialize()
        await self.roi_calculator.initialize()
        await self.insight_generator.initialize()
        await self.predictive_engine.initialize()
        await self.websocket_manager.initialize()
        await self.api_key_manager.initialize()
        await self.audit_system.initialize()
        await self.webhook_manager.initialize()
        
        logger.info("Advanced Analytics Dashboard API started successfully")
    
    async def _shutdown(self):
        """Application shutdown tasks"""
        logger.info("Shutting down Advanced Analytics Dashboard API...")
        
        # Cleanup components
        await self.websocket_manager.cleanup()
        await self.webhook_manager.cleanup()
        await self.audit_system.cleanup()
        
        logger.info("Advanced Analytics Dashboard API shutdown complete")
    
    async def _setup_middleware(self):
        """Setup middleware stack"""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure appropriately for production
        )
        
        # Custom middleware
        self.app.add_middleware(ErrorHandlerMiddleware)
        self.app.add_middleware(AuditMiddleware, audit_system=self.audit_system)
        self.app.add_middleware(AuthMiddleware, api_key_manager=self.api_key_manager)
        self.app.add_middleware(RateLimitMiddleware, rate_limits=self.rate_limits)
    
    async def _setup_routes(self):
        """Setup API routes"""
        
        # Include route modules
        self.app.include_router(dashboard_router, prefix="/api/v1")
        self.app.include_router(roi_router, prefix="/api/v1")
        self.app.include_router(insight_router, prefix="/api/v1")
        self.app.include_router(predictive_router, prefix="/api/v1")
        self.app.include_router(advanced_analytics_router, prefix="/api/v1")
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "components": {
                    "dashboard_manager": await self.dashboard_manager.health_check(),
                    "roi_calculator": await self.roi_calculator.health_check(),
                    "insight_generator": await self.insight_generator.health_check(),
                    "predictive_engine": await self.predictive_engine.health_check(),
                    "websocket_manager": await self.websocket_manager.health_check(),
                    "webhook_manager": await self.webhook_manager.health_check()
                }
            }
        
        # API info endpoint
        @self.app.get("/api/info")
        async def api_info():
            """API information endpoint"""
            return {
                "name": "Advanced Analytics Dashboard API",
                "version": "1.0.0",
                "description": "Comprehensive API for executive analytics, ROI tracking, and business intelligence",
                "endpoints": {
                    "rest": "/api/v1",
                    "graphql": "/graphql",
                    "websocket": "/ws",
                    "webhooks": "/webhooks",
                    "docs": "/docs",
                    "redoc": "/redoc"
                },
                "features": [
                    "Executive Dashboards",
                    "ROI Tracking",
                    "AI Insights",
                    "Predictive Analytics",
                    "Real-time Updates",
                    "Multi-source Integration",
                    "Custom Templates",
                    "Automated Reporting"
                ]
            }
    
    async def _setup_graphql(self):
        """Setup GraphQL API"""
        from graphql import build_schema
        from starlette.graphql import GraphQLApp
        
        # Create GraphQL app
        graphql_app = GraphQLApp(
            schema=analytics_schema,
            context_value={"resolvers": self.analytics_resolvers}
        )
        
        # Mount GraphQL endpoint
        self.app.mount("/graphql", graphql_app)
        
        # GraphQL playground
        @self.app.get("/graphql/playground")
        async def graphql_playground():
            """GraphQL Playground interface"""
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>GraphQL Playground</title>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/graphql-playground-react/build/static/css/index.css" />
            </head>
            <body>
                <div id="root">
                    <style>
                        body { margin: 0; font-family: Open Sans, sans-serif; overflow: hidden; }
                        #root { height: 100vh; }
                    </style>
                </div>
                <script src="https://cdn.jsdelivr.net/npm/graphql-playground-react/build/static/js/middleware.js"></script>
            </body>
            </html>
            """)
    
    async def _setup_webhooks(self):
        """Setup webhook system"""
        
        # Webhook registration endpoint
        @self.app.post("/api/v1/webhooks")
        async def register_webhook(webhook_config: Dict[str, Any]):
            """Register a new webhook"""
            try:
                webhook_id = await self.webhook_manager.register_webhook(webhook_config)
                return {"webhook_id": webhook_id, "status": "registered"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Webhook management endpoints
        @self.app.get("/api/v1/webhooks")
        async def list_webhooks():
            """List all registered webhooks"""
            return await self.webhook_manager.list_webhooks()
        
        @self.app.get("/api/v1/webhooks/{webhook_id}")
        async def get_webhook(webhook_id: str):
            """Get webhook details"""
            webhook = await self.webhook_manager.get_webhook(webhook_id)
            if not webhook:
                raise HTTPException(status_code=404, detail="Webhook not found")
            return webhook
        
        @self.app.put("/api/v1/webhooks/{webhook_id}")
        async def update_webhook(webhook_id: str, updates: Dict[str, Any]):
            """Update webhook configuration"""
            success = await self.webhook_manager.update_webhook(webhook_id, updates)
            if not success:
                raise HTTPException(status_code=404, detail="Webhook not found")
            return {"status": "updated"}
        
        @self.app.delete("/api/v1/webhooks/{webhook_id}")
        async def delete_webhook(webhook_id: str):
            """Delete a webhook"""
            success = await self.webhook_manager.delete_webhook(webhook_id)
            if not success:
                raise HTTPException(status_code=404, detail="Webhook not found")
            return {"status": "deleted"}
        
        # Webhook testing endpoint
        @self.app.post("/api/v1/webhooks/{webhook_id}/test")
        async def test_webhook(webhook_id: str, test_payload: Optional[Dict] = None):
            """Test a webhook"""
            result = await self.webhook_manager.test_webhook(webhook_id, test_payload)
            return result
        
        # Webhook delivery status
        @self.app.get("/api/v1/webhooks/{webhook_id}/deliveries")
        async def get_webhook_deliveries(webhook_id: str):
            """Get webhook delivery history"""
            deliveries = await self.webhook_manager.get_delivery_history(webhook_id)
            return {"deliveries": deliveries}
    
    async def _setup_websockets(self):
        """Setup WebSocket connections"""
        
        @self.app.websocket("/ws/dashboard/{dashboard_id}")
        async def dashboard_websocket(websocket, dashboard_id: str):
            """WebSocket endpoint for real-time dashboard updates"""
            await self.websocket_manager.handle_dashboard_connection(websocket, dashboard_id)
        
        @self.app.websocket("/ws/insights")
        async def insights_websocket(websocket):
            """WebSocket endpoint for real-time insights"""
            await self.websocket_manager.handle_insights_connection(websocket)
        
        @self.app.websocket("/ws/alerts")
        async def alerts_websocket(websocket):
            """WebSocket endpoint for real-time alerts"""
            await self.websocket_manager.handle_alerts_connection(websocket)
    
    async def _setup_documentation(self):
        """Setup custom API documentation"""
        
        @self.app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html():
            """Custom Swagger UI with enhanced styling"""
            return get_swagger_ui_html(
                openapi_url="/openapi.json",
                title="Advanced Analytics Dashboard API - Documentation",
                swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui-bundle.js",
                swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui.css",
            )
        
        @self.app.get("/redoc", include_in_schema=False)
        async def redoc_html():
            """ReDoc documentation"""
            return get_redoc_html(
                openapi_url="/openapi.json",
                title="Advanced Analytics Dashboard API - Documentation",
            )
        
        @self.app.get("/openapi.json", include_in_schema=False)
        async def custom_openapi():
            """Custom OpenAPI schema with enhanced metadata"""
            if self.app.openapi_schema:
                return self.app.openapi_schema
            
            openapi_schema = get_openapi(
                title="Advanced Analytics Dashboard API",
                version="1.0.0",
                description="""
                ## Advanced Analytics Dashboard API
                
                Comprehensive API for executive analytics, ROI tracking, business intelligence, 
                and automated insight generation.
                
                ### Features
                - **Executive Dashboards**: Role-specific dashboards with real-time metrics
                - **ROI Tracking**: Comprehensive financial impact analysis
                - **AI Insights**: Automated insight generation with natural language explanations
                - **Predictive Analytics**: Forecasting and scenario modeling
                - **Real-time Updates**: WebSocket-based live data streaming
                - **Multi-source Integration**: Connect to ERP, CRM, BI tools, and cloud platforms
                - **Custom Templates**: Industry-specific dashboard templates
                - **Automated Reporting**: Scheduled report generation and distribution
                
                ### Authentication
                API uses API key authentication. Include your API key in the `Authorization` header:
                ```
                Authorization: Bearer YOUR_API_KEY
                ```
                
                ### Rate Limits
                - **Default**: 100 requests per minute
                - **Premium**: 1,000 requests per minute  
                - **Enterprise**: 10,000 requests per minute
                
                ### WebSocket Endpoints
                - `/ws/dashboard/{dashboard_id}` - Real-time dashboard updates
                - `/ws/insights` - Real-time insights stream
                - `/ws/alerts` - Real-time alerts and notifications
                
                ### GraphQL
                GraphQL endpoint available at `/graphql` with playground at `/graphql/playground`
                """,
                routes=self.app.routes,
            )
            
            # Add custom metadata
            openapi_schema["info"]["contact"] = {
                "name": "ScrollIntel Support",
                "email": "support@scrollintel.com",
                "url": "https://scrollintel.com/support"
            }
            
            openapi_schema["info"]["license"] = {
                "name": "Commercial License",
                "url": "https://scrollintel.com/license"
            }
            
            # Add security schemes
            openapi_schema["components"]["securitySchemes"] = {
                "ApiKeyAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "API Key"
                }
            }
            
            # Add global security requirement
            openapi_schema["security"] = [{"ApiKeyAuth": []}]
            
            # Add tags
            openapi_schema["tags"] = [
                {
                    "name": "Dashboards",
                    "description": "Executive dashboard management and real-time metrics"
                },
                {
                    "name": "ROI Tracking",
                    "description": "Financial impact analysis and ROI calculations"
                },
                {
                    "name": "AI Insights",
                    "description": "Automated insight generation and natural language explanations"
                },
                {
                    "name": "Predictive Analytics",
                    "description": "Forecasting, scenario modeling, and risk prediction"
                },
                {
                    "name": "Data Integration",
                    "description": "Multi-source data connectors and normalization"
                },
                {
                    "name": "Templates",
                    "description": "Dashboard templates and customization"
                },
                {
                    "name": "Webhooks",
                    "description": "Webhook registration and management"
                },
                {
                    "name": "System",
                    "description": "Health checks and system information"
                }
            ]
            
            self.app.openapi_schema = openapi_schema
            return self.app.openapi_schema


# Global API instance
analytics_api = AnalyticsDashboardAPI()

# Factory function for creating the app
async def create_analytics_dashboard_app() -> FastAPI:
    """Factory function to create the Analytics Dashboard API application"""
    return await analytics_api.create_app()

# For direct usage
def get_app() -> FastAPI:
    """Get the FastAPI application instance"""
    if analytics_api.app is None:
        raise RuntimeError("Application not initialized. Call create_analytics_dashboard_app() first.")
    return analytics_api.app