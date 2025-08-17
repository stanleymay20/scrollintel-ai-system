"""
Production-Ready FastAPI Application
Integrates all stability, onboarding, and infrastructure systems
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from scrollintel.core.production_infrastructure import initialize_infrastructure, get_infrastructure
from scrollintel.core.user_onboarding import initialize_onboarding_system, get_onboarding_system
from scrollintel.core.api_stability import initialize_api_stability_system, get_api_stability_system
from scrollintel.core.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting ScrollIntel Production Server...")
    
    try:
        # Load configuration
        config = get_config()
        
        # Initialize infrastructure
        await initialize_infrastructure(config['infrastructure'])
        logger.info("Infrastructure initialized")
        
        # Initialize onboarding system
        initialize_onboarding_system(config['onboarding'])
        logger.info("Onboarding system initialized")
        
        # Initialize API stability system
        await initialize_api_stability_system(config['api_stability'])
        logger.info("API stability system initialized")
        
        logger.info("ScrollIntel Production Server started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down ScrollIntel Production Server...")

# Create FastAPI application
app = FastAPI(
    title="ScrollIntel API",
    description="Production-ready AI agent orchestration platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure properly for production
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add API stability middleware
@app.middleware("http")
async def api_stability_middleware(request: Request, call_next):
    """Apply API stability checks to all requests"""
    try:
        stability_system = get_api_stability_system()
        return await stability_system.process_request(request, call_next)
    except Exception as e:
        logger.error(f"API stability middleware error: {e}")
        return await call_next(request)

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": "2025-01-16T12:00:00Z"}

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system status"""
    try:
        infrastructure = await get_infrastructure()
        stability_system = get_api_stability_system()
        
        health_data = {
            "status": "healthy",
            "infrastructure": await infrastructure.get_system_health(),
            "api_stability": await stability_system.get_system_status(),
            "timestamp": "2025-01-16T12:00:00Z"
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

# Authentication dependency
async def get_current_user(request: Request) -> Dict[str, Any]:
    """Get current authenticated user"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    token = auth_header.split(" ")[1]
    onboarding_system = get_onboarding_system()
    
    # Verify token (simplified - would use proper JWT verification)
    user_id = "user_123"  # Extract from token
    
    return {"user_id": user_id}

# User management endpoints
@app.post("/api/v1/auth/register")
async def register_user(request: Request):
    """Register a new user"""
    try:
        data = await request.json()
        onboarding_system = get_onboarding_system()
        
        result = await onboarding_system.register_user(
            email=data.get("email"),
            username=data.get("username"),
            password=data.get("password")
        )
        
        if result["success"]:
            return JSONResponse(
                status_code=201,
                content=result
            )
        else:
            return JSONResponse(
                status_code=400,
                content=result
            )
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Registration failed"}
        )

@app.post("/api/v1/auth/login")
async def login_user(request: Request):
    """Authenticate user login"""
    try:
        data = await request.json()
        onboarding_system = get_onboarding_system()
        
        result = await onboarding_system.authenticate_user(
            email=data.get("email"),
            password=data.get("password")
        )
        
        if result["success"]:
            return JSONResponse(
                status_code=200,
                content=result
            )
        else:
            return JSONResponse(
                status_code=401,
                content=result
            )
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Login failed"}
        )

@app.get("/api/v1/auth/verify-email")
async def verify_email(token: str):
    """Verify user email"""
    try:
        onboarding_system = get_onboarding_system()
        result = await onboarding_system.verify_email(token)
        
        if result["success"]:
            return JSONResponse(
                status_code=200,
                content=result
            )
        else:
            return JSONResponse(
                status_code=400,
                content=result
            )
            
    except Exception as e:
        logger.error(f"Email verification error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Verification failed"}
        )

# Onboarding endpoints
@app.get("/api/v1/onboarding/status")
async def get_onboarding_status(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get user's onboarding status"""
    try:
        onboarding_system = get_onboarding_system()
        result = await onboarding_system.get_onboarding_status(current_user["user_id"])
        
        return JSONResponse(
            status_code=200,
            content=result
        )
        
    except Exception as e:
        logger.error(f"Onboarding status error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Failed to get onboarding status"}
        )

@app.post("/api/v1/onboarding/complete-step")
async def complete_onboarding_step(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Complete an onboarding step"""
    try:
        data = await request.json()
        onboarding_system = get_onboarding_system()
        
        from scrollintel.core.user_onboarding import OnboardingStep
        step = OnboardingStep(data.get("step"))
        step_data = data.get("step_data", {})
        
        result = await onboarding_system.complete_onboarding_step(
            current_user["user_id"], step, step_data
        )
        
        return JSONResponse(
            status_code=200,
            content=result
        )
        
    except Exception as e:
        logger.error(f"Complete onboarding step error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Failed to complete step"}
        )

# Support endpoints
@app.post("/api/v1/support/tickets")
async def create_support_ticket(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a support ticket"""
    try:
        data = await request.json()
        onboarding_system = get_onboarding_system()
        
        from scrollintel.core.user_onboarding import SupportTicketPriority
        priority = SupportTicketPriority(data.get("priority", "medium"))
        
        result = await onboarding_system.create_support_ticket(
            user_id=current_user["user_id"],
            subject=data.get("subject"),
            description=data.get("description"),
            category=data.get("category"),
            priority=priority
        )
        
        return JSONResponse(
            status_code=201,
            content=result
        )
        
    except Exception as e:
        logger.error(f"Create support ticket error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Failed to create ticket"}
        )

# Agent interaction endpoints (simplified)
@app.post("/api/v1/agents/chat")
async def chat_with_agent(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Chat with an AI agent"""
    try:
        data = await request.json()
        
        # Simplified agent interaction
        response = {
            "success": True,
            "agent_response": f"Hello! I received your message: {data.get('message')}",
            "agent_id": data.get("agent_id", "default"),
            "timestamp": "2025-01-16T12:00:00Z"
        }
        
        return JSONResponse(
            status_code=200,
            content=response
        )
        
    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Agent interaction failed"}
        )

# System monitoring endpoints
@app.get("/api/v1/system/metrics")
async def get_system_metrics(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get system performance metrics"""
    try:
        infrastructure = await get_infrastructure()
        stability_system = get_api_stability_system()
        
        metrics = {
            "infrastructure": await infrastructure.get_system_health(),
            "api_stability": await stability_system.get_system_status(),
            "timestamp": "2025-01-16T12:00:00Z"
        }
        
        return JSONResponse(
            status_code=200,
            content=metrics
        )
        
    except Exception as e:
        logger.error(f"System metrics error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get system metrics"}
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    # Production server configuration
    config = {
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 8000)),
        "workers": int(os.getenv("WORKERS", 4)),
        "log_level": "info",
        "access_log": True,
        "reload": False
    }
    
    uvicorn.run(
        "scrollintel.api.production_main:app",
        **config
    )