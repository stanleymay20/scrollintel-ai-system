"""
Security middleware for FastAPI routes in EXOUSIA system.
Handles authentication, authorization, and security headers.
"""

import time
import asyncio
from typing import Optional, Callable, Dict, Any, List
from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..core.interfaces import SecurityContext, SecurityError
from .auth import authenticator
from .session import session_manager
from .permissions import permission_checker, Permission, ResourceType, Action
from .audit import audit_logger, AuditAction


class SecurityMiddleware(BaseHTTPMiddleware):
    """Main security middleware for request processing."""
    
    def __init__(self, app, excluded_paths: Optional[List[str]] = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/health",
            "/auth/login",
            "/auth/register"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security middleware."""
        start_time = time.time()
        
        # Skip security for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        try:
            # Extract security context
            context = await self._extract_security_context(request)
            
            # Add context to request state
            request.state.security_context = context
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            # Log successful request
            processing_time = time.time() - start_time
            await self._log_request(request, context, True, processing_time)
            
            return response
            
        except SecurityError as e:
            # Log security error
            await self._log_security_error(request, str(e))
            
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Authentication failed",
                    "message": str(e),
                    "timestamp": time.time()
                }
            )
        
        except HTTPException as e:
            # Log HTTP error
            await self._log_request(request, None, False, time.time() - start_time, str(e))
            raise
        
        except Exception as e:
            # Log unexpected error
            await self._log_request(request, None, False, time.time() - start_time, str(e))
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "timestamp": time.time()
                }
            )
    
    async def _extract_security_context(self, request: Request) -> SecurityContext:
        """Extract security context from request."""
        # Try to get token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise SecurityError("Authorization header missing")
        
        if not auth_header.startswith("Bearer "):
            raise SecurityError("Invalid authorization header format")
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        # Validate JWT token
        context = authenticator.get_user_from_token(token)
        if not context:
            raise SecurityError("Invalid or expired token")
        
        # Get client IP address
        client_ip = self._get_client_ip(request)
        context.ip_address = client_ip
        
        # Try to validate session if session ID is available
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            session_context = await session_manager.validate_session(session_id, client_ip)
            if session_context:
                context.session_id = session_id
                # Update context with session data
                context.role = session_context.role
                context.permissions = session_context.permissions
        
        return context
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers (when behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"
    
    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    async def _log_request(self, request: Request, context: Optional[SecurityContext],
                          success: bool, processing_time: float,
                          error_message: Optional[str] = None) -> None:
        """Log request for audit purposes."""
        try:
            details = {
                "method": request.method,
                "path": str(request.url.path),
                "query_params": dict(request.query_params),
                "processing_time": processing_time,
                "user_agent": request.headers.get("User-Agent", "")
            }
            
            await audit_logger.log(
                action="api.request",
                resource_type="api",
                resource_id=str(request.url.path),
                details=details,
                user_id=context.user_id if context else None,
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("User-Agent", ""),
                session_id=context.session_id if context else None,
                success=success,
                error_message=error_message
            )
        except Exception:
            # Don't let audit logging errors affect the main request
            pass
    
    async def _log_security_error(self, request: Request, error_message: str) -> None:
        """Log security errors."""
        try:
            details = {
                "method": request.method,
                "path": str(request.url.path),
                "user_agent": request.headers.get("User-Agent", ""),
                "authorization_header_present": bool(request.headers.get("Authorization"))
            }
            
            await audit_logger.log(
                action=AuditAction.PERMISSION_DENIED,
                resource_type="api",
                resource_id=str(request.url.path),
                details=details,
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("User-Agent", ""),
                success=False,
                error_message=error_message
            )
        except Exception:
            # Don't let audit logging errors affect the main request
            pass


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, List[float]] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting."""
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.request_counts:
            self.request_counts[client_ip] = [
                req_time for req_time in self.request_counts[client_ip]
                if current_time - req_time < 60  # Keep requests from last minute
            ]
        else:
            self.request_counts[client_ip] = []
        
        # Check rate limit
        if len(self.request_counts[client_ip]) >= self.requests_per_minute:
            # Log rate limit exceeded
            await audit_logger.log(
                action=AuditAction.RATE_LIMIT_EXCEEDED,
                resource_type="api",
                resource_id=str(request.url.path),
                ip_address=client_ip,
                details={
                    "requests_per_minute": self.requests_per_minute,
                    "current_requests": len(self.request_counts[client_ip])
                },
                success=False,
                error_message="Rate limit exceeded"
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed",
                    "retry_after": 60
                }
            )
        
        # Add current request
        self.request_counts[client_ip].append(current_time)
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"


class PermissionDependency:
    """Dependency for checking permissions in FastAPI routes."""
    
    def __init__(self, permission: Permission):
        self.permission = permission
    
    def __call__(self, request: Request) -> SecurityContext:
        """Check permission and return security context."""
        context = getattr(request.state, "security_context", None)
        if not context:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        if not permission_checker.check_permission(context, self.permission):
            # Log permission denied
            asyncio.create_task(
                audit_logger.log_permission_denied(
                    context=context,
                    action=self.permission.value,
                    resource_type="api",
                    resource_id=str(request.url.path)
                )
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {self.permission.value}"
            )
        
        return context


class ResourceAccessDependency:
    """Dependency for checking resource access in FastAPI routes."""
    
    def __init__(self, resource_type: ResourceType, action: Action):
        self.resource_type = resource_type
        self.action = action
    
    def __call__(self, request: Request, resource_id: Optional[str] = None) -> SecurityContext:
        """Check resource access and return security context."""
        context = getattr(request.state, "security_context", None)
        if not context:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        if not permission_checker.check_resource_access(
            context, self.resource_type, self.action, resource_id
        ):
            # Log permission denied
            asyncio.create_task(
                audit_logger.log_permission_denied(
                    context=context,
                    action=f"{self.resource_type.value}:{self.action.value}",
                    resource_type=self.resource_type.value,
                    resource_id=resource_id
                )
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: {self.action.value} on {self.resource_type.value}"
            )
        
        return context


# Convenience functions for creating dependencies
def require_permission(permission: Permission):
    """Create a permission dependency."""
    return PermissionDependency(permission)


def require_resource_access(resource_type: ResourceType, action: Action):
    """Create a resource access dependency."""
    return ResourceAccessDependency(resource_type, action)


# Common permission dependencies
require_admin = require_permission(Permission.SYSTEM_CONFIG)
require_user_management = require_permission(Permission.USER_CREATE)
require_agent_execution = require_permission(Permission.AGENT_EXECUTE)
require_model_training = require_permission(Permission.MODEL_TRAIN)
require_audit_access = require_permission(Permission.AUDIT_READ)


# HTTP Bearer security scheme
security_scheme = HTTPBearer(
    scheme_name="Bearer",
    description="JWT Bearer token authentication"
)