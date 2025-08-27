"""
Authentication Middleware for Advanced Analytics Dashboard API

This module provides API key authentication and authorization middleware.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time

from ...core.api_key_manager import APIKeyManager, APIKey, APIKeyStatus

logger = logging.getLogger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API key validation"""
    
    def __init__(self, app, api_key_manager: APIKeyManager):
        super().__init__(app)
        self.api_key_manager = api_key_manager
        
        # Paths that don't require authentication
        self.public_paths = {
            "/health",
            "/api/info",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/graphql/playground"
        }
        
        # Paths that require specific permissions
        self.admin_paths = {
            "/api/v1/admin",
            "/api/v1/webhooks",
            "/api/v1/api-keys"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request through authentication middleware"""
        start_time = time.time()
        
        try:
            # Skip authentication for public paths
            if self._is_public_path(request.url.path):
                response = await call_next(request)
                return response
            
            # Extract API key from request
            api_key = await self._extract_api_key(request)
            
            if not api_key:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Authentication required",
                        "message": "API key is required. Include it in the Authorization header as 'Bearer YOUR_API_KEY'"
                    }
                )
            
            # Validate API key
            key_info = await self._validate_api_key(api_key)
            
            if not key_info:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Invalid API key",
                        "message": "The provided API key is invalid or has been revoked"
                    }
                )
            
            # Check if key is active
            if key_info.status != APIKeyStatus.ACTIVE:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "API key inactive",
                        "message": f"API key status: {key_info.status.value}"
                    }
                )
            
            # Check permissions for admin paths
            if self._is_admin_path(request.url.path) and not key_info.is_admin:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Insufficient permissions",
                        "message": "Admin access required for this endpoint"
                    }
                )
            
            # Check rate limits
            if not await self._check_rate_limits(key_info, request):
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": "API key has exceeded rate limits. Please try again later."
                    }
                )
            
            # Add authentication info to request state
            request.state.api_key = key_info
            request.state.user_id = key_info.user_id
            request.state.organization_id = key_info.organization_id
            
            # Process request
            response = await call_next(request)
            
            # Update API key usage statistics
            await self._update_usage_stats(key_info, request, response, time.time() - start_time)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication middleware error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Authentication error",
                    "message": "An error occurred during authentication"
                }
            )
    
    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (doesn't require authentication)"""
        # Exact match
        if path in self.public_paths:
            return True
        
        # Prefix match for static files and docs
        public_prefixes = ["/static/", "/docs/", "/redoc/"]
        for prefix in public_prefixes:
            if path.startswith(prefix):
                return True
        
        return False
    
    def _is_admin_path(self, path: str) -> bool:
        """Check if path requires admin permissions"""
        for admin_path in self.admin_paths:
            if path.startswith(admin_path):
                return True
        return False
    
    async def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request"""
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header:
            if auth_header.startswith("Bearer "):
                return auth_header[7:]  # Remove "Bearer " prefix
            elif auth_header.startswith("ApiKey "):
                return auth_header[7:]  # Remove "ApiKey " prefix
        
        # Check X-API-Key header
        api_key_header = request.headers.get("X-API-Key")
        if api_key_header:
            return api_key_header
        
        # Check query parameter (less secure, for development only)
        api_key_param = request.query_params.get("api_key")
        if api_key_param:
            logger.warning("API key provided in query parameter - this is insecure for production")
            return api_key_param
        
        return None
    
    async def _validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate API key and return key information"""
        try:
            return await self.api_key_manager.validate_key(api_key)
        except Exception as e:
            logger.error(f"Error validating API key: {str(e)}")
            return None
    
    async def _check_rate_limits(self, key_info: APIKey, request: Request) -> bool:
        """Check if request is within rate limits"""
        try:
            return await self.api_key_manager.check_rate_limit(
                key_info.id,
                request.url.path,
                request.method
            )
        except Exception as e:
            logger.error(f"Error checking rate limits: {str(e)}")
            return True  # Allow request if rate limit check fails
    
    async def _update_usage_stats(self, key_info: APIKey, request: Request, response: Response, duration: float):
        """Update API key usage statistics"""
        try:
            await self.api_key_manager.record_usage(
                key_info.id,
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                response_time=duration,
                request_size=int(request.headers.get("content-length", 0)),
                response_size=len(response.body) if hasattr(response, 'body') else 0
            )
        except Exception as e:
            logger.error(f"Error updating usage stats: {str(e)}")

class APIKeyAuth:
    """API Key authentication dependency"""
    
    def __init__(self, api_key_manager: APIKeyManager):
        self.api_key_manager = api_key_manager
        self.security = HTTPBearer()
    
    async def __call__(self, credentials: HTTPAuthorizationCredentials = None) -> APIKey:
        """Authenticate API key from credentials"""
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="API key required"
            )
        
        key_info = await self.api_key_manager.validate_key(credentials.credentials)
        
        if not key_info:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        if key_info.status != APIKeyStatus.ACTIVE:
            raise HTTPException(
                status_code=401,
                detail=f"API key status: {key_info.status.value}"
            )
        
        return key_info

class AdminAuth:
    """Admin authentication dependency"""
    
    def __init__(self, api_key_auth: APIKeyAuth):
        self.api_key_auth = api_key_auth
    
    async def __call__(self, key_info: APIKey = None) -> APIKey:
        """Authenticate admin API key"""
        if not key_info:
            key_info = await self.api_key_auth()
        
        if not key_info.is_admin:
            raise HTTPException(
                status_code=403,
                detail="Admin access required"
            )
        
        return key_info

class OrganizationAuth:
    """Organization-based authentication dependency"""
    
    def __init__(self, api_key_auth: APIKeyAuth):
        self.api_key_auth = api_key_auth
    
    async def __call__(self, organization_id: str, key_info: APIKey = None) -> APIKey:
        """Authenticate API key for specific organization"""
        if not key_info:
            key_info = await self.api_key_auth()
        
        if key_info.organization_id != organization_id and not key_info.is_admin:
            raise HTTPException(
                status_code=403,
                detail="Access denied for this organization"
            )
        
        return key_info

# Utility functions for extracting auth info from request

def get_current_user_id(request: Request) -> Optional[str]:
    """Get current user ID from request state"""
    return getattr(request.state, 'user_id', None)

def get_current_organization_id(request: Request) -> Optional[str]:
    """Get current organization ID from request state"""
    return getattr(request.state, 'organization_id', None)

def get_current_api_key(request: Request) -> Optional[APIKey]:
    """Get current API key info from request state"""
    return getattr(request.state, 'api_key', None)

def is_admin_user(request: Request) -> bool:
    """Check if current user is admin"""
    api_key = get_current_api_key(request)
    return api_key.is_admin if api_key else False