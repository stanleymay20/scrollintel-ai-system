"""
API Key Middleware for ScrollIntel Launch MVP.
Handles API key authentication, rate limiting, and usage tracking.
"""

import time
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.security.utils import get_authorization_scheme_param
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from ...core.api_key_manager import APIKeyManager
from ...models.database import get_db
from ...models.api_key_models import APIKey


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API key authentication, rate limiting, and usage tracking.
    """
    
    def __init__(self, app, excluded_paths: Optional[list] = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/api/v1/auth",
            "/api/v1/keys"  # Key management endpoints use session auth
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through API key authentication and tracking.
        """
        # Skip middleware for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Skip if not an API request
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        
        # Get database session
        db: Session = next(get_db())
        manager = APIKeyManager(db)
        
        try:
            # Extract API key from Authorization header
            api_key = self._extract_api_key(request)
            if not api_key:
                # Allow requests without API key for now (session auth)
                return await call_next(request)
            
            # Validate API key
            api_key_obj = manager.validate_api_key(api_key)
            if not api_key_obj:
                return self._create_error_response(
                    status.HTTP_401_UNAUTHORIZED,
                    "Invalid API key"
                )
            
            # Check rate limits
            rate_limit_status = manager.check_rate_limit(api_key_obj)
            if not rate_limit_status['allowed']:
                return self._create_rate_limit_response(rate_limit_status, api_key_obj)
            
            # Add API key info to request state
            request.state.api_key = api_key_obj
            request.state.rate_limit_status = rate_limit_status
            
            # Record request start time
            start_time = time.time()
            
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Record usage
            await self._record_usage(
                manager, api_key_obj, request, response, response_time_ms
            )
            
            # Add rate limit headers
            self._add_rate_limit_headers(response, rate_limit_status, api_key_obj)
            
            return response
            
        except Exception as e:
            # Log error and return generic error response
            print(f"API Key Middleware Error: {str(e)}")
            return self._create_error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Internal server error"
            )
        finally:
            db.close()
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from Authorization header."""
        authorization = request.headers.get("Authorization")
        if not authorization:
            return None
        
        scheme, credentials = get_authorization_scheme_param(authorization)
        if scheme.lower() != "bearer":
            return None
        
        return credentials
    
    def _create_error_response(self, status_code: int, detail: str) -> Response:
        """Create error response."""
        return Response(
            content=f'{{"error": "{detail}"}}',
            status_code=status_code,
            media_type="application/json"
        )
    
    def _create_rate_limit_response(
        self, 
        rate_limit_status: dict, 
        api_key: APIKey
    ) -> Response:
        """Create rate limit exceeded response."""
        headers = {
            "X-RateLimit-Limit-Minute": str(api_key.rate_limit_per_minute),
            "X-RateLimit-Remaining-Minute": str(rate_limit_status['minute']['remaining']),
            "X-RateLimit-Reset-Minute": rate_limit_status['minute']['reset_at'].isoformat(),
            "X-RateLimit-Limit-Hour": str(api_key.rate_limit_per_hour),
            "X-RateLimit-Remaining-Hour": str(rate_limit_status['hour']['remaining']),
            "X-RateLimit-Reset-Hour": rate_limit_status['hour']['reset_at'].isoformat(),
            "X-RateLimit-Limit-Day": str(api_key.rate_limit_per_day),
            "X-RateLimit-Remaining-Day": str(rate_limit_status['day']['remaining']),
            "X-RateLimit-Reset-Day": rate_limit_status['day']['reset_at'].isoformat(),
        }
        
        return Response(
            content='{"error": "Rate limit exceeded"}',
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            media_type="application/json",
            headers=headers
        )
    
    def _add_rate_limit_headers(
        self, 
        response: Response, 
        rate_limit_status: dict, 
        api_key: APIKey
    ):
        """Add rate limit headers to response."""
        response.headers["X-RateLimit-Limit-Minute"] = str(api_key.rate_limit_per_minute)
        response.headers["X-RateLimit-Remaining-Minute"] = str(rate_limit_status['minute']['remaining'])
        response.headers["X-RateLimit-Reset-Minute"] = rate_limit_status['minute']['reset_at'].isoformat()
        
        response.headers["X-RateLimit-Limit-Hour"] = str(api_key.rate_limit_per_hour)
        response.headers["X-RateLimit-Remaining-Hour"] = str(rate_limit_status['hour']['remaining'])
        response.headers["X-RateLimit-Reset-Hour"] = rate_limit_status['hour']['reset_at'].isoformat()
        
        response.headers["X-RateLimit-Limit-Day"] = str(api_key.rate_limit_per_day)
        response.headers["X-RateLimit-Remaining-Day"] = str(rate_limit_status['day']['remaining'])
        response.headers["X-RateLimit-Reset-Day"] = rate_limit_status['day']['reset_at'].isoformat()
    
    async def _record_usage(
        self,
        manager: APIKeyManager,
        api_key: APIKey,
        request: Request,
        response: Response,
        response_time_ms: float
    ):
        """Record API usage for tracking and billing."""
        try:
            # Get request size
            request_size = 0
            if hasattr(request, 'body'):
                body = await request.body()
                request_size = len(body) if body else 0
            
            # Get response size
            response_size = 0
            if hasattr(response, 'body'):
                response_size = len(response.body) if response.body else 0
            
            # Get client info
            client_ip = request.client.host if request.client else None
            user_agent = request.headers.get("User-Agent")
            
            # Record usage
            manager.record_api_usage(
                api_key=api_key,
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                request_size_bytes=request_size,
                response_size_bytes=response_size,
                ip_address=client_ip,
                user_agent=user_agent,
                request_metadata={
                    'query_params': dict(request.query_params),
                    'headers': dict(request.headers)
                }
            )
        except Exception as e:
            # Log error but don't fail the request
            print(f"Failed to record API usage: {str(e)}")


class APIKeyRateLimiter:
    """
    Standalone rate limiter for API keys.
    """
    
    def __init__(self, db: Session):
        self.manager = APIKeyManager(db)
    
    async def check_rate_limit(self, api_key: str) -> dict:
        """
        Check rate limit for an API key.
        
        Returns:
            Dict with rate limit status
        """
        api_key_obj = self.manager.validate_api_key(api_key)
        if not api_key_obj:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        return self.manager.check_rate_limit(api_key_obj)
    
    async def record_request(
        self,
        api_key: str,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        **kwargs
    ):
        """
        Record API request for usage tracking.
        """
        api_key_obj = self.manager.validate_api_key(api_key)
        if api_key_obj:
            self.manager.record_api_usage(
                api_key=api_key_obj,
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time_ms=response_time_ms,
                **kwargs
            )