import os
"""
Data Product API Middleware

Comprehensive middleware for authentication, rate limiting, logging,
and security for data product APIs.
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
from functools import wraps

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
import redis
from sqlalchemy.orm import Session

from scrollintel.models.database import get_db
from scrollintel.models.data_product_models import AccessLevel

logger = logging.getLogger(__name__)

# Configuration
JWT_SECRET = "your-secret-key"  # In production, use environment variable
JWT_ALGORITHM = "HS256"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")  # In production, use environment variable

# Redis client for distributed rate limiting
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    logger.warning(f"Redis not available, using in-memory rate limiting: {e}")
    redis_client = None

# In-memory fallback for rate limiting
memory_rate_limit_storage = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with Redis support"""
    
    def __init__(self, app, default_rate_limit: int = 1000, window_seconds: int = 3600):
        super().__init__(app)
        self.default_rate_limit = default_rate_limit
        self.window_seconds = window_seconds
        
        # Endpoint-specific rate limits
        self.endpoint_limits = {
            "POST:/api/v1/data-products/": {"limit": 50, "window": 3600},
            "PUT:/api/v1/data-products/": {"limit": 30, "window": 3600},
            "DELETE:/api/v1/data-products/": {"limit": 10, "window": 3600},
            "GET:/api/v1/data-products/": {"limit": 200, "window": 3600},
            "GET:/api/v1/data-products/search/": {"limit": 100, "window": 3600},
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting"""
        client_ip = self._get_client_ip(request)
        endpoint_key = f"{request.method}:{request.url.path}"
        
        # Get rate limit for endpoint
        endpoint_config = self.endpoint_limits.get(endpoint_key, {
            "limit": self.default_rate_limit,
            "window": self.window_seconds
        })
        
        # Check rate limit
        if not await self._check_rate_limit(client_ip, endpoint_key, endpoint_config):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {endpoint_config['limit']} per {endpoint_config['window']} seconds",
                    "retry_after": endpoint_config['window']
                },
                headers={"Retry-After": str(endpoint_config['window'])}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self._get_remaining_requests(client_ip, endpoint_key, endpoint_config)
        response.headers["X-RateLimit-Limit"] = str(endpoint_config['limit'])
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + endpoint_config['window'])
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host
    
    async def _check_rate_limit(self, client_ip: str, endpoint_key: str, config: Dict) -> bool:
        """Check if request is within rate limit"""
        key = f"rate_limit:{client_ip}:{endpoint_key}"
        current_time = int(time.time())
        window_start = current_time - config['window']
        
        if redis_client:
            try:
                # Use Redis for distributed rate limiting
                pipe = redis_client.pipeline()
                
                # Remove old entries
                pipe.zremrangebyscore(key, 0, window_start)
                
                # Count current requests
                pipe.zcard(key)
                
                # Add current request
                pipe.zadd(key, {str(current_time): current_time})
                
                # Set expiration
                pipe.expire(key, config['window'])
                
                results = pipe.execute()
                current_count = results[1]
                
                return current_count < config['limit']
                
            except Exception as e:
                logger.error(f"Redis rate limiting error: {e}")
                # Fall back to memory-based rate limiting
                return self._memory_rate_limit_check(client_ip, endpoint_key, config)
        else:
            return self._memory_rate_limit_check(client_ip, endpoint_key, config)
    
    def _memory_rate_limit_check(self, client_ip: str, endpoint_key: str, config: Dict) -> bool:
        """Memory-based rate limiting fallback"""
        key = f"{client_ip}:{endpoint_key}"
        current_time = time.time()
        window_start = current_time - config['window']
        
        # Clean old requests
        memory_rate_limit_storage[key] = [
            req_time for req_time in memory_rate_limit_storage[key]
            if req_time > window_start
        ]
        
        # Check limit
        if len(memory_rate_limit_storage[key]) >= config['limit']:
            return False
        
        # Add current request
        memory_rate_limit_storage[key].append(current_time)
        return True
    
    async def _get_remaining_requests(self, client_ip: str, endpoint_key: str, config: Dict) -> int:
        """Get remaining requests for client"""
        key = f"rate_limit:{client_ip}:{endpoint_key}"
        
        if redis_client:
            try:
                current_count = redis_client.zcard(key)
                return max(0, config['limit'] - current_count)
            except Exception:
                pass
        
        # Memory fallback
        memory_key = f"{client_ip}:{endpoint_key}"
        current_count = len(memory_rate_limit_storage.get(memory_key, []))
        return max(0, config['limit'] - current_count)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """JWT authentication middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.security = HTTPBearer()
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "GET:/api/v1/data-products/health",
            "GET:/docs",
            "GET:/redoc",
            "GET:/openapi.json"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with authentication"""
        endpoint_key = f"{request.method}:{request.url.path}"
        
        # Skip authentication for public endpoints
        if endpoint_key in self.public_endpoints:
            return await call_next(request)
        
        # Extract and verify token
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Missing or invalid authorization header"}
                )
            
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            
            # Add user info to request state
            request.state.user_id = payload.get("sub")
            request.state.permissions = payload.get("permissions", [])
            request.state.user_info = payload
            
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={"error": "Token has expired"}
            )
        except jwt.InvalidTokenError:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid token"}
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return JSONResponse(
                status_code=401,
                content={"error": "Authentication failed"}
            )
        
        return await call_next(request)


class AccessControlMiddleware(BaseHTTPMiddleware):
    """Role-based access control middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        
        # Endpoint permissions mapping
        self.endpoint_permissions = {
            "POST:/api/v1/data-products/": "data_product:create",
            "GET:/api/v1/data-products/": "data_product:read",
            "PUT:/api/v1/data-products/": "data_product:update",
            "DELETE:/api/v1/data-products/": "data_product:delete",
            "POST:/api/v1/data-products/*/provenance": "data_product:manage_provenance",
            "PUT:/api/v1/data-products/*/quality-metrics": "data_product:manage_quality",
            "PUT:/api/v1/data-products/*/bias-assessment": "data_product:manage_bias",
            "PUT:/api/v1/data-products/*/verification": "data_product:verify",
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with access control"""
        # Skip if no user info (handled by auth middleware)
        if not hasattr(request.state, 'user_id'):
            return await call_next(request)
        
        endpoint_key = f"{request.method}:{request.url.path}"
        required_permission = self._get_required_permission(endpoint_key)
        
        if required_permission:
            user_permissions = getattr(request.state, 'permissions', [])
            
            if required_permission not in user_permissions:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Insufficient permissions",
                        "required_permission": required_permission
                    }
                )
        
        return await call_next(request)
    
    def _get_required_permission(self, endpoint_key: str) -> Optional[str]:
        """Get required permission for endpoint"""
        # Direct match
        if endpoint_key in self.endpoint_permissions:
            return self.endpoint_permissions[endpoint_key]
        
        # Pattern matching for dynamic routes
        for pattern, permission in self.endpoint_permissions.items():
            if "*" in pattern:
                pattern_parts = pattern.split("*")
                if (endpoint_key.startswith(pattern_parts[0]) and 
                    endpoint_key.endswith(pattern_parts[1])):
                    return permission
        
        return None


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Audit logging middleware for data product operations"""
    
    def __init__(self, app):
        super().__init__(app)
        self.audit_logger = logging.getLogger("audit")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with audit logging"""
        start_time = time.time()
        
        # Capture request details
        request_data = {
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "path": str(request.url.path),
            "query_params": dict(request.query_params),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent"),
            "user_id": getattr(request.state, 'user_id', None)
        }
        
        # Process request
        response = await call_next(request)
        
        # Capture response details
        processing_time = time.time() - start_time
        
        audit_data = {
            **request_data,
            "status_code": response.status_code,
            "processing_time_ms": round(processing_time * 1000, 2),
            "response_size": len(response.body) if hasattr(response, 'body') else 0
        }
        
        # Log based on operation type
        if self._is_sensitive_operation(request.method, request.url.path):
            self.audit_logger.info(f"SENSITIVE_OPERATION: {json.dumps(audit_data)}")
        elif response.status_code >= 400:
            self.audit_logger.warning(f"ERROR_RESPONSE: {json.dumps(audit_data)}")
        else:
            self.audit_logger.info(f"API_REQUEST: {json.dumps(audit_data)}")
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host
    
    def _is_sensitive_operation(self, method: str, path: str) -> bool:
        """Check if operation is sensitive and requires detailed logging"""
        sensitive_patterns = [
            ("POST", "/api/v1/data-products/"),
            ("PUT", "/api/v1/data-products/"),
            ("DELETE", "/api/v1/data-products/"),
            ("PUT", "/verification"),
            ("POST", "/provenance")
        ]
        
        for sensitive_method, pattern in sensitive_patterns:
            if method == sensitive_method and pattern in path:
                return True
        
        return False


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response"""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


# Utility functions for token generation and validation
def generate_access_token(user_id: str, permissions: List[str], expires_in: int = 3600) -> str:
    """Generate JWT access token"""
    payload = {
        "sub": user_id,
        "permissions": permissions,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(seconds=expires_in)
    }
    
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def validate_token(token: str) -> Dict[str, Any]:
    """Validate JWT token and return payload"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_permissions(required_permissions: List[str]):
    """Decorator to require specific permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request') or next((arg for arg in args if isinstance(arg, Request)), None)
            
            if not request or not hasattr(request.state, 'permissions'):
                raise HTTPException(status_code=401, detail="Authentication required")
            
            user_permissions = request.state.permissions
            missing_permissions = [perm for perm in required_permissions if perm not in user_permissions]
            
            if missing_permissions:
                raise HTTPException(
                    status_code=403,
                    detail=f"Missing required permissions: {', '.join(missing_permissions)}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Rate limiting decorator for specific endpoints
def endpoint_rate_limit(max_requests: int, window_seconds: int = 3600):
    """Rate limiting decorator for specific endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request') or next((arg for arg in args if isinstance(arg, Request)), None)
            
            if not request:
                return await func(*args, **kwargs)
            
            client_ip = request.client.host
            endpoint_key = f"{request.method}:{request.url.path}"
            key = f"endpoint_rate_limit:{client_ip}:{endpoint_key}"
            
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            if redis_client:
                try:
                    # Use Redis for rate limiting
                    pipe = redis_client.pipeline()
                    pipe.zremrangebyscore(key, 0, window_start)
                    pipe.zcard(key)
                    pipe.zadd(key, {str(current_time): current_time})
                    pipe.expire(key, window_seconds)
                    
                    results = pipe.execute()
                    current_count = results[1]
                    
                    if current_count >= max_requests:
                        raise HTTPException(
                            status_code=429,
                            detail=f"Rate limit exceeded. Max {max_requests} requests per {window_seconds} seconds"
                        )
                        
                except Exception as e:
                    logger.error(f"Redis rate limiting error: {e}")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator