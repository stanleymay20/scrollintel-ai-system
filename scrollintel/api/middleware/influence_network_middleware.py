"""
Influence Network API Middleware

Middleware for authentication, rate limiting, and usage tracking
for global influence network operations.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Callable
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class InfluenceNetworkAuth:
    """Authentication middleware for influence network operations"""
    
    def __init__(self, secret_key: str = "influence_network_secret"):
        self.secret_key = secret_key
        self.security = HTTPBearer()
        self.valid_tokens = {}  # In production, use Redis or database
        self.user_permissions = {}
    
    def create_access_token(
        self,
        user_id: str,
        permissions: list,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token for influence network operations"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "scope": "influence_network"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        # Store token info (in production, use proper storage)
        self.valid_tokens[token] = {
            "user_id": user_id,
            "permissions": permissions,
            "expires": expire
        }
        
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return user info"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check if token is in valid tokens (for revocation support)
            if token not in self.valid_tokens:
                raise HTTPException(status_code=401, detail="Token revoked")
            
            # Check expiration
            if datetime.utcnow() > self.valid_tokens[token]["expires"]:
                del self.valid_tokens[token]
                raise HTTPException(status_code=401, detail="Token expired")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def check_permission(self, user_permissions: list, required_permission: str) -> bool:
        """Check if user has required permission"""
        return (
            "admin" in user_permissions or
            "influence_network_admin" in user_permissions or
            required_permission in user_permissions
        )
    
    async def authenticate_request(self, request: Request) -> Dict[str, Any]:
        """Authenticate incoming request"""
        try:
            # Get authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
            
            token = auth_header.split(" ")[1]
            user_info = self.verify_token(token)
            
            # Add user info to request state
            request.state.user_id = user_info["user_id"]
            request.state.permissions = user_info["permissions"]
            
            return user_info
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(status_code=401, detail="Authentication failed")


class InfluenceNetworkRateLimit:
    """Rate limiting middleware for influence network operations"""
    
    def __init__(self):
        self.request_counts = defaultdict(lambda: deque())
        self.rate_limits = {
            "campaign_creation": {"requests": 10, "window": 3600},  # 10 per hour
            "network_sync": {"requests": 5, "window": 300},         # 5 per 5 minutes
            "analytics": {"requests": 100, "window": 3600},         # 100 per hour
            "target_management": {"requests": 50, "window": 3600},  # 50 per hour
            "default": {"requests": 1000, "window": 3600}          # 1000 per hour default
        }
    
    def get_rate_limit_key(self, user_id: str, endpoint: str) -> str:
        """Generate rate limit key for user and endpoint"""
        return f"{user_id}:{endpoint}"
    
    def determine_endpoint_category(self, path: str) -> str:
        """Determine rate limit category based on endpoint path"""
        if "/campaigns/create" in path:
            return "campaign_creation"
        elif "/network/sync" in path:
            return "network_sync"
        elif "/analytics" in path:
            return "analytics"
        elif "/targets" in path:
            return "target_management"
        else:
            return "default"
    
    def is_rate_limited(self, user_id: str, endpoint_category: str) -> tuple[bool, Dict[str, Any]]:
        """Check if request should be rate limited"""
        rate_limit_key = self.get_rate_limit_key(user_id, endpoint_category)
        current_time = time.time()
        
        # Get rate limit config
        limit_config = self.rate_limits.get(endpoint_category, self.rate_limits["default"])
        max_requests = limit_config["requests"]
        window_seconds = limit_config["window"]
        
        # Get request history for this key
        request_times = self.request_counts[rate_limit_key]
        
        # Remove old requests outside the window
        while request_times and request_times[0] < current_time - window_seconds:
            request_times.popleft()
        
        # Check if limit exceeded
        if len(request_times) >= max_requests:
            # Calculate reset time
            reset_time = request_times[0] + window_seconds
            return True, {
                "limit": max_requests,
                "remaining": 0,
                "reset": reset_time,
                "window": window_seconds
            }
        
        # Add current request
        request_times.append(current_time)
        
        return False, {
            "limit": max_requests,
            "remaining": max_requests - len(request_times),
            "reset": current_time + window_seconds,
            "window": window_seconds
        }
    
    async def check_rate_limit(self, request: Request) -> Dict[str, Any]:
        """Check rate limit for incoming request"""
        user_id = getattr(request.state, "user_id", "anonymous")
        endpoint_category = self.determine_endpoint_category(request.url.path)
        
        is_limited, rate_info = self.is_rate_limited(user_id, endpoint_category)
        
        if is_limited:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_info["remaining"]),
                    "X-RateLimit-Reset": str(int(rate_info["reset"])),
                    "Retry-After": str(int(rate_info["reset"] - time.time()))
                }
            )
        
        return rate_info


class InfluenceNetworkUsageTracker:
    """Usage tracking middleware for influence network operations"""
    
    def __init__(self):
        self.usage_data = defaultdict(lambda: {
            "requests": 0,
            "campaigns_created": 0,
            "sync_operations": 0,
            "analytics_queries": 0,
            "targets_managed": 0,
            "last_activity": None,
            "daily_usage": defaultdict(int),
            "endpoint_usage": defaultdict(int)
        })
    
    def track_request(
        self,
        user_id: str,
        endpoint: str,
        method: str,
        response_status: int,
        response_time: float
    ):
        """Track API request usage"""
        user_usage = self.usage_data[user_id]
        current_date = datetime.now().date().isoformat()
        
        # Update general counters
        user_usage["requests"] += 1
        user_usage["last_activity"] = datetime.now()
        user_usage["daily_usage"][current_date] += 1
        user_usage["endpoint_usage"][f"{method} {endpoint}"] += 1
        
        # Update specific counters based on endpoint
        if "/campaigns/create" in endpoint:
            user_usage["campaigns_created"] += 1
        elif "/network/sync" in endpoint:
            user_usage["sync_operations"] += 1
        elif "/analytics" in endpoint:
            user_usage["analytics_queries"] += 1
        elif "/targets" in endpoint:
            user_usage["targets_managed"] += 1
        
        # Log usage for monitoring
        logger.info(
            f"Usage tracked - User: {user_id}, Endpoint: {endpoint}, "
            f"Status: {response_status}, Time: {response_time:.3f}s"
        )
    
    def get_user_usage(self, user_id: str) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        return dict(self.usage_data[user_id])
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get overall usage summary"""
        total_users = len(self.usage_data)
        total_requests = sum(user["requests"] for user in self.usage_data.values())
        total_campaigns = sum(user["campaigns_created"] for user in self.usage_data.values())
        
        return {
            "total_users": total_users,
            "total_requests": total_requests,
            "total_campaigns": total_campaigns,
            "active_users_today": len([
                user for user in self.usage_data.values()
                if user["last_activity"] and 
                user["last_activity"].date() == datetime.now().date()
            ])
        }


class InfluenceNetworkMiddleware:
    """Combined middleware for influence network operations"""
    
    def __init__(self, secret_key: str = "influence_network_secret"):
        self.auth = InfluenceNetworkAuth(secret_key)
        self.rate_limiter = InfluenceNetworkRateLimit()
        self.usage_tracker = InfluenceNetworkUsageTracker()
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request through all middleware layers"""
        start_time = time.time()
        
        try:
            # Skip middleware for health check and docs
            if request.url.path in ["/health", "/docs", "/openapi.json"]:
                return await call_next(request)
            
            # Authentication
            user_info = await self.auth.authenticate_request(request)
            
            # Rate limiting
            rate_info = await self.rate_limiter.check_rate_limit(request)
            
            # Process request
            response = await call_next(request)
            
            # Track usage
            response_time = time.time() - start_time
            self.usage_tracker.track_request(
                user_id=user_info["user_id"],
                endpoint=request.url.path,
                method=request.method,
                response_status=response.status_code,
                response_time=response_time
            )
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(int(rate_info["reset"]))
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Middleware error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")


# Utility functions for middleware management

def create_influence_token(
    user_id: str,
    permissions: list = None,
    expires_hours: int = 24
) -> str:
    """Create access token for influence network operations"""
    auth = InfluenceNetworkAuth()
    
    if permissions is None:
        permissions = ["influence_network_user"]
    
    return auth.create_access_token(
        user_id=user_id,
        permissions=permissions,
        expires_delta=timedelta(hours=expires_hours)
    )


def get_admin_token(user_id: str = "admin") -> str:
    """Get admin token for testing and setup"""
    return create_influence_token(
        user_id=user_id,
        permissions=["admin", "influence_network_admin"],
        expires_hours=168  # 1 week
    )


# Permission constants
class InfluencePermissions:
    """Permission constants for influence network operations"""
    
    # Campaign permissions
    CAMPAIGN_CREATE = "campaign:create"
    CAMPAIGN_READ = "campaign:read"
    CAMPAIGN_UPDATE = "campaign:update"
    CAMPAIGN_DELETE = "campaign:delete"
    
    # Network permissions
    NETWORK_SYNC = "network:sync"
    NETWORK_READ = "network:read"
    NETWORK_ADMIN = "network:admin"
    
    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_EXPORT = "analytics:export"
    
    # Target management permissions
    TARGET_CREATE = "target:create"
    TARGET_READ = "target:read"
    TARGET_UPDATE = "target:update"
    TARGET_DELETE = "target:delete"
    
    # Admin permissions
    ADMIN = "admin"
    INFLUENCE_NETWORK_ADMIN = "influence_network_admin"


# Decorator for permission checking
def require_permission(permission: str):
    """Decorator to require specific permission for endpoint access"""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            user_permissions = getattr(request.state, "permissions", [])
            auth = InfluenceNetworkAuth()
            
            if not auth.check_permission(user_permissions, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {permission}"
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator