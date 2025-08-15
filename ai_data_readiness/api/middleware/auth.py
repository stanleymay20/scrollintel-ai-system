"""Authentication and authorization middleware."""

from fastapi import HTTPException, Depends, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
import time
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# JWT configuration (should be in environment variables in production)
JWT_SECRET = "your-secret-key"  # Use environment variable in production
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API requests."""
    
    def __init__(self, app):
        super().__init__(app)
        self.excluded_paths = {
            "/",
            "/health",
            "/api/v1/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/register"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process authentication for incoming requests."""
        # Skip authentication for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Skip authentication for OPTIONS requests
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Extract authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return Response(
                content='{"error": {"code": 401, "message": "Authorization header required"}}',
                status_code=401,
                media_type="application/json"
            )
        
        try:
            # Validate token
            token = auth_header.replace("Bearer ", "")
            payload = verify_token(token)
            
            # Add user info to request state
            request.state.user = payload
            
        except HTTPException as e:
            return Response(
                content=f'{{"error": {{"code": {e.status_code}, "message": "{e.detail}"}}}}',
                status_code=e.status_code,
                media_type="application/json"
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return Response(
                content='{"error": {"code": 401, "message": "Invalid authentication"}}',
                status_code=401,
                media_type="application/json"
            )
        
        return await call_next(request)


def create_access_token(data: Dict[str, Any]) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = time.time() + (JWT_EXPIRATION_HOURS * 3600)
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        # Check if token is expired
        if payload.get("exp", 0) < time.time():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        
        return payload
        
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def get_current_user(request: Request) -> Dict[str, Any]:
    """Get current user from request state."""
    if not hasattr(request.state, 'user'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not authenticated"
        )
    return request.state.user


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            user = get_current_user(request)
            user_permissions = user.get("permissions", [])
            
            if permission not in user_permissions and "admin" not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class User:
    """User model for authentication."""
    
    def __init__(self, user_id: str, username: str, email: str, permissions: list = None):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.permissions = permissions or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "permissions": self.permissions
        }
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions or "admin" in self.permissions


# Mock user database (replace with real database in production)
MOCK_USERS = {
    "admin": {
        "user_id": "1",
        "username": "admin",
        "email": "admin@example.com",
        "password": "admin123",  # Should be hashed in production
        "permissions": ["admin", "read", "write", "delete"]
    },
    "user": {
        "user_id": "2",
        "username": "user",
        "email": "user@example.com",
        "password": "user123",  # Should be hashed in production
        "permissions": ["read", "write"]
    }
}


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user credentials."""
    user_data = MOCK_USERS.get(username)
    if not user_data or user_data["password"] != password:
        return None
    
    return User(
        user_id=user_data["user_id"],
        username=user_data["username"],
        email=user_data["email"],
        permissions=user_data["permissions"]
    )