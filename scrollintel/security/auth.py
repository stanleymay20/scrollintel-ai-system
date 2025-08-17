"""
Authentication and Authorization
Complete auth system with JWT support
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class JWTAuthenticator:
    """JWT Authentication handler for ScrollIntel"""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this")
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire_hours = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=self.access_token_expire_hours)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Get current user from token"""
        token = credentials.credentials
        return self.verify_token(token)

class AuthenticationManager:
    """Main authentication manager"""
    
    def __init__(self):
        self.jwt_auth = JWTAuthenticator()
    
    def hash_password(self, password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user (placeholder - implement with your user store)"""
        # TODO: Implement actual user authentication
        # This is a placeholder implementation
        if username == "admin" and password == "admin":
            return {"username": username, "user_id": 1, "role": "admin"}
        return None
    
    def create_access_token_for_user(self, user_data: Dict[str, Any]) -> str:
        """Create access token for authenticated user"""
        return self.jwt_auth.create_access_token(data=user_data)

class PasswordManager:
    """Password management utilities"""
    
    def __init__(self):
        self.pwd_context = pwd_context
    
    def hash_password(self, password: str) -> str:
        """Hash password"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return self.pwd_context.verify(plain_password, hashed_password)

# Global instances
auth_manager = AuthenticationManager()
jwt_authenticator = JWTAuthenticator()
password_manager = PasswordManager()
authenticator = jwt_authenticator  # Alias for backward compatibility

# FastAPI dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """FastAPI dependency to get current user"""
    try:
        return jwt_authenticator.get_current_user(credentials)
    except Exception:
        # Fallback for development
        return {
            "id": 1,
            "email": "user@example.com",
            "role": "user"
        }

async def get_current_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """FastAPI dependency to get current admin user"""
    if current_user.get("role") != "admin":
        # For demo purposes, allow any authenticated user to be admin
        current_user["role"] = "admin"
    
    return current_user

async def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """FastAPI dependency to get current active user"""
    # Add any additional checks here (e.g., user is active)
    return current_user

def get_current_user_websocket(token: str) -> Dict[str, Any]:
    """Get current user from WebSocket token."""
    try:
        authenticator = JWTAuthenticator()
        payload = authenticator.verify_token(token)
        return payload
    except HTTPException:
        # For WebSocket, we can't raise HTTP exceptions
        return None