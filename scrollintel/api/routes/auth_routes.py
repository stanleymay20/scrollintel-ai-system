"""
Authentication routes for ScrollIntel API.
Handles user login, registration, and token management.
"""

import time
from typing import Optional
from datetime import timedelta

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr, Field

from ...core.interfaces import SecurityContext, UserRole
from ...security.auth import authenticator, PasswordManager
from ...security.session import session_manager
from ...security.audit import audit_logger, AuditAction


# Request/Response models
class LoginRequest(BaseModel):
    """Login request model."""
    email: EmailStr
    password: str = Field(..., min_length=1)
    remember_me: bool = False


class RegisterRequest(BaseModel):
    """Registration request model."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)
    role: Optional[UserRole] = UserRole.VIEWER


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: dict


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request model."""
    current_password: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)


def create_auth_router() -> APIRouter:
    """Create authentication router."""
    
    router = APIRouter()
    security = HTTPBearer()
    
    @router.post("/login", response_model=TokenResponse)
    async def login(request: LoginRequest):
        """Authenticate user and return tokens."""
        try:
            # TODO: Implement actual user authentication with database
            # For now, we'll create a mock user for testing
            
            # Validate credentials (mock implementation)
            if request.email == "admin@scrollintel.com" and request.password == "admin123":
                # Create mock user object
                class MockUser:
                    def __init__(self, id, email, role, permissions):
                        self.id = id
                        self.email = email
                        self.role = role
                        self.permissions = permissions
                
                mock_user = MockUser(
                    id="admin-user-id",
                    email=request.email,
                    role=UserRole.ADMIN,
                    permissions=["*"]  # Admin has all permissions
                )
                
                # Generate tokens
                expires_delta = timedelta(days=7) if request.remember_me else None
                access_token = authenticator.create_access_token(mock_user, expires_delta)
                refresh_token = authenticator.create_refresh_token(mock_user)
                
                # Create session (handle case where Redis is not available)
                try:
                    session_id = await session_manager.create_session(
                        user_id=mock_user.id,
                        role=mock_user.role,
                        permissions=mock_user.permissions,
                        ip_address="127.0.0.1",  # Will be set by middleware
                        user_agent="API Client"
                    )
                except Exception:
                    # If session creation fails, use a mock session ID
                    session_id = "mock-session-id"
                
                # Log successful login
                await audit_logger.log(
                    action=AuditAction.USER_LOGIN,
                    resource_type="auth",
                    resource_id=mock_user.id,
                    user_id=mock_user.id,
                    details={"email": request.email, "remember_me": request.remember_me},
                    success=True
                )
                
                return TokenResponse(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    expires_in=3600 * 24 * (7 if request.remember_me else 1),
                    user_info={
                        "id": mock_user.id,
                        "email": mock_user.email,
                        "role": mock_user.role.value,
                        "permissions": mock_user.permissions
                    }
                )
            else:
                # Log failed login attempt
                await audit_logger.log(
                    action=AuditAction.USER_LOGIN,
                    resource_type="auth",
                    resource_id=request.email,
                    details={"email": request.email},
                    success=False,
                    error_message="Invalid credentials"
                )
                
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            await audit_logger.log(
                action=AuditAction.USER_LOGIN,
                resource_type="auth",
                resource_id=request.email,
                details={"email": request.email},
                success=False,
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Login failed due to server error"
            )
    
    @router.post("/register", response_model=TokenResponse)
    async def register(request: RegisterRequest):
        """Register a new user."""
        try:
            # Validate password confirmation
            if request.password != request.confirm_password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Passwords do not match"
                )
            
            # Validate password strength
            if not PasswordManager.validate_password_strength(request.password):
                requirements = PasswordManager.generate_password_requirements()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "message": "Password does not meet security requirements",
                        "requirements": requirements
                    }
                )
            
            # TODO: Implement actual user registration with database
            # For now, we'll return an error indicating registration is not implemented
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="User registration is not yet implemented. Please contact administrator."
            )
            
        except HTTPException:
            raise
        except Exception as e:
            await audit_logger.log(
                action=AuditAction.USER_REGISTER,
                resource_type="auth",
                resource_id=request.email,
                details={"email": request.email, "role": request.role.value if request.role else None},
                success=False,
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed due to server error"
            )
    
    @router.post("/refresh", response_model=TokenResponse)
    async def refresh_token(request: RefreshTokenRequest):
        """Refresh access token using refresh token."""
        try:
            # TODO: Implement actual token refresh with database
            # For now, return not implemented
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Token refresh is not yet implemented"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token refresh failed due to server error"
            )
    
    @router.post("/logout")
    async def logout(token: str = Depends(security)):
        """Logout user and invalidate session."""
        try:
            # Extract user context from token
            context = authenticator.get_user_from_token(token.credentials)
            
            if context:
                # Invalidate session
                if context.session_id:
                    await session_manager.invalidate_session(context.session_id)
                
                # Log logout
                await audit_logger.log(
                    action=AuditAction.USER_LOGOUT,
                    resource_type="auth",
                    resource_id=context.user_id,
                    user_id=context.user_id,
                    details={},
                    success=True
                )
            
            return {"message": "Successfully logged out"}
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Logout failed due to server error"
            )
    
    @router.get("/me")
    async def get_current_user(token: str = Depends(security)):
        """Get current user information."""
        try:
            context = authenticator.get_user_from_token(token.credentials)
            
            if not context:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
            
            # TODO: Fetch full user information from database
            # For now, return information from token
            return {
                "id": context.user_id,
                "role": context.role.value,
                "permissions": context.permissions,
                "session_id": context.session_id
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get user information"
            )
    
    @router.post("/change-password")
    async def change_password(
        request: PasswordChangeRequest,
        token: str = Depends(security)
    ):
        """Change user password."""
        try:
            context = authenticator.get_user_from_token(token.credentials)
            
            if not context:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
            
            # Validate password confirmation
            if request.new_password != request.confirm_password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="New passwords do not match"
                )
            
            # Validate new password strength
            if not PasswordManager.validate_password_strength(request.new_password):
                requirements = PasswordManager.generate_password_requirements()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "message": "New password does not meet security requirements",
                        "requirements": requirements
                    }
                )
            
            # TODO: Implement actual password change with database
            # For now, return not implemented
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Password change is not yet implemented"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            await audit_logger.log(
                action=AuditAction.PASSWORD_CHANGE,
                resource_type="auth",
                resource_id=context.user_id if 'context' in locals() else "unknown",
                user_id=context.user_id if 'context' in locals() else None,
                details={},
                success=False,
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password change failed due to server error"
            )
    
    @router.get("/password-requirements")
    async def get_password_requirements():
        """Get password security requirements."""
        return {
            "requirements": PasswordManager.generate_password_requirements(),
            "description": "Password must meet all of the following requirements"
        }
    
    return router

# Create the router instance
router = create_auth_router()