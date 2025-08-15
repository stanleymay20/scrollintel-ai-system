"""Authentication routes."""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime

from ..models.requests import LoginRequest
from ..models.responses import LoginResponse
from ..middleware.auth import authenticate_user, create_access_token

router = APIRouter()


@router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return access token."""
    try:
        # Authenticate user
        user = authenticate_user(request.username, request.password)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )
        
        # Create access token
        token_data = {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "permissions": user.permissions
        }
        access_token = create_access_token(token_data)
        
        return LoginResponse(
            success=True,
            message="Login successful",
            access_token=access_token,
            token_type="bearer",
            expires_in=24 * 3600,  # 24 hours
            user=user.to_dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@router.post("/auth/logout")
async def logout():
    """Logout user (client-side token removal)."""
    return {
        "success": True,
        "message": "Logout successful. Please remove the token from client storage."
    }